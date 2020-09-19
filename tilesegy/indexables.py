from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union, cast

import numpy as np
import tiledb

from ._singledispatchmethod import singledispatchmethod  # type: ignore

Index = Union[int, slice]


class Indexable(ABC):
    @abstractmethod
    def __len__(self) -> int:
        ...  # pragma: nocover

    @abstractmethod
    def __getitem__(self, i: Index) -> Any:
        ...  # pragma: nocover


class Header(Indexable):
    def __init__(self, tdb: tiledb.Array):
        self._tdb = tdb

    def __len__(self) -> int:
        return len(self._tdb)

    @singledispatchmethod
    def __getitem__(self, i: object) -> None:
        raise NotImplementedError(f"Cannot index by {i.__class__}")  # pragma: nocover

    @__getitem__.register(int)
    def _get_one(self, i: int) -> int:
        return cast(int, self._tdb[i].item())

    @__getitem__.register(slice)
    def _get_many(self, i: slice) -> List[int]:
        return cast(List[int], self._tdb[i].tolist())


class Headers(Indexable):
    def __init__(self, tdb: tiledb.Array):
        self._tdb = tdb

    def __len__(self) -> int:
        return len(self._tdb)

    @singledispatchmethod
    def __getitem__(self, i: object) -> None:
        raise NotImplementedError(f"Cannot index by {i.__class__}")  # pragma: nocover

    @__getitem__.register(int)
    def _get_one(self, i: int) -> Dict[str, int]:
        return cast(Dict[str, int], self._tdb[i])

    @__getitem__.register(slice)
    def _get_many(self, i: slice) -> List[Dict[str, int]]:
        headers = self._tdb[i]
        keys = headers.keys()
        columns = [v.tolist() for v in headers.values()]
        return [dict(zip(keys, row)) for row in zip(*columns)]


class TraceDepth(Indexable):
    def __init__(self, tdb: tiledb.Array):
        self._tdb = tdb

    def __len__(self) -> int:
        return cast(int, self._tdb.shape[1])

    def __getitem__(self, i: Index) -> np.ndarray:
        data = self._tdb[:, i]
        return data.swapaxes(0, 1) if data.ndim == 2 else data


class Traces(Indexable):
    def __init__(self, data: tiledb.Array, headers: tiledb.Array):
        self._data = data
        self._headers = headers

    def __len__(self) -> int:
        return cast(int, np.asarray(self._data.shape[:-1]).prod())

    def __getitem__(
        self, i: Union[Index, Tuple[Index, Index]]
    ) -> Union[np.number, np.ndarray]:
        trace = self._data[i]
        return trace if trace.ndim > 0 else trace[()]

    @property
    def headers(self) -> Headers:
        return Headers(self._headers)

    def header(self, name: str) -> Header:
        return Header(tiledb.DenseArray(self._headers.uri, attr=name))


class StructuredTraces(Traces):
    def __getitem__(
        self, i: Union[Index, Tuple[Index, Index]]
    ) -> Union[np.number, np.ndarray]:
        if not isinstance(i, tuple):
            i = i, slice(None)
        traces = self._get_traces(*i)
        return traces if traces.ndim > 0 else traces[()]

    @singledispatchmethod
    def _get_traces(self, t: object, s: Index) -> np.ndarray:
        raise NotImplementedError(f"Cannot index by {t.__class__}")  # pragma: nocover

    @_get_traces.register(int)
    def _get_one_trace(self, t: int, s: Index) -> np.ndarray:
        # we store data as (fast, offset, slow), trace is (fast, slow, offset) order
        fast, offset, slow, _ = self._data.shape
        fast, slow, offset = np.unravel_index(t, (fast, slow, offset))
        return self._data[(fast, offset, slow, s)]

    @_get_traces.register(slice)
    def _get_slice_traces(self, t: slice, s: Index) -> np.ndarray:
        return np.stack(self._get_traces(i, s) for i in range(len(self))[t])


class Lines(Indexable):
    def __init__(
        self,
        dim_name: str,
        labels: np.ndarray,
        offsets: np.ndarray,
        data: tiledb.Array,
        headers: tiledb.Array,
    ):
        self._dim_name = dim_name
        self._label_indexer = LabelIndexer(labels)
        self._offset_indexer = LabelIndexer(offsets)
        self._default_offset = offsets[0]
        self._data = data
        self._headers = headers

    def __str__(self) -> str:
        return f"Lines({self._dim_name!r})"

    def __len__(self) -> int:
        return cast(int, self._data.shape[self._dims.index(self._dim_name)])

    def __getitem__(self, i: Union[Index, Tuple[Index, Index]]) -> np.ndarray:
        if isinstance(i, tuple):
            labels, offsets = i
        else:
            labels, offsets = i, self._default_offset

        offsets_dim = "offsets"
        labels_dim = self._dim_name
        dims = self._dims

        composite_index: List[Index] = [slice(None)] * 4
        composite_index[dims.index(labels_dim)] = self._label_indexer[labels]
        composite_index[dims.index(offsets_dim)] = self._offset_indexer[offsets]
        data = self._data[tuple(composite_index)]

        if not isinstance(labels, slice):
            dims.remove(labels_dim)
        if not isinstance(offsets, slice):
            dims.remove(offsets_dim)

        # ensure the labels dim is the first axis (if present)
        try:
            labels_axis = dims.index(labels_dim)
            if labels_axis != 0:
                data = data.swapaxes(0, labels_axis)
                labels_axis = 0
        except ValueError:
            labels_axis = -1

        # ensure the offsets dim is right after the labels
        try:
            offsets_axis = dims.index(offsets_dim)
            if offsets_axis != labels_axis + 1:
                data = data.swapaxes(labels_axis + 1, offsets_axis)
        except ValueError:
            pass

        # for samples, if at least one axis is swapped, the last two dims are (slow, fast)
        # if so, swap them to (fast, slow)
        if labels_dim == "samples" and len(dims) > 2:
            data = data.swapaxes(-1, -2)

        return data

    _dims = property(lambda self: [dim.name for dim in self._data.schema.domain])


class LabelIndexer:
    def __init__(self, labels: np.ndarray):
        if not issubclass(labels.dtype.type, np.integer):
            raise ValueError("labels should be integers")
        if len(np.unique(labels)) != len(labels):
            raise ValueError(f"labels should not contain duplicates: {labels}")
        self._labels = labels
        self._min_label = labels.min()
        self._max_label = labels.max() + 1
        self._sorter = labels.argsort()

    @singledispatchmethod
    def __getitem__(self, label: int) -> int:
        indices = np.flatnonzero(label == self._labels)
        assert indices.size <= 1, indices
        if indices.size == 0:
            raise ValueError(f"{label} is not in labels")
        return int(indices[0])

    @__getitem__.register(slice)
    def _get_slice(self, label_slice: slice) -> slice:
        start, stop, step = label_slice.start, label_slice.stop, label_slice.step
        min_label = self._min_label
        if step is None or step > 0:  # increasing step
            if start is None or start < min_label:
                start = min_label
        else:  # decreasing step
            if stop is None or stop < min_label - 1:
                stop = min_label - 1

        label_range = np.arange(*slice(start, stop, step).indices(self._max_label))
        indices = self._sorter[
            self._labels.searchsorted(label_range, sorter=self._sorter)
        ]
        indices = indices[self._labels[indices] == label_range]
        if len(indices) == 0:
            raise ValueError(f"{label_slice} has no overlap with labels")

        start = indices[0]
        step = indices[1] - start if len(indices) > 1 else 1
        stop = indices[-1] + (1 if step > 0 else -1)
        if (np.arange(start, stop, step) != indices).any():
            raise ValueError(
                f"Label indices for {label_slice} is not a slice: {indices}"
            )

        return slice(start, stop, step)
