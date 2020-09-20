import itertools as it
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
        # for single sample segyio returns an array of size 1 instead of scalar
        if isinstance(i, tuple) and not isinstance(i[1], slice):
            i = i[0], slice(i[1], i[1] + 1)
        return self._data[i]

    @property
    def headers(self) -> Headers:
        return Headers(self._headers)

    def header(self, name: str) -> Header:
        return Header(tiledb.DenseArray(self._headers.uri, attr=name))


class StructuredTraces(Traces):
    def __getitem__(
        self, i: Union[Index, Tuple[Index, Index]]
    ) -> Union[np.number, np.ndarray]:
        if isinstance(i, tuple):
            trace_index, samples = i
            # for single sample segyio returns an array of size 1 instead of scalar
            if not isinstance(samples, slice):
                samples = slice(samples, samples + 1)
        else:
            trace_index, samples = i, slice(None)

        shape = self._data.shape[:-1]
        if isinstance(trace_index, slice):
            # get indices in 1D (trace index) and 3D (fast-slow-offset indices)
            raveled_indices = np.arange(len(self))[trace_index]
            unraveled_indices = np.unravel_index(raveled_indices, shape)
            unique_unraveled_indices = tuple(map(np.unique, unraveled_indices))

            # get the hypercube (fast-slow-offset-samples) for the cartesian product of
            # unique_unraveled_indices and reshape it to 2D (trace-samples)
            traces = self._data[
                (*map(array_to_slice, unique_unraveled_indices), samples)
            ]
            traces = traces.reshape(np.array(traces.shape[:-1]).prod(), -1)

            # select the requested subset of indices from the cartesian product
            points = frozenset(zip(*unraveled_indices))
            selected_product_indices = [
                i
                for i, point in enumerate(it.product(*unique_unraveled_indices))
                if point in points
            ]
            traces = traces[selected_product_indices]
        else:
            traces = self._data[(*np.unravel_index(trace_index, shape), samples)]

        return traces


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

        return array_to_slice(indices)


def array_to_slice(a: np.ndarray) -> slice:
    start = a[0]
    step = a[1] - start if len(a) > 1 else 1
    stop = a[-1] + (1 if step > 0 else -1)
    if not np.array_equal(np.arange(start, stop, step), a):
        raise ValueError(f"Array is not a range: {a}")
    return slice(start, stop, step)
