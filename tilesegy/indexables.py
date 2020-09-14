from typing import Dict, List, Tuple, Union, cast

import numpy as np
import tiledb

from ._singledispatchmethod import singledispatchmethod  # type: ignore

Index = Union[int, slice]


class Sized:
    def __init__(self, tdb: tiledb.Array):
        self._tdb = tdb

    def __len__(self) -> int:
        return len(self._tdb)


class Header(Sized):
    @singledispatchmethod
    def __getitem__(self, i: object) -> None:
        raise NotImplementedError(f"Cannot index by {i.__class__}")  # pragma: nocover

    @__getitem__.register(int)
    def _get_one(self, i: int) -> int:
        return cast(int, self._tdb[i].item())

    @__getitem__.register(slice)
    def _get_many(self, i: slice) -> List[int]:
        return cast(List[int], self._tdb[i].tolist())


class Headers(Sized):
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


class Traces:
    def __init__(self, data_tdb: tiledb.Array, headers_tdb: tiledb.Array):
        self._data_tdb = data_tdb
        self._headers_tdb = headers_tdb

    def __len__(self) -> int:
        return len(self._data_tdb)

    def __getitem__(
        self, i: Union[Index, Tuple[Index, Index]]
    ) -> Union[np.number, np.ndarray]:
        return self._data_tdb[i]

    @property
    def headers(self) -> Headers:
        return Headers(self._headers_tdb)

    def header(self, name: str) -> Header:
        return Header(tiledb.DenseArray(self._headers_tdb.uri, attr=name))


class Lines:
    def __init__(
        self,
        data_tdb: tiledb.Array,
        headers_tdb: tiledb.Array,
        labels: np.ndarray,
        offsets: np.ndarray,
        dimension: int,
    ):
        self._data_tdb = data_tdb
        self._headers_tdb = headers_tdb
        self._labels = labels
        self._offsets = offsets
        self._dim = dimension
        self._label_indexer = LabelIndexer(self._labels)
        self._offset_indexer = LabelIndexer(self._offsets)

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    def __len__(self) -> int:
        return cast(int, self._data_tdb.shape[self._dim])

    def __getitem__(self, i: Union[Index, Tuple[Index, Index]]) -> np.ndarray:
        if isinstance(i, tuple):
            labels, offsets = i
        else:
            labels = i
            offsets = self._offsets[0]
        label_indices = self._label_indexer[labels]
        offset_indices = self._offset_indexer[offsets]

        composite_idx: List[Index] = [slice(None)] * 4
        composite_idx[self._dim] = label_indices
        composite_idx[2] = offset_indices
        data = self._data_tdb[tuple(composite_idx)]

        # TODO: Simplify this logic
        if isinstance(label_indices, slice) and self._dim != 0:
            data = data.swapaxes(0, self._dim)
        if isinstance(offset_indices, slice):
            if isinstance(label_indices, slice):
                data = data.swapaxes(1, 2)
            else:
                data = data.swapaxes(0, 1)

        return data


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
    def __getitem__(self, label: object) -> int:
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
