import itertools as it
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union, cast

import numpy as np
import tiledb

from ._singledispatchmethod import singledispatchmethod  # type: ignore
from .utils import LabelIndexer, ensure_slice

Index = Union[int, np.integer, slice]


class Indexable(ABC):
    def __init__(self, tdb: tiledb.Array):
        self._tdb = tdb

    @abstractmethod
    def __len__(self) -> int:
        ...  # pragma: nocover

    @abstractmethod
    def __getitem__(self, i: Index) -> Any:
        ...  # pragma: nocover


class Attributes(Indexable):
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


class Header(Indexable):
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
    def __len__(self) -> int:
        return cast(int, self._tdb.shape[1])

    def __getitem__(self, i: Index) -> np.ndarray:
        data = self._tdb[:, i]
        return data.swapaxes(0, 1) if data.ndim == 2 else data


class Trace(Indexable):
    def __len__(self) -> int:
        return cast(int, np.asarray(self._tdb.shape[:-1]).prod())

    def __getitem__(self, i: Union[Index, Tuple[Index, Index]]) -> np.ndarray:
        # for single sample segyio returns an array of size 1 instead of scalar
        if isinstance(i, tuple):
            i = i[0], ensure_slice(i[1])
        return self._tdb[i]


class StructuredTrace(Trace):
    def __getitem__(self, i: Union[Index, Tuple[Index, Index]]) -> np.ndarray:
        if isinstance(i, tuple):
            # for single sample segyio returns an array of size 1 instead of scalar
            trace_index, samples = i[0], ensure_slice(i[1])
        else:
            trace_index, samples = i, slice(None)

        shape = self._tdb.shape[:-1]
        if isinstance(trace_index, slice):
            # get indices in 1D (trace index) and 3D (fast-slow-offset indices)
            raveled_indices = np.arange(len(self))[trace_index]
            unraveled_indices = np.unravel_index(raveled_indices, shape)
            unique_unraveled_indices = tuple(map(np.unique, unraveled_indices))

            # get the hypercube (fast-slow-offset-samples) for the cartesian product of
            # unique_unraveled_indices and reshape it to 2D (trace-samples)
            traces = self._tdb[(*map(ensure_slice, unique_unraveled_indices), samples)]
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
            traces = self._tdb[(*np.unravel_index(trace_index, shape), samples)]

        return traces


class Line(Indexable):
    def __init__(
        self, dim_name: str, labels: np.ndarray, offsets: np.ndarray, tdb: tiledb.Array,
    ):
        super().__init__(tdb)
        self._dim_name = dim_name
        self._label_indexer = LabelIndexer(labels)
        self._offset_indexer = LabelIndexer(offsets)
        self._default_offset = offsets[0]

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._dim_name!r})"

    def __len__(self) -> int:
        return cast(int, self._tdb.shape[self._dims.index(self._dim_name)])

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
        data = self._tdb[tuple(composite_index)]

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

    _dims = property(lambda self: [dim.name for dim in self._tdb.schema.domain])
