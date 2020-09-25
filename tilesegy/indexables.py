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


class Trace(Indexable):
    def __len__(self) -> int:
        return cast(int, np.asarray(self._tdb.shape[:-1]).prod())

    def __getitem__(self, i: Union[Index, Tuple[Index, Index]]) -> np.ndarray:
        if isinstance(i, tuple):
            # for single sample segyio returns an array of size 1 instead of scalar
            trace_index, samples = i[0], ensure_slice(i[1])
        else:
            trace_index, samples = i, slice(None)

        shape = self._tdb.shape[:-1]
        if len(shape) == 1:
            # unstructured tilesegy: already indexed by trace
            return self._tdb[trace_index, samples]

        if not isinstance(trace_index, slice):
            # translate single trace_index to fast-slow-offset index
            return self._tdb[(*np.unravel_index(trace_index, shape), samples)]

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
        return traces[selected_product_indices]


class Header(Indexable):
    def __len__(self) -> int:
        return cast(int, np.asarray(self._shape).prod())

    @singledispatchmethod
    def __getitem__(self, i: object) -> None:
        raise NotImplementedError(f"Cannot index by {i.__class__}")  # pragma: nocover

    @__getitem__.register(int)
    def _get_one(self, i: int) -> Dict[str, int]:
        return cast(Dict[str, int], self._tdb[np.unravel_index(i, self._shape)])

    @__getitem__.register(slice)
    def _get_many(self, i: slice) -> List[Dict[str, int]]:
        shape = self._shape
        if len(shape) == 1:
            # unstructured tilesegy: already indexed by trace
            headers = self._tdb[i]
        else:
            # get indices in 1D (trace index) and 3D (fast-slow-offset indices)
            raveled_indices = np.arange(len(self))[i]
            unraveled_indices = np.unravel_index(raveled_indices, shape)
            unique_unraveled_indices = tuple(map(np.unique, unraveled_indices))

            # find the requested subset of indices from the cartesian product
            points = frozenset(zip(*unraveled_indices))
            selected_product_indices = [
                i
                for i, point in enumerate(it.product(*unique_unraveled_indices))
                if point in points
            ]

            # get the hypercube (fast-slow-offset-samples) for the cartesian product of
            # unique_unraveled_indices and reshape it to 2D (trace-samples)
            headers = self._tdb[tuple(map(ensure_slice, unique_unraveled_indices))]
            for key, value in headers.items():
                headers[key] = value.reshape(-1)[selected_product_indices]

        keys = headers.keys()
        columns = [v.tolist() for v in headers.values()]
        return [dict(zip(keys, row)) for row in zip(*columns)]

    @property
    def _shape(self) -> Tuple[int, ...]:
        return cast(Tuple[int, ...], self._tdb.shape)


class Attributes(Header):
    def __getitem__(self, i: Index) -> np.ndarray:
        # for single sample segyio returns an array of size 1 instead of scalar
        i = ensure_slice(i)

        shape = self._shape
        if len(shape) == 1:
            # unstructured tilesegy: already indexed by trace
            return self._tdb[i]
        else:
            # get indices in 1D (trace index) and 3D (fast-slow-offset indices)
            raveled_indices = np.arange(len(self))[i]
            unraveled_indices = np.unravel_index(raveled_indices, shape)
            unique_unraveled_indices = tuple(map(np.unique, unraveled_indices))

            # find the requested subset of indices from the cartesian product
            points = frozenset(zip(*unraveled_indices))
            selected_product_indices = [
                i
                for i, point in enumerate(it.product(*unique_unraveled_indices))
                if point in points
            ]

            # get the hypercube (fast-slow-offset-samples) for the cartesian product of
            # unique_unraveled_indices and reshape it to 2D (trace-samples)
            values = self._tdb[tuple(map(ensure_slice, unique_unraveled_indices))]
            return values.reshape(-1)[selected_product_indices]


class Line(Indexable):
    def __init__(
        self,
        dim_name: str,
        labels: np.ndarray,
        offsets: np.ndarray,
        tdb: tiledb.Array,
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

        return data

    _dims = property(lambda self: [dim.name for dim in self._tdb.schema.domain])


class Depth(Indexable):
    def __len__(self) -> int:
        return cast(int, self._tdb.shape[-1])

    def __getitem__(self, i: Index) -> np.ndarray:
        data = self._tdb[:, i]
        return data.swapaxes(0, 1) if data.ndim == 2 else data


class StructuredDepth(Depth):
    def __getitem__(self, i: Index) -> np.ndarray:
        # segyio depth doesn't currently support offsets; pick the first one
        # https://github.com/equinor/segyio/issues/474
        try:
            data = self._tdb[..., 0, i]
        except IndexError as ex:
            raise TypeError(
                f"depth indices must be integers or slices, not {i.__class__.__name__}"
            ) from ex

        if data.ndim == 3:
            # (fast, slow, samples) -> (samples, fast, slow)
            data = data.swapaxes(0, 2).swapaxes(1, 2)
        return data
