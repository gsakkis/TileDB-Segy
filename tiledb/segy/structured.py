import itertools as it
from functools import singledispatchmethod
from typing import List, Tuple, Union, cast, overload

import numpy as np
from segyio import TraceSortingFormat

import tiledb

from .types import (
    Ellipsis,
    ExtendedIndex,
    ExtendedIndices,
    Index,
    NestedFieldList,
    cached_property,
)
from .unstructured import Header, Segy, TraceIndexer


class StructuredTraceIndexer(TraceIndexer):
    def __getitem__(self, trace_index: Index) -> Tuple[ExtendedIndices, ExtendedIndex]:
        if isinstance(trace_index, int):
            return np.unravel_index(trace_index, self._shape), Ellipsis

        # get indices in 1D (trace index) and 3D (fast-slow-offset indices)
        raveled_indices = np.arange(len(self))[trace_index]
        unraveled_indices = np.unravel_index(raveled_indices, self._shape)
        unique_unraveled_indices = tuple(map(np.unique, unraveled_indices))
        if (trace_index.step or 1) < 0:
            unique_unraveled_indices = tuple(map(np.flip, unique_unraveled_indices))
        bounding_box = cast(
            Tuple[List[int]], tuple(map(list, unique_unraveled_indices))
        )

        # find the requested subset of indices from the cartesian product
        points = frozenset(zip(*unraveled_indices))
        post_reshape_indices = [
            i
            for i, point in enumerate(it.product(*unique_unraveled_indices))
            if point in points
        ]
        return bounding_box, post_reshape_indices


class LabelIndexer:
    def __init__(self, labels: np.ndarray):
        if not issubclass(labels.dtype.type, np.integer):
            raise ValueError("labels should be integers")
        if len(np.unique(labels)) != len(labels):
            raise ValueError(f"labels should not contain duplicates: {labels}")
        self._labels = labels
        self._min_label = int(labels.min())
        self._max_label = int(labels.max() + 1)
        self._sorter = labels.argsort()

    @singledispatchmethod
    def _get_index(self, i: Index) -> Union[int, List[int]]:
        raise TypeError(f"Cannot index by {i.__class__}")

    @overload
    @_get_index.register(np.integer)
    @_get_index.register
    def __getitem__(self, label: int) -> int:
        indices = np.flatnonzero(label == self._labels)
        assert indices.size <= 1, indices
        if indices.size == 0:
            raise ValueError(f"{label} is not in labels")
        return int(indices[0])

    @overload
    @_get_index.register
    def __getitem__(self, label_slice: slice) -> List[int]:
        start, stop, step = label_slice.start, label_slice.stop, label_slice.step
        increasing = step is None or step > 0
        if start is None and increasing:
            start = self._min_label
        elif stop is None and not increasing:
            stop = self._min_label - 1

        label_range = np.arange(*slice(start, stop, step).indices(self._max_label))
        indices = self._sorter[
            self._labels.searchsorted(label_range, sorter=self._sorter)
        ]
        return list(indices[self._labels[indices] == label_range])

    def __getitem__(self, i: Index) -> Union[int, List[int]]:
        return self._get_index(i)


class Line:
    def __init__(
        self,
        dim_name: str,
        labels: np.ndarray,
        offsets: np.ndarray,
        tdb: tiledb.Array,
    ):
        self.name = dim_name
        self._tdb = tdb
        self._label_indexer = LabelIndexer(labels)
        self._offset_indexer = LabelIndexer(offsets)
        self._default_offset = offsets[0]

    def __len__(self) -> int:
        return cast(int, self._tdb.shape[self._dims.index(self.name)])

    def __getitem__(self, i: Union[Index, Tuple[Index, Index]]) -> np.ndarray:
        if isinstance(i, tuple):
            labels, offsets = i
        else:
            labels, offsets = i, self._default_offset
        data = self._tdb[self._get_tdb_indices(labels, offsets)]
        data = self._moveaxis(data, labels, offsets)
        return data

    _dims = property(lambda self: [dim.name for dim in self._tdb.schema.domain])

    def _get_tdb_indices(self, labels: Index, offsets: Index) -> ExtendedIndices:
        dims = self._dims
        composite_index: List[ExtendedIndex] = [slice(None)] * self._tdb.ndim
        composite_index[dims.index(self.name)] = self._label_indexer[labels]
        composite_index[dims.index("offsets")] = self._offset_indexer[offsets]
        return tuple(composite_index)

    def _moveaxis(self, data: np.ndarray, labels: Index, offsets: Index) -> np.ndarray:
        labels_dim = self.name
        offsets_dim = "offsets"

        dims = self._dims
        if not isinstance(labels, slice):
            dims.remove(labels_dim)
        if not isinstance(offsets, slice):
            dims.remove(offsets_dim)

        # ensure the labels dim is the first axis (if present)
        try:
            labels_axis = dims.index(labels_dim)
            if labels_axis != 0:
                data = np.moveaxis(data, labels_axis, 0)
                labels_axis = 0
        except ValueError:
            labels_axis = -1

        # ensure the offsets dim is right after the labels
        try:
            offsets_axis = dims.index(offsets_dim)
            if offsets_axis != labels_axis + 1:
                data = np.moveaxis(data, offsets_axis, labels_axis + 1)
        except ValueError:
            pass

        return data


class HeaderLine(Line):
    def __getitem__(self, i: Union[Index, Tuple[Index, Index]]) -> NestedFieldList:
        if isinstance(i, tuple):
            labels, offsets = i
        else:
            labels, offsets = i, self._default_offset
        header_dicts = self._tdb[self._get_tdb_indices(labels, offsets)]
        header_keys = header_dicts.keys()
        data = np.stack(list(header_dicts.values()))
        data = np.moveaxis(data, 0, -1)
        data = self._moveaxis(data, labels, offsets)
        data = np.apply_along_axis(lambda v: dict(zip(header_keys, v)), -1, data)
        return cast(NestedFieldList, data.tolist())


class Gather:
    def __init__(
        self,
        ilines: np.ndarray,
        xlines: np.ndarray,
        offsets: np.ndarray,
        tdb: tiledb.Array,
    ):
        self._tdb = tdb
        self._iline_indexer = LabelIndexer(ilines)
        self._xline_indexer = LabelIndexer(xlines)
        self._offset_indexer = LabelIndexer(offsets)
        self._default_offset = offsets[0] if len(offsets) == 1 else slice(None)

    def __getitem__(self, t: Tuple[Index, ...]) -> np.ndarray:
        if len(t) == 3:
            ilines, xlines, offsets = t
        else:
            ilines, xlines = t
            offsets = self._default_offset

        dims = tuple(dim.name for dim in self._tdb.schema.domain)
        composite_index: List[ExtendedIndex] = [slice(None)] * 3
        composite_index[dims.index("ilines")] = self._iline_indexer[ilines]
        composite_index[dims.index("xlines")] = self._xline_indexer[xlines]
        composite_index[dims.index("offsets")] = self._offset_indexer[offsets]

        data = self._tdb[tuple(composite_index)]
        # segyio returns always (ilines, xlines); convert from (fast, slow) if necessary
        if (
            dims[0] == "xlines"
            and isinstance(ilines, slice)
            and isinstance(xlines, slice)
        ):
            data = data.swapaxes(0, 1)
        return data


class StructuredSegy(Segy):
    _indexer_cls = StructuredTraceIndexer

    @cached_property
    def iline(self) -> Line:
        return Line("ilines", self.ilines, self.offsets, self._data)

    @cached_property
    def xline(self) -> Line:
        return Line("xlines", self.xlines, self.offsets, self._data)

    @cached_property
    def fast(self) -> Line:
        if self.sorting == TraceSortingFormat.INLINE_SORTING:
            return self.iline
        assert self.sorting == TraceSortingFormat.CROSSLINE_SORTING
        return self.xline

    @cached_property
    def slow(self) -> Line:
        if self.sorting == TraceSortingFormat.INLINE_SORTING:
            return self.xline
        assert self.sorting == TraceSortingFormat.CROSSLINE_SORTING
        return self.iline

    @cached_property
    def gather(self) -> Gather:
        return Gather(self.ilines, self.xlines, self.offsets, self._data)

    @cached_property
    def offsets(self) -> np.ndarray:
        return self._meta_to_numpy("offsets", dtype="intc")

    @cached_property
    def ilines(self) -> np.ndarray:
        return self._meta_to_numpy("ilines", dtype="intc")

    @cached_property
    def xlines(self) -> np.ndarray:
        return self._meta_to_numpy("xlines", dtype="intc")

    @cached_property
    def header(self) -> Header:
        header = super().header
        for attr, name in ("iline", "ilines"), ("xline", "xlines"):
            line = HeaderLine(name, getattr(self, name), self.offsets, self._headers)
            setattr(header, attr, line)
        return header

    def cube(self) -> np.ndarray:
        if self.sorting == TraceSortingFormat.INLINE_SORTING:
            fast, slow = self.ilines, self.xlines
        else:
            fast, slow = self.xlines, self.ilines
        dims = list(map(len, (fast, slow, self.samples)))
        offsets = len(self.offsets)
        if offsets > 1:
            dims.insert(2, offsets)
        return self.trace[:].reshape(dims)
