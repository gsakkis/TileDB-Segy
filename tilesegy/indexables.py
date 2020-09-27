from typing import Dict, List, Tuple, Type, Union, cast

import numpy as np
import tiledb

from .singledispatchmethod import singledispatchmethod  # type: ignore
from .utils import Index, Int, LabelIndexer, TraceIndexer, ensure_slice

Field = Dict[str, int]


class TraceIndexable:
    def __init__(self, tdb: tiledb.Array, indexer: TraceIndexer):
        self._tdb = tdb
        self._indexer = indexer

    def __len__(self) -> int:
        return len(self._indexer)


class Trace(TraceIndexable):
    def __init__(self, tdb: tiledb.Array, indexer_cls: Type[TraceIndexer]):
        super().__init__(tdb, indexer_cls(tdb.shape[:-1]))

    def __getitem__(self, i: Union[Index, Tuple[Index, Index]]) -> np.ndarray:
        if isinstance(i, tuple):
            # for single sample segyio returns an array of size 1 instead of scalar
            trace_index, samples = i[0], ensure_slice(i[1])
        else:
            trace_index, samples = i, slice(None)

        bounding_box, post_reshape_indices = self._indexer[trace_index]
        traces = self._tdb[(*bounding_box, samples)]
        if len(traces.shape) > 2:
            traces = traces.reshape(np.array(traces.shape[:-1]).prod(), -1)
        return traces[post_reshape_indices]


class Header(TraceIndexable):
    def __init__(self, tdb: tiledb.Array, indexer_cls: Type[TraceIndexer]):
        super().__init__(tdb, indexer_cls(tdb.shape))

    @singledispatchmethod
    def __getitem__(self, i: object) -> None:
        raise NotImplementedError(f"Cannot index by {i.__class__}")  # pragma: nocover

    @__getitem__.register(int)
    @__getitem__.register(np.integer)
    def _get_one(self, i: Int) -> Field:
        return cast(Field, self[ensure_slice(i)][0])

    @__getitem__.register(slice)
    def _get_many(self, i: slice) -> List[Field]:
        bounding_box, post_reshape_indices = self._indexer[i]
        header_arrays = self._tdb[bounding_box]
        keys = header_arrays.keys()
        columns = [header_arrays[key].reshape(-1)[post_reshape_indices] for key in keys]
        return [dict(zip(keys, row)) for row in zip(*columns)]


class Attributes(TraceIndexable):
    def __init__(self, tdb: tiledb.Array, indexer_cls: Type[TraceIndexer]):
        super().__init__(tdb, indexer_cls(tdb.shape))

    def __getitem__(self, i: Index) -> np.ndarray:
        bounding_box, post_reshape_indices = self._indexer[ensure_slice(i)]
        return self._tdb[bounding_box].reshape(-1)[post_reshape_indices]


class Line:
    def __init__(
        self,
        dim_name: str,
        labels: np.ndarray,
        offsets: np.ndarray,
        tdb: tiledb.Array,
    ):
        self._tdb = tdb
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
        data = self._tdb[self._get_tdb_indices(labels, offsets)]
        data = self._moveaxis(data, labels, offsets)
        return data

    _dims = property(lambda self: [dim.name for dim in self._tdb.schema.domain])

    def _get_tdb_indices(self, labels: Index, offsets: Index) -> Tuple[Index, ...]:
        dims = self._dims
        composite_index: List[Index] = [slice(None)] * self._tdb.ndim
        composite_index[dims.index(self._dim_name)] = self._label_indexer[labels]
        composite_index[dims.index("offsets")] = self._offset_indexer[offsets]
        return tuple(composite_index)

    def _moveaxis(self, data: np.ndarray, labels: Index, offsets: Index) -> np.ndarray:
        labels_dim = self._dim_name
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


NestedFieldList = Union[List[Field], List[List[Field]], List[List[List[Field]]]]


class HeaderLine(Line):
    def __getitem__(self, i: Union[Index, Tuple[Index, Index]]) -> NestedFieldList:
        if isinstance(i, tuple):
            labels, offsets = i
        else:
            labels, offsets = i, self._default_offset
        header_dicts = self._tdb[self._get_tdb_indices(labels, offsets)]
        header_keys = header_dicts.keys()
        data = np.stack(header_dicts.values())
        data = np.moveaxis(data, 0, -1)
        data = self._moveaxis(data, labels, offsets)
        data = np.apply_along_axis(lambda v: dict(zip(header_keys, v)), -1, data)
        return cast(NestedFieldList, data.tolist())


class Depth:
    def __init__(self, tdb: tiledb.Array):
        self._tdb = tdb

    def __len__(self) -> int:
        return cast(int, self._tdb.shape[-1])

    def __getitem__(self, i: Index) -> np.ndarray:
        ndim = self._tdb.ndim
        if ndim > 2:
            # segyio doesn't currently support offset indexing for depth; always selects the first
            # https://github.com/equinor/segyio/issues/474
            data = self._tdb[..., 0, i]
            ndim -= 1
        else:
            data = self._tdb[..., i]
        # move samples as first dimension
        return np.moveaxis(data, -1, 0) if data.ndim == ndim else data
