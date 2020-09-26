from typing import Dict, List, Tuple, Type, Union, cast

import numpy as np
import tiledb

from ._singledispatchmethod import singledispatchmethod  # type: ignore
from .utils import Index, Int, LabelIndexer, TraceIndexer, ensure_slice


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
    def _get_one(self, i: Int) -> Dict[str, int]:
        return cast(Dict[str, int], self[ensure_slice(i)][0])

    @__getitem__.register(slice)
    def _get_many(self, i: slice) -> List[Dict[str, int]]:
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

    _dims = property(lambda self: [dim.name for dim in self._tdb.schema.domain])


class Depth:
    def __init__(self, tdb: tiledb.Array):
        self._tdb = tdb

    def __len__(self) -> int:
        return cast(int, self._tdb.shape[-1])

    def __getitem__(self, i: Index) -> np.ndarray:
        data = self._tdb[:, i]
        # (traces, samples) -> (samples, traces)
        return np.moveaxis(data, 1, 0) if data.ndim == 2 else data


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

        # (fast, slow, samples) -> (samples, fast, slow)
        return np.moveaxis(data, 2, 0) if data.ndim == 3 else data
