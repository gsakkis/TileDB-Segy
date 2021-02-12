import os
from pathlib import PurePath
from types import TracebackType
from typing import List, Optional, Tuple, Type, Union, cast

import numpy as np
import wrapt
from segyio import TraceSortingFormat

import tiledb

from .singledispatchmethod import singledispatchmethod  # type: ignore
from .types import Ellipsis, Field, Index, cached_property, ellipsis
from .utils import ensure_slice


class TraceIndexer:
    def __init__(self, shape: Tuple[int, ...]):
        self._shape = shape

    def __len__(self) -> int:
        return int(np.asarray(self._shape).prod())

    def __getitem__(
        self, trace_index: Index
    ) -> Tuple[Tuple[Index, ...], Union[int, List[int], ellipsis]]:
        """
        Given a trace index, return a `(bounding_box, post_reshape_indices)` tuple where:
        - `bounding_box` is a tuple of (int or slice) indices for each dimension in shape that
          enclose all data of the requested `trace_index`.
        - `post_reshape_indices` is a list of indices to select from the reshaped 1-dimensional
          bounding box in order to get the requested `trace_index` data. It may also be ellipsis
          (...) if the whole bounding box is to be selected.
        """
        return (trace_index,), Ellipsis


class TraceIndexable:
    def __init__(self, tdb: tiledb.Array, indexer: TraceIndexer):
        self._tdb = tdb
        self._indexer = indexer

    def __len__(self) -> int:
        return len(self._indexer)


class Trace(TraceIndexable):
    def __getitem__(self, i: Union[Index, Tuple[Index, Index]]) -> np.ndarray:
        if isinstance(i, tuple):
            trace_index, samples = i
        else:
            trace_index, samples = i, slice(None)

        bounding_box, post_reshape_indices = self._indexer[trace_index]
        traces = self._tdb[(*bounding_box, ensure_slice(samples))]

        if not (isinstance(trace_index, slice) or isinstance(samples, slice)):
            # convert to scalar (https://github.com/equinor/segyio/issues/475)
            return traces[0]

        if traces.ndim > 2:
            traces = traces.reshape(np.array(traces.shape[:-1]).prod(), -1)
        return traces[post_reshape_indices]


class Header(TraceIndexable):
    @singledispatchmethod
    def __getitem__(self, i: object) -> None:
        raise TypeError(f"Cannot index by {i.__class__}")

    @__getitem__.register(int)
    @__getitem__.register(np.integer)
    def _get_one(self, i: int) -> Field:
        return cast(Field, self[ensure_slice(i)][0])

    @__getitem__.register(slice)
    def _get_many(self, i: slice) -> List[Field]:
        bounding_box, post_reshape_indices = self._indexer[i]
        header_arrays = self._tdb[bounding_box]
        keys = header_arrays.keys()
        columns = [header_arrays[key].reshape(-1)[post_reshape_indices] for key in keys]
        return [dict(zip(keys, row)) for row in zip(*columns)]


class Attributes(TraceIndexable):
    def __getitem__(self, i: Index) -> np.ndarray:
        bounding_box, post_reshape_indices = self._indexer[ensure_slice(i)]
        return self._tdb[bounding_box].reshape(-1)[post_reshape_indices]


class Depth:
    def __init__(self, tdb: tiledb.Array):
        self._tdb = tdb

    def __len__(self) -> int:
        return cast(int, self._tdb.shape[-1])

    def __getitem__(self, i: Index) -> np.ndarray:
        if not isinstance(i, (int, slice)):
            raise TypeError(
                f"depth indices must be integers or slices, not {i.__class__.__name__}"
            )
        ndim = self._tdb.ndim
        if ndim > 2:
            # segyio currently selects the first offset for depth slicing
            # https://github.com/equinor/segyio/issues/474
            data = self._tdb[..., 0, i]
            ndim -= 1
        else:
            data = self._tdb[..., i]
        # move samples as first dimension
        return np.moveaxis(data, -1, 0) if data.ndim == ndim else data


class TiledbArrayWrapper(wrapt.ObjectProxy):
    """
    TileDB array wrapper that provides standard python/numpy semantics for
    indexing slices with negative step.
    """

    def __getitem__(self, i: Union[ellipsis, Index, Tuple[Index, ...]]) -> np.ndarray:
        return self.__wrapped__[self._normalize_index(i, self.shape[0])]

    @singledispatchmethod
    def _normalize_index(
        self, i: Union[int, ellipsis], size: int
    ) -> Union[int, ellipsis]:
        return i if i is Ellipsis or i >= 0 else size + i

    @_normalize_index.register(slice)
    def _normalize_slice(self, s: slice, size: int) -> slice:
        start, stop, step = s.indices(size)
        if step < 0:
            start, stop = stop + 1, start + 1
        return slice(start, stop, step)

    @_normalize_index.register(tuple)
    def _normalize_tuple(self, t: Tuple[Index, ...], size: int) -> Tuple[Index, ...]:
        has_ellipsis = t.count(Ellipsis)
        if has_ellipsis > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        if has_ellipsis and len(self.shape) > len(t):
            ellipsis_index = t.index(Ellipsis)
            shape = list(self.shape)
            del shape[ellipsis_index : ellipsis_index + len(shape) - len(t)]
        else:
            shape = self.shape
        return tuple(map(self._normalize_index, t, shape))


class Segy:
    _indexer_cls: Type[TraceIndexer] = TraceIndexer

    def __init__(
        self,
        data: tiledb.Array,
        headers: tiledb.Array,
        uri: Optional[PurePath] = None,
        ctx: Optional[tiledb.Ctx] = None,
    ):
        self._data = TiledbArrayWrapper(data)
        self._headers = TiledbArrayWrapper(headers)
        self._uri = uri or PurePath(os.devnull)
        self._ctx = ctx

    @property
    def uri(self) -> PurePath:
        return self._uri

    @cached_property
    def sorting(self) -> Optional[TraceSortingFormat]:
        sorting = TraceSortingFormat(self._data.meta["sorting"])
        return sorting if sorting != TraceSortingFormat.UNKNOWN_SORTING else None

    @cached_property
    def bin(self) -> Field:
        bin_headers = dict(self._headers.meta.items())
        del bin_headers["__text__"]
        return bin_headers

    @cached_property
    def text(self) -> Tuple[bytes, ...]:
        text_headers = self._headers.meta["__text__"]
        assert len(text_headers) % 3200 == 0, len(text_headers)
        return tuple(
            text_headers[i : i + 3200] for i in range(0, len(text_headers), 3200)
        )

    @cached_property
    def samples(self) -> np.ndarray:
        return self._meta_to_numpy("samples")

    @cached_property
    def trace(self) -> Trace:
        return Trace(self._data, self._indexer_cls(self._data.shape[:-1]))

    @cached_property
    def header(self) -> Header:
        return Header(self._headers, self._indexer_cls(self._headers.shape))

    def attributes(self, name: str) -> Attributes:
        tdb = tiledb.open(self._headers.uri, attr=name, ctx=self._ctx)
        return Attributes(TiledbArrayWrapper(tdb), self._indexer_cls(tdb.shape))

    @cached_property
    def depth_slice(self) -> Depth:
        return Depth(self._data)

    def dt(self, fallback: float = 4000.0) -> float:
        return self._data.meta["dt"] or fallback

    def close(self) -> None:
        self._headers.close()
        self._data.close()
        # remove all cached properties
        for attr in list(self.__dict__.keys()):
            if isinstance(getattr(self.__class__, attr, None), cached_property):
                delattr(self, attr)

    def __enter__(self) -> "Segy":
        return self

    def __exit__(
        self, type: Type[Exception], value: Exception, traceback: TracebackType
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self._uri}')"

    def _meta_to_numpy(
        self, meta_key: str, dtype: Union[np.dtype, str, None] = None
    ) -> np.ndarray:
        values = self._data.meta[meta_key]
        if not isinstance(values, tuple):
            values = (values,)
        return np.array(values, dtype)
