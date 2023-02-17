import os
from functools import singledispatchmethod
from pathlib import PurePath
from types import TracebackType
from typing import List, Optional, Tuple, Type, Union, cast, overload

import numpy as np
from segyio import TraceSortingFormat

import tiledb

from .tdbwrapper import MultiAttrArrayWrapper, SingleAttrArrayWrapper
from .types import (
    Ellipsis,
    ExtendedIndex,
    ExtendedIndices,
    Field,
    Index,
    cached_property,
)


def ensure_slice(i: Index) -> slice:
    return i if isinstance(i, slice) else slice(i, i + 1)


class TraceIndexer:
    def __init__(self, shape: Tuple[int, ...]):
        self._shape = shape

    def __len__(self) -> int:
        return int(np.asarray(self._shape).prod())

    def __getitem__(self, trace_index: Index) -> Tuple[ExtendedIndices, ExtendedIndex]:
        """
        Given a trace index, return a `(bounding_box, post_reshape_indices)` tuple where:
        - `bounding_box` is a tuple of (int or slice) indices for each dimension in shape
            that enclose all data of the requested `trace_index`.
        - `post_reshape_indices` is a list of indices to select from the reshaped 1D
            bounding box in order to get the requested `trace_index` data. It may also
            be ellipsis (...) if the whole bounding box is to be selected.
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
    def _get_index(self, i: Index) -> Union[Field, List[Field]]:
        raise TypeError(f"Cannot index by {i.__class__}")

    @overload
    @_get_index.register
    def __getitem__(self, i: int) -> Field:
        return self[i : i + 1][0]

    @overload
    @_get_index.register
    def __getitem__(self, i: slice) -> List[Field]:
        bounding_box, post_reshape_indices = self._indexer[i]
        header_arrays = self._tdb[bounding_box]
        keys = header_arrays.keys()
        columns = [header_arrays[key].reshape(-1)[post_reshape_indices] for key in keys]
        return [dict(zip(keys, row)) for row in zip(*columns)]

    def __getitem__(self, i: Index) -> Union[Field, List[Field]]:
        return self._get_index(i)


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


class Segy:
    _indexer_cls: Type[TraceIndexer] = TraceIndexer

    def __init__(
        self,
        data: tiledb.Array,
        headers: tiledb.Array,
        uri: Optional[PurePath] = None,
    ):
        self._data = SingleAttrArrayWrapper(data, attr="trace")
        self._headers = MultiAttrArrayWrapper(headers)
        self._uri = uri or PurePath(os.devnull)

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
        return Attributes(
            SingleAttrArrayWrapper(self._headers.__wrapped__, attr=name),
            self._indexer_cls(self._headers.shape),
        )

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
