__all__ = ["open", "Segy", "StructuredSegy"]

import os
from pathlib import PurePath
from types import TracebackType
from typing import TYPE_CHECKING, Optional, Tuple, Type, Union

import numpy as np
import urlpath
from segyio import TraceSortingFormat

import tiledb

if TYPE_CHECKING:  # pragma: nocover
    cached_property = property
else:
    from cached_property import cached_property

from .indexables import (
    Attributes,
    Depth,
    Field,
    Gather,
    Header,
    HeaderLine,
    Line,
    Trace,
)
from .utils import StructuredTraceIndexer, TraceIndexer


class Segy:
    _indexer_cls: Type[TraceIndexer] = TraceIndexer

    def __init__(
        self,
        data: tiledb.Array,
        headers: tiledb.Array,
        uri: Optional[PurePath] = None,
        ctx: Optional[tiledb.Ctx] = None,
    ):
        self._data = data
        self._headers = headers
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
        return Trace(self._data, self._indexer_cls)

    @cached_property
    def header(self) -> Header:
        return Header(self._headers, self._indexer_cls)

    def attributes(self, name: str) -> Attributes:
        return Attributes(
            tiledb.open(self._headers.uri, attr=name, ctx=self._ctx), self._indexer_cls
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


URI = Union[str, PurePath]


def open(uri: URI, config: Optional[tiledb.Config] = None) -> Segy:
    uri = urlpath.URL(uri) if not isinstance(uri, PurePath) else uri
    ts = open2(uri / "data", uri / "headers", config)
    ts._uri = uri
    return ts


def open2(
    data_uri: URI, headers_uri: URI, config: Optional[tiledb.Config] = None
) -> Segy:
    ctx = tiledb.Ctx(config)
    data = tiledb.open(str(data_uri), attr="trace", ctx=ctx)
    headers = tiledb.open(str(headers_uri), ctx=ctx)
    if data.schema.domain.has_dim("traces"):
        cls = Segy
    else:
        cls = StructuredSegy
    return cls(data, headers, ctx=ctx)
