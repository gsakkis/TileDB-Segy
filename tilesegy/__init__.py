__all__ = ["open", "TileSegy", "StructuredTileSegy"]

from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Optional, Tuple, Type, Union

import numpy as np
import tiledb
from segyio import TraceSortingFormat

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


class TileSegy:
    _indexer_cls: Type[TraceIndexer] = TraceIndexer

    def __init__(self, uri: Path, headers: tiledb.Array, data: tiledb.Array):
        self._uri = uri
        self._headers = headers
        self._data = data

    @property
    def uri(self) -> Path:
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
            tiledb.DenseArray(self._headers.uri, attr=name), self._indexer_cls
        )

    @cached_property
    def depth_slice(self) -> Depth:
        return Depth(self._data)

    def close(self) -> None:
        self._headers.close()
        self._data.close()
        # remove all cached properties
        for attr in list(self.__dict__.keys()):
            if isinstance(getattr(self.__class__, attr, None), cached_property):
                delattr(self, attr)

    def __enter__(self) -> "TileSegy":
        return self

    def __exit__(
        self, type: Type[Exception], value: Exception, traceback: TracebackType
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self._uri)!r})"

    def _meta_to_numpy(
        self, meta_key: str, dtype: Union[np.dtype, str, None] = None
    ) -> np.ndarray:
        values = self._data.meta[meta_key]
        if not isinstance(values, tuple):
            values = (values,)
        return np.array(values, dtype)


class StructuredTileSegy(TileSegy):
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


def open(uri: Union[str, Path]) -> TileSegy:
    uri = Path(uri) if not isinstance(uri, Path) else uri
    headers = tiledb.DenseArray(str(uri / "headers"))
    data = tiledb.DenseArray(str(uri / "data"), attr="trace")
    if data.schema.domain.has_dim("traces"):
        cls = TileSegy
    else:
        cls = StructuredTileSegy
    return cls(uri, headers, data)
