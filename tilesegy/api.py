from pathlib import Path
from types import TracebackType
from typing import Dict, List, Type

import numpy as np
import tiledb

from .indexables import Traces


class TileSegy:
    def __init__(self, uri: Path, headers: tiledb.Array, data: tiledb.Array):
        self._uri = uri
        self._headers = headers
        self._data = data

    @property
    def uri(self) -> Path:
        return self._uri

    @property
    def bin(self) -> Dict[str, int]:
        bin_headers = dict(self._headers.meta.items())
        del bin_headers["__text__"]
        return bin_headers

    @property
    def text(self) -> List[bytes]:
        text_headers = self._headers.meta["__text__"]
        assert len(text_headers) % 3200 == 0, len(text_headers)
        return [text_headers[i : i + 3200] for i in range(0, len(text_headers), 3200)]

    @property
    def samples(self) -> np.ndarray:
        return np.asarray(self._data.meta["samples"])

    @property
    def traces(self) -> Traces:
        return Traces(self._data, self._headers)

    def close(self) -> None:
        self._headers.close()
        self._data.close()

    def __enter__(self) -> "TileSegy":
        return self

    def __exit__(
        self, type: Type[Exception], value: Exception, traceback: TracebackType
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self._uri)!r})"
