__all__ = ["open", "Segy", "StructuredSegy"]

from pathlib import PurePath
from typing import Optional, Union

import urlpath

import tiledb

from .structured import StructuredSegy
from .unstructured import Segy

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
    return cls(data, headers)
