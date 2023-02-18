__all__ = ["open", "Segy", "StructuredSegy"]

from pathlib import PurePath
from typing import Union

import urlpath

import tiledb

from .structured import StructuredSegy
from .unstructured import Segy

URI = Union[str, PurePath]


def open(uri: URI) -> Segy:
    uri = urlpath.URL(uri) if not isinstance(uri, PurePath) else uri
    ts = open2(uri / "data", uri / "headers")
    ts._uri = uri
    return ts


def open2(data_uri: URI, headers_uri: URI) -> Segy:
    data = tiledb.open(str(data_uri), attr="trace")
    headers = tiledb.open(str(headers_uri))
    if data.schema.domain.has_dim("traces"):
        cls = Segy
    else:
        cls = StructuredSegy
    return cls(data, headers)
