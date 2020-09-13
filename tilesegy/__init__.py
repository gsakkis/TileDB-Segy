from pathlib import Path
from typing import Union

import tiledb

from .api import StructuredTileSegy, TileSegy


def open(uri: Union[str, Path]) -> TileSegy:
    uri = Path(uri) if not isinstance(uri, Path) else uri
    headers = tiledb.DenseArray(str(uri / "headers"))
    data = tiledb.DenseArray(str(uri / "data"))
    if data.schema.domain.has_dim("traces"):
        cls = TileSegy
    else:
        cls = StructuredTileSegy
    return cls(uri, headers, data)
