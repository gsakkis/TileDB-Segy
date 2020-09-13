from pathlib import Path
from typing import Union

import tiledb

from .api import TileSegy


def open(uri: Union[str, Path]) -> TileSegy:
    if not isinstance(uri, Path):
        uri = Path(uri)
    headers = tiledb.DenseArray(str(uri / "headers"))
    data = tiledb.DenseArray(str(uri / "data"))
    return TileSegy(uri, headers, data)
