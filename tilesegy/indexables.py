from typing import Dict, List, Tuple, Union

import numpy as np
import tiledb

Index = Union[int, slice]


class Sized:
    def __init__(self, tdb: tiledb.Array):
        self._tdb = tdb

    def __len__(self) -> int:
        return len(self._tdb)


class Header(Sized):
    def __getitem__(self, i: Index) -> Union[int, List[int]]:
        values = self._tdb[i]
        return np.asscalar(values) if isinstance(i, int) else values.tolist()  # type: ignore


class Headers(Sized):
    def __getitem__(self, i: Index) -> Union[Dict[str, int], List[Dict[str, int]]]:
        headers = self._tdb[i]
        if not isinstance(i, int):
            keys = headers.keys()
            columns = [v.tolist() for v in headers.values()]
            headers = [dict(zip(keys, row)) for row in zip(*columns)]
        return headers  # type: ignore


class Traces(Sized):
    def __init__(self, traces_tdb: tiledb.Array, headers_tdb: tiledb.Array):
        super().__init__(traces_tdb)
        self._headers_tdb = headers_tdb

    def __getitem__(
        self, i: Union[Index, Tuple[Index, Index]]
    ) -> Union[np.number, np.ndarray]:
        return self._tdb[i]

    @property
    def headers(self) -> Headers:
        return Headers(self._headers_tdb)

    def header(self, name: str) -> Header:
        return Header(tiledb.DenseArray(self._headers_tdb.uri, attr=name))
