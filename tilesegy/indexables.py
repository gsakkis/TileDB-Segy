from typing import Dict, List, Tuple, Union

import numpy as np
import tiledb

from ._singledispatchmethod import singledispatchmethod  # type: ignore

Index = Union[int, slice]


class Sized:
    def __init__(self, tdb: tiledb.Array):
        self._tdb = tdb

    def __len__(self) -> int:
        return len(self._tdb)


class Header(Sized):
    @singledispatchmethod
    def __getitem__(self, i: object) -> None:
        raise NotImplementedError(f"Cannot index by {i.__class__}")  # pragma: nocover

    @__getitem__.register(int)
    def _get_one(self, i: int) -> int:
        return np.asscalar(self._tdb[i])  # type: ignore

    @__getitem__.register(slice)
    def _get_many(self, i: slice) -> List[int]:
        return self._tdb[i].tolist()  # type: ignore


class Headers(Sized):
    @singledispatchmethod
    def __getitem__(self, i: object) -> None:
        raise NotImplementedError(f"Cannot index by {i.__class__}")  # pragma: nocover

    @__getitem__.register(int)
    def _get_one(self, i: int) -> Dict[str, int]:
        return self._tdb[i]  # type: ignore

    @__getitem__.register(slice)
    def _get_many(self, i: slice) -> List[Dict[str, int]]:
        headers = self._tdb[i]
        keys = headers.keys()
        columns = [v.tolist() for v in headers.values()]
        return [dict(zip(keys, row)) for row in zip(*columns)]


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
