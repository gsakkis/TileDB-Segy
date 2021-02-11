import itertools as it
from collections import abc
from functools import singledispatch
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pytest
import segyio
from filelock import FileLock
from segyio import SegyFile, TraceSortingFormat

import tiledb.segy
from tiledb.segy import Segy, cli

from .segyio_utils import generate_structured_segy, generate_unstructured_segy

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)

UNSTRUCTURED_SEGY_COMBOS = {
    "sorting": [TraceSortingFormat.UNKNOWN_SORTING],
    "traces": [6300],
    "samples": [10],
}
STRUCTURED_SEGY_COMBOS = {
    "sorting": [
        TraceSortingFormat.CROSSLINE_SORTING,
        TraceSortingFormat.INLINE_SORTING,
    ],
    "ilines": [28],
    "xlines": [90],
    "offsets": [1, 5],
    "samples": [10],
}


def iter_tsgy_sgy_files(
    structured: Optional[bool] = None, multiple_offsets: Optional[bool] = None
) -> Iterator[Tuple[Segy, SegyFile]]:
    if structured is None:
        yield from iter_tsgy_sgy_files(False, multiple_offsets)
        yield from iter_tsgy_sgy_files(True, multiple_offsets)
        return

    generate_segy: Callable[..., None]
    if structured:
        combos = STRUCTURED_SEGY_COMBOS
        generate_segy = generate_structured_segy
    else:
        combos = UNSTRUCTURED_SEGY_COMBOS
        generate_segy = generate_unstructured_segy

    for values in it.product(*combos.values()):
        kwargs = dict(zip(combos.keys(), values))
        if (
            structured
            and multiple_offsets is not None
            and bool(multiple_offsets) == (kwargs["offsets"] < 2)
        ):
            continue

        basename = "-".join("{}={}".format(*item) for item in kwargs.items())
        sgy_path = FIXTURES_DIR / (basename + ".sgy")
        tsgy_path = FIXTURES_DIR / (basename + ".tsgy")

        with FileLock(str(FIXTURES_DIR / (basename + ".lock"))):
            if not sgy_path.exists():
                generate_segy(sgy_path, **kwargs)
            if not tsgy_path.exists():
                cli.main(list(map(str, [sgy_path, tsgy_path])))

        yield tiledb.segy.open(tsgy_path), segyio.open(sgy_path, strict=False)


def parametrize_segys(
    tiledb_segy_name: str,
    segyio_file_name: str,
    structured: Optional[bool] = None,
    multiple_offsets: Optional[bool] = None,
) -> Any:
    return pytest.mark.parametrize(
        (tiledb_segy_name, segyio_file_name),
        iter_tsgy_sgy_files(structured, multiple_offsets),
        ids=lambda x: x.uri.stem if isinstance(x, Segy) else None,
    )


def assert_equal_arrays(
    a: Union[np.ndarray, np.number],
    b: Union[np.ndarray, np.number],
    reshape: bool = False,
) -> None:
    assert a.dtype == b.dtype
    if isinstance(a, np.number) or isinstance(b, np.number):
        assert isinstance(a, np.number) and isinstance(b, np.number)
        assert a == b
    elif reshape:
        assert a.ndim == b.ndim + 1
        assert a.shape[2:] == b.shape[1:]
        b = b.reshape(a.shape)
    else:
        assert a.ndim == b.ndim
        assert a.shape == b.shape
    np.testing.assert_array_equal(a, b)


def iter_slices(i: int, j: int) -> Iterable[slice]:
    assert 0 <= i < j
    slice_args = it.chain(
        # non-negative start stop step
        it.product((None, i), (None, j), (None, 2)),
        # non-negative start stop, negative step
        it.product((None, j - 1), (None, i - 1) if i > 0 else (None,), (-1, -2)),
        # negative start and/or stop
        [(-1, None, None), (None, -1, None), (-2, -1, None), (-1, -2, -1)],
    )
    return it.starmap(slice, slice_args)


@singledispatch
def stringify_keys(o: object) -> Any:
    raise TypeError(f"Cannot stringify_keys for {o.__class__}")


@stringify_keys.register(abc.Mapping)
def _stringify_keys_mapping(d: Mapping[int, int]) -> Mapping[str, int]:
    return {str(k): v for k, v in d.items()}


@stringify_keys.register(abc.Iterable)
def _stringify_keys_iter(s: Iterable[Mapping[int, int]]) -> List[Mapping[str, int]]:
    return list(map(stringify_keys, s))
