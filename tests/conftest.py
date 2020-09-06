import itertools as it
import shutil
from pathlib import Path
from typing import Iterator, Mapping, Tuple

import pytest
import segyio
import tiledb
from _pytest.fixtures import SubRequest

from tilesegy.create import segy_to_tiledb

from .segyio_utils import generate_structured_segy, generate_unstructured_segy

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CLEANUP = False


_segy_combos = [
    (False, dict(zip(["traces", "samples"], combo)))
    for combo in it.product([6300], [10])
]
_segy_combos.extend(
    (True, dict(zip(["ilines", "xlines", "offsets", "samples", "sorting"], combo)))
    for combo in it.product(
        [70],
        [90],
        [1, 2],
        [10],
        [
            segyio.TraceSortingFormat.INLINE_SORTING,
            segyio.TraceSortingFormat.CROSSLINE_SORTING,
        ],
    )
)


def _serialize_request_param(param: Tuple[bool, Mapping[str, int]]) -> str:
    structured, kwargs = param
    value = "structured-" if structured else "unstructured-"
    value += "-".join("{}={}".format(*item) for item in kwargs.items())
    return value


@pytest.fixture(  # type: ignore
    scope="session", params=_segy_combos, ids=_serialize_request_param,
)
def segy_file(request: SubRequest) -> Iterator[segyio.SegyFile]:
    structured, kwargs = request.param
    path = FIXTURES_DIR / (_serialize_request_param(request.param) + ".sgy")
    if not path.exists():
        if structured:
            generate_structured_segy(path, **kwargs)
        else:
            generate_unstructured_segy(path, **kwargs)
    with segyio.open(path, ignore_geometry=not structured) as segy_file:
        yield segy_file
    if CLEANUP:
        path.unlink()


@pytest.fixture(scope="session")  # type: ignore
def tilesegy(segy_file: segyio.SegyFile) -> Iterator[Path]:
    path = Path(segy_file._filename).with_suffix(".tdb")
    if not path.exists():
        segy_to_tiledb(
            segy_file,
            str(path),
            tile_size=1024 ** 2,
            config=tiledb.Config({"sm.consolidation.buffer_size": 500000}),
        )
    yield path
    if CLEANUP:
        shutil.rmtree(path)
