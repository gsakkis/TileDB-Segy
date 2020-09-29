import itertools as it
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import pytest
import segyio
from segyio import SegyFile, TraceSortingFormat

import tilesegy
from tilesegy import TileSegy, cli

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


def iter_segyfiles(
    structured: Optional[bool] = None, multiple_offsets: Optional[bool] = None
) -> Iterator[SegyFile]:
    if structured is None:
        yield from iter_segyfiles(False, multiple_offsets)
        yield from iter_segyfiles(True, multiple_offsets)
        return

    generate_segy: Callable[..., None]
    if structured:
        combos = STRUCTURED_SEGY_COMBOS
        generate_segy = generate_structured_segy
    else:
        combos = UNSTRUCTURED_SEGY_COMBOS
        generate_segy = generate_unstructured_segy
    keys = combos.keys()
    for values in it.product(*combos.values()):
        kwargs = dict(zip(keys, values))
        if (
            structured
            and multiple_offsets is not None
            and bool(multiple_offsets) == (kwargs["offsets"] < 2)
        ):
            continue
        filename = "-".join("{}={}".format(*item) for item in kwargs.items()) + ".sgy"
        path = FIXTURES_DIR / filename
        if not path.exists():
            generate_segy(path, **kwargs)
        yield segyio.open(path, ignore_geometry=not structured)


def get_tilesegy(segy_file: SegyFile) -> TileSegy:
    inpath = Path(segy_file._filename)
    outpath = inpath.with_suffix(".tsgy")
    if not outpath.exists():
        cli.main(list(map(str, [inpath, outpath])))
    return tilesegy.open(outpath)


def parametrize_tilesegy_segyfiles(
    tilesegy_name: str,
    segyfile_name: str,
    structured: Optional[bool] = None,
    multiple_offsets: Optional[bool] = None,
) -> Any:
    return pytest.mark.parametrize(
        (tilesegy_name, segyfile_name),
        (
            (get_tilesegy(segy_file), segy_file)
            for segy_file in iter_segyfiles(structured, multiple_offsets)
        ),
        ids=lambda x: Path(x.uri).stem if isinstance(x, TileSegy) else None,
    )
