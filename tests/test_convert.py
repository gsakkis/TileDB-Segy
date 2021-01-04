import pytest
import segyio
from segyio import TraceSortingFormat

from tiledb.segy.convert import SegyFileConverter

from .segyio_utils import generate_structured_segy, generate_unstructured_segy


def test_convert_unstructured_segy(tmp_path):
    segy_path = tmp_path / "input.segy"
    generate_unstructured_segy(segy_path, traces=10, samples=10)
    tsgy_path = segy_path.with_suffix(".tsgy")
    with segyio.open(segy_path, ignore_geometry=True) as segy_file:
        SegyFileConverter(segy_file, tile_size=1000).to_tiledb(tsgy_path)


@pytest.mark.parametrize(
    "sorting", [TraceSortingFormat.INLINE_SORTING, TraceSortingFormat.CROSSLINE_SORTING]
)
def test_convert_structured_segy(sorting, tmp_path):
    segy_path = tmp_path / "input.segy"
    generate_structured_segy(
        segy_path,
        ilines=20,
        xlines=30,
        offsets=1,
        samples=10,
        sorting=sorting,
    )
    tsgy_path = segy_path.with_suffix(".tsgy")
    with segyio.open(segy_path) as segy_file:
        SegyFileConverter(segy_file, tile_size=1000).to_tiledb(tsgy_path)
