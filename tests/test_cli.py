from unittest import mock

import pytest

from tiledb.segy import cli


@mock.patch("segyio.open")
@mock.patch("tiledb.segy.cli.SegyFileConverter")
def test_minimal(SegyFileConverter, segyio_open, tmp_path):
    cli.main([str(tmp_path)])

    assert segyio_open.call_count == 1
    assert segyio_open.call_args[0] == ()
    assert segyio_open.call_args[1] == dict(
        filename=tmp_path, iline=189, xline=193, endian="big", strict=False
    )

    assert SegyFileConverter.call_count == 1
    converter_kwargs = SegyFileConverter.call_args[1]
    assert set(converter_kwargs.keys()) == {"tile_size", "config"}
    assert converter_kwargs["tile_size"] == 4_000_000
    assert converter_kwargs["config"]["sm.consolidation.buffer_size"] == str(5_000_000)

    to_tiledb = SegyFileConverter.return_value.to_tiledb
    assert to_tiledb.call_count == 1
    assert to_tiledb.call_args[0] == (tmp_path.with_suffix(".tsgy"),)
    assert to_tiledb.call_args[1] == {}


@mock.patch("segyio.su.open")
@mock.patch("tiledb.segy.cli.SegyFileConverter")
def test_maximal(SegyFileConverter, segyio_open, tmp_path):
    output_path = tmp_path.with_suffix(".tdb")
    output_path.mkdir()
    cli.main(
        [
            str(tmp_path),
            str(output_path),
            "--overwrite",
            "--geometry=unstructured",
            "--su",
            "--iline=42",
            "--xline=67",
            "--endian=little",
            "--tile-size=1234567",
            "--consolidation-buffersize=987654321",
        ]
    )

    assert segyio_open.call_count == 1
    assert segyio_open.call_args[0] == ()
    assert segyio_open.call_args[1] == dict(
        filename=tmp_path, iline=42, xline=67, endian="little", ignore_geometry=True
    )

    assert SegyFileConverter.call_count == 1
    converter_kwargs = SegyFileConverter.call_args[1]
    assert set(converter_kwargs.keys()) == {"tile_size", "config"}
    assert converter_kwargs["tile_size"] == 1234567
    assert converter_kwargs["config"]["sm.consolidation.buffer_size"] == "987654321"

    to_tiledb = SegyFileConverter.return_value.to_tiledb
    assert to_tiledb.call_count == 1
    assert to_tiledb.call_args[0] == (output_path,)
    assert to_tiledb.call_args[1] == {}


def test_output_exists_error(tmp_path):
    tmp_path.with_suffix(".tsgy").mkdir()
    with pytest.raises(SystemExit):
        cli.main([str(tmp_path)])

    output_path = tmp_path.with_suffix(".tdb")
    output_path.mkdir()
    with pytest.raises(SystemExit):
        cli.main([str(tmp_path), str(output_path)])
