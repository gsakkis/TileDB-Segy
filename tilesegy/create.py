import logging
import os
from collections import namedtuple
from typing import Collection, Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import tiledb
from segyio import SegyFile, TraceField
from segyio.field import Field

Number = Union[int, float, np.number]
TypedTraceField = namedtuple("TypedTraceField", ["name", "enum", "dtype"])


def iter_typed_trace_fields(
    exclude: Collection[Union[TraceField, str]] = ()
) -> Iterator[TypedTraceField]:
    all_fields = TraceField.enums()
    include_names = set(map(str, Field._tr_keys))
    for f in exclude:
        include_names.remove(str(f))
    size2dtype = {2: np.dtype(np.int16), 4: np.dtype(np.int32)}
    for f, f2 in zip(all_fields, all_fields[1:]):
        name = str(f)
        if name in include_names:
            yield TypedTraceField(name, f, size2dtype[int(f2) - int(f)])


TRACE_FIELDS = tuple(iter_typed_trace_fields())
TRACE_FIELD_ENUMS = tuple(f.enum for f in TRACE_FIELDS)
TRACE_FIELD_NAMES = tuple(f.name for f in TRACE_FIELDS)
TRACE_FIELD_DTYPES = tuple(f.dtype for f in TRACE_FIELDS)
TRACE_FIELDS_SIZE = sum(dtype.itemsize for dtype in TRACE_FIELD_DTYPES)
TRACE_FIELD_FILTERS = (
    tiledb.BitWidthReductionFilter(),
    tiledb.ByteShuffleFilter(),
    tiledb.LZ4Filter(),
)

logger = logging.getLogger(__name__)


def create(uri: str, segy_file: SegyFile, tile_size: int) -> None:
    if tiledb.object_type(uri) != "group":
        tiledb.group_create(uri)

    headers_uri = os.path.join(uri, "headers")
    if tiledb.object_type(headers_uri) != "array":
        _create_headers_array(headers_uri, segy_file, tile_size)

    data_uri = os.path.join(uri, "data")
    if tiledb.object_type(data_uri) != "array":
        _create_data_array(data_uri, segy_file, tile_size)


def _create_headers_array(uri: str, segy_file: SegyFile, tile_size: int) -> None:
    schema = _get_headers_schema(segy_file, tile_size)
    logger.info(f"header schema: {schema}")
    tiledb.DenseArray.create(uri, schema)
    with tiledb.DenseArray(uri, mode="w") as tdb:
        _fill_headers(tdb, segy_file, tile_size)
    tiledb.consolidate(uri)
    tiledb.vacuum(uri)


def _create_data_array(uri: str, segy_file: SegyFile, tile_size: int) -> None:
    schema = _get_data_schema(segy_file, tile_size)
    logger.info(f"data schema: {schema}")
    tiledb.DenseArray.create(uri, schema)
    with tiledb.DenseArray(uri, mode="w") as tdb:
        _fill_data(tdb, segy_file, tile_size)
    tiledb.consolidate(uri)
    tiledb.vacuum(uri)


def _get_headers_schema(segy_file: SegyFile, tile_size: int) -> tiledb.ArraySchema:
    if segy_file.unstructured:
        dims = _get_unstructured_header_dims(segy_file, tile_size)
    else:
        dims = _get_structured_header_dims(segy_file, tile_size)
    return tiledb.ArraySchema(
        domain=tiledb.Domain(*dims),
        attrs=[
            tiledb.Attr(f.name, f.dtype, filters=TRACE_FIELD_FILTERS)
            for f in TRACE_FIELDS
        ],
    )


def _get_data_schema(segy_file: SegyFile, tile_size: int) -> tiledb.ArraySchema:
    if segy_file.unstructured:
        dims = _get_unstructured_data_dims(segy_file, tile_size)
    else:
        dims = _get_structured_data_dims(segy_file, tile_size)
    return tiledb.ArraySchema(
        domain=tiledb.Domain(*dims), attrs=[tiledb.Attr(dtype=segy_file.dtype)]
    )


def _get_unstructured_header_dims(
    segy_file: SegyFile, tile_size: int
) -> Sequence[tiledb.Dim]:
    domain = (0, segy_file.tracecount - 1)
    return [
        tiledb.Dim(
            name="traces",
            domain=domain,
            dtype=np.uint64,
            tile=np.clip(tile_size // TRACE_FIELDS_SIZE, 1, segy_file.tracecount),
        ),
    ]


def _get_structured_header_dims(
    segy_file: SegyFile, tile_size: int
) -> Sequence[tiledb.Dim]:
    ilines, xlines, offsets = map(
        len, (segy_file.ilines, segy_file.xlines, segy_file.offsets)
    )
    dtype = np.uintc

    if segy_file.fast is segy_file.iline:
        iline_tile = np.clip(tile_size // (xlines * TRACE_FIELDS_SIZE), 1, ilines)
        xline_tile = xlines
    else:
        iline_tile = ilines
        xline_tile = np.clip(tile_size // (ilines * TRACE_FIELDS_SIZE), 1, xlines)

    dims = [
        tiledb.Dim(
            name="ilines", domain=(0, ilines - 1), dtype=dtype, tile=iline_tile,
        ),
        tiledb.Dim(
            name="xlines", domain=(0, xlines - 1), dtype=dtype, tile=xline_tile,
        ),
    ]
    if offsets > 1:
        dims.append(
            tiledb.Dim(name="offsets", domain=(0, offsets - 1), dtype=dtype, tile=1)
        )
    return dims


def _get_unstructured_data_dims(
    segy_file: SegyFile, tile_size: int
) -> Sequence[tiledb.Dim]:
    samples = len(segy_file.samples)
    cell_size = segy_file.dtype.itemsize
    dtype = np.uint64
    return [
        tiledb.Dim(
            name="traces",
            domain=(0, segy_file.tracecount - 1),
            dtype=dtype,
            tile=np.clip(tile_size // (samples * cell_size), 1, segy_file.tracecount),
        ),
        tiledb.Dim(
            name="samples",
            domain=(0, samples - 1),
            dtype=dtype,
            tile=np.clip(tile_size // cell_size, 1, samples),
        ),
    ]


def _get_structured_data_dims(
    segy_file: SegyFile, tile_size: int
) -> Sequence[tiledb.Dim]:
    cell_size = segy_file.dtype.itemsize
    ilines, xlines, offsets, samples = map(
        len, (segy_file.ilines, segy_file.xlines, segy_file.offsets, segy_file.samples)
    )
    dtype = np.uintc

    if segy_file.fast is segy_file.iline:
        iline_tile = np.clip(tile_size // (xlines * samples * cell_size), 1, ilines)
        xline_tile = xlines
    else:
        iline_tile = ilines
        xline_tile = np.clip(tile_size // (ilines * samples * cell_size), 1, xlines)

    dims = [
        tiledb.Dim(
            name="ilines", domain=(0, ilines - 1), dtype=dtype, tile=iline_tile,
        ),
        tiledb.Dim(
            name="xlines", domain=(0, xlines - 1), dtype=dtype, tile=xline_tile,
        ),
        tiledb.Dim(
            name="samples",
            domain=(0, samples - 1),
            dtype=dtype,
            tile=np.clip(tile_size // cell_size, 1, samples),
        ),
    ]
    if offsets > 1:
        dims.insert(
            2, tiledb.Dim(name="offsets", domain=(0, offsets - 1), dtype=dtype, tile=1),
        )
    return dims


def _fill_headers(tdb: tiledb.Array, segy_file: SegyFile, tile_size: int) -> None:
    tdb.meta["__text__"] = b"".join(segy_file.text)
    for k, v in segy_file.bin.items():
        tdb.meta[str(k)] = v
    if segy_file.unstructured:
        _fill_unstructured_trace_headers(tdb, segy_file, tile_size)
    else:
        _fill_structured_trace_headers(tdb, segy_file, tile_size)


def _fill_unstructured_trace_headers(
    tdb: tiledb.Array, segy_file: SegyFile, tile_size: int
) -> None:
    step = np.clip(tile_size // TRACE_FIELDS_SIZE, 1, segy_file.tracecount)
    for sl in _iter_slices(segy_file.tracecount, step):
        headers = [np.zeros(sl.stop - sl.start, dtype) for dtype in TRACE_FIELD_DTYPES]
        for i, field in enumerate(segy_file.header[sl]):
            getfield, buf = field.getfield, field.buf
            for key, header in zip(TRACE_FIELD_ENUMS, headers):
                v = getfield(buf, key)
                if v:
                    header[i] = v
        tdb[sl] = dict(zip(TRACE_FIELD_NAMES, headers))


def _fill_structured_trace_headers(
    tdb: tiledb.Array, segy_file: SegyFile, tile_size: int
) -> None:
    ilines, xlines = map(len, (segy_file.ilines, segy_file.xlines))
    if segy_file.fast is segy_file.iline:
        step = np.clip(tile_size // (xlines * TRACE_FIELDS_SIZE), 1, ilines)
    else:
        step = np.clip(tile_size // (ilines * TRACE_FIELDS_SIZE), 1, xlines)

    if tdb.schema.domain.has_dim("offsets"):
        for i_offset, offset in enumerate(segy_file.offsets):
            for islice, xslice, subcube in _iter_subcube_headers(
                segy_file, step, offset
            ):
                tdb[islice, xslice, i_offset] = subcube
    else:
        for islice, xslice, subcube in _iter_subcube_headers(segy_file, step):
            tdb[islice, xslice] = subcube


def _iter_subcube_headers(
    segy_file: SegyFile, step: int, offset: Optional[int] = None
) -> Iterator[Tuple[slice, slice, np.ndarray]]:
    if segy_file.fast is segy_file.iline:
        fast_headers = segy_file.header.iline
        fast_lines = segy_file.ilines
        num_slow_lines = len(segy_file.xlines)
        axis = 0
    else:
        fast_headers = segy_file.header.xline
        fast_lines = segy_file.xlines
        num_slow_lines = len(segy_file.ilines)
        axis = 1
    if offset is None:
        offset = segy_file.fast.default_offset
    islice = xslice = slice(None, None)
    for fast_slice in _iter_slices(len(fast_lines), step):
        headers = [
            np.zeros((fast_slice.stop - fast_slice.start, num_slow_lines), dtype)
            for dtype in TRACE_FIELD_DTYPES
        ]
        for i, line_id in enumerate(fast_lines[fast_slice]):
            for j, field in enumerate(fast_headers[line_id, offset]):
                getfield, buf = field.getfield, field.buf
                for key, header in zip(TRACE_FIELD_ENUMS, headers):
                    v = getfield(buf, key)
                    if v:
                        header[i, j] = v
        if axis == 0:
            islice = fast_slice
        else:
            xslice = fast_slice
            headers = [h.T for h in headers]
        yield islice, xslice, dict(zip(TRACE_FIELD_NAMES, headers))


def _fill_data(tdb: tiledb.Array, segy_file: SegyFile, tile_size: int) -> None:
    tdb.meta["samples"] = segy_file.samples.tolist()
    if segy_file.unstructured:
        _fill_unstructured_data(tdb, segy_file, tile_size)
    else:
        tdb.meta["ilines"] = segy_file.ilines.tolist()
        tdb.meta["xlines"] = segy_file.xlines.tolist()
        if tdb.schema.domain.has_dim("offsets"):
            tdb.meta["offsets"] = segy_file.offsets.tolist()
        _fill_structured_data(tdb, segy_file, tile_size)


def _fill_unstructured_data(
    tdb: tiledb.Array, segy_file: SegyFile, tile_size: int
) -> None:
    dtype = segy_file.dtype
    step = np.clip(
        tile_size // (len(segy_file.samples) * dtype.itemsize), 1, segy_file.tracecount
    )
    for sl in _iter_slices(segy_file.tracecount, step):
        tdb[sl] = segy_file.trace.raw[sl]


def _fill_structured_data(
    tdb: tiledb.Array, segy_file: SegyFile, tile_size: int
) -> None:
    cell_size = segy_file.dtype.itemsize
    ilines, xlines, samples = map(
        len, (segy_file.ilines, segy_file.xlines, segy_file.samples)
    )
    if segy_file.fast is segy_file.iline:
        step = np.clip(tile_size // (xlines * samples * cell_size), 1, ilines,)
    else:
        step = np.clip(tile_size // (ilines * samples * cell_size), 1, xlines,)

    if tdb.schema.domain.has_dim("offsets"):
        for i_offset, offset in enumerate(segy_file.offsets):
            for islice, xslice, subcube in _iter_subcubes(segy_file, step, offset):
                tdb[islice, xslice, i_offset] = subcube
    else:
        for islice, xslice, subcube in _iter_subcubes(segy_file, step):
            tdb[islice, xslice] = subcube


def _iter_subcubes(
    segy_file: SegyFile, step: int, offset: Optional[int] = None
) -> Iterator[Tuple[slice, slice, np.ndarray]]:
    fast_line = segy_file.fast
    if fast_line is segy_file.iline:
        fast_lines = segy_file.ilines
        axis = 0
    else:
        fast_lines = segy_file.xlines
        axis = 1
    if offset is None:
        offset = fast_line.default_offset
    islice = xslice = slice(None, None)
    for fast_slice in _iter_slices(len(fast_lines), step):
        subcube = np.stack(
            [fast_line[i, offset] for i in fast_lines[fast_slice]], axis=axis
        )
        if axis == 0:
            islice = fast_slice
        else:
            xslice = fast_slice
        yield islice, xslice, subcube


def _iter_slices(size: int, step: int) -> Iterator[slice]:
    r = range(0, size, step)
    yield from map(slice, r, r[1:])
    yield slice(r[-1], size)


if __name__ == "__main__":
    import sys

    import segyio

    segy_file, output_dir, ignore_geometry = sys.argv[1:]
    with segyio.open(segy_file, ignore_geometry=int(ignore_geometry)) as segy_file:
        create(output_dir, segy_file, 4 * 1024 ** 2)
