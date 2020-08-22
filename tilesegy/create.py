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
MAX_TILESIZE = 2 ** 16

logger = logging.getLogger(__name__)


def create(uri: str, segy_file: SegyFile, chunk_bytes: Optional[int] = None) -> None:
    if tiledb.object_type(uri) != "group":
        tiledb.group_create(uri)

    headers_uri = os.path.join(uri, "headers")
    if tiledb.object_type(headers_uri) != "array":
        _create_headers_array(headers_uri, segy_file, chunk_bytes)

    data_uri = os.path.join(uri, "data")
    if tiledb.object_type(data_uri) != "array":
        _create_data_array(data_uri, segy_file, chunk_bytes)


def _create_headers_array(
    uri: str, segy_file: SegyFile, chunk_bytes: Optional[int] = None
) -> None:
    schema = _get_headers_schema(segy_file)
    logger.info(f"header schema: {schema}")
    tiledb.DenseArray.create(uri, schema)
    with tiledb.DenseArray(uri, mode="w") as tdb:
        _fill_headers(tdb, segy_file, chunk_bytes)
    tiledb.consolidate(uri)
    tiledb.vacuum(uri)


def _create_data_array(
    uri: str, segy_file: SegyFile, chunk_bytes: Optional[int] = None
) -> None:
    schema = _get_data_schema(segy_file)
    logger.info(f"data schema: {schema}")
    tiledb.DenseArray.create(uri, schema)
    with tiledb.DenseArray(uri, mode="w") as tdb:
        _fill_data(tdb, segy_file, chunk_bytes)
    tiledb.consolidate(uri)
    tiledb.vacuum(uri)


def _get_headers_schema(segy_file: SegyFile) -> tiledb.ArraySchema:
    if segy_file.unstructured:
        dims = _get_unstructured_header_dims(segy_file)
    else:
        dims = _get_structured_header_dims(segy_file)
    return tiledb.ArraySchema(
        domain=tiledb.Domain(*dims),
        attrs=[
            tiledb.Attr(f.name, f.dtype, filters=TRACE_FIELD_FILTERS)
            for f in TRACE_FIELDS
        ],
    )


def _get_data_schema(segy_file: SegyFile) -> tiledb.ArraySchema:
    if segy_file.unstructured:
        dims = _get_unstructured_data_dims(segy_file)
    else:
        dims = _get_structured_data_dims(segy_file)
    return tiledb.ArraySchema(
        domain=tiledb.Domain(*dims), attrs=[tiledb.Attr(dtype=segy_file.dtype)]
    )


def _get_unstructured_header_dims(segy_file: SegyFile) -> Sequence[tiledb.Dim]:
    domain = (0, segy_file.tracecount - 1)
    return [
        tiledb.Dim(
            name="traces",
            domain=domain,
            dtype=_find_shortest_dtype(domain),
            tile=np.clip(MAX_TILESIZE // TRACE_FIELDS_SIZE, 1, segy_file.tracecount),
        ),
    ]


def _get_structured_header_dims(segy_file: SegyFile) -> Sequence[tiledb.Dim]:
    ilines, xlines, offsets = segy_file.ilines, segy_file.xlines, segy_file.offsets
    domains = [(0, len(ilines) - 1), (0, len(xlines) - 1)]
    if len(offsets) > 1:
        domains.append((0, len(offsets) - 1))
    dtype = _find_shortest_dtype(sum(domains, ()))
    dims = [
        tiledb.Dim(
            name="ilines",
            domain=domains[0],
            dtype=dtype,
            tile=np.clip(
                MAX_TILESIZE // (len(xlines) * TRACE_FIELDS_SIZE), 1, len(ilines),
            ),
        ),
        tiledb.Dim(
            name="xlines",
            domain=domains[1],
            dtype=dtype,
            tile=np.clip(
                MAX_TILESIZE // (len(ilines) * TRACE_FIELDS_SIZE), 1, len(xlines),
            ),
        ),
    ]
    if len(domains) == 3:
        dims.append(tiledb.Dim(name="offsets", domain=domains[2], dtype=dtype, tile=1))
    return dims


def _get_unstructured_data_dims(segy_file: SegyFile) -> Sequence[tiledb.Dim]:
    num_samples = len(segy_file.samples)
    cell_size = segy_file.dtype.itemsize
    domains = [(0, segy_file.tracecount - 1), (0, num_samples - 1)]
    dtype = _find_shortest_dtype(sum(domains, ()))
    return [
        tiledb.Dim(
            name="traces",
            domain=domains[0],
            dtype=dtype,
            tile=np.clip(
                MAX_TILESIZE // (num_samples * cell_size), 1, segy_file.tracecount
            ),
        ),
        tiledb.Dim(
            name="samples",
            domain=domains[1],
            dtype=dtype,
            tile=np.clip(MAX_TILESIZE // cell_size, 1, num_samples),
        ),
    ]


def _get_structured_data_dims(segy_file: SegyFile) -> Sequence[tiledb.Dim]:
    num_samples = len(segy_file.samples)
    cell_size = segy_file.dtype.itemsize
    ilines, xlines, offsets = segy_file.ilines, segy_file.xlines, segy_file.offsets
    domains = [(0, len(ilines) - 1), (0, len(xlines) - 1), (0, num_samples - 1)]
    if len(offsets) > 1:
        domains.append((0, len(offsets) - 1))
    dtype = _find_shortest_dtype(sum(domains, ()))
    dims = [
        tiledb.Dim(
            name="ilines",
            domain=domains[0],
            dtype=dtype,
            tile=np.clip(
                MAX_TILESIZE // (len(xlines) * num_samples * cell_size), 1, len(ilines),
            ),
        ),
        tiledb.Dim(
            name="xlines",
            domain=domains[1],
            dtype=dtype,
            tile=np.clip(
                MAX_TILESIZE // (len(ilines) * num_samples * cell_size), 1, len(xlines),
            ),
        ),
        tiledb.Dim(
            name="samples",
            domain=domains[2],
            dtype=dtype,
            tile=np.clip(MAX_TILESIZE // cell_size, 1, num_samples),
        ),
    ]
    if len(domains) == 4:
        dims.insert(
            2, tiledb.Dim(name="offsets", domain=domains[3], dtype=dtype, tile=1)
        )
    return dims


def _fill_headers(
    tdb: tiledb.Array, segy_file: SegyFile, chunk_bytes: Optional[int] = None
) -> None:
    for i, text in enumerate(segy_file.text):
        tdb.meta[f"text_{i}"] = bytes(text)
    for k, v in segy_file.bin.items():
        tdb.meta[f"bin_{k}"] = v
    if segy_file.unstructured:
        _fill_unstructured_trace_headers(tdb, segy_file, chunk_bytes)
    else:
        _fill_structured_trace_headers(tdb, segy_file, chunk_bytes)


def _fill_unstructured_trace_headers(
    tdb: tiledb.Array, segy_file: SegyFile, chunk_bytes: Optional[int] = None
) -> None:
    if chunk_bytes is not None:
        step = np.clip(chunk_bytes // TRACE_FIELDS_SIZE, 1, segy_file.tracecount)
    else:
        step = segy_file.tracecount
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
    tdb: tiledb.Array, segy_file: SegyFile, chunk_bytes: Optional[int] = None
) -> None:
    ilines, xlines = segy_file.ilines, segy_file.xlines
    if segy_file.fast is segy_file.iline:
        if chunk_bytes is not None:
            step = np.clip(
                chunk_bytes // (len(xlines) * TRACE_FIELDS_SIZE), 1, len(ilines),
            )
        else:
            step = len(ilines)
    else:
        if chunk_bytes is not None:
            step = np.clip(
                chunk_bytes // (len(ilines) * TRACE_FIELDS_SIZE), 1, len(xlines),
            )
        else:
            step = len(xlines)

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
        slow_lines = segy_file.xlines
        axis = 0
    else:
        fast_headers = segy_file.header.xline
        fast_lines = segy_file.xlines
        slow_lines = segy_file.ilines
        axis = 1
    if offset is None:
        offset = segy_file.fast.default_offset
    islice = xslice = slice(None, None)
    for fast_slice in _iter_slices(len(fast_lines), step):
        headers = [
            np.zeros((fast_slice.stop - fast_slice.start, len(slow_lines)), dtype)
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


def _fill_data(
    tdb: tiledb.Array, segy_file: SegyFile, chunk_bytes: Optional[int] = None
) -> None:
    tdb.meta["samples"] = segy_file.samples.tolist()
    if segy_file.unstructured:
        _fill_unstructured_data(tdb, segy_file, chunk_bytes)
    else:
        tdb.meta["ilines"] = segy_file.ilines.tolist()
        tdb.meta["xlines"] = segy_file.xlines.tolist()
        if tdb.schema.domain.has_dim("offsets"):
            tdb.meta["offsets"] = segy_file.offsets.tolist()
        _fill_structured_data(tdb, segy_file, chunk_bytes)


def _fill_unstructured_data(
    tdb: tiledb.Array, segy_file: SegyFile, chunk_bytes: Optional[int] = None
) -> None:
    num_samples = len(segy_file.samples)
    dtype = segy_file.dtype
    if chunk_bytes is not None:
        step = np.clip(
            chunk_bytes // (num_samples * dtype.itemsize), 1, segy_file.tracecount
        )
    else:
        step = segy_file.tracecount
    for sl in _iter_slices(segy_file.tracecount, step):
        tdb[sl] = segy_file.trace.raw[sl]


def _fill_structured_data(
    tdb: tiledb.Array, segy_file: SegyFile, chunk_bytes: Optional[int] = None
) -> None:
    num_samples = len(segy_file.samples)
    cell_size = segy_file.dtype.itemsize
    ilines, xlines = segy_file.ilines, segy_file.xlines
    if segy_file.fast is segy_file.iline:
        if chunk_bytes is not None:
            step = np.clip(
                chunk_bytes // (len(xlines) * num_samples * cell_size), 1, len(ilines),
            )
        else:
            step = len(ilines)
    else:
        if chunk_bytes is not None:
            step = np.clip(
                chunk_bytes // (len(ilines) * num_samples * cell_size), 1, len(xlines),
            )
        else:
            step = len(xlines)

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


def _find_shortest_dtype(values: Collection[Number]) -> np.dtype:
    min_value = min(values)
    max_value = max(values)
    for dt in (
        np.uint8,
        np.int8,
        np.uint16,
        np.int16,
        np.uint32,
        np.int32,
        np.uint64,
        np.int64,
    ):
        info = np.iinfo(dt)
        if info.min <= min_value <= max_value <= info.max:
            return dt


def _iter_slices(size: int, step: int) -> Iterator[slice]:
    r = range(0, size, step)
    yield from map(slice, r, r[1:])
    yield slice(r[-1], size)


if __name__ == "__main__":
    import sys

    import segyio

    segy_file, output_dir, ignore_geometry = sys.argv[1:]
    with segyio.open(segy_file, ignore_geometry=int(ignore_geometry)) as segy_file:
        create(output_dir, segy_file)
