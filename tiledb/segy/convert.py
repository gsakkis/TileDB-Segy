import copy
from abc import ABC, abstractmethod
from collections import namedtuple
from contextlib import contextmanager
from pathlib import PurePath
from typing import Any, Iterable, Iterator, Union

import numpy as np
import segyio
import segyio.tools
from cached_property import cached_property
from urlpath import URL

import tiledb

TypedTraceField = namedtuple("TypedTraceField", ["name", "enum", "dtype"])


def iter_typed_trace_fields() -> Iterator[TypedTraceField]:
    all_fields = segyio.TraceField.enums()
    include_names = set(map(str, segyio.field.Field._tr_keys))
    size2dtype = {2: np.dtype(np.int16), 4: np.dtype(np.int32)}
    for f, f2 in zip(all_fields, all_fields[1:]):
        name = str(f)
        if name in include_names:
            yield TypedTraceField(name, f, size2dtype[int(f2) - int(f)])


TRACE_FIELDS = tuple(iter_typed_trace_fields())
TRACE_FIELD_ENUMS = tuple(int(f.enum) for f in TRACE_FIELDS)
TRACE_FIELD_NAMES = tuple(f.name for f in TRACE_FIELDS)
TRACE_FIELD_DTYPES = tuple(f.dtype for f in TRACE_FIELDS)
TRACE_FIELDS_SIZE = sum(dtype.itemsize for dtype in TRACE_FIELD_DTYPES)
TRACE_FIELD_FILTERS = (
    tiledb.BitWidthReductionFilter(),
    tiledb.ByteShuffleFilter(),
    tiledb.LZ4Filter(),
)


class ExtendedSegyFile(segyio.SegyFile):
    @cached_property
    def trace_size(self) -> int:
        return len(self._samples) * int(self._dtype.itemsize)

    @cached_property
    def fast_headerline(self) -> segyio.line.HeaderLine:
        return self._header.iline if self.is_inline else self._header.xline

    @cached_property
    def fast_lines(self) -> np.ndarray:
        return self._ilines if self.is_inline else self._xlines

    @cached_property
    def slow_lines(self) -> np.ndarray:
        return self._xlines if self.is_inline else self._ilines

    @cached_property
    def is_inline(self) -> bool:
        assert not self.unstructured
        return bool(self.sorting == segyio.TraceSortingFormat.INLINE_SORTING)


class SegyFileConverter(ABC):
    def __new__(cls, segy_file: segyio.SegyFile, **kwargs: Any) -> "SegyFileConverter":
        if cls is SegyFileConverter:
            if segy_file.unstructured:
                cls = UnstructuredSegyFileConverter
            else:
                cls = StructuredSegyFileConverter
        return super().__new__(cls)

    def __init__(self, segy_file: segyio.SegyFile, *, tile_size: int):
        if not isinstance(segy_file, ExtendedSegyFile):
            segy_file = copy.copy(segy_file)
            segy_file.__class__ = ExtendedSegyFile
        self.segy_file = segy_file
        self.tile_size = tile_size

    def to_tiledb(self, uri: Union[str, PurePath]) -> None:
        uri = URL(uri) if not isinstance(uri, PurePath) else uri

        if tiledb.object_type(str(uri)) != "group":
            tiledb.group_create(str(uri))

        headers_uri = str(uri / "headers")
        if tiledb.object_type(headers_uri) != "array":
            dims = self._get_dims(TRACE_FIELDS_SIZE)
            header_schema = tiledb.ArraySchema(
                domain=tiledb.Domain(*dims),
                sparse=False,
                attrs=[
                    tiledb.Attr(f.name, f.dtype, filters=TRACE_FIELD_FILTERS)
                    for f in TRACE_FIELDS
                ],
            )
            with self._tiledb_array(headers_uri, header_schema) as tdb:
                self._fill_headers(tdb)

        data_uri = str(uri / "data")
        if tiledb.object_type(data_uri) != "array":
            samples = len(self.segy_file.samples)
            sample_dtype = self.segy_file.dtype
            sample_size = sample_dtype.itemsize
            dims = list(self._get_dims(sample_size * samples))
            dims.append(
                tiledb.Dim(
                    name="samples",
                    domain=(0, samples - 1),
                    dtype=dims[0].dtype,
                    tile=np.clip(self.tile_size // sample_size, 1, samples),
                )
            )
            data_schema = tiledb.ArraySchema(
                domain=tiledb.Domain(*dims),
                sparse=False,
                attrs=[
                    tiledb.Attr("trace", sample_dtype, filters=(tiledb.LZ4Filter(),))
                ],
            )
            with self._tiledb_array(data_uri, data_schema) as tdb:
                self._fill_data(tdb)

    @contextmanager
    def _tiledb_array(
        self, uri: str, schema: tiledb.ArraySchema
    ) -> Iterator[tiledb.Array]:
        tiledb.Array.create(uri, schema)
        with tiledb.open(uri, mode="w") as tdb:
            yield tdb

    @abstractmethod
    def _get_dims(self, trace_size: int) -> Iterable[tiledb.Dim]:
        """Get the tiledb schema dimensions"""

    @abstractmethod
    def _fill_headers(self, tdb: tiledb.Array) -> None:
        tdb.meta["__text__"] = b"".join(self.segy_file.text)
        for k, v in self.segy_file.bin.items():
            tdb.meta[str(k)] = v

    @abstractmethod
    def _fill_data(self, tdb: tiledb.Array) -> None:
        tdb.meta["sorting"] = (
            self.segy_file.sorting or segyio.TraceSortingFormat.UNKNOWN_SORTING
        )
        tdb.meta["samples"] = self.segy_file.samples.tolist()
        tdb.meta["dt"] = segyio.tools.dt(self.segy_file, fallback_dt=0)


class UnstructuredSegyFileConverter(SegyFileConverter):
    def _get_dims(self, trace_size: int) -> Iterable[tiledb.Dim]:
        traces = self.segy_file.tracecount
        return [
            tiledb.Dim(
                name="traces",
                domain=(0, traces - 1),
                dtype=np.uint64,
                tile=np.clip(self.tile_size // trace_size, 1, traces),
            ),
        ]

    def _fill_headers(self, tdb: tiledb.Array) -> None:
        super()._fill_headers(tdb)
        traces = self.segy_file.tracecount
        get_header = self.segy_file.header
        step = np.clip(self.tile_size // TRACE_FIELDS_SIZE, 1, traces)
        for sl in iter_slices(traces, step):
            headers = [
                np.zeros(sl.stop - sl.start, dtype) for dtype in TRACE_FIELD_DTYPES
            ]
            for i, field in enumerate(get_header[sl]):
                getfield, buf = field.getfield, field.buf
                for key, header in zip(TRACE_FIELD_ENUMS, headers):
                    v = getfield(buf, key)
                    if v:
                        header[i] = v
            tdb[sl] = dict(zip(TRACE_FIELD_NAMES, headers))

    def _fill_data(self, tdb: tiledb.Array) -> None:
        super()._fill_data(tdb)
        raw_trace = self.segy_file.trace.raw
        traces = self.segy_file.tracecount
        step = np.clip(self.tile_size // self.segy_file.trace_size, 1, traces)
        for sl in iter_slices(traces, step):
            tdb[sl] = raw_trace[sl]


class StructuredSegyFileConverter(SegyFileConverter):
    def _get_dims(self, trace_size: int) -> Iterable[tiledb.Dim]:
        dtype = np.uintc
        slow_lines = len(self.segy_file.slow_lines)
        if self.segy_file.is_inline:
            fast_dim, slow_dim = "ilines", "xlines"
        else:
            fast_dim, slow_dim = "xlines", "ilines"
        return [
            tiledb.Dim(
                name=fast_dim,
                domain=(0, len(self.segy_file.fast_lines) - 1),
                dtype=dtype,
                tile=self._fast_tile(trace_size),
            ),
            tiledb.Dim(
                name=slow_dim,
                domain=(0, slow_lines - 1),
                dtype=dtype,
                tile=slow_lines,
            ),
            tiledb.Dim(
                name="offsets",
                domain=(0, len(self.segy_file.offsets) - 1),
                dtype=dtype,
                tile=1,
            ),
        ]

    def _fill_headers(self, tdb: tiledb.Array) -> None:
        super()._fill_headers(tdb)
        step = self._fast_tile(TRACE_FIELDS_SIZE)
        fast_lines = self.segy_file.fast_lines
        slow_lines = self.segy_file.slow_lines
        fast_headerline = self.segy_file.fast_headerline
        for offset_idx, offset in enumerate(self.segy_file.offsets):
            for sl in iter_slices(len(fast_lines), step):
                slice_lines = fast_lines[sl]
                cubes = [
                    np.zeros((len(slice_lines), len(slow_lines)), dtype)
                    for dtype in TRACE_FIELD_DTYPES
                ]
                for i, line in enumerate(slice_lines):
                    for j, field in enumerate(fast_headerline[line, offset]):
                        getfield, buf = field.getfield, field.buf
                        for key, cube in zip(TRACE_FIELD_ENUMS, cubes):
                            v = getfield(buf, key)
                            if v:
                                cube[i, j] = v
                tdb[sl, :, offset_idx] = dict(zip(TRACE_FIELD_NAMES, cubes))

    def _fill_data(self, tdb: tiledb.Array) -> None:
        super()._fill_data(tdb)
        for key in "ilines", "xlines", "offsets":
            tdb.meta[key] = getattr(self.segy_file, key).tolist()

        step = self._fast_tile(self.segy_file.trace_size)
        fast_lines = self.segy_file.fast_lines
        get_line = self.segy_file.fast
        for offset_idx, offset in enumerate(self.segy_file.offsets):
            for sl in iter_slices(len(fast_lines), step):
                # get_line[i, offset] returns a (slow, samples) 2D array
                # reshape it to a (slow, offsets, samples) 3D array
                lines = [np.expand_dims(get_line[i, offset], 1) for i in fast_lines[sl]]
                tdb[sl, :, offset_idx, :] = np.stack(lines)

    def _fast_tile(self, trace_size: int) -> int:
        num_fast, num_slow = map(
            len, (self.segy_file.fast_lines, self.segy_file.slow_lines)
        )
        return int(np.clip(self.tile_size // (num_slow * trace_size), 1, num_fast))


def iter_slices(size: int, step: int) -> Iterator[slice]:
    r = range(0, size, step)
    yield from map(slice, r, r[1:])
    yield slice(r[-1], size)
