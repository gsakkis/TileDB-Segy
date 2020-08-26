import os
from abc import ABC, abstractmethod
from collections import namedtuple
from contextlib import contextmanager
from typing import Iterable, Iterator, Optional, Tuple, Type, Union

import numpy as np
import tiledb
from segyio import SegyFile, TraceField, TraceSortingFormat
from segyio.field import Field
from segyio.line import HeaderLine

TypedTraceField = namedtuple("TypedTraceField", ["name", "enum", "dtype"])


def iter_typed_trace_fields(
    exclude: Iterable[Union[TraceField, str]] = ()
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


def segy_to_tiledb(
    segy_file: SegyFile,
    uri: str,
    *,
    tile_size: int,
    config: Optional[tiledb.Config] = None,
) -> None:
    cls: Type[SegyFileConverter]
    if segy_file.sorting == TraceSortingFormat.INLINE_SORTING:
        cls = InlineSegyFileConverter
    elif segy_file.sorting == TraceSortingFormat.CROSSLINE_SORTING:
        cls = CrosslineSegyFileConverter
    else:
        cls = UnstructuredSegyFileConverter

    cls(segy_file, tile_size, config).to_tiledb(uri)


class SegyFileConverter(ABC):
    def __init__(
        self,
        segy_file: SegyFile,
        tile_size: int,
        config: Optional[tiledb.Config] = None,
    ):
        self.segy_file = segy_file
        self.tile_size = tile_size
        self.config = config

    def to_tiledb(self, uri: str) -> None:
        if tiledb.object_type(uri) != "group":
            tiledb.group_create(uri)

        headers_uri = os.path.join(uri, "headers")
        if tiledb.object_type(headers_uri) != "array":
            with self._tiledb_array(headers_uri, self.header_schema) as tdb:
                self._fill_headers(tdb)

        data_uri = os.path.join(uri, "data")
        if tiledb.object_type(data_uri) != "array":
            with self._tiledb_array(data_uri, self.data_schema) as tdb:
                self._fill_data(tdb)

    @property
    def header_schema(self) -> tiledb.ArraySchema:
        dims = self._get_dims(TRACE_FIELDS_SIZE)
        return tiledb.ArraySchema(
            domain=tiledb.Domain(*dims),
            attrs=[
                tiledb.Attr(f.name, f.dtype, filters=TRACE_FIELD_FILTERS)
                for f in TRACE_FIELDS
            ],
        )

    @property
    def data_schema(self) -> tiledb.ArraySchema:
        sample_size, samples = self._sample_size_count
        dims = list(self._get_dims(sample_size * samples))
        dims.append(
            tiledb.Dim(
                name="samples",
                domain=(0, samples - 1),
                dtype=dims[0].dtype,
                tile=np.clip(self.tile_size // sample_size, 1, samples),
            )
        )
        return tiledb.ArraySchema(
            domain=tiledb.Domain(*dims),
            attrs=[tiledb.Attr(dtype=self.segy_file.dtype)],
        )

    @property
    def _sample_size_count(self) -> Tuple[int, int]:
        return self.segy_file.dtype.itemsize, len(self.segy_file.samples)

    @property
    def _trace_size(self) -> int:
        sample_size, samples = self._sample_size_count
        return sample_size * samples

    @contextmanager
    def _tiledb_array(
        self, uri: str, schema: tiledb.ArraySchema
    ) -> Iterator[tiledb.Array]:
        tiledb.DenseArray.create(uri, schema)
        with tiledb.DenseArray(uri, mode="w") as tdb:
            yield tdb
        tiledb.consolidate(uri, config=self.config)
        tiledb.vacuum(uri)

    @abstractmethod
    def _get_dims(self, trace_size: int) -> Iterable[tiledb.Dim]:
        ...

    @abstractmethod
    def _fill_headers(self, tdb: tiledb.Array) -> None:
        tdb.meta["__text__"] = b"".join(self.segy_file.text)
        for k, v in self.segy_file.bin.items():
            tdb.meta[str(k)] = v

    @abstractmethod
    def _fill_data(self, tdb: tiledb.Array) -> None:
        tdb.meta["samples"] = self.segy_file.samples.tolist()


class UnstructuredSegyFileConverter(SegyFileConverter):
    @property
    def _trace_count(self) -> int:
        return int(self.segy_file.tracecount)

    def _get_dims(self, trace_size: int) -> Iterable[tiledb.Dim]:
        traces = self._trace_count
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
        traces = self._trace_count
        get_header = self.segy_file.header
        step = np.clip(self.tile_size // TRACE_FIELDS_SIZE, 1, traces)
        for sl in _iter_slices(traces, step):
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
        traces = self._trace_count
        step = np.clip(self.tile_size // self._trace_size, 1, traces)
        for sl in _iter_slices(traces, step):
            tdb[sl] = raw_trace[sl]


class StructuredSegyFileConverter(SegyFileConverter):
    @property
    def _iline_count(self) -> int:
        return len(self.segy_file.ilines)

    @property
    def _xline_count(self) -> int:
        return len(self.segy_file.xlines)

    def _get_dims(self, trace_size: int) -> Iterable[tiledb.Dim]:
        dtype = np.uintc
        dims = [
            tiledb.Dim(
                name="ilines",
                domain=(0, self._iline_count - 1),
                dtype=dtype,
                tile=self._iline_tile(trace_size),
            ),
            tiledb.Dim(
                name="xlines",
                domain=(0, self._xline_count - 1),
                dtype=dtype,
                tile=self._xline_tile(trace_size),
            ),
        ]
        offsets = len(self.segy_file.offsets)
        if offsets > 1:
            dims.append(
                tiledb.Dim(
                    name="offsets", domain=(0, offsets - 1), dtype=dtype, tile=1,
                )
            )
        return dims

    def _fill_headers(self, tdb: tiledb.Array) -> None:
        super()._fill_headers(tdb)

        is_inline = self.segy_file.sorting == TraceSortingFormat.INLINE_SORTING
        step = self._fast_tile(TRACE_FIELDS_SIZE)
        get_headerline = self._fast_headerline
        num_slow_lines = len(self._slow_lines)

        for hyperslice, offset, lines in self._iter_hyperslice_offset_lines(step):
            headers = [
                np.zeros((len(lines), num_slow_lines), dtype)
                for dtype in TRACE_FIELD_DTYPES
            ]
            for i, line in enumerate(lines):
                for j, field in enumerate(get_headerline[line, offset]):
                    getfield, buf = field.getfield, field.buf
                    for key, header in zip(TRACE_FIELD_ENUMS, headers):
                        v = getfield(buf, key)
                        if v:
                            header[i, j] = v
            if not is_inline:
                headers = [h.T for h in headers]
            tdb[hyperslice] = dict(zip(TRACE_FIELD_NAMES, headers))

    def _fill_data(self, tdb: tiledb.Array) -> None:
        super()._fill_data(tdb)

        tdb.meta["ilines"] = self.segy_file.ilines.tolist()
        tdb.meta["xlines"] = self.segy_file.xlines.tolist()
        if tdb.schema.domain.has_dim("offsets"):
            tdb.meta["offsets"] = self.segy_file.offsets.tolist()

        is_inline = self.segy_file.sorting == TraceSortingFormat.INLINE_SORTING
        get_line = self.segy_file.fast
        step = self._fast_tile(self._trace_size)
        for hyperslice, offset, lines in self._iter_hyperslice_offset_lines(step):
            subcube = np.stack([get_line[i, offset] for i in lines])
            if not is_inline:
                subcube = subcube.swapaxes(0, 1)
            tdb[hyperslice] = subcube

    def _iter_hyperslice_offset_lines(
        self, step: int
    ) -> Iterator[Tuple[Tuple[Union[slice, int], ...], int, np.ndarray]]:
        is_inline = self.segy_file.sorting == TraceSortingFormat.INLINE_SORTING
        fast_lines = self._fast_lines
        full_slice = slice(None, None)
        if len(self.segy_file.offsets) > 1:
            for i_offset, offset in enumerate(self.segy_file.offsets):
                for fast_slice in _iter_slices(len(fast_lines), step):
                    hyperslice = [fast_slice, full_slice]
                    if not is_inline:
                        hyperslice.reverse()
                    yield (*hyperslice, i_offset), offset, fast_lines[fast_slice]
        else:
            offset = self.segy_file.fast.default_offset
            for fast_slice in _iter_slices(len(fast_lines), step):
                hyperslice = [fast_slice, full_slice]
                if not is_inline:
                    hyperslice.reverse()
                yield tuple(hyperslice), offset, fast_lines[fast_slice]

    @abstractmethod
    def _iline_tile(self, trace_size: int) -> int:
        ...

    @abstractmethod
    def _xline_tile(self, trace_size: int) -> int:
        ...

    @abstractmethod
    def _fast_tile(self, trace_size: int) -> int:
        ...

    @property
    @abstractmethod
    def _fast_headerline(self) -> HeaderLine:
        ...

    @property
    @abstractmethod
    def _fast_lines(self) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def _slow_lines(self) -> np.ndarray:
        ...


class InlineSegyFileConverter(StructuredSegyFileConverter):
    def _iline_tile(self, trace_size: int) -> int:
        return int(
            np.clip(
                self.tile_size // (self._xline_count * trace_size), 1, self._iline_count
            )
        )

    def _xline_tile(self, trace_size: int) -> int:
        return self._xline_count

    _fast_tile = _iline_tile
    _fast_headerline = property(lambda self: self.segy_file.header.iline)
    _fast_lines = property(lambda self: self.segy_file.ilines)
    _slow_lines = property(lambda self: self.segy_file.xlines)


class CrosslineSegyFileConverter(StructuredSegyFileConverter):
    def _iline_tile(self, trace_size: int) -> int:
        return self._iline_count

    def _xline_tile(self, trace_size: int) -> int:
        return int(
            np.clip(
                self.tile_size // (self._iline_count * trace_size), 1, self._xline_count
            )
        )

    _fast_tile = _xline_tile
    _fast_headerline = property(lambda self: self.segy_file.header.xline)
    _fast_lines = property(lambda self: self.segy_file.xlines)
    _slow_lines = property(lambda self: self.segy_file.ilines)


def _iter_slices(size: int, step: int) -> Iterator[slice]:
    r = range(0, size, step)
    yield from map(slice, r, r[1:])
    yield slice(r[-1], size)


def main() -> None:
    import shutil
    import sys

    import segyio

    segy_file, output_dir = sys.argv[1:]
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    with segyio.open(segy_file, strict=False) as segy_file:
        segy_to_tiledb(
            segy_file,
            output_dir,
            tile_size=4 * 1024 ** 2,
            config=tiledb.Config({"sm.consolidation.buffer_size": 500000}),
        )


if __name__ == "__main__":
    main()
