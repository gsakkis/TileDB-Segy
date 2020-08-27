__all__ = ["segy_to_tiledb"]

import copy
import os
from abc import ABC, abstractmethod
from collections import namedtuple
from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, Optional, Type, Union

import numpy as np
import segyio
import tiledb

TypedTraceField = namedtuple("TypedTraceField", ["name", "enum", "dtype"])


def iter_typed_trace_fields(
    exclude: Iterable[Union[segyio.TraceField, str]] = ()
) -> Iterator[TypedTraceField]:
    all_fields = segyio.TraceField.enums()
    include_names = set(map(str, segyio.field.Field._tr_keys))
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
    segy_file: segyio.SegyFile,
    uri: str,
    *,
    tile_size: int,
    config: Optional[tiledb.Config] = None,
) -> None:
    if not isinstance(segy_file, ExtendedSegyFile):
        segy_file = copy.copy(segy_file)
        segy_file.__class__ = ExtendedSegyFile

    cls: Type[SegyFileConverter]
    if segy_file.sorting == segyio.TraceSortingFormat.INLINE_SORTING:
        cls = InlineSegyFileConverter
    elif segy_file.sorting == segyio.TraceSortingFormat.CROSSLINE_SORTING:
        cls = CrosslineSegyFileConverter
    else:
        cls = UnstructuredSegyFileConverter

    cls(segy_file, tile_size, config).to_tiledb(uri)


class ExtendedSegyFile(segyio.SegyFile):

    trace_size = property(lambda self: len(self._samples) * int(self._dtype.itemsize))
    fast_headerline = property(
        lambda self: self._header.iline if self._is_inline else self._header.xline
    )
    fast_lines = property(
        lambda self: self._ilines if self._is_inline else self._xlines
    )
    slow_lines = property(
        lambda self: self._xlines if self._is_inline else self._ilines
    )

    @property
    def _is_inline(self) -> bool:
        if self.sorting == segyio.TraceSortingFormat.INLINE_SORTING:
            return True
        if self.sorting == segyio.TraceSortingFormat.CROSSLINE_SORTING:
            return False
        raise RuntimeError("Unknown sorting.")


class SegyFileConverter(ABC):
    def __init__(
        self,
        segy_file: ExtendedSegyFile,
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
        sample_size = self.segy_file.dtype.itemsize
        samples = len(self.segy_file.samples)
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

    @contextmanager
    def _tiledb_array(
        self, uri: str, schema: tiledb.ArraySchema
    ) -> Iterator[tiledb.Array]:
        tiledb.DenseArray.create(uri, schema)
        with tiledb.DenseArray(uri, mode="w") as tdb:
            yield tdb
        tiledb.consolidate(uri, config=self.config)
        tiledb.vacuum(uri, config=self.config)

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
        tiles = [self._fast_tile(trace_size), len(self.segy_file.slow_lines)]
        if self.segy_file.fast is self.segy_file.xline:
            tiles.reverse()
        itile, xtile = tiles
        dtype = np.uintc
        return [
            tiledb.Dim(
                name="ilines",
                domain=(0, len(self.segy_file.ilines) - 1),
                dtype=dtype,
                tile=itile,
            ),
            tiledb.Dim(
                name="xlines",
                domain=(0, len(self.segy_file.xlines) - 1),
                dtype=dtype,
                tile=xtile,
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
                attr_cubes = dict(zip(TRACE_FIELD_NAMES, cubes))
                self._fill_header_slice(tdb, sl, offset_idx, attr_cubes)

    def _fill_data(self, tdb: tiledb.Array) -> None:
        super()._fill_data(tdb)
        for key in "ilines", "xlines", "offsets":
            tdb.meta[key] = getattr(self.segy_file, key).tolist()

        step = self._fast_tile(self.segy_file.trace_size)
        fast_lines = self.segy_file.fast_lines
        get_line = self.segy_file.fast
        for offset_idx, offset in enumerate(self.segy_file.offsets):
            for sl in iter_slices(len(fast_lines), step):
                cube = np.stack([get_line[i, offset] for i in fast_lines[sl]])
                self._fill_data_slice(tdb, sl, offset_idx, cube)

    def _fast_tile(self, trace_size: int) -> int:
        num_fast, num_slow = map(
            len, (self.segy_file.fast_lines, self.segy_file.slow_lines)
        )
        return int(np.clip(self.tile_size // (num_slow * trace_size), 1, num_fast))

    @abstractmethod
    def _fill_header_slice(
        self,
        tdb: tiledb.Array,
        fast_slice: slice,
        offset_idx: int,
        attr_cubes: Dict[str, np.ndarray],
    ) -> None:
        ...

    @abstractmethod
    def _fill_data_slice(
        self, tdb: tiledb.Array, fast_slice: slice, offset_idx: int, cube: np.ndarray
    ) -> None:
        ...


class InlineSegyFileConverter(StructuredSegyFileConverter):
    def _fill_header_slice(
        self,
        tdb: tiledb.Array,
        fast_slice: slice,
        offset_idx: int,
        attr_cubes: Dict[str, np.ndarray],
    ) -> None:
        tdb[fast_slice, :, offset_idx] = attr_cubes

    def _fill_data_slice(
        self, tdb: tiledb.Array, fast_slice: slice, offset_idx: int, cube: np.ndarray
    ) -> None:
        tdb[fast_slice, :, offset_idx] = cube


class CrosslineSegyFileConverter(StructuredSegyFileConverter):
    def _fill_header_slice(
        self,
        tdb: tiledb.Array,
        fast_slice: slice,
        offset_idx: int,
        attr_cubes: Dict[str, np.ndarray],
    ) -> None:
        for k, v in attr_cubes.items():
            attr_cubes[k] = v.swapaxes(0, 1)
        tdb[:, fast_slice, offset_idx] = attr_cubes

    def _fill_data_slice(
        self, tdb: tiledb.Array, fast_slice: slice, offset_idx: int, cube: np.ndarray
    ) -> None:
        tdb[:, fast_slice, offset_idx] = cube.swapaxes(0, 1)


def iter_slices(size: int, step: int) -> Iterator[slice]:
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
