from functools import singledispatch
from typing import Any, Iterable, List, Mapping, Union
from unittest.mock import PropertyMock, patch

import numpy as np
import pytest
from segyio import SegyFile, TraceField, TraceSortingFormat
from tiledb.libtiledb import TileDBError

import tilesegy
from tests.conftest import parametrize_tilesegy_segyfiles
from tilesegy import StructuredTileSegy, TileSegy


def assert_equal_arrays(
    a: Union[np.ndarray, np.number],
    b: Union[np.ndarray, np.number],
) -> None:
    assert a.dtype == b.dtype
    if isinstance(a, np.number) or isinstance(b, np.number):
        assert isinstance(a, np.number) and isinstance(b, np.number)
        assert a == b
    else:
        assert a.ndim == b.ndim
        assert a.shape == b.shape
    np.testing.assert_array_equal(a, b)


def segy_gen_to_array(segy_gen: Iterable[np.ndarray]) -> np.ndarray:
    return np.array(list(map(np.copy, segy_gen)))


@singledispatch
def stringify_keys(o: object) -> Any:
    raise TypeError(f"Cannot stringify_keys for {o.__class__}")


@stringify_keys.register(Mapping)
def _stringify_keys_mapping(d: Mapping[int, int]) -> Mapping[str, int]:
    return {str(k): v for k, v in d.items()}


@stringify_keys.register(Iterable)
def _stringify_keys_iter(s: Iterable[Mapping[int, int]]) -> List[Mapping[str, int]]:
    return list(map(stringify_keys, s))


def iter_slices(i: int, j: int) -> Iterable[slice]:
    return slice(None, None), slice(None, j), slice(i, None), slice(i, j)


class TestTileSegy:
    @parametrize_tilesegy_segyfiles("t", "s")
    def test_sorting(self, t: TileSegy, s: SegyFile) -> None:
        assert t.sorting == s.sorting

    @parametrize_tilesegy_segyfiles("t", "s")
    def test_bin(self, t: TileSegy, s: SegyFile) -> None:
        assert t.bin == stringify_keys(s.bin)

    @parametrize_tilesegy_segyfiles("t", "s")
    def test_text(self, t: TileSegy, s: SegyFile) -> None:
        assert t.text == list(s.text)

    @parametrize_tilesegy_segyfiles("t", "s")
    def test_samples(self, t: TileSegy, s: SegyFile) -> None:
        assert_equal_arrays(t.samples, s.samples)

    @parametrize_tilesegy_segyfiles("t", "s")
    def test_close(self, t: TileSegy, s: SegyFile) -> None:
        t.bin
        t.close()
        with pytest.raises(TileDBError):
            t.bin

    @parametrize_tilesegy_segyfiles("t", "s")
    def test_context_manager(self, t: TileSegy, s: SegyFile) -> None:
        with tilesegy.open(t.uri) as t2:
            t2.bin
        with pytest.raises(TileDBError):
            t2.bin

    @parametrize_tilesegy_segyfiles("t", "s")
    def test_repr(self, t: TileSegy, s: SegyFile) -> None:
        if s.unstructured:
            assert repr(t) == f"TileSegy('{str(t.uri)}')"
        else:
            assert repr(t) == f"StructuredTileSegy('{str(t.uri)}')"

    @parametrize_tilesegy_segyfiles("t", "s")
    def test_trace(self, t: TileSegy, s: SegyFile) -> None:
        assert len(t.trace) == len(s.trace) == s.tracecount

        i = np.random.randint(0, s.tracecount // 2)
        j = np.random.randint(i + 1, s.tracecount)
        x = np.random.randint(0, len(s.samples) // 2)
        y = np.random.randint(x + 1, len(s.samples))

        # one trace, all samples
        assert_equal_arrays(t.trace[i], s.trace[i])

        # one trace, one sample
        assert_equal_arrays(t.trace[i, x], s.trace[i, x])

        # one trace, slice samples
        for sl in iter_slices(x, y):
            assert_equal_arrays(t.trace[i, sl], s.trace[i, sl])

        for sl1 in iter_slices(i, j):
            try:
                # slices traces, all samples
                assert_equal_arrays(t.trace[sl1], segy_gen_to_array(s.trace[sl1]))
                # slices traces, one sample
                assert_equal_arrays(t.trace[sl1, x], segy_gen_to_array(s.trace[sl1, x]))
                # slices traces, slice samples
                for sl2 in iter_slices(x, y):
                    assert_equal_arrays(
                        t.trace[sl1, sl2], segy_gen_to_array(s.trace[sl1, sl2])
                    )
            except NotImplementedError as ex:
                pytest.xfail(str(ex))

        with pytest.raises((IndexError, TypeError)):
            t.trace[float(i), 0]

    @parametrize_tilesegy_segyfiles("t", "s")
    def test_header(self, t: TileSegy, s: SegyFile) -> None:
        assert len(t.header) == len(s.header)

        i = np.random.randint(0, s.tracecount // 2)
        assert t.header[i] == stringify_keys(s.header[i])
        for sl in slice(None, 3), slice(-3, None), slice(i, i + 3):
            try:
                assert t.header[sl] == stringify_keys(s.header[sl])
            except NotImplementedError as ex:
                pytest.xfail(str(ex))

        with pytest.raises(TypeError):
            t.header[i, 0]

    @parametrize_tilesegy_segyfiles("t", "s")
    def test_attributes(self, t: TileSegy, s: SegyFile) -> None:
        str_attr = "TraceNumber"
        t_attrs = t.attributes(str_attr)
        s_attrs = s.attributes(getattr(TraceField, str_attr))
        assert len(t_attrs) == len(s_attrs)

        i = np.random.randint(0, s.tracecount // 2)
        j = np.random.randint(i + 1, s.tracecount)
        assert_equal_arrays(t_attrs[i], s_attrs[i])
        for sl in iter_slices(i, j):
            try:
                assert_equal_arrays(t_attrs[sl], s_attrs[sl])
            except NotImplementedError as ex:
                pytest.xfail(str(ex))

        with pytest.raises(TypeError):
            t_attrs[i, 0]

    @parametrize_tilesegy_segyfiles("t", "s")
    def test_depth(self, t: StructuredTileSegy, s: SegyFile) -> None:
        assert len(t.depth) == len(s.depth_slice)

        i = np.random.randint(0, len(s.samples) // 2)
        j = np.random.randint(i + 1, len(s.samples))
        # one depth
        assert_equal_arrays(t.depth[i], s.depth_slice[i])
        # slice depths
        for sl in iter_slices(i, j):
            assert_equal_arrays(t.depth[sl], segy_gen_to_array(s.depth_slice[sl]))


class TestStructuredTileSegy:
    @parametrize_tilesegy_segyfiles("t", "s", structured=True)
    def test_offsets(self, t: StructuredTileSegy, s: SegyFile) -> None:
        assert_equal_arrays(t.offsets, s.offsets)

    @parametrize_tilesegy_segyfiles("t", "s", structured=True)
    def test_fast(self, t: StructuredTileSegy, s: SegyFile) -> None:
        if s.fast is s.iline:
            assert str(t.fast) == "Line('ilines')"
        else:
            assert s.fast is s.xline
            assert str(t.fast) == "Line('xlines')"

        with patch.object(
            StructuredTileSegy,
            "sorting",
            PropertyMock(return_value=TraceSortingFormat.UNKNOWN_SORTING),
        ):
            with pytest.raises(RuntimeError):
                t.fast

    @pytest.mark.parametrize("lines", ["ilines", "xlines"])
    @parametrize_tilesegy_segyfiles("t", "s", structured=True)
    def test_lines(self, lines: str, t: StructuredTileSegy, s: SegyFile) -> None:
        assert_equal_arrays(getattr(t, lines), getattr(s, lines))

    @pytest.mark.parametrize("line,lines", [("iline", "ilines"), ("xline", "xlines")])
    @parametrize_tilesegy_segyfiles("t", "s", structured=True)
    def test_line(
        self,
        line: str,
        lines: str,
        t: StructuredTileSegy,
        s: SegyFile,
    ) -> None:
        t_line, s_line = getattr(t, line), getattr(s, line)
        assert len(t_line) == len(s_line)

        i, j = np.sort(np.random.choice(getattr(s, lines), 2, replace=False))
        x = np.random.choice(s.offsets)

        # one line, first offset
        assert_equal_arrays(t_line[i], s_line[i])
        # one line, x offset
        assert_equal_arrays(t_line[i, x], s_line[i, x])

        for sl in iter_slices(i, j):
            # slice lines, first offset
            assert_equal_arrays(t_line[sl], segy_gen_to_array(s_line[sl]))
            # slice lines, x offset
            assert_equal_arrays(t_line[sl, x], segy_gen_to_array(s_line[sl, x]))

    @pytest.mark.parametrize("line,lines", [("iline", "ilines"), ("xline", "xlines")])
    @parametrize_tilesegy_segyfiles("t", "s", structured=True, multiple_offsets=True)
    def test_line_multiple_offsets(
        self,
        line: str,
        lines: str,
        t: StructuredTileSegy,
        s: SegyFile,
    ) -> None:
        t_line, s_line = getattr(t, line), getattr(s, line)
        i, j = np.sort(np.random.choice(getattr(s, lines), 2, replace=False))
        for sl2 in iter_slices(s.offsets[1], s.offsets[3]):
            # one line, slice offsets
            assert_equal_arrays(t_line[i, sl2], segy_gen_to_array(s_line[i, sl2]))

            for sl1 in iter_slices(i, j):
                # slice lines, slice offsets
                sliced_t = t_line[sl1, sl2]
                sliced_s = segy_gen_to_array(s_line[sl1, sl2])
                # segyio flattens the (lines, offsets) dimensions - unflatten them
                assert_equal_arrays(sliced_t, sliced_s.reshape(sliced_t.shape))

    @pytest.mark.parametrize("line,lines", [("iline", "ilines"), ("xline", "xlines")])
    @parametrize_tilesegy_segyfiles("t", "s", structured=True)
    def test_header_line(
        self,
        line: str,
        lines: str,
        t: StructuredTileSegy,
        s: SegyFile,
    ) -> None:
        t_line, s_line = getattr(t.header, line), getattr(s.header, line)
        assert len(t_line) == len(s_line)

        lines = getattr(s, lines)
        i = np.random.choice(lines)
        x = np.random.choice(s.offsets)

        # one line, first offset
        assert t_line[i] == stringify_keys(s_line[i])
        # one line, x offset
        assert t_line[i, x] == stringify_keys(s_line[i, x])

        for sl in slice(None, lines[2]), slice(lines[-2], None), slice(i, i + 2):
            # slice lines, first offset
            assert t_line[sl] == stringify_keys(s_line[sl])
            # slice lines, x offset
            assert t_line[sl, x] == stringify_keys(s_line[sl, x])

    @pytest.mark.parametrize("line,lines", [("iline", "ilines"), ("xline", "xlines")])
    @parametrize_tilesegy_segyfiles("t", "s", structured=True, multiple_offsets=True)
    def test_header_line_multiple_offsets(
        self,
        line: str,
        lines: str,
        t: StructuredTileSegy,
        s: SegyFile,
    ) -> None:
        t_line, s_line = getattr(t.header, line), getattr(s.header, line)
        lines = getattr(s, lines)
        i = np.random.choice(lines)
        o1, o2 = s.offsets[1], s.offsets[3]
        for sl2 in slice(None, o1), slice(o2, None), slice(o1, o2):
            # one line, slice offsets
            assert t_line[i, sl2] == stringify_keys(s_line[i, sl2])

            for sl1 in slice(None, lines[2]), slice(lines[-2], None), slice(i, i + 2):
                # slice lines, slice offsets
                # segyio flattens the (lines, offsets) dimensions
                t_line_sliced = [line for lines in t_line[sl1, sl2] for line in lines]
                s_line_sliced = stringify_keys(s_line[sl1, sl2])
                assert t_line_sliced == s_line_sliced

    @parametrize_tilesegy_segyfiles("t", "s", structured=True)
    def test_depth(self, t: StructuredTileSegy, s: SegyFile) -> None:
        # segyio doesn't currently support offset indexing for depth
        # https://github.com/equinor/segyio/issues/474
        i = np.random.randint(0, len(s.samples) // 2)
        j = np.random.randint(i + 1, len(s.samples))
        x = np.random.choice(s.offsets)

        # one line, x offset
        with pytest.raises(IndexError):
            t.depth[i, x]
        with pytest.raises(TypeError):
            s.depth_slice[i, x]

        for sl in iter_slices(i, j):
            # slice lines, x offset
            with pytest.raises(IndexError):
                t.depth[sl, x]
            with pytest.raises(TypeError):
                s.depth_slice[sl, x]

    @parametrize_tilesegy_segyfiles("t", "s", structured=True, multiple_offsets=True)
    def test_depth_multiple_offsets(self, t: StructuredTileSegy, s: SegyFile) -> None:
        # segyio doesn't currently support offset indexing for depth
        # https://github.com/equinor/segyio/issues/474
        i = np.random.randint(0, len(s.samples) // 2)
        j = np.random.randint(i + 1, len(s.samples))

        for sl2 in iter_slices(s.offsets[1], s.offsets[3]):
            # one depth, slice offsets
            with pytest.raises(IndexError):
                t.depth[i, sl2]
            with pytest.raises(TypeError):
                s.depth_slice[i, sl2]

            for sl1 in iter_slices(i, j):
                # slice depths, slice offsets
                with pytest.raises(IndexError):
                    t.depth[sl1, sl2]
                with pytest.raises(TypeError):
                    s.depth_slice[sl1, sl2]
