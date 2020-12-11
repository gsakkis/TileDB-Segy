from collections import abc
from functools import singledispatch
from typing import Any, Iterable, List, Mapping, Union

import numpy as np
import pytest
import segyio.tools
from segyio import SegyFile, TraceField, TraceSortingFormat
from tiledb.libtiledb import TileDBError

import tiledb.segy
from tests.conftest import parametrize_segys
from tiledb.segy import Segy, StructuredSegy
from tiledb.segy.utils import Index


def assert_equal_arrays(
    a: Union[np.ndarray, np.number],
    b: Union[np.ndarray, np.number],
    reshape: bool = False,
) -> None:
    assert a.dtype == b.dtype
    if isinstance(a, np.number) or isinstance(b, np.number):
        assert isinstance(a, np.number) and isinstance(b, np.number)
        assert a == b
    elif reshape:
        assert a.ndim == b.ndim + 1
        assert a.shape[2:] == b.shape[1:]
        b = b.reshape(a.shape)
    else:
        assert a.ndim == b.ndim
        assert a.shape == b.shape
    np.testing.assert_array_equal(a, b)


def as_array(obj: Union[np.ndarray, Iterable[np.ndarray]]) -> np.ndarray:
    if not isinstance(obj, np.ndarray):
        obj = np.array(list(map(np.copy, obj)))
    return obj


@singledispatch
def stringify_keys(o: object) -> Any:
    raise TypeError(f"Cannot stringify_keys for {o.__class__}")


@stringify_keys.register(abc.Mapping)
def _stringify_keys_mapping(d: Mapping[int, int]) -> Mapping[str, int]:
    return {str(k): v for k, v in d.items()}


@stringify_keys.register(abc.Iterable)
def _stringify_keys_iter(s: Iterable[Mapping[int, int]]) -> List[Mapping[str, int]]:
    return list(map(stringify_keys, s))


def iter_slices(i: int, j: int) -> Iterable[slice]:
    return slice(None, None), slice(None, j), slice(i, None), slice(i, j)


class TestSegy:
    @parametrize_segys("t", "s")
    def test_sorting(self, t: Segy, s: SegyFile) -> None:
        assert t.sorting == s.sorting

    @parametrize_segys("t", "s")
    def test_bin(self, t: Segy, s: SegyFile) -> None:
        assert t.bin == stringify_keys(s.bin)

    @parametrize_segys("t", "s")
    def test_text(self, t: Segy, s: SegyFile) -> None:
        assert t.text == tuple(s.text)

    @parametrize_segys("t", "s")
    def test_samples(self, t: Segy, s: SegyFile) -> None:
        assert_equal_arrays(t.samples, s.samples)

    @parametrize_segys("t", "s")
    def test_close(self, t: Segy, s: SegyFile) -> None:
        t.bin
        t.close()
        with pytest.raises(TileDBError):
            t.bin

    @parametrize_segys("t", "s")
    def test_context_manager(self, t: Segy, s: SegyFile) -> None:
        with tiledb.segy.open(t.uri) as t2:
            t2.bin
        with pytest.raises(TileDBError):
            t2.bin

    @parametrize_segys("t", "s")
    def test_repr(self, t: Segy, s: SegyFile) -> None:
        if s.unstructured:
            assert repr(t) == f"Segy('{t.uri}')"
        else:
            assert repr(t) == f"StructuredSegy('{t.uri}')"

    @parametrize_segys("t", "s")
    def test_trace(self, t: Segy, s: SegyFile) -> None:
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
                assert_equal_arrays(t.trace[sl1], as_array(s.trace[sl1]))
                # slices traces, one sample
                assert_equal_arrays(t.trace[sl1, x], as_array(s.trace[sl1, x]))
                # slices traces, slice samples
                for sl2 in iter_slices(x, y):
                    assert_equal_arrays(t.trace[sl1, sl2], as_array(s.trace[sl1, sl2]))
            except NotImplementedError as ex:
                pytest.xfail(str(ex))

        with pytest.raises((IndexError, TypeError)):
            t.trace[float(i), 0]

    @parametrize_segys("t", "s")
    def test_header(self, t: Segy, s: SegyFile) -> None:
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

    @parametrize_segys("t", "s")
    def test_attributes(self, t: Segy, s: SegyFile) -> None:
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

    @parametrize_segys("t", "s")
    def test_depth_slice(self, t: Segy, s: SegyFile) -> None:
        assert len(t.depth_slice) == len(s.depth_slice)

        i = np.random.randint(0, len(s.samples) // 2)
        j = np.random.randint(i + 1, len(s.samples))
        # one depth
        assert_equal_arrays(t.depth_slice[i], s.depth_slice[i])
        # slice depths
        for sl in iter_slices(i, j):
            assert_equal_arrays(t.depth_slice[sl], as_array(s.depth_slice[sl]))

    @parametrize_segys("t", "s")
    def test_dt(self, t: Segy, s: SegyFile) -> None:
        assert t.dt() == segyio.tools.dt(s)
        assert t.dt(fallback=1234) == segyio.tools.dt(s, fallback_dt=1234)


class TestStructuredSegy:
    @parametrize_segys("t", "s", structured=True)
    def test_offsets(self, t: StructuredSegy, s: SegyFile) -> None:
        assert_equal_arrays(t.offsets, s.offsets)

    @parametrize_segys("t", "s", structured=True)
    def test_fast(self, t: StructuredSegy, s: SegyFile) -> None:
        if s.sorting == TraceSortingFormat.INLINE_SORTING:
            assert t.fast.name == "ilines"
        else:
            assert t.fast.name == "xlines"

    @parametrize_segys("t", "s", structured=True)
    def test_slow(self, t: StructuredSegy, s: SegyFile) -> None:
        if s.sorting == TraceSortingFormat.INLINE_SORTING:
            assert t.slow.name == "xlines"
        else:
            assert t.slow.name == "ilines"

    @pytest.mark.parametrize("lines", ["ilines", "xlines"])
    @parametrize_segys("t", "s", structured=True)
    def test_lines(self, lines: str, t: StructuredSegy, s: SegyFile) -> None:
        assert_equal_arrays(getattr(t, lines), getattr(s, lines))

    @pytest.mark.parametrize("line,lines", [("iline", "ilines"), ("xline", "xlines")])
    @parametrize_segys("t", "s", structured=True)
    def test_line(
        self,
        line: str,
        lines: str,
        t: StructuredSegy,
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
            assert_equal_arrays(t_line[sl], as_array(s_line[sl]))
            # slice lines, x offset
            assert_equal_arrays(t_line[sl, x], as_array(s_line[sl, x]))

    @pytest.mark.parametrize("line,lines", [("iline", "ilines"), ("xline", "xlines")])
    @parametrize_segys("t", "s", structured=True, multiple_offsets=True)
    def test_line_multiple_offsets(
        self,
        line: str,
        lines: str,
        t: StructuredSegy,
        s: SegyFile,
    ) -> None:
        t_line, s_line = getattr(t, line), getattr(s, line)
        i, j = np.sort(np.random.choice(getattr(s, lines), 2, replace=False))
        for sl2 in iter_slices(s.offsets[1], s.offsets[3]):
            # one line, slice offsets
            assert_equal_arrays(t_line[i, sl2], as_array(s_line[i, sl2]))

            for sl1 in iter_slices(i, j):
                # slice lines, slice offsets
                # segyio flattens the (lines, offsets) dimensions into one
                assert_equal_arrays(
                    t_line[sl1, sl2], as_array(s_line[sl1, sl2]), reshape=True
                )

    @pytest.mark.parametrize("line,lines", [("iline", "ilines"), ("xline", "xlines")])
    @parametrize_segys("t", "s", structured=True)
    def test_header_line(
        self,
        line: str,
        lines: str,
        t: StructuredSegy,
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
    @parametrize_segys("t", "s", structured=True, multiple_offsets=True)
    def test_header_line_multiple_offsets(
        self,
        line: str,
        lines: str,
        t: StructuredSegy,
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

    @parametrize_segys("t", "s", structured=True)
    def test_depth_slice(self, t: StructuredSegy, s: SegyFile) -> None:
        # segyio doesn't currently support offset indexing for depth_slice
        # https://github.com/equinor/segyio/issues/474
        i = np.random.randint(0, len(s.samples) // 2)
        j = np.random.randint(i + 1, len(s.samples))
        x = np.random.choice(s.offsets)

        # one line, x offset
        with pytest.raises(IndexError):
            t.depth_slice[i, x]
        with pytest.raises(TypeError):
            s.depth_slice[i, x]

        for sl in iter_slices(i, j):
            # slice lines, x offset
            with pytest.raises(IndexError):
                t.depth_slice[sl, x]
            with pytest.raises(TypeError):
                s.depth_slice[sl, x]

    @parametrize_segys("t", "s", structured=True, multiple_offsets=True)
    def test_depth_slice_many_offsets(self, t: StructuredSegy, s: SegyFile) -> None:
        # segyio doesn't currently support offset indexing for depth_slice
        # https://github.com/equinor/segyio/issues/474
        i = np.random.randint(0, len(s.samples) // 2)
        j = np.random.randint(i + 1, len(s.samples))

        for sl2 in iter_slices(s.offsets[1], s.offsets[3]):
            # one depth, slice offsets
            with pytest.raises(IndexError):
                t.depth_slice[i, sl2]
            with pytest.raises(TypeError):
                s.depth_slice[i, sl2]

            for sl1 in iter_slices(i, j):
                # slice depths, slice offsets
                with pytest.raises(IndexError):
                    t.depth_slice[sl1, sl2]
                with pytest.raises(TypeError):
                    s.depth_slice[sl1, sl2]

    @parametrize_segys("t", "s", structured=True)
    def test_gather(self, t: StructuredSegy, s: SegyFile) -> None:
        i = np.random.choice(s.ilines)
        i_slices = [
            slice(None, s.ilines[2]),
            slice(s.ilines[-2], None),
            slice(i, i + 2),
        ]
        x = np.random.choice(s.xlines)
        x_slices = [
            slice(None, s.xlines[3]),
            slice(s.xlines[-3], None),
            slice(x, x + 3),
        ]

        # single iline/xline
        self._assert_equal_gather(t, s, i, x)

        # single iline / slice xlines
        for sl in x_slices:
            self._assert_equal_gather(t, s, i, sl)

        # slice ilines / single xlines
        for sl in i_slices:
            self._assert_equal_gather(t, s, sl, x)

        # slice ilines/xlines
        for sl1 in i_slices:
            for sl2 in x_slices:
                self._assert_equal_gather(t, s, sl1, sl2)

    @parametrize_segys("t", "s", structured=True)
    def test_cube(self, t: StructuredSegy, s: SegyFile) -> None:
        assert_equal_arrays(t.cube(), segyio.tools.cube(s))

    def _assert_equal_gather(
        self, t: StructuredSegy, s: SegyFile, i: Index, x: Index
    ) -> None:
        o = s.offsets[0]
        # segyio flattens the (ilines, xlines) dimensions into one
        reshape = isinstance(i, slice) and isinstance(x, slice)
        assert_equal_arrays(t.gather[i, x], as_array(s.gather[i, x]), reshape)
        assert_equal_arrays(t.gather[i, x, o], as_array(s.gather[i, x, o]), reshape)
        if len(s.offsets) > 1:
            for sl in iter_slices(s.offsets[1], s.offsets[3]):
                assert_equal_arrays(
                    t.gather[i, x, sl], as_array(s.gather[i, x, sl]), reshape
                )
        else:
            assert_equal_arrays(t.gather[i, x, :], as_array(s.gather[i, x, :]), reshape)
