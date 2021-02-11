from typing import Any, Type, Union

import numpy as np
import pytest
import segyio.tools
from segyio import SegyFile, TraceSortingFormat

from tiledb.segy.structured import Index, LabelIndexer, StructuredSegy

from .conftest import (
    assert_equal_arrays,
    iter_slices,
    parametrize_segys,
    stringify_keys,
)

Int = Union[int, np.integer]
int_types = (
    int,
    np.int64,
    np.int32,
    np.int16,
    np.int8,
    np.uint64,
    np.uint32,
    np.uint16,
    np.uint8,
)
collect = segyio.tools.collect


class TestLabelIndexer:
    def test_invalid_array_duplicates(self) -> None:
        pytest.raises(ValueError, LabelIndexer, np.array([1, 2, 3, 2]))

    @pytest.mark.parametrize("dtype", [float, np.float32, np.bool_])
    def test_invalid_array_dtype(self, dtype: Type[Any]) -> None:
        pytest.raises(ValueError, LabelIndexer, np.array([1, 2, 3], dtype))

    @pytest.mark.parametrize("obj", [3.1, "3", None, (1, 2)])
    def test_get_invalid_type(self, obj: Any) -> None:
        indexer = LabelIndexer(np.array([1, 2, 3]))
        with pytest.raises(TypeError):
            indexer[obj]

    @pytest.mark.parametrize("dtype", int_types)
    def test_get_int(self, dtype: Type[Int]) -> None:
        array = np.array([10, 21, 32], dtype)
        indexer = LabelIndexer(array)
        assert indexer[array[0]] == 0
        assert indexer[array[1]] == 1
        assert indexer[array[2]] == 2
        with pytest.raises(ValueError):
            indexer[dtype(42)]

    @pytest.mark.parametrize("dtype", int_types)
    def test_get_slice_increasing_array(self, dtype: Type[Int]) -> None:
        array = np.array([10, 13, 17, 21, 26, 29, 31], dtype)
        indexer = LabelIndexer(array)

        assert indexer[5:21] == slice(0, 3, 1)
        assert indexer[5:22] == slice(0, 4, 1)
        assert indexer[11:21] == slice(1, 3, 1)
        assert indexer[13:23:2] == slice(1, 4, 1)
        assert indexer[13:33:8] == slice(1, 6, 2)

        label_slice = slice(10, 32, 7)
        assert np.array_equal(
            indexer._label_slice_to_indices(label_slice), np.array([0, 2, 6])
        )
        with pytest.raises(ValueError):
            indexer[label_slice]

        assert indexer[20:5:-1] == slice(2, None, -1)
        assert indexer[21:5:-1] == slice(3, None, -1)
        assert indexer[21:12:-1] == slice(3, 0, -1)
        assert indexer[19:12:-2] == slice(2, 0, -1)
        assert indexer[29:10:-8] == slice(5, 0, -2)

        label_slice = slice(31, 20, -5)
        assert np.array_equal(
            indexer._label_slice_to_indices(label_slice), np.array([6, 4, 3])
        )
        with pytest.raises(ValueError):
            indexer[label_slice]

    @pytest.mark.parametrize("dtype", int_types)
    def test_get_slice_decreasing_array(self, dtype: Type[Int]) -> None:
        array = np.array([31, 29, 26, 21, 17, 13, 10], dtype)
        indexer = LabelIndexer(array)

        assert indexer[5:21] == slice(6, 3, -1)
        assert indexer[5:22] == slice(6, 2, -1)
        assert indexer[11:21] == slice(5, 3, -1)
        assert indexer[13:23:2] == slice(5, 2, -1)
        assert indexer[13:33:8] == slice(5, 0, -2)

        label_slice = slice(10, 32, 7)
        assert np.array_equal(
            indexer._label_slice_to_indices(label_slice), np.array([6, 4, 0])
        )
        with pytest.raises(ValueError):
            indexer[label_slice]

        assert indexer[20:5:-1] == slice(4, 7, 1)
        assert indexer[21:5:-1] == slice(3, 7, 1)
        assert indexer[21:12:-1] == slice(3, 6, 1)
        assert indexer[19:12:-2] == slice(4, 6, 1)
        assert indexer[29:10:-8] == slice(1, 6, 2)

        label_slice = slice(31, 20, -5)
        assert np.array_equal(
            indexer._label_slice_to_indices(label_slice), np.array([0, 2, 3])
        )
        with pytest.raises(ValueError):
            indexer[label_slice]


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
            assert_equal_arrays(t_line[sl], collect(s_line[sl]))
            # slice lines, x offset
            assert_equal_arrays(t_line[sl, x], collect(s_line[sl, x]))

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
            assert_equal_arrays(t_line[i, sl2], collect(s_line[i, sl2]))

            for sl1 in iter_slices(i, j):
                # slice lines, slice offsets
                # segyio flattens the (lines, offsets) dimensions into one
                assert_equal_arrays(
                    t_line[sl1, sl2], collect(s_line[sl1, sl2]), reshape=True
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
        assert_equal_arrays(t.gather[i, x], collect(s.gather[i, x]), reshape)
        assert_equal_arrays(t.gather[i, x, o], collect(s.gather[i, x, o]), reshape)
        if len(s.offsets) > 1:
            for sl in iter_slices(s.offsets[1], s.offsets[3]):
                # TODO: remove this condition when https://github.com/equinor/segyio/pull/500 is merged
                if sl.start is not None or sl.step in (None, 1):
                    assert_equal_arrays(
                        t.gather[i, x, sl], collect(s.gather[i, x, sl]), reshape
                    )
        else:
            assert_equal_arrays(t.gather[i, x, :], collect(s.gather[i, x, :]), reshape)
