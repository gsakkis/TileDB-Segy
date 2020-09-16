from typing import Iterator, Mapping

import numpy as np
import pytest
from segyio import SegyFile, TraceField
from tiledb.libtiledb import TileDBError

import tilesegy
from tests.conftest import parametrize_tilesegy_segyfiles, parametrize_tilesegys
from tilesegy import StructuredTileSegy, TileSegy


def assert_equal_arrays(a: np.ndarray, b: np.ndarray, reshape: bool = False) -> None:
    assert a.dtype == b.dtype
    if reshape:
        assert a.ndim == b.ndim + 1
        assert a.shape[0] * a.shape[1] == b.shape[0]
        assert a.shape[-2:] == b.shape[-2:]
        b = b.reshape(a.shape)
    else:
        assert a.ndim == b.ndim
        assert a.shape == b.shape
    np.testing.assert_array_equal(a, b)


def segy_gen_to_array(segy_gen: Iterator[np.ndarray]) -> np.ndarray:
    return np.array(list(map(np.copy, segy_gen)))


def stringify_keys(d: Mapping[int, int]) -> Mapping[str, int]:
    return {str(k): v for k, v in d.items()}


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

    @parametrize_tilesegys("t")
    def test_close(self, t: TileSegy) -> None:
        t.bin
        t.close()
        with pytest.raises(TileDBError):
            t.bin

    @parametrize_tilesegys("t")
    def test_context_manager(self, t: TileSegy) -> None:
        with tilesegy.open(t.uri) as t2:
            t2.bin
        with pytest.raises(TileDBError):
            t2.bin

    @parametrize_tilesegys("t", structured=False)
    def test_repr(self, t: TileSegy) -> None:
        assert repr(t) == f"TileSegy('{str(t.uri)}')"


class TestTileSegyTraces:
    @parametrize_tilesegy_segyfiles("t", "s", structured=False)
    def test_len(self, t: TileSegy, s: SegyFile) -> None:
        assert len(t.trace) == len(s.trace) == s.tracecount

    @parametrize_tilesegy_segyfiles("t", "s", structured=False)
    def test_get_one_trace_all_samples(self, t: TileSegy, s: SegyFile) -> None:
        i = np.random.randint(0, s.tracecount)
        assert_equal_arrays(t.trace[i], s.trace[i])

    @parametrize_tilesegy_segyfiles("t", "s", structured=False)
    def test_get_one_trace_one_sample(self, t: TileSegy, s: SegyFile) -> None:
        i = np.random.randint(0, s.tracecount)
        x = np.random.randint(0, len(s.samples))
        assert t.trace[i, x] == s.trace[i, x]

    @parametrize_tilesegy_segyfiles("t", "s", structured=False)
    def test_get_one_trace_slice_samples(self, t: TileSegy, s: SegyFile) -> None:
        i = np.random.randint(0, s.tracecount)
        x = np.random.randint(0, len(s.samples) // 2)
        y = np.random.randint(x + 1, len(s.samples))

        assert_equal_arrays(t.trace[i, :], s.trace[i, :])
        assert_equal_arrays(t.trace[i, x:], s.trace[i, x:])
        assert_equal_arrays(t.trace[i, :y], s.trace[i, :y])
        assert_equal_arrays(t.trace[i, x:y], s.trace[i, x:y])

    @parametrize_tilesegy_segyfiles("t", "s", structured=False)
    def test_get_slice_traces_all_samples(self, t: TileSegy, s: SegyFile) -> None:
        i = np.random.randint(0, s.tracecount // 2)
        j = np.random.randint(i + 1, s.tracecount)

        assert_equal_arrays(t.trace[:], segy_gen_to_array(s.trace[:]))
        assert_equal_arrays(t.trace[i:], segy_gen_to_array(s.trace[i:]))
        assert_equal_arrays(t.trace[:j], segy_gen_to_array(s.trace[:j]))
        assert_equal_arrays(t.trace[i:j], segy_gen_to_array(s.trace[i:j]))

    @parametrize_tilesegy_segyfiles("t", "s", structured=False)
    def test_get_slice_traces_one_sample(self, t: TileSegy, s: SegyFile) -> None:
        i = np.random.randint(0, s.tracecount // 2)
        j = np.random.randint(i + 1, s.tracecount)
        x = np.random.randint(0, len(s.samples))

        assert_equal_arrays(t.trace[:, x], np.fromiter(s.trace[:, x], s.dtype))
        assert_equal_arrays(t.trace[i:, x], np.fromiter(s.trace[i:, x], s.dtype))
        assert_equal_arrays(t.trace[:j, x], np.fromiter(s.trace[:j, x], s.dtype))
        assert_equal_arrays(t.trace[i:j, x], np.fromiter(s.trace[i:j, x], s.dtype))

    @parametrize_tilesegy_segyfiles("t", "s", structured=False)
    def test_get_slice_traces_slice_samples(self, t: TileSegy, s: SegyFile) -> None:
        i = np.random.randint(0, s.tracecount // 2)
        j = np.random.randint(i + 1, s.tracecount)

        x = np.random.randint(0, len(s.samples) // 2)
        y = np.random.randint(x + 1, len(s.samples))

        assert_equal_arrays(t.trace[:, :], segy_gen_to_array(s.trace[:, :]))
        assert_equal_arrays(t.trace[:, x:], segy_gen_to_array(s.trace[:, x:]))
        assert_equal_arrays(t.trace[:, :y], segy_gen_to_array(s.trace[:, :y]))
        assert_equal_arrays(t.trace[:, x:y], segy_gen_to_array(s.trace[:, x:y]))

        assert_equal_arrays(t.trace[i:, :], segy_gen_to_array(s.trace[i:, :]))
        assert_equal_arrays(t.trace[i:, x:], segy_gen_to_array(s.trace[i:, x:]))
        assert_equal_arrays(t.trace[i:, :y], segy_gen_to_array(s.trace[i:, :y]))
        assert_equal_arrays(t.trace[i:, x:y], segy_gen_to_array(s.trace[i:, x:y]))

        assert_equal_arrays(t.trace[:j, :], segy_gen_to_array(s.trace[:j, :]))
        assert_equal_arrays(t.trace[:j, x:], segy_gen_to_array(s.trace[:j, x:]))
        assert_equal_arrays(t.trace[:j, :y], segy_gen_to_array(s.trace[:j, :y]))
        assert_equal_arrays(t.trace[:j, x:y], segy_gen_to_array(s.trace[:j, x:y]))

        assert_equal_arrays(t.trace[i:j, :], segy_gen_to_array(s.trace[i:j, :]))
        assert_equal_arrays(t.trace[i:j, x:], segy_gen_to_array(s.trace[i:j, x:]))
        assert_equal_arrays(t.trace[i:j, :y], segy_gen_to_array(s.trace[i:j, :y]))
        assert_equal_arrays(t.trace[i:j, x:y], segy_gen_to_array(s.trace[i:j, x:y]))

    @parametrize_tilesegy_segyfiles("t", "s", structured=False)
    def test_headers(self, t: TileSegy, s: SegyFile) -> None:
        i = np.random.randint(0, s.tracecount // 2)
        j = i + 20

        assert len(t.trace.headers) == len(s.header)
        assert t.trace.headers[i] == stringify_keys(s.header[i])
        assert t.trace.headers[i:j] == list(map(stringify_keys, s.header[i:j]))

    @parametrize_tilesegy_segyfiles("t", "s", structured=False)
    def test_header(self, t: TileSegy, s: SegyFile) -> None:
        str_attr = "TraceNumber"
        t_attrs = t.trace.header(str_attr)
        s_attrs = s.attributes(getattr(TraceField, str_attr))

        i = np.random.randint(0, s.tracecount // 2)
        j = np.random.randint(i + 1, s.tracecount)

        assert len(t_attrs) == len(s_attrs)
        assert t_attrs[i] == s_attrs[i]
        assert t_attrs[i:] == s_attrs[i:].tolist()
        assert t_attrs[:j] == s_attrs[:j].tolist()
        assert t_attrs[i:j] == s_attrs[i:j].tolist()


class TestStructuredTileSegy:
    @parametrize_tilesegys("t", structured=True)
    def test_repr(self, t: StructuredTileSegy) -> None:
        assert repr(t) == f"StructuredTileSegy('{str(t.uri)}')"

    @parametrize_tilesegy_segyfiles("t", "s", structured=True)
    def test_offsets(self, t: StructuredTileSegy, s: SegyFile) -> None:
        assert_equal_arrays(t.offsets, s.offsets)

    @parametrize_tilesegy_segyfiles("t", "s", structured=True)
    def test_ilines(self, t: StructuredTileSegy, s: SegyFile) -> None:
        assert_equal_arrays(t.ilines, s.ilines)

    @parametrize_tilesegy_segyfiles("t", "s", structured=True)
    def test_xlines(self, t: StructuredTileSegy, s: SegyFile) -> None:
        assert_equal_arrays(t.xlines, s.xlines)

    @parametrize_tilesegy_segyfiles("t", "s", structured=True)
    def test_fast(self, t: StructuredTileSegy, s: SegyFile) -> None:
        if s.fast is s.iline:
            assert str(t.fast) == "Lines('ilines')"
        else:
            assert s.fast is s.xline
            assert str(t.fast) == "Lines('xlines')"


class TestStructuredTileSegyIlines:
    @parametrize_tilesegy_segyfiles("t", "s", structured=True)
    def test_len(self, t: StructuredTileSegy, s: SegyFile) -> None:
        assert len(t.iline) == len(s.iline)

    @parametrize_tilesegy_segyfiles("t", "s", structured=True)
    def test_get_one_line_one_offset(self, t: StructuredTileSegy, s: SegyFile) -> None:
        i = np.random.choice(s.ilines)
        x = np.random.choice(s.offsets)

        assert_equal_arrays(t.iline[i], s.iline[i])
        assert_equal_arrays(t.iline[i, x], s.iline[i, x])

    @parametrize_tilesegy_segyfiles("t", "s", structured=True)
    def test_get_slice_lines_one_offset(
        self, t: StructuredTileSegy, s: SegyFile
    ) -> None:
        i, j = np.sort(np.random.choice(s.ilines, 2, replace=False))
        x = np.random.choice(s.offsets)

        assert_equal_arrays(t.iline[:], segy_gen_to_array(s.iline[:]))
        assert_equal_arrays(t.iline[i:], segy_gen_to_array(s.iline[i:]))
        assert_equal_arrays(t.iline[:j], segy_gen_to_array(s.iline[:j]))
        assert_equal_arrays(t.iline[i:j], segy_gen_to_array(s.iline[i:j]))

        assert_equal_arrays(t.iline[:, x], segy_gen_to_array(s.iline[:, x]))
        assert_equal_arrays(t.iline[i:, x], segy_gen_to_array(s.iline[i:, x]))
        assert_equal_arrays(t.iline[:j, x], segy_gen_to_array(s.iline[:j, x]))
        assert_equal_arrays(t.iline[i:j, x], segy_gen_to_array(s.iline[i:j, x]))

    @parametrize_tilesegy_segyfiles("t", "s", structured=True)
    def test_get_one_line_slice_offsets(
        self, t: StructuredTileSegy, s: SegyFile
    ) -> None:
        if len(s.offsets) == 1:
            pytest.skip("single offset segy")

        i = np.random.choice(s.ilines)
        x, y = s.offsets[1], s.offsets[3]

        assert_equal_arrays(t.iline[i, :], segy_gen_to_array(s.iline[i, :]))
        assert_equal_arrays(t.iline[i, x:], segy_gen_to_array(s.iline[i, x:]))
        assert_equal_arrays(t.iline[i, :y], segy_gen_to_array(s.iline[i, :y]))
        assert_equal_arrays(t.iline[i, x:y], segy_gen_to_array(s.iline[i, x:y]))

    @parametrize_tilesegy_segyfiles("t", "s", structured=True)
    def test_get_slice_lines_slice_offsets(
        self, t: StructuredTileSegy, s: SegyFile
    ) -> None:
        if len(s.offsets) == 1:
            pytest.skip("single offset segy")

        i, j = np.sort(np.random.choice(s.ilines, 2, replace=False))
        x, y = s.offsets[1], s.offsets[3]

        assert_equal_arrays(
            t.iline[:, :], segy_gen_to_array(s.iline[:, :]), reshape=True
        )
        assert_equal_arrays(
            t.iline[:, x:], segy_gen_to_array(s.iline[:, x:]), reshape=True
        )
        assert_equal_arrays(
            t.iline[:, :y], segy_gen_to_array(s.iline[:, :y]), reshape=True
        )
        assert_equal_arrays(
            t.iline[:, x:y], segy_gen_to_array(s.iline[:, x:y]), reshape=True
        )

        assert_equal_arrays(
            t.iline[i:, :], segy_gen_to_array(s.iline[i:, :]), reshape=True
        )
        assert_equal_arrays(
            t.iline[i:, x:], segy_gen_to_array(s.iline[i:, x:]), reshape=True
        )
        assert_equal_arrays(
            t.iline[i:, :y], segy_gen_to_array(s.iline[i:, :y]), reshape=True
        )
        assert_equal_arrays(
            t.iline[i:, x:y], segy_gen_to_array(s.iline[i:, x:y]), reshape=True
        )

        assert_equal_arrays(
            t.iline[:j, :], segy_gen_to_array(s.iline[:j, :]), reshape=True
        )
        assert_equal_arrays(
            t.iline[:j, x:], segy_gen_to_array(s.iline[:j, x:]), reshape=True
        )
        assert_equal_arrays(
            t.iline[:j, :y], segy_gen_to_array(s.iline[:j, :y]), reshape=True
        )
        assert_equal_arrays(
            t.iline[:j, x:y], segy_gen_to_array(s.iline[:j, x:y]), reshape=True
        )

        assert_equal_arrays(
            t.iline[i:j, :], segy_gen_to_array(s.iline[i:j, :]), reshape=True
        )
        assert_equal_arrays(
            t.iline[i:j, x:], segy_gen_to_array(s.iline[i:j, x:]), reshape=True
        )
        assert_equal_arrays(
            t.iline[i:j, :y], segy_gen_to_array(s.iline[i:j, :y]), reshape=True
        )
        assert_equal_arrays(
            t.iline[i:j, x:y], segy_gen_to_array(s.iline[i:j, x:y]), reshape=True
        )


class TestStructuredTileSegyXlines:
    @parametrize_tilesegy_segyfiles("t", "s", structured=True)
    def test_len(self, t: StructuredTileSegy, s: SegyFile) -> None:
        assert len(t.xline) == len(s.xline)

    @parametrize_tilesegy_segyfiles("t", "s", structured=True)
    def test_get_one_line_one_offset(self, t: StructuredTileSegy, s: SegyFile) -> None:
        i = np.random.choice(s.xlines)
        x = np.random.choice(s.offsets)

        assert_equal_arrays(t.xline[i], s.xline[i])
        assert_equal_arrays(t.xline[i, x], s.xline[i, x])

    @parametrize_tilesegy_segyfiles("t", "s", structured=True)
    def test_get_slice_lines_one_offset(
        self, t: StructuredTileSegy, s: SegyFile
    ) -> None:
        i, j = np.sort(np.random.choice(s.xlines, 2, replace=False))
        x = np.random.choice(s.offsets)

        assert_equal_arrays(t.xline[:], segy_gen_to_array(s.xline[:]))
        assert_equal_arrays(t.xline[i:], segy_gen_to_array(s.xline[i:]))
        assert_equal_arrays(t.xline[:j], segy_gen_to_array(s.xline[:j]))
        assert_equal_arrays(t.xline[i:j], segy_gen_to_array(s.xline[i:j]))

        assert_equal_arrays(t.xline[:, x], segy_gen_to_array(s.xline[:, x]))
        assert_equal_arrays(t.xline[i:, x], segy_gen_to_array(s.xline[i:, x]))
        assert_equal_arrays(t.xline[:j, x], segy_gen_to_array(s.xline[:j, x]))
        assert_equal_arrays(t.xline[i:j, x], segy_gen_to_array(s.xline[i:j, x]))

    @parametrize_tilesegy_segyfiles("t", "s", structured=True)
    def test_get_one_line_slice_offsets(
        self, t: StructuredTileSegy, s: SegyFile
    ) -> None:
        if len(s.offsets) == 1:
            pytest.skip("single offset segy")

        i = np.random.choice(s.xlines)
        x, y = s.offsets[1], s.offsets[3]

        assert_equal_arrays(t.xline[i, :], segy_gen_to_array(s.xline[i, :]))
        assert_equal_arrays(t.xline[i, x:], segy_gen_to_array(s.xline[i, x:]))
        assert_equal_arrays(t.xline[i, :y], segy_gen_to_array(s.xline[i, :y]))
        assert_equal_arrays(t.xline[i, x:y], segy_gen_to_array(s.xline[i, x:y]))

    @parametrize_tilesegy_segyfiles("t", "s", structured=True)
    def test_get_slice_lines_slice_offsets(
        self, t: StructuredTileSegy, s: SegyFile
    ) -> None:
        if len(s.offsets) == 1:
            pytest.skip("single offset segy")

        i, j = np.sort(np.random.choice(s.xlines, 2, replace=False))
        x, y = s.offsets[1], s.offsets[3]

        assert_equal_arrays(
            t.xline[:, :], segy_gen_to_array(s.xline[:, :]), reshape=True
        )
        assert_equal_arrays(
            t.xline[:, x:], segy_gen_to_array(s.xline[:, x:]), reshape=True
        )
        assert_equal_arrays(
            t.xline[:, :y], segy_gen_to_array(s.xline[:, :y]), reshape=True
        )
        assert_equal_arrays(
            t.xline[:, x:y], segy_gen_to_array(s.xline[:, x:y]), reshape=True
        )

        assert_equal_arrays(
            t.xline[i:, :], segy_gen_to_array(s.xline[i:, :]), reshape=True
        )
        assert_equal_arrays(
            t.xline[i:, x:], segy_gen_to_array(s.xline[i:, x:]), reshape=True
        )
        assert_equal_arrays(
            t.xline[i:, :y], segy_gen_to_array(s.xline[i:, :y]), reshape=True
        )
        assert_equal_arrays(
            t.xline[i:, x:y], segy_gen_to_array(s.xline[i:, x:y]), reshape=True
        )

        assert_equal_arrays(
            t.xline[:j, :], segy_gen_to_array(s.xline[:j, :]), reshape=True
        )
        assert_equal_arrays(
            t.xline[:j, x:], segy_gen_to_array(s.xline[:j, x:]), reshape=True
        )
        assert_equal_arrays(
            t.xline[:j, :y], segy_gen_to_array(s.xline[:j, :y]), reshape=True
        )
        assert_equal_arrays(
            t.xline[:j, x:y], segy_gen_to_array(s.xline[:j, x:y]), reshape=True
        )

        assert_equal_arrays(
            t.xline[i:j, :], segy_gen_to_array(s.xline[i:j, :]), reshape=True
        )
        assert_equal_arrays(
            t.xline[i:j, x:], segy_gen_to_array(s.xline[i:j, x:]), reshape=True
        )
        assert_equal_arrays(
            t.xline[i:j, :y], segy_gen_to_array(s.xline[i:j, :y]), reshape=True
        )
        assert_equal_arrays(
            t.xline[i:j, x:y], segy_gen_to_array(s.xline[i:j, x:y]), reshape=True
        )


class TestStructuredTileSegyDepths:
    @parametrize_tilesegy_segyfiles("t", "s", structured=True)
    def test_len(self, t: StructuredTileSegy, s: SegyFile) -> None:
        assert len(t.depth) == len(s.depth_slice)

    @parametrize_tilesegy_segyfiles("t", "s", structured=True)
    def test_get_one_line(self, t: StructuredTileSegy, s: SegyFile) -> None:
        i = np.random.randint(0, len(s.samples))
        assert_equal_arrays(t.depth[i], s.depth_slice[i])

    @parametrize_tilesegy_segyfiles("t", "s", structured=True)
    def test_get_slice_lines(self, t: StructuredTileSegy, s: SegyFile) -> None:
        i = np.random.randint(0, len(s.samples) // 2)
        j = np.random.randint(i + 1, len(s.samples))

        assert_equal_arrays(t.depth[:], segy_gen_to_array(s.depth_slice[:]))
        assert_equal_arrays(t.depth[i:], segy_gen_to_array(s.depth_slice[i:]))
        assert_equal_arrays(t.depth[:j], segy_gen_to_array(s.depth_slice[:j]))
        assert_equal_arrays(t.depth[i:j], segy_gen_to_array(s.depth_slice[i:j]))
