from typing import Iterator, Mapping

import numpy as np
import pytest
from segyio import SegyFile, TraceField
from tiledb.libtiledb import TileDBError

from tests.conftest import parametrize_tilesegy_segyfiles, parametrize_tilesegys
from tilesegy.api import TileSegy


def assert_equal_arrays(a: np.ndarray, b: np.ndarray) -> None:
    np.testing.assert_array_equal(a, b)
    assert a.dtype is b.dtype


def segy_gen_to_array(segy_gen: Iterator[np.ndarray]) -> np.ndarray:
    return np.array([a.copy() for a in segy_gen])


def stringify_keys(d: Mapping[int, int]) -> Mapping[str, int]:
    return {str(k): v for k, v in d.items()}


class TestTileSegy:
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
        with TileSegy(t.uri) as t2:
            t2.bin
        with pytest.raises(TileDBError):
            t2.bin

    @parametrize_tilesegys("t")
    def test_repr(self, t: TileSegy) -> None:
        assert repr(t) == f"TileSegy('{str(t.uri)}')"


class TestTileSegyTraces:
    @parametrize_tilesegy_segyfiles("t", "s", structured=False)
    def test_len(self, t: TileSegy, s: SegyFile) -> None:
        assert len(t.traces) == len(s.trace) == s.tracecount

    @parametrize_tilesegy_segyfiles("t", "s", structured=False)
    def test_get_one_trace_all_samples(self, t: TileSegy, s: SegyFile) -> None:
        i = np.random.randint(0, s.tracecount)
        assert_equal_arrays(t.traces[i], s.trace[i])

    @parametrize_tilesegy_segyfiles("t", "s", structured=False)
    def test_get_one_trace_one_sample(self, t: TileSegy, s: SegyFile) -> None:
        i = np.random.randint(0, s.tracecount)
        x = np.random.randint(0, len(s.samples))
        assert t.traces[i, x] == s.trace[i, x]

    @parametrize_tilesegy_segyfiles("t", "s", structured=False)
    def test_get_one_trace_slice_samples(self, t: TileSegy, s: SegyFile) -> None:
        i = np.random.randint(0, s.tracecount)
        x = np.random.randint(0, len(s.samples) // 2)
        y = np.random.randint(x + 1, len(s.samples))

        assert_equal_arrays(t.traces[i, x:], s.trace[i, x:])
        assert_equal_arrays(t.traces[i, :y], s.trace[i, :y])
        assert_equal_arrays(t.traces[i, x:y], s.trace[i, x:y])

    @parametrize_tilesegy_segyfiles("t", "s", structured=False)
    def test_get_slice_traces_all_samples(self, t: TileSegy, s: SegyFile) -> None:
        i = np.random.randint(0, s.tracecount // 2)
        j = np.random.randint(i + 1, s.tracecount)

        assert_equal_arrays(t.traces[i:], segy_gen_to_array(s.trace[i:]))
        assert_equal_arrays(t.traces[:j], segy_gen_to_array(s.trace[:j]))
        assert_equal_arrays(t.traces[i:j], segy_gen_to_array(s.trace[i:j]))

    @parametrize_tilesegy_segyfiles("t", "s", structured=False)
    def test_get_slice_traces_one_sample(self, t: TileSegy, s: SegyFile) -> None:
        i = np.random.randint(0, s.tracecount // 2)
        j = np.random.randint(i + 1, s.tracecount)
        x = np.random.randint(0, len(s.samples))

        assert_equal_arrays(t.traces[i:, x], np.fromiter(s.trace[i:, x], s.dtype))
        assert_equal_arrays(t.traces[:j, x], np.fromiter(s.trace[:j, x], s.dtype))
        assert_equal_arrays(t.traces[i:j, x], np.fromiter(s.trace[i:j, x], s.dtype))

    @parametrize_tilesegy_segyfiles("t", "s", structured=False)
    def test_get_slice_traces_slice_samples(self, t: TileSegy, s: SegyFile) -> None:
        i = np.random.randint(0, s.tracecount // 2)
        j = np.random.randint(i + 1, s.tracecount)

        x = np.random.randint(0, len(s.samples) // 2)
        y = np.random.randint(x + 1, len(s.samples))

        assert_equal_arrays(t.traces[i:, x:], segy_gen_to_array(s.trace[i:, x:]))
        assert_equal_arrays(t.traces[i:, :y], segy_gen_to_array(s.trace[i:, :y]))
        assert_equal_arrays(t.traces[i:, x:y], segy_gen_to_array(s.trace[i:, x:y]))

        assert_equal_arrays(t.traces[:j, x:], segy_gen_to_array(s.trace[:j, x:]))
        assert_equal_arrays(t.traces[:j, :y], segy_gen_to_array(s.trace[:j, :y]))
        assert_equal_arrays(t.traces[:j, x:y], segy_gen_to_array(s.trace[:j, x:y]))

        assert_equal_arrays(t.traces[i:j, x:], segy_gen_to_array(s.trace[i:j, x:]))
        assert_equal_arrays(t.traces[i:j, :y], segy_gen_to_array(s.trace[i:j, :y]))
        assert_equal_arrays(t.traces[i:j, x:y], segy_gen_to_array(s.trace[i:j, x:y]))

    @parametrize_tilesegy_segyfiles("t", "s", structured=False)
    def test_headers(self, t: TileSegy, s: SegyFile) -> None:
        i = np.random.randint(0, s.tracecount // 2)
        j = i + 20

        assert len(t.traces.headers) == len(s.header)
        assert t.traces.headers[i] == stringify_keys(s.header[i])
        assert t.traces.headers[i:j] == list(map(stringify_keys, s.header[i:j]))

    @parametrize_tilesegy_segyfiles("t", "s", structured=False)
    def test_header(self, t: TileSegy, s: SegyFile) -> None:
        str_attr = "TraceNumber"
        t_attrs = t.traces.header(str_attr)
        s_attrs = s.attributes(getattr(TraceField, str_attr))

        i = np.random.randint(0, s.tracecount // 2)
        j = np.random.randint(i + 1, s.tracecount)

        assert len(t_attrs) == len(s_attrs)
        assert t_attrs[i] == s_attrs[i]
        assert t_attrs[i:] == s_attrs[i:].tolist()
        assert t_attrs[:j] == s_attrs[:j].tolist()
        assert t_attrs[i:j] == s_attrs[i:j].tolist()
