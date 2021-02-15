import numpy as np
import pytest
import segyio.tools
from segyio import SegyFile, TraceField
from tiledb.libtiledb import TileDBError

import tiledb.segy
from tiledb.segy import Segy

from .conftest import (
    assert_equal_arrays,
    iter_slices,
    parametrize_segys,
    stringify_keys,
)

collect = segyio.tools.collect


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

        # one trace, one sample (scalar)
        assert_equal_arrays(t.trace[i, x], s.trace[i, x])

        # one-slice trace, one sample: these are (1,1) arrays, not scalar
        assert_equal_arrays(t.trace[i : i + 1, x], collect(s.trace[i : i + 1, x]))
        assert_equal_arrays(
            t.trace[j : j - 1 : -1, x], collect(s.trace[j : j - 1 : -1, x])
        )

        # one trace, slice samples
        for sl in iter_slices(x, y):
            assert_equal_arrays(t.trace[i, sl], s.trace[i, sl])

        for sl1 in iter_slices(i, j):
            # slices traces, all samples
            assert_equal_arrays(t.trace[sl1], collect(s.trace[sl1]))
            # slices traces, one sample
            assert_equal_arrays(t.trace[sl1, x], collect(s.trace[sl1, x]))
            # slices traces, slice samples
            for sl2 in iter_slices(x, y):
                assert_equal_arrays(t.trace[sl1, sl2], collect(s.trace[sl1, sl2]))

    @parametrize_segys("t", "s")
    def test_header(self, t: Segy, s: SegyFile) -> None:
        assert len(t.header) == len(s.header)

        i = np.random.randint(0, s.tracecount // 2)
        assert t.header[i] == stringify_keys(s.header[i])

        slices = [
            slice(None, 3),
            slice(-3, None),
            slice(i, i + 3),
            slice(3, None, -1),
            slice(None, -3, -1),
            slice(i + 3, i, -1),
        ]
        for sl in slices:
            assert t.header[sl] == stringify_keys(s.header[sl])

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
            assert_equal_arrays(t_attrs[sl], s_attrs[sl])

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
            assert_equal_arrays(t.depth_slice[sl], collect(s.depth_slice[sl]))

    @parametrize_segys("t", "s")
    def test_dt(self, t: Segy, s: SegyFile) -> None:
        assert t.dt() == segyio.tools.dt(s)
        assert t.dt(fallback=1234) == segyio.tools.dt(s, fallback_dt=1234)
