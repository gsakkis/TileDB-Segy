import numpy as np
import pytest

from tilesegy.utils import ensure_slice


class TestEnsureSlice:
    @pytest.mark.parametrize("obj", [None, 3.14, "hey"])
    def test_invalid_type(self, obj):
        pytest.raises(TypeError, ensure_slice, obj)

    @pytest.mark.parametrize("start", [1, None])
    @pytest.mark.parametrize("stop", [10, None])
    @pytest.mark.parametrize("step", [2, -1, None])
    def test_slice(self, start, stop, step):
        s = slice(start, stop, step)
        assert ensure_slice(s) is s

    @pytest.mark.parametrize(
        "cls",
        [
            int,
            np.int64,
            np.int32,
            np.int16,
            np.int8,
            np.uint64,
            np.uint32,
            np.uint16,
            np.uint8,
        ],
    )
    def test_int_or_singleton_array(self, cls):
        i = cls(5)
        assert ensure_slice(i) == slice(5, 6)
        assert ensure_slice(np.array(i)) == slice(5, 6)
        assert ensure_slice(np.array([i])) == slice(5, 6)

    @pytest.mark.parametrize("dtype", [float, np.float32, np.bool_])
    def test_invalid_array_dtype(self, dtype):
        a = np.zeros(10, dtype=dtype)
        assert pytest.raises(ValueError, ensure_slice, a)

    @pytest.mark.parametrize("shape", [(4, 5), (1, 5), (5, 1), (1, 2, 3)])
    def test_invalid_array_ndim(self, shape):
        a = np.zeros(shape, dtype=np.int32)
        assert pytest.raises(ValueError, ensure_slice, a)

    def test_invalid_array_empty(self):
        assert pytest.raises(ValueError, ensure_slice, np.array([], int))

    @pytest.mark.parametrize("a", [(1, 5, 3, 2), (10, 5, 7, 8)])
    def test_invalid_array_non_monotonic(self, a):
        assert pytest.raises(ValueError, ensure_slice, np.array(a, int))

    @pytest.mark.parametrize("a", [(1, 5, 7, 9), (10, 7, 6, 5)])
    def test_invalid_array_non_fixed_step(self, a):
        assert pytest.raises(NotImplementedError, ensure_slice, np.array(a, int))

    @pytest.mark.parametrize("a", [(2, 3, 4), (1, 5, 9, 13), (1, 0, -1), (7, 4, 1, -2)])
    def test_valid_array(self, a):
        a = np.array(a, int)
        s = ensure_slice(a)
        assert np.array_equal(np.arange(s.start, s.stop, s.step), a)
