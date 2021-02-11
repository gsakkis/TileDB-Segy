from typing import Any, Optional, Tuple, Type

import numpy as np
import pytest

from tiledb.segy.utils import ensure_slice


class TestEnsureSlice:
    @pytest.mark.parametrize("obj", [None, 3.14, "hey"])
    def test_invalid_type(self, obj: Any) -> None:
        pytest.raises(TypeError, ensure_slice, obj)

    @pytest.mark.parametrize("start", [1, None])
    @pytest.mark.parametrize("stop", [10, None])
    @pytest.mark.parametrize("step", [2, -1, None])
    def test_slice(
        self, start: Optional[int], stop: Optional[int], step: Optional[int]
    ) -> None:
        s = slice(start, stop, step)
        assert ensure_slice(s) is s

    def test_int_or_singleton_array(self) -> None:
        assert ensure_slice(5) == slice(5, 6)
        assert ensure_slice(np.array(5)) == slice(5, 6)
        assert ensure_slice(np.array([5])) == slice(5, 6)

    @pytest.mark.parametrize("dtype", [float, np.float32, np.bool_])
    def test_invalid_array_dtype(self, dtype: Type[Any]) -> None:
        a = np.zeros(10, dtype=dtype)
        pytest.raises(ValueError, ensure_slice, a)

    @pytest.mark.parametrize("shape", [(4, 5), (1, 5), (5, 1), (1, 2, 3)])
    def test_invalid_array_ndim(self, shape: Tuple[int, ...]) -> None:
        a = np.zeros(shape, dtype=np.int32)
        pytest.raises(ValueError, ensure_slice, a)

    def test_invalid_array_empty(self) -> None:
        pytest.raises(ValueError, ensure_slice, np.array([], int))

    @pytest.mark.parametrize("a", [(1, 0, -1), (7, 4, 1, -2)])
    def test_invalid_array_negative(self, a: Tuple[int, ...]) -> None:
        pytest.raises(ValueError, ensure_slice, np.array(a, int))

    @pytest.mark.parametrize("a", [(1, 5, 3, 2), (10, 5, 7, 8)])
    def test_invalid_array_non_monotonic(self, a: Tuple[int, ...]) -> None:
        pytest.raises(ValueError, ensure_slice, np.array(a, int))

    @pytest.mark.parametrize("a", [(1, 5, 7, 9), (10, 7, 6, 5)])
    def test_invalid_array_non_range(self, a: Tuple[int, ...]) -> None:
        pytest.raises(ValueError, ensure_slice, np.array(a, int))

    @pytest.mark.parametrize("a", [(2, 3, 4), (1, 5, 9, 13), (7, 4, 1), (6, 3, 0)])
    def test_valid_array(self, a: Tuple[int, ...]) -> None:
        a = np.array(a, int)
        s = ensure_slice(a)
        assert np.array_equal(np.arange(max(a) + 1)[s], a)
