import numpy as np
import pytest

from tiledb.segy.utils import LabelIndexer, ensure_slice

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


class TestLabelIndexer:
    def test_invalid_array_duplicates(self):
        pytest.raises(ValueError, LabelIndexer, np.array([1, 2, 3, 2]))

    @pytest.mark.parametrize("dtype", [float, np.float32, np.bool_])
    def test_invalid_array_dtype(self, dtype):
        pytest.raises(ValueError, LabelIndexer, np.array([1, 2, 3], dtype))

    @pytest.mark.parametrize("obj", [3.1, "3", None, (1, 2)])
    def test_get_invalid_type(self, obj):
        indexer = LabelIndexer(np.array([1, 2, 3]))
        with pytest.raises(TypeError):
            indexer[obj]

    @pytest.mark.parametrize("dtype", int_types)
    def test_get_int(self, dtype):
        array = np.array([10, 21, 32], dtype)
        indexer = LabelIndexer(array)
        assert indexer[array[0]] == 0
        assert indexer[array[1]] == 1
        assert indexer[array[2]] == 2
        with pytest.raises(ValueError):
            indexer[dtype(42)]

    @pytest.mark.parametrize("dtype", int_types)
    def test_get_slice_increasing_array(self, dtype):
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

        assert indexer[20:5:-1] == slice(2, -1, -1)
        assert indexer[21:5:-1] == slice(3, -1, -1)
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
    def test_get_slice_decreasing_array(self, dtype):
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

    @pytest.mark.parametrize("cls", int_types)
    def test_int_or_singleton_array(self, cls):
        i = cls(5)
        assert ensure_slice(i) == slice(5, 6)
        assert ensure_slice(np.array(i)) == slice(5, 6)
        assert ensure_slice(np.array([i])) == slice(5, 6)

    @pytest.mark.parametrize("dtype", [float, np.float32, np.bool_])
    def test_invalid_array_dtype(self, dtype):
        a = np.zeros(10, dtype=dtype)
        pytest.raises(ValueError, ensure_slice, a)

    @pytest.mark.parametrize("shape", [(4, 5), (1, 5), (5, 1), (1, 2, 3)])
    def test_invalid_array_ndim(self, shape):
        a = np.zeros(shape, dtype=np.int32)
        pytest.raises(ValueError, ensure_slice, a)

    def test_invalid_array_empty(self):
        pytest.raises(ValueError, ensure_slice, np.array([], int))

    @pytest.mark.parametrize("a", [(1, 5, 3, 2), (10, 5, 7, 8)])
    def test_invalid_array_non_monotonic(self, a):
        pytest.raises(ValueError, ensure_slice, np.array(a, int))

    @pytest.mark.parametrize("a", [(1, 5, 7, 9), (10, 7, 6, 5)])
    def test_invalid_array_non_range(self, a):
        pytest.raises(ValueError, ensure_slice, np.array(a, int))

    @pytest.mark.parametrize("a", [(2, 3, 4), (1, 5, 9, 13), (1, 0, -1), (7, 4, 1, -2)])
    def test_valid_array(self, a):
        a = np.array(a, int)
        s = ensure_slice(a)
        assert np.array_equal(np.arange(s.start, s.stop, s.step), a)
