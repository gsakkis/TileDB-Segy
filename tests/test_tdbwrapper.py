from typing import Mapping, Union

import numpy as np
import pytest

import tiledb
from tiledb.segy.tdbwrapper import MultiAttrArrayWrapper, SingleAttrArrayWrapper

from .conftest import assert_equal_arrays, iter_slices


@pytest.fixture(scope="module")
def np_array() -> np.ndarray:
    shape = (6, 4, 2, 3)
    return np.arange(np.product(shape)).reshape(shape)


@pytest.fixture(scope="module", params=[True, False], ids=["single_attr", "multi_attr"])
def tdb_wrapper(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    np_array: np.ndarray,
) -> tiledb.Array:
    uri = str(tmp_path_factory.mktemp("array"))
    with tiledb.DenseArray.from_numpy(uri, np_array) as tdb:
        if request.param:
            yield SingleAttrArrayWrapper(tdb, "")
        else:
            yield MultiAttrArrayWrapper(tdb)


def assert_equal_array_or_dict(
    array: np.ndarray, array_or_dict: Union[np.ndarray, Mapping[str, np.ndarray]]
) -> None:
    if isinstance(array_or_dict, Mapping):
        assert list(array_or_dict.keys()) == [""]
        assert_equal_arrays(array, array_or_dict[""])
    else:
        assert_equal_arrays(array, array_or_dict)


class TestTiledbArrayWrapper:
    def test_too_many_indices_error(
        self, np_array: np.ndarray, tdb_wrapper: tiledb.Array
    ) -> None:
        for indexable in np_array, tdb_wrapper:
            with pytest.raises(IndexError):
                indexable[0, 0, 0, 0, 0]
            with pytest.raises(IndexError):
                indexable[0, :, 0, :, 0]

    def test_ellipsis(self, np_array: np.ndarray, tdb_wrapper: tiledb.Array) -> None:
        assert_equal_array_or_dict(np_array[...], tdb_wrapper[...])
        for indexable in np_array, tdb_wrapper:
            with pytest.raises(IndexError):
                indexable[..., ...]
            with pytest.raises(IndexError):
                indexable[..., 1, ...]
            with pytest.raises(IndexError):
                indexable[1, ..., 2, ...]

    def test_int(self, np_array: np.ndarray, tdb_wrapper: tiledb.Array) -> None:
        assert_equal_array_or_dict(np_array[0], tdb_wrapper[0])
        assert_equal_array_or_dict(np_array[-1], tdb_wrapper[-1])

    def test_slice(self, np_array: np.ndarray, tdb_wrapper: tiledb.Array) -> None:
        for sl in iter_slices(1, 5):
            assert_equal_array_or_dict(np_array[sl], tdb_wrapper[sl])

    def test_list(self, np_array: np.ndarray, tdb_wrapper: tiledb.Array) -> None:
        assert_equal_array_or_dict(np_array[[1, 4, 3]], tdb_wrapper[[1, 4, 3]])
        assert_equal_array_or_dict(
            np_array[np.ix_([1, 4, 3], [2, 3])], tdb_wrapper[[1, 4, 3], [2, 3]]
        )

    def test_tuple(self, np_array: np.ndarray, tdb_wrapper: tiledb.Array) -> None:
        assert_equal_array_or_dict(np_array[1, 2], tdb_wrapper[1, 2])
        assert_equal_array_or_dict(np_array[1, ...], tdb_wrapper[1, ...])
        assert_equal_array_or_dict(np_array[..., 1], tdb_wrapper[..., 1])
        assert_equal_array_or_dict(np_array[1, ..., 1], tdb_wrapper[1, ..., 1])

        assert_equal_array_or_dict(np_array[[1, 4, 3], 1], tdb_wrapper[[1, 4, 3], 1])
        assert_equal_array_or_dict(np_array[1, [2, 0]], tdb_wrapper[1, [2, 0]])
        assert_equal_array_or_dict(np_array[..., [2, 0]], tdb_wrapper[..., [2, 0]])

        for sl0 in iter_slices(1, 5):
            assert_equal_array_or_dict(np_array[sl0, 2], tdb_wrapper[sl0, 2])
            assert_equal_array_or_dict(np_array[sl0, ...], tdb_wrapper[sl0, ...])
            assert_equal_array_or_dict(np_array[sl0, ..., 1], tdb_wrapper[sl0, ..., 1])
            for sl3 in iter_slices(0, 2):
                assert_equal_array_or_dict(
                    np_array[sl0, 2, 1, sl3], tdb_wrapper[sl0, 2, 1, sl3]
                )
                assert_equal_array_or_dict(
                    np_array[sl0, 2, ..., sl3], tdb_wrapper[sl0, 2, ..., sl3]
                )
                assert_equal_array_or_dict(
                    np_array[sl0, ..., 1, sl3], tdb_wrapper[sl0, ..., 1, sl3]
                )
                assert_equal_array_or_dict(
                    np_array[sl0, ..., sl3], tdb_wrapper[sl0, ..., sl3]
                )
