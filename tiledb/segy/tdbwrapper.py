"""
TileDB array wrappers with Numpy-like indexing semantics, either for a single
attribute (`SingleAttrArrayWrapper`) or for many (`MultiAttrArrayWrapper`).

Such wrapper instances can be indexed by:
- an integer
- a slice (negative start/stop/step are allowed)
- a list of integers
- ellipsis
- a tuple of any of the above, one for each dimension.

**Note**: In case of multiple list indices, their cross product is indexed. This is unlike
Numpy sequence-like indexing that expects all sequences to have (or can be broadcast to)
the same shape. That is, `a[[1,3], [2,5]]` for a TileDB array wrapper is equivalent to
`a[np.ix_([1,3], [2,5])]` for a Numpy array.
"""

from typing import Dict, Union

import numpy as np
import wrapt

import tiledb

from .types import Ellipsis, ExtendedIndex, ExtendedIndices


class SingleAttrArrayWrapper(wrapt.ObjectProxy):
    """
    Provides Numpy-like indexing semantics across a single attribute of a TileDB array.
    """

    def __init__(self, array: tiledb.Array, attr: str):
        super().__init__(array)
        self._self_attr = attr

    def __getitem__(self, indices: Union[ExtendedIndex, ExtendedIndices]) -> np.ndarray:
        if not isinstance(indices, tuple):
            indices = (indices,)
        query = self.query(attrs=(self._self_attr,))
        return _np_multi_index(self, query, *indices)[self._self_attr]


class MultiAttrArrayWrapper(wrapt.ObjectProxy):
    """
    Provides Numpy-like indexing semantics across all attributes of a TileDB array.
    """

    def __init__(self, array: tiledb.Array, *attrs: str):
        super().__init__(array)
        self._self_attrs = attrs

    def __getitem__(
        self, indices: Union[ExtendedIndex, ExtendedIndices]
    ) -> Dict[str, np.ndarray]:
        if not isinstance(indices, tuple):
            indices = (indices,)
        query = self.query(attrs=self._self_attrs or None)
        return _np_multi_index(self, query, *indices)


def _np_multi_index(
    array: tiledb.Array,
    query: tiledb.libtiledb.Query,
    *_indices: ExtendedIndex,
) -> Dict[str, np.ndarray]:
    missing = array.ndim - len(_indices)
    if missing < 0:
        raise IndexError(
            f"too many indices for array: array is {array.ndim}-dimensional, "
            f"but {len(_indices)} were indexed"
        )

    indices = list(_indices)
    try:
        e_idx = indices.index(Ellipsis)
    except ValueError:
        # extend full slices (:) for missing indices
        indices.extend(slice(None) for _ in range(missing))
    else:
        if indices.count(Ellipsis) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        # replace (...) with missing+1 full slices (:)
        indices[e_idx : e_idx + 1] = (slice(None) for _ in range(missing + 1))

    # adapt indices to TileDB’s range semantics
    int_axis = []
    assert len(indices) == len(array.shape)
    for i, (index, dim_size) in enumerate(zip(indices, array.shape)):
        if isinstance(index, (int, np.integer)):
            int_axis.append(i)
            if index < 0:
                indices[i] += dim_size
        elif isinstance(index, slice):
            if index.step in (None, 1):
                start, stop, step = index.indices(dim_size)
                # multi_index slice ranges are inclusive of the stop point
                indices[i] = slice(start, stop - 1)
            else:
                # Stepped slice ranges are not currently supported
                # convert to explicit list of indices
                indices[i] = list(range(dim_size)[index])

    result_dict: Dict[str, np.ndarray] = query.multi_index[tuple(indices)]
    if int_axis:
        # squeeze out the dimensions indexed by integers
        result_dict = {k: v.squeeze(tuple(int_axis)) for k, v in result_dict.items()}
    return result_dict
