__all__ = ["LabelIndexer", "ensure_slice"]

from functools import singledispatch
from typing import Union

import numpy as np

from ._singledispatchmethod import singledispatchmethod  # type: ignore


class LabelIndexer:
    def __init__(self, labels: np.ndarray):
        if not issubclass(labels.dtype.type, np.integer):
            raise ValueError("labels should be integers")
        if len(np.unique(labels)) != len(labels):
            raise ValueError(f"labels should not contain duplicates: {labels}")
        self._labels = labels
        self._min_label = labels.min()
        self._max_label = labels.max() + 1
        self._sorter = labels.argsort()

    @singledispatchmethod
    def __getitem__(self, label: int) -> int:
        indices = np.flatnonzero(label == self._labels)
        assert indices.size <= 1, indices
        if indices.size == 0:
            raise ValueError(f"{label} is not in labels")
        return int(indices[0])

    @__getitem__.register(slice)
    def _get_slice(self, label_slice: slice) -> slice:
        start, stop, step = label_slice.start, label_slice.stop, label_slice.step
        min_label = self._min_label
        if step is None or step > 0:  # increasing step
            if start is None or start < min_label:
                start = min_label
        else:  # decreasing step
            if stop is None or stop < min_label - 1:
                stop = min_label - 1

        label_range = np.arange(*slice(start, stop, step).indices(self._max_label))
        indices = self._sorter[
            self._labels.searchsorted(label_range, sorter=self._sorter)
        ]
        indices = indices[self._labels[indices] == label_range]
        return ensure_slice(indices)


@singledispatch
def ensure_slice(obj: object) -> slice:
    raise NotImplementedError(f"Cannot convert {obj.__class__} to slice")


@ensure_slice.register(slice)
def ensure_slice_identity(s: slice) -> slice:
    return s


@ensure_slice.register(int)
@ensure_slice.register(np.integer)
def ensure_slice_int(i: Union[int, np.integer]) -> slice:
    return slice(i, i + 1)


@ensure_slice.register(np.ndarray)
def ensure_slice_array(a: np.ndarray) -> slice:
    if not issubclass(a.dtype.type, np.integer):
        raise ValueError("Non-integer arrays cannot be converted to slice")
    if a.ndim > 1:
        raise ValueError(f"{a.ndim}D arrays cannot be converted to slice")
    if a.ndim == 1 and len(a) == 0:
        raise ValueError("Empty array cannot be converted to slice")
    if a.ndim == 0 or len(a) == 1:
        return ensure_slice(a.item())

    diffs = a[1:] - a[:-1]
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValueError(
            "Non-monotonically increasing or decreasing array cannot be converted to slice"
        )
    unique_diffs = np.unique(diffs)
    if len(unique_diffs) > 1:
        raise ValueError(
            "Array with non-fixed step between elements cannot be converted to slice"
        )

    start = a[0]
    step = unique_diffs[0]
    stop = a[-1] + (1 if step > 0 else -1)
    return slice(start, stop, step)
