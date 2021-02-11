from functools import singledispatch

import numpy as np


class MultiSliceError(ValueError):
    pass


@singledispatch
def ensure_slice(obj: object) -> slice:
    raise TypeError(f"Cannot convert {obj.__class__} to slice")


@ensure_slice.register(slice)
def _ensure_slice_slice(s: slice) -> slice:
    return s


@ensure_slice.register(int)
@ensure_slice.register(np.integer)
def _ensure_slice_int(i: int) -> slice:
    return slice(i, i + 1)


@ensure_slice.register(np.ndarray)
def _ensure_slice_array(a: np.ndarray) -> slice:
    if not issubclass(a.dtype.type, np.integer):
        raise ValueError("Non-integer array cannot be converted to slice")
    if a.ndim > 1:
        raise ValueError(f"{a.ndim}D array cannot be converted to slice")
    if a.ndim == 1 and len(a) == 0:
        raise ValueError("Empty array cannot be converted to slice")
    if a.min() < 0:
        raise ValueError("Array with negative indices cannot be converted to slice")
    if a.ndim == 0 or len(a) == 1:
        return ensure_slice(a.item())

    diffs = a[1:] - a[:-1]
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValueError(
            "Non-monotonically increasing or decreasing array cannot be converted to slice"
        )
    unique_diffs = np.unique(diffs)
    if len(unique_diffs) > 1:
        raise MultiSliceError("Array is not convertible to a single range")

    step = unique_diffs[0]
    start = a[0]
    if step > 0:
        stop = a[-1] + 1
    elif a[-1] > 0:
        stop = a[-1] - 1
    else:
        stop = None
    return slice(start, stop, step)
