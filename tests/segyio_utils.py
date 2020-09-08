import itertools as it
import pathlib
from typing import Union

import numpy as np
import segyio

Path = Union[pathlib.Path, str]
from_array_inline = segyio.tools.from_array


def from_array_crossline(
    filename: Path,
    data: np.ndarray,
    iline: int = 189,
    xline: int = 193,
    format: segyio.SegySampleFormat = segyio.SegySampleFormat.IBM_FLOAT_4_BYTE,
    dt: int = 4000,
    delrt: int = 0,
) -> None:
    dt = int(dt)
    delrt = int(delrt)
    data = np.asarray(data)
    dimensions = len(data.shape)
    if dimensions not in range(3, 5):
        raise ValueError(f"Expected 3 or 4 dimensions, {dimensions} was given")

    spec = segyio.spec()
    spec.iline = iline
    spec.xline = xline
    spec.format = format
    spec.sorting = segyio.TraceSortingFormat.CROSSLINE_SORTING
    spec.ilines = list(range(1, np.size(data, 0) + 1))
    spec.xlines = list(range(1, np.size(data, 1) + 1))
    if dimensions == 3:
        spec.samples = list(range(np.size(data, 2)))
    else:
        spec.offsets = list(range(1, np.size(data, 2) + 1))
        spec.samples = list(range(np.size(data, 3)))

    samplecount = len(spec.samples)
    with segyio.create(filename, spec) as f:
        tr = 0
        for xlno, xl in enumerate(spec.xlines):
            for ilno, il in enumerate(spec.ilines):
                for offno, off in enumerate(spec.offsets):
                    f.header[tr] = {
                        segyio.su.tracf: tr,
                        segyio.su.cdpt: tr,
                        segyio.su.offset: off,
                        segyio.su.ns: samplecount,
                        segyio.su.dt: dt,
                        segyio.su.delrt: delrt,
                        segyio.su.iline: il,
                        segyio.su.xline: xl,
                    }
                    if dimensions == 3:
                        f.trace[tr] = data[ilno, xlno, :]
                    else:
                        f.trace[tr] = data[ilno, xlno, offno, :]
                    tr += 1
        f.bin.update(tsort=spec.sorting, hdt=dt, dto=dt)


def from_array_unstructured(
    filename: Path,
    data: np.ndarray,
    format: segyio.SegySampleFormat = segyio.SegySampleFormat.IBM_FLOAT_4_BYTE,
    dt: int = 4000,
    delrt: int = 0,
) -> None:
    dt = int(dt)
    delrt = int(delrt)
    data = np.asarray(data)
    dimensions = len(data.shape)
    if dimensions != 2:
        raise ValueError(f"Expected 2 dimensions, {dimensions} was given")

    spec = segyio.spec()
    spec.format = format
    spec.sorting = segyio.TraceSortingFormat.UNKNOWN_SORTING
    spec.tracecount = np.size(data, 0)
    spec.samples = list(range(np.size(data, 1)))
    samplecount = len(spec.samples)
    with segyio.create(filename, spec) as f:
        for tr in range(f.tracecount):
            f.header[tr] = {
                segyio.su.tracf: tr,
                segyio.su.cdpt: tr,
                segyio.su.ns: samplecount,
                segyio.su.dt: dt,
                segyio.su.delrt: delrt,
            }
            f.trace[tr] = data[tr, :]
        f.bin.update(tsort=spec.sorting, hdt=dt, dto=dt)


def generate_structured_segy(
    path: Path,
    ilines: int,
    xlines: int,
    offsets: int,
    samples: int,
    sorting: segyio.TraceSortingFormat,
) -> None:
    data = np.zeros((ilines, xlines, offsets, samples), np.float32)
    for i, x, o, s in it.product(*map(range, data.shape)):
        data[i, x, o, s] = o * 1000 + i * 10 + x + s / 10
    if offsets == 1:
        data = data.squeeze(axis=2)
    if sorting == segyio.TraceSortingFormat.INLINE_SORTING:
        from_array_inline(path, data)
    else:
        from_array_crossline(path, data)


def generate_unstructured_segy(
    path: Path,
    traces: int,
    samples: int,
    sorting: segyio.TraceSortingFormat = segyio.TraceSortingFormat.UNKNOWN_SORTING,
) -> None:
    assert sorting == segyio.TraceSortingFormat.UNKNOWN_SORTING
    data = np.zeros((traces, samples), np.float32)
    for t, s in it.product(*map(range, data.shape)):
        data[t, s] = t + s / 10
    from_array_unstructured(path, data)
