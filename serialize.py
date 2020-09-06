import json
import os
import sys
from typing import Any, Mapping

import numpy as np
import segyio
import tiledb
from segyio import SegyFile


def serialize_tilesegy(uri: str) -> Mapping[str, Any]:
    serialized = {}

    data_uri = os.path.join(uri, "data")
    with tiledb.DenseArray(data_uri) as tdb:
        for key in "samples", "ilines", "xlines", "offsets":
            if tdb.schema.domain.has_dim(key):
                value = tdb.meta[key]
                if not isinstance(value, tuple):
                    value = (value,)
                serialized[key] = value
        data = tdb[:]
        if len(serialized.get("offsets", ())) == 1:
            data = data.squeeze(axis=2)
        serialized["data"] = data.tolist()

    headers_uri = os.path.join(uri, "headers")
    with tiledb.DenseArray(headers_uri) as tdb:
        serialized["bin"] = dict(tdb.meta.items())
        text = serialized["bin"].pop("__text__").decode(errors="replace")
        assert len(text) % 3200 == 0, len(text)
        serialized["text"] = [text[i : i + 3200] for i in range(0, len(text), 3200)]

        headers = tdb[:]
        keys = headers.keys()
        columns = [v.tolist() for v in headers.values()]
        if tdb.schema.domain.has_dim("traces"):
            serialized["headers"] = [dict(zip(keys, row)) for row in zip(*columns)]
        else:
            ilines = range(len(serialized["ilines"]))
            xlines = range(len(serialized["xlines"]))
            offsets = range(len(serialized["offsets"]))
            if len(offsets) > 1:

                def serialize_header(i: int, x: int) -> Any:
                    return [
                        {key: columns[k][i][x][o] for k, key in enumerate(keys)}
                        for o in offsets
                    ]

            else:

                def serialize_header(i: int, x: int) -> Any:
                    return {key: columns[k][i][x][0] for k, key in enumerate(keys)}

            serialized["headers"] = [
                [serialize_header(i, x) for x in xlines] for i in ilines
            ]

    return serialized


def serialize_segyfile(segy_file: SegyFile) -> Mapping[str, Any]:
    serialized = {
        "bin": stringify_keys(segy_file.bin),
        "text": [bytes(s).decode(errors="replace") for s in segy_file.text],
        "samples": segy_file.samples.tolist(),
    }
    if segy_file.unstructured:
        serialized["data"] = segy_file.trace.raw[:].tolist()
        serialized["headers"] = list(map(stringify_keys, segy_file.header))
    else:
        serialized["ilines"] = segy_file.ilines.tolist()
        serialized["xlines"] = segy_file.xlines.tolist()
        serialized["offsets"] = segy_file.offsets.tolist()
        serialized["data"] = segyio.tools.cube(segy_file)
        serialized["headers"] = cube_headers(segy_file)
        for k in "data", "headers":
            if segy_file.sorting == segyio.TraceSortingFormat.CROSSLINE_SORTING:
                serialized[k] = serialized[k].swapaxes(0, 1)
            serialized[k] = serialized[k].tolist()

    return serialized


def cube_headers(segy_file: SegyFile) -> np.ndarray:
    if segy_file.fast is segy_file.iline:
        fast = segy_file.ilines
        slow = segy_file.xlines
    else:
        fast = segy_file.xlines
        slow = segy_file.ilines
    fast, slow, offs = map(len, (fast, slow, segy_file.offsets))
    dims = (fast, slow) if offs == 1 else (fast, slow, offs)
    headers = list(map(stringify_keys, segy_file.header))
    cube = np.array(headers).reshape(dims)
    return cube


def stringify_keys(d: Mapping[Any, Any]) -> Mapping[Any, Any]:
    return {str(k): v for k, v in d.items()}


if __name__ == "__main__":
    path = sys.argv[1]
    if os.path.isfile(path):
        with segyio.open(path, strict=False) as segy_file:
            serialized = serialize_segyfile(segy_file)
    else:
        serialized = serialize_tilesegy(path)
    print(json.dumps(serialized))
