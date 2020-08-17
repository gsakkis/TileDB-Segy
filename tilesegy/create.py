import os
from typing import Collection, Sequence, Union

import numpy as np
import tiledb
from segyio import SegyFile

Number = Union[int, float, np.number]

TRACE_FIELD_DTYPES = {
    "TRACE_SEQUENCE_LINE": np.int32,
    "TRACE_SEQUENCE_FILE": np.int32,
    "FieldRecord": np.int32,
    "TraceNumber": np.int32,
    "EnergySourcePoint": np.int32,
    "CDP": np.int32,
    "CDP_TRACE": np.int32,
    "TraceIdentificationCode": np.int16,
    "NSummedTraces": np.int16,
    "NStackedTraces": np.int16,
    "DataUse": np.int16,
    "offset": np.int32,
    "ReceiverGroupElevation": np.int32,
    "SourceSurfaceElevation": np.int32,
    "SourceDepth": np.int32,
    "ReceiverDatumElevation": np.int32,
    "SourceDatumElevation": np.int32,
    "SourceWaterDepth": np.int32,
    "GroupWaterDepth": np.int32,
    "ElevationScalar": np.int16,
    "SourceGroupScalar": np.int16,
    "SourceX": np.int32,
    "SourceY": np.int32,
    "GroupX": np.int32,
    "GroupY": np.int32,
    "CoordinateUnits": np.int16,
    "WeatheringVelocity": np.int16,
    "SubWeatheringVelocity": np.int16,
    "SourceUpholeTime": np.int16,
    "GroupUpholeTime": np.int16,
    "SourceStaticCorrection": np.int16,
    "GroupStaticCorrection": np.int16,
    "TotalStaticApplied": np.int16,
    "LagTimeA": np.int16,
    "LagTimeB": np.int16,
    "DelayRecordingTime": np.int16,
    "MuteTimeStart": np.int16,
    "MuteTimeEND": np.int16,
    "TRACE_SAMPLE_COUNT": np.int16,
    "TRACE_SAMPLE_INTERVAL": np.int16,
    "GainType": np.int16,
    "InstrumentGainConstant": np.int16,
    "InstrumentInitialGain": np.int16,
    "Correlated": np.int16,
    "SweepFrequencyStart": np.int16,
    "SweepFrequencyEnd": np.int16,
    "SweepLength": np.int16,
    "SweepType": np.int16,
    "SweepTraceTaperLengthStart": np.int16,
    "SweepTraceTaperLengthEnd": np.int16,
    "TaperType": np.int16,
    "AliasFilterFrequency": np.int16,
    "AliasFilterSlope": np.int16,
    "NotchFilterFrequency": np.int16,
    "NotchFilterSlope": np.int16,
    "LowCutFrequency": np.int16,
    "HighCutFrequency": np.int16,
    "LowCutSlope": np.int16,
    "HighCutSlope": np.int16,
    "YearDataRecorded": np.int16,
    "DayOfYear": np.int16,
    "HourOfDay": np.int16,
    "MinuteOfHour": np.int16,
    "SecondOfMinute": np.int16,
    "TimeBaseCode": np.int16,
    "TraceWeightingFactor": np.int16,
    "GeophoneGroupNumberRoll1": np.int16,
    "GeophoneGroupNumberFirstTraceOrigField": np.int16,
    "GeophoneGroupNumberLastTraceOrigField": np.int16,
    "GapSize": np.int16,
    "OverTravel": np.int16,
    "CDP_X": np.int32,
    "CDP_Y": np.int32,
    "INLINE_3D": np.int32,
    "CROSSLINE_3D": np.int32,
    "ShotPoint": np.int32,
    "ShotPointScalar": np.int16,
    "TraceValueMeasurementUnit": np.int16,
    "TransductionConstantMantissa": np.int32,
    "TransductionConstantPower": np.int16,
    "TransductionUnit": np.int16,
    "TraceIdentifier": np.int16,
    "ScalarTraceHeader": np.int16,
    "SourceType": np.int16,
    "SourceEnergyDirectionMantissa": np.int32,
    "SourceEnergyDirectionExponent": np.int16,
    "SourceMeasurementMantissa": np.int32,
    "SourceMeasurementExponent": np.int16,
    "SourceMeasurementUnit": np.int16,
}
TRACE_FIELDS_SIZE = sum(
    np.dtype(dtype).itemsize for dtype in TRACE_FIELD_DTYPES.values()
)
TRACE_FIELD_FILTERS = (
    tiledb.BitWidthReductionFilter(),
    tiledb.ByteShuffleFilter(),
    tiledb.LZ4Filter(),
)
MAX_TILESIZE = 2 ** 16


def create(uri: str, segy_file: SegyFile) -> None:
    if tiledb.object_type(uri) != "group":
        tiledb.group_create(uri)

    headers_uri = os.path.join(uri, "headers")
    if tiledb.object_type(headers_uri) != "array":
        schema = _get_headers_schema(segy_file)
        print(f"header schema: {schema}")
        tiledb.DenseArray.create(headers_uri, schema)

    data_uri = os.path.join(uri, "data")
    if tiledb.object_type(data_uri) != "array":
        schema = _get_data_schema(segy_file)
        print(f"data schema: {schema}")
        tiledb.DenseArray.create(data_uri, schema)


def _get_headers_schema(segy_file: SegyFile) -> tiledb.ArraySchema:
    if segy_file.unstructured:
        dims = _get_unstructured_header_dims(segy_file)
    else:
        dims = _get_structured_header_dims(segy_file)
    return tiledb.ArraySchema(
        domain=tiledb.Domain(*dims),
        attrs=[
            tiledb.Attr(name, dtype, filters=TRACE_FIELD_FILTERS)
            for name, dtype in TRACE_FIELD_DTYPES.items()
        ],
    )


def _get_data_schema(segy_file: SegyFile) -> tiledb.ArraySchema:
    if segy_file.unstructured:
        dims = _get_unstructured_data_dims(segy_file)
    else:
        dims = _get_structured_data_dims(segy_file)
    return tiledb.ArraySchema(
        domain=tiledb.Domain(*dims), attrs=[tiledb.Attr(dtype=segy_file.dtype)]
    )


def _get_unstructured_header_dims(segy_file: SegyFile) -> Sequence[tiledb.Dim]:
    domain = (0, segy_file.tracecount - 1)
    return [
        tiledb.Dim(
            name="traces",
            domain=domain,
            dtype=_find_shortest_dtype(domain),
            tile=np.clip(MAX_TILESIZE // TRACE_FIELDS_SIZE, 1, segy_file.tracecount),
        ),
    ]


def _get_structured_header_dims(segy_file: SegyFile) -> Sequence[tiledb.Dim]:
    ilines, xlines, offsets = segy_file.ilines, segy_file.xlines, segy_file.offsets
    domains = [(0, len(ilines) - 1), (0, len(xlines) - 1)]
    if len(offsets) > 1:
        domains.append((0, len(offsets) - 1))
    dtype = _find_shortest_dtype(sum(domains, ()))
    dims = [
        tiledb.Dim(
            name="ilines",
            domain=domains[0],
            dtype=dtype,
            tile=np.clip(
                MAX_TILESIZE // (len(xlines) * TRACE_FIELDS_SIZE), 1, len(ilines),
            ),
        ),
        tiledb.Dim(
            name="xlines",
            domain=domains[1],
            dtype=dtype,
            tile=np.clip(
                MAX_TILESIZE // (len(ilines) * TRACE_FIELDS_SIZE), 1, len(xlines),
            ),
        ),
    ]
    if len(domains) == 3:
        dims.append(tiledb.Dim(name="offsets", domain=domains[2], dtype=dtype, tile=1))
    return dims


def _get_unstructured_data_dims(segy_file: SegyFile) -> Sequence[tiledb.Dim]:
    num_samples = len(segy_file.samples)
    cell_size = segy_file.dtype.itemsize
    domains = [(0, segy_file.tracecount - 1), (0, num_samples - 1)]
    dtype = _find_shortest_dtype(sum(domains, ()))
    return [
        tiledb.Dim(
            name="traces",
            domain=domains[0],
            dtype=dtype,
            tile=np.clip(
                MAX_TILESIZE // (num_samples * cell_size), 1, segy_file.tracecount
            ),
        ),
        tiledb.Dim(
            name="samples",
            domain=domains[1],
            dtype=dtype,
            tile=np.clip(MAX_TILESIZE // cell_size, 1, num_samples),
        ),
    ]


def _get_structured_data_dims(segy_file: SegyFile) -> Sequence[tiledb.Dim]:
    num_samples = len(segy_file.samples)
    cell_size = segy_file.dtype.itemsize
    ilines, xlines, offsets = segy_file.ilines, segy_file.xlines, segy_file.offsets
    domains = [(0, len(ilines) - 1), (0, len(xlines) - 1), (0, num_samples - 1)]
    if len(offsets) > 1:
        domains.append((0, len(offsets) - 1))
    dtype = _find_shortest_dtype(sum(domains, ()))
    dims = [
        tiledb.Dim(
            name="ilines",
            domain=domains[0],
            dtype=dtype,
            tile=np.clip(
                MAX_TILESIZE // (len(xlines) * num_samples * cell_size), 1, len(ilines),
            ),
        ),
        tiledb.Dim(
            name="xlines",
            domain=domains[1],
            dtype=dtype,
            tile=np.clip(
                MAX_TILESIZE // (len(ilines) * num_samples * cell_size), 1, len(xlines),
            ),
        ),
        tiledb.Dim(
            name="samples",
            domain=domains[2],
            dtype=dtype,
            tile=np.clip(MAX_TILESIZE // cell_size, 1, num_samples),
        ),
    ]
    if len(domains) == 4:
        dims.append(tiledb.Dim(name="offsets", domain=domains[3], dtype=dtype, tile=1))
    return dims


def _find_shortest_dtype(values: Collection[Number]) -> np.dtype:
    min_value = min(values)
    max_value = max(values)
    for dt in (
        np.uint8,
        np.int8,
        np.uint16,
        np.int16,
        np.uint32,
        np.int32,
        np.uint64,
        np.int64,
    ):
        info = np.iinfo(dt)
        if info.min <= min_value <= max_value <= info.max:
            return dt


if __name__ == "__main__":
    import sys

    import segyio

    segy_file, output_dir, ignore_geometry = sys.argv[1:]
    with segyio.open(segy_file, ignore_geometry=int(ignore_geometry)) as segy_file:
        create(output_dir, segy_file)
