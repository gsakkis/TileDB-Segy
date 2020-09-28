"""Convert a segy file to tilesegy format"""

import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, RawTextHelpFormatter
from pathlib import Path

import segyio
import tiledb

from .convert import SegyFileConverter


class HelpFormatter(RawTextHelpFormatter, ArgumentDefaultsHelpFormatter):
    pass


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=HelpFormatter, description=__doc__)
    parser.add_argument("input", type=Path, help="Input segy file path")
    parser.add_argument(
        "output", type=Path, nargs="?", help="Output tilesegy directory path"
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists",
    )
    parser.add_argument(
        "-g",
        "--geometry",
        choices=["auto", "structured", "unstructured"],
        default="auto",
        help="""Geometry of the converted tilesegy:
- auto (default): same as the input segy.
  Calls `segyio.open()` with `strict=False`.
- structured: same as `auto` but abort if a geometry cannot be inferred.
  Calls `segyio.open()` with `strict=True`.
- unstructured: opt out on building geometry information.
  Calls `segyio.open()` with `ignore_geometry=True`.
""",
    )

    segyio_args = parser.add_argument_group("segyio options")
    segyio_args.add_argument(
        "--iline",
        type=int,
        default=189,
        help="Inline number field in the trace headers",
    )
    segyio_args.add_argument(
        "--xline",
        type=int,
        default=193,
        help="Crossline number field in the trace headers",
    )
    segyio_args.add_argument(
        "--endian",
        choices=["big", "msb", "little", "lsb"],
        default="big",
        help="File endianness, big/msb (default) or little/lsb",
    )

    tiledb_args = parser.add_argument_group("tiledb options")
    tiledb_args.add_argument(
        "-s",
        "--tile-size",
        type=int,
        default=4_000_000,
        help="Tile size in bytes.\n"
        "Larger tile size improves disk access time at the cost of higher memory",
    )
    tiledb_args.add_argument(
        "--consolidation-buffersize",
        type=int,
        default=5_000_000,
        help="The size in bytes of the attribute buffers used during consolidation",
    )

    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    if not args.output:
        args.output = args.input.with_suffix(".tsgy")
    if args.output.exists():
        if args.overwrite:
            shutil.rmtree(args.output)
        else:
            parser.error(f"{args.output} already exists")

    segyio_kwargs = dict(
        filename=args.input,
        iline=args.iline,
        xline=args.xline,
        endian=args.endian,
    )
    if args.geometry == "unstructured":
        segyio_kwargs["ignore_geometry"] = True
    else:
        segyio_kwargs["strict"] = args.geometry != "auto"

    converter_kwargs = dict(
        tile_size=args.tile_size,
        config=tiledb.Config(
            {"sm.consolidation.buffer_size": args.consolidation_buffersize}
        ),
    )
    with segyio.open(**segyio_kwargs) as f:
        converter = SegyFileConverter(f, **converter_kwargs)  # type: ignore
        converter.to_tiledb(args.output)


if __name__ == "__main__":
    main()
