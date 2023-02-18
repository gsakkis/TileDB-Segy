"""Convert a SEG-Y file to tiledb-segy format"""

import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from typing import List, Optional

import segyio

from .convert import SegyFileConverter


class HelpFormatter(RawTextHelpFormatter, ArgumentDefaultsHelpFormatter):
    pass


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=HelpFormatter, description=__doc__)
    parser.add_argument("input", type=Path, help="Input SEG-Y file path")
    parser.add_argument("output", type=Path, nargs="?", help="Output directory path")
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
        help="""Output geometry:
- auto: same as the input SEG-Y.
- structured: same as `auto` but abort if a geometry cannot be inferred.
- unstructured: opt out on building geometry information.
""",
    )

    segyio_args = parser.add_argument_group("segyio options")
    segyio_args.add_argument(
        "--su",
        action="store_true",
        help="Open a seismic unix file instead of SEG-Y",
    )
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

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = get_parser()
    args = parser.parse_args(argv)

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

    open_seismic = segyio.su.open if args.su else segyio.open
    with open_seismic(**segyio_kwargs) as f:
        converter = SegyFileConverter(f, tile_size=args.tile_size)  # type: ignore
        converter.to_tiledb(args.output)


if __name__ == "__main__":  # pragma: nocover
    main()
