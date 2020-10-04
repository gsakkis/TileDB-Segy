# tilesegy #

Tilesegy is a small MIT licensed Python library for easy interaction with seismic
data, powered by [TileDB](https://github.com/TileDB-Inc/TileDB). It combines an
intuitive, [segyio](https://github.com/equinor/segyio)-like API with a powerful
storage engine.


## Feature summary ##

### Available features  ###
  * Converting from SEG-Y and Seismic Unix formatted seismic data to TileDB arrays.
  * Simple and powerful read-only API, closely modeled after segyio.
  * 100% unit test coverage.
  * Fully type-annotated.

### Currently missing features ###
  * API for write operations.
  * Converting back to SEG-Y.
  * TileDB configuration and performance tuning.
  * Comprehensive documentation.
  * Real-world usage.


## Installation ##

Tilesegy can be installed

- from [PyPI](https://pypi.org/project/tilesegy/) by `pip`:

      pip install tilesegy

- from source by cloning the [Git](https://github.com/gsakkis/tilesegy) repository:

      git clone https://github.com/gsakkis/tilesegy.git
      cd tilesegy
      python setup.py install

  You may run the test suite with:

      python setup.py test


## Converting from SEG-Y ##

Tilesegy comes with a commandline interface (CLI) called `segy2tiledb` for converting
SEG-Y and Seismic Unix formatted files to TileDB formatted arrays. At minimum it takes
an input file and generates a tilesegy directory at the same parent directory with the
input and extension `.tsgy`:

    $ segy2tiledb a123.segy
    $ du -sh a123.*
    73M a123.sgy
    55M a123.tsgy

To see the full list of options run:

    $ segy2tiledb -h
    usage: segy2tiledb [-h] [-o] [-g {auto,structured,unstructured}] [--su]
                       [--iline ILINE] [--xline XLINE]
                       [--endian {big,msb,little,lsb}] [-s TILE_SIZE]
                       [--consolidation-buffersize CONSOLIDATION_BUFFERSIZE]
                       input [output]

    Convert a segy file to tilesegy format

    positional arguments:
      input                 Input segy file path
      output                Output tilesegy directory path (default: None)

    optional arguments:
      -h, --help            show this help message and exit
      -o, --overwrite       Overwrite the output directory if it already exists (default: False)
      -g {auto,structured,unstructured}, --geometry {auto,structured,unstructured}
                            Geometry of the converted tilesegy:
                            - auto: same as the input segy.
                            - structured: same as `auto` but abort if a geometry cannot be inferred.
                            - unstructured: opt out on building geometry information.
                             (default: auto)

    segyio options:
      --su                  Open a seismic unix file instead of SEG-Y (default: False)
      --iline ILINE         Inline number field in the trace headers (default: 189)
      --xline XLINE         Crossline number field in the trace headers (default: 193)
      --endian {big,msb,little,lsb}
                            File endianness, big/msb (default) or little/lsb (default: big)

    tiledb options:
      -s TILE_SIZE, --tile-size TILE_SIZE
                            Tile size in bytes.
                            Larger tile size improves disk access time at the cost of higher memory (default: 4000000)
      --consolidation-buffersize CONSOLIDATION_BUFFERSIZE
                            The size in bytes of the attribute buffers used during consolidation (default: 5000000)


## API ##

Tilesegy generally follows the segyio API so you may consult its excellent
[documentation](https://segyio.readthedocs.io/en/latest/index.html) to learn about
the public attributes (`ilines`, `xlines`, `offsets`, `samples`) and addressing modes
(`trace`, `header`, `attributes`', `iline`, `xline`, `fast`, `slow`, `depth_slice`,
`gather`,  `text`, `bin`).

You can find usage examples in the included [Jupyter notebook](https://github.com/gsakkis/tilesegy/blob/master/tutorial.ipynb).

The following list outlines the main differences from segyio:

- Probably the biggest difference is that addressing modes that return a generator of
  numpy arrays in segyio, in tilesegy return a single numpy array of higher dimension(s).
  For example, in a SEG-Y with 50 ilines, 20 xlines, 100 samples, and 3 offsets:
  - `f.iline[0:5]`:
    - `segyio` returns a generator that yields 5 2D numpy arrays of (20, 100) shape
    - `tilesegy` returns a 3D numpy array of (5, 20, 100) shape
  - `f.iline[0:5, :]`:
    - `segyio` returns a generator that yields 15 2D numpy arrays of (20, 100) shape
    - `tilesegy` returns a 4D numpy array of (5, 3, 20, 100) shape

- The mappings returned by `bin`, `header` and `attributes(name)` have string keys
  instead of `segyio.TraceField` enums or integers.

- `tilesegy.open(dir_path)`, the `segyio.open(file_path)` equivalent, does not currently
  take any optional parameters (e.g. `strict` or `ignore_geometry`).

- `tilesegy` exposes two classes, `TileSegy` for unstructured SEG-Y and
  `StructuredTileSegy` for structured.
  - `StructuredTileSegy` extends `TileSegy`, so the whole unstructured API is inherited
    by the structured.
  - All attributes and addressing modes specific to structured files (e.g. `ilines` or
    `gather`) are available only to `StructuredTileSegy`. In contrast `segyio` returns
    `None` or raises an exception if these properties are accessed on unstructured files.
  - There is no `unstructured` attibute; use `not isinstance(f, StructuredTileSegy)` instead.

- There is no `tracecount` attribute; use `len(trace)` instead.
- There is no `ext_headers` attribute; use `len(text[1:])` instead.
