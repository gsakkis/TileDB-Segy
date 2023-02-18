<a href="https://tiledb.com"><img src="https://github.com/TileDB-Inc/TileDB/raw/dev/doc/source/_static/tiledb-logo_color_no_margin_@4x.png" alt="TileDB logo" width="400"></a>

<a href="https://github.com/TileDB-Inc/TileDB-Segy"><img alt="GitHub Actions status" src="https://github.com/TileDB-Inc/TileDB-Segy/workflows/CI/badge.svg"></a>

# TileDB-Segy #

TileDB-Segy is a small MIT licensed Python library for easy interaction with seismic
data, powered by [TileDB](https://docs.tiledb.com/). It combines an intuitive,
[segyio](https://github.com/equinor/segyio)-like API with a powerful storage engine.


## Feature summary ##

### Available features  ###
  * Converting from SEG-Y and Seismic Unix formatted seismic data to TileDB arrays.
  * Simple and powerful read-only API, closely modeled after `segyio`.
  * 100% unit test coverage.
  * Fully type-annotated.

### Currently missing features ###
  * API for write operations.
  * Converting back to SEG-Y.
  * TileDB configuration and performance tuning.
  * Comprehensive documentation.
  * Real-world usage.


## Installation ##

TileDB-Segy can be installed:

- from [PyPI](https://pypi.org/project/tiledb-segy/) by `pip`:

      pip install tiledb-segy

- from source by cloning the [Git](https://github.com/TileDB-Inc/TileDB-Segy) repository:

      git clone https://github.com/TileDB-Inc/TileDB-Segy.git
      cd TileDB-Segy
      pip install .

  You may run the test suite with:

      python setup.py test


## Converting from SEG-Y ##

TileDB-Segy comes with a commandline interface (CLI) called `segy2tiledb` for converting
SEG-Y and Seismic Unix formatted files to TileDB formatted arrays. At minimum it takes
an input file and generates a directory at the same parent directory with the input and
extension `.tsgy`:

    $ segy2tiledb a123.segy
    $ du -sh a123.*
    73M a123.sgy
    55M a123.tsgy

To see the full list of options run:

    $ segy2tiledb -h
    usage: segy2tiledb [-h] [-o] [-g {auto,structured,unstructured}] [--su]
                       [--iline ILINE] [--xline XLINE]
                       [--endian {big,msb,little,lsb}] [-s TILE_SIZE]
                       input [output]

    Convert a SEG-Y file to tiledb-segy format

    positional arguments:
      input                 Input SEG-Y file path
      output                Output directory path (default: None)

    optional arguments:
      -h, --help            show this help message and exit
      -o, --overwrite       Overwrite the output directory if it already exists (default: False)
      -g {auto,structured,unstructured}, --geometry {auto,structured,unstructured}
                            Output geometry:
                            - auto: same as the input SEG-Y.
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


## API ##

TileDB-Segy generally follows the `segyio` API; you may consult its
[documentation](https://segyio.readthedocs.io/en/latest/index.html) to learn about
the public attributes (`ilines`, `xlines`, `offsets`, `samples`) and addressing modes
(`trace`, `header`, `attributes`', `iline`, `xline`, `fast`, `slow`, `depth_slice`,
`gather`,  `text`, `bin`).

You can find usage examples in the following Jupyter notebooks:
- [TileDB-Segy tutorial](https://github.com/TileDB-Inc/TileDB-Segy/blob/master/notebooks/tutorial.ipynb)
- [Seismic inversion of real data](https://github.com/TileDB-Inc/TileDB-Segy/blob/master/notebooks/seismic_inversion.ipynb)

### Differences from segyio ###

- Addressing modes that return a generator of numpy arrays in `segyio`, in `tiledb-segy`
  they return a single numpy array of higher dimension. For example, in a SEG-Y with
  50 ilines, 20 xlines, 100 samples, and 3 offsets:
  - `f.iline[0:5]`:
    - `segyio` returns a generator that yields 5 2D numpy arrays of (20, 100) shape
    - `tiledb-segy` returns a 3D numpy array of (5, 20, 100) shape
  - `f.iline[0:5, :]`:
    - `segyio` returns a generator that yields 15 2D numpy arrays of (20, 100) shape
    - `tiledb-segy` returns a 4D numpy array of (5, 3, 20, 100) shape

- The mappings returned by `bin`, `header` and `attributes(name)` have string keys
  instead of `segyio.TraceField` enums or integers.

- `tiledb.segy.open(dir_path)`, the `segyio.open(file_path)` equivalent, does not
  take any optional parameters (e.g. `strict` or `ignore_geometry`).

- Unstructured and structured SEG-Y are represented as instances of two different classes,
  `tiledb.segy.Segy` and `tiledb.segy.StructuredSegy` respectively.
  - `StructuredSegy` extends `Segy`, so the whole unstructured API is inherited
    by the structured.
  - All attributes and addressing modes specific to structured files (e.g. `ilines` or
    `gather`) are available only to `StructuredSegy`. In contrast `segyio` returns
    `None` or raises an exception if these properties are accessed on unstructured files.
  - [`segyio.tools.dt`](https://segyio.readthedocs.io/en/latest/segyio.html#segyio.tools.dt)
    is exposed as `Segy.dt(fallback=4000.0)` method.
  - [`segyio.tools.cube`](https://segyio.readthedocs.io/en/latest/segyio.html#segyio.tools.cube)
    is exposed as `StructuredSegy.cube()` method.
  - There is no `unstructured` attribute; use `not isinstance(f, StructuredSegy)` instead.

- There is no `tracecount` attribute; use `len(trace)` instead.

- There is no `ext_headers` attribute; use `len(text[1:])` instead.
