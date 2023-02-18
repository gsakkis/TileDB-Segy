from pathlib import Path

from setuptools import setup

with Path(__file__).parent.joinpath("README.md").open() as f:
    long_description = f.read()

setup(
    name="tiledb-segy",
    version="0.3.1",
    description="Python library for fast access to seismic data using TileDB",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TileDB-Inc/TileDB-Segy",
    author="George Sakkis",
    author_email="george.sakkis@gmail.com",
    packages=["tiledb.segy"],
    install_requires=[
        "cached_property",
        "segyio>=1.9.6",
        "tiledb>=0.8.3",
        "urlpath",
        "wrapt",
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-coverage", "pytest-xdist", "filelock"],
    entry_points={"console_scripts": ["segy2tiledb=tiledb.segy.cli:main"]},
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries",
        "Topic :: Utilities",
    ],
)
