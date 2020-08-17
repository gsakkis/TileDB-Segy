from setuptools import setup

setup(
    name="tilesegy",
    version="0.0.1",
    description="SEG-Y file API based on TileDB",
    url="https://github.com/gsakkis/tilesegy",
    author="George Sakkis",
    author_email="george.sakkis@gmail.com",
    packages=["tilesegy"],
    install_requires=["segyio", "tiledb"],
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries",
        "Topic :: Utilities",
    ],
)
