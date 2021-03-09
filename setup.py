#!/usr/bin/env/python
# -*- coding: utf-8 -*-
"""Setuptools file."""
from setuptools import setup

with open("swn/__init__.py", "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip("\"'")
        elif line.startswith("__author__"):
            author = line.split("=")[1].strip().strip("\"'")
        elif line.startswith("__email__"):
            author_email = line.split("=")[1].strip().strip("\"'")

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='swn',
    version=version,
    description="Surface water network",
    long_description=long_description,
    author=author,
    author_email=author_email,
    url='https://github.com/mwtoews/surface-water-network',
    license='BSD',
    packages=['swn'],
    package_data={'': ['tests/data/*']},
    python_requires='>=3.6',
    install_requires=['geopandas', 'pyproj>=2.0', 'rtree', 'shapely'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Hydrology',
    ],
)
