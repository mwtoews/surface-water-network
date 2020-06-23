#!/usr/bin/env/python
# -*- coding: utf-8 -*-
"""Setuptools file."""
from setuptools import setup

from swn import \
    __version__ as version, __doc__ as long_description, \
    __author__ as author, __email__ as author_email

setup(
    name='swn',
    version=version,
    description=long_description.split('\n')[0],
    long_description=long_description,
    author=author,
    author_email=author_email,
    url='https://github.com/mwtoews/surface-water-network',
    license='BSD',
    packages=['swn'],
    package_data={'': ['tests/data/*']},
    python_requires='>=3.5',
    install_requires=['geopandas'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Hydrology',
    ],
)
