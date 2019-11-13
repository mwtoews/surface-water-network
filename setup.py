#!/usr/bin/env/python
# -*- coding: utf-8 -*-
from setuptools import setup

from swn import __version__ as version, __doc__ as long_description

setup(
    name='swn',
    version=version,
    description='Surface water network',
    long_description=long_description,
    author='Mike Taves',
    author_email='mwtoews@gmail.com',
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
