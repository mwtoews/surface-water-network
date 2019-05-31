# -*- coding: utf-8 -*-
import geopandas
import os
import pytest
import sys
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import swn

datadir = os.path.join('tests', 'data')


@pytest.fixture(scope='session', autouse=True)
def costal_lines():
    shp_srs = os.path.join(datadir, 'DN2_Coastal_strahler1z_stream_vf.shp')
    lines = geopandas.read_file(shp_srs)
    lines.set_index('nzsegment', inplace=True)
    return lines


@pytest.fixture(scope='session', autouse=True)
def costal_swn(costal_lines):
    return swn.SurfaceWaterNetwork(costal_lines)
