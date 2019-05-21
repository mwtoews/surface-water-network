# -*- coding: utf-8 -*-
import geopandas
import pytest
import os
try:
    import rtree
except ImportError:
    rtree = False

import sys
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import swn

datadir = os.path.join('tests', 'data')


@pytest.fixture
def dn():
    shp_srs = os.path.join(datadir, 'DN2_Coastal_strahler1z_stream_vf.shp')
    lines = geopandas.read_file(shp_srs)
    lines.set_index('nzsegment', inplace=True)
    return swn.SurfaceWaterNetwork(lines)


def test_init(dn):
    assert len(dn) == 304
    assert dn.END_NODE == 0
    if rtree:
        assert dn.lines_idx is not None
    else:
        assert dn.lines_idx is None
    assert dn.reaches.index is dn.lines.index

    assert dn.outlets.values.tolist() == [3046700, 3046737, 3046736]
