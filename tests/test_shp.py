# -*- coding: utf-8 -*-
import geopandas
import os
import pytest
import numpy as np
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
    assert len(dn.headwater) == 154
    assert set(dn.headwater).issuperset([3046700, 3046802, 3050418, 3048102])
    assert list(dn.outlets) == [3046700, 3046737, 3046736]
    nto = dn.reaches['num_to_outlet']
    assert nto.min() == 1
    assert nto.max() == 32
    assert list(nto[nto == 1].index) == [3046700, 3046737, 3046736]
    assert list(nto[nto == 2].index) == [3046539, 3046745, 3046951, 3046952]
    assert list(nto[nto == 31].index) == [3050175, 3050176, 3050337, 3050338]
    assert list(nto[nto == 32].index) == [3050413, 3050418]
    cat_group = dn.reaches.groupby('cat_group').count()['to_node']
    assert len(cat_group) == 3
    assert dict(cat_group) == {3046700: 1, 3046737: 173, 3046736: 130}
    ln = dn.reaches['length_to_outlet']
    np.testing.assert_almost_equal(ln.min(), 42.43659279)
    np.testing.assert_almost_equal(ln.max(), 21077.7486858)
    # supplied LENGTHDOWN is similar
    res = dn.lines['LENGTHDOWN'] - ln + dn.lines.geometry.length
    np.testing.assert_almost_equal(res.min(), 0.0)
    np.testing.assert_almost_equal(res.max(), 15.00362636)
    assert list(dn.reaches['sequence'])[:6] == [141, 222, 151, 217, 139, 131]
    assert list(dn.reaches['sequence'])[-6:] == [156, 4, 155, 1, 3, 2]
    assert dn.reaches['numiter'].min() == 0
    assert dn.reaches['numiter'].max() == 4
    stream_order = dn.reaches.groupby('stream_order').count()['to_node']
    assert len(stream_order) == 5
    assert dict(stream_order) == {1: 154, 2: 72, 3: 46, 4: 28, 5: 4}
    np.testing.assert_array_equal(
        dn.reaches['stream_order'], dn.lines['StreamOrde'])


def test_accumulate_values(dn):
    catarea = dn.accumulate_values(dn.lines['CATAREA'])
    res = catarea - dn.lines['CUM_AREA']
    assert res.min() > -7.0
    assert res.max() < 7.0
    assert catarea.name == 'accumulated_CATAREA'
