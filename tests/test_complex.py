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
def lines():
    shp_srs = os.path.join(datadir, 'DN2_Coastal_strahler1z_stream_vf.shp')
    lines = geopandas.read_file(shp_srs)
    lines.set_index('nzsegment', inplace=True)
    return lines


@pytest.fixture
def dn(lines):
    return swn.SurfaceWaterNetwork(lines)


def test_init(dn, lines):
    assert len(dn.warnings) == 1
    assert dn.warnings[0].startswith('ending coordinate ')
    assert dn.warnings[0].endswith(
            ' matches end nodes: ' + str(set([3046736, 3046737])))
    assert len(dn.errors) == 0
    assert len(dn) == 304
    assert dn.has_z is True
    assert dn.END_NODE == 0
    assert dn.reaches.index is dn.index
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
    res = lines['LENGTHDOWN'] + lines.geometry.length - ln
    np.testing.assert_almost_equal(res.min(), 0.0)
    np.testing.assert_almost_equal(res.max(), 15.00362636)
    assert list(dn.reaches['sequence'])[:6] == [149, 225, 152, 222, 145, 142]
    assert list(dn.reaches['sequence'])[-6:] == [156, 4, 155, 1, 3, 2]
    stream_order = dn.reaches.groupby('stream_order').count()['to_node']
    assert len(stream_order) == 5
    assert dict(stream_order) == {1: 154, 2: 72, 3: 46, 4: 28, 5: 4}
    np.testing.assert_array_equal(
        dn.reaches['stream_order'], lines['StreamOrde'])


def test_accumulate_values(dn, lines):
    catarea = dn.accumulate_values(lines['CATAREA'])
    res = catarea - lines['CUM_AREA']
    assert res.min() > -7.0
    assert res.max() < 7.0
    assert catarea.name == 'accumulated_CATAREA'


@pytest.fixture
def m():
    flopy = pytest.importorskip('flopy')
    return flopy.modflow.Modflow.load('h.nam', model_ws=datadir, check=False)


def test_nothing(m):
    assert m.modelgrid.extent == (1802000.0, 1819000.0, 5861000.0, 5879000.0)
