# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd
import pytest
from hashlib import md5
try:
    import flopy
except ImportError:
    pytest.skip("skipping tests that require flopy", allow_module_level=True)

from .common import swn, wkt_to_geoseries


datadir = os.path.join('tests', 'data')


@pytest.fixture
def n3d():
    # same valid network used in test_basic
    lines = wkt_to_geoseries([
        'LINESTRING Z (60 100 14, 60  80 12)',
        'LINESTRING Z (40 130 15, 60 100 14)',
        'LINESTRING Z (70 130 15, 60 100 14)',
    ])
    return swn.SurfaceWaterNetwork(lines)


@pytest.fixture
def n2d():
    lines = wkt_to_geoseries([
        'LINESTRING (60 100, 60  80)',
        'LINESTRING (40 130, 60 100)',
        'LINESTRING (70 130, 60 100)',
    ])
    return swn.SurfaceWaterNetwork(lines)


def test_process_flopy_instance_errors(n3d):
    n = n3d
    n.segments = n.segments.copy()
    with pytest.raises(
            ValueError,
            match=r'must be a flopy\.modflow\.mf\.Modflow object'):
        n.process_flopy(object())

    m = flopy.modflow.Modflow()
    with pytest.raises(ValueError, match='DIS package required'):
        n.process_flopy(m)

    flopy.modflow.ModflowDis(
        m, nlay=1, nrow=3, ncol=2, nper=4, delr=20.0, delc=20.0)
    with pytest.raises(ValueError, match='BAS6 package required'):
        n.process_flopy(m)

    flopy.modflow.ModflowBas(m)

    m.modelgrid.set_coord_info(epsg=2193)
    n.segments.crs = {'init': 'epsg:27200'}
    # with pytest.raises(
    #        ValueError,
    #        match='CRS for segments and modelgrid are different'):
    #    n.process_flopy(m)

    n.segments.crs = None
    with pytest.raises(
            ValueError,
            match='modelgrid extent does not cover segments extent'):
        n.process_flopy(m)

    m.modelgrid.set_coord_info(xoff=30.0, yoff=70.0)

    with pytest.raises(ValueError, match='ibound_action must be one of'):
        n.process_flopy(m, ibound_action='foo')

    with pytest.raises(ValueError, match='flow must be a dict or DataFrame'):
        n.process_flopy(m, flow=1.1)

    with pytest.raises(
            ValueError,
            match=r'flow\.index must be a pandas\.DatetimeIndex'):
        n.process_flopy(m, flow=pd.DataFrame({'1': [1.1]}))

    with pytest.raises(
            ValueError,
            match='flow DataFrame length is different than nper'):
        n.process_flopy(
            m, flow=pd.DataFrame(
                    {'1': [1.1]},
                    index=pd.DatetimeIndex(['1970-01-01'])))

    with pytest.raises(
            ValueError,
            match=r'flow\.index does not match expected \(1970\-01\-01, 1970'):
        n.process_flopy(
            m, flow=pd.DataFrame(
                    {'1': 1.1},  # starts on the wrong day
                    index=pd.DatetimeIndex(['1970-01-02'] * 4) +
                    pd.TimedeltaIndex(range(4), 'days')))

    with pytest.raises(
            ValueError,
            match=r'flow\.columns\.dtype must be same as segments\.index\.dt'):
        n.process_flopy(
            m, flow=pd.DataFrame(
                    {'s1': 1.1},  # can't convert key to int
                    index=pd.DatetimeIndex(['1970-01-01'] * 4) +
                    pd.TimedeltaIndex(range(4), 'days')))

    with pytest.raises(
            ValueError,
            match=r'flow\.columns not found in segments\.index'):
        n.process_flopy(
            m, flow=pd.DataFrame(
                    {'3': 1.1},  # segnum 3 does not exist
                    index=pd.DatetimeIndex(['1970-01-01'] * 4) +
                    pd.TimedeltaIndex(range(4), 'days')))

    # finally success!
    n.process_flopy(
        m, flow=pd.DataFrame(
                {'1': 1.1, '3': 1.1},  # segnum 3 ignored
                index=pd.DatetimeIndex(['1970-01-01'] * 4) +
                pd.TimedeltaIndex(range(4), 'days')))


def test_process_flopy_n3d(n3d, tmpdir_factory):
    r"""
        .___.___.
        :  1:  2:  row 0
        :__\:_/_:
        :   \/  :  row 1
        :__ :0__:
        :   :|  :  row 2
        :___:|__:
      col 0 ' col 1

      Segment IDs: 0 (bottom), 1 & 2 (top)
    """
    n = n3d
    m = flopy.modflow.Modflow()
    flopy.modflow.ModflowDis(
        m, nlay=1, nrow=3, ncol=2, delr=20.0, delc=20.0, top=15.0, botm=10.0,
        xul=30.0, yul=130.0)
    flopy.modflow.ModflowBas(m)
    n.process_flopy(m)
    # Data set 1c
    assert abs(m.sfr.nstrm) == 7
    assert m.sfr.nss == 3
    assert m.sfr.const == 86400.0
    # Data set 2
    # Base-0
    assert list(m.sfr.reach_data.node) == [0, 1, 3, 1, 3, 3, 5]
    assert list(m.sfr.reach_data.k) == [0, 0, 0, 0, 0, 0, 0]
    assert list(m.sfr.reach_data.i) == [0, 0, 1, 0, 1, 1, 2]
    assert list(m.sfr.reach_data.j) == [0, 1, 1, 1, 1, 1, 1]
    # Base-1
    assert list(m.sfr.reach_data.reachID) == [1, 2, 3, 4, 5, 6, 7]
    assert list(m.sfr.reach_data.iseg) == [1, 1, 1, 2, 2, 3, 3]
    assert list(m.sfr.reach_data.ireach) == [1, 2, 3, 1, 2, 1, 2]
    np.testing.assert_array_almost_equal(
        m.sfr.reach_data.rchlen,
        [18.027756, 6.009252, 12.018504, 21.081851, 10.540926, 10.0, 10.0])
    np.testing.assert_array_almost_equal(
        m.sfr.reach_data.strtop,
        [14.75, 14.416667, 14.166667, 14.666667, 14.166667, 13.5, 12.5])
    np.testing.assert_array_almost_equal(
        m.sfr.reach_data.slope,
        [0.027735, 0.027735, 0.027735, 0.031622775, 0.031622775, 0.1, 0.1])
    # Repeat, but with min_slope enforced
    sfr_unit = m.sfr.unit_number[0]
    m.remove_package('sfr')
    if sfr_unit in m.package_units:
        m.package_units.remove(sfr_unit)
    n.process_flopy(m, min_slope=0.03)
    np.testing.assert_array_almost_equal(
        m.sfr.reach_data.rchlen,
        [18.027756, 6.009252, 12.018504, 21.081851, 10.540926, 10.0, 10.0])
    np.testing.assert_array_almost_equal(
        m.sfr.reach_data.strtop,
        [14.75, 14.416667, 14.166667, 14.666667, 14.166667, 13.5, 12.5])
    np.testing.assert_array_almost_equal(
        m.sfr.reach_data.slope,
        [0.03, 0.03, 0.03, 0.031622775, 0.031622775, 0.1, 0.1])
    sd = m.sfr.segment_data[0]
    assert list(sd.nseg) == [1, 2, 3]
    assert list(sd.icalc) == [1, 1, 1]
    assert list(sd.outseg) == [3, 3, 0]
    assert list(sd.iupseg) == [0, 0, 0]
    assert list(sd.iprior) == [0, 0, 0]
    # TODO: more tests needed
    # Write output files
    outdir = tmpdir_factory.mktemp('n3d')
    m.model_ws = str(outdir)
    m.write_input()
    # m.sfr.write_file(str(outdir.join('file.sfr')))
    n.grid_cells.to_file(str(outdir.join('grid_cells.shp')))
    n.reaches.to_file(str(outdir.join('reaches.shp')))


def test_process_flopy_n2d(n2d, tmpdir_factory):
    # similar to 3D version, but getting information from model
    n = n2d
    m = flopy.modflow.Modflow()
    top = np.array([
        [16.0, 15.0],
        [15.0, 15.0],
        [14.0, 14.0],
    ])
    flopy.modflow.ModflowDis(
        m, nlay=1, nrow=3, ncol=2, delr=20.0, delc=20.0, top=top, botm=10.0,
        xul=30.0, yul=130.0)
    flopy.modflow.ModflowBas(m)
    n.process_flopy(m)
    # Data set 1c
    assert abs(m.sfr.nstrm) == 7
    assert m.sfr.nss == 3
    # Data set 2
    np.testing.assert_array_almost_equal(
        m.sfr.reach_data.rchlen,
        [18.027756, 6.009252, 12.018504, 21.081851, 10.540926, 10.0, 10.0])
    np.testing.assert_array_equal(
        m.sfr.reach_data.strtop,
        [16.0, 15.0, 15.0, 15.0, 15.0, 15.0, 14.0])
    np.testing.assert_array_almost_equal(
        m.sfr.reach_data.slope,
        [0.070710681, 0.05, 0.025, 0.05, 0.025, 0.025, 0.05])
    # Repeat, but with min_slope enforced
    sfr_unit = m.sfr.unit_number[0]
    m.remove_package('sfr')
    if sfr_unit in m.package_units:
        m.package_units.remove(sfr_unit)
    n.process_flopy(m, min_slope=0.03)
    np.testing.assert_array_almost_equal(
        m.sfr.reach_data.rchlen,
        [18.027756, 6.009252, 12.018504, 21.081851, 10.540926, 10.0, 10.0])
    np.testing.assert_array_equal(
        m.sfr.reach_data.strtop,
        [16.0, 15.0, 15.0, 15.0, 15.0, 15.0, 14.0])
    np.testing.assert_array_almost_equal(
        m.sfr.reach_data.slope,
        [0.070710681, 0.05, 0.03, 0.05, 0.03, 0.03, 0.05])
    sd = m.sfr.segment_data[0]
    assert list(sd.nseg) == [1, 2, 3]
    assert list(sd.icalc) == [1, 1, 1]
    assert list(sd.outseg) == [3, 3, 0]
    assert list(sd.iupseg) == [0, 0, 0]
    assert list(sd.iprior) == [0, 0, 0]
    # TODO: more tests needed
    # Write output files
    outdir = tmpdir_factory.mktemp('n2d')
    m.model_ws = str(outdir)
    m.write_input()
    # m.sfr.write_file(str(outdir.join('file.sfr')))
    n.grid_cells.to_file(str(outdir.join('grid_cells.shp')))
    n.reaches.to_file(str(outdir.join('reaches.shp')))


@pytest.fixture
def costal_flopy_m():
    return flopy.modflow.Modflow.load('h.nam', model_ws=datadir, check=False)


def test_costal_process_flopy(costal_swn, costal_flopy_m):
    n = costal_swn
    m = costal_flopy_m
    # Make sure this is the model we are thinking of
    assert m.modelgrid.extent == (1802000.0, 1819000.0, 5861000.0, 5879000.0)
    n.process_flopy(m)
    assert len(n.reaches) == 370
    assert m.bas6.ibound[0].array.sum() == 178
    assert md5(m.bas6.ibound.array.astype('B').tostring()).hexdigest() == \
        '3ca05914e75b930252257dd331111d95'


def test_costal_process_flopy_ibound_modify(costal_swn):
    n = costal_swn
    m = flopy.modflow.Modflow.load('h.nam', model_ws=datadir, check=False)
    n.process_flopy(m, ibound_action='modify')
    assert len(n.reaches) == 626
    assert m.bas6.ibound[0].array.sum() == 243
    assert md5(m.bas6.ibound.array.astype('B').tostring()).hexdigest() == \
        '9666d6aff9266196a26c3726e5c1da94'


def test_process_flopy_lines_on_boundaries():
    m = flopy.modflow.Modflow()
    flopy.modflow.ModflowDis(
        m, nrow=3, ncol=3, delr=100, delc=100, xul=0, yul=300)
    flopy.modflow.ModflowBas(m)
    lines = wkt_to_geoseries([
        'LINESTRING (  0 320, 100 200)',
        'LINESTRING (100 200, 100 150, 150 100)',
        'LINESTRING (100 280, 100 200)',
        'LINESTRING (250 250, 150 150, 150 100)',
        'LINESTRING (150 100, 200   0, 300   0)',
    ])
    n = swn.SurfaceWaterNetwork(lines)
    n.process_flopy(m)
    assert m.sfr.nss == 5
    # TODO: code needs to be improved for this type of case
    # assert abs(m.sfr.nstrm) == 8
