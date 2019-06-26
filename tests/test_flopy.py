# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest
from hashlib import md5
from shapely import wkt
try:
    import flopy
except ImportError:
    pytest.skip("skipping tests that require flopy", allow_module_level=True)

from .common import datadir, swn, wkt_to_geoseries


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
            match=r'length of flow \(1\) is different than nper \(4\)'):
        n.process_flopy(
            m, flow=pd.DataFrame(
                    {'1': [1.1]},
                    index=pd.DatetimeIndex(['1970-01-01'])))

    with pytest.raises(
            ValueError,
            match=r'flow\.index must be a pandas\.DatetimeIndex'):
        n.process_flopy(m, flow=pd.DataFrame({'1': [1.1] * 4}))

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


def test_process_flopy_n3d_defaults(n3d, tmpdir_factory):
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
    np.testing.assert_array_equal(m.sfr.reach_data.strthick, [1.0] * 7)
    np.testing.assert_array_equal(m.sfr.reach_data.strhc1, [1.0] * 7)
    # Data set 6
    assert len(m.sfr.segment_data) == 1
    sd = m.sfr.segment_data[0]
    np.testing.assert_array_equal(sd.nseg, [1, 2, 3])
    np.testing.assert_array_equal(sd.icalc, [1, 1, 1])
    np.testing.assert_array_equal(sd.outseg, [3, 3, 0])
    np.testing.assert_array_equal(sd.iupseg, [0, 0, 0])
    np.testing.assert_array_equal(sd.iprior, [0, 0, 0])
    np.testing.assert_array_almost_equal(sd.flow, [0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(sd.runoff, [0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(sd.etsw, [0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(sd.pptsw, [0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(sd.roughch, [0.024, 0.024, 0.024])
    np.testing.assert_array_almost_equal(sd.hcond1, [1.0, 1.0, 1.0])
    np.testing.assert_array_almost_equal(sd.thickm1, [1.0, 1.0, 1.0])
    np.testing.assert_array_almost_equal(sd.width1, [10.0, 10.0, 10.0])
    np.testing.assert_array_almost_equal(sd.hcond2, [1.0, 1.0, 1.0])
    np.testing.assert_array_almost_equal(sd.thickm2, [1.0, 1.0, 1.0])
    np.testing.assert_array_almost_equal(sd.width2, [10.0, 10.0, 10.0])
    # Write output files
    outdir = tmpdir_factory.mktemp('n3d')
    m.model_ws = str(outdir)
    m.write_input()
    n.grid_cells.to_file(str(outdir.join('grid_cells.shp')))
    n.reaches.to_file(str(outdir.join('reaches.shp')))


def test_process_flopy_n3d_vars(n3d, tmpdir_factory):
    # Repeat, but with min_slope enforced, and other options
    n = n3d
    m = flopy.modflow.Modflow()
    flopy.modflow.ModflowDis(
        m, nlay=1, nrow=3, ncol=2, delr=20.0, delc=20.0, top=15.0, botm=10.0,
        xul=30.0, yul=130.0)
    flopy.modflow.ModflowBas(m)
    n.process_flopy(m, min_slope=0.03, hyd_cond1=12, thickness1=2.0,
                    inflow={0: 96.6, 1: 97.7, 2: 98.8}, flow={0: 5.4},
                    runoff={1: 5}, etsw={0: 3, 1: 9.1, 2: 6}, pptsw={2: 8.8})
    np.testing.assert_array_almost_equal(
        m.sfr.reach_data.rchlen,
        [18.027756, 6.009252, 12.018504, 21.081851, 10.540926, 10.0, 10.0])
    np.testing.assert_array_almost_equal(
        m.sfr.reach_data.strtop,
        [14.75, 14.416667, 14.166667, 14.666667, 14.166667, 13.5, 12.5])
    np.testing.assert_array_almost_equal(
        m.sfr.reach_data.slope,
        [0.03, 0.03, 0.03, 0.031622775, 0.031622775, 0.1, 0.1])
    np.testing.assert_array_equal(m.sfr.reach_data.strthick, [2.0] * 7)
    np.testing.assert_array_equal(m.sfr.reach_data.strhc1, [12.0] * 7)
    # Data set 6
    assert len(m.sfr.segment_data) == 1
    sd = m.sfr.segment_data[0]
    np.testing.assert_array_equal(sd.nseg, [1, 2, 3])
    np.testing.assert_array_equal(sd.icalc, [1, 1, 1])
    np.testing.assert_array_equal(sd.outseg, [3, 3, 0])
    np.testing.assert_array_equal(sd.iupseg, [0, 0, 0])
    np.testing.assert_array_equal(sd.iprior, [0, 0, 0])
    # note that 'inflow' was effectivley ignored, as is expected
    np.testing.assert_array_almost_equal(sd.flow, [0.0, 0.0, 5.4])
    np.testing.assert_array_almost_equal(sd.runoff, [5.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(sd.etsw, [9.1, 6.0, 3.0])
    np.testing.assert_array_almost_equal(sd.pptsw, [0.0, 8.8, 0.0])
    np.testing.assert_array_almost_equal(sd.roughch, [0.024, 0.024, 0.024])
    np.testing.assert_array_almost_equal(sd.hcond1, [12.0, 12.0, 12.0])
    np.testing.assert_array_almost_equal(sd.thickm1, [2.0, 2.0, 2.0])
    np.testing.assert_array_almost_equal(sd.width1, [10.0, 10.0, 10.0])
    np.testing.assert_array_almost_equal(sd.hcond2, [12.0, 12.0, 12.0])
    np.testing.assert_array_almost_equal(sd.thickm2, [2.0, 2.0, 2.0])
    np.testing.assert_array_almost_equal(sd.width2, [10.0, 10.0, 10.0])
    # Write output files
    outdir = tmpdir_factory.mktemp('n3d')
    m.model_ws = str(outdir)
    m.write_input()
    n.grid_cells.to_file(str(outdir.join('grid_cells.shp')))
    n.reaches.to_file(str(outdir.join('reaches.shp')))


def test_process_flopy_n2d_defaults(n2d, tmpdir_factory):
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
    sd = m.sfr.segment_data[0]
    assert list(sd.nseg) == [1, 2, 3]
    assert list(sd.icalc) == [1, 1, 1]
    assert list(sd.outseg) == [3, 3, 0]
    assert list(sd.iupseg) == [0, 0, 0]
    # See test_process_flopy_n3d_defaults for other checks
    # Write output files
    outdir = tmpdir_factory.mktemp('n2d')
    m.model_ws = str(outdir)
    m.write_input()
    n.grid_cells.to_file(str(outdir.join('grid_cells.shp')))
    n.reaches.to_file(str(outdir.join('reaches.shp')))


def test_process_flopy_n2d_min_slope(n2d, tmpdir_factory):
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
    # See test_process_flopy_n3d_defaults for other checks
    # Write output files
    outdir = tmpdir_factory.mktemp('n2d')
    m.model_ws = str(outdir)
    m.write_input()
    n.grid_cells.to_file(str(outdir.join('grid_cells.shp')))
    n.reaches.to_file(str(outdir.join('reaches.shp')))


def test_set_elevations(n2d, tmpdir_factory):
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
    _ = n.fix_segment_elevs(min_incise=0.2, min_slope=1.e-4)
    _ = n.reconcile_reach_strtop()
    _ = n.set_topbot_elevs_at_reaches()
    # n.plot_reaches_above(m, 'all', plot_bottom=True, points2=None)
    n.fix_reach_elevs()
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
    sd = m.sfr.segment_data[0]
    assert list(sd.nseg) == [1, 2, 3]
    assert list(sd.icalc) == [1, 1, 1]
    assert list(sd.outseg) == [3, 3, 0]
    assert list(sd.iupseg) == [0, 0, 0]
    # See test_process_flopy_n3d_defaults for other checks
    # Write output files
    outdir = tmpdir_factory.mktemp('n2d')
    m.model_ws = str(outdir)
    m.write_input()
    n.grid_cells.to_file(str(outdir.join('grid_cells.shp')))
    n.reaches.to_file(str(outdir.join('reaches.shp')))


def test_reach_barely_outside_ibound():
    n = swn.SurfaceWaterNetwork(wkt_to_geoseries([
        'LINESTRING (15 125, 70 90, 120 120, 130 90, '
        '150 110, 180 90, 190 110, 290 80)'
    ]))
    m = flopy.modflow.Modflow()
    flopy.modflow.ModflowDis(
        m, nrow=2, ncol=3, delr=100.0, delc=100.0, xul=0.0, yul=200.0)
    flopy.modflow.ModflowBas(m, ibound=np.array([[1, 1, 1], [0, 0, 0]]))
    n.process_flopy(m, reach_include_fraction=0.8)
    # Data set 1c
    assert abs(m.sfr.nstrm) == 3
    assert m.sfr.nss == 1
    # Data set 2
    assert list(m.sfr.reach_data.i) == [0, 0, 0]
    assert list(m.sfr.reach_data.j) == [0, 1, 2]
    assert list(m.sfr.reach_data.iseg) == [1, 1, 1]
    assert list(m.sfr.reach_data.ireach) == [1, 2, 3]
    np.testing.assert_array_almost_equal(
        m.sfr.reach_data.rchlen, [100.177734, 152.08736, 93.96276], 5)
    expected_reaches_geom = wkt_to_geoseries([
        'LINESTRING (15 125, 54.3 100, 70 90, 86.7 100, 100 108)',
        'LINESTRING (100 108, 120 120, 126.7 100, 130 90, 140 100, 150 110, '
        '165 100, 180 90, 185 100, 190 110, 200 107)',
        'LINESTRING (200 107, 223.3 100, 290 80)'])
    expected_reaches_geom.index += 1
    assert n.reaches.geom_almost_equals(expected_reaches_geom, 0).all()


@pytest.fixture
def costal_flopy_m():
    return flopy.modflow.Modflow.load('h.nam', model_ws=datadir, check=False)


def check_number_sum_hex(a, n, h):
    a = np.ceil(a).astype(np.int64)
    assert a.sum() == n
    ah = md5(a.tostring()).hexdigest()
    assert ah.startswith(h), '{0} does not start with {1}'.format(ah, h)


def test_costal_process_flopy(
        costal_swn, costal_flopy_m, clostal_flow_m, tmpdir_factory):
    n = costal_swn
    assert len(n) == 304
    m = costal_flopy_m
    # Make sure this is the model we are thinking of
    assert m.modelgrid.extent == (1802000.0, 1819000.0, 5861000.0, 5879000.0)
    n.process_flopy(m, inflow=clostal_flow_m)
    # Check remaining reaches added that are inside model domain
    reach_geom = n.reaches.loc[
        n.reaches['segnum'] == 3047735, 'geometry'].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 980.5448069140768)
    # These should be split between two cells
    reach_geoms = n.reaches.loc[
        n.reaches['segnum'] == 3047750, 'geometry']
    assert len(reach_geoms) == 2
    np.testing.assert_almost_equal(reach_geoms.iloc[0].length, 204.90164560019)
    np.testing.assert_almost_equal(reach_geoms.iloc[1].length, 789.59872070638)
    # This reach should not be extended, the remainder is too far away
    reach_geom = n.reaches.loc[
        n.reaches['segnum'] == 3047762, 'geometry'].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 261.4644731621629)
    # This reach should not be extended, the remainder is too long
    reach_geom = n.reaches.loc[
        n.reaches['segnum'] == 3047926, 'geometry'].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 237.72893664132727)
    # Data set 1c
    assert abs(m.sfr.nstrm) == 369
    assert m.sfr.nss == 184
    # Data set 2
    # check_number_sum_hex(
    #    m.sfr.reach_data.node, 49998, '29eb6a019a744893ceb5a09294f62638')
    # check_number_sum_hex(
    #    m.sfr.reach_data.k, 0, '213581ea1c4e2fa86e66227673da9542')
    # check_number_sum_hex(
    #    m.sfr.reach_data.i, 2690, 'be41f95d2eb64b956cc855304f6e5e1d')
    # check_number_sum_hex(
    #    m.sfr.reach_data.j, 4268, '4142617f1cbd589891e9c4033efb0243')
    # check_number_sum_hex(
    #    m.sfr.reach_data.reachID, 68635, '2a512563b164c76dfc605a91b10adae1')
    # check_number_sum_hex(
    #    m.sfr.reach_data.iseg, 34415, '48c4129d78c344d2e8086cd6971c16f7')
    # check_number_sum_hex(
    #    m.sfr.reach_data.ireach, 687, '233b71e88260cddb374e28ed197dfab0')
    # check_number_sum_hex(
    #    m.sfr.reach_data.rchlen, 159871, '776ed1ced406c7de9cfe502181dc8e97')
    # check_number_sum_hex(
    #    m.sfr.reach_data.strtop, 4266, '572a5ef53cd2c69f5d467f1056ee7579')
    # check_number_sum_hex(
    #   m.sfr.reach_data.slope * 999, 2945, '91c54e646fec7af346c0979167789316')
    # check_number_sum_hex(
    #    m.sfr.reach_data.strthick, 370, '09fd95bcbfe7c6309694157904acac68')
    # check_number_sum_hex(
    #    m.sfr.reach_data.strhc1, 370, '09fd95bcbfe7c6309694157904acac68')
    # Data set 6
    assert len(m.sfr.segment_data) == 1
    sd = m.sfr.segment_data[0]
    assert sd.flow.sum() > 0.0
    assert sd.pptsw.sum() == 0.0
    # check_number_sum_hex(
    #    sd.nseg, 17020, '55968016ecfb4e995fb5591bce55fea0')
    # check_number_sum_hex(
    #    sd.icalc, 184, '1e57e4eaa6f22ada05f4d8cd719e7876')
    # check_number_sum_hex(
    #    sd.outseg, 24372, '46730406d031de87aad40c2d13921f6a')
    # check_number_sum_hex(
    #    sd.iupseg, 0, 'f7e23bb7abe5b9603e8212ad467155bd')
    # check_number_sum_hex(
    #    sd.iprior, 0, 'f7e23bb7abe5b9603e8212ad467155bd')
    # check_number_sum_hex(
    #    sd.flow, 4009, '49b48704587dc36d5d6f6295569eabd6')
    # check_number_sum_hex(
    #    sd.runoff, 0, 'f7e23bb7abe5b9603e8212ad467155bd')
    # check_number_sum_hex(
    #    sd.etsw, 0, 'f7e23bb7abe5b9603e8212ad467155bd')
    # check_number_sum_hex(
    #    sd.pptsw, 0, 'f7e23bb7abe5b9603e8212ad467155bd')
    # check_number_sum_hex(
    #    sd.roughch * 1000, 4416, 'a1a620fac8f5a6cbed3cc49aa2b90467')
    # check_number_sum_hex(
    #    sd.hcond1, 184, '1e57e4eaa6f22ada05f4d8cd719e7876')
    # check_number_sum_hex(
    #    sd.thickm1, 184, '1e57e4eaa6f22ada05f4d8cd719e7876')
    # check_number_sum_hex(
    #    sd.width1, 1840, '5749f425818b3b18e395b2a432520a4e')
    # check_number_sum_hex(
    #    sd.hcond2, 184, '1e57e4eaa6f22ada05f4d8cd719e7876')
    # check_number_sum_hex(
    #    sd.thickm2, 184, '1e57e4eaa6f22ada05f4d8cd719e7876')
    # check_number_sum_hex(
    #    sd.width2, 1840, '5749f425818b3b18e395b2a432520a4e')
    # Check other packages
    check_number_sum_hex(
        m.bas6.ibound.array, 509, 'c4135a084b2593e0b69c148136a3ad6d')
    # Write output files
    outdir = tmpdir_factory.mktemp('costal')
    m.model_ws = str(outdir)
    m.write_input()
    n.grid_cells.to_file(str(outdir.join('grid_cells.shp')))
    n.reaches.to_file(str(outdir.join('reaches.shp')))


def test_costal_elevations(
        costal_swn, costal_flopy_m, clostal_flow_m, tmpdir_factory):
    n = costal_swn
    assert len(n) == 304
    m = costal_flopy_m
    # Make sure this is the model we are thinking of
    assert m.modelgrid.extent == (1802000.0, 1819000.0, 5861000.0, 5879000.0)
    n.process_flopy(m, inflow=clostal_flow_m)
    _ = n.set_topbot_elevs_at_reaches()
    n.plot_reaches_above(m, 'all', plot_bottom=True, points2=None)
    _ = n.fix_segment_elevs(min_incise=0.2, min_slope=1.e-4)
    _ = n.reconcile_reach_strtop()
    _ = n.set_topbot_elevs_at_reaches()
    n.plot_reaches_above(m, 'all', plot_bottom=True, points2=None)
    n.fix_reach_elevs()
    n.plot_reaches_above(m, 'all', plot_bottom=True, points2=None)


def test_costal_reduced_process_flopy(
        costal_lines_gdf, costal_flopy_m, clostal_flow_m, tmpdir_factory):
    n = swn.SurfaceWaterNetwork(costal_lines_gdf.geometry)
    assert len(n) == 304
    m = costal_flopy_m
    # Modify swn object
    n.remove(
        condition=n.segments['stream_order'] == 1,
        segnums=n.query(upstream=3047927))
    assert len(n) == 130
    n.process_flopy(m, inflow=clostal_flow_m)
    # These should be split between two cells
    reach_geoms = n.reaches.loc[
        n.reaches['segnum'] == 3047750, 'geometry']
    assert len(reach_geoms) == 2
    np.testing.assert_almost_equal(reach_geoms.iloc[0].length, 204.90164560019)
    np.testing.assert_almost_equal(reach_geoms.iloc[1].length, 789.59872070638)
    # This reach should not be extended, the remainder is too far away
    reach_geom = n.reaches.loc[
        n.reaches['segnum'] == 3047762, 'geometry'].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 261.4644731621629)
    # This reach should not be extended, the remainder is too long
    reach_geom = n.reaches.loc[
        n.reaches['segnum'] == 3047926, 'geometry'].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 237.72893664132727)
    # Data set 1c
    assert abs(m.sfr.nstrm) == 195
    assert m.sfr.nss == 94
    # Data set 2
    # check_number_sum_hex(
    #    m.sfr.reach_data.node, 49998, '29eb6a019a744893ceb5a09294f62638')
    # check_number_sum_hex(
    #    m.sfr.reach_data.k, 0, '213581ea1c4e2fa86e66227673da9542')
    # check_number_sum_hex(
    #    m.sfr.reach_data.i, 2690, 'be41f95d2eb64b956cc855304f6e5e1d')
    # check_number_sum_hex(
    #    m.sfr.reach_data.j, 4268, '4142617f1cbd589891e9c4033efb0243')
    # check_number_sum_hex(
    #    m.sfr.reach_data.reachID, 68635, '2a512563b164c76dfc605a91b10adae1')
    # check_number_sum_hex(
    #    m.sfr.reach_data.iseg, 34415, '48c4129d78c344d2e8086cd6971c16f7')
    # check_number_sum_hex(
    #    m.sfr.reach_data.ireach, 687, '233b71e88260cddb374e28ed197dfab0')
    # check_number_sum_hex(
    #    m.sfr.reach_data.rchlen, 159871, '776ed1ced406c7de9cfe502181dc8e97')
    # check_number_sum_hex(
    #    m.sfr.reach_data.strtop, 4266, '572a5ef53cd2c69f5d467f1056ee7579')
    # check_number_sum_hex(
    #   m.sfr.reach_data.slope * 999, 2945, '91c54e646fec7af346c0979167789316')
    # check_number_sum_hex(
    #    m.sfr.reach_data.strthick, 370, '09fd95bcbfe7c6309694157904acac68')
    # check_number_sum_hex(
    #    m.sfr.reach_data.strhc1, 370, '09fd95bcbfe7c6309694157904acac68')
    # Data set 6
    assert len(m.sfr.segment_data) == 1
    sd = m.sfr.segment_data[0]
    assert sd.flow.sum() > 0.0
    assert sd.pptsw.sum() == 0.0
    # check_number_sum_hex(
    #    sd.nseg, 17020, '55968016ecfb4e995fb5591bce55fea0')
    # check_number_sum_hex(
    #    sd.icalc, 184, '1e57e4eaa6f22ada05f4d8cd719e7876')
    # check_number_sum_hex(
    #    sd.outseg, 24372, '46730406d031de87aad40c2d13921f6a')
    # check_number_sum_hex(
    #    sd.iupseg, 0, 'f7e23bb7abe5b9603e8212ad467155bd')
    # check_number_sum_hex(
    #    sd.iprior, 0, 'f7e23bb7abe5b9603e8212ad467155bd')
    # check_number_sum_hex(
    #    sd.flow, 4009, '49b48704587dc36d5d6f6295569eabd6')
    # check_number_sum_hex(
    #    sd.runoff, 0, 'f7e23bb7abe5b9603e8212ad467155bd')
    # check_number_sum_hex(
    #    sd.etsw, 0, 'f7e23bb7abe5b9603e8212ad467155bd')
    # check_number_sum_hex(
    #    sd.pptsw, 0, 'f7e23bb7abe5b9603e8212ad467155bd')
    # check_number_sum_hex(
    #    sd.roughch * 1000, 4416, 'a1a620fac8f5a6cbed3cc49aa2b90467')
    # check_number_sum_hex(
    #    sd.hcond1, 184, '1e57e4eaa6f22ada05f4d8cd719e7876')
    # check_number_sum_hex(
    #    sd.thickm1, 184, '1e57e4eaa6f22ada05f4d8cd719e7876')
    # check_number_sum_hex(
    #    sd.width1, 1840, '5749f425818b3b18e395b2a432520a4e')
    # check_number_sum_hex(
    #    sd.hcond2, 184, '1e57e4eaa6f22ada05f4d8cd719e7876')
    # check_number_sum_hex(
    #    sd.thickm2, 184, '1e57e4eaa6f22ada05f4d8cd719e7876')
    # check_number_sum_hex(
    #    sd.width2, 1840, '5749f425818b3b18e395b2a432520a4e')
    # Check other packages
    # Write output files
    outdir = tmpdir_factory.mktemp('costal')
    m.model_ws = str(outdir)
    m.write_input()
    n.grid_cells.to_file(str(outdir.join('grid_cells.shp')))
    n.reaches.to_file(str(outdir.join('reaches.shp')))


def test_costal_process_flopy_ibound_modify(
        costal_swn, clostal_flow_m, tmpdir_factory):
    n = costal_swn
    assert len(n) == 304
    m = flopy.modflow.Modflow.load('h.nam', model_ws=datadir, check=False)
    n.process_flopy(m, ibound_action='modify', inflow=clostal_flow_m)
    # Check a remaining reach added that is outside model domain
    reach_geom = n.reaches.loc[
        n.reaches['segnum'] == 3048565, 'geometry'].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 647.316024023105)
    expected_geom = wkt.loads(
        'LINESTRING Z (1819072.5 5869685.1 4, 1819000 5869684.9 5.7, '
        '1818997.5 5869684.9 5.8, 1818967.5 5869654.9 5, '
        '1818907.5 5869654.8 4, 1818877.6 5869624.7 5, 1818787.5 5869624.5 6, '
        '1818757.6 5869594.5 5.1, 1818697.6 5869594.4 5.7, '
        '1818667.6 5869564.3 6.2, 1818607.6 5869564.2 4.7, '
        '1818577.6 5869534.1 5.6, 1818487.6 5869534 6.2)')
    reach_geom.almost_equals(expected_geom, 0)
    # Data set 1c
    assert abs(m.sfr.nstrm) == 626
    assert m.sfr.nss == 304
    # Data set 2
    check_number_sum_hex(
        m.sfr.reach_data.node, 95964, '52c2df8cb982061c4c0a39bbf865926f')
    check_number_sum_hex(
        m.sfr.reach_data.k, 0, '975d4ebfcacc6428ed80b7e319ed023a')
    check_number_sum_hex(
        m.sfr.reach_data.i, 5307, '7ad41ac8568ac5e45bbb95a89a50da12')
    check_number_sum_hex(
        m.sfr.reach_data.j, 5745, 'fc24e43745e3e09f5e84f63b07d32473')
    check_number_sum_hex(
        m.sfr.reach_data.reachID, 196251, '46356d0cbb4563e5d882e5fd2639c3e8')
    check_number_sum_hex(
        m.sfr.reach_data.iseg, 94974, '7bd775afa62ce9818fa6b1f715ecbb27')
    check_number_sum_hex(
        m.sfr.reach_data.ireach, 1173, '8008ac0cb8bf371c37c3e51236e44fd4')
    check_number_sum_hex(
        m.sfr.reach_data.rchlen, 255531, '72f89892d6e5e03c53106792e2695084')
    check_number_sum_hex(
        m.sfr.reach_data.strtop, 24142, 'bc96d80acc1b59c4d50759301ae2392a')
    check_number_sum_hex(
        m.sfr.reach_data.slope * 500, 6593, '0306817657dc6c85cb65c93f3fa15af0')
    check_number_sum_hex(
        m.sfr.reach_data.strthick, 626, 'a3aa65f110b20b57fc7f445aa743759f')
    check_number_sum_hex(
        m.sfr.reach_data.strhc1, 626, 'a3aa65f110b20b57fc7f445aa743759f')
    # Data set 6
    assert len(m.sfr.segment_data) == 1
    sd = m.sfr.segment_data[0]
    check_number_sum_hex(
        sd.nseg, 46360, '22126069af5cfa16460d6b5ee2c9e25e')
    check_number_sum_hex(
        sd.icalc, 304, '3665cd80c97966d0a740f0845e8b50e6')
    check_number_sum_hex(
        sd.outseg, 69130, 'bfd96b95f0d9e7c4cfa67fac834dcf37')
    check_number_sum_hex(
        sd.iupseg, 0, 'd6c6d43a06a3923eac7f03dcfe16f437')
    check_number_sum_hex(
        sd.iprior, 0, 'd6c6d43a06a3923eac7f03dcfe16f437')
    check_number_sum_hex(
        sd.flow, 0, 'd6c6d43a06a3923eac7f03dcfe16f437')
    check_number_sum_hex(
        sd.runoff, 0, 'd6c6d43a06a3923eac7f03dcfe16f437')
    check_number_sum_hex(
        sd.etsw, 0, 'd6c6d43a06a3923eac7f03dcfe16f437')
    check_number_sum_hex(
        sd.pptsw, 0, 'd6c6d43a06a3923eac7f03dcfe16f437')
    check_number_sum_hex(
        sd.roughch * 1000, 7296, 'fde9b5ef3863e60a5173b5949d495c09')
    check_number_sum_hex(
        sd.hcond1, 304, '3665cd80c97966d0a740f0845e8b50e6')
    check_number_sum_hex(
        sd.thickm1, 304, '3665cd80c97966d0a740f0845e8b50e6')
    check_number_sum_hex(
        sd.width1, 3040, '65f2c05e33613b359676244036d86689')
    check_number_sum_hex(
        sd.hcond2, 304, '3665cd80c97966d0a740f0845e8b50e6')
    check_number_sum_hex(
        sd.thickm2, 304, '3665cd80c97966d0a740f0845e8b50e6')
    check_number_sum_hex(
        sd.width2, 3040, '65f2c05e33613b359676244036d86689')
    # Check other packages
    check_number_sum_hex(
        m.bas6.ibound.array, 574, 'c1ae1c2676e735281aeed2c9301ec84c')


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
