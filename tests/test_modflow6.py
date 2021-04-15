# -*- coding: utf-8 -*-
import geopandas
import numpy as np
import os
import pandas as pd
import pickle
from hashlib import md5
from shapely import wkt
from shapely.geometry import Point
from shutil import which
from textwrap import dedent

import pytest

from .conftest import datadir, matplotlib, plt

try:
    import flopy
except ImportError:
    pytest.skip("skipping tests that require flopy", allow_module_level=True)

import swn
import swn.modflow
from swn.file import gdf_to_shapefile
from swn.spatial import interp_2d_to_3d, force_2d, wkt_to_geoseries

mf6_exe = which('mf6')
requires_mf6 = pytest.mark.skipif(not mf6_exe, reason='requires mf6')
if mf6_exe is None:
    mfnwt_exe = 'mf6'

# same valid network used in test_basic
n3d_lines = wkt_to_geoseries([
    'LINESTRING Z (60 100 14, 60  80 12)',
    'LINESTRING Z (40 130 15, 60 100 14)',
    'LINESTRING Z (70 130 15, 60 100 14)',
])


@pytest.fixture
def n3d():
    return swn.SurfaceWaterNetwork.from_lines(n3d_lines)


@pytest.fixture
def n2d():
    return swn.SurfaceWaterNetwork.from_lines(force_2d(n3d_lines))


def test_init_errors():
    with pytest.raises(ValueError, match="expected 'logger' to be Logger"):
        swn.SwnMf6(object())

@pytest.mark.xfail
def test_from_swn_flopy_errors(n3d):
    n = n3d
    n.segments = n.segments.copy()
    sim = flopy.mf6.MFSimulation(exe_name=mf6_exe)
    _ = flopy.mf6.ModflowTdis(sim, nper=4)
    m = flopy.mf6.ModflowGwf(sim)
    _ = flopy.mf6.ModflowGwfdis(
        m, nlay=1, nrow=3, ncol=2, delr=20.0, delc=20.0)

    with pytest.raises(
            ValueError,
            match='swn must be a SurfaceWaterNetwork object'):
        swn.SwnMf6.from_swn_flopy(object(), m)

    m.modelgrid.set_coord_info(epsg=2193)
    n.segments.crs = {'init': 'epsg:27200'}
    # with pytest.raises(
    #        ValueError,
    #        match='CRS for segments and modelgrid are different'):
    #    nm = swn.SwnMf6.from_swn_flopy(n, m)

    n.segments.crs = None
    with pytest.raises(
            ValueError,
            match='modelgrid extent does not cover segments extent'):
        swn.SwnMf6.from_swn_flopy(n, m)

    m.modelgrid.set_coord_info(xoff=30.0, yoff=70.0)

    with pytest.raises(ValueError, match='idomain_action must be one of'):
        swn.SwnMf6.from_swn_flopy(n, m, idomain_action='foo')

    with pytest.raises(ValueError, match='flow must be a dict or DataFrame'):
        swn.SwnMf6.from_swn_flopy(n, m, flow=1.1)

    with pytest.raises(
            ValueError,
            match=r'length of flow \(1\) is different than nper \(4\)'):
        swn.SwnMf6.from_swn_flopy(
            n, m, flow=pd.DataFrame(
                    {'1': [1.1]},
                    index=pd.DatetimeIndex(['1970-01-01'])))

    with pytest.raises(
            ValueError,
            match=r'flow\.index must be a pandas\.DatetimeIndex'):
        swn.SwnMf6.from_swn_flopy(
            n, m, flow=pd.DataFrame({'1': [1.1] * 4}))

    with pytest.raises(
            ValueError,
            match=r'flow\.index does not match expected \(1970\-01\-01, 1970'):
        swn.SwnMf6.from_swn_flopy(
            n, m, flow=pd.DataFrame(
                    {'1': 1.1},  # starts on the wrong day
                    index=pd.DatetimeIndex(['1970-01-02'] * 4) +
                    pd.TimedeltaIndex(range(4), 'days')))

    with pytest.raises(
            ValueError,
            match=r'flow\.columns\.dtype must be same as segments\.index\.dt'):
        swn.SwnMf6.from_swn_flopy(
            n, m, flow=pd.DataFrame(
                    {'s1': 1.1},  # can't convert key to int
                    index=pd.DatetimeIndex(['1970-01-01'] * 4) +
                    pd.TimedeltaIndex(range(4), 'days')))

    with pytest.raises(
            ValueError,
            match=r'flow\.columns \(or keys\) not found in segments\.index'):
        swn.SwnMf6.from_swn_flopy(
            n, m, flow=pd.DataFrame(
                    {'3': 1.1},  # segnum 3 does not exist
                    index=pd.DatetimeIndex(['1970-01-01'] * 4) +
                    pd.TimedeltaIndex(range(4), 'days')))

    # finally success!
    swn.SwnMf6.from_swn_flopy(
        n, m, flow=pd.DataFrame(
                {'1': 1.1, '3': 1.1},  # segnum 3 ignored
                index=pd.DatetimeIndex(['1970-01-01'] * 4) +
                pd.TimedeltaIndex(range(4), 'days')))


@pytest.mark.xfail
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
    outdir = tmpdir_factory.mktemp('n3d')
    # Create a simple MODFLOW model
    m = flopy.modflow.Modflow(version='mf2005', exe_name=mf2005_exe)
    _ = flopy.modflow.ModflowDis(
        m, nlay=1, nrow=3, ncol=2, delr=20.0, delc=20.0, top=15.0, botm=10.0,
        xul=30.0, yul=130.0)
    _ = flopy.modflow.ModflowOc(
        m, stress_period_data={
            (0, 0): ['print head', 'save head', 'save budget']})
    _ = flopy.modflow.ModflowBas(m, strt=15.0, stoper=5.0)
    _ = flopy.modflow.ModflowSip(m)
    _ = flopy.modflow.ModflowLpf(m, ipakcb=52, laytyp=0, hk=1e-2)
    _ = flopy.modflow.ModflowRch(m, ipakcb=52, rech=1e-4)
    nm = swn.SwnMf6.from_swn_flopy(n3d, m)
    m.sfr.ipakcb = 52
    m.sfr.istcb2 = -53
    m.add_output_file(53, extension='sfo', binflag=True)
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
    assert repr(nm) == dedent('''\
        <SwnMf6: flopy mf2005 'modflowtest'
          7 in reaches (reachID): [1, 2, ..., 6, 7]
          3 in segment_data (nseg): [1, 2, 3]
            3 from segments: [1, 2, 0]
            no diversions
          1 stress period with perlen: [1.0] />''')
    if matplotlib:
        _ = nm.plot()
        plt.close()
    # Run model and read outputs
    m.model_ws = str(outdir)
    m.write_input()
    success, buff = m.run_model()
    assert success
    hds_fname = str(outdir.join(m.name + '.hds'))
    cbc_fname = str(outdir.join(m.name + '.cbc'))
    sfo_fname = str(outdir.join(m.name + '.sfo'))
    heads = read_head(hds_fname)
    sl = read_budget(cbc_fname, 'STREAM LEAKAGE', nm.reaches, 'sfrleakage')
    sf = read_budget(sfo_fname, 'STREAMFLOW OUT', nm.reaches, 'sfr_Q')
    # Write some files
    nm.reaches.to_file(str(outdir.join('reaches.shp')))
    nm.grid_cells.to_file(str(outdir.join('grid_cells.shp')))
    gdf_to_shapefile(nm.segments, outdir.join('segments.shp'))
    # Check results
    assert heads.shape == (1, 3, 2)
    np.testing.assert_array_almost_equal(
        heads,
        np.array([[
                [14.604243, 14.409589],
                [14.172486, 13.251323],
                [13.861891, 12.751296]]], np.float32))
    np.testing.assert_array_almost_equal(
        sl['q'],
        np.array([-0.00859839, 0.00420513, 0.00439326, 0.0, 0.0,
                  -0.12359641, -0.12052996], np.float32))
    np.testing.assert_array_almost_equal(
        sf['q'],
        np.array([0.00859839, 0.00439326, 0.0, 0.0, 0.0,
                  0.12359641, 0.24412636], np.float32))


@pytest.mark.xfail
def test_model_property(n3d):
    nm = swn.SwnMf6()
    with pytest.raises(
            ValueError, match="'model' must be a flopy Modflow object"):
        nm.model = None

    m = flopy.modflow.Modflow()
    with pytest.raises(ValueError, match='DIS package required'):
        nm.model = m

    _ = flopy.modflow.ModflowDis(
        m, nlay=1, nrow=3, ncol=2, delr=20.0, delc=20.0, top=15.0, botm=10.0,
        xul=30.0, yul=130.0, start_datetime='2001-02-03')

    with pytest.raises(ValueError, match='BAS6 package required'):
        nm.model = m

    _ = flopy.modflow.ModflowBas(m, strt=15.0, stoper=5.0)

    # Success!
    nm.model = m

    pd.testing.assert_index_equal(
        nm.time_index,
        pd.DatetimeIndex(['2001-02-03'], dtype='datetime64[ns]'))

    # Swap model with same and with another
    nm.model = m
    m2 = flopy.modflow.Modflow()
    _ = flopy.modflow.ModflowDis(m2)
    _ = flopy.modflow.ModflowBas(m2)
    nm.model = m2


@pytest.mark.xfail
def test_set_segment_data():
    # Note that set_segment_data is used both internally and externally
    # Create a local swn object to modify
    n3d = swn.SurfaceWaterNetwork.from_lines(n3d_lines)
    # manually add outside flow from extra segnums, referenced with inflow
    n3d.segments.at[1, 'from_segnums'] = {3, 4}
    # Create a simple MODFLOW model object (don't write/run)
    m = flopy.modflow.Modflow(version='mf2005', exe_name=mf2005_exe)
    _ = flopy.modflow.ModflowDis(
        m, nlay=1, nrow=3, ncol=2, delr=20.0, delc=20.0, top=15.0, botm=10.0,
        xul=30.0, yul=130.0)
    _ = flopy.modflow.ModflowBas(m, strt=15.0, stoper=5.0)
    nm = swn.SwnMf6.from_swn_flopy(
        n3d, m, min_slope=0.03, hyd_cond1=2, thickness1=2.0,
        inflow={3: 9.6, 4: 9.7}, flow={1: 18.4},
        runoff={1: 5}, pptsw={2: 1.8}, etsw={0: 0.01, 1: 0.02, 2: 0.03})
    # Check only data set 6
    assert m.sfr.nss == 3
    assert len(m.sfr.segment_data) == 1
    sd = m.sfr.segment_data[0]
    np.testing.assert_array_equal(sd.nseg, [1, 2, 3])
    np.testing.assert_array_equal(sd.icalc, [1, 1, 1])
    np.testing.assert_array_equal(sd.outseg, [3, 3, 0])
    np.testing.assert_array_equal(sd.iupseg, [0, 0, 0])
    np.testing.assert_array_equal(sd.iprior, [0, 0, 0])
    # note that 'inflow' gets added to nseg 1 flow
    assert 'inflow_segnums' in nm.segment_data.columns
    np.testing.assert_array_equal(
        nm.segment_data.inflow_segnums, [set([3, 4]), None, None])
    np.testing.assert_array_almost_equal(
        sd.flow, np.array([18.4 + 9.6 + 9.7, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_almost_equal(sd.runoff, [5.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(sd.etsw, [0.02, 0.03, 0.01])
    np.testing.assert_array_almost_equal(sd.pptsw, [0.0, 1.8, 0.0])
    np.testing.assert_array_almost_equal(sd.roughch, [0.024, 0.024, 0.024])
    np.testing.assert_array_almost_equal(sd.hcond1, [2.0, 2.0, 2.0])
    np.testing.assert_array_almost_equal(sd.thickm1, [2.0, 2.0, 2.0])
    np.testing.assert_array_almost_equal(sd.width1, [10.0, 10.0, 10.0])
    np.testing.assert_array_almost_equal(sd.hcond2, [2.0, 2.0, 2.0])
    np.testing.assert_array_almost_equal(sd.thickm2, [2.0, 2.0, 2.0])
    np.testing.assert_array_almost_equal(sd.width2, [10.0, 10.0, 10.0])
    assert repr(nm) == dedent('''\
        <SwnMf6: flopy mf2005 'modflowtest'
          7 in reaches (reachID): [1, 2, ..., 6, 7]
          3 in segment_data (nseg): [1, 2, 3]
            3 from segments: [1, 2, 0]
            no diversions
          1 stress period with perlen: [1.0] />''')
    if matplotlib:
        _ = nm.plot()
        plt.close()

    # Re-write segment_data without arguments
    nm.set_segment_data()
    assert m.sfr.nss == 3
    assert len(m.sfr.segment_data) == 1
    sd = m.sfr.segment_data[0]
    np.testing.assert_array_equal(sd.nseg, [1, 2, 3])
    np.testing.assert_array_equal(sd.icalc, [1, 1, 1])
    np.testing.assert_array_equal(sd.outseg, [3, 3, 0])
    np.testing.assert_array_equal(sd.iupseg, [0, 0, 0])
    np.testing.assert_array_equal(sd.iprior, [0, 0, 0])
    # note that 'inflow' gets added to nseg 1 flow
    assert 'inflow_segnums' not in nm.segment_data.columns
    # These timeseries sets are now all zero
    np.testing.assert_array_equal(sd.flow, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(sd.runoff, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(sd.etsw, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(sd.pptsw, [0.0, 0.0, 0.0])
    # And these stationary datasets are unchanged
    np.testing.assert_array_almost_equal(sd.roughch, [0.024, 0.024, 0.024])
    np.testing.assert_array_almost_equal(sd.hcond1, [2.0, 2.0, 2.0])
    np.testing.assert_array_almost_equal(sd.thickm1, [2.0, 2.0, 2.0])
    np.testing.assert_array_almost_equal(sd.width1, [10.0, 10.0, 10.0])
    np.testing.assert_array_almost_equal(sd.hcond2, [2.0, 2.0, 2.0])
    np.testing.assert_array_almost_equal(sd.thickm2, [2.0, 2.0, 2.0])
    np.testing.assert_array_almost_equal(sd.width2, [10.0, 10.0, 10.0])
    assert repr(nm) == dedent('''\
        <SwnMf6: flopy mf2005 'modflowtest'
          7 in reaches (reachID): [1, 2, ..., 6, 7]
          3 in segment_data (nseg): [1, 2, 3]
            3 from segments: [1, 2, 0]
            no diversions
          1 stress period with perlen: [1.0] />''')
    if matplotlib:
        _ = nm.plot()
        plt.close()

    # Re-write segment_data with arguments
    # note that this network does not have diversions, so abstraction term will
    # not be used and generate a warning
    nm.set_segment_data(
        abstraction={0: 123.0}, runoff={1: 5.5}, inflow={3: 9.6, 4: 9.7})
    assert m.sfr.nss == 3
    assert len(m.sfr.segment_data) == 1
    sd = m.sfr.segment_data[0]
    np.testing.assert_array_equal(sd.nseg, [1, 2, 3])
    np.testing.assert_array_equal(sd.icalc, [1, 1, 1])
    np.testing.assert_array_equal(sd.outseg, [3, 3, 0])
    np.testing.assert_array_equal(sd.iupseg, [0, 0, 0])
    np.testing.assert_array_equal(sd.iprior, [0, 0, 0])
    assert 'inflow_segnums' in nm.segment_data.columns
    np.testing.assert_array_equal(
        nm.segment_data.inflow_segnums, [set([3, 4]), None, None])
    np.testing.assert_array_almost_equal(
        sd.flow, np.array([9.6 + 9.7, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_almost_equal(sd.runoff, [5.5, 0.0, 0.0])
    np.testing.assert_array_equal(sd.etsw, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(sd.pptsw, [0.0, 0.0, 0.0])
    # And these stationary datasets are unchanged
    np.testing.assert_array_almost_equal(sd.roughch, [0.024, 0.024, 0.024])
    np.testing.assert_array_almost_equal(sd.hcond1, [2.0, 2.0, 2.0])
    np.testing.assert_array_almost_equal(sd.thickm1, [2.0, 2.0, 2.0])
    np.testing.assert_array_almost_equal(sd.width1, [10.0, 10.0, 10.0])
    np.testing.assert_array_almost_equal(sd.hcond2, [2.0, 2.0, 2.0])
    np.testing.assert_array_almost_equal(sd.thickm2, [2.0, 2.0, 2.0])
    np.testing.assert_array_almost_equal(sd.width2, [10.0, 10.0, 10.0])

    # Use another model with multiple stress periods
    m = flopy.modflow.Modflow(version='mf2005', exe_name=mf2005_exe)
    _ = flopy.modflow.ModflowDis(
        m, nlay=1, nrow=3, ncol=2, nper=4, delr=20.0, delc=20.0,
        top=15.0, botm=10.0, xul=30.0, yul=130.0)
    _ = flopy.modflow.ModflowBas(m, strt=15.0, stoper=5.0)
    nm = swn.SwnMf6.from_swn_flopy(
        n3d, m,
        inflow={3: 9.6, 4: 9.7}, flow={1: [18.4, 13.1, 16.4, 9.2]},
        runoff={1: 5}, pptsw={2: [1.8, 0.2, 1.3, 0.9]},
        etsw={0: 0.01, 1: 0.02, 2: [0.03, 0.02, 0.03, 0.01]})
    # Check only data set 6
    assert m.sfr.nss == 3
    assert len(m.sfr.segment_data) == 4
    for iper in range(4):
        sd = m.sfr.segment_data[iper]
        np.testing.assert_array_equal(sd.nseg, [1, 2, 3])
        np.testing.assert_array_equal(sd.icalc, [1, 1, 1])
        np.testing.assert_array_equal(sd.outseg, [3, 3, 0])
        np.testing.assert_array_equal(sd.iupseg, [0, 0, 0])
        np.testing.assert_array_equal(sd.iprior, [0, 0, 0])
        np.testing.assert_array_almost_equal(sd.roughch, [0.024, 0.024, 0.024])
        np.testing.assert_array_almost_equal(sd.hcond1, [1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(sd.thickm1, [1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(sd.width1, [10.0, 10.0, 10.0])
        np.testing.assert_array_almost_equal(sd.hcond2, [1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(sd.thickm2, [1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(sd.width2, [10.0, 10.0, 10.0])
    assert 'inflow_segnums' in nm.segment_data.columns
    np.testing.assert_array_equal(
        nm.segment_data.inflow_segnums, [set([3, 4]), None, None])
    sd = m.sfr.segment_data[0]
    np.testing.assert_array_almost_equal(sd.flow, [9.6 + 9.7 + 18.4, 0.0, 0.0])
    np.testing.assert_array_almost_equal(sd.runoff, [5.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(sd.etsw, [0.02, 0.03, 0.01])
    np.testing.assert_array_almost_equal(sd.pptsw, [0.0, 1.8, 0.0])
    sd = m.sfr.segment_data[1]
    np.testing.assert_array_almost_equal(
        sd.flow, np.array([9.6 + 9.7 + 13.1, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_almost_equal(sd.runoff, [5.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(sd.etsw, [0.02, 0.02, 0.01])
    np.testing.assert_array_almost_equal(sd.pptsw, [0.0, 0.2, 0.0])
    sd = m.sfr.segment_data[2]
    np.testing.assert_array_almost_equal(
        sd.flow, np.array([9.6 + 9.7 + 16.4, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_almost_equal(sd.runoff, [5.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(sd.etsw, [0.02, 0.03, 0.01])
    np.testing.assert_array_almost_equal(sd.pptsw, [0.0, 1.3, 0.0])
    sd = m.sfr.segment_data[3]
    np.testing.assert_array_almost_equal(
        sd.flow, np.array([9.6 + 9.7 + 9.2, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_almost_equal(sd.runoff, [5.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(sd.etsw, [0.02, 0.01, 0.01])
    np.testing.assert_array_almost_equal(sd.pptsw, [0.0, 0.9, 0.0])
    assert repr(nm) == dedent('''\
        <SwnMf6: flopy mf2005 'modflowtest'
          7 in reaches (reachID): [1, 2, ..., 6, 7]
          3 in segment_data (nseg): [1, 2, 3]
            3 from segments: [1, 2, 0]
            no diversions
          4 stress periods with perlen: [1.0, 1.0, 1.0, 1.0] />''')
    if matplotlib:
        _ = nm.plot()
        plt.close()

    # Check errors
    with pytest.raises(ValueError, match='flow must be a dict or DataFrame'):
        nm.set_segment_data(flow=1.1)

    m.dis.nper = 4  # TODO: is this alowed?
    with pytest.raises(
            ValueError,
            match=r'length of flow \(1\) is different than nper \(4\)'):
        nm.set_segment_data(
            flow=pd.DataFrame(
                {'1': [1.1]},
                index=pd.DatetimeIndex(['1970-01-01'])))

    with pytest.raises(
            ValueError,
            match=r'flow\.index must be a pandas\.DatetimeIndex'):
        nm.set_segment_data(flow=pd.DataFrame({'1': [1.1] * 4}))

    with pytest.raises(
            ValueError,
            match=r'flow\.index does not match expected \(1970\-01\-01, 1970'):
        nm.set_segment_data(
            flow=pd.DataFrame(
                {'1': 1.1},  # starts on the wrong day
                index=pd.DatetimeIndex(['1970-01-02'] * 4) +
                pd.TimedeltaIndex(range(4), 'days')))

    with pytest.raises(
            ValueError,
            match=r'flow\.columns\.dtype must be same as segments\.index\.dt'):
        nm.set_segment_data(
            flow=pd.DataFrame(
                {'s1': 1.1},  # can't convert key to int
                index=pd.DatetimeIndex(['1970-01-01'] * 4) +
                pd.TimedeltaIndex(range(4), 'days')))

    with pytest.raises(
            ValueError,
            match=r'flow\.columns \(or keys\) not found in segments\.index'):
        nm.set_segment_data(
            flow=pd.DataFrame(
                {'3': 1.1},  # segnum 3 does not exist
                index=pd.DatetimeIndex(['1970-01-01'] * 4) +
                pd.TimedeltaIndex(range(4), 'days')))


@pytest.mark.xfail
def test_process_flopy_n3d_vars(tmpdir_factory):
    # Repeat, but with min_slope enforced, and other options
    outdir = tmpdir_factory.mktemp('n3d')
    # Create a local swn object to modify
    n3d = swn.SurfaceWaterNetwork.from_lines(n3d_lines)
    # manually add outside flow from extra segnums, referenced with inflow
    n3d.segments.at[1, 'from_segnums'] = {3, 4}
    # Create a simple MODFLOW model
    m = flopy.modflow.Modflow(version='mf2005', exe_name=mf2005_exe)
    _ = flopy.modflow.ModflowDis(
        m, nlay=1, nrow=3, ncol=2, delr=20.0, delc=20.0, top=15.0, botm=10.0,
        xul=30.0, yul=130.0)
    _ = flopy.modflow.ModflowOc(
        m, stress_period_data={
            (0, 0): ['print head', 'save head', 'save budget']})
    _ = flopy.modflow.ModflowBas(m, strt=15.0, stoper=5.0)
    _ = flopy.modflow.ModflowSip(m)
    _ = flopy.modflow.ModflowLpf(m, ipakcb=52, laytyp=0, hk=1.0)
    _ = flopy.modflow.ModflowRch(m, ipakcb=52, rech=0.01)
    nm = swn.SwnMf6.from_swn_flopy(
        n3d, m, min_slope=0.03, hyd_cond1=2, thickness1=2.0,
        inflow={3: 9.6, 4: 9.7}, flow={1: 18.4},
        runoff={1: 5}, pptsw={2: 1.8}, etsw={0: 0.01, 1: 0.02, 2: 0.03})
    m.sfr.ipakcb = 52
    m.sfr.istcb2 = -53
    m.add_output_file(53, extension='sfo', binflag=True)
    # Data set 2
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
    np.testing.assert_array_equal(m.sfr.reach_data.strhc1, [2.0] * 7)
    # Data set 6
    assert len(m.sfr.segment_data) == 1
    sd = m.sfr.segment_data[0]
    np.testing.assert_array_equal(sd.nseg, [1, 2, 3])
    np.testing.assert_array_equal(sd.icalc, [1, 1, 1])
    np.testing.assert_array_equal(sd.outseg, [3, 3, 0])
    np.testing.assert_array_equal(sd.iupseg, [0, 0, 0])
    np.testing.assert_array_equal(sd.iprior, [0, 0, 0])
    # note that 'inflow' gets added to nseg 1 flow
    np.testing.assert_array_equal(
        nm.segment_data.inflow_segnums, [set([3, 4]), None, None])
    np.testing.assert_array_almost_equal(sd.flow, [18.4 + 9.6 + 9.7, 0.0, 0.0])
    np.testing.assert_array_almost_equal(sd.runoff, [5.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(sd.etsw, [0.02, 0.03, 0.01])
    np.testing.assert_array_almost_equal(sd.pptsw, [0.0, 1.8, 0.0])
    np.testing.assert_array_almost_equal(sd.roughch, [0.024, 0.024, 0.024])
    np.testing.assert_array_almost_equal(sd.hcond1, [2.0, 2.0, 2.0])
    np.testing.assert_array_almost_equal(sd.thickm1, [2.0, 2.0, 2.0])
    np.testing.assert_array_almost_equal(sd.width1, [10.0, 10.0, 10.0])
    np.testing.assert_array_almost_equal(sd.hcond2, [2.0, 2.0, 2.0])
    np.testing.assert_array_almost_equal(sd.thickm2, [2.0, 2.0, 2.0])
    np.testing.assert_array_almost_equal(sd.width2, [10.0, 10.0, 10.0])
    assert repr(nm) == dedent('''\
        <SwnMf6: flopy mf2005 'modflowtest'
          7 in reaches (reachID): [1, 2, ..., 6, 7]
          3 in segment_data (nseg): [1, 2, 3]
            3 from segments: [1, 2, 0]
            no diversions
          1 stress period with perlen: [1.0] />''')
    if matplotlib:
        _ = nm.plot()
        plt.close()
    # Run model and read outputs
    m.model_ws = str(outdir)
    m.write_input()
    success, buff = m.run_model()
    assert success
    hds_fname = str(outdir.join(m.name + '.hds'))
    cbc_fname = str(outdir.join(m.name + '.cbc'))
    sfo_fname = str(outdir.join(m.name + '.sfo'))
    heads = read_head(hds_fname)
    sl = read_budget(cbc_fname, 'STREAM LEAKAGE', nm.reaches, 'sfrleakage')
    sf = read_budget(sfo_fname, 'STREAMFLOW OUT', nm.reaches, 'sfr_Q')
    # Write some files
    nm.reaches.to_file(str(outdir.join('reaches.shp')))
    nm.grid_cells.to_file(str(outdir.join('grid_cells.shp')))
    gdf_to_shapefile(nm.segments, outdir.join('segments.shp'))
    # Check results
    assert heads.shape == (1, 3, 2)
    np.testing.assert_array_almost_equal(
        heads,
        np.array([[
                [14.620145, 14.489456],
                [14.494376, 13.962832],
                [14.100152, 12.905928]]], np.float32))
    np.testing.assert_array_almost_equal(
        sl['q'],
        np.array([-2.717792, -4.734348, 36.266556, 2.713955, 30.687397,
                  -70.960304, -15.255642], np.float32))
    np.testing.assert_array_almost_equal(
        sf['q'],
        np.array([39.31224, 43.67807, 6.67448, 370.4348, 526.3218,
                  602.95654, 617.21216], np.float32))


@pytest.mark.xfail
def test_process_flopy_n2d_defaults(n2d, tmpdir_factory):
    # similar to 3D version, but getting information from model
    outdir = tmpdir_factory.mktemp('n2d')
    # Create a simple MODFLOW model
    top = np.array([
        [16.0, 15.0],
        [15.0, 15.0],
        [14.0, 14.0],
    ])
    m = flopy.modflow.Modflow(version='mf2005', exe_name=mf2005_exe)
    flopy.modflow.ModflowDis(
        m, nlay=1, nrow=3, ncol=2, delr=20.0, delc=20.0, top=top, botm=10.0,
        xul=30.0, yul=130.0)
    _ = flopy.modflow.ModflowOc(
        m, stress_period_data={
            (0, 0): ['print head', 'save head', 'save budget']})
    _ = flopy.modflow.ModflowBas(m, strt=top, stoper=5.0)
    _ = flopy.modflow.ModflowSip(m)
    _ = flopy.modflow.ModflowLpf(m, ipakcb=52, laytyp=0, hk=1.0)
    _ = flopy.modflow.ModflowRch(m, ipakcb=52, rech=0.01)
    nm = swn.SwnMf6.from_swn_flopy(n2d, m)
    m.sfr.ipakcb = 52
    m.sfr.istcb2 = -53
    m.add_output_file(53, extension='sfo', binflag=True)
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
    assert repr(nm) == dedent('''\
        <SwnMf6: flopy mf2005 'modflowtest'
          7 in reaches (reachID): [1, 2, ..., 6, 7]
          3 in segment_data (nseg): [1, 2, 3]
            3 from segments: [1, 2, 0]
            no diversions
          1 stress period with perlen: [1.0] />''')
    if matplotlib:
        _ = nm.plot()
        plt.close()
    # Run model
    m.model_ws = str(outdir)
    m.write_input()
    success, buff = m.run_model()
    assert not success
    # Error/warning: upstream elevation is equal to downstream, slope is zero
    # TODO: improve processing to correct elevation errors
    nm.reaches.to_file(str(outdir.join('reaches.shp')))
    nm.grid_cells.to_file(str(outdir.join('grid_cells.shp')))
    gdf_to_shapefile(nm.segments, outdir.join('segments.shp'))


@pytest.mark.xfail
def test_process_flopy_n2d_min_slope(n2d, tmpdir_factory):
    outdir = tmpdir_factory.mktemp('n2d')
    # Create a simple MODFLOW model
    top = np.array([
        [16.0, 15.0],
        [15.0, 15.0],
        [14.0, 14.0],
    ])
    m = flopy.modflow.Modflow(version='mf2005', exe_name=mf2005_exe)
    _ = flopy.modflow.ModflowDis(
        m, nlay=1, nrow=3, ncol=2, delr=20.0, delc=20.0, top=top, botm=10.0,
        xul=30.0, yul=130.0)
    _ = flopy.modflow.ModflowOc(
        m, stress_period_data={
            (0, 0): ['print head', 'save head', 'save budget']})
    _ = flopy.modflow.ModflowBas(m, strt=top, stoper=5.0)
    _ = flopy.modflow.ModflowSip(m)
    _ = flopy.modflow.ModflowLpf(m, ipakcb=52, laytyp=0, hk=1.0)
    _ = flopy.modflow.ModflowRch(m, ipakcb=52, rech=0.01)
    nm = swn.SwnMf6.from_swn_flopy(n2d, m, min_slope=0.03)
    m.sfr.ipakcb = 52
    m.sfr.istcb2 = -53
    # Data set 2
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
    # Run model and read outputs
    m.model_ws = str(outdir)
    m.write_input()
    success, buff = m.run_model()
    assert not success
    # Error/warning: upstream elevation is equal to downstream, slope is zero
    # TODO: improve processing to correct elevation errors
    nm.reaches.to_file(str(outdir.join('reaches.shp')))
    nm.grid_cells.to_file(str(outdir.join('grid_cells.shp')))
    gdf_to_shapefile(nm.segments, outdir.join('segments.shp'))


@pytest.mark.xfail
def test_process_flopy_interp_2d_to_3d(tmpdir_factory):
    # similar to 3D version, but getting information from model
    outdir = tmpdir_factory.mktemp('interp_2d_to_3d')
    # Create a simple MODFLOW model
    top = np.array([
        [16.0, 15.0],
        [15.0, 15.0],
        [14.0, 14.0],
    ])
    m = flopy.modflow.Modflow(version='mf2005', exe_name=mf2005_exe)
    flopy.modflow.ModflowDis(
        m, nlay=1, nrow=3, ncol=2, delr=20.0, delc=20.0, top=top, botm=10.0,
        xul=30.0, yul=130.0)
    _ = flopy.modflow.ModflowOc(
        m, stress_period_data={
            (0, 0): ['print head', 'save head', 'save budget']})
    _ = flopy.modflow.ModflowBas(m, strt=top, stoper=5.0)
    _ = flopy.modflow.ModflowSip(m)
    _ = flopy.modflow.ModflowLpf(m, ipakcb=52, laytyp=0, hk=1.0)
    _ = flopy.modflow.ModflowRch(m, ipakcb=52, rech=0.01)
    gt = swn.modflow.geotransform_from_flopy(m)
    n = swn.SurfaceWaterNetwork.from_lines(interp_2d_to_3d(n3d_lines, top, gt))
    n.adjust_elevation_profile()
    nm = swn.SwnMf6.from_swn_flopy(n, m)
    m.sfr.ipakcb = 52
    m.sfr.istcb2 = -53
    m.add_output_file(53, extension='sfo', binflag=True)
    # Data set 1c
    assert abs(m.sfr.nstrm) == 7
    assert m.sfr.nss == 3
    # Data set 2
    np.testing.assert_array_almost_equal(
        m.sfr.reach_data.rchlen,
        [18.027756, 6.009252, 12.018504, 21.081851, 10.540926, 10.0, 10.0])
    np.testing.assert_array_almost_equal(
        m.sfr.reach_data.strtop,
        [15.742094, 15.39822, 15.140314, 14.989459, 14.973648, 14.726283,
         14.242094])
    np.testing.assert_array_almost_equal(
        m.sfr.reach_data.slope,
        [0.02861207, 0.02861207, 0.02861207, 0.001, 0.001, 0.04841886,
         0.04841886])
    sd = m.sfr.segment_data[0]
    assert list(sd.nseg) == [1, 2, 3]
    assert list(sd.icalc) == [1, 1, 1]
    assert list(sd.outseg) == [3, 3, 0]
    assert list(sd.iupseg) == [0, 0, 0]
    # See test_process_flopy_n3d_defaults for other checks
    # Run model and read outputs
    m.model_ws = str(outdir)
    m.write_input()
    success, buff = m.run_model()
    assert success
    hds_fname = str(outdir.join(m.name + '.hds'))
    cbc_fname = str(outdir.join(m.name + '.cbc'))
    sfo_fname = str(outdir.join(m.name + '.sfo'))
    heads = read_head(hds_fname)
    sl = read_budget(cbc_fname, 'STREAM LEAKAGE', nm.reaches, 'sfrleakage')
    sf = read_budget(sfo_fname, 'STREAMFLOW OUT', nm.reaches, 'sfr_Q')
    # Write some files
    nm.reaches.to_file(str(outdir.join('reaches.shp')))
    nm.grid_cells.to_file(str(outdir.join('grid_cells.shp')))
    gdf_to_shapefile(nm.segments, outdir.join('segments.shp'))
    # Check results
    assert heads.shape == (1, 3, 2)
    np.testing.assert_array_almost_equal(
        heads,
        np.array([[
                [15.595171, 15.015385],
                [15.554525, 14.750549],
                [15.509117, 14.458664]]], np.float32))
    np.testing.assert_array_almost_equal(
        sl['q'],
        np.array([-0.61594236, 0.61594236, 0.0, -6.4544363, 6.4544363,
                  -14.501283, -9.499095], np.float32))
    np.testing.assert_array_almost_equal(
        sf['q'],
        np.array([0.61594236, 0.0,  0.0, 6.4544363, 0.0,
                  14.501283, 24.000378], np.float32))


@pytest.mark.xfail
def test_set_elevations(n2d, tmpdir_factory):
    # similar to 3D version, but getting information from model
    outdir = tmpdir_factory.mktemp('n2d')
    # Create a simple MODFLOW model
    top = np.array([
        [16.0, 15.0],
        [15.0, 15.0],
        [14.0, 14.0],
    ])
    m = flopy.modflow.Modflow(version='mf2005', exe_name=mf2005_exe)
    _ = flopy.modflow.ModflowDis(
        m, nlay=1, nrow=3, ncol=2, delr=20.0, delc=20.0, top=top, botm=10.0,
        xul=30.0, yul=130.0)
    _ = flopy.modflow.ModflowOc(
        m, stress_period_data={
            (0, 0): ['print head', 'save head', 'save budget']})
    _ = flopy.modflow.ModflowBas(m, strt=top, stoper=5.0)
    _ = flopy.modflow.ModflowSip(m)
    _ = flopy.modflow.ModflowLpf(m, ipakcb=52, laytyp=0, hk=1.0)
    _ = flopy.modflow.ModflowRch(m, ipakcb=52, rech=0.01)
    nm = swn.SwnMf6.from_swn_flopy(n2d, m)
    # fix elevations
    _ = nm.set_topbot_elevs_at_reaches()
    seg_data = nm.set_segment_data(return_dict=True)
    reach_data = nm.get_reach_data()
    _ = flopy.modflow.mfsfr2.ModflowSfr2(
        model=m, reach_data=reach_data, segment_data=seg_data)
    if matplotlib:
        nm.plot_reaches_above(m, 'all', plot_bottom=True)
        plt.close()
    _ = nm.fix_segment_elevs(min_incise=0.2, min_slope=1.e-4)
    _ = nm.reconcile_reach_strtop()
    seg_data = nm.set_segment_data(return_dict=True)
    reach_data = nm.get_reach_data()
    _ = flopy.modflow.mfsfr2.ModflowSfr2(
        model=m, reach_data=reach_data, segment_data=seg_data)
    if matplotlib:
        nm.plot_reaches_above(m, 'all', plot_bottom=True)
        plt.close()
        nm.plot_reaches_above(m, 1)
        plt.close()
    _ = nm.set_topbot_elevs_at_reaches()
    nm.fix_reach_elevs()
    seg_data = nm.set_segment_data(return_dict=True)
    reach_data = nm.get_reach_data()
    _ = flopy.modflow.mfsfr2.ModflowSfr2(
        model=m, reach_data=reach_data, segment_data=seg_data)
    if matplotlib:
        nm.plot_reaches_above(m, 'all', plot_bottom=True)
        plt.close()
    m.sfr.ipakcb = 52
    m.sfr.istcb2 = -53
    m.add_output_file(53, extension='sfo', binflag=True)
    # Data set 1c
    assert abs(m.sfr.nstrm) == 7
    assert m.sfr.nss == 3
    # Data set 2
    np.testing.assert_array_almost_equal(
        m.sfr.reach_data.rchlen,
        [18.027756, 6.009252, 12.018504, 21.081851, 10.540926, 10.0, 10.0])
    # TODO testy testy
    # np.testing.assert_array_equal(
    #     m.sfr.reach_data.strtop,
    #     [16.0, 15.0, 15.0, 15.0, 15.0, 15.0, 14.0])
    np.testing.assert_array_almost_equal(
        m.sfr.reach_data.slope,
        [0.070710681, 0.05, 0.025, 0.05, 0.025, 0.025, 0.05])
    sd = m.sfr.segment_data[0]
    assert list(sd.nseg) == [1, 2, 3]
    assert list(sd.icalc) == [1, 1, 1]
    assert list(sd.outseg) == [3, 3, 0]
    assert list(sd.iupseg) == [0, 0, 0]
    # See test_process_flopy_n3d_defaults for other checks
    # Run model
    m.model_ws = str(outdir)
    m.write_input()
    success, buff = m.run_model()
    assert success
    hds_fname = str(outdir.join(m.name + '.hds'))
    cbc_fname = str(outdir.join(m.name + '.cbc'))
    sfo_fname = str(outdir.join(m.name + '.sfo'))
    heads = read_head(hds_fname)
    sl = read_budget(cbc_fname, 'STREAM LEAKAGE', nm.reaches, 'sfrleakage')
    sf = read_budget(sfo_fname, 'STREAMFLOW OUT', nm.reaches, 'sfr_Q')
    # Write some files
    nm.reaches.to_file(str(outdir.join('reaches.shp')))
    nm.grid_cells.to_file(str(outdir.join('grid_cells.shp')))
    gdf_to_shapefile(nm.segments, outdir.join('segments.shp'))
    # Check results
    assert heads.shape == (1, 3, 2)
    np.testing.assert_array_almost_equal(
        heads,
        np.array([[
                [15.4999275, 14.832507],
                [15.434015, 14.678202],
                [15.303412, 14.1582985]]], np.float32))
    np.testing.assert_array_almost_equal(
        sl['q'],
        np.array([0.0, 0.0, 0.0, -6.8689923, 6.8689923,
                  -13.108882, -10.891137], np.float32))
    np.testing.assert_array_almost_equal(
        sf['q'],
        np.array([0.0, 0.0, 0.0, 6.8689923, 0.0,
                  13.108882, 24.00002], np.float32))


@pytest.mark.xfail
def test_reach_barely_outside_idomain():
    n = swn.SurfaceWaterNetwork.from_lines(wkt_to_geoseries([
        'LINESTRING (15 125, 70 90, 120 120, 130 90, '
        '150 110, 180 90, 190 110, 290 80)'
    ]))
    m = flopy.modflow.Modflow(version='mf2005', exe_name=mf2005_exe)
    flopy.modflow.ModflowDis(
        m, nrow=2, ncol=3, delr=100.0, delc=100.0, xul=0.0, yul=200.0)
    flopy.modflow.ModflowBas(m, idomain=np.array([[1, 1, 1], [0, 0, 0]]))
    nm = swn.SwnMf6.from_swn_flopy(n, m, reach_include_fraction=0.8)
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
    assert nm.reaches.geom_almost_equals(expected_reaches_geom, 0).all()
    assert repr(nm) == dedent('''\
        <SwnMf6: flopy mf2005 'modflowtest'
          3 in reaches (reachID): [1, 2, 3]
          1 in segment_data (nseg): [1]
            1 from segments: [0]
            no diversions
          1 stress period with perlen: [1.0] />''')
    if matplotlib:
        _ = nm.plot()
        plt.close()


def check_number_sum_hex(a, n, h):
    a = np.ceil(a).astype(np.int64)
    assert a.sum() == n
    ah = md5(a.tobytes()).hexdigest()
    assert ah.startswith(h), '{0} does not start with {1}'.format(ah, h)


@pytest.mark.xfail
def test_coastal_process_flopy(tmpdir_factory,
                               coastal_lines_gdf, coastal_flow_m):
    outdir = tmpdir_factory.mktemp('coastal')
    # Load a MODFLOW model
    m = flopy.modflow.Modflow.load(
        'h.nam', version='mfnwt', exe_name=mfnwt_exe, model_ws=datadir,
        check=False)
    m.model_ws = str(outdir)
    # this model works without SFR
    m.write_input()
    success, buff = m.run_model()
    assert success
    # Create a SWN with adjusted elevation profiles
    n = swn.SurfaceWaterNetwork.from_lines(coastal_lines_gdf.geometry)
    n.adjust_elevation_profile()
    nm = swn.SwnMf6.from_swn_flopy(n, m, inflow=coastal_flow_m)
    m.sfr.unit_number = [24]  # WARNING: unit 17 of package SFR already in use
    m.sfr.ipakcb = 50
    m.sfr.istcb2 = -51
    m.add_output_file(51, extension='sfo', binflag=True)
    # and breaks with default SFR due to elevation errors
    m.write_input()
    success, buff = m.run_model()
    assert not success
    # Check dataframes
    assert len(nm.segments) == 304
    assert nm.segments['in_model'].sum() == 184
    # Check remaining reaches added that are inside model domain
    reach_geom = nm.reaches.loc[
        nm.reaches['segnum'] == 3047735, 'geometry'].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 980.5448069140768)
    # These should be split between two cells
    reach_geoms = nm.reaches.loc[
        nm.reaches['segnum'] == 3047750, 'geometry']
    assert len(reach_geoms) == 2
    np.testing.assert_almost_equal(reach_geoms.iloc[0].length, 204.90164560019)
    np.testing.assert_almost_equal(reach_geoms.iloc[1].length, 789.59872070638)
    # This reach should not be extended, the remainder is too far away
    reach_geom = nm.reaches.loc[
        nm.reaches['segnum'] == 3047762, 'geometry'].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 261.4644731621629)
    # This reach should not be extended, the remainder is too long
    reach_geom = nm.reaches.loc[
        nm.reaches['segnum'] == 3047926, 'geometry'].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 237.72893664132727)
    # Data set 1c
    assert abs(m.sfr.nstrm) == 296
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
        m.dis.idomain.array, 509, 'c4135a084b2593e0b69c148136a3ad6d')
    assert repr(nm) == dedent('''\
    <SwnMf6: flopy mfnwt 'h'
      296 in reaches (reachID): [1, 2, ..., 295, 296]
      184 in segment_data (nseg): [1, 2, ..., 183, 184]
        184 from segments (61% used): [3049818, 3049819, ..., 3046952, 3046736]
        no diversions
      1 stress period with perlen: [1.0] />''')
    if matplotlib:
        _ = nm.plot()
        plt.close()
    # Write output files
    nm.grid_cells.to_file(str(outdir.join('grid_cells.shp')))
    nm.reaches.to_file(str(outdir.join('reaches.shp')))
    gdf_to_shapefile(nm.segments, outdir.join('segments.shp'))


@pytest.mark.xfail
def test_coastal_elevations(coastal_swn, coastal_flow_m, tmpdir_factory):
    outdir = tmpdir_factory.mktemp('coastal')
    # Load a MODFLOW model
    m = flopy.modflow.Modflow.load(
        'h.nam', version='mfnwt', exe_name=mfnwt_exe, model_ws=datadir,
        check=False)
    m.model_ws = str(outdir)
    nm = swn.SwnMf6.from_swn_flopy(coastal_swn, m, inflow=coastal_flow_m)
    _ = nm.set_topbot_elevs_at_reaches()
    seg_data = nm.set_segment_data(return_dict=True)
    reach_data = nm.get_reach_data()
    flopy.modflow.mfsfr2.ModflowSfr2(
        model=m, reach_data=reach_data, segment_data=seg_data)
    if matplotlib:
        nm.plot_reaches_above(m, 'all', plot_bottom=False)
        plt.close()
    # handy to set a max elevation that a stream can be
    _ = nm.get_seg_ijk()
    tops = nm.get_top_elevs_at_segs().top_up
    max_str_z = tops.describe()['75%']
    if matplotlib:
        for seg in nm.segment_data.index[nm.segment_data.index.isin([1, 18])]:
            nm.plot_reaches_above(m, seg)
            plt.close()
    _ = nm.fix_segment_elevs(min_incise=0.2, min_slope=1.e-4,
                             max_str_z=max_str_z)
    _ = nm.reconcile_reach_strtop()
    seg_data = nm.set_segment_data(return_dict=True)
    reach_data = nm.get_reach_data()
    flopy.modflow.mfsfr2.ModflowSfr2(
        model=m, reach_data=reach_data, segment_data=seg_data)
    if matplotlib:
        nm.plot_reaches_above(m, 'all', plot_bottom=False)
        plt.close()
        for seg in nm.segment_data.index[nm.segment_data.index.isin([1, 18])]:
            nm.plot_reaches_above(m, seg)
            plt.close()
    _ = nm.set_topbot_elevs_at_reaches()
    nm.fix_reach_elevs()
    seg_data = nm.set_segment_data(return_dict=True)
    reach_data = nm.get_reach_data()
    flopy.modflow.mfsfr2.ModflowSfr2(
        model=m, reach_data=reach_data, segment_data=seg_data)
    if matplotlib:
        nm.plot_reaches_above(m, 'all', plot_bottom=False)
        plt.close()
        for seg in nm.segment_data.index[nm.segment_data.index.isin([1, 18])]:
            nm.plot_reaches_above(m, seg)
            plt.close()
    m.sfr.unit_number = [24]
    m.sfr.ipakcb = 50
    m.sfr.istcb2 = -51
    m.add_output_file(51, extension='sfo', binflag=True)
    # Run model
    m.model_ws = str(outdir)
    m.write_input()
    success, buff = m.run_model()
    assert success


@pytest.mark.xfail
def test_coastal_reduced_process_flopy(
        coastal_lines_gdf, coastal_flow_m, tmpdir_factory):
    outdir = tmpdir_factory.mktemp('coastal')
    n = swn.SurfaceWaterNetwork.from_lines(coastal_lines_gdf.geometry)
    assert len(n) == 304
    # Modify swn object
    n.remove(
        condition=n.segments['stream_order'] == 1,
        segnums=n.query(upstream=3047927))
    assert len(n) == 130
    # Load a MODFLOW model
    m = flopy.modflow.Modflow.load(
        'h.nam', version='mfnwt', exe_name=mfnwt_exe, model_ws=datadir,
        check=False)
    nm = swn.SwnMf6.from_swn_flopy(n, m, inflow=coastal_flow_m)
    # These should be split between two cells
    reach_geoms = nm.reaches.loc[
        nm.reaches['segnum'] == 3047750, 'geometry']
    assert len(reach_geoms) == 2
    np.testing.assert_almost_equal(reach_geoms.iloc[0].length, 204.90164560019)
    np.testing.assert_almost_equal(reach_geoms.iloc[1].length, 789.59872070638)
    # This reach should not be extended, the remainder is too far away
    reach_geom = nm.reaches.loc[
        nm.reaches['segnum'] == 3047762, 'geometry'].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 261.4644731621629)
    # This reach should not be extended, the remainder is too long
    reach_geom = nm.reaches.loc[
        nm.reaches['segnum'] == 3047926, 'geometry'].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 237.72893664132727)
    # Data set 1c
    assert abs(m.sfr.nstrm) == 154
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
    assert repr(nm) == dedent('''\
    <SwnMf6: flopy mfnwt 'h'
      154 in reaches (reachID): [1, 2, ..., 153, 154]
      94 in segment_data (nseg): [1, 2, ..., 93, 94]
        94 from segments (72% used): [3049802, 3049683, ..., 3046952, 3046736]
        no diversions
      1 stress period with perlen: [1.0] />''')
    if matplotlib:
        _ = nm.plot()
        plt.close()
    # Run model and read outputs
    m.model_ws = str(outdir)
    m.sfr.unit_number = [24]
    m.sfr.ipakcb = 50
    m.sfr.istcb2 = -51
    m.add_output_file(51, extension='sfo', binflag=True)
    m.write_input()
    success, buff = m.run_model()
    assert not success
    # Error/warning: upstream elevation is equal to downstream, slope is zero
    # TODO: improve processing to correct elevation errors
    nm.grid_cells.to_file(str(outdir.join('grid_cells.shp')))
    nm.reaches.to_file(str(outdir.join('reaches.shp')))
    gdf_to_shapefile(nm.segments, outdir.join('segments.shp'))


@pytest.mark.xfail
def test_coastal_process_flopy_idomain_modify(coastal_swn, coastal_flow_m,
                                             tmpdir_factory):
    outdir = tmpdir_factory.mktemp('coastal')
    # Load a MODFLOW model
    m = flopy.modflow.Modflow.load(
        'h.nam', version='mfnwt', exe_name=mfnwt_exe, model_ws=datadir,
        check=False)
    nm = swn.SwnMf6.from_swn_flopy(
        coastal_swn, m, idomain_action='modify', inflow=coastal_flow_m)
    assert len(nm.segments) == 304
    assert nm.segments['in_model'].sum() == 304
    # Check a remaining reach added that is outside model domain
    reach_geom = nm.reaches.loc[
        nm.reaches['segnum'] == 3048565, 'geometry'].iloc[0]
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
    assert abs(m.sfr.nstrm) == 478
    assert m.sfr.nss == 304
    # Data set 2
    # check_number_sum_hex(
    #     m.sfr.reach_data.node, 95964, '52c2df8cb982061c4c0a39bbf865926f')
    # check_number_sum_hex(
    #     m.sfr.reach_data.k, 0, '975d4ebfcacc6428ed80b7e319ed023a')
    # check_number_sum_hex(
    #     m.sfr.reach_data.i, 5307, '7ad41ac8568ac5e45bbb95a89a50da12')
    # check_number_sum_hex(
    #     m.sfr.reach_data.j, 5745, 'fc24e43745e3e09f5e84f63b07d32473')
    # check_number_sum_hex(
    #     m.sfr.reach_data.reachID, 196251, '46356d0cbb4563e5d882e5fd2639c3e8')
    # check_number_sum_hex(
    #     m.sfr.reach_data.iseg, 94974, '7bd775afa62ce9818fa6b1f715ecbb27')
    # check_number_sum_hex(
    #     m.sfr.reach_data.ireach, 1173, '8008ac0cb8bf371c37c3e51236e44fd4')
    # check_number_sum_hex(
    #     m.sfr.reach_data.rchlen, 255531, '72f89892d6e5e03c53106792e2695084')
    # check_number_sum_hex(
    #     m.sfr.reach_data.strtop, 24142, 'bc96d80acc1b59c4d50759301ae2392a')
    # check_number_sum_hex(
    #     m.sfr.reach_data.slope * 500, 6593, '0306817657dc6c85cb65c93f3fa15a')
    # check_number_sum_hex(
    #     m.sfr.reach_data.strthick, 626, 'a3aa65f110b20b57fc7f445aa743759f')
    # check_number_sum_hex(
    #     m.sfr.reach_data.strhc1, 626, 'a3aa65f110b20b57fc7f445aa743759f')
    # Data set 6
    assert len(m.sfr.segment_data) == 1
    sd = m.sfr.segment_data[0]
    del sd
    # check_number_sum_hex(
    #     sd.nseg, 46360, '22126069af5cfa16460d6b5ee2c9e25e')
    # check_number_sum_hex(
    #     sd.icalc, 304, '3665cd80c97966d0a740f0845e8b50e6')
    # check_number_sum_hex(
    #     sd.outseg, 69130, 'bfd96b95f0d9e7c4cfa67fac834dcf37')
    # check_number_sum_hex(
    #     sd.iupseg, 0, 'd6c6d43a06a3923eac7f03dcfe16f437')
    # check_number_sum_hex(
    #     sd.iprior, 0, 'd6c6d43a06a3923eac7f03dcfe16f437')
    # check_number_sum_hex(
    #     sd.flow, 0, 'd6c6d43a06a3923eac7f03dcfe16f437')
    # check_number_sum_hex(
    #     sd.runoff, 0, 'd6c6d43a06a3923eac7f03dcfe16f437')
    # check_number_sum_hex(
    #     sd.etsw, 0, 'd6c6d43a06a3923eac7f03dcfe16f437')
    # check_number_sum_hex(
    #     sd.pptsw, 0, 'd6c6d43a06a3923eac7f03dcfe16f437')
    # check_number_sum_hex(
    #     sd.roughch * 1000, 7296, 'fde9b5ef3863e60a5173b5949d495c09')
    # check_number_sum_hex(
    #     sd.hcond1, 304, '3665cd80c97966d0a740f0845e8b50e6')
    # check_number_sum_hex(
    #     sd.thickm1, 304, '3665cd80c97966d0a740f0845e8b50e6')
    # check_number_sum_hex(
    #     sd.width1, 3040, '65f2c05e33613b359676244036d86689')
    # check_number_sum_hex(
    #     sd.hcond2, 304, '3665cd80c97966d0a740f0845e8b50e6')
    # check_number_sum_hex(
    #     sd.thickm2, 304, '3665cd80c97966d0a740f0845e8b50e6')
    # check_number_sum_hex(
    #     sd.width2, 3040, '65f2c05e33613b359676244036d86689')
    # Check other packages
    check_number_sum_hex(
        m.dis.idomain.array, 572, 'd353560128577b37f730562d2f89c025')
    assert repr(nm) == dedent('''\
        <SwnMf6: flopy mfnwt 'h'
          478 in reaches (reachID): [1, 2, ..., 477, 478]
          304 in segment_data (nseg): [1, 2, ..., 303, 304]
            304 from segments: [3050413, 3050418, ..., 3046952, 3046736]
            no diversions
          1 stress period with perlen: [1.0] />''')
    if matplotlib:
        _ = nm.plot()
        plt.close()
    # Run model and read outputs
    m.model_ws = str(outdir)
    m.sfr.unit_number = [24]
    m.sfr.ipakcb = 50
    m.sfr.istcb2 = -51
    m.add_output_file(51, extension='sfo', binflag=True)
    m.write_input()
    success, buff = m.run_model()
    assert not success
    # Error/warning: upstream elevation is equal to downstream, slope is zero
    # TODO: improve processing to correct elevation errors
    nm.reaches.to_file(str(outdir.join('reaches.shp')))
    nm.grid_cells.to_file(str(outdir.join('grid_cells.shp')))
    gdf_to_shapefile(nm.segments, outdir.join('segments.shp'))


@pytest.mark.xfail
def test_process_flopy_lines_on_boundaries():
    m = flopy.modflow.Modflow(version='mf2005', exe_name=mf2005_exe)
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
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    nm = swn.SwnMf6.from_swn_flopy(n, m)
    if matplotlib:
        _ = nm.plot()
        plt.close()
    assert m.sfr.nss == 5
    # TODO: code needs to be improved for this type of case
    assert abs(m.sfr.nstrm) == 8


@pytest.mark.xfail
def test_process_flopy_diversion(tmpdir_factory):
    outdir = tmpdir_factory.mktemp('diversion')
    # Create a simple MODFLOW model
    m = flopy.modflow.Modflow(version='mf2005', exe_name=mf2005_exe)
    top = np.array([
        [16.0, 15.0],
        [15.0, 15.0],
        [14.0, 14.0],
    ])
    _ = flopy.modflow.ModflowDis(
        m, nlay=1, nrow=3, ncol=2, delr=20.0, delc=20.0, top=top, botm=10.0,
        xul=30.0, yul=130.0)
    _ = flopy.modflow.ModflowOc(
        m, stress_period_data={
            (0, 0): ['print head', 'save head', 'save budget']})
    _ = flopy.modflow.ModflowBas(m, strt=top)
    _ = flopy.modflow.ModflowDe4(m)
    _ = flopy.modflow.ModflowLpf(m, ipakcb=52, laytyp=0)
    gt = swn.modflow.geotransform_from_flopy(m)
    n = swn.SurfaceWaterNetwork.from_lines(interp_2d_to_3d(n3d_lines, top, gt))
    n.adjust_elevation_profile()
    diversions = geopandas.GeoDataFrame(geometry=[
        Point(58, 97), Point(62, 97), Point(61, 89), Point(59, 89)])
    n.set_diversions(diversions=diversions)

    # With zero specified flow for all terms
    nm = swn.SwnMf6.from_swn_flopy(n, m, hyd_cond1=0.0)
    m.sfr.ipakcb = 52
    m.sfr.istcb2 = 54
    m.add_output_file(54, extension='sfl', binflag=False)
    # Data set 1c
    assert abs(m.sfr.nstrm) == 11
    assert m.sfr.nss == 7
    # Data set 2
    np.testing.assert_array_almost_equal(
        m.sfr.reach_data.rchlen,
        [18.027756, 6.009252, 12.018504, 21.081852, 10.540926, 10.0, 10.0,
         1.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_almost_equal(
        m.sfr.reach_data.strtop,
        [15.742094, 15.39822, 15.140314, 14.989459, 14.973648, 14.726283,
         14.242094, 15.0, 15.0, 14.0, 14.0])
    np.testing.assert_array_almost_equal(
        m.sfr.reach_data.slope,
        [0.02861207, 0.02861207, 0.02861207, 0.001, 0.001, 0.04841886,
         0.04841886, 0.0, 0.0, 0.0, 0.0])
    sd = m.sfr.segment_data[0]
    np.testing.assert_array_equal(sd.nseg, [1, 2, 3, 4, 5, 6, 7])
    np.testing.assert_array_equal(sd.icalc,  [1, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(sd.outseg,  [3, 3, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(sd.iupseg,  [0, 0, 0, 1, 2, 3, 3])
    np.testing.assert_array_equal(sd.iprior,  [0, 0, 0, 0, 0, 0, 0])
    assert repr(nm) == dedent('''\
        <SwnMf6: flopy mf2005 'modflowtest'
          11 in reaches (reachID): [1, 2, ..., 10, 11]
          7 in segment_data (nseg): [1, 2, ..., 6, 7]
            3 from segments: [1, 2, 0]
            4 from diversions[0, 1, 2, 3]
          1 stress period with perlen: [1.0] />''')
    if matplotlib:
        _ = nm.plot()
        plt.close()

    # Run model and read outputs
    m.model_ws = str(outdir)
    m.write_input()
    success, buff = m.run_model()
    assert success
    cbc_fname = str(outdir.join(m.name + '.cbc'))
    sfl_fname = str(outdir.join(m.name + '.sfl'))
    sl = read_budget(cbc_fname, 'STREAM LEAKAGE', nm.reaches, 'sfrleakage')
    sfl = read_sfl(sfl_fname, nm.reaches)
    # Write some files
    nm.reaches.to_file(str(outdir.join('reaches.shp')))
    nm.grid_cells.to_file(str(outdir.join('grid_cells.shp')))
    gdf_to_shapefile(nm.segments, outdir.join('segments.shp'))
    # Check results
    assert (sl['q'] == 0.0).all()
    assert (sfl['Qin'] == 0.0).all()
    assert (sfl['Qaquifer'] == 0.0).all()
    assert (sfl['Qout'] == 0.0).all()
    assert (sfl['Qovr'] == 0.0).all()
    assert (sfl['Qprecip'] == 0.0).all()
    assert (sfl['Qet'] == 0.0).all()
    # Don't check stage, depth or gradient
    np.testing.assert_array_almost_equal(
        nm.reaches['width'],
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    assert (nm.reaches['Cond'] == 0.0).all()

    # Route some flow from headwater segments
    m.sfr.segment_data = nm.set_segment_data(
        flow={1: 2, 2: 3}, return_dict=True)
    m.sfr.write_file()
    success, buff = m.run_model()
    assert success
    sl = read_budget(cbc_fname, 'STREAM LEAKAGE', nm.reaches, 'sfrleakage')
    sfl = read_sfl(sfl_fname, nm.reaches)
    expected_flow = np.array(
        [2.0, 2.0, 2.0, 3.0, 3.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0])
    assert (sl['q'] == 0.0).all()
    np.testing.assert_almost_equal(sfl['Qin'], expected_flow)
    assert (sfl['Qaquifer'] == 0.0).all()
    np.testing.assert_almost_equal(sfl['Qout'], expected_flow)
    assert (sfl['Qovr'] == 0.0).all()
    assert (sfl['Qprecip'] == 0.0).all()
    assert (sfl['Qet'] == 0.0).all()

    # Same, but with abstraction
    m.sfr.segment_data = nm.set_segment_data(
        abstraction={0: 1.1}, flow={1: 2, 2: 3}, return_dict=True)
    m.sfr.write_file()
    success, buff = m.run_model()
    assert success
    sl = read_budget(cbc_fname, 'STREAM LEAKAGE', nm.reaches, 'sfrleakage')
    sfl = read_sfl(sfl_fname, nm.reaches)
    expected_flow = np.array(
        [2.0, 2.0, 2.0, 3.0, 3.0, 3.9, 3.9, 1.1, 0.0, 0.0, 0.0])
    assert (sl['q'] == 0.0).all()
    np.testing.assert_almost_equal(sfl['Qin'], expected_flow)
    assert (sfl['Qaquifer'] == 0.0).all()
    np.testing.assert_almost_equal(sfl['Qout'], expected_flow)
    assert (sfl['Qovr'] == 0.0).all()
    assert (sfl['Qprecip'] == 0.0).all()
    assert (sfl['Qet'] == 0.0).all()

    # More abstraction with dry streams
    m.sfr.segment_data = nm.set_segment_data(
        abstraction={0: 1.1, 1: 3.3}, flow={1: 2, 2: 3}, return_dict=True)
    m.sfr.write_file()
    success, buff = m.run_model()
    assert success
    sl = read_budget(cbc_fname, 'STREAM LEAKAGE', nm.reaches, 'sfrleakage')
    sfl = read_sfl(sfl_fname, nm.reaches)
    expected_flow = np.array(
        [2.0, 2.0, 2.0, 3.0, 3.0, 0.9, 0.9, 1.1, 3.0, 0.0, 0.0])
    assert (sl['q'] == 0.0).all()
    np.testing.assert_almost_equal(sfl['Qin'], expected_flow)
    assert (sfl['Qaquifer'] == 0.0).all()
    np.testing.assert_almost_equal(sfl['Qout'], expected_flow)
    assert (sfl['Qovr'] == 0.0).all()
    assert (sfl['Qprecip'] == 0.0).all()
    assert (sfl['Qet'] == 0.0).all()


@pytest.mark.xfail
def test_pickle(tmp_path):
    # Create a simple MODFLOW model
    m = flopy.modflow.Modflow(version='mf2005', exe_name=mf2005_exe)
    top = np.array([
        [16.0, 15.0],
        [15.0, 15.0],
        [14.0, 14.0],
    ])
    _ = flopy.modflow.ModflowDis(
        m, nlay=1, nrow=3, ncol=2, delr=20.0, delc=20.0, top=top, botm=10.0,
        xul=30.0, yul=130.0)
    _ = flopy.modflow.ModflowOc(
        m, stress_period_data={
            (0, 0): ['print head', 'save head', 'save budget']})
    _ = flopy.modflow.ModflowBas(m, strt=15.0, stoper=5.0)
    _ = flopy.modflow.ModflowSip(m)
    _ = flopy.modflow.ModflowLpf(m, ipakcb=52, laytyp=0, hk=1e-2)
    _ = flopy.modflow.ModflowRch(m, ipakcb=52, rech=1e-4)
    gt = swn.modflow.geotransform_from_flopy(m)
    n = swn.SurfaceWaterNetwork.from_lines(interp_2d_to_3d(n3d_lines, top, gt))
    n.adjust_elevation_profile()
    nm1 = swn.SwnMf6.from_swn_flopy(n, m)
    # use pickle dumps / loads methods
    data = pickle.dumps(nm1)
    nm2 = pickle.loads(data)
    assert nm1 != nm2
    assert nm2.model is None
    nm2.model = m
    assert nm1 == nm2
    # use to_pickle / from_pickle methods
    diversions = geopandas.GeoDataFrame(geometry=[
        Point(58, 97), Point(62, 97), Point(61, 89), Point(59, 89)])
    n.set_diversions(diversions=diversions)
    nm3 = swn.SwnMf6.from_swn_flopy(n, m, hyd_cond1=0.0)
    nm3.to_pickle(tmp_path / "nm4.pickle")
    nm4 = swn.SwnMf6.from_pickle(tmp_path / "nm4.pickle", m)
    assert nm3 == nm4


@requires_mf6
def test_mf6(tmpdir_factory, coastal_lines_gdf, coastal_flow_m):
    outdir = tmpdir_factory.mktemp('coastal')
    # Load a MODFLOW model
    sim = flopy.mf6.MFSimulation.load(
        "mfsim.nam", sim_ws=os.path.join(datadir, "mf6_coastal"),
        exe_name=mf6_exe)
    m = sim.get_model("mf6h")
    sim.set_sim_path("{}".format(outdir))
    # this model works without SFR -- Actually doesn't! (mfnwt lies)
    sim.write_simulation()
    success, buff = sim.run_simulation()
    assert success
    return
    # Create a SWN with adjusted elevation profiles
    n = swn.SurfaceWaterNetwork.from_lines(coastal_lines_gdf.geometry)
    n.adjust_elevation_profile()
    nm = swn.SwnMf6.from_swn_flopy(n, m)
    nm.set_sfr_data()
    # m.sfr.unit_number = [24]  # WARNING: unit 17 of package SFR already in use
    # m.sfr.ipakcb = 50
    # m.sfr.istcb2 = -51
    # m.add_output_file(51, extension='sfo', binflag=True)
    # and breaks with default SFR due to elevation errors
    sim.write_simulation()
    success, buff = sim.run_simulation()
    assert not success
    # Check dataframes
    assert len(nm.segments) == 304
    assert nm.segments['in_model'].sum() == 184
    # Check remaining reaches added that are inside model domain
    reach_geom = nm.reaches.loc[
        nm.reaches['segnum'] == 3047735, 'geometry'].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 980.5448069140768)
    # These should be split between two cells
    reach_geoms = nm.reaches.loc[
        nm.reaches['segnum'] == 3047750, 'geometry']
    assert len(reach_geoms) == 2
    np.testing.assert_almost_equal(reach_geoms.iloc[0].length, 204.90164560019)
    np.testing.assert_almost_equal(reach_geoms.iloc[1].length, 789.59872070638)
    # This reach should not be extended, the remainder is too far away
    reach_geom = nm.reaches.loc[
        nm.reaches['segnum'] == 3047762, 'geometry'].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 261.4644731621629)
    # This reach should not be extended, the remainder is too long
    reach_geom = nm.reaches.loc[
        nm.reaches['segnum'] == 3047926, 'geometry'].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 237.72893664132727)
    # Data set 1c
    assert abs(m.sfr.nstrm) == 296
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
        m.dis.idomain.array, 509, 'c4135a084b2593e0b69c148136a3ad6d')
    assert repr(nm) == dedent('''\
    <SwnMf6: flopy mfnwt 'h'
      296 in reaches (reachID): [1, 2, ..., 295, 296]
      184 in segment_data (nseg): [1, 2, ..., 183, 184]
        184 from segments (61% used): [3049818, 3049819, ..., 3046952, 3046736]
        no diversions
      1 stress period with perlen: [1.0] />''')
    if matplotlib:
        _ = nm.plot()
        plt.close()
    # Write output files
    nm.grid_cells.to_file(str(outdir.join('grid_cells.shp')))
    nm.reaches.to_file(str(outdir.join('reaches.shp')))
    gdf_to_shapefile(nm.segments, outdir.join('segments.shp'))