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
    mf6_exe = 'mf6'

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


def read_head(hed_fname, reaches=None):
    """Reads MODFLOW Head file

    If reaches is not None, it is modified inplace to add a 'head' column

    Returns numpy array
    """
    with flopy.utils.HeadFile(hed_fname) as b:
        data = b.get_data()
    if reaches is not None:
        reaches['head'] = data[reaches['k'], reaches['i'], reaches['j']]
    return data


def read_budget(bud_fname, text, reaches=None, colname=None):
    """Reads MODFLOW cell-by-cell file

    If reaches is not None, it is modified inplace to add data in 'colname'

    Returns numpy array
    """
    with flopy.utils.CellBudgetFile(bud_fname) as b:
        urn = [x.strip() for x in b.get_unique_record_names(True)]
        try:
            res = b.get_data(text=text)
        except Exception as e:
            raise Exception(f"cannot read text={text}; use one of {urn}\n{e}")
        if len(res) != 1:
            from warnings import warn
            warn('get_data(text={!r}) returned more than one array'
                 .format(text))
        data = res[0]
    if reaches is not None:
        if isinstance(data, np.recarray) and 'q' in data.dtype.names:
            reaches[colname] = data['q']
        else:
            reaches[colname] = data[reaches['k'], reaches['i'], reaches['j']]
    return data


def test_init_errors():
    with pytest.raises(ValueError, match="expected 'logger' to be Logger"):
        swn.SwnMf6(object())


def test_from_swn_flopy_errors(n3d):
    n = n3d
    n.segments = n.segments.copy()
    sim = flopy.mf6.MFSimulation(exe_name=mf6_exe)
    _ = flopy.mf6.ModflowTdis(sim, nper=4, time_units="days")
    m = flopy.mf6.ModflowGwf(sim)
    _ = flopy.mf6.ModflowGwfdis(
        m, nlay=1, nrow=3, ncol=2, delr=20.0, delc=20.0, idomain=1)

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

    # finally success!
    swn.SwnMf6.from_swn_flopy(n, m)


def test_process_flopy_n3d_defaults(n3d, tmpdir_factory):
    outdir = tmpdir_factory.mktemp('n3d')
    # Create a simple MODFLOW model
    sim = flopy.mf6.MFSimulation(exe_name=mf6_exe, sim_ws=str(outdir))
    _ = flopy.mf6.ModflowIms(
        sim, outer_maximum=600, inner_maximum=100,
        outer_dvclose=1e-6, rcloserecord=0.1, relaxation_factor=1.0)
    _ = flopy.mf6.ModflowTdis(sim, nper=1, time_units="days")
    m = flopy.mf6.ModflowGwf(sim, print_flows=True, save_flows=True)
    _ = flopy.mf6.ModflowGwfdis(
        m, nlay=1, nrow=3, ncol=2,
        delr=20.0, delc=20.0, length_units="meters",
        idomain=1, top=15.0, botm=10.0,
        xorigin=30.0, yorigin=70.0)
    _ = flopy.mf6.ModflowGwfoc(
        m, head_filerecord="model.hds", budget_filerecord="model.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")])
    _ = flopy.mf6.ModflowGwfic(m, strt=15.0)
    _ = flopy.mf6.ModflowGwfnpf(m, k=1e-2, save_flows=True)
    _ = flopy.mf6.ModflowGwfrcha(m, recharge=1e-4, save_flows=True)
    nm = swn.SwnMf6.from_swn_flopy(n3d, m)
    # check object reaches
    r = nm.reaches
    assert len(r) == 7
    assert r.index.name == "rno"
    assert list(r.index) == [1, 2, 3, 4, 5, 6, 7]
    # row and col are base-0
    assert list(r.row) == [0, 0, 1, 0, 1, 1, 2]
    assert list(r.col) == [0, 1, 1, 1, 1, 1, 1]
    assert list(r.segnum) == [1, 1, 1, 2, 2, 0, 0]
    assert list(r.to_rno) == [2, 3, 6, 5, 6, 7, 0]
    assert list(r.from_rnos) == [set(), {1}, {2}, set(), {4}, {3, 5}, {6}]
    assert list(r.to_div) == [0, 0, 0, 0, 0, 0, 0]
    with pytest.raises(KeyError, match='missing 5 reach dataset'):
        nm._packagedata_df("native")
    nm.set_reach_data_from_series("man", 0.024)
    nm.set_reach_data_from_series("rbth", 1.0)
    nm.set_reach_data_from_series("rhk", 1e-4)
    nm.reaches["rtp"] = nm.reaches["lsz_avg"]
    nm.set_reach_data_from_series("rwid", 10.0)
    # TODO rgrd ?
    np.testing.assert_array_almost_equal(
        nm.reaches["rtp"],
        [14.75, 14.416667, 14.166667, 14.666667, 14.166667, 13.5, 12.5])
    # check PACKAGEDATA
    rn = nm._packagedata_df("native")
    rf = nm._packagedata_df("flopy")
    assert list(rn.ncon) == [1, 2, 2, 1, 2, 3, 1]
    assert list(rn.ndv) == [0, 0, 0, 0, 0, 0, 0]
    # native is one based, k,i,j are str
    assert list(rn.index) == [1, 2, 3, 4, 5, 6, 7]
    assert list(rn.k) == ['1', '1', '1', '1', '1', '1', '1']
    assert list(rn.i) == ['1', '1', '2', '1', '2', '2', '3']
    assert list(rn.j) == ['1', '2', '2', '2', '2', '2', '2']
    # flopy has zero based, cellid with tuple
    assert list(rf.index) == [0, 1, 2, 3, 4, 5, 6]
    assert list(rf.cellid) == \
        [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 0, 1), (0, 1, 1), (0, 1, 1),
         (0, 2, 1)]
    np.testing.assert_array_almost_equal(
        rn.rlen,
        [18.027756, 6.009252, 12.018504, 21.081851, 10.540926, 10.0, 10.0])
    # Write native MF6 file and flopy datasets
    nm.write_packagedata(os.path.join(outdir, "packagedata.dat"))
    fpd = nm.flopy_packagedata
    # Don't check everything
    assert isinstance(fpd, list)
    assert len(fpd) == 7
    assert isinstance(fpd[0], list)
    assert len(fpd[0]) == 12
    assert fpd[-1][0] == 6
    assert fpd[-1][1] == (0, 2, 1)
    # check CONNECTIONDATA
    cn = nm._connectiondata_series("native")
    cf = nm._connectiondata_series("flopy")
    assert list(cn.index) == [1, 2, 3, 4, 5, 6, 7]
    assert list(cn) == [[-2], [1, -3], [2, -6], [-5], [4, -6], [3, 5, -7], [6]]
    assert list(cf.index) == [0, 1, 2, 3, 4, 5, 6]
    assert list(cf) == [[-1], [0, -2], [1, -5], [-4], [3, -5], [2, 4, -6], [5]]
    # Write native MF6 file and flopy datasets
    nm.write_connectiondata(os.path.join(outdir, "connectiondata.dat"))
    assert nm.flopy_connectiondata == \
        [[0, -1], [1, 0, -2], [2, 1, -5], [3, -4], [4, 3, -5], [5, 2, 4, -6],
         [6, 5]]
    # Use with flopy
    sfr_spd = {}
    _ = flopy.mf6.ModflowGwfsfr(
        m, save_flows=True,
        stage_filerecord="model.sfr.bin",
        budget_filerecord="model.sfr.bud",
        maximum_iterations=100,
        maximum_picard_iterations=10,
        nreaches=len(nm.reaches),
        packagedata={"filename": "packagedata.dat"},
        connectiondata={"filename": "connectiondata.dat"},
        perioddata=sfr_spd,
    )
    assert repr(nm) == dedent('''\
        <SwnMf6: flopy mf6 'model'
          7 in reaches (rno): [1, 2, ..., 6, 7]
          1 stress period with perlen: [1.0] days />''')
    if matplotlib:
        _ = nm.plot()
        plt.close()
    # Run model and read outputs
    sim.write_simulation()
    success, buff = sim.run_simulation()
    assert success
    hds_fname = str(outdir.join("model.hds"))
    cbc_fname = str(outdir.join("model.cbc"))
    sfo_fname = str(outdir.join("model.sfr.bud"))
    head = read_head(hds_fname)
    sl = read_budget(cbc_fname, 'SFR', nm.reaches, 'sfrleakage')
    sf = read_budget(sfo_fname, 'GWF', nm.reaches, 'sfr_Q')
    # Write some files
    gdf_to_shapefile(nm.reaches, outdir.join('reaches.shp'))
    gdf_to_shapefile(nm.segments, outdir.join('segments.shp'))
    nm.grid_cells.to_file(str(outdir.join('grid_cells.shp')))
    # Check results
    assert head.shape == (1, 3, 2)
    np.testing.assert_array_almost_equal(
        head,
        np.array([[
            [17.10986328, 16.77658646],
            [17.49119547, 16.81176342],
            [17.75195983, 17.2127244]]]))
    np.testing.assert_array_almost_equal(
        sl['q'],
        np.array([
            -0.04240276, -0.01410363, -0.03159486, -0.04431906, -0.02773725,
            -0.03292868, -0.04691372]))
    np.testing.assert_array_almost_equal(
        sf['q'],
        np.array([
            0.04240276, 0.01410362, 0.03159486, 0.04431904, 0.02773725,
            0.03292867, 0.04691372]))


def test_model_property(n3d):
    nm = swn.SwnMf6()
    with pytest.raises(
            ValueError, match="'model' must be a flopy.mf6.MFModel object"):
        nm.model = None

    sim = flopy.mf6.MFSimulation(exe_name=mf6_exe)
    m = flopy.mf6.MFModel(sim)

    with pytest.raises(ValueError, match='TDIS package required'):
        nm.model = m

    _ = flopy.mf6.ModflowTdis(
        sim, nper=1, time_units="days", start_date_time="2001-02-03")

    with pytest.raises(ValueError, match='DIS package required'):
        nm.model = m

    _ = flopy.mf6.ModflowGwfdis(
        m, nlay=1, nrow=3, ncol=2,
        delr=20.0, delc=20.0, length_units="meters",
        idomain=1, top=15.0, botm=10.0,
        xorigin=30.0, yorigin=70.0)

    # Success!
    nm.model = m

    pd.testing.assert_index_equal(
        nm.time_index,
        pd.DatetimeIndex(['2001-02-03'], dtype='datetime64[ns]'))

    # Swap model with same and with another
    m2 = flopy.mf6.ModflowGwf(sim, modelname="another")
    _ = flopy.mf6.ModflowGwfdis(m2)
    nm.model = m2


def test_set_reach_data_from_array(n3d):
    sim = flopy.mf6.MFSimulation(exe_name=mf6_exe)
    _ = flopy.mf6.ModflowTdis(sim, nper=1, time_units="days")
    m = flopy.mf6.ModflowGwf(sim, print_flows=True, save_flows=True)
    _ = flopy.mf6.ModflowGwfdis(
        m, nlay=1, nrow=3, ncol=2,
        delr=20.0, delc=20.0, length_units="meters",
        idomain=1, top=15.0, botm=10.0,
        xorigin=30.0, yorigin=70.0)
    nm = swn.SwnMf6.from_swn_flopy(n3d, m)
    ar = np.arange(6).reshape((3, 2)) + 8.0
    nm.set_reach_data_from_array("test", ar)
    assert list(nm.reaches["test"]) == \
        [8.0, 9.0, 11.0, 9.0, 11.0, 11.0, 13.0]


def test_set_reach_data_from_series():
    sim = flopy.mf6.MFSimulation(exe_name=mf6_exe)
    _ = flopy.mf6.ModflowTdis(sim, nper=1, time_units="days")
    m = flopy.mf6.ModflowGwf(sim, print_flows=True, save_flows=True)
    _ = flopy.mf6.ModflowGwfdis(
        m, nlay=1, nrow=3, ncol=2,
        delr=20.0, delc=20.0, length_units="meters",
        idomain=1, top=15.0, botm=10.0,
        xorigin=30.0, yorigin=70.0)
    n3d = swn.SurfaceWaterNetwork.from_lines(n3d_lines)
    n3d.segments["upstream_area"] = n3d.segments["upstream_length"] ** 2 * 100
    n3d.estimate_width()
    nm = swn.SwnMf6.from_swn_flopy(n3d, m)
    nm.set_reach_data_from_series("const_var", 9)
    assert list(nm.reaches["const_var"]) == [9.0] * 7
    nm.set_reach_data_from_series("rwd", n3d.segments.width)
    np.testing.assert_array_almost_equal(
        nm.reaches["rwd"],
        np.array([
            1.89765007, 2.07299816, 2.20450922, 1.91205787, 2.19715192,
            2.29218327, 2.29218327]))
    nm.set_reach_data_from_series("rwd", n3d.segments.width, pd.Series(5))
    np.testing.assert_array_almost_equal(
        nm.reaches["rwd"],
        np.array([
            1.89765007, 2.07299816, 2.20450922, 1.91205787, 2.19715192,
            2.96913745, 4.32304582]))
    k = pd.Series([1, 10, 100], dtype=float)
    nm.set_reach_data_from_series("rhk", k)
    np.testing.assert_array_almost_equal(
        nm.reaches["rhk"],
        np.array([7.75, 4.75, 2.5, 67., 17.5, 1., 1.]))
    nm.set_reach_data_from_series("rhk", k, log10=True)
    np.testing.assert_array_almost_equal(
        nm.reaches["rhk"],
        np.array([
            5.62341325, 2.61015722, 1.46779927, 21.5443469, 2.15443469,
            1., 1.]))
    nm.set_reach_data_from_series("rhk", k, pd.Series(1000), log10=True)
    np.testing.assert_array_almost_equal(
        nm.reaches["rhk"],
        np.array([
            5.62341325,2.61015722, 1.46779927, 21.5443469, 2.15443469,
            5.62341325,177.827941]))


def test_process_flopy_n2d_defaults(n2d, tmpdir_factory):
    # similar to 3D version, but getting information from model
    outdir = tmpdir_factory.mktemp('n2d')
    # Create a simple MODFLOW model
    top = np.array([
        [16.0, 15.0],
        [15.0, 15.0],
        [14.0, 14.0],
    ])
    sim = flopy.mf6.MFSimulation(exe_name=mf6_exe, sim_ws=str(outdir))
    _ = flopy.mf6.ModflowIms(
        sim, outer_maximum=600, inner_maximum=100,
        outer_dvclose=1e-6, rcloserecord=0.1, relaxation_factor=1.0)
    _ = flopy.mf6.ModflowTdis(sim, nper=1, time_units="days")
    m = flopy.mf6.ModflowGwf(sim, print_flows=True, save_flows=True)
    _ = flopy.mf6.ModflowGwfdis(
        m, nlay=1, nrow=3, ncol=2,
        delr=20.0, delc=20.0, length_units="meters",
        idomain=1, top=top, botm=10.0,
        xorigin=30.0, yorigin=70.0)
    _ = flopy.mf6.ModflowGwfoc(
        m, head_filerecord="model.hds", budget_filerecord="model.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")])
    _ = flopy.mf6.ModflowGwfic(m, strt=top)
    _ = flopy.mf6.ModflowGwfnpf(m, k=1.0, save_flows=True)
    _ = flopy.mf6.ModflowGwfrcha(m, recharge=0.01, save_flows=True)
    nm = swn.SwnMf6.from_swn_flopy(n2d, m)
    # check object reaches
    r = nm.reaches
    assert len(r) == 7
    # row and col are base-0
    assert list(r.row) == [0, 0, 1, 0, 1, 1, 2]
    assert list(r.col) == [0, 1, 1, 1, 1, 1, 1]
    assert list(r.segnum) == [1, 1, 1, 2, 2, 0, 0]
    assert list(r.to_rno) == [2, 3, 6, 5, 6, 7, 0]
    # See test_process_flopy_n3d_defaults for other checks
    nm.set_reach_data_from_series("man", 0.024)
    nm.set_reach_data_from_series("rbth", 2.0)
    nm.set_reach_data_from_series("rhk", 2.0)
    nm.set_reach_data_from_series("rwid", 10.0)
    nm.set_reach_data_from_array("rtp", m.dis.top.array - 1.0)
    np.testing.assert_array_almost_equal(
        nm.reaches["rtp"],
        [15., 14., 14., 14., 14., 14., 13.])
    # Use with flopy
    sfr_spd = {}
    _ = flopy.mf6.ModflowGwfsfr(
        m, save_flows=True,
        stage_filerecord="model.sfr.bin",
        budget_filerecord="model.sfr.bud",
        maximum_iterations=100,
        maximum_picard_iterations=10,
        nreaches=len(nm.reaches),
        packagedata=nm.flopy_packagedata,
        connectiondata=nm.flopy_connectiondata,
        perioddata=sfr_spd,
    )
    if mf6_exe:
        # Run model
        sim.write_simulation()
        success, buff = sim.run_simulation()
        assert success
        # check outputs?


def check_number_sum_hex(a, n, h):
    a = np.ceil(a).astype(np.int64)
    assert a.sum() == n
    ah = md5(a.tobytes()).hexdigest()
    assert ah.startswith(h), '{0} does not start with {1}'.format(ah, h)


@requires_mf6
def test_coastal_process_flopy(
        tmpdir_factory, coastal_lines_gdf, coastal_flow_m):
    outdir = tmpdir_factory.mktemp('coastal')
    # Load a MODFLOW model
    sim = flopy.mf6.MFSimulation.load(
        "mfsim.nam", sim_ws=os.path.join(datadir, "mf6_coastal"),
        exe_name=mf6_exe)
    sim.set_sim_path(str(outdir))
    m = sim.get_model("h")
    # this model runs without SFR
    sim.write_simulation()
    success, buff = sim.run_simulation()
    assert success
    # Create a SWN with adjusted elevation profiles
    n = swn.SurfaceWaterNetwork.from_lines(coastal_lines_gdf.geometry)
    n.adjust_elevation_profile()
    nm = swn.SwnMf6.from_swn_flopy(n, m)
    nm.set_reach_data_from_series("man", 0.024)
    nm.set_reach_data_from_series("rbth", 2.0)
    nm.set_reach_data_from_series("rhk", 2.0)
    nm.set_reach_data_from_series("rwid", 10.0)
    nm.reaches["rtp"] = nm.reaches["lsz_avg"]
    sfr_spd = {}
    _ = flopy.mf6.ModflowGwfsfr(
        m, save_flows=True,
        stage_filerecord="h.sfr.bin",
        budget_filerecord="h.sfr.bud",
        maximum_iterations=100,
        maximum_picard_iterations=10,
        nreaches=len(nm.reaches),
        packagedata=nm.flopy_packagedata,
        connectiondata=nm.flopy_connectiondata,
        perioddata=sfr_spd,
    )
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
    # Check other packages
    check_number_sum_hex(
        m.dis.idomain.array, 509, 'c4135a084b2593e0b69c148136a3ad6d')
    assert repr(nm) == dedent('''\
    <SwnMf6: flopy mf6 'h'
      296 in reaches (rno): [1, 2, ..., 295, 296]
      1 stress period with perlen: [1.0] days />''')
    if matplotlib:
        _ = nm.plot()
        plt.close()
    # Write output files
    gdf_to_shapefile(nm.reaches, outdir.join('reaches.shp'))
    gdf_to_shapefile(nm.segments, outdir.join('segments.shp'))
    nm.grid_cells.to_file(str(outdir.join('grid_cells.shp')))


@requires_mf6
def test_coastal_elevations(coastal_swn, coastal_flow_m, tmpdir_factory):
    outdir = tmpdir_factory.mktemp('coastal')
    # Load a MODFLOW model
    sim = flopy.mf6.MFSimulation.load(
        "mfsim.nam", sim_ws=os.path.join(datadir, "mf6_coastal"),
        exe_name=mf6_exe)
    sim.set_sim_path(str(outdir))
    m = sim.get_model("h")
    nm = swn.SwnMf6.from_swn_flopy(coastal_swn, m)
    nm.set_reach_data_from_series("man", 0.024)
    nm.set_reach_data_from_series("rbth", 2.0)
    nm.set_reach_data_from_series("rhk", 2.0)
    nm.set_reach_data_from_series("rwid", 10.0)
    nm.reaches["rtp"] = nm.reaches["lsz_avg"]
    # TODO: inflow=coastal_flow_m
    # TODO: complete elevation adjustments; see older MODFLOW methods
    return
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


@requires_mf6
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
    sim = flopy.mf6.MFSimulation.load(
        "mfsim.nam", sim_ws=os.path.join(datadir, "mf6_coastal"),
        exe_name=mf6_exe)
    sim.set_sim_path(str(outdir))
    m = sim.get_model("h")
    nm = swn.SwnMf6.from_swn_flopy(n, m)
    # TODO: inflow=coastal_flow_m
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
    assert len(nm.reaches) == 154
    assert repr(nm) == dedent('''\
    <SwnMf6: flopy mf6 'h'
      154 in reaches (rno): [1, 2, ..., 153, 154]
      1 stress period with perlen: [1.0] days />''')
    if matplotlib:
        _ = nm.plot()
        plt.close()
    # Run model and read outputs
    nm.set_reach_data_from_series("man", 0.024)
    nm.set_reach_data_from_series("rbth", 2.0)
    nm.set_reach_data_from_series("rhk", 2.0)
    nm.set_reach_data_from_series("rwid", 10.0)
    nm.reaches["rtp"] = nm.reaches["lsz_avg"]
    sfr_spd = {}
    _ = flopy.mf6.ModflowGwfsfr(
        m, save_flows=True,
        stage_filerecord="h.sfr.bin",
        budget_filerecord="h.sfr.bud",
        maximum_iterations=100,
        maximum_picard_iterations=10,
        nreaches=len(nm.reaches),
        packagedata=nm.flopy_packagedata,
        connectiondata=nm.flopy_connectiondata,
        perioddata=sfr_spd,
    )
    sim.write_simulation()
    success, buff = sim.run_simulation()
    assert not success
    # Error/warning: upstream elevation is equal to downstream, slope is zero
    # TODO: improve processing to correct elevation errors
    gdf_to_shapefile(nm.reaches, outdir.join('reaches.shp'))
    gdf_to_shapefile(nm.segments, outdir.join('segments.shp'))
    nm.grid_cells.to_file(str(outdir.join('grid_cells.shp')))


@requires_mf6
def test_coastal_process_flopy_idomain_modify(
        coastal_swn, coastal_flow_m, tmpdir_factory):
    outdir = tmpdir_factory.mktemp('coastal')
    # Load a MODFLOW model
    sim = flopy.mf6.MFSimulation.load(
        "mfsim.nam", sim_ws=os.path.join(datadir, "mf6_coastal"),
        exe_name=mf6_exe)
    sim.set_sim_path(str(outdir))
    m = sim.get_model("h")
    nm = swn.SwnMf6.from_swn_flopy(
        coastal_swn, m, idomain_action='modify')
    # TODO: inflow=coastal_flow_m
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
    assert len(nm.reaches) == 478
    # Check other packages
    check_number_sum_hex(
        m.dis.idomain.array, 572, 'd353560128577b37f730562d2f89c025')
    assert repr(nm) == dedent('''\
        <SwnMf6: flopy mf6 'h'
          478 in reaches (rno): [1, 2, ..., 477, 478]
          1 stress period with perlen: [1.0] days />''')
    if matplotlib:
        _ = nm.plot()
        plt.close()
    # Run model and read outputs
    nm.set_reach_data_from_series("man", 0.024)
    nm.set_reach_data_from_series("rbth", 2.0)
    nm.set_reach_data_from_series("rhk", 2.0)
    nm.set_reach_data_from_series("rwid", 10.0)
    nm.reaches["rtp"] = nm.reaches["lsz_avg"]
    sfr_spd = {}
    _ = flopy.mf6.ModflowGwfsfr(
        m, save_flows=True,
        stage_filerecord="h.sfr.bin",
        budget_filerecord="h.sfr.bud",
        maximum_iterations=100,
        maximum_picard_iterations=10,
        nreaches=len(nm.reaches),
        packagedata=nm.flopy_packagedata,
        connectiondata=nm.flopy_connectiondata,
        perioddata=sfr_spd,
    )
    sim.write_simulation()
    success, buff = sim.run_simulation()
    assert not success
    # Error/warning: upstream elevation is equal to downstream, slope is zero
    # TODO: improve processing to correct elevation errors
    gdf_to_shapefile(nm.reaches, outdir.join('reaches.shp'))
    gdf_to_shapefile(nm.segments, outdir.join('segments.shp'))
    nm.grid_cells.to_file(str(outdir.join('grid_cells.shp')))


@pytest.mark.xfail
def test_process_flopy_diversion(tmpdir_factory):
    outdir = tmpdir_factory.mktemp('diversion')
    # Create a simple MODFLOW model
    m = flopy.modflow.Modflow(version='mf6', exe_name=mf6_exe)
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
        <SwnMf6: flopy mf6 'modflowtest'
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
    gdf_to_shapefile(nm.reaches, outdir.join('reaches.shp'))
    gdf_to_shapefile(nm.segments, outdir.join('segments.shp'))
    nm.grid_cells.to_file(str(outdir.join('grid_cells.shp')))
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


def test_pickle(tmp_path):
    # Create a simple MODFLOW model
    sim = flopy.mf6.MFSimulation(exe_name=mf6_exe)
    top = np.array([
        [16.0, 15.0],
        [15.0, 15.0],
        [14.0, 14.0],
    ])
    _ = flopy.mf6.ModflowTdis(sim, nper=1, time_units="days")
    m = flopy.mf6.ModflowGwf(sim)
    _ = flopy.mf6.ModflowGwfdis(
        m, nlay=1, nrow=3, ncol=2,
        delr=20.0, delc=20.0, length_units="meters",
        idomain=1, top=top, botm=10.0,
        xorigin=30.0, yorigin=70.0)
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
    nm3 = swn.SwnMf6.from_swn_flopy(n, m)
    nm3.to_pickle(tmp_path / "nm4.pickle")
    nm4 = swn.SwnMf6.from_pickle(tmp_path / "nm4.pickle", m)
    assert nm3 == nm4
