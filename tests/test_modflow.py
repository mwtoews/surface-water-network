import pickle
from hashlib import md5
from shutil import which
from textwrap import dedent

import geopandas
import numpy as np
import pandas as pd
import pytest
from shapely import wkt
from shapely.geometry import Point

import swn
from swn.file import gdf_to_shapefile
from swn.spatial import force_2d, interp_2d_to_3d

if __name__ != "__main__":
    from .conftest import datadir, matplotlib, plt
else:
    from conftest import datadir, matplotlib, plt
try:
    import flopy
except ImportError:
    pytest.skip("skipping tests that require flopy", allow_module_level=True)


mfnwt_exe = which("mfnwt")
mf2005_exe = which("mf2005")
requires_mfnwt = pytest.mark.skipif(not mfnwt_exe, reason="requires mfnwt")
requires_mf2005 = pytest.mark.skipif(not mf2005_exe, reason="requires mf2005")

# same valid network used in test_basic
n3d_lines = geopandas.GeoSeries.from_wkt([
    "LINESTRING Z (60 100 14, 60  80 12)",
    "LINESTRING Z (40 130 15, 60 100 14)",
    "LINESTRING Z (70 130 15, 60 100 14)",
])


def get_basic_swn(has_z: bool = True, has_diversions: bool = False):
    if has_z:
        n = swn.SurfaceWaterNetwork.from_lines(n3d_lines)
    else:
        n = swn.SurfaceWaterNetwork.from_lines(force_2d(n3d_lines))
    if has_diversions:
        diversions = geopandas.GeoDataFrame(geometry=[
            Point(58, 100), Point(62, 100), Point(61, 89), Point(59, 89)])
        n.set_diversions(diversions=diversions)
    return n


def get_basic_modflow(
        outdir=".", with_top: bool = False, nper: int = 1,
        hk=1e-2, rech=1e-4):
    """Returns a basic Flopy MODFLOW model"""
    if with_top:
        top = np.array([
            [16.0, 15.0],
            [15.0, 15.0],
            [14.0, 14.0],
        ])
    else:
        top = 15.0
    m = flopy.modflow.Modflow(
        version="mf2005", exe_name=mf2005_exe, model_ws=outdir)
    flopy.modflow.ModflowDis(
        m, nlay=1, nrow=3, ncol=2, nper=nper,
        delr=20.0, delc=20.0, top=top, botm=10.0,
        xul=30.0, yul=130.0)
    _ = flopy.modflow.ModflowBas(m, strt=top, stoper=5.0)
    _ = flopy.modflow.ModflowSip(m)
    _ = flopy.modflow.ModflowLpf(m, ipakcb=52, laytyp=0, hk=hk)
    _ = flopy.modflow.ModflowRch(m, ipakcb=52, rech=rech)
    _ = flopy.modflow.ModflowOc(
        m, stress_period_data={
            (0, 0): ["print head", "save head", "save budget"]})
    return m


def read_head(hed_fname, reaches=None):
    """Reads MODFLOW Head file

    If reaches is not None, it is modified inplace to add a "head" column

    Returns numpy array
    """
    with flopy.utils.HeadFile(hed_fname) as b:
        data = b.get_data()
    if reaches is not None:
        reaches["head"] = data[reaches["k"], reaches["i"], reaches["j"]]
    return data


def read_budget(bud_fname, text, reaches=None, colname=None):
    """Reads MODFLOW cell-by-cell file

    If reaches is not None, it is modified inplace to add data in "colname"

    Returns numpy array
    """
    with flopy.utils.CellBudgetFile(bud_fname) as b:
        res = b.get_data(text=text)
        if len(res) != 1:
            from warnings import warn
            warn(f"get_data(text={text!r}) returned more than one array")
        data = res[0]
    if reaches is not None:
        if isinstance(data, np.recarray) and "q" in data.dtype.names:
            reaches[colname] = data["q"]
        else:
            reaches[colname] = data[reaches["k"], reaches["i"], reaches["j"]]
    return data


def read_sfl(sfl_fname, reaches=None):
    """Reads MODFLOW stream flow listing ASCII file

    If reaches is not None, it is modified inplace to add new columns

    Returns DataFrame of stream flow listing file
    """
    sfl = flopy.utils.SfrFile(sfl_fname).get_dataframe()
    # this index modification is only valid for steady models
    if sfl.index.name is None:
        sfl.index += 1
        sfl.index.name = "reachID"
    if "col16" in sfl.columns:
        sfl.rename(columns={"col16": "gradient"}, inplace=True)
    dont_copy = ["layer", "row", "column", "segment", "reach", "k", "i", "j"]
    if reaches is not None:
        if not (reaches.index == sfl.index).all():
            raise IndexError("reaches.index is different")
        for cn in sfl.columns:
            if cn == "kstpkper":  # split tuple into two columns
                reaches["kstp"] = sfl[cn].apply(lambda x: x[0])
                reaches["kper"] = sfl[cn].apply(lambda x: x[1])
            elif cn not in dont_copy:
                reaches[cn] = sfl[cn]
    return sfl


def test_init_errors():
    with pytest.raises(ValueError, match="expected 'logger' to be Logger"):
        swn.SwnModflow(object())


def test_from_swn_flopy_errors():
    n = get_basic_swn()
    m = flopy.modflow.Modflow(version="mf2005", exe_name=mf2005_exe)
    _ = flopy.modflow.ModflowDis(
        m, nlay=1, nrow=3, ncol=2, nper=4, delr=20.0, delc=20.0)

    with pytest.raises(
            ValueError,
            match="swn must be a SurfaceWaterNetwork object"):
        swn.SwnModflow.from_swn_flopy(object(), m)

    _ = flopy.modflow.ModflowBas(m)

    m.modelgrid.set_coord_info(epsg=2193)
    # n.segments.crs = {"init": "epsg:27200"}
    # with pytest.raises(
    #        ValueError,
    #        match="CRS for segments and modelgrid are different"):
    #    nm = swn.SwnModflow.from_swn_flopy(n, m)

    n.segments.crs = None
    with pytest.raises(
            ValueError,
            match="modelgrid extent does not cover segments extent"):
        swn.SwnModflow.from_swn_flopy(n, m)

    m.modelgrid.set_coord_info(xoff=30.0, yoff=70.0)

    with pytest.raises(ValueError, match="ibound_action must be one of"):
        swn.SwnModflow.from_swn_flopy(n, m, ibound_action="foo")


@pytest.mark.parametrize("has_diversions", [False, True], ids=["nodiv", "div"])
def test_new_segment_data(has_diversions):
    n = get_basic_swn(has_diversions=has_diversions)
    m = get_basic_modflow()
    nm = swn.SwnModflow.from_swn_flopy(n, m)
    assert nm.segment_data is None
    assert nm.segment_data_ts is None
    nm.new_segment_data()
    assert nm.segment_data_ts == {}
    assert (nm.segment_data.icalc == 0).all()
    if has_diversions:
        pd.testing.assert_index_equal(
            nm.segment_data.index,
            pd.Index([1, 2, 3, 4, 5, 6, 7], name="nseg"))
        assert list(nm.segment_data.segnum) == [1, 2, 0, -1, -1, -1, -1]
        assert list(nm.segment_data.divid) == [0, 0, 0, 0, 1, 2, 3]
        assert list(nm.segment_data.outseg) == [3, 3, 0, 0, 0, 0, 0]
        assert list(nm.segment_data.iupseg) == [0, 0, 0, 1, 2, 3, 3]
    else:
        pd.testing.assert_index_equal(
            nm.segment_data.index,
            pd.Index([1, 2, 3], name="nseg"))
        assert list(nm.segment_data.segnum) == [1, 2, 0]
        assert "divid" not in nm.segment_data.columns
        assert list(nm.segment_data.outseg) == [3, 3, 0]
        assert list(nm.segment_data.iupseg) == [0, 0, 0]


@requires_mf2005
def test_n3d_defaults(tmp_path):
    n = get_basic_swn()
    m = get_basic_modflow(tmp_path)
    nm = swn.SwnModflow.from_swn_flopy(n, m)
    nm.default_segment_data()
    nm.set_sfr_obj(ipakcb=52, istcb2=-53)
    assert m.sfr.ipakcb == 52
    assert m.sfr.istcb2 == -53
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
        [14.75, 14.416667, 14.16666667, 14.66666667, 14.16666667, 13.5, 12.5])
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
    np.testing.assert_array_almost_equal(sd.elevup, [14.75, 14.66666667, 13.5])
    np.testing.assert_array_almost_equal(sd.width1, [10.0, 10.0, 10.0])
    np.testing.assert_array_almost_equal(sd.hcond2, [1.0, 1.0, 1.0])
    np.testing.assert_array_almost_equal(sd.thickm2, [1.0, 1.0, 1.0])
    np.testing.assert_array_almost_equal(
        sd.elevdn, [14.16666667, 14.16666667, 12.5])
    np.testing.assert_array_almost_equal(sd.width2, [10.0, 10.0, 10.0])
    assert repr(nm) == dedent("""\
        <SwnModflow: flopy mf2005 'modflowtest'
          7 in reaches (reachID): [1, 2, ..., 6, 7]
          3 in segment_data (nseg): [1, 2, 3]
            3 from segments: [1, 2, 0]
          1 stress period with perlen: [1.0] />""")
    if matplotlib:
        _ = nm.plot()
        plt.close()
    # Run model and read outputs
    m.model_ws = str(tmp_path)
    m.write_input()
    success, buff = m.run_model()
    assert success
    heads = read_head(tmp_path / "modflowtest.hds")
    sl = read_budget(tmp_path / "modflowtest.cbc",
                     "STREAM LEAKAGE", nm.reaches, "sfrleakage")
    sf = read_budget(tmp_path / "modflowtest.sfr.bin",
                     "STREAMFLOW OUT", nm.reaches, "sfr_Q")
    # Write some files
    nm.grid_cells.to_file(tmp_path / "grid_cells.shp")
    gdf_to_shapefile(nm.reaches, tmp_path / "reaches.shp")
    gdf_to_shapefile(nm.segments, tmp_path / "segments.shp")
    # Check results
    assert heads.shape == (1, 3, 2)
    np.testing.assert_array_almost_equal(
        heads,
        np.array([[
                [14.604243, 14.409589],
                [14.172486, 13.251323],
                [13.861891, 12.751296]]], np.float32))
    np.testing.assert_array_almost_equal(
        sl["q"],
        np.array([-0.00859839, 0.00420513, 0.00439326, 0.0, 0.0,
                  -0.12359641, -0.12052996], np.float32))
    np.testing.assert_array_almost_equal(
        sf["q"],
        np.array([0.00859839, 0.00439326, 0.0, 0.0, 0.0,
                  0.12359641, 0.24412636], np.float32))


def test_model_property():
    nm = swn.SwnModflow()
    with pytest.raises(
            ValueError, match="model must be a flopy Modflow object"):
        nm.model = 0

    m = flopy.modflow.Modflow()
    with pytest.raises(ValueError, match="DIS package required"):
        nm.model = m

    _ = flopy.modflow.ModflowDis(
        m, nlay=1, nrow=3, ncol=2, delr=20.0, delc=20.0, top=15.0, botm=10.0,
        xul=30.0, yul=130.0, start_datetime="2001-02-03")

    with pytest.raises(ValueError, match="BAS6 package required"):
        nm.model = m

    _ = flopy.modflow.ModflowBas(m, strt=15.0, stoper=5.0)

    assert not hasattr(nm, "time_index")
    assert not hasattr(nm, "grid_cells")

    # Success!
    nm.model = m

    pd.testing.assert_index_equal(
        nm.time_index,
        pd.DatetimeIndex(["2001-02-03"], dtype="datetime64[ns]"))
    assert nm.grid_cells.shape == (6, 2)

    # Swap model with same and with another
    # same object
    nm.model = m

    dis_args = {
        "nper": 1, "nlay": 1, "nrow": 3, "ncol": 2, "delr": 20.0, "delc": 20.0,
        "xul": 30.0, "yul": 130.0, "start_datetime": "2001-03-02"}
    m = flopy.modflow.Modflow()
    _ = flopy.modflow.ModflowDis(m, **dis_args)
    _ = flopy.modflow.ModflowBas(m)
    # this is allowed
    nm.model = m

    dis_args_replace = {
        "nper": 2, "nrow": 4, "ncol": 3, "delr": 30.0, "delc": 40.0,
        "xul": 20.0, "yul": 120.0}
    for vn, vr in dis_args_replace.items():
        # print(f"{vn}: {vr}")
        dis_args_use = dis_args.copy()
        dis_args_use[vn] = vr
        m = flopy.modflow.Modflow()
        _ = flopy.modflow.ModflowDis(m, **dis_args_use)
        _ = flopy.modflow.ModflowBas(m)
        # this is not allowed
        with pytest.raises(AttributeError, match="properties are too differe"):
            nm.model = m


def test_time_index():
    n = get_basic_swn()
    m = get_basic_modflow(nper=12)
    m.dis.start_datetime = "1999-07-01"
    m.dis.perlen = [31, 31, 30, 31, 30, 31, 31, 29, 31, 30, 31, 30]
    nm = swn.SwnModflow.from_swn_flopy(n, m)
    assert nm.time_index.freqstr == "MS"  # "month start" or <MonthBegin>
    assert list(nm.time_index.day) == [1] * 12
    assert list(nm.time_index.month) == list((np.arange(12) + 6) % 12 + 1)
    assert list(nm.time_index.year) == [1999] * 6 + [2000] * 6


def test_segment_data_property():
    n = get_basic_swn(has_diversions=True)
    m = get_basic_modflow(nper=2)
    nm = swn.SwnModflow.from_swn_flopy(n, m)
    assert nm.segment_data is None
    assert nm.segment_data_ts is None
    nm.segment_data = None
    nm.segment_data_ts = None
    assert nm.segment_data is None
    assert nm.segment_data_ts is None
    sd = pd.DataFrame(index=pd.RangeIndex(7) + 1)
    nm.segment_data = sd
    assert nm.segment_data is sd
    assert nm.segment_data_ts is None
    sd_ts = {"data": pd.DataFrame(index=[0, 1])}
    nm.segment_data_ts = sd_ts
    assert nm.segment_data is sd
    assert nm.segment_data_ts is sd_ts
    nm.segment_data = None
    assert nm.segment_data is None
    assert nm.segment_data_ts is sd_ts
    nm.segment_data_ts = None
    assert nm.segment_data_ts is None
    # Errors
    with pytest.raises(ValueError, match="segment_data must be a DataFrame o"):
        nm.segment_data = []
    with pytest.raises(ValueError, match="segment_data nseg index is unexpec"):
        nm.segment_data = pd.DataFrame()
    with pytest.raises(ValueError, match="segment_data_ts must be a dict or "):
        nm.segment_data_ts = pd.DataFrame()
    with pytest.raises(ValueError, match="segment_data_ts key 'data' must be"):
        nm.segment_data_ts = {"data": []}


@pytest.mark.parametrize(
    "has_z", [False, True], ids=["n2d", "n3d"])
def test_default_segment_data(has_z):
    n = get_basic_swn(has_z=has_z)
    m = get_basic_modflow()
    nm = swn.SwnModflow.from_swn_flopy(n, m)
    assert nm.segment_data is None
    assert nm.segment_data_ts is None
    nm.default_segment_data()
    assert nm.segment_data is not None
    assert nm.segment_data_ts == {}
    sd = nm.segment_data
    assert sd.index.name == "nseg"
    np.testing.assert_array_equal(sd.index, [1, 2, 3])
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
    if has_z:
        expected_elevup = [14.75, 14.66666667, 13.5]
        expected_elevdn = [14.16666667, 14.16666667, 12.5]
    else:
        expected_elevup = [15.0, 15.0, 15.0]
        expected_elevdn = [15.0, 15.0, 15.0]
    np.testing.assert_array_almost_equal(sd.elevup, expected_elevup)
    np.testing.assert_array_almost_equal(sd.width1, [10.0, 10.0, 10.0])
    np.testing.assert_array_almost_equal(sd.hcond2, [1.0, 1.0, 1.0])
    np.testing.assert_array_almost_equal(sd.thickm2, [1.0, 1.0, 1.0])
    np.testing.assert_array_almost_equal(sd.elevdn, expected_elevdn)
    np.testing.assert_array_almost_equal(sd.width2, [10.0, 10.0, 10.0])

    # auto determine width
    n.catchments = geopandas.GeoSeries.from_wkt([
        "POLYGON ((35 100, 75 100, 75  80, 35  80, 35 100))",
        "POLYGON ((35 135, 60 135, 60 100, 35 100, 35 135))",
        "POLYGON ((60 135, 75 135, 75 100, 60 100, 60 135))",
    ])
    nm = swn.SwnModflow.from_swn_flopy(n, m)
    nm.default_segment_data()
    sd = nm.segment_data
    np.testing.assert_array_almost_equal(
        sd.width1, [1.4456947376374667, 1.439700753532406, 1.4615011177787172])
    np.testing.assert_array_almost_equal(
        sd.width2, [1.4456947376374667, 1.439700753532406, 1.4615011177787172])


def test_set_segment_data_from_scalar():
    n = get_basic_swn(has_diversions=True)
    m = get_basic_modflow(nper=2)
    nm = swn.SwnModflow.from_swn_flopy(n, m)
    nm.set_segment_data_from_scalar("icalc", 0)
    assert list(nm.segment_data.icalc) == [0, 0, 0, 0, 0, 0, 0]
    nm.set_segment_data_from_scalar("icalc", 1, "segments")
    assert list(nm.segment_data.icalc) == [1, 1, 1, 0, 0, 0, 0]
    nm.set_segment_data_from_scalar("icalc", 2, "diversions")
    assert list(nm.segment_data.icalc) == [1, 1, 1, 2, 2, 2, 2]

    # check that segment_data_ts item is dropped by this method
    nm.segment_data_ts["flow"] = 1.2
    assert nm.segment_data_ts["flow"] == 1.2
    nm.set_segment_data_from_scalar("flow", 2.3)
    assert "flow" not in nm.segment_data_ts

    # check errors
    with pytest.raises(KeyError, match="could not find 'nope'"):
        nm.set_segment_data_from_scalar("nope", 1.0)
    with pytest.raises(ValueError, match="data is not scalar"):
        nm.set_segment_data_from_scalar("flow", [1.0])
    with pytest.raises(ValueError, match="'which' should be one of"):
        nm.set_segment_data_from_scalar("flow", 1.0, "nope")


def test_set_segment_data_from_segments():
    n = get_basic_swn(has_diversions=True)
    n.segments["upstream_area"] = n.segments["upstream_length"] ** 2 * 100
    n.estimate_width()
    m = get_basic_modflow(nper=2)
    nm = swn.SwnModflow.from_swn_flopy(n, m)

    nm.new_segment_data()
    assert list(nm.segment_data.icalc) == [0, 0, 0, 0, 0, 0, 0]
    assert list(nm.segment_data.width1) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # scalar -- most other tests are in test_set_segment_data_from_scalar
    nm.set_segment_data_from_segments("icalc", 1)
    assert list(nm.segment_data.icalc) == [1, 1, 1, 0, 0, 0, 0]

    # dict
    nm.set_segment_data_from_segments("flow", {0: 0.1, 2: 1.2, 1: 2.3})
    assert list(nm.segment_data.flow) == [2.3, 1.2, 0.1, 0.0, 0.0, 0.0, 0.0]
    nm.set_segment_data_from_segments("flow", {})
    assert list(nm.segment_data.flow) == [2.3, 1.2, 0.1, 0.0, 0.0, 0.0, 0.0]
    nm.set_segment_data_from_segments("flow", {0: 4.0})
    assert list(nm.segment_data.flow) == [2.3, 1.2, 4.0, 0.0, 0.0, 0.0, 0.0]
    # errors
    with pytest.raises(KeyError, match="dict has a disjoint segnum set"):
        nm.set_segment_data_from_segments("flow", {"1": 0.1})
    with pytest.raises(KeyError, match="dict has a disjoint segnum set"):
        nm.set_segment_data_from_segments("flow", {3: 0.1})
    with pytest.raises(KeyError, match="dict has 1 key not found in segnum"):
        nm.set_segment_data_from_segments("flow", {3: 0.1, 2: 1.2})

    # series
    nm.set_segment_data_from_segments("flow", pd.Series([1.1, 2.2, 3.3]))
    assert list(nm.segment_data.flow) == [2.2, 3.3, 1.1, 0.0, 0.0, 0.0, 0.0]
    nm.set_segment_data_from_segments("flow", pd.Series([], dtype=float))
    assert list(nm.segment_data.flow) == [2.2, 3.3, 1.1, 0.0, 0.0, 0.0, 0.0]
    nm.set_segment_data_from_segments("flow", pd.Series([4.0], index=[1]))
    assert list(nm.segment_data.flow) == [4.0, 3.3, 1.1, 0.0, 0.0, 0.0, 0.0]
    nm.set_segment_data_from_segments("width1", n.segments.width)
    np.testing.assert_array_almost_equal(
        nm.segment_data.width1,
        [1.766139, 1.721995, 2.292183, 0.0, 0.0, 0.0, 0.0])

    # frame
    assert "runoff" not in nm.segment_data_ts
    nm.set_segment_data_from_segments(
        "runoff",
        pd.DataFrame(index=nm.time_index))
    assert "runoff" not in nm.segment_data_ts
    nm.set_segment_data_from_segments(
        "runoff",
        pd.DataFrame({0: [1.1, 2.2]}, index=nm.time_index))
    assert list(nm.segment_data.runoff) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pd.testing.assert_frame_equal(
        nm.segment_data_ts["runoff"],
        pd.DataFrame({3: [1.1, 2.2]}, index=nm.time_index))
    nm.set_segment_data_from_segments(
        "runoff",
        pd.DataFrame(index=nm.time_index))
    pd.testing.assert_frame_equal(
        nm.segment_data_ts["runoff"],
        pd.DataFrame({3: [1.1, 2.2]}, index=nm.time_index))

    # check that segment_data_ts item is not dropped by this method
    nm.set_segment_data_from_segments("runoff", {0: 0.1})
    assert "runoff" in nm.segment_data_ts

    # check errors
    with pytest.raises(TypeError, match="missing 1 required positional"):
        nm.set_segment_data_from_segments(1)
    with pytest.raises(ValueError, match="name must be str typ"):
        nm.set_segment_data_from_segments(1, 2)


def test_set_segment_data_from_diversions():
    n = get_basic_swn(has_diversions=True)
    m = get_basic_modflow(nper=2)
    nm = swn.SwnModflow.from_swn_flopy(n, m)

    nm.new_segment_data()
    assert list(nm.segment_data.icalc) == [0, 0, 0, 0, 0, 0, 0]
    assert list(nm.segment_data.abstraction) == [0.0] * 7

    # scalar -- most other tests are in test_set_segment_data_from_scalar
    nm.set_segment_data_from_diversions("abstraction", 2.2)
    assert list(nm.segment_data.abstraction) == \
        [0.0, 0.0, 0.0, 2.2, 2.2, 2.2, 2.2]

    # dict
    nm.set_segment_data_from_diversions(
        "abstraction", {0: 0.1, 2: 1.2, 1: 2.3})
    assert list(nm.segment_data.abstraction) == \
        [0.0, 0.0, 0.0, 0.1, 2.3, 1.2, 2.2]
    nm.set_segment_data_from_diversions("abstraction", {})
    assert list(nm.segment_data.abstraction) == \
        [0.0, 0.0, 0.0, 0.1, 2.3, 1.2, 2.2]
    nm.set_segment_data_from_diversions("abstraction", {0: 4.0})
    assert list(nm.segment_data.abstraction) == \
        [0.0, 0.0, 0.0, 4.0, 2.3, 1.2, 2.2]

    # errors
    with pytest.raises(KeyError, match="dict has a disjoint divid set"):
        nm.set_segment_data_from_diversions("abstraction", {"1": 0.1})
    with pytest.raises(KeyError, match="dict has a disjoint divid set"):
        nm.set_segment_data_from_diversions("abstraction", {5: 0.1})
    with pytest.raises(KeyError, match="dict has 1 key not found in divid"):
        nm.set_segment_data_from_diversions("abstraction", {5: 0.1, 2: 1.2})

    # series
    nm.set_segment_data_from_scalar("abstraction", 0.0)
    nm.set_segment_data_from_diversions(
        "abstraction", pd.Series([1.1, 2.2, 3.3, 4.4]))
    assert list(nm.segment_data.abstraction) == \
        [0.0, 0.0, 0.0, 1.1, 2.2, 3.3, 4.4]
    nm.set_segment_data_from_diversions(
        "abstraction", pd.Series([], dtype=float))
    assert list(nm.segment_data.abstraction) == \
        [0.0, 0.0, 0.0, 1.1, 2.2, 3.3, 4.4]
    nm.set_segment_data_from_diversions(
        "abstraction", pd.Series([4.0], index=[1]))
    assert list(nm.segment_data.abstraction) == \
        [0.0, 0.0, 0.0, 1.1, 4.0, 3.3, 4.4]
    nm.set_segment_data_from_diversions(
        "abstraction", n.diversions.dist_to_seg)
    np.testing.assert_array_almost_equal(
        nm.segment_data.abstraction,
        [0.0, 0.0, 0.0, 1.664101, 1.897367, 1.0, 1.0])

    # frame
    nm.set_segment_data_from_scalar("abstraction", 0.0)
    assert "abstraction" not in nm.segment_data_ts
    nm.set_segment_data_from_diversions(
        "abstraction",
        pd.DataFrame(index=nm.time_index))
    assert "abstraction" not in nm.segment_data_ts
    nm.set_segment_data_from_diversions(
        "abstraction",
        pd.DataFrame({0: [1.1, 2.2]}, index=nm.time_index))
    assert list(nm.segment_data.abstraction) == [0.0] * 7
    pd.testing.assert_frame_equal(
        nm.segment_data_ts["abstraction"],
        pd.DataFrame({4: [1.1, 2.2]}, index=nm.time_index))
    nm.set_segment_data_from_diversions(
        "abstraction",
        pd.DataFrame(index=nm.time_index))
    pd.testing.assert_frame_equal(
        nm.segment_data_ts["abstraction"],
        pd.DataFrame({4: [1.1, 2.2]}, index=nm.time_index))

    # check that segment_data_ts item is not dropped by this method
    nm.set_segment_data_from_diversions("abstraction", {0: 0.1})
    assert "abstraction" in nm.segment_data_ts

    # check errors
    with pytest.raises(TypeError, match="missing 1 required positional"):
        nm.set_segment_data_from_diversions(1)
    with pytest.raises(ValueError, match="name must be str typ"):
        nm.set_segment_data_from_diversions(1, 2)


@pytest.mark.parametrize(
    "nper,inflow,expected",
    [(1, {3: 9.6, 4: 9.7}, {1: 19.3}),
     (1, {}, {}),
     (2, {3: 9.6, 4: 9.7}, {1: 19.3}),
     (2, {}, {}),
     ])
def test_set_segment_data_inflow(nper, inflow, expected):
    n = get_basic_swn()
    n.segments.at[1, "from_segnums"] = {3, 4}
    m = get_basic_modflow(nper=nper)
    nm = swn.SwnModflow.from_swn_flopy(n, m)

    assert "inflow_segnums" not in nm.segments.columns
    assert nm.segment_data is None
    nm.set_segment_data_inflow(inflow)
    if inflow:
        assert list(nm.segments.inflow_segnums) == [set(), {3, 4}, set()]
        assert list(nm.segment_data.inflow_segnums) == [{3, 4}, set(), set()]
    else:
        assert "inflow_segnums" not in nm.segments.columns
        assert "inflow_segnums" not in nm.segment_data.columns

    expected = pd.Series(expected, dtype=float)
    if hasattr(nm.segment_data_ts, "inflow"):
        pd.testing.assert_frame_equal(
            nm.segment_data["inflow"],
            pd.DataFrame(expected, index=nm.time_index))
    else:
        expected_series = pd.Series(
            0.0, index=nm.segment_data.index, name="inflow")
        expected_series.update(expected)
        pd.testing.assert_series_equal(
            nm.segment_data["inflow"],
            expected_series)

    if matplotlib:
        _ = nm.plot()
        plt.close()


@requires_mf2005
def test_n3d_vars(tmp_path):
    # Repeat, but with min_slope enforced, and other options
    n = get_basic_swn()
    # manually add outside flow from extra segnums, referenced with inflow
    n.segments.at[1, "from_segnums"] = {3, 4}
    m = get_basic_modflow(tmp_path, hk=1.0, rech=0.01)
    nm = swn.SwnModflow.from_swn_flopy(n, m)
    nm.set_reach_slope(min_slope=0.03)
    nm.default_segment_data(hyd_cond1=2, thickness1=2.0)
    nm.set_segment_data_inflow({3: 9.6, 4: 9.7})
    nm.set_segment_data_from_segments("flow", {1: 18.4})
    nm.set_segment_data_from_segments("runoff", {1: 5})
    nm.set_segment_data_from_segments("pptsw", {2: 1.8})
    nm.set_segment_data_from_segments("etsw", {0: 0.01, 1: 0.02, 2: 0.03})

    nm.set_sfr_obj(ipakcb=52, istcb2=-53)
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
    # note that "inflow" gets added to nseg 1 flow
    np.testing.assert_array_equal(
        nm.segment_data.inflow_segnums, [{3, 4}, set(), set()])
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
    assert repr(nm) == dedent("""\
        <SwnModflow: flopy mf2005 'modflowtest'
          7 in reaches (reachID): [1, 2, ..., 6, 7]
          3 in segment_data (nseg): [1, 2, 3]
            3 from segments: [1, 2, 0]
          1 stress period with perlen: [1.0] />""")
    if matplotlib:
        _ = nm.plot()
        plt.close()
    # Run model and read outputs
    m.model_ws = str(tmp_path)
    m.write_input()
    success, buff = m.run_model()
    assert success
    heads = read_head(tmp_path / "modflowtest.hds")
    sl = read_budget(tmp_path / "modflowtest.cbc",
                     "STREAM LEAKAGE", nm.reaches, "sfrleakage")
    sf = read_budget(tmp_path / "modflowtest.sfr.bin",
                     "STREAMFLOW OUT", nm.reaches, "sfr_Q")
    # Write some files
    nm.grid_cells.to_file(tmp_path / "grid_cells.shp")
    gdf_to_shapefile(nm.reaches, tmp_path / "reaches.shp")
    gdf_to_shapefile(nm.segments, tmp_path / "segments.shp")
    # Check results
    assert heads.shape == (1, 3, 2)
    np.testing.assert_array_almost_equal(
        heads,
        np.array([[
                [14.620145, 14.489456],
                [14.494376, 13.962832],
                [14.100152, 12.905928]]], np.float32))
    np.testing.assert_array_almost_equal(
        sl["q"],
        np.array([-2.717792, -4.734348, 36.266556, 2.713955, 30.687397,
                  -70.960304, -15.255642], np.float32))
    np.testing.assert_array_almost_equal(
        sf["q"],
        np.array([39.31224, 43.67807, 6.67448, 370.4348, 526.3218,
                  602.95654, 617.21216], np.float32))


@requires_mf2005
def test_n2d_defaults(tmp_path):
    # similar to 3D version, but getting information from model
    n = get_basic_swn(has_z=False)
    m = get_basic_modflow(tmp_path, with_top=True, hk=1.0, rech=0.01)
    nm = swn.SwnModflow.from_swn_flopy(n, m)
    nm.default_segment_data()
    nm.set_sfr_obj(ipakcb=52, istcb2=-53)
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
    # See test_n3d_defaults for other checks
    assert repr(nm) == dedent("""\
        <SwnModflow: flopy mf2005 'modflowtest'
          7 in reaches (reachID): [1, 2, ..., 6, 7]
          3 in segment_data (nseg): [1, 2, 3]
            3 from segments: [1, 2, 0]
          1 stress period with perlen: [1.0] />""")
    if matplotlib:
        _ = nm.plot()
        plt.close()
    # Run model
    m.model_ws = str(tmp_path)
    m.write_input()
    success, buff = m.run_model()
    assert not success
    # Error/warning: upstream elevation is equal to downstream, slope is zero
    # TODO: improve processing to correct elevation errors
    nm.grid_cells.to_file(tmp_path / "grid_cells.shp")
    gdf_to_shapefile(nm.reaches, tmp_path / "reaches.shp")
    gdf_to_shapefile(nm.segments, tmp_path / "segments.shp")


@requires_mf2005
def test_n2d_min_slope(tmp_path):
    n = get_basic_swn(has_z=False)
    m = get_basic_modflow(tmp_path, with_top=True, hk=1.0, rech=0.01)
    nm = swn.SwnModflow.from_swn_flopy(n, m)
    nm.set_reach_slope(min_slope=0.03)
    nm.default_segment_data()

    nm.set_sfr_obj(ipakcb=52, istcb2=-53)
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
    # See test_n3d_defaults for other checks
    # Run model and read outputs
    m.model_ws = str(tmp_path)
    m.write_input()
    success, buff = m.run_model()
    assert not success
    # Error/warning: upstream elevation is equal to downstream, slope is zero
    # TODO: improve processing to correct elevation errors
    nm.grid_cells.to_file(tmp_path / "grid_cells.shp")
    gdf_to_shapefile(nm.reaches, tmp_path / "reaches.shp")
    gdf_to_shapefile(nm.segments, tmp_path / "segments.shp")


@requires_mf2005
def test_set_elevations(tmp_path):
    n = get_basic_swn(has_z=False)
    m = get_basic_modflow(tmp_path, with_top=True, hk=1.0, rech=0.01)
    nm = swn.SwnModflow.from_swn_flopy(n, m)
    nm.default_segment_data()
    # fix elevations
    _ = nm.add_model_topbot_to_reaches()
    if matplotlib:
        nm.plot_reaches_vs_model("all", plot_bottom=True)
        for seg in nm.reaches.segnum.unique():
            nm.plot_profile(
                seg, upstream=True, downstream=True
            )
    # Make sure segment ends are sensible relative to model elevations
    # Also ensure segments flow downstream
    # and downstream segments are below upstream segments
    _ = nm.fix_segment_elevs(min_incise=0.2, min_slope=1.e-4)
    # pass segment elevation back to update reach elevations
    _ = nm.reconcile_reach_strtop()
    if matplotlib:
        nm.plot_reaches_vs_model("all", plot_bottom=True)
        nm.plot_reaches_vs_model(1)
        for seg in nm.reaches.segnum.unique():
            nm.plot_profile(
                seg, upstream=True, downstream=True
            )
    _ = nm.add_model_topbot_to_reaches()
    nm.fix_reach_elevs()
    if matplotlib:
        nm.plot_reaches_vs_model("all", plot_bottom=True)
        for seg in nm.reaches.segnum.unique():
            nm.plot_profile(
                seg, upstream=True, downstream=True
            )
    nm.set_sfr_obj(ipakcb=52, istcb2=-53)
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
    # See test_n3d_defaults for other checks
    # Run model
    m.model_ws = str(tmp_path)
    m.write_input()
    success, buff = m.run_model()
    assert success
    heads = read_head(tmp_path / "modflowtest.hds")
    sl = read_budget(tmp_path / "modflowtest.cbc",
                     "STREAM LEAKAGE", nm.reaches, "sfrleakage")
    sf = read_budget(tmp_path / "modflowtest.sfr.bin",
                     "STREAMFLOW OUT", nm.reaches, "sfr_Q")
    # Write some files
    nm.grid_cells.to_file(tmp_path / "grid_cells.shp")
    gdf_to_shapefile(nm.reaches, tmp_path / "reaches.shp")
    gdf_to_shapefile(nm.segments, tmp_path / "segments.shp")
    # Check results
    assert heads.shape == (1, 3, 2)
    np.testing.assert_array_almost_equal(
        heads,
        np.array([[
                [15.4999275, 14.832507],
                [15.434015, 14.678202],
                [15.303412, 14.1582985]]], np.float32))
    np.testing.assert_array_almost_equal(
        sl["q"],
        np.array([0.0, 0.0, 0.0, -6.8689923, 6.8689923,
                  -13.108882, -10.891137], np.float32))
    np.testing.assert_array_almost_equal(
        sf["q"],
        np.array([0.0, 0.0, 0.0, 6.8689923, 0.0,
                  13.108882, 24.00002], np.float32))


def check_number_sum_hex(a, n, h):
    a = np.ceil(a).astype(np.int64)
    assert a.sum() == n
    ah = md5(a.tobytes()).hexdigest()
    assert ah.startswith(h), f"{ah} does not start with {h}"


@requires_mfnwt
def test_coastal(tmp_path, coastal_lines_gdf, coastal_flow_m):
    m = flopy.modflow.Modflow.load(
        "h.nam", version="mfnwt", exe_name=mfnwt_exe, model_ws=datadir,
        check=False)
    m.model_ws = str(tmp_path)
    # this model works without SFR
    m.write_input()
    success, buff = m.run_model()
    assert success
    # Create a SWN with adjusted elevation profiles
    n = swn.SurfaceWaterNetwork.from_lines(coastal_lines_gdf.geometry)
    n.adjust_elevation_profile()
    nm = swn.SwnModflow.from_swn_flopy(n, m)
    nm.default_segment_data()
    nm.set_segment_data_inflow(coastal_flow_m)

    nm.set_sfr_obj(ipakcb=50, istcb2=-51, unit_number=24)
    # WARNING: unit 17 of package SFR already in use
    # and breaks with default SFR due to elevation errors
    assert m.sfr.unit_number == [24]
    m.write_input()
    success, buff = m.run_model()
    assert not success
    # Check dataframes
    assert len(nm.segments) == 304
    assert nm.segments["in_model"].sum() == 184
    # Check remaining reaches added that are inside model domain
    reach_geom = nm.reaches.loc[
        nm.reaches["segnum"] == 3047735, "geometry"].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 980.5448069140768)
    # These should be split between two cells
    reach_geoms = nm.reaches.loc[
        nm.reaches["segnum"] == 3047750, "geometry"]
    assert len(reach_geoms) == 2
    np.testing.assert_almost_equal(reach_geoms.iloc[0].length, 204.90164560019)
    np.testing.assert_almost_equal(reach_geoms.iloc[1].length, 789.59872070638)
    # This reach should not be extended, the remainder is too far away
    reach_geom = nm.reaches.loc[
        nm.reaches["segnum"] == 3047762, "geometry"].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 261.4644731621629)
    # This reach should not be extended, the remainder is too long
    reach_geom = nm.reaches.loc[
        nm.reaches["segnum"] == 3047926, "geometry"].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 237.72893664132727)
    # Data set 1c
    assert abs(m.sfr.nstrm) == 297
    assert m.sfr.nss == 185
    # Data set 2
    # check_number_sum_hex(
    #    m.sfr.reach_data.node, 49998, "29eb6a019a744893ceb5a09294f62638")
    # check_number_sum_hex(
    #    m.sfr.reach_data.k, 0, "213581ea1c4e2fa86e66227673da9542")
    # check_number_sum_hex(
    #    m.sfr.reach_data.i, 2690, "be41f95d2eb64b956cc855304f6e5e1d")
    # check_number_sum_hex(
    #    m.sfr.reach_data.j, 4268, "4142617f1cbd589891e9c4033efb0243")
    # check_number_sum_hex(
    #    m.sfr.reach_data.reachID, 68635, "2a512563b164c76dfc605a91b10adae1")
    # check_number_sum_hex(
    #    m.sfr.reach_data.iseg, 34415, "48c4129d78c344d2e8086cd6971c16f7")
    # check_number_sum_hex(
    #    m.sfr.reach_data.ireach, 687, "233b71e88260cddb374e28ed197dfab0")
    # check_number_sum_hex(
    #    m.sfr.reach_data.rchlen, 159871, "776ed1ced406c7de9cfe502181dc8e97")
    # check_number_sum_hex(
    #    m.sfr.reach_data.strtop, 4266, "572a5ef53cd2c69f5d467f1056ee7579")
    # check_number_sum_hex(
    #   m.sfr.reach_data.slope * 999, 2945, "91c54e646fec7af346c0979167789316")
    # check_number_sum_hex(
    #    m.sfr.reach_data.strthick, 370, "09fd95bcbfe7c6309694157904acac68")
    # check_number_sum_hex(
    #    m.sfr.reach_data.strhc1, 370, "09fd95bcbfe7c6309694157904acac68")
    # Data set 6
    assert len(m.sfr.segment_data) == 1
    sd = m.sfr.segment_data[0]
    assert sd.flow.sum() > 0.0
    assert sd.pptsw.sum() == 0.0
    np.testing.assert_array_equal(sd.width1, sd.width2)
    # check_number_sum_hex(
    #    sd.nseg, 17020, "55968016ecfb4e995fb5591bce55fea0")
    # check_number_sum_hex(
    #    sd.icalc, 184, "1e57e4eaa6f22ada05f4d8cd719e7876")
    # check_number_sum_hex(
    #    sd.outseg, 24372, "46730406d031de87aad40c2d13921f6a")
    # check_number_sum_hex(
    #    sd.iupseg, 0, "f7e23bb7abe5b9603e8212ad467155bd")
    # check_number_sum_hex(
    #    sd.iprior, 0, "f7e23bb7abe5b9603e8212ad467155bd")
    # check_number_sum_hex(
    #    sd.flow, 4009, "49b48704587dc36d5d6f6295569eabd6")
    # check_number_sum_hex(
    #    sd.runoff, 0, "f7e23bb7abe5b9603e8212ad467155bd")
    # check_number_sum_hex(
    #    sd.etsw, 0, "f7e23bb7abe5b9603e8212ad467155bd")
    # check_number_sum_hex(
    #    sd.pptsw, 0, "f7e23bb7abe5b9603e8212ad467155bd")
    # check_number_sum_hex(
    #    sd.roughch * 1000, 4416, "a1a620fac8f5a6cbed3cc49aa2b90467")
    # check_number_sum_hex(
    #    sd.hcond1, 184, "1e57e4eaa6f22ada05f4d8cd719e7876")
    # check_number_sum_hex(
    #    sd.thickm1, 184, "1e57e4eaa6f22ada05f4d8cd719e7876")
    # check_number_sum_hex(
    #    sd.width1, 1840, "5749f425818b3b18e395b2a432520a4e")
    # check_number_sum_hex(
    #    sd.hcond2, 184, "1e57e4eaa6f22ada05f4d8cd719e7876")
    # check_number_sum_hex(
    #    sd.thickm2, 184, "1e57e4eaa6f22ada05f4d8cd719e7876")
    # check_number_sum_hex(
    #    sd.width2, 1840, "5749f425818b3b18e395b2a432520a4e")
    # Check other packages
    check_number_sum_hex(
        m.bas6.ibound.array, 509, "c4135a084b2593e0b69c148136a3ad6d")
    assert repr(nm) == dedent("""\
    <SwnModflow: flopy mfnwt 'h'
      297 in reaches (reachID): [1, 2, ..., 296, 297]
      185 in segment_data (nseg): [1, 2, ..., 184, 185]
        185 from segments (nzsegment) (61% used): [3049818, 3049819, ..., 3046952, 3046736]
      1 stress period with perlen: [1.0] />""")  # noqa
    if matplotlib:
        _ = nm.plot()
        plt.close()
    # Write output files
    nm.grid_cells.to_file(tmp_path / "grid_cells.shp")
    gdf_to_shapefile(nm.reaches, tmp_path / "reaches.shp")
    gdf_to_shapefile(nm.segments, tmp_path / "segments.shp")


@requires_mfnwt
def test_coastal_elevations(coastal_swn, coastal_flow_m, tmp_path):
    def _make_plot_sequence():
        if matplotlib:
            nm.plot_reaches_vs_model("all", plot_bottom=True)
            for seg in nm.segment_data.loc[
                nm.segment_data.index.isin([1, 18]), "segnum"
            ]:
                nm.plot_reaches_vs_model(seg, plot_bottom=True)
                plt.close()

    m = flopy.modflow.Modflow.load(
        "h.nam", version="mfnwt", exe_name=mfnwt_exe, model_ws=datadir,
        check=False)
    m.model_ws = str(tmp_path)
    nm = swn.SwnModflow.from_swn_flopy(coastal_swn, m)
    nm.default_segment_data()
    nm.set_segment_data_inflow(coastal_flow_m)
    _ = nm.add_model_topbot_to_reaches()
    _make_plot_sequence()
    # handy to set a max elevation that a stream can be
    _ = nm.get_seg_ijk()
    tops = nm.get_top_elevs_at_segs().top_up
    max_str_z = tops.describe()["75%"]
    _ = nm.fix_segment_elevs(min_incise=0.2, min_slope=1.e-4,
                             max_str_z=max_str_z)
    _ = nm.reconcile_reach_strtop()
    _make_plot_sequence()

    _ = nm.add_model_topbot_to_reaches()
    nm.fix_reach_elevs()
    _make_plot_sequence()

    nm.set_sfr_obj(ipakcb=50, istcb2=-51)
    m.sfr.unit_number = [24]
    m.add_output_file(51, extension="sfo", binflag=True)
    # Run model
    m.model_ws = str(tmp_path)
    m.write_input()
    success, buff = m.run_model()
    assert success


@requires_mfnwt
def test_coastal_reduced(coastal_lines_gdf, coastal_flow_m, tmp_path):
    n = swn.SurfaceWaterNetwork.from_lines(coastal_lines_gdf.geometry)
    assert len(n) == 304
    # Modify swn object
    n.remove(
        condition=n.segments["stream_order"] == 1,
        segnums=n.gather_segnums(upstream=3047927))
    assert len(n) == 130
    # Load a MODFLOW model
    m = flopy.modflow.Modflow.load(
        "h.nam", version="mfnwt", exe_name=mfnwt_exe, model_ws=datadir,
        check=False)
    nm = swn.SwnModflow.from_swn_flopy(n, m)
    nm.default_segment_data()
    nm.set_segment_data_inflow(coastal_flow_m)  # no inflow should result
    np.testing.assert_equal(nm.segment_data.inflow, np.zeros(94))
    # These should be split between two cells
    reach_geoms = nm.reaches.loc[
        nm.reaches["segnum"] == 3047750, "geometry"]
    assert len(reach_geoms) == 2
    np.testing.assert_almost_equal(reach_geoms.iloc[0].length, 204.90164560019)
    np.testing.assert_almost_equal(reach_geoms.iloc[1].length, 789.59872070638)
    # This reach should not be extended, the remainder is too far away
    reach_geom = nm.reaches.loc[
        nm.reaches["segnum"] == 3047762, "geometry"].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 261.4644731621629)
    # This reach should not be extended, the remainder is too long
    reach_geom = nm.reaches.loc[
        nm.reaches["segnum"] == 3047926, "geometry"].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 237.72893664132727)
    nm.set_sfr_obj()
    # Data set 1c
    assert abs(m.sfr.nstrm) == 154
    assert m.sfr.nss == 94
    # Data set 2
    # check_number_sum_hex(
    #    m.sfr.reach_data.node, 49998, "29eb6a019a744893ceb5a09294f62638")
    # check_number_sum_hex(
    #    m.sfr.reach_data.k, 0, "213581ea1c4e2fa86e66227673da9542")
    # check_number_sum_hex(
    #    m.sfr.reach_data.i, 2690, "be41f95d2eb64b956cc855304f6e5e1d")
    # check_number_sum_hex(
    #    m.sfr.reach_data.j, 4268, "4142617f1cbd589891e9c4033efb0243")
    # check_number_sum_hex(
    #    m.sfr.reach_data.reachID, 68635, "2a512563b164c76dfc605a91b10adae1")
    # check_number_sum_hex(
    #    m.sfr.reach_data.iseg, 34415, "48c4129d78c344d2e8086cd6971c16f7")
    # check_number_sum_hex(
    #    m.sfr.reach_data.ireach, 687, "233b71e88260cddb374e28ed197dfab0")
    # check_number_sum_hex(
    #    m.sfr.reach_data.rchlen, 159871, "776ed1ced406c7de9cfe502181dc8e97")
    # check_number_sum_hex(
    #    m.sfr.reach_data.strtop, 4266, "572a5ef53cd2c69f5d467f1056ee7579")
    # check_number_sum_hex(
    #   m.sfr.reach_data.slope * 999, 2945, "91c54e646fec7af346c0979167789316")
    # check_number_sum_hex(
    #    m.sfr.reach_data.strthick, 370, "09fd95bcbfe7c6309694157904acac68")
    # check_number_sum_hex(
    #    m.sfr.reach_data.strhc1, 370, "09fd95bcbfe7c6309694157904acac68")
    # Data set 6
    assert len(m.sfr.segment_data) == 1
    sd = m.sfr.segment_data[0]
    assert sd.flow.sum() > 0.0
    assert sd.pptsw.sum() == 0.0
    # check_number_sum_hex(
    #    sd.nseg, 17020, "55968016ecfb4e995fb5591bce55fea0")
    # check_number_sum_hex(
    #    sd.icalc, 184, "1e57e4eaa6f22ada05f4d8cd719e7876")
    # check_number_sum_hex(
    #    sd.outseg, 24372, "46730406d031de87aad40c2d13921f6a")
    # check_number_sum_hex(
    #    sd.iupseg, 0, "f7e23bb7abe5b9603e8212ad467155bd")
    # check_number_sum_hex(
    #    sd.iprior, 0, "f7e23bb7abe5b9603e8212ad467155bd")
    # check_number_sum_hex(
    #    sd.flow, 4009, "49b48704587dc36d5d6f6295569eabd6")
    # check_number_sum_hex(
    #    sd.runoff, 0, "f7e23bb7abe5b9603e8212ad467155bd")
    # check_number_sum_hex(
    #    sd.etsw, 0, "f7e23bb7abe5b9603e8212ad467155bd")
    # check_number_sum_hex(
    #    sd.pptsw, 0, "f7e23bb7abe5b9603e8212ad467155bd")
    # check_number_sum_hex(
    #    sd.roughch * 1000, 4416, "a1a620fac8f5a6cbed3cc49aa2b90467")
    # check_number_sum_hex(
    #    sd.hcond1, 184, "1e57e4eaa6f22ada05f4d8cd719e7876")
    # check_number_sum_hex(
    #    sd.thickm1, 184, "1e57e4eaa6f22ada05f4d8cd719e7876")
    # check_number_sum_hex(
    #    sd.width1, 1840, "5749f425818b3b18e395b2a432520a4e")
    # check_number_sum_hex(
    #    sd.hcond2, 184, "1e57e4eaa6f22ada05f4d8cd719e7876")
    # check_number_sum_hex(
    #    sd.thickm2, 184, "1e57e4eaa6f22ada05f4d8cd719e7876")
    # check_number_sum_hex(
    #    sd.width2, 1840, "5749f425818b3b18e395b2a432520a4e")
    assert repr(nm) == dedent("""\
    <SwnModflow: flopy mfnwt 'h'
      154 in reaches (reachID): [1, 2, ..., 153, 154]
      94 in segment_data (nseg): [1, 2, ..., 93, 94]
        94 from segments (nzsegment) (72% used): [3049802, 3049683, ..., 3046952, 3046736]
      1 stress period with perlen: [1.0] />""")  # noqa
    if matplotlib:
        _ = nm.plot()
        plt.close()
    # Run model and read outputs
    m.model_ws = str(tmp_path)
    m.sfr.unit_number = [24]
    m.sfr.ipakcb = 50
    m.sfr.istcb2 = -51
    m.add_output_file(51, extension="sfo", binflag=True)
    m.write_input()
    success, buff = m.run_model()
    assert not success
    # Error/warning: upstream elevation is equal to downstream, slope is zero
    # TODO: improve processing to correct elevation errors
    nm.grid_cells.to_file(tmp_path / "grid_cells.shp")
    gdf_to_shapefile(nm.reaches, tmp_path / "reaches.shp")
    gdf_to_shapefile(nm.segments, tmp_path / "segments.shp")


@requires_mfnwt
def test_coastal_ibound_modify(coastal_swn, coastal_flow_m, tmp_path):
    m = flopy.modflow.Modflow.load(
        "h.nam", version="mfnwt", exe_name=mfnwt_exe, model_ws=datadir,
        check=False)
    nm = swn.SwnModflow.from_swn_flopy(coastal_swn, m, ibound_action="modify")
    nm.default_segment_data()
    nm.set_segment_data_inflow(coastal_flow_m)
    assert len(nm.segments) == 304
    assert nm.segments["in_model"].sum() == 304
    # Check a remaining reach added that is outside model domain
    reach_geom = nm.reaches.loc[
        nm.reaches["segnum"] == 3048565, "geometry"].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 647.316024023105)
    expected_geom = wkt.loads(
        "LINESTRING Z (1819072.5 5869685.1 4, 1819000 5869684.9 5.7, "
        "1818997.5 5869684.9 5.8, 1818967.5 5869654.9 5, "
        "1818907.5 5869654.8 4, 1818877.6 5869624.7 5, 1818787.5 5869624.5 6, "
        "1818757.6 5869594.5 5.1, 1818697.6 5869594.4 5.7, "
        "1818667.6 5869564.3 6.2, 1818607.6 5869564.2 4.7, "
        "1818577.6 5869534.1 5.6, 1818487.6 5869534 6.2)")
    reach_geom.equals_exact(expected_geom, 0)
    nm.set_sfr_obj()
    # Data set 1c
    assert abs(m.sfr.nstrm) == 478
    assert m.sfr.nss == 304
    # Data set 2
    # check_number_sum_hex(
    #     m.sfr.reach_data.node, 95964, "52c2df8cb982061c4c0a39bbf865926f")
    # check_number_sum_hex(
    #     m.sfr.reach_data.k, 0, "975d4ebfcacc6428ed80b7e319ed023a")
    # check_number_sum_hex(
    #     m.sfr.reach_data.i, 5307, "7ad41ac8568ac5e45bbb95a89a50da12")
    # check_number_sum_hex(
    #     m.sfr.reach_data.j, 5745, "fc24e43745e3e09f5e84f63b07d32473")
    # check_number_sum_hex(
    #     m.sfr.reach_data.reachID, 196251, "46356d0cbb4563e5d882e5fd2639c3e8")
    # check_number_sum_hex(
    #     m.sfr.reach_data.iseg, 94974, "7bd775afa62ce9818fa6b1f715ecbb27")
    # check_number_sum_hex(
    #     m.sfr.reach_data.ireach, 1173, "8008ac0cb8bf371c37c3e51236e44fd4")
    # check_number_sum_hex(
    #     m.sfr.reach_data.rchlen, 255531, "72f89892d6e5e03c53106792e2695084")
    # check_number_sum_hex(
    #     m.sfr.reach_data.strtop, 24142, "bc96d80acc1b59c4d50759301ae2392a")
    # check_number_sum_hex(
    #     m.sfr.reach_data.slope * 500, 6593, "0306817657dc6c85cb65c93f3fa15a")
    # check_number_sum_hex(
    #     m.sfr.reach_data.strthick, 626, "a3aa65f110b20b57fc7f445aa743759f")
    # check_number_sum_hex(
    #     m.sfr.reach_data.strhc1, 626, "a3aa65f110b20b57fc7f445aa743759f")
    # Data set 6
    assert len(m.sfr.segment_data) == 1
    sd = m.sfr.segment_data[0]
    del sd
    # check_number_sum_hex(
    #     sd.nseg, 46360, "22126069af5cfa16460d6b5ee2c9e25e")
    # check_number_sum_hex(
    #     sd.icalc, 304, "3665cd80c97966d0a740f0845e8b50e6")
    # check_number_sum_hex(
    #     sd.outseg, 69130, "bfd96b95f0d9e7c4cfa67fac834dcf37")
    # check_number_sum_hex(
    #     sd.iupseg, 0, "d6c6d43a06a3923eac7f03dcfe16f437")
    # check_number_sum_hex(
    #     sd.iprior, 0, "d6c6d43a06a3923eac7f03dcfe16f437")
    # check_number_sum_hex(
    #     sd.flow, 0, "d6c6d43a06a3923eac7f03dcfe16f437")
    # check_number_sum_hex(
    #     sd.runoff, 0, "d6c6d43a06a3923eac7f03dcfe16f437")
    # check_number_sum_hex(
    #     sd.etsw, 0, "d6c6d43a06a3923eac7f03dcfe16f437")
    # check_number_sum_hex(
    #     sd.pptsw, 0, "d6c6d43a06a3923eac7f03dcfe16f437")
    # check_number_sum_hex(
    #     sd.roughch * 1000, 7296, "fde9b5ef3863e60a5173b5949d495c09")
    # check_number_sum_hex(
    #     sd.hcond1, 304, "3665cd80c97966d0a740f0845e8b50e6")
    # check_number_sum_hex(
    #     sd.thickm1, 304, "3665cd80c97966d0a740f0845e8b50e6")
    # check_number_sum_hex(
    #     sd.width1, 3040, "65f2c05e33613b359676244036d86689")
    # check_number_sum_hex(
    #     sd.hcond2, 304, "3665cd80c97966d0a740f0845e8b50e6")
    # check_number_sum_hex(
    #     sd.thickm2, 304, "3665cd80c97966d0a740f0845e8b50e6")
    # check_number_sum_hex(
    #     sd.width2, 3040, "65f2c05e33613b359676244036d86689")
    # Check other packages
    check_number_sum_hex(
        m.bas6.ibound.array, 572, "d353560128577b37f730562d2f89c025")
    assert repr(nm) == dedent("""\
        <SwnModflow: flopy mfnwt 'h'
          478 in reaches (reachID): [1, 2, ..., 477, 478]
          304 in segment_data (nseg): [1, 2, ..., 303, 304]
            304 from segments (nzsegment): [3050413, 3050418, ..., 3046952, 3046736]
          1 stress period with perlen: [1.0] />""")  # noqa
    if matplotlib:
        _ = nm.plot()
        plt.close()
    # Run model and read outputs
    m.model_ws = str(tmp_path)
    m.sfr.unit_number = [24]
    m.sfr.ipakcb = 50
    m.sfr.istcb2 = -51
    m.add_output_file(51, extension="sfo", binflag=True)
    m.write_input()
    success, buff = m.run_model()
    assert not success
    # Error/warning: upstream elevation is equal to downstream, slope is zero
    # TODO: improve processing to correct elevation errors
    nm.grid_cells.to_file(tmp_path / "grid_cells.shp")
    gdf_to_shapefile(nm.reaches, tmp_path / "reaches.shp")
    gdf_to_shapefile(nm.segments, tmp_path / "segments.shp")


@requires_mf2005
def test_include_downstream_reach_outside_model(tmp_path):
    m = get_basic_modflow(tmp_path, with_top=True)
    m.remove_package("rch")
    m.remove_package("sip")
    _ = flopy.modflow.ModflowDe4(m)
    m.bas6.ibound = np.array([[1, 1], [1, 1], [1, 0]])
    gt = swn.modflow.geotransform_from_flopy(m)
    lines = interp_2d_to_3d(
        geopandas.GeoSeries.from_wkt([
            "LINESTRING (60 89, 60 80)",
            "LINESTRING (40 130, 60 89)",
            "LINESTRING (70 130, 60 89)",
        ]), m.dis.top.array, gt)
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    nm = swn.SwnModflow.from_swn_flopy(n, m)
    nm.default_segment_data(hyd_cond1=0.0)
    nm.set_segment_data_from_segments("inflow", {1: 1.2, 2: 3.4})
    nm.set_sfr_obj()
    m.sfr.istcb2 = 54
    m.add_output_file(54, extension="sfl", binflag=False)
    # Data set 1c
    assert abs(m.sfr.nstrm) == 5
    assert m.sfr.nss == 3
    # Data set 2
    np.testing.assert_array_equal(m.sfr.reach_data.k, [0, 0, 0, 0, 0])
    np.testing.assert_array_equal(m.sfr.reach_data.i, [0, 1, 0, 1, 1])
    np.testing.assert_array_equal(m.sfr.reach_data.j, [0, 1, 1, 1, 1])
    np.testing.assert_array_equal(m.sfr.reach_data.iseg, [1, 1, 2, 2, 3])
    np.testing.assert_array_equal(m.sfr.reach_data.ireach, [1, 2, 1, 2, 1])
    np.testing.assert_array_almost_equal(
        m.sfr.reach_data.rchlen,
        [22.53083, 23.087149, 20.58629, 21.615604, 9.0])
    np.testing.assert_array_almost_equal(
        m.sfr.reach_data.strtop,
        [15.4927845, 14.849314, 14.865853, 14.548374, 14.225])
    np.testing.assert_array_almost_equal(
        m.sfr.reach_data.slope,
        [0.033977833, 0.033977833, 0.01303259, 0.01303259, 0.05])
    np.testing.assert_array_equal(m.sfr.reach_data.outreach, [2, 5, 4, 5, 0])
    sd = m.sfr.segment_data[0]
    np.testing.assert_array_equal(sd.nseg, [1, 2, 3])
    np.testing.assert_array_equal(sd.icalc, [1, 1, 0])
    np.testing.assert_array_equal(sd.outseg, [3, 3, 0])
    np.testing.assert_array_almost_equal(sd.flow, [1.2, 3.4, 0.0])
    np.testing.assert_array_almost_equal(sd.roughch, [0.024, 0.024, 0.024])
    if matplotlib:
        _ = nm.plot()
        plt.close()

    # Run model and read outputs
    m.write_input()
    success, buff = m.run_model()
    assert success
    sfl_fname = tmp_path / "modflowtest.sfl"
    sfl = read_sfl(sfl_fname, nm.reaches)
    # Write some files
    nm.grid_cells.to_file(tmp_path / "grid_cells.shp")
    gdf_to_shapefile(nm.reaches, tmp_path / "reaches.shp")
    gdf_to_shapefile(nm.segments, tmp_path / "segments.shp")
    # Check results
    np.testing.assert_array_equal(sfl["Qin"], [1.2, 1.2, 3.4, 3.4, 4.6])
    np.testing.assert_array_equal(sfl["Qout"], [1.2, 1.2, 3.4, 3.4, 4.6])


@requires_mf2005
def test_diversions(tmp_path):
    m = get_basic_modflow(tmp_path, with_top=True)
    m.remove_package("rch")
    m.remove_package("sip")
    _ = flopy.modflow.ModflowDe4(m)
    gt = swn.modflow.geotransform_from_flopy(m)
    lsz = interp_2d_to_3d(n3d_lines, m.dis.top.array, gt)
    n = swn.SurfaceWaterNetwork.from_lines(lsz)
    n.adjust_elevation_profile()
    diversions = geopandas.GeoDataFrame(geometry=[
        Point(58, 100), Point(62, 100), Point(61, 89), Point(59, 89)])
    n.set_diversions(diversions=diversions)

    nm = swn.SwnModflow.from_swn_flopy(n, m)
    nm.default_segment_data(hyd_cond1=0.0)
    nm.set_sfr_obj()
    m.sfr.ipakcb = 52
    m.sfr.istcb2 = 54
    m.add_output_file(54, extension="sfl", binflag=False)
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
         0.04841886, 0.001, 0.001, 0.001, 0.001])
    sd = m.sfr.segment_data[0]
    np.testing.assert_array_equal(sd.nseg, [1, 2, 3, 4, 5, 6, 7])
    np.testing.assert_array_equal(sd.icalc,  [1, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(sd.outseg,  [3, 3, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(sd.iupseg,  [0, 0, 0, 1, 2, 3, 3])
    np.testing.assert_array_equal(sd.iprior,  [0, 0, 0, 0, 0, 0, 0])
    assert repr(nm) == dedent("""\
        <SwnModflow: flopy mf2005 'modflowtest'
          11 in reaches (reachID): [1, 2, ..., 10, 11]
          7 in segment_data (nseg): [1, 2, ..., 6, 7]
            3 from segments: [1, 2, 0]
            4 from diversions: [0, 1, 2, 3]
          1 stress period with perlen: [1.0] />""")
    if matplotlib:
        _ = nm.plot()
        plt.close()

    # Run model and read outputs
    m.model_ws = str(tmp_path)
    m.write_input()
    success, buff = m.run_model()
    assert success
    cbc_fname = tmp_path / "modflowtest.cbc"
    sfl_fname = tmp_path / "modflowtest.sfl"
    sl = read_budget(cbc_fname, "STREAM LEAKAGE", nm.reaches, "sfrleakage")
    sfl = read_sfl(sfl_fname, nm.reaches)
    # Write some files
    nm.grid_cells.to_file(tmp_path / "grid_cells.shp")
    gdf_to_shapefile(nm.reaches[~nm.reaches.diversion],
                     tmp_path / "reaches.shp")
    gdf_to_shapefile(nm.segments, tmp_path / "segments.shp")
    # Check results
    assert (sl["q"] == 0.0).all()
    assert (sfl["Qin"] == 0.0).all()
    assert (sfl["Qaquifer"] == 0.0).all()
    assert (sfl["Qout"] == 0.0).all()
    assert (sfl["Qovr"] == 0.0).all()
    assert (sfl["Qprecip"] == 0.0).all()
    assert (sfl["Qet"] == 0.0).all()
    # Don't check stage, depth or gradient
    np.testing.assert_array_almost_equal(
        nm.reaches["width"],
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    assert (nm.reaches["Cond"] == 0.0).all()

    # Route some flow from headwater segments
    nm.set_segment_data_from_segments("flow", {1: 2, 2: 3})
    m.sfr.segment_data = nm.flopy_segment_data()
    np.testing.assert_array_almost_equal(
        m.sfr.segment_data[0]["flow"],
        [2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    m.sfr.write_file()
    success, buff = m.run_model()
    assert success
    sl = read_budget(cbc_fname, "STREAM LEAKAGE", nm.reaches, "sfrleakage")
    sfl = read_sfl(sfl_fname, nm.reaches)
    expected_flow = np.array(
        [2.0, 2.0, 2.0, 3.0, 3.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0])
    assert (sl["q"] == 0.0).all()
    np.testing.assert_almost_equal(sfl["Qin"], expected_flow)
    assert (sfl["Qaquifer"] == 0.0).all()
    np.testing.assert_almost_equal(sfl["Qout"], expected_flow)
    assert (sfl["Qovr"] == 0.0).all()
    assert (sfl["Qprecip"] == 0.0).all()
    assert (sfl["Qet"] == 0.0).all()

    # Same, but with abstraction
    nm.set_segment_data_from_diversions("abstraction", {0: 1.1})
    m.sfr.segment_data = nm.flopy_segment_data()
    np.testing.assert_array_almost_equal(
        m.sfr.segment_data[0]["flow"],
        [2.0, 3.0, 0.0, 1.1, 0.0, 0.0, 0.0])
    m.sfr.write_file()
    success, buff = m.run_model()
    assert success
    sl = read_budget(cbc_fname, "STREAM LEAKAGE", nm.reaches, "sfrleakage")
    sfl = read_sfl(sfl_fname, nm.reaches)
    expected_flow = np.array(
        [2.0, 2.0, 2.0, 3.0, 3.0, 3.9, 3.9, 1.1, 0.0, 0.0, 0.0])
    assert (sl["q"] == 0.0).all()
    np.testing.assert_almost_equal(sfl["Qin"], expected_flow)
    assert (sfl["Qaquifer"] == 0.0).all()
    np.testing.assert_almost_equal(sfl["Qout"], expected_flow)
    assert (sfl["Qovr"] == 0.0).all()
    assert (sfl["Qprecip"] == 0.0).all()
    assert (sfl["Qet"] == 0.0).all()

    # More abstraction with dry streams
    nm.set_segment_data_from_diversions("abstraction", {1: 3.3})
    m.sfr.segment_data = nm.flopy_segment_data()
    np.testing.assert_array_almost_equal(
        m.sfr.segment_data[0]["flow"],
        [2.0, 3.0, 0.0, 1.1, 3.3, 0.0, 0.0])
    m.sfr.write_file()
    success, buff = m.run_model()
    assert success
    sl = read_budget(cbc_fname, "STREAM LEAKAGE", nm.reaches, "sfrleakage")
    sfl = read_sfl(sfl_fname, nm.reaches)
    expected_flow = np.array(
        [2.0, 2.0, 2.0, 3.0, 3.0, 0.9, 0.9, 1.1, 3.0, 0.0, 0.0])
    assert (sl["q"] == 0.0).all()
    np.testing.assert_almost_equal(sfl["Qin"], expected_flow)
    assert (sfl["Qaquifer"] == 0.0).all()
    np.testing.assert_almost_equal(sfl["Qout"], expected_flow)
    assert (sfl["Qovr"] == 0.0).all()
    assert (sfl["Qprecip"] == 0.0).all()
    assert (sfl["Qet"] == 0.0).all()


def test_pickle(tmp_path):
    m = get_basic_modflow(tmp_path, with_top=True)
    m.remove_package("rch")
    m.remove_package("sip")
    _ = flopy.modflow.ModflowDe4(m)
    gt = swn.modflow.geotransform_from_flopy(m)
    lsz = interp_2d_to_3d(n3d_lines, m.dis.top.array, gt)
    n = swn.SurfaceWaterNetwork.from_lines(lsz)
    n.adjust_elevation_profile()
    nm1 = swn.SwnModflow.from_swn_flopy(n, m)
    # use pickle dumps / loads methods
    data = pickle.dumps(nm1)
    nm2 = pickle.loads(data)
    assert nm1 != nm2
    assert nm2.swn is None
    assert nm2.model is None
    nm2.swn = n
    nm2.model = m
    assert nm1 == nm2
    # use to_pickle / from_pickle methods
    diversions = geopandas.GeoDataFrame(geometry=[
        Point(58, 100), Point(62, 100), Point(61, 89), Point(59, 89)])
    n.set_diversions(diversions=diversions)
    nm3 = swn.SwnModflow.from_swn_flopy(n, m)
    nm3.default_segment_data(hyd_cond1=0.0)
    nm3.to_pickle(tmp_path / "nm4.pickle")
    nm4 = swn.SwnModflow.from_pickle(tmp_path / "nm4.pickle", n, m)
    assert nm3 == nm4


def test_route_reaches():
    n1 = get_basic_swn(has_diversions=True)
    lines2 = list(n1.segments.geometry)
    lines2.append(wkt.loads("LINESTRING (40 90, 50 80)"))
    n = swn.SurfaceWaterNetwork.from_lines(geopandas.GeoSeries(lines2))
    m = get_basic_modflow(with_top=False)
    nm = swn.SwnModflow.from_swn_flopy(n, m)
    assert nm.route_reaches(8, 8) == [8]
    assert nm.route_reaches(7, 8) == [7, 8]
    assert nm.route_reaches(8, 7) == [8, 7]
    assert nm.route_reaches(2, 7) == [2, 3, 7]
    assert nm.route_reaches(7, 2) == [7, 3, 2]
    assert nm.route_reaches(1, 7) == [1, 2, 3, 7]
    assert nm.route_reaches(7, 1) == [7, 3, 2, 1]
    assert nm.route_reaches(2, 4, allow_indirect=True) == [2, 3, 5, 4]
    # errors
    with pytest.raises(IndexError, match="invalid start reachID 0"):
        nm.route_reaches(0, 1)
    with pytest.raises(IndexError, match="invalid end reachID 0"):
        nm.route_reaches(1, 0)
    with pytest.raises(ConnectionError, match="1 does not connect to 4"):
        nm.route_reaches(1, 4)
    with pytest.raises(ConnectionError, match="reach networks are disjoint"):
        nm.route_reaches(1, 6)
    with pytest.raises(ConnectionError, match="reach networks are disjoint"):
        nm.route_reaches(1, 6, allow_indirect=True)
    # TODO: diversions?


def test_get_flopy_modflow_package():
    from swn.modflow._swnmodflow import get_flopy_modflow_package

    assert get_flopy_modflow_package("drn") == flopy.modflow.ModflowDrn
    with pytest.raises(AttributeError):
        get_flopy_modflow_package("no")


def test_package_period_frame():
    n = get_basic_swn()
    m = get_basic_modflow()
    nm = swn.SwnModflow.from_swn_flopy(n, m)

    with pytest.raises(
            KeyError, match="2 reach series needed for ModflowDrn: elev, con"):
        nm.package_period_frame("drn", "native")

    nm.set_reach_data_from_array("elev", m.dis.top.array - 1.0)
    nm.reaches["rlen"] = (nm.reaches.length).round(2)
    nm.reaches["cond"] = (nm.reaches.length * 10.0).round(2)

    # default, without auxiliary
    exp_native = pd.DataFrame(
        {
            "k": [1] * 7,
            "i": [1, 1, 2, 1, 2, 2, 3],
            "j": [1, 2, 2, 2, 2, 2, 2],
            "elev": np.array([14.0] * 7, np.float32),
            "cond": [180.28, 60.09, 120.19, 210.82, 105.41, 100.0, 100.0],
        },
        index=pd.MultiIndex.from_tuples(
            [(1, rid + 1) for rid in range(7)], names=["per", "reachID"])
    )
    exp_flopy = exp_native.copy()
    exp_flopy[list("ijk")] -= 1
    exp_flopy.index = pd.MultiIndex.from_tuples(
        [(0, rid + 1) for rid in range(7)], names=["per", "reachID"])
    pd.testing.assert_frame_equal(
        nm.package_period_frame("drn", "native"),
        exp_native)
    pd.testing.assert_frame_equal(
        nm.package_period_frame("drn", "flopy"),
        exp_flopy)

    # with auxiliary
    rlen = [18.03, 6.01, 12.02, 21.08, 10.54, 10.0, 10.0]
    exp_native["rlen"] = rlen
    exp_flopy["rlen"] = rlen
    pd.testing.assert_frame_equal(
        nm.package_period_frame("drn", "native", auxiliary="rlen"),
        exp_native)
    pd.testing.assert_frame_equal(
        nm.package_period_frame("drn", "flopy", auxiliary="rlen"),
        exp_flopy)


def test_write_package_period(tmp_path):
    n = get_basic_swn()
    m = get_basic_modflow(tmp_path)
    nm = swn.SwnModflow.from_swn_flopy(n, m)

    nm.set_reach_data_from_array("elev", m.dis.top.array - 1.0)
    nm.reaches["rlen"] = (nm.reaches.length).round(2)
    nm.reaches["cond"] = (nm.reaches.length * 10.0).round(2)

    fname_tpl = tmp_path / "drn_period_{:02d}.dat"
    fname = tmp_path / "drn_period_01.dat"
    assert not fname.exists()

    # default, without auxiliary
    nm.write_package_period("drn", fname_tpl)
    file_l = fname.read_text().splitlines()
    assert file_l[0].split() == ["1", "1", "1", "14.0", "180.28"]
    assert len(file_l) == 7

    # with auxiliary
    nm.write_package_period("drn", fname_tpl, auxiliary="rlen")
    assert fname.read_text().splitlines()[0].split() == \
        ["1", "1", "1", "14.0", "180.28", "18.03"]

    # add dummy package, to be overwritten
    _ = flopy.modflow.ModflowDrn(
        m, stress_period_data={
            0: flopy.modflow.ModflowDrn.get_empty(len(nm.reaches))})
    m.write_input()
    drn_fname = (tmp_path / "modflowtest.drn")
    assert drn_fname.exists()

    # Run model and read outputs
    if mf2005_exe:
        # more precise cond without rounding
        nm.reaches["cond"] = nm.reaches.length * 10.0
        nm.write_package_period("drn", fname_tpl)
        drn_fname.write_text(dedent(f"""\
            7 52
            7 0
            OPEN/CLOSE {fname.name}
        """))
        success, buff = m.run_model()
        assert success
        dl = read_budget(tmp_path / "modflowtest.cbc", "DRAINS")
        expected_q = np.array(
            [-0.06402918, -0.00885882, -0.02372583, -0.03096424, -0.02075164,
             -0.01971569, -0.07200407], np.float32)
        np.testing.assert_almost_equal(dl["q"], expected_q)
        assert "RLEN" not in dl.dtype.names
        assert "RLEN".ljust(16) not in dl.dtype.names

        # with auxiliary
        nm.write_package_period("drn", fname_tpl, auxiliary="rlen")
        drn_fname.write_text(dedent(f"""\
            7 52 AUX RLEN
            7 0
            OPEN/CLOSE {fname.name}
        """))
        success, buff = m.run_model()
        assert success
        dl = read_budget(tmp_path / "modflowtest.cbc", "DRAINS")
        np.testing.assert_almost_equal(dl["q"], expected_q)
        rlen_name = "RLEN"
        if rlen_name not in dl.dtype.names:
            rlen_name = "RLEN".ljust(16)
        assert rlen_name in dl.dtype.names
        np.testing.assert_almost_equal(
            dl[rlen_name], nm.reaches["rlen"].astype(np.float32))


def test_flopy_package_period(tmp_path):
    n = get_basic_swn()
    m = get_basic_modflow(tmp_path)
    nm = swn.SwnModflow.from_swn_flopy(n, m)

    nm.set_reach_data_from_array("elev", m.dis.top.array - 1.0)
    nm.reaches["rlen"] = (nm.reaches.length).round(2)
    nm.reaches["cond"] = (nm.reaches.length * 10.0).round(2)

    # default, without auxiliary
    exp_flopy = pd.DataFrame(
        {
            "k": [0] * 7,
            "i": [0, 0, 1, 0, 1, 1, 2],
            "j": [0, 1, 1, 1, 1, 1, 1],
            "elev": np.array([14.0] * 7, np.float32),
            "cond": np.array(
                [180.28, 60.09, 120.19, 210.82, 105.41, 100.0, 100.0],
                np.float32),
        },
        index=pd.MultiIndex.from_tuples(
            [(1, rid + 1) for rid in range(7)], names=["per", "reachID"])
    )
    ret = nm.flopy_package_period("drn")
    assert type(ret) == dict
    assert list(ret.keys()) == [0]
    np.testing.assert_array_equal(ret[0], exp_flopy.to_records(index=False))

    # with auxiliary
    ret = nm.flopy_package_period("drn", auxiliary="rlen")
    assert type(ret) == dict
    assert list(ret.keys()) == [0]
    np.testing.assert_array_equal(
        ret[0],
        exp_flopy.assign(rlen=np.array(
            [18.03, 6.01, 12.02, 21.08, 10.54, 10.0, 10.0], np.float32
        )).to_records(index=False))

    # Run model and read outputs
    if mf2005_exe:
        _ = nm.set_package_obj("drn", ipakcb=52)
        m.write_input()
        success, buff = m.run_model()
        assert success
        dl = read_budget(tmp_path / "modflowtest.cbc", "DRAINS")
        expected_q = np.array(
            [-0.06402918, -0.00886265, -0.02358327, -0.03124261, -0.02074423,
             -0.01962166, -0.07200401], np.float32)
        np.testing.assert_almost_equal(dl["q"], expected_q)
        assert "RLEN" not in dl.dtype.names
        assert "RLEN".ljust(16) not in dl.dtype.names

        # with auxiliary
        _ = nm.set_package_obj("drn", ipakcb=52, auxiliary="rlen")
        m.write_input()
        # edit file before running, due to flopy shortcomming
        drn_fname = tmp_path / "modflowtest.drn"
        txt = drn_fname.read_text().splitlines()
        txt[1] += " AUX RLEN"
        drn_fname.write_text("\n".join(txt))
        success, buff = m.run_model()
        assert success
        dl = read_budget(tmp_path / "modflowtest.cbc", "DRAINS")
        np.testing.assert_almost_equal(dl["q"], expected_q)
        rlen_name = "RLEN"
        if rlen_name not in dl.dtype.names:
            rlen_name = "RLEN".ljust(16)
        assert rlen_name in dl.dtype.names
        np.testing.assert_almost_equal(
            dl[rlen_name], nm.reaches["rlen"].astype(np.float32))
