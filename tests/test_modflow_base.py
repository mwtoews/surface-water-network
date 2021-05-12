# -*- coding: utf-8 -*-
"""This test suit checks the abstract base class for SwnModflow.

MODFLOW models are not run. See test_modflow.py and test_modflow6.py
for similar, but running models.
"""
import geopandas
import numpy as np
import pandas as pd
from hashlib import md5
from shapely import wkt
from shapely.geometry import Point
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

# same valid network used in test_basic
n3d_lines = wkt_to_geoseries([
    "LINESTRING Z (60 100 14, 60  80 12)",
    "LINESTRING Z (40 130 15, 60 100 14)",
    "LINESTRING Z (70 130 15, 60 100 14)",
])


@pytest.fixture
def n3d():
    return swn.SurfaceWaterNetwork.from_lines(n3d_lines)


@pytest.fixture
def n2d():
    return swn.SurfaceWaterNetwork.from_lines(force_2d(n3d_lines))

def get_basic_m(with_top: bool = False):
    """Returns a basic Flopy MODFLOW model"""
    if with_top:
        top = np.array([
            [16.0, 15.0],
            [15.0, 15.0],
            [14.0, 14.0],
        ])
    else:
        top=15.0
    m = flopy.modflow.Modflow()
    flopy.modflow.ModflowDis(
        m, nlay=1, nrow=3, ncol=2, delr=20.0, delc=20.0, top=top, botm=10.0,
        xul=30.0, yul=130.0)
    _ = flopy.modflow.ModflowBas(m)
    return m


def test_from_swn_flopy_n3d_defaults(n3d, tmpdir_factory):
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
    m = get_basic_m()
    nm = swn.SwnModflow.from_swn_flopy(n3d, m)

    # Check that "SFR" package was not automatically added
    assert m.sfr is None
    assert nm.segment_data is None

    # Check segments
    pd.testing.assert_index_equal(n3d.segments.index, nm.segments.index)
    assert list(nm.segments.in_model) == [True, True, True]

    # Check grid cells
    assert list(nm.grid_cells.index.names) == ["i", "j"]
    assert list(nm.grid_cells.index) == \
        [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    assert list(nm.grid_cells.ibound) == [1, 1, 1, 1, 1, 1]

    # Check reaches
    assert list(nm.reaches.index) == [1, 2, 3, 4, 5, 6, 7]
    assert list(nm.reaches.segnum) == [1, 1, 1, 2, 2, 0, 0]
    # Base-0
    assert list(nm.reaches.k) == [0, 0, 0, 0, 0, 0, 0]
    assert list(nm.reaches.i) == [0, 0, 1, 0, 1, 1, 2]
    assert list(nm.reaches.j) == [0, 1, 1, 1, 1, 1, 1]
    # Base-1
    assert list(nm.reaches.iseg) == [1, 1, 1, 2, 2, 3, 3]
    assert list(nm.reaches.ireach) == [1, 2, 3, 1, 2, 1, 2]
    # Other data
    np.testing.assert_array_almost_equal(
        nm.reaches.segndist,
        [0.25, 0.58333333, 0.8333333333, 0.333333333, 0.833333333, 0.25, 0.75])
    np.testing.assert_array_almost_equal(
        nm.reaches.geometry.length,
        [18.027756, 6.009252, 12.018504, 21.081851, 10.540926, 10.0, 10.0])

    if matplotlib:
        _ = nm.plot()
        plt.close()

    # Write some files
    outdir = tmpdir_factory.mktemp("n3d")
    nm.reaches.to_file(str(outdir.join("reaches.shp")))
    nm.grid_cells.to_file(str(outdir.join("grid_cells.shp")))
    gdf_to_shapefile(nm.segments, outdir.join("segments.shp"))

def test_set_reach_slope_n3d(n3d):
    m = get_basic_m()
    nm = swn.SwnModflow.from_swn_flopy(n3d, m)

    assert "slope" not in nm.reaches.columns
    nm.set_reach_slope()  # default method="auto" is "linestringz_ab"
    np.testing.assert_array_almost_equal(
        nm.reaches.slope,
        [0.027735, 0.027735, 0.027735, 0.03162277, 0.03162277, 0.1, 0.1])
    nm.set_reach_slope("grid_top", 0.01)
    np.testing.assert_array_almost_equal(
        nm.reaches.slope,
        [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])


def test_from_swn_flopy_n2d_defaults(n2d):
    # similar to 3D version, but getting Z information from model
    m = get_basic_m(with_top=True)
    nm = swn.SwnModflow.from_swn_flopy(n2d, m)

    # Check that "SFR" package was not automatically added
    assert m.sfr is None
    assert nm.segment_data is None

    # Check segments
    pd.testing.assert_index_equal(n2d.segments.index, nm.segments.index)
    assert list(nm.segments.in_model) == [True, True, True]

    # Check grid cells
    assert list(nm.grid_cells.index.names) == ["i", "j"]
    assert list(nm.grid_cells.index) == \
        [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    assert list(nm.grid_cells.ibound) == [1, 1, 1, 1, 1, 1]

    # Check reaches
    assert list(nm.reaches.index) == [1, 2, 3, 4, 5, 6, 7]
    assert list(nm.reaches.segnum) == [1, 1, 1, 2, 2, 0, 0]
    # Base-0
    assert list(nm.reaches.k) == [0, 0, 0, 0, 0, 0, 0]
    assert list(nm.reaches.i) == [0, 0, 1, 0, 1, 1, 2]
    assert list(nm.reaches.j) == [0, 1, 1, 1, 1, 1, 1]
    # Base-1
    assert list(nm.reaches.iseg) == [1, 1, 1, 2, 2, 3, 3]
    assert list(nm.reaches.ireach) == [1, 2, 3, 1, 2, 1, 2]
    # Other data
    np.testing.assert_array_almost_equal(
        nm.reaches.segndist,
        [0.25, 0.58333333, 0.8333333333, 0.333333333, 0.833333333, 0.25, 0.75])
    np.testing.assert_array_almost_equal(
        nm.reaches.geometry.length,
        [18.027756, 6.009252, 12.018504, 21.081851, 10.540926, 10.0, 10.0])

    if matplotlib:
        _ = nm.plot()
        plt.close()

def test_set_reach_slope_n2d(n2d):
    m = get_basic_m(with_top=True)
    nm = swn.SwnModflow.from_swn_flopy(n2d, m)

    assert "slope" not in nm.reaches.columns
    nm.set_reach_slope()  # default method="auto" is "grid_top"
    np.testing.assert_array_almost_equal(
        nm.reaches.slope,
        [0.070711, 0.05, 0.025, 0.05, 0.025, 0.025, 0.05])
    with pytest.raises(ValueError, match="method linestringz_ab requested"):
        nm.set_reach_slope("linestringz_ab")

def test_interp_2d_to_3d():
    m = get_basic_m(with_top=True)
    gt = swn.modflow.geotransform_from_flopy(m)
    assert gt == (30.0, 20.0, 0.0, 130.0, 0.0, -20.0)
    # Interpolate the line from the top of the model
    lsz = interp_2d_to_3d(n3d_lines, m.dis.top.array, gt)
    n = swn.SurfaceWaterNetwork.from_lines(lsz)
    n.adjust_elevation_profile()
    nm = swn.SwnModflow.from_swn_flopy(n, m)
    nm.set_reach_slope()
    np.testing.assert_array_almost_equal(
        nm.reaches.lsz_avg,  # aka strtop or rtop
        [15.742094, 15.39822, 15.140314, 14.989459, 14.973648, 14.726283,
         14.242094])
    np.testing.assert_array_almost_equal(
        nm.reaches.slope,
        [0.02861207, 0.02861207, 0.02861207, 0.001, 0.001, 0.04841886,
         0.04841886])


def test_reach_barely_outside_ibound():
    n = swn.SurfaceWaterNetwork.from_lines(wkt_to_geoseries([
        "LINESTRING (15 125, 70 90, 120 120, 130 90, "
        "150 110, 180 90, 190 110, 290 80)"
    ]))
    m = flopy.modflow.Modflow()
    flopy.modflow.ModflowDis(
        m, nrow=2, ncol=3, delr=100.0, delc=100.0, xul=0.0, yul=200.0)
    flopy.modflow.ModflowBas(m, ibound=np.array([[1, 1, 1], [0, 0, 0]]))
    nm = swn.SwnModflow.from_swn_flopy(n, m, reach_include_fraction=0.8)

    assert len(nm.reaches) == 3
    assert list(nm.reaches.segnum) == [0, 0, 0]
    assert list(nm.reaches.i) == [0, 0, 0]
    assert list(nm.reaches.j) == [0, 1, 2]
    assert list(nm.reaches.iseg) == [1, 1, 1]
    assert list(nm.reaches.ireach) == [1, 2, 3]
    np.testing.assert_array_almost_equal(
        nm.reaches.rchlen, [100.177734, 152.08736, 93.96276], 5)
    expected_reaches_geom = wkt_to_geoseries([
        "LINESTRING (15 125, 54.3 100, 70 90, 86.7 100, 100 108)",
        "LINESTRING (100 108, 120 120, 126.7 100, 130 90, 140 100, 150 110, "
        "165 100, 180 90, 185 100, 190 110, 200 107)",
        "LINESTRING (200 107, 223.3 100, 290 80)"])
    expected_reaches_geom.index += 1
    assert nm.reaches.geom_almost_equals(expected_reaches_geom, 0).all()
    assert repr(nm) == dedent("""\
        <SwnModflow: flopy mf2005 'modflowtest'
          3 in reaches (reachID): [1, 2, 3]
          1 stress period with perlen: [1.0] />""")
    if matplotlib:
        _ = nm.plot()
        plt.close()


def check_number_sum_hex(a, n, h):
    a = np.ceil(a).astype(np.int64)
    assert a.sum() == n
    ah = md5(a.tobytes()).hexdigest()
    assert ah.startswith(h), "{0} does not start with {1}".format(ah, h)


def test_coastal(coastal_lines_gdf):
    m = flopy.modflow.Modflow.load(
        "h.nam", version="mfnwt", model_ws=datadir, check=False)

    # Create a SWN with adjusted elevation profiles
    n = swn.SurfaceWaterNetwork.from_lines(coastal_lines_gdf.geometry)
    n.adjust_elevation_profile()
    nm = swn.SwnModflow.from_swn_flopy(n, m)

    # Check dataframes
    assert len(nm.reaches) == 296
    assert len(np.unique(nm.reaches.iseg)) == 184
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

    assert repr(nm) == dedent("""\
    <SwnModflow: flopy mfnwt 'h'
      296 in reaches (reachID): [1, 2, ..., 295, 296]
      1 stress period with perlen: [1.0] />""")

    if matplotlib:
        _ = nm.plot()
        plt.close()


@pytest.mark.xfail
def test_coastal_elevations(coastal_swn):
    # Load a MODFLOW model
    m = flopy.modflow.Modflow.load(
        "h.nam", version="mfnwt", model_ws=datadir, check=False)
    nm = swn.SwnModflow.from_swn_flopy(coastal_swn, m)

    # handy to set a max elevation that a stream can be
    _ = nm.get_seg_ijk()
    tops = nm.get_top_elevs_at_segs().top_up
    max_str_z = tops.describe()["75%"]
    if matplotlib:
        for seg in nm.segment_data.index[nm.segment_data.index.isin([1, 18])]:
            nm.plot_reaches_above(m, seg)
            plt.close()
    _ = nm.fix_segment_elevs(min_incise=0.2, min_slope=1.e-4,
                             max_str_z=max_str_z)
    _ = nm.reconcile_reach_strtop()
    seg_data = nm.set_segment_data(return_dict=True)
    reach_data = nm.flopy_reach_data
    flopy.modflow.mfsfr2.ModflowSfr2(
        model=m, reach_data=reach_data, segment_data=seg_data)
    if matplotlib:
        nm.plot_reaches_above(m, "all", plot_bottom=False)
        plt.close()
        for seg in nm.segment_data.index[nm.segment_data.index.isin([1, 18])]:
            nm.plot_reaches_above(m, seg)
            plt.close()
    _ = nm.set_topbot_elevs_at_reaches()
    nm.fix_reach_elevs()
    seg_data = nm.set_segment_data(return_dict=True)
    reach_data = nm.flopy_reach_data
    flopy.modflow.mfsfr2.ModflowSfr2(
        model=m, reach_data=reach_data, segment_data=seg_data)
    if matplotlib:
        nm.plot_reaches_above(m, "all", plot_bottom=False)
        plt.close()
        for seg in nm.segment_data.index[nm.segment_data.index.isin([1, 18])]:
            nm.plot_reaches_above(m, seg)
            plt.close()


def test_coastal_reduced(coastal_lines_gdf):
    n = swn.SurfaceWaterNetwork.from_lines(coastal_lines_gdf.geometry)
    assert len(n) == 304
    # Modify swn object
    n.remove(
        condition=n.segments["stream_order"] == 1,
        segnums=n.query(upstream=3047927))
    assert len(n) == 130
    # Load a MODFLOW model
    m = flopy.modflow.Modflow.load(
        "h.nam", version="mfnwt", model_ws=datadir, check=False)
    nm = swn.SwnModflow.from_swn_flopy(n, m)

    # Check dataframes
    assert len(nm.reaches) == 154
    assert len(np.unique(nm.reaches.iseg)) == 94
    assert len(nm.segments) == 130
    assert nm.segments["in_model"].sum() == 94

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

    assert repr(nm) == dedent("""\
    <SwnModflow: flopy mfnwt 'h'
      154 in reaches (reachID): [1, 2, ..., 153, 154]
      1 stress period with perlen: [1.0] />""")

    if matplotlib:
        _ = nm.plot()
        plt.close()


def test_coastal_ibound_modify(coastal_swn):
    # Load a MODFLOW model
    m = flopy.modflow.Modflow.load(
        "h.nam", version="mfnwt", model_ws=datadir, check=False)
    nm = swn.SwnModflow.from_swn_flopy(coastal_swn, m, ibound_action="modify")

    # Check dataframes
    assert len(nm.reaches) == 478
    assert len(np.unique(nm.reaches.iseg)) == 304
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
    reach_geom.almost_equals(expected_geom, 0)

    # Check modified IBOUND
    check_number_sum_hex(
        m.bas6.ibound.array, 572, "d353560128577b37f730562d2f89c025")
    assert repr(nm) == dedent("""\
        <SwnModflow: flopy mfnwt 'h'
          478 in reaches (reachID): [1, 2, ..., 477, 478]
          1 stress period with perlen: [1.0] />""")
    if matplotlib:
        _ = nm.plot()
        plt.close()


@pytest.mark.xfail
def test_lines_on_boundaries():
    m = flopy.modflow.Modflow()
    _ = flopy.modflow.ModflowDis(
        m, nrow=3, ncol=3, delr=100, delc=100, xul=0, yul=300)
    _ = flopy.modflow.ModflowBas(m)
    lines = wkt_to_geoseries([
        "LINESTRING (  0 320, 100 200)",
        "LINESTRING (100 200, 100 150, 150 100)",
        "LINESTRING (100 280, 100 200)",
        "LINESTRING (250 250, 150 150, 150 100)",
        "LINESTRING (150 100, 200   0, 300   0)",
    ])
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    nm = swn.SwnModflow.from_swn_flopy(n, m)

    if matplotlib:
        _ = nm.plot()
        plt.close()

    # Check dataframes
    # TODO: code needs to be improved for this type of case
    assert len(nm.reaches) == 7
    assert len(np.unique(nm.reaches.iseg)) == 5
    assert len(nm.segments) == 5
    assert nm.segments["in_model"].sum() == 5


def test_diversions():
    m = get_basic_m(with_top=True)
    gt = swn.modflow.geotransform_from_flopy(m)
    assert gt == (30.0, 20.0, 0.0, 130.0, 0.0, -20.0)
    # Interpolate the line from the top of the model
    lsz = interp_2d_to_3d(n3d_lines, m.dis.top.array, gt)
    n = swn.SurfaceWaterNetwork.from_lines(lsz)
    diversions = geopandas.GeoDataFrame(geometry=[
        Point(58, 97), Point(62, 97), Point(61, 89), Point(59, 89)])
    n.set_diversions(diversions=diversions)
    n.adjust_elevation_profile()
    nm = swn.SwnModflow.from_swn_flopy(n, m)

    # Check dataframes
    assert len(nm.reaches) == 11
    assert len(np.unique(nm.reaches.iseg)) == 7
    assert len(nm.diversions) == 4
    assert nm.diversions["in_model"].sum() == 4
    assert len(nm.segments) == 3
    assert nm.segments["in_model"].sum() == 3

    # Check reaches
    assert list(nm.reaches.index) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert list(nm.reaches.segnum) == [1, 1, 1, 2, 2, 0, 0, -1, -1, -1, -1]
    # Base-0
    assert list(nm.reaches.k) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert list(nm.reaches.i) == [0, 0, 1, 0, 1, 1, 2, 1, 1, 2, 2]
    assert list(nm.reaches.j) == [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # Base-1
    assert list(nm.reaches.iseg) == [1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7]
    assert list(nm.reaches.ireach) == [1, 2, 3, 1, 2, 1, 2, 1, 1, 1, 1]

    # Other data
    np.testing.assert_array_almost_equal(
        nm.reaches.segndist,
        [0.25, 0.58333333, 0.8333333333, 0.333333333, 0.833333333, 0.25, 0.75,
         0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(
        nm.reaches.rchlen,
        [18.027756, 6.009252, 12.018504, 21.081852, 10.540926, 10.0, 10.0,
         1.0, 1.0, 1.0, 1.0])

    assert "slope" not in nm.reaches.columns
    nm.set_reach_slope("auto", 0.001)
    np.testing.assert_array_almost_equal(
        nm.reaches.slope,
        [0.02861207, 0.02861207, 0.02861207, 0.001, 0.001, 0.04841886,
         0.04841886, 0.001, 0.001, 0.001, 0.001])

    assert repr(nm) == dedent("""\
        <SwnModflow: flopy mf2005 'modflowtest'
          11 in reaches (reachID): [1, 2, ..., 10, 11]
          1 stress period with perlen: [1.0] />""")
    if matplotlib:
        _ = nm.plot()
        plt.close()
