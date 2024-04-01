"""This test suit checks the abstract base class for SwnModflow.

Also check functionality in modflow._misc too.

MODFLOW models are not run. See test_modflow.py and test_modflow6.py
for similar, but running models.
"""

import logging
from hashlib import md5
from textwrap import dedent

import geopandas
import numpy as np
import pandas as pd
import pytest
from shapely import wkt
from shapely.geometry import Point

import swn
import swn.modflow
from swn.spatial import force_2d, interp_2d_to_3d

from .conftest import datadir, matplotlib, plt

try:
    import flopy
except ImportError:
    pytest.skip("skipping tests that require flopy", allow_module_level=True)


# same valid network used in test_basic
n3d_lines = geopandas.GeoSeries.from_wkt(
    [
        "LINESTRING Z (60 100 14, 60  80 12)",
        "LINESTRING Z (40 130 15, 60 100 14)",
        "LINESTRING Z (70 130 15, 60 100 14)",
    ]
)


def get_basic_swn(has_z: bool = True, has_diversions: bool = False):
    if has_z:
        n = swn.SurfaceWaterNetwork.from_lines(n3d_lines)
    else:
        n = swn.SurfaceWaterNetwork.from_lines(force_2d(n3d_lines))
    if has_diversions:
        diversions = geopandas.GeoDataFrame(
            geometry=[Point(58, 100), Point(62, 100), Point(61, 89), Point(59, 89)]
        )
        n.set_diversions(diversions=diversions)
    return n


def get_basic_modflow(with_top: bool = False, nper: int = 1):
    """Returns a basic Flopy MODFLOW model"""
    if with_top:
        top = np.array(
            [
                [16.0, 15.0],
                [15.0, 15.0],
                [14.0, 14.0],
            ]
        )
    else:
        top = 15.0
    m = flopy.modflow.Modflow()
    flopy.modflow.ModflowDis(
        m,
        nlay=1,
        nrow=3,
        ncol=2,
        nper=nper,
        delr=20.0,
        delc=20.0,
        top=top,
        botm=10.0,
        xul=30.0,
        yul=130.0,
    )
    _ = flopy.modflow.ModflowBas(m)
    return m


def test_swn_property():
    n = get_basic_swn(False)
    nm = swn.SwnModflow()
    assert nm.swn is None
    nm.swn = n
    assert nm.swn is not None
    assert nm.swn is n
    with pytest.raises(AttributeError, match="swn property can only be set o"):
        nm.swn = n


def test_segments_property():
    nm = swn.SwnModflow()
    assert nm.segments is None
    gdf = geopandas.GeoDataFrame()
    nm.segments = gdf
    assert nm.segments is not None
    assert nm.segments is gdf
    nm.segments = None
    assert nm.segments is None
    with pytest.raises(ValueError, match="segments must be a GeoDataFrame or"):
        nm.segments = pd.DataFrame()


def test_diversions_property():
    nm = swn.SwnModflow()
    assert nm.diversions is None
    gdf = geopandas.GeoDataFrame()
    nm.diversions = gdf
    assert nm.diversions is not None
    assert nm.diversions is gdf
    df = pd.DataFrame()
    nm.diversions = df
    assert nm.diversions is not None
    assert nm.diversions is df
    nm.diversions = None
    assert nm.diversions is None
    with pytest.raises(ValueError, match="diversions must be a GeoDataFrame,"):
        nm.diversions = {}


def test_reaches_property():
    nm = swn.SwnModflow()
    assert nm.reaches is None
    gdf = geopandas.GeoDataFrame()
    nm.reaches = gdf
    assert nm.reaches is not None
    assert nm.reaches is gdf
    nm.reaches = None
    assert nm.reaches is None
    with pytest.raises(ValueError, match="reaches must be a GeoDataFrame or "):
        nm.reaches = pd.DataFrame()


@pytest.mark.parametrize("has_z", [False, True], ids=["n2d", "n3d"])
def test_from_swn_flopy_defaults(has_z):
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
    n = get_basic_swn(has_z=has_z)
    m = get_basic_modflow()
    nm = swn.SwnModflow.from_swn_flopy(n, m)

    # Check that "SFR" package was not automatically added
    assert m.sfr is None
    assert nm.segment_data is None
    assert nm.segment_data_ts is None

    # Check segments
    pd.testing.assert_index_equal(n.segments.index, nm.segments.index)
    assert list(nm.segments.in_model) == [True, True, True]

    # Check grid cells
    assert list(nm.grid_cells.index.names) == ["i", "j"]
    assert list(nm.grid_cells.index) == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
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
        [0.25, 0.58333333, 0.8333333333, 0.333333333, 0.833333333, 0.25, 0.75],
    )
    np.testing.assert_array_almost_equal(
        nm.reaches.geometry.length,
        [18.027756, 6.009252, 12.018504, 21.081851, 10.540926, 10.0, 10.0],
    )

    if matplotlib:
        _ = nm.plot()
        plt.close()


def test_set_reach_data_from_segments():
    n = get_basic_swn()
    m = get_basic_modflow(with_top=False)
    n.segments["upstream_area"] = n.segments["upstream_length"] ** 2 * 100
    n.estimate_width()
    nm = swn.SwnModflow.from_swn_flopy(n, m)
    reach_idx = pd.RangeIndex(7, name="reachID") + 1
    k = pd.Series([1, 10, 100], dtype=float)
    # scalar
    nm.set_reach_data_from_segments("const_var", 9)
    pd.testing.assert_series_equal(
        nm.reaches["const_var"], pd.Series(9, name="const_var", index=reach_idx)
    )
    nm.set_reach_data_from_segments("const_var", 9.0)
    pd.testing.assert_series_equal(
        nm.reaches["const_var"], pd.Series(9.0, name="const_var", index=reach_idx)
    )
    nm.set_reach_data_from_segments("const_var", "%SCALE%")
    pd.testing.assert_series_equal(
        nm.reaches["const_var"], pd.Series("%SCALE%", name="const_var", index=reach_idx)
    )
    # series with no interpolation
    nm.set_reach_data_from_segments("sequence", n.segments.sequence)
    pd.testing.assert_series_equal(
        nm.reaches["sequence"],
        pd.Series([1, 1, 1, 2, 2, 3, 3], name="sequence", index=reach_idx),
    )
    nm.set_reach_data_from_segments(
        "boundname", (n.segments.index.to_series() + 100).astype(str)
    )
    pd.testing.assert_series_equal(
        nm.reaches["boundname"],
        pd.Series(
            ["101", "101", "101", "102", "102", "100", "100"],
            name="boundname",
            index=reach_idx,
        ),
    )
    nm.set_reach_data_from_segments("width", n.segments.width, method="constant")
    pd.testing.assert_series_equal(
        nm.reaches["width"],
        pd.Series(
            [1.766139, 1.766139, 1.766139, 1.721995, 1.721995, 2.292183, 2.292183],
            name="width",
            index=reach_idx,
        ),
    )
    # series with interpolation
    nm.set_reach_data_from_segments("var", n.segments.width, method="continuous")
    pd.testing.assert_series_equal(
        nm.reaches["var"],
        pd.Series(
            [
                1.89765007,
                2.07299816,
                2.20450922,
                1.91205787,
                2.19715192,
                2.29218327,
                2.29218327,
            ],
            name="var",
            index=reach_idx,
        ),
    )
    nm.set_reach_data_from_segments("width", n.segments.width, method="additive")
    pd.testing.assert_series_equal(
        nm.reaches["width"],
        pd.Series(
            [
                1.61475323,
                1.41290554,
                1.26151976,
                1.52519257,
                1.229988656,
                2.29218327,
                2.29218327,
            ],
            name="width",
            index=reach_idx,
        ),
    )
    nm.set_reach_data_from_segments(
        "width", n.segments.width, pd.Series(5), method="additive"
    )
    pd.testing.assert_series_equal(
        nm.reaches["width"],
        pd.Series(
            [
                1.61475323,
                1.41290554,
                1.26151976,
                1.52519257,
                1.229988656,
                2.96913745,
                4.32304582,
            ],
            name="width",
            index=reach_idx,
        ),
    )
    nm.set_reach_data_from_segments("strhc1", k)
    pd.testing.assert_series_equal(
        nm.reaches["strhc1"],
        pd.Series(
            [7.75, 4.75, 2.5, 67.0, 17.5, 1.0, 1.0], name="strhc1", index=reach_idx
        ),
    )
    nm.set_reach_data_from_segments("strhc1", k, log=True)
    pd.testing.assert_series_equal(
        nm.reaches["strhc1"],
        pd.Series(
            [5.62341325, 2.61015722, 1.46779927, 21.5443469, 2.15443469, 1.0, 1.0],
            name="strhc1",
            index=reach_idx,
        ),
    )
    nm.set_reach_data_from_segments("strhc1", k.astype(int), 1000, log=True)
    pd.testing.assert_series_equal(
        nm.reaches["strhc1"],
        pd.Series(
            [
                5.62341325,
                2.61015722,
                1.46779927,
                21.5443469,
                2.15443469,
                5.62341325,
                177.827941,
            ],
            name="strhc1",
            index=reach_idx,
        ),
    )
    expected_width = pd.Series(
        [
            1.897650,
            2.072998,
            2.204509,
            1.912058,
            2.197152,
            4.219137,
            8.073046,
        ],
        name="width",
        index=reach_idx,
    )
    nm.set_reach_data_from_segments("width", n.segments.width, 10.0)
    pd.testing.assert_series_equal(nm.reaches["width"], expected_width)
    nm.set_reach_data_from_segments("width", n.segments.width, "10")
    pd.testing.assert_series_equal(nm.reaches["width"], expected_width)
    # misc errors
    with pytest.raises(ValueError, match="name must be a str type"):
        nm.set_reach_data_from_segments(1, 2)


@pytest.mark.parametrize("has_diversions", [False, True], ids=["nodiv", "div"])
def test_set_reach_slope_n3d(has_diversions):
    n = get_basic_swn(has_z=True, has_diversions=has_diversions)
    m = get_basic_modflow(with_top=False)
    nm = swn.SwnModflow.from_swn_flopy(n, m)
    assert "slope" not in nm.reaches.columns

    nm.set_reach_slope()  # default method="auto" is "zcoord_ab"
    expected = [0.027735, 0.027735, 0.027735, 0.03162277, 0.03162277, 0.1, 0.1]
    if has_diversions:
        expected += [0.001, 0.001, 0.001, 0.001]
    np.testing.assert_array_almost_equal(nm.reaches.slope, expected)

    # Both "grid_top" and "rch_len" use the grid top, which is flat
    # so the same result is expected, with only min_slope repeated
    min_slope = 0.01
    expected = [min_slope] * 7
    if has_diversions:
        expected += [min_slope] * 4
    nm.set_reach_slope("grid_top", min_slope)
    np.testing.assert_array_almost_equal(nm.reaches.slope, expected)
    nm.set_reach_slope("rch_len", min_slope)
    np.testing.assert_array_almost_equal(nm.reaches.slope, expected)


@pytest.mark.parametrize("has_diversions", [False, True], ids=["nodiv", "div"])
def test_set_reach_slope_n2d(has_diversions):
    n = get_basic_swn(has_z=False, has_diversions=has_diversions)
    m = get_basic_modflow(with_top=True)
    nm = swn.SwnModflow.from_swn_flopy(n, m)
    assert "slope" not in nm.reaches.columns

    nm.set_reach_slope()  # default method="auto" is "grid_top"
    expected = [0.070711, 0.05, 0.025, 0.05, 0.025, 0.025, 0.05]
    if has_diversions:
        expected += [0.025, 0.025, 0.05, 0.05]
    np.testing.assert_array_almost_equal(nm.reaches.slope, expected)

    nm.set_reach_slope("rch_len")
    expected = [0.078446, 0.16641, 0.041603, 0.047434, 0.047434, 0.05, 0.1]
    if has_diversions:
        expected += [0.5, 0.5, 1.0, 1.0]
    np.testing.assert_array_almost_equal(nm.reaches.slope, expected)

    with pytest.raises(ValueError, match="method zcoord_ab requested"):
        nm.set_reach_slope("zcoord_ab")


def test_geotransform_from_flopy():
    m = get_basic_modflow(with_top=True)
    gt = swn.modflow.geotransform_from_flopy(m)
    assert gt == (30.0, 20.0, 0.0, 130.0, 0.0, -20.0)
    # Interpolate the line from the top of the model
    lsz = interp_2d_to_3d(n3d_lines, m.dis.top.array, gt)
    n = swn.SurfaceWaterNetwork.from_lines(lsz)
    n.adjust_elevation_profile()
    nm = swn.SwnModflow.from_swn_flopy(n, m)
    nm.set_reach_slope()
    np.testing.assert_array_almost_equal(
        nm.reaches.zcoord_avg,  # aka strtop or rtop
        [15.742094, 15.39822, 15.140314, 14.989459, 14.973648, 14.726283, 14.242094],
    )
    np.testing.assert_array_almost_equal(
        nm.reaches.slope,
        [0.02861207, 0.02861207, 0.02861207, 0.001, 0.001, 0.04841886, 0.04841886],
    )


def test_reach_barely_outside_ibound():
    n = swn.SurfaceWaterNetwork.from_lines(
        geopandas.GeoSeries.from_wkt(
            [
                "LINESTRING (15 125, 70 90, 120 120, 130 90, "
                "150 110, 180 90, 190 110, 290 80)"
            ]
        )
    )
    m = flopy.modflow.Modflow()
    flopy.modflow.ModflowDis(
        m, nrow=2, ncol=3, delr=100.0, delc=100.0, xul=0.0, yul=200.0
    )
    flopy.modflow.ModflowBas(m, ibound=np.array([[1, 1, 1], [0, 0, 0]]))
    nm = swn.SwnModflow.from_swn_flopy(n, m, reach_include_fraction=0.8)

    assert len(nm.reaches) == 3
    assert list(nm.reaches.segnum) == [0, 0, 0]
    assert list(nm.reaches.i) == [0, 0, 0]
    assert list(nm.reaches.j) == [0, 1, 2]
    assert list(nm.reaches.iseg) == [1, 1, 1]
    assert list(nm.reaches.ireach) == [1, 2, 3]
    np.testing.assert_array_almost_equal(
        nm.reaches.rchlen, [100.177734, 152.08736, 93.96276], 5
    )
    expected_reaches_geom = geopandas.GeoSeries.from_wkt(
        [
            "LINESTRING (15 125, 54.3 100, 70 90, 86.7 100, 100 108)",
            "LINESTRING (100 108, 120 120, 126.7 100, 130 90, 140 100, 150 110, "
            "165 100, 180 90, 185 100, 190 110, 200 107)",
            "LINESTRING (200 107, 223.3 100, 290 80)",
        ]
    )
    expected_reaches_geom.index += 1
    assert nm.reaches.geom_equals_exact(expected_reaches_geom, 0.1).all()
    assert repr(nm) == dedent(
        """\
        <SwnModflow: flopy mf2005 'modflowtest'
          3 in reaches (reachID): [1, 2, 3]
          1 stress period with perlen: [1.0] />"""
    )
    if matplotlib:
        _ = nm.plot()
        plt.close()


def test_linemerge_reaches():
    n = swn.SurfaceWaterNetwork.from_lines(
        geopandas.GeoSeries.from_wkt(
            [
                "LINESTRING (30 180, 80 170, 120 210, 140 210, 190 110, "
                "205 90, 240 60, 255 35)"
            ]
        )
    )
    m = flopy.modflow.Modflow()
    _ = flopy.modflow.ModflowDis(
        m, nrow=3, ncol=3, delr=100.0, delc=100.0, xul=0.0, yul=300.0
    )
    _ = flopy.modflow.ModflowBas(m)
    nm = swn.SwnModflow.from_swn_flopy(n, m)

    assert len(nm.reaches) == 5
    assert list(nm.reaches.i) == [1, 1, 0, 1, 2]
    assert list(nm.reaches.j) == [0, 1, 1, 1, 2]
    np.testing.assert_array_almost_equal(
        nm.reaches.rchlen, [79.274, 14.142, 45.322, 115.206, 85.669], 3
    )
    expected_reaches_geom = geopandas.GeoSeries.from_wkt(
        [
            "LINESTRING (30 180, 80 170, 100 190)",
            "LINESTRING (100 190, 110 200)",
            "LINESTRING (110 200, 120 210, 140 210, 145 200)",
            "LINESTRING (145 200, 190 110, 197.5 100, 198.75 98.33333333333334)",
            "LINESTRING (198.75 98.33333333333334, 200 96.66666666666667, "
            "205 90, 240 60, 255 35)",
        ]
    )
    expected_reaches_geom.index += 1
    assert nm.reaches.geom_equals_exact(expected_reaches_geom, 0).all()
    assert repr(nm) == dedent(
        """\
        <SwnModflow: flopy mf2005 'modflowtest'
          5 in reaches (reachID): [1, 2, ..., 4, 5]
          1 stress period with perlen: [1.0] />"""
    )
    if matplotlib:
        _ = nm.plot()
        plt.close()


def test_linemerge_reaches_2():
    n = swn.SurfaceWaterNetwork.from_lines(
        geopandas.GeoSeries.from_wkt(
            [
                "LINESTRING(103.48 235.46,103.48 179.46,95.48 171.46,95.48 163.46,"
                "103.48 155.46,103.48 139.46,119.48 123.46,199.48 123.46,"
                "207.48 115.46,215.48 115.46,223.48 107.46,239.48 107.46,247.48 99.46,"
                "255.48 99.46,271.48 83.46,271.48 75.46,279.48 67.46,279.48 59.46,"
                "287.48 51.46,287.48 43.46,295.48 35.46,295.48 3.46,303.48 -4.54)"
            ]
        )
    )
    m = flopy.modflow.Modflow()
    _ = flopy.modflow.ModflowDis(
        m, nrow=2, ncol=3, delr=100.0, delc=100.0, xul=0, yul=200
    )
    _ = flopy.modflow.ModflowBas(m, ibound=np.array([[1, 1, 0], [1, 1, 1]]))
    nm = swn.SwnModflow.from_swn_flopy(n, m)

    assert len(nm.reaches) == 4
    assert list(nm.reaches.i) == [0, 0, 0, 1]
    assert list(nm.reaches.j) == [1, 0, 1, 2]
    np.testing.assert_array_almost_equal(
        nm.reaches.rchlen, [25.461, 20.784, 134.863, 178.51], 3
    )
    assert repr(nm) == dedent(
        """\
        <SwnModflow: flopy mf2005 'modflowtest'
          4 in reaches (reachID): [1, 2, 3, 4]
          1 stress period with perlen: [1.0] />"""
    )
    if matplotlib:
        _ = nm.plot()
        plt.close()


def test_linemerge_reaches_3():
    lines = geopandas.GeoSeries.from_wkt(
        [
            "LINESTRING(-9 295,63 295,71 303,87 303,95 295,119 295,143 271,"
            "143 255,151 247,167 247,175 239,183 239,191 247,255 247,263 255,"
            "271 255,279 247,295 247,303 239,303 207,271 175,255 175,247 167,"
            "239 167,215 143,215 135,207 127,207 119,199 111,199 87,191 79,"
            "191 39,207 23,223 23,231 31,239 31,255 47,263 47,271 55,279 55,"
            "287 63,295 63,303 71,311 71,319 79,327 79,335 87,343 87,351 95,"
            "367 95,375 103,391 103,399 111,423 111,439 95)",
            "LINESTRING(431 415,391 375,391 303,383 295,383 287,375 279,375 263,"
            "383 255,391 255,399 247,431 247,439 255,447 255,455 263,463 263,"
            "471 271,495 271,527 239,527 231,535 223,535 215,527 207,527 199,"
            "503 175,495 175,487 167,479 167,447 135,447 119,439 111,439 95)",
            "LINESTRING(439 95,439 71,471 39,503 39,511 47,551 47,607 -9)",
        ]
    )
    lines.index += 100
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    m = flopy.modflow.Modflow()
    _ = flopy.modflow.ModflowDis(
        m, nrow=4, ncol=6, delr=100.0, delc=100.0, xul=0, yul=400
    )
    _ = flopy.modflow.ModflowBas(m)
    nm = swn.SwnModflow.from_swn_flopy(n, m)

    assert len(nm.reaches) == 22
    assert list(nm.reaches.segnum) == [100] * 13 + [101] * 7 + [102] * 2
    assert list(nm.reaches.i) == (
        [1, 0, 1, 1, 1, 1, 1, 2, 3, 3, 3, 2, 2, 0, 0, 1, 1, 1, 2, 2, 3, 3]
    )
    assert list(nm.reaches.j) == (
        [0, 0, 0, 1, 2, 3, 2, 2, 1, 2, 3, 3, 4, 4, 3, 3, 4, 5, 5, 4, 4, 5]
    )
    assert list(nm.reaches.iseg) == [1] * 13 + [2] * 7 + [3] * 2
    assert list(nm.reaches.ireach) == (
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1, 2, 3, 4, 5, 6, 7, 1, 2]
    )
    np.testing.assert_array_almost_equal(
        nm.reaches.rchlen,
        [
            79.1,
            24.5,
            12.1,
            135.9,
            108.7,
            40.5,
            5.7,
            143.2,
            88.0,
            121.5,
            85.3,
            32.6,
            45.6,
            43.8,
            89.0,
            74.0,
            112.0,
            83.8,
            37.9,
            112.9,
            98.3,
            133.5,
        ],
        1,
    )
    assert repr(nm) == dedent(
        """\
        <SwnModflow: flopy mf2005 'modflowtest'
          22 in reaches (reachID): [1, 2, ..., 21, 22]
          1 stress period with perlen: [1.0] />"""
    )
    if matplotlib:
        _ = nm.plot(cmap="nipy_spectral")
        plt.close()


def check_number_sum_hex(a, n, h):
    a = np.ceil(a).astype(np.int64)
    assert a.sum() == n
    ah = md5(a.tobytes()).hexdigest()
    assert ah.startswith(h), f"{ah} does not start with {h}"


def test_coastal(coastal_lines_gdf):
    m = flopy.modflow.Modflow.load(
        "h.nam", version="mfnwt", model_ws=datadir, check=False
    )

    # Create a SWN with adjusted elevation profiles
    n = swn.SurfaceWaterNetwork.from_lines(coastal_lines_gdf.geometry)
    n.adjust_elevation_profile()
    nm = swn.SwnModflow.from_swn_flopy(n, m)

    # Check dataframes
    assert len(nm.reaches) == 297
    assert len(np.unique(nm.reaches.iseg)) == 185
    assert len(nm.segments) == 304
    assert nm.segments["in_model"].sum() == 184

    # Check remaining reaches added that are inside model domain
    reach_geom = nm.reaches.loc[nm.reaches["segnum"] == 3047735, "geometry"].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 980.5448069140768)
    # These should be split between two cells
    reach_geoms = nm.reaches.loc[nm.reaches["segnum"] == 3047750, "geometry"]
    assert len(reach_geoms) == 2
    np.testing.assert_almost_equal(reach_geoms.iloc[0].length, 204.90164560019)
    np.testing.assert_almost_equal(reach_geoms.iloc[1].length, 789.59872070638)
    # This reach should not be extended, the remainder is too far away
    reach_geom = nm.reaches.loc[nm.reaches["segnum"] == 3047762, "geometry"].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 261.4644731621629)
    # This reach should not be extended, the remainder is too long
    reach_geom = nm.reaches.loc[nm.reaches["segnum"] == 3047926, "geometry"].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 237.72893664132727)

    assert repr(nm) == dedent(
        """\
        <SwnModflow: flopy mfnwt 'h'
          297 in reaches (reachID): [1, 2, ..., 296, 297]
          1 stress period with perlen: [1.0] />"""
    )

    if matplotlib:
        _ = nm.plot()
        plt.close()


@pytest.mark.xfail
def test_coastal_elevations(coastal_swn):
    # Load a MODFLOW model
    m = flopy.modflow.Modflow.load(
        "h.nam", version="mfnwt", model_ws=datadir, check=False
    )
    nm = swn.SwnModflow.from_swn_flopy(coastal_swn, m)

    # handy to set a max elevation that a stream can be
    _ = nm.get_seg_ijk()
    tops = nm.get_top_elevs_at_segs().top_up
    max_str_z = tops.describe()["75%"]
    if matplotlib:
        for seg in nm.segment_data.index[nm.segment_data.index.isin([1, 18])]:
            nm.plot_reaches_vs_model(seg)
            plt.close()
    _ = nm.fix_segment_elevs(min_incise=0.2, min_slope=1.0e-4, max_str_z=max_str_z)
    _ = nm.reconcile_reach_strtop()
    seg_data = nm.flopy_segment_data()
    reach_data = nm.flopy_reach_data()
    flopy.modflow.mfsfr2.ModflowSfr2(
        model=m, reach_data=reach_data, segment_data=seg_data
    )
    if matplotlib:
        nm.plot_reaches_vs_model("all", plot_bottom=False)
        plt.close()
        for seg in nm.segment_data.index[nm.segment_data.index.isin([1, 18])]:
            nm.plot_reaches_vs_model(seg)
            plt.close()
    _ = nm.add_model_topbot_to_reaches()
    nm.fix_reach_elevs()
    seg_data = nm.flopy_segment_data()
    reach_data = nm.flopy_reach_data()
    flopy.modflow.mfsfr2.ModflowSfr2(
        model=m, reach_data=reach_data, segment_data=seg_data
    )
    if matplotlib:
        nm.plot_reaches_vs_model("all", plot_bottom=False)
        plt.close()
        for seg in nm.segment_data.index[nm.segment_data.index.isin([1, 18])]:
            nm.plot_reaches_vs_model(seg)
            plt.close()


def test_coastal_reduced(coastal_lines_gdf):
    n = swn.SurfaceWaterNetwork.from_lines(coastal_lines_gdf.geometry)
    assert len(n) == 304
    # Modify swn object
    n.remove(
        condition=n.segments["stream_order"] == 1,
        segnums=n.gather_segnums(upstream=3047927),
    )
    assert len(n) == 130
    # Load a MODFLOW model
    m = flopy.modflow.Modflow.load(
        "h.nam", version="mfnwt", model_ws=datadir, check=False
    )
    nm = swn.SwnModflow.from_swn_flopy(n, m)

    # Check dataframes
    assert len(nm.reaches) == 154
    assert len(np.unique(nm.reaches.iseg)) == 94
    assert len(nm.segments) == 130
    assert nm.segments["in_model"].sum() == 94

    # These should be split between two cells
    reach_geoms = nm.reaches.loc[nm.reaches["segnum"] == 3047750, "geometry"]
    assert len(reach_geoms) == 2
    np.testing.assert_almost_equal(reach_geoms.iloc[0].length, 204.90164560019)
    np.testing.assert_almost_equal(reach_geoms.iloc[1].length, 789.59872070638)
    # This reach should not be extended, the remainder is too far away
    reach_geom = nm.reaches.loc[nm.reaches["segnum"] == 3047762, "geometry"].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 261.4644731621629)
    # This reach should not be extended, the remainder is too long
    reach_geom = nm.reaches.loc[nm.reaches["segnum"] == 3047926, "geometry"].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 237.72893664132727)

    assert repr(nm) == dedent(
        """\
        <SwnModflow: flopy mfnwt 'h'
          154 in reaches (reachID): [1, 2, ..., 153, 154]
          1 stress period with perlen: [1.0] />"""
    )

    if matplotlib:
        _ = nm.plot()
        plt.close()


def test_coastal_ibound_modify(coastal_swn):
    # Load a MODFLOW model
    m = flopy.modflow.Modflow.load(
        "h.nam", version="mfnwt", model_ws=datadir, check=False
    )
    nm = swn.SwnModflow.from_swn_flopy(coastal_swn, m, ibound_action="modify")

    # Check dataframes
    assert len(nm.reaches) == 478
    assert len(np.unique(nm.reaches.iseg)) == 304
    assert len(nm.segments) == 304
    assert nm.segments["in_model"].sum() == 304
    assert not (nm.reaches.prev_ibound == 1).all()

    # Check a remaining reach added that is outside model domain
    reach_geom = nm.reaches.loc[nm.reaches["segnum"] == 3048565, "geometry"].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 647.316024023105)
    expected_geom = wkt.loads(
        "LINESTRING Z (1819072.5 5869685.1 4, 1819000 5869684.9 5.7, "
        "1818997.5 5869684.9 5.8, 1818967.5 5869654.9 5, "
        "1818907.5 5869654.8 4, 1818877.6 5869624.7 5, 1818787.5 5869624.5 6, "
        "1818757.6 5869594.5 5.1, 1818697.6 5869594.4 5.7, "
        "1818667.6 5869564.3 6.2, 1818607.6 5869564.2 4.7, "
        "1818577.6 5869534.1 5.6, 1818487.6 5869534 6.2)"
    )
    reach_geom.equals_exact(expected_geom, 0)

    # Check modified IBOUND
    check_number_sum_hex(m.bas6.ibound.array, 572, "d353560128577b37f730562d2f89c025")
    assert repr(nm) == dedent(
        """\
        <SwnModflow: flopy mfnwt 'h'
          478 in reaches (reachID): [1, 2, ..., 477, 478]
          1 stress period with perlen: [1.0] />"""
    )
    if matplotlib:
        _ = nm.plot()
        plt.close()


@pytest.mark.xfail
def test_lines_on_boundaries():
    m = flopy.modflow.Modflow()
    _ = flopy.modflow.ModflowDis(m, nrow=3, ncol=3, delr=100, delc=100, xul=0, yul=300)
    _ = flopy.modflow.ModflowBas(m)
    lines = geopandas.GeoSeries.from_wkt(
        [
            "LINESTRING (  0 320, 100 200)",
            "LINESTRING (100 200, 100 150, 150 100)",
            "LINESTRING (100 280, 100 200)",
            "LINESTRING (250 250, 150 150, 150 100)",
            "LINESTRING (150 100, 200   0, 300   0)",
        ]
    )
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
    m = get_basic_modflow(with_top=True)
    gt = swn.modflow.geotransform_from_flopy(m)
    assert gt == (30.0, 20.0, 0.0, 130.0, 0.0, -20.0)
    # Interpolate the line from the top of the model
    lsz = interp_2d_to_3d(n3d_lines, m.dis.top.array, gt)
    n = swn.SurfaceWaterNetwork.from_lines(lsz)
    diversions = geopandas.GeoDataFrame(
        geometry=[Point(58, 100), Point(62, 100), Point(61, 89), Point(59, 89)]
    )
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
        [0.25, 7 / 12, 5 / 6, 1 / 3, 5 / 6, 0.25, 0.75, 0.0, 0.0, 0.0, 0.0],
    )
    np.testing.assert_array_almost_equal(
        nm.reaches.rchlen,
        [
            18.027756,
            6.009252,
            12.018504,
            21.081852,
            10.540926,
            10.0,
            10.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
    )

    assert "slope" not in nm.reaches.columns
    nm.set_reach_slope("auto", 0.001)
    np.testing.assert_array_almost_equal(
        nm.reaches.slope,
        [
            0.02861207,
            0.02861207,
            0.02861207,
            0.001,
            0.001,
            0.04841886,
            0.04841886,
            0.001,
            0.001,
            0.001,
            0.001,
        ],
    )

    assert repr(nm) == dedent(
        """\
        <SwnModflow: flopy mf2005 'modflowtest'
          11 in reaches (reachID): [1, 2, ..., 10, 11]
          1 stress period with perlen: [1.0] />"""
    )
    if matplotlib:
        _ = nm.plot()
        plt.close()


def test_transform_data_from_dict():
    from swn.modflow._misc import transform_data_to_series_or_frame as f

    time_index = pd.DatetimeIndex(["2000-07-01", "2000-07-02"])
    mapping = pd.Series(
        [1, 2, 3], name="nseg", index=pd.Index([1, 2, 0], int, name="segnum")
    )

    # returns series
    pd.testing.assert_series_equal(f({}, float, time_index), pd.Series(dtype=float))

    pd.testing.assert_series_equal(
        f({}, float, time_index, mapping), pd.Series(dtype=float)
    )

    pd.testing.assert_series_equal(
        f({0: 10, 1: 11.1}, float, time_index), pd.Series([10.0, 11.1], index=[0, 1])
    )

    pd.testing.assert_series_equal(
        f({101: 10, 100: 11.1}, int, time_index),
        pd.Series([10, 11], index=[101, 100], dtype=int),
    )

    pd.testing.assert_series_equal(
        f({0: 10, 1: 11}, float, time_index, mapping),
        pd.Series([10.0, 11.0], index=[3, 1]),
    )

    pd.testing.assert_series_equal(
        f({0: 10, 1: 11}, float, time_index, mapping, {1}), pd.Series([10.0], index=[3])
    )

    pd.testing.assert_series_equal(
        f({0: 10, 1: 11, 3: 12}, float, time_index, mapping, {3}),
        pd.Series([10.0, 11.0], index=[3, 1]),
    )

    # errors returning series
    with pytest.raises(KeyError, match="dict has a disjoint segnum set"):
        f({3: 10}, float, time_index, mapping)
    with pytest.raises(KeyError, match="dict has a disjoint segnum set"):
        f({"0": 10, "1": 11}, float, time_index, mapping),
    with pytest.raises(KeyError, match="dict has 1 key not found in segnum"):
        f({0: 1.1, 3: 10}, float, time_index, mapping)
    with pytest.raises(KeyError, match="dict has 2 keys not found in segnum"):
        f({0: 1.1, 3: 10, 10: 1}, float, time_index, mapping)

    # returns frame
    pd.testing.assert_frame_equal(
        f({0: [10, 10], 1: [11.1, 12.2]}, float, time_index),
        pd.DataFrame({0: [10, 10], 1: [11.1, 12.2]}, dtype=float, index=time_index),
    )

    pd.testing.assert_frame_equal(
        f({101: [10, 10], 100: [11.1, 12.2]}, int, time_index),
        pd.DataFrame({101: [10, 10], 100: [11, 12]}, dtype=int, index=time_index),
    )

    pd.testing.assert_frame_equal(
        f({0: [10, 10], 1: [11.1, 12.2]}, float, time_index, mapping),
        pd.DataFrame({3: [10, 10], 1: [11.1, 12.2]}, dtype=float, index=time_index),
    )

    pd.testing.assert_frame_equal(
        f({0: [10, 10], 1: [11.1, 12.2]}, float, time_index, mapping, {1}),
        pd.DataFrame({3: [10, 10]}, dtype=float, index=time_index),
    )

    pd.testing.assert_frame_equal(
        f({0: [10, 10], 1: [11.1, 12.2]}, float, time_index, mapping, {3}),
        pd.DataFrame({3: [10, 10], 1: [11.1, 12.2]}, dtype=float, index=time_index),
    )

    # errors returning frame
    with pytest.raises(ValueError, match="mixture of iterable and scalar val"):
        f({1: 10, 2: [1, 2]}, float, time_index)
    with pytest.raises(ValueError, match="inconsistent lengths found in dict"):
        f({1: [10], 2: [1, 2]}, float, time_index)
    with pytest.raises(ValueError, match="length of dict series does not mat"):
        f({1: [10], 2: [1]}, float, time_index)
    with pytest.raises(KeyError, match="dict has a disjoint segnum set"):
        f({3: [10, 10]}, float, time_index, mapping)
    with pytest.raises(KeyError, match="dict has a disjoint segnum set"):
        f({"0": [10, 10], "1": [11.1, 12.2]}, float, time_index, mapping)
    with pytest.raises(KeyError, match="dict has 1 key not found in segnum"):
        f({0: [0, 0], 3: [0, 0]}, float, time_index, mapping)
    with pytest.raises(KeyError, match="dict has 2 keys not found in segnum"):
        f({0: [0, 0], 3: [0, 0], 10: [0, 0]}, float, time_index, mapping)


def test_transform_data_from_series():
    from swn.modflow._misc import transform_data_to_series_or_frame as f

    time_index = pd.DatetimeIndex(["2000-07-01", "2000-07-02"])
    mapping = pd.Series(
        [1, 2, 3], name="nseg", index=pd.Index([1, 2, 0], name="segnum")
    )

    pd.testing.assert_series_equal(
        f(pd.Series(dtype=float), float, time_index), pd.Series(dtype=float)
    )

    pd.testing.assert_series_equal(
        f(pd.Series(dtype=float), float, time_index, mapping), pd.Series(dtype=float)
    )

    pd.testing.assert_series_equal(
        f(pd.Series([10.0, 11.1], index=[0, 1]), float, time_index),
        pd.Series([10.0, 11.1], index=[0, 1]),
    )

    pd.testing.assert_series_equal(
        f(pd.Series([10, 11], index=[101, 100]), int, time_index),
        pd.Series([10, 11], index=[101, 100], dtype=int),
    )

    pd.testing.assert_series_equal(
        f(pd.Series([set(), {1, 2}], index=[101, 100]), set, time_index),
        pd.Series([set(), {1, 2}], index=[101, 100]),
    )

    pd.testing.assert_series_equal(
        f(pd.Series([10.0, 11.0], index=[0, 1]), float, time_index, mapping),
        pd.Series([10.0, 11.0], index=[3, 1]),
    )

    pd.testing.assert_series_equal(
        f(pd.Series([10, 11], index=["0", "1"]), int, time_index, mapping),
        pd.Series([10, 11], index=[3, 1], dtype=int),
    )

    pd.testing.assert_series_equal(
        f(
            pd.Series([10.0, 11.0, 12.0], index=[0, 1, 2]),
            float,
            time_index,
            mapping,
            {2},
        ),
        pd.Series([10.0, 11.0], index=[3, 1]),
    )

    # errors
    # with pytest.raises(ValueError,match="dtype for series cannot be object"):
    #    f(pd.Series([[1]]), float, time_index)
    with pytest.raises(ValueError, match="cannot cast index.dtype to int64"):
        f(pd.Series([10, 11], index=["0", "1A"]), int, time_index, mapping)
    with pytest.raises(KeyError, match="series has a disjoint segnum set"):
        f(pd.Series([3], index=[10]), float, time_index, mapping)
    with pytest.raises(KeyError, match="series has 1 key not found in segnum"):
        f(pd.Series([2, 3], index=[0, 3]), float, time_index, mapping)
    with pytest.raises(KeyError, match="series has 2 keys not found in segnu"):
        f(pd.Series([2, 3, 4], index=[0, 3, 10]), float, time_index, mapping)


def test_transform_data_from_frame():
    from swn.modflow._misc import transform_data_to_series_or_frame as f

    time_index = pd.DatetimeIndex(["2000-07-01", "2000-07-02"])
    mapping = pd.Series(
        [1, 2, 3], name="nseg", index=pd.Index([1, 2, 0], name="segnum")
    )

    pd.testing.assert_frame_equal(
        f(pd.DataFrame(dtype=float, index=time_index), float, time_index),
        pd.DataFrame(dtype=float, index=time_index),
    )

    pd.testing.assert_frame_equal(
        f(pd.DataFrame(dtype=float, index=[0, 1]), float, time_index),
        pd.DataFrame(dtype=float, index=time_index),
    )

    pd.testing.assert_frame_equal(
        f(pd.DataFrame(dtype=float, index=[0, 1]), float, time_index, mapping),
        pd.DataFrame(dtype=float, index=time_index),
    )

    pd.testing.assert_frame_equal(
        f(
            pd.DataFrame({0: [10, 10], 1: [11.1, 12.2]}, index=time_index),
            float,
            time_index,
        ),
        pd.DataFrame({0: [10, 10], 1: [11.1, 12.2]}, dtype=float, index=time_index),
    )

    pd.testing.assert_frame_equal(
        f(pd.DataFrame({0: [10, 10], 1: [11.1, 12.2]}), float, time_index),
        pd.DataFrame({0: [10, 10], 1: [11.1, 12.2]}, dtype=float, index=time_index),
    )

    pd.testing.assert_frame_equal(
        f(
            pd.DataFrame({101: [10, 10], 100: [11.1, 12.2]}, index=time_index),
            int,
            time_index,
        ),
        pd.DataFrame({101: [10, 10], 100: [11, 12]}, dtype=int, index=time_index),
    )

    pd.testing.assert_frame_equal(
        f(pd.DataFrame({101: [10, 10], 100: [11.1, 12.2]}), int, time_index),
        pd.DataFrame({101: [10, 10], 100: [11, 12]}, dtype=int, index=time_index),
    )

    pd.testing.assert_frame_equal(
        f(
            pd.DataFrame({0: [10, 10], 1: [11.1, 12.2]}, index=time_index),
            float,
            time_index,
            mapping,
        ),
        pd.DataFrame({3: [10, 10], 1: [11.1, 12.2]}, dtype=float, index=time_index),
    )

    pd.testing.assert_frame_equal(
        f(pd.DataFrame({0: [10, 10], 1: [11.1, 12.2]}), float, time_index, mapping),
        pd.DataFrame({3: [10, 10], 1: [11.1, 12.2]}, dtype=float, index=time_index),
    )

    pd.testing.assert_frame_equal(
        f(
            pd.DataFrame({0: [10, 10], 1: [11.1, 12.2]}, index=time_index),
            float,
            time_index,
            mapping,
            {2},
        ),
        pd.DataFrame({3: [10, 10], 1: [11.1, 12.2]}, dtype=float, index=time_index),
    )

    pd.testing.assert_frame_equal(
        f(
            pd.DataFrame({0: [10, 10], 1: [11.1, 12.2], 2: [13, 14]}),
            float,
            time_index,
            mapping,
            {2, 1},
        ),
        pd.DataFrame({3: [10, 10]}, dtype=float, index=time_index),
    )

    # errors returning frame
    with pytest.raises(ValueError, match="frame index should be a DatetimeIn"):
        f(pd.DataFrame({3: [10, 10]}, index=[1, 2]), float, time_index, mapping)
    with pytest.raises(ValueError, match="length of frame index does not mat"):
        f(
            pd.DataFrame({3: [10]}, index=pd.DatetimeIndex(["2000-07-03"])),
            float,
            time_index,
            mapping,
        )
    with pytest.raises(ValueError, match="frame index does not match time in"):
        f(
            pd.DataFrame(
                {3: [10, 10]}, index=pd.DatetimeIndex(["2000-07-03", "2000-07-04"])
            ),
            float,
            time_index,
            mapping,
        )
    with pytest.raises(ValueError, match="cannot cast columns.dtype to int64"):
        f(
            pd.DataFrame({"0": [10, 10], "1A": [11.1, 12.2]}),
            float,
            time_index,
            mapping,
        ),
    with pytest.raises(KeyError, match="frame has a disjoint segnum set"):
        f(pd.DataFrame({3: [10, 10]}, index=time_index), float, time_index, mapping)
    with pytest.raises(KeyError, match="frame has 1 key not found in segnum"):
        f(
            pd.DataFrame({0: [0, 0], 3: [0, 0]}, index=time_index),
            float,
            time_index,
            mapping,
        )
    with pytest.raises(KeyError, match="frame has 2 keys not found in segnum"):
        f(
            pd.DataFrame({0: [0, 0], 3: [0, 0], 10: [0, 0]}, index=time_index),
            float,
            time_index,
            mapping,
        )


def test_get_segments_inflow():
    m = get_basic_modflow()
    n = get_basic_swn()
    nm = swn.SwnModflow.from_swn_flopy(n, m)

    pd.testing.assert_series_equal(nm._get_segments_inflow({}), pd.Series(dtype=float))

    pd.testing.assert_frame_equal(
        nm._get_segments_inflow(pd.DataFrame(dtype=float, index=[0])),
        pd.DataFrame(dtype=float, index=nm.time_index),
    )

    pd.testing.assert_series_equal(
        nm._get_segments_inflow({4: 1.1}), pd.Series(dtype=float)
    )

    pd.testing.assert_frame_equal(
        nm._get_segments_inflow({4: [1.1]}),
        pd.DataFrame(dtype=float, index=nm.time_index),
    )

    nm.segments.from_segnums.at[1] = {4}

    pd.testing.assert_series_equal(
        nm._get_segments_inflow({4: 1.1}), pd.Series({1: 1.1})
    )

    pd.testing.assert_frame_equal(
        nm._get_segments_inflow({4: [1.1]}),
        pd.DataFrame({1: [1.1]}, index=nm.time_index),
    )

    pd.testing.assert_series_equal(
        nm._get_segments_inflow({1: 2.2, 4: 1.1}), pd.Series({1: 1.1})
    )

    pd.testing.assert_frame_equal(
        nm._get_segments_inflow({1: [2.2], 4: [1.1]}),
        pd.DataFrame({1: [1.1]}, index=nm.time_index),
    )

    nm.segments.from_segnums.at[1] = {4, 5}

    pd.testing.assert_series_equal(
        nm._get_segments_inflow({5: 2.2, 4: 1.1}), pd.Series({1: 3.3})
    )

    pd.testing.assert_frame_equal(
        nm._get_segments_inflow({5: [2.2], 4: [1.1]}),
        pd.DataFrame({1: [3.3]}, index=nm.time_index),
    )

    m = get_basic_modflow(nper=2)
    nm = swn.SwnModflow.from_swn_flopy(n, m)

    pd.testing.assert_series_equal(nm._get_segments_inflow({}), pd.Series(dtype=float))

    pd.testing.assert_frame_equal(
        nm._get_segments_inflow(pd.DataFrame(dtype=float, index=[0, 1])),
        pd.DataFrame(dtype=float, index=nm.time_index),
    )

    pd.testing.assert_series_equal(
        nm._get_segments_inflow({4: 1.1}), pd.Series(dtype=float)
    )

    pd.testing.assert_frame_equal(
        nm._get_segments_inflow({4: [1.1, 2.2]}),
        pd.DataFrame(dtype=float, index=nm.time_index),
    )

    nm.segments.from_segnums.at[1] = {4}

    pd.testing.assert_series_equal(
        nm._get_segments_inflow({4: 1.1}), pd.Series({1: 1.1})
    )

    pd.testing.assert_frame_equal(
        nm._get_segments_inflow({4: [1.1, 2.2]}),
        pd.DataFrame({1: [1.1, 2.2]}, index=nm.time_index),
    )

    pd.testing.assert_series_equal(
        nm._get_segments_inflow({1: 2.2, 4: 1.1}), pd.Series({1: 1.1})
    )

    pd.testing.assert_frame_equal(
        nm._get_segments_inflow({1: [2.2, 2.3], 4: [1.1, 1.2]}),
        pd.DataFrame({1: [1.1, 1.2]}, index=nm.time_index),
    )

    nm.segments.from_segnums.at[1] = {4, 5}

    pd.testing.assert_series_equal(
        nm._get_segments_inflow({5: 2.2, 4: 1.1}), pd.Series({1: 3.3})
    )

    pd.testing.assert_frame_equal(
        nm._get_segments_inflow({5: [2.2, 2.3], 4: [1.1, 1.2]}),
        pd.DataFrame({1: [3.3, 3.5]}, index=nm.time_index),
    )


def test_get_location_frame_reach_info(caplog):
    m = get_basic_modflow(with_top=True)
    gt = swn.modflow.geotransform_from_flopy(m)
    lsz = interp_2d_to_3d(n3d_lines, m.dis.top.array, gt)
    n = swn.SurfaceWaterNetwork.from_lines(lsz)
    n.adjust_elevation_profile()
    diversions = geopandas.GeoSeries(
        [Point(58, 100), Point(62, 100), Point(61, 89), Point(59, 89)]
    )
    # n.set_diversions(diversions=geopandas.GeoDataFrame(geometry=diversions))
    nm = swn.SwnModflow.from_swn_flopy(n, m)

    loc_gdf = n.locate_geoms(diversions)
    r_df = nm.get_location_frame_reach_info(loc_gdf)
    expected_df = pd.DataFrame(
        data={
            "reachID": [3, 5, 7, 7],
            "k": [0] * 4,
            "i": [1, 1, 2, 2],
            "j": [1, 1, 1, 1],
            "iseg": [1, 2, 3, 3],
            "ireach": [3, 2, 2, 2],
            "dist_to_reach": [1.6641005886756874, 1.897366596101, 1.0, 1.0],
        }
    )
    pd.testing.assert_frame_equal(r_df, expected_df)

    # With DataFrame
    loc_df = pd.DataFrame(loc_gdf.drop(columns="geometry"))
    r_df = nm.get_location_frame_reach_info(loc_df)
    assert "dist_to_reach" not in r_df.columns
    pd.testing.assert_frame_equal(r_df, expected_df.drop(columns="dist_to_reach"))

    # With optional geom_loc_df
    r_df = nm.get_location_frame_reach_info(loc_df, geom_loc_df=diversions)
    pd.testing.assert_frame_equal(r_df, expected_df)

    # With DataFrame and optional geom_loc_df

    # Change downstream_bias -- no change with most values
    r_df = nm.get_location_frame_reach_info(loc_gdf, downstream_bias=0.01)
    pd.testing.assert_frame_equal(r_df, expected_df)
    r_df = nm.get_location_frame_reach_info(loc_gdf, downstream_bias=-0.01)
    pd.testing.assert_frame_equal(r_df, expected_df)
    r_df = nm.get_location_frame_reach_info(loc_gdf, downstream_bias=0.5)
    pd.testing.assert_frame_equal(r_df, expected_df)
    assert list(r_df.reachID) == [3, 5, 7, 7]
    r_df = nm.get_location_frame_reach_info(loc_gdf, downstream_bias=-0.5)
    assert list(r_df.reachID) == [3, 5, 6, 6]

    # Test with non-zero downstream_bias and optional geom_loc_df
    r_df = nm.get_location_frame_reach_info(
        loc_df, downstream_bias=-0.5, geom_loc_df=diversions
    )
    assert list(r_df.reachID) == [3, 5, 6, 6]

    # Consider points outside model
    ext_points = pd.concat(
        [
            diversions,
            geopandas.GeoSeries([Point(1, 2), Point(2000, 3000)], index=[4, 5]),
        ]
    )
    expected_ext_df = pd.concat(
        [
            expected_df,
            pd.DataFrame(
                data={
                    "reachID": [7, 4],
                    "k": [0] * 2,
                    "i": [2, 0],
                    "j": [1, 1],
                    "iseg": [3, 2],
                    "ireach": [2, 1],
                    "dist_to_reach": [97.8008179924892, 3458.583525086535],
                },
                index=[4, 5],
            ),
        ]
    )
    loc_df = n.locate_geoms(ext_points)
    r_df = nm.get_location_frame_reach_info(loc_df)
    pd.testing.assert_frame_equal(r_df, expected_ext_df)
    assert list(loc_df.index) == [0, 1, 2, 3, 4, 5]
    loc_df.loc[loc_df.dist_to_seg > 100.0, "segnum"] = n.END_SEGNUM
    with caplog.at_level(logging.WARNING):
        r_df = nm.get_location_frame_reach_info(loc_df)
        assert "location index missing 1 match: 5" in caplog.messages[-1]
    assert list(r_df.index) == [0, 1, 2, 3, 4]
    pd.testing.assert_frame_equal(r_df, expected_ext_df.drop(index=5))
    loc_df.loc[loc_df.dist_to_seg > 10.0, "segnum"] = n.END_SEGNUM
    with caplog.at_level(logging.WARNING):
        r_df = nm.get_location_frame_reach_info(loc_df)
        assert "location index missing 2 matches: 4, 5" in caplog.messages[-1]
    assert list(r_df.index) == [0, 1, 2, 3]
    pd.testing.assert_frame_equal(r_df, expected_ext_df.drop(index=[4, 5]))

    # misc
    loc_df = n.locate_geoms(diversions)
    loc_df["reachID"] = r_df.reachID
    with caplog.at_level(logging.INFO):
        r_df = nm.get_location_frame_reach_info(loc_df)
        assert "resetting reachID from location frame" in caplog.messages[-1]
    pd.testing.assert_frame_equal(r_df, expected_df)
    # errors
    with pytest.raises(TypeError, match="loc_df must be a GeoDataFrame or Da"):
        nm.get_location_frame_reach_info(loc_df.index)
    with pytest.raises(ValueError, match="loc_df must have 'segnum' column"):
        nm.get_location_frame_reach_info(loc_df[["geometry", "method"]])
    with pytest.raises(ValueError, match="downstream_bias must be between -1"):
        nm.get_location_frame_reach_info(loc_df, 8)
    with pytest.raises(ValueError, match="downstream_bias must be between -1"):
        nm.get_location_frame_reach_info(loc_df, -2)


def test_get_location_frame_reach_info_coastal(caplog, coastal_swn, coastal_points):
    n = coastal_swn
    m = flopy.modflow.Modflow.load(
        "h.nam", version="mfnwt", model_ws=datadir, check=False
    )
    nm = swn.SwnModflow.from_swn_flopy(n, m)
    # edits to make better examples
    coastal_points = coastal_points.copy()
    coastal_points.loc[9] = Point(1811234, 5874806)
    loc_gdf = n.locate_geoms(coastal_points)
    r_df = nm.get_location_frame_reach_info(loc_gdf)
    np.testing.assert_array_almost_equal(
        loc_gdf["dist_to_seg"], r_df.pop("dist_to_reach")
    )
    expected_df = pd.DataFrame(
        index=pd.Series([1, 2, 3, 5, 6, 7, 8, 9, 4, 10], name="id"),
        data={
            "reachID": [268, 284, 117, 158, 168, 178, 286, 294, 214, 268],
            "k": [0] * 10,
            "i": [8, 3, 6, 9, 13, 6, 2, 4, 1, 8],
            "j": [12, 10, 10, 9, 10, 15, 10, 9, 11, 12],
            "iseg": [166, 179, 73, 99, 107, 109, 180, 184, 129, 166],
            "ireach": [1, 1, 1, 1, 1, 2, 1, 3, 1, 1],
        },
    )
    pd.testing.assert_frame_equal(r_df, expected_df)
    # df2 = pd.concat([loc_gdf, r_df], axis=1)
    # assert not pd.isnull(df2).any().any()

    # With DataFrame
    loc_df = pd.DataFrame(loc_gdf.drop(columns="geometry"))
    r_df = nm.get_location_frame_reach_info(loc_df)
    assert "dist_to_reach" not in r_df.columns
    pd.testing.assert_frame_equal(r_df, expected_df)

    # With optional geom_loc_df
    r_df = nm.get_location_frame_reach_info(loc_df, geom_loc_df=coastal_points)
    np.testing.assert_array_almost_equal(
        loc_df["dist_to_seg"], r_df.pop("dist_to_reach")
    )
    pd.testing.assert_frame_equal(r_df, expected_df)

    # With DataFrame and optional geom_loc_df

    # Change downstream_bias
    # expect no change with near zero
    r_df = nm.get_location_frame_reach_info(loc_gdf, downstream_bias=0.01)
    assert list(r_df.reachID) == list(expected_df.reachID)
    diff = (loc_gdf["dist_to_seg"] - r_df["dist_to_reach"]).abs()
    np.testing.assert_allclose(diff.max(), 5e-10, rtol=0.5)
    r_df = nm.get_location_frame_reach_info(loc_gdf, downstream_bias=-0.01)
    assert list(r_df.reachID) == list(expected_df.reachID)
    # expect minor changes
    r_df = nm.get_location_frame_reach_info(loc_gdf, downstream_bias=0.1)
    assert list(r_df.reachID) == [269, 285, 117, 158, 168, 178, 286, 294, 214, 268]
    diff = (loc_gdf["dist_to_seg"] - r_df["dist_to_reach"]).abs()
    np.testing.assert_allclose(diff.max(), 0.94459247)
    # expect more changes
    r_df = nm.get_location_frame_reach_info(loc_gdf, downstream_bias=0.5)
    assert list(r_df.reachID) == [269, 285, 117, 158, 168, 178, 286, 295, 214, 268]
    diff = (loc_gdf["dist_to_seg"] - r_df["dist_to_reach"]).abs()
    np.testing.assert_allclose(diff.max(), 98.10718)
    # negative upstream match bias
    r_df = nm.get_location_frame_reach_info(loc_gdf, downstream_bias=-0.5)
    assert list(r_df.reachID) == [268, 284, 117, 158, 168, 177, 286, 294, 214, 268]
    diff = (loc_gdf["dist_to_seg"] - r_df["dist_to_reach"]).abs()
    np.testing.assert_allclose(diff.max(), 38.063676)

    # Test with non-zero downstream_bias and optional geom_loc_df
    r_df = nm.get_location_frame_reach_info(
        loc_df, downstream_bias=-0.5, geom_loc_df=coastal_points
    )
    assert list(r_df.reachID) == [268, 284, 117, 158, 168, 177, 286, 294, 214, 268]
    diff = (loc_df["dist_to_seg"] - r_df["dist_to_reach"]).abs()
    np.testing.assert_allclose(diff.max(), 38.063676)

    # Add a points outside model
    ext_points = pd.concat(
        [
            coastal_points.loc[8:],
            geopandas.GeoSeries(
                [Point(1810531, 5869152), Point(1806822.5, 5869173.5)],
                index=pd.Index([11, 12], name="id"),
            ),
        ]
    )
    loc_df = n.locate_geoms(ext_points)
    assert list(loc_df.index) == [8, 9, 4, 10, 11, 12]
    with caplog.at_level(logging.WARNING):
        r_df = nm.get_location_frame_reach_info(loc_df)
        assert "location id missing 2 matches: 11, 12" in caplog.messages[-1]
    assert list(r_df.index) == [8, 9, 4, 10]
    pd.testing.assert_frame_equal(
        r_df.drop(columns="dist_to_reach"), expected_df.loc[8:]
    )
