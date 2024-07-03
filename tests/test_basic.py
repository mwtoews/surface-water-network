import logging
from textwrap import dedent

import geopandas
import geopandas.testing
import numpy as np
import pandas as pd
import pytest
from shapely import wkt
from shapely.geometry import LineString, Point

import swn
from swn.compat import GEOPANDAS_GE_100, ignore_shapely_warnings_for_object_array
from swn.spatial import force_2d, round_coords

from .conftest import matplotlib, plt

# a few static objects (not fixtures)

valid_lines_list = [
    "LINESTRING Z (60 100 14, 60  80 12)",
    "LINESTRING Z (40 130 15, 60 100 14)",
    "LINESTRING Z (70 130 15, 60 100 14)",
]

valid_lines = geopandas.GeoSeries.from_wkt(valid_lines_list)

valid_df = pd.DataFrame(valid_lines.values, columns=["geometry"])

valid_polygons = geopandas.GeoSeries.from_wkt(
    [
        "POLYGON ((35 100, 75 100, 75  80, 35  80, 35 100))",
        "POLYGON ((35 135, 60 135, 60 100, 35 100, 35 135))",
        "POLYGON ((60 135, 75 135, 75 100, 60 100, 60 135))",
    ]
)


@pytest.fixture
def valid_n():
    return swn.SurfaceWaterNetwork.from_lines(valid_lines)


def test_init_errors():
    with pytest.raises(ValueError, match="segments must be a GeoDataFrame"):
        swn.SurfaceWaterNetwork(object())
    with pytest.raises(ValueError, match="segments must be a GeoDataFrame"):
        swn.SurfaceWaterNetwork(valid_df)


def test_from_lines_errors():
    with pytest.raises(ValueError, match="lines must be a GeoSeries"):
        swn.SurfaceWaterNetwork.from_lines(object())
    with pytest.raises(ValueError, match="lines must be a GeoSeries"):
        swn.SurfaceWaterNetwork.from_lines(valid_df)
    with pytest.raises(ValueError, match="one or more lines are required"):
        swn.SurfaceWaterNetwork.from_lines(valid_lines[0:0])


def test_init_geom_type():
    wkt_list = valid_lines_list[:]
    wkt_list[1] = "MULTILINESTRING Z ((70 130 15, 60 100 14))"
    lines = geopandas.GeoSeries.from_wkt(wkt_list)
    with pytest.raises(ValueError, match="lines must all be LineString types"):
        swn.SurfaceWaterNetwork.from_lines(lines)


def test_init_defaults(valid_n):
    n = valid_n
    assert n.logger is not None
    assert len(n.warnings) == 0
    assert len(n.errors) == 0
    assert len(n) == 3
    assert n.catchments is None
    assert n.has_z is True
    assert n.END_SEGNUM == -1
    assert list(n.segments.index) == [0, 1, 2]
    assert list(n.segments["to_segnum"]) == [-1, 0, 0]
    assert list(n.segments["from_segnums"]) == [{1, 2}, set(), set()]
    assert list(n.segments["cat_group"]) == [0, 0, 0]
    assert list(n.segments["num_to_outlet"]) == [1, 2, 2]
    np.testing.assert_allclose(
        n.segments["dist_to_outlet"], [20.0, 56.05551, 51.622776]
    )
    assert list(n.segments["sequence"]) == [3, 1, 2]
    assert list(n.segments["stream_order"]) == [2, 1, 1]
    np.testing.assert_allclose(
        n.segments["upstream_length"], [87.67828936, 36.05551275, 31.6227766]
    )
    assert "upstream_area" not in n.segments.columns
    assert "width" not in n.segments.columns
    assert n.headwater.to_list() == [1, 2]
    assert n.outlets.to_list() == [0]
    assert n.to_segnums.to_dict() == {1: 0, 2: 0}
    assert n.from_segnums.to_dict() == {0: {1, 2}}
    # more pedantic checks
    pd.testing.assert_index_equal(n.headwater, pd.Index([1, 2]))
    pd.testing.assert_index_equal(n.outlets, pd.Index([0]))
    pd.testing.assert_series_equal(
        n.to_segnums, pd.Series({1: 0, 2: 0}, name="to_segnum")
    )
    pd.testing.assert_series_equal(
        n.from_segnums, pd.Series({0: {1, 2}}, name="from_segnums")
    )
    n.adjust_elevation_profile()
    assert len(n.messages) == 0
    assert str(n) == repr(n)
    assert repr(n) == dedent(
        """\
        <SurfaceWaterNetwork: with Z coordinates
          3 segments: [0, 1, 2]
          2 headwater: [1, 2]
          1 outlets: [0]
          no diversions />"""
    )
    if matplotlib:
        _ = n.plot()
        plt.close()


def test_segments(valid_n):
    assert valid_n.segments is valid_n._segments
    assert isinstance(valid_n.segments, geopandas.GeoDataFrame)
    # columns are checked in other tests
    with pytest.raises(
        AttributeError, match="can't set attribute|object has no setter"
    ):
        valid_n.segments = None


def test_init_2D_geom():
    lines = force_2d(valid_lines)
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    assert len(n.warnings) == 0
    assert len(n.errors) == 0
    assert len(n) == 3
    assert n.has_z is False
    assert list(n.segments.index) == [0, 1, 2]
    assert list(n.segments["to_segnum"]) == [-1, 0, 0]
    assert list(n.segments["cat_group"]) == [0, 0, 0]
    assert list(n.segments["num_to_outlet"]) == [1, 2, 2]
    np.testing.assert_allclose(
        n.segments["dist_to_outlet"], [20.0, 56.05551, 51.622776]
    )
    assert list(n.segments["sequence"]) == [3, 1, 2]
    assert list(n.segments["stream_order"]) == [2, 1, 1]
    np.testing.assert_allclose(
        n.segments["upstream_length"], [87.67828936, 36.05551275, 31.6227766]
    )
    assert n.headwater.to_list() == [1, 2]
    assert n.outlets.to_list() == [0]
    assert repr(n) == dedent(
        """\
        <SurfaceWaterNetwork:
          3 segments: [0, 1, 2]
          2 headwater: [1, 2]
          1 outlets: [0]
          no diversions />"""
    )
    if matplotlib:
        _ = n.plot()
        plt.close()


def test_init_mismatch_3D():
    # Match in 2D, but not in Z dimension
    lines = geopandas.GeoSeries.from_wkt(
        [
            "LINESTRING Z (70 130 15, 60 100 14)",
            "LINESTRING Z (60 100 14, 60  80 12)",
            "LINESTRING Z (40 130 15, 60 100 13)",
        ]
    )
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    assert len(n.warnings) == 1
    assert (
        n.warnings[0]
        == "end of segment 2 matches start of segment 1 in 2D, but not in Z dimension"
    )
    assert len(n.errors) == 0
    assert len(n) == 3
    assert n.has_z is True
    assert list(n.segments.index) == [0, 1, 2]
    assert repr(n) == dedent(
        """\
        <SurfaceWaterNetwork: with Z coordinates
          3 segments: [0, 1, 2]
          2 headwater: [0, 2]
          1 outlets: [1]
          no diversions />"""
    )
    if matplotlib:
        _ = n.plot()
        plt.close()


def test_init_reversed_lines():
    # same as the working lines, but reversed in the opposite direction
    lines = valid_lines.geometry.apply(lambda x: LineString(x.coords[::-1]))
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    assert len(n.warnings) == 0
    assert len(n.errors) == 2
    assert n.errors[0] == "segment 0 has more than one downstream segments: [1, 2]"
    assert n.errors[1] == "starting coordinate (60.0, 100.0) matches start segment: {1}"
    assert len(n) == 3
    assert n.has_z is True
    assert list(n.segments.index) == [0, 1, 2]
    # This is all non-sense
    assert list(n.segments["to_segnum"]) == [1, -1, -1]
    assert list(n.segments["cat_group"]) == [1, 1, 2]
    assert list(n.segments["num_to_outlet"]) == [2, 1, 1]
    np.testing.assert_allclose(
        n.segments["dist_to_outlet"], [56.05551, 36.05551, 31.622776]
    )
    assert list(n.segments["sequence"]) == [1, 3, 2]
    assert list(n.segments["stream_order"]) == [1, 1, 1]
    np.testing.assert_allclose(
        n.segments["upstream_length"], [20.0, 56.05551275, 31.6227766]
    )
    assert n.headwater.to_list() == [0, 2]
    assert n.outlets.to_list() == [1, 2]
    assert n.to_segnums.to_dict() == {0: 1}
    assert n.from_segnums.to_dict() == {1: {0}}
    assert repr(n) == dedent(
        """\
        <SurfaceWaterNetwork: with Z coordinates
          3 segments: [0, 1, 2]
          2 headwater: [0, 2]
          2 outlets: [1, 2]
          no diversions />"""
    )
    if matplotlib:
        _ = n.plot()
        plt.close()


def test_init_all_converge():
    # Lines all converge to the same place
    lines = geopandas.GeoSeries.from_wkt(
        [
            "LINESTRING Z (40 130 15, 60 100 15)",
            "LINESTRING Z (70 130 14, 60 100 14)",
            "LINESTRING Z (60  80 12, 60 100 14)",
        ]
    )
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    assert len(n.warnings) == 1
    # Note: ending segment 0 matches end of segment 1 in 2D,
    # but not in Z dimension
    assert (
        n.warnings[0]
        == "ending coordinate (60.0, 100.0) matches end segments: {0, 1, 2}"
    )
    assert len(n.errors) == 0
    assert len(n) == 3
    assert n.has_z is True
    assert list(n.segments.index) == [0, 1, 2]
    assert list(n.segments["to_segnum"]) == [-1, -1, -1]
    assert list(n.segments["cat_group"]) == [0, 1, 2]
    assert list(n.segments["num_to_outlet"]) == [1, 1, 1]
    np.testing.assert_allclose(
        n.segments["dist_to_outlet"], [36.05551, 31.622776, 20.0]
    )
    assert list(n.segments["sequence"]) == [1, 2, 3]
    assert list(n.segments["stream_order"]) == [1, 1, 1]
    np.testing.assert_allclose(
        n.segments["upstream_length"], [36.05551, 31.622776, 20.0]
    )
    assert n.headwater.to_list() == [0, 1, 2]
    assert n.outlets.to_list() == [0, 1, 2]
    assert n.to_segnums.to_dict() == {}
    assert n.from_segnums.to_dict() == {}
    assert repr(n) == dedent(
        """\
        <SurfaceWaterNetwork: with Z coordinates
          3 segments: [0, 1, 2]
          3 headwater: [0, 1, 2]
          3 outlets: [0, 1, 2]
          no diversions />"""
    )
    if matplotlib:
        _ = n.plot()
        plt.close()


def test_init_all_diverge():
    # Lines all diverge from the same place
    lines = geopandas.GeoSeries.from_wkt(
        [
            "LINESTRING Z (60 100 15, 40 130 14)",
            "LINESTRING Z (60 100 16, 70 130 14)",
            "LINESTRING Z (60 100 15, 60  80 12)",
        ]
    )
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    assert len(n.warnings) == 0
    # Note: starting segment 0 matches start of segment 1 in 2D,
    # but not in Z dimension
    assert len(n.errors) == 1
    assert (
        n.errors[0]
        == "starting coordinate (60.0, 100.0) matches start segments: {0, 1, 2}"
    )
    assert len(n) == 3
    assert n.has_z is True
    assert list(n.segments.index) == [0, 1, 2]
    assert list(n.segments["to_segnum"]) == [-1, -1, -1]
    assert list(n.segments["num_to_outlet"]) == [1, 1, 1]
    assert list(n.segments["cat_group"]) == [0, 1, 2]
    np.testing.assert_allclose(
        n.segments["dist_to_outlet"], [36.05551, 31.622776, 20.0]
    )
    assert list(n.segments["sequence"]) == [1, 2, 3]
    assert list(n.segments["stream_order"]) == [1, 1, 1]
    np.testing.assert_allclose(
        n.segments["upstream_length"], [36.05551, 31.622776, 20.0]
    )
    assert n.headwater.to_list() == [0, 1, 2]
    assert n.outlets.to_list() == [0, 1, 2]
    assert n.to_segnums.to_dict() == {}
    assert n.from_segnums.to_dict() == {}
    assert repr(n) == dedent(
        """\
        <SurfaceWaterNetwork: with Z coordinates
          3 segments: [0, 1, 2]
          3 headwater: [0, 1, 2]
          3 outlets: [0, 1, 2]
          no diversions />"""
    )
    if matplotlib:
        _ = n.plot()
        plt.close()


def test_init_line_connects_to_middle():
    lines = geopandas.GeoSeries.from_wkt(
        [
            "LINESTRING Z (40 130 15, 60 100 14, 60 80 12)",
            "LINESTRING Z (70 130 15, 60 100 14)",
        ]
    )
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    assert len(n.warnings) == 0
    assert len(n.errors) == 1
    assert n.errors[0] == "segment 1 connects to the middle of segment 0"
    assert len(n) == 2
    assert n.has_z is True
    assert list(n.segments.index) == [0, 1]
    assert list(n.segments["to_segnum"]) == [-1, -1]
    assert list(n.segments["cat_group"]) == [0, 1]
    assert list(n.segments["num_to_outlet"]) == [1, 1]
    np.testing.assert_allclose(n.segments["dist_to_outlet"], [56.05551, 31.622776])
    assert list(n.segments["sequence"]) == [1, 2]
    assert list(n.segments["stream_order"]) == [1, 1]
    np.testing.assert_allclose(n.segments["upstream_length"], [56.05551, 31.622776])
    assert n.headwater.to_list() == [0, 1]
    assert n.outlets.to_list() == [0, 1]
    assert n.to_segnums.to_dict() == {}
    assert n.from_segnums.to_dict() == {}
    assert repr(n) == dedent(
        """\
        <SurfaceWaterNetwork: with Z coordinates
          2 segments: [0, 1]
          2 headwater: [0, 1]
          2 outlets: [0, 1]
          no diversions />"""
    )
    if matplotlib:
        _ = n.plot()
        plt.close()


def test_to_segnums(valid_n):
    # check series in propery method
    pd.testing.assert_series_equal(
        valid_n.to_segnums, pd.Series([0, 0], name="to_segnum", index=pd.Index([1, 2]))
    )
    # check series in segments frame
    pd.testing.assert_series_equal(
        valid_n.segments["to_segnum"], pd.Series([-1, 0, 0], name="to_segnum")
    )

    # Rebuild network using named index
    lines = valid_lines.copy()
    lines.index.name = "idx"
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    pd.testing.assert_series_equal(
        n.to_segnums,
        pd.Series([0, 0], name="to_segnum", index=pd.Index([1, 2], name="idx")),
    )
    pd.testing.assert_series_equal(
        n.segments["to_segnum"],
        pd.Series([-1, 0, 0], name="to_segnum", index=pd.RangeIndex(0, 3, name="idx")),
    )


def test_from_segnums(valid_n):
    # check series in propery method
    pd.testing.assert_series_equal(
        valid_n.from_segnums, pd.Series([{1, 2}], name="from_segnums")
    )
    # check series in segments frame
    pd.testing.assert_series_equal(
        valid_n.segments["from_segnums"],
        pd.Series([{1, 2}, set(), set()], name="from_segnums"),
    )

    # Rebuild network using named index
    lines = valid_lines.copy()
    lines.index.name = "idx"
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    pd.testing.assert_series_equal(
        n.from_segnums,
        pd.Series([{1, 2}], name="from_segnums", index=pd.Index([0], name="idx")),
    )
    pd.testing.assert_series_equal(
        n.segments["from_segnums"],
        pd.Series(
            [{1, 2}, set(), set()],
            name="from_segnums",
            index=pd.RangeIndex(0, 3, name="idx"),
        ),
    )


def test_dict(valid_n):
    # via __iter__
    d = dict(valid_n)
    assert list(d.keys()) == (
        ["class", "segments", "END_SEGNUM", "catchments", "diversions"]
    )
    assert d["class"] == "SurfaceWaterNetwork"
    assert isinstance(d["segments"], geopandas.GeoDataFrame)
    assert list(d["segments"].index) == [0, 1, 2]
    assert list(d["segments"].columns) == [
        "geometry",
        "to_segnum",
        "from_segnums",
        "cat_group",
        "num_to_outlet",
        "dist_to_outlet",
        "sequence",
        "stream_order",
        "upstream_length",
    ]
    assert d["END_SEGNUM"] == -1
    assert d["catchments"] is None
    assert d["diversions"] is None


def test_copy_lines(valid_n):
    n1 = valid_n
    n2 = swn.SurfaceWaterNetwork(valid_n.segments, valid_n.END_SEGNUM)
    assert n1 is not n2


def test_copy_lines_polygons(valid_n):
    n1 = valid_n
    n2 = swn.SurfaceWaterNetwork(valid_n.segments, valid_n.END_SEGNUM)
    assert n1 is not n2


def test_eq_lines(valid_n):
    n1 = valid_n
    n2 = swn.SurfaceWaterNetwork(valid_n.segments, valid_n.END_SEGNUM)
    assert len(n1) == len(n2) == 3
    assert n1 == n2


def test_ne_lines(valid_n):
    n1 = valid_n
    n2 = swn.SurfaceWaterNetwork(valid_n.segments, valid_n.END_SEGNUM)
    n2.remove(segnums=[1])
    assert len(n1) != len(n2)
    assert n1 != n2


def test_init_geoseries():
    gs = geopandas.GeoSeries.from_wkt(valid_lines_list, name="foo")
    n = swn.SurfaceWaterNetwork.from_lines(gs)
    assert len(n.warnings) == 0
    assert len(n.errors) == 0
    assert len(n) == 3
    assert list(n.segments.index) == [0, 1, 2]
    assert n.has_z is True
    v = pd.Series([3.0, 2.0, 4.0])
    a = n.accumulate_values(v)
    pd.testing.assert_series_equal(a, pd.Series({0: 9.0, 1: 2.0, 2: 4.0}))


def test_init_segments_loc():
    lines = geopandas.GeoSeries.from_wkt(
        [
            "LINESTRING (60 100, 60  80)",
            "LINESTRING (40 130, 60 100)",
            "LINESTRING (70 130, 60 100)",
            "LINESTRING (60  80, 70  70)",
        ]
    )
    lines.index += 100
    n1 = swn.SurfaceWaterNetwork.from_lines(lines)
    assert list(n1.outlets) == [103]
    # Create by slicing another GeoDataFrame, but check to_segnums
    n2 = swn.SurfaceWaterNetwork(n1.segments.loc[100:102])
    assert len(n2.segments) == 3
    assert list(n2.outlets) == [100]
    assert n2.to_segnums.to_dict() == {101: 100, 102: 100}


def test_accumulate_values_must_be_series(valid_n):
    with pytest.raises(ValueError, match="values must be a pandas Series"):
        valid_n.accumulate_values([3.0, 2.0, 4.0])


def test_accumulate_values_different_index(valid_n):
    # indexes don't completely overlap
    v = pd.Series([3.0, 2.0, 4.0])
    v.index += 1
    with pytest.raises(ValueError, match="index is different"):
        valid_n.accumulate_values(v)
    # indexes overlap, but have a different sequence
    v = pd.Series([3.0, 2.0, 4.0]).sort_values()
    with pytest.raises(ValueError, match="index is different"):
        valid_n.accumulate_values(v)


def test_accumulate_values_expected(valid_n):
    v = pd.Series([2.0, 3.0, 4.0])
    a = valid_n.accumulate_values(v)
    pd.testing.assert_series_equal(a, pd.Series({0: 9.0, 1: 3.0, 2: 4.0}))
    v = pd.Series([3.0, 2.0, 4.0], name="vals")
    a = valid_n.accumulate_values(v)
    pd.testing.assert_series_equal(
        a, pd.Series({0: 9.0, 1: 2.0, 2: 4.0}, name="accumulated_vals")
    )


def test_init_polygons():
    expected_area = [800.0, 875.0, 525.0]
    expected_upstream_area = [2200.0, 875.0, 525.0]
    expected_upstream_width = [1.4615, 1.4457, 1.4397]
    # from GeoSeries
    n = swn.SurfaceWaterNetwork.from_lines(valid_lines, valid_polygons)
    assert n.catchments is not None
    np.testing.assert_array_almost_equal(n.catchments.area, expected_area)
    np.testing.assert_array_almost_equal(
        n.segments["upstream_area"], expected_upstream_area
    )
    np.testing.assert_array_almost_equal(
        n.segments["width"], expected_upstream_width, 4
    )
    # from GeoDataFrame
    n = swn.SurfaceWaterNetwork.from_lines(valid_lines, valid_polygons)
    assert n.catchments is not None
    np.testing.assert_array_almost_equal(n.catchments.area, expected_area)
    np.testing.assert_array_almost_equal(
        n.segments["upstream_area"], expected_upstream_area
    )
    np.testing.assert_array_almost_equal(
        n.segments["width"], expected_upstream_width, 4
    )
    # manual upstream area calculation
    np.testing.assert_array_almost_equal(
        n.accumulate_values(n.catchments.area), expected_upstream_area
    )
    assert repr(n) == dedent(
        """\
        <SurfaceWaterNetwork: with Z coordinates and catchment polygons
          3 segments: [0, 1, 2]
          2 headwater: [1, 2]
          1 outlets: [0]
          no diversions />"""
    )
    if matplotlib:
        _ = n.plot()
        plt.close()
    # check error
    with pytest.raises(ValueError, match="polygons must be a GeoSeries or None"):
        swn.SurfaceWaterNetwork.from_lines(valid_lines, 1.0)


def test_catchments_property():
    # also vary previous test with 2D lines
    n = swn.SurfaceWaterNetwork.from_lines(force_2d(valid_lines))
    assert n.catchments is None
    n.catchments = valid_polygons
    assert n.catchments is not None
    np.testing.assert_array_almost_equal(n.catchments.area, [800.0, 875.0, 525.0])
    np.testing.assert_array_almost_equal(
        n.segments["upstream_area"], [2200.0, 875.0, 525.0]
    )
    np.testing.assert_array_almost_equal(
        n.segments["width"], [1.4615, 1.4457, 1.4397], 4
    )
    assert repr(n) == dedent(
        """\
        <SurfaceWaterNetwork: with catchment polygons
          3 segments: [0, 1, 2]
          2 headwater: [1, 2]
          1 outlets: [0]
          no diversions />"""
    )
    if matplotlib:
        _ = n.plot()
        plt.close()
    # unset property
    n.catchments = None
    assert n.catchments is None
    # check errors
    with pytest.raises(ValueError, match="catchments must be a GeoSeries or None"):
        n.catchments = 1.0
    with pytest.raises(
        ValueError, match=r"catchments\.index is different than for segments"
    ):
        n.catchments = valid_polygons.iloc[1:]


def test_set_diversions_geodataframe():
    n = swn.SurfaceWaterNetwork.from_lines(valid_lines)
    diversions = geopandas.GeoDataFrame(
        geometry=[Point(58, 97), Point(62, 97), Point(61, 89), Point(59, 89)]
    )
    # check errors
    with pytest.raises(AttributeError, match=r"use \'set_diversions\(\)\' method"):
        n.diversions = diversions
    with pytest.raises(ValueError, match=r"a \[Geo\]DataFrame is expected"):
        n.set_diversions([1])
    with pytest.raises(ValueError, match=r"a \[Geo\]DataFrame is expected"):
        n.set_diversions(diversions.geometry)
    with pytest.raises(ValueError, match="does not appear to be spatial"):
        n.set_diversions(geopandas.GeoDataFrame([1]))
    # normal operation
    assert n.diversions is None
    # use -ve downstream_bias to replicte previous behaviour
    n.set_diversions(diversions, downstream_bias=-0.5)
    assert n.diversions is not None
    expected = geopandas.GeoDataFrame(
        {
            "method": ["nearest"] * 4,
            "from_segnum": [1, 2, 0, 0],
            "seg_ndist": [1.0, 1.0, 0.55, 0.55],
            "dist_to_seg": [3.605551, 3.605551, 1.0, 1.0],
        },
        geometry=diversions.geometry,
        index=diversions.index,
    )
    assert set(n.diversions.columns) == set(expected.columns)
    pd.testing.assert_frame_equal(n.diversions[expected.columns], expected)
    np.testing.assert_array_equal(n.segments["diversions"], [{2, 3}, {0}, {1}])
    # defalut parameters
    n.set_diversions(diversions)
    expected = geopandas.GeoDataFrame(
        {
            "method": ["nearest"] * 4,
            "from_segnum": [0, 0, 0, 0],
            "seg_ndist": [0.15, 0.15, 0.55, 0.55],
            "dist_to_seg": [2.0, 2.0, 1.0, 1.0],
        },
        geometry=diversions.geometry,
        index=diversions.index,
    )
    pd.testing.assert_frame_equal(n.diversions[expected.columns], expected)
    # Unset
    n.set_diversions(None)
    assert n.diversions is None
    assert "diversions" not in n.segments.columns
    # Try again with min_stream_order option
    n.set_diversions(diversions, min_stream_order=2)
    pd.testing.assert_frame_equal(n.diversions[expected.columns], expected)
    np.testing.assert_array_equal(
        n.segments["diversions"], [{0, 1, 2, 3}, set(), set()]
    )
    # Try again, but use 'from_segnum' column
    diversions = geopandas.GeoDataFrame(
        {"from_segnum": [0, 2]}, geometry=[Point(55, 97), Point(68, 105)]
    )
    n.set_diversions(diversions)
    expected = geopandas.GeoDataFrame(
        {
            "from_segnum": [0, 2],
            "seg_ndist": [0.15, 0.77],
            "dist_to_seg": [5.0, 6.008328],
        },
        geometry=diversions.geometry,
        index=diversions.index,
    )
    pd.testing.assert_frame_equal(n.diversions[expected.columns], expected)
    np.testing.assert_array_equal(n.segments["diversions"], [{0}, set(), {1}])
    assert repr(n) == dedent(
        """\
        <SurfaceWaterNetwork: with Z coordinates
          3 segments: [0, 1, 2]
          2 headwater: [1, 2]
          1 outlets: [0]
          2 diversions (as GeoDataFrame): [0, 1] />"""
    )
    if matplotlib:
        _ = n.plot()
        plt.close()


def test_set_diversions_dataframe():
    n = swn.SurfaceWaterNetwork.from_lines(valid_lines)
    diversions = pd.DataFrame({"from_segnum": [0, 2]})
    # check errors
    with pytest.raises(AttributeError, match=r"use \'set_diversions\(\)\' method"):
        n.diversions = diversions
    with pytest.raises(ValueError, match=r"a \[Geo\]DataFrame is expected"):
        n.set_diversions(diversions.from_segnum)
    with pytest.raises(ValueError, match="does not appear to be spatial"):
        n.set_diversions(pd.DataFrame([1]))
    # normal operation
    assert n.diversions is None
    n.set_diversions(diversions)
    assert n.diversions is not None
    expected = pd.DataFrame(
        {
            "from_segnum": [0, 2],
            "seg_ndist": [1.0] * 2,
        }
    )
    assert set(n.diversions.columns) == set(expected.columns)
    pd.testing.assert_frame_equal(n.diversions[expected.columns], expected)
    np.testing.assert_array_equal(n.segments["diversions"], [{0}, set(), {1}])
    assert repr(n) == dedent(
        """\
        <SurfaceWaterNetwork: with Z coordinates
          3 segments: [0, 1, 2]
          2 headwater: [1, 2]
          1 outlets: [0]
          2 diversions (as DataFrame): [0, 1] />"""
    )
    if matplotlib:
        _ = n.plot()
        plt.close()
    # Unset
    n.set_diversions(None)
    assert n.diversions is None
    assert "diversions" not in n.segments.columns
    # add column to dataframe that should be preserved
    diversions["seg_ndist"] = [0.7, 0.3]
    expected["seg_ndist"] = [0.7, 0.3]
    n.set_diversions(diversions)
    pd.testing.assert_frame_equal(n.diversions[expected.columns], expected)
    if matplotlib:
        _ = n.plot()
        plt.close()


def test_estimate_width():
    n = swn.SurfaceWaterNetwork.from_lines(valid_lines, valid_polygons)
    # defaults
    np.testing.assert_array_almost_equal(
        n.segments["width"], [1.4615, 1.4457, 1.4397], 4
    )
    # float a and b
    n.estimate_width(1.4, 0.6)
    np.testing.assert_array_almost_equal(
        n.segments["width"], [1.4254, 1.4146, 1.4108], 4
    )
    # integer a and b
    n.estimate_width(1, 1)
    np.testing.assert_array_almost_equal(
        n.segments["width"], [1.0022, 1.0009, 1.0005], 4
    )
    # a and b are Series per segment
    n.estimate_width([1.2, 1.8, 1.4], [0.4, 0.7, 0.6])
    np.testing.assert_array_almost_equal(n.segments["width"], [1.2006, 1.8, 1.4], 4)
    # based on Series
    n.estimate_width(upstream_area=n.segments["upstream_area"] / 2)
    np.testing.assert_array_almost_equal(
        n.segments["width"], [1.4489, 1.4379, 1.4337], 4
    )
    # defaults (again)
    n.estimate_width()
    np.testing.assert_array_almost_equal(
        n.segments["width"], [1.4615, 1.4457, 1.4397], 4
    )
    # based on a column, where upstream_area is not available
    n2 = swn.SurfaceWaterNetwork.from_lines(valid_lines)
    assert "width" not in n2.segments.columns
    assert "upstream_area" not in n2.segments.columns
    n2.segments["foo_upstream"] = n2.segments["upstream_length"] * 25
    n2.estimate_width(upstream_area="foo_upstream")
    np.testing.assert_array_almost_equal(
        n2.segments["width"], [1.4614, 1.4461, 1.4444], 4
    )
    # check errors
    with pytest.raises(ValueError, match="unknown use for upstream_area"):
        n.estimate_width(upstream_area=3)
    with pytest.raises(
        ValueError, match=r"'upstream_area' not found in segments\.columns"
    ):
        n2.estimate_width()


def test_segments_series(valid_n):
    n = valid_n
    errmsg = "index is different than for segments"
    # from scalar
    pd.testing.assert_series_equal(n.segments_series(8.0), pd.Series([8.0] * 3))
    pd.testing.assert_series_equal(
        n.segments_series(8, name="eight"), pd.Series([8, 8, 8], name="eight")
    )
    pd.testing.assert_series_equal(n.segments_series("$VAL$"), pd.Series(["$VAL$"] * 3))
    # from list
    pd.testing.assert_series_equal(n.segments_series([3, 4, 5]), pd.Series([3, 4, 5]))
    pd.testing.assert_series_equal(
        n.segments_series([3, 4, 5], name="list"), pd.Series([3, 4, 5], name="list")
    )
    pd.testing.assert_series_equal(
        n.segments_series(["$VAL1$", "$VAL2$", "$VAL3$"]),
        pd.Series(["$VAL1$", "$VAL2$", "$VAL3$"]),
    )
    with pytest.raises(ValueError, match=errmsg):
        n.segments_series([8])
    with pytest.raises(ValueError, match=errmsg):
        n.segments_series([3, 4, 5, 6])
    # from dict
    pd.testing.assert_series_equal(
        n.segments_series({0: 1.1, 1: 2.2, 2: 3.3}), pd.Series([1.1, 2.2, 3.3])
    )
    with pytest.raises(ValueError, match=errmsg):
        n.segments_series({0: 1.1, 1: 2.2, 3: 3.3})
    # from Series
    s = pd.Series([2.0, 3.0, 4.0])
    pd.testing.assert_series_equal(n.segments_series(s), s)
    s = pd.Series([2.0, 3.0, 4.0], name="foo")
    pd.testing.assert_series_equal(n.segments_series(s), s)
    # now break it
    s.index += 1
    with pytest.raises(ValueError, match=errmsg):
        n.segments_series(s)
    # misc error
    with pytest.raises(ValueError, match="expected value to be scalar, list,"):
        n.segments_series(object())


def test_pair_segments_frame(valid_n):
    n = valid_n
    errmsg_value = "index is different than for segments"
    errmsg_value_out = "value_out.index is not a subset of segments.index"
    errmsg_value_out_expected = "expected value_out to be scalar, dict or Seri"
    # from scalar
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(8.0, method="continuous"),
        pd.DataFrame({1: 8.0, 2: 8.0}, index=[0, 1, 2]),
    )
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(8.0, name="foo"),
        pd.DataFrame({"foo1": 8.0, "foo2": 8.0}, index=[0, 1, 2]),
    )
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(8, 9), pd.DataFrame({1: 8, 2: [9, 8, 8]})
    )
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(8, 9, name="foo"),
        pd.DataFrame({"foo1": 8, "foo2": [9, 8, 8]}),
    )
    # from list
    pd.testing.assert_frame_equal(
        n.pair_segments_frame([3, 4, 5]), pd.DataFrame({1: [3, 4, 5], 2: 3})
    )
    pd.testing.assert_frame_equal(
        n.pair_segments_frame([3, 4, 5], name="foo"),
        pd.DataFrame({"foo1": [3, 4, 5], "foo2": 3}),
    )
    pd.testing.assert_frame_equal(
        n.pair_segments_frame([3, 4, 5], {0: 6}),
        pd.DataFrame({1: [3, 4, 5], 2: [6, 3, 3]}),
    )
    pd.testing.assert_frame_equal(
        n.pair_segments_frame([3, 4, 5], 6, "foo"),
        pd.DataFrame({"foo1": [3, 4, 5], "foo2": [6, 3, 3]}),
    )
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(["$VAL1$", "$VAL2$", "$VAL3$"]),
        pd.DataFrame({1: ["$VAL1$", "$VAL2$", "$VAL3$"], 2: "$VAL1$"}),
    )
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(["v1", "v2", "v3"], "o1"),
        pd.DataFrame({1: ["v1", "v2", "v3"], 2: ["o1", "v1", "v1"]}),
    )
    with pytest.raises(ValueError, match=errmsg_value):
        n.pair_segments_frame([3, 4])
    with pytest.raises(ValueError, match=errmsg_value_out_expected):
        n.pair_segments_frame([3, 4, 5], [6, 7])
    # from Series
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(pd.Series([3, 4, 5])), pd.DataFrame({1: [3, 4, 5], 2: 3})
    )
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(pd.Series([3, 4, 5]), name="foo"),
        pd.DataFrame({"foo1": [3, 4, 5], "foo2": 3}),
    )
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(pd.Series([3, 4, 5]), pd.Series(6)),
        pd.DataFrame({1: [3, 4, 5], 2: [6, 3, 3]}),
    )
    # mixed with dict
    pd.testing.assert_frame_equal(
        n.pair_segments_frame([3, 4, 5], {0: 6}),
        pd.DataFrame({1: [3, 4, 5], 2: [6, 3, 3]}),
    )
    with pytest.raises(ValueError, match=errmsg_value_out):
        n.pair_segments_frame(1, {3: 2})
    # discontinuous series
    pd.testing.assert_frame_equal(
        n.pair_segments_frame([3, 4, 5], {0: 6, 1: 7, 2: 8}),
        pd.DataFrame({1: [3, 4, 5], 2: [6, 7, 8]}),
    )
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(1, {0: 2, 2: 3}),
        pd.DataFrame({1: [1, 1, 1], 2: [2, 1, 3]}),
    )
    pd.testing.assert_frame_equal(
        n.pair_segments_frame([3.0, 4.0, 5.0], {1: 7, 2: 8}),
        pd.DataFrame({1: [3.0, 4.0, 5.0], 2: [3.0, 7.0, 8.0]}),
    )
    with pytest.raises(ValueError, match=errmsg_value_out):
        n.pair_segments_frame(1, {0: 6, 3: 8})
    # constant
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(1, method="constant"),
        pd.DataFrame({1: [1, 1, 1], 2: [1, 1, 1]}),
    )
    pd.testing.assert_frame_equal(
        n.pair_segments_frame([3, 2, 1], method="constant"),
        pd.DataFrame({1: [3, 2, 1], 2: [3, 2, 1]}),
    )
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(1, 2, name="x", method="constant"),
        pd.DataFrame({"x1": [1, 1, 1], "x2": [2, 1, 1]}),
    )
    # additive
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(1, method="additive"),
        pd.DataFrame({1: [1.0, 1.0, 1.0], 2: [1.0, 0.5, 0.5]}),
    )
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(1, {0: 2, 1: 10}, name="foo", method="additive"),
        pd.DataFrame({"foo1": [1.0, 1.0, 1.0], "foo2": [2.0, 10.0, 0.5]}),
    )
    pd.testing.assert_frame_equal(
        n.pair_segments_frame([10.0, 2.0, 3.0], name="foo", method="additive"),
        pd.DataFrame({"foo1": [10.0, 2.0, 3.0], "foo2": [10.0, 4.0, 6.0]}),
    )
    # misc errors
    with pytest.raises(ValueError, match="method must be one of "):
        n.pair_segments_frame(1, method="nope")
    with pytest.raises(ValueError, match="expected value to be scalar, list,"):
        n.pair_segments_frame(object())
    with pytest.raises(ValueError, match=errmsg_value_out_expected):
        n.pair_segments_frame(1, object())


def test_remove_condition():
    n = swn.SurfaceWaterNetwork.from_lines(valid_lines, valid_polygons)
    assert len(n) == 3
    # manually copy these, to compare later
    n.segments["orig_upstream_length"] = n.segments["upstream_length"]
    n.segments["orig_upstream_area"] = n.segments["upstream_area"]
    n.segments["orig_width"] = n.segments["width"]
    n.segments["orig_stream_order"] = n.segments["stream_order"]
    n.remove(n.segments["upstream_area"] <= 1000.0)
    assert len(n) == 1
    assert len(n.segments) == 1
    assert len(n.catchments) == 1
    assert list(n.segments.index) == [0]
    assert n.segments.at[0, "from_segnums"] == {1, 2}
    np.testing.assert_almost_equal(n.segments.at[0, "upstream_length"], 87.67828936)
    np.testing.assert_almost_equal(n.segments.at[0, "upstream_area"], 2200.0)
    np.testing.assert_almost_equal(n.segments.at[0, "width"], 1.4615, 4)
    # Manually re-trigger these
    n.evaluate_upstream_length()
    n.evaluate_upstream_area()
    n.estimate_width()
    np.testing.assert_almost_equal(n.segments.at[0, "upstream_length"], 20.0)
    np.testing.assert_almost_equal(n.segments.at[0, "upstream_area"], 800.0)
    np.testing.assert_almost_equal(n.segments.at[0, "width"], 1.4445, 4)
    np.testing.assert_almost_equal(
        n.segments.at[0, "orig_upstream_length"], 87.67828936
    )
    np.testing.assert_almost_equal(n.segments.at[0, "orig_upstream_area"], 2200.0)
    np.testing.assert_almost_equal(n.segments.at[0, "orig_width"], 1.4615, 4)
    # Stays the same
    pd.testing.assert_series_equal(
        n.segments["stream_order"],
        n.segments["orig_stream_order"],
        check_names=False,
    )


def test_remove_segnums():
    n = swn.SurfaceWaterNetwork.from_lines(valid_lines)
    assert len(n) == 3
    n.remove(segnums=[1])
    assert len(n) == 2
    assert len(n.segments) == 2
    assert list(n.segments.index) == [0, 2]
    assert n.segments.at[0, "from_segnums"] == {1, 2}
    # repeats are ok
    n = swn.SurfaceWaterNetwork.from_lines(valid_lines)
    n.remove(segnums=[1, 2, 1])
    assert len(n) == 1
    assert len(n.segments) == 1
    assert list(n.segments.index) == [0]
    assert n.segments.at[0, "from_segnums"] == {1, 2}


def test_remove_errors(valid_n):
    n = valid_n
    assert len(n) == 3
    n.remove()  # no segments selected to remove; no changes made
    assert len(n) == 3
    n.remove(n.segments["dist_to_outlet"] > 100.0)  # dito, none selected
    assert len(n) == 3
    with pytest.raises(
        IndexError, match=r"1 segnums not found in segments\.index: \[3\]"
    ):
        n.remove(segnums=[3])
    with pytest.raises(
        ValueError, match="all segments were selected to remove; must keep at least "
    ):
        n.remove(segnums=[0, 1, 2])


# https://commons.wikimedia.org/wiki/File:Flussordnung_(Strahler).svg
fluss_gs = geopandas.GeoSeries(
    wkt.loads(
        """\
MULTILINESTRING(
    (380 490, 370 420), (300 460, 370 420), (370 420, 420 330),
    (190 250, 280 270), (225 180, 280 270), (280 270, 420 330),
    (420 330, 584 250), (520 220, 584 250), (584 250, 710 160),
    (740 270, 710 160), (735 350, 740 270), (880 320, 740 270),
    (925 370, 880 320), (974 300, 880 320), (760 460, 735 350),
    (650 430, 735 350), (710 160, 770 100), (700  90, 770 100),
    (770 100, 820  40))
"""
    ).geoms
)


@pytest.fixture
def fluss_n():
    return swn.SurfaceWaterNetwork.from_lines(fluss_gs)


def test_fluss_n(fluss_n):
    n = fluss_n
    assert len(n.warnings) == 0
    assert len(n.errors) == 0
    assert len(n) == 19
    assert list(n.segments.index) == list(range(19))
    assert list(n.segments["to_segnum"]) == (
        [2, 2, 6, 5, 5, 6, 8, 8, 16, 16, 9, 9, 11, 11, 10, 10, 18, 18, -1]
    )
    assert list(n.segments["cat_group"]) == [18] * 19
    assert list(n.segments["num_to_outlet"]) == (
        [6, 6, 5, 6, 6, 5, 4, 4, 3, 3, 4, 4, 5, 5, 5, 5, 2, 2, 1]
    )
    assert list(n.segments["sequence"]) == (
        [4, 3, 12, 2, 1, 11, 15, 9, 16, 17, 14, 13, 6, 5, 8, 7, 18, 10, 19]
    )
    assert list(n.segments["stream_order"]) == (
        [1, 1, 2, 1, 1, 2, 3, 1, 3, 3, 2, 2, 1, 1, 1, 1, 4, 1, 4]
    )
    assert n.headwater.to_list() == [0, 1, 3, 4, 7, 12, 13, 14, 15, 17]
    assert n.outlets.to_list() == [18]
    assert n.to_segnums.to_dict() == {
        0: 2,
        1: 2,
        2: 6,
        3: 5,
        4: 5,
        5: 6,
        6: 8,
        7: 8,
        8: 16,
        9: 16,
        10: 9,
        11: 9,
        12: 11,
        13: 11,
        14: 10,
        15: 10,
        16: 18,
        17: 18,
    }

    assert n.from_segnums.to_dict() == {
        16: {8, 9},
        2: {0, 1},
        5: {3, 4},
        6: {2, 5},
        8: {6, 7},
        9: {10, 11},
        10: {14, 15},
        11: {12, 13},
        18: {16, 17},
    }
    assert repr(n) == dedent(
        """\
        <SurfaceWaterNetwork:
          19 segments: [0, 1, ..., 17, 18]
          10 headwater: [0, 1, ..., 15, 17]
          1 outlets: [18]
          no diversions />"""
    )
    if matplotlib:
        _ = n.plot()
        plt.close()


def test_route_segnums(fluss_n):
    n = fluss_n
    assert n.route_segnums(18, 18) == [18]
    assert n.route_segnums(17, 18) == [17, 18]
    assert n.route_segnums(18, 17) == [18, 17]
    assert n.route_segnums(6, 16) == [6, 8, 16]
    assert n.route_segnums(16, 6) == [16, 8, 6]
    assert n.route_segnums(15, 9) == [15, 10, 9]
    assert n.route_segnums(9, 15) == [9, 10, 15]
    assert n.route_segnums(0, 1, allow_indirect=True) == [0, 1]
    assert n.route_segnums(0, 11, allow_indirect=True) == [0, 2, 6, 8, 9, 11]
    # errors
    with pytest.raises(IndexError, match="invalid start segnum -1"):
        n.route_segnums(-1, 0)
    with pytest.raises(IndexError, match="invalid end segnum -1"):
        n.route_segnums(0, -1)
    with pytest.raises(ConnectionError, match="0 does not connect to 1"):
        n.route_segnums(0, 1)
    with pytest.raises(ConnectionError, match="3 does not connect to 9"):
        n.route_segnums(3, 9)
    n2 = swn.SurfaceWaterNetwork.from_lines(n.segments.geometry.iloc[0:4])
    with pytest.raises(ConnectionError, match="segment networks are disjoint"):
        n2.route_segnums(0, 3)
    with pytest.raises(ConnectionError, match="segment networks are disjoint"):
        n2.route_segnums(0, 3, allow_indirect=True)


def test_query_deprecated(fluss_n):
    n = fluss_n
    with pytest.deprecated_call():
        assert set(n.query(upstream=[2])) == {0, 1, 2}
    with pytest.deprecated_call():
        assert n.query(downstream=0) == [2, 6, 8, 16, 18]


def test_fluss_n_gather_segnums_upstream(fluss_n):
    n = fluss_n
    assert set(n.gather_segnums(upstream=0)) == {0}
    assert set(n.gather_segnums(upstream=[2])) == {0, 1, 2}
    assert set(n.gather_segnums(upstream=8)) == {0, 1, 2, 3, 4, 5, 6, 7, 8}
    assert set(n.gather_segnums(upstream=9)) == {9, 10, 11, 12, 13, 14, 15}
    assert set(n.gather_segnums(upstream=17)) == {17}
    assert len(set(n.gather_segnums(upstream=18))) == 19
    # with barriers
    assert len(set(n.gather_segnums(upstream=18, barrier=17))) == 18
    assert len(set(n.gather_segnums(upstream=18, barrier=9))) == 13
    assert set(n.gather_segnums(upstream=9, barrier=8)) == {9, 10, 11, 12, 13, 14, 15}
    assert set(n.gather_segnums(upstream=16, barrier=[9, 5])) == (
        {0, 1, 2, 5, 6, 7, 8, 9, 16}
    )
    # break it
    with pytest.raises(
        IndexError, match=r"upstream segnum \-1 not found in segments\.index"
    ):
        n.gather_segnums(upstream=-1)
    with pytest.raises(
        IndexError, match=r"2 upstream segments not found in segments\.index: \[19, "
    ):
        n.gather_segnums(upstream=[18, 19, 20])
    with pytest.raises(
        IndexError, match=r"barrier segnum \-1 not found in segments\.index"
    ):
        n.gather_segnums(upstream=18, barrier=-1)
    with pytest.raises(
        IndexError, match=r"1 barrier segment not found in segments\.index: \[\-1\]"
    ):
        n.gather_segnums(upstream=18, barrier=[-1, 15])


def test_fluss_n_gather_segnums_downstream(fluss_n):
    n = fluss_n
    assert n.gather_segnums(downstream=0) == [2, 6, 8, 16, 18]
    assert n.gather_segnums(downstream=[2]) == [6, 8, 16, 18]
    assert n.gather_segnums(downstream=8) == [16, 18]
    assert n.gather_segnums(downstream=9) == [16, 18]
    assert n.gather_segnums(downstream=17) == [18]
    assert n.gather_segnums(downstream=18) == []
    assert set(n.gather_segnums(downstream=7, gather_upstream=True)) == (
        {0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}
    )
    assert set(n.gather_segnums(downstream=8, gather_upstream=True)) == (
        {9, 10, 11, 12, 13, 14, 15, 16, 17, 18}
    )
    assert set(n.gather_segnums(downstream=[9], gather_upstream=True)) == (
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18}
    )
    assert n.gather_segnums(downstream=18, gather_upstream=True) == []
    assert set(n.gather_segnums(downstream=0, gather_upstream=True, barrier=8)) == (
        {1, 2, 3, 4, 5, 6, 8}
    )
    with pytest.raises(
        IndexError, match=r"downstream segnum \-1 not found in segments\.index"
    ):
        n.gather_segnums(downstream=-1)


def test_locate_geoms_in_basic_swn(caplog):
    ls = geopandas.GeoSeries.from_wkt(
        [
            "LINESTRING (60 100, 60  80)",
            "LINESTRING (40 130, 60 100)",
            "LINESTRING (80 130, 60 100)",
        ]
    )
    ls.index += 101
    n = swn.SurfaceWaterNetwork.from_lines(ls)
    gs = geopandas.GeoSeries.from_wkt(
        [
            "POINT (60 200)",
            "POINT (61 200)",
            "POINT (60 100)",
            "LINESTRING (50 110, 70 110)",
            "POLYGON ((58 102, 63 102, 60 94, 58 102))",
            "POINT EMPTY",
            "POINT (60 90)",
        ]
    )
    gs.index += 11
    r = n.locate_geoms(gs)
    e = pd.DataFrame(
        {
            "method": "nearest",
            "segnum": [102, 103, 101, 102, 101, 0, 101],
            "seg_ndist": [0.0, 0.0, 0.0, 2 / 3.0, 0.0, np.nan, 0.5],
            "dist_to_seg": [72.80109889, 72.53275122, 0.0, 0.0, 0.0, np.nan, 0.0],
        },
        index=[11, 12, 13, 14, 15, 16, 17],
    )
    e.at[16, "method"] = "empty"
    # check everything except the empty geometry
    r2 = r.drop(index=16)
    e2 = e.drop(index=16)
    pd.testing.assert_frame_equal(r2[e2.columns], e2)
    pd.testing.assert_series_equal(r2.dist_to_seg, r2.length, check_names=False)
    assert list(r.geom_type.unique()) == ["LineString"]
    a = r2.geometry.apply(lambda x: Point(*x.coords[0]))
    assert (a.distance(gs.drop(index=16)) == 0.0).all()
    b = r2.geometry.apply(lambda x: Point(*x.coords[-1]))
    seg_geoms = n.segments.geometry[r2.segnum]
    if GEOPANDAS_GE_100:
        seg_mls = seg_geoms.union_all()
    else:
        seg_mls = seg_geoms.unary_union
    assert (b.distance(seg_mls) < 1e-10).all()
    # now check the empty geometry
    for k in e.keys():
        rv = r.at[16, k]
        re = e.at[16, k]
        assert (rv == re) or (pd.isna(rv) and pd.isna(re))
    assert r.at[16, "geometry"].is_empty
    # check downstream_bias parameter
    assert list(n.locate_geoms(gs, downstream_bias=0.01).segnum) == (
        [102, 103, 101, 102, 101, 0, 101]
    )
    assert list(n.locate_geoms(gs, downstream_bias=-0.01).segnum) == (
        [102, 103, 102, 102, 101, 0, 101]
    )
    assert list(n.locate_geoms(gs, downstream_bias=0.5).segnum) == (
        [102, 103, 101, 102, 101, 0, 101]
    )
    assert list(n.locate_geoms(gs, downstream_bias=-0.5).segnum) == (
        [102, 103, 102, 102, 102, 0, 101]
    )
    assert list(n.locate_geoms(gs, downstream_bias=1).segnum) == (
        [102, 103, 101, 101, 101, 0, 101]
    )
    assert list(n.locate_geoms(gs, downstream_bias=-1).segnum) == (
        [102, 103, 102, 102, 102, 0, 101]
    )
    # check override parameter
    r = n.locate_geoms(gs, override={11: 103, 12: 103})
    r2 = r.drop(index=16)
    e2 = e.drop(index=16)
    e2.loc[11, "segnum"] = 103
    e2.loc[[11, 12], "method"] = "override"
    pd.testing.assert_frame_equal(r2[e2.columns], e2)
    with caplog.at_level(logging.WARNING):
        r = n.locate_geoms(gs, override={1: 103})
        assert "1 override key" in caplog.messages[-1]
    with caplog.at_level(logging.WARNING):
        r = n.locate_geoms(gs, override={11: 13})
        assert "1 override value" in caplog.messages[-1]
    # check special value None
    r = n.locate_geoms(gs, override={11: None})
    assert r.at[11, "geometry"].is_empty
    assert r.at[11, "method"] == "override"
    assert r.at[11, "segnum"] == n.END_SEGNUM
    assert pd.isna(r.at[11, "seg_ndist"])
    assert pd.isna(r.at[11, "dist_to_seg"])
    pd.testing.assert_frame_equal(
        r[e.columns].drop(index=[11, 16]), e.drop(index=[11, 16])
    )


@pytest.fixture
def coastal_geom():
    xy = [
        (1811503.1, 5874071.2),
        (1806234.3, 5869114.8),
        (1804222, 5870087),
        (1802814, 5867160),
    ]
    df = pd.DataFrame(xy, columns=["x", "y"])
    gs = geopandas.GeoSeries(geopandas.points_from_xy(df.x, df.y))
    with ignore_shapely_warnings_for_object_array():
        gs[len(gs)] = wkt.loads(
            """POLYGON ((1812532 5872498, 1812428 5872361, 1812561 5872390,
                         1812532 5872498))"""
        )
        gs[len(gs)] = wkt.loads(
            """POLYGON ((1814560 5875222, 1814655 5875215, 1814580 5875114,
                         1814560 5875222))"""
        )
        gs[len(gs)] = wkt.loads(
            """POLYGON ((1812200 5876599, 1812213 5876629, 1812219 5876606,
                         1812200 5876599))"""
        )
    gs.index += 101
    gs.set_crs("EPSG:2193", inplace=True)
    return gs


def test_locate_geoms_only_lines(coastal_geom, coastal_swn):
    r = coastal_swn.locate_geoms(coastal_geom)
    e = pd.DataFrame(
        {
            "method": "nearest",
            "segnum": [3047364, 3048663, 3048249, 3049113, 3047736, 3047145, 3046745],
            "seg_ndist": [0.5954064, 0.0974058, 0.279147, 0.0, 0.684387, 0.0, 0.541026],
            "dist_to_seg": [
                80.25,
                315.519943,
                364.7475,
                586.13982,
                203.144242,
                76.995938,
                13.84302,
            ],
        },
        index=[101, 102, 103, 104, 105, 106, 107],
    )
    pd.testing.assert_frame_equal(r[e.columns], e)
    pd.testing.assert_series_equal(r.dist_to_seg, r.length, check_names=False)
    assert list(r.geom_type.unique()) == ["LineString"]
    assert (r.geometry.apply(lambda g: len(g.coords)) == 2).all()
    a = r.geometry.interpolate(0.0)
    b = r.geometry.interpolate(1.0, normalized=True)
    seg_geoms = coastal_swn.segments.geometry[r.segnum]
    if GEOPANDAS_GE_100:
        seg_mls = seg_geoms.union_all()
    else:
        seg_mls = seg_geoms.unary_union
    assert (a.distance(coastal_geom) < 1e-10).all()
    assert (a.distance(seg_mls) > 0.0).all()
    assert (b.distance(coastal_geom) > 0.0).all()
    assert (b.distance(seg_mls) < 1e-10).all()
    r1 = coastal_swn.locate_geoms(coastal_geom, min_stream_order=1)
    pd.testing.assert_frame_equal(r[e.columns], e)
    assert list(r1.segnum) == (
        [3047364, 3048663, 3048249, 3049113, 3047736, 3047145, 3046745]
    )
    r2 = coastal_swn.locate_geoms(coastal_geom, min_stream_order=2)
    assert list(r2.segnum) == (
        [3046952, 3048504, 3048552, 3049106, 3047365, 3047040, 3046745]
    )
    r2 = coastal_swn.locate_geoms(coastal_geom, min_stream_order=2)
    assert list(r2.segnum) == (
        [3046952, 3048504, 3048552, 3049106, 3047365, 3047040, 3046745]
    )
    r6 = coastal_swn.locate_geoms(coastal_geom, min_stream_order=6)
    assert list(r6.segnum) == (
        [3046952, 3047683, 3047683, 3047683, 3046952, 3046736, 3046736]
    )


def test_locate_geoms_with_catchments(coastal_geom, coastal_swn_w_poly):
    r = coastal_swn_w_poly.locate_geoms(coastal_geom)
    e = pd.DataFrame(
        {
            "method": [
                "catchment",
                "catchment",
                "nearest",
                "nearest",
                "catchment",
                "catchment",
                "catchment",
            ],
            "segnum": [3046952, 3048504, 3048249, 3049113, 3047737, 3047145, 3046745],
            "seg_ndist": [
                0.3356222,
                0.087803396575,
                0.279147,
                0.0,
                0.798942725,
                0.0,
                0.541026,
            ],
            "dist_to_seg": [
                169.243199,
                496.662756,
                364.7475,
                586.1398,
                247.3825,
                76.995938,
                13.84302,
            ],
        },
        index=[101, 102, 103, 104, 105, 106, 107],
    )
    pd.testing.assert_frame_equal(r[e.columns], e)
    pd.testing.assert_series_equal(r.dist_to_seg, r.length, check_names=False)
    assert list(r.geom_type.unique()) == ["LineString"]
    assert (r.geometry.apply(lambda g: len(g.coords)) == 2).all()
    a = r.geometry.interpolate(0.0)
    b = r.geometry.interpolate(1.0, normalized=True)
    seg_geoms = coastal_swn_w_poly.segments.geometry[r.segnum]
    if GEOPANDAS_GE_100:
        seg_mls = seg_geoms.union_all()
    else:
        seg_mls = seg_geoms.unary_union
    assert (a.distance(coastal_geom) < 1e-10).all()
    assert (a.distance(seg_mls) > 0.0).all()
    assert (b.distance(coastal_geom) > 0.0).all()
    assert (b.distance(seg_mls) < 1e-10).all()
    r1 = coastal_swn_w_poly.locate_geoms(coastal_geom, min_stream_order=1)
    pd.testing.assert_frame_equal(r[e.columns], e)
    assert list(r1.segnum) == (
        [3046952, 3048504, 3048249, 3049113, 3047737, 3047145, 3046745]
    )
    r2 = coastal_swn_w_poly.locate_geoms(coastal_geom, min_stream_order=2)
    assert list(r2.segnum) == (
        [3046952, 3048504, 3048552, 3049106, 3047365, 3047040, 3046745]
    )
    r2 = coastal_swn_w_poly.locate_geoms(coastal_geom, min_stream_order=2)
    assert list(r2.segnum) == (
        [3046952, 3048504, 3048552, 3049106, 3047365, 3047040, 3046745]
    )
    r6 = coastal_swn_w_poly.locate_geoms(coastal_geom, min_stream_order=6)
    assert list(r6.segnum) == (
        [3046736, 3046736, 3047683, 3047683, 3046737, 3046737, 3046736]
    )


def test_aggregate_fluss_headwater(fluss_n):
    n = fluss_n
    assert len(n) == 19
    na = n.aggregate([17, 4])
    assert len(na.warnings) == 0
    assert len(na.errors) == 0
    assert len(na) == 2
    assert list(na.segments.index) == [17, 4]
    assert list(na.headwater) == [17, 4]
    assert list(na.outlets) == [17, 4]
    assert list(na.segments["agg_patch"]) == [[17], [4]]
    assert list(na.segments["agg_path"]) == [[17], [4]]
    assert list(na.segments["agg_unpath"]) == [[], []]


def test_aggregate_fluss_headwater_and_middle(fluss_n):
    n = fluss_n
    assert len(n) == 19
    na = n.aggregate([17, 18])
    assert len(na.warnings) == 0
    assert len(na.errors) == 0
    assert len(na) == 3
    assert list(na.segments.index) == [17, 18, 16]
    assert list(na.headwater) == [17, 16]
    assert list(na.outlets) == [18]
    assert [set(x) for x in na.segments["agg_patch"]] == [
        {17},
        {18},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    ]
    assert list(na.segments["agg_path"]) == [[17], [18], [16, 8, 6, 5, 4]]
    assert list(na.segments["agg_unpath"]) == [[], [16, 17], [9, 7, 2, 3]]


def test_aggregate_fluss_two_middle(fluss_n):
    n = fluss_n
    assert len(n) == 19
    na = n.aggregate([8, 9])
    assert len(na.warnings) == 1
    assert len(na.errors) == 0
    assert len(na) == 2
    assert list(na.segments.index) == [8, 9]
    assert list(na.headwater) == [8, 9]
    assert list(na.outlets) == [8, 9]
    assert [set(x) for x in na.segments["agg_patch"]] == [
        {0, 1, 2, 3, 4, 5, 6, 7, 8},
        {9, 10, 11, 12, 13, 14, 15},
    ]
    assert list(na.segments["agg_path"]) == [[8, 6, 5, 4], [9, 11, 13]]
    assert list(na.segments["agg_unpath"]) == [[7, 2, 3], [10, 12]]


def test_aggregate_fluss_disconnected(fluss_n):
    n = fluss_n
    assert len(n) == 19
    na = n.aggregate([5, 10, 17])
    assert len(na.warnings) == 0
    assert len(na.errors) == 0
    assert len(na) == 3
    assert list(na.segments.index) == [5, 10, 17]
    assert list(na.headwater) == [5, 10, 17]
    assert list(na.outlets) == [5, 10, 17]
    assert [set(x) for x in na.segments["agg_patch"]] == [{3, 4, 5}, {10, 14, 15}, {17}]
    assert list(na.segments["agg_path"]) == [[5, 4], [10, 15], [17]]
    assert list(na.segments["agg_unpath"]) == [[3], [14], []]


def test_aggregate_fluss_coarse(fluss_n):
    n = fluss_n
    assert len(n) == 19
    na = n.aggregate([5, 10, 18])
    assert len(na.warnings) == 0
    assert len(na.errors) == 0
    # extra junctions need to be added
    assert len(na) == 7
    assert list(na.segments.index) == [5, 10, 18, 8, 2, 9, 11]
    assert list(na.headwater) == [5, 10, 2, 11]
    assert list(na.outlets) == [18]
    assert [set(x) for x in na.segments["agg_patch"]] == [
        {3, 4, 5},
        {10, 14, 15},
        {16, 17, 18},
        {6, 7, 8},
        {0, 1, 2},
        {9},
        {11, 12, 13},
    ]
    assert list(na.segments["agg_path"]) == (
        [[5, 4], [10, 15], [18, 16], [8, 6], [2, 1], [9], [11, 13]]
    )
    assert list(na.segments["agg_unpath"]) == (
        [[3], [14], [17, 8, 9], [7, 2, 5], [0], [10, 11], [12]]
    )


def test_coarsen_fluss(fluss_n):
    # level 1 gets back the same network, but with 'traced_segnums'
    nc1 = fluss_n.coarsen(1)
    assert len(nc1) == 19
    assert nc1.catchments is None
    pd.testing.assert_frame_equal(
        fluss_n.segments, nc1.segments.drop(columns="traced_segnums")
    )
    pd.testing.assert_series_equal(
        nc1.segments["traced_segnums"],
        nc1.segments.index.to_series().apply(lambda x: [x]),
        check_names=False,
    )

    # coarsen to level 2
    nc2 = fluss_n.coarsen(2)
    expected_nc2 = geopandas.GeoDataFrame(
        {
            "to_segnum": [8, 8, 18, 18, 9, 9, 0],
            "from_segnums": [set(), set(), {2, 5}, {10, 11}, set(), set(), {8, 9}],
            "stream_order": [2, 2, 3, 3, 2, 2, 4],
            "traced_segnums": [[2], [5], [6, 8], [9], [10], [11], [16, 18]],
        },
        geometry=geopandas.GeoSeries.from_wkt(
            [
                "LINESTRING (370 420, 420 330)",
                "LINESTRING (280 270, 420 330)",
                "LINESTRING (420 330, 584 250, 710 160)",
                "LINESTRING (740 270, 710 160)",
                "LINESTRING (735 350, 740 270)",
                "LINESTRING (880 320, 740 270)",
                "LINESTRING (710 160, 770 100, 820 40)",
            ],
        ),
    ).set_index(pd.Index([2, 5, 8, 9, 10, 11, 18]))
    cols = ["geometry", "to_segnum", "from_segnums", "stream_order", "traced_segnums"]
    geopandas.testing.assert_geodataframe_equal(nc2.segments[cols], expected_nc2[cols])

    # coarsen to level 3
    nc3 = fluss_n.coarsen(3)
    expected_nc3 = geopandas.GeoDataFrame(
        {
            "to_segnum": [18, 18, 0],
            "from_segnums": [set(), set(), {8, 9}],
            "stream_order": [3, 3, 4],
            "traced_segnums": [[6, 8], [9], [16, 18]],
        },
        geometry=geopandas.GeoSeries.from_wkt(
            [
                "LINESTRING (420 330, 584 250, 710 160)",
                "LINESTRING (740 270, 710 160)",
                "LINESTRING (710 160, 770 100, 820 40)",
            ],
        ),
    ).set_index(pd.Index([8, 9, 18]))
    geopandas.testing.assert_geodataframe_equal(nc3.segments[cols], expected_nc3[cols])

    # coarsen to level 4
    nc4 = fluss_n.coarsen(4)
    expected_nc4 = geopandas.GeoDataFrame(
        {
            "to_segnum": [0],
            "from_segnums": [set()],
            "stream_order": [4],
            "traced_segnums": [[16, 18]],
        },
        geometry=geopandas.GeoSeries.from_wkt(
            [
                "LINESTRING (710 160, 770 100, 820 40)",
            ],
        ),
    ).set_index(pd.Index([18]))
    geopandas.testing.assert_geodataframe_equal(nc4.segments[cols], expected_nc4[cols])

    # can't coarsen to level 5
    with pytest.raises(ValueError, match="no segments found"):
        fluss_n.coarsen(5)


def test_adjust_elevation_profile_errors(valid_n):
    with pytest.raises(ValueError, match="min_slope must be greater than zero"):
        valid_n.adjust_elevation_profile(0.0)

    n2d = swn.SurfaceWaterNetwork.from_lines(force_2d(valid_lines))
    with pytest.raises(AttributeError, match="line geometry does not have Z dimension"):
        n2d.adjust_elevation_profile()

    min_slope = pd.Series(2.0 / 1000, index=valid_n.segments.index)
    min_slope[1] = 3.0 / 1000
    min_slope.index += 1
    with pytest.raises(ValueError, match="index is different"):
        valid_n.adjust_elevation_profile(min_slope)


def test_adjust_elevation_profile_min_slope_float(valid_n):
    valid_n.adjust_elevation_profile(2.0 / 1000)


def test_adjust_elevation_profile_min_slope_series(valid_n):
    min_slope = pd.Series(2.0 / 1000, index=valid_n.segments.index)
    min_slope[1] = 3.0 / 1000
    valid_n.adjust_elevation_profile(min_slope)


def test_adjust_elevation_profile_no_change():
    lines = geopandas.GeoSeries.from_wkt(["LINESTRING Z (0 0 8, 1 0 7, 2 0 6)"])
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    n.adjust_elevation_profile()
    assert len(n.messages) == 0
    assert (lines == n.segments.geometry).all()
    expected_profiles = geopandas.GeoSeries.from_wkt(["LINESTRING (0 8, 1 7, 2 6)"])
    assert (n.profiles == expected_profiles).all()


def test_adjust_elevation_profile_use_min_slope():
    lines = geopandas.GeoSeries.from_wkt(["LINESTRING Z (0 0 8, 1 0 9)"])
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    n.adjust_elevation_profile()
    n.profiles = round_coords(n.profiles)
    n.segments.geometry = round_coords(n.segments.geometry)
    assert len(n.messages) == 1
    assert n.messages[0] == "segment 0: adjusted 1 coordinate elevation by 1.001"
    expected = geopandas.GeoSeries.from_wkt(["LINESTRING Z (0 0 8, 1 0 7.999)"])
    assert (expected == n.segments.geometry).all()
    expected_profiles = geopandas.GeoSeries.from_wkt(["LINESTRING (0 8, 1 7.999)"])
    assert (n.profiles == expected_profiles).all()

    lines = geopandas.GeoSeries.from_wkt(["LINESTRING Z (0 0 8, 1 0 9, 2 0 6)"])
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    n.adjust_elevation_profile(0.1)
    assert len(n.messages) == 1
    assert n.messages[0] == "segment 0: adjusted 1 coordinate elevation by 1.100"
    expected = geopandas.GeoSeries.from_wkt(["LINESTRING Z (0 0 8, 1 0 7.9, 2 0 6)"])
    assert (expected == n.segments.geometry).all()
    expected_profiles = geopandas.GeoSeries.from_wkt(["LINESTRING (0 8, 1 7.9, 2 6)"])
    assert (n.profiles == expected_profiles).all()

    lines = geopandas.GeoSeries.from_wkt(["LINESTRING Z (0 0 8, 1 0 5, 2 0 6, 3 0 5)"])
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    n.adjust_elevation_profile(0.2)
    assert len(n.messages) == 1
    assert (
        n.messages[0]
        == "segment 0: adjusted 2 coordinate elevations between 0.400 and 1.200"
    )
    expected = geopandas.GeoSeries.from_wkt(
        ["LINESTRING Z (0 0 8, 1 0 5, 2 0 4.8, 3 0 4.6)"]
    )
    assert (expected == n.segments.geometry).all()
    expected_profiles = geopandas.GeoSeries.from_wkt(
        ["LINESTRING (0 8, 1 5, 2 4.8, 3 4.6)"]
    )
    assert (n.profiles == expected_profiles).all()
