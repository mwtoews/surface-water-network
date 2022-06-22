import geopandas
import numpy as np
import pandas as pd
import pytest
from shapely import wkt
from shapely.geometry import Point

import swn
import swn.spatial as spatial


def test_get_sindex():
    xy = geopandas.GeoDataFrame(geometry=geopandas.points_from_xy([0], [1]))
    # disable these tests due to difficulty in determinig if available or not
    # assert spatial.get_sindex(xy) is None
    # assert spatial.get_sindex(xy.geometry) is None
    xy = geopandas.GeoDataFrame(
        geometry=geopandas.points_from_xy(
            range(spatial.rtree_threshold), range(spatial.rtree_threshold)))
    if spatial.rtree:
        assert spatial.get_sindex(xy) is not None
        assert spatial.get_sindex(xy.geometry) is not None


def test_interp_2d_to_3d():
    grid = np.array([
        [18.0, 15.0],
        [14.0, 13.0],
        [12.0, 10.0],
    ])
    gt = (30.0, 20.0, 0.0, 130.0, 0.0, -20.0)
    # Test points within
    x = np.tile(np.arange(30, 80, 10), 7)
    y = np.repeat(np.arange(70, 140, 10)[::-1], 5)
    expected_z = [
        18.0, 18.0, 16.5, 15.0, 15.0,
        18.0, 18.0, 16.5, 15.0, 15.0,
        16.0, 16.0, 15.0, 14.0, 14.0,
        14.0, 14.0, 13.5, 13.0, 13.0,
        13.0, 13.0, 12.25, 11.5, 11.5,
        12.0, 12.0, 11.0, 10.0, 10.0,
        12.0, 12.0, 11.0, 10.0, 10.0]
    gs2d = geopandas.GeoSeries([Point(xy) for xy in zip(x, y)])
    gs3d = spatial.interp_2d_to_3d(gs2d, grid, gt)
    np.testing.assert_array_equal(gs3d.apply(lambda g: g.z), expected_z)
    # plt.imshow(np.array(expected_z).reshape((7, 5)))
    # plt.imshow(grid)
    # Test inside corners
    x_in = [30, 70, 70, 30]
    y_in = [130, 130, 70, 70]
    gs3d = spatial.interp_2d_to_3d(
            geopandas.GeoSeries([Point(xy) for xy in zip(x_in, y_in)]),
            grid, gt)
    np.testing.assert_array_equal(gs3d.apply(lambda g: g.z), [18, 15, 10, 12])
    # Points outside shoud raise an exception
    x_out = [29, 71, 71, 29]
    y_out = [131, 131, 69, 69]
    outside_combs = list(zip(x_out, y_out)) + list(zip(x_in, y_out)) + \
        list(zip(x_out, y_in))
    for pt in outside_combs:
        with pytest.raises(ValueError, match='coordinates are outside grid'):
            spatial.interp_2d_to_3d(geopandas.GeoSeries([Point(pt)]), grid, gt)


def test_get_crs():
    assert spatial.get_crs(None) is None
    assert spatial.get_crs(2193) is not None
    assert spatial.get_crs('EPSG:2193') is not None
    assert spatial.get_crs('+init=EPSG:2193') is not None
    assert spatial.get_crs({'init': 'EPSG:2193'}) is not None
    assert spatial.get_crs({'init': 'EPSG:2193'}) is not None
    assert spatial.get_crs(
        {'proj': 'tmerc', 'lat_0': 0, 'lon_0': 173, 'k': 0.9996,
         'x_0': 1600000, 'y_0': 10000000, 'ellps': 'GRS80', 'units': 'm',
         'no_defs': None, 'type': 'crs'}) is not None


def test_compare_crs():
    assert spatial.compare_crs(None, None) == (None, None, True)
    assert spatial.compare_crs(2193, None)[2] is False
    assert spatial.compare_crs(2193, 'EPSG:2193')[2] is True
    assert spatial.compare_crs(2193, {'init': 'EPSG:2193'})[2] is True


def test_bias_substring():
    line = spatial.wkt_to_geoseries(["LINESTRING (0 0, 10 0)"])
    pd.testing.assert_series_equal(
        spatial.bias_substring(line, 0, end_cut=0), line)
    pd.testing.assert_series_equal(
        spatial.bias_substring(line, -1, end_cut=0),
        spatial.wkt_to_geoseries(["POINT (10 0)"]))
    pd.testing.assert_series_equal(
        spatial.bias_substring(line, 1, end_cut=0),
        spatial.wkt_to_geoseries(["POINT (0 0)"]))
    pd.testing.assert_series_equal(
        spatial.bias_substring(line, 0.5, end_cut=0),
        spatial.wkt_to_geoseries(["LINESTRING (0 0, 5 0)"]))
    pd.testing.assert_series_equal(
        spatial.bias_substring(line, -0.2, end_cut=0),
        spatial.wkt_to_geoseries(["LINESTRING (2 0, 10 0)"]))
    pd.testing.assert_series_equal(
        spatial.bias_substring(line, 0, end_cut=0.1),
        spatial.wkt_to_geoseries(["LINESTRING (1 0, 9 0)"]))
    pd.testing.assert_series_equal(
        spatial.bias_substring(line, 0.5, end_cut=0.1),
        spatial.wkt_to_geoseries(["LINESTRING (1 0, 5 0)"]))
    pd.testing.assert_series_equal(
        spatial.bias_substring(line, -0.2, end_cut=0.1),
        spatial.wkt_to_geoseries(["LINESTRING (2 0, 9 0)"]))
    pd.testing.assert_series_equal(
        spatial.bias_substring(line, -1, end_cut=0.1),
        spatial.wkt_to_geoseries(["POINT (9 0)"]))
    pd.testing.assert_series_equal(
        spatial.bias_substring(line, 1, end_cut=0.1),
        spatial.wkt_to_geoseries(["POINT (1 0)"]))
    pd.testing.assert_series_equal(
        spatial.bias_substring(line, 0.2, end_cut=0.4),
        spatial.wkt_to_geoseries(["LINESTRING (4 0, 6 0)"]))
    pd.testing.assert_series_equal(
        spatial.bias_substring(line, -0.2, end_cut=0.4),
        spatial.wkt_to_geoseries(["LINESTRING (4 0, 6 0)"]))
    pd.testing.assert_series_equal(
        spatial.bias_substring(line, -0.2, end_cut=0.5),
        spatial.wkt_to_geoseries(["POINT (5 0)"]))
    # errors
    with pytest.raises(TypeError, match="xpected 'gs' as an instance of GeoS"):
        spatial.bias_substring(line.to_frame(), 0)
    with pytest.raises(ValueError, match="must be between -1 and 1"):
        spatial.bias_substring(line, 2)
    with pytest.raises(ValueError, match="must between 0 and 0.5"):
        spatial.bias_substring(line, 0, 1)


def test_bias_substring_confluence():
    lines = swn.spatial.wkt_to_geoseries([
        "LINESTRING (60 100, 60 80)",
        "LINESTRING (40 130, 60 100)",
        "LINESTRING (70 130, 60 100)"])
    pts = swn.spatial.wkt_to_geoseries([
        "POINT (58 97)", "POINT (62 97)", "POINT (59 89)"])

    def nearest_pt_idx(lines):
        ret = []
        for pt in pts:
            ret.append(lines.distance(pt).sort_values().index[0])
        return ret

    assert nearest_pt_idx(lines) == [0, 0, 0]
    assert nearest_pt_idx(spatial.bias_substring(lines, 1)) == [0, 0, 0]
    assert nearest_pt_idx(spatial.bias_substring(lines, -0.3)) == [0, 0, 0]
    assert nearest_pt_idx(spatial.bias_substring(lines, -0.4)) == [1, 2, 0]
    assert nearest_pt_idx(spatial.bias_substring(lines, -1)) == [1, 2, 0]
    assert nearest_pt_idx(spatial.bias_substring(lines, -0.4, end_cut=0)) == \
        [1, 1, 0]


def test_find_segnum_in_swn_only_lines(coastal_swn):
    """Test deprecated function."""
    n = coastal_swn
    # query tuple
    with pytest.deprecated_call():
        r = spatial.find_segnum_in_swn(n, (1811503.1, 5874071.2))
    assert len(r) == 1
    assert r.index[0] == 0
    assert r.segnum[0] == 3047364
    assert round(r.dist_to_segnum[0], 1) == 80.3
    # query list of tuples
    geom = [
        (1806234.3, 5869114.8),
        (1804222, 5870087),
        (1802814, 5867160)]
    with pytest.deprecated_call():
        r = spatial.find_segnum_in_swn(n, geom)
    assert len(r) == 3
    assert list(r.segnum) == [3048663, 3048249, 3049113]
    assert list(r.dist_to_segnum.round(1)) == [315.5, 364.7, 586.1]
    # query GeoSeries
    polygon = wkt.loads('''\
    POLYGON ((1815228 5869053, 1815267 5869021, 1815120 5868936,
              1815228 5869053))''')
    gs = geopandas.GeoSeries(
        [polygon, Point(1814271, 5869525)], index=[101, 102])
    with pytest.deprecated_call():
        r = spatial.find_segnum_in_swn(n, gs)
    assert len(r) == 2
    assert list(r.segnum) == [3048690, 3048372]
    assert list(r.dist_to_segnum.round(1)) == [73.4, 333.8]


def test_find_segnum_in_swn_with_catchments(coastal_swn_w_poly):
    """Test deprecated function."""
    n = coastal_swn_w_poly
    # query tuple
    with pytest.deprecated_call():
        r = spatial.find_segnum_in_swn(n, (1811503.1, 5874071.2))
    assert len(r) == 1
    assert r.index[0] == 0
    assert r.segnum[0] == 3046952
    assert round(r.dist_to_segnum[0], 1) == 169.2
    assert r.is_within_catchment[0]
    # query list of tuples
    geom = [
        (1806234.3, 5869114.8),
        (1804222, 5870087),
        (1802814, 5867160)]
    with pytest.deprecated_call():
        r = spatial.find_segnum_in_swn(n, geom)
    assert len(r) == 3
    assert list(r.segnum) == [3048504, 3048249, 3049113]
    assert list(r.dist_to_segnum.round(1)) == [496.7, 364.7, 586.1]
    assert list(r.is_within_catchment) == [True, False, False]
    # query GeoSeries
    polygon = wkt.loads('''\
    POLYGON ((1815228 5869053, 1815267 5869021, 1815120 5868936,
              1815228 5869053))''')
    gs = geopandas.GeoSeries(
        [polygon, Point(1814271, 5869525)], index=[101, 102])
    with pytest.deprecated_call():
        r = spatial.find_segnum_in_swn(n, gs)
    assert len(r) == 2
    assert list(r.segnum) == [3048690, 3048482]
    assert list(r.dist_to_segnum.round(1)) == [73.4, 989.2]
    assert list(r.is_within_catchment) == [True, True]


def test_find_location_pairs(coastal_points, coastal_swn):
    loc_df = coastal_swn.locate_geoms(coastal_points)
    pairs = spatial.find_location_pairs(loc_df, coastal_swn)
    assert pairs == {(1, 2), (2, 8), (3, 2), (5, 10), (6, 10), (7, 2), (10, 1)}
    # use a subset of loc_df, as a DataFrame
    sub_df = loc_df.loc[loc_df.dist_to_seg < 20, ["segnum", "seg_ndist"]]
    pairs = spatial.find_location_pairs(sub_df, coastal_swn)
    assert pairs == {(3, 8), (5, 8), (6, 8)}
    # errors
    with pytest.raises(TypeError, match="expected 'n' as an instance of Surf"):
        spatial.find_location_pairs(loc_df, False)
    with pytest.raises(TypeError, match="loc_df must be a GeoDataFrame or"):
        spatial.find_location_pairs(loc_df.segnum, coastal_swn)
    with pytest.raises(ValueError, match="loc_df must have 'segnum' column"):
        spatial.find_location_pairs(loc_df[["method"]], coastal_swn)
    with pytest.raises(ValueError, match="loc_df must have 'seg_ndist' colum"):
        spatial.find_location_pairs(loc_df[["segnum"]], coastal_swn)
    loc_df.segnum += 10
    with pytest.raises(ValueError, match="loc_df has segnum values not foun"):
        spatial.find_location_pairs(loc_df, coastal_swn)


def test_location_pair_geoms(coastal_points, coastal_swn):
    loc_df = coastal_swn.locate_geoms(coastal_points)
    pairs = spatial.find_location_pairs(loc_df, coastal_swn)
    geoms = spatial.location_pair_geoms(pairs, loc_df, coastal_swn)
    assert isinstance(geoms, dict)
    assert set(geoms.keys()) == pairs
    gs = geopandas.GeoSeries(geoms).sort_index()
    assert list(gs.geom_type.unique()) == ["LineString"]
    np.testing.assert_array_almost_equal(
        gs.length.values,
        [5930.74, 921.94, 3968.87, 3797.21, 6768.95, 8483.27, 279.65],
        decimal=2
    )
    # use a subset of loc_df, as a DataFrame
    sub_df = loc_df.loc[loc_df.dist_to_seg < 20, ["segnum", "seg_ndist"]]
    pairs = spatial.find_location_pairs(sub_df, coastal_swn)
    geoms = spatial.location_pair_geoms(pairs, loc_df, coastal_swn)
    assert set(geoms.keys()) == pairs
    gs = geopandas.GeoSeries(geoms).sort_index()
    assert list(gs.geom_type.unique()) == ["LineString"]
    np.testing.assert_array_almost_equal(
        gs.length.values,
        [4890.81221721, 10929.54403336, 13901.28305631]
    )
    # errors
    with pytest.raises(TypeError, match="expected 'n' as an instance of Surf"):
        spatial.location_pair_geoms(pairs, loc_df, False)
    with pytest.raises(TypeError, match="loc_df must be a GeoDataFrame or"):
        spatial.location_pair_geoms(pairs, loc_df.segnum, coastal_swn)
    with pytest.raises(ValueError, match="loc_df must have 'segnum' column"):
        spatial.location_pair_geoms(pairs, loc_df[["method"]], coastal_swn)
    with pytest.raises(ValueError, match="loc_df must have 'seg_ndist' colum"):
        spatial.location_pair_geoms(pairs, loc_df[["segnum"]], coastal_swn)
    loc_df.segnum += 10
    with pytest.raises(ValueError, match="loc_df has segnum values not found"):
        spatial.location_pair_geoms(pairs, loc_df, coastal_swn)
    n2 = swn.SurfaceWaterNetwork.from_lines(
        coastal_swn.segments.geometry.iloc[0:4])
    with pytest.raises(ValueError, match="loc_df has segnum values not found"):
        spatial.location_pair_geoms(pairs, loc_df, n2)
