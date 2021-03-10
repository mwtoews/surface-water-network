# -*- coding: utf-8 -*-
import geopandas
import numpy as np
from shapely import wkt
from shapely.geometry import Point

import pytest

import swn.spatial as spatial
from swn import SurfaceWaterNetwork


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


def test_find_segnum_in_swn_only_lines(coastal_swn):
    n = coastal_swn
    # querey tuple
    r = spatial.find_segnum_in_swn(n, (1811503.1, 5874071.2))
    assert len(r) == 1
    assert r.index[0] == 0
    assert r.segnum[0] == 3047364
    assert round(r.dist_to_segnum[0], 1) == 80.3
    # querey list of tuples
    geom = [
        (1806234.3, 5869114.8),
        (1804222, 5870087),
        (1802814, 5867160)]
    r = spatial.find_segnum_in_swn(n, geom)
    assert len(r) == 3
    assert list(r.segnum) == [3048663, 3048249, 3049113]
    assert list(r.dist_to_segnum.round(1)) == [315.5, 364.7, 586.1]
    # query GeoSeries
    poylgon = wkt.loads('''\
    POLYGON ((1815228 5869053, 1815267 5869021, 1815120 5868936,
              1815228 5869053))''')
    gs = geopandas.GeoSeries(
        [poylgon, Point(1814271, 5869525)], index=[101, 102])
    r = spatial.find_segnum_in_swn(n, gs)
    assert len(r) == 2
    assert list(r.segnum) == [3048690, 3048372]
    assert list(r.dist_to_segnum.round(1)) == [73.4, 333.8]


def test_find_segnum_in_swn_with_catchments(
        coastal_lines_gdf, coastal_polygons_gdf):
    n = SurfaceWaterNetwork.from_lines(
        coastal_lines_gdf.geometry, coastal_polygons_gdf.geometry)
    # querey tuple
    r = spatial.find_segnum_in_swn(n, (1811503.1, 5874071.2))
    assert len(r) == 1
    assert r.index[0] == 0
    assert r.segnum[0] == 3046952
    assert round(r.dist_to_segnum[0], 1) == 169.2
    assert r.is_within_catchment[0]
    # querey list of tuples
    geom = [
        (1806234.3, 5869114.8),
        (1804222, 5870087),
        (1802814, 5867160)]
    r = spatial.find_segnum_in_swn(n, geom)
    assert len(r) == 3
    assert list(r.segnum) == [3048504, 3048249, 3049113]
    assert list(r.dist_to_segnum.round(1)) == [496.7, 364.7, 586.1]
    assert list(r.is_within_catchment) == [True, False, False]
    # query GeoSeries
    poylgon = wkt.loads('''\
    POLYGON ((1815228 5869053, 1815267 5869021, 1815120 5868936,
              1815228 5869053))''')
    gs = geopandas.GeoSeries(
        [poylgon, Point(1814271, 5869525)], index=[101, 102])
    r = spatial.find_segnum_in_swn(n, gs)
    assert len(r) == 2
    assert list(r.segnum) == [3048690, 3048482]
    assert list(r.dist_to_segnum.round(1)) == [73.4, 989.2]
    assert list(r.is_within_catchment) == [True, True]
