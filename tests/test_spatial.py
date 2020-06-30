# -*- coding: utf-8 -*-
import geopandas
import numpy as np
from shapely.geometry import Point

import pytest

import swn.spatial as spatial


def test_get_sindex():
    xy = geopandas.GeoDataFrame(geometry=geopandas.points_from_xy([0], [1]))
    assert spatial.get_sindex(xy) is None
    assert spatial.get_sindex(xy.geometry) is None
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
    assert spatial.get_crs('2193') is not None
    assert spatial.get_crs('EPSG:2193') is not None
    assert spatial.get_crs('+init=EPSG:2193') is not None
    assert spatial.get_crs({'init': 'EPSG:2193'}) is not None


def test_compare_crs():
    assert spatial.compare_crs(None, None) == (None, None, True)
    assert spatial.compare_crs(2193, None)[2] is False
    assert spatial.compare_crs(2193, 'EPSG:2193')[2] is True
    assert spatial.compare_crs(2193, {'init': 'EPSG:2193'})[2] is True
