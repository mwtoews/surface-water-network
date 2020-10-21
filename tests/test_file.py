# -*- coding: utf-8 -*-
import geopandas
import numpy as np
import os
import pandas as pd
import pickle
from shapely.geometry import Point

import pytest
from .conftest import datadir

import swn
from swn.spatial import wkt_to_geoseries

# same valid network used in test_basic
n3d_lines = wkt_to_geoseries([
    'LINESTRING Z (60 100 14, 60  80 12)',
    'LINESTRING Z (40 130 15, 60 100 14)',
    'LINESTRING Z (70 130 15, 60 100 14)',
])

valid_polygons = wkt_to_geoseries([
    'POLYGON ((35 100, 75 100, 75  80, 35  80, 35 100))',
    'POLYGON ((35 135, 60 135, 60 100, 35 100, 35 135))',
    'POLYGON ((60 135, 75 135, 75 100, 60 100, 60 135))',
])

diversions = geopandas.GeoDataFrame(geometry=[
    Point(58, 97), Point(62, 97), Point(61, 89), Point(59, 89)])


def test_topnet2ts(coastal_flow_ts):
    pytest.importorskip('netCDF4')
    nc_fname = 'streamq_20170115_20170128_topnet_03046727_strahler1.nc'
    flow = swn.file.topnet2ts(os.path.join(datadir, nc_fname), 'mod_flow')
    assert flow.shape == (14, 304)
    # convert from m3/s to m3/day
    flow *= 24 * 60 * 60
    # remove time and truncat to closest day
    try:
        flow.index = flow.index.floor('d')
    except AttributeError:
        # older pandas
        flow.index = pd.to_datetime(
            flow.index.map(lambda x: x.strftime('%Y-%m-%d')))
    # Compare against CSV version of this data
    assert flow.shape == (14, 304)
    np.testing.assert_array_equal(flow.columns, coastal_flow_ts.columns)
    np.testing.assert_array_equal(flow.index, coastal_flow_ts.index)
    np.testing.assert_array_almost_equal(flow, coastal_flow_ts, 2)


def test_pickle(tmp_path):
    # use pickle dumps / loads methods
    n1 = swn.SurfaceWaterNetwork.from_lines(n3d_lines)
    data = pickle.dumps(n1)
    n2 = pickle.loads(data)
    assert n1 == n2
    # use to_pickle / from_pickle methods
    n3 = swn.SurfaceWaterNetwork.from_lines(n3d_lines, valid_polygons)
    n3.to_pickle(tmp_path / "n4.pickle")
    n4 = swn.SurfaceWaterNetwork.from_pickle(tmp_path / "n4.pickle")
    assert n3 == n4
    assert n2 != n4
    n1.set_diversions(diversions)
    n1.to_pickle(tmp_path / "n5.pickle")
    n5 = swn.SurfaceWaterNetwork.from_pickle(tmp_path / "n5.pickle")
    assert n1 != n2
    assert n1 == n5
