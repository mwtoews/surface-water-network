# -*- coding: utf-8 -*-
"""Common code for testing
"""
import geopandas
import os
import pandas as pd
import pytest

# Import this local package for tests
import sys
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import swn  # noqa: E402

datadir = os.path.join('tests', 'data')


@pytest.fixture(scope='session', autouse=True)
def coastal_lines_gdf():
    shp_srs = os.path.join(datadir, 'DN2_Coastal_strahler1z_stream_vf.shp')
    gdf = geopandas.read_file(shp_srs)
    gdf.set_index('nzsegment', inplace=True)
    return gdf


@pytest.fixture(scope='module')
def coastal_polygons_gdf(coastal_lines_gdf):
    shp_srs = os.path.join(datadir, 'DN2_Coastal_strahler1_vf.shp')
    polygons = geopandas.read_file(shp_srs)
    polygons.set_index('nzsegment', inplace=True)
    # repair the shapefile by filling in the missing data
    for segnum in [3046737, 3047026, 3047906, 3048995, 3049065]:
        line = coastal_lines_gdf.loc[segnum]
        polygons.loc[segnum] = {
            'HydroID': line['HydroID'],
            'GridID': 0,
            'OBJECTID': 0,
            'nzreach_re': line['nzreach_re'],
            'Shape_Leng': 0.0,
            'Shape_Area': 0.0,
            'Area': 0.0,
            'X84': 0.0,
            'Y84': 0.0,
            'geometry': line['geometry'].centroid.buffer(20.0, 1),
            # wkt.loads('POLYGON EMPTY')
        }
    return polygons.reindex(index=coastal_lines_gdf.index)


@pytest.fixture(scope='session', autouse=True)
def coastal_points_gdf():
    shp_srs = os.path.join(datadir, 'Coastal_SW_abstraction.shp')
    gdf = geopandas.read_file(shp_srs)
    gdf.set_index('id', inplace=True)
    return gdf


@pytest.fixture(scope='module')
def coastal_swn(coastal_lines_gdf):
    return swn.SurfaceWaterNetwork(coastal_lines_gdf.geometry)


@pytest.fixture(scope='session', autouse=True)
def coastal_flow_ts():
    csv_fname = 'streamq_20170115_20170128_topnet_03046727_m3day.csv'
    ts = pd.read_csv(os.path.join(datadir, csv_fname), index_col=0)
    ts.columns = ts.columns.astype(int)
    ts.index = pd.to_datetime(ts.index)
    return ts


@pytest.fixture(scope='module')
def coastal_flow_m(coastal_flow_ts):
    flow_m = pd.DataFrame(coastal_flow_ts.mean(0)).T
    # flow_m.index = pd.DatetimeIndex(['2000-01-01'])
    return flow_m
