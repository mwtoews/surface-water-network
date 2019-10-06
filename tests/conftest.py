# -*- coding: utf-8 -*-
import geopandas
import os
import pandas as pd
import pytest
import sys

from shapely import wkt
try:
    import rtree
except ImportError:
    rtree = False

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import swn


datadir = os.path.join('tests', 'data')


# Helper functions
def wkt_to_dataframe(wkt_list, geom_name='geometry'):
    df = pd.DataFrame({'wkt': wkt_list})
    df[geom_name] = df['wkt'].apply(wkt.loads)
    return df


def wkt_to_geodataframe(wkt_list, geom_name='geometry'):
    return geopandas.GeoDataFrame(
            wkt_to_dataframe(wkt_list, geom_name), geometry=geom_name)


def wkt_to_geoseries(wkt_list, geom_name=None):
    geom = geopandas.GeoSeries([wkt.loads(x) for x in wkt_list])
    if geom_name is not None:
        geom.name = geom_name
    return geom


@pytest.fixture(scope='session', autouse=True)
def costal_lines_gdf():
    shp_srs = os.path.join(datadir, 'DN2_Coastal_strahler1z_stream_vf.shp')
    gdf = geopandas.read_file(shp_srs)
    gdf.set_index('nzsegment', inplace=True)
    return gdf


@pytest.fixture(scope='session', autouse=True)
def costal_polygons_gdf(costal_lines_gdf):
    shp_srs = os.path.join(datadir, 'DN2_Coastal_strahler1_vf.shp')
    polygons = geopandas.read_file(shp_srs)
    polygons.set_index('nzsegment', inplace=True)
    # repair the shapefile by filling in the missing data
    for segnum in [3046737, 3047026, 3047906, 3048995, 3049065]:
        line = costal_lines_gdf.loc[segnum]
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
    return polygons.reindex(index=costal_lines_gdf.index)


@pytest.fixture(scope='session', autouse=True)
def costal_swn(costal_lines_gdf):
    return swn.SurfaceWaterNetwork(costal_lines_gdf.geometry)


@pytest.fixture(scope='session', autouse=True)
def clostal_flow_ts():
    csv_fname = 'streamq_20170115_20170128_topnet_03046727_m3day.csv'
    ts = pd.read_csv(os.path.join(datadir, csv_fname), index_col=0)
    ts.columns = ts.columns.astype(int)
    ts.index = pd.to_datetime(ts.index)
    return ts


@pytest.fixture(scope='session', autouse=True)
def clostal_flow_m(clostal_flow_ts):
    flow_m = pd.DataFrame(clostal_flow_ts.mean(0)).T
    # flow_m.index = pd.DatetimeIndex(['2000-01-01'])
    return flow_m
