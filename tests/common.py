# -*- coding: utf-8 -*-
import geopandas
import os
import pandas as pd
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
