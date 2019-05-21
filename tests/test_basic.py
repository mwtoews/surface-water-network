# -*- coding: utf-8 -*-
import geopandas
import pandas as pd
import pytest
import os
from shapely import wkt
try:
    import rtree
except ImportError:
    rtree = False

import sys
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import swn


# Helper functions
def wkt_to_df(wkt_list, geom_name='geometry'):
    df = pd.DataFrame({'wkt': wkt_list})
    df[geom_name] = df['wkt'].apply(wkt.loads)
    return df


def wkt_to_gdf(wkt_list, geom_name='geometry'):
    return geopandas.GeoDataFrame(
            wkt_to_df(wkt_list, geom_name), geometry=geom_name)


@pytest.fixture
def df():
    # valid network
    return wkt_to_df([
        'LINESTRING Z (40 130 15, 60 100 14)',
        'LINESTRING Z (70 130 15, 60 100 14)',
        'LINESTRING Z (60 100 14, 60 80 12)',
    ])


@pytest.fixture
def lines(df):
    return geopandas.GeoDataFrame(df, geometry='geometry')


def test_init_object():
    with pytest.raises(ValueError, match='lines must be a GeoDataFrame'):
        swn.SurfaceWaterNetwork(object())


def test_init_dataframe(df):
    with pytest.raises(ValueError, match='lines must be a GeoDataFrame'):
        swn.SurfaceWaterNetwork(df)


def test_init_zero_lines(lines):
    with pytest.raises(ValueError, match='one or more lines are required'):
        swn.SurfaceWaterNetwork(lines[0:0])


def test_init_geom_type(df):
    wkt = df['wkt'].tolist()
    wkt[1] = 'MULTILINESTRING Z ((70 130 15, 60 100 14))'
    lines = wkt_to_gdf(wkt)
    with pytest.raises(ValueError, match='lines must all be LineString types'):
        swn.SurfaceWaterNetwork(lines)


def test_init_2D_geom(df):
    # Rewrite WKT as 2D
    lines = wkt_to_gdf(
        df['wkt'].apply(wkt.loads).apply(wkt.dumps, output_dimension=2))
    with pytest.raises(ValueError, match='lines must all have Z dimension'):
        swn.SurfaceWaterNetwork(lines)


def test_init_mismatch_3D():
    # Match in 2D, but not in Z-dimension
    lines = wkt_to_gdf([
        'LINESTRING Z (40 130 15, 60 100 13)',
        'LINESTRING Z (70 130 15, 60 100 14)',
        'LINESTRING Z (60 100 14, 60 80 12)',
    ])
    n = swn.SurfaceWaterNetwork(lines)
    assert len(n.warnings) == 1
    assert n.warnings[0] == 'node 0 matches 2 in 2D, but not in Z-dimension'
    assert len(n.errors) == 0
    assert n.outlets.values.tolist() == [2]


def test_init_all_converge():
    # Lines all converge to the same place. Should this be a warning / error?
    lines = wkt_to_gdf([
        'LINESTRING Z (40 130 15, 60 100 15)',
        'LINESTRING Z (70 130 14, 60 100 14)',
        'LINESTRING Z (60 80 12, 60 100 14)',
    ])
    n = swn.SurfaceWaterNetwork(lines)
    assert len(n.warnings) == 0
    assert len(n.errors) == 0
    assert n.outlets.values.tolist() == [0, 1, 2]


def test_init_defaults(lines):
    n = swn.SurfaceWaterNetwork(lines)
    assert n.logger is not None
    assert len(n) == 3
    assert n.END_NODE == -1
    assert n.lines_idx is None
    assert n.reaches.index is n.lines.index
    assert n.reaches['to_node'].values.tolist() == [2, 2, -1]
    assert n.outlets.values.tolist() == [2]
    assert len(n.warnings) == 0
    assert len(n.errors) == 0
