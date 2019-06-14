# -*- coding: utf-8 -*-
import geopandas
import os
import pytest

from .common import swn

datadir = os.path.join('tests', 'data')


@pytest.fixture(scope='session', autouse=True)
def costal_lines():
    shp_srs = os.path.join(datadir, 'DN2_Coastal_strahler1z_stream_vf.shp')
    lines = geopandas.read_file(shp_srs)
    lines.set_index('nzsegment', inplace=True)
    return lines


@pytest.fixture(scope='session', autouse=True)
def costal_polygons(costal_lines):
    shp_srs = os.path.join(datadir, 'DN2_Coastal_strahler1_vf.shp')
    polygons = geopandas.read_file(shp_srs)
    polygons.set_index('nzsegment', inplace=True)
    # repair the shapefile by filling in the missing data
    for segnum in [3046737, 3047026, 3047906, 3048995, 3049065]:
        line = costal_lines.loc[segnum]
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
            'geometry': line['geometry'].buffer(1.0),
            # wkt.loads('POLYGON EMPTY')
        }
    return polygons.reindex(index=costal_lines.index)


@pytest.fixture(scope='session', autouse=True)
def costal_swn(costal_lines):
    return swn.SurfaceWaterNetwork(costal_lines)
