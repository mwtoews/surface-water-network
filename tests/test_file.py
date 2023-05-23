import pickle

import geopandas
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

import swn

from .conftest import PANDAS_MAJOR_VERSION, datadir


# same valid network used in test_basic
n3d_lines = geopandas.GeoSeries.from_wkt([
    'LINESTRING Z (60 100 14, 60  80 12)',
    'LINESTRING Z (40 130 15, 60 100 14)',
    'LINESTRING Z (70 130 15, 60 100 14)',
])

valid_polygons = geopandas.GeoSeries.from_wkt([
    'POLYGON ((35 100, 75 100, 75  80, 35  80, 35 100))',
    'POLYGON ((35 135, 60 135, 60 100, 35 100, 35 135))',
    'POLYGON ((60 135, 75 135, 75 100, 60 100, 60 135))',
])

diversions = geopandas.GeoDataFrame(geometry=[
    Point(58, 97), Point(62, 97), Point(61, 89), Point(59, 89)])


def test_topnet2ts(coastal_flow_ts):
    pytest.importorskip("netCDF4")
    nc_fname = "streamq_20170115_20170128_topnet_03046727_strahler1.nc"
    # read variable, and convert from m3/s to m3/day
    flow = swn.file.topnet2ts(datadir / nc_fname, "mod_flow", mult=86400)
    assert flow.shape == (14, 304)
    assert list(pd.unique(flow.dtypes)) == [np.float32]
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
    # errors
    with pytest.raises(KeyError, match="'nope' not found in dataset; use one"):
        swn.file.topnet2ts(datadir / nc_fname, "nope")
    with pytest.raises(IndexError, match="index exceeds dimension bounds"):
        swn.file.topnet2ts(datadir / nc_fname, "mod_flow", run=1)


def test_gdf_to_shapefile(tmp_path, coastal_swn):
    gdf = coastal_swn.segments.copy()
    fname = tmp_path / "segments.shp"
    swn.file.gdf_to_shapefile(gdf, fname)
    assert "from_segnums" in gdf.columns
    assert "from_seg" not in gdf.columns
    shp = geopandas.read_file(fname)
    assert "from_segnums" not in shp.columns
    assert "from_seg" in shp.columns


def test_pickle_lines():
    n1 = swn.SurfaceWaterNetwork.from_lines(n3d_lines)
    data = pickle.dumps(n1)
    n2 = pickle.loads(data)
    assert n1 == n2


def test_pickle_lines_polygons():
    n1 = swn.SurfaceWaterNetwork.from_lines(n3d_lines, valid_polygons)
    data = pickle.dumps(n1)
    n2 = pickle.loads(data)
    assert n1 == n2


def test_pickle_lines_diversions():
    n1 = swn.SurfaceWaterNetwork.from_lines(n3d_lines)
    n1.set_diversions(diversions)
    data = pickle.dumps(n1)
    n2 = pickle.loads(data)
    assert n1 == n2


def test_pickle_file_methods(tmp_path):
    # use to_pickle / from_pickle methods
    n1 = swn.SurfaceWaterNetwork.from_lines(n3d_lines, valid_polygons)
    n1.set_diversions(diversions)
    n1.to_pickle(tmp_path / "n2.pickle")
    n2 = swn.SurfaceWaterNetwork.from_pickle(tmp_path / "n2.pickle")
    assert n1 == n2


def test_read_write_formatted_frame(tmp_path):
    df = pd.DataFrame({
        "value1": [-1e10, -1e-10, 0, 1e-10, 1, 1000],
        "value2": [1, 10, 100, 1000, 10000, 100000],
        "value3": ["first one", "two", "three", np.nan, "five", "six"],
        }, index=[1, 12, 33, 40, 450, 6267])
    df.index.name = "rno"

    # test default write method
    fname = tmp_path / "file.dat"
    swn.file.write_formatted_frame(df, fname)
    # check first line
    with fname.open() as f:
        header = f.readline()
    assert header == "# rno        value1  value2  value3\n"

    # test read method
    df2 = swn.file.read_formatted_frame(fname).set_index("rno")
    pd.testing.assert_frame_equal(df, df2)

    # similar checks without comment char
    swn.file.write_formatted_frame(df, fname, comment_header=False)
    # check first line
    with fname.open() as f:
        header = f.readline()
    assert header == "rno         value1  value2  value3\n"

    # test read method
    df2 = swn.file.read_formatted_frame(fname).set_index("rno")
    pd.testing.assert_frame_equal(df, df2)

    # without index
    swn.file.write_formatted_frame(
        df, fname, comment_header=True, index=False)
    # check first line
    with fname.open() as f:
        header = f.readline()
    if PANDAS_MAJOR_VERSION >= 2:
        expected = "#      value1  value2  value3\n"
    else:
        expected = "#      value1  value2 value3\n"
    assert header == expected

    # test read method
    df2 = swn.file.read_formatted_frame(fname)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), df2)

    swn.file.write_formatted_frame(
        df, fname, comment_header=False, index=False)
    # check first line
    with fname.open() as f:
        header = f.readline()
    assert header == expected.replace("#", " ")

    # test read method
    df2 = swn.file.read_formatted_frame(fname)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), df2)

    # empty data frame
    df = df.iloc[0:0]
    swn.file.write_formatted_frame(df, fname)
    assert fname.read_text() == "# rno value1 value2 value3\n"
    df2 = swn.file.read_formatted_frame(fname).set_index("rno")
    pd.testing.assert_frame_equal(
        df, df2, check_index_type=False, check_dtype=False)

    swn.file.write_formatted_frame(df, fname, index=False)
    assert fname.read_text() == "# value1 value2 value3\n"
    df2 = swn.file.read_formatted_frame(fname)
    pd.testing.assert_frame_equal(
        df.reset_index(drop=True), df2,
        check_index_type=False, check_dtype=False)

    swn.file.write_formatted_frame(df, fname, comment_header=False)
    assert fname.read_text() == "rno value1 value2 value3\n"
    df2 = swn.file.read_formatted_frame(fname).set_index("rno")
    pd.testing.assert_frame_equal(
        df, df2, check_index_type=False, check_dtype=False)

    swn.file.write_formatted_frame(
        df, fname, comment_header=False, index=False)
    assert fname.read_text() == "value1 value2 value3\n"
    df2 = swn.file.read_formatted_frame(fname)
    pd.testing.assert_frame_equal(
        df.reset_index(drop=True), df2,
        check_index_type=False, check_dtype=False)
