import pickle
from textwrap import dedent

import geopandas
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

import swn

from .conftest import PANDAS_VESRSION_TUPLE, datadir

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
    assert np.issubdtype(shp["nzsegment"].dtype, np.integer)
    assert np.issubdtype(shp["from_seg"].dtype, np.object_)
    assert np.issubdtype(shp["to_seg"].dtype, np.integer)
    assert np.issubdtype(shp["upstr_len"].dtype, np.floating)
    shp.set_index("nzsegment", inplace=True)
    assert list(shp.columns) == \
        ["to_seg", "from_seg", "cat_group", "num_to_out", "dst_to_out",
         "sequence", "strm_order", "upstr_len", "geometry"]
    assert gdf.shape == shp.shape


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
    df1 = pd.DataFrame({
        "value1": [-1e10, -1e-10, 0, 1e-10, 1, 1000],
        "value2": [1, 10, 100, 1000, 10000, 100000],
        "value3": ["first one", "two", "three", np.nan, "five", "six"],
        }, index=[1, 12, 33, 40, 450, 6267])
    df1.index.name = "rno"

    # test default write method
    fname = tmp_path / "file.dat"
    swn.file.write_formatted_frame(df1, fname)
    lines = fname.read_text().splitlines()
    assert len(lines) == 7
    # check each line
    expected = dedent("""\
        # rno        value1  value2  value3
        1     -1.000000e+10       1  'first one'
        12    -1.000000e-10      10   two
        33     0.000000e+00     100   three
        40     1.000000e-10    1000
        450    1.000000e+00   10000   five
        6267   1.000000e+03  100000   six
    """)
    for ln, expected_line in enumerate(expected.splitlines()):
        assert lines[ln] == expected_line

    # test read method
    df2 = swn.file.read_formatted_frame(fname).set_index("rno")
    pd.testing.assert_frame_equal(df1, df2)

    # similar checks without comment char
    swn.file.write_formatted_frame(df1, fname, comment_header=False)
    lines = fname.read_text().splitlines()
    assert len(lines) == 7
    assert lines[0] == "rno         value1  value2  value3"

    # test read method
    df2 = swn.file.read_formatted_frame(fname).set_index("rno")
    pd.testing.assert_frame_equal(df1, df2)

    # without index
    swn.file.write_formatted_frame(
        df1, fname, comment_header=True, index=False)
    lines = fname.read_text().splitlines()
    assert len(lines) == 7
    # check first line, space between object columns differ between versions!
    if (2, 0, 0) <= PANDAS_VESRSION_TUPLE <= (2, 0, 2):
        expected = "#      value1  value2  value3"
    else:
        expected = "#      value1  value2 value3"
    assert lines[0] == expected
    df2 = swn.file.read_formatted_frame(fname)
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2)

    swn.file.write_formatted_frame(
        df1, fname, comment_header=False, index=False)
    lines = fname.read_text().splitlines()
    assert len(lines) == 7
    assert lines[0] == expected.replace("#", " ")
    df2 = swn.file.read_formatted_frame(fname)
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2)

    # empty data frame, no rows
    df = df1.iloc[0:0]
    swn.file.write_formatted_frame(df, fname)
    lines = fname.read_text().splitlines()
    assert len(lines) == 1
    assert lines[0] == "# rno value1 value2 value3"
    df2 = swn.file.read_formatted_frame(fname).set_index("rno")
    pd.testing.assert_frame_equal(
        df, df2, check_index_type=False, check_dtype=False)

    swn.file.write_formatted_frame(df, fname, index=False)
    lines = fname.read_text().splitlines()
    assert len(lines) == 1
    assert lines[0] == "# value1 value2 value3"
    df2 = swn.file.read_formatted_frame(fname)
    pd.testing.assert_frame_equal(
        df.reset_index(drop=True), df2,
        check_index_type=False, check_dtype=False)

    swn.file.write_formatted_frame(df, fname, comment_header=False)
    lines = fname.read_text().splitlines()
    assert len(lines) == 1
    assert lines[0] == "rno value1 value2 value3"
    df2 = swn.file.read_formatted_frame(fname).set_index("rno")
    pd.testing.assert_frame_equal(
        df, df2, check_index_type=False, check_dtype=False)

    swn.file.write_formatted_frame(
        df, fname, comment_header=False, index=False)
    lines = fname.read_text().splitlines()
    assert len(lines) == 1
    assert lines[0] == "value1 value2 value3"
    df2 = swn.file.read_formatted_frame(fname)
    pd.testing.assert_frame_equal(
        df.reset_index(drop=True), df2,
        check_index_type=False, check_dtype=False)

    # empty data frame, no columns
    df = df1.iloc[:, 0:0]
    swn.file.write_formatted_frame(df, fname)
    lines = fname.read_text().splitlines()
    assert len(lines) == 7
    assert lines[0] == "# rno"
    df2 = swn.file.read_formatted_frame(fname).set_index("rno")
    pd.testing.assert_frame_equal(df, df2)

    swn.file.write_formatted_frame(df, fname, index=False)
    lines = fname.read_text().splitlines()
    assert len(lines) == 7
    assert lines[0] == "#"
    df2 = swn.file.read_formatted_frame(fname)
    pd.testing.assert_frame_equal(
        df.reset_index(drop=True), df2, check_column_type=False)

    swn.file.write_formatted_frame(df, fname, comment_header=False)
    lines = fname.read_text().splitlines()
    assert len(lines) == 7
    assert lines[0] == "rno"
    df2 = swn.file.read_formatted_frame(fname).set_index("rno")
    pd.testing.assert_frame_equal(df, df2)

    swn.file.write_formatted_frame(
        df, fname, comment_header=False, index=False)
    lines = fname.read_text().splitlines()
    assert len(lines) == 7
    assert lines[0] == ""
    df2 = swn.file.read_formatted_frame(fname)
    pd.testing.assert_frame_equal(
        df.reset_index(drop=True), df2, check_column_type=False)

    # other checks with different frames
    df = pd.DataFrame({
        "k": pd.Series([1, 2], dtype=object),
        "i": pd.Series([1, 1], dtype=object),
        "j": pd.Series([6, 12], dtype=object),
        "elev": [14.0, -2.0],
        })

    swn.file.write_formatted_frame(
        df, fname, comment_header=True, index=True)
    header = fname.read_text().splitlines()[0]
    assert header == "# index  k  i   j  elev"
    df2 = swn.file.read_formatted_frame(fname)
    assert df.index.name is None
    df.index.name = "index"
    pd.testing.assert_frame_equal(
        df, df2.set_index("index"),
        check_dtype=False)

    swn.file.write_formatted_frame(
        df, fname, comment_header=True, index=False)
    header = fname.read_text().splitlines()[0]
    assert header == "#k i  j  elev"
    df2 = swn.file.read_formatted_frame(fname)
    pd.testing.assert_frame_equal(
        df.reset_index(drop=True), df2,
        check_dtype=False)
