from textwrap import dedent

import geopandas
import numpy as np
import pandas as pd
import pytest
from shapely import wkt
from shapely.geometry import LineString, Point

import swn
from swn.spatial import force_2d, round_coords, wkt_to_geoseries

from .conftest import matplotlib, plt

# a few static objects (not fixtures)

valid_lines_list = [
    'LINESTRING Z (60 100 14, 60  80 12)',
    'LINESTRING Z (40 130 15, 60 100 14)',
    'LINESTRING Z (70 130 15, 60 100 14)',
]

valid_df = swn.spatial.wkt_to_dataframe(valid_lines_list)

valid_lines = wkt_to_geoseries(valid_lines_list)

valid_polygons = wkt_to_geoseries([
    'POLYGON ((35 100, 75 100, 75  80, 35  80, 35 100))',
    'POLYGON ((35 135, 60 135, 60 100, 35 100, 35 135))',
    'POLYGON ((60 135, 75 135, 75 100, 60 100, 60 135))',
])


@pytest.fixture
def valid_n():
    return swn.SurfaceWaterNetwork.from_lines(valid_lines)


def test_init_errors():
    with pytest.raises(ValueError, match='segments must be a GeoDataFrame'):
        swn.SurfaceWaterNetwork(object())
    with pytest.raises(ValueError, match='segments must be a GeoDataFrame'):
        swn.SurfaceWaterNetwork(valid_df)


def test_from_lines_errors():
    with pytest.raises(ValueError, match='lines must be a GeoSeries'):
        swn.SurfaceWaterNetwork.from_lines(object())
    with pytest.raises(ValueError, match='lines must be a GeoSeries'):
        swn.SurfaceWaterNetwork.from_lines(valid_df)
    with pytest.raises(ValueError, match='one or more lines are required'):
        swn.SurfaceWaterNetwork.from_lines(valid_lines[0:0])


def test_init_geom_type():
    wkt_list = valid_lines_list[:]
    wkt_list[1] = 'MULTILINESTRING Z ((70 130 15, 60 100 14))'
    lines = wkt_to_geoseries(wkt_list)
    with pytest.raises(ValueError, match='lines must all be LineString types'):
        swn.SurfaceWaterNetwork.from_lines(lines)


def test_init_defaults(valid_n):
    n = valid_n
    assert n.logger is not None
    assert len(n.warnings) == 0
    assert len(n.errors) == 0
    assert len(n) == 3
    assert n.catchments is None
    assert n.has_z is True
    assert n.END_SEGNUM == -1
    assert list(n.segments.index) == [0, 1, 2]
    assert list(n.segments['to_segnum']) == [-1, 0, 0]
    assert list(n.segments['from_segnums']) == [{1, 2}, set(), set()]
    assert list(n.segments['cat_group']) == [0, 0, 0]
    assert list(n.segments['num_to_outlet']) == [1, 2, 2]
    np.testing.assert_allclose(
        n.segments['dist_to_outlet'], [20.0, 56.05551, 51.622776])
    assert list(n.segments['sequence']) == [3, 1, 2]
    assert list(n.segments['stream_order']) == [2, 1, 1]
    np.testing.assert_allclose(
        n.segments['upstream_length'], [87.67828936, 36.05551275, 31.6227766])
    assert 'upstream_area' not in n.segments.columns
    assert 'width' not in n.segments.columns
    assert list(n.headwater) == [1, 2]
    assert list(n.outlets) == [0]
    assert dict(n.to_segnums) == {1: 0, 2: 0}
    assert dict(n.from_segnums) == {0: {1, 2}}
    n.adjust_elevation_profile()
    assert len(n.messages) == 0
    assert str(n) == repr(n)
    assert repr(n) == dedent('''\
        <SurfaceWaterNetwork: with Z coordinates
          3 segments: [0, 1, 2]
          2 headwater: [1, 2]
          1 outlets: [0]
          no diversions />''')
    if matplotlib:
        _ = n.plot()
        plt.close()


def test_segments(valid_n):
    assert valid_n.segments is valid_n._segments
    assert isinstance(valid_n.segments, geopandas.GeoDataFrame)
    # columns are checked in other tests
    with pytest.raises(AttributeError, match="can't set attribute"):
        valid_n.segments = None


def test_init_2D_geom():
    lines = force_2d(valid_lines)
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    assert len(n.warnings) == 0
    assert len(n.errors) == 0
    assert len(n) == 3
    assert n.has_z is False
    assert list(n.segments.index) == [0, 1, 2]
    assert list(n.segments['to_segnum']) == [-1, 0, 0]
    assert list(n.segments['cat_group']) == [0, 0, 0]
    assert list(n.segments['num_to_outlet']) == [1, 2, 2]
    np.testing.assert_allclose(
        n.segments['dist_to_outlet'], [20.0, 56.05551, 51.622776])
    assert list(n.segments['sequence']) == [3, 1, 2]
    assert list(n.segments['stream_order']) == [2, 1, 1]
    np.testing.assert_allclose(
        n.segments['upstream_length'], [87.67828936, 36.05551275, 31.6227766])
    assert list(n.headwater) == [1, 2]
    assert list(n.outlets) == [0]
    assert repr(n) == dedent('''\
        <SurfaceWaterNetwork:
          3 segments: [0, 1, 2]
          2 headwater: [1, 2]
          1 outlets: [0]
          no diversions />''')
    if matplotlib:
        _ = n.plot()
        plt.close()


def test_init_mismatch_3D():
    # Match in 2D, but not in Z dimension
    lines = wkt_to_geoseries([
        'LINESTRING Z (70 130 15, 60 100 14)',
        'LINESTRING Z (60 100 14, 60  80 12)',
        'LINESTRING Z (40 130 15, 60 100 13)',
    ])
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    assert len(n.warnings) == 1
    assert n.warnings[0] == \
        'end of segment 2 matches start of segment 1 in 2D, but not in '\
        'Z dimension'
    assert len(n.errors) == 0
    assert len(n) == 3
    assert n.has_z is True
    assert list(n.segments.index) == [0, 1, 2]
    assert repr(n) == dedent('''\
        <SurfaceWaterNetwork: with Z coordinates
          3 segments: [0, 1, 2]
          2 headwater: [0, 2]
          1 outlets: [1]
          no diversions />''')
    if matplotlib:
        _ = n.plot()
        plt.close()


def test_init_reversed_lines():
    # same as the working lines, but reversed in the opposite direction
    lines = valid_lines.geometry.apply(lambda x: LineString(x.coords[::-1]))
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    assert len(n.warnings) == 0
    assert len(n.errors) == 2
    assert n.errors[0] == \
        'segment 0 has more than one downstream segments: [1, 2]'
    assert n.errors[1] == \
        'starting coordinate (60.0, 100.0) matches start segment: {1}'
    assert len(n) == 3
    assert n.has_z is True
    assert list(n.segments.index) == [0, 1, 2]
    # This is all non-sense
    assert list(n.segments['to_segnum']) == [1, -1, -1]
    assert list(n.segments['cat_group']) == [1, 1, 2]
    assert list(n.segments['num_to_outlet']) == [2, 1, 1]
    np.testing.assert_allclose(
        n.segments['dist_to_outlet'], [56.05551, 36.05551, 31.622776])
    assert list(n.segments['sequence']) == [1, 3, 2]
    assert list(n.segments['stream_order']) == [1, 1, 1]
    np.testing.assert_allclose(
        n.segments['upstream_length'], [20.0, 56.05551275, 31.6227766])
    assert list(n.headwater) == [0, 2]
    assert list(n.outlets) == [1, 2]
    assert dict(n.to_segnums) == {0: 1}
    assert dict(n.from_segnums) == {1: {0}}
    assert repr(n) == dedent('''\
        <SurfaceWaterNetwork: with Z coordinates
          3 segments: [0, 1, 2]
          2 headwater: [0, 2]
          2 outlets: [1, 2]
          no diversions />''')
    if matplotlib:
        _ = n.plot()
        plt.close()


def test_init_all_converge():
    # Lines all converge to the same place
    lines = wkt_to_geoseries([
        'LINESTRING Z (40 130 15, 60 100 15)',
        'LINESTRING Z (70 130 14, 60 100 14)',
        'LINESTRING Z (60  80 12, 60 100 14)',
    ])
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    assert len(n.warnings) == 1
    # Note: ending segment 0 matches end of segment 1 in 2D,
    # but not in Z dimension
    assert n.warnings[0] == \
        'ending coordinate (60.0, 100.0) matches end segments: {0, 1, 2}'
    assert len(n.errors) == 0
    assert len(n) == 3
    assert n.has_z is True
    assert list(n.segments.index) == [0, 1, 2]
    assert list(n.segments['to_segnum']) == [-1, -1, -1]
    assert list(n.segments['cat_group']) == [0, 1, 2]
    assert list(n.segments['num_to_outlet']) == [1, 1, 1]
    np.testing.assert_allclose(
        n.segments['dist_to_outlet'], [36.05551, 31.622776, 20.0])
    assert list(n.segments['sequence']) == [1, 2, 3]
    assert list(n.segments['stream_order']) == [1, 1, 1]
    np.testing.assert_allclose(
        n.segments['upstream_length'], [36.05551, 31.622776, 20.0])
    assert list(n.headwater) == [0, 1, 2]
    assert list(n.outlets) == [0, 1, 2]
    assert dict(n.to_segnums) == {}
    assert dict(n.from_segnums) == {}
    assert repr(n) == dedent('''\
        <SurfaceWaterNetwork: with Z coordinates
          3 segments: [0, 1, 2]
          3 headwater: [0, 1, 2]
          3 outlets: [0, 1, 2]
          no diversions />''')
    if matplotlib:
        _ = n.plot()
        plt.close()


def test_init_all_diverge():
    # Lines all diverge from the same place
    lines = wkt_to_geoseries([
        'LINESTRING Z (60 100 15, 40 130 14)',
        'LINESTRING Z (60 100 16, 70 130 14)',
        'LINESTRING Z (60 100 15, 60  80 12)',
    ])
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    assert len(n.warnings) == 0
    # Note: starting segment 0 matches start of segment 1 in 2D,
    # but not in Z dimension
    assert len(n.errors) == 1
    assert n.errors[0] == \
        'starting coordinate (60.0, 100.0) matches start segments: {0, 1, 2}'
    assert len(n) == 3
    assert n.has_z is True
    assert list(n.segments.index) == [0, 1, 2]
    assert list(n.segments['to_segnum']) == [-1, -1, -1]
    assert list(n.segments['num_to_outlet']) == [1, 1, 1]
    assert list(n.segments['cat_group']) == [0, 1, 2]
    np.testing.assert_allclose(
        n.segments['dist_to_outlet'], [36.05551, 31.622776, 20.0])
    assert list(n.segments['sequence']) == [1, 2, 3]
    assert list(n.segments['stream_order']) == [1, 1, 1]
    np.testing.assert_allclose(
        n.segments['upstream_length'], [36.05551, 31.622776, 20.0])
    assert list(n.headwater) == [0, 1, 2]
    assert list(n.outlets) == [0, 1, 2]
    assert dict(n.to_segnums) == {}
    assert dict(n.from_segnums) == {}
    assert repr(n) == dedent('''\
        <SurfaceWaterNetwork: with Z coordinates
          3 segments: [0, 1, 2]
          3 headwater: [0, 1, 2]
          3 outlets: [0, 1, 2]
          no diversions />''')
    if matplotlib:
        _ = n.plot()
        plt.close()


def test_init_line_connects_to_middle():
    lines = wkt_to_geoseries([
        'LINESTRING Z (40 130 15, 60 100 14, 60 80 12)',
        'LINESTRING Z (70 130 15, 60 100 14)',
    ])
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    assert len(n.warnings) == 0
    assert len(n.errors) == 1
    assert n.errors[0] == 'segment 1 connects to the middle of segment 0'
    assert len(n) == 2
    assert n.has_z is True
    assert list(n.segments.index) == [0, 1]
    assert list(n.segments['to_segnum']) == [-1, -1]
    assert list(n.segments['cat_group']) == [0, 1]
    assert list(n.segments['num_to_outlet']) == [1, 1]
    np.testing.assert_allclose(
        n.segments['dist_to_outlet'], [56.05551, 31.622776])
    assert list(n.segments['sequence']) == [1, 2]
    assert list(n.segments['stream_order']) == [1, 1]
    np.testing.assert_allclose(
        n.segments['upstream_length'], [56.05551, 31.622776])
    assert list(n.headwater) == [0, 1]
    assert list(n.outlets) == [0, 1]
    assert dict(n.to_segnums) == {}
    assert dict(n.from_segnums) == {}
    assert repr(n) == dedent('''\
        <SurfaceWaterNetwork: with Z coordinates
          2 segments: [0, 1]
          2 headwater: [0, 1]
          2 outlets: [0, 1]
          no diversions />''')
    if matplotlib:
        _ = n.plot()
        plt.close()


def test_to_segnums(valid_n):
    # check series in propery method
    pd.testing.assert_series_equal(
        valid_n.to_segnums,
        pd.Series([0, 0], name="to_segnum", index=pd.Int64Index([1, 2])))
    # check series in segments frame
    pd.testing.assert_series_equal(
        valid_n.segments["to_segnum"],
        pd.Series([-1, 0, 0], name="to_segnum"))

    # Rebuild network using named index
    lines = valid_lines.copy()
    lines.index.name = "idx"
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    pd.testing.assert_series_equal(
        n.to_segnums,
        pd.Series([0, 0], name="to_segnum",
                  index=pd.Int64Index([1, 2], name="idx")))
    pd.testing.assert_series_equal(
        n.segments["to_segnum"],
        pd.Series([-1, 0, 0], name="to_segnum",
                  index=pd.RangeIndex(0, 3, name="idx")))


def test_from_segnums(valid_n):
    # check series in propery method
    pd.testing.assert_series_equal(
        valid_n.from_segnums,
        pd.Series([{1, 2}], name="from_segnums"))
    # check series in segments frame
    pd.testing.assert_series_equal(
        valid_n.segments["from_segnums"],
        pd.Series([{1, 2}, set(), set()], name="from_segnums"))

    # Rebuild network using named index
    lines = valid_lines.copy()
    lines.index.name = "idx"
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    pd.testing.assert_series_equal(
        n.from_segnums,
        pd.Series([{1, 2}], name="from_segnums",
                  index=pd.Int64Index([0], name="idx")))
    pd.testing.assert_series_equal(
        n.segments["from_segnums"],
        pd.Series([{1, 2}, set(), set()], name="from_segnums",
                  index=pd.RangeIndex(0, 3, name="idx")))


def test_dict(valid_n):
    # via __iter__
    d = dict(valid_n)
    assert list(d.keys()) == \
        ['class', 'segments', 'END_SEGNUM', 'catchments', 'diversions']
    assert d['class'] == 'SurfaceWaterNetwork'
    assert isinstance(d['segments'], geopandas.GeoDataFrame)
    assert list(d['segments'].index) == [0, 1, 2]
    assert list(d['segments'].columns) == \
        ['geometry', 'to_segnum', 'from_segnums', 'cat_group', 'num_to_outlet',
         'dist_to_outlet', 'sequence', 'stream_order', 'upstream_length']
    assert d['END_SEGNUM'] == -1
    assert d['catchments'] is None
    assert d['diversions'] is None


def test_copy_lines(valid_n):
    n1 = valid_n
    n2 = swn.SurfaceWaterNetwork(valid_n.segments, valid_n.END_SEGNUM)
    assert n1 is not n2


def test_copy_lines_polygons(valid_n):
    n1 = valid_n
    n2 = swn.SurfaceWaterNetwork(valid_n.segments, valid_n.END_SEGNUM)
    assert n1 is not n2


def test_eq_lines(valid_n):
    n1 = valid_n
    n2 = swn.SurfaceWaterNetwork(valid_n.segments, valid_n.END_SEGNUM)
    assert len(n1) == len(n2) == 3
    assert n1 == n2


def test_ne_lines(valid_n):
    n1 = valid_n
    n2 = swn.SurfaceWaterNetwork(valid_n.segments, valid_n.END_SEGNUM)
    n2.remove(segnums=[1])
    assert len(n1) != len(n2)
    assert n1 != n2


def test_init_geoseries():
    gs = wkt_to_geoseries(valid_lines_list, geom_name='foo')
    n = swn.SurfaceWaterNetwork.from_lines(gs)
    assert len(n.warnings) == 0
    assert len(n.errors) == 0
    assert len(n) == 3
    assert list(n.segments.index) == [0, 1, 2]
    assert n.has_z is True
    v = pd.Series([3.0, 2.0, 4.0])
    a = n.accumulate_values(v)
    assert dict(a) == {0: 9.0, 1: 2.0, 2: 4.0}


def test_init_segments_loc():
    lines = wkt_to_geoseries([
        "LINESTRING (60 100, 60  80)",
        "LINESTRING (40 130, 60 100)",
        "LINESTRING (70 130, 60 100)",
        "LINESTRING (60  80, 70  70)",
    ])
    lines.index += 100
    n1 = swn.SurfaceWaterNetwork.from_lines(lines)
    assert list(n1.outlets) == [103]
    # Create by slicing another GeoDataFrame, but check to_segnums
    n2 = swn.SurfaceWaterNetwork(n1.segments.loc[100:102])
    assert len(n2.segments) == 3
    assert list(n2.outlets) == [100]
    assert dict(n2.to_segnums) == {101: 100, 102: 100}


def test_accumulate_values_must_be_series(valid_n):
    with pytest.raises(ValueError, match='values must be a pandas Series'):
        valid_n.accumulate_values([3.0, 2.0, 4.0])


def test_accumulate_values_different_index(valid_n):
    # indexes don't completely overlap
    v = pd.Series([3.0, 2.0, 4.0])
    v.index += 1
    with pytest.raises(ValueError, match='index is different'):
        valid_n.accumulate_values(v)
    # indexes overlap, but have a different sequence
    v = pd.Series([3.0, 2.0, 4.0]).sort_values()
    with pytest.raises(ValueError, match='index is different'):
        valid_n.accumulate_values(v)


def test_accumulate_values_expected(valid_n):
    v = pd.Series([2.0, 3.0, 4.0])
    a = valid_n.accumulate_values(v)
    assert dict(a) == {0: 9.0, 1: 3.0, 2: 4.0}
    assert a.name is None


def test_init_polygons():
    expected_area = [800.0, 875.0, 525.0]
    expected_upstream_area = [2200.0, 875.0, 525.0]
    expected_upstream_width = [1.4615, 1.4457, 1.4397]
    # from GeoSeries
    n = swn.SurfaceWaterNetwork.from_lines(valid_lines, valid_polygons)
    assert n.catchments is not None
    np.testing.assert_array_almost_equal(n.catchments.area, expected_area)
    np.testing.assert_array_almost_equal(
            n.segments['upstream_area'], expected_upstream_area)
    np.testing.assert_array_almost_equal(
            n.segments['width'], expected_upstream_width, 4)
    # from GeoDataFrame
    n = swn.SurfaceWaterNetwork.from_lines(valid_lines, valid_polygons)
    assert n.catchments is not None
    np.testing.assert_array_almost_equal(n.catchments.area, expected_area)
    np.testing.assert_array_almost_equal(
            n.segments['upstream_area'], expected_upstream_area)
    np.testing.assert_array_almost_equal(
            n.segments['width'], expected_upstream_width, 4)
    # manual upstream area calculation
    np.testing.assert_array_almost_equal(
        n.accumulate_values(n.catchments.area), expected_upstream_area)
    assert repr(n) == dedent('''\
        <SurfaceWaterNetwork: with Z coordinates and catchment polygons
          3 segments: [0, 1, 2]
          2 headwater: [1, 2]
          1 outlets: [0]
          no diversions />''')
    if matplotlib:
        _ = n.plot()
        plt.close()
    # check error
    with pytest.raises(
            ValueError,
            match='polygons must be a GeoSeries or None'):
        swn.SurfaceWaterNetwork.from_lines(valid_lines, 1.0)


def test_catchments_property():
    # also vary previous test with 2D lines
    n = swn.SurfaceWaterNetwork.from_lines(force_2d(valid_lines))
    assert n.catchments is None
    n.catchments = valid_polygons
    assert n.catchments is not None
    np.testing.assert_array_almost_equal(
            n.catchments.area, [800.0, 875.0, 525.0])
    np.testing.assert_array_almost_equal(
            n.segments['upstream_area'], [2200.0, 875.0, 525.0])
    np.testing.assert_array_almost_equal(
            n.segments['width'], [1.4615, 1.4457, 1.4397], 4)
    assert repr(n) == dedent('''\
        <SurfaceWaterNetwork: with catchment polygons
          3 segments: [0, 1, 2]
          2 headwater: [1, 2]
          1 outlets: [0]
          no diversions />''')
    if matplotlib:
        _ = n.plot()
        plt.close()
    # unset property
    n.catchments = None
    assert n.catchments is None
    # check errors
    with pytest.raises(
            ValueError,
            match='catchments must be a GeoSeries or None'):
        n.catchments = 1.0
    with pytest.raises(
            ValueError,
            match=r'catchments\.index is different than for segments'):
        n.catchments = valid_polygons.iloc[1:]


def test_set_diversions_geodataframe():
    n = swn.SurfaceWaterNetwork.from_lines(valid_lines)
    diversions = geopandas.GeoDataFrame(geometry=[
        Point(58, 97), Point(62, 97), Point(61, 89), Point(59, 89)])
    # check errors
    with pytest.raises(
            AttributeError,
            match=r'use \'set_diversions\(\)\' method'):
        n.diversions = diversions
    with pytest.raises(
            ValueError,
            match=r'a \[Geo\]DataFrame is expected'):
        n.set_diversions([1])
    with pytest.raises(
            ValueError,
            match=r'a \[Geo\]DataFrame is expected'):
        n.set_diversions(diversions.geometry)
    with pytest.raises(
            ValueError,
            match='does not appear to be spatial'):
        n.set_diversions(geopandas.GeoDataFrame([1]))
    # normal operation
    assert n.diversions is None
    n.set_diversions(diversions)
    assert n.diversions is not None
    np.testing.assert_array_almost_equal(
        n.diversions['dist_end'], [3.605551, 3.605551, 9.055385, 9.055385])
    np.testing.assert_array_almost_equal(
        n.diversions['dist_line'], [3.605551, 3.605551, 1.0, 1.0])
    np.testing.assert_array_equal(
        n.diversions['from_segnum'], [1, 2, 0, 0])
    np.testing.assert_array_equal(
        n.segments['diversions'], [{2, 3}, {0}, {1}])
    # Unset
    n.set_diversions(None)
    assert n.diversions is None
    assert 'diversions' not in n.segments.columns
    # Try again with min_stream_order option
    n.set_diversions(diversions, min_stream_order=2)
    assert n.diversions is not None
    np.testing.assert_array_almost_equal(
        n.diversions['dist_end'], [17.117243, 17.117243, 9.055385, 9.055385])
    np.testing.assert_array_equal(
        n.diversions['dist_line'], [2.0, 2.0, 1.0, 1.0])
    np.testing.assert_array_equal(
        n.diversions['from_segnum'], [0, 0, 0, 0])
    np.testing.assert_array_equal(
        n.segments['diversions'], [{0, 1, 2, 3}, set(), set()])
    # Try again, but use 'from_segnum' column
    diversions = geopandas.GeoDataFrame(
        {'from_segnum': [0, 2]}, geometry=[Point(55, 97), Point(68, 105)])
    n.set_diversions(diversions)
    assert n.diversions is not None
    np.testing.assert_array_almost_equal(
        n.diversions['dist_end'], [17.720045,  9.433981])
    np.testing.assert_array_almost_equal(
        n.diversions['dist_line'], [5.0, 6.008328])
    np.testing.assert_array_equal(
        n.diversions['from_segnum'], [0, 2])
    np.testing.assert_array_equal(
        n.segments['diversions'], [{0}, set(), {1}])
    assert repr(n) == dedent('''\
        <SurfaceWaterNetwork: with Z coordinates
          3 segments: [0, 1, 2]
          2 headwater: [1, 2]
          1 outlets: [0]
          2 diversions (as GeoDataFrame): [0, 1] />''')
    if matplotlib:
        _ = n.plot()
        plt.close()


def test_set_diversions_dataframe():
    n = swn.SurfaceWaterNetwork.from_lines(valid_lines)
    diversions = pd.DataFrame({'from_segnum': [0, 2]})
    # check errors
    with pytest.raises(
            AttributeError,
            match=r'use \'set_diversions\(\)\' method'):
        n.diversions = diversions
    with pytest.raises(
            ValueError,
            match=r'a \[Geo\]DataFrame is expected'):
        n.set_diversions(diversions.from_segnum)
    with pytest.raises(
            ValueError,
            match='does not appear to be spatial'):
        n.set_diversions(pd.DataFrame([1]))
    # normal operation
    assert n.diversions is None
    n.set_diversions(diversions)
    assert n.diversions is not None
    assert 'dist_end' not in n.diversions.columns
    assert 'dist_line' not in n.diversions.columns
    np.testing.assert_array_equal(n.diversions['from_segnum'], [0, 2])
    np.testing.assert_array_equal(
        n.segments['diversions'], [{0}, set(), {1}])
    assert repr(n) == dedent('''\
        <SurfaceWaterNetwork: with Z coordinates
          3 segments: [0, 1, 2]
          2 headwater: [1, 2]
          1 outlets: [0]
          2 diversions (as DataFrame): [0, 1] />''')
    if matplotlib:
        _ = n.plot()
        plt.close()
    # Unset
    n.set_diversions(None)
    assert n.diversions is None
    assert 'diversions' not in n.segments.columns


def test_estimate_width():
    n = swn.SurfaceWaterNetwork.from_lines(valid_lines, valid_polygons)
    # defaults
    np.testing.assert_array_almost_equal(
            n.segments['width'], [1.4615, 1.4457, 1.4397], 4)
    # float a and b
    n.estimate_width(1.4, 0.6)
    np.testing.assert_array_almost_equal(
            n.segments['width'], [1.4254, 1.4146, 1.4108], 4)
    # integer a and b
    n.estimate_width(1, 1)
    np.testing.assert_array_almost_equal(
            n.segments['width'], [1.0022, 1.0009, 1.0005], 4)
    # a and b are Series per segment
    n.estimate_width([1.2, 1.8, 1.4], [0.4, 0.7, 0.6])
    np.testing.assert_array_almost_equal(
            n.segments['width'], [1.2006, 1.8, 1.4], 4)
    # based on Series
    n.estimate_width(upstream_area=n.segments['upstream_area'] / 2)
    np.testing.assert_array_almost_equal(
            n.segments['width'], [1.4489, 1.4379, 1.4337], 4)
    # defaults (again)
    n.estimate_width()
    np.testing.assert_array_almost_equal(
            n.segments['width'], [1.4615, 1.4457, 1.4397], 4)
    # based on a column, where upstream_area is not available
    n2 = swn.SurfaceWaterNetwork.from_lines(valid_lines)
    assert 'width' not in n2.segments.columns
    assert 'upstream_area' not in n2.segments.columns
    n2.segments['foo_upstream'] = n2.segments['upstream_length'] * 25
    n2.estimate_width(upstream_area='foo_upstream')
    np.testing.assert_array_almost_equal(
            n2.segments['width'], [1.4614, 1.4461, 1.4444], 4)
    # check errors
    with pytest.raises(
            ValueError,
            match='unknown use for upstream_area'):
        n.estimate_width(upstream_area=3)
    with pytest.raises(
            ValueError,
            match=r"'upstream_area' not found in segments\.columns"):
        n2.estimate_width()


def test_segments_series(valid_n):
    n = valid_n
    errmsg = "index is different than for segments"
    # from scalar
    pd.testing.assert_series_equal(
        n.segments_series(8.0),
        pd.Series([8.0] * 3))
    pd.testing.assert_series_equal(
        n.segments_series(8, name="eight"),
        pd.Series([8, 8, 8], name="eight"))
    pd.testing.assert_series_equal(
        n.segments_series("$VAL$"),
        pd.Series(["$VAL$"] * 3))
    # from list
    pd.testing.assert_series_equal(
        n.segments_series([3, 4, 5]),
        pd.Series([3, 4, 5]))
    pd.testing.assert_series_equal(
        n.segments_series([3, 4, 5], name="list"),
        pd.Series([3, 4, 5], name="list"))
    pd.testing.assert_series_equal(
        n.segments_series(["$VAL1$", "$VAL2$", "$VAL3$"]),
        pd.Series(["$VAL1$", "$VAL2$", "$VAL3$"]))
    with pytest.raises(ValueError, match=errmsg):
        n.segments_series([8])
    with pytest.raises(ValueError, match=errmsg):
        n.segments_series([3, 4, 5, 6])
    # from dict
    pd.testing.assert_series_equal(
        n.segments_series({0: 1.1, 1: 2.2, 2: 3.3}),
        pd.Series([1.1, 2.2, 3.3]))
    with pytest.raises(ValueError, match=errmsg):
        n.segments_series({0: 1.1, 1: 2.2, 3: 3.3})
    # from Series
    s = pd.Series([2.0, 3.0, 4.0])
    pd.testing.assert_series_equal(n.segments_series(s), s)
    s = pd.Series([2.0, 3.0, 4.0], name="foo")
    pd.testing.assert_series_equal(n.segments_series(s), s)
    # now break it
    s.index += 1
    with pytest.raises(ValueError, match=errmsg):
        n.segments_series(s)
    # misc error
    with pytest.raises(ValueError, match="expected value to be scalar, list,"):
        n.segments_series(object())


def test_pair_segments_frame(valid_n):
    n = valid_n
    errmsg_value = "index is different than for segments"
    errmsg_value_out = "value_out.index is not a subset of segments.index"
    errmsg_value_out_expected = "expected value_out to be scalar, dict or Seri"
    # from scalar
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(8.0, method="continuous"),
        pd.DataFrame({1: 8.0, 2: 8.0}, index=[0, 1, 2]))
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(8.0, name="foo"),
        pd.DataFrame({"foo1": 8.0, "foo2": 8.0}, index=[0, 1, 2]))
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(8, 9),
        pd.DataFrame({1: 8, 2: [9, 8, 8]}))
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(8, 9, name="foo"),
        pd.DataFrame({"foo1": 8, "foo2": [9, 8, 8]}))
    # from list
    pd.testing.assert_frame_equal(
        n.pair_segments_frame([3, 4, 5]),
        pd.DataFrame({1: [3, 4, 5], 2: 3}))
    pd.testing.assert_frame_equal(
        n.pair_segments_frame([3, 4, 5], name="foo"),
        pd.DataFrame({"foo1": [3, 4, 5], "foo2": 3}))
    pd.testing.assert_frame_equal(
        n.pair_segments_frame([3, 4, 5], {0: 6}),
        pd.DataFrame({1: [3, 4, 5], 2: [6, 3, 3]}))
    pd.testing.assert_frame_equal(
        n.pair_segments_frame([3, 4, 5], 6, "foo"),
        pd.DataFrame({"foo1": [3, 4, 5], "foo2": [6, 3, 3]}))
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(["$VAL1$", "$VAL2$", "$VAL3$"]),
        pd.DataFrame({1: ["$VAL1$", "$VAL2$", "$VAL3$"], 2: "$VAL1$"}))
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(["v1", "v2", "v3"], "o1"),
        pd.DataFrame({1: ["v1", "v2", "v3"], 2: ["o1", "v1", "v1"]}))
    with pytest.raises(ValueError, match=errmsg_value):
        n.pair_segments_frame([3, 4])
    with pytest.raises(ValueError, match=errmsg_value_out_expected):
        n.pair_segments_frame([3, 4, 5], [6, 7])
    # from Series
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(pd.Series([3, 4, 5])),
        pd.DataFrame({1: [3, 4, 5], 2: 3}))
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(pd.Series([3, 4, 5]), name="foo"),
        pd.DataFrame({"foo1": [3, 4, 5], "foo2": 3}))
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(pd.Series([3, 4, 5]), pd.Series(6)),
        pd.DataFrame({1: [3, 4, 5], 2: [6, 3, 3]}))
    # mixed with dict
    pd.testing.assert_frame_equal(
        n.pair_segments_frame([3, 4, 5], {0: 6}),
        pd.DataFrame({1: [3, 4, 5], 2: [6, 3, 3]}))
    with pytest.raises(ValueError, match=errmsg_value_out):
        n.pair_segments_frame(1, {3: 2})
    # discontinuous series
    pd.testing.assert_frame_equal(
        n.pair_segments_frame([3, 4, 5], {0: 6, 1: 7, 2: 8}),
        pd.DataFrame({1: [3, 4, 5], 2: [6, 7, 8]}))
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(1, {0: 2, 2: 3}),
        pd.DataFrame({1: [1, 1, 1], 2: [2, 1, 3]}))
    pd.testing.assert_frame_equal(
        n.pair_segments_frame([3.0, 4.0, 5.0], {1: 7, 2: 8}),
        pd.DataFrame({1: [3.0, 4.0, 5.0], 2: [3.0, 7.0, 8.0]}))
    with pytest.raises(ValueError, match=errmsg_value_out):
        n.pair_segments_frame(1, {0: 6, 3: 8})
    # constant
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(1, method="constant"),
        pd.DataFrame({1: [1, 1, 1], 2: [1, 1, 1]}))
    pd.testing.assert_frame_equal(
        n.pair_segments_frame([3, 2, 1], method="constant"),
        pd.DataFrame({1: [3, 2, 1], 2: [3, 2, 1]}))
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(1, 2, name="x", method="constant"),
        pd.DataFrame({"x1": [1, 1, 1], "x2": [2, 1, 1]}))
    # additive
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(1, method="additive"),
        pd.DataFrame({1: [1, 1, 1], 2: [1.0, 0.5, 0.5]}))
    pd.testing.assert_frame_equal(
        n.pair_segments_frame(1, {0: 2, 1: 10}, name="foo", method="additive"),
        pd.DataFrame({"foo1": [1, 1, 1], "foo2": [2.0, 10.0, 0.5]}))
    pd.testing.assert_frame_equal(
        n.pair_segments_frame([10.0, 2.0, 3.0], name="foo", method="additive"),
        pd.DataFrame({"foo1": [10.0, 2.0, 3.0], "foo2": [10.0, 4.0, 6.0]}))
    # misc errors
    with pytest.raises(ValueError, match="method must be one of "):
        n.pair_segments_frame(1, method="nope")
    with pytest.raises(ValueError, match="expected value to be scalar, list,"):
        n.pair_segments_frame(object())
    with pytest.raises(ValueError, match=errmsg_value_out_expected):
        n.pair_segments_frame(1, object())


def test_remove_condition():
    n = swn.SurfaceWaterNetwork.from_lines(valid_lines, valid_polygons)
    assert len(n) == 3
    # manually copy these, to compare later
    n.segments['orig_upstream_length'] = n.segments['upstream_length']
    n.segments['orig_upstream_area'] = n.segments['upstream_area']
    n.segments['orig_width'] = n.segments['width']
    n.remove(n.segments['upstream_area'] <= 1000.0)
    assert len(n) == 1
    assert len(n.segments) == 1
    assert len(n.catchments) == 1
    assert list(n.segments.index) == [0]
    assert n.segments.at[0, 'from_segnums'] == {1, 2}
    np.testing.assert_almost_equal(
        n.segments.at[0, 'upstream_length'], 87.67828936)
    np.testing.assert_almost_equal(
        n.segments.at[0, 'upstream_area'], 2200.0)
    np.testing.assert_almost_equal(
        n.segments.at[0, 'width'], 1.4615, 4)
    # Manually re-trigger these
    n.evaluate_upstream_length()
    n.evaluate_upstream_area()
    n.estimate_width()
    np.testing.assert_almost_equal(
        n.segments.at[0, 'upstream_length'], 20.0)
    np.testing.assert_almost_equal(
        n.segments.at[0, 'upstream_area'], 800.0)
    np.testing.assert_almost_equal(
        n.segments.at[0, 'width'], 1.4445, 4)
    np.testing.assert_almost_equal(
        n.segments.at[0, 'orig_upstream_length'], 87.67828936)
    np.testing.assert_almost_equal(
        n.segments.at[0, 'orig_upstream_area'], 2200.0)
    np.testing.assert_almost_equal(
        n.segments.at[0, 'orig_width'], 1.4615, 4)


def test_remove_segnums():
    n = swn.SurfaceWaterNetwork.from_lines(valid_lines)
    assert len(n) == 3
    n.remove(segnums=[1])
    assert len(n) == 2
    assert len(n.segments) == 2
    assert list(n.segments.index) == [0, 2]
    assert n.segments.at[0, 'from_segnums'] == {1, 2}
    # repeats are ok
    n = swn.SurfaceWaterNetwork.from_lines(valid_lines)
    n.remove(segnums=[1, 2, 1])
    assert len(n) == 1
    assert len(n.segments) == 1
    assert list(n.segments.index) == [0]
    assert n.segments.at[0, 'from_segnums'] == {1, 2}


def test_remove_errors(valid_n):
    n = valid_n
    assert len(n) == 3
    n.remove()  # no segments selected to remove; no changes made
    assert len(n) == 3
    n.remove(n.segments['dist_to_outlet'] > 100.0)  # dito, none selected
    assert len(n) == 3
    with pytest.raises(
            IndexError,
            match=r'1 segnums not found in segments\.index: \[3\]'):
        n.remove(segnums=[3])
    with pytest.raises(
            ValueError,
            match='all segments were selected to remove; must keep at least '):
        n.remove(segnums=[0, 1, 2])


# https://commons.wikimedia.org/wiki/File:Flussordnung_(Strahler).svg
fluss_gs = geopandas.GeoSeries(wkt.loads('''\
MULTILINESTRING(
    (380 490, 370 420), (300 460, 370 420), (370 420, 420 330),
    (190 250, 280 270), (225 180, 280 270), (280 270, 420 330),
    (420 330, 584 250), (520 220, 584 250), (584 250, 710 160),
    (740 270, 710 160), (735 350, 740 270), (880 320, 740 270),
    (925 370, 880 320), (974 300, 880 320), (760 460, 735 350),
    (650 430, 735 350), (710 160, 770 100), (700  90, 770 100),
    (770 100, 820  40))
''').geoms)


@pytest.fixture
def fluss_n():
    return swn.SurfaceWaterNetwork.from_lines(fluss_gs)


def test_fluss_n(fluss_n):
    n = fluss_n
    assert len(n.warnings) == 0
    assert len(n.errors) == 0
    assert len(n) == 19
    assert list(n.segments.index) == list(range(19))
    assert list(n.segments['to_segnum']) == \
        [2, 2, 6, 5, 5, 6, 8, 8, 16, 16, 9, 9, 11, 11, 10, 10, 18, 18, -1]
    assert list(n.segments['cat_group']) == [18] * 19
    assert list(n.segments['num_to_outlet']) == \
        [6, 6, 5, 6, 6, 5, 4, 4, 3, 3, 4, 4, 5, 5, 5, 5, 2, 2, 1]
    assert list(n.segments['sequence']) == \
        [4, 3, 12, 2, 1, 11, 15, 9, 16, 17, 14, 13, 6, 5, 8, 7, 18, 10, 19]
    assert list(n.segments['stream_order']) == \
        [1, 1, 2, 1, 1, 2, 3, 1, 3, 3, 2, 2, 1, 1, 1, 1, 4, 1, 4]
    assert list(n.headwater) == [0, 1, 3, 4, 7, 12, 13, 14, 15, 17]
    assert list(n.outlets) == [18]
    assert dict(n.to_segnums) == \
        {0: 2, 1: 2, 2: 6, 3: 5, 4: 5, 5: 6, 6: 8, 7: 8, 8: 16, 9: 16, 10: 9,
         11: 9, 12: 11, 13: 11, 14: 10, 15: 10, 16: 18, 17: 18}
    assert dict(n.from_segnums) == \
        {16: {8, 9}, 2: {0, 1}, 5: {3, 4}, 6: {2, 5}, 8: {6, 7}, 9: {10, 11},
         10: {14, 15}, 11: {12, 13}, 18: {16, 17}}
    assert repr(n) == dedent('''\
        <SurfaceWaterNetwork:
          19 segments: [0, 1, ..., 17, 18]
          10 headwater: [0, 1, ..., 15, 17]
          1 outlets: [18]
          no diversions />''')
    if matplotlib:
        _ = n.plot()
        plt.close()


def test_fluss_n_query_upstream(fluss_n):
    n = fluss_n
    assert set(n.query(upstream=0)) == {0}
    assert set(n.query(upstream=[2])) == {0, 1, 2}
    assert set(n.query(upstream=8)) == {0, 1, 2, 3, 4, 5, 6, 7, 8}
    assert set(n.query(upstream=9)) == {9, 10, 11, 12, 13, 14, 15}
    assert set(n.query(upstream=17)) == {17}
    assert len(set(n.query(upstream=18))) == 19
    # with barriers
    assert len(set(n.query(upstream=18, barrier=17))) == 18
    assert len(set(n.query(upstream=18, barrier=9))) == 13
    assert set(n.query(upstream=9, barrier=8)) == {9, 10, 11, 12, 13, 14, 15}
    assert set(n.query(upstream=16, barrier=[9, 5])) == \
        {0, 1, 2, 5, 6, 7, 8, 9, 16}
    # break it
    with pytest.raises(
            IndexError,
            match=r'upstream segnum \-1 not found in segments\.index'):
        n.query(upstream=-1)
    with pytest.raises(
            IndexError,
            match=r'2 upstream segments not found in segments\.index: \[19, '):
        n.query(upstream=[18, 19, 20])
    with pytest.raises(
            IndexError,
            match=r'barrier segnum \-1 not found in segments\.index'):
        n.query(upstream=18, barrier=-1)
    with pytest.raises(
            IndexError,
            match=r'1 barrier segment not found in segments\.index: \[\-1\]'):
        n.query(upstream=18, barrier=[-1, 15])


def test_fluss_n_query_downstream(fluss_n):
    n = fluss_n
    assert n.query(downstream=0) == [2, 6, 8, 16, 18]
    assert n.query(downstream=[2]) == [6, 8, 16, 18]
    assert n.query(downstream=8) == [16, 18]
    assert n.query(downstream=9) == [16, 18]
    assert n.query(downstream=17) == [18]
    assert n.query(downstream=18) == []
    assert set(n.query(downstream=7, gather_upstream=True)) == \
        {0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}
    assert set(n.query(downstream=8, gather_upstream=True)) == \
        {9, 10, 11, 12, 13, 14, 15, 16, 17, 18}
    assert set(n.query(downstream=[9], gather_upstream=True)) == \
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18}
    assert n.query(downstream=18, gather_upstream=True) == []
    assert set(n.query(downstream=0, gather_upstream=True, barrier=8)) == \
        {1, 2, 3, 4, 5, 6, 8}
    with pytest.raises(
            IndexError,
            match=r'downstream segnum \-1 not found in segments\.index'):
        n.query(downstream=-1)


def test_aggregate_fluss_headwater(fluss_n):
    n = fluss_n
    assert len(n) == 19
    na = n.aggregate([17, 4])
    assert len(na.warnings) == 0
    assert len(na.errors) == 0
    assert len(na) == 2
    assert list(na.segments.index) == [17, 4]
    assert list(na.headwater) == [17, 4]
    assert list(na.outlets) == [17, 4]
    assert list(na.segments['agg_patch']) == [[17], [4]]
    assert list(na.segments['agg_path']) == [[17], [4]]
    assert list(na.segments['agg_unpath']) == [[], []]


def test_aggregate_fluss_headwater_and_middle(fluss_n):
    n = fluss_n
    assert len(n) == 19
    na = n.aggregate([17, 18])
    assert len(na.warnings) == 0
    assert len(na.errors) == 0
    assert len(na) == 3
    assert list(na.segments.index) == [17, 18, 16]
    assert list(na.headwater) == [17, 16]
    assert list(na.outlets) == [18]
    assert [set(x) for x in na.segments['agg_patch']] == \
        [{17}, {18},
         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}]
    assert list(na.segments['agg_path']) == [[17], [18], [16, 8, 6, 5, 4]]
    assert list(na.segments['agg_unpath']) == [[], [16, 17], [9, 7, 2, 3]]


def test_aggregate_fluss_two_middle(fluss_n):
    n = fluss_n
    assert len(n) == 19
    na = n.aggregate([8, 9])
    assert len(na.warnings) == 1
    assert len(na.errors) == 0
    assert len(na) == 2
    assert list(na.segments.index) == [8, 9]
    assert list(na.headwater) == [8, 9]
    assert list(na.outlets) == [8, 9]
    assert [set(x) for x in na.segments['agg_patch']] == \
        [{0, 1, 2, 3, 4, 5, 6, 7, 8}, {9, 10, 11, 12, 13, 14, 15}]
    assert list(na.segments['agg_path']) == [[8, 6, 5, 4], [9, 11, 13]]
    assert list(na.segments['agg_unpath']) == [[7, 2, 3], [10, 12]]


def test_aggregate_fluss_disconnected(fluss_n):
    n = fluss_n
    assert len(n) == 19
    na = n.aggregate([5, 10, 17])
    assert len(na.warnings) == 0
    assert len(na.errors) == 0
    assert len(na) == 3
    assert list(na.segments.index) == [5, 10, 17]
    assert list(na.headwater) == [5, 10, 17]
    assert list(na.outlets) == [5, 10, 17]
    assert [set(x) for x in na.segments['agg_patch']] == \
        [{3, 4, 5}, {10, 14, 15}, {17}]
    assert list(na.segments['agg_path']) == [[5, 4], [10, 15], [17]]
    assert list(na.segments['agg_unpath']) == [[3], [14], []]


def test_aggregate_fluss_coarse(fluss_n):
    n = fluss_n
    assert len(n) == 19
    na = n.aggregate([5, 10, 18])
    assert len(na.warnings) == 0
    assert len(na.errors) == 0
    # extra junctions need to be added
    assert len(na) == 7
    assert list(na.segments.index) == [5, 10, 18, 8, 2, 9, 11]
    assert list(na.headwater) == [5, 10, 2,  11]
    assert list(na.outlets) == [18]
    assert [set(x) for x in na.segments['agg_patch']] == \
        [{3, 4, 5}, {10, 14, 15}, {16, 17, 18}, {6, 7, 8}, {0, 1, 2}, {9},
         {11, 12, 13}]
    assert list(na.segments['agg_path']) == \
        [[5, 4], [10, 15], [18, 16], [8, 6], [2, 1], [9], [11, 13]]
    assert list(na.segments['agg_unpath']) == \
        [[3], [14], [17, 8, 9], [7, 2, 5], [0], [10, 11], [12]]


def test_adjust_elevation_profile_errors(valid_n):
    with pytest.raises(
            ValueError,
            match='min_slope must be greater than zero'):
        valid_n.adjust_elevation_profile(0.0)

    n2d = swn.SurfaceWaterNetwork.from_lines(force_2d(valid_lines))
    with pytest.raises(
            AttributeError,
            match='line geometry does not have Z dimension'):
        n2d.adjust_elevation_profile()

    min_slope = pd.Series(2./1000, index=valid_n.segments.index)
    min_slope[1] = 3./1000
    min_slope.index += 1
    with pytest.raises(ValueError, match='index is different'):
        valid_n.adjust_elevation_profile(min_slope)


def test_adjust_elevation_profile_min_slope_float(valid_n):
    valid_n.adjust_elevation_profile(2./1000)


def test_adjust_elevation_profile_min_slope_series(valid_n):
    min_slope = pd.Series(2./1000, index=valid_n.segments.index)
    min_slope[1] = 3./1000
    valid_n.adjust_elevation_profile(min_slope)


def test_adjust_elevation_profile_no_change():
    lines = wkt_to_geoseries(['LINESTRING Z (0 0 8, 1 0 7, 2 0 6)'])
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    n.adjust_elevation_profile()
    assert len(n.messages) == 0
    assert (lines == n.segments.geometry).all()
    expected_profiles = wkt_to_geoseries(['LINESTRING (0 8, 1 7, 2 6)'])
    assert (n.profiles == expected_profiles).all()


def test_adjust_elevation_profile_use_min_slope():
    lines = wkt_to_geoseries(['LINESTRING Z (0 0 8, 1 0 9)'])
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    n.adjust_elevation_profile()
    n.profiles = round_coords(n.profiles)
    n.segments.geometry = round_coords(n.segments.geometry)
    assert len(n.messages) == 1
    assert n.messages[0] == \
        'segment 0: adjusted 1 coordinate elevation by 1.001'
    expected = wkt_to_geoseries(['LINESTRING Z (0 0 8, 1 0 7.999)'])
    assert (expected == n.segments.geometry).all()
    expected_profiles = wkt_to_geoseries(['LINESTRING (0 8, 1 7.999)'])
    assert (n.profiles == expected_profiles).all()

    lines = wkt_to_geoseries(['LINESTRING Z (0 0 8, 1 0 9, 2 0 6)'])
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    n.adjust_elevation_profile(0.1)
    assert len(n.messages) == 1
    assert n.messages[0] == \
        'segment 0: adjusted 1 coordinate elevation by 1.100'
    expected = wkt_to_geoseries(['LINESTRING Z (0 0 8, 1 0 7.9, 2 0 6)'])
    assert (expected == n.segments.geometry).all()
    expected_profiles = wkt_to_geoseries(['LINESTRING (0 8, 1 7.9, 2 6)'])
    assert (n.profiles == expected_profiles).all()

    lines = wkt_to_geoseries(['LINESTRING Z (0 0 8, 1 0 5, 2 0 6, 3 0 5)'])
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    n.adjust_elevation_profile(0.2)
    assert len(n.messages) == 1
    assert n.messages[0] == \
        'segment 0: adjusted 2 coordinate elevations between 0.400 and 1.200'
    expected = wkt_to_geoseries(
            ['LINESTRING Z (0 0 8, 1 0 5, 2 0 4.8, 3 0 4.6)'])
    assert (expected == n.segments.geometry).all()
    expected_profiles = wkt_to_geoseries(
            ['LINESTRING (0 8, 1 5, 2 4.8, 3 4.6)'])
    assert (n.profiles == expected_profiles).all()
