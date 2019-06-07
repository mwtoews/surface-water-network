# -*- coding: utf-8 -*-
import pandas as pd
import pytest
import numpy as np
from shapely import wkt
from shapely.geometry import LineString

from .common import swn, wkt_to_dataframe, wkt_to_geoseries


@pytest.fixture
def wkt_list():
    # valid network
    return [
        'LINESTRING Z (60 100 14, 60  80 12)',
        'LINESTRING Z (40 130 15, 60 100 14)',
        'LINESTRING Z (70 130 15, 60 100 14)',
    ]


@pytest.fixture
def df(wkt_list):
    return wkt_to_dataframe(wkt_list)


@pytest.fixture
def lines(wkt_list):
    return wkt_to_geoseries(wkt_list)


@pytest.fixture
def n(lines):
    return swn.SurfaceWaterNetwork(lines)


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
    wkt_list = df['wkt'].tolist()
    wkt_list[1] = 'MULTILINESTRING Z ((70 130 15, 60 100 14))'
    lines = wkt_to_geoseries(wkt_list)
    with pytest.raises(ValueError, match='lines must all be LineString types'):
        swn.SurfaceWaterNetwork(lines)


def test_init_defaults(n):
    assert n.logger is not None
    assert len(n.warnings) == 0
    assert len(n.errors) == 0
    assert len(n) == 3
    assert n.has_z is True
    assert n.END_SEGNUM == -1
    assert n.index is n.segments.index
    assert list(n.index) == [0, 1, 2]
    assert list(n.segments['to_segnum']) == [-1, 0, 0]
    assert list(n.segments['cat_group']) == [0, 0, 0]
    assert list(n.segments['num_to_outlet']) == [1, 2, 2]
    np.testing.assert_allclose(
        n.segments['length_to_outlet'], [20.0, 56.05551, 51.622776])
    assert list(n.segments['sequence']) == [3, 1, 2]
    assert list(n.segments['stream_order']) == [2, 1, 1]
    assert list(n.headwater) == [1, 2]
    assert list(n.outlets) == [0]
    assert dict(n.to_segnums) == {1: 0, 2: 0}
    assert n.from_segnums == {0: set([1, 2])}
    n.adjust_elevation_profile()
    assert len(n.messages) == 0


def test_init_2D_geom(df):
    # Rewrite WKT as 2D
    lines = wkt_to_geoseries(
        df['wkt'].apply(wkt.loads).apply(wkt.dumps, output_dimension=2))
    n = swn.SurfaceWaterNetwork(lines)
    assert len(n.warnings) == 0
    assert len(n.errors) == 0
    assert len(n) == 3
    assert n.has_z is False
    assert list(n.index) == [0, 1, 2]
    assert list(n.segments['to_segnum']) == [-1, 0, 0]
    assert list(n.segments['cat_group']) == [0, 0, 0]
    assert list(n.segments['num_to_outlet']) == [1, 2, 2]
    np.testing.assert_allclose(
        n.segments['length_to_outlet'], [20.0, 56.05551, 51.622776])
    assert list(n.segments['sequence']) == [3, 1, 2]
    assert list(n.segments['stream_order']) == [2, 1, 1]
    assert list(n.headwater) == [1, 2]
    assert list(n.outlets) == [0]


def test_init_mismatch_3D():
    # Match in 2D, but not in Z dimension
    lines = wkt_to_geoseries([
        'LINESTRING Z (70 130 15, 60 100 14)',
        'LINESTRING Z (60 100 14, 60  80 12)',
        'LINESTRING Z (40 130 15, 60 100 13)',
    ])
    n = swn.SurfaceWaterNetwork(lines)
    assert len(n.warnings) == 1
    assert n.warnings[0] == \
        'end of segment 2 matches start of segment 1 in 2D, but not in '\
        'Z dimension'
    assert len(n.errors) == 0
    assert len(n) == 3
    assert n.has_z is True
    assert list(n.index) == [0, 1, 2]


def test_init_reversed_lines(lines):
    # same as the working lines, but reversed in the opposite direction
    lines = lines.geometry.apply(lambda x: LineString(x.coords[::-1]))
    n = swn.SurfaceWaterNetwork(lines)
    assert len(n.warnings) == 0
    assert len(n.errors) == 2
    assert n.errors[0] == \
        'segment 0 has more than one downstream segments: [1, 2]'
    assert n.errors[1] == \
        'starting coordinate (60.0, 100.0) matches start segment: ' + \
        str(set([1]))
    assert len(n) == 3
    assert n.has_z is True
    assert list(n.index) == [0, 1, 2]
    # This is all non-sense
    assert list(n.segments['to_segnum']) == [1, -1, -1]
    assert list(n.segments['cat_group']) == [1, 1, 2]
    assert list(n.segments['num_to_outlet']) == [2, 1, 1]
    np.testing.assert_allclose(
        n.segments['length_to_outlet'], [56.05551, 36.05551, 31.622776])
    assert list(n.segments['sequence']) == [1, 3, 2]
    assert list(n.segments['stream_order']) == [1, 1, 1]
    assert list(n.headwater) == [0, 2]
    assert list(n.outlets) == [1, 2]
    assert dict(n.to_segnums) == {0: 1}
    assert n.from_segnums == {1: set([0])}


def test_init_all_converge():
    # Lines all converge to the same place
    lines = wkt_to_geoseries([
        'LINESTRING Z (40 130 15, 60 100 15)',
        'LINESTRING Z (70 130 14, 60 100 14)',
        'LINESTRING Z (60  80 12, 60 100 14)',
    ])
    n = swn.SurfaceWaterNetwork(lines)
    assert len(n.warnings) == 5
    assert n.warnings[0] == \
        'ending segment 0 matches end of segment 1 '\
        'in 2D, but not in Z dimension'
    assert n.warnings[4] == \
        'ending coordinate (60.0, 100.0) matches end segments: ' + \
        str(set([0, 1, 2]))
    assert len(n.errors) == 0
    assert len(n) == 3
    assert n.has_z is True
    assert list(n.index) == [0, 1, 2]
    assert list(n.segments['to_segnum']) == [-1, -1, -1]
    assert list(n.segments['cat_group']) == [0, 1, 2]
    assert list(n.segments['num_to_outlet']) == [1, 1, 1]
    np.testing.assert_allclose(
        n.segments['length_to_outlet'], [36.05551, 31.622776, 20.0])
    assert list(n.segments['sequence']) == [1, 2, 3]
    assert list(n.segments['stream_order']) == [1, 1, 1]
    assert list(n.headwater) == [0, 1, 2]
    assert list(n.outlets) == [0, 1, 2]
    assert dict(n.to_segnums) == {}
    assert n.from_segnums == {}


def test_init_all_diverge():
    # Lines all diverge from the same place
    lines = wkt_to_geoseries([
        'LINESTRING Z (60 100 15, 40 130 14)',
        'LINESTRING Z (60 100 16, 70 130 14)',
        'LINESTRING Z (60 100 15, 60  80 12)',
    ])
    n = swn.SurfaceWaterNetwork(lines)
    assert len(n.warnings) == 4
    assert n.warnings[0] == \
        'starting segment 0 matches start of segment 1 in 2D, '\
        'but not in Z dimension'
    assert len(n.errors) == 1
    assert n.errors[0] == \
        'starting coordinate (60.0, 100.0) matches start segments: ' + \
        str(set([0, 1, 2]))
    assert len(n) == 3
    assert n.has_z is True
    assert list(n.index) == [0, 1, 2]
    assert list(n.segments['to_segnum']) == [-1, -1, -1]
    assert list(n.segments['num_to_outlet']) == [1, 1, 1]
    assert list(n.segments['cat_group']) == [0, 1, 2]
    np.testing.assert_allclose(
        n.segments['length_to_outlet'], [36.05551, 31.622776, 20.0])
    assert list(n.segments['sequence']) == [1, 2, 3]
    assert list(n.segments['stream_order']) == [1, 1, 1]
    assert list(n.headwater) == [0, 1, 2]
    assert list(n.outlets) == [0, 1, 2]
    assert dict(n.to_segnums) == {}
    assert n.from_segnums == {}


def test_init_line_connects_to_middle():
    lines = wkt_to_geoseries([
        'LINESTRING Z (40 130 15, 60 100 14, 60 80 12)',
        'LINESTRING Z (70 130 15, 60 100 14)',
    ])
    n = swn.SurfaceWaterNetwork(lines)
    assert len(n.warnings) == 0
    assert len(n.errors) == 1
    assert n.errors[0] == 'segment 1 connects to the middle of segment 0'
    assert len(n) == 2
    assert n.has_z is True
    assert list(n.index) == [0, 1]
    assert list(n.segments['to_segnum']) == [-1, -1]
    assert list(n.segments['cat_group']) == [0, 1]
    assert list(n.segments['num_to_outlet']) == [1, 1]
    np.testing.assert_allclose(
        n.segments['length_to_outlet'], [56.05551, 31.622776])
    assert list(n.segments['sequence']) == [1, 2]
    assert list(n.segments['stream_order']) == [1, 1]
    assert list(n.headwater) == [0, 1]
    assert list(n.outlets) == [0, 1]
    assert dict(n.to_segnums) == {}
    assert n.from_segnums == {}


def test_init_geoseries(wkt_list):
    gs = wkt_to_geoseries(wkt_list, geom_name='foo')
    n = swn.SurfaceWaterNetwork(gs)
    assert len(n.warnings) == 0
    assert len(n.errors) == 0
    assert len(n) == 3
    assert list(n.index) == [0, 1, 2]
    assert n.has_z is True
    v = pd.Series([3.0, 2.0, 4.0])
    a = n.accumulate_values(v)
    assert dict(a) == {0: 9.0, 1: 2.0, 2: 4.0}


def test_accumulate_values_must_be_series(n):
    with pytest.raises(ValueError, match='values must be a pandas Series'):
        n.accumulate_values([3.0, 2.0, 4.0])


def test_accumulate_values_different_index(n):
    # indexes don't completely overlap
    v = pd.Series([3.0, 2.0, 4.0])
    v.index += 1
    with pytest.raises(ValueError, match='index is different'):
        n.accumulate_values(v)
    # indexes overlap, but have a different sequence
    v = pd.Series([3.0, 2.0, 4.0]).sort_values()
    with pytest.raises(ValueError, match='index is different'):
        n.accumulate_values(v)


def test_accumulate_values_expected(n):
    v = pd.Series([2.0, 3.0, 4.0])
    a = n.accumulate_values(v)
    assert dict(a) == {0: 9.0, 1: 3.0, 2: 4.0}
    assert a.name is None


def test_segment_series(n):
    # from scalar
    v = n.segment_series(8.0)
    assert list(v.index) == [0, 1, 2]
    assert list(v) == [8.0, 8.0, 8.0]
    assert v.name is None
    v = n.segment_series('$VAL$')
    assert list(v) == ['$VAL$', '$VAL$', '$VAL$']
    # from list
    v = n.segment_series([3, 4, 5])
    assert list(v.index) == [0, 1, 2]
    assert list(v) == [3, 4, 5]
    assert v.name is None
    v = n.segment_series(['$VAL1$', '$VAL2$', '$VAL3$'])
    assert list(v.index) == [0, 1, 2]
    assert list(v) == ['$VAL1$', '$VAL2$', '$VAL3$']
    assert v.name is None
    # from Series
    s = pd.Series([2.0, 3.0, 4.0])
    v = n.segment_series(s)
    assert list(v.index) == [0, 1, 2]
    assert list(v) == [2.0, 3.0, 4.0]
    assert v.name is None
    s.name = 'foo'
    v = n.segment_series(s)
    assert v.name == 'foo'
    # now break it
    s.index += 1
    with pytest.raises(ValueError,
                       match='index is different than for segments'):
        n.segment_series(s)


def test_outlet_series():
    # make a network with outlet on index 2
    n = swn.SurfaceWaterNetwork(wkt_to_geoseries([
        'LINESTRING Z (40 130 15, 60 100 14)',
        'LINESTRING Z (70 130 15, 60 100 14)',
        'LINESTRING Z (60 100 14, 60  80 12)',
    ]))
    # from scalar
    v = n.outlet_series(8.0)
    assert list(v.index) == [2]
    assert list(v) == [8.0]
    assert v.name is None
    v = n.outlet_series('$VAL$')
    assert list(v) == ['$VAL$']
    # from list
    v = n.outlet_series([8])
    assert list(v.index) == [2]
    assert list(v) == [8]
    assert v.name is None
    v = n.outlet_series(['$VAL_out$'])
    assert list(v.index) == [2]
    assert list(v) == ['$VAL_out$']
    assert v.name is None
    # from Series
    s = pd.Series([5.0], index=[2])
    v = n.outlet_series(s)
    assert list(v.index) == [2]
    assert list(v) == [5.0]
    assert v.name is None
    s.name = 'foo'
    v = n.outlet_series(s)
    assert v.name == 'foo'
    # now break it
    s.index -= 1
    with pytest.raises(ValueError,
                       match='index is different than for outlets'):
        n.outlet_series(s)


def test_pair_segment_values(n):
    # from scalar
    p = n.pair_segment_values(8.0)
    assert list(p.columns) == [0, 1]
    assert list(p.index) == [0, 1, 2]
    expected = np.ones((3, 2)) * 8.0
    np.testing.assert_equal(p.to_numpy(), expected)
    p = n.pair_segment_values(8.0, 9.0)
    assert list(p.columns) == [0, 1]
    assert list(p.index) == [0, 1, 2]
    expected[0, 1] = 9.0
    np.testing.assert_equal(p.to_numpy(), expected)
    # from list
    p = n.pair_segment_values([3, 4, 5])
    assert list(p.columns) == [0, 1]
    assert list(p.index) == [0, 1, 2]
    expected = np.array([
            [3, 3],
            [4, 3],
            [5, 3]])
    np.testing.assert_equal(p.to_numpy(), expected)
    p = n.pair_segment_values([3, 4, 5], [6])
    assert list(p.columns) == [0, 1]
    assert list(p.index) == [0, 1, 2]
    expected[0, 1] = 6
    np.testing.assert_equal(p.to_numpy(), expected)
    p = n.pair_segment_values([3, 4, 5], 6)
    assert list(p.columns) == [0, 1]
    assert list(p.index) == [0, 1, 2]
    np.testing.assert_equal(p.to_numpy(), expected)
    p = n.pair_segment_values(['$VAL1$', '$VAL2$', '$VAL3$'])
    assert list(p.columns) == [0, 1]
    assert list(p.index) == [0, 1, 2]
    expected = np.array([
            ['$VAL1$', '$VAL1$'],
            ['$VAL2$', '$VAL1$'],
            ['$VAL3$', '$VAL1$']])
    np.testing.assert_equal(p.to_numpy(), expected)
    p = n.pair_segment_values(['$VAL1$', '$VAL2$', '$VAL3$'], ['$OUT1$'])
    assert list(p.columns) == [0, 1]
    assert list(p.index) == [0, 1, 2]
    expected[0, 1] = '$OUT1$'
    np.testing.assert_equal(p.to_numpy(), expected)
    return
    # TODO: from Series


def test_adjust_elevation_profile_min_slope_float(n):
    n.adjust_elevation_profile(2./1000)


def test_adjust_elevation_profile_min_slope_series(n):
    min_slope = pd.Series(2./1000, index=n.index)
    min_slope[1] = 3./1000
    n.adjust_elevation_profile(min_slope)


def test_adjust_elevation_profile_different_index(n):
    min_slope = pd.Series(2./1000, index=n.index)
    min_slope[1] = 3./1000
    min_slope.index += 1
    with pytest.raises(ValueError, match='index is different'):
        n.adjust_elevation_profile(min_slope)


def test_adjust_elevation_profile_min_slope_gt_zero(n):
    with pytest.raises(ValueError,
                       match='min_slope must be greater than zero'):
        n.adjust_elevation_profile(-2./1000)


def test_adjust_elevation_profile_no_change():
    lines = wkt_to_geoseries(['LINESTRING Z (0 0 8, 1 0 7, 2 0 6)'])
    n = swn.SurfaceWaterNetwork(lines)
    n.adjust_elevation_profile()
    assert len(n.messages) == 0
    assert (lines == n.segments.geometry).all()
    expected_profiles = wkt_to_geoseries(['LINESTRING (2 8, 1 7, 0 6)'])
    assert (n.profiles == expected_profiles).all()


def test_adjust_elevation_profile_use_min_slope():
    lines = wkt_to_geoseries(['LINESTRING Z (0 0 8, 1 0 9)'])
    n = swn.SurfaceWaterNetwork(lines)
    n.adjust_elevation_profile()
    assert len(n.messages) == 1
    assert n.messages[0] == 'adjusting 1 coordinate elevation in segment 0'
    expected = wkt_to_geoseries(['LINESTRING Z (0 0 8, 1 0 7.999)'])
    assert (expected == n.segments.geometry).all()
    expected_profiles = wkt_to_geoseries(['LINESTRING (1 8, 0 7.999)'])
    assert (n.profiles == expected_profiles).all()

    lines = wkt_to_geoseries(['LINESTRING Z (0 0 8, 1 0 9, 2 0 6)'])
    n = swn.SurfaceWaterNetwork(lines)
    n.adjust_elevation_profile(0.1)
    assert len(n.messages) == 1
    assert n.messages[0] == 'adjusting 1 coordinate elevation in segment 0'
    expected = wkt_to_geoseries(['LINESTRING Z (0 0 8, 1 0 7.9, 2 0 6)'])
    assert (expected == n.segments.geometry).all()
    expected_profiles = wkt_to_geoseries(['LINESTRING (2 8, 1 7.9, 0 6)'])
    assert (n.profiles == expected_profiles).all()

    lines = wkt_to_geoseries(['LINESTRING Z (0 0 8, 1 0 5, 2 0 6, 3 0 5)'])
    n = swn.SurfaceWaterNetwork(lines)
    n.adjust_elevation_profile(0.2)
    assert len(n.messages) == 1
    assert n.messages[0] == 'adjusting 2 coordinate elevations in segment 0'
    expected = wkt_to_geoseries(
            ['LINESTRING Z (0 0 8, 1 0 5, 2 0 4.8, 3 0 4.6)'])
    assert (expected == n.segments.geometry).all()
    expected_profiles = wkt_to_geoseries(
            ['LINESTRING (3 8, 2 5, 1 4.8, 0 4.6)'])
    assert (n.profiles == expected_profiles).all()


# def test_process_flopy_required(n):
#    with pytest.raises(ImportError, match='this method requires flopy'):
#        n.process_flopy(object())
