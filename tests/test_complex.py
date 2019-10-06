# -*- coding: utf-8 -*-
import numpy as np

from .conftest import swn


def test_init(coastal_swn, coastal_lines_gdf):
    n = coastal_swn
    assert len(n.warnings) == 1
    assert n.warnings[0].startswith('ending coordinate ')
    assert n.warnings[0].endswith(
            ' matches end segments: ' + str(set([3046736, 3046737])))
    assert len(n.errors) == 0
    assert len(n) == 304
    assert n.has_z is True
    assert n.END_SEGNUM == 0
    assert 0 not in n.segments.index
    assert 3046700 in n.segments.index
    assert len(n.headwater) == 154
    assert set(n.headwater).issuperset([3046700, 3046802, 3050418, 3048102])
    assert list(n.outlets) == [3046700, 3046737, 3046736]
    to_segnums = n.to_segnums
    to_segnums_d = dict(to_segnums)
    assert len(to_segnums) == 301
    assert len(n.from_segnums) == 150
    assert n.END_SEGNUM not in to_segnums_d
    assert n.END_SEGNUM not in n.from_segnums
    # one segment catchment
    assert 3046700 not in to_segnums_d
    assert 3046700 not in n.from_segnums
    # near outlet
    assert to_segnums_d[3046539] == 3046737
    assert to_segnums_d[3046745] == 3046737
    assert n.from_segnums[3046737] == set([3046539, 3046745])
    # at hedwater
    assert to_segnums_d[3047898] == 3047762
    assert to_segnums_d[3047899] == 3047762
    assert n.from_segnums[3047762] == set([3047898, 3047899])
    # three tributaries
    assert to_segnums_d[3048237] == 3048157
    assert to_segnums_d[3048250] == 3048157
    assert to_segnums_d[3048251] == 3048157
    assert n.from_segnums[3048157] == set([3048237, 3048250, 3048251])
    nto = n.segments['num_to_outlet']
    assert nto.min() == 1
    assert nto.max() == 32
    assert list(nto[nto == 1].index) == [3046700, 3046737, 3046736]
    assert list(nto[nto == 2].index) == [3046539, 3046745, 3046951, 3046952]
    assert list(nto[nto == 31].index) == [3050175, 3050176, 3050337, 3050338]
    assert list(nto[nto == 32].index) == [3050413, 3050418]
    cat_group = n.segments.groupby('cat_group').count()['to_segnum']
    assert len(cat_group) == 3
    assert dict(cat_group) == {3046700: 1, 3046737: 173, 3046736: 130}
    ln = n.segments['dist_to_outlet']
    np.testing.assert_almost_equal(ln.min(), 42.437, 3)
    np.testing.assert_almost_equal(ln.mean(), 11105.741, 3)
    np.testing.assert_almost_equal(ln.max(), 21077.749, 3)
    # supplied LENGTHDOWN is similar
    res = coastal_lines_gdf['LENGTHDOWN'] + \
        coastal_lines_gdf.geometry.length - ln
    np.testing.assert_almost_equal(res.min(), 0.0)
    np.testing.assert_almost_equal(res.max(), 15.00362636)
    assert list(n.segments['sequence'])[:6] == [149, 225, 152, 222, 145, 142]
    assert list(n.segments['sequence'])[-6:] == [156, 4, 155, 1, 3, 2]
    stream_order = n.segments.groupby('stream_order').count()['to_segnum']
    assert len(stream_order) == 5
    assert dict(stream_order) == {1: 154, 2: 72, 3: 46, 4: 28, 5: 4}
    np.testing.assert_array_equal(
        n.segments['stream_order'], coastal_lines_gdf['StreamOrde'])
    ul = n.segments['upstream_length']
    np.testing.assert_almost_equal(ul.min(), 45.010, 3)
    np.testing.assert_almost_equal(ul.mean(), 11381.843, 3)
    np.testing.assert_almost_equal(ul.max(), 144764.575, 3)
    assert 'upstream_area' not in n.segments.columns
    assert 'width' not in n.segments.columns


def test_accumulate_values(coastal_swn, coastal_lines_gdf):
    n = coastal_swn
    catarea = n.accumulate_values(coastal_lines_gdf['CATAREA'])
    res = catarea - coastal_lines_gdf['CUM_AREA']
    assert res.min() > -7.0
    assert res.max() < 7.0
    assert catarea.name == 'accumulated_CATAREA'


def test_catchment_polygons(coastal_lines_gdf, coastal_polygons_gdf):
    lines = coastal_lines_gdf.geometry
    polygons = coastal_polygons_gdf.geometry
    n = swn.SurfaceWaterNetwork(lines, polygons)
    cat_areas = n.catchments.area
    # only consider areas where polygons existed (some were filled in)
    nonzero = coastal_polygons_gdf['Area'] != 0.0
    np.testing.assert_allclose(
        cat_areas[nonzero], coastal_polygons_gdf.loc[nonzero, 'Area'])
    # this has a wider margin of error, but still relatively small
    up_cat_areas = n.accumulate_values(cat_areas)
    np.testing.assert_allclose(
        up_cat_areas, coastal_lines_gdf['CUM_AREA'], atol=7800.0)
    ua = n.segments['upstream_area']
    np.testing.assert_almost_equal(ua.min(), 180994.763, 3)
    np.testing.assert_almost_equal(ua.mean(), 7437127.120, 3)
    np.testing.assert_almost_equal(ua.max(), 100625129.836, 3)
    w = n.segments['width']
    np.testing.assert_almost_equal(w.min(), 1.831, 3)
    np.testing.assert_almost_equal(w.mean(), 3.435, 3)
    np.testing.assert_almost_equal(w.max(), 12.420, 3)
