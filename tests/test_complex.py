# -*- coding: utf-8 -*-
import numpy as np


def test_init(costal_swn, costal_lines):
    n = costal_swn
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
    ln = n.segments['length_to_outlet']
    np.testing.assert_almost_equal(ln.min(), 42.43659279)
    np.testing.assert_almost_equal(ln.max(), 21077.7486858)
    # supplied LENGTHDOWN is similar
    res = costal_lines['LENGTHDOWN'] + costal_lines.geometry.length - ln
    np.testing.assert_almost_equal(res.min(), 0.0)
    np.testing.assert_almost_equal(res.max(), 15.00362636)
    assert list(n.segments['sequence'])[:6] == [149, 225, 152, 222, 145, 142]
    assert list(n.segments['sequence'])[-6:] == [156, 4, 155, 1, 3, 2]
    stream_order = n.segments.groupby('stream_order').count()['to_segnum']
    assert len(stream_order) == 5
    assert dict(stream_order) == {1: 154, 2: 72, 3: 46, 4: 28, 5: 4}
    np.testing.assert_array_equal(
        n.segments['stream_order'], costal_lines['StreamOrde'])


def test_accumulate_values(costal_swn, costal_lines):
    n = costal_swn
    catarea = n.accumulate_values(costal_lines['CATAREA'])
    res = catarea - costal_lines['CUM_AREA']
    assert res.min() > -7.0
    assert res.max() < 7.0
    assert catarea.name == 'accumulated_CATAREA'
