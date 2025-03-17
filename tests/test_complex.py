from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

import swn

from .conftest import matplotlib, plt


def test_init(coastal_swn, coastal_lines_gdf):
    n = coastal_swn
    assert len(n.warnings) == 1
    assert n.warnings[0].startswith("ending coordinate ")
    assert n.warnings[0].endswith(" matches end segments: {3046736, 3046737}")
    assert len(n.errors) == 0
    assert len(n) == 304
    assert n.has_z is True
    assert n.END_SEGNUM == 0
    assert 0 not in n.segments.index
    assert 3046700 in n.segments.index
    assert len(n.headwater) == 154
    assert set(n.headwater).issuperset({3046700, 3046802, 3050418, 3048102})
    assert list(n.outlets) == [3046700, 3046737, 3046736]
    to_segnums = n.to_segnums
    from_segnums = n.from_segnums
    assert to_segnums.index.name == "nzsegment"
    assert from_segnums.index.name == "nzsegment"
    assert len(to_segnums) == 301
    assert len(from_segnums) == 150
    assert n.END_SEGNUM not in to_segnums
    assert n.END_SEGNUM not in from_segnums
    # one segment catchment
    assert 3046700 not in to_segnums
    assert 3046700 not in from_segnums
    # near outlet
    assert to_segnums[3046539] == 3046737
    assert to_segnums[3046745] == 3046737
    assert from_segnums[3046737] == {3046539, 3046745}
    # at hedwater
    assert to_segnums[3047898] == 3047762
    assert to_segnums[3047899] == 3047762
    assert from_segnums[3047762] == {3047898, 3047899}
    # three tributaries
    assert to_segnums[3048237] == 3048157
    assert to_segnums[3048250] == 3048157
    assert to_segnums[3048251] == 3048157
    assert from_segnums[3048157] == {3048237, 3048250, 3048251}
    nto = n.segments["num_to_outlet"]
    assert nto.min() == 1
    assert nto.max() == 32
    assert list(nto[nto == 1].index) == [3046700, 3046737, 3046736]
    assert list(nto[nto == 2].index) == [3046539, 3046745, 3046951, 3046952]
    assert list(nto[nto == 31].index) == [3050175, 3050176, 3050337, 3050338]
    assert list(nto[nto == 32].index) == [3050413, 3050418]
    cat_group = n.segments.groupby("cat_group").count()["to_segnum"]
    assert len(cat_group) == 3
    assert cat_group.to_dict() == {3046700: 1, 3046737: 173, 3046736: 130}
    ln = n.segments["dist_to_outlet"]
    np.testing.assert_almost_equal(ln.min(), 42.437, 3)
    np.testing.assert_almost_equal(ln.mean(), 11105.741, 3)
    np.testing.assert_almost_equal(ln.max(), 21077.749, 3)
    # supplied LENGTHDOWN is similar
    res = coastal_lines_gdf["LENGTHDOWN"] + coastal_lines_gdf.geometry.length - ln
    np.testing.assert_almost_equal(res.min(), 0.0)
    np.testing.assert_almost_equal(res.max(), 15.00362636)
    assert list(n.segments["sequence"])[:6] == [149, 225, 152, 222, 145, 142]
    assert list(n.segments["sequence"])[-6:] == [156, 4, 155, 1, 3, 2]
    stream_order = n.segments.groupby("stream_order").count()["to_segnum"]
    assert len(stream_order) == 5
    assert stream_order.to_dict() == {1: 154, 2: 72, 3: 46, 4: 28, 5: 4}
    np.testing.assert_array_equal(
        n.segments["stream_order"], coastal_lines_gdf["StreamOrde"]
    )
    ul = n.segments["upstream_length"]
    np.testing.assert_almost_equal(ul.min(), 45.010, 3)
    np.testing.assert_almost_equal(ul.mean(), 11381.843, 3)
    np.testing.assert_almost_equal(ul.max(), 144764.575, 3)
    assert "upstream_area" not in n.segments.columns
    assert "width" not in n.segments.columns
    assert repr(n) == dedent(
        """\
        <SurfaceWaterNetwork: with Z coordinates
          304 segments: [3046409, 3046455, ..., 3050338, 3050418]
          154 headwater: [3046409, 3046542, ..., 3050338, 3050418]
          3 outlets: [3046700, 3046737, 3046736]
          no diversions />"""
    )
    if matplotlib:
        _ = n.plot()
        plt.close()


def test_accumulate_values(coastal_swn, coastal_lines_gdf):
    n = coastal_swn
    catarea = n.accumulate_values(coastal_lines_gdf["CATAREA"])
    res = catarea - coastal_lines_gdf["CUM_AREA"]
    assert res.min() > -7.0
    assert res.max() < 7.0
    assert catarea.name == "accumulated_CATAREA"


def test_catchment_polygons(coastal_lines_gdf, coastal_polygons_gdf):
    lines = coastal_lines_gdf.geometry
    polygons = coastal_polygons_gdf.geometry
    n = swn.SurfaceWaterNetwork.from_lines(lines, polygons)
    cat_areas = n.catchments.area
    # only consider areas where polygons existed (some were filled in)
    nonzero = coastal_polygons_gdf["Area"] != 0.0
    np.testing.assert_allclose(
        cat_areas[nonzero], coastal_polygons_gdf.loc[nonzero, "Area"]
    )
    # this has a wider margin of error, but still relatively small
    up_cat_areas = n.accumulate_values(cat_areas)
    np.testing.assert_allclose(up_cat_areas, coastal_lines_gdf["CUM_AREA"], atol=7800.0)
    ua = n.segments["upstream_area"]
    np.testing.assert_almost_equal(ua.min(), 180994.763, 3)
    np.testing.assert_almost_equal(ua.mean(), 7437127.120, 3)
    np.testing.assert_almost_equal(ua.max(), 100625129.836, 3)
    w = n.segments["width"]
    np.testing.assert_almost_equal(w.min(), 1.831, 3)
    np.testing.assert_almost_equal(w.mean(), 3.435, 3)
    np.testing.assert_almost_equal(w.max(), 12.420, 3)
    assert repr(n) == dedent(
        """\
        <SurfaceWaterNetwork: with Z coordinates and catchment polygons
          304 segments: [3046409, 3046455, ..., 3050338, 3050418]
          154 headwater: [3046409, 3046542, ..., 3050338, 3050418]
          3 outlets: [3046700, 3046737, 3046736]
          no diversions />"""
    )
    if matplotlib:
        _ = n.plot()
        plt.close()


def test_coarsen_coastal_swn_w_poly(coastal_swn_w_poly):
    # level 1 gets back the same network, but with 'traced_segnums'
    # and 'catchment_segnums'
    nc1 = coastal_swn_w_poly.coarsen(1)
    assert len(nc1) == 304
    assert nc1.catchments is not None
    assert nc1.catchments.area.sum() == pytest.approx(165924652.6749345)
    pd.testing.assert_frame_equal(
        coastal_swn_w_poly.segments,
        nc1.segments.drop(columns=["traced_segnums", "catchment_segnums"]),
    )
    pd.testing.assert_series_equal(
        nc1.segments["traced_segnums"],
        nc1.segments.index.to_series().apply(lambda x: [x]),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        nc1.segments["catchment_segnums"],
        nc1.segments.index.to_series().apply(lambda x: [x]),
        check_names=False,
    )

    # coarsen to level 2
    nc2 = coastal_swn_w_poly.coarsen(2)
    assert len(nc2) == 66
    assert nc2.catchments.area.sum() == pytest.approx(165492135.76667663)
    assert set(nc2.segments.stream_order.unique()) == {2, 3, 4, 5}
    # validate one item
    item = nc2.segments.loc[3046539]
    assert item.geometry.length == pytest.approx(5670.908519171191)
    assert item.traced_segnums == [3046456, 3046455, 3046539]
    assert item.catchment_segnums == [
        3046456,
        3046604,
        3046605,
        3046455,
        3046409,
        3046539,
        3046542,
    ]

    # coarsen to level 3
    nc3 = coastal_swn_w_poly.coarsen(3)
    assert len(nc3) == 14
    assert nc3.catchments.area.sum() == pytest.approx(165491123.14988112)
    assert set(nc3.segments.stream_order.unique()) == {3, 4, 5}

    # coarsen to level 4
    nc4 = coastal_swn_w_poly.coarsen(4)
    assert len(nc4) == 4
    assert nc4.catchments.area.sum() == pytest.approx(165491123.14988112)
    assert set(nc4.segments.stream_order.unique()) == {4, 5}
