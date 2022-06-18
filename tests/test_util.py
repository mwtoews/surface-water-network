import geopandas
import pandas as pd
import pytest
from shapely.geometry import LineString

from swn.util import abbr_str, is_location_frame


def test_abbr_str():
    assert abbr_str(list(range(15))) == \
        "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]"
    assert abbr_str(list(range(16))) == \
        "[0, 1, 2, 3, 4, 5, 6, 7, ..., 9, 10, 11, 12, 13, 14, 15]"
    assert abbr_str(list(range(17))) == \
        "[0, 1, 2, 3, 4, 5, 6, 7, ..., 10, 11, 12, 13, 14, 15, 16]"
    assert abbr_str(set(range(4)), limit=4) == "{0, 1, 2, 3}"
    assert abbr_str(set(range(5)), limit=4) == "{0, 1, ..., 3, 4}"
    assert abbr_str(set(range(6)), limit=4) == "{0, 1, ..., 4, 5}"


def test_is_location_frame():
    gdf = geopandas.GeoDataFrame(
        {"segnum": [1, 2, 3], "seg_ndist": [0.1, 1.0, 0.0]},
        geometry=[
            LineString([(0, 0), (1, 1)]),
            LineString([(2, 2), (3, 3)]),
            LineString()],
    )
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    assert is_location_frame(gdf) is True
    assert is_location_frame(gdf, True) is True
    assert is_location_frame(gdf, False) is True
    assert is_location_frame(gdf, False) is True
    assert is_location_frame(df, False) is True
    assert is_location_frame(gdf.drop(columns="geometry"), False) is True
    with pytest.raises(TypeError, match="must be a GeoDataFrame"):
        is_location_frame(object())
    with pytest.raises(TypeError, match="must be a GeoDataFrame"):
        is_location_frame(df, True)
    with pytest.raises(ValueError, match="does not have geometry data"):
        is_location_frame(gdf.drop(columns="geometry"), True)
    with pytest.raises(ValueError, match="must have 'segnum' column"):
        is_location_frame(gdf.drop(columns="segnum"))
    with pytest.raises(ValueError, match="must have 'seg_ndist' column"):
        is_location_frame(gdf.drop(columns="seg_ndist"))
    gdf.geometry.at[2] = gdf.geometry.at[1].centroid
    with pytest.raises(ValueError, match="geometry expected to be LineString"):
        is_location_frame(gdf, True)
    gdf.geometry.at[2] = LineString([(0, 0), (1, 1), (3, 3)])
    with pytest.raises(
            ValueError, match="geometry expected to only have two coordinate"):
        is_location_frame(gdf, True)
