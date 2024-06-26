"""Test compat module."""

import geopandas
from shapely.geometry import Point

from swn import compat


def test_sjoin_idx_names():
    noname_idx_df = geopandas.GeoDataFrame(geometry=[Point(0, 1)])
    left_idx_name, right_idx_name = compat.sjoin_idx_names(noname_idx_df, noname_idx_df)
    assert left_idx_name == "index"
    assert right_idx_name == "index_right"
    sj = geopandas.sjoin(noname_idx_df, noname_idx_df)
    assert sj.index.name is None
    assert left_idx_name not in sj.columns
    assert right_idx_name in sj.columns

    name_idx_df = noname_idx_df.copy()
    name_idx_df.index.name = "idx"
    left_idx_name, right_idx_name = compat.sjoin_idx_names(name_idx_df, name_idx_df)
    if compat.GEOPANDAS_GE_100:
        assert left_idx_name == "idx_left"
        assert right_idx_name == "idx_right"
    else:
        assert left_idx_name == "idx"
        assert right_idx_name == "index_right"
    sj = geopandas.sjoin(name_idx_df, name_idx_df)
    assert sj.index.name == left_idx_name
    assert left_idx_name not in sj.columns
    assert right_idx_name in sj.columns

    left_idx_name, right_idx_name = compat.sjoin_idx_names(noname_idx_df, name_idx_df)
    assert left_idx_name == "index"
    if compat.GEOPANDAS_GE_100:
        assert right_idx_name == "idx"
    else:
        assert right_idx_name == "index_right"
    sj = geopandas.sjoin(noname_idx_df, name_idx_df)
    assert sj.index.name is None
    assert left_idx_name not in sj.columns
    assert right_idx_name in sj.columns

    left_idx_name, right_idx_name = compat.sjoin_idx_names(name_idx_df, noname_idx_df)
    assert left_idx_name == "idx"
    assert right_idx_name == "index_right"
    sj = geopandas.sjoin(name_idx_df, noname_idx_df)
    assert sj.index.name == left_idx_name
    assert left_idx_name not in sj.columns
    assert right_idx_name in sj.columns
