"""Test compat module."""

import geopandas
import numpy as np
import pandas as pd
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


def test_dataframe_str_na_zero_na():
    # no missing values, just make sure correct str-type (for pandas 3+)
    zero_na = pd.DataFrame({"a": ["one", "two"], "b": [1, 2]})
    zero_na_check = zero_na.copy()
    try:
        zero_na_check["a"] = zero_na_check["a"].astype(pd.StringDtype(na_value=pd.NA))
    except TypeError:
        zero_na_check["a"] = zero_na_check["a"].astype(pd.StringDtype())
    pd.testing.assert_frame_equal(compat.dataframe_str_na(zero_na), zero_na_check)
    # repeat with object dtype
    zero_na["a"] = zero_na["a"].astype(object)
    pd.testing.assert_frame_equal(compat.dataframe_str_na(zero_na), zero_na_check)


def test_dataframe_str_na_one_none():
    # only one missing value, originally supplied as None
    one_none = pd.DataFrame({"a": ["one", None], "b": [np.nan, 2.0]})
    one_none_check = one_none.copy()
    try:
        one_none_check["a"] = one_none_check["a"].astype(pd.StringDtype(na_value=pd.NA))
    except TypeError:
        one_none_check["a"] = one_none_check["a"].astype(pd.StringDtype())
        one_none_check.loc[1, "a"] = pd.NA
    pd.testing.assert_frame_equal(compat.dataframe_str_na(one_none), one_none_check)
    # repeat with object dtype
    one_none["a"] = one_none["a"].astype(object)
    pd.testing.assert_frame_equal(compat.dataframe_str_na(one_none), one_none_check)


def test_dataframe_str_na_one_na():
    # only one missing value, originally supplied as NA
    one_na = pd.DataFrame({"a": ["one", pd.NA], "b": [np.nan, 2]})
    one_na_check = pd.DataFrame({"a": ["one", pd.NA], "b": [np.nan, 2]})
    try:
        one_na_check["a"] = one_na_check["a"].astype(pd.StringDtype(na_value=pd.NA))
    except TypeError:
        one_na_check["a"] = one_na_check["a"].astype(pd.StringDtype())
    pd.testing.assert_frame_equal(compat.dataframe_str_na(one_na), one_na_check)
    # repeat with object dtype
    one_na["a"] = one_na["a"].astype(object)
    pd.testing.assert_frame_equal(compat.dataframe_str_na(one_na), one_na_check)


def test_dataframe_str_na_all_na():
    # all missing values supplied as NA stay as NA
    all_na = pd.DataFrame({"a": [pd.NA, pd.NA], "b": [1, pd.NA]})
    all_na_check = all_na.copy()
    try:
        all_na_check["a"] = all_na_check["a"].astype(pd.StringDtype(na_value=pd.NA))
    except TypeError:
        all_na_check["a"] = all_na_check["a"].astype(pd.StringDtype())
    pd.testing.assert_frame_equal(compat.dataframe_str_na(all_na), all_na_check)
    # repeat with object dtype
    all_na["a"] = all_na["a"].astype(object)
    pd.testing.assert_frame_equal(compat.dataframe_str_na(all_na), all_na_check)


def test_dataframe_str_na_all_none():
    # all missing values supplied as None change to NA
    all_none = pd.DataFrame({"a": [None, None], "b": [1, None]})
    all_none_check = pd.DataFrame({"a": [pd.NA, pd.NA], "b": [1.0, np.nan]})
    all_none_check["a"] = pd.NA
    try:
        all_none_check["a"] = all_none_check["a"].astype(pd.StringDtype(na_value=pd.NA))
    except TypeError:
        all_none_check["a"] = all_none_check["a"].astype(pd.StringDtype())
    pd.testing.assert_frame_equal(compat.dataframe_str_na(all_none), all_none_check)
    # repeat with object dtype
    all_none["a"] = all_none["a"].astype(object)
    pd.testing.assert_frame_equal(compat.dataframe_str_na(all_none), all_none_check)
