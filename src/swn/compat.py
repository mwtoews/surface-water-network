"""Compatibility module."""

import contextlib
import warnings

import geopandas
import numpy as np
import pandas as pd
import shapely
from packaging.version import Version

NUMPY_GE_121 = Version(np.__version__) >= Version("1.21")

if shapely is not None:
    SHAPELY_GE_20 = Version(shapely.__version__) >= Version("2.0")
    SHAPELY_LT_18 = Version(shapely.__version__) < Version("1.8")
else:
    SHAPELY_GE_20 = False
    SHAPELY_LT_18 = False

shapely_warning = None
if shapely is not None:
    try:
        from shapely.errors import ShapelyDeprecationWarning as shapely_warning
    except ImportError:
        pass

if shapely_warning is not None and not SHAPELY_GE_20:

    @contextlib.contextmanager
    def ignore_shapely_warnings_for_object_array():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "Iteration|The array interface|__len__",
                shapely_warning,
            )
            if NUMPY_GE_121:
                # warning from numpy for existing Shapely releases (this is
                # fixed with Shapely 1.8)
                warnings.filterwarnings(
                    "ignore",
                    "An exception was ignored while fetching",
                    DeprecationWarning,
                )
            yield

elif SHAPELY_LT_18 and NUMPY_GE_121:

    @contextlib.contextmanager
    def ignore_shapely_warnings_for_object_array():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "An exception was ignored while fetching",
                DeprecationWarning,
            )
            yield

else:

    @contextlib.contextmanager
    def ignore_shapely_warnings_for_object_array():
        yield


GEOPANDAS_GE_100 = Version(geopandas.__version__) >= Version("1.0.0")


def sjoin_idx_names(left_df, right_df):
    """Return left and right index names from :py:func:`geopandas.sjoin`.

    Handles breaking change from geopandas 1.0.0.
    """
    left_idx_name = left_df.index.name or "index"
    if GEOPANDAS_GE_100:
        right_idx_name = right_df.index.name or "index"
        # add _left/_right if needed
        if left_df.index.name and (
            left_idx_name == right_idx_name or left_idx_name in right_df.columns
        ):
            left_idx_name += "_left"
        if (
            right_df.index.name is None
            or right_idx_name in left_df.columns
            or right_idx_name == left_df.index.name
        ):
            right_idx_name += "_right"
    else:
        right_idx_name = "index_right"
    return left_idx_name, right_idx_name


def dataframe_str_na(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with consistent str-type columns.

    A str-type column is detected if all rows are NA or if the non-NA
    rows are str-type. These are cast with :py:class:`pandas.StringDtype` with
    (where feasible) `na_value=pd.NA` instead of NaN.
    """
    df = df.copy()
    for name, dtype in df.dtypes.items():
        if pd.api.types.is_object_dtype(dtype):
            isna = df[name].isna()
            types = set(df.loc[~isna, name].map(type).unique())
            if isna.all() or types == set({str}):
                try:
                    df[name] = df[name].astype(pd.StringDtype(na_value=pd.NA))
                except TypeError:
                    df[name] = df[name].astype(pd.StringDtype())
                    if isna.any():
                        df.loc[isna, name] = pd.NA
        elif pd.api.types.is_string_dtype(dtype):
            if hasattr(dtype, "na_value"):
                if dtype.na_value is not pd.NA:
                    try:
                        # pandas 3+: change to NA missing type (probably from NaN)
                        df[name] = df[name].astype(pd.StringDtype(na_value=pd.NA))
                    except TypeError:
                        df[name] = df[name].astype(pd.StringDtype())
            elif (isna := df[name].isna()).any():
                # pandas 2: ensure missing is NA
                df.loc[isna, name] = pd.NA
    return df
