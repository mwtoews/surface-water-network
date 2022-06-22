"""Miscellaneous utility functions that don't fit anywhere else."""

__all__ = ["abbr_str", "is_location_frame"]

import geopandas
import pandas as pd


def abbr_str(lst, limit=15):
    """Return str of list that is abbreviated (if necessary)."""
    if isinstance(lst, list):
        is_set = False
    elif isinstance(lst, set):
        is_set = True
        lst = list(lst)
    else:
        raise TypeError(type(lst))
    if len(lst) <= limit:
        res = ', '.join(str(x) for x in lst)
    else:
        left = limit // 2
        right = left
        if left + right != limit:
            left += 1
        res = ', '.join(
            [str(x) for x in lst[:left]] + ['...'] +
            [str(x) for x in lst[-right:]])
    if is_set:
        return f"{{{res}}}"
    else:
        return f"[{res}]"


def is_location_frame(loc_df, geom_required=True):
    """Check if it is a location data frame, raise Exceptions if necessary.

    Parameters
    ----------
    loc_df : geopandas.GeoDataFrame or pandas.DataFrame
        Location [geo] dataframe, created by
        :py:meth:`SurfaceWaterNetwork.locate_geoms`.
    geom_required : bool, default True
        If True, loc_df must be a GeoDataFrame with a geometry column.

    Returns
    -------
    bool
        Always True, unless there is an exception.

    Raises
    ------
    TypeError
        Input must be either a GeoDataFrame or DataFrame (if geometry is not
        required).
    ValueError
        If the [geo] data frame has insufficent data.
    """
    if geom_required:
        if not isinstance(loc_df, geopandas.GeoDataFrame):
            raise TypeError("loc_df must be a GeoDataFrame")
        try:
            non_empty = ~loc_df.is_empty
        except AttributeError:
            raise TypeError("loc_df must be a GeoDataFrame")
    else:
        if not isinstance(loc_df, (geopandas.GeoDataFrame, pd.DataFrame)):
            raise TypeError("loc_df must be a GeoDataFrame or DataFrame")
    loc_df_columns = loc_df.columns
    if "segnum" not in loc_df_columns:
        raise ValueError(
            "loc_df must have 'segnum' column; "
            "was it created by n.locate_geoms?")
    elif "seg_ndist" not in loc_df_columns:
        raise ValueError(
            "loc_df must have 'seg_ndist' column; "
            "was it created by n.locate_geoms?")
    if geom_required:
        if not (loc_df[non_empty].geom_type == "LineString").all():
            raise ValueError("geometry expected to be LineString")
        num_coords = loc_df[non_empty].geometry.apply(lambda g: len(g.coords))
        if not (num_coords == 2).all():
            raise ValueError("geometry expected to only have two coordinates")
    return True
