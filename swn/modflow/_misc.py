"""Misc utilities for MODFLOW."""

__all__ = [
    "geotransform_from_flopy",
    "transform_data_to_series_or_frame",
]

from collections.abc import Iterable

import pandas as pd

from ..util import abbr_str


def sfr_rec_to_df(sfr):
    """Convert flopy rec arrays for ds2 and ds6 to pandas dataframes."""
    d = sfr.segment_data
    # multi index
    reform = {(i, j): d[i][j] for i in d.keys() for j in d[i].dtype.names}
    segdatadf = pd.DataFrame.from_dict(reform)
    segdatadf.columns.names = ['kper', 'col']
    reachdatadf = pd.DataFrame.from_records(sfr.reach_data)
    return segdatadf, reachdatadf


def sfr_dfs_to_rec(model, segdatadf, reachdatadf, set_outreaches=False,
                   get_slopes=True, minslope=None):
    """Convert sfr ds6 and ds2 to model sfr rec.

    Function to convert sfr ds6 (seg data) and ds2 (reach data) to model.sfr
    rec arrays option to update slopes from reachdata dataframes
    """
    if get_slopes:
        print('Getting slopes')
        if minslope is None:
            minslope = 1.0e-4
            print(f'using default minslope of {minslope}')
        else:
            print(f'using specified minslope of {minslope}')
    # segs ds6
    # multiindex
    g = segdatadf.groupby(level=0, axis=1)  # group multi index df by kper
    model.sfr.segment_data = g.apply(
        lambda k: k.xs(k.name, axis=1).to_records(index=False)).to_dict()
    # # reaches ds2
    model.sfr.reach_data = reachdatadf.to_records(index=False)
    if set_outreaches:
        # flopy method to set/fix outreaches from segment routing
        # and reach number information
        model.sfr.set_outreaches()
    if get_slopes:
        model.sfr.get_slopes(minimum_slope=minslope)
    # as of 08/03/2018 flopy plotting of sfr plots whatever is in
    # stress_period_data; add
    model.sfr.stress_period_data.data[0] = model.sfr.reach_data


def geotransform_from_flopy(m):
    """Return GDAL-style geotransform from flopy model."""
    try:
        import flopy
    except ImportError:
        raise ImportError('this method requires flopy')
    if not isinstance(m, (flopy.mbase.BaseModel, flopy.mf6.MFModel)):
        raise TypeError("'m' must be a flopy model")
    mg = m.modelgrid
    if mg.angrot != 0.0:
        raise NotImplementedError('rotated grids not supported')
    if mg.delr.min() != mg.delr.max():
        raise ValueError('delr not uniform')
    if mg.delc.min() != mg.delc.max():
        raise ValueError('delc not uniform')
    a = mg.delr[0]
    b = 0.0
    c = mg.xoffset
    d = 0.0
    e = -mg.delc[0]
    f = mg.yoffset - e * mg.nrow
    # GDAL order of affine transformation coefficients
    return c, a, b, f, d, e


def transform_data_to_series_or_frame(
        data, dtype, time_index, mapping=None, ignore=None):
    """Check and transform "data" to a Series or time-varying DataFrame.

    Parameters
    ----------
    data: dict, pandas.Series, or pandas.DataFrame
        Variable input to transform. If a dict with scalar, this will be
        transformed to a series. If a dict with iterable, this will be
        transformed to a frame. Keep series or data frame types, but
        potentially modify data types and index or column names.
    dtype: type
        Data type to cast series or frame, such as float or int.
    time_index: pandas.DatetimeIndex
        Date time index for the end of each stress period.
    mapping: pandas.Series, optional, default None
        If provided, map identifiers for series index or frame columns.
    ignore: set, optional, default None
        If provided, ignore index items from mapping.

    Returns
    -------
    pandas.Series or pandas.DataFrame

    Raises
    ------
    KeyError
        If mapping is provided, check the dict keys, series index or frame
        columns to ensure they can be mapped to a new index. Raise KeyError
        if they are disjoint or only partially match.

    ValueError
        Raised if there is an issue with the dimensions, time index, or
        data types.
    """
    has_mapping = mapping is not None
    has_ignore = ignore is not None and len(ignore) > 0

    def check_keys(keys, keys_name, which):
        if keys_name is None:
            keys_s = set(keys)
        else:
            try:
                new_index = getattr(keys, keys_name)\
                    .astype(mapping.index.dtype)
                setattr(keys, keys_name, new_index)
            except (ValueError, TypeError):
                raise ValueError(
                      f"cannot cast {keys_name}.dtype to "
                      f"{mapping.index.dtype}")
            keys_s = set(new_index.values)
        idxname = mapping.index.name
        index_s = set(mapping.index)
        if has_ignore:
            index_s.update(ignore)
        if keys_s.isdisjoint(index_s):
            raise KeyError(
                f"{which} has a disjoint {idxname} set")
        diff_s = keys_s.difference(index_s)
        if diff_s:
            ldiff = len(diff_s)
            raise KeyError(
                "{} has {} key{} not found in {}: {}"
                .format(which, ldiff, "" if ldiff == 1 else "s",
                        idxname, abbr_str(diff_s)))
        return

    def return_series(data):
        do_astype = True
        if isinstance(data, dict):
            try:
                series = pd.Series(data, dtype=dtype)
                do_astype = False
            except ValueError:
                data = pd.Series(data)
        if do_astype:
            try:
                series = data.astype(dtype)
            except TypeError:
                series = data
        if has_mapping:
            if has_ignore:
                series.drop(index=ignore, errors="ignore", inplace=True)
            series.index = series.index.map(mapping)
        return series

    def return_frame(data):
        try:
            frame = data.astype(dtype)
        except TypeError:
            frame = data
        if has_mapping:
            if has_ignore:
                frame.drop(columns=ignore, errors="ignore", inplace=True)
            frame.columns = frame.columns.map(mapping)
        return frame

    if isinstance(data, dict):
        if len(data) == 0:
            return pd.Series(dtype=dtype)
        if has_mapping:
            check_keys(data.keys(), None, "dict")
        iterable_vals = list(isinstance(v, Iterable) for v in data.values())
        if any(iterable_vals):
            if not all(iterable_vals):
                raise ValueError("mixture of iterable and scalar values")
            val_lens = list(len(v) for v in data.values())
            if min(val_lens) != max(val_lens):
                raise ValueError("inconsistent lengths found in dict series")
            elif max(val_lens) != len(time_index):
                raise ValueError(
                    "length of dict series does not match time index")
            return return_frame(pd.DataFrame(data, index=time_index))
        return return_series(data)
    elif isinstance(data, pd.Series):
        if len(data) == 0:
            return pd.Series(dtype=dtype)
        # elif np.issubdtype(data.dtype, object):
        #    raise ValueError("dtype for series cannot be object")
        if has_mapping:
            check_keys(data, "index", "series")
        return return_series(data)
    elif isinstance(data, pd.DataFrame):
        if len(data.index) != len(time_index):
            raise ValueError("length of frame index does not match time index")
        elif isinstance(data.index, pd.DatetimeIndex):
            if not (data.index == time_index).all():
                raise ValueError("frame index does not match time index")
        elif (data.index == pd.RangeIndex(len(data.index))).all():
            # if this is a default index, just substitute time_index in
            data.index = time_index
        else:
            raise ValueError(
                "frame index should be a DatetimeIndex or a RangeIndex")
        if len(data.columns) == 0:
            return pd.DataFrame(dtype=dtype, index=time_index)
        if has_mapping:
            check_keys(data, "columns", "frame")
        return return_frame(data)
    raise ValueError("data must be a dict, Series or DataFrame")


def invert_series(series):
    """Inverts a series to swap index and values."""
    series_name = series.name
    assert series_name, series_name
    index_name = series.index.name
    assert index_name, index_name
    return series.reset_index().\
        set_index(series_name, verify_integrity=True)[index_name]


def tile_series_as_frame(series, index):
    """Repeats series along index."""
    frame = pd.concat([series.to_frame()] * len(index), axis=1).T
    frame.index = index
    return frame
