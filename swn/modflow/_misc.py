# -*- coding: utf-8 -*-
"""Misc utilities for MODFLOW."""

__all__ = ["geotransform_from_flopy"]

import pandas as pd


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
            print('using default minslope of {}'.format(minslope))
        else:
            print('using specified minslope of {}'.format(minslope))
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
    if not isinstance(m, flopy.mbase.BaseModel):
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
