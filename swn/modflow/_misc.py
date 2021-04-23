# -*- coding: utf-8 -*-
"""Misc utilities for MODFLOW."""

__all__ = ["geotransform_from_flopy"]

import pandas as pd
import numpy as np


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


def set_outreaches(reach_data, seg_data):
    """
    Determine the outreach for each SFR reach (requires a reachID
    column in reach_data). Uses the segment routing specified in segdata?!? TODO?.
    """
    rd = reach_data.sort_values(["iseg", "ireach"])
    # ensure that each segment starts with reach 1
    # reach_data = reset_reaches(reach_data)
    # ensure that all outsegs are segments, outlets, or negative (lakes)
    # seg_data = repair_outsegs(seg_data)
    # rd = reach_data
    outseg = seg_data.set_index('nseg').outseg.to_dict()  # make_graph(seg_data)  # TODO
    reach1IDs = dict(
        zip(rd[rd.ireach == 1].iseg, rd[rd.ireach == 1].index)
    )
    outreach = []
    for i in range(len(rd)):
        # if at the end of reach data or current segment
        if i + 1 == len(rd) or rd.ireach.values[i + 1] == 1:
            nextseg = outseg[rd.iseg.values[i]]  # get next segment
            if nextseg > 0:  # current reach is not an outlet
                nextrchid = reach1IDs[
                    nextseg
                ]  # get reach 1 of next segment
            else:
                nextrchid = 0
        else:  # otherwise, it's the next reachID
            nextrchid = rd.index[i + 1]
        outreach.append(nextrchid)
    rd["outreach"] = outreach
    return rd
