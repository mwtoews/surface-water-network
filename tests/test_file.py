# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd

import pytest
from .conftest import datadir

import swn


def test_topnet2ts(coastal_flow_ts):
    pytest.importorskip('netCDF4')
    nc_fname = 'streamq_20170115_20170128_topnet_03046727_strahler1.nc'
    flow = swn.file.topnet2ts(os.path.join(datadir, nc_fname), 'mod_flow')
    assert flow.shape == (14, 304)
    # convert from m3/s to m3/day
    flow *= 24 * 60 * 60
    # remove time and truncat to closest day
    try:
        flow.index = flow.index.floor('d')
    except AttributeError:
        # older pandas
        flow.index = pd.to_datetime(
            flow.index.map(lambda x: x.strftime('%Y-%m-%d')))
    # Compare against CSV version of this data
    assert flow.shape == (14, 304)
    np.testing.assert_array_equal(flow.columns, coastal_flow_ts.columns)
    np.testing.assert_array_equal(flow.index, coastal_flow_ts.index)
    np.testing.assert_array_almost_equal(flow, coastal_flow_ts, 2)
