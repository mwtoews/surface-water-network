# -*- coding: utf-8 -*-
import numpy as np
import os
import pytest
from hashlib import md5
try:
    import flopy
except ImportError:
    pytest.skip("skipping tests that require flopy", allow_module_level=True)

from .common import swn, wkt_to_geoseries


datadir = os.path.join('tests', 'data')


def test_process_flopy_instance_errors():
    # same valid network used in test_basic
    lines = wkt_to_geoseries([
        'LINESTRING Z (60 100 14, 60  80 12)',
        'LINESTRING Z (40 130 15, 60 100 14)',
        'LINESTRING Z (70 130 15, 60 100 14)',
    ])
    n = swn.SurfaceWaterNetwork(lines)

    with pytest.raises(ValueError,
                       match=r'must be a flopy\.modflow\.mf\.Modflow object'):
        n.process_flopy(object())

    m = flopy.modflow.Modflow()
    with pytest.raises(ValueError, match='DIS package required'):
        n.process_flopy(m)

    flopy.modflow.ModflowDis(m, xul=10000, yul=10000)
    with pytest.raises(ValueError, match='BAS6 package required'):
        n.process_flopy(m)

    flopy.modflow.ModflowBas(m)
    with pytest.raises(ValueError, match='modelgrid extent does not cover '
                       'segments extent'):
        n.process_flopy(m)

    m.modelgrid.set_coord_info(xoff=0.0, yoff=0.0)
    with pytest.raises(ValueError, match='ibound_action must be one of'):
        n.process_flopy(m, ibound_action='foo')


@pytest.fixture
def costal_flopy_m():
    return flopy.modflow.Modflow.load('h.nam', model_ws=datadir, check=False)


def test_costal_flopy_m(costal_flopy_m):
    # Make sure this is the model we are thinking of
    m = costal_flopy_m
    assert m.modelgrid.extent == (1802000.0, 1819000.0, 5861000.0, 5879000.0)


def test_process_flopy(costal_swn, costal_flopy_m):
    n = costal_swn
    m = costal_flopy_m
    n.process_flopy(m)
    assert len(n.reaches) == 370
    assert m.bas6.ibound[0].array.sum() == 178
    assert md5(m.bas6.ibound.array.astype('B').tostring()).hexdigest() == \
        '3ca05914e75b930252257dd331111d95'


def test_process_flopy_ibound_modify(costal_swn):
    n = costal_swn
    m = flopy.modflow.Modflow.load('h.nam', model_ws=datadir, check=False)
    n.process_flopy(m, ibound_action='modify')
    assert len(n.reaches) == 626
    assert m.bas6.ibound[0].array.sum() == 243
    assert md5(m.bas6.ibound.array.astype('B').tostring()).hexdigest() == \
        '9666d6aff9266196a26c3726e5c1da94'
