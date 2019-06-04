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


@pytest.fixture
def n():
    # same valid network used in test_basic
    lines = wkt_to_geoseries([
        'LINESTRING Z (60 100 14, 60  80 12)',
        'LINESTRING Z (40 130 15, 60 100 14)',
        'LINESTRING Z (70 130 15, 60 100 14)',
    ])
    return swn.SurfaceWaterNetwork(lines)


def test_process_flopy_instance_errors(n):

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


def test_process_flopy_n(n, tmpdir_factory):
    r"""
        .___.___.
        :  1:  2:  row 0
        :__\:_/_:
        :   \/  :  row 1
        :__ :0__:
        :   :|  :  row 2
        :___:|__:
      col 0 ' col 1

      Segment IDs: 0 (bottom), 1 & 2 (top)
    """
    m = flopy.modflow.Modflow()
    flopy.modflow.ModflowDis(
        m, nrow=3, ncol=2, delr=20, delc=20, xul=30, yul=130)
    flopy.modflow.ModflowBas(m)
    n.process_flopy(m)
    sfr = m.sfr
    # Data set 1c
    assert abs(sfr.nstrm) == 7
    assert sfr.nss == 3
    assert sfr.const == 86400.0
    # Data set 2
    # Base-0
    assert list(sfr.reach_data.node) == [0, 1, 3, 1, 3, 3, 5]
    assert list(sfr.reach_data.k) == [0, 0, 0, 0, 0, 0, 0]
    assert list(sfr.reach_data.i) == [0, 0, 1, 0, 1, 1, 2]
    assert list(sfr.reach_data.j) == [0, 1, 1, 1, 1, 1, 1]
    # Base-1
    assert list(sfr.reach_data.reachID) == [1, 2, 3, 4, 5, 6, 7]
    assert list(sfr.reach_data.iseg) == [1, 1, 1, 2, 2, 3, 3]
    assert list(sfr.reach_data.ireach) == [1, 2, 3, 1, 2, 1, 2]
    np.testing.assert_array_almost_equal(
        n.reach_data.rchlen,
        [18.027756, 6.009252, 12.018504, 21.081851, 10.540926, 10.0, 10.0])
    sd = m.sfr.segment_data[0]
    assert list(sd.nseg) == [1, 2, 3]
    assert list(sd.icalc) == [1, 1, 1]
    assert list(sd.outseg) == [3, 3, 0]
    assert list(sd.iupseg) == [0, 0, 0]
    assert list(sd.iprior) == [0, 0, 0]
    # TODO: more tests needed
    # Write output files
    outdir = tmpdir_factory.mktemp('out')
    m.model_ws = str(outdir)
    m.write_input()
    # sfr.write_file(str(outdir.join('file.sfr')))
    n.grid_cells.to_file(str(outdir.join('grid_cells.shp')))
    n.reaches.to_file(str(outdir.join('reaches.shp')))


@pytest.fixture
def costal_flopy_m():
    return flopy.modflow.Modflow.load('h.nam', model_ws=datadir, check=False)


def test_costal_process_flopy(costal_swn, costal_flopy_m):
    n = costal_swn
    m = costal_flopy_m
    # Make sure this is the model we are thinking of
    assert m.modelgrid.extent == (1802000.0, 1819000.0, 5861000.0, 5879000.0)
    n.process_flopy(m)
    assert len(n.reaches) == 370
    assert m.bas6.ibound[0].array.sum() == 178
    assert md5(m.bas6.ibound.array.astype('B').tostring()).hexdigest() == \
        '3ca05914e75b930252257dd331111d95'


def test_costal_process_flopy_ibound_modify(costal_swn):
    n = costal_swn
    m = flopy.modflow.Modflow.load('h.nam', model_ws=datadir, check=False)
    n.process_flopy(m, ibound_action='modify')
    assert len(n.reaches) == 626
    assert m.bas6.ibound[0].array.sum() == 243
    assert md5(m.bas6.ibound.array.astype('B').tostring()).hexdigest() == \
        '9666d6aff9266196a26c3726e5c1da94'


def test_process_flopy_lines_on_boundaries():
    m = flopy.modflow.Modflow()
    flopy.modflow.ModflowDis(
        m, nrow=3, ncol=3, delr=100, delc=100, xul=0, yul=300)
    flopy.modflow.ModflowBas(m)
    lines = wkt_to_geoseries([
        'LINESTRING (  0 320, 100 200)',
        'LINESTRING (100 200, 100 150, 150 100)',
        'LINESTRING (100 280, 100 200)',
        'LINESTRING (250 250, 150 150, 150 100)',
        'LINESTRING (150 100, 200   0, 300   0)',
    ])
    n = swn.SurfaceWaterNetwork(lines)
    n.process_flopy(m)
    assert m.sfr.nss == 5
    # TODO: code needs to be improved for this type of case
    # assert abs(m.sfr.nstrm) == 8
