import pickle
from hashlib import md5
from shutil import which
from textwrap import dedent

import geopandas
import numpy as np
import pandas as pd
import pytest
from shapely import wkt
from shapely.geometry import Point

import swn
from swn.file import gdf_to_shapefile
from swn.spatial import force_2d, interp_2d_to_3d

from .conftest import datadir, matplotlib, plt

try:
    import flopy

    for block in flopy.mf6.ModflowGwfsfr.dfn:
        if block[0] == "block packagedata" and block[1] != "name packagedata":
            ridxname = block[1][5:]
            break
    else:
        raise ValueError("cannot determine reach index name for GWF-SFR")
    to_ridxname = f"to_{ridxname}"
    from_ridxsname = f"from_{ridxname}s"
    div_to_ridxsname = f"div_to_{ridxname}s"
    div_from_ridxname = f"div_from_{ridxname}"

except ImportError:
    pytest.skip("skipping tests that require flopy", allow_module_level=True)


mf6_exe = which("mf6")
requires_mf6 = pytest.mark.skipif(not mf6_exe, reason="requires mf6")

# same valid network used in test_basic
n3d_lines = geopandas.GeoSeries.from_wkt(
    [
        "LINESTRING Z (60 100 14, 60  80 12)",
        "LINESTRING Z (40 130 15, 60 100 14)",
        "LINESTRING Z (70 130 15, 60 100 14)",
    ]
)


def get_basic_swn(has_z: bool = True, has_diversions: bool = False):
    if has_z:
        n = swn.SurfaceWaterNetwork.from_lines(n3d_lines)
    else:
        n = swn.SurfaceWaterNetwork.from_lines(force_2d(n3d_lines))
    if has_diversions:
        diversions = geopandas.GeoDataFrame(
            geometry=[Point(58, 100), Point(62, 100), Point(61, 91), Point(59, 91)]
        )
        n.set_diversions(diversions=diversions)
    return n


def get_basic_modflow(outdir=".", with_top: bool = False, nper: int = 1):
    """Returns a basic Flopy MODFLOW 6 simulation and gwf model."""
    if with_top:
        top = np.array(
            [
                [16.0, 15.0],
                [15.0, 15.0],
                [14.0, 14.0],
            ]
        )
    else:
        top = 15.0
    sim = flopy.mf6.MFSimulation(exe_name=mf6_exe, sim_ws=str(outdir))
    _ = flopy.mf6.ModflowTdis(sim, nper=nper, time_units="days")
    gwf = flopy.mf6.ModflowGwf(sim, print_flows=True, save_flows=True)
    _ = flopy.mf6.ModflowIms(
        sim,
        outer_maximum=600,
        inner_maximum=100,
        outer_dvclose=1e-6,
        rcloserecord=0.1,
        relaxation_factor=1.0,
    )
    _ = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=1,
        nrow=3,
        ncol=2,
        delr=20.0,
        delc=20.0,
        length_units="meters",
        idomain=1,
        top=top,
        botm=10.0,
        xorigin=30.0,
        yorigin=70.0,
    )
    _ = flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord="model.hds",
        budget_filerecord="model.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    _ = flopy.mf6.ModflowGwfic(gwf, strt=15.0)
    _ = flopy.mf6.ModflowGwfnpf(gwf, k=1e-2, save_flows=True)
    _ = flopy.mf6.ModflowGwfrcha(gwf, recharge=1e-4, save_flows=True)
    return sim, gwf


def write_list(fname, data):
    with open(fname, "w") as f:
        for line in data:
            f.write(" " + " ".join(str(x) for x in line) + "\n")


def read_head(hed_fname, reaches=None):
    """Reads MODFLOW Head file

    If reaches is not None, it is modified inplace to add a "head" column

    Returns numpy array
    """
    with flopy.utils.HeadFile(hed_fname) as b:
        data = b.get_data()
    if reaches is not None:
        reaches["head"] = data[reaches["k"], reaches["i"], reaches["j"]]
    return data


def read_budget(bud_fname, text, reaches=None, colname=None):
    """Reads MODFLOW cell-by-cell file

    If reaches is not None, it is modified inplace to add data in "colname"

    Returns numpy array
    """
    with flopy.utils.CellBudgetFile(bud_fname) as b:
        urn = [x.strip() for x in b.get_unique_record_names(True)]
        try:
            res = b.get_data(text=text)
        except Exception as e:
            raise Exception(f"cannot read text={text}; use one of {urn}\n{e}")
        if len(res) != 1:
            from warnings import warn

            warn(f"get_data(text={text!r}) returned more than one array")
        data = res[0]
    if reaches is not None:
        if isinstance(data, np.recarray) and "q" in data.dtype.names:
            coldata = data["q"]
        else:
            coldata = data[reaches["k"], reaches["i"], reaches["j"]]
        if "mask" in reaches.columns:
            reaches.loc[~reaches["mask"], colname] = coldata
        else:
            reaches[colname] = coldata
    return data


def test_init_errors():
    with pytest.raises(ValueError, match="expected 'logger' to be Logger"):
        swn.SwnMf6(object())


def test_from_swn_flopy_errors():
    n = get_basic_swn()
    sim = flopy.mf6.MFSimulation(exe_name=mf6_exe)
    _ = flopy.mf6.ModflowTdis(sim, nper=4, time_units="days")
    gwf = flopy.mf6.ModflowGwf(sim)
    _ = flopy.mf6.ModflowGwfdis(
        gwf, nlay=1, nrow=3, ncol=2, delr=20.0, delc=20.0, idomain=1
    )

    with pytest.raises(ValueError, match="swn must be a SurfaceWaterNetwork object"):
        swn.SwnMf6.from_swn_flopy(object(), gwf)

    gwf.modelgrid.set_coord_info(epsg=2193)
    # n.segments.crs = {"init": "epsg:27200"}
    # with pytest.raises(
    #        ValueError,
    #        match="CRS for segments and modelgrid are different"):
    #    nm = swn.SwnMf6.from_swn_flopy(n, gwf)

    n.segments.crs = None
    with pytest.raises(
        ValueError, match="modelgrid extent does not cover segments extent"
    ):
        swn.SwnMf6.from_swn_flopy(n, gwf)

    gwf.modelgrid.set_coord_info(xoff=30.0, yoff=70.0)

    with pytest.raises(ValueError, match="idomain_action must be one of"):
        swn.SwnMf6.from_swn_flopy(n, gwf, idomain_action="foo")

    # finally success!
    swn.SwnMf6.from_swn_flopy(n, gwf)


@pytest.mark.parametrize("has_diversions", [False, True], ids=["nodiv", "div"])
def test_n3d_defaults(tmp_path, has_diversions):
    n = get_basic_swn(has_diversions=has_diversions)
    sim, gwf = get_basic_modflow(tmp_path, with_top=False)
    nm = swn.SwnMf6.from_swn_flopy(n, gwf)
    # check object reaches
    nodiv_expected = pd.DataFrame(
        {
            "segnum": [1, 1, 1, 2, 2, 0, 0],
            # k,i,j are base-0
            "k": 0,
            "i": [0, 0, 1, 0, 1, 1, 2],
            "j": [0, 1, 1, 1, 1, 1, 1],
            "iseg": [1, 1, 1, 2, 2, 3, 3],
            "ireach": [1, 2, 3, 1, 2, 1, 2],
            "rlen": [18.027756, 6.009252, 12.018504, 21.081851, 10.5409255, 10.0, 10.0],
            to_ridxname: [2, 3, 6, 5, 6, 7, 0],
            from_ridxsname: [set(), {1}, {2}, set(), {4}, {3, 5}, {6}],
        },
        index=pd.RangeIndex(1, 8, name=ridxname),
    )
    if not has_diversions:
        pd.testing.assert_frame_equal(
            nm.reaches[nodiv_expected.columns], nodiv_expected
        )
    else:
        div_expected = pd.concat(
            [
                nodiv_expected,
                pd.DataFrame(
                    {
                        "segnum": -1,
                        "k": 0,
                        "i": [1, 1, 1, 1],
                        "j": [1, 1, 1, 1],
                        "iseg": 0,
                        "ireach": 0,
                        "rlen": 1.0,
                        to_ridxname: 0,
                        from_ridxsname: [set(), set(), set(), set()],
                    },
                    index=pd.RangeIndex(8, 12, name=ridxname),
                ),
            ]
        )
        div_expected["diversion"] = [False] * 7 + [True] * 4
        div_expected["divid"] = [0] * 7 + [0, 1, 2, 3]
        div_expected[div_from_ridxname] = [0] * 7 + [3, 5, 6, 6]
        div_expected[div_to_ridxsname] = [
            set(),
            set(),
            {8},
            set(),
            {9},
            {10, 11},
            set(),
            set(),
            set(),
            set(),
            set(),
        ]
        pd.testing.assert_frame_equal(nm.reaches[div_expected.columns], div_expected)
    with pytest.raises(KeyError, match="missing 6 packagedata reaches series"):
        nm.packagedata_frame("native")
    nm.set_reach_slope()
    with pytest.raises(KeyError, match="missing 5 packagedata reaches series"):
        nm.packagedata_frame("native")
    nm.default_packagedata(hyd_cond1=1e-4)
    nodiv_expected = pd.DataFrame(
        {
            "rwid": 10.0,
            "rgrd": [
                0.027735,
                0.027735,
                0.027735,
                0.0316227766,
                0.0316227766,
                0.1,
                0.1,
            ],
            "rtp": [14.75, 14.416667, 14.166667, 14.666667, 14.166667, 13.5, 12.5],
            "rbth": 1.0,
            "rhk": 1e-4,
            "man": 0.024,
            "ncon": [1, 2, 2, 1, 2, 3, 1],
            "ndv": 0,
        },
        index=pd.RangeIndex(1, 8, name=ridxname),
    )
    if not has_diversions:
        pd.testing.assert_frame_equal(
            nm.packagedata_frame("native")[nodiv_expected.columns], nodiv_expected
        )
    else:
        div_expected = pd.concat(
            [
                nodiv_expected,
                pd.DataFrame(
                    {
                        "rwid": 1.0,
                        "rgrd": 0.001,
                        "rtp": [15.0, 15.0, 15.0, 15.0],
                        "rbth": 1.0,
                        "rhk": 0.0,
                        "man": 0.024,
                        "ncon": 1,
                        "ndv": 0,
                    },
                    index=pd.RangeIndex(8, 12, name=ridxname),
                ),
            ]
        )
        div_expected["ncon"] = [1, 2, 3, 1, 3, 5, 1, 1, 1, 1, 1]
        div_expected["ndv"] = [0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0]
        pd.testing.assert_frame_equal(
            nm.packagedata_frame("native")[div_expected.columns], div_expected
        )
    # Write native MF6 file and flopy datasets
    nm.write_packagedata(tmp_path / "packagedata.dat")
    fpd = nm.flopy_packagedata()
    # Don't check everything
    assert isinstance(fpd, list)
    assert len(fpd) == 11 if has_diversions else 7
    assert isinstance(fpd[0], list)
    assert len(fpd[0]) == 12
    assert fpd[-1][0] == 10 if has_diversions else 6
    assert fpd[0][1] == (0, 0, 0)
    # Write native MF6 file and flopy datasets
    nm.write_connectiondata(tmp_path / "connectiondata.dat")
    nodiv_expected = [
        [0, -1],
        [1, 0, -2],
        [2, 1, -5],
        [3, -4],
        [4, 3, -5],
        [5, 2, 4, -6],
        [6, 5],
    ]
    if not has_diversions:
        assert nm.flopy_connectiondata() == nodiv_expected
    else:
        div_expected = nodiv_expected + [[7, 2], [8, 4], [9, 5], [10, 5]]
        div_expected[2].append(-7)
        div_expected[4].append(-8)
        div_expected[5].extend([-9, -10])
        assert nm.flopy_connectiondata() == div_expected
    nm.write_diversions(tmp_path / "diversions.dat")
    if not has_diversions:
        assert nm.diversions_frame("native").shape == (0, 4)
        assert nm.flopy_diversions() == []
    else:
        pd.testing.assert_frame_equal(
            nm.diversions_frame("native"),
            pd.DataFrame(
                {
                    ridxname: [3, 5, 6, 6],
                    "idv": [1, 1, 1, 2],
                    "iconr": [8, 9, 10, 11],
                    "cprior": "upto",
                }
            ),
        )
        expected = [
            [2, 0, 7, "upto"],
            [4, 0, 8, "upto"],
            [5, 0, 9, "upto"],
            [5, 1, 10, "upto"],
        ]
        assert nm.flopy_diversions() == expected
    if not has_diversions:
        assert repr(nm) == dedent(
            f"""\
            <SwnMf6: flopy mf6 'model'
              7 in reaches ({ridxname}): [1, 2, ..., 6, 7]
              1 stress period with perlen: [1.0] days />"""
        )
    else:
        assert repr(nm) == dedent(
            f"""\
            <SwnMf6: flopy mf6 'model'
              11 in reaches ({ridxname}): [1, 2, ..., 10, 11]
              4 in diversions (iconr): [8, 9, 10, 11]
              1 stress period with perlen: [1.0] days />"""
        )
    if matplotlib:
        _ = nm.plot()
        plt.close()
    # Use with flopy
    for use_open_close in [False, True]:
        gwf.remove_package("sfr")
        if use_open_close:
            # gwf.name = "openclose"
            # gwf.rename_all_packages("openclose")
            if has_diversions:
                diversions = {"filename": "diversions.dat"}
            else:
                diversions = None
            nm.set_sfr_obj(
                print_input=True,
                save_flows=True,
                stage_filerecord="model.sfr.bin",
                budget_filerecord="model.sfr.bud",
                maximum_iterations=100,
                maximum_picard_iterations=10,
                packagedata={"filename": "packagedata.dat"},
                connectiondata={"filename": "connectiondata.dat"},
                diversions=diversions,
                perioddata={},
            )
        else:
            nm.set_sfr_obj(
                print_input=True,
                save_flows=True,
                stage_filerecord="model.sfr.bin",
                budget_filerecord="model.sfr.bud",
                maximum_iterations=100,
                maximum_picard_iterations=10,
                perioddata={},
            )
        # Run model and read outputs
        if mf6_exe:
            sim.write_simulation()
            success, buff = sim.run_simulation()
            assert success
    # Write some files
    gdf_to_shapefile(
        nm.reaches[nm.reaches.geom_type == "LineString"], tmp_path / "reaches.shp"
    )
    gdf_to_shapefile(nm.segments, tmp_path / "segments.shp")
    nm.grid_cells.to_file(tmp_path / "grid_cells.shp")
    if not mf6_exe:
        return
    # Check results
    head = read_head(tmp_path / "model.hds")
    sl = read_budget(tmp_path / "model.cbc", "SFR", nm.reaches, "sfrleakage")
    sf = read_budget(tmp_path / "model.sfr.bud", "GWF", nm.reaches, "sfr_Q")
    print(pd.DataFrame(sl))
    print(pd.DataFrame(sf))
    assert head.shape == (1, 3, 2)
    np.testing.assert_array_almost_equal(
        head,
        np.array(
            [
                [
                    [17.10986328, 16.77658646],
                    [17.49119547, 16.81176342],
                    [17.75195983, 17.2127244],
                ]
            ]
        ),
    )
    expected = np.array(
        [
            -0.04240276,
            -0.01410363,
            -0.03159486,
            -0.04431906,
            -0.02773725,
            -0.03292868,
            -0.04691372,
        ]
    )
    if has_diversions:
        expected = np.concatenate([expected, np.zeros(4, float)])
    np.testing.assert_array_almost_equal(sl["q"], expected)
    expected = np.array(
        [
            0.04240276,
            0.01410362,
            0.03159486,
            0.04431904,
            0.02773725,
            0.03292867,
            0.04691372,
        ]
    )
    if has_diversions:
        expected = np.concatenate([expected, np.zeros(4, float)])
    np.testing.assert_array_almost_equal(sf["q"], expected)


def test_model_property():
    nm = swn.SwnMf6()
    with pytest.raises(ValueError, match="model must be a flopy.mf6.MFModel object"):
        nm.model = 0

    sim = flopy.mf6.MFSimulation()
    gwf = flopy.mf6.MFModel(sim)

    with pytest.raises(ValueError, match="TDIS package required"):
        nm.model = gwf

    _ = flopy.mf6.ModflowTdis(
        sim, nper=1, time_units="days", start_date_time="2001-02-03"
    )

    with pytest.raises(ValueError, match="DIS package required"):
        nm.model = gwf

    _ = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=1,
        nrow=3,
        ncol=2,
        delr=20.0,
        delc=20.0,
        length_units="meters",
        top=15.0,
        botm=10.0,
        xorigin=30.0,
        yorigin=70.0,
    )

    with pytest.raises(ValueError, match="DIS idomain has no data"):
        nm.model = gwf

    gwf.dis.idomain.set_data(1)

    assert not hasattr(nm, "time_index")
    assert not hasattr(nm, "grid_cells")

    # Success!
    nm.model = gwf

    pd.testing.assert_index_equal(
        nm.time_index, pd.DatetimeIndex(["2001-02-03"], dtype="datetime64[ns]")
    )
    assert nm.grid_cells.shape == (6, 2)

    # Swap model with same and with another
    # same object
    nm.model = gwf

    tdis_args = {"nper": 1, "time_units": "days", "start_date_time": "2001-02-03"}
    dis_args = {
        "nlay": 1,
        "nrow": 3,
        "ncol": 2,
        "delr": 20.0,
        "delc": 20.0,
        "length_units": "meters",
        "xorigin": 30.0,
        "yorigin": 70.0,
        "idomain": 1,
    }
    sim = flopy.mf6.MFSimulation()
    gwf = flopy.mf6.MFModel(sim)
    _ = flopy.mf6.ModflowTdis(sim, **tdis_args)
    _ = flopy.mf6.ModflowGwfdis(gwf, **dis_args)
    # this is allowed
    nm.model = gwf

    tdis_args_replace = {"nper": 2}
    for vn, vr in tdis_args_replace.items():
        print(f"{vn}: {vr}")
        tdis_args_use = tdis_args.copy()
        tdis_args_use[vn] = vr
        sim = flopy.mf6.MFSimulation()
        gwf = flopy.mf6.MFModel(sim)
        _ = flopy.mf6.ModflowTdis(sim, **tdis_args_use)
        _ = flopy.mf6.ModflowGwfdis(gwf, **dis_args)
        # this is not allowed
        with pytest.raises(AttributeError, match="properties are too differe"):
            nm.model = gwf
    dis_args_replace = {
        "nrow": 4,
        "ncol": 3,
        "delr": 30.0,
        "delc": 40.0,
        "xorigin": 20.0,
        "yorigin": 60.0,
    }
    for vn, vr in dis_args_replace.items():
        dis_args_use = dis_args.copy()
        dis_args_use[vn] = vr
        sim = flopy.mf6.MFSimulation()
        gwf = flopy.mf6.MFModel(sim)
        _ = flopy.mf6.ModflowTdis(sim, **tdis_args)
        _ = flopy.mf6.ModflowGwfdis(gwf, **dis_args_use)
        # this is not allowed
        with pytest.raises(AttributeError, match="properties are too differe"):
            nm.model = gwf


def test_time_index():
    n = get_basic_swn()
    sim, gwf = get_basic_modflow(nper=12)
    sim.tdis.start_date_time.set_data("1999-07-01")
    perioddata = np.ones(12, dtype=sim.tdis.perioddata.dtype)
    perioddata["perlen"] = [31, 31, 30, 31, 30, 31, 31, 29, 31, 30, 31, 30]
    sim.tdis.perioddata.set_data(perioddata)
    nm = swn.SwnMf6.from_swn_flopy(n, gwf)
    assert nm.time_index.freqstr == "MS"  # "month start" or <MonthBegin>
    assert list(nm.time_index.day) == [1] * 12
    assert list(nm.time_index.month) == list((np.arange(12) + 6) % 12 + 1)
    assert list(nm.time_index.year) == [1999] * 6 + [2000] * 6


def test_set_reach_data_from_array():
    n = get_basic_swn()
    sim, gwf = get_basic_modflow(with_top=False)
    nm = swn.SwnMf6.from_swn_flopy(n, gwf)
    ar = np.arange(6).reshape((3, 2)) + 8.0
    nm.set_reach_data_from_array("test", ar)
    assert list(nm.reaches["test"]) == [8.0, 9.0, 11.0, 9.0, 11.0, 11.0, 13.0]


def test_n2d_defaults(tmp_path):
    # similar to 3D version, but getting information from model
    n = get_basic_swn(has_z=False)
    sim, gwf = get_basic_modflow(tmp_path, with_top=True)
    nm = swn.SwnMf6.from_swn_flopy(n, gwf)
    # check object reaches
    r = nm.reaches
    assert len(r) == 7
    # i and j are base-0
    assert list(r.i) == [0, 0, 1, 0, 1, 1, 2]
    assert list(r.j) == [0, 1, 1, 1, 1, 1, 1]
    assert list(r.segnum) == [1, 1, 1, 2, 2, 0, 0]
    assert list(r[to_ridxname]) == [2, 3, 6, 5, 6, 7, 0]
    nm.default_packagedata(hyd_cond1=2.0, thickness1=2.0)
    nm.set_reach_data_from_array("rtp", gwf.dis.top.array - 1.0)
    np.testing.assert_array_almost_equal(
        nm.reaches["rlen"],
        [18.027756, 6.009252, 12.018504, 21.081851, 10.540926, 10.0, 10.0],
    )
    np.testing.assert_array_almost_equal(nm.reaches["rwid"], [10.0] * 7)
    np.testing.assert_array_almost_equal(
        nm.reaches["rgrd"], [0.070711, 0.05, 0.025, 0.05, 0.025, 0.025, 0.05]
    )
    np.testing.assert_array_almost_equal(
        nm.reaches["rtp"], [15.0, 14.0, 14.0, 14.0, 14.0, 14.0, 13.0]
    )
    np.testing.assert_array_almost_equal(nm.reaches["rbth"], [2.0] * 7)
    np.testing.assert_array_almost_equal(nm.reaches["rhk"], [2.0] * 7)
    np.testing.assert_array_almost_equal(nm.reaches["man"], [0.024] * 7)
    # Use with flopy
    nm.set_sfr_obj(
        print_input=True,
        save_flows=True,
        stage_filerecord="model.sfr.bin",
        budget_filerecord="model.sfr.bud",
        maximum_iterations=100,
        maximum_picard_iterations=10,
    )
    if mf6_exe:
        # Run model
        sim.write_simulation()
        success, buff = sim.run_simulation()
        assert success
        # check outputs?


def test_packagedata(tmp_path):
    n = get_basic_swn()
    sim, gwf = get_basic_modflow(tmp_path)
    nm = swn.SwnMf6.from_swn_flopy(n, gwf)
    nm.default_packagedata()
    nm.set_sfr_obj()

    partial_expected_cols = [
        "rlen",
        "rwid",
        "rgrd",
        "rtp",
        "rbth",
        "rhk",
        "man",
        "ncon",
        "ustrf",
        "ndv",
    ]
    assert (
        list(gwf.sfr.packagedata.array.dtype.names)
        == [ridxname, "cellid"] + partial_expected_cols
    )

    # Check pandas frames
    rn = nm.packagedata_frame("native")
    rf = nm.packagedata_frame("flopy")
    assert list(rn.columns) == ["k", "i", "j"] + partial_expected_cols
    assert list(rf.columns) == ["cellid"] + partial_expected_cols
    assert list(rn.ncon) == [1, 2, 2, 1, 2, 3, 1]
    assert list(rn.ndv) == [0, 0, 0, 0, 0, 0, 0]
    # native is one based, k,i,j are str
    assert list(rn.index) == [1, 2, 3, 4, 5, 6, 7]
    assert list(rn.k) == ["1", "1", "1", "1", "1", "1", "1"]
    assert list(rn.i) == ["1", "1", "2", "1", "2", "2", "3"]
    assert list(rn.j) == ["1", "2", "2", "2", "2", "2", "2"]
    # flopy has zero based, cellid with tuple
    assert list(rf.index) == [0, 1, 2, 3, 4, 5, 6]
    assert list(rf.cellid) == [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 1),
        (0, 0, 1),
        (0, 1, 1),
        (0, 1, 1),
        (0, 2, 1),
    ]
    nm.write_packagedata(tmp_path / "packagedata.dat")

    # With one auxiliary str
    gwf.remove_package("sfr")
    nm.reaches["var1"] = np.arange(len(nm.reaches), dtype=float) * 12.0
    nm.set_sfr_obj(auxiliary="var1")
    assert list(gwf.sfr.packagedata.array.dtype.names) == [
        ridxname,
        "cellid",
    ] + partial_expected_cols + ["var1"]
    np.testing.assert_array_almost_equal(
        gwf.sfr.packagedata.array["var1"], [0.0, 12.0, 24.0, 36.0, 48.0, 60.0, 72.0]
    )
    nm.write_packagedata(tmp_path / "packagedata_one_aux.dat", auxiliary="var1")

    # With auxiliary list
    gwf.remove_package("sfr")
    nm.reaches["var1"] = np.arange(len(nm.reaches), dtype=float) * 12.0
    nm.reaches["var2"] = np.arange(len(nm.reaches)) * 11
    nm.set_sfr_obj(auxiliary=["var1", "var2"])
    assert list(gwf.sfr.packagedata.array.dtype.names) == [
        ridxname,
        "cellid",
    ] + partial_expected_cols + ["var1", "var2"]
    np.testing.assert_array_almost_equal(
        gwf.sfr.packagedata.array["var1"], [0.0, 12.0, 24.0, 36.0, 48.0, 60.0, 72.0]
    )
    np.testing.assert_array_almost_equal(
        gwf.sfr.packagedata.array["var2"], [0, 11, 22, 33, 44, 55, 66]
    )
    nm.write_packagedata(
        tmp_path / "packagedata_two_aux.dat", auxiliary=["var1", "var2"]
    )

    # With boundname
    gwf.remove_package("sfr")
    nm.reaches["boundname"] = nm.reaches["segnum"]
    nm.set_sfr_obj()
    assert list(gwf.sfr.packagedata.array.dtype.names) == [
        ridxname,
        "cellid",
    ] + partial_expected_cols + ["boundname"]
    assert list(gwf.sfr.packagedata.array["boundname"]) == (
        ["1", "1", "1", "2", "2", "0", "0"]
    )
    nm.write_packagedata(tmp_path / "packagedata_boundname.dat")

    # With auxiliary and boundname
    gwf.remove_package("sfr")
    nm.reaches["boundname"] = nm.reaches["segnum"].astype(str)
    nm.reaches.boundname.at[1] = "another reach"
    nm.reaches.boundname.at[2] = "longname" * 6
    nm.reaches["var1"] = np.arange(len(nm.reaches), dtype=float) * 12.0
    nm.set_sfr_obj(auxiliary=["var1"])
    assert list(gwf.sfr.packagedata.array.dtype.names) == [
        ridxname,
        "cellid",
    ] + partial_expected_cols + ["var1", "boundname"]
    np.testing.assert_array_almost_equal(
        gwf.sfr.packagedata.array["var1"], [0.0, 12.0, 24.0, 36.0, 48.0, 60.0, 72.0]
    )
    assert list(gwf.sfr.packagedata.array["boundname"]) == (
        ["another reach", "longname" * 5, "1", "2", "2", "0", "0"]
    )
    nm.write_packagedata(tmp_path / "packagedata_aux_boundname.dat", auxiliary=["var1"])


def test_connectiondata(tmp_path):
    n = get_basic_swn()
    sim, gwf = get_basic_modflow(tmp_path)
    nm = swn.SwnMf6.from_swn_flopy(n, gwf)
    nm.default_packagedata()
    nm.set_sfr_obj()

    # Check pandas series
    cn = nm.connectiondata_series("native")
    cf = nm.connectiondata_series("flopy")
    assert list(cn.index) == [1, 2, 3, 4, 5, 6, 7]
    assert list(cn) == [[-2], [1, -3], [2, -6], [-5], [4, -6], [3, 5, -7], [6]]
    assert list(cf.index) == [0, 1, 2, 3, 4, 5, 6]
    assert list(cf) == [[-1], [0, -2], [1, -5], [-4], [3, -5], [2, 4, -6], [5]]

    nm.write_connectiondata(tmp_path / "connectiondata.dat")


def check_number_sum_hex(a, n, h):
    a = np.ceil(a).astype(np.int64)
    assert a.sum() == n
    ah = md5(a.tobytes()).hexdigest()
    assert ah.startswith(h), f"{ah} does not start with {h}"


@requires_mf6
def test_coastal(tmp_path, coastal_lines_gdf, coastal_flow_m):
    # Load a MODFLOW model
    sim = flopy.mf6.MFSimulation.load(
        "mfsim.nam", sim_ws=str(datadir / "mf6_coastal"), exe_name=mf6_exe
    )
    sim.set_sim_path(str(tmp_path))
    gwf = sim.get_model("h")
    # this model runs without SFR
    sim.write_simulation()
    success, buff = sim.run_simulation()
    assert success
    # Create a SWN with adjusted elevation profiles
    n = swn.SurfaceWaterNetwork.from_lines(coastal_lines_gdf.geometry)
    n.adjust_elevation_profile()
    nm = swn.SwnMf6.from_swn_flopy(n, gwf)
    nm.default_packagedata(hyd_cond1=2.0, thickness1=2.0)
    nm.set_sfr_obj(
        save_flows=True,
        stage_filerecord="h.sfr.bin",
        budget_filerecord="h.sfr.bud",
        maximum_iterations=100,
        maximum_picard_iterations=10,
    )
    sim.write_simulation()
    success, buff = sim.run_simulation()
    assert not success  # failed run
    # Check dataframes
    assert len(nm.segments) == 304
    assert nm.segments["in_model"].sum() == 184
    assert len(nm.reaches) == 297
    # Check remaining reaches added that are inside model domain
    reach_geom = nm.reaches.loc[nm.reaches["segnum"] == 3047735, "geometry"].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 980.5448069140768)
    # These should be split between two cells
    reach_geoms = nm.reaches.loc[nm.reaches["segnum"] == 3047750, "geometry"]
    assert len(reach_geoms) == 2
    np.testing.assert_almost_equal(reach_geoms.iloc[0].length, 204.90164560019)
    np.testing.assert_almost_equal(reach_geoms.iloc[1].length, 789.59872070638)
    # This reach should not be extended, the remainder is too far away
    reach_geom = nm.reaches.loc[nm.reaches["segnum"] == 3047762, "geometry"].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 261.4644731621629)
    # This reach should not be extended, the remainder is too long
    reach_geom = nm.reaches.loc[nm.reaches["segnum"] == 3047926, "geometry"].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 237.72893664132727)
    # Check other packages
    check_number_sum_hex(gwf.dis.idomain.array, 509, "c4135a084b2593e0b69c148136a3ad6d")
    assert repr(nm) == dedent(
        f"""\
        <SwnMf6: flopy mf6 'h'
          297 in reaches ({ridxname}): [1, 2, ..., 296, 297]
          1 stress period with perlen: [1.0] days />"""
    )
    if matplotlib:
        _ = nm.plot()
        plt.close()
    # Write output files
    gdf_to_shapefile(nm.reaches, tmp_path / "reaches.shp")
    gdf_to_shapefile(nm.segments, tmp_path / "segments.shp")
    nm.grid_cells.to_file(tmp_path / "grid_cells.shp")


@requires_mf6
def test_coastal_elevations(coastal_swn, coastal_flow_m, tmp_path):
    def _make_plot_sequence():
        if matplotlib:
            nm.plot_reaches_vs_model("all", plot_bottom=True)
            for seg in [3049818, 3049378]:
                nm.plot_reaches_vs_model(seg, plot_bottom=True)
                plt.close()

    # Load a MODFLOW model
    sim = flopy.mf6.MFSimulation.load(
        "mfsim.nam", sim_ws=str(datadir / "mf6_coastal"), exe_name=mf6_exe
    )
    sim.set_sim_path(str(tmp_path))
    gwf = sim.get_model("h")
    nm = swn.SwnMf6.from_swn_flopy(coastal_swn, gwf)
    nm.default_packagedata(hyd_cond1=2.0, thickness1=2.0)
    # TODO: inflow=coastal_flow_m
    _ = nm.add_model_topbot_to_reaches()
    _make_plot_sequence()
    # handy to set a max elevation that a stream can be
    # MF6 no longer segment based so this is not approp:
    # _ = nm.get_seg_ijk()
    # tops = nm.get_top_elevs_at_segs().top_up
    # max_str_z = tops.describe()["75%"]
    # _ = nm.fix_segment_elevs(min_incise=0.2, min_slope=1.e-4,
    #                          max_str_z=max_str_z)
    # _ = nm.reconcile_reach_strtop()
    # _make_plot_sequence()
    #
    # _ = nm.add_model_topbot_to_reaches()
    nm.fix_reach_elevs(direction="upstream")
    _make_plot_sequence()
    nm.fix_reach_elevs(direction="downstream")
    _make_plot_sequence()
    nm.set_sfr_obj(
        save_flows=True,
        stage_filerecord="h.sfr.bin",
        budget_filerecord="h.sfr.bud",
        maximum_iterations=100,
        maximum_picard_iterations=10,
        unit_conversion=86400,
    )
    # sim.ims.outer_dvclose = 1e-2
    # sim.ims.inner_dvclose = 1e-3
    sim.write_simulation()
    success, buff = sim.run_simulation()
    assert success
    if matplotlib:
        plt.close()
    # TODO: complete elevation adjustments; see older MODFLOW methods


@requires_mf6
def test_coastal_reduced(coastal_lines_gdf, coastal_flow_m, tmp_path):
    n = swn.SurfaceWaterNetwork.from_lines(coastal_lines_gdf.geometry)
    assert len(n) == 304
    # Modify swn object
    n.remove(
        condition=n.segments["stream_order"] == 1,
        segnums=n.gather_segnums(upstream=3047927),
    )
    assert len(n) == 130
    # Load a MODFLOW model
    sim = flopy.mf6.MFSimulation.load(
        "mfsim.nam", sim_ws=str(datadir / "mf6_coastal"), exe_name=mf6_exe
    )
    sim.set_sim_path(str(tmp_path))
    gwf = sim.get_model("h")
    nm = swn.SwnMf6.from_swn_flopy(n, gwf)
    # TODO: inflow=coastal_flow_m
    # These should be split between two cells
    reach_geoms = nm.reaches.loc[nm.reaches["segnum"] == 3047750, "geometry"]
    assert len(reach_geoms) == 2
    np.testing.assert_almost_equal(reach_geoms.iloc[0].length, 204.90164560019)
    np.testing.assert_almost_equal(reach_geoms.iloc[1].length, 789.59872070638)
    # This reach should not be extended, the remainder is too far away
    reach_geom = nm.reaches.loc[nm.reaches["segnum"] == 3047762, "geometry"].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 261.4644731621629)
    # This reach should not be extended, the remainder is too long
    reach_geom = nm.reaches.loc[nm.reaches["segnum"] == 3047926, "geometry"].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 237.72893664132727)
    assert len(nm.reaches) == 154
    assert repr(nm) == dedent(
        f"""\
        <SwnMf6: flopy mf6 'h'
          154 in reaches ({ridxname}): [1, 2, ..., 153, 154]
          1 stress period with perlen: [1.0] days />"""
    )
    if matplotlib:
        _ = nm.plot()
        plt.close()
    # Run model and read outputs
    nm.default_packagedata(hyd_cond1=2.0, thickness1=2.0)
    nm.set_sfr_obj(
        save_flows=True,
        stage_filerecord="h.sfr.bin",
        budget_filerecord="h.sfr.bud",
        maximum_iterations=100,
        maximum_picard_iterations=10,
    )
    sim.write_simulation()
    success, buff = sim.run_simulation()
    assert not success
    # Error/warning: upstream elevation is equal to downstream, slope is zero
    # TODO: improve processing to correct elevation errors
    gdf_to_shapefile(nm.reaches, tmp_path / "reaches.shp")
    gdf_to_shapefile(nm.segments, tmp_path / "segments.shp")
    nm.grid_cells.to_file(tmp_path / "grid_cells.shp")


@requires_mf6
def test_coastal_idomain_modify(coastal_swn, coastal_flow_m, tmp_path):
    # Load a MODFLOW model
    sim = flopy.mf6.MFSimulation.load(
        "mfsim.nam", sim_ws=str(datadir / "mf6_coastal"), exe_name=mf6_exe
    )
    sim.set_sim_path(str(tmp_path))
    gwf = sim.get_model("h")
    nm = swn.SwnMf6.from_swn_flopy(coastal_swn, gwf, idomain_action="modify")
    # TODO: inflow=coastal_flow_m
    assert len(nm.segments) == 304
    assert nm.segments["in_model"].sum() == 304
    # Check a remaining reach added that is outside model domain
    reach_geom = nm.reaches.loc[nm.reaches["segnum"] == 3048565, "geometry"].iloc[0]
    np.testing.assert_almost_equal(reach_geom.length, 647.316024023105)
    expected_geom = wkt.loads(
        "LINESTRING Z (1819072.5 5869685.1 4, 1819000 5869684.9 5.7, "
        "1818997.5 5869684.9 5.8, 1818967.5 5869654.9 5, "
        "1818907.5 5869654.8 4, 1818877.6 5869624.7 5, 1818787.5 5869624.5 6, "
        "1818757.6 5869594.5 5.1, 1818697.6 5869594.4 5.7, "
        "1818667.6 5869564.3 6.2, 1818607.6 5869564.2 4.7, "
        "1818577.6 5869534.1 5.6, 1818487.6 5869534 6.2)"
    )
    reach_geom.equals_exact(expected_geom, 0)
    # Data set 1c
    assert len(nm.reaches) == 478
    # Check other packages
    check_number_sum_hex(gwf.dis.idomain.array, 572, "d353560128577b37f730562d2f89c025")
    assert repr(nm) == dedent(
        f"""\
        <SwnMf6: flopy mf6 'h'
          478 in reaches ({ridxname}): [1, 2, ..., 477, 478]
          1 stress period with perlen: [1.0] days />"""
    )
    if matplotlib:
        _ = nm.plot()
        plt.close()
    # Run model and read outputs
    nm.default_packagedata(hyd_cond1=2.0, thickness1=2.0)
    nm.set_sfr_obj(
        save_flows=True,
        stage_filerecord="h.sfr.bin",
        budget_filerecord="h.sfr.bud",
        maximum_iterations=100,
        maximum_picard_iterations=10,
    )
    sim.write_simulation()
    success, buff = sim.run_simulation()
    assert not success
    # Error/warning: upstream elevation is equal to downstream, slope is zero
    # TODO: improve processing to correct elevation errors
    gdf_to_shapefile(nm.reaches, tmp_path / "reaches.shp")
    gdf_to_shapefile(nm.segments, tmp_path / "segments.shp")
    nm.grid_cells.to_file(tmp_path / "grid_cells.shp")


@requires_mf6
def test_include_downstream_reach_outside_model(tmp_path):
    sim, gwf = get_basic_modflow(tmp_path, with_top=True)
    gwf.remove_package("rch")
    gwf.dis.idomain.set_data([[1, 1], [1, 1], [1, 0]])
    gt = swn.modflow.geotransform_from_flopy(gwf)
    lines = interp_2d_to_3d(
        geopandas.GeoSeries.from_wkt(
            [
                "LINESTRING (60 89, 60 80)",
                "LINESTRING (40 130, 60 89)",
                "LINESTRING (70 130, 60 89)",
            ]
        ),
        gwf.dis.top.array,
        gt,
    )
    n = swn.SurfaceWaterNetwork.from_lines(lines)
    nm = swn.SwnMf6.from_swn_flopy(n, gwf)
    nm.default_packagedata(hyd_cond1=0.0)
    perioddata = [
        [0, "inflow", 1.2],
        [2, "inflow", 3.4],
    ]
    nm.set_sfr_obj(
        print_input=True,
        save_flows=True,
        budget_filerecord="model.sfr.bud",
        maximum_iterations=100,
        maximum_picard_iterations=10,
        perioddata=perioddata,
    )
    assert gwf.sfr.nreaches.data == 5
    dat = gwf.sfr.packagedata.array
    np.testing.assert_array_equal(dat[ridxname], [0, 1, 2, 3, 4])
    cellid = np.array(dat.cellid, dtype=[("k", int), ("i", int), ("j", int)])
    np.testing.assert_array_equal(cellid["k"], [0, 0, 0, 0, 0])
    np.testing.assert_array_equal(cellid["i"], [0, 1, 0, 1, 1])
    np.testing.assert_array_equal(cellid["j"], [0, 1, 1, 1, 1])
    np.testing.assert_array_almost_equal(
        dat.rlen, [22.53083, 23.087149, 20.58629, 21.615604, 9.0]
    )
    np.testing.assert_array_almost_equal(
        dat.rgrd, [0.033977833, 0.033977833, 0.01303259, 0.01303259, 0.05]
    )
    np.testing.assert_array_almost_equal(
        dat.rtp, [15.4927845, 14.849314, 14.865853, 14.548374, 14.225]
    )
    np.testing.assert_array_equal(dat.rbth, [1.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(dat.rhk, [0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(dat.man, [0.024, 0.024, 0.024, 0.024, 0.024])
    np.testing.assert_array_equal(dat.ncon, [1, 2, 1, 2, 2])
    np.testing.assert_array_equal(dat.ndv, [0, 0, 0, 0, 0])
    assert list(nm.connectiondata_series("native")) == (
        [[-2], [1, -5], [-4], [3, -5], [2, 4]]
    )
    if matplotlib:
        _ = nm.plot()
        plt.close()

    # Run model and read outputs
    sim.write_simulation()
    success, buff = sim.run_simulation()
    assert success
    sfr_bud_fname = tmp_path / "model.sfr.bud"
    read_budget(sfr_bud_fname, "EXT-INFLOW", nm.reaches, colname="Qin")
    read_budget(sfr_bud_fname, "EXT-OUTFLOW", nm.reaches, colname="Qout")
    # Check results
    np.testing.assert_array_equal(nm.reaches["Qin"], [1.2, 0.0, 3.4, 0.0, 0.0])
    np.testing.assert_array_equal(nm.reaches["Qout"], [0.0, 0.0, 0.0, 0.0, -4.6])


def test_n3d_defaults_with_div_on_outlet(tmp_path):
    """Test special case to handle, possible MODFLOW6 bug."""
    n = get_basic_swn()
    diversions = geopandas.GeoDataFrame(
        geometry=[Point(58, 100), Point(62, 100), Point(61, 89), Point(59, 89)]
    )
    n.set_diversions(diversions=diversions)
    sim, gwf = get_basic_modflow(tmp_path, with_top=False)
    nm = swn.SwnMf6.from_swn_flopy(n, gwf)
    # check object reaches
    expected = pd.DataFrame(
        {
            "segnum": [1, 1, 1, 2, 2, 0, 0, -1, -1, -1, -1, 0],
            "k": 0,
            "i": [0, 0, 1, 0, 1, 1, 2, 1, 1, 2, 2, 2],
            "j": [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "iseg": [1, 1, 1, 2, 2, 3, 3, 0, 0, 0, 0, 3],
            "ireach": [1, 2, 3, 1, 2, 1, 2, 0, 0, 0, 0, 3],
            "rlen": [
                18.027756,
                6.009252,
                12.018504,
                21.081851,
                10.5409255,
                10.0,
                10.0,
                1.0,
                1.0,
                1.0,
                1.0,
                10.0,
            ],
            to_ridxname: [2, 3, 6, 5, 6, 7, 12, 0, 0, 0, 0, 0],
            from_ridxsname: [
                set(),
                {1},
                {2},
                set(),
                {4},
                {3, 5},
                {6},
                set(),
                set(),
                set(),
                set(),
                {7},
            ],
            div_from_ridxname: [0, 0, 0, 0, 0, 0, 0, 3, 5, 7, 7, 0],
            div_to_ridxsname: [
                set(),
                set(),
                {8},
                set(),
                {9},
                set(),
                {10, 11},
                set(),
                set(),
                set(),
                set(),
                set(),
            ],
            "mask": [False] * 11 + [True],
        },
        index=pd.RangeIndex(1, 13, name=ridxname),
    )
    pd.testing.assert_frame_equal(nm.reaches[expected.columns], expected)
    nm.default_packagedata(hyd_cond1=1e-4)
    expected = pd.DataFrame(
        {
            "rwid": [10.0] * 7 + [1.0] * 4 + [10.0],
            "rgrd": [0.027735, 0.027735, 0.027735, 0.0316227766, 0.0316227766, 0.1, 0.1]
            + [0.001] * 5,
            "rtp": [14.75, 14.416667, 14.166667, 14.666667, 14.166667, 13.5, 12.5]
            + [15.0] * 5,
            "rbth": 1.0,
            "rhk": [1e-4] * 7 + [0.0] * 4 + [1e-4],
            "man": 0.024,
            "ncon": [1, 2, 3, 1, 3, 3, 4, 1, 1, 1, 1, 1],
            "ndv": [0, 0, 1, 0, 1, 0, 2, 0, 0, 0, 0, 0],
        },
        index=pd.RangeIndex(1, 13, name=ridxname),
    )
    pd.testing.assert_frame_equal(
        nm.packagedata_frame("native")[expected.columns], expected
    )
    # Write native MF6 file and flopy datasets
    nm.write_packagedata(tmp_path / "packagedata.dat")
    fpd = nm.flopy_packagedata()
    # Don't check everything
    assert len(fpd) == 12
    assert fpd[11][1] == "NONE"
    # Write native MF6 file and flopy datasets
    nm.write_connectiondata(tmp_path / "connectiondata.dat")
    expected = [
        [0, -1],
        [1, 0, -2],
        [2, 1, -5, -7],
        [3, -4],
        [4, 3, -5, -8],
        [5, 2, 4, -6],
        [6, 5, -11, -9, -10],
        [7, 2],
        [8, 4],
        [9, 6],
        [10, 6],
        [11, 6],
    ]
    assert nm.flopy_connectiondata() == expected
    nm.write_diversions(tmp_path / "diversions.dat")
    pd.testing.assert_frame_equal(
        nm.diversions_frame("native"),
        pd.DataFrame(
            {
                ridxname: [3, 5, 7, 7],
                "idv": [1, 1, 1, 2],
                "iconr": [8, 9, 10, 11],
                "cprior": "upto",
            }
        ),
    )
    expected = [
        [2, 0, 7, "upto"],
        [4, 0, 8, "upto"],
        [6, 0, 9, "upto"],
        [6, 1, 10, "upto"],
    ]
    assert nm.flopy_diversions() == expected
    assert repr(nm) == dedent(
        f"""\
        <SwnMf6: flopy mf6 'model'
          12 in reaches ({ridxname}): [1, 2, ..., 11, 12]
          4 in diversions (iconr): [8, 9, 10, 11]
          1 stress period with perlen: [1.0] days />"""
    )
    if matplotlib:
        _ = nm.plot()
        plt.close()
    # Use with flopy
    diversions = None
    for use_open_close in [False, True]:
        gwf.remove_package("sfr")
        if use_open_close:
            nm.set_sfr_obj(
                print_input=True,
                save_flows=True,
                stage_filerecord="model.sfr.bin",
                budget_filerecord="model.sfr.bud",
                maximum_iterations=100,
                maximum_picard_iterations=10,
                packagedata={"filename": "packagedata.dat"},
                connectiondata={"filename": "connectiondata.dat"},
                diversions={"filename": "diversions.dat"},
                perioddata={},
            )
        else:
            nm.set_sfr_obj(
                print_input=True,
                save_flows=True,
                stage_filerecord="model.sfr.bin",
                budget_filerecord="model.sfr.bud",
                maximum_iterations=100,
                maximum_picard_iterations=10,
                packagedata=nm.flopy_packagedata(),
                connectiondata=nm.flopy_connectiondata(),
                diversions=nm.flopy_diversions(),
                perioddata={},
            )
        if mf6_exe:
            # Run model and read outputs
            sim.write_simulation()
            success, buff = sim.run_simulation()
            assert success
    # Write some files
    gdf_to_shapefile(
        nm.reaches[nm.reaches.geom_type == "LineString"], tmp_path / "reaches.shp"
    )
    if not mf6_exe:
        return
    # Check results
    head = read_head(tmp_path / "model.hds")
    sl = read_budget(tmp_path / "model.cbc", "SFR", nm.reaches, "sfrleakage")
    sf = read_budget(tmp_path / "model.sfr.bud", "GWF", nm.reaches, "sfr_Q")
    assert head.shape == (1, 3, 2)
    np.testing.assert_array_almost_equal(
        head,
        np.array(
            [
                [
                    [17.10986328, 16.77658646],
                    [17.49119547, 16.81176342],
                    [17.75195983, 17.2127244],
                ]
            ]
        ),
    )
    expected = np.array(
        [
            -0.04240276,
            -0.01410363,
            -0.03159486,
            -0.04431906,
            -0.02773725,
            -0.03292868,
            -0.04691372,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    np.testing.assert_array_almost_equal(sl["q"], expected)
    expected = np.array(
        [
            0.04240276,
            0.01410362,
            0.03159486,
            0.04431904,
            0.02773725,
            0.03292867,
            0.04691372,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    np.testing.assert_array_almost_equal(sf["q"], expected)


@requires_mf6
def test_diversions(tmp_path):
    sim, gwf = get_basic_modflow(tmp_path, with_top=True)
    gwf.remove_package("rch")
    gt = swn.modflow.geotransform_from_flopy(gwf)
    lsz = interp_2d_to_3d(n3d_lines, gwf.dis.top.array, gt)
    n = swn.SurfaceWaterNetwork.from_lines(lsz)
    n.adjust_elevation_profile()
    diversions = geopandas.GeoDataFrame(
        geometry=[Point(58, 100), Point(62, 100), Point(61, 91), Point(59, 91)],
        index=["SW 1", "SW 2", "SW 3", "SW 4"],
    )
    n.set_diversions(diversions=diversions)

    nm = swn.SwnMf6.from_swn_flopy(n, gwf)
    nm.reaches["boundname"] = nm.reaches["divid"]
    assert list(nm.diversions[ridxname]) == [3, 5, 6, 6]
    assert list(nm.diversions["idv"]) == [1, 1, 1, 2]

    # Check optional parameter
    nm2 = swn.SwnMf6.from_swn_flopy(n, gwf, diversion_downstream_bias=0.3)
    assert list(nm2.diversions[ridxname]) == [3, 5, 7, 7]
    assert list(nm2.diversions["idv"]) == [1, 1, 1, 2]

    # Assemble MODFLOW SFR data
    nm.default_packagedata(hyd_cond1=0.0)
    nm.write_packagedata(tmp_path / "packagedata.dat")
    nm.write_connectiondata(tmp_path / "connectiondata.dat")
    nm.write_diversions(tmp_path / "diversions.dat")

    # Extra checks with formatted frame read/write methods
    df = nm.packagedata_frame("native")
    for col in "kij":
        df[col] = df[col].astype(np.int64)
    pd.testing.assert_frame_equal(
        df,
        swn.file.read_formatted_frame(tmp_path / "packagedata.dat").set_index(ridxname),
    )
    pd.testing.assert_frame_equal(
        nm.diversions_frame("native").reset_index(drop=True),
        swn.file.read_formatted_frame(tmp_path / "diversions.dat"),
    )

    # Route some flow from headwater reaches
    perioddata = [
        [1, "inflow", 2.0],
        [4, "inflow", 3.0],
    ]
    write_list(tmp_path / "perioddata.dat", perioddata)

    nm.set_sfr_obj(
        print_input=True,
        print_flows=True,
        save_flows=True,
        stage_filerecord="model.sfr.bin",
        budget_filerecord="model.sfr.bud",
        maximum_iterations=100,
        maximum_picard_iterations=10,
        packagedata={"filename": "packagedata.dat"},
        connectiondata={"filename": "connectiondata.dat"},
        diversions={"filename": "diversions.dat"},
        perioddata={0: {"filename": "perioddata.dat"}},
    )
    sim.write_simulation()
    success, buff = sim.run_simulation()
    res = pd.DataFrame(index=nm.reaches.index)
    _ = read_budget(tmp_path / "model.sfr.bud", "EXT-INFLOW", res, "Qin")
    _ = read_budget(tmp_path / "model.sfr.bud", "EXT-OUTFLOW", res, "Qout")
    expected = pd.DataFrame(
        {
            "Qin": [2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Qout": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=nm.reaches.index,
    )
    pd.testing.assert_frame_equal(res, expected)

    # With abstraction
    perioddata = [[1, "inflow", 2.0], [4, "inflow", 3.0], [5, "diversion", 1, 1.1]]
    write_list(tmp_path / "perioddata.dat", perioddata)
    success, buff = sim.run_simulation()
    np.testing.assert_array_almost_equal(
        read_budget(tmp_path / "model.sfr.bud", "EXT-OUTFLOW").q,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.9, 0.0, -1.1, 0.0, 0.0],
    )
    # More abstraction with dry streams
    perioddata = [[1, "inflow", 2.0], [4, "inflow", 3.0], [5, "diversion", 1, 3.3]]
    write_list(tmp_path / "perioddata.dat", perioddata)
    success, buff = sim.run_simulation()
    np.testing.assert_array_almost_equal(
        read_budget(tmp_path / "model.sfr.bud", "EXT-OUTFLOW").q,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, -3.0, 0.0, 0.0],
    )


def test_pickle(tmp_path):
    sim, gwf = get_basic_modflow(tmp_path, with_top=True)
    gt = swn.modflow.geotransform_from_flopy(gwf)
    lsz = interp_2d_to_3d(n3d_lines, gwf.dis.top.array, gt)
    n = swn.SurfaceWaterNetwork.from_lines(lsz)
    n.adjust_elevation_profile()
    nm1 = swn.SwnMf6.from_swn_flopy(n, gwf)
    # use pickle dumps / loads methods
    data = pickle.dumps(nm1)
    nm2 = pickle.loads(data)
    assert nm1 != nm2
    assert nm2.swn is None
    assert nm2.model is None
    nm2.swn = n
    nm2.model = gwf
    assert nm1 == nm2
    # use to_pickle / from_pickle methods
    diversions = geopandas.GeoDataFrame(
        geometry=[Point(58, 100), Point(62, 100), Point(61, 89), Point(59, 89)]
    )
    n.set_diversions(diversions=diversions)
    nm3 = swn.SwnMf6.from_swn_flopy(n, gwf)
    nm3.to_pickle(tmp_path / "nm4.pickle")
    nm4 = swn.SwnMf6.from_pickle(tmp_path / "nm4.pickle", n, gwf)
    assert nm3 == nm4

    # Issue 31
    with pytest.raises(TypeError, match="swn property must be an instance o"):
        swn.SwnMf6.from_pickle(tmp_path / "nm4.pickle", gwf)
    with pytest.raises(AttributeError, match="swn property can only be set o"):
        nm2.swn = n


def test_route_reaches():
    n1 = get_basic_swn(has_diversions=True)
    lines2 = list(n1.segments.geometry)
    lines2.append(wkt.loads("LINESTRING (40 90, 50 80)"))
    n = swn.SurfaceWaterNetwork.from_lines(geopandas.GeoSeries(lines2))
    sim, gwf = get_basic_modflow(with_top=False)
    nm = swn.SwnMf6.from_swn_flopy(n, gwf)
    assert nm.route_reaches(8, 8) == [8]
    assert nm.route_reaches(7, 8) == [7, 8]
    assert nm.route_reaches(8, 7) == [8, 7]
    assert nm.route_reaches(2, 7) == [2, 3, 7]
    assert nm.route_reaches(7, 2) == [7, 3, 2]
    assert nm.route_reaches(1, 7) == [1, 2, 3, 7]
    assert nm.route_reaches(7, 1) == [7, 3, 2, 1]
    assert nm.route_reaches(2, 4, allow_indirect=True) == [2, 3, 5, 4]
    # errors
    with pytest.raises(IndexError, match=f"invalid start {ridxname} 0"):
        nm.route_reaches(0, 1)
    with pytest.raises(IndexError, match=f"invalid end {ridxname} 0"):
        nm.route_reaches(1, 0)
    with pytest.raises(ConnectionError, match="1 does not connect to 4"):
        nm.route_reaches(1, 4)
    with pytest.raises(ConnectionError, match="reach networks are disjoint"):
        nm.route_reaches(1, 6)
    with pytest.raises(ConnectionError, match="reach networks are disjoint"):
        nm.route_reaches(1, 6, allow_indirect=True)
    # TODO: diversions?


def test_gather_reaches(fluss_n):
    n = fluss_n
    sim = flopy.mf6.MFSimulation()
    _ = flopy.mf6.ModflowTdis(sim, nper=1, time_units="days")
    gwf = flopy.mf6.ModflowGwf(sim)
    _ = flopy.mf6.ModflowIms(sim)
    _ = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=2,
        nrow=9,
        ncol=16,
        delr=50.0,
        delc=50.0,
        length_units="meters",
        idomain=1,
        top=20.0,
        botm=10.0,
        xorigin=200.0,
        yorigin=50.0,
    )
    _ = flopy.mf6.ModflowGwfoc(gwf)
    _ = flopy.mf6.ModflowGwfnpf(gwf, k=1e-2, save_flows=True)

    nm = swn.SwnMf6.from_swn_flopy(n, gwf)

    # upstream
    assert set(nm.gather_reaches(upstream=10)) == {10}
    assert set(nm.gather_reaches(upstream=[33])) == {7, 8, 9, 10, 11, 33}
    assert set(nm.gather_reaches(upstream=52)) == (
        set(range(1, 12)) | {25, 26} | set(range(29, 37)) | set(range(44, 53))
    )
    assert set(nm.gather_reaches(upstream=53)) == (
        set(range(12, 25)) | set(range(37, 44)) | {53}
    )
    assert set(nm.gather_reaches(upstream=28)) == {27, 28}
    assert set(nm.gather_reaches(upstream=60)) == set(range(1, 61))
    # with barriers
    assert set(nm.gather_reaches(upstream=60, barrier=27)) == set(range(1, 61)) - {27}
    assert set(nm.gather_reaches(upstream=60, barrier=28)) == (
        set(range(1, 61)) - {27, 28}
    )
    assert set(nm.gather_reaches(upstream=55, barrier=52)) == set(
        nm.gather_reaches(upstream=55)
    )
    assert set(nm.gather_reaches(upstream=58, barrier=[55, 31])) == (
        set(range(7, 12))
        | {25, 26, 56, 57, 58}
        | set(range(32, 37))
        | set(range(44, 53))
    )
    # downstream
    assert nm.gather_reaches(downstream=10) == (
        [11, 33, 34, 35, 36] + list(range(44, 53)) + list(range(56, 61))
    )
    assert nm.gather_reaches(downstream=[35]) == (
        [36] + list(range(44, 53)) + list(range(56, 61))
    )
    assert nm.gather_reaches(downstream=52) == list(range(56, 61))
    assert nm.gather_reaches(downstream=55) == list(range(56, 61))
    assert nm.gather_reaches(downstream=27) == [28, 59, 60]
    assert nm.gather_reaches(downstream=60) == []
    assert set(nm.gather_reaches(downstream=25, gather_upstream=True)) == (
        set(range(1, 61)) - {25}
    )
    assert set(nm.gather_reaches(downstream=[51], gather_upstream=True)) == (
        set(range(12, 25)) | {27, 28} | set(range(37, 44)) | set(range(52, 61))
    )
    assert nm.gather_reaches(downstream=60, gather_upstream=True) == []
    assert set(nm.gather_reaches(downstream=10, gather_upstream=True, barrier=52)) == (
        set(range(1, 10)) | {11, 25, 26} | set(range(29, 37)) | set(range(44, 52))
    )

    # break it
    with pytest.raises(
        IndexError, match=r"upstream (rno|ifno) 0 not found in reaches\.index"
    ):
        nm.gather_reaches(upstream=0)
    with pytest.raises(
        IndexError, match=r"2 upstream reaches not found in reaches\.index: \[0, 61"
    ):
        nm.gather_reaches(upstream=[0, 2, 61])
    with pytest.raises(
        IndexError, match=r"barrier (rno|ifno) 0 not found in reaches\.index"
    ):
        nm.gather_reaches(upstream=18, barrier=0)
    with pytest.raises(
        IndexError, match=r"1 barrier reach not found in reaches\.index: \[0\]"
    ):
        nm.gather_reaches(upstream=18, barrier=[0, 10])
    with pytest.raises(
        IndexError, match=r"downstream (rno|ifno) 0 not found in reaches\.index"
    ):
        nm.gather_reaches(downstream=0)


def test_get_flopy_mf6_package():
    from swn.modflow._swnmf6 import get_flopy_mf6_package

    assert get_flopy_mf6_package("drn") == flopy.mf6.ModflowGwfdrn
    assert get_flopy_mf6_package("GWT/SRC") == flopy.mf6.ModflowGwtsrc
    with pytest.raises(AttributeError):
        get_flopy_mf6_package("no")


def test_package_period_frame():
    n = get_basic_swn()
    sim, gwf = get_basic_modflow()
    nm = swn.SwnMf6.from_swn_flopy(n, gwf)

    with pytest.raises(
        KeyError, match="missing 2 ModflowGwfdrn reaches series: elev, c"
    ):
        nm.package_period_frame("drn", "native")

    nm.set_reach_data_from_array("elev", gwf.dis.top.data - 1.0)
    nm.reaches["rlen"] = (nm.reaches.length).round(2)
    nm.reaches["cond"] = (nm.reaches.length * 10.0).round(2)

    # default, without auxiliary or boundname
    exp_native = pd.DataFrame(
        {
            "k": ["1"] * 7,
            "i": ["1", "1", "2", "1", "2", "2", "3"],
            "j": ["1", "2", "2", "2", "2", "2", "2"],
            "elev": [14.0] * 7,
            "cond": [180.28, 60.09, 120.19, 210.82, 105.41, 100.0, 100.0],
        },
        index=pd.MultiIndex.from_tuples(
            [(1, ridx + 1) for ridx in range(7)], names=["per", ridxname]
        ),
    )
    exp_flopy = pd.DataFrame(
        {
            "cellid": [
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 1),
                (0, 0, 1),
                (0, 1, 1),
                (0, 1, 1),
                (0, 2, 1),
            ],
            "elev": [14.0] * 7,
            "cond": [180.28, 60.09, 120.19, 210.82, 105.41, 100.0, 100.0],
        },
        index=pd.MultiIndex.from_tuples(
            [(0, ridx) for ridx in range(7)], names=["per", ridxname]
        ),
    )
    pd.testing.assert_frame_equal(nm.package_period_frame("drn", "native"), exp_native)
    pd.testing.assert_frame_equal(nm.package_period_frame("drn", "flopy"), exp_flopy)

    # with auxiliary
    pd.testing.assert_frame_equal(
        nm.package_period_frame("drn", "native", auxiliary="rlen"),
        exp_native.assign(rlen=[18.03, 6.01, 12.02, 21.08, 10.54, 10.0, 10.0]),
    )
    pd.testing.assert_frame_equal(
        nm.package_period_frame("drn", "flopy", auxiliary="rlen"),
        exp_flopy.assign(rlen=[18.03, 6.01, 12.02, 21.08, 10.54, 10.0, 10.0]),
    )

    # with boundname column, but boundname=False
    nm.reaches["boundname"] = nm.reaches.segnum
    pd.testing.assert_frame_equal(
        nm.package_period_frame("drn", "native", boundname=False), exp_native
    )
    pd.testing.assert_frame_equal(
        nm.package_period_frame("drn", "flopy", boundname=False), exp_flopy
    )

    # with boundname
    pd.testing.assert_frame_equal(
        nm.package_period_frame("drn", "native"),
        exp_native.assign(boundname=["1", "1", "1", "2", "2", "0", "0"]),
    )
    pd.testing.assert_frame_equal(
        nm.package_period_frame("drn", "flopy"),
        exp_flopy.assign(boundname=["1", "1", "1", "2", "2", "0", "0"]),
    )

    # with boundname and auxiliary
    assert list(nm.package_period_frame("drn", "native", auxiliary="rlen").columns) == (
        ["k", "i", "j", "elev", "cond", "rlen", "boundname"]
    )
    assert list(nm.package_period_frame("drn", "flopy", auxiliary="rlen").columns) == (
        ["cellid", "elev", "cond", "rlen", "boundname"]
    )


def test_write_package_period(tmp_path):
    n = get_basic_swn()
    sim, gwf = get_basic_modflow(tmp_path)
    nm = swn.SwnMf6.from_swn_flopy(n, gwf)

    nm.set_reach_data_from_array("elev", gwf.dis.top.data - 1.0)
    nm.reaches["rlen"] = (nm.reaches.length).round(2)
    nm.reaches["cond"] = (nm.reaches.length * 10.0).round(2)

    fname_tpl = tmp_path / "drn_period_{:02d}.dat"
    fname = tmp_path / "drn_period_01.dat"
    assert not fname.exists()

    expected = ["#k", "i", "j", "elev", "cond"]

    # default, without auxiliary or boundname
    nm.write_package_period("drn", fname_tpl)
    lines = fname.read_text().splitlines()
    assert lines[0].split() == expected
    assert len(lines) == 8

    # with auxiliary
    nm.write_package_period("drn", fname_tpl, auxiliary="rlen")
    assert fname.read_text().splitlines()[0].split() == expected + ["rlen"]

    # with boundname column, but boundname=False
    nm.reaches["boundname"] = nm.reaches.segnum
    nm.write_package_period("drn", fname_tpl, boundname=False)
    assert "boundname" not in fname.read_text().splitlines()[0]

    # with boundname
    nm.write_package_period("drn", fname_tpl)
    assert fname.read_text().splitlines()[0].split() == expected + ["boundname"]

    # with boundname and auxiliary
    nm.write_package_period("drn", fname_tpl, auxiliary="rlen")
    assert fname.read_text().splitlines()[0].split() == expected + ["rlen", "boundname"]

    # Run model and read outputs
    if mf6_exe:
        # more precise cond without rounding
        nm.reaches["cond"] = nm.reaches.length * 10.0
        nm.write_package_period("drn", fname_tpl)
        _ = nm.set_package_obj(
            "drn", pname="swn_drn", stress_period_data={0: {"filename": fname.name}}
        )
        sim.write_simulation()
        success, buff = sim.run_simulation()
        assert success
        dl = read_budget(tmp_path / "model.cbc", "DRN")
        assert "RLEN" not in dl.dtype.names
        expected_q = np.array(
            [
                -0.0639901,
                -0.0088755,
                -0.0236359,
                -0.0311373,
                -0.02073,
                -0.0196662,
                -0.071965,
            ]
        )
        np.testing.assert_almost_equal(dl["q"], expected_q, decimal=7)

        # use the handy "auxmultname" feature
        nm.reaches["cond"] = 10.0
        nm.write_package_period("drn", fname_tpl, auxiliary="rlen")
        _ = nm.set_package_obj(
            "drn",
            pname="swn_drn",
            auxmultname="rlen",
            stress_period_data={0: {"filename": fname.name}},
        )
        sim.write_simulation()
        success, buff = sim.run_simulation()
        assert success
        dl = read_budget(tmp_path / "model.cbc", "DRN")
        assert "RLEN" in dl.dtype.names
        np.testing.assert_almost_equal(dl["q"], expected_q, decimal=5)


def test_flopy_package_period(tmp_path):
    n = get_basic_swn()
    sim, gwf = get_basic_modflow(tmp_path)
    nm = swn.SwnMf6.from_swn_flopy(n, gwf)

    nm.set_reach_data_from_array("elev", gwf.dis.top.data - 1.0)
    nm.reaches["rlen"] = (nm.reaches.length).round(2)
    nm.reaches["cond"] = (nm.reaches.length * 10.0).round(2)

    def df2dic(df):
        return {0: [list(x) for x in df.to_records(index=False)]}

    # default, without auxiliary or boundname
    exp_flopy = pd.DataFrame(
        {
            "cellid": [
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 1),
                (0, 0, 1),
                (0, 1, 1),
                (0, 1, 1),
                (0, 2, 1),
            ],
            "elev": [14.0] * 7,
            "cond": [180.28, 60.09, 120.19, 210.82, 105.41, 100.0, 100.0],
        },
        index=pd.MultiIndex.from_tuples(
            [(0, ridx) for ridx in range(7)], names=["per", ridxname]
        ),
    )
    assert nm.flopy_package_period("drn") == df2dic(exp_flopy)

    # with auxiliary
    assert nm.flopy_package_period("drn", auxiliary="rlen") == df2dic(
        exp_flopy.assign(rlen=[18.03, 6.01, 12.02, 21.08, 10.54, 10.0, 10.0])
    )

    # with boundname column, but boundname=False
    nm.reaches["boundname"] = nm.reaches.segnum
    assert nm.flopy_package_period("drn", boundname=False) == df2dic(exp_flopy)

    # with boundname
    assert nm.flopy_package_period("drn") == df2dic(
        exp_flopy.assign(boundname=["1", "1", "1", "2", "2", "0", "0"])
    )

    # with boundname and auxiliary
    assert nm.flopy_package_period("drn", auxiliary="rlen")[0][0] == (
        [(0, 0, 0), 14.0, 180.28, 18.03, "1"]
    )

    # Run model and read outputs
    if mf6_exe:
        # more precise cond without rounding
        nm.reaches["cond"] = nm.reaches.length * 10.0
        _ = nm.set_package_obj("drn", pname="swn_drn")
        sim.write_simulation()
        success, buff = sim.run_simulation()
        assert success
        dl = read_budget(tmp_path / "model.cbc", "DRN")
        assert "RLEN" not in dl.dtype.names
        expected_q = np.array(
            [
                -0.0639901,
                -0.0088755,
                -0.0236359,
                -0.0311373,
                -0.02073,
                -0.0196662,
                -0.071965,
            ]
        )
        np.testing.assert_almost_equal(dl["q"], expected_q, decimal=7)

        # use the handy "auxmultname" feature
        nm.reaches["cond"] = 10.0
        _ = nm.set_package_obj("drn", pname="swn_drn", auxmultname="rlen")
        sim.write_simulation()
        success, buff = sim.run_simulation()
        assert success
        dl = read_budget(tmp_path / "model.cbc", "DRN")
        assert "RLEN" in dl.dtype.names
        np.testing.assert_almost_equal(dl["q"], expected_q, decimal=5)
