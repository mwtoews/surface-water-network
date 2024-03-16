"""Interface for flopy's implementation for MODFLOW 6."""

__all__ = [
    "SwnMf6",
]

import os
from copy import deepcopy
from itertools import zip_longest

import numpy as np
import pandas as pd
from shapely import wkt

from ..compat import ignore_shapely_warnings_for_object_array
from ..file import write_formatted_frame
from ..util import abbr_str
from ._base import SwnModflowBase

try:
    import matplotlib
except ImportError:
    matplotlib = False


class SwnMf6(SwnModflowBase):
    """Surface water network adaptor for MODFLOW 6.

    Attributes
    ----------
    swn : swn.SurfaceWaterNetwork
        Instance of a SurfaceWaterNetwork.
    model : flopy.mf6.ModflowGwf
        Instance of flopy.mf6.ModflowGwf
    segments : geopandas.GeoDataFrame
        Copied from swn.segments, but with additional columns added
    reaches : geopandas.GeoDataFrame
        Similar to structure in model.sfr.reaches with index "rno" or "ifno",
        ordered and starting from 1. Contains geometry and other columns
        not used by flopy.
    tsvar : dict
        Dataframe of time-varying data for MODFLOW SFR, key is dataset name.
    diversions :  geopandas.GeoDataFrame, pd.DataFrame or None
        Copied from swn.diversions, if set/defined.
    logger : logging.Logger
        Logger to show messages.

    """

    def __init__(self, logger=None):
        """Initialise SwnMf6.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger to show messages.
        """
        super().__init__(logger)
        self.tsvar = {}  # time-varying data
        # all other properties added afterwards

    @classmethod
    def from_swn_flopy(
        cls,
        swn,
        model,
        idomain_action="freeze",
        reach_include_fraction=0.2,
        diversion_downstream_bias=0.0,
    ):
        """Create a MODFLOW 6 SFR structure from a surface water network.

        Parameters
        ----------
        swn : swn.SurfaceWaterNetwork
            Instance of a SurfaceWaterNetwork.
        model : flopy.mf6.ModflowGwf
            Instance of a flopy MODFLOW 6 groundwater flow model.
        idomain_action : str, optional
            Action to handle IDOMAIN:
                - ``freeze`` : Freeze IDOMAIN, but clip streams to fit bounds.
                - ``modify`` : Modify IDOMAIN to fit streams, where possible.
        reach_include_fraction : float or pandas.Series, optional
            Fraction of cell size used as a threshold distance to determine if
            reaches outside the active grid should be included to a cell.
            Based on the furthest distance of the line and cell geometries.
            Default 0.2 (e.g. for a 100 m grid cell, this is 20 m).
        diversion_downstream_bias : float, default 0.0
            A bias used for spatial location matching that increase the
            likelihood of finding downstream reaches of a segnum if positive,
            and upstream reaches if negative. Valid range is -1.0 to 1.0.
            Default 0.0 is no bias, matching to the closest reach.

        Returns
        -------
        obj : swn.SwnMf6 object
        """
        if idomain_action not in ("freeze", "modify"):
            raise ValueError("idomain_action must be one of freeze or modify")

        obj = super().from_swn_flopy(
            swn=swn,
            model=model,
            domain_action=idomain_action,
            reach_include_fraction=reach_include_fraction,
        )

        # Evaluate connections, assume only converging network
        to_segnums_d = swn.to_segnums.to_dict()
        has_diversions = obj.diversions is not None
        if has_diversions:
            reaches_segnum_s = set(obj.reaches[~obj.reaches.diversion].segnum)
        else:
            reaches_segnum_s = set(obj.reaches.segnum)

        def find_next_ridx(segnum):
            if segnum in to_segnums_d:
                to_segnum = to_segnums_d[segnum]
                if to_segnum in reaches_segnum_s:
                    sel = obj.reaches["segnum"] == to_segnum
                    return obj.reaches[sel].index[0]
                else:  # recurse downstream
                    return find_next_ridx(to_segnum)
            else:
                return 0

        def get_to_ridx():
            if segnum == next_segnum:
                return next_ridx
            else:
                return find_next_ridx(segnum)

        ridxname = obj.reach_index_name
        to_ridxname = f"to_{ridxname}"
        from_ridxsname = f"from_{ridxname}s"
        obj.reaches[to_ridxname] = -1
        if has_diversions:
            from_ridxname = f"from_{ridxname}"
            div_to_ridxsname = f"div_to_{ridxname}s"
            div_from_ridxname = f"div_from_{ridxname}"
            segnum_iter = obj.reaches.loc[~obj.reaches.diversion, "segnum"].items()
        else:
            segnum_iter = obj.reaches["segnum"].items()
        ridx, segnum = next(segnum_iter)
        for next_ridx, next_segnum in segnum_iter:
            obj.reaches.at[ridx, to_ridxname] = get_to_ridx()
            ridx, segnum = next_ridx, next_segnum
        next_segnum = swn.END_SEGNUM
        obj.reaches.at[ridx, to_ridxname] = get_to_ridx()
        if has_diversions:
            obj.reaches.loc[obj.reaches["diversion"], to_ridxname] = 0
        assert obj.reaches[to_ridxname].min() >= 0

        # Populate from_ set
        obj.reaches[from_ridxsname] = [set() for _ in range(len(obj.reaches))]
        to_ridxs = obj.reaches.loc[obj.reaches[to_ridxname] > 0, to_ridxname]
        for k, v in to_ridxs.items():
            obj.reaches.at[v, from_ridxsname].add(k)

        # Refresh diversions if set
        if has_diversions:
            div_sel = obj.diversions["in_model"]
            # populate (ridx, idv) from their match to non-diversion reaches
            diversions_in_model = obj.diversions[div_sel]
            r_df = obj.get_location_frame_reach_info(
                diversions_in_model.rename(columns={"from_segnum": "segnum"})[
                    ["segnum", "seg_ndist"]
                ],
                downstream_bias=diversion_downstream_bias,
                geom_loc_df=getattr(diversions_in_model, "geometry", None),
            )
            obj.diversions[ridxname] = 0  # valid from 1
            obj.diversions.loc[r_df.index, ridxname] = r_df[ridxname]
            # evaluate idv, which is betwen 1 and ndv
            obj.diversions["idv"] = 0
            obj.diversions.loc[div_sel, "idv"] = 1
            ridx_counts = obj.diversions[div_sel].groupby(ridxname).count()["in_model"]
            for ridx, count in ridx_counts[ridx_counts > 1].items():
                obj.diversions.loc[
                    obj.diversions[ridxname] == ridx, "idv"
                ] = obj.diversions.idv[obj.diversions[ridxname] == ridx].cumsum()
            # cross-reference iconr to ridx used as a reach
            diversion_reaches = (
                obj.reaches.loc[obj.reaches.diversion].reset_index().set_index("divid")
            )
            obj.diversions["iconr"] = diversion_reaches[ridxname]
            # Also put data into reaches frame
            obj.reaches[div_from_ridxname] = 0
            rdiv = (
                obj.diversions.loc[div_sel, [ridxname, "iconr"]]
                .rename(columns={ridxname: from_ridxname, "iconr": ridxname})
                .reset_index()
                .set_index(ridxname)
            )
            obj.reaches.loc[rdiv.index, div_from_ridxname] = rdiv[from_ridxname]
            obj.reaches[div_to_ridxsname] = [set() for _ in range(len(obj.reaches))]
            to_ridxs = obj.reaches.loc[
                obj.reaches[div_from_ridxname] > 0, div_from_ridxname
            ]
            for k, v in to_ridxs.items():
                obj.reaches.at[v, div_to_ridxsname].add(k)

            # Workaround potential MODFLOW6 bug where diversions cannot attach
            # to outlet reach, so add another reach
            sel = (
                (obj.reaches[to_ridxname] == 0)
                & (~obj.reaches["diversion"])
                & (obj.reaches[div_to_ridxsname].apply(len) > 0)
            )
            if sel.any() and "mask" not in obj.reaches.columns:
                obj.reaches["mask"] = False
                if swn.has_z:
                    empty_geom = wkt.loads("linestring z empty")
                else:
                    empty_geom = wkt.loads("linestring empty")
            for ridx in sel[sel].index:
                new_ridx = len(obj.reaches) + 1
                # Correction to this reach
                obj.reaches.loc[ridx, to_ridxname] = new_ridx
                # Use as template for new reach
                reach_d = obj.reaches.loc[ridx].to_dict()
                reach_d.update(
                    {
                        "geometry": empty_geom,
                        "mask": True,
                        "ireach": reach_d["ireach"] + 1,
                        to_ridxname: 0,
                        from_ridxsname: {ridx},
                        div_to_ridxsname: set(),
                    }
                )
                with ignore_shapely_warnings_for_object_array():
                    obj.reaches.loc[new_ridx] = reach_d

        # Set 1.0 for most, 0.0 for head and diversion nodes
        obj.reaches["ustrf"] = 1.0
        zero_from_ridxs = obj.reaches[from_ridxsname].apply(len) == 0
        obj.reaches.loc[zero_from_ridxs, "ustrf"] = 0.0

        return obj

    def __repr__(self):
        """Return string representation of SwnModflow object."""
        model = self.model
        if model is None:
            model_info = "model not set"
            sp_info = "model not set"
        else:
            tdis = self.model.simulation.tdis
            nper = tdis.nper.data
            model_info = f"flopy {self.model.version} {self.model.name!r}"
            sp_info = "{} stress period{} with perlen: {} {}".format(
                nper,
                "" if nper == 1 else "s",
                abbr_str(list(tdis.perioddata.array["perlen"]), 4),
                tdis.time_units.data,
            )
        s = f"<{self.__class__.__name__}: {model_info}\n"
        reaches = self.reaches
        if reaches is not None:
            s += "  {} in reaches ({}): {}\n".format(
                len(reaches), reaches.index.name, abbr_str(list(reaches.index), 4)
            )
        diversions = self.diversions
        if diversions is not None:
            diversions_in_model = self.diversions[self.diversions["in_model"]]
            s += "  {} in diversions (iconr): {}\n".format(
                len(diversions_in_model), abbr_str(list(diversions_in_model.iconr), 4)
            )
        s += f"  {sp_info} />"
        return s

    def __eq__(self, other):
        """Return true if objects are equal."""
        import flopy

        try:
            for (ak, av), (bk, bv) in zip_longest(iter(self), iter(other)):
                if ak != bk:
                    return False
                is_none = (av is None, bv is None)
                if all(is_none):
                    continue
                elif any(is_none):
                    return False
                elif type(av) != type(bv):
                    return False
                elif isinstance(av, pd.DataFrame):
                    pd.testing.assert_frame_equal(av, bv)
                elif isinstance(av, pd.Series):
                    pd.testing.assert_series_equal(av, bv)
                elif isinstance(av, flopy.mf6.MFModel):
                    # basic test
                    assert str(av) == str(bv)
                else:
                    assert av == bv
            return True
        except (AssertionError, TypeError, ValueError):
            return False

    def __iter__(self):
        """Return object datasets with an iterator."""
        yield from super().__iter__()
        yield "tsvar", self.tsvar

    def __setstate__(self, state):
        """Set object attributes from pickle loads."""
        super().__setstate__(state)
        self.tsvar = state["tsvar"]

    def _init_package_df(self, *, style, defcols_names, auxiliary):
        """Return an initial DataFrame of PACKAGEDATA or STRESS_PERIOD_DATA."""
        if auxiliary is not None:
            if isinstance(auxiliary, str):
                auxiliary = [auxiliary]
            defcols_names += auxiliary
        dat = pd.DataFrame(self.reaches.copy())
        dat["idomain"] = self.model.dis.idomain.array[dat["k"], dat["i"], dat["j"]]
        cellid_none = dat["idomain"] < 1
        if "mask" in dat.columns:
            cellid_none |= dat["mask"]
        kij_l = list("kij")
        if style == "native":
            # Convert from zero-based to one-based notation
            dat[kij_l] += 1
            # use kij unstead of cellid
            idx = defcols_names.index("cellid")
            defcols_names[idx : idx + 1] = kij_l
            # convert kij to str, and store NONE in k, if needed
            dat[kij_l] = dat[kij_l].astype(str)
            if cellid_none.any():
                dat.loc[cellid_none, "k"] = "NONE"
                dat.loc[cellid_none, ["i", "j"]] = ""
        elif style == "flopy":
            # make cellid into tuple
            dat["cellid"] = dat[kij_l].to_records(index=False).tolist()
            if cellid_none.any():
                dat.loc[cellid_none, "cellid"] = "NONE"
            # Convert ridx from one-based to zero-based notation
            dat.index -= 1
        else:
            raise ValueError("'style' must be either 'native' or 'flopy'")
        return dat

    def _final_package_df(self, what, dat, *, defcols_names, boundname):
        """Return a final DataFrame of PACKAGEDATA or STRESS_PERIOD_DATA."""
        if boundname is None:
            boundname = "boundname" in dat.columns
        if boundname:
            defcols_names.append("boundname")
            if (na := dat["boundname"].isna()).any():
                dat.loc[~na, "boundname"] = dat.loc[~na, "boundname"].astype(str)
            else:
                dat["boundname"] = dat["boundname"].astype(str)
            # Check and enforce 40 character limit
            sel = dat.boundname.str.len() > 40
            if sel.any():
                self.logger.warning(
                    "clipping %d boundname entries to 40 character limit", sel.sum()
                )
                dat.loc[sel, "boundname"] = dat.loc[sel, "boundname"].str.slice(stop=40)
        # check missing columns
        missing = []
        for name in defcols_names:
            if name not in dat.columns:
                missing.append(name)
        if missing:
            missing_list = ", ".join(missing)
            raise KeyError(f"missing {len(missing)} {what}: {missing_list}")
        return dat.loc[:, defcols_names]

    def packagedata_frame(self, style: str, auxiliary: list = [], boundname=None):
        """Return DataFrame of PACKAGEDATA for MODFLOW 6 SFR.

        This DataFrame is derived from the reaches DataFrame.

        Parameters
        ----------
        style : str
            If "native", all indicies (including kij) use one-based notation.
            Also use k,i,j columns (as str) rather than cellid.
            If "flopy", all indices (including rno/ifno) use zero-based notation.
            Also use cellid as a tuple.
        auxiliary : str, list, optional
            String or list of auxiliary variable names. Default [].
        boundname : bool, optional
            Default None will determine if "boundname" column is added if
            available in reaches.columns.

        Returns
        -------
        DataFrame

        See Also
        --------
        SwnMf6.write_packagedata : Write native file.
        SwnMf6.flopy_packagedata : List of lists for flopy.

        Examples
        --------
        >>> import flopy
        >>> import geopandas
        >>> import swn
        >>> lines = geopandas.GeoSeries.from_wkt([
        ...    "LINESTRING (60 100, 60  80)",
        ...    "LINESTRING (40 130, 60 100)",
        ...    "LINESTRING (70 130, 60 100)"])
        >>> lines.index += 100
        >>> n = swn.SurfaceWaterNetwork.from_lines(lines)
        >>> sim = flopy.mf6.MFSimulation()
        >>> _ = flopy.mf6.ModflowTdis(sim, nper=1, time_units="days")
        >>> gwf = flopy.mf6.ModflowGwf(sim)
        >>> _ = flopy.mf6.ModflowGwfdis(
        ...     gwf, nrow=3, ncol=2, delr=20.0, delc=20.0, idomain=1,
        ...     length_units="meters", xorigin=30.0, yorigin=70.0)
        >>> nm = swn.SwnMf6.from_swn_flopy(n, gwf)
        >>> nm.default_packagedata()
        >>> nm.reaches["boundname"] = nm.reaches["segnum"]
        >>> nm.reaches["aux1"] = 2.0 + nm.reaches.index / 10.0
        >>> nm.packagedata_frame("native", auxiliary="aux1")
             k  i  j       rlen  rwid   rgrd  ...    man  ncon  ustrf  ndv  aux1  boundname
        ifno                                  ...
        1    1  1  1  18.027756  10.0  0.001  ...  0.024     1    0.0    0   2.1        101
        2    1  1  2   6.009252  10.0  0.001  ...  0.024     2    1.0    0   2.2        101
        3    1  2  2  12.018504  10.0  0.001  ...  0.024     2    1.0    0   2.3        101
        4    1  1  2  21.081851  10.0  0.001  ...  0.024     1    0.0    0   2.4        102
        5    1  2  2  10.540926  10.0  0.001  ...  0.024     2    1.0    0   2.5        102
        6    1  2  2  10.000000  10.0  0.001  ...  0.024     3    1.0    0   2.6        100
        7    1  3  2  10.000000  10.0  0.001  ...  0.024     1    1.0    0   2.7        100
        <BLANKLINE>
        [7 rows x 15 columns]
        >>> nm.packagedata_frame("flopy", boundname=False).drop(columns="man")
                 cellid       rlen  rwid   rgrd  rtp  rbth  rhk  ncon  ustrf  ndv
        ifno
        0     (0, 0, 0)  18.027756  10.0  0.001  1.0   1.0  1.0     1    0.0    0
        1     (0, 0, 1)   6.009252  10.0  0.001  1.0   1.0  1.0     2    1.0    0
        2     (0, 1, 1)  12.018504  10.0  0.001  1.0   1.0  1.0     2    1.0    0
        3     (0, 0, 1)  21.081851  10.0  0.001  1.0   1.0  1.0     1    0.0    0
        4     (0, 1, 1)  10.540926  10.0  0.001  1.0   1.0  1.0     2    1.0    0
        5     (0, 1, 1)  10.000000  10.0  0.001  1.0   1.0  1.0     3    1.0    0
        6     (0, 2, 1)  10.000000  10.0  0.001  1.0   1.0  1.0     1    1.0    0
        """  # noqa
        from flopy.mf6 import ModflowGwfsfr as Mf6Sfr

        defcols_names = [dt[0] for dt in Mf6Sfr.packagedata.dtype(self.model)]
        for idx in ['rno', 'ifno']:
            if idx in defcols_names:
                defcols_names.remove(idx)  # this is the index
        dat = self._init_package_df(
            style=style, defcols_names=defcols_names, auxiliary=auxiliary)
        # MF6 > 6.2.2 changed from rno to ifno
        # TODO: track ifno all the way through?
        if 'ifno' in defcols_names:
            dat.index.name = 'rno'
        if "rlen" not in dat.columns:
            dat.loc[:, "rlen"] = dat.geometry.length
        dat["ncon"] = dat[from_ridxsname].apply(len) + (dat[to_ridxname] > 0).astype(
            int
        )
        dat["ndv"] = 0
        if self.diversions is not None:
            div_to_ridxsname = f"div_to_{ridxname}s"
            div_from_ridxname = f"div_from_{ridxname}"
            dat["ncon"] += dat[div_to_ridxsname].apply(len) + (
                dat[div_from_ridxname] > 0
            ).astype(int)
            ndv = (
                self.diversions[self.diversions["in_model"]]
                .groupby(ridxname)
                .count()
                .in_model
            )
            if style == "flopy":
                ndv.index -= 1
            dat.loc[ndv.index, "ndv"] = ndv
        return self._final_package_df(
            "packagedata reaches series",
            dat,
            defcols_names=defcols_names,
            boundname=boundname,
        )

    def write_packagedata(self, fname: str, auxiliary: list = [], boundname=None):
        """
        Write PACKAGEDATA file for MODFLOW 6 SFR.

        Parameters
        ----------
        fname : str
            Output file name.
        auxiliary : str, list, optional
            String or list of auxiliary variable names. Default [].
        boundname : bool, optional
            Default None will determine if "boundname" column is added if
            available in reaches.columns.

        Returns
        -------
        None

        See Also
        --------
        swn.file.read_formatted_frame : Read formatted file as a DataFrame.

        """
        pn = self.packagedata_frame("native", auxiliary=auxiliary, boundname=boundname)
        write_formatted_frame(pn, fname)

    def flopy_packagedata(self, auxiliary: list = [], boundname=None):
        """Return list of lists for flopy.

        Parameters
        ----------
        auxiliary : str, list, optional
            String or list of auxiliary variable names. Default [].
        boundname : bool, optional
            Default None will determine if "boundname" column is added if
            available in reaches.columns.

        Returns
        -------
        list

        """
        df = self.packagedata_frame("flopy", auxiliary=auxiliary, boundname=boundname)
        return [list(x) for x in df.itertuples()]

    def connectiondata_series(self, style: str):
        """Return Series of CONNECTIONDATA for MODFLOW 6 SFR.

        Parameters
        ----------
        style : str
            If "native", all indicies (including kij) use one-based notation.
            If "flopy", all indices (including rno/ifno) use zero-based notation.
        """

        def nonzerolst(x, neg=False):
            if neg:
                return [-x] if x > 0 else []
            else:
                return [x] if x > 0 else []

        ridxname = self.reach_index_name
        from_ridxsname = f"from_{ridxname}s"
        to_ridxname = f"to_{ridxname}"
        if self.diversions is not None:
            div_from_ridxname = f"div_from_{ridxname}"
            div_to_ridxsname = f"div_to_{ridxname}s"
            res = (
                self.reaches[from_ridxsname].apply(sorted)
                + self.reaches[div_from_ridxname].apply(nonzerolst, neg=False)
                + self.reaches[to_ridxname].apply(nonzerolst, neg=True)
                + self.reaches[div_to_ridxsname]
                .apply(sorted)
                .apply(lambda x: [-i for i in x])
            )
        else:
            res = self.reaches[from_ridxsname].apply(sorted) + self.reaches[
                to_ridxname
            ].apply(nonzerolst, neg=True)

        if style == "native":
            # keep one-based notation, but convert list to str
            return res
        elif style == "flopy":
            # Convert ridx from one-based to zero-based notation
            res.index -= 1
            return res.apply(lambda x: [v - 1 if v > 0 else v + 1 for v in x])
        else:
            raise ValueError("'style' must be either 'native' or 'flopy'")

    def write_connectiondata(self, fname: str):
        """
        Write CONNECTIONDATA file for MODFLOW 6 SFR.

        Parameters
        ----------
        fname : str
            Output file name.

        Returns
        -------
        None

        """
        cn = self.connectiondata_series("native")
        icn = [f"ic{n+1}" for n in range(cn.apply(len).max())]
        rowfmt = f"{{:>{len(str(cn.index.max()))}}} {{}}\n"
        ridxlen = 1 + len(str(len(self.reaches)))
        cn = cn.apply(lambda x: " ".join(str(v).rjust(ridxlen) for v in x))
        with open(fname, "w") as f:
            f.write(f"# {self.reach_index_name} {' '.join(icn)}\n")
            for ridx, ic in cn.items():
                f.write(rowfmt.format(ridx, ic))

    def flopy_connectiondata(self):
        """Return list of lists for flopy.

        Returns
        -------
        list

        """
        s = self.connectiondata_series("flopy")
        return (s.index.to_series().apply(lambda x: list([x])) + s).to_list()

    def diversions_frame(self, style: str):
        """Return DataFrame of DIVERSIONS for MODFLOW 6 SFR.

        This DataFrame is derived from :py:attr:`diversions`.

        Parameters
        ----------
        style : str
            If "native", all indicies use one-based notation.
            If "flopy", all indices use zero-based notation.

        Returns
        -------
        DataFrame

        See Also
        --------
        SwnMf6.write_diversions : Write native file.
        SwnMf6.flopy_diversions : List of lists for flopy.

        Examples
        --------
        >>> import flopy
        >>> import geopandas
        >>> import swn
        >>> lines = geopandas.GeoSeries.from_wkt([
        ...    "LINESTRING (60 100, 60  80)",
        ...    "LINESTRING (40 130, 60 100)",
        ...    "LINESTRING (70 130, 60 100)"])
        >>> lines.index += 100
        >>> n = swn.SurfaceWaterNetwork.from_lines(lines)
        >>> diversions = geopandas.GeoSeries.from_wkt([
        ...    "POINT (58 100)", "POINT (62 100)",
        ...    "POINT (59 95)", "POINT (61 92)"]).to_frame("geometry")
        >>> diversions.index += 11
        >>> n.set_diversions(diversions)
        >>> sim = flopy.mf6.MFSimulation()
        >>> _ = flopy.mf6.ModflowTdis(sim, nper=1, time_units="days")
        >>> gwf = flopy.mf6.ModflowGwf(sim)
        >>> _ = flopy.mf6.ModflowGwfdis(
        ...     gwf, nrow=3, ncol=2, delr=20.0, delc=20.0, idomain=1,
        ...     length_units="meters", xorigin=30.0, yorigin=70.0)
        >>> nm = swn.SwnMf6.from_swn_flopy(n, gwf)
        >>> nm.diversions_frame("native")
            ifno  idv  iconr cprior
        11    3     1      8   upto
        12    5     1      9   upto
        13    6     1     10   upto
        14    6     2     11   upto
        >>> nm.diversions_frame("flopy")
            ifno  idv  iconr cprior
        11     2    0      7   upto
        12     4    0      8   upto
        13     5    0      9   upto
        14     5    1     10   upto
        """  # noqa
        from flopy.mf6 import ModflowGwfsfr as Mf6Sfr

        defcols_dtype = Mf6Sfr.diversions.empty(self.model).dtype
        if self.diversions is None:
            self.logger.warning("diversions not set")
            return pd.DataFrame(np.recarray(0, dtype=defcols_dtype))
        defcols_names = list(defcols_dtype.names)
        # MF6 > 6.2.2 changed from rno to ifno
        # TODO: track ifno all the way through?
        if 'ifno' in defcols_names:
            dat.index.name = 'rno'
        dat = pd.DataFrame(self.diversions[self.diversions["in_model"]].copy())
        if "cprior" not in dat.columns:
            self.logger.info("diversions missing cprior; assuming UPTO")
            dat["cprior"] = "upto"
        # checking missing columns
        div_columns = set(dat.columns)
        missing = set(defcols_names).difference(div_columns)
        if missing:
            missing_l = []
            for name in defcols_names:
                if name not in div_columns:
                    missing_l.append(name)
            raise KeyError(
                "missing {} diversions dataset(s): {}".format(
                    len(missing_l), ", ".join(sorted(missing_l))
                )
            )
        if style == "native":
            pass
        elif style == "flopy":
            # Convert ridx from one-based to zero-based notation
            dat[[self.reach_index_name, "idv", "iconr"]] -= 1
        else:
            raise ValueError("'style' must be either 'native' or 'flopy'")
        return dat[defcols_names]

    def write_diversions(self, fname: str):
        """Write DIVERSIONS file for MODFLOW 6 SFR.

        File can be used in a ``OPEN/CLOSE`` statement for a DIVERSIONS block,
        which can be set with ``diversions={"filename": "diversions.dat"}`` in
        :py:meth:`set_sfr_obj`.

        This method is based on :py:meth:`diversions_frame`.

        Parameters
        ----------
        fname : str
            Output file name.

        See Also
        --------
        swn.file.read_formatted_frame : Read formatted file as a DataFrame.
        """
        dat = self.diversions_frame("native")
        write_formatted_frame(dat, fname, index=False)

    def flopy_diversions(self):
        """Return list of lists for flopy.

        This method is based on :py:meth:`diversions_frame`.

        Returns
        -------
        list

        """
        dat = self.diversions_frame("flopy")
        return [list(x) for x in dat.itertuples(index=False)]

    def package_period_frame(
        self, package: str, style: str, auxiliary: list = [], boundname=None
    ):
        """Return DataFrame of PERIOD data for MODFLOW 6 packages.

        This DataFrame is derived from the reaches DataFrame, and is
        implemented using flopy.mf6 package's ``stress_period_data``.

        Parameters
        ----------
        package : str
            MODFLOW 6 package name, such as "drn" for DRAIN, or "GWT/SRC" for
            Mass Source Loading.
        style : str
            If "native", all indicies (including kij) use one-based notation.
            Also use k,i,j columns (as str) rather than cellid.
            If "flopy", all indices (including rno/ifno) use zero-based notation.
            Also use cellid as a tuple.
        auxiliary : str, list, optional
            String or list of auxiliary variable names. Default [].
        boundname : bool, optional
            Default None will determine if "boundname" column is added if
            available in reaches.columns.

        Returns
        -------
        DataFrame

        See Also
        --------
        SwnMf6.write_package_period : Write native file.
        SwnMf6.flopy_package_period : Dict of lists of lists for flopy.

        Examples
        --------
        >>> import flopy
        >>> import geopandas
        >>> import swn
        >>> lines = geopandas.GeoSeries.from_wkt([
        ...    "LINESTRING (60 100, 60  80)",
        ...    "LINESTRING (40 130, 60 100)",
        ...    "LINESTRING (70 130, 60 100)"])
        >>> lines.index += 100
        >>> n = swn.SurfaceWaterNetwork.from_lines(lines)
        >>> sim = flopy.mf6.MFSimulation()
        >>> _ = flopy.mf6.ModflowTdis(sim, nper=1, time_units="days")
        >>> gwf = flopy.mf6.ModflowGwf(sim)
        >>> _ = flopy.mf6.ModflowGwfdis(
        ...     gwf, nrow=3, ncol=2, delr=20.0, delc=20.0, top=10, botm=0,
        ...     idomain=1, length_units="meters", xorigin=30.0, yorigin=70.0)
        >>> nm = swn.SwnMf6.from_swn_flopy(n, gwf)
        >>> nm.set_reach_data_from_array("elev", gwf.dis.top.array)
        >>> nm.reaches["dlen"] = nm.reaches.length
        >>> nm.reaches["cond"] = nm.reaches.dlen * 10.0
        >>> nm.reaches["boundname"] = nm.reaches["segnum"]
        >>> nm.package_period_frame("drn", "native", auxiliary="dlen")
                 k  i  j  elev        cond       dlen boundname
        per ifno
        1   1    1  1  1  10.0  180.277564  18.027756       101
            2    1  1  2  10.0   60.092521   6.009252       101
            3    1  2  2  10.0  120.185043  12.018504       101
            4    1  1  2  10.0  210.818511  21.081851       102
            5    1  2  2  10.0  105.409255  10.540926       102
            6    1  2  2  10.0  100.000000  10.000000       100
            7    1  3  2  10.0  100.000000  10.000000       100
        >>> nm.package_period_frame("drn","flopy", boundname=False)
                    cellid  elev        cond
        per ifno
        0   0    (0, 0, 0)  10.0  180.277564
            1    (0, 0, 1)  10.0   60.092521
            2    (0, 1, 1)  10.0  120.185043
            3    (0, 0, 1)  10.0  210.818511
            4    (0, 1, 1)  10.0  105.409255
            5    (0, 1, 1)  10.0  100.000000
            6    (0, 2, 1)  10.0  100.000000
        """  # noqa
        Mf6pak = get_flopy_mf6_package(package)
        lst_tpl = Mf6pak.stress_period_data
        defcols_names = [dt[0] for dt in lst_tpl.dtype(self.model)]
        # MF6 > 6.2.2 changed from rno to ifno
        # TODO: track ifno all the way through?
        if 'ifno' == defcols_names.index.name:
            dat.index.name = 'rno'
        dat = self._init_package_df(
            style=style, defcols_names=defcols_names, auxiliary=auxiliary
        )
        dat = self._final_package_df(
            f"{Mf6pak.__name__} reaches series",
            dat,
            defcols_names=defcols_names,
            boundname=boundname,
        )
        if self.model.simulation.tdis.nper.data != 1:
            self.logger.warning(
                "only preparing PERIOD data for the first stress period of %s",
                self.model.simulation.tdis.nper.data,
            )
        dat["per"] = 1
        if style == "flopy":
            dat["per"] -= 1
        return dat.reset_index().set_index(["per", self.reach_index_name])

    def write_package_period(
        self, package: str, fname, auxiliary: list = [], boundname=None
    ):
        """
        Write PERIOD file for MODFLOW 6 packages, to be used within OPEN/CLOSE.

        Parameters
        ----------
        package : str
            MODFLOW 6 package name, such as "drn" for DRAIN, or "GWT/SRC" for
            Mass Source Loading.
        fname : str or path-like
            Output file name, with str formatting for stress period number
            which starts numbering from 1. For example, "drn_sp{:02d}.txt" to
            create "drn_sp01.txt" for the first stress period.
        auxiliary : str, list, optional
            String or list of auxiliary variable names. Default [].
        boundname : bool, optional
            Default None will determine if "boundname" column is added if
            available in reaches.columns.

        Returns
        -------
        None

        See Also
        --------
        swn.file.read_formatted_frame : Read formatted file as a DataFrame.
        SwnMf6.package_period_frame : Dataframe generator.
        SwnMf6.flopy_package_period : Dict of lists of lists for flopy.

        Examples
        --------
        >>> import flopy
        >>> import geopandas
        >>> import pandas as pd
        >>> import swn
        >>> from tempfile import TemporaryDirectory
        >>> from pathlib import Path
        >>> lines = geopandas.GeoSeries.from_wkt([
        ...    "LINESTRING (60 100, 60  80)",
        ...    "LINESTRING (40 130, 60 100)",
        ...    "LINESTRING (70 130, 60 100)"])
        >>> lines.index += 100
        >>> n = swn.SurfaceWaterNetwork.from_lines(lines)
        >>> tmp_dir_obj = TemporaryDirectory()
        >>> dir = Path(tmp_dir_obj.name)
        >>> sim = flopy.mf6.MFSimulation(sim_ws=str(dir))
        >>> _ = flopy.mf6.ModflowTdis(sim, nper=1, time_units="days")
        >>> gwf = flopy.mf6.ModflowGwf(sim)
        >>> _ = flopy.mf6.ModflowGwfdis(
        ...     gwf, nrow=3, ncol=2, delr=20.0, delc=20.0, top=10, botm=0,
        ...     idomain=1, length_units="meters", xorigin=30.0, yorigin=70.0)
        >>> nm = swn.SwnMf6.from_swn_flopy(n, gwf)
        >>> nm.set_reach_data_from_array("elev", gwf.dis.top.array)
        >>> nm.reaches["dlen"] = nm.reaches.length.round(3)
        >>> nm.reaches["cond"] = 15.0
        >>> nm.reaches["boundname"] = nm.reaches["segnum"]
        >>> fname_tpl = dir / "model_drn_{:02d}.dat"
        >>> fname_01 = dir / "model_drn_01.dat"
        >>> nm.write_package_period("drn", fname_tpl, auxiliary="dlen")
        >>> print(fname_01.read_text(), end="")
        #k i j  elev  cond   dlen boundname
         1 1 1  10.0  15.0 18.028 101
         1 1 2  10.0  15.0  6.009 101
         1 2 2  10.0  15.0 12.019 101
         1 1 2  10.0  15.0 21.082 102
         1 2 2  10.0  15.0 10.541 102
         1 2 2  10.0  15.0 10.000 100
         1 3 2  10.0  15.0 10.000 100
        >>> drn = nm.set_package_obj(
        ...    "drn", pname="swn_drn", auxmultname="dlen",
        ...    stress_period_data={0: {"filename": fname_01.name}})
        >>> sim.write_simulation(silent=True)
        >>> success, buff = sim.run_simulation()  # doctest: +SKIP
        >>> tmp_dir_obj.cleanup()

        """
        spdf = self.package_period_frame(
            package, "native", auxiliary=auxiliary, boundname=boundname
        )
        if isinstance(fname, os.PathLike):
            fname = os.fsdecode(fname)
        for per, df in spdf.groupby("per"):
            per_fname = fname.format(per)
            self.logger.debug("writing period file %s", per_fname)
            write_formatted_frame(df, per_fname, index=False)

    def flopy_package_period(self, package: str, auxiliary: list = [], boundname=None):
        """Return dict of lists of lists for flopy's stress_period_data.

        Parameters
        ----------
        package : str
            MODFLOW 6 package name, such as "drn" for DRAIN, or "GWT/SRC" for
            Mass Source Loading.
        auxiliary : str, list, optional
            String or list of auxiliary variable names. Default [].
        boundname : bool, optional
            Default None will determine if "boundname" column is added if
            available in reaches.columns.

        Returns
        -------
        dict

        See Also
        --------
        SwnMf6.package_period_frame : Dataframe generator.
        SwnMf6.write_package_period : Write native file.

        Examples
        --------
        >>> import flopy
        >>> import geopandas
        >>> import pandas as pd
        >>> import swn
        >>> lines = geopandas.GeoSeries.from_wkt([
        ...    "LINESTRING (60 100, 60  80)",
        ...    "LINESTRING (40 130, 60 100)",
        ...    "LINESTRING (70 130, 60 100)"])
        >>> lines.index += 100
        >>> n = swn.SurfaceWaterNetwork.from_lines(lines)
        >>> sim = flopy.mf6.MFSimulation()
        >>> _ = flopy.mf6.ModflowTdis(sim, nper=1, time_units="days")
        >>> gwf = flopy.mf6.ModflowGwf(sim)
        >>> _ = flopy.mf6.ModflowGwfdis(
        ...     gwf, nrow=3, ncol=2, delr=20.0, delc=20.0, top=10, botm=0,
        ...     idomain=1, length_units="meters", xorigin=30.0, yorigin=70.0)
        >>> nm = swn.SwnMf6.from_swn_flopy(n, gwf)
        >>> nm.set_reach_data_from_array("elev", gwf.dis.top.array)
        >>> nm.reaches["dlen"] = nm.reaches.length.round(3)
        >>> nm.reaches["cond"] = 15.0
        >>> nm.reaches["boundname"] = nm.reaches["segnum"]
        >>> drn = nm.set_package_obj("drn", auxmultname="dlen")
        >>> pd.DataFrame(drn.stress_period_data.data[0])
              cellid  elev  cond    dlen boundname
        0  (0, 0, 0)  10.0  15.0  18.028       101
        1  (0, 0, 1)  10.0  15.0   6.009       101
        2  (0, 1, 1)  10.0  15.0  12.019       101
        3  (0, 0, 1)  10.0  15.0  21.082       102
        4  (0, 1, 1)  10.0  15.0  10.541       102
        5  (0, 1, 1)  10.0  15.0  10.000       100
        6  (0, 2, 1)  10.0  15.0  10.000       100

        """
        spdf = self.package_period_frame(
            package, "flopy", auxiliary=auxiliary, boundname=boundname
        )
        spd = {}
        for per, df in spdf.groupby("per"):
            spd[per] = [list(x) for x in df.itertuples(index=False)]
        return spd

    _tsvar_meta = pd.DataFrame(
        [
            ("status", "str"),
            ("manning", "ts"),
            ("stage", "ts"),
            ("inflow", "ts"),
            ("rainfall", "ts"),
            ("evaporation", "ts"),
            ("runoff", "ts"),
            ("divflow", "ts"),
            ("upstream_fraction", "float"),
            ("auxname", "str"),
            ("auxval", "ts"),
        ],
        columns=["name", "type"],
    ).set_index("name")

    def _check_tsvar_name(self, name: str):
        """Helper method to check name used for set_tsvar_* methods."""
        # TODO
        raise NotImplementedError("method is not finished")
        if not isinstance(name, str):
            raise ValueError("name must be str type")
        elif name not in self._tsvar_meta.index:
            names = ", ".join(repr(n) for n in self._tsvar_meta.index)
            raise KeyError(f"could not find {name!r} in {names}")

    def set_tsvar_from_segments(self, name: str, data, where: str = "start"):
        """Set time-varying data from data defined at segments.

        Parameters
        ----------
        name : str
            Name for dataset to use in a tsvar dict.
        data : dict, pandas.Series or pandas.DataFrame
            Data to assigned to tsvar, indexed by segnum.
        where : str, optional, default "start"
            For segments represented by multiple reaches, pick which reach
            to represent data. Default "start" is the upper-most reach.
        """
        # TODO
        raise NotImplementedError("method is not finished")
        self._check_tsvar_name(name)
        if not np.isscalar(data):
            raise ValueError("only non-scalar data can be set")
        if where != "start":
            raise NotImplementedError("only 'start' is currently implemented")

    def default_packagedata(
        self,
        hyd_cond1=1.0,
        hyd_cond_out=None,
        thickness1=1.0,
        thickness_out=None,
        width1=None,
        width_out=None,
        roughch=0.024,
    ):
        """High-level function to set MODFLOW 6 SFR PACKAGEDATA defaults.

        Parameters
        ----------
        hyd_cond1 : float or pandas.Series, optional
            Hydraulic conductivity of the streambed, as a global or per top of
            each segment. Used for reach rhk. Default 1.
        hyd_cond_out : None, float or pandas.Series, optional
            Similar to thickness1, but for the hydraulic conductivity of each
            segment outlet. If None (default), the same hyd_cond1 value for the
            top of the outlet segment is used for the bottom.
        thickness1 : float or pandas.Series, optional
            Thickness of the streambed, as a global or per top of each segment.
            Used for reach rbth. Default 1.
        thickness_out : None, float or pandas.Series, optional
            Similar to thickness1, but for the bottom of each segment outlet.
            If None (default), the same thickness1 value for the top of the
            outlet segment is used for the bottom.
        width1 : float or pandas.Series, optional
            Channel width, as a global or per top of each segment. Used for
            reach rwid. Default None will first try finding "width" from
            "segments", otherwise will use 10.
        width_out : None, float or pandas.Series, optional
            Similar to width1, but for the bottom of each segment outlet.
            If None (default), use a constant width1 value for segment.
        roughch : float or pandas.Series, optional
            Manning's roughness coefficient for the channel. If float, then
            this is a global value, otherwise it is per-segment with a Series.
            Default 0.024.

        Returns
        -------
        None
        """
        self.logger.info("default_packagedata: using high-level function")

        if width1 is None:
            # Determine width based on available data
            if "width" in self.segments.columns:
                width1 = self.segments.width
                action = "taken from segments frame, with range %.3f to %.3f"
                action_args = width1.min(), width1.max()
            else:
                width1 = 10.0
                action = "not found in segments frame; using default %s"
                action_args = (width1,)
            self.logger.info(f"default_packagedata: 'rwd' {action}", *action_args)
        self.set_reach_data_from_segments("rwid", width1, width_out, method="constant")

        if "rgrd" not in self.reaches.columns:
            self.logger.info(
                "default_packagedata: 'rgrd' not yet evaluated, setting with "
                "set_reach_slope()"
            )
            self.set_reach_slope()

        # TODO: add a similar method for rtp? set_reaches_top()?

        # Get stream values from top of model
        self.set_reach_data_from_array("rtp", self.model.dis.top.array)
        if "zcoord_avg" in self.reaches.columns:
            # Created by set_reach_slope(); be aware of NaN
            nn = ~self.reaches.zcoord_avg.isnull()
            self.reaches.loc[nn, "rtp"] = self.reaches.loc[nn, "zcoord_avg"]

        # Assign remaining reach datasets
        self.set_reach_data_from_segments("rbth", thickness1, thickness_out)
        self.set_reach_data_from_segments("rhk", hyd_cond1, hyd_cond_out, True)
        self.set_reach_data_from_segments("man", roughch)
        if self.diversions is not None:
            diversion_reaches = self.reaches["diversion"]
            self.reaches.loc[diversion_reaches, "rwid"] = 1.0
            self.reaches.loc[diversion_reaches, "rhk"] = 0.0

    def set_sfr_obj(self, auxiliary=None, boundnames=None, **kwds):
        """Set MODFLOW 6 SFR package data to flopy model.

        Parameters
        ----------
        auxiliary : list, optional
            List of auxiliary names, which must be columns in the reaches
            frame.
        boundnames : bool, optional
            Sets the BOUNDAMES option, with names provided by a "boundname"
            column of the reaches frame. Default None will set this True
            if column exists.
        **kwds : dict, optional
            Passed to flopy.mf6.ModflowGwfsfr.

        Returns
        -------
        flopy.mf6.ModflowGwfsfr

        """
        import flopy

        if auxiliary is not None:
            kwds["auxiliary"] = auxiliary
        if boundnames is None:
            boundnames = "boundname" in self.reaches.columns
        if boundnames:
            kwds["boundnames"] = True
        kwds["nreaches"] = len(self.reaches)
        if "packagedata" not in kwds:
            kwds["packagedata"] = self.flopy_packagedata(
                auxiliary=auxiliary, boundname=boundnames
            )
        if "connectiondata" not in kwds:
            kwds["connectiondata"] = self.flopy_connectiondata()
        if self.diversions is not None and "diversions" not in kwds:
            kwds["diversions"] = self.flopy_diversions()

        return flopy.mf6.ModflowGwfsfr(self.model, **kwds)

    def set_package_obj(self, package: str, auxiliary=None, boundnames=None, **kwds):
        """Set MODFLOW 6 package data to flopy model.

        Parameters
        ----------
        package : str
            MODFLOW 6 package name, such as "drn" for DRAIN, or "GWT/SRC" for
            Mass Source Loading.
        auxiliary : list, optional
            List of auxiliary names, which must be columns in the reaches
            frame.
        boundnames : bool, optional
            Sets the BOUNDAMES option, with names provided by a "boundname"
            column of the reaches frame. Default None will set this True
            if column exists.
        **kwds : dict, optional
            Passed to flopy.mf6 package.

        Returns
        -------
        flopy.mf6.mfpackage.MFPackage
            Subclass instance.

        """
        if auxiliary is None and "auxmultname" in kwds:
            auxiliary = kwds["auxmultname"]
        if auxiliary is not None:
            kwds["auxiliary"] = auxiliary
        if boundnames is None:
            boundnames = "boundname" in self.reaches.columns
        if boundnames:
            kwds["boundnames"] = True
        if "stress_period_data" not in kwds:
            kwds["stress_period_data"] = self.flopy_package_period(
                package=package, auxiliary=auxiliary, boundname=boundnames
            )
        if "maxbound" not in kwds:
            kwds["maxbound"] = len(self.reaches)

        Mf6pak = get_flopy_mf6_package(package)
        # self.logger.debug("%s called with kwds: %s", package, kwds)
        return Mf6pak(self.model, **kwds)

    def _segbyseg_elevs(
        self, minslope=1e-4, fix_dis=True, minthick=0.5, min_incise=0.2, max_str_z=None
    ):
        """
        Fix reach elevations but by using segment definition and setting sane
        segment elevations first.

        Need to ensure reach elevation is:
            0. below the top
            1. below the upstream reach
            2. above the minimum slope to the bottom reach elevation
            3. above the base of layer 1
        segment by segment, reach by reach! Fun!
        """
        raise NotImplementedError("method not finished")

        def _check_reach_v_laybot(r, botms, buffer=1.0, rbed_elev=None):
            if rbed_elev is None:
                rbed_elev = r.strtop - r.strthick
            if (rbed_elev - buffer) < r.bot:
                # if new strtop is below layer one
                # drop bottom of layer one to accomodate stream
                # (top, bed thickness and buffer)
                new_elev = rbed_elev - buffer
                self.logger.debug(
                    "seg %s reach %s @ %s is below layer 1 bottom @ %s",
                    seg,
                    r.ireach,
                    rbed_elev,
                    r.bot,
                )
                self.logger.debug(
                    (
                        "dropping layer 1 bottom to %s to accommodate stream "
                        "@ i = %s, j = %s",
                        new_elev,
                        r.i,
                        r.j,
                    )
                )
                botms[0, r.i, r.j] = new_elev
            return botms

        # fix segments first
        # 0. Segments are below the model top
        # 1. Segments flow downstream
        # 2. Downstream segments are below upstream segments
        self.fix_segment_elevs(
            min_incise=min_incise, min_slope=minslope, max_str_z=max_str_z
        )
        self.reconcile_reach_strtop()
        buffer = 1.0  # 1 m (buffer to leave at the base of layer 1 -
        # also helps with precision issues)
        # make sure elevations are up-to-date
        # recalculate REACH strtop elevations
        self.reconcile_reach_strtop()
        self.add_model_topbot_to_reaches()
        # top read from dis as float32 so comparison need to be with like
        reachsel = self.reaches["top"] <= self.reaches["strtop"]
        reach_ij = tuple(self.reaches[["i", "j"]].values.T)
        self.logger.info(
            "%s segments with reaches above model top",
            self.reaches[reachsel]["iseg"].unique().shape[0],
        )
        # get segments with reaches above the top surface
        segsabove = (
            self.reaches[reachsel].groupby("iseg").size().sort_values(ascending=False)
        )
        # get incision gradient from segment elevups and elevdns
        # ("diff_up" and "diff_dn" are the incisions of the top and
        # bottom reaches from the segment data)
        self.segment_data["incgrad"] = (
            self.segment_data["diff_up"] - self.segment_data["diff_dn"]
        ) / self.segment_data["seglen"]
        # copy of layer 1 bottom (for updating to fit in stream reaches)
        layerbots = self.model.dis.botm.array.copy()
        # loop over each segment
        for seg in self.segment_data.index:  # (all segs)
            # selection for segment in reachdata and seg data
            rsel = self.reaches["iseg"] == seg
            segsel = self.segment_data.index == seg

            if seg in segsabove.index:
                # check top and bottom reaches are above layer 1 bottom
                # (not adjusting elevations of reaches)
                for reach in self.reaches[rsel].iloc[[0, -1]].itertuples():
                    layerbots = _check_reach_v_laybot(reach, layerbots, buffer)
                # apparent optimised incision based
                # on the incision gradient for the segment
                self.reaches.loc[rsel, "strtop_incopt"] = self.reaches.loc[
                    rsel, "top"
                ].subtract(self.segment_data.loc[segsel, "diff_up"].values[0]) + (
                    self.reaches.loc[rsel, "cmids"].subtract(
                        self.reaches.loc[rsel, "cmids"].values[0]
                    )
                    * self.segment_data.loc[segsel, "incgrad"].values[0]
                )
                # falls apart when the top elevation is not monotonically
                # decreasing down the segment (/always!)

                # bottom reach elevation:
                botreach_strtop = self.reaches[rsel]["strtop"].values[-1]
                # total segment length
                seglen = self.reaches[rsel]["seglen"].values[-1]
                # botreach_slope = minslope  # minimum slope of segment
                # top reach elevation and "length?":
                upreach_strtop = self.reaches[rsel]["strtop"].values[0]
                upreach_cmid = self.reaches[rsel]["cmids"].values[0]
                # use top reach as starting point

                # loop over reaches in segement from second to penultimate
                # (dont want to move elevup or elevdn)
                for reach in self.reaches[rsel][1:-1].itertuples():
                    # strtop that would result from minimum slope
                    # from upstream reach
                    strtop_withminslope = upreach_strtop - (
                        (reach.cmids - upreach_cmid) * minslope
                    )
                    # strtop that would result from minimum slope
                    # from bottom reach
                    strtop_min2bot = botreach_strtop + (
                        (seglen - reach.cmids) * minslope
                    )
                    # check "optimum incision" is below upstream elevation
                    # and above the minimum slope to the bottom reach
                    if reach.strtop_incopt < strtop_min2bot:
                        # strtop would give too shallow a slope to
                        # the bottom reach (not moving bottom reach)
                        self.logger.debug(
                            "seg %s reach %s, incopt is \\/ below minimum "
                            "slope from bottom reach elevation",
                            seg,
                            reach.ireach,
                        )
                        self.logger.debug("setting elevation to minslope from bottom")
                        # set to minimum slope from outreach
                        self.reaches.at[reach.Index, "strtop"] = strtop_min2bot
                        # update upreach for next iteration
                        upreach_strtop = strtop_min2bot
                    elif reach.strtop_incopt > strtop_withminslope:
                        # strtop would be above upstream or give
                        # too shallow a slope from upstream
                        self.logger.debug(
                            "seg %s reach %s, incopt /\\ above upstream",
                            seg,
                            reach.ireach,
                        )
                        self.logger.debug("setting elevation to minslope from upstream")
                        # set to minimum slope from upstream reach
                        self.reaches.at[reach.Index, "strtop"] = strtop_withminslope
                        # update upreach for next iteration
                        upreach_strtop = strtop_withminslope
                    else:
                        # strtop might be ok to set to "optimum incision"
                        self.logger.debug(
                            "seg %s reach %s, incopt is -- below upstream "
                            "reach and above the bottom reach",
                            seg,
                            reach.ireach,
                        )
                        # CHECK FIRST:
                        # if optimium incision would place it
                        # below the bottom of layer 1
                        if reach.strtop_incopt - reach.strthick < reach.bot + buffer:
                            # opt - stream thickness lower than layer 1 bottom
                            # (with a buffer)
                            self.logger.debug(
                                "seg %s reach %s, incopt - bot is x\\/ "
                                "below layer 1 bottom",
                                seg,
                                reach.ireach,
                            )
                            if (
                                reach.bot + reach.strthick + buffer
                                > strtop_withminslope
                            ):
                                # if layer bottom would put reach above
                                # upstream reach we can only set to
                                # minimum slope from upstream
                                self.logger.debug(
                                    "setting elevation to minslope from upstream"
                                )
                                self.reaches.at[
                                    reach.Index, "strtop"
                                ] = strtop_withminslope
                                upreach_strtop = strtop_withminslope
                            else:
                                # otherwise we can move reach so that it
                                # fits into layer 1
                                new_elev = reach.bot + reach.strthick + buffer
                                self.logger.debug(
                                    "setting elevation to %s, above layer 1 bottom",
                                    new_elev,
                                )
                                # set reach top so that it is above layer 1
                                # bottom with a buffer
                                # (allowing for bed thickness)
                                self.reaches.at[reach.Index, "strtop"] = (
                                    reach.bot + buffer + reach.strthick
                                )
                                upreach_strtop = new_elev
                        else:
                            # strtop ok to set to "optimum incision"
                            # set to "optimum incision"
                            self.logger.debug("setting elevation to incopt")
                            self.reaches.at[reach.Index, "strtop"] = reach.strtop_incopt
                            upreach_strtop = reach.strtop_incopt
                    # check if new stream top is above layer 1 with a buffer
                    # (allowing for bed thickness)
                    reachbed_elev = upreach_strtop - reach.strthick
                    layerbots = _check_reach_v_laybot(
                        reach, layerbots, buffer, reachbed_elev
                    )
                    upreach_cmid = reach.cmids
                    # upreach_slope=reach.slope
            else:
                # For segments that do not have reaches above top
                # check if reaches are below layer 1
                self.logger.debug("seg %s is always downstream and below the top", seg)
                for reach in self.reaches[rsel].itertuples():
                    reachbed_elev = reach.strtop - reach.strthick
                    layerbots = _check_reach_v_laybot(
                        reach, layerbots, buffer, reachbed_elev
                    )
            # OH CRAP need to update dis bottoms in reach df!
            # self.reaches["top"] = layerbots[
            #     tuple(self.reaches[["i", "j"]].values.T)]
            self.reaches["bot"] = layerbots[0][reach_ij]
        if fix_dis:
            # fix dis for incised reaches
            for k in range(self.model.dis.nlay - 1):
                laythick = (
                    layerbots[k] - layerbots[k + 1]
                )  # first one is layer 1 bottom - layer 2 bottom
                self.logger.debug("checking layer %s thicknesses", k + 2)
                thincells = laythick < minthick
                self.logger.debug("%s cells less than %s", thincells.sum(), minthick)
                laythick[thincells] = minthick
                layerbots[k + 1] = layerbots[k] - laythick
            self.model.dis.botm.set_data(layerbots)

    def _reachbyreach_elevs(
        self,
        minslope=1e-4,
        minincise=0.2,
        minthick=0.5,
        fix_dis=True,
        direction="downstream",
    ):
        """
        Fix reach elevations by just working from headwater to outlet
        (ignoring segment divisions).

        Need to ensure reach elevation is:
            0. below the top
            1. below the upstream reach
            2. above the minimum slope to the bottom reach elevation
            3. above the base of layer 1
        (nwt method went seg-by-seg then reach-by-reach)

        Parameters
        ----------
        minslope : float, defai;t 1e-4
            The minimum allowable slope between adjacent reaches
            (to be considered flowing downstream).
        minincise : float, default 0.2
            The minimum allowable incision of a reach below the model top.
        minthick : float, default 0.5
            The minimum thickness of stream bed. Will try to ensure that this
            is available below stream top before layer bottom.
        fix_dis : bool, default True
            Move layer elevations down where it is not possible to honor
            minimum slope without going below layer 1 bottom.
        direction : {'upstream', 'downstream'}
            NOT IMPLEMENTED
            Select whether elevations are set from ensuring a minimum slope
            in a upstream or downstream direction.
            If 'upstream' will honor elevation at outlet reach
                (if in model layer) and work upstream ensuring minimum slope.
            If 'downstream' will honor elevation at headwater reach
                (if in model layer) and work downstream ensuring minimum slope.
            Default: 'downstream',
            set elevations from headwaters to outlets.

        Returns
        -------

        """

        def _check_reach_v_laybot(r, botms, buffer=1.0, rbed_elev=None):
            if rbed_elev is None:
                rbed_elev = r.rtp - r.rbth
            if (rbed_elev - buffer) < r.bot:
                # if new strtop is below layer one
                # drop bottom of layer one to accomodate stream
                # (top, bed thickness and buffer)
                new_elev = rbed_elev - buffer
                name = getattr(r, "Index", None) or getattr(r, "name")
                self.logger.debug(
                    "reach %s @ %s is below layer 1 bottom @ %s", name, rbed_elev, r.bot
                )
                self.logger.debug(
                    "dropping layer 1 bottom to %s to accommodate stream "
                    "@ i = %s, j = %s",
                    new_elev,
                    r.i,
                    r.j,
                )
                botms[0, r.i, r.j] = new_elev
            return botms

        ridxname = self.reach_index_name
        from_ridxsname = f"from_{ridxname}s"
        to_ridxname = f"to_{ridxname}"
        if direction == "upstream":
            ustrm = True
        else:
            ustrm = False
        buffer = 1.0  # 1 m (buffer to leave at the base of layer 1 -
        # also helps with precision issues)
        # make sure elevations are up-to-date
        # recalculate REACH strtop elevations
        # self.reconcile_reach_strtop()
        self.add_model_topbot_to_reaches()
        # top read from dis as float32 so comparison need to be with like
        reachsel = self.reaches["top"] <= self.reaches["rtp"]
        reach_ij = tuple(self.reaches[["i", "j"]].values.T)
        self.logger.info("%s reaches above model top", reachsel.sum())
        # copy of layer 1 bottom (for updating to fit in stream reaches)
        layerbots = self.model.dis.botm.array.copy()
        sel = self.reaches[from_ridxsname] == set()
        if self.diversions is not None:
            sel &= ~self.reaches["diversion"]
        headreaches = self.reaches.loc[sel]
        # through reaches DF
        for hdrch in headreaches.itertuples():
            # check if head reach above model top
            if hdrch.rtp > hdrch.top - minincise:
                # set to below model top
                upreach_rtp = hdrch.top - minincise
                self.reaches.at[hdrch.Index, "rtp"] = upreach_rtp
            else:
                upreach_rtp = hdrch.rtp
            inc_up = hdrch.top - upreach_rtp
            # get profile of reaches from this headwater
            dsegs = self._swn.gather_segnums(downstream=hdrch.segnum)
            segs = [hdrch.segnum] + dsegs
            reaches = self.reaches.loc[self.reaches.segnum.isin(segs)].sort_index()
            # get outflow reach for this profile
            # maybe can't rely on it being the last one
            # the sort_index() should order (assuming ridx increases downstream)
            # so last should be to_rno == 0
            assert (
                reaches.iloc[-1][to_ridxname] == 0
            ), "reach numbers possibly not increasing downstream"
            outflow = reaches.iloc[-1]
            # check if outflow above model top
            if outflow.rtp > outflow.top - minincise:
                # set below model top
                botreach_rtp = outflow.top - minincise
                self.reaches.at[outflow.name, "rtp"] = botreach_rtp
            else:
                botreach_rtp = outflow.rtp
            inc_dn = outflow.top - botreach_rtp
            # total profile length
            totlen = reaches.rlen.sum()
            reaches["mid_dist"] = reaches["rlen"].cumsum() - reaches["rlen"] / 2.0
            if ustrm:  # switch order
                reaches = reaches.sort_index(ascending=False)
                prevreach_top = botreach_rtp
                # get mid length for each reach
                prevreach_mid = reaches.iloc[0].mid_dist
            else:
                prevreach_top = upreach_rtp
                prevreach_mid = reaches.iloc[0].mid_dist

            # get incision gradient from headwater and outflow incision
            # ("inc_up" and "inc_dn" are the incisions of the top and
            # bottom reaches) # TODO is this stil meaningfull?
            incgrad = (inc_up - inc_dn) / totlen
            # apparent optimised incision based
            # on the incision gradient for the segment
            reaches["strtop_incopt"] = (reaches.top - inc_up) + (
                (reaches.mid_dist - prevreach_mid) * incgrad
            )
            layerbots = _check_reach_v_laybot(reaches.iloc[0], layerbots, buffer)
            # loop over current profile from second to penultimate
            # (dont want to move endpoints)
            # TODO maybe more flexibility on bottom reach incision
            for reach in reaches.iloc[1:].itertuples():
                # strtop that would result from minimum slope
                # from upstream reach
                rtp = reach.rtp
                strtop_withminslope = prevreach_top - (
                    (reach.mid_dist - prevreach_mid) * minslope
                )
                if rtp > reach.top - minincise:
                    # current reach rtp is above the model top
                    self.logger.debug(
                        "reach %s, rtp is: /\\ above model top", reach.Index
                    )
                    minslope_cond = (
                        reach.strtop_incopt < strtop_withminslope
                        if ustrm
                        else reach.strtop_incopt > strtop_withminslope
                    )
                    if minslope_cond:
                        # incision according to incision gradient would be
                        # above upstream/below downstream or give too shallow
                        # a slope from previous:
                        self.logger.debug(
                            "reach %s, incopt is: %s",
                            reach.Index,
                            "\\/ below downstream" if ustrm else "/\\ above upstream",
                        )
                        self.logger.debug("setting elevation to minslope from previous")
                        # set to minimum slope from previous reach
                        self.reaches.at[reach.Index, "rtp"] = strtop_withminslope
                    else:
                        # rtp might be ok to set to "optimum incision"
                        self.logger.debug(
                            "reach %s, incopt is: %s",
                            reach.Index,
                            "above downstream" if ustrm else "below upstream",
                        )
                        # CHECK FIRST:
                        # if optimium incision would place it
                        # below the bottom of layer 1
                        if (reach.strtop_incopt - reach.rbth) < reach.bot + buffer:
                            # opt - stream thickness lower than layer 1 bottom
                            # (with a buffer)
                            self.logger.debug(
                                "reach %s, incopt is: x\\/ below layer 1 bottom",
                                reach.Index,
                            )
                            new_elev = reach.bot + reach.rbth + buffer
                            minslope_cond = (
                                new_elev < strtop_withminslope
                                if ustrm
                                else new_elev > strtop_withminslope
                            )
                            if minslope_cond:
                                # if layer bottom would put reach above
                                # upstream reach/below downstream reach we
                                # can only set to minimum slope from previous
                                self.logger.debug(
                                    "setting elevation to minslope from previous"
                                )
                                self.reaches.at[
                                    reach.Index, "rtp"
                                ] = strtop_withminslope
                                # this may still leave us below the
                                # bottom of layer
                            else:
                                # otherwise we can move reach so that it
                                # fits into layer 1
                                self.logger.debug(
                                    "setting elevation to %s, above layer 1 bottom",
                                    new_elev,
                                )
                                # set reach top so that it is above layer 1
                                # bottom with a buffer
                                # (allowing for bed thickness)
                                self.reaches.at[reach.Index, "rtp"] = new_elev
                        else:
                            # strtop ok to set to "optimum incision"
                            # set to "optimum incision"
                            self.logger.debug("setting elevation to incopt")
                            self.reaches.at[reach.Index, "rtp"] = reach.strtop_incopt
                else:
                    self.logger.debug("reach %s, rtp is: below model top", reach.Index)
                    minslope_cond = (
                        rtp < strtop_withminslope
                        if ustrm
                        else rtp > strtop_withminslope
                    )
                    if (rtp - reach.rbth) < reach.bot + buffer:
                        # rtp is below the bottom of layer 1
                        self.logger.debug(
                            "reach %s, rtp is: x\\/ below layer 1 bottom", reach.Index
                        )
                        new_elev = reach.bot + reach.rbth + buffer
                        minslope_cond = (
                            new_elev < strtop_withminslope
                            if ustrm
                            else new_elev > strtop_withminslope
                        )
                        if minslope_cond:
                            # if layer bottom would put reach above
                            # upstream reach we can only set to
                            # minimum slope from upstream
                            self.logger.debug(
                                "setting elevation to minslope from upstream"
                            )
                            self.reaches.at[reach.Index, "rtp"] = strtop_withminslope
                            # this may still leave us below the
                            # bottom of layer
                        else:
                            # otherwise we can move reach so that it
                            # fits into layer 1
                            self.logger.debug(
                                "setting elevation to %s, above layer 1 bottom",
                                new_elev,
                            )
                            # set reach top so that it is above layer 1
                            # bottom with a buffer
                            # (allowing for bed thickness)
                            self.reaches.at[reach.Index, "rtp"] = new_elev
                    elif minslope_cond:
                        # (rtp < strtop_withminslope
                        # if ustrm
                        # else rtp > strtop_withminslope)
                        # below top but above/below minslope from
                        # upstream/downstream
                        self.logger.debug(
                            "reach %s, rtp is: /\\ above upstream", reach.Index
                        )
                        self.logger.debug("setting elevation to minslope from upstream")
                        # set to minimum slope from previous reach
                        self.reaches.at[reach.Index, "rtp"] = strtop_withminslope
                    # else it is above above bottom, below top and downstream
                # update upreach for next iteration
                prevreach_top = self.reaches.at[reach.Index, "rtp"]
                # check if new stream top is above layer 1 with a buffer
                # (allowing for bed thickness)
                reachbed_elev = prevreach_top - reach.rbth
                layerbots = _check_reach_v_laybot(
                    reach, layerbots, buffer, reachbed_elev
                )
                prevreach_mid = reach.mid_dist
                # upreach_slope=reach.slope
            self.reaches["bot"] = layerbots[0][reach_ij]
        if fix_dis:
            # fix dis for incised reaches
            for k in range(self.model.dis.nlay.data - 1):
                laythick = (
                    layerbots[k] - layerbots[k + 1]
                )  # first one is layer 1 bottom - layer 2 bottom
                self.logger.debug("checking layer %s thicknesses", k + 2)
                thincells = laythick < minthick
                self.logger.debug("%s cells less than %s", thincells.sum(), minthick)
                laythick[thincells] = minthick
                layerbots[k + 1] = layerbots[k] - laythick
            self.model.dis.botm.set_data(layerbots)


    def _fix_dis(self, buffer=0.5, move_top=True):
            botm = self.model.dis.botm.get_data()
            top = self.model.dis.top.get_data()
            rdf = self.reaches.copy()

            rdf['botm0'] = rdf[['i','j']].apply(lambda x: botm[0,x[0],x[1]], axis=1)
            rdf['maxbot'] = rdf[['rtp', 'rbth']].apply(lambda x: x[0] - x[1] - buffer, axis=1)
            rdf['bdz'] = rdf['botm0'] - rdf['maxbot']
            rdf['bdz'].clip(0, None, inplace=True)
            rdf['rno'] = rdf.index
            # need to get min bdz for each cell (unique 'ij')
            rdf['ij'] = rdf[['i','j']].apply(lambda x: (x[0],x[1]), axis=1)
            rdf = rdf.select_dtypes(include=[np.number, tuple]).groupby(['ij']).max()
            rdf.set_index('rno', inplace=True)
            # shift all layers in model column, TODO: enforce min layer thickness instead?
            for r in rdf.index:
                botm[:,rdf.loc[r, 'i'], rdf.loc[r, 'j']] = botm[:,rdf.loc[r, 'i'], rdf.loc[r, 'j']] - rdf.loc[r,'bdz']
                if move_top:
                    top[rdf.loc[r, 'i'], rdf.loc[r, 'j']] = top[rdf.loc[r, 'i'], rdf.loc[r, 'j']] - rdf.loc[r, 'bdz']
            # may do funny things to flopy external file reference?
            self.model.dis.botm.set_data(botm)
            self.model.dis.top.set_data(top)

    def _to_rno_elevs(self, minslope=0.0001, minincise=0.2,
                      minthick=0.5, buffer=0.5, fix_dis=True):
        """
        Attempt to set reach elevations, pandas approach:
            0. ensure minimum incision
            1. find minimum dz to next reach rtp-minslope*(delr+delc)/2
            2. adjust rtp of receiving reaches that are too high (while loop)
            3. optionally fix discretization in external method

        Parameters
        ----------
        minslope : float, default 1e-4
            The minimum allowable slope between adjacent reaches
            (to be considered flowing downstream).
        minincise : float, default 0.2
            The minimum allowable incision of a reach below the model top.
        minthick : float, default 0.5
            The minimum thickness of stream bed. Ensure that this
            is available below stream top before layer bottom or model will fail.
        buffer : float, default 0.5
            The minimum cell thickness between the bottom of stream bed
            (rtp-minthick) and the bottom of the cell below
        fix_dis : bool, default True
            Move layer elevations down where it is not possible to honor
            minimum slope without going below layer 1 bottom.

        Returns
        -------
        None

        """
        to_ridxname = f"to_{self.reach_index_name}"

        def get_to_rtp():
            intp = (rdf['to_rno'] != 0) & (rdf['to_rno'].isin(rdf.index))
            trno = rdf.loc[intp, 'to_rno']
            trno.drop_duplicates(inplace=True)
            trtp = rdf.loc[trno.index, 'rtp']
            rdf['rno'] = rdf.index
            ndf = rdf[['rno','to_rno']].merge(trtp,left_on='to_rno',right_on='rno',how='outer')
            ndf.dropna(subset=['rno','to_rno'], inplace=True)
            ndf[['rno','to_rno']] = ndf[['rno','to_rno']].astype(int)
            ndf.set_index('rno', inplace=True, drop=True)
            rdf['to_rtp'] = ndf['rtp']
            rdf.loc[~intp, 'to_rtp'] = -9999

        # copy some data
        top = self.model.dis.top.array.copy()
        delr = self.model.dis.delr.data.copy()
        delc = self.model.dis.delc.data.copy()

        if 'rtp' not in self.reaches.columns:
            self.set_reach_data_from_array('rtp', top)
            self.reaches['rtp'] = self.reaches['rtp'] - minincise
        rdf = self.reaches.copy()
        icols = rdf.columns.to_list()

        # add some necessary columns to rdf
        # attempt to sort of addresses local grid refinement?
        rdf.loc[:,"mindz"] = minslope * \
            (delr[rdf.loc[:, "j"]] + delc[rdf.loc[:, "i"]]) / 2.0
        if "rbth" not in rdf.columns:
            rdf["rbth"] = minthick
            icols.append("rbth")
        if "incise" not in rdf.columns:
            rdf["incise"] = minincise
            icols.append("incise")

        rdf['rbth'].where(rdf['rbth'] > minthick, minthick, inplace=True)
        # initial criteria
        get_to_rtp()
        #rdf['to_rtp'] = rdf['to_rno'].apply(lambda x: rdf.loc[x, 'rtp'] if x in rdf.index else -10)
        rdf['mx_to_rtp'] = rdf['rtp'] - rdf['mindz']
        sel = rdf['to_rtp'] > rdf['mx_to_rtp']
        loop = 0
        while sel.sum() > 0 and loop < 10000:
            # this gives geodataframe
            mx_rtp = rdf.loc[sel, ['to_rno','mx_to_rtp']].groupby('to_rno').min()
            # so make it a series, exclude mx_rtp==0
            mx_rtp = mx_rtp['mx_to_rtp']
            mx_rtp.drop([_ for _ in mx_rtp.index if _ not in rdf.index], inplace=True)
            # reset rtp values in rdf
            rdf.loc[mx_rtp.index, 'rtp'] = mx_rtp.values
            # report
            self.logger.debug("%s changed in loop %s", sel.sum(), loop)
            loop += 1
            # prep for next
            get_to_rtp()
            rdf['mx_to_rtp'] = rdf['rtp'] - rdf['mindz']
            sel = rdf['to_rtp'] > rdf['mx_to_rtp']
        if loop >= 10000:
            # maybe stronger warning, kill it?
            self.logger.debug("to_rno_elev did not find final solution after %s loops", loop)
        # copy here removes pd warning about setting value on copy
        self.reaches.loc[:,icols] = rdf.loc[:, icols]
        if fix_dis:
            self._fix_dis(buffer)
        self.add_model_topbot_to_reaches()

    def fix_reach_elevs(
        self,
        minslope=1e-4,
        minincise=0.2,
        minthick=0.5,
        buffer=0.1,
        fix_dis=True,
        direction="downstream",
        segbyseg=False,
        to_rno_elevs=False,
    ):
        """Fix reach elevations.

        Notes
        -----
        Need to ensure reach elevation is:
            0. below the top
            1. below the upstream reach
            2. above the minimum slope to the bottom reach elevation
            3. above the base of layer 1
        in modflow6 only reaches so maybe we should go just reach by reach
        (nwt method went seg-by-seg then reach-by-reach)

        Parameters
        ----------
        minslope : float, default 1e-4
            The minimum allowable slope between adjacent reaches
            (to be considered flowing downstream).
        minincise : float, default 0.2
            The minimum allowable incision of a reach below the model top.
        minthick : float, default 0.5
            The minimum thickness of stream bed. Will try to ensure that this
            is available below stream top before layer bottom.
        buffer : float, default 0.5
            The minimum cell thickness between the bottom of stream bed
            (rtp-minthick) and the bottom of the layer 1 cell
        fix_dis : bool, default True
            Move layer elevations down where it is not possible to honor
            minimum slope without going below layer 1 bottom.
        direction : {"upstream", "downstream", "both"}, default "downstream"
            Select whether elevations are set from ensuring a minimum slope
            in a upstream or downstream direction.
            If "upstream" will honor elevation at outlet reach
                (if in model layer) and work upstream ensuring minimum slope.
            If "downstream" will honor elevation at headwater reach
                (if in model layer) and work downstream ensuring minimum slope.
            If "both" will iterate "upstream" first and then "downstream",
                handy is no constraint on stream elevations.
        segbyseg : bool, default False
            NOT IMPLEMENTED
            Sets elevation of reaches segment by segment.
            If True, will attempt to honor the elevations specified at the
            upstream and downstream ends of each input line segment
            (if they are in the top model layer and appropriately downstream).
            If False will only attempt to honor elevations/incision at
            headwaters and outlets.
        to_rno_elevs : bool, default False
            attempt to quickly ensure rtp of the downstream reach
            is lower than rtp of the upstream reach

        Returns
        -------
        None

        """
        if segbyseg:
            raise NotImplementedError("option 'segbyseg=True' not finished")
            self._segbyseg_elevs(minslope, fix_dis, minthick)
        elif to_rno_elevs:
            self._to_rno_elevs(minslope, minincise, minthick, buffer, fix_dis)
        else:
            if direction == "both":
                direction = ["upstream", "downstream"]
            else:
                direction = [direction]
            for d in direction:
                self._reachbyreach_elevs(minslope, minincise, minthick, fix_dis, d)
        return

    def route_reaches(self, start, end, *, allow_indirect=False):
        """Return a list of reach numbers that connect a pair of reaches.

        Parameters
        ----------
        start, end : any
            Start and end reach numbers (rno or ifno).
        allow_indirect : bool, default False
            If True, allow the route to go downstream from start to a
            confluence, then route upstream to end. Defalut False allows
            only a direct route along a single direction up or down.

        Returns
        -------
        list

        Raises
        ------
        IndexError
            If start and/or end reach numbers are not valid.
        ConnecionError
            If start and end reach numbers do not connect.

        See Also
        --------
        gather_reaches : Query multiple reaches up and downstream.

        Examples
        --------
        >>> import flopy
        >>> import geopandas
        >>> import swn
        >>> lines = geopandas.GeoSeries.from_wkt([
        ...    "LINESTRING (60 100, 60  80)",
        ...    "LINESTRING (40 130, 60 100)",
        ...    "LINESTRING (70 130, 60 100)"])
        >>> lines.index += 100
        >>> n = swn.SurfaceWaterNetwork.from_lines(lines)
        >>> sim = flopy.mf6.MFSimulation()
        >>> _ = flopy.mf6.ModflowTdis(sim, nper=1, time_units="days")
        >>> gwf = flopy.mf6.ModflowGwf(sim)
        >>> _ = flopy.mf6.ModflowGwfdis(
        ...     gwf, nrow=3, ncol=2, delr=20.0, delc=20.0, idomain=1,
        ...     length_units="meters", xorigin=30.0, yorigin=70.0)
        >>> nm = swn.SwnMf6.from_swn_flopy(n, gwf)
        >>> nm.reaches[["iseg", "ireach", "to_ifno", "from_ifnos", "segnum"]]
             iseg  ireach  to_ifno from_ifnos  segnum
        ifno
        1       1       1        2         {}     101
        2       1       2        3        {1}     101
        3       1       3        6        {2}     101
        4       2       1        5         {}     102
        5       2       2        6        {4}     102
        6       3       1        7     {3, 5}     100
        7       3       2        0        {6}     100
        >>> nm.route_reaches(1, 7)
        [1, 2, 3, 6, 7]
        >>> nm.reaches.loc[nm.route_reaches(1, 7), ["i", "j", "rlen"]]
             i  j       rlen
        ifno
        1    0  0  18.027756
        2    0  1   6.009252
        3    1  1  12.018504
        6    1  1  10.000000
        7    2  1  10.000000
        """  # noqa
        if start not in self.reaches.index:
            raise IndexError(f"invalid start {self.reach_index_name} {start}")
        if end not in self.reaches.index:
            raise IndexError(f"invalid end {self.reach_index_name} {end}")
        if start == end:
            return [start]
        to_ridxname = f"to_{self.reach_index_name}"
        to_ridxs = dict(self.reaches.loc[self.reaches[to_ridxname] != 0, to_ridxname])

        def go_downstream(ridx):
            yield ridx
            if ridx in to_ridxs:
                yield from go_downstream(to_ridxs[ridx])

        con1 = list(go_downstream(start))
        try:
            # start is upstream, end is downstream
            return con1[: (con1.index(end) + 1)]
        except ValueError:
            pass
        con2 = list(go_downstream(end))
        set2 = set(con2)
        set1 = set(con1)
        if set1.issubset(set2):
            # start is downstream, end is upstream
            drop = set1.intersection(set2)
            drop.remove(start)
            while drop:
                drop.remove(con2.pop(-1))
            return list(reversed(con2))
        common = list(set1.intersection(set2))
        if not allow_indirect or not common:
            msg = f"{start} does not connect to {end}"
            if not common:
                msg += " -- reach networks are disjoint"
            raise ConnectionError(msg)
        # find the most upstream common ridx or "confluence"
        ridx = common.pop()
        idx1 = con1.index(ridx)
        idx2 = con2.index(ridx)
        while common:
            ridx = common.pop()
            tmp1 = con1.index(ridx)
            if tmp1 < idx1:
                idx1 = tmp1
                idx2 = con2.index(ridx)
        return con1[:idx1] + list(reversed(con2[:idx2]))

    def gather_reaches(
        self, *, upstream=[], downstream=[], barrier=[], gather_upstream=False
    ):
        """Return reaches upstream (inclusive) and downstream (exclusive).

        Parameters
        ----------
        upstream, downstream : int or list, default []
            Reach number(s) (rno or ifno) from reaches.index to search from.
        barriers : int or list, default []
            Reach number(s) that cannot be traversed past.
        gather_upstream : bool, default False
            Gather upstream from all other downstream reaches.

        Returns
        -------
        list

        See Also
        --------
        route_reaches :
            Return a list of reaches that connect a pair of reaches.

        Examples
        --------
        >>> import flopy
        >>> import geopandas
        >>> import swn
        >>> lines = geopandas.GeoSeries.from_wkt([
        ...    "LINESTRING (60 100, 60  80)",
        ...    "LINESTRING (40 130, 60 100)",
        ...    "LINESTRING (70 130, 60 100)"])
        >>> lines.index += 100
        >>> n = swn.SurfaceWaterNetwork.from_lines(lines)
        >>> sim = flopy.mf6.MFSimulation()
        >>> _ = flopy.mf6.ModflowTdis(sim, nper=1, time_units="days")
        >>> gwf = flopy.mf6.ModflowGwf(sim)
        >>> _ = flopy.mf6.ModflowGwfdis(
        ...     gwf, nrow=3, ncol=2, delr=20.0, delc=20.0, idomain=1,
        ...     length_units="meters", xorigin=30.0, yorigin=70.0)
        >>> nm = swn.SwnMf6.from_swn_flopy(n, gwf)
        >>> nm.reaches[["iseg", "ireach", "to_ifno", "from_ifnos", "segnum"]]
             iseg  ireach  to_ifno from_ifnos  segnum
        ifno
        1       1       1        2         {}     101
        2       1       2        3        {1}     101
        3       1       3        6        {2}     101
        4       2       1        5         {}     102
        5       2       2        6        {4}     102
        6       3       1        7     {3, 5}     100
        7       3       2        0        {6}     100
        >>> nm.gather_reaches(upstream=6)
        [6, 3, 2, 1, 5, 4]
        >>> nm.gather_reaches(downstream=4)
        [5, 6, 7]
        """
        reaches_set = set(self.reaches.index)

        def check_and_return_list(var, name):
            if isinstance(var, list):
                if not reaches_set.issuperset(var):
                    diff = list(sorted(set(var).difference(reaches_set)))
                    raise IndexError(
                        f"{len(diff)} {name} "
                        f"reach{'' if len(diff) == 1 else 'es'} "
                        f"not found in reaches.index: {abbr_str(diff)}"
                    )
                return var
            else:
                if var not in reaches_set:
                    raise IndexError(
                        f"{name} {self.reach_index_name} {var} not found in reaches.index"
                    )
                return [var]

        def go_upstream(ridx):
            yield ridx
            for from_ridx in from_ridxs.get(ridx, []):
                yield from go_upstream(from_ridx)

        def go_downstream(ridx):
            yield ridx
            if ridx in to_ridxs:
                yield from go_downstream(to_ridxs[ridx])

        to_ridx_name = f"to_{self.reach_index_name}"
        to_ridxs = dict(self.reaches.loc[self.reaches[to_ridx_name] != 0, to_ridx_name])
        from_ridxs = self.reaches[f"from_{self.reach_index_name}s"]
        # Note that `.copy(deep=True)` does not work; use deepcopy
        from_ridxs = from_ridxs[from_ridxs.apply(len) > 0].apply(deepcopy)
        for barrier in check_and_return_list(barrier, "barrier"):
            for ridx in from_ridxs.get(barrier, []):
                del to_ridxs[ridx]
            from_ridxs[to_ridxs[barrier]].remove(barrier)
            del to_ridxs[barrier]

        ridxs = []
        for ridx in check_and_return_list(upstream, "upstream"):
            upridxs = list(go_upstream(ridx))
            ridxs += upridxs  # ridx inclusive
        for ridx in check_and_return_list(downstream, "downstream"):
            downidxs = list(go_downstream(ridx))
            ridxs += downidxs[1:]  # ridx exclusive
            if gather_upstream:
                for ridx in downidxs[1:]:
                    for from_ridx in from_ridxs.get(ridx, []):
                        if from_ridx not in downidxs:
                            upridxs = list(go_upstream(from_ridx))
                            ridxs += upridxs
        return ridxs


def get_flopy_mf6_package(name: str):
    """Returns a flopy.mf6 package.

    Parameters
    ----------
    package : str
        MODFLOW 6 package name, such as "drn" for DRAIN, or "GWT/SRC" for
        Mass Source Loading.

    Returns
    -------
    flopy.mf6.mfpackage.MFPackage
        Subclass object.
    """
    import flopy.mf6

    if not isinstance(name, str):
        raise ValueError(f"packge must be a str type; found {type(name)!r}")
    try:
        return getattr(flopy.mf6, name)
    except AttributeError:
        pass
    # try (e.g.) "GWF-DRN" -> "Gwfdrn"
    name = "".join(x for x in list(name) if x.isalpha())
    try:
        return getattr(flopy.mf6, f"Modflow{name.title()}")
    except AttributeError:
        pass
    # try (e.g.) "DRN" -> "drn"
    return getattr(flopy.mf6, f"ModflowGwf{name.lower()}")
