# -*- coding: utf-8 -*-
"""Interface for flopy's implementation for MODFLOW 6."""

__all__ = ["SwnMf6"]

import numpy as np
import pandas as pd
from itertools import zip_longest
from textwrap import dedent

try:
    import matplotlib
except ImportError:
    matplotlib = False

from swn.util import abbr_str
from swn.modflow._base import SwnModflowBase


class SwnMf6(SwnModflowBase):
    """Surface water network adaptor for MODFLOW 6.

    Attributes
    ----------
    model : flopy.mf6.ModflowGwf
        Instance of flopy.mf6.ModflowGwf
    segments : geopandas.GeoDataFrame
        Copied from swn.segments, but with additional columns added
    reaches : geopandas.GeoDataFrame
        Similar to structure in model.sfr.reaches with index "rno",
        ordered and starting from 1. Contains geometry and other columns
        not used by flopy.
    reaches_ts : dict
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
            cls, swn, model, idomain_action="freeze",
            reach_include_fraction=0.2):
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

        Returns
        -------
        obj : swn.SwnMf6 object
        """
        if idomain_action not in ("freeze", "modify"):
            raise ValueError("idomain_action must be one of freeze or modify")

        obj = super().from_swn_flopy(
            swn=swn, model=model, domain_action=idomain_action,
            reach_include_fraction=reach_include_fraction)

        # Add more information to reaches
        obj.reaches.index.name = "rno"
        obj.reaches["rlen"] = obj.reaches.geometry.length

        # Evaluate connections
        # Assume only converging network
        to_segnums_d = swn.to_segnums.to_dict()
        reaches_segnum_s = set(obj.reaches["segnum"])

        def find_next_rno(segnum):
            if segnum in to_segnums_d:
                to_segnum = to_segnums_d[segnum]
                if to_segnum in reaches_segnum_s:
                    sel = obj.reaches["segnum"] == to_segnum
                    return obj.reaches[sel].index[0]
                else:  # recurse downstream
                    return find_next_rno(to_segnum)
            else:
                return 0

        def get_to_rno():
            if segnum == next_segnum:
                return next_rno
            else:
                return find_next_rno(segnum)

        obj.reaches["to_rno"] = -1
        segnum_iter = obj.reaches["segnum"].iteritems()
        rno, segnum = next(segnum_iter)
        for next_rno, next_segnum in segnum_iter:
            obj.reaches.at[rno, "to_rno"] = get_to_rno()
            rno, segnum = next_rno, next_segnum
        next_segnum = swn.END_SEGNUM
        obj.reaches.at[rno, "to_rno"] = get_to_rno()
        assert obj.reaches.to_rno.min() >= 0

        # Populate from_rnos set
        obj.reaches["from_rnos"] = [set() for _ in range(len(obj.reaches))]
        to_rnos = obj.reaches.loc[obj.reaches["to_rno"] != 0, "to_rno"]
        for k, v in to_rnos.items():
            obj.reaches.at[v, "from_rnos"].add(k)

        # TODO: Diversions not handled (yet)
        obj.reaches["to_div"] = 0
        obj.reaches["ustrf"] = 1.

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
            model_info = "flopy {} {!r}".format(
                self.model.version, self.model.name)
            sp_info = "{} stress period{} with perlen: {} {}".format(
                nper, "" if nper == 1 else "s",
                abbr_str(list(tdis.perioddata.array["perlen"]), 4),
                tdis.time_units.data)
        reaches = self.reaches
        if reaches is None:
            reaches_info = "reaches not set"
        else:
            reaches_info = "{} in reaches ({}): {}".format(
                len(self.reaches), self.reaches.index.name,
                abbr_str(list(self.reaches.index), 4))
        return dedent("""\
            <{}: {}
              {}
              {} />""".format(
            self.__class__.__name__, model_info, reaches_info, sp_info))

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

    @SwnModflowBase.model.setter
    def model(self, model):
        """Set model property."""
        import flopy
        if not (isinstance(model, flopy.mf6.mfmodel.MFModel)):
            raise ValueError(
                "'model' must be a flopy.mf6.MFModel object; found "
                + str(type(model)))
        sim = model.simulation
        if "tdis" not in sim.package_key_dict.keys():
            raise ValueError("TDIS package required")
        if "dis" not in model.package_type_dict.keys():
            raise ValueError("DIS package required")
        _model = getattr(self, "_model", None)
        if _model is not None and _model is not model:
            self.logger.info("swapping 'model' object")
        self._model = model
        # Build stress period DataFrame from modflow model
        stress_df = pd.DataFrame(
            {"perlen": sim.tdis.perioddata.array.perlen})
        modeltime = self.model.modeltime
        stress_df["duration"] = pd.TimedeltaIndex(
            stress_df["perlen"].cumsum(), modeltime.time_units)
        stress_df["start"] = pd.to_datetime(modeltime.start_datetime)
        stress_df["end"] = stress_df["duration"] + stress_df.at[0, "start"]
        stress_df.loc[1:, "start"] = stress_df["end"].iloc[:-1].values
        self._stress_df = stress_df  # keep this for debugging
        self.time_index = pd.DatetimeIndex(stress_df["start"]).copy()
        self.time_index.name = None

    def packagedata_df(self, style: str, auxiliary: list = [], boundname=None):
        """Return DataFrame of PACKAGEDATA for MODFLOW 6 SFR.

        This DataFrame is derived from the reaches DataFrame.

        Parameters
        ----------
        style : str
            If "native", all indicies (including kij) use one-based notation.
            Also use k,i,j columns (as str) rather than cellid.
            If "flopy", all indices (including rno) use zero-based notation.
            Also use cellid as a tuple.
        auxiliary : str, list, optional
            String or list of auxiliary variable names. Default [].
        boundname : bool, optional
            Default None will determine if "boundname" column is added if
            available in reaches.columns.

        Returns
        -------
        DataFrame

        """
        from flopy.mf6 import ModflowGwfsfr as Mf6Sfr
        defcols_names = list(Mf6Sfr.packagedata.empty(self.model).dtype.names)
        defcols_names.remove("rno")  # this is the index
        if auxiliary is not None:
            if isinstance(auxiliary, str):
                auxiliary = [auxiliary]
            defcols_names += auxiliary
        dat = pd.DataFrame(self.reaches.copy())
        dat["idomain"] = \
            self.model.dis.idomain.array[dat["k"], dat["i"], dat["j"]]
        cellid_none = dat["idomain"] < 1
        kij_l = list("kij")
        if style == "native":
            # Convert from zero-based to one-based notation
            dat[kij_l] += 1
            # use kij unstead of cellid
            idx = defcols_names.index("cellid")
            defcols_names[idx:idx + 1] = kij_l
            # convert kij to str, and store NONE in k, if needed
            dat[kij_l] = dat[kij_l].astype(str)
            if cellid_none.any():
                dat.loc[cellid_none, "k"] = "NONE"
                dat.loc[cellid_none, ["i", "j"]] = ""
        elif style == "flopy":
            # Convert rno from one-based to zero-based notation
            dat.index -= 1
            # make cellid into tuple
            dat["cellid"] = dat[kij_l].to_records(index=False).tolist()
            if cellid_none.any():
                dat.loc[cellid_none, "cellid"] = None
        else:
            raise ValueError("'style' must be either 'native' or 'flopy'")
        if "rlen" not in dat.columns:
            dat.loc[:, "rlen"] = dat.geometry.length
        dat["ncon"] = (
            dat.from_rnos.apply(len) +
            (dat.to_rno > 0).astype(int)
        )
        dat["ndv"] = (dat.to_div > 0).astype(int)
        if boundname is None:
            boundname = "boundname" in dat.columns
        if boundname:
            defcols_names.append("boundname")
            dat["boundname"] = dat["boundname"].astype(str)
            # Check and enforce 40 character limit
            sel = dat.boundname.str.len() > 40
            if sel.any():
                self.logger.warning(
                    "clipping %d boundname entries to 40 character limit",
                    sel.sum())
                dat.loc[sel, "boundname"] = \
                    dat.loc[sel, "boundname"].str.slice(stop=40)
        # checking missing columns
        reach_columns = set(dat.columns)
        missing = set(defcols_names).difference(reach_columns)
        if missing:
            missing_l = []
            for name in defcols_names:
                if name not in reach_columns:
                    missing_l.append(name)
            raise KeyError(
                "missing {} reach dataset(s): {}"
                .format(len(missing_l), ", ".join(sorted(missing_l))))
        return dat.loc[:, defcols_names]

    def write_packagedata(
            self, fname: str, auxiliary: list = [], boundname=None):
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

        """
        pn = self.packagedata_df(
            "native", auxiliary=auxiliary, boundname=boundname)
        pn.index.name = "# rno"
        formatters = None
        if "boundname" in pn.columns:
            # Add single quotes for any items with spaces
            sel = pn.boundname.str.contains(" ")
            if sel.any():
                pn.loc[sel, "boundname"] = pn.loc[sel, "boundname"].map(
                    lambda x: "'{}'".format(x))
            # left-justify column
            formatters = {
                "boundname": "{{:<{}s}}".format(
                    pn["boundname"].str.len().max()).format}
        with open(fname, "w") as f:
            pn.reset_index().to_string(
                f, header=True, index=False, formatters=formatters)
            f.write("\n")

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
        df = self.packagedata_df(
            "flopy", auxiliary=auxiliary, boundname=boundname)
        return [list(x) for x in df.itertuples()]

    def _connectiondata_series(self, style: str):
        """Return Series of CONNECTIONDATA for MODFLOW 6 SFR.

        Parameters
        ----------
        style : str
            If "native", all indicies (including kij) use one-based notation.
            If "flopy", all indices (including rno) use zero-based notation.
        """
        r = (self.reaches["from_rnos"].apply(sorted)
             + self.reaches["to_rno"].apply(lambda x: [-x] if x > 0 else []))
        if self.reaches["to_div"].any():
            r += self.reaches["to_div"].apply(lambda x: [-x] if x > 0 else [])
        if style == "native":
            # keep one-based notation, but convert list to str
            return r
        elif style == "flopy":
            # Convert rno from one-based to zero-based notation
            r.index -= 1
            return r.apply(lambda x: [v - 1 if v > 0 else v + 1 for v in x])
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
        cn = self._connectiondata_series("native")
        icn = [f"ic{n+1}" for n in range(cn.apply(len).max())]
        rowfmt = "{:>" + str(len(str(cn.index.max()))) + "} {}\n"
        rnolen = 1 + len(str(len(self.reaches)))
        cn = cn.apply(lambda x: " ".join(str(v).rjust(rnolen) for v in x))
        with open(fname, "w") as f:
            f.write("# rno " + " ".join(icn) + "\n")
            for rno, ic in cn.iteritems():
                f.write(rowfmt.format(rno, ic))

    def flopy_connectiondata(self):
        """Return list of lists for flopy.

        Returns
        -------
        list

        """
        s = self._connectiondata_series("flopy")
        return (s.index.to_series().apply(lambda x: list([x])) + s).to_list()

    # TODO: add methods for diversions for flopy and writing file
    def write_diversions(self, fname: str):
        """Write DIVERSIONS file for MODFLOW 6 SFR."""
        raise NotImplementedError()

    def flopy_diversions(self):
        """Return list of lists for flopy.

        Returns
        -------
        list

        """
        raise NotImplementedError()

    _tsvar_meta = pd.DataFrame([
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
    ], columns=["name", "type"]).set_index("name")

    def _check_tsvar_name(self, name: str):
        """Helper method to check name used for set_tsvar_* methods."""
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
        self._check_tsvar_name(name)
        if not np.isscalar(data):
            raise ValueError("only non-scalar data can be set")
        if where != "start":
            raise NotImplementedError("only 'start' is currently implemented")

    def default_packagedata(
            self, hyd_cond1=1., hyd_cond_out=None,
            thickness1=1., thickness_out=None, width1=None, width_out=None,
            roughch=0.024):
        """High-level function to set MODFLOW 6 SFR PACKAGEDATA defaults.

        Parameters
        ----------
        hyd_cond1 : float or pandas.Series, optional
            Hydraulic conductivity of the streambed, as a global or per top of
            each segment. Used for reach rhk.
            Default 1.
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
            If None (default), the same width1 value for the top of the
            outlet segment is used for the bottom.
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
            self.logger.info(
                "default_packagedata: 'rwd' " + action, *action_args)
        self.set_reach_data_from_series("rwid", width1, width_out)

        if "rgrd" not in self.reaches.columns:
            self.logger.info(
                "default_packagedata: 'rgrd' not yet evaluated, setting with "
                "set_reach_slope()")
            self.set_reach_slope()

        # TODO: add a similar method for rtp? set_reaches_top()?

        # Get stream values from top of model
        self.set_reach_data_from_array("rtp", self.model.dis.top.array)
        if "zcoord_avg" in self.reaches.columns:
            # Created by set_reach_slope(); be aware of NaN
            nn = ~self.reaches.zcoord_avg.isnull()
            self.reaches.loc[nn, "rtp"] = self.reaches.loc[nn, "zcoord_avg"]

        # Assign remaining reach datasets
        self.set_reach_data_from_series("rbth", thickness1, thickness_out)
        self.set_reach_data_from_series("rhk", hyd_cond1, hyd_cond_out, True)
        self.set_reach_data_from_series("man", roughch)

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
        **kwargs : dict, optional
            Passed to flopy.mf6.ModflowGwfsfr.
        """
        import flopy

        if auxiliary is not None:
            kwds["auxiliary"] = auxiliary
        if boundnames is None:
            boundnames = "boundname" in self.reaches.columns
        if boundnames:
            kwds["boundnames"] = True
        if "packagedata" not in kwds:
            kwds["packagedata"] = self.flopy_packagedata(
                auxiliary=auxiliary, boundname=boundnames)
        if "connectiondata" not in kwds:
            kwds["connectiondata"] = self.flopy_connectiondata()

        flopy.mf6.ModflowGwfsfr(
            self.model,
            nreaches=len(self.reaches),
            **kwds)
