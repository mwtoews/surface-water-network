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
from swn.modflow._base import _SwnModflow


class SwnMf6(_SwnModflow):
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
        self.reaches_ts = {}
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

    # def __iter__(self):
    #    """Return object datasets with an iterator."""
    #    super().__iter__()

    # def __setstate__(self, state):
    #    """Set object attributes from pickle loads."""
    #    super().__setstate__(state)

    @_SwnModflow.model.setter
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

    def _packagedata_df(self, style: str):
        """Return DataFrame of PACKAGEDATA for MODFLOW 6 SFR.

        This DataFrame is derived from the reaches DataFrame.

        Parameters
        ----------
        style : str
            If "native", all indicies (including kij) use one-based notation.
            Also use k,i,j columns (as str) rather than cellid.
            If "flopy", all indices (including rno) use zero-based notation.
            Also use cellid as a tuple.

        Returns
        -------
        DataFrame

        """
        from flopy.mf6 import ModflowGwfsfr as Mf6Sfr
        defcols_names = list(Mf6Sfr.packagedata.empty(self.model).dtype.names)
        defcols_names.remove("rno")  # this is the index
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

    def write_packagedata(self, fname: str):
        """
        Write PACKAGEDATA file for MODFLOW 6 SFR.

        Parameters
        ----------
        fname : str
            Output file name.

        Returns
        -------
        None

        """
        pn = self._packagedata_df("native")
        pn.index.name = "# rno"
        with open(fname, "w") as f:
            pn.reset_index().to_string(f, header=True, index=False)

    @property
    def flopy_packagedata(self):
        """Return list of lists for flopy."""
        df = self._packagedata_df("flopy")
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

    @property
    def flopy_connectiondata(self):
        """Return list of lists for flopy."""
        s = self._connectiondata_series("flopy")
        return (s.index.to_series().apply(lambda x: list([x])) + s).to_list()

    # TODO: add methods for diversions for flopy and writing file
    def write_diversions(self, fname: str):
        """Write DIVERSIONS file for MODFLOW 6 SFR."""
        raise NotImplementedError()

    @property
    def flopy_diversions(self):
        """Return list of lists for flopy."""
        raise NotImplementedError()

    def set_period(name, data):
        """Set PERIOD data.
        """

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


    def set_sfr_obj(self, **kwds):
        """Set MODFLOW 6 SFR package data to flopy model.

        Parameters
        ----------
        **kwargs : dict, optional
            Passed to flopy.mf6.ModflowGwfsfr.
        """
        import flopy

        flopy.mf6.ModflowGwfsfr(
            self.model,
            nreaches=len(self.reaches),
            packagedata=self.flopy_packagedata,
            connectiondata=self.flopy_connectiondata,
            perioddata=self.spd,
            **kwds)

    def get_reach_data(self):
        """Return numpy.recarray for flopy's ModflowSfr2 reach_data.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.recarray

        """
        from flopy.modflow.mfsfr2 import ModflowSfr2
        # Build reach_data for Data Set 2
        reach_data_names = []
        for name in ModflowSfr2.get_default_reach_dtype().names:
            if name in self.reaches.columns:
                reach_data_names.append(name)
        reach_data = pd.DataFrame(self.reaches[reach_data_names])
        return reach_data.to_records(index=True)

    def get_seg_ijk(self):
        """Get the upstream and downstream segment k,i,j."""
        topidx = self.reaches["ireach"] == 1
        kij_df = self.reaches[topidx][["iseg", "k", "i", "j"]].sort_values(
            "iseg")
        idx_name = self.segment_data.index.name or "index"
        self.segment_data = self.segment_data.reset_index().merge(
            kij_df, left_on="nseg", right_on="iseg", how="left").drop(
            "iseg", axis=1).set_index(idx_name)
        self.segment_data.rename(
            columns={"k": "k_up", "i": "i_up", "j": "j_up"}, inplace=True)
        # seg bottoms
        btmidx = self.reaches.groupby("iseg")["ireach"].transform(max) == \
            self.reaches["ireach"]
        kij_df = self.reaches[btmidx][["iseg", "k", "i", "j"]].sort_values(
            "iseg")

        self.segment_data = self.segment_data.reset_index().merge(
            kij_df, left_on="nseg", right_on="iseg", how="left").drop(
            "iseg", axis=1).set_index(idx_name)
        self.segment_data.rename(
            columns={"k": "k_dn", "i": "i_dn", "j": "j_dn"}, inplace=True)
        return self.segment_data[[
            "k_up", "i_up", "j_up", "k_dn", "i_dn", "j_dn"]]

    def get_top_elevs_at_segs(self, m=None):
        """
        Get topsurface elevations associated with segment up and dn elevations.

        Adds elevation of model top at
        upstream and downstream ends of each segment
        :param m: modeflow model with active dis package
        :return: Adds "top_up" and "top_dn" columns to segment data dataframe
        """
        if m is None:
            m = self.model
        assert m.sfr is not None, "need sfr package"
        self.segment_data["top_up"] = m.dis.top.array[
            tuple(self.segment_data[["i_up", "j_up"]].values.T)]
        self.segment_data["top_dn"] = m.dis.top.array[
            tuple(self.segment_data[["i_dn", "j_dn"]].values.T)]
        return self.segment_data[["top_up", "top_dn"]]

    def get_segment_incision(self):
        """
        Calculate the upstream and downstream incision of the segment.

        :return:
        """
        self.segment_data["diff_up"] = (self.segment_data["top_up"] -
                                        self.segment_data["elevup"])
        self.segment_data["diff_dn"] = (self.segment_data["top_dn"] -
                                        self.segment_data["elevdn"])
        return self.segment_data[["diff_up", "diff_dn"]]

    def set_seg_minincise(self, minincise=0.2, max_str_z=None):
        """
        Set segment elevation to have the minumum incision from the top.

        :param minincise: Desired minimum incision
        :param max_str_z: Optional parameter to prevent streams at
        high elevations (forces incision to max_str_z)
        :return: incisions at the upstream and downstream end of each segment
        """
        sel = self.segment_data["diff_up"] < minincise
        self.segment_data.loc[sel, "elevup"] = (self.segment_data.loc[
                                                    sel, "top_up"] - minincise)
        sel = self.segment_data["diff_dn"] < minincise
        self.segment_data.loc[sel, "elevdn"] = (self.segment_data.loc[
                                                    sel, "top_dn"] - minincise)
        if max_str_z is not None:
            sel = self.segment_data["elevup"] > max_str_z
            self.segment_data.loc[sel, "elevup"] = max_str_z
            sel = self.segment_data["elevdn"] > max_str_z
            self.segment_data.loc[sel, "elevdn"] = max_str_z
        # recalculate incisions
        updown_incision = self.get_segment_incision()
        return updown_incision

    def get_segment_length(self):
        """
        Get segment length from accumulated reach lengths.

        :return:
        """
        # extract segment length for calculating minimun drop later
        reaches = self.reaches[["geometry", "iseg", "rchlen"]].copy()
        seglen = reaches.groupby("iseg")["rchlen"].sum()
        self.segment_data.loc[seglen.index, "seglen"] = seglen
        return seglen

    def get_outseg_elev(self):
        """Get the max elevup from all downstream segments for each segment."""
        self.segment_data["outseg_elevup"] = self.segment_data.outseg.apply(
            lambda x: self.segment_data.loc[
                self.segment_data.index == x].elevup).max(axis=1)
        return self.segment_data["outseg_elevup"]

    def set_outseg_elev_for_seg(self, seg):
        """Set outseg elevation for segment.

        Gets all the defined outseg_elevup associated with a specific segment
        (multiple upstream segements route to one segment)
        Returns a df with all the calculated outseg elevups for each segment.
        .min(axis=1) is a good way to collapse to a series
        :param seg: Pandas Series containing one row of seg_data dataframe
        :return: Returns a df of the outseg_elev up values
        where current segment is listed as an outseg
        """
        # downstreambuffer = 0.001 # 1mm
        # find where seg is listed as outseg
        outsegsel = self.segment_data["outseg"] == seg.name
        # set outseg elevup
        outseg_elevup = self.segment_data.loc[outsegsel, "outseg_elevup"]
        return outseg_elevup

    def minslope_seg(self, seg, *args):
        """
        Force segment to have minumim slope (check for backward flowing segs).

        Moves downstream end down (vertically, more incision)
        to acheive minimum slope.
        :param seg: Pandas Series containing one row of seg_data dataframe
        :param args: desired minumum slope
        :return: Pandas Series with new downstream elevation and
        associated outseg_elevup
        """
        # segdata_df = args[0]
        minslope = args[0]
        downstreambuffer = 0.001  # 1mm
        up = seg.elevup
        dn = np.nan
        outseg_up = np.nan
        # prefer slope derived from surface
        surfslope = (seg.top_up-seg.top_dn)/(10.*seg.seglen)
        prefslope = np.max([surfslope, minslope])
        if seg.outseg > 0.0:
            # select outflow segment for current seg and pull out elevup
            outsegsel = self.segment_data.index == seg.outseg
            outseg_elevup = self.segment_data.loc[outsegsel, "elevup"]
            down = outseg_elevup.values[0]
            if down >= up - (seg.seglen * prefslope):
                # downstream elevation too high
                dn = up - (seg.seglen * prefslope)  # set to minslope
                outseg_up = up - (seg.seglen * prefslope) - downstreambuffer
                print("Segment {}, outseg = {}, old outseg_elevup = {}, "
                      "new outseg_elevup = {}"
                      .format(seg.name, seg.outseg,
                              seg.outseg_elevup, outseg_up))
            else:
                dn = down
                outseg_up = down - downstreambuffer
        else:
            # must be an outflow segment
            down = seg.elevdn
            if down > up - (seg.seglen * prefslope):
                dn = up - (seg.seglen * prefslope)
                print("Outflow Segment {}, outseg = {}, old elevdn = {}, "
                      "new elevdn = {}"
                      .format(seg.name, seg.outseg, seg.elevdn, dn))
            else:
                dn = down
        # this returns a DF once the apply is done!
        return pd.Series({"nseg": seg.name, "elevdn": dn,
                          "outseg_elevup": outseg_up})

    def set_forward_segs(self, min_slope=1.e-4):
        """Set minimum slope in forwards direction.

        Ensure slope of all segment is at least min_slope
        in the downstream direction.
        Moves down the network correcting downstream elevations if necessary
        :param min_slope: Desired minimum slope
        :return: and updated segment data df
        """
        # upper most segments (not referenced as outsegs)
        # segdata_df = self.segment_data.sort_index(axis=1)
        segsel = ~self.segment_data.index.isin(self.segment_data["outseg"])
        while segsel.sum() > 0:
            print("Checking elevdn and outseg_elevup for {} segments"
                  .format(segsel.sum()))
            # get elevdn and outseg_elevups with a minimum slope constraint
            # index should align with self.segment_data index
            # not applying directly allows us to filter out nans
            tmp = self.segment_data.assign(_="").loc[segsel].apply(
                self.minslope_seg, args=[min_slope], axis=1)
            ednsel = tmp[tmp["elevdn"].notna()].index
            oeupsel = tmp[tmp["outseg_elevup"].notna()].index
            # set elevdn and outseg_elevup
            self.segment_data.loc[ednsel, "elevdn"] = tmp.loc[ednsel, "elevdn"]
            self.segment_data.loc[oeupsel, "outseg_elevup"] = \
                tmp.loc[oeupsel, "outseg_elevup"]
            # get `elevups` for outflow segs from `outseg_elevups`
            # taking `min` ensures outseg elevup is below all inflow elevdns
            tmp2 = self.segment_data.apply(
                self.set_outseg_elev_for_seg, axis=1).min(axis=1)
            tmp2 = pd.DataFrame(tmp2, columns=["elevup"])
            # update `elevups`
            eupsel = tmp2[tmp2.loc[:, "elevup"].notna()].index
            self.segment_data.loc[eupsel, "elevup"] = \
                tmp2.loc[eupsel, "elevup"]
            # get list of next outsegs
            segsel = self.segment_data.index.isin(
                self.segment_data.loc[segsel, "outseg"])
        return self.segment_data

    def fix_segment_elevs(self, min_incise=0.2, min_slope=1.e-4,
                          max_str_z=None):
        """
        Provide wrapper function for calculating SFR segment elevations.

        Calls series of functions to process and move sfr segment elevations,
        to try to ensure:
            0. Segments are below the model top
            1. Segments flow downstream
            2. Downstream segments are below upstream segments
        :param min_slope: desired minimum slope for segment
        :param min_incise: desired minimum incision (in model units)
        :return: segment data dataframe
        """
        kijcols = {"k_up", "i_up", "j_up", "k_dn", "i_dn", "j_dn"}
        dif = kijcols - set(self.segment_data.columns)
        if len(dif) > 1:
            # some missing
            # drop others
            others = kijcols - dif
            self.segment_data.drop(others, axis=0, inplace=True)
            # get model locations for segments ends
            _ = self.get_seg_ijk()
        # get model cell elevations at seg ends
        _ = self.get_top_elevs_at_segs()
        # get current segment incision at seg ends
        _ = self.get_segment_incision()
        # move segments end elevation down to achieve minimum incision
        _ = self.set_seg_minincise(minincise=min_incise, max_str_z=max_str_z)
        # get the elevations of downstream segments
        _ = self.get_outseg_elev()
        # get segment length from reach lengths
        _ = self.get_segment_length()
        # ensure downstream ends are below upstream ends
        # and reconcile upstream elevation of downstream segments
        self.set_forward_segs(min_slope=min_slope)
        # reassess segment incision after processing.
        self.get_segment_incision()
        return self.segment_data

    def reconcile_reach_strtop(self):
        """
        Recalculate reach strtop elevations after moving segment elevations.

        :return: None
        """
        def reach_elevs(seg):
            """Return reach properties.

            Calculate reach elevation from segment slope and
            reach length along segment.
            :param seg: one row of reach data dataframe grouped by segment
            :return: reaches by segment with strtop adjusted
            """
            segsel = self.segment_data.index == seg.name

            seg_elevup = self.segment_data.loc[segsel, "elevup"].values[0]
            seg_slope = self.segment_data.loc[segsel, "Zslope"].values[0]

            # interpolate reach lengths to cell centres
            cmids = seg.seglen.shift().fillna(0.0) + seg.rchlen.multiply(0.5)
            cmids.iat[0] = 0.0
            cmids.iat[-1] = seg.seglen.iloc[-1]
            seg["cmids"] = cmids  # cummod+(seg.rchlen *0.5)
            # calculate reach strtops
            seg["strtop"] = seg["cmids"].multiply(seg_slope) + seg_elevup
            # seg["slope"]= #!!!!! use m.sfr.get_slopes() method
            return seg
        self.segment_data["Zslope"] = \
            ((self.segment_data["elevdn"] - self.segment_data["elevup"]) /
             self.segment_data["seglen"])
        segs = self.reaches.groupby("iseg")
        self.reaches["seglen"] = segs.rchlen.cumsum()
        self.reaches = segs.apply(reach_elevs)
        return self.reaches

    def set_topbot_elevs_at_reaches(self, m=None):
        """
        Get top and bottom elevation of the cell containing a reach.

        :param m: Modflow model
        :return: dataframe with reach cell top and bottom elevations
        """
        if m is None:
            m = self.model
        self.reaches["top"] = m.dis.top.array[
            tuple(self.reaches[["i", "j"]].values.T)]
        self.reaches["bot"] = m.dis.botm[0].array[
            tuple(self.reaches[["i", "j"]].values.T)]
        return self.reaches[["top", "bot"]]

    def fix_reach_elevs(self, minslope=0.0001, fix_dis=True, minthick=0.5):
        """Fix reach elevations.

        Need to ensure reach elevation is:
            0. below the top
            1. below the upstream reach
            2. above the minimum slope to the bottom reach elevation
            3. above the base of layer 1
        segment by segment, reach by reach! Fun!

        :return:
        """
        def _check_reach_v_laybot(r, botms, buffer=1.0, rbed_elev=None):
            if rbed_elev is None:
                rbed_elev = r.strtop - r.strthick
            if (rbed_elev - buffer) < r.bot:
                # if new strtop is below layer one
                # drop bottom of layer one to accomodate stream
                # (top, bed thickness and buffer)
                new_elev = rbed_elev - buffer
                print("seg {} reach {} @ {} "
                      "is below layer 1 bottom @ {}"
                      .format(seg, r.ireach, rbed_elev,
                              r.bot))
                print("    dropping layer 1 bottom to {} "
                      "to accommodate stream @ i = {}, j = {}"
                      .format(new_elev, r.i, r.j))
                botms[0, r.i, r.j] = new_elev
            return botms

        buffer = 1.0  # 1 m (buffer to leave at the base of layer 1 -
        # also helps with precision issues)
        # make sure elevations are up-to-date
        # recalculate REACH strtop elevations
        self.reconcile_reach_strtop()
        _ = self.set_topbot_elevs_at_reaches()
        # top read from dis as float32 so comparison need to be with like
        reachsel = self.reaches["top"] <= self.reaches["strtop"]
        reach_ij = tuple(self.reaches[["i", "j"]].values.T)
        print("{} segments with reaches above model top".format(
            self.reaches[reachsel]["iseg"].unique().shape[0]))
        # get segments with reaches above the top surface
        segsabove = self.reaches[reachsel].groupby(
            "iseg").size().sort_values(ascending=False)
        # get incision gradient from segment elevups and elevdns
        # ("diff_up" and "diff_dn" are the incisions of the top and
        # bottom reaches from the segment data)
        self.segment_data["incgrad"] = \
            ((self.segment_data["diff_up"] - self.segment_data["diff_dn"]) /
             self.segment_data["seglen"])
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
                self.reaches.loc[rsel, "strtop_incopt"] = \
                    self.reaches.loc[rsel, "top"].subtract(
                        self.segment_data.loc[segsel, "diff_up"].values[0]) + \
                    (self.reaches.loc[rsel, "cmids"].subtract(
                        self.reaches.loc[rsel, "cmids"].values[0]) *
                     self.segment_data.loc[segsel, "incgrad"].values[0])
                # falls apart when the top elevation is not monotonically
                # decreasing down the segment (/always!)

                # bottom reach elevation:
                botreach_strtop = self.reaches[rsel]["strtop"].values[-1]
                # total segment length
                seglen = self.reaches[rsel]["seglen"].values[-1]
                botreach_slope = minslope  # minimum slope of segment
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
                            (reach.cmids - upreach_cmid) * minslope)
                    # strtop that would result from minimum slope
                    # from bottom reach
                    strtop_min2bot = botreach_strtop + (
                            (seglen - reach.cmids) * minslope)
                    # check "optimum incision" is below upstream elevation
                    # and above the minimum slope to the bottom reach
                    if reach.strtop_incopt < strtop_min2bot:
                        # strtop would give too shallow a slope to
                        # the bottom reach (not moving bottom reach)
                        print("seg {} reach {}, incopt is \\/ below minimum "
                              "slope from bottom reach elevation"
                              .format(seg, reach.ireach))
                        print("    setting elevation to minslope from bottom")
                        # set to minimum slope from outreach
                        self.reaches.at[
                            reach.Index, "strtop"] = strtop_min2bot
                        # update upreach for next iteration
                        upreach_strtop = strtop_min2bot
                    elif reach.strtop_incopt > strtop_withminslope:
                        # strtop would be above upstream or give
                        # too shallow a slope from upstream
                        print("seg {} reach {}, incopt /\\ above upstream"
                              .format(seg, reach.ireach))
                        print("    setting elevation to minslope from "
                              "upstream")
                        # set to minimum slope from upstream reach
                        self.reaches.at[
                            reach.Index, "strtop"] = strtop_withminslope
                        # update upreach for next iteration
                        upreach_strtop = strtop_withminslope
                    else:
                        # strtop might be ok to set to "optimum incision"
                        print("seg {} reach {}, incopt is -- below upstream "
                              "reach and above the bottom reach"
                              .format(seg, reach.ireach))
                        # CHECK FIRST:
                        # if optimium incision would place it
                        # below the bottom of layer 1
                        if reach.strtop_incopt - reach.strthick < \
                                reach.bot + buffer:
                            # opt - stream thickness lower than layer 1 bottom
                            # (with a buffer)
                            print("seg {} reach {}, incopt - bot is x\\/ "
                                  "below layer 1 bottom"
                                  .format(seg, reach.ireach))
                            if reach.bot + reach.strthick + buffer > \
                                    strtop_withminslope:
                                # if layer bottom would put reach above
                                # upstream reach we can only set to
                                # minimum slope from upstream
                                print("    setting elevation to minslope "
                                      "from upstream")
                                self.reaches.at[reach.Index, "strtop"] = \
                                    strtop_withminslope
                                upreach_strtop = strtop_withminslope
                            else:
                                # otherwise we can move reach so that it
                                # fits into layer 1
                                new_elev = reach.bot + reach.strthick + buffer
                                print("    setting elevation to {}, above "
                                      "layer 1 bottom".format(new_elev))
                                # set reach top so that it is above layer 1
                                # bottom with a buffer
                                # (allowing for bed thickness)
                                self.reaches.at[reach.Index, "strtop"] = \
                                    reach.bot + buffer + reach.strthick
                                upreach_strtop = new_elev
                        else:
                            # strtop ok to set to "optimum incision"
                            # set to "optimum incision"
                            print("    setting elevation to incopt")
                            self.reaches.at[
                                reach.Index, "strtop"] = reach.strtop_incopt
                            upreach_strtop = reach.strtop_incopt
                    # check if new stream top is above layer 1 with a buffer
                    # (allowing for bed thickness)
                    reachbed_elev = upreach_strtop - reach.strthick
                    layerbots = _check_reach_v_laybot(reach, layerbots, buffer,
                                                      reachbed_elev)
                    upreach_cmid = reach.cmids
                    # upreach_slope=reach.slope
            else:
                # For segments that do not have reaches above top
                # check if reaches are below layer 1
                print("seg {} is always downstream and below the top"
                      .format(seg))
                for reach in self.reaches[rsel].itertuples():
                    reachbed_elev = reach.strtop - reach.strthick
                    layerbots = _check_reach_v_laybot(reach, layerbots, buffer,
                                                      reachbed_elev)
            # OH CRAP need to update dis bottoms in reach df!
            # self.reaches["top"] = layerbots[
            #     tuple(self.reaches[["i", "j"]].values.T)]
            self.reaches["bot"] = layerbots[0][reach_ij]
        if fix_dis:
            # fix dis for incised reaches
            for lay in range(self.model.dis.nlay - 1):
                laythick = layerbots[lay] - layerbots[
                    lay + 1]  # first one is layer 1 bottom - layer 2 bottom
                print("checking layer {} thicknesses".format(lay + 2))
                thincells = laythick < minthick
                print("{} cells less than {}"
                      .format(thincells.sum(), minthick))
                laythick[thincells] = minthick
                layerbots[lay + 1] = layerbots[lay] - laythick
            self.model.dis.botm = layerbots

    def sfr_plot(self, model, sfrar, dem, points=None, points2=None,
                 label=None):
        """Plot sfr."""
        from swn.modflow._modelplot import ModelPlot
        p = ModelPlot(model)
        p._add_plotlayer(dem, label="Elevation (m)")
        p._add_sfr(sfrar, cat_cmap=False, cbar=True,
                   label=label)
        return p

    def plot_reaches_above(self, model, seg, dem=None,
                           plot_bottom=False, points2=None):
        """Plot sfr reaches above."""
        # ensure reach elevations are up-to-date
        _ = self.set_topbot_elevs_at_reaches()
        dis = model.dis
        sfr = model.sfr
        if dem is None:
            dem = np.ma.array(
                dis.top.array, mask=model.dis.idomain.array[0] == 0)
        sfrar = np.ma.zeros(dis.top.array.shape, "f")
        sfrar.mask = np.ones(sfrar.shape)
        lay1reaches = self.reaches.loc[
            self.reaches.k.apply(lambda x: x == 1)]
        points = None
        if lay1reaches.shape[0] > 0:
            points = lay1reaches[["i", "j"]]
        # segsel=reachdata["iseg"].isin(segsabove.index)
        if seg == "all":
            segsel = np.ones((self.reaches.shape[0]), dtype=bool)
        else:
            segsel = self.reaches["iseg"] == seg
        sfrar[tuple((self.reaches[segsel][["i", "j"]]
                     .values.T).tolist())] = \
            (self.reaches[segsel]["top"] -
             self.reaches[segsel]["strtop"]).tolist()
        # .mask = np.ones(sfrar.shape)
        vtop = self.sfr_plot(model, sfrar, dem, points=points, points2=points2,
                             label="str below top (m)")
        if seg != "all":
            sfr.plot_path(seg)
        if plot_bottom:
            dembot = np.ma.array(dis.botm.array[0],
                                 mask=model.dis.idomain.array[0] == 0)
            sfrarbot = np.ma.zeros(dis.botm.array[0].shape, "f")
            sfrarbot.mask = np.ones(sfrarbot.shape)
            sfrarbot[tuple((self.reaches[segsel][["i", "j"]]
                            .values.T).tolist())] = \
                (self.reaches[segsel]["strtop"] -
                 self.reaches[segsel]["bot"]).tolist()
            # .mask = np.ones(sfrar.shape)
            vbot = self.sfr_plot(model, sfrarbot, dembot, points=points,
                                 points2=points2, label="str above bottom (m)")
        else:
            vbot = None
        return vtop, vbot
