# -*- coding: utf-8 -*-
"""Interface for flopy's implementation for MODFLOW 6."""

__all__ = ["SwnMf6"]

from itertools import zip_longest
from textwrap import dedent

import numpy as np
import pandas as pd

from swn.modflow._base import SwnModflowBase
from swn.util import abbr_str

try:
    import matplotlib
except ImportError:
    matplotlib = False


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
                "'model' must be a flopy.mf6.MFModel object; found " +
                str(type(model)))
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

    def packagedata_frame(
            self, style: str, auxiliary: list = [], boundname=None):
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

        See also
        --------
        SwnMf6.write_packagedata : Write native file.
        SwnMf6.flopy_packagedata : List of lists for flopy.

        Examples
        --------
        >>> import flopy
        >>> import swn
        >>> from swn.spatial import wkt_to_geoseries
        >>> lines = wkt_to_geoseries([
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
        rno                                   ...                                          
        1    1  1  1  18.027756  10.0  0.001  ...  0.024     1    1.0    0   2.1        101
        2    1  1  2   6.009252  10.0  0.001  ...  0.024     2    1.0    0   2.2        101
        3    1  2  2  12.018504  10.0  0.001  ...  0.024     2    1.0    0   2.3        101
        4    1  1  2  21.081851  10.0  0.001  ...  0.024     1    1.0    0   2.4        102
        5    1  2  2  10.540926  10.0  0.001  ...  0.024     2    1.0    0   2.5        102
        6    1  2  2  10.000000  10.0  0.001  ...  0.024     3    1.0    0   2.6        100
        7    1  3  2  10.000000  10.0  0.001  ...  0.024     1    1.0    0   2.7        100
        <BLANKLINE>
        [7 rows x 15 columns]
        >>> nm.packagedata_frame("flopy", boundname=False)
                cellid       rlen  rwid   rgrd  rtp  rbth  rhk    man  ncon  ustrf  ndv
        rno                                                                            
        0    (0, 0, 0)  18.027756  10.0  0.001  1.0   1.0  1.0  0.024     1    1.0    0
        1    (0, 0, 1)   6.009252  10.0  0.001  1.0   1.0  1.0  0.024     2    1.0    0
        2    (0, 1, 1)  12.018504  10.0  0.001  1.0   1.0  1.0  0.024     2    1.0    0
        3    (0, 0, 1)  21.081851  10.0  0.001  1.0   1.0  1.0  0.024     1    1.0    0
        4    (0, 1, 1)  10.540926  10.0  0.001  1.0   1.0  1.0  0.024     2    1.0    0
        5    (0, 1, 1)  10.000000  10.0  0.001  1.0   1.0  1.0  0.024     3    1.0    0
        6    (0, 2, 1)  10.000000  10.0  0.001  1.0   1.0  1.0  0.024     1    1.0    0
        """  # noqa
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
        pn = self.packagedata_frame(
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
        df = self.packagedata_frame(
            "flopy", auxiliary=auxiliary, boundname=boundname)
        return [list(x) for x in df.itertuples()]

    def connectiondata_series(self, style: str):
        """Return Series of CONNECTIONDATA for MODFLOW 6 SFR.

        Parameters
        ----------
        style : str
            If "native", all indicies (including kij) use one-based notation.
            If "flopy", all indices (including rno) use zero-based notation.
        """
        r = (self.reaches["from_rnos"].apply(sorted) +
             self.reaches["to_rno"].apply(lambda x: [-x] if x > 0 else []))
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
        cn = self.connectiondata_series("native")
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
        s = self.connectiondata_series("flopy")
        return (s.index.to_series().apply(lambda x: list([x])) + s).to_list()

    # TODO: add methods for diversions for flopy and writing file
    def write_diversions(self, fname: str):
        """Write DIVERSIONS file for MODFLOW 6 SFR."""
        # TODO
        raise NotImplementedError("method is not finished")

    def flopy_diversions(self):
        """Return list of lists for flopy.

        Returns
        -------
        list

        """
        # TODO
        raise NotImplementedError("method is not finished")

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
            self, hyd_cond1=1., hyd_cond_out=None,
            thickness1=1., thickness_out=None, width1=None, width_out=None,
            roughch=0.024):
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
            self.logger.info(
                "default_packagedata: 'rwd' " + action, *action_args)
        self.set_reach_data_from_segments(
            "rwid", width1, width_out, method="constant")

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
        self.set_reach_data_from_segments("rbth", thickness1, thickness_out)
        self.set_reach_data_from_segments("rhk", hyd_cond1, hyd_cond_out, True)
        self.set_reach_data_from_segments("man", roughch)

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

    def _segbyseg_elevs(self, minslope=0.0001, fix_dis=True, minthick=0.5,
                        min_incise=0.2, max_str_z=None):
        """
        Fix reach elevations but by using segment definition and setting sane
        segment elevations first.

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
        # fix segments first
        # 0. Segments are below the model top
        # 1. Segments flow downstream
        # 2. Downstream segments are below upstream segments
        _ = self.fix_segment_elevs(min_incise=min_incise,
                                   min_slope=minslope,
                                   max_str_z=max_str_z)
        _ = self.reconcile_reach_strtop()
        buffer = 1.0  # 1 m (buffer to leave at the base of layer 1 -
        # also helps with precision issues)
        # make sure elevations are up-to-date
        # recalculate REACH strtop elevations
        self.reconcile_reach_strtop()
        _ = self.add_model_topbot_to_reaches()
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
            for k in range(self.model.dis.nlay - 1):
                laythick = layerbots[k] - layerbots[
                    k + 1]  # first one is layer 1 bottom - layer 2 bottom
                print("checking layer {} thicknesses".format(k + 2))
                thincells = laythick < minthick
                print("{} cells less than {}"
                      .format(thincells.sum(), minthick))
                laythick[thincells] = minthick
                layerbots[k + 1] = layerbots[k] - laythick
            self.model.dis.botm = layerbots

    def _reachbyreach_elevs(self, minslope=0.0001, minincise=0.2, minthick=0.5,
                            fix_dis=True, direction='downstream'):
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
        minslope : float,
            The minimum allowable slope between adjacent reaches
            (to be considered flowing downstream).
            Default: 1e-4
        minincise : float,
            The minimum allowable incision of a reach below the model top.
            Default : 0.2
        minthick : float,
            The minimum thickness of stream bed. Will try to ensure that this is
            available below stream top before layer bottom.
            Default: 0.5
        fix_dis : bool,
            Move layer elevations down where it is not possible to honor
            minimum slope without going below layer 1 bottom.
            Default: True
        direction : `str`, 'upstream' or 'downstream',
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
                print("reach {} @ {} "
                      "is below layer 1 bottom @ {}"
                      .format(r.Index, rbed_elev, r.bot))
                print("    dropping layer 1 bottom to {} "
                      "to accommodate stream @ i = {}, j = {}"
                      .format(new_elev, r.i, r.j))
                botms[0, r.i, r.j] = new_elev
            return botms

        if direction == 'upstream':
            ustrm = True
        else:
            ustrm = False
        buffer = 1.0  # 1 m (buffer to leave at the base of layer 1 -
        # also helps with precision issues)
        # make sure elevations are up-to-date
        # recalculate REACH strtop elevations
        # self.reconcile_reach_strtop()
        _ = self.add_model_topbot_to_reaches()
        # top read from dis as float32 so comparison need to be with like
        reachsel = self.reaches["top"] <= self.reaches["rtp"]
        reach_ij = tuple(self.reaches[["i", "j"]].values.T)
        print("{} reaches above model top".format(reachsel.sum()))
        # copy of layer 1 bottom (for updating to fit in stream reaches)
        layerbots = self.model.dis.botm.array.copy()
        headreaches = self.reaches.loc[self.reaches.from_rnos == set()]
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
            dsegs = self._swn.query(downstream=hdrch.segnum)
            segs = [hdrch.segnum] + dsegs
            reaches = self.reaches.loc[self.reaches.segnum.isin(segs)].sort_index()
            # get outflow reach for this profile
            # maybe can't rely on it being the last one
            # the sort_index() should order (assuming rno increases downstream)
            # so last should be to_rno == 0
            assert reaches.iloc[-1].to_rno == 0, ("reach numbers possibly not "
                                                  "increasing downstream")
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
            reaches['mid_dist'] = reaches['rlen'].cumsum() - reaches['rlen'] / 2
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
            incgrad = ((inc_up - inc_dn) / totlen)
            # apparent optimised incision based
            # on the incision gradient for the segment
            reaches["strtop_incopt"] = (
                    (reaches.top - inc_up) +
                    ((reaches.mid_dist - prevreach_mid) * incgrad)
            )
            layerbots = _check_reach_v_laybot(reaches.iloc[0], layerbots, buffer)
            # loop over current profile from second to penultimate
            # (dont want to move endpoints)
            for reach in reaches.iloc[1:].itertuples():  # TODO maybe more flexibility on bottom reach incision
                # strtop that would result from minimum slope
                # from upstream reach
                rtp = reach.rtp
                strtop_withminslope = prevreach_top - (
                        (reach.mid_dist - prevreach_mid) * minslope)
                if rtp > reach.top - minincise:
                    # current reach rtp is above the model top
                    print(f"reach {reach.Index}, rtp is: /\\ above model top")
                    minslope_cond = (
                        reach.strtop_incopt < strtop_withminslope
                        if ustrm else
                        reach.strtop_incopt > strtop_withminslope
                    )
                    if minslope_cond:
                        # incision according to incision gradient would be
                        # above upstream/below downstream or give too shallow
                        # a slope from previous:
                        s = '\\/ below downstream' if ustrm else '/\\ above upstream'
                        print(f"--reach {reach.Index}, incopt is: {s}")
                        print("--setting elevation to minslope from "
                              "previous")
                        # set to minimum slope from previous reach
                        self.reaches.at[reach.Index, "rtp"] = strtop_withminslope
                    else:
                        # rtp might be ok to set to "optimum incision"
                        s = '-- above downstream' if ustrm else '-- below upstream'
                        print(f"--reach {reach.Index}, incopt is: {s}")
                        # CHECK FIRST:
                        # if optimium incision would place it
                        # below the bottom of layer 1
                        if (reach.strtop_incopt - reach.rbth) < reach.bot + buffer:
                            # opt - stream thickness lower than layer 1 bottom
                            # (with a buffer)
                            print(f"----reach {reach.Index}, incopt is: x\\/ "
                                  "below layer 1 bottom")
                            minslope_cond = (
                                reach.bot + reach.rbth + buffer < strtop_withminslope
                                if ustrm else
                                reach.bot + reach.rbth + buffer > strtop_withminslope
                            )
                            if minslope_cond:
                                # if layer bottom would put reach above
                                # upstream reach/below downstream reach
                                # we can only set to minimum slope from previous
                                print("------setting elevation to minslope "
                                      "from previous")
                                self.reaches.at[
                                    reach.Index, "rtp"] = strtop_withminslope
                                # this may still leave us below the
                                # bottom of layer
                            else:
                                # otherwise we can move reach so that it
                                # fits into layer 1
                                new_elev = reach.bot + reach.rbth + buffer
                                print(f"------setting elevation to {new_elev}, "
                                      f"above layer 1 bottom")
                                # set reach top so that it is above layer 1
                                # bottom with a buffer
                                # (allowing for bed thickness)
                                self.reaches.at[reach.Index, "rtp"] = new_elev
                        else:
                            # strtop ok to set to "optimum incision"
                            # set to "optimum incision"
                            print("----setting elevation to incopt")
                            self.reaches.at[
                                reach.Index, "rtp"] = reach.strtop_incopt
                else:
                    print(f"reach {reach.Index}, rtp is: -- below model top")
                    minslope_cond = (
                        rtp < strtop_withminslope
                        if ustrm else
                        rtp > strtop_withminslope
                    )
                    if (rtp - reach.rbth) < reach.bot + buffer:
                        # rtp is below the bottom of layer 1
                        print(f"--reach {reach.Index}, rtp is: x\\/ "
                              "below layer 1 bottom")
                        minslope_cond = (
                            reach.bot + reach.rbth + buffer < strtop_withminslope
                            if ustrm else
                            reach.bot + reach.rbth + buffer > strtop_withminslope
                        )
                        if minslope_cond:
                            # if layer bottom would put reach above
                            # upstream reach we can only set to
                            # minimum slope from upstream
                            print("----setting elevation to minslope "
                                  "from upstream")
                            self.reaches.at[
                                reach.Index, "rtp"] = strtop_withminslope
                            # this may still leave us below the
                            # bottom of layer
                        else:
                            # otherwise we can move reach so that it
                            # fits into layer 1
                            new_elev = reach.bot + reach.rbth + buffer
                            print(f"----setting elevation to {new_elev}, "
                                  f"above layer 1 bottom")
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
                        print(f"--reach {reach.Index}, rtp is: /\\ "
                              f"above upstream")
                        print("----setting elevation to minslope from upstream")
                        # set to minimum slope from previous reach
                        self.reaches.at[reach.Index, "rtp"] = strtop_withminslope
                    # else it is above above bottom, below top and downstream
                # update upreach for next iteration
                prevreach_top = self.reaches.at[reach.Index, "rtp"]
                # check if new stream top is above layer 1 with a buffer
                # (allowing for bed thickness)
                reachbed_elev = prevreach_top - reach.rbth
                layerbots = _check_reach_v_laybot(reach, layerbots, buffer,
                                                  reachbed_elev)
                prevreach_mid = reach.mid_dist
                # upreach_slope=reach.slope
            self.reaches["bot"] = layerbots[0][reach_ij]
        if fix_dis:
            # fix dis for incised reaches
            for k in range(self.model.dis.nlay.data - 1):
                laythick = layerbots[k] - layerbots[
                    k + 1]  # first one is layer 1 bottom - layer 2 bottom
                print("checking layer {} thicknesses".format(k + 2))
                thincells = laythick < minthick
                print("{} cells less than {}"
                      .format(thincells.sum(), minthick))
                laythick[thincells] = minthick
                layerbots[k + 1] = layerbots[k] - laythick
            self.model.dis.botm = layerbots

    def _auto_reach_elevs(self, minslope=0.0001, minincise=0.2, minthick=0.5,
                            buffer=0.5, fix_dis=True):
        """
        Wes's hacky attempt to set reach elevations. Doesn't really ensure anything,
        but the goal is:
        
            0. get a list of cells with to_rtp > rtp-minslope*(delr+delc)/2
            1. drop rtp of downstream reach (to_rno) when higher than rtp (of rno)
            2. grab all offensive reaches downstream of rno
            3. check and fix all layer bottoms to be above minthick+buffer
            4. adjust top, up only, to accomodate minincision or rtp above cell top

        Parameters
        ----------
        minslope : float,
            The minimum allowable slope between adjacent reaches
            (to be considered flowing downstream).
            Default: 1e-4
        minincise : float,
            The minimum allowable incision of a reach below the model top.
            Default : 0.2
        minthick : float,
            The minimum thickness of stream bed. Will try to ensure that this is
            available below stream top before layer bottom.
            Default: 0.5
        buffer : float,
            The minimum cell thickness between the bottom of stream bed (rtp-minthick)
            and the bottom of the layer 1 cell
            default: 0.5
        fix_dis : bool,
            Move layer elevations down where it is not possible to honor
            minimum slope without going below layer 1 bottom.
            Default: True

        Returns
        -------

        """

        # copy some data
        top=self.model.dis.top.array.copy()
        botm=self.model.dis.botm.array.copy()
        delr=self.model.dis.delr.data.copy()
        delc=self.model.dis.delc.data.copy()
        rdf=self.reaches.copy()
        icols=rdf.columns.to_list()
        

        # add some columns to rdf
        rdf['ij']=rdf.apply(lambda x: (int(x['i']),int(x['j'])),axis=1)
        rdf['mindz']=minslope*(delr[rdf.loc[:,'j']]+delc[rdf.loc[:,'i']])/2
        if 'rbth' not in rdf.columns:                
            rdf['rbth']=minthick
            icols.append('rbth')
        # potentially reach specific props        
        for idx,r in rdf.iterrows():
            if 'rtp' not in rdf.columns:
                rdf.loc[idx,'rtp']=top[r['ij']]
                icols.append('rtp')
            rdf.loc[idx,'rbth']=np.max([r['rbth'],minthick])
            trno=int(r['to_rno'])
            if trno!=0:
                rdf.loc[idx,'to_rtp']=rdf.loc[trno,'rtp']
        
        # start loop
        loop=0
        cont=True
        while cont:
            bad_reaches=[i for i in rdf.index if rdf.loc[i,'to_rtp'] > \
                        rdf.loc[i,'rtp']-rdf.loc[i,'mindz']]
            loop=loop+1
            chg=0
            for br in bad_reaches:
                rno=br
                trno=int(rdf.loc[br,'to_rno'])
                chglist=[]
                if trno!=0:
                    #count how many downstream reaches offend                    
                    # keep track of changes in elev
                    dzlist=[rdf.loc[rno,'mindz']]
                    while trno!=0 and rdf.loc[trno,'rtp']>rdf.loc[rno,'rtp']-np.sum(dzlist):
                        # keep list of dz in case another inflowing stream is even lower
                        chglist.append(trno)
                        nelev=rdf.loc[rno,'rtp']-np.sum(dzlist)
                        # set to_rtp and rtp
                        rdf.loc[rno,'to_rtp']=nelev                
                        rdf.loc[trno,'rtp']=nelev
                        # get new to_rno 
                        rno=trno
                        trno=rdf.loc[rno,'to_rno']
                        dzlist.append(rdf.loc[rno,'mindz'])
                        
                    # now adjust layering if necessary
                    if len(chglist)>0 and fix_dis:
                        for r in chglist:                            
                            # bump top elev up to rtp+minincise if need be
                            if top[rdf.loc[r,'ij']]<rdf.loc[r,'rtp']:
                                top[rdf.loc[r,'ij']]=rdf.loc[r,'rtp']+minincise
                            # bump bottoms down if needed
                            maxbot=rdf.loc[r,'rtp']--buffer
                            if botm[0][rdf.loc[r,'ij']]>=maxbot:
                                botdz=botm[0][rdf.loc[r,'ij']]-maxbot
                                for b in range(0,botm.shape[0]):
                                    botm[b][rdf.loc[r,'ij']]=botm[b][rdf.loc[r,'ij']]-botdz
                                
                chg=chg+len(chglist)
            if chg==0:
                cont=False
            else:
                print('{} changed in loop {}'.format(chg,loop))
        setattr(self,'reaches',rdf[icols])


    def fix_reach_elevs(self, minslope=0.0001, minincise=0.2, minthick=0.5, buffer=0.1,
                        fix_dis=True, direction='downstream', segbyseg=False,
                        autoreach=False):
        """
        Fix reach elevations.
        Need to ensure reach elevation is:
            0. below the top
            1. below the upstream reach
            2. above the minimum slope to the bottom reach elevation
            3. above the base of layer 1
        in modflow6 only reaches so maybe we should go just reach by reach
        (nwt method went seg-by-seg then reach-by-reach)

        Parameters
        ----------
        minslope : float,
            The minimum allowable slope between adjacent reaches
            (to be considered flowing downstream).
            Default: 1e-4
        minincise : float,
            The minimum allowable incision of a reach below the model top.
            Default : 0.2
        minthick : float,
            The minimum thickness of stream bed. Will try to ensure that this is
            available below stream top before layer bottom.
            Default: 0.5
        buffer : float,
            The minimum cell thickness between the bottom of stream bed (rtp-minthick)
            and the bottom of the layer 1 cell
            default: 0.5
        fix_dis : bool,
            Move layer elevations down where it is not possible to honor
            minimum slope without going below layer 1 bottom.
            Default: True
        direction : `str`, 'upstream', 'downstream' or 'both'
            Select whether elevations are set from ensuring a minimum slope
            in a upstream or downstream direction.
            If 'upstream' will honor elevation at outlet reach
                (if in model layer) and work upstream ensuring minimum slope.
            If 'downstream' will honor elevation at headwater reach
                (if in model layer) and work downstream ensuring minimum slope.
            If 'both' will iterate 'upstream' first and then 'downstrem',
                handy is no constraint on stream elevations.
            Default: 'downstream',
            set elevations from headwaters to outlets.
        segbyseg : bool,
            NOT IMPLEMENTED
            Sets elevation of reaches segment by segment.
            If True, will attempt to honor the elevations specified at the
            upstream and downstream ends of each input line segment
            (if they are in the top model layer and appropriately downstream).
            If False will only attempt to honor elevations/incision at
            headwaters and outlets. Default: False
            Default is
        autoreach : bool,
            likely a better name and many improvements to be made
            Wes's attempt to quickly ensure rtp of the downstream reach
            is lower than rtp of the upstream reach

        Returns
        -------

        """
        if segbyseg:
            raise NotImplementedError
            self._segbyseg_elevs(minslope, fix_dis, minthick)
        elif autoreach:
            self._auto_reach_elevs(minslope, minincise, minthick, buffer, fix_dis)
        else:
            if direction == 'both':
                direction = ['upstream', 'downstream']
            else:
                direction = [direction]
            for d in direction:
                self._reachbyreach_elevs(minslope, minincise, minthick, fix_dis,
                                         d)
        return