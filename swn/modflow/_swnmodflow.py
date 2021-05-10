# -*- coding: utf-8 -*-
"""Interface for flopy's implementation for MODFLOW."""

__all__ = ["SwnModflow"]

import numpy as np
import pandas as pd
from itertools import zip_longest
from shapely.geometry import LineString

try:
    import matplotlib
except ImportError:
    matplotlib = False

from swn.modflow._base import _SwnModflow
from swn.util import abbr_str


class SwnModflow(_SwnModflow):
    """Surface water network adaptor for MODFLOW.

    Attributes
    ----------
    model : flopy.modflow.Modflow
        Instance of flopy.modflow.Modflow
    segments : geopandas.GeoDataFrame
        Copied from swn.segments, but with additional columns added
    segment_data : pandas.DataFrame
        Simialr to structure in model.sfr.segment_data, but for one stress
        period. Transient data (where applicable) will show summary statistics.
        The index is 'nseg', ordered and starting from 1. An additional column
        'segnum' is used to identify segments, and if defined,
        abstraction/diversion identifiers, where iupseg != 0.
    reaches : geopandas.GeoDataFrame
        Similar to structure in model.sfr.reach_data with index 'reachID',
        ordered and starting from 1. Contains geometry and other columns
        not used by flopy. Use flopy_reach_data for use with flopy.
    diversions :  geopandas.GeoDataFrame, pd.DataFrame or None
        Copied from swn.diversions, if set/defined.
    logger : logging.Logger
        Logger to show messages.

    """

    def __init__(self, logger=None):
        """Initialise SwnModflow.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger to show messages.
        """
        super().__init__(logger)
        # set empty properties for now
        self.reaches = None
        self.segment_data = None
        self.diversions = None
        # all other properties added afterwards

    @classmethod
    def from_swn_flopy(
            cls, swn, model, ibound_action="freeze",
            reach_include_fraction=0.2):
        """Create a MODFLOW SFR structure from a surface water network.

        Parameters
        ----------
        swn : swn.SurfaceWaterNetwork
            Instance of a SurfaceWaterNetwork.
        model : flopy.modflow.Modflow
            Instance of a flopy MODFLOW model with DIS and BAS6 packages.
        ibound_action : str, optional
            Action to handle IBOUND:
                - ``freeze`` : Freeze IBOUND, but clip streams to fit bounds.
                - ``modify`` : Modify IBOUND to fit streams, where possible.
        reach_include_fraction : float or pandas.Series, optional
            Fraction of cell size used as a threshold distance to determine if
            reaches outside the active grid should be included to a cell.
            Based on the furthest distance of the line and cell geometries.
            Default 0.2 (e.g. for a 100 m grid cell, this is 20 m).

        Returns
        -------
        obj : swn.SwnModflow object
        """
        if ibound_action not in ("freeze", "modify"):
            raise ValueError("ibound_action must be one of freeze or modify")

        obj = super().from_swn_flopy(
            swn=swn, model=model, domain_action=ibound_action,
            reach_include_fraction=reach_include_fraction)

        # Add more information to reaches
        obj.reaches.index.name = "reachID"
        obj.reaches["rchlen"] = obj.reaches.geometry.length

        return obj

    def set_sfr_data(
            self, hyd_cond1=1., hyd_cond_out=None,
            thickness1=1., thickness_out=None, width1=10., width_out=None,
            roughch=0.024, abstraction={}, inflow={}, flow={}, runoff={},
            etsw={}, pptsw={}):
        """Prepare data required for SFR package, and set model.sfr propery.

        Also populate reach_data and segment_data objects.

        Parameters
        ----------
        hyd_cond1 : float or pandas.Series, optional
            Hydraulic conductivity of the streambed, as a global or per top of
            each segment. Used for either STRHC1 or HCOND1/HCOND2 outputs.
            Default 1.
        hyd_cond_out : None, float or pandas.Series, optional
            Similar to thickness1, but for the hydraulic conductivity of each
            segment outlet. If None (default), the same hyd_cond1 value for the
            top of the outlet segment is used for the bottom.
        thickness1 : float or pandas.Series, optional
            Thickness of the streambed, as a global or per top of each segment.
            Used for either STRTHICK or THICKM1/THICKM2 outputs. Default 1.
        thickness_out : None, float or pandas.Series, optional
            Similar to thickness1, but for the bottom of each segment outlet.
            If None (default), the same thickness1 value for the top of the
            outlet segment is used for the bottom.
        width1 : float or pandas.Series, optional
            Channel width, as a global or per top of each segment. Used for
            WIDTH1/WIDTH2 outputs. Default 10.
        width_out : None, float or pandas.Series, optional
            Similar to width1, but for the bottom of each segment outlet.
            If None (default), the same width1 value for the top of the
            outlet segment is used for the bottom.
        roughch : float or pandas.Series, optional
            Manning's roughness coefficient for the channel. If float, then
            this is a global value, otherwise it is per-segment with a Series.
            Default 0.024.
        abstraction : dict or pandas.DataFrame, optional
            See generate_segment_data for details.
            Default is {} (no abstraction from diversions).
        inflow : dict or pandas.DataFrame, optional
            See generate_segment_data for details.
            Default is {} (no outside inflow added to flow term).
        flow : dict or pandas.DataFrame, optional
            See generate_segment_data. Default is {} (zero).
        runoff : dict or pandas.DataFrame, optional
            See generate_segment_data. Default is {} (zero).
        etsw : dict or pandas.DataFrame, optional
            See generate_segment_data. Default is {} (zero).
        pptsw : dict or pandas.DataFrame, optional
            See generate_segment_data. Default is {} (zero).

        Returns
        -------
        None
        """
        import flopy
        swn = self._swn
        model = self.model
        dis = model.dis

        if "slope" not in self.reaches.columns:
            self.logger.info(
                "'slope' not yet evaluated, setting with set_reach_slope()")
            self.set_reach_slope()

        # Column names common to segments and segment_data
        segment_cols = [
            "roughch",
            "hcond1", "thickm1", "elevup", "width1",
            "hcond2", "thickm2", "elevdn", "width2"]
        # Tidy any previous attempts
        for col in segment_cols:
            if col in self.segments.columns:
                del self.segments[col]
        # Combine pairs of series for each segment
        more_segment_columns = pd.concat([
            swn._pair_segment_values(hyd_cond1, hyd_cond_out, "hcond"),
            swn._pair_segment_values(thickness1, thickness_out, "thickm"),
            swn._pair_segment_values(width1, width_out, name="width")
        ], axis=1, copy=False)
        for name, series in more_segment_columns.iteritems():
            self.segments[name] = series
        self.segments["roughch"] = swn._segment_series(roughch)

        # Interpolate segment properties to each reach
        self.set_reach_data_from_series("thickm", thickness1, thickness_out)
        self.set_reach_data_from_series("strthick", thickness1, thickness_out)
        self.set_reach_data_from_series(
            "strhc1", hyd_cond1, hyd_cond_out, log10=True)

        self.reaches["strtop"] = 0.0
        if swn.has_z:
            for reachID, geom in self.reaches.geometry.iteritems():
                if geom.is_empty or not isinstance(geom, LineString):
                    continue
                # Get strtop from LineString mid-point Z
                zm = geom.interpolate(0.5, normalized=True).z
                self.reaches.at[reachID, "strtop"] = zm
        else:
            # Get stream values from top of model
            self.set_reach_data_from_array("strtop", dis.top.array)

        has_diversions = self.diversions is not None
        cols = ["iseg", "segnum"]
        if has_diversions:
            cols += ["diversion", "divid"]
        # Build segment_data for Data Set 6
        self.segment_data = self.reaches[cols].drop_duplicates().rename(
            columns={"iseg": "nseg"})
        # index changes from "reachID", to "segnum", to finally "nseg"
        self.segment_data["icalc"] = 1  # assumption for all streams
        self.segment_data["outseg"] = 0
        if has_diversions:
            rows = ~self.segment_data.diversion
            segnum2nseg_d = self.segment_data.loc[rows].set_index(
                "segnum")["nseg"].to_dict()
            self.segment_data.loc[rows, "outseg"] = \
                self.segment_data.loc[rows, "segnum"].map(
                    lambda x: segnum2nseg_d.get(
                        self.segments.loc[x, "to_segnum"], 0))
        else:
            segnum2nseg_d = \
                self.segment_data.set_index("segnum")["nseg"].to_dict()
            self.segment_data["outseg"] = self.segment_data["segnum"].map(
                lambda x: segnum2nseg_d.get(
                    self.segments.loc[x, "to_segnum"], 0))
        self.segment_data["iupseg"] = 0  # handle diversions next
        self.segment_data["iprior"] = 0
        self.segment_data["flow"] = 0.0
        self.segment_data["runoff"] = 0.0
        self.segment_data["etsw"] = 0.0
        self.segment_data["pptsw"] = 0.0
        # upper elevation from the first and last reachID items from reaches
        self.segment_data["elevup"] = \
            self.reaches.loc[self.segment_data.index, "strtop"]
        self.segment_data["elevdn"] = self.reaches.loc[
            self.reaches.groupby(["iseg"]).ireach.idxmax().values,
            "strtop"].values
        self.segment_data.set_index("segnum", drop=False, inplace=True)
        # copy several columns over (except "elevup" and "elevdn", for now)
        segment_cols.remove("elevup")
        segment_cols.remove("elevdn")
        self.segment_data[segment_cols] = self.segments[segment_cols]
        # now use nseg as primary index, not reachID or segnum
        self.segment_data.set_index("nseg", inplace=True)
        self.segment_data.sort_index(inplace=True)

        # Process diversions / SW takes
        if has_diversions:
            # Add columns for ICALC=0
            self.segment_data["depth1"] = 0.0
            self.segment_data["depth2"] = 0.0
            # workaround for coercion issue
            self.segment_data["foo"] = ""
            itr = self.diversions[self.diversions.in_model].iterrows()
            for divid, divn in itr:
                # Get indexes
                sel = (
                    (self.segment_data.divid == divid) &
                    (self.segment_data.diversion))
                if sel.sum() == 0:
                    raise KeyError(f"divid {divid} not found in segment_data")
                elif sel.sum() > 1:
                    raise KeyError(f"more than one divid {divid} found")
                nseg = self.segment_data.index[sel][0]
                sel = (self.reaches.divid == divid) & (self.reaches.diversion)
                if sel.sum() == 0:
                    raise KeyError(f"divid {divid} not found in reaches")
                elif sel.sum() > 1:
                    raise KeyError(f"more than one divid {divid} found")
                reach_idx = self.reaches.index[sel][0]

                iupseg = segnum2nseg_d[divn.from_segnum]
                assert iupseg != 0, iupseg
                rchlen = 1.0  # length required
                thickm = 1.0  # thickness required
                hcond = 0.0  # don't allow GW exchange
                segd = self.segment_data.loc[nseg]
                self.segment_data.loc[nseg] = segd
                segd.at["icalc"] = 0  # stream depth is specified
                segd.at["outseg"] = 0
                segd.at["iupseg"] = iupseg
                segd.at["iprior"] = 0  # normal behaviour for SW takes
                segd.at["flow"] = 0.0  # abstraction assigned later
                segd.at["runoff"] = 0.0
                segd.at["etsw"] = 0.0
                segd.at["pptsw"] = 0.0
                segd.at["roughch"] = 0.0  # not used
                segd.at["hcond1"] = hcond
                segd.at["hcond2"] = hcond
                segd.at["thickm1"] = thickm
                segd.at["thickm2"] = thickm
                segd.at["width1"] = 0.0  # not used
                segd.at["width2"] = 0.0
                reach = self.reaches.loc[reach_idx]
                i, j = reach[["i", "j"]]
                strtop = dis.top[i, j]
                reach.at["iseg"] = nseg
                reach.at["ireach"] = 1
                reach.at["rchlen"] = rchlen
                reach.at["min_slope"] = 0.0
                reach.at["slope"] = 0.0
                reach.at["strthick"] = thickm
                reach.at["strtop"] = strtop
                reach.at["strhc1"] = hcond
                depth = strtop + thickm
                segd.at["depth1"] = depth
                segd.at["depth2"] = depth
                self.reaches.loc[reach_idx] = reach
                self.segment_data.loc[nseg] = segd
            # end of coercion workaround
            self.segment_data.drop("foo", axis=1, inplace=True)

        # Create flopy Sfr2 package
        segment_data = self.set_segment_data(
            abstraction=abstraction, inflow=inflow,
            flow=flow, runoff=runoff, etsw=etsw, pptsw=pptsw, return_dict=True)
        flopy.modflow.mfsfr2.ModflowSfr2(
            model=self.model,
            reach_data=self.flopy_reach_data,
            segment_data=segment_data)

    def __repr__(self):
        """Return string representation of SwnModflow object."""
        model = self.model
        if model is None:
            model_info = "model not set"
            sp_info = "model not set"
        else:
            dis = self.model.dis
            nper = dis.nper
            model_info = "flopy {} {!r}".format(
                self.model.version, self.model.name)
            sp_info = "{} stress period{} with perlen: {}".format(
                nper, '' if nper == 1 else 's',
                abbr_str(list(dis.perlen), 4))
        s = "<{}: {}\n".format(self.__class__.__name__, model_info)
        reaches = self.reaches
        if reaches is not None:
            s += "  {} in reaches ({}): {}\n".format(
                len(self.reaches), self.reaches.index.name,
                abbr_str(list(self.reaches.index), 4))
        segment_data = self.segment_data
        if segment_data is not None:
            s += "  {} in segment_data ({}): {}\n".format(
                len(segment_data), segment_data.index.name,
                abbr_str(list(segment_data.index), 4))
            is_diversion = segment_data["iupseg"] != 0
            segnum_l = list(segment_data.loc[~is_diversion, "segnum"])
            s += "    {} from segments".format(len(segnum_l))
            segnum_index_name = self.segments.index.name
            if segnum_index_name is not None:
                s += " ({})".format(segnum_index_name)
            if set(segnum_l) != set(self.segments.index):
                s += " ({:.0%} used)".format(
                    len(segnum_l) / float(len(self.segments)))
            s += ": " + abbr_str(segnum_l, 4) + "\n"
            if is_diversion.any() and self.diversions is not None:
                divid_l = list(self.segment_data.loc[is_diversion, "divid"])
                s += "    {} from diversions".format(len(divid_l))
                divid_index_name = self.diversions.index.name
                if divid_index_name is not None:
                    s += " ({})".format(divid_index_name)
                if set(divid_l) != set(self.diversions.index):
                    s += " ({:.0%} used)".format(
                        len(divid_l) / float(len(self.diversions)))
                s += ": " + abbr_str(divid_l, 4) + "\n"
        s += "  {} />".format(sp_info)
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
                elif isinstance(av, flopy.modflow.Modflow):
                    # basic test
                    assert str(av) == str(bv)
                else:
                    assert av == bv
            return True
        except (AssertionError, TypeError, ValueError):
            return False

    def __iter__(self):
        """Return object datasets with an iterator."""
        yield "class", self.__class__.__name__
        yield "segments", self.segments
        yield "segment_data", self.segment_data
        yield "reaches", self.reaches
        yield "diversions", self.diversions
        yield "model", self.model

    def __setstate__(self, state):
        """Set object attributes from pickle loads."""
        if not isinstance(state, dict):
            raise ValueError("expected 'dict'; found {!r}".format(type(state)))
        elif "class" not in state:
            raise KeyError("state does not have 'class' key")
        elif state["class"] != self.__class__.__name__:
            raise ValueError("expected state class {!r}; found {!r}"
                             .format(state["class"], self.__class__.__name__))
        self.__init__()
        self.segments = state["segments"]
        self.segment_data = state["segment_data"]
        self.reaches = state["reaches"]
        self.diversions = state["diversions"]
        # Note: model must be set outsie of this method

    @_SwnModflow.model.setter
    def model(self, model):
        """Set model property from flopy.modflow.Modflow."""
        import flopy
        if not isinstance(model, flopy.modflow.Modflow):
            raise ValueError(
                "'model' must be a flopy Modflow object; found "
                + str(type(model)))
        elif not model.has_package("DIS"):
            raise ValueError("DIS package required")
        elif not model.has_package("BAS6"):
            raise ValueError("BAS6 package required")
        _model = getattr(self, "_model", None)
        if _model is not None and _model is not model:
            self.logger.info("swapping 'model' object")
        self._model = model
        # Build stress period DataFrame from modflow model
        stress_df = pd.DataFrame({"perlen": self.model.dis.perlen.array})
        modeltime = self.model.modeltime
        stress_df["duration"] = pd.TimedeltaIndex(
            stress_df["perlen"].cumsum(), modeltime.time_units)
        stress_df["start"] = pd.to_datetime(modeltime.start_datetime)
        stress_df["end"] = stress_df["duration"] + stress_df.at[0, "start"]
        stress_df.loc[1:, "start"] = stress_df["end"].iloc[:-1].values
        self._stress_df = stress_df  # keep this for debugging
        self.time_index = pd.DatetimeIndex(stress_df["start"]).copy()
        self.time_index.name = None

    @property
    def flopy_reach_data(self):
        """Return numpy.recarray for flopy's ModflowSfr2 reach_data.

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

    def set_segment_data(self, abstraction={}, inflow={}, flow={}, runoff={},
                         etsw={}, pptsw={}, return_dict=False):
        """
        Set timeseries data in segment_data required for flopy's ModflowSfr2.

        This method does two things:

            1. Updates sfr.segment_data, which is a dict of rec.array
               for each stress period.
            2. Updates summary statistics in segment_data if there are more
               than one stress period, otherwise values are kept for one
               stress period.

        Other stationary data members that are part of segment_data
        (e.g. hcond1, elevup, etc.) are not modified.

        Parameters
        ----------
        abstraction : dict or pandas.DataFrame, optional
            Surface water abstraction from diversions. Default is {} (zero).
            Keys are matched to diversions index.
        inflow : dict or pandas.DataFrame, optional
            Streamflow at the bottom of each segment, which is used to to
            determine the streamflow entering the upstream end of a segment if
            it is not part of the SFR network. Internal flows are ignored.
            A dict can be used to provide constant values to segnum
            identifiers. If a DataFrame is passed for a model with more than
            one stress period, the index must be a DatetimeIndex aligned with
            the start of each model stress period.
            Default is {} (no outside inflow added to flow term).
        flow : dict or pandas.DataFrame, optional
            Flow to the top of each segment. This is added to any inflow,
            which is handled separately. This can be negative for withdrawls.
            Default is {} (zero).
        runoff : dict or pandas.DataFrame, optional
            Runoff to each segment. Default is {} (zero).
        etsw : dict or pandas.DataFrame, optional
            Evapotranspiration removed from each segment. Default is {} (zero).
        pptsw : dict or pandas.DataFrame, optional
            Precipitation added to each segment. Default is {} (zero).
        return_dict : bool, optional
            If True, return segment_data instead of setting the sfr object.
            Default False, which implies that an sfr object exists.

        Returns
        -------
        None or dict (if return_dict is True)

        """
        from flopy.modflow.mfsfr2 import ModflowSfr2
        # Build stress period DataFrame from modflow model
        dis = self.model.dis
        stress_df = pd.DataFrame({"perlen": dis.perlen.array})
        modeltime = self.model.modeltime
        stress_df["duration"] = pd.TimedeltaIndex(
                stress_df["perlen"].cumsum(), modeltime.time_units)
        stress_df["start"] = pd.to_datetime(modeltime.start_datetime)
        stress_df["end"] = stress_df["duration"] + stress_df.at[0, "start"]
        stress_df.loc[1:, "start"] = stress_df["end"].iloc[:-1].values
        # Consider all IDs from segments/diversions
        segments_segnums = set(self.segments.index)
        has_diversions = self.diversions is not None
        if has_diversions:
            diversions_divids = set(self.diversions.index)
        else:
            diversions_divids = set()

        def check_ts(data, name):
            """Return DataFrame with index along nper.

            Columns are either segnum or divid (checked later).
            """
            if isinstance(data, dict):
                data = pd.DataFrame(data, index=stress_df["start"])
            elif not isinstance(data, pd.DataFrame):
                raise ValueError(
                    "{0} must be a dict or DataFrame".format(name))
            data.index.name = name  # handy for debugging
            if len(data) != dis.nper:
                raise ValueError(
                    "length of {0} ({1}) is different than nper ({2})"
                    .format(name, len(data), dis.nper))
            if dis.nper > 1:  # check DatetimeIndex
                if not isinstance(data.index, pd.DatetimeIndex):
                    raise ValueError(
                        "{0}.index must be a pandas.DatetimeIndex"
                        .format(name))
                elif not (data.index == stress_df["start"]).all():
                    try:
                        t = stress_df["start"].to_string(
                                index=False, max_rows=5).replace("\n", ", ")
                    except TypeError:
                        t = abbr_str(list(stress_df["start"]))
                    raise ValueError(
                        "{0}.index does not match expected ({1})"
                        .format(name, t))
            # Also do basic check of column IDs against diversions/segments
            if name == "abstraction":
                if not has_diversions:
                    if len(data.columns) > 0:
                        self.logger.error(
                            "abstraction provided, but diversions are not "
                            "defined for the surface water network")
                        data.drop(data.columns, axis=1, inplace=True)
                    return data
                parent = self.diversions
                parent_name = "diversions"
                parent_s = diversions_divids
            else:
                parent = self.segments
                parent_name = "segments"
                parent_s = segments_segnums
            try:
                data.columns = data.columns.astype(parent.index.dtype)
            except (ValueError, TypeError):
                raise ValueError(
                    "{0}.columns.dtype must be same as {1}.index.dtype"
                    .format(name, parent_name))
            data_id_s = set(data.columns)
            if len(data_id_s) > 0:
                if data_id_s.isdisjoint(parent_s):
                    msg = "{0}.columns (or keys) not found in {1}.index: {2}"\
                        .format(name, parent_name, abbr_str(data_id_s))
                    if name == "inflow":
                        self.logger.warning(msg)
                    else:
                        raise ValueError(msg)
                if name != "inflow":  # some segnums accumulate outside flow
                    not_found = data_id_s.difference(parent_s)
                    if not data_id_s.issubset(parent_s):
                        self.logger.warning(
                            "dropping %s of %s %s.columns, which are "
                            "not found in %s.index: %s",
                            len(not_found), len(data_id_s), name,
                            parent_name, abbr_str(data_id_s))
                        data.drop(not_found, axis=1, inplace=True)
            return data

        self.logger.debug("checking timeseries data against modflow model")
        abstraction = check_ts(abstraction, "abstraction")
        inflow = check_ts(inflow, "inflow")
        flow = check_ts(flow, "flow")
        runoff = check_ts(runoff, "runoff")
        etsw = check_ts(etsw, "etsw")
        pptsw = check_ts(pptsw, "pptsw")

        # Translate segnum/divid to nseg
        is_diversion = self.segment_data["iupseg"] != 0
        if is_diversion.any():
            divid2nseg = self.segment_data[is_diversion]\
                .reset_index().set_index("divid")["nseg"]
            divid2nseg_d = divid2nseg.to_dict()
        else:
            divid2nseg_d = {}
        segnum2nseg = self.segment_data[~is_diversion]\
            .reset_index().set_index("segnum")["nseg"]
        segnum2nseg_d = segnum2nseg.to_dict()
        segnum_s = set(segnum2nseg_d.keys())

        def map_nseg(data, name):
            data_id_s = set(data.columns)
            if len(data_id_s) == 0:
                return data
            if name == "abstraction":
                colid2nseg_d = divid2nseg_d
                parent_descr = "diversions"
            else:
                colid2nseg_d = segnum2nseg_d
                parent_descr = "regular segments"
            colid_s = set(colid2nseg_d.keys())
            not_found = data_id_s.difference(colid_s)
            if not data_id_s.issubset(colid_s):
                self.logger.warning(
                    "dropping %s of %s %s.columns, which are "
                    "not found in segment_data.index for %s",
                    len(not_found), len(data_id_s), name,
                    parent_descr)
                data.drop(not_found, axis=1, inplace=True)
            return data.rename(columns=colid2nseg_d)

        self.logger.debug("mapping segnum/divid to segment_data.index (nseg)")
        abstraction = map_nseg(abstraction, "abstraction")
        flow = map_nseg(flow, "flow")
        runoff = map_nseg(runoff, "runoff")
        etsw = map_nseg(etsw, "etsw")
        pptsw = map_nseg(pptsw, "pptsw")

        self.logger.debug("accumulating inflow from outside network")
        # Create an "inflows" DataFrame calculated from combining "inflow"
        inflows = pd.DataFrame(index=inflow.index)
        has_inflow = len(inflow.columns) > 0
        missing_inflow_segnums = []
        if has_inflow:
            self.segment_data["inflow_segnums"] = None
        elif "inflow_segnums" in self.segment_data:
            self.segment_data.drop("inflow_segnums", axis=1, inplace=True)
        # Determine upstream flows needed for each SFR segment
        for segnum in self.segment_data.loc[~is_diversion, "segnum"]:
            nseg = segnum2nseg_d[segnum]
            from_segnums = self.segments.at[segnum, "from_segnums"]
            if not from_segnums:
                continue
            # gather segments outside SFR network
            outside_segnums = from_segnums.difference(segnum_s)
            if not outside_segnums:
                continue
            if has_inflow:
                inflow_series = pd.Series(0.0, index=inflow.index)
                inflow_segnums = set()
                for from_segnum in outside_segnums:
                    try:
                        inflow_series += inflow[from_segnum]
                        inflow_segnums.add(from_segnum)
                    except KeyError:
                        self.logger.warning(
                            "flow from segment %s not provided by inflow term "
                            "(needed for %s)", from_segnum, segnum)
                        missing_inflow_segnums.append(from_segnum)
                if inflow_segnums:
                    inflows[nseg] = inflow_series
                    self.segment_data.at[nseg, "inflow_segnums"] = \
                        inflow_segnums
            else:
                missing_inflow_segnums += outside_segnums
        if not has_inflow and len(missing_inflow_segnums) > 0:
            self.logger.warning(
                "inflow from %d segnums are needed to determine flow from "
                "outside SFR network: %s", len(missing_inflow_segnums),
                abbr_str(missing_inflow_segnums))
        # Append extra columns to segment_data that are used by flopy
        segment_column_names = []
        nss = len(self.segment_data)
        for name, dtype in ModflowSfr2.get_default_segment_dtype().descr:
            if name == "nseg":  # skip adding the index
                continue
            segment_column_names.append(name)
            if name not in self.segment_data.columns:
                self.segment_data[name] = np.zeros(nss, dtype=dtype)
        # Re-assmble stress period dict for flopy, with iper keys
        segment_data = {}
        has_abstraction = len(abstraction.columns) > 0
        has_inflows = len(inflows.columns) > 0
        has_flow = len(flow.columns) > 0
        has_runoff = len(runoff.columns) > 0
        has_etsw = len(etsw.columns) > 0
        has_pptsw = len(pptsw.columns) > 0
        for iper in range(dis.nper):
            # Store data for each stress period
            self.segment_data["flow"] = 0.0
            self.segment_data["runoff"] = 0.0
            self.segment_data["etsw"] = 0.0
            self.segment_data["pptsw"] = 0.0
            if has_abstraction:
                item = abstraction.iloc[iper]
                self.segment_data.loc[item.index, "flow"] = item
            if has_inflows:
                item = inflows.iloc[iper]
                self.segment_data.loc[item.index, "flow"] += item
            if has_flow:
                item = flow.iloc[iper]
                self.segment_data.loc[item.index, "flow"] += item
            if has_runoff:
                item = runoff.iloc[iper]
                self.segment_data.loc[item.index, "runoff"] = item
            if has_etsw:
                item = etsw.iloc[iper]
                self.segment_data.loc[item.index, "etsw"] = item
            if has_pptsw:
                item = pptsw.iloc[iper]
                self.segment_data.loc[item.index, "pptsw"] = item
            segment_data[iper] = self.segment_data[segment_column_names]\
                .to_records(index=True)  # index is nseg

        # For models with more than one stress period, evaluate summary stats
        if dis.nper > 1:
            # Remove time-varying data from last stress period
            self.segment_data.drop(
                ["flow", "runoff", "etsw", "pptsw"], axis=1, inplace=True)

            def add_summary_stats(name, df):
                if len(df.columns) == 0:
                    return
                self.segment_data[name + "_min"] = 0.0
                self.segment_data[name + "_mean"] = 0.0
                self.segment_data[name + "_max"] = 0.0
                min_v = df.min(0)
                mean_v = df.mean(0)
                max_v = df.max(0)
                self.segment_data.loc[min_v.index, name + "_min"] = min_v
                self.segment_data.loc[mean_v.index, name + "_mean"] = mean_v
                self.segment_data.loc[max_v.index, name + "_max"] = max_v

            add_summary_stats("abstraction", abstraction)
            add_summary_stats("inflow", inflows)
            add_summary_stats("flow", flow)
            add_summary_stats("runoff", runoff)
            add_summary_stats("etsw", etsw)
            add_summary_stats("pptsw", pptsw)

        if return_dict:
            return segment_data
        else:
            self.model.sfr.segment_data = segment_data

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
                dis.top.array, mask=model.bas6.ibound.array[0] == 0)
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
                                 mask=model.bas6.ibound.array[0] == 0)
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
