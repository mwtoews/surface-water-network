# -*- coding: utf-8 -*-
"""Interface for flopy's implementation for MODFLOW."""

__all__ = ["SwnModflow"]

import inspect
from itertools import zip_longest

import numpy as np
import pandas as pd

from swn.modflow._base import SwnModflowBase
from swn.modflow._misc import invert_series, transform_data_to_series_or_frame
from swn.util import abbr_str

try:
    import matplotlib
except ImportError:
    matplotlib = False


class SwnModflow(SwnModflowBase):
    """Surface water network adaptor for MODFLOW.

    Attributes
    ----------
    model : flopy.modflow.Modflow
        Instance of flopy.modflow.Modflow
    segments : geopandas.GeoDataFrame
        Copied from swn.segments, but with additional columns added
    segment_data : pd.DataFrame or None
        Dataframe of stationary data for MODFLOW SFR, index is nseg, ordered
        and starting from 1. Additional column "segnum" is usedto identify
        segments, and if present, "divid" to identify diversions, where
        iupseg != 0.
    segment_data_ts : dict or None
        Dataframe of time-varying data for MODFLOW SFR, key is dataset name.
    reaches : geopandas.GeoDataFrame
        Similar to structure in model.sfr.reach_data with index "reachID",
        ordered and starting from 1. Contains geometry and other columns
        not used by flopy. Use flopy_reach_data() for use with flopy.
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
        self.segment_data = None
        self.segment_data_ts = None
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
        sel = obj.reaches["rchlen"] == 0
        if sel.any():
            # zero lengths not permitted
            obj.reaches.loc[sel, "rchlen"] = 1.0

        return obj

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
                nper, "" if nper == 1 else "s",
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
        yield from super().__iter__()
        yield "segment_data", self.segment_data
        yield "segment_data_ts", self.segment_data_ts

    def __setstate__(self, state):
        """Set object attributes from pickle loads."""
        super().__setstate__(state)
        self.segment_data = state["segment_data"]
        self.segment_data_ts = state["segment_data_ts"]

    @SwnModflowBase.model.setter
    def model(self, model):
        """Set model property from flopy.modflow.Modflow."""
        import flopy
        if not isinstance(model, flopy.modflow.Modflow):
            raise ValueError(
                "'model' must be a flopy Modflow object; found " +
                str(type(model)))
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

    def new_segment_data(self):
        """Generate an empty segment_data DataFrame.

        Several other columns are added, including:
            - "segnum" - index for segments DataFrame
            - "inflow" - positive inflow rate for upstream external flow
        Additionally, if diversions are set:
            - "divid" - index for diversions DataFrame
            - "abstraction" - positive abstraction rate

        Only the following are determined:
            - "nseg" - 1, 2, ...,  nss
            - "iupseg" - 0 for most segments, except diversions
            - "outseg" - outflowing nseg

        Returns
        -------
        pandas.DataFrame
        """
        from flopy.modflow.mfsfr2 import ModflowSfr2
        if self.segment_data is None:
            self.logger.info("creating new segment_data")
        else:
            self.logger.warning("resetting new segment_data")
        self.segment_data_ts = {}
        seg_dtype = ModflowSfr2.get_default_segment_dtype()
        # upgrade from single to double precision
        seg_dtype = np.dtype(
            [(n, d.replace("f4", "f8")) for n, d in list(seg_dtype.descr)])
        iseg_gb = self.reaches.groupby("iseg")
        segnums = iseg_gb["segnum"].first()
        self.segment_data = pd.DataFrame(np.zeros(len(segnums), seg_dtype))
        self.segment_data.nseg = segnums.index
        self.segment_data.set_index("nseg", inplace=True)
        # Add extra columns
        self.segment_data.insert(0, "segnum", segnums)
        self.segment_data["inflow"] = 0.0
        has_diversions = self.diversions is not None
        if has_diversions:
            self.segment_data.insert(1, "divid", iseg_gb["divid"].first())
            self.segment_data["abstraction"] = 0.0
            nseg2segnum = self.reaches.loc[
                ~self.reaches.diversion].groupby("iseg")["segnum"].first()
            segnum2nseg = invert_series(nseg2segnum)
            divid2segnum = self.diversions.loc[
                self.diversions.in_model, "from_segnum"]
            sel = ~self.segment_data.segnum.isin(self.segments.in_model.index)
            self.segment_data.loc[sel, "iupseg"] = self.segment_data.loc[
                sel, "divid"].apply(lambda d: segnum2nseg[divid2segnum[d]])
        else:
            segnum2nseg = invert_series(self.segment_data["segnum"])
        # Evaluate outseg
        segnum2nseg_d = segnum2nseg.to_dict()
        if has_diversions:
            sel = self.segment_data.iupseg == 0
            self.segment_data.loc[sel, "outseg"] = \
                self.segment_data.loc[sel, "segnum"].map(
                    lambda x: segnum2nseg_d.get(
                        self.segments.loc[x, "to_segnum"], 0))
        else:
            self.segment_data["outseg"] = self.segment_data["segnum"].map(
                lambda x: segnum2nseg_d.get(
                    self.segments.loc[x, "to_segnum"], 0))

    def _check_segment_data_name(self, name: str):
        """Helper method to check name used for set_segment_data_* methods."""
        if not isinstance(name, str):
            raise ValueError("name must be str type")
        if self.segment_data is None:
            self.new_segment_data()
        if name == "nseg":
            raise ValueError("'nseg' can't be set")
        elif name not in self.segment_data.columns:
            cols = ", ".join(repr(n) for n in self.segment_data.columns)
            raise KeyError(f"could not find {name!r} in {cols}")

    def set_segment_data_from_scalar(
            self, name: str, data, which: str = "all"):
        """Set segment_data from a scalar.

        This method can be used to set data that does not vary in time.

        Parameters
        ----------
        name : str
            Name for dataset, from segment_data columns.
        data : int or float
            Data to assign to each segment. If a float, this value
            is a constant. If a pandas Series, then this is applied for
            each segment.
        which : str, default = "all"
            Determine which segment_data rows should be set as "segments",
            "diversions" (determined from IUPSEG), or "all".
        """
        self._check_segment_data_name(name)
        if not np.isscalar(data):
            raise ValueError(repr(name) + " data is not scalar")
        self.logger.debug(
            "setting scalar segment_data[%r] = %r for %s", name, data, which)
        if which == "all":
            self.segment_data[name] = data
        elif which == "segments":
            self.segment_data.loc[self.segment_data.iupseg == 0, name] = data
        elif which == "diversions":
            self.segment_data.loc[self.segment_data.iupseg != 0, name] = data
        else:
            raise ValueError(
                "'which' should be one of 'all', 'segments' or 'diversions'")
        if name in self.segment_data_ts:
            self.logger.warning("dropping %r from segment_data_ts", name)
            del self.segment_data_ts[name]
        return

    def _set_segment_data(self, name, data):
        """Helper function for set_* methods for frame or tsvar data."""
        caller = inspect.stack()[1].function
        # dfref = getattr(self, dfname)  # dataframe of series for stationary
        assert self.segment_data is not None
        if isinstance(data, pd.Series):
            if len(data.index) == 0:
                self.logger.debug("%s: empty %r series", caller, name)
                return
            self.logger.debug(
                "%s: setting segment_data[%r] from series with shape %s",
                caller, name, data.shape)
            self.segment_data.loc[data.index, name] = data
        elif isinstance(data, pd.DataFrame):
            if len(data.columns) == 0:
                self.logger.debug("%s: empty %r frame", caller, name)
                return
            if name in name in self.segment_data_ts:
                self.logger.debug(
                    "%s: combining %r with existing segment_data_ts",
                    caller, name)
                existing = self.segment_data_ts[name]
                existing.loc[data.index, data.columns] = data
                self.segment_data_ts[name] = existing
            else:
                self.logger.debug(
                    "%s: creating new %r in segment_data_ts", caller, name)
                self.segment_data_ts[name] = data
        else:
            raise NotImplementedError("expected a series or frame")

    def set_segment_data_from_segments(self, name: str, data):
        """Set segment_data from a series indexed by segments.

        Parameters
        ----------
        name : str
            Name for dataset, from segment_data columns.
        data : int, float, dict, pandas.Series or pandas.DataFrame
            Data to assigned from segments. If a pandas Series, then this is
            applied for each index matched by segnum. If a dict, then
            each item is applied for each key matched by segnum.
        """
        self._check_segment_data_name(name)
        if np.isscalar(data):
            self.set_segment_data_from_scalar(name, data, "segments")
            return
        # Prepare mapping between segnum <-> nseg
        nseg2segnum = self.segment_data.loc[
            self.segment_data.iupseg == 0, "segnum"]
        mapping = invert_series(nseg2segnum)
        ignore = set(self.segments.index[~self.segments.in_model])
        dtype = self.segment_data[name].dtype
        data = transform_data_to_series_or_frame(
            data, dtype, self.time_index, mapping, ignore)
        self._set_segment_data(name, data)

    def set_segment_data_from_diversions(self, name: str, data):
        """Set segment_data from a series indexed by diversions.

        Parameters
        ----------
        name : str
            Name for dataset, from segment_data columns.
        data : float, dict, pandas.Series or pandas.DataFrame
            Data to assigned from diversions. If a pandas Series, then this is
            applied for each index matched by divid. If a dict, then
            each item is applied for each key matched by divid.
        """
        self._check_segment_data_name(name)
        if np.isscalar(data):
            self.set_segment_data_from_scalar(name, data, "diversions")
            return
        # Prepare mapping between divid <-> nseg
        nseg2divid = self.segment_data.loc[
            self.segment_data.iupseg != 0, "divid"]
        mapping = invert_series(nseg2divid)
        ignore = set(self.diversions.index[~self.diversions.in_model])
        dtype = self.segment_data[name].dtype
        data = transform_data_to_series_or_frame(
            data, dtype, self.time_index, mapping, ignore)
        self._set_segment_data(name, data)

    def set_segment_data_inflow(self, data):
        """Set segment_data inflow data upstream of the model.

        Parameters
        ----------
        data: dict, pandas.Series or pandas.DataFrame
            Time series of flow for segnums either inside or outside model,
            indexed by segnum.
        """
        inflow = self._get_segments_inflow(data)
        self.set_segment_data_from_segments("inflow", inflow)
        if "inflow_segnums" in self.segments:
            self.segment_data["inflow_segnums"] = \
                [set() for _ in range(len(self.segment_data))]
            self.set_segment_data_from_segments(
                "inflow_segnums", self.segments["inflow_segnums"])

    def flopy_reach_data(self):
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

    def flopy_segment_data(self):
        """Return dict of numpy.recarray for flopy's ModflowSfr2 segment_data.

        This method relies of segnum and divid values being mapped to nseg,
        and stored in segment_data for stationary and segment_data_ts
        for time-varying data.

        Importantly, the final FLOW term is determined by combining
        abstraction and inflow terms.

        Parameters
        ----------
        None

        Returns
        -------
        dict

        """
        from flopy.modflow.mfsfr2 import ModflowSfr2
        if self.segment_data is None:
            self.logger.warning(
                "'segment_data' was not set; using default values")
            self.new_segment_data()
        seg_dtype = ModflowSfr2.get_default_segment_dtype()
        seg_names = list(seg_dtype.names[1:])  # everything except nseg

        has_diversions = self.diversions is not None
        # Re-assmble stress period dict for flopy, with iper keys
        segment_data = {}
        for iper, tidx in enumerate(self.time_index):
            # Store data for each stress period
            seg_df = self.segment_data.copy()
            # Update any time-varying components
            for key, df in self.segment_data_ts.items():
                ts = df.loc[tidx]
                seg_df.loc[ts.index, key] = ts
            # Combine terms for FLOW
            if has_diversions:
                if seg_df.abstraction.any():
                    # Add abstraction to FLOW term
                    seg_df.flow += seg_df.abstraction
            if seg_df.inflow.any():
                # Add external inflow to FLOW term
                seg_df.flow += seg_df.inflow
            segment_data[iper] = seg_df[seg_names].to_records(index=True)
        return segment_data

    def default_segment_data(
            self, hyd_cond1=1., hyd_cond_out=None,
            thickness1=1., thickness_out=None, width1=None, width_out=None,
            roughch=0.024):
        """High-level function to set default segment_data for MODFLOW SFR.

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
            WIDTH1/WIDTH2 outputs. Default None will first try finding
            "width" from "segments", otherwise will use 10.
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
        self.logger.info("default_segment_data: using high-level function")
        if self.segment_data is None:
            self.new_segment_data()

        if "slope" not in self.reaches.columns:
            self.logger.info(
                "default_segment_data: 'slope' not yet evaluated, setting "
                "with set_reach_slope()")
            self.set_reach_slope()

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
                "default_segment_data: 'width' " + action, *action_args)

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
        swn = self._swn
        more_segment_columns = pd.concat([
            swn.pair_segments_frame(hyd_cond1, hyd_cond_out, "hcond"),
            swn.pair_segments_frame(thickness1, thickness_out, "thickm"),
            swn.pair_segments_frame(
                width1, width_out, name="width", method="constant")
        ], axis=1, copy=False)
        for name, series in more_segment_columns.iteritems():
            self.segments[name] = series
        self.segments["roughch"] = swn.segments_series(roughch)

        # Interpolate segment properties to each reach
        self.set_reach_data_from_segments(
            "thickm", thickness1, thickness_out)
        self.set_reach_data_from_segments(
            "strthick", thickness1, thickness_out)
        self.set_reach_data_from_segments(
            "strhc1", hyd_cond1, hyd_cond_out, log=True)

        # Get stream values from top of model
        self.set_reach_data_from_array("strtop", self.model.dis.top.array)
        if "zcoord_avg" in self.reaches.columns:
            # Created by set_reach_slope(); be aware of NaN
            nn = ~self.reaches.zcoord_avg.isnull()
            self.reaches.loc[nn, "strtop"] = self.reaches.loc[nn, "zcoord_avg"]

        has_diversions = self.diversions is not None
        # Assume all streams will use ICALC=1
        self.set_segment_data_from_scalar("icalc", 1, "segments")
        # upper elevation from the first and last reachID items from reaches
        iseg_gb = self.reaches.groupby(["iseg"])
        self.segment_data["elevup"] = \
            self.reaches.loc[iseg_gb.ireach.idxmin(), "strtop"].values
        self.segment_data["elevdn"] = \
            self.reaches.loc[iseg_gb.ireach.idxmax(), "strtop"].values
        # copy several columns over (except "elevup" and "elevdn")
        segment_cols.remove("elevup")
        segment_cols.remove("elevdn")
        for name in segment_cols:
            self.set_segment_data_from_segments(name, self.segments[name])

        # Process diversions / SW takes
        if has_diversions:
            # Assume all diversions will use ICALC=0
            self.set_segment_data_from_scalar("icalc", 0, "diversions")
            # thickness and depth required
            self.set_segment_data_from_scalar("thickm1", 1.0, "diversions")
            self.set_segment_data_from_scalar("thickm2", 1.0, "diversions")
            self.set_segment_data_from_scalar("depth1", 1.0, "diversions")
            self.set_segment_data_from_scalar("depth2", 1.0, "diversions")

    def set_sfr_obj(self, **kwds):
        """Set MODFLOW SFR package data to flopy model.

        Parameters
        ----------
        **kwargs : dict, optional
            Passed to flopy.modflow.mfsfr2.ModflowSfr2.
        """
        import flopy

        flopy.modflow.mfsfr2.ModflowSfr2(
            model=self.model,
            reach_data=self.flopy_reach_data(),
            segment_data=self.flopy_segment_data(),
            **kwds)

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
