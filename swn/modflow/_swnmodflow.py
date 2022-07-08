"""Interface for flopy's implementation for MODFLOW."""

__all__ = ["SwnModflow"]

import inspect
from itertools import zip_longest

import numpy as np
import pandas as pd

from ..util import abbr_str
from ._base import SwnModflowBase
from ._misc import invert_series, transform_data_to_series_or_frame

try:
    import matplotlib
except ImportError:
    matplotlib = False


class SwnModflow(SwnModflowBase):
    """Surface water network adaptor for MODFLOW.

    Attributes
    ----------
    swn : swn.SurfaceWaterNetwork
        Instance of a SurfaceWaterNetwork.
    model : flopy.modflow.Modflow
        Instance of flopy.modflow.Modflow
    segments : geopandas.GeoDataFrame
        Copied from swn.segments, but with additional columns added
    segment_data : pd.DataFrame or None
        Dataframe of stationary data for MODFLOW SFR, index is nseg, ordered
        and starting from 1. Additional column "segnum" is used to identify
        segments, and if present, "divid" to identify diversions, where
        iupseg != 0.
    segment_data_ts : dict or None
        Dataframe of time-varying data for MODFLOW SFR, key is dataset name.
    reaches : geopandas.GeoDataFrame
        Similar to structure in model.sfr.reach_data with index "reachID",
        ordered and starting from 1. Contains geometry and other columns
        not used by flopy. Use :py:meth:`flopy_reach_data`
        for use with flopy.
    diversions :  geopandas.GeoDataFrame, pandas.DataFrame or None
        Copied from ``swn.diversions``, if set/defined.
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

        Examples
        --------
        >>> import flopy
        >>> import swn
        >>> lines = swn.spatial.wkt_to_geoseries([
        ...    "LINESTRING (60 100, 60  80)",
        ...    "LINESTRING (40 130, 60 100)",
        ...    "LINESTRING (70 130, 60 100)"])
        >>> lines.index += 100
        >>> n = swn.SurfaceWaterNetwork.from_lines(lines)
        >>> m = flopy.modflow.Modflow(version="mf2005")
        >>> _ = flopy.modflow.ModflowDis(
        ...     m, nrow=3, ncol=2, delr=20.0, delc=20.0, xul=30.0, yul=130.0,
        ...     top=15.0, botm=10.0)
        >>> _ = flopy.modflow.ModflowBas(m)
        >>> nm = swn.modflow.SwnModflow.from_swn_flopy(n, m)
        >>> print(nm)
        <SwnModflow: flopy mf2005 'modflowtest'
          7 in reaches (reachID): [1, 2, ..., 6, 7]
          1 stress period with perlen: [1.0] />
        >>> print(nm.reaches[["segnum", "i", "j", "iseg", "ireach", "rchlen"]])
                 segnum  i  j  iseg  ireach     rchlen
        reachID                                       
        1           101  0  0     1       1  18.027756
        2           101  0  1     1       2   6.009252
        3           101  1  1     1       3  12.018504
        4           102  0  1     2       1  21.081851
        5           102  1  1     2       2  10.540926
        6           100  1  1     3       1  10.000000
        7           100  2  1     3       2  10.000000
        """  # noqa
        if ibound_action not in ("freeze", "modify"):
            raise ValueError("ibound_action must be one of freeze or modify")

        obj = super().from_swn_flopy(
            swn=swn, model=model, domain_action=ibound_action,
            reach_include_fraction=reach_include_fraction)

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
            model_info = f"flopy {self.model.version} {self.model.name!r}"
            sp_info = "{} stress period{} with perlen: {}".format(
                nper, "" if nper == 1 else "s",
                abbr_str(list(dis.perlen), 4))
        s = f"<{self.__class__.__name__}: {model_info}\n"
        reaches = self.reaches
        if reaches is not None:
            s += "  {} in reaches ({}): {}\n".format(
                len(reaches), reaches.index.name,
                abbr_str(list(reaches.index), 4))
        segment_data = self.segment_data
        if segment_data is not None:
            s += "  {} in segment_data ({}): {}\n".format(
                len(segment_data), segment_data.index.name,
                abbr_str(list(segment_data.index), 4))
            is_diversion = segment_data["iupseg"] != 0
            segnum_l = list(segment_data.loc[~is_diversion, "segnum"])
            s += f"    {len(segnum_l)} from segments"
            segnum_index_name = self.segments.index.name
            if segnum_index_name is not None:
                s += f" ({segnum_index_name})"
            if set(segnum_l) != set(self.segments.index):
                s += f" ({len(segnum_l) / float(len(self.segments)):.0%} used)"
            s += f": {abbr_str(segnum_l, 4)}\n"
            if is_diversion.any() and self.diversions is not None:
                divid_l = list(self.segment_data.loc[is_diversion, "divid"])
                s += f"    {len(divid_l)} from diversions"
                divid_index_name = self.diversions.index.name
                if divid_index_name is not None:
                    s += f" ({divid_index_name})"
                if set(divid_l) != set(self.diversions.index):
                    frac = len(divid_l) / float(len(self.diversions))
                    s += f" ({frac:.0%} used)"
                s += f": {abbr_str(divid_l, 4)}\n"
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

    @property
    def segment_data(self):
        """Dataframe of segment data.

        The structure of the dataframe is created by
        :py:meth:`new_segment_data`. Time-varying data is stored in
        :py:attr:`segment_data_ts`.

        Attributes
        ----------
        nseg : int, index
            SFR stream segment index number, starting from 1.
        segnum, divid : int
            Index from original surface water network segments, and diversions
            (if used).
        icalc, outseg, iupseg, iprior, nstrpts, flow, runoff, etsw, pptsw, \
        roughch, roughbk, cdpth, fdpth, awdth, bwdth, hcond1, thickm1, \
        elevup, width1, depth1, thts1, thti1, eps1, uhc1, hcond2, thickm2, \
        elevdn, width2, depth2, thts2, thti2, eps2, uhc2 : int, float
            SFR inputs, as documented for MODFLOW.
        inflow : float
            Positive inflow rate from upstream external flow.
        abstraction : float, optional
            Positive abstraction rate for diversions, if used.

        See Also
        --------
        new_segment_data : Create an empty segment data frame.
        default_segment_data : High-level frame constructor for segment data.
        """
        return getattr(self, "_segment_data", None)

    @segment_data.setter
    def segment_data(self, value):
        if value is None:
            if hasattr(self, "_segment_data"):
                delattr(self, "_segment_data")
            return
        elif not isinstance(value, pd.DataFrame):
            raise ValueError(
                "segment_data must be a DataFrame or None; "
                f"found {type(value)!r}")
        # check index values
        try:
            pd.testing.assert_index_equal(
                value.index, pd.Index(np.arange(self.reaches.iseg.max()) + 1),
                check_names=False)
        except AssertionError as e:
            raise ValueError(f"segment_data nseg index is unexpected: {e!s}")
        # but don't check index name or column names
        self._segment_data = value

    @property
    def segment_data_ts(self):
        """Dict of dataframes of time-varying segment data.

        Keys for data names are the columns from :py:attr:`segment_data`.

        This attribute is reset to an empty dict by
        :py:meth:`new_segment_data`.

        See Also
        --------
        set_segment_data_from_scalar : Set all segment data to one value.
        set_segment_data_from_segments : Set all segment data from segments.
        set_segment_data_from_diversions: Set all segment data from diversions.
        """
        return getattr(self, "_segment_data_ts", None)

    @segment_data_ts.setter
    def segment_data_ts(self, value):
        if value is None:
            if hasattr(self, "_segment_data_ts"):
                delattr(self, "_segment_data_ts")
            return
        elif not isinstance(value, dict):
            raise ValueError(
                "segment_data_ts must be a dict or None; "
                f"found {type(value)!r}")
        for k, v in value.items():
            if not isinstance(v, pd.DataFrame):
                raise ValueError(
                    f"segment_data_ts key {k!r} must be a DataFrame; "
                    f"found {type(v)!r}")
        self._segment_data_ts = value

    def new_segment_data(self):
        """Generate empty segment data.

        Notes
        -----
        All values are zero except for:

            - nseg - 1, 2, ...,  nss
            - iupseg - 0 for most segments, except diversions
            - outseg - outflowing nseg

        If :py:attr:`segment_data` is already set, subsequent calls will
        reset the DataFrame and :py:attr:`segment_data_ts`.

        Examples
        --------
        >>> import flopy
        >>> import swn
        >>> lines = swn.spatial.wkt_to_geoseries([
        ...    "LINESTRING (60 100, 60  80)",
        ...    "LINESTRING (40 130, 60 100)",
        ...    "LINESTRING (70 130, 60 100)"])
        >>> lines.index += 100
        >>> n = swn.SurfaceWaterNetwork.from_lines(lines)
        >>> m = flopy.modflow.Modflow(version="mf2005")
        >>> _ = flopy.modflow.ModflowDis(
        ...     m, nrow=3, ncol=2, delr=20.0, delc=20.0, xul=30.0, yul=130.0,
        ...     top=15.0, botm=10.0)
        >>> _ = flopy.modflow.ModflowBas(m)
        >>> nm = swn.modflow.SwnModflow.from_swn_flopy(n, m)
        >>> nm.new_segment_data()
        >>> print(nm.segment_data[["segnum", "icalc", "outseg", "elevup"]])
              segnum  icalc  outseg  elevup
        nseg                               
        1        101      0       3     0.0
        2        102      0       3     0.0
        3        100      0       0     0.0

        See Also
        --------
        default_segment_data : High-level frame constructor for segment data.
        """  # noqa
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
        segment_data = pd.DataFrame(np.zeros(len(segnums), seg_dtype))
        segment_data.nseg = segnums.index
        segment_data.set_index("nseg", inplace=True)
        # Add extra columns
        segment_data.insert(0, "segnum", segnums)
        segment_data["inflow"] = 0.0
        has_diversions = self.diversions is not None
        if has_diversions:
            segment_data.insert(1, "divid", iseg_gb["divid"].first())
            segment_data["abstraction"] = 0.0
            nseg2segnum = self.reaches.loc[
                ~self.reaches.diversion].groupby("iseg")["segnum"].first()
            segnum2nseg = invert_series(nseg2segnum)
            divid2segnum = self.diversions.loc[
                self.diversions.in_model, "from_segnum"]
            sel = ~segment_data.segnum.isin(self.segments.in_model.index)
            segment_data.loc[sel, "iupseg"] = segment_data.loc[
                sel, "divid"].apply(lambda d: segnum2nseg[divid2segnum[d]])
        else:
            segnum2nseg = invert_series(segment_data["segnum"])
        # Evaluate outseg
        segnum2nseg_d = segnum2nseg.to_dict()
        if has_diversions:
            sel = segment_data.iupseg == 0
            segment_data.loc[sel, "outseg"] = \
                segment_data.loc[sel, "segnum"].map(
                    lambda x: segnum2nseg_d.get(
                        self.segments.loc[x, "to_segnum"], 0))
        else:
            segment_data["outseg"] = segment_data["segnum"].map(
                lambda x: segnum2nseg_d.get(
                    self.segments.loc[x, "to_segnum"], 0))
        self._segment_data = segment_data

    def _check_segment_data_name(self, name: str):
        """Check name used for set_segment_data_* methods."""
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
        """Set segment data from a scalar.

        This method can be used to set :py:attr:`segment_data`, which does not
        vary in time. It does not modify :py:attr:`segment_data_ts`.

        Parameters
        ----------
        name : str
            Name for dataset, from :py:attr:`segment_data` columns.
        data : int or float
            Data to assign to each segment. If a float, this value
            is a constant. If a pandas Series, then this is applied for
            each segment.
        which : str, default = "all"
            Determine which segment_data rows should be set as "segments",
            "diversions" (determined from IUPSEG), or "all".

        Examples
        --------
        >>> import flopy
        >>> import swn
        >>> lines = swn.spatial.wkt_to_geoseries([
        ...    "LINESTRING (60 100, 60  80)",
        ...    "LINESTRING (40 130, 60 100)",
        ...    "LINESTRING (70 130, 60 100)"])
        >>> lines.index += 100
        >>> n = swn.SurfaceWaterNetwork.from_lines(lines)
        >>> m = flopy.modflow.Modflow(version="mf2005")
        >>> _ = flopy.modflow.ModflowDis(
        ...     m, nrow=3, ncol=2, delr=20.0, delc=20.0, xul=30.0, yul=130.0,
        ...     top=15.0, botm=10.0)
        >>> _ = flopy.modflow.ModflowBas(m)
        >>> nm = swn.modflow.SwnModflow.from_swn_flopy(n, m)
        >>> nm.set_segment_data_from_scalar("icalc", 1)
        >>> print(nm.segment_data[["segnum", "icalc"]])
              segnum  icalc
        nseg               
        1        101      1
        2        102      1
        3        100      1

        See Also
        --------
        set_segment_data_from_segments : Set all segment data from segments.
        set_segment_data_from_diversions: Set all segment data from diversions.
        """  # noqa
        self._check_segment_data_name(name)
        if not np.isscalar(data):
            raise ValueError(f"{name!r} data is not scalar")
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
        """Set segment data from a series indexed by segments.

        Modifies :py:attr:`segment_data` or :py:attr:`segment_data_ts`.

        Parameters
        ----------
        name : str
            Name for dataset, from :py:attr:`segment_data` columns.
        data : int, float, dict, pandas.Series or pandas.DataFrame
            Data to assigned from segments. If a pandas Series, then this is
            applied for each index matched by segnum. If a dict, then
            each item is applied for each key matched by segnum.

        Examples
        --------
        >>> import flopy
        >>> import swn
        >>> lines = swn.spatial.wkt_to_geoseries([
        ...    "LINESTRING (60 100, 60  80)",
        ...    "LINESTRING (40 130, 60 100)",
        ...    "LINESTRING (70 130, 60 100)"])
        >>> lines.index += 100
        >>> n = swn.SurfaceWaterNetwork.from_lines(lines)
        >>> m = flopy.modflow.Modflow(version="mf2005")
        >>> _ = flopy.modflow.ModflowDis(
        ...     m, nrow=3, ncol=2, delr=20.0, delc=20.0, xul=30.0, yul=130.0,
        ...     top=15.0, botm=10.0)
        >>> _ = flopy.modflow.ModflowBas(m)
        >>> nm = swn.modflow.SwnModflow.from_swn_flopy(n, m)
        >>> nm.set_segment_data_from_segments("runoff", {101: 2.2})
        >>> print(nm.segment_data[["segnum", "runoff"]])
              segnum  runoff
        nseg                
        1        101     2.2
        2        102     0.0
        3        100     0.0
        >>> nm.set_segment_data_from_segments("runoff", {100: 3.3})
        >>> print(nm.segment_data[["segnum", "runoff"]])
              segnum  runoff
        nseg                
        1        101     2.2
        2        102     0.0
        3        100     3.3

        See Also
        --------
        set_segment_data_from_scalar : Set all segment data to one value.
        set_segment_data_from_diversions: Set all segment data from diversions.
        """  # noqa
        self._check_segment_data_name(name)
        if np.isscalar(data):
            self.set_segment_data_from_scalar(name, data, "segments")
            return
        # Prepare mapping between segnum <-> nseg
        nseg2segnum = self.segment_data.loc[
            self.segment_data.iupseg == 0, "segnum"]
        mapping = invert_series(nseg2segnum)
        ignore = set(self.segments.index[~self.segments.in_model])\
            .difference(nseg2segnum.values)
        dtype = self.segment_data[name].dtype
        data = transform_data_to_series_or_frame(
            data, dtype, self.time_index, mapping, ignore)
        self._set_segment_data(name, data)

    def set_segment_data_from_diversions(self, name: str, data):
        """Set segment data from a series indexed by diversions.

        Modifies :py:attr:`segment_data` or :py:attr:`segment_data_ts`.

        Parameters
        ----------
        name : str
            Name for dataset, from :py:attr:`segment_data` columns.
        data : float, dict, pandas.Series or pandas.DataFrame
            Data to assigned from diversions. If a pandas Series, then this is
            applied for each index matched by divid. If a dict, then
            each item is applied for each key matched by divid.

        Examples
        --------
        >>> import flopy
        >>> import swn
        >>> lines = swn.spatial.wkt_to_geoseries([
        ...    "LINESTRING (60 100, 60  80)",
        ...    "LINESTRING (40 130, 60 100)",
        ...    "LINESTRING (70 130, 60 100)"])
        >>> lines.index += 100
        >>> n = swn.SurfaceWaterNetwork.from_lines(lines)
        >>> diversions = swn.spatial.wkt_to_geoseries([
        ...    "POINT (58 100)",
        ...    "POINT (58 100)"]).to_frame("geometry")
        >>> diversions.index += 10
        >>> diversions["rate"] = [1.1, 2.2]
        >>> n.set_diversions(diversions=diversions)
        >>> m = flopy.modflow.Modflow(version="mf2005")
        >>> _ = flopy.modflow.ModflowDis(
        ...     m, nrow=3, ncol=2, delr=20.0, delc=20.0, xul=30.0, yul=130.0,
        ...     top=15.0, botm=10.0)
        >>> _ = flopy.modflow.ModflowBas(m)
        >>> nm = swn.modflow.SwnModflow.from_swn_flopy(n, m)
        >>> nm.set_segment_data_from_diversions("abstraction", diversions.rate)
        >>> print(nm.segment_data[["divid", "iupseg", "abstraction"]])
              divid  iupseg  abstraction
        nseg                            
        1         0       0          0.0
        2         0       0          0.0
        3         0       0          0.0
        4        10       1          1.1
        5        11       1          2.2

        See Also
        --------
        set_segment_data_from_scalar : Set all segment data to one value.
        set_segment_data_from_segments : Set all segment data from segments.
        """  # noqa
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
        """Set :py:attr:`segment_data` inflow data upstream of the model.

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

        Examples
        --------
        >>> import flopy
        >>> import swn
        >>> lines = swn.spatial.wkt_to_geoseries([
        ...    "LINESTRING Z (60 100 14, 60  80 12)",
        ...    "LINESTRING Z (40 130 15, 60 100 14)",
        ...    "LINESTRING Z (70 130 15, 60 100 14)"])
        >>> lines.index += 100
        >>> n = swn.SurfaceWaterNetwork.from_lines(lines)
        >>> m = flopy.modflow.Modflow(version="mf2005")
        >>> _ = flopy.modflow.ModflowDis(
        ...     m, nrow=3, ncol=2, delr=20.0, delc=20.0, xul=30.0, yul=130.0,
        ...     top=15.0, botm=10.0)
        >>> _ = flopy.modflow.ModflowBas(m)
        >>> nm = swn.modflow.SwnModflow.from_swn_flopy(n, m)
        >>> nm.default_segment_data()
        >>> print(nm)
        <SwnModflow: flopy mf2005 'modflowtest'
          7 in reaches (reachID): [1, 2, ..., 6, 7]
          3 in segment_data (nseg): [1, 2, 3]
            3 from segments: [101, 102, 100]
          1 stress period with perlen: [1.0] />
        >>> print(nm.segment_data[["segnum", "icalc", "elevup", "elevdn"]])
              segnum  icalc     elevup     elevdn
        nseg                                     
        1        101      1  14.750000  14.166667
        2        102      1  14.666667  14.166667
        3        100      1  13.500000  12.500000
        >>> nm.default_segment_data(width1=2, width_out=4)
        >>> print(nm.segment_data[["segnum", "icalc", "width1", "width2"]])
              segnum  icalc  width1  width2
        nseg                               
        1        101      1     2.0     2.0
        2        102      1     2.0     2.0
        3        100      1     2.0     4.0

        See Also
        --------
        new_segment_data : Create an empty segment data frame.
        """  # noqa
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
                f"default_segment_data: 'width' {action}", *action_args)

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

        # Segments not in the model should try to be inactive
        not_in_model_sel = ~self.segments.in_model
        if not_in_model_sel.any():
            idx = not_in_model_sel.index[not_in_model_sel]
            segment_data_sel = self.segment_data.segnum.isin(idx)
            self.segment_data.loc[segment_data_sel, "icalc"] = 0
            self.segment_data.loc[segment_data_sel, "hcond1"] = 0.0
            # thickness and depth required
            # self.segment_data.loc[segment_data_sel, "thickm1"] = 1.0
            # self.segment_data.loc[segment_data_sel, "thickm2"] = 1.0
            self.segment_data.loc[segment_data_sel, "depth1"] = 1.0
            self.segment_data.loc[segment_data_sel, "depth2"] = 1.0

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
        """
        Get the k,i,j location of upstream and downstream segments
        for each segment

        Returns
        -------
        pandas DataFrame with
        """
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

    def get_top_elevs_at_segs(self):
        """
        Get topsurface elevations associated with segment up and dn elevations.

        Returns
        -------
        pandas.DataFrame
            With "top_up" and "top_dn" series for the elevation model top
            at upstream and downstream ends of each segment.

        """
        dis = self.model.dis
        # assert m.sfr is not None, "need sfr package"
        self.segment_data["top_up"] = dis.top.array[
            tuple(self.segment_data[["i_up", "j_up"]].values.T)]
        self.segment_data["top_dn"] = dis.top.array[
            tuple(self.segment_data[["i_dn", "j_dn"]].values.T)]
        return self.segment_data[["top_up", "top_dn"]]

    def get_segment_incision(self):
        """
        Calculate the upstream and downstream incision of the segment.

        Returns
        -------
        pandas.DataFrame
            with "diff_up" and "diff_dn" series.
        """
        self.segment_data["diff_up"] = (self.segment_data["top_up"] -
                                        self.segment_data["elevup"])
        self.segment_data["diff_dn"] = (self.segment_data["top_dn"] -
                                        self.segment_data["elevdn"])
        return self.segment_data[["diff_up", "diff_dn"]]

    def set_seg_minincise(self, minincise=0.2, max_str_z=None):
        """
        Set segment elevation to have the minimum incision from the top.

        Parameters
        ----------
        minincise : float, default 0.2
            Desired minimum incision
        max_str_z : float, default None
            Optional parameter to prevent streams at high elevations
            (forces incision to max_str_z)

        Returns
        -------
        pandas.DataFrame
            incisions at the upstream and downstream end of each segment
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
        # extract segment length for calculating minimum drop later
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
        (multiple upstream segments route to one segment)
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
        Force segment to have minimum slope (check for backward flowing segs).

        Moves downstream end down (vertically, more incision)
        to acheive minimum slope.

        Parameters
        ----------
        seg : pandas.Series
            Series containing one row of seg_data dataframe
        *args: tuple
            Desired minumum slope

        Returns
        -------
        pandas.Series:
            Series with new downstream elevation and associated outseg_elevup
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
                self.logger.debug(
                    "segment %s, outseg = %s, old outseg_elevup = %s, "
                    "new outseg_elevup = %s",
                    seg.name, seg.outseg, seg.outseg_elevup, outseg_up)
            else:
                dn = down
                outseg_up = down - downstreambuffer
        else:
            # must be an outflow segment
            down = seg.elevdn
            if down > up - (seg.seglen * prefslope):
                dn = up - (seg.seglen * prefslope)
                self.logger.debug(
                    "outflow segment %s, outseg = %s, old elevdn = %s, "
                    "new elevdn = %s", seg.name, seg.outseg, seg.elevdn, dn)
            else:
                dn = down
        # this returns a DF once the apply is done!
        return pd.Series({"nseg": seg.name, "elevdn": dn,
                          "outseg_elevup": outseg_up})

    def set_forward_segs(self, min_slope=1e-4):
        """Set minimum slope in forwards direction.

        Notes
        -----
        Ensure slope of all segment is at least min_slope
        in the downstream direction.
        Moves down the network correcting downstream elevations if necessary

        Parameters
        ----------
        min_slope : float, default 1e-4
            Desired minimum slope

        Returns
        -------
        pandas.DataFrame
            segment_data with updated values
        """
        # upper most segments (not referenced as outsegs)
        # segdata_df = self.segment_data.sort_index(axis=1)
        segsel = ~self.segment_data.index.isin(self.segment_data["outseg"])
        while segsel.sum() > 0:
            self.logger.info(
                "Checking elevdn and outseg_elevup for %s segments",
                segsel.sum())
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

    def fix_segment_elevs(self, min_incise=0.2, min_slope=1e-4,
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
        TODO: assess if this has not been super-seeded by other functions!
        """
        kijcols = {"k_up", "i_up", "j_up", "k_dn", "i_dn", "j_dn"}
        dif = kijcols - set(self.segment_data.columns)
        if len(dif) > 1:
            # some missing
            # drop others
            others = kijcols - dif
            self.segment_data.drop(others, axis=0, inplace=True)
            # get model locations for segments ends
            self.get_seg_ijk()
        # get model cell elevations at seg ends
        self.get_top_elevs_at_segs()
        # get current segment incision at seg ends
        self.get_segment_incision()
        # move segments end elevation down to achieve minimum incision
        self.set_seg_minincise(minincise=min_incise, max_str_z=max_str_z)
        # get the elevations of downstream segments
        self.get_outseg_elev()
        # get segment length from reach lengths
        self.get_segment_length()
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

    def set_topbot_elevs_at_reaches(self):
        """
        Legacy method to add model top and bottom information to reaches.

        .. deprecated:: 1.0
            Use :py:meth:`add_model_topbot_to_reaches` instead.

        Returns
        -------
        pandas.DataFrame
            with reach cell top and bottom elevations
        """
        import warnings

        msg = ("`set_topbot_elevs_at_reaches()` is deprecated, "
               "use `add_model_topbot_to_reaches()`")
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        self.logger.warning(msg)
        return self.add_model_topbot_to_reaches()

    def fix_reach_elevs(self, minslope=1e-4, fix_dis=True, minthick=0.5):
        """Fix reach elevations.

        Notes
        -----
        Need to ensure reach elevation is:
            0. below the top
            1. below the upstream reach
            2. above the minimum slope to the bottom reach elevation
            3. above the base of layer 1
        segment by segment, reach by reach! Fun!

        Parameters
        ----------
        minslope : float, default 1e-4
            The minimum allowable slope between adjacent reaches
            (to be considered flowing downstream).
        fix_dis : bool, default True
            Move layer elevations down where it is not possible to honor
            minimum slope without going below layer 1 bottom.
        minthick : float, default 0.5
            The minimum thickness of stream bed. Will try to ensure that this
            is available below stream top before layer bottom.

        Returns
        -------
        None

        """
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
                    seg, r.ireach, rbed_elev, r.bot)
                self.logger.debug(
                    "dropping layer 1 bottom to %s to accommodate stream "
                    "@ i = %s, j = %s", new_elev, r.i, r.j)
                botms[0, r.i, r.j] = new_elev
            return botms

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
            self.reaches[reachsel]["iseg"].unique().shape[0])
        # get segments with reaches above the top surface
        segsabove = self.reaches[reachsel].groupby(
            "iseg")["iseg"].size().sort_values(ascending=False)
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
                        self.logger.debug(
                            "seg %s reach %s, incopt is \\/ below minimum "
                            "slope from bottom reach elevation",
                            seg, reach.ireach)
                        self.logger.debug(
                            "setting elevation to minslope from bottom")
                        # set to minimum slope from outreach
                        self.reaches.at[
                            reach.Index, "strtop"] = strtop_min2bot
                        # update upreach for next iteration
                        upreach_strtop = strtop_min2bot
                    elif reach.strtop_incopt > strtop_withminslope:
                        # strtop would be above upstream or give
                        # too shallow a slope from upstream
                        self.logger.debug(
                            "seg %s reach %s, incopt /\\ above upstream",
                            seg, reach.ireach)
                        self.logger.debug(
                            "setting elevation to minslope from upstream")
                        # set to minimum slope from upstream reach
                        self.reaches.at[
                            reach.Index, "strtop"] = strtop_withminslope
                        # update upreach for next iteration
                        upreach_strtop = strtop_withminslope
                    else:
                        # strtop might be ok to set to "optimum incision"
                        self.logger.debug(
                            "seg %s reach %s, incopt is -- below upstream "
                            "reach and above the bottom reach",
                            seg, reach.ireach)
                        # CHECK FIRST:
                        # if optimium incision would place it
                        # below the bottom of layer 1
                        if reach.strtop_incopt - reach.strthick < \
                                reach.bot + buffer:
                            # opt - stream thickness lower than layer 1 bottom
                            # (with a buffer)
                            self.logger.debug(
                                "seg %s reach %s, incopt - bot is x\\/ "
                                "below layer 1 bottom", seg, reach.ireach)
                            if reach.bot + reach.strthick + buffer > \
                                    strtop_withminslope:
                                # if layer bottom would put reach above
                                # upstream reach we can only set to
                                # minimum slope from upstream
                                self.logger.debug(
                                    "setting elevation to minslope "
                                    "from upstream")
                                self.reaches.at[reach.Index, "strtop"] = \
                                    strtop_withminslope
                                upreach_strtop = strtop_withminslope
                            else:
                                # otherwise we can move reach so that it
                                # fits into layer 1
                                new_elev = reach.bot + reach.strthick + buffer
                                self.logger.debug(
                                    "setting elevation to %s, above "
                                    "layer 1 bottom", new_elev)
                                # set reach top so that it is above layer 1
                                # bottom with a buffer
                                # (allowing for bed thickness)
                                self.reaches.at[reach.Index, "strtop"] = \
                                    reach.bot + buffer + reach.strthick
                                upreach_strtop = new_elev
                        else:
                            # strtop ok to set to "optimum incision"
                            # set to "optimum incision"
                            self.logger.debug("setting elevation to incopt")
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
                self.logger.debug(
                    "seg %s is always downstream and below the top", seg)
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
                self.logger.debug("checking layer %s thicknesses", k + 2)
                thincells = laythick < minthick
                self.logger.debug(
                    "%s cells less than %s", thincells.sum(), minthick)
                laythick[thincells] = minthick
                layerbots[k + 1] = layerbots[k] - laythick
            self.model.dis.botm = layerbots

    def plot_reaches_above(
            self, model=None, seg="all", dem=None, plot_bottom=False):
        """Plot map of stream elevations relative to model surfaces.

        .. deprecated:: 1.0
            Use :py:meth:`plot_reaches_vs_model` instead.

        Parameters
        ----------
        model : flopy MODFLOW model instance, default None
            With at least dis and bas6 -- so currently <MF6 method
        seg : int or str, default "all"
            Specific segment number to plot (sfr iseg/nseg)
        dem : array_like, default None
            For using as plot background -- assumes same (nrow, ncol)
            dimensions as model layer
        plot_bottom : bool, default False
            Also plot stream bed elevation relative to the bottom of layer 1

        Returns
        -------
        vtop, vbot : ModelPlot objects containing matplotlib fig and axes

        """
        import warnings

        msg = ("`plot_reaches_above()` is deprecated, "
               "use `plot_reaches_vs_model()`")
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        self.logger.warning(msg)
        if model is not None:
            self.logger.warning(
                'no longer using `model` parameter. Instead using model '
                'associated with swnmodflow class object, '
                'changed to `plot_reaches_vs_model()`',
            )
        vtop, vbot = self.plot_reaches_vs_model(
            seg,
            dem,
            plot_bottom
        )
        return vtop, vbot
        # ensure reach elevations are up-to-date
        self.add_model_topbot_to_reaches()  # TODO: check required first
        dis = model.dis
        # sfr = model.sfr
        if dem is None:
            dem = np.ma.array(
                dis.top.array, mask=model.bas6.ibound.array[0] == 0)
        sfrar = np.ma.zeros(dis.top.array.shape, "f")
        sfrar.mask = np.ones(sfrar.shape)
        if seg == "all":
            segsel = np.ones((self.reaches.shape[0]), dtype=bool)
        else:
            segsel = self.reaches["segnum"] == seg
        self.reaches['tmp_tdif'] = (self.reaches["top"] -
                                    self.reaches["strtop"])
        sfrar[
            tuple(self.reaches[segsel][["i", "j"]].values.T.tolist())
        ] = self.reaches.loc[segsel, 'tmp_tdif'].tolist()
        # .mask = np.ones(sfrar.shape)
        vtop = self.sfr_plot(
            model, sfrar, dem,
            label="str below\ntop (m)",
            lines=self.reaches.loc[segsel, ['geometry', 'tmp_tdif']]
        )

        if seg != "all":
            self.plot_profile(seg, upstream=True, downstream=True)
        if plot_bottom:
            dembot = np.ma.array(dis.botm.array[0],
                                 mask=model.bas6.ibound.array[0] == 0)
            sfrarbot = np.ma.zeros(dis.botm.array[0].shape, "f")
            sfrarbot.mask = np.ones(sfrarbot.shape)
            self.reaches['tmp_bdif'] = (self.reaches["strtop"] -
                                        self.reaches["bot"])
            sfrarbot[
                tuple(self.reaches.loc[segsel, ["i", "j"]].values.T.tolist())
            ] = self.reaches.loc[segsel, 'tmp_bdif'].tolist()
            # .mask = np.ones(sfrar.shape)
            vbot = self.sfr_plot(
                model, sfrarbot, dembot,
                label="str above\nbottom (m)",
                lines=self.reaches.loc[segsel, ['geometry', 'tmp_bdif']]
            )
        else:
            vbot = None
        return vtop, vbot

    def route_reaches(self, start, end, *, allow_indirect=False):
        """Return a list of reach identifiers that connect a pair of reaches.

        Parameters
        ----------
        start, end : any
            Start and end reach identifiers (reachID) used by FloPy.
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
            If start and/or end reach identifiers are not valid.
        ConnecionError
            If start and end reach identifiers do not connect.

        Examples
        --------
        >>> import flopy
        >>> import swn
        >>> lines = swn.spatial.wkt_to_geoseries([
        ...    "LINESTRING (60 100, 60  80)",
        ...    "LINESTRING (40 130, 60 100)",
        ...    "LINESTRING (70 130, 60 100)"])
        >>> lines.index += 100
        >>> n = swn.SurfaceWaterNetwork.from_lines(lines)
        >>> m = flopy.modflow.Modflow(version="mf2005")
        >>> _ = flopy.modflow.ModflowDis(
        ...     m, nrow=3, ncol=2, delr=20.0, delc=20.0, xul=30.0, yul=130.0,
        ...     top=15.0, botm=10.0)
        >>> _ = flopy.modflow.ModflowBas(m)
        >>> nm = swn.modflow.SwnModflow.from_swn_flopy(n, m)
        >>> nm.reaches[["iseg", "ireach", "i", "j", "segnum"]]
                 iseg  ireach  i  j  segnum
        reachID                            
        1           1       1  0  0     101
        2           1       2  0  1     101
        3           1       3  1  1     101
        4           2       1  0  1     102
        5           2       2  1  1     102
        6           3       1  1  1     100
        7           3       2  2  1     100
        >>> sel = nm.route_reaches(1, 7)
        >>> sel
        [1, 2, 3, 6, 7]
        >>> nm.reaches.loc[sel, ["iseg", "ireach", "i", "j", "rchlen"]]
                 iseg  ireach  i  j     rchlen
        reachID                               
        1           1       1  0  0  18.027756
        2           1       2  0  1   6.009252
        3           1       3  1  1  12.018504
        6           3       1  1  1  10.000000
        7           3       2  2  1  10.000000
        """  # noqa
        if start not in self.reaches.index:
            raise IndexError(f"invalid start reachID {start}")
        if end not in self.reaches.index:
            raise IndexError(f"invalid end reachID {end}")
        if start == end:
            return [start]

        # Find connections with reachID in model
        if "diversion" in self.reaches.columns:
            segnum_iseg_df = self.reaches.loc[
                ~self.reaches.diversion, ["segnum", "iseg"]]
        else:
            segnum_iseg_df = self.reaches[["segnum", "iseg"]]
        segnum_iseg = segnum_iseg_df.drop_duplicates().set_index(
            "segnum", verify_integrity=True).iseg

        to_reachids = {}
        for segnum, iseg in segnum_iseg.iteritems():
            sel = self.reaches.index[self.reaches.iseg == iseg]
            to_reachids.update(dict(zip(sel[0:-1], sel[1:])))
            next_segnum = self.segments.to_segnum[segnum]
            next_reachids = self.reaches.index[
                self.reaches.segnum == next_segnum]
            if len(next_reachids) > 0:
                to_reachids[sel[-1]] = next_reachids[0]

        def go_downstream(rid):
            yield rid
            if rid in to_reachids:
                yield from go_downstream(to_reachids[rid])

        con1 = list(go_downstream(start))
        try:
            # start is upstream, end is downstream
            return con1[:(con1.index(end) + 1)]
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
        # find the most upstream common reachID or "confluence"
        reachID = common.pop()
        idx1 = con1.index(reachID)
        idx2 = con2.index(reachID)
        while common:
            reachID = common.pop()
            tmp1 = con1.index(reachID)
            if tmp1 < idx1:
                idx1 = tmp1
                idx2 = con2.index(reachID)
        return con1[:idx1] + list(reversed(con2[:idx2]))
