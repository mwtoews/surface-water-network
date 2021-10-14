"""Core functionality of surface water network package."""

__all__ = ["SurfaceWaterNetwork"]

import pickle
from itertools import zip_longest
from math import sqrt
from textwrap import dedent

import geopandas
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import cascaded_union, linemerge

from swn.compat import ignore_shapely_warnings_for_object_array
from swn.spatial import get_sindex
from swn.util import abbr_str


class SurfaceWaterNetwork:
    """Surface water network class.

    Attributes
    ----------
    END_SEGNUM : int
        Special segment number that indicates a line end, default is
        usually 0. This number is not part of segments.index.
    logger : logging.Logger
        Logger to show messages.
    warnings : list
        List of warning messages.
    errors : list
        List of error messages.

    """

    def __init__(self, segments, END_SEGNUM=0, logger=None):
        """
        Initialise SurfaceWaterNetwork.

        Parameters
        ----------
        segments : geopandas.GeoDataFrame
            Primary GeoDataFrame for stream segments. Index is treated as a
            segment number or ID.
        END_SEGNUM : int
            Special segment number that indicates a line end, default is
            0. This number is not part of segments.index.
        logger : logging.Logger, optional
            Logger to show messages.

        """
        from swn.logger import get_logger, logging
        if logger is None:
            self.logger = get_logger(self.__class__.__name__)
        elif isinstance(logger, logging.Logger):
            self.logger = logger
        else:
            raise ValueError(
                f"expected 'logger' to be Logger; found {type(logger)!r}")
        self.logger.info('creating new %s object', self.__class__.__name__)
        if not isinstance(segments, geopandas.GeoDataFrame):
            raise ValueError(
                f'segments must be a GeoDataFrame; found {type(segments)!r}')
        self._segments = segments
        self.END_SEGNUM = END_SEGNUM
        if self.END_SEGNUM in self.segments.index:
            self.logger.error(
                "END_SEGNUM %r found in segments.index", self.END_SEGNUM)
        notin = ~self.to_segnums.isin(self.segments.index)
        if notin.any():
            self._segments = self._segments.copy()
            self.logger.warning(
                "correcting %d to_segnum not found in segments.index",
                notin.sum())
            self.segments.loc[notin, "to_segnum"] = self.END_SEGNUM
        # all other properties added afterwards

    def __len__(self):
        """Return number of segments."""
        return len(self._segments.index)

    def __repr__(self):
        """Return string representation of surface water network."""
        modifiers = []
        if self.has_z:
            modifiers.append('Z coordinates')
        if self.catchments is not None:
            modifiers.append('catchment polygons')
        if modifiers:
            with_modifiers = f" with {modifiers[0]}"
            if len(modifiers) == 2:
                with_modifiers += f" and {modifiers[1]}"
        else:
            with_modifiers = ''
        segments = list(self.segments.index)
        hw_l = list(self.headwater)
        out_l = list(self.outlets)
        diversions = self.diversions
        if diversions is None:
            diversions_line = 'no diversions'
        else:
            div_l = list(diversions.index)
            diversions_line = '{} diversions (as {}): {}'.format(
                len(div_l), diversions.__class__.__name__, abbr_str(div_l, 4))
        return dedent(f'''\
            <{self.__class__.__name__}:{with_modifiers}
              {len(segments)} segments: {abbr_str(segments, 4)}
              {len(hw_l)} headwater: {abbr_str(hw_l, 4)}
              {len(out_l)} outlets: {abbr_str(out_l, 4)}
              {diversions_line} />''')

    def __eq__(self, other):
        """Return true if objects are equal."""
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
                else:
                    assert av == bv
            return True
        except (AssertionError, TypeError, ValueError):
            return False

    def __iter__(self):
        """Return object datasets with an iterator."""
        yield "class", self.__class__.__name__
        yield "segments", self.segments
        yield "END_SEGNUM", self.END_SEGNUM
        yield "catchments", self.catchments
        yield "diversions", self.diversions

    def __getstate__(self):
        """Serialize object attributes for pickle dumps."""
        return dict(self)

    def __setstate__(self, state):
        """Set object attributes from pickle loads."""
        if not isinstance(state, dict):
            raise ValueError(f"expected 'dict'; found {type(state)!r}")
        elif "class" not in state:
            raise KeyError("state does not have 'class' key")
        elif state["class"] != self.__class__.__name__:
            raise ValueError("expected state class {!r}; found {!r}"
                             .format(state["class"], self.__class__.__name__))
        self.__init__(state["segments"], state["END_SEGNUM"])
        catchments = state["catchments"]
        if catchments is not None:
            self.catchments = catchments
        diversions = state["diversions"]
        if diversions is not None:
            self.set_diversions(diversions)

    @classmethod
    def from_lines(cls, lines, polygons=None):
        """
        Create and evaluate a new SurfaceWaterNetwork from lines for segments.

        Parameters
        ----------
        lines : geopandas.GeoSeries
            Input lines of surface water network. Geometries must be
            ``LINESTRING`` or ``LINESTRING Z``. Index is used for segment
            numbers. The geometry is copied to the segments property.
        polygons : geopandas.GeoSeries, optional
            Optional input polygons of surface water catchments. Geometries
            must be ``POLYGON``. Index must be the same as ``segments.index``.

        Examples
        --------
        >>> import swn
        >>> from swn.spatial import wkt_to_geoseries
        >>> lines = wkt_to_geoseries([
        ...    "LINESTRING (60 100, 60  80)",
        ...    "LINESTRING (40 130, 60 100)",
        ...    "LINESTRING (70 130, 60 100)"])
        >>> lines.index += 100
        >>> n = swn.SurfaceWaterNetwork.from_lines(lines)
        >>> n
        <SurfaceWaterNetwork:
          3 segments: [100, 101, 102]
          2 headwater: [101, 102]
          1 outlets: [100]
          no diversions />

        """
        if not isinstance(lines, geopandas.GeoSeries):
            raise ValueError('lines must be a GeoSeries')
        elif len(lines) == 0:
            raise ValueError('one or more lines are required')
        elif not (lines.geom_type == 'LineString').all():
            raise ValueError('lines must all be LineString types')
        # Create a new GeoDataFrame with a copy of line's geometry
        segments = geopandas.GeoDataFrame(geometry=lines)
        if not (polygons is None or isinstance(polygons, geopandas.GeoSeries)):
            raise ValueError('polygons must be a GeoSeries or None')
        if polygons is not None:
            if (len(polygons.index) != len(lines.index) or
                    not (polygons.index == lines.index).all()):
                raise ValueError(
                    'polygons.index is different than lines.index')
        if segments.index.min() > 0:
            END_SEGNUM = 0
        else:
            END_SEGNUM = segments.index.min() - 1
        segments['to_segnum'] = END_SEGNUM
        obj = cls(segments=segments, END_SEGNUM=END_SEGNUM)
        del segments, END_SEGNUM  # dereference local copies
        obj.errors = []
        obj.warnings = []
        obj.logger.debug("creating start/end points and spatial join objets")
        start_pts = obj.segments.interpolate(0.0, normalized=True)
        end_pts = obj.segments.interpolate(1.0, normalized=True)
        start_df = start_pts.to_frame("start").set_geometry("start")
        end_df = end_pts.to_frame("end").set_geometry("end")
        segidxname = obj.segments.index.name or "index"
        # This is the main component of the algorithm
        jxn = pd.DataFrame(
            geopandas.sjoin(end_df, start_df, "inner", "intersects")
            .drop(columns="end").reset_index()
            .rename(columns={segidxname: "end", "index_right": "start"}))
        # Group end points to start points, list should only have 1 item
        to_segnum_l = jxn.groupby("end")["start"].agg(list)
        to_segnum = to_segnum_l.apply(lambda x: x[0])
        obj.segments.loc[to_segnum.index, "to_segnum"] = to_segnum.values
        headwater = obj.headwater
        outlets = obj.outlets

        # A few checks
        sel = to_segnum_l.apply(len) > 1
        for segnum1, to_segnums in to_segnum_l.loc[sel].iteritems():
            m = ('segment %s has more than one downstream segments: %s',
                 segnum1, str(to_segnums))
            obj.logger.error(*m)
            obj.errors.append(m[0] % m[1:])
        if obj.has_z:
            # Check if match is in 2D but not 3D
            jxn["start_z"] = start_pts.loc[jxn.start].z.values
            jxn["end_z"] = end_pts.loc[jxn.end].z.values
            for r in jxn.query("start_z != end_z").itertuples():
                m = ('end of segment %s matches start of segment %s in '
                     '2D, but not in Z dimension', r.end, r.start)
                obj.logger.warning(*m)
                obj.warnings.append(m[0] % m[1:])

        obj.logger.debug(
            'checking %d headwater segments and %d outlet segments',
            len(headwater), len(outlets))
        # Find outlets that join to a single coodinate
        multi_outlets = set()
        out_pts = end_pts.loc[outlets].to_frame("out").set_geometry("out")
        jout = pd.DataFrame(
            geopandas.sjoin(out_pts, out_pts, "inner").reset_index()
            .rename(columns={segidxname: "out1", "index_right": "out2"}))\
            .query("out1 != out2")
        if jout.size > 0:
            # Just evaluate 2D tuple to find segnums with same location
            outsets = jout.assign(xy=jout.out.map(lambda g: (g.x, g.y)))\
                .drop(columns="out").groupby("xy").agg(set)
            for r in outsets.itertuples():
                v = r.out1
                m = ("ending coordinate %s matches end segment%s: %s",
                     r.Index, "s" if len(v) != 1 else "", v)
                obj.logger.warning(*m)
                obj.warnings.append(m[0] % m[1:])
                multi_outlets |= v
        # Find outlets that join to middle of other segments
        joutseg = pd.DataFrame(
            geopandas.sjoin(out_pts, obj.segments[["geometry"]], "inner")
            .drop(columns="out").reset_index()
            .rename(columns={segidxname: "out", "index_right": "segnum"}))
        for r in joutseg.query("out != segnum").itertuples():
            if r.out in multi_outlets:
                continue
            m = ('segment %s connects to the middle of segment %s',
                 r.out, r.segnum)
            obj.logger.error(*m)
            obj.errors.append(m[0] % m[1:])
        # Find headwater that join to a single coodinate
        hw_pts = start_pts.loc[headwater].to_frame("hw").set_geometry("hw")
        jhw = pd.DataFrame(
            geopandas.sjoin(hw_pts, start_df, "inner").reset_index()
            .rename(columns={segidxname: "hw1", "index_right": "start"}))\
            .query("hw1 != start")
        obj.jhw = jhw
        if jhw.size > 0:
            hwsets = jhw.assign(xy=jhw.hw.map(lambda g: (g.x, g.y)))\
                .drop(columns="hw").groupby("xy").agg(set)
            for r in hwsets.itertuples():
                v = r.start
                m = ("starting coordinate %s matches start segment%s: %s",
                     r.Index, "s" if len(v) != 1 else "", v)
                obj.logger.warning(*m)
                obj.errors.append(m[0] % m[1:])

        # Store from_segnums set to segments GeoDataFrame
        from_segnums = obj.from_segnums
        obj.segments["from_segnums"] = from_segnums
        sel = obj.segments.from_segnums.isna()
        if sel.any():
            obj.segments.loc[sel, "from_segnums"] = \
                [set() for _ in range(sel.sum())]
        obj.logger.debug('evaluating segments upstream from %d outlet%s',
                         len(outlets), 's' if len(outlets) != 1 else '')
        obj.segments['cat_group'] = obj.END_SEGNUM
        obj.segments['num_to_outlet'] = 0
        obj.segments['dist_to_outlet'] = 0.0

        # Recursive function that accumulates information upstream
        def resurse_upstream(segnum, cat_group, num, dist):
            obj.segments.at[segnum, 'cat_group'] = cat_group
            num += 1
            obj.segments.at[segnum, 'num_to_outlet'] = num
            dist += obj.segments.geometry[segnum].length
            obj.segments.at[segnum, 'dist_to_outlet'] = dist
            # Branch to zero or more upstream segments
            for from_segnum in from_segnums.get(segnum, []):
                resurse_upstream(from_segnum, cat_group, num, dist)

        for segnum in obj.segments.loc[outlets].index:
            resurse_upstream(segnum, segnum, 0, 0.0)

        obj.logger.debug('evaluating downstream sequence')
        obj.segments['sequence'] = 0
        obj.segments['stream_order'] = 0
        # obj.segments['numiter'] = 0  # should be same as stream_order
        # Sort headwater segments from the furthest from outlet to closest
        search_order = ['num_to_outlet', 'dist_to_outlet']
        furthest_upstream = obj.segments.loc[headwater]\
            .sort_values(search_order, ascending=False).index
        sequence = pd.Series(
            np.arange(len(furthest_upstream)) + 1, index=furthest_upstream)
        obj.segments.loc[sequence.index, 'sequence'] = sequence
        obj.segments.loc[sequence.index, 'stream_order'] = 1
        completed = set(headwater)
        sequence = int(sequence.max())
        for numiter in range(1, obj.segments['num_to_outlet'].max() + 1):
            # Gather segments downstream from completed upstream set
            downstream = set(
                obj.segments.loc[completed, 'to_segnum'])\
                .difference(completed.union([obj.END_SEGNUM]))
            # Sort them to evaluate the furthest first
            downstream_sorted = obj.segments.loc[downstream]\
                .sort_values(search_order, ascending=False).index
            for segnum in downstream_sorted:
                if from_segnums[segnum].issubset(completed):
                    sequence += 1
                    obj.segments.at[segnum, 'sequence'] = sequence
                    # self.segments.at[segnum, 'numiter'] = numiter
                    # See Strahler stream order definition
                    up_ord = list(
                        obj.segments.loc[
                            list(from_segnums[segnum]), 'stream_order'])
                    max_ord = max(up_ord)
                    if up_ord.count(max_ord) > 1:
                        obj.segments.at[segnum, 'stream_order'] = max_ord + 1
                    else:
                        obj.segments.at[segnum, 'stream_order'] = max_ord
                    completed.add(segnum)
            if obj.segments['sequence'].min() > 0:
                break
        obj.logger.debug('sequence evaluated with %d iterations', numiter)
        # Don't do this: self.segments.sort_values('sequence', inplace=True)
        obj.evaluate_upstream_length()
        if polygons is not None:
            obj.catchments = polygons
        return obj

    @property
    def segments(self):
        """GeoDataFrame of stream segments derived from the lines input.

        This GeoDataFrame is created by
        :py:meth:`swn.SurfaceWaterNetwork.from_lines`.

        Attributes
        ----------
        index : any
            Unique index for each segment, with an optional name attribute.
            Copied from ``lines.index``. This is often defined externally,
            and is used to relate and exchange stream information.
        geometry : geometry
            LineString or LineStringZ geometries, copied from ``lines``.
        to_segnum : same type as index
            Index to which segment connects downstream to. If this is an
            outlet, the value is ``END_SEGNUM``.
        from_segnums : set
            A set of zero or more indexes from which a segment connects
            upstream from. This will be an empty set (pandas displays this
            as ``{}`` rather than ``set()``), then it is a headwater segment.
        cat_group : same type as index
            Catchment group for each outlet.
        num_to_outlet : int
            Number of segments to the outlet.
        dist_to_outlet : float
            Distance to outlet.
        sequence : int
            Unique downstream sequence.
        stream_order : int
            Strahler number.
        upstream_length : float
            Total lengths of streams upstream. See
            :py:meth:`swn.SurfaceWaterNetwork.evaluate_upstream_width`
            for details.
        upstream_area : float
            If catchments are defined, this is the upstream catchment area.
            See :py:meth:`swn.SurfaceWaterNetwork.evaluate_upstream_area`
            for details.
        width : float
            If upstream_area is defined, this is an estimate of stream width.
            See :py:meth:`swn.SurfaceWaterNetwork.estimate_width` for details.
        diversions : set
            If diversions are defined, this is a set of zero or more indexes
            to which the segment connects to.

        Examples
        --------
        >>> import swn
        >>> from swn.spatial import wkt_to_geoseries
        >>> lines = wkt_to_geoseries([
        ...    "LINESTRING (60 100, 60  80)",
        ...    "LINESTRING (40 130, 60 100)",
        ...    "LINESTRING (70 130, 60 100)"])
        >>> lines.index += 100
        >>> n = swn.SurfaceWaterNetwork.from_lines(lines)
        >>> n.segments.columns
        Index(['geometry', 'to_segnum', 'from_segnums', 'cat_group', 'num_to_outlet',
               'dist_to_outlet', 'sequence', 'stream_order', 'upstream_length'],
              dtype='object')
        >>> n.segments[["to_segnum", "from_segnums", "cat_group", "num_to_outlet",
        ...             "sequence", "stream_order"]]
             to_segnum from_segnums  cat_group  num_to_outlet  sequence  stream_order
        100          0   {101, 102}        100              1         3             2
        101        100           {}        100              2         1             1
        102        100           {}        100              2         2             1
        """  # noqa
        return getattr(self, '_segments')

    @property
    def catchments(self):
        """Polygon GeoSeries of surface water catchments.

        Catchment polygons are optional, and are set with a GeoSeries with
        a matching index to ``segments.index``.

        To unset this property, use ``n.catchments = None``.

        Examples
        --------
        >>> import swn
        >>> from swn.spatial import wkt_to_geoseries
        >>> lines = wkt_to_geoseries([
        ...    "LINESTRING (60 100, 60  80)",
        ...    "LINESTRING (40 130, 60 100)",
        ...    "LINESTRING (70 130, 60 100)"])
        >>> lines.index += 100
        >>> polygons = wkt_to_geoseries([
        ...    "POLYGON ((35 100, 75 100, 75  80, 35  80, 35 100))",
        ...    "POLYGON ((35 135, 60 135, 60 100, 35 100, 35 135))",
        ...    "POLYGON ((60 135, 75 135, 75 100, 60 100, 60 135))"])
        >>> polygons.index += 100
        >>> n = swn.SurfaceWaterNetwork.from_lines(lines, polygons)
        >>> n
        <SurfaceWaterNetwork: with catchment polygons
          3 segments: [100, 101, 102]
          2 headwater: [101, 102]
          1 outlets: [100]
          no diversions />
        >>> n.segments[["to_segnum", "from_segnums", "upstream_area", "width"]]
             to_segnum from_segnums  upstream_area     width
        100          0   {101, 102}         2200.0  1.461501
        101        100           {}          875.0  1.445695
        102        100           {}          525.0  1.439701

        """
        return getattr(self, '_catchments', None)

    @catchments.setter
    def catchments(self, value):
        if value is None:
            if hasattr(self, '_catchments'):
                delattr(self, '_catchments')
            return
        elif not isinstance(value, geopandas.GeoSeries):
            raise ValueError(
                f'catchments must be a GeoSeries or None; found {type(value)!r}')
        segments_index = self.segments.index
        if (len(value.index) != len(segments_index) or
                not (value.index == segments_index).all()):
            raise ValueError(
                'catchments.index is different than for segments')
        # TODO: check extent overlaps
        setting_new = self.catchments is None
        self._catchments = value
        if setting_new:
            self.evaluate_upstream_area()
            self.estimate_width()

    @property
    def diversions(self):
        """[Geo]DataFrame of surface water diversions.

        Use :py:meth:`swn.SurfaceWaterNetwork.set_diversions` to set this
        property.

        Attributes
        ----------
        index : any
            Unique index for each diversion, with an optional name attribute.
            Copied from ``lines.index``. This is often defined externally,
            and is used to relate and exchange stream information.
        geometry : geometry
            If a GeoDataFrame was used to define diversions, this is a copy
            of the GeoSeries.
        from_segnum : same type as segments.index
            Index of upstream segment from which diversion is connected to.
        dist_end : float
            Distance to most-downstream-end of segment described by
            ``from_segnum``.
        dist_end : float
            Distance to segment line described by ``from_segnum``.
        """
        return getattr(self, '_diversions', None)

    @diversions.setter
    def diversions(self, value):
        raise AttributeError("use 'set_diversions()' method")

    def set_diversions(self, diversions, min_stream_order=None):
        """Set surface water diversion locations.

        This method checks or assigns a segment number to divert surface water
        flow from, and adds other columns to the diversions and segments data
        frames.

        If a ``from_segnum`` column exists, these values are checked against
        the index for segments, and adds/updates ``dist`` (where possible) to
        describe the distance from the diversion to the end of the segment.

        If ``from_segnum`` is not provided and a GeoDataFrame is provided, then
        the closest segment end is identified, using optional min_stream_order
        preferentially select higher-order stream segments.

        Parameters
        ----------
        diversions : pandas.DataFrame, geopandas.GeoDataFrame or None
            Data frame of surface water diversions, a modified copy kept as a
            ``diversions`` property. Use None to unset this property.
        min_stream_order : int, default None
            Finds stream segments with a minimum stream order.

        """
        if diversions is None:
            self._diversions = None
            if 'diversions' in self.segments:
                self.segments.drop('diversions', axis=1, inplace=True)
            return

        if not isinstance(diversions, (geopandas.GeoDataFrame, pd.DataFrame)):
            raise ValueError('a [Geo]DataFrame is expected')

        is_spatial = (
            isinstance(diversions, geopandas.GeoDataFrame) and
            'geometry' in diversions.columns and
            (~diversions.is_empty).all())

        if is_spatial:
            # Make sure CRS is the same as segments (if defined)
            diversions_crs = getattr(diversions, 'crs', None)
            segments_crs = getattr(self.segments, 'crs', None)
            if diversions_crs != segments_crs:
                self.logger.warning(
                    'CRS for diversions and segments are different: '
                    '%s vs. %s', diversions_crs, segments_crs)
        if 'from_segnum' in diversions.columns:
            self.logger.debug(
                "checking existing 'from_segnum' column for diversions")
            sn = set(self.segments.index)
            dn = set(diversions['from_segnum'])
            if not sn.issuperset(dn):
                diff = dn.difference(sn)
                raise ValueError(
                    f"{len(diff)} 'from_segnum' are not found in "
                    f"segments.index: {abbr_str(diff)}")
            self._diversions = diversions.copy()
            if is_spatial:
                cols = ["dist_end", "dist_line"]
                self._diversions[cols] = np.nan
                geom_name = self._diversions.geometry.name
                for item in self._diversions.itertuples():
                    div_geom = getattr(item, geom_name)
                    seg_geom = self.segments.geometry[item.from_segnum]
                    dist_end = div_geom.distance(Point(seg_geom.coords[-1]))
                    dist_line = div_geom.distance(seg_geom)
                    self._diversions.loc[item.Index, cols] = \
                        [dist_end, dist_line]
        elif is_spatial:
            self.logger.debug(
                'assigning columns for diversions based on spatial distances')
            self._diversions = diversions.copy()
            self._diversions['from_segnum'] = self.END_SEGNUM
            self._diversions['dist_end'] = np.nan
            self._diversions['dist_line'] = np.nan
            if min_stream_order is not None:
                sel = self.segments['stream_order'] >= min_stream_order
                if not sel.any():
                    raise ValueError(
                        "'from_segnum' value too high (nothing selected)")
                seg_lines = self.segments.loc[sel, 'geometry']
            else:
                seg_lines = self.segments.geometry
            seg_lines_sindex = get_sindex(seg_lines)
            seg_ends = seg_lines.apply(lambda g: Point(g.coords[-1]))
            # a point that is 0.1 m upstream from end
            seg_near_ends = seg_lines.interpolate(-0.1, False)
            for idx, geom in self._diversions.geometry.iteritems():
                # build table of distances to nearest lines and end points

                if seg_lines_sindex:
                    sel = list(seg_lines_sindex.nearest(
                                geom.coords[0], num_results=8))
                    dists = pd.DataFrame(
                        {
                            'dist_end': seg_ends.iloc[sel].distance(geom),
                            'dist_line': seg_lines.iloc[sel].distance(geom),
                            'seg_near_ends':
                                seg_near_ends.iloc[sel].distance(geom),
                        },
                        index=seg_lines.iloc[sel].index).sort_values(
                            ['dist_end', 'dist_line', 'seg_near_ends'])
                else:  # slower processing with of all seg_lines
                    dists = pd.DataFrame(
                        {
                            'dist_end': seg_ends.distance(geom),
                            'dist_line': seg_lines.distance(geom),
                            'seg_near_ends': seg_near_ends.distance(geom),
                        },
                        index=seg_lines.index).sort_values(
                            ['dist_end', 'dist_line', 'seg_near_ends'])

                # assign closest segnum
                self._diversions.loc[
                    idx, ['from_segnum', 'dist_end', 'dist_line']] = \
                    [dists.index[0]] + \
                    list(dists.iloc[0][['dist_end', 'dist_line']])
        else:
            raise ValueError(
                "'gdf' does not appear to be spatial or have a 'from_segnum' "
                "column")
        # Update segments column to mirror this information
        self.segments['diversions'] = None
        for segnum in self._diversions['from_segnum'].unique():
            sel = self._diversions['from_segnum'] == segnum
            self.segments.at[segnum, 'diversions'] = \
                set(self._diversions.loc[sel].index)
        sel = self.segments.diversions.isna()
        if sel.any():
            self.segments.loc[sel, "diversions"] = \
                [set() for _ in range(sel.sum())]

    @property
    def has_z(self):
        """Return True if all segment lines have Z dimension."""
        return bool(self.segments.geometry.apply(lambda x: x.has_z).all())

    @property
    def headwater(self):
        """Return index of headwater segments."""
        return self.segments.index[
                ~self.segments.index.isin(self.segments['to_segnum'])]

    @property
    def outlets(self):
        """Return index of outlets.

        Determined where ``n.segments.to_segnum == n.END_SEGNUM`
        """
        return self.segments.index[
                self.segments['to_segnum'] == self.END_SEGNUM]

    @property
    def to_segnums(self):
        """Return Series of segnum to connect downstream.

        Determined from ``n.segments.to_segnum``.
        """
        return self.segments.loc[
            self.segments['to_segnum'] != self.END_SEGNUM, 'to_segnum']

    @property
    def from_segnums(self):
        """Return partial Series of a set of segnums to connect upstream."""
        series = self.to_segnums.to_frame(0).reset_index()\
            .groupby(0).aggregate(set).iloc[:, 0]
        series.index.name = self.segments.index.name
        series.name = "from_segnums"
        return series

    def query(self, upstream=[], downstream=[], barrier=[],
              gather_upstream=False):
        """Return segnums upstream (inclusive) and downstream (exclusive).

        Parameters
        ----------
        upstream, downstream : int or list, default []
            Segment number(s) from segments.index to search from.
        barriers : int or list, default []
            Segment number(s) that cannot be traversed past.
        gather_upstream : bool, default False
            Gather upstream from all other downstream segments.

        Returns
        -------
        list

        """
        segments_index = self.segments.index
        segments_set = set(segments_index)

        def check_and_return_list(var, name):
            if isinstance(var, list):
                if not segments_set.issuperset(var):
                    diff = list(sorted(set(var).difference(segments_set)))
                    raise IndexError(
                        f"{len(diff)} {name} "
                        f"segment{'' if len(diff) == 1 else 's'} "
                        f"not found in segments.index: {abbr_str(diff)}")
                return var
            else:
                if var not in segments_index:
                    raise IndexError(
                        f'{name} segnum {var} not found in segments.index')
                return [var]

        def go_upstream(segnum):
            yield segnum
            for from_segnum in from_segnums.get(segnum, []):
                yield from go_upstream(from_segnum)

        def go_downstream(segnum):
            yield segnum
            if segnum in to_segnums:
                yield from go_downstream(to_segnums[segnum])

        to_segnums = dict(self.to_segnums)
        from_segnums = self.from_segnums
        for barrier in check_and_return_list(barrier, 'barrier'):
            try:
                del from_segnums[barrier]
            except KeyError:  # this is a tributary, remove value
                from_segnums[to_segnums[barrier]].remove(barrier)
            del to_segnums[barrier]

        segnums = []
        for segnum in check_and_return_list(upstream, 'upstream'):
            upsegnums = list(go_upstream(segnum))
            segnums += upsegnums  # segnum inclusive
        for segnum in check_and_return_list(downstream, 'downstream'):
            downsegnums = list(go_downstream(segnum))
            segnums += downsegnums[1:]  # segnum exclusive
            if gather_upstream:
                for segnum in downsegnums[1:]:
                    for from_segnum in from_segnums.get(segnum, []):
                        if from_segnum not in downsegnums:
                            upsegnums = list(go_upstream(from_segnum))
                            segnums += upsegnums
        return segnums

    def aggregate(self, segnums, follow_up='upstream_length'):
        """Aggregate segments (and catchments) to a coarser network of segnums.

        Parameters
        ----------
        segnums : list
            List of segment numbers to aggregate. Must be unique.
        follow_up : str
            Column name in segments used to determine the descending sort
            order used to determine which segment to follow up headwater
            catchments. Default 'upstream_length'. Another good candidate
            is 'upstream_area', if catchment polygons are available.

        Returns
        -------
        SurfaceWaterNetwork
            Columns 'agg_patch' and 'agg_path' are added to segments to
            provide a segnum list from the original surface water network
            to the aggregated object. Column 'agg_unpath' lists other
            segnums that flow into 'agg_path'. Also 'from_segnums' is updated
            to reflect the uppermost segment.

        """
        if (isinstance(follow_up, str) and follow_up in self.segments.columns):
            pass
        else:
            raise ValueError(
                f'{follow_up!r} not found in segments.columns')
        junctions = list(segnums)
        junctions_set = set(junctions)
        if len(junctions) != len(junctions_set):
            raise ValueError('list of segnums is not unique')
        segments_index_set = set(self.segments.index)
        if not junctions_set.issubset(segments_index_set):
            diff = junctions_set.difference(segments_index_set)
            raise IndexError(
                f"{len(diff)} segnums not found in "
                f"segments.index: {abbr_str(diff)}")
        self.logger.debug(
            'aggregating at least %d segnums (junctions)', len(junctions))
        from_segnums = self.from_segnums
        to_segnums = dict(self.to_segnums)

        # trace down from each segnum to the outlet - keep this step simple
        traced_segnums = list()

        def trace_down(segnum):
            if segnum is not None and segnum not in traced_segnums:
                traced_segnums.append(segnum)
                trace_down(to_segnums.get(segnum, None))

        for segnum in junctions:
            trace_down(segnum)

        self.logger.debug(
            'traced down initial junctions to assemble %d traced segnums: %s',
            len(traced_segnums), abbr_str(traced_segnums))

        # trace up from each junction, add extra junctions as needed
        traset = set(traced_segnums)
        extra_junctions = []

        def trace_up(segnum):
            if segnum not in from_segnums:
                return
            up_segnums = from_segnums[segnum]
            segsij = up_segnums.intersection(junctions_set)
            segsit = up_segnums.intersection(traset)
            if len(segsij) > 0:
                segsdj = up_segnums.difference(junctions_set)
                segsdjdt = segsdj.difference(segsit)
                for up_segnum in sorted(segsdjdt):
                    self.logger.debug(
                        'adding extra junction %s above segnum %s',
                        up_segnum, segnum)
                    extra_junctions.append(up_segnum)
                    # but don't follow up these, as it's untraced
            elif len(segsit) > 1:
                for up_segnum in sorted(segsit):
                    self.logger.debug(
                        'adding extra junction %s above fork at segnum %s',
                        up_segnum, segnum)
                    extra_junctions.append(up_segnum)
                    trace_up(up_segnum)
            elif len(segsit) == 1:
                trace_up(segsit.pop())

        for segnum in junctions:
            trace_up(segnum)

        if len(extra_junctions) == 0:
            self.logger.debug('traced up; no extra junctions added')
        else:
            junctions += extra_junctions
            junctions_set = set(junctions)
            if len(extra_junctions) == 1:
                self.logger.debug(
                    'traced up; added 1 extra junction: %s', extra_junctions)
            else:
                self.logger.debug(
                    'traced up; added %d extra junctions: %s',
                    len(extra_junctions), abbr_str(extra_junctions))

        # aggregate segnums above each junction, build larger polygons
        def up_patch_segnums(segnum):
            yield segnum
            for up_segnum in from_segnums.get(segnum, []):
                if up_segnum not in junctions_set:
                    yield from up_patch_segnums(up_segnum)

        trbset = traset.difference(junctions_set)

        # aggregate segnums in a path across an internal subcatchment
        def up_path_internal_segnums(segnum):
            yield segnum
            up_segnums = from_segnums.get(segnum, set()).intersection(trbset)
            if len(up_segnums) == 1:
                yield from up_path_internal_segnums(up_segnums.pop())

        # aggregate segnums in a path up a headwater, choosing untraced path
        def up_path_headwater_segnums(segnum):
            yield segnum
            up_segnums = from_segnums.get(segnum, set())
            if len(up_segnums) == 1:
                yield from up_path_headwater_segnums(up_segnums.pop())
            elif len(up_segnums) > 1:
                up_segnum = self.segments.loc[up_segnums].sort_values(
                    follow_up, ascending=False).index[0]
                # self.logger.debug('untraced path %s: %s -> %s',
                #                   segnum, up_segnums, up_segnum)
                yield from up_path_headwater_segnums(up_segnum)

        junctions_goto = {s: to_segnums.get(s, None) for s in junctions}
        agg_patch = pd.Series(dtype=object)
        agg_path = pd.Series(dtype=object)
        agg_unpath = pd.Series(dtype=object)
        if self.catchments is not None:
            polygons = pd.Series(dtype=object)
        else:
            polygons = None
        lines = pd.Series(dtype=object)

        for segnum in junctions:
            catchment_segnums = list(up_patch_segnums(segnum))
            agg_patch.at[segnum] = catchment_segnums
            if polygons is not None:
                polygons.at[segnum] = cascaded_union(
                        list(self.catchments.loc[catchment_segnums]))
            # determine if headwater or not, if any other junctions go to it
            is_headwater = True
            for key, value in junctions_goto.items():
                if value in catchment_segnums:
                    is_headwater = False
                    break
            if is_headwater:
                agg_path_l = list(up_path_headwater_segnums(segnum))
            else:  # internal
                agg_path_l = list(up_path_internal_segnums(segnum))
            agg_path.at[segnum] = agg_path_l
            # gather unfollowed paths, e.g. to accumulate flow
            agg_path_s = set(agg_path_l)
            agg_unpath_l = []
            for aseg in agg_path_l:
                agg_unpath_l += sorted(
                        from_segnums.get(aseg, set()).difference(agg_path_s))
            agg_unpath.at[segnum] = agg_unpath_l
            # agg_path_l.reverse()
            seg_geom = linemerge(list(
                self.segments.loc[reversed(agg_path_l), 'geometry']))
            with ignore_shapely_warnings_for_object_array():
                lines.at[segnum] = seg_geom

        # Create GeoSeries and copy a few other properties
        lines = geopandas.GeoSeries(lines, crs=self.segments.crs)
        lines.index.name = self.segments.index.name
        if polygons is not None:
            polygons = geopandas.GeoSeries(polygons, crs=self.catchments.crs)
            polygons.index.name = self.catchments.index.name
            txt = ' and polygons'
        else:
            txt = ''
        self.logger.debug('aggregated %d lines%s', len(junctions), txt)

        na = SurfaceWaterNetwork.from_lines(lines, polygons)
        na.segments['agg_patch'] = agg_patch
        na.segments['agg_path'] = agg_path
        na.segments['agg_unpath'] = agg_unpath
        return na

    def accumulate_values(self, values):
        """Accumulate values down the stream network.

        For example, calculate cumulative upstream catchment area for each
        segment.

        Parameters
        ----------
        values : pandas.Series
            Series of values that align with the index.

        Returns
        -------
        pandas.Series
            Accumulated values.

        """
        if not isinstance(values, pd.Series):
            raise ValueError('values must be a pandas Series')
        elif (len(values.index) != len(self.segments.index) or
                not (values.index == self.segments.index).all()):
            raise ValueError('index is different')
        accum = values.copy()
        if accum.name is not None:
            accum.name = f"accumulated_{accum.name}"
        segnumset = set(self.segments.index)
        for segnum in self.segments.sort_values('sequence').index:
            from_segnums = self.segments.at[segnum, 'from_segnums']
            if from_segnums:
                from_segnums = from_segnums.intersection(segnumset)
                if from_segnums:
                    accum[segnum] += sum(accum[s] for s in from_segnums)
        return accum

    def evaluate_upstream_length(self):
        """Evaluate upstream length of segments, adds to segments."""
        self.logger.debug('evaluating upstream length')
        self.segments['upstream_length'] = \
            self.accumulate_values(self.segments.length)

    def evaluate_upstream_area(self):
        """Evaluate upstream area from catchments, adds to segments."""
        self.logger.debug('evaluating upstream area')
        if self.catchments is None:
            raise ValueError("can't evaluate upstream area without catchments")
        self.segments['upstream_area'] = \
            self.accumulate_values(self.catchments.area)

    def estimate_width(self, a=1.42, b=0.52, upstream_area='upstream_area'):
        """Estimate stream width based on upstream area and fitting parameters.

        The column 'upstream_area' (in m^2) must exist in segments, as
        automatically calculated if provided catchments (polygons), or
        appended manually.

        Stream width is based on the formula:

            width = a + (upstream_area / 1e6) ** b

        Parameters
        ----------
        a, b : float or pandas.Series, optional
            Fitting parameters, with defaults a=1.42, and b=0.52
        upstream_area : str or pandas.Series
            Column name in segments (default ``upstream_area``) to use for
            upstream area (in m^2), or series of upstream areas.

        Returns
        -------
        None

        """
        self.logger.debug('evaluating width')
        if isinstance(upstream_area, pd.Series):
            upstream_area_km2 = upstream_area / 1e6
        elif isinstance(upstream_area, str):
            if upstream_area not in self.segments.columns:
                raise ValueError(
                    f'{upstream_area!r} not found in segments.columns')
            upstream_area_km2 = self.segments[upstream_area] / 1e6
        else:
            raise ValueError('unknown use for upstream_area')
        if not isinstance(a, (float, int)):
            a = self.segments_series(a, "a")
        if not isinstance(b, (float, int)):
            b = self.segments_series(a, "b")
        self.segments['width'] = a + upstream_area_km2 ** b

    def segments_series(self, value, name=None):
        """Generate a Series along the segments index.

        Parameters
        ----------
        value : scalar, list, dict or pandas.Series
            If value is a Series, it is checked to ensure it is has the same
            index as :py:attr:`SurfaceWaterNetwork.segments`. Otherwise value
            as a scalar, list or dict is cast as a Series with
            ``segments.index``.
        name : str, default None
            Name used for series, if provided.

        Returns
        -------
        pandas.Series

        Examples
        --------
        >>> import swn
        >>> from swn.spatial import wkt_to_geoseries
        >>> lines = wkt_to_geoseries([
        ...    "LINESTRING (60 100, 60  80)",
        ...    "LINESTRING (40 130, 60 100)",
        ...    "LINESTRING (70 130, 60 100)"])
        >>> n = swn.SurfaceWaterNetwork.from_lines(lines)
        >>> n.segments_series(1.2)
        0    1.2
        1    1.2
        2    1.2
        dtype: float64
        >>> n.segments_series([3, 2, 1], "codes")
        0    3
        1    2
        2    1
        Name: codes, dtype: int64
        """
        segments_index = self.segments.index
        if np.isscalar(value):
            value = pd.Series(value, index=segments_index)
        elif isinstance(value, pd.Series):
            pass
        elif isinstance(value, (list, dict)):
            value = pd.Series(value)
        else:
            raise ValueError(
                "expected value to be scalar, list, dict or Series")
        if (len(value.index) != len(segments_index) or
                not (value.index == segments_index).all()):
            raise ValueError('index is different than for segments')
        if name is not None:
            value.name = name
        return value

    def pair_segments_frame(
            self, value, value_out=None, name=None, method="continuous"):
        """Generate a DataFrame of paired values for top/bottom of segments.

        The first value applies to the top of each segment, and the bottom
        value is determined from the top of the value it connects to.

        Parameters
        ----------
        value : scalar, list, dict or pandas.Series
            Value to assign to the upstream or top end of each segment.
            If value is a Series, it is checked to ensure it is has the same
            index as :py:attr:`SurfaceWaterNetwork.segments`. Otherwise value
            as a scalar, list or dict is cast as a Series with
            ``segments.index``.
        value_out : None (default), scalar, dict or pandas.Series
            If None (default), the value used for the bottom is determined
            using ``method``. Otherwise, ``value_out`` can be directly
            specified with a Series or dict, indexed by segnum. If
            ``value_out`` is a scalar, it is cast as a Series with
            ``outlets.index``. This option is normally specified for outlets,
            but will overwrite any other values determined by ``method``.
        name : str, default None
            Base name used for the column names, if provided. For example,
            ``name="foo"`` will create columns "foo1" and "foo2".
        method : str, default "continuous"
            This option determines how ``value_out`` values should be
            determined, if not specified. Choose one of:
              - ``continuous`` (default): downstream value is evaluated to be
                the same as the upstream value it connects to. This allows a
                continuous network of values along the networks, such as
                elevation.
              - ``constant`` : ``value_out`` is the same as ``value``.
              - ``additive`` : downstream value is evaluated to be a fraction
                of tributaries that add to the upstream value it connects to.
                Proportions of values for each tributary are preserved, but
                lengths of segments are ignored.

        Returns
        -------
        pandas.DataFrame
            Resulting DataFrame has two columns for top (1) and bottom (2) of
            each segment.

        Examples
        --------
        >>> import swn
        >>> from swn.spatial import wkt_to_geoseries
        >>> lines = wkt_to_geoseries([
        ...    "LINESTRING (60 100, 60  80)",
        ...    "LINESTRING (40 130, 60 100)",
        ...    "LINESTRING (70 130, 60 100)"])
        >>> n = swn.SurfaceWaterNetwork.from_lines(lines)
        >>> n.pair_segments_frame([3, 4, 5])
           1  2
        0  3  3
        1  4  3
        2  5  3
        >>> n.pair_segments_frame([4.0, 3.0, 5.0], {0: 6.0}, name="var")
           var1  var2
        0   4.0   6.0
        1   3.0   4.0
        2   5.0   4.0
        >>> n.pair_segments_frame([10.0, 2.0, 3.0], name="add",
        ...                       method="additive")
           add1  add2
        0  10.0  10.0
        1   2.0   4.0
        2   3.0   6.0
        """
        supported_methods = ["continuous", "constant", "additive"]
        if method not in supported_methods:
            raise ValueError(f"method must be one of {supported_methods}")
        value = self.segments_series(value, name=name)
        df = pd.concat([value, value], axis=1)
        if value.name is not None:
            df.columns = df.columns.str.cat(["1", "2"])
        else:
            df.columns += 1
        c1, c2 = df.columns
        if method == "continuous":
            to_segnums = self.to_segnums
            df.loc[to_segnums.index, c2] = df.loc[to_segnums, c1].values
        elif method == "additive":
            for segnum, from_segnums in self.from_segnums.iteritems():
                if len(from_segnums) <= 1:
                    continue
                # get proportions of upstream values
                from_value1 = df.loc[sorted(from_segnums), c1]
                from_value1_prop = from_value1 / from_value1.sum()
                from_value2 = df.loc[segnum, c1] * from_value1_prop
                for from_segnum, v2 in from_value2.iteritems():
                    df.loc[from_segnum, c2] = v2
        elif method == "constant":
            pass
        if value_out is not None:
            if np.isscalar(value_out):
                # generate only for outlets
                value_out = pd.Series(value_out, index=self.outlets)
            elif isinstance(value_out, pd.Series):
                pass
            elif isinstance(value_out, dict):
                value_out = pd.Series(value_out)
            else:
                raise ValueError(
                    "expected value_out to be scalar, dict or Series")
            if not set(value_out.index).issubset(set(df.index)):
                raise ValueError(
                    "value_out.index is not a subset of segments.index")
            df.loc[value_out.index, c2] = value_out
        return df

    def adjust_elevation_profile(self, min_slope=1./1000):
        """Check and adjust (if necessary) Z coordinates of elevation profiles.

        Parameters
        ----------
        min_slope : float or pandas.Series, optional
            Minimum downwards slope imposed on segments. If float, then this is
            a global value, otherwise it is per-segment with a Series.
            Default 1./1000 (or 0.001).

        """
        min_slope = self.segments_series(min_slope)
        if (min_slope <= 0.0).any():
            raise ValueError('min_slope must be greater than zero')
        elif not self.has_z:
            raise AttributeError('line geometry does not have Z dimension')

        geom_name = self.segments.geometry.name
        from_segnums = self.from_segnums
        to_segnums = dict(self.to_segnums)
        modified_d = {}  # key is segnum, value is drop amount (+ve is down)
        self.messages = []

        # Build elevation profile list storing three values per coordinate:
        #   [dx, elev], where dx is 2D distance from upstream coord
        #   elev is the adjusted elevation
        profile_d = {}  # key is segnum, value is list of profile tuples
        for segnum, geom in self.segments.geometry.iteritems():
            coords = geom.coords[:]  # coordinates
            x0, y0, z0 = coords[0]  # upstream coordinate
            profile = [[0.0, z0]]
            for idx, (x1, y1, z1) in enumerate(coords[1:], 1):
                dx = sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
                profile.append([dx, z1])
                x0, y0, z0 = x1, y1, z1
            profile_d[segnum] = profile
        # Process elevation adjustments in flow sequence
        for segnum in self.segments.sort_values('sequence').index:
            if segnum not in modified_d:
                modified_d[segnum] = []
            profile = profile_d[segnum]
            # Get upstream last coordinates, and check if drop0 needs updating
            dx, z0 = profile[0]
            if segnum in from_segnums:
                last_zs = [profile_d[n][-1][1] for n in from_segnums[segnum]]
                if last_zs:
                    last_zs_min = min(last_zs)
                    if last_zs_min < z0:
                        drop = z0 - last_zs_min
                        z0 -= drop
                        profile[0][1] = z0
                        modified_d[segnum].append(drop)
                    elif min(last_zs) > z0:
                        raise NotImplementedError('unexpected scenario')
            # Check and enforce minimum slope for remaining coords
            for idx, (dx, z1) in enumerate(profile[1:], 1):
                dz = z0 - z1
                slope = dz / dx
                if slope < min_slope[segnum]:
                    drop = z1 - z0 + dx * min_slope[segnum]
                    z1 -= drop
                    profile[idx][1] = z1
                    modified_d[segnum].append(drop)
                    # print('adj', z0 + drop0, dx * min_slope[segnum], drop)
                z0 = z1
            # Ensure last coordinate matches other segments that end here
            if segnum in to_segnums:
                beside_segnums = from_segnums[to_segnums[segnum]]
                if beside_segnums:
                    last_zs = [profile_d[n][-1][1] for n in beside_segnums]
                    last_zs_min = min(last_zs)
                    if max(last_zs) != last_zs_min:
                        for bsegnum in beside_segnums:
                            if profile_d[bsegnum][-1][1] != last_zs_min:
                                drop = profile_d[bsegnum][-1][1] - last_zs_min
                                profile_d[bsegnum][-1][1] -= drop
                                if bsegnum not in modified_d:
                                    modified_d[bsegnum] = [drop]
                                elif len(modified_d[bsegnum]) == 0:
                                    modified_d[bsegnum].append(drop)
                                else:
                                    modified_d[bsegnum][-1] += drop
        # Adjust geometries and report some information on adjustments
        profiles = []
        for segnum in self.segments.index:
            profile = profile_d[segnum]
            modified = modified_d[segnum]
            if modified:
                if len(modified) == 1:
                    msg = (
                        'segment %s: adjusted 1 coordinate elevation by %.3f',
                        segnum, modified[0])
                else:
                    msg = (
                        'segment %s: adjusted %d coordinate elevations between'
                        ' %.3f and %.3f', segnum, len(modified),
                        min(modified), max(modified))
                self.logger.debug(*msg)
                self.messages.append(msg[0] % msg[1:])
                coords = self.segments.geometry[segnum].coords[:]
                dist = 0.0
                for idx in range(len(profile)):
                    dist += profile[idx][0]
                    profile[idx][0] = dist
                    coords[idx] = coords[idx][0:2] + (profile[idx][1],)
                self.segments.at[segnum, geom_name] = LineString(coords)
            else:
                self.logger.debug('segment %s: not adjusted', segnum)
                dist = 0.0
                for idx in range(len(profile)):
                    dist += profile[idx][0]
                    profile[idx][0] = dist
            profiles.append(LineString(profile))
        self.profiles = geopandas.GeoSeries(
                profiles, index=self.segments.index)

    def remove(self, condition=False, segnums=[]):
        """Remove segments (and catchments).

        Parameters
        ----------
        condition : bool or pandas.Series
            Series of bool for each segment index, where True is to remove.
            Combined with 'segnums'. Defaut False (keep all).
        segnums : list
            List of segnums to remove. Combined with 'condition'. Default [].

        Returns
        -------
        None

        """
        condition = self.segments_series(condition, "condition").astype(bool)
        if condition.any():
            self.logger.debug(
                'selecting %d segnum(s) based on a condition',
                condition.sum())
        sel = condition.copy()
        segments_index = self.segments.index
        if len(segnums) > 0:
            segnums_set = set(segnums)
            if not segnums_set.issubset(segments_index):
                diff = list(sorted(segnums_set.difference(segments_index)))
                raise IndexError(
                    f"{len(diff)} segnums not found in "
                    f"segments.index: {abbr_str(diff)}")
            self.logger.debug(
                'selecting %d segnum(s) based on a list', len(segnums_set))
            sel |= segments_index.isin(segnums_set)
        if sel.all():
            raise ValueError(
                'all segments were selected to remove; must keep at least one')
        elif (~sel).all():
            self.logger.info('no segments selected to remove; no changes made')
        else:
            assert sel.any()
            self.logger.info(
                'removing %d of %d segments (%.2f%%)', sel.sum(),
                len(segments_index), sel.sum() * 100.0 / len(segments_index))
            self._segments = self.segments.loc[~sel]
            if self.catchments is not None:
                self.catchments = self.catchments.loc[~sel]
        return

    def plot(self, column='stream_order', sort_column='sequence',
             cmap='viridis_r', legend=False):
        """Plot map of surface water network.

        Shows map of surface water network lines, with points showing,
        headwater (green dots), outlets (navy dots), and if present, diversion
        locations with a blue dashed line to the diversion location at the
        end of the segment line.

        Parameters
        ----------
        column : str, default "stream_order"
            Column from segments to use with ``map``.
            See also ``legend`` to help interpret values.
        sort_column : str, default "sequence"
            Column from segments to sort values.
        cmap : str, default "viridis_r"
            Matplotlib color map.
        legend : bool, default False
            Show legend for `column`.

        Returns
        -------
        AxesSubplot

        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        segments = self.segments
        if sort_column:
            segments = self.segments.sort_values(sort_column)
        segments.plot(
            column=column, label='segments', legend=legend, ax=ax, cmap=cmap)

        outet_points = geopandas.GeoSeries(
            self.segments.loc[self.outlets].geometry.apply(
                lambda g: Point(g.coords[-1])))
        outet_points.plot(
            ax=ax, label='outlet', marker='o', color='navy')

        headwater_points = geopandas.GeoSeries(
            self.segments.loc[self.headwater].geometry.apply(
                lambda g: Point(g.coords[0])))
        headwater_points.plot(
            ax=ax, label='headwater', marker='.', color='green')

        diversions = self.diversions
        if diversions is not None:
            diversions_is_spatial = (
                isinstance(diversions, geopandas.GeoDataFrame) and
                'geometry' in diversions.columns and
                (~diversions.is_empty).all())
            if diversions_is_spatial:
                diversion_points = diversions.geometry
            else:
                diversion_points = self.segments.loc[
                    self.diversions['from_segnum']].geometry.apply(
                        lambda g: Point(g.coords[-1]))
            diversion_points.plot(
                ax=ax, label='diversion', marker='+', color='red')
            if diversions_is_spatial:
                diversion_lines = []
                for item in self.diversions.itertuples():
                    p = self.segments.loc[item.from_segnum].geometry.coords[-1]
                    diversion_lines.append(
                        LineString([item.geometry.coords[0], p]))
                diversion_lines = geopandas.GeoSeries(diversion_lines)
                diversion_lines.plot(
                    ax=ax, label='diversion lines',
                    linestyle='--', color='deepskyblue')
        return ax

    def to_pickle(self, path, protocol=pickle.HIGHEST_PROTOCOL):
        """Pickle (serialize) object to file.

        Parameters
        ----------
        path : str
            File path where the pickled object will be stored.
        protocol : int
            Default is pickle.HIGHEST_PROTOCOL.

        See also
        --------
        SurfaceWaterNetwork.from_pickle : Read file.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=protocol)

    @classmethod
    def from_pickle(cls, path):
        """Read a pickled format from a file.

        Parameters
        ----------
        path : str
            File path where the pickled object will be stored.

        See also
        --------
        SurfaceWaterNetwork.to_pickle : Save file.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj
