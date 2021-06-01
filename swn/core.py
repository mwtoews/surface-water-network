# -*- coding: utf-8 -*-
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

from swn.spatial import get_sindex
from swn.util import abbr_str


class SurfaceWaterNetwork(object):
    """Surface water network class.

    Attributes
    ----------
    segments : geopandas.GeoDataFrame
        Primary GeoDataFrame created from 'lines' input, containing
        attributes evaluated during initialisation. Index is treated as a
        segment number or ID.
    catchments : geopandas.GeoSeries
        Catchments created from optional 'polygons' input, containing only
        the catchment polygon. Index must match segments.
    diversions : geopandas.GeoDataFrame or pd.DataFrame
        [Geo]DataFrame created from 'diversions' input, containing geometry
        (if available) for a surface water abstraction location, and the
        connected segment number from where the diversion occurs.
    END_SEGNUM : int
        Special segment number that indicates a line end, default is
        usually 0. This number is not part of segments.index.
    has_z : bool
        Property that indicates all segment lines have Z dimension coordinates.
    headwater : pandas.core.index.Int64Index
        Head water segment numbers at top of cachment.
    outlets : pandas.core.index.Int64Index
        Index segment numbers for each outlet.
    to_segnums : pandas.core.series.Series
        Series of segnum identifiers that connect downstream.
    logger : logging.Logger
        Logger to show messages.
    warnings : list
        List of warning messages.
    errors : list
        List of error messages.

    """

    END_SEGNUM = None
    segments = None
    logger = None
    warnings = None
    errors = None

    def __len__(self):
        """Return number of segments."""
        return len(self.segments.index)

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
                "expected 'logger' to be Logger; found " + str(type(logger)))
        self.logger.info('creating new %s object', self.__class__.__name__)
        if not isinstance(segments, geopandas.GeoDataFrame):
            raise ValueError(
                'segments must be a GeoDataFrame; found {!r}'
                .format(type(segments)))
        self.segments = segments
        self.END_SEGNUM = END_SEGNUM
        if self.END_SEGNUM in self.segments.index:
            self.logger.error(
                "END_SEGNUM %r found in segments.index", self.END_SEGNUM)
        notin = ~self.to_segnums.isin(self.segments.index)
        if notin.any():
            self.segments = self.segments.copy()
            self.logger.warning(
                "correcting %d to_segnum not found in segments.index",
                notin.sum())
            self.segments.loc[notin, "to_segnum"] = self.END_SEGNUM
        # all other properties added afterwards

    @classmethod
    def from_lines(cls, lines, polygons=None):
        """
        Create and evaluate a new SurfaceWaterNetwork from lines for segments.

        Parameters
        ----------
        lines : geopandas.GeoSeries
            Input lines of surface water network. Geometries must be
            'LINESTRING' or 'LINESTRING Z'. Index is used for segment numbers.
            The geometry is copied to the segments property.
        polygons : geopandas.GeoSeries, optional
            Optional input polygons of surface water catchments. Geometries
            must be 'POLYGON'. Index must be the same as segments.index.

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
        segments_sindex = get_sindex(segments)
        if segments.index.min() > 0:
            END_SEGNUM = 0
        else:
            END_SEGNUM = segments.index.min() - 1
        segments['to_segnum'] = END_SEGNUM
        obj = cls(segments=segments, END_SEGNUM=END_SEGNUM)
        del segments, END_SEGNUM  # dereference local copies
        obj.errors = []
        obj.warnings = []
        # Cartesian join of segments to find where ends connect to
        obj.logger.debug('finding connections between pairs of segment lines')
        for segnum1, geom1 in obj.segments.geometry.iteritems():
            end1_coord = geom1.coords[-1]  # downstream end
            end1_coord2d = end1_coord[0:2]
            end1_pt = Point(*end1_coord)
            if segments_sindex:
                subsel = segments_sindex.intersection(end1_coord2d)
                sub = obj.segments.iloc[sorted(subsel)]
            else:  # slow scan of all segments
                sub = obj.segments
            to_segnums = []
            for segnum2, geom2 in sub.geometry.iteritems():
                if segnum1 == segnum2:
                    continue
                start2_coord = geom2.coords[0]
                end2_coord = geom2.coords[-1]
                if end1_coord == start2_coord:
                    to_segnums.append(segnum2)  # perfect 3D match
                elif end1_coord2d == start2_coord[0:2]:
                    to_segnums.append(segnum2)
                    m = ('end of segment %s matches start of segment %s in '
                         '2D, but not in Z dimension', segnum1, segnum2)
                    obj.logger.warning(*m)
                    obj.warnings.append(m[0] % m[1:])
                elif (geom2.distance(end1_pt) < 1e-6 and
                      Point(*end2_coord).distance(end1_pt) > 1e-6):
                    m = ('segment %s connects to the middle of segment %s',
                         segnum1, segnum2)
                    obj.logger.error(*m)
                    obj.errors.append(m[0] % m[1:])
            if len(to_segnums) > 1:
                m = ('segment %s has more than one downstream segments: %s',
                     segnum1, str(to_segnums))
                obj.logger.error(*m)
                obj.errors.append(m[0] % m[1:])
            if len(to_segnums) > 0:
                # do not diverge flow; only flow to one downstream segment
                obj.segments.at[segnum1, 'to_segnum'] = to_segnums[0]
        # Store from_segnums set to segments GeoDataFrame
        from_segnums = obj.from_segnums
        obj.segments["from_segnums"] = from_segnums
        sel = obj.segments.from_segnums.isna()
        if sel.any():
            obj.segments.loc[sel, "from_segnums"] = \
                [set() for _ in range(sel.sum())]
        outlets = obj.outlets
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

        # Check to see if headwater and outlets have common locations
        headwater = obj.headwater
        obj.logger.debug(
            'checking %d headwater segments and %d outlet segments',
            len(headwater), len(outlets))
        start_coords = {}  # key: 2D coord, value: list of segment numbers
        for segnum1, geom1 in obj.segments.loc[headwater].geometry.iteritems():
            start1_coord = geom1.coords[0]
            start1_coord2d = start1_coord[0:2]
            if segments_sindex:
                subsel = segments_sindex.intersection(start1_coord2d)
                sub = obj.segments.iloc[sorted(subsel)]
            else:  # slow scan of all segments
                sub = obj.segments
            for segnum2, geom2 in sub.geometry.iteritems():
                if segnum1 == segnum2:
                    continue
                start2_coord = geom2.coords[0]
                match = False
                if start1_coord == start2_coord:
                    match = True  # perfect 3D match
                elif start1_coord2d == start2_coord[0:2]:
                    match = True
                    m = ('starting segment %s matches start of segment %s in '
                         '2D, but not in Z dimension', segnum1, segnum2)
                    obj.logger.warning(*m)
                    obj.warnings.append(m[0] % m[1:])
                if match:
                    if start1_coord2d in start_coords:
                        start_coords[start1_coord2d].add(segnum2)
                    else:
                        start_coords[start1_coord2d] = {segnum2}
        for key in start_coords.keys():
            v = start_coords[key]
            m = ('starting coordinate %s matches start segment%s: %s',
                 key, 's' if len(v) != 1 else '', v)
            obj.logger.error(*m)
            obj.errors.append(m[0] % m[1:])

        end_coords = {}  # key: 2D coord, value: list of segment numbers
        for segnum1, geom1 in obj.segments.loc[outlets].geometry.iteritems():
            end1_coord = geom1.coords[-1]
            end1_coord2d = end1_coord[0:2]
            if segments_sindex:
                subsel = segments_sindex.intersection(end1_coord2d)
                sub = obj.segments.iloc[sorted(subsel)]
            else:  # slow scan of all segments
                sub = obj.segments
            for segnum2, geom2 in sub.geometry.iteritems():
                if segnum1 == segnum2:
                    continue
                end2_coord = geom2.coords[-1]
                match = False
                if end1_coord == end2_coord:
                    match = True  # perfect 3D match
                elif end1_coord2d == end2_coord[0:2]:
                    match = True
                    m = ('ending segment %s matches end of segment %s in 2D, '
                         'but not in Z dimension', segnum1, segnum2)
                    obj.logger.warning(*m)
                    obj.warnings.append(m[0] % m[1:])
                if match:
                    if end1_coord2d in end_coords:
                        end_coords[end1_coord2d].add(segnum2)
                    else:
                        end_coords[end1_coord2d] = {segnum2}
        for key in end_coords.keys():
            v = end_coords[key]
            m = ('ending coordinate %s matches end segment%s: %s',
                 key, 's' if len(v) != 1 else '', v)
            obj.logger.warning(*m)
            obj.warnings.append(m[0] % m[1:])

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

    def __repr__(self):
        """Return string representation of surface water network."""
        modifiers = []
        if self.has_z:
            modifiers.append('Z coordinates')
        if self.catchments is not None:
            modifiers.append('catchment polygons')
        if modifiers:
            with_modifiers = ' with ' + modifiers[0]
            if len(modifiers) == 2:
                with_modifiers += ' and ' + modifiers[1]
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
        return dedent('''\
            <{}:{}
              {} segments: {}
              {} headwater: {}
              {} outlets: {}
              {} />'''.format(
            self.__class__.__name__, with_modifiers,
            len(segments), abbr_str(segments, 4),
            len(hw_l), abbr_str(hw_l, 4),
            len(out_l), abbr_str(out_l, 4),
            diversions_line))

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
            raise ValueError("expected 'dict'; found {!r}".format(type(state)))
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

    def plot(self, column='stream_order', sort_column='sequence',
             cmap='viridis_r', legend=False):
        """Plot map of surface water network.

        Shows map of surface water network lines, with points showing,
        headwater (green dots), outlets (navy dots), and if present, diversion
        locations with a blue dashed line to the diversion location at the
        end of the segment line.

        Parameters
        ----------
        column : str
            Column from segments to use with 'cmap'; default 'stream_order'.
            See also 'legend' to help interpret values.
        sort_column : str
            Column from segments to sort values; default 'sequence'.
        cmap : str
            Matplotlib color map; default 'viridis_r',
        legend : bool
            Show legend for 'column'; default False.

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

    @property
    def catchments(self):
        """Return Polygon GeoSeries of surface water catchments or None."""
        return getattr(self, '_catchments', None)

    @catchments.setter
    def catchments(self, value):
        if value is None:
            if hasattr(self, '_catchments'):
                delattr(self, '_catchments')
            return
        elif not isinstance(value, geopandas.GeoSeries):
            raise ValueError(
                'catchments must be a GeoSeries or None; found {!r}'
                .format(type(value)))
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
        """Return [Geo]DataFrame of surface water diversions or None."""
        return getattr(self, '_diversions', None)

    @diversions.setter
    def diversions(self, value):
        raise AttributeError("use 'set_diversions()' method")

    def set_diversions(self, diversions, min_stream_order=None):
        """Set surface water diversion locations.

        This method checks or assigns a segment number to divert surface water
        flow from, and adds other columns to the diversions and segments data
        frames.

        If a 'from_segnum' column exists, these values are checked against the
        index for segments, and adds/updates 'dist' (where possible) to
        describe the distance from the diversion to the end of the segment.

        If 'from_segnum' is not provided and a GeoDataFrame is provided, then
        the closest segment end is identified, using optional min_stream_order
        preferentially select higher-order stream segments.

        Parameters
        ----------
        diversions : pd.DataFrame, geopandas.GeoDataFrame or None
            Data frame of surface water diversions, a modified copy kept as a
            'diversions' property. Use None to unset this property.
        min_stream_order : int, optional
            Finds stream segments with a minimum stream order. Default None.

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
                    '{0} vs. {1}'.format(diversions_crs, segments_crs))
        if 'from_segnum' in diversions.columns:
            self.logger.debug(
                "checking existing 'from_segnum' column for diversions")
            sn = set(self.segments.index)
            dn = set(diversions['from_segnum'])
            if not sn.issuperset(dn):
                diff = dn.difference(sn)
                raise ValueError(
                    "{0} 'from_segnum' are not found in segments.index: {1}"
                    .format(len(diff), abbr_str(diff)))
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
        """Return index of outlets."""
        return self.segments.index[
                self.segments['to_segnum'] == self.END_SEGNUM]

    @property
    def to_segnums(self):
        """Return Series of segnum to connect downstream."""
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
        upstream, downstream : int or list, optional
            Segmnet number(s) from segments.index to search from. Default [].
        barriers : int or list, optional
            Segment number(s) that cannot be traversed past. Default [].
        gather_upstream : bool
            Gather upstream from all other downstream segments. Default False.

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
                        '{0} {1} segment{2} not found in segments.index: {3}'
                        .format(len(diff), name, '' if len(diff) == 1 else 's',
                                abbr_str(diff)))
                return var
            else:
                if var not in segments_index:
                    raise IndexError(
                        '{0} segnum {1} not found in segments.index'
                        .format(name, var))
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
            catchments. Default 'upstream_length'. Anothr good candidate
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
                '{!r} not found in segments.columns'.format(follow_up))
        junctions = list(segnums)
        junctions_set = set(junctions)
        if len(junctions) != len(junctions_set):
            raise ValueError('list of segnums is not unique')
        segments_index_set = set(self.segments.index)
        if not junctions_set.issubset(segments_index_set):
            diff = junctions_set.difference(segments_index_set)
            raise IndexError(
                '{0} segnums not found in segments.index: {1}'
                .format(len(diff), abbr_str(diff)))
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

        # aggregate segnums in a path accross an internal subcatchment
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
            lines.at[segnum] = linemerge(list(
                    self.segments.loc[reversed(agg_path_l), 'geometry']))

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
        values : pandas.core.series.Series
            Series of values that align with the index.

        Returns
        -------
        pandas.core.series.Series
            Accumulated values.

        """
        if not isinstance(values, pd.Series):
            raise ValueError('values must be a pandas Series')
        elif (len(values.index) != len(self.segments.index) or
                not (values.index == self.segments.index).all()):
            raise ValueError('index is different')
        accum = values.copy()
        try:
            accum.name = 'accumulated_' + accum.name
        except TypeError:
            pass
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
        upstream_area : str or pd.Series
            Column name in segments (default 'upstream_area') to use for
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
                    '{!r} not found in segments.columns'.format(upstream_area))
            upstream_area_km2 = self.segments[upstream_area] / 1e6
        else:
            raise ValueError('unknown use for upstream_area')
        if not isinstance(a, (float, int)):
            a = self._segment_series(a, 'a')
        if not isinstance(b, (float, int)):
            b = self._segment_series(a, 'b')
        self.segments['width'] = a + upstream_area_km2 ** b

    def _segment_series(self, value, name=None):
        """Return a pandas.Series along the segment index.

        Parameters
        ----------
        value : float or pandas.Series
            If value is a float, it is repated for each segment, otherwise
            a Series is used, with a check to ensure it is the same index.
        name : str, optional
            Name used for series, if provided. Default is None.

        Returns
        -------
        pandas.Series

        """
        segments_index = self.segments.index
        if not isinstance(value, pd.Series):
            value = pd.Series(value, index=segments_index)
        elif (len(value.index) != len(segments_index) or
                not (value.index == segments_index).all()):
            raise ValueError('index is different than for segments')
        if name is not None:
            value.name = name
        return value

    def _outlet_series(self, value):
        """Return a pandas.Series along the outlet index.

        Parameters
        ----------
        value : float or pandas.Series
            If value is a float, it is repated for each outlet, otherwise
            a Series is used, with a check to ensure it is the same index.

        Returns
        -------
        pandas.Series

        """
        outlets_index = self.outlets
        if not isinstance(value, pd.Series):
            value = pd.Series(value, index=outlets_index)
        elif (len(value.index) != len(outlets_index) or
                not (value.index == outlets_index).all()):
            raise ValueError('index is different than for outlets')
        return value

    def _pair_segment_values(self, value1, outlet_value=None, name=None):
        """Return a pair of values that connect the segments.

        The first value applies to the top of each segment, and the bottom
        value is determined from the top of the value it connects to. Special
        consideration is applied to outlet segments.

        Parameters
        ----------
        value1 : float or pandas.Series
            Value to assign to the top of each segment.
        outlet_value : None, float or pandas.Series
            If None (default), the value used for the bottom of outlet segments
            is assumed to be the same as the top. Otherwise, a Series
            for each outlet can be specified.
        name : str, optional
            Base name used for each series pair, if provided. Default is None.

        Returns
        -------
        pandas.DataFrame
            Resulting DataFrame has two columns for top (1) and bottom (2) of
            each segment.

        """
        value1 = self._segment_series(value1, name=name)
        df = pd.concat([value1, value1], axis=1)
        if value1.name is not None:
            df.columns = df.columns.str.cat(['1', '2'])
        else:
            df.columns += 1
        to_segnums = self.to_segnums
        c1, c2 = df.columns
        df.loc[to_segnums.index, c2] = df.loc[to_segnums, c1].values
        if outlet_value is not None:
            outlet_value = self._outlet_series(outlet_value)
            df.loc[outlet_value.index, c2] = outlet_value
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
        min_slope = self._segment_series(min_slope)
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
        condition = self._segment_series(condition, 'condition').astype(bool)
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
                    '{0} segnums not found in segments.index: {1}'
                    .format(len(diff), abbr_str(diff)))
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
            self.segments = self.segments.loc[~sel]
            if self.catchments is not None:
                self.catchments = self.catchments.loc[~sel]
        return

    def to_pickle(self, path, protocol=pickle.HIGHEST_PROTOCOL):
        """Pickle (serialize) object to file.

        Parameters
        ----------
        path : str
            File path where the pickled object will be stored.
        protocol : int
            Default is pickle.HIGHEST_PROTOCOL.

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

        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj
