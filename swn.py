# -*- coding: utf-8 -*-
import geopandas
import logging
import numpy as np
import pandas as pd
try:
    from geopandas.tools import sjoin
except ImportError:
    sjoin = False
from fiona import crs as fiona_crs
from math import sqrt
from shapely import wkt
from shapely.geometry import LineString, Point, Polygon, box
from matplotlib import pyplot as plt
from matplotlib import colors, cm
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.ops import cascaded_union, linemerge

try:
    from osgeo import gdal
except ImportError:
    gdal = False
try:
    import rtree
    from rtree.index import Index as RTreeIndex
except ImportError:
    rtree = False
try:
    import flopy
except ImportError:
    flopy = False

__version__ = '0.1'
__author__ = 'Mike Toews'

module_logger = logging.getLogger(__name__)
if __name__ not in [_.name for _ in module_logger.handlers]:
    if logging.root.handlers:
        module_logger.addHandler(logging.root.handlers[0])
    else:
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s:%(message)s',
            '%H:%M:%S')
        handler = logging.StreamHandler()
        handler.name = __name__
        handler.setFormatter(formatter)
        module_logger.addHandler(handler)
        del formatter, handler

# default threshold size of geometries when Rtree index is built
_rtree_threshold = 100


def get_sindex(gdf):
    """Helper function to get or build a spatial index

    Particularly useful for geopandas<0.2.0
    """
    assert isinstance(gdf, geopandas.GeoDataFrame)
    has_sindex = hasattr(gdf, 'sindex')
    if has_sindex:
        sindex = gdf.geometry.sindex
    elif rtree and len(gdf) >= _rtree_threshold:
        # Manually populate a 2D spatial index for speed
        sindex = RTreeIndex()
        # slow, but reliable
        for idx, (segnum, row) in enumerate(gdf.bounds.iterrows()):
            sindex.add(idx, tuple(row))
    else:
        sindex = None
    return sindex


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        mask = np.ma.getmask(value)
        return np.ma.masked_array(np.interp(value, x, y), mask=mask)


class SurfaceWaterNetwork(object):
    """Surface water network

    Attributes
    ----------
    segments : geopandas.GeoDataFrame
        Primary GeoDataFrame created from 'lines' input, containing
        attributes evaluated during initialisation. Index is treated as a
        segment number or ID.
    catchments : geopandas.GeoSeries
        Catchments created from optional 'polygons' input, containing only
        the catchment polygon. Index must match segments.
    END_SEGNUM : int
        Special segment number that indicates a line end, default is usually 0.
        This number is not part of the index.
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
    index = None
    END_SEGNUM = None
    segments = None
    logger = None
    warnings = None
    errors = None

    def __len__(self):
        return len(self.segments.index)

    def __init__(self, lines, polygons=None, logger=None):
        """
        Initialise SurfaceWaterNetwork and evaluate segments

        Parameters
        ----------
        lines : geopandas.GeoSeries
            Input lines of surface water network. Geometries must be
            'LINESTRING' or 'LINESTRING Z'. Index is used for segment numbers.
            The geometry is copied to the segments property.
        polygons : geopandas.GeoSeries, optional
            Input polygons of surface water catchments. Geometries must be
            'POLYGON'. Index must be the same as segment.index.
        logger : logging.Logger, optional
            Logger to show messages.
        """
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.handlers = module_logger.handlers
            self.logger.setLevel(module_logger.level)
        if not isinstance(lines, geopandas.GeoSeries):
            raise ValueError('lines must be a GeoSeries')
        elif len(lines) == 0:
            raise ValueError('one or more lines are required')
        elif not (lines.geom_type == 'LineString').all():
            raise ValueError('lines must all be LineString types')
        # Create a new GeoDataFrame with a copy of line's geometry
        self.segments = geopandas.GeoDataFrame(geometry=lines)
        self.logger.info('creating network with %d segments', len(self))
        if isinstance(polygons, geopandas.GeoSeries):
            self.catchments = polygons.copy()
        elif polygons is not None:
            raise ValueError(
                'polygons must be a GeoSeries or None')
        segments_sindex = get_sindex(self.segments)
        if self.segments.index.min() > 0:
            self.END_SEGNUM = 0
        else:
            self.END_SEGNUM = self.segments.index.min() - 1
        self.segments['to_segnum'] = self.END_SEGNUM
        self.errors = []
        self.warnings = []
        # Cartesian join of segments to find where ends connect to
        self.logger.debug('finding connections between pairs of segment lines')
        for segnum1, geom1 in self.segments.geometry.iteritems():
            end1_coord = geom1.coords[-1]  # downstream end
            end1_coord2d = end1_coord[0:2]
            end1_pt = Point(*end1_coord)
            if segments_sindex:
                subsel = segments_sindex.intersection(end1_coord2d)
                sub = self.segments.iloc[sorted(subsel)]
            else:  # slow scan of all segments
                sub = self.segments
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
                    self.logger.warning(*m)
                    self.warnings.append(m[0] % m[1:])
                elif (geom2.distance(end1_pt) < 1e-6 and
                      Point(*end2_coord).distance(end1_pt) > 1e-6):
                    m = ('segment %s connects to the middle of segment %s',
                         segnum1, segnum2)
                    self.logger.error(*m)
                    self.errors.append(m[0] % m[1:])
            if len(to_segnums) > 1:
                m = ('segment %s has more than one downstream segments: %s',
                     segnum1, str(to_segnums))
                self.logger.error(*m)
                self.errors.append(m[0] % m[1:])
            if len(to_segnums) > 0:
                # do not diverge flow; only flow to one downstream segment
                self.segments.at[segnum1, 'to_segnum'] = to_segnums[0]
        # Store from_segnums list to segments GeoDataFrame
        self.segments['from_segnums'] = None
        from_segnums = self.from_segnums
        for k, v in from_segnums.items():
            self.segments.at[k, 'from_segnums'] = v
        outlets = self.outlets
        self.logger.debug('evaluating segments upstream from %d outlet%s',
                          len(outlets), 's' if len(outlets) != 1 else '')
        self.segments['cat_group'] = self.END_SEGNUM
        self.segments['num_to_outlet'] = 0
        self.segments['dist_to_outlet'] = 0.0

        # Recursive function that accumulates information upstream
        def resurse_upstream(segnum, cat_group, num, dist):
            self.segments.at[segnum, 'cat_group'] = cat_group
            num += 1
            self.segments.at[segnum, 'num_to_outlet'] = num
            dist += self.segments.geometry[segnum].length
            self.segments.at[segnum, 'dist_to_outlet'] = dist
            # Branch to zero or more upstream segments
            for from_segnum in from_segnums.get(segnum, []):
                resurse_upstream(from_segnum, cat_group, num, dist)

        for segnum in self.segments.loc[outlets].index:
            resurse_upstream(segnum, segnum, 0, 0.0)

        # Check to see if headwater and outlets have common locations
        headwater = self.headwater
        self.logger.debug(
            'checking %d headwater segments and %d outlet segments',
            len(headwater), len(outlets))
        start_coords = {}  # key: 2D coord, value: list of segment numbers
        for segnum1, geom1 in self.\
                segments.loc[headwater].geometry.iteritems():
            start1_coord = geom1.coords[0]
            start1_coord2d = start1_coord[0:2]
            if segments_sindex:
                subsel = segments_sindex.intersection(start1_coord2d)
                sub = self.segments.iloc[sorted(subsel)]
            else:  # slow scan of all segments
                sub = self.segments
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
                    self.logger.warning(*m)
                    self.warnings.append(m[0] % m[1:])
                if match:
                    if start1_coord2d in start_coords:
                        start_coords[start1_coord2d].add(segnum2)
                    else:
                        start_coords[start1_coord2d] = set([segnum2])
        for key in start_coords.keys():
            v = start_coords[key]
            m = ('starting coordinate %s matches start segment%s: %s',
                 key, 's' if len(v) != 1 else '', v)
            self.logger.error(*m)
            self.errors.append(m[0] % m[1:])

        end_coords = {}  # key: 2D coord, value: list of segment numbers
        for segnum1, geom1 in self.segments.loc[outlets].geometry.iteritems():
            end1_coord = geom1.coords[-1]
            end1_coord2d = end1_coord[0:2]
            if segments_sindex:
                subsel = segments_sindex.intersection(end1_coord2d)
                sub = self.segments.iloc[sorted(subsel)]
            else:  # slow scan of all segments
                sub = self.segments
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
                    self.logger.warning(*m)
                    self.warnings.append(m[0] % m[1:])
                if match:
                    if end1_coord2d in end_coords:
                        end_coords[end1_coord2d].add(segnum2)
                    else:
                        end_coords[end1_coord2d] = set([segnum2])
        for key in end_coords.keys():
            v = end_coords[key]
            m = ('ending coordinate %s matches end segment%s: %s',
                 key, 's' if len(v) != 1 else '', v)
            self.logger.warning(*m)
            self.warnings.append(m[0] % m[1:])

        self.logger.debug('evaluating downstream sequence')
        self.segments['sequence'] = 0
        self.segments['stream_order'] = 0
        # self.segments['numiter'] = 0  # should be same as stream_order
        # Sort headwater segments from the furthest from outlet to closest
        search_order = ['num_to_outlet', 'dist_to_outlet']
        furthest_upstream = self.segments.loc[headwater]\
            .sort_values(search_order, ascending=False).index
        sequence = pd.Series(
            np.arange(len(furthest_upstream)) + 1, index=furthest_upstream)
        self.segments.loc[sequence.index, 'sequence'] = sequence
        self.segments.loc[sequence.index, 'stream_order'] = 1
        completed = set(headwater)
        sequence = int(sequence.max())
        for numiter in range(1, self.segments['num_to_outlet'].max() + 1):
            # Gather segments downstream from completed upstream set
            downstream = set(
                self.segments.loc[completed, 'to_segnum'])\
                .difference(completed.union([self.END_SEGNUM]))
            # Sort them to evaluate the furthest first
            downstream_sorted = self.segments.loc[downstream]\
                .sort_values(search_order, ascending=False).index
            for segnum in downstream_sorted:
                if from_segnums[segnum].issubset(completed):
                    sequence += 1
                    self.segments.at[segnum, 'sequence'] = sequence
                    # self.segments.at[segnum, 'numiter'] = numiter
                    up_ord = list(
                        self.segments.loc[
                            list(from_segnums[segnum]), 'stream_order'])
                    max_ord = max(up_ord)
                    if up_ord.count(max_ord) > 1:
                        self.segments.at[segnum, 'stream_order'] = max_ord + 1
                    else:
                        self.segments.at[segnum, 'stream_order'] = max_ord
                    completed.add(segnum)
            if self.segments['sequence'].min() > 0:
                break
        self.logger.debug('sequence evaluated with %d iterations', numiter)
        # Don't do this: self.segments.sort_values('sequence', inplace=True)

    @classmethod
    def init_from_gdal(cls, lines_srs, elevation_srs=None):
        """
        Initialise SurfaceWaterNetwork from GDAL source datasets

        Parameters
        ----------
        lines_srs : str
            Path to open vector GDAL dataset of stream network lines.
        elevation_srs : str, optional
            Path to open raster GDAL dataset of elevation. If not provided,
            then Z dimension from lines used. Not implemented yet.
        """
        if not gdal:
            raise ImportError('this method requires GDAL')
        lines_ds = gdal.Open(lines_srs, gdal.GA_ReadOnly)
        if lines_ds is None:
            raise IOError('cannot open lines: {}'.format(lines_srs))
        logger = logging.getLogger(cls.__class__.__name__)
        logger.handlers = module_logger.handlers
        logger.setLevel(module_logger.level)
        logger.info('reading lines from: %s', lines_srs)
        projection = lines_ds.GetProjection()
        if elevation_srs is None:
            elevation_ds = None
        else:
            logger.info('reading elevation from: %s', elevation_srs)
            elevation_ds = gdal.Open(elevation_srs, gdal.GA_ReadOnly)
            if elevation_ds is None:
                raise IOError('cannot open elevation: {}'.format(elevation_ds))
            elif elevation_ds.RasterCount != 1:
                logger.warning(
                    'expected 1 raster band for elevation, found %s',
                    elevation_ds.RasterCount)
            band = elevation_ds.GetRasterBand(1)
            elevation = np.ma.array(band.ReadAsArray(), np.float64, copy=True)
            nodata = band.GetNoDataValue()
            elevation_ds = band = None  # close raster
            if nodata is not None:
                elevation.mask = elevation == nodata
            raise NotImplementedError('nothing done with elevation yet')
        return cls(projection=projection, logger=logger)

    @property
    def catchments(self):
        """Returns Polygon GeoSeries of surface water catchments or None"""
        return getattr(self, '_catchments', None)

    @catchments.setter
    def catchments(self, value):
        if value is None:
            delattr(self, '_catchments')
            return
        elif not isinstance(value, geopandas.GeoSeries):
            raise ValueError(
                'catchments must be a GeoSeries or None')
        segments_index = self.segments.index
        if (len(value.index) != len(segments_index) or
                not (value.index == segments_index).all()):
            raise ValueError(
                'catchments.index is different than for segments')
        # TODO: check extent overlaps
        self._catchments = value

    @property
    def has_z(self):
        """Returns True if all segment lines have Z dimension"""
        return bool(self.segments.geometry.apply(lambda x: x.has_z).all())

    @property
    def headwater(self):
        """Returns index of headwater segments"""
        return self.segments.index[
                ~self.segments.index.isin(self.segments['to_segnum'])]

    @property
    def outlets(self):
        """Returns index of outlets"""
        return self.segments.index[
                self.segments['to_segnum'] == self.END_SEGNUM]

    @property
    def to_segnums(self):
        """Returns Series of segnum to connect downstream"""
        return self.segments.loc[
            self.segments['to_segnum'] != self.END_SEGNUM, 'to_segnum']

    @property
    def from_segnums(self):
        """Returns dict of a set of segnums to connect upstream"""
        to_segnums = self.to_segnums
        from_segnums = {}
        for k, v in to_segnums.items():
            from_segnums.setdefault(v, set()).add(k)
        return from_segnums

    def query(self, upstream=[], downstream=[], barrier=[],
              gather_upstream=False):
        """Returns segnums upstream (inclusive) and downstream (exclusive)

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
                    diff = set(var).difference(segments_set)
                    raise IndexError(
                        '{0} {1} segment{2} not found in segments.index: {3}'
                        .format(len(diff), name, '' if len(diff) == 1 else 's',
                                list(diff)[:5]))
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

    def aggregate(self, segnums):
        """Aggregate segments (and catchments) to a coarser network of segnums

        Parameters
        ----------
        segnums : list
            List of segmnet numbers to aggregate. Must be unique.

        Returns
        -------
        SurfaceWaterNetwork
        """
        segnums_set = set(segnums)
        if len(segnums) != len(segnums_set):
            raise ValueError('list of segnums is not unique')
        segments_index_set = set(self.segments.index)
        if not segnums_set.issubset(segments_index_set):
            diff = segnums_set.difference(segments_index_set)
            raise IndexError(
                '{0} segnums not found in segments.index: {1}'
                .format(len(diff), list(diff)[:5]))
        from_segnums = self.from_segnums
        if 'length' in self.segments.columns:
            del self.segments['length']
        self.segments['length'] = self.segments.geometry.length
        order_keys = ['stream_order', 'length']

        def up_line(segnum):
            yield self.segments.geometry.at[segnum]
            if segnum in from_segnums:
                sel = list(from_segnums[segnum].difference(segnums_set))
                if len(sel) > 0:
                    if len(sel) == 1:
                        next_segnum = sel[0]
                    elif len(sel) > 1:
                        # pick the longest line with highest stream order
                        next_segnum = self.segments.loc[
                            from_segnums[segnum], order_keys]\
                            .sort_values(order_keys, ascending=False).index[0]
                    yield from up_line(next_segnum)

        def up_poly(segnum):
            yield self.catchments.geometry.at[segnum]
            for from_segnum in from_segnums.get(segnum, []):
                if from_segnum not in segnums_set:
                    yield from up_poly(from_segnum)

        lines = geopandas.GeoSeries()
        if self.catchments is not None:
            polygons = geopandas.GeoSeries()
        else:
            polygons = None
        for segnum in segnums:
            lines.loc[segnum] = linemerge(list(up_line(segnum)))
            if polygons is not None:
                polygons.loc[segnum] = cascaded_union(list(up_poly(segnum)))
        return SurfaceWaterNetwork(lines, polygons)

    def accumulate_values(self, values):
        """Accumulate values down the stream network

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
        for segnum in self.segments.sort_values('sequence').index:
            from_segnums = self.segments.at[segnum, 'from_segnums']
            if from_segnums:
                accum[segnum] += accum[list(from_segnums)].sum()
        return accum

    def segment_series(self, value, name=None):
        """Returns a pandas.Series along the segment index

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

    def outlet_series(self, value):
        """Returns a pandas.Series along the outlet index

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

    def pair_segment_values(self, value1, outlet_value=None, name=None):
        """Returns a pair of values that connect the segments

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
        value1 = self.segment_series(value1, name=name)
        df = pd.concat([value1, value1], axis=1)
        if value1.name is not None:
            df.columns = df.columns.str.cat(['1', '2'])
        else:
            df.columns += 1
        to_segnums = self.to_segnums
        c1, c2 = df.columns
        df.loc[to_segnums.index, c2] = df.loc[to_segnums, c1].values
        if outlet_value is not None:
            outlet_value = self.outlet_series(outlet_value)
            df.loc[outlet_value.index, c2] = outlet_value
        return df

    def adjust_elevation_profile(self, min_slope=1./1000):
        """Check and adjust (if necessary) Z coordinates of elevation profiles

        Parameters
        ----------
        min_slope : float or pandas.Series, optional
            Minimum downwards slope imposed on segments. If float, then this is
            a global value, otherwise it is per-segment with a Series.
            Default 1./1000 (or 0.001).
        """
        min_slope = self.segment_series(min_slope)
        if (min_slope < 0.0).any():
            raise ValueError('min_slope must be greater than zero')
        elif not self.has_z:
            raise AttributeError('line geometry does not have Z dimension')
        geom_name = self.segments.geometry.name
        # Build elevation profiles as 2D coordinates,
        # where X is 2D distance from downstream coordinate and Y is elevation
        profiles = []
        self.messages = []
        for segnum, geom in self.segments.geometry.iteritems():
            modified = 0
            coords = geom.coords[:]  # coordinates
            x0, y0, z0 = coords[0]  # upstream coordinate
            dist = geom.length  # total 2D distance from downstream coordinate
            profile_coords = [(dist, z0)]
            for idx, (x1, y1, z1) in enumerate(coords[1:], 1):
                dz = z0 - z1
                dx = sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
                dist -= dx
                # Check and enforce minimum slope
                slope = dz / dx
                if slope < min_slope[segnum]:
                    modified += 1
                    z1 = z0 - dx * min_slope[segnum]
                    coords[idx] = (x1, y1, z1)
                profile_coords.append((dist, z1))
                x0, y0, z0 = x1, y1, z1
            if modified > 0:
                m = ('adjusting %d coordinate elevation%s in segment %s',
                     modified, 's' if modified != 1 else '', segnum)
                self.logger.debug(*m)
                self.messages.append(m[0] % m[1:])
            if modified:
                self.segments.at[segnum, geom_name] = LineString(coords)
            profiles.append(LineString(profile_coords))
        self.profiles = geopandas.GeoSeries(profiles)
        return
        # TODO: adjust connected segments
        # Find minimum elevation, then force any segs downstream to flow down
        self.profiles.geometry.bounds.miny.sort_values()

        # lines = self.segments
        numiter = 0
        while True:
            numiter += 1

    def remove(self, condition=False, segnums=[]):
        """Remove segments (and catchments)

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
        condition = self.segment_series(condition, 'condition').astype(bool)
        if condition.any():
            self.logger.debug(
                'selecting %d segnum(s) based on a condition',
                condition.sum())
        sel = condition.copy()
        segments_index = self.segments.index
        if len(segnums) > 0:
            segnums_set = set(segnums)
            if not segnums_set.issubset(segments_index):
                diff = segnums_set.difference(segments_index)
                raise IndexError(
                    '{0} segnums not found in segments.index: {1}'
                    .format(len(diff), list(diff)[:5]))
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

    def process_flopy(self, m, ibound_action='freeze', min_slope=1./1000,
                      reach_include_fraction=0.2,
                      hyd_cond1=1., hyd_cond_out=None,
                      thickness1=1., thickness_out=None,
                      width1=10., width_out=None, roughch=0.024,
                      inflow={}, flow={}, runoff={}, etsw={}, pptsw={}):
        """Process MODFLOW groundwater model information from flopy

        Parameters
        ----------
        m : flopy.modflow.mf.Modflow
            Instance of a flopy MODFLOW model with DIS and BAS6 packages.
        ibound_action : str, optional
            Action to handle IBOUND:
                - ``freeze`` : Freeze IBOUND, but clip streams to fit bounds.
                - ``modify`` : Modify IBOUND to fit streams, where possible.
        min_slope : float or pandas.Series, optional
            Minimum downwards slope imposed on segments. If float, then this is
            a global value, otherwise it is per-segment with a Series.
            Default 1./1000 (or 0.001).
        reach_include_fraction : float or pandas.Series, optional
            Fraction of cell size used as a threshold distance to determine if
            reaches outside the active grid should be included to a cell.
            Based on the furthest distance of the line and cell geometries.
            Default 0.2 (e.g. for a 100 m grid cell, this is 20 m).
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
        """
        if not flopy:
            raise ImportError('this method requires flopy')
        elif not isinstance(m, flopy.modflow.mf.Modflow):
            raise ValueError('m must be a flopy.modflow.mf.Modflow object')
        elif ibound_action not in ('freeze', 'modify'):
            raise ValueError('ibound_action must be one of freeze or modify')
        elif not m.has_package('DIS'):
            raise ValueError('DIS package required')
        elif not m.has_package('BAS6'):
            raise ValueError('BAS6 package required')
        # Build stress period DataFrame from modflow model
        stress_df = pd.DataFrame({'perlen': m.dis.perlen.array})
        stress_df['duration'] = pd.TimedeltaIndex(
                stress_df['perlen'].cumsum(), m.modeltime.time_units)
        stress_df['start'] = pd.to_datetime(m.modeltime.start_datetime)
        stress_df['end'] = stress_df['duration'] + stress_df.at[0, 'start']
        stress_df.loc[1:, 'start'] = stress_df['end'].iloc[:-1].values
        segments_segnums = set(self.segments.index)

        def check_ts(data, name, drop_segnums=True):
            if isinstance(data, dict):
                data = pd.DataFrame(data, index=stress_df['start'])
            elif not isinstance(data, pd.DataFrame):
                raise ValueError(
                    '{0} must be a dict or DataFrame'.format(name))
            if len(data) != m.dis.nper:
                raise ValueError(
                    'length of {0} ({1}) is different than nper ({2})'
                    .format(name, len(data), m.dis.nper))
            if m.dis.nper > 1:  # check DatetimeIndex
                if not isinstance(data.index, pd.DatetimeIndex):
                    raise ValueError(
                        '{0}.index must be a pandas.DatetimeIndex'
                        .format(name))
                elif not (data.index == stress_df['start']).all():
                    try:
                        t = stress_df['start'].to_string(
                                index=False, max_rows=5).replace('\n', ', ')
                    except TypeError:
                        t = ', '.join([str(x)
                                       for x in list(stress_df['start'][:5])])
                    raise ValueError(
                        '{0}.index does not match expected ({1})'
                        .format(name, t))
            try:
                data.columns = data.columns.astype(self.segments.index.dtype)
            except TypeError:
                raise ValueError(
                    '{0}.columns.dtype must be same as segments.index.dtype'
                    .format(name))
            data_segnums = set(data.columns)
            if len(data_segnums) > 0:
                if data_segnums.isdisjoint(segments_segnums):
                    raise ValueError(
                        '{0}.columns not found in segments.index'.format(name))
                if drop_segnums:
                    not_found = data_segnums.difference(segments_segnums)
                    if not data_segnums.issubset(segments_segnums):
                        self.logger.warning(
                            'dropping %s of %s %s.columns, which are '
                            'not found in segments.index',
                            len(not_found), len(data_segnums), name)
                        data.drop(not_found, axis=1, inplace=True)
            return data

        self.logger.debug('checking timeseries data against modflow model')
        inflow = check_ts(inflow, 'inflow', drop_segnums=False)
        flow = check_ts(flow, 'flow')
        runoff = check_ts(runoff, 'runoff')
        etsw = check_ts(etsw, 'etsw')
        pptsw = check_ts(pptsw, 'pptsw')
        self.logger.debug('building model grid cell geometries')
        # Make sure model CRS and segments CRS are the same (if defined)
        crs = None
        segments_crs = getattr(self.segments.geometry, 'crs', None)
        modelgrid_crs = None
        if m.modelgrid.proj4 is not None:
            modelgrid_crs = fiona_crs.from_string(m.modelgrid.proj4)
        if (segments_crs is not None and modelgrid_crs is not None and
                segments_crs != modelgrid_crs):
            self.logger.warning(
                'CRS for segments and modelgrid are different: {0} vs. {1}'
                .format(segments_crs, modelgrid_crs))
        crs = segments_crs or modelgrid_crs
        # Make sure their extents overlap
        minx, maxx, miny, maxy = m.modelgrid.extent
        model_bbox = box(minx, miny, maxx, maxy)
        rstats = self.segments.bounds.describe()
        segments_bbox = box(
                rstats.loc['min', 'minx'], rstats.loc['min', 'miny'],
                rstats.loc['max', 'maxx'], rstats.loc['max', 'maxy'])
        if model_bbox.disjoint(segments_bbox):
            raise ValueError('modelgrid extent does not cover segments extent')
        # More careful check of overlap of lines with grid polygons
        cols, rows = np.meshgrid(np.arange(m.dis.ncol), np.arange(m.dis.nrow))
        ibound = m.bas6.ibound[0].array.copy()
        ibound_modified = 0
        grid_df = pd.DataFrame({'row': rows.flatten(), 'col': cols.flatten()})
        grid_df.set_index(['row', 'col'], inplace=True)
        grid_df['ibound'] = ibound.flatten()
        if ibound_action == 'freeze' and (ibound == 0).any():
            # Remove any inactive grid cells from analysis
            grid_df = grid_df.loc[grid_df['ibound'] != 0]
        # Determine grid cell size
        col_size = np.median(m.dis.delr.array)
        if m.dis.delr.array.min() != m.dis.delr.array.max():
            self.logger.warning(
                'assuming constant column spacing %s', col_size)
        row_size = np.median(m.dis.delc.array)
        if m.dis.delc.array.min() != m.dis.delc.array.max():
            self.logger.warning(
                'assuming constant row spacing %s', row_size)
        cell_size = (row_size + col_size) / 2.0
        # Note: m.modelgrid.get_cell_vertices(row, col) is slow!
        xv = m.modelgrid.xvertices
        yv = m.modelgrid.yvertices
        r, c = [np.array(s[1])
                for s in grid_df.reset_index()[['row', 'col']].iteritems()]
        cell_verts = zip(
            zip(xv[r, c], yv[r, c]),
            zip(xv[r, c + 1], yv[r, c + 1]),
            zip(xv[r + 1, c + 1], yv[r + 1, c + 1]),
            zip(xv[r + 1, c], yv[r + 1, c])
        )
        self.grid_cells = grid_cells = geopandas.GeoDataFrame(
            grid_df, geometry=[Polygon(r) for r in cell_verts], crs=crs)
        self.logger.debug('evaluating reach data on model grid')
        grid_sindex = get_sindex(grid_cells)
        reach_include = self.segment_series(reach_include_fraction) * cell_size
        # Make an empty DataFrame for reaches
        reach_df = pd.DataFrame(columns=['geometry'])
        reach_df.insert(1, column='segnum',
                        value=pd.Series(dtype=self.segments.index.dtype))
        reach_df.insert(2, column='dist', value=pd.Series(dtype=float))
        reach_df.insert(3, column='row', value=pd.Series(dtype=int))
        reach_df.insert(4, column='col', value=pd.Series(dtype=int))

        # general helper function
        def append_reach_df(segnum, line, row, col, reach_geom):
            if line.has_z:
                # intersection(line) does not preserve Z coords,
                # but line.interpolate(d) works as expected
                reach_geom = LineString(line.interpolate(
                    line.project(Point(c))) for c in reach_geom.coords)
            # Get a point from the middle of the reach_geom
            reach_mid_pt = reach_geom.interpolate(0.5, normalized=True)
            reach_df.loc[len(reach_df.index)] = {
                'geometry': reach_geom,
                'segnum': segnum,
                'dist': line.project(reach_mid_pt, normalized=True),
                'row': row,
                'col': col,
            }

        # recusive helper functions
        def append_reach(segnum, row, col, line, reach_geom):
            if reach_geom.geom_type == 'LineString':
                append_reach_df(segnum, line, row, col, reach_geom)
            elif reach_geom.geom_type.startswith('Multi'):
                for sub_reach_geom in reach_geom.geoms:  # recurse
                    append_reach(segnum, row, col, line, sub_reach_geom)
            else:
                raise NotImplementedError(reach_geom.geom_type)

        def reassign_reach(segnum, line, rem):
            if rem.geom_type == 'LineString':
                threshold = cell_size * 2.0
                if rem.length > threshold:
                    self.logger.debug(
                        'remaining line segment from %s too long to merge '
                        '(%.1f > %.1f)', segnum, rem.length, threshold)
                    return
                if grid_sindex:
                    bbox_match = sorted(grid_sindex.intersection(rem.bounds))
                    sub = grid_cells.geometry.iloc[bbox_match]
                else:  # slow scan of all cells
                    sub = grid_cells.geometry
                assert len(sub) > 0, len(sub)
                matches = []
                for (row, col), grid_geom in sub.iteritems():
                    if grid_geom.touches(rem):
                        matches.append((row, col, grid_geom))
                if len(matches) == 0:
                    return
                threshold = reach_include[segnum]
                # Build a tiny DataFrame for just the remaining coordinates
                rem_c = pd.DataFrame({
                    'pt': [Point(c) for c in rem.coords[:]]
                })
                if len(matches) == 1:  # merge it with adjacent cell
                    row, col, grid_geom = matches[0]
                    mdist = rem_c['pt'].apply(
                                    lambda p: grid_geom.distance(p)).max()
                    if mdist > threshold:
                        self.logger.debug(
                            'remaining line segment from %s too far away to '
                            'merge (%.1f > %.1f)', segnum, mdist, threshold)
                        return
                    append_reach_df(segnum, line, row, col, rem)
                elif len(matches) == 2:  # complex: need to split it
                    if len(rem_c) == 2:
                        # If this is a simple line with two coords, split it
                        rem_c.index = [0, 2]
                        rem_c.loc[1] = {
                            'pt': rem.interpolate(0.5, normalized=True)}
                        rem_c.sort_index(inplace=True)
                        rem = LineString(list(rem_c['pt']))  # rebuild
                    # first match assumed to be touching the start of the line
                    if rem_c.at[0, 'pt'].touches(matches[1][2]):
                        matches.reverse()
                    rem_c['d1'] = rem_c['pt'].apply(
                                    lambda p: p.distance(matches[0][2]))
                    rem_c['d2'] = rem_c['pt'].apply(
                                    lambda p: p.distance(matches[1][2]))
                    rem_c['dm'] = rem_c[['d1', 'd2']].min(1)
                    mdist = rem_c['dm'].max()
                    if mdist > threshold:
                        self.logger.debug(
                            'remaining line segment from %s too far away to '
                            'merge (%.1f > %.1f)', segnum, mdist, threshold)
                        return
                    # try a simple split where distances switch
                    ds = rem_c['d1'] < rem_c['d2']
                    idx = ds[ds].index[-1]
                    # ensure it's not the index of either end
                    if idx == 0:
                        idx = 1
                    elif idx == len(rem_c) - 1:
                        idx = len(rem_c) - 2
                    row, col = matches[0][0:2]
                    rem1 = LineString(rem.coords[:(idx + 1)])
                    append_reach_df(segnum, line, row, col, rem1)
                    row, col = matches[1][0:2]
                    rem2 = LineString(rem.coords[idx:])
                    append_reach_df(segnum, line, row, col, rem2)
                else:
                    self.logger.error(
                        'how does this happen? Segments from %d touching %d '
                        'grid cells', segnum, len(matches))
            elif rem.geom_type.startswith('Multi'):
                for sub_rem_geom in rem.geoms:  # recurse
                    reassign_reach(segnum, line, sub_rem_geom)
            else:
                raise NotImplementedError(reach_geom.geom_type)

        drop_reach_ids = []
        for segnum, line in self.segments.geometry.iteritems():
            remaining_line = line
            if grid_sindex:
                bbox_match = sorted(grid_sindex.intersection(line.bounds))
                if not bbox_match:
                    continue
                sub = grid_cells.geometry.iloc[bbox_match]
            else:  # slow scan of all cells
                sub = grid_cells.geometry
            for (row, col), grid_geom in sub.iteritems():
                reach_geom = grid_geom.intersection(line)
                if not reach_geom.is_empty and reach_geom.geom_type != 'Point':
                    remaining_line = remaining_line.difference(grid_geom)
                    append_reach(segnum, row, col, line, reach_geom)
                    if ibound_action == 'modify' and ibound[row, col] == 0:
                        ibound_modified += 1
                        ibound[row, col] = 1
            if line is not remaining_line and remaining_line.length > 0:
                # Determine if any remaining portions of the line can be used
                reassign_reach(segnum, line, remaining_line)
            # Potentially merge a few reaches for each segnum/row/col
            gb = reach_df.loc[reach_df['segnum'] == segnum]\
                .groupby(['row', 'col'])['geometry'].apply(list).copy()
            for (row, col), geoms in gb.iteritems():
                if len(geoms) > 1:
                    geom = linemerge(geoms)
                    if geom.geom_type == 'MultiLineString':
                        # workaround for odd floating point issue
                        geom = linemerge([wkt.loads(g.wkt) for g in geoms])
                    if geom.geom_type == 'LineString':
                        sel = ((reach_df['segnum'] == segnum) &
                               (reach_df['row'] == row) &
                               (reach_df['col'] == col))
                        drop_reach_ids += list(sel.index[sel])
                        self.logger.debug(
                            'merging %d reaches for segnum %s at (%s, %s)',
                            sel.sum(), segnum, row, col)
                        append_reach_df(segnum, line, row, col, geom)
                    else:
                        self.logger.debug(
                            'attempted to merge segnum %s at (%s, %s), however'
                            ' geometry was %s', segnum, row, col, geom.wkt)
        if drop_reach_ids:
            reach_df.drop(drop_reach_ids, axis=0, inplace=True)
        # TODO: Some reaches match multiple cells if they share a border
        self.reaches = geopandas.GeoDataFrame(
            reach_df, geometry='geometry', crs=crs)
        if ibound_action == 'modify':
            if ibound_modified:
                self.logger.debug(
                    'updating %d cells from IBOUND array for top layer',
                    ibound_modified)
                m.bas6.ibound[0] = ibound
                self.reaches = self.reaches.merge(
                    grid_df[['ibound']],
                    left_on=['row', 'col'], right_index=True)
                self.reaches.rename(
                        columns={'ibound': 'prev_ibound'}, inplace=True)
            else:
                self.reaches['prev_ibound'] = 1
        # Assign segment data
        self.segments['min_slope'] = self.segment_series(min_slope)
        if (self.segments['min_slope'] < 0.0).any():
            raise ValueError('min_slope must be greater than zero')
        # Column names common to segments and segment_data
        segment_cols = [
            'roughch',
            'hcond1', 'thickm1', 'elevup', 'width1',
            'hcond2', 'thickm2', 'elevdn', 'width2']
        # Tidy any previous attempts
        for col in segment_cols:
            if col in self.segments.columns:
                del self.segments[col]
        # Combine pairs of series for each segment
        self.segments = pd.concat([
            self.segments,
            self.pair_segment_values(hyd_cond1, hyd_cond_out, 'hcond'),
            self.pair_segment_values(thickness1, thickness_out, 'thickm'),
            self.pair_segment_values(width1, width_out, name='width')
        ], axis=1, copy=False)
        self.segments['roughch'] = self.segment_series(roughch)
        # Add information from segments
        self.reaches = self.reaches.merge(
            self.segments[['sequence', 'min_slope']], 'left',
            left_on='segnum', right_index=True)
        self.reaches.sort_values(['sequence', 'dist'], inplace=True)
        # Interpolate segment properties to each reach
        self.reaches['strthick'] = 0.0
        self.reaches['strhc1'] = 0.0
        for segnum, seg in self.segments.iterrows():
            sel = self.reaches['segnum'] == segnum
            if seg['thickm1'] == seg['thickm2']:
                val = seg['thickm1']
            else:  # linear interpolate to mid points
                tk1 = seg['thickm1']
                tk2 = seg['thickm2']
                dtk = tk2 - tk1
                val = dtk * self.reaches.loc[sel, 'dist'] + tk1
            self.reaches.loc[sel, 'strthick'] = val
            if seg['hcond1'] == seg['hcond2']:
                val = seg['hcond1']
            else:  # linear interpolate to mid points in log-10 space
                lhc1 = np.log10(seg['hcond1'])
                lhc2 = np.log10(seg['hcond2'])
                dlhc = lhc2 - lhc1
                val = 10 ** (dlhc * self.reaches.loc[sel, 'dist'] + lhc1)
            self.reaches.loc[sel, 'strhc1'] = val
        del self.reaches['sequence']
        # del self.reaches['dist']
        # Use MODFLOW SFR dataset 2 terms ISEG and IREACH, counting from 1
        self.reaches['iseg'] = 0
        self.reaches['ireach'] = 0
        iseg = ireach = 0
        prev_segnum = None
        for idx, segnum in self.reaches['segnum'].iteritems():
            if segnum != prev_segnum:
                iseg += 1
                ireach = 0
            ireach += 1
            self.reaches.at[idx, 'iseg'] = iseg
            self.reaches.at[idx, 'ireach'] = ireach
            prev_segnum = segnum
        self.reaches.reset_index(inplace=True, drop=True)
        self.reaches.index += 1
        self.reaches.index.name = 'reachID'  # starts at one
        self.reaches['strtop'] = 0.0
        self.reaches['slope'] = 0.0
        if self.has_z:
            for reachID, item in self.reaches.iterrows():
                geom = item.geometry
                # Get Z from each end
                z0 = geom.coords[0][2]
                z1 = geom.coords[-1][2]
                dz = z0 - z1
                dx = geom.length
                slope = dz / dx
                self.reaches.at[reachID, 'slope'] = slope
                # Get strtop from LineString mid-point Z
                zm = geom.interpolate(0.5, normalized=True).z
                self.reaches.at[reachID, 'strtop'] = zm
        else:
            r = self.reaches['row'].values
            c = self.reaches['col'].values
            # Estimate slope from top and grid spacing
            px, py = np.gradient(m.dis.top.array, col_size, row_size)
            grid_slope = np.sqrt(px ** 2 + py ** 2)
            self.reaches['slope'] = grid_slope[r, c]
            # Get stream values from top of model
            self.reaches['strtop'] = m.dis.top.array[r, c]
        # Enforce min_slope
        sel = self.reaches['slope'] < self.reaches['min_slope']
        if sel.any():
            self.logger.warning('enforcing min_slope for %d reaches (%.2f%%)',
                                sel.sum(), 100.0 * sel.sum() / len(sel))
            self.reaches.loc[sel, 'slope'] = self.reaches.loc[sel, 'min_slope']
        # Build reach_data for Data Set 2
        # See flopy.modflow.ModflowSfr2.get_default_reach_dtype()
        self.reach_data = pd.DataFrame(
            self.reaches[[
                'row', 'col', 'iseg', 'ireach',
                'strtop', 'slope', 'strthick', 'strhc1']])
        if not hasattr(self.reaches.geometry, 'geom_type'):
            # workaround needed for reaches.to_file()
            self.reaches.geometry.geom_type = self.reaches.geom_type
        self.reach_data.rename(columns={'row': 'i', 'col': 'j'}, inplace=True)
        self.reach_data.insert(0, column='k', value=0)  # only top layer
        self.reach_data.insert(5, column='rchlen', value=self.reaches.length)
        # Build segment_data for Data Set 6
        self.segment_data = self.reaches[['iseg', 'segnum']].drop_duplicates()
        self.segment_data['elevup'] = self.reaches.loc[
            self.segment_data.index].strtop
        self.segment_data['elevdn'] = self.reaches.loc[
            self.reaches.groupby(['iseg']).ireach.idxmax().values
        ].strtop.values
        self.segment_data.rename(columns={'iseg': 'nseg'}, inplace=True)
        self.segment_data.set_index('segnum', inplace=True)
        self.segment_data.index.name = self.segments.index.name
        self.segment_data['icalc'] = 1  # assumption
        # Translate 'to_segnum' to 'outseg' via 'nseg'
        self.segment_data['outseg'] = self.segment_data.index.map(
            lambda x: self.segment_data.loc[
                self.segments.loc[x, 'to_segnum'], 'nseg'] if
            self.segments.loc[
                x, 'to_segnum'] in self.segment_data.index else 0.).values
        self.segment_data['iupseg'] = 0  # no diversions (yet)
        self.segment_data['iprior'] = 0
        self.segment_data['flow'] = 0.0
        self.segment_data['runoff'] = 0.0
        self.segment_data['etsw'] = 0.0
        self.segment_data['pptsw'] = 0.0
        # copy several columns over (except 'elevup' and 'elevdn', for now)
        segment_cols.remove('elevup')
        segment_cols.remove('elevdn')
        self.segment_data[segment_cols] = self.segments[segment_cols]
        # Create an 'inflows' DataFrame calculated from combining 'inflow'
        inflows = pd.DataFrame(index=inflow.index)
        has_inflow = len(inflow.columns) > 0
        missing_inflow = 0
        self.segments['inflow_segnums'] = None
        # Determine upstream flows needed for each SFR segment
        sfr_segnums = set(self.segment_data.index)
        for segnum in self.segment_data.index:
            from_segnums = self.segments.at[segnum, 'from_segnums']
            if not from_segnums:
                continue
            # gather segments outside SFR network
            outside_segnums = from_segnums.difference(sfr_segnums)
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
                            'flow from segment %s not provided by inflow term '
                            '(needed for %s)', from_segnum, segnum)
                        missing_inflow += 1
                if inflow_segnums:
                    inflows[segnum] = inflow_series
                    self.segments.at[segnum, 'inflow_segnums'] = inflow_segnums
            else:
                missing_inflow += len(outside_segnums)
        if not has_inflow and missing_inflow > 0:
            self.logger.warning(
                'inflow from %d segnums are needed to determine flow from '
                'outside SFR network', missing_inflow)
        segment_data = {}
        has_inflows = len(inflows.columns) > 0
        has_flow = len(flow.columns) > 0
        has_runoff = len(runoff.columns) > 0
        has_etsw = len(etsw.columns) > 0
        has_pptsw = len(pptsw.columns) > 0
        for iper in range(m.dis.nper):
            # Store data for each stress period
            self.segment_data['flow'] = 0.0
            self.segment_data['runoff'] = 0.0
            self.segment_data['etsw'] = 0.0
            self.segment_data['pptsw'] = 0.0
            if has_inflows:
                item = inflows.iloc[iper]
                self.segment_data.loc[item.index, 'flow'] += item
            if has_flow:
                item = flow.iloc[iper]
                self.segment_data.loc[item.index, 'flow'] += item
            if has_runoff:
                item = runoff.iloc[iper]
                self.segment_data.loc[item.index, 'runoff'] = item
            if has_etsw:
                item = etsw.iloc[iper]
                self.segment_data.loc[item.index, 'etsw'] = item
            if has_pptsw:
                item = pptsw.iloc[iper]
                self.segment_data.loc[item.index, 'pptsw'] = item
            segment_data[iper] = self.segment_data.to_records(index=False)
        # Create flopy Sfr2 package
        flopy.modflow.mfsfr2.ModflowSfr2(
                model=m,
                reach_data=self.reach_data.to_records(index=True),
                segment_data=segment_data)
        self.model = m
        # For models with more than one stress period, evaluate summary stats
        if m.dis.nper > 1:
            # Remove time-varying data from last stress period
            self.segment_data.drop(
                ['flow', 'runoff', 'etsw', 'pptsw'], axis=1, inplace=True)

            def add_summary_stats(name, df):
                if len(df.columns) == 0:
                    return
                self.segment_data[name + '_min'] = 0.0
                self.segment_data[name + '_mean'] = 0.0
                self.segment_data[name + '_max'] = 0.0
                min_v = df.min(0)
                mean_v = df.mean(0)
                max_v = df.max(0)
                self.segment_data.loc[min_v.index, name + '_min'] = min_v
                self.segment_data.loc[mean_v.index, name + '_mean'] = mean_v
                self.segment_data.loc[max_v.index, name + '_max'] = max_v

            add_summary_stats('inflow', inflows)
            add_summary_stats('flow', flow)
            add_summary_stats('runoff', runoff)
            add_summary_stats('etsw', etsw)
            add_summary_stats('pptsw', pptsw)

    def get_seg_ijk(self):
        """
        This will just get the upstream and downstream segment k,i,j
        """
        topidx = self.reach_data['ireach'] == 1
        kij_df = self.reach_data[topidx][['iseg', 'k', 'i', 'j']].sort_values(
            'iseg')
        self.segment_data = self.segment_data.merge(
            kij_df, left_on='nseg', right_on='iseg', how='left').drop(
            'iseg', axis=1)
        self.segment_data.rename(
            columns={"k": "k_up", "i": "i_up", "j": "j_up"}, inplace=True)
        # seg bottoms
        btmidx = self.reach_data.groupby('iseg')['ireach'].transform(max) == \
                 self.reach_data['ireach']
        kij_df = self.reach_data[btmidx][['iseg', 'k', 'i', 'j']].sort_values(
            'iseg')

        self.segment_data = self.segment_data.merge(
            kij_df, left_on='nseg', right_on='iseg', how='left').drop(
            'iseg', axis=1)
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
        :return: Adds 'top_up' and 'top_dn' columns to segment data dataframe
        """
        if m is None:
            m = self.model
        assert m.sfr is not None, "need sfr package"
        self.segment_data['top_up'] = m.dis.top.array[
            tuple(self.segment_data[['i_up', 'j_up']].values.T)]
        self.segment_data['top_dn'] = m.dis.top.array[
            tuple(self.segment_data[['i_dn', 'j_dn']].values.T)]
        return self.segment_data[['top_up', 'top_dn']]

    def get_segment_incision(self):
        """
        Calculates the upstream and downstream incision of the segment
        :return:
        """
        self.segment_data['diff_up'] = \
            self.segment_data['top_up'] - self.segment_data['elevup']
        self.segment_data['diff_dn'] = \
            self.segment_data['top_dn'] - self.segment_data['elevdn']
        return self.segment_data[['diff_up', 'diff_dn']]

    def set_seg_minincise(self, minincise=0.2):
        """
        Set segment elevation so that they have the minumum incision
        from the top surface
        :param minincise: Desired minimum incision
        :return: incisions at the upstream and downstream end of each segment
        """
        sel = self.segment_data['diff_up'] < minincise
        self.segment_data.loc[sel, 'elevup'] = \
            self.segment_data.loc[sel, 'top_up'] - minincise
        sel = self.segment_data['diff_dn'] < minincise
        self.segment_data.loc[sel, 'elevdn'] = \
            self.segment_data.loc[sel, 'top_dn'] - minincise
        # recalculate incisions
        updown_incision = self.get_segment_incision()
        return updown_incision

    def get_segment_length(self):
        """
        Get segment length from accumulated reach lengths
        :return:
        """
        # extract segment length for calculating minimun drop later
        seglen = self.reach_data.groupby('iseg').rchlen.sum()
        self.segment_data = self.segment_data.merge(seglen, left_on='nseg',
                                                    right_index=True,
                                                    how='left')
        self.segment_data.rename(columns={'rchlen': "seglen"}, inplace=True)
        return self.segment_data['seglen']

    def get_outseg_elev(self):
        """Get the maximum elevup associated with all downstream segments
        for each segment"""
        self.segment_data['outseg_elevup'] = self.segment_data.outseg.apply(
            lambda x: self.segment_data.loc[
                self.segment_data.nseg == x].elevup).max(axis=1)
        return self.segment_data['outseg_elevup']

    def set_outseg_elev_for_seg(self, seg):
        """
        gets all the defined outseg_elevup associated with a specific segment
        (multiple upstream segements route to one segment)
        Returns a df with all the calculated outseg elevups for each segment
        .min(axis=1) is a good way to collapse to a series
        :param seg: Pandas Series containing one row of seg_data dataframe
        :return: Returns a df of the outseg_elev up values
        where current segment is listed as an outseg
        """
        # downstreambuffer = 0.001 # 1mm
        # find where seg is listed as outseg
        outsegsel = self.segment_data['outseg'] == seg.nseg
        # set outseg elevup
        outseg_elevup = self.segment_data.loc[outsegsel, 'outseg_elevup']
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
        if seg.outseg > 0.0:
            # select outflow segment for current seg and pull out elevup
            outsegsel = self.segment_data['nseg'] == seg.outseg
            outseg_elevup = self.segment_data.loc[outsegsel, 'elevup']
            down = outseg_elevup.values[0]
            if down >= seg.elevup - (seg.seglen * minslope):
                # downstream elevation too high
                dn = up - (seg.seglen * minslope)  # set to minslope
                outseg_up = up - (seg.seglen * minslope) - downstreambuffer
                print(f'Segment {int(seg.nseg)}, outseg = {int(seg.outseg)}, '
                      f'old outseg_elevup = {seg.outseg_elevup}, '
                      f'new outseg_elevup = {outseg_up}')
            else:
                dn = down
                outseg_up = down - downstreambuffer
        else:
            # must be an outflow segment
            down = seg.elevdn
            if down > up - (seg.seglen * minslope):
                dn = up - (seg.seglen * minslope)
                print(f'Outflow Segment {int(seg.nseg)}, '
                      f'outseg = {int(seg.outseg)}, '
                      f'old elevdn = {seg.elevdn}, '
                      f'new elevdn = {dn}')
            else:
                dn = down
        return pd.Series({'nseg': seg.nseg, 'elevdn': dn,
                          'outseg_elevup': outseg_up})  # this returns a DF once the apply is done!

    def set_forward_segs(self, min_slope=1.e-4):
        """
        Ensure slope of all segment is at least min_slope
        in the downstream direction.
        Moves down the network correcting downstream elevations if necessary
        :param min_slope: Desired minimum slope
        :return: and updated segment data df
        """
        ### upper most segments (not referenced as outsegs)
        # segdata_df = self.segment_data.sort_index(axis=1)
        segsel = ~self.segment_data['nseg'].isin(self.segment_data['outseg'])
        while segsel.sum() > 0:
            print(
                f'Checking elevdn and outseg_elevup '
                f'for {int(segsel.sum())} segments')
            # get elevdn and outseg_elevups with a minimum slope constraint
            # index should align with self.segment_data index
            # not applying directly allows us to filter out nans
            tmp = self.segment_data.loc[segsel].apply(
                self.minslope_seg, args=[min_slope], axis=1)
            ednsel = tmp[tmp['elevdn'].notna()].index
            oeupsel = tmp[tmp['outseg_elevup'].notna()].index
            # set elevdn and outseg_elevup
            self.segment_data.loc[ednsel, 'elevdn'] = tmp.loc[ednsel, 'elevdn']
            self.segment_data.loc[oeupsel, 'outseg_elevup'] = \
                tmp.loc[oeupsel, 'outseg_elevup']
            # get `elevups` for outflow segs from `outseg_elevups`
            # taking `min` ensures outseg elevup is below all inflow elevdns
            tmp2 = self.segment_data.apply(
                self.set_outseg_elev_for_seg, axis=1).min(axis=1)
            tmp2 = pd.DataFrame(tmp2, columns=['elevup'])
            # update `elevups`
            eupsel = tmp2[tmp2.loc[:, 'elevup'].notna()].index
            self.segment_data.loc[eupsel, 'elevup'] = tmp2.loc[eupsel, 'elevup']
            # get list of next outsegs
            segsel = self.segment_data['nseg'].isin(
                self.segment_data.loc[segsel, 'outseg'])
        return self.segment_data

    def fix_segment_elevs(self, min_incise=0.2, min_slope=1.e-4):
        """
        Wrapper function for calculating SFR segment elevations.
        Calls series of functions to process and move sfr segment elevations,
        to try to ensure:
            0. Segments are below the model top
            1. Segments flow downstream
            2. Downstream segments are below upstream segments
        :param min_slope: desired minimum slope for segment
        :param min_incise: desired minimum incision (in model units)
        :return: segment data dataframe
        """
        # get model locations for segments ends
        _ = self.get_seg_ijk()
        # get model cell elevations at seg ends
        _ = self.get_top_elevs_at_segs()
        # get current segment incision at seg ends
        _ = self.get_segment_incision()
        # move segments end elevation down to achieve minimum incision
        _ = self.set_seg_minincise(minincise=min_incise)
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
        recalculate reach strtop elevations after moving segment elevations
        :return: Reach data df
        """
        def reach_elevs(seg):
            """
            Calculate reach elevation from segment slope and
            reach length along segment.
            :param seg: one row of reach data dataframe grouped by segment
            :return: reaches by segment with strtop adjusted
            """
            segsel = self.segment_data['nseg'] == seg.name

            seg_elevup = self.segment_data.loc[segsel, 'elevup'].values[0]
            seg_slope = self.segment_data.loc[segsel, 'Zslope'].values[0]

            # interpolate reach lengths to cell centres
            cmids = seg.seglen.shift().fillna(0.0) + seg.rchlen.multiply(0.5)
            cmids.iat[0] = 0.0
            cmids.iat[-1] = seg.seglen.iloc[-1]
            seg['cmids'] = cmids  # cummod+(seg.rchlen *0.5)
            # calculate reach strtops
            seg['strtop'] = seg['cmids'].multiply(seg_slope) + seg_elevup
            # seg['slope']= #!!!!! use m.sfr.get_slopes() method
            return seg
        self.segment_data['Zslope'] = (self.segment_data['elevdn'] -
                                       self.segment_data['elevup']) / \
                                       self.segment_data['seglen']
        segs = self.reach_data.groupby('iseg')
        self.reach_data['seglen'] = segs.rchlen.cumsum()
        self.reach_data = segs.apply(reach_elevs)
        return self.reach_data

    def set_topbot_elevs_at_reaches(self, m=None):
        """
        get top and bottom elevation of the cell containing a reach
        :param m: Modflow model
        :return: dataframe with reach cell top and bottom elevations
        """
        if m is None:
            m = self.model
        self.reach_data['top'] = m.dis.top.array[
            tuple(self.reach_data[['i', 'j']].values.T)]
        self.reach_data['bot'] = m.dis.botm[0].array[
            tuple(self.reach_data[['i', 'j']].values.T)]
        return self.reach_data[['top', 'bot']]

    def fix_reach_elevs(self, minslope=0.0001, fix_dis=True, minthick=0.5):
        """
        need to ensure reach elevation is:
            0. below the top
            1. below the upstream reach
            2. above the minimum slope to the bottom reach elevation
            3. above the base of layer 1
        segment by segment, reach by reach! Fun!

        :return:
        """
        buffer = 1.0  # 1 m (buffer to leave at the base of layer 1 -
        # also helps with precision issues)
        # top read from dis as float32 so comparison need to be with like
        reachsel = self.reach_data['top'] <= self.reach_data['strtop']
        print('{} segments with reaches above model top'.format(
            self.reach_data[reachsel]['iseg'].unique().shape[0]))
        # get segments with reaches above the top surface
        segsabove = self.reach_data[reachsel].groupby(
            'iseg').size().sort_values(ascending=False)
        # get incision gradient from segment elevups and elevdns
        # ('diff_up' and 'diff_dn' are the incisions of the top and
        # bottom reaches from the segment data)
        self.segment_data['incgrad'] = (self.segment_data['diff_up'] -
                                        self.segment_data['diff_dn']) / \
                                       self.segment_data['seglen']
        # copy of layer 1 bottom (for updating to fit in stream reaches)
        layerbots = self.model.dis.botm.array.copy()
        # loop over each segment
        for seg in self.segment_data['nseg'].values:  # (all segs)
            # selection for segment in reachdata and seg data
            rsel = self.reach_data['iseg'] == seg
            segsel = self.segment_data['nseg'] == seg

            if seg in segsabove:
                # check top and bottom reaches are above layer 1 bottom
                # (not adjusting elevations of reaches)
                for reach in self.reach_data[rsel].iloc[[0, -1]].itertuples():
                    if reach.strtop - reach.strthick < reach.bot + buffer:
                        # drop bottom of layer one to accomodate stream
                        print(f'seg {seg} reach {reach.ireach} '
                              f'is below layer 1 bottom')
                        print(
                            f'dropping layer 1 bottom to '
                            f'{reach.strtop - reach.strthick - buffer} '
                            f'to accomodate stream @ i = '
                            f'{reach.i}, j = {reach.j}')
                        layerbots[0, reach.i, reach.j] = \
                            reach.strtop - reach.strthick - buffer
                # apparent optimised incision based
                # on the incision gradient for the segment
                self.reach_data.loc[rsel, 'strtop_incopt'] = \
                    self.reach_data.loc[rsel, 'top'].subtract(
                        self.segment_data.loc[segsel, 'diff_up'].values[0]) + \
                    (self.reach_data.loc[rsel, 'cmids'].subtract(
                        self.reach_data.loc[rsel, 'cmids'].values[0]) *
                     self.segment_data.loc[segsel, 'incgrad'].values[0])
                # falls apart when the top elevation is not monotonically
                # decreasing down the segment (/always!)

                # bottom reach elevation:
                botreach_strtop = self.reach_data[rsel]['strtop'].values[-1]
                # total segment length
                seglen = self.reach_data[rsel]['seglen'].values[-1]
                botreach_slope = minslope  # minimum slope of segment
                # top reach elevation and "length?":
                upreach_strtop = self.reach_data[rsel]['strtop'].values[0]
                upreach_cmid = self.reach_data[rsel]['cmids'].values[0]
                # use top reach as starting point

                # loop over reaches in segement from second to penultimate
                # (dont want to move elevup or elevdn)
                for reach in self.reach_data[rsel][1:-1].itertuples():
                    # strtop that would result from minimum slope
                    # from upstream reach
                    strtop_withminslope = upreach_strtop - (
                            (reach.cmids - upreach_cmid) * minslope)
                    # strtop that would result from minimum slope
                    # from bottom reach
                    strtop_min2bot = botreach_strtop + (
                            (seglen - reach.cmids) * minslope)
                    # check 'optimum incision' is below upstream elevation
                    # and above the minimum slope to the bottom reach
                    if reach.strtop_incopt < strtop_min2bot:
                        # strtop would give too shallow a slope to
                        # the bottom reach (not moving bottom reach)
                        print(f'seg {seg} reach {reach.ireach}, '
                              f'incopt is \\/ below minimum slope '
                              f'from bottom reach elevation')
                        print('setting elevation to minslope from bottom')
                        # set to minimum slope from outreach
                        self.reach_data.at[
                            reach.Index, 'strtop'] = strtop_min2bot
                        # update upreach for next iteration
                        upreach_strtop = strtop_min2bot
                    elif reach.strtop_incopt > strtop_withminslope:
                        # strtop would be above upstream or give
                        # too shallow a slope from upstream
                        print(f'seg {seg} reach {reach.ireach}, '
                              f'incopt /\\ above upstream')
                        print('setting elevation to minslope from upstream')
                        # set to minimum slope from upstream reach
                        self.reach_data.at[
                            reach.Index, 'strtop'] = strtop_withminslope
                        # update upreach for next iteration
                        upreach_strtop = strtop_withminslope
                    else:
                        # strtop might be ok to set to 'optimum incision'
                        print(f'seg {seg} reach {reach.ireach}, '
                              f'incopt is -- below upstream reach and above '
                              f'the bottom reach')
                        # CHECK FIRST:
                        # if optimium incision would place it
                        # below the bottom of layer 1
                        if reach.strtop_incopt - reach.strthick < \
                                reach.bot + buffer:
                            # opt - stream thickness lower than layer 1 bottom
                            # (with a buffer)
                            print(f'seg {seg} reach {reach.ireach}, '
                                  f'incopt - bot is x\\/ below layer 1 bottom')
                            if reach.bot + reach.strthick + buffer > \
                                    strtop_withminslope:
                                # if layer bottom would put reach above
                                # upstream reach we can only set to
                                # minimum slope from upstream
                                print('setting elevation to minslope '
                                      'from upstream')
                                self.reach_data.at[reach.Index, 'strtop'] = \
                                    strtop_withminslope
                                upreach_strtop = strtop_withminslope
                            else:
                                # otherwise we can move reach so that it
                                # fits into layer 1
                                print(f'setting elevation to '
                                      f'{reach.bot + reach.strthick + buffer}'
                                      f', above layer 1 bottom')
                                # set reach top so that it is above layer 1
                                # bottom with a buffer
                                # (allowing for bed thickness)
                                self.reach_data.at[reach.Index, 'strtop'] = \
                                    reach.bot + buffer + reach.strthick
                                upreach_strtop = reach.bot + buffer + \
                                                 reach.strthick
                        else:
                            # strtop ok to set to 'optimum incision'
                            # set to "optimum incision"
                            print('setting elevation to incopt')
                            self.reach_data.at[
                                reach.Index, 'strtop'] = reach.strtop_incopt
                            upreach_strtop = reach.strtop_incopt
                    # check if new stream top is above layer 1 with a buffer
                    # (allowing for bed thickness)
                    if upreach_strtop < reach.bot + buffer + reach.strthick:
                        # if new strtop is below layer one
                        # drop bottom of layer one to accomodate stream
                        # (top, bed thickness and buffer)
                        print(f'dropping layer 1 bottom to '
                              f'{upreach_strtop - reach.strthick - buffer} '
                              f'to accomodate stream @ i = '
                              f'{reach.i}, j = {reach.j}')
                        layerbots[0, reach.i, reach.j] = \
                            upreach_strtop - reach.strthick - buffer
                    upreach_cmid = reach.cmids
                    # upreach_slope=reach.slope
            else:
                # For segments that do not have reaches above top
                # check if reaches are below layer 1
                print(f'seg {seg} is always downstream and below the top')
                for reach in self.reach_data[rsel].itertuples():
                    if reach.strtop < reach.bot + buffer + reach.strthick:
                        # strtop is below layer one
                        # drop bottom of layer one to accomodate stream
                        # (top, bed thickness and buffer)
                        print(f'seg {seg} reach {reach.ireach} '
                              f'is below layer 1 bottom')
                        print(f'dropping layer 1 bottom to '
                              f'{reach.strtop - reach.strthick - buffer} '
                              f'to accomodate stream @ i = '
                              f'{reach.i}, j = {reach.j}')
                        layerbots[0, reach.i, reach.j] = \
                            reach.strtop - reach.strthick - buffer
            print('')  # printout spacer
        if fix_dis:
            # fix dis for incised reaches
            for lay in range(self.model.dis.nlay - 1):
                laythick = layerbots[lay] - layerbots[
                    lay + 1]  # first one is layer 1 bottom - layer 2 bottom
                print('checking layer {} thicknesses'.format(lay + 2))
                thincells = laythick < minthick
                print('{} cells less than {}'.format(thincells.sum(), minthick))
                laythick[thincells] = minthick
                layerbots[lay + 1] = layerbots[lay] - laythick
            self.model.dis.botm = layerbots

    def sfr_plot(self, model, sfrar, dem, points=None, points2=None,
                 label=None):
        mprj = ccrs.epsg(2193)
        crs_longlat = ccrs.PlateCarree()
        extent = model.modelgrid.extent  # # get model spacial reference extent - local utm
        domain_extent = crs_longlat.transform_points(
            mprj, np.array(extent[0:2]),
            np.array(extent[2:])).T.flatten()[0:4]
        p = ModelPlot(model, domain_extent)
        p._add_plotlayer(dem, label="Elevation (m)")
        p._add_sfr(sfrar, cat_cmap=False, colorbar=True,
                   cbar_label=label)

    def plot_reaches_above(self, model, seg, dem=None,
                           plot_bottom=False, points2=None):
        dis = model.dis
        sfr = model.sfr
        if dem is None:
            dem = np.ma.array(dis.top.array, mask=model.bas6.ibound.array[0] == 0)
        sfrar = np.ma.zeros(dis.top.array.shape, 'f')
        sfrar.mask = np.ones(sfrar.shape)
        lay1reaches = self.reach_data.loc[self.reach_data.k.apply(lambda x: x == 1)]
        points = None
        if lay1reaches.shape[0] > 0:
            points = lay1reaches[['i', 'j']]
        # segsel=reachdata['iseg'].isin(segsabove.index)
        if seg == 'all':
            segsel = np.ones((self.reach_data.shape[0]), dtype=bool)
        else:
            segsel = self.reach_data['iseg'] == seg
        sfrar[tuple((self.reach_data[segsel][['i', 'j']].values.T).tolist())] = \
            (self.reach_data[segsel]['top'] - self.reach_data[segsel][
                'strtop']).tolist()  # .mask = np.ones(sfrar.shape)
        self.sfr_plot(model, sfrar, dem, points=points, points2=points2,
                      label="str below top (m)")
        if seg != 'all':
            sfr.plot_path(seg)
        if plot_bottom:
            dembot = np.ma.array(dis.botm.array[0],
                                 mask=model.bas6.ibound.array[0] == 0)
            sfrarbot = np.ma.zeros(dis.botm.array[0].shape, 'f')
            sfrarbot.mask = np.ones(sfrarbot.shape)
            sfrarbot[
                tuple((self.reach_data[segsel][['i', 'j']].values.T).tolist())] = \
                (self.reach_data[segsel]['strtop'] - self.reach_data[segsel][
                    'bot']).tolist()  # .mask = np.ones(sfrar.shape)
            self.sfr_plot(model, sfrarbot, dembot, points=points,
                          points2=points2, label="str above bottom (m)")


class ModelPlot(object):
    """
    Object for plotting array style results
    """

    def __init__(self, model, domain_extent, fig=None, ax=None, figsize=None):
        """
        Container to help with results plotting functions
        :param model: flopy modflow model object
        :param fig: matplotlib figure handle
        :param ax: Cartopy GeoAxes with .projection attribute
        :param domain_extent: np.array of array([long-left , long-right, lat-up, lat-dn])
        """

        self.model = model
        self.domain_extent = domain_extent
        self.xg = self.model.modelgrid.xvertices
        self.yg = self.model.modelgrid.yvertices
        self.extent = self.model.modelgrid.extent

        if figsize is None:
            figsize = (8, 8)
        # if no figure or axes based initialise figure
        if fig is None or ax is None:
            plt.rc('font', size=10)
            try:
                self.mprj = ccrs.epsg(2193) #TODO - flexi
            except:
                print("error getting model projection")
                exit()
            crs_longlat = ccrs.PlateCarree()
            self.fig, self.ax = plt.subplots(figsize=figsize,
                                             subplot_kw=dict(
                                                 projection=self.mprj))  # empty figure container cartopy geoaxes
        else:
            self.fig = fig
            self.ax = ax
            self.mprj = self.ax.projection  # map projection
        # self._get_base_ax()
        self._set_divider()

    # def _get_base_ax(self):
    #     """
    #     Define the plot axis based on the extent of the domain
    #     """
    #     import cartopy.io.img_tiles as cimgt
    #
    #     terrain = cimgt.Stamen('terrain-background')
    #     self.ax.set_extent(self.domain_extent)
    #
    #     # Add the Stamen data at zoom level 8.
    #     self.ax.add_image(terrain, 9, cmap="Greys")
    #
    #     # rivers = cartopy.feature.NaturalEarthFeature(
    #     #     category='physical', name='rivers_lake_centerlines',
    #     #     scale='10m', facecolor='none', edgecolor='b')
    #     # ax.add_feature(rivers, linewidth=0.5,)#,transform=ccrs.PlateCarree())
    #     self._label_utm_grid()

    # def _label_utm_grid(self):
    #     """
    #     Label axes as UTM
    #     Warning: should only use with small area UTM maps
    #     """
    #     for val, label in zip(self.ax.get_xticks(), self.ax.get_xticklabels()):
    #         label.set_text(str(val))
    #         label.set_position((val, 0))
    #         label.set_rotation(90)
    #
    #     for val, label in zip(self.ax.get_yticks(), self.ax.get_yticklabels()):
    #         label.set_text(str(val))
    #         label.set_position((0, val))
    #
    #     self.ax.tick_params(bottom=True, top=True, left=True, right=True,
    #                         labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    #
    #     self.ax.xaxis.set_visible(True)
    #     self.ax.yaxis.set_visible(True)
    #     self.ax.set_xlabel("Easting ($m$)")
    #     self.ax.set_ylabel("Northing ($m$)")
    #     self.ax.grid(True)

    @staticmethod
    def _get_range(k):
        kmin = np.min(k)
        kmax = np.max(k)
        krange = kmax - kmin
        if krange < np.abs(0.05 * ((kmin + kmax)/2)):
            vmin = ((kmin + kmax)/2) - np.abs(0.025 * ((kmin + kmax)/2))
            vmax = ((kmin + kmax)/2) + np.abs(0.025 * ((kmin + kmax)/2))
        else:
            vmin = kmin
            vmax = kmax
        return vmin, vmax

    def _get_cbar_props(self):
        """
        Get properties for next colorbar and its label based on the number of axis already in plot
        :return: divider padding, label locations for use when appending divider axes and setting colorbar label.
        """
        numax = len(self.ax.figure.axes)
        if np.mod(numax, 2) == 1:
            # odd number of axes so even number of cbars
            if numax == 1:
                # no colorbar yet - small pad
                divider_props = dict(pad=0.2)
            else:
                divider_props = dict(pad=0.5)
            props = dict(labelpad=-40, y=1.01, rotation=0, ha='center', va='bottom')
        else:
            divider_props = dict(pad=0.5)
            props = dict(labelpad=-40, y=-0.01, rotation=0, ha='center', va='top')
        return divider_props, props

    def _set_divider(self):
        """
        initiate a mpl divider for the colorbar location
        """
        self.divider = make_axes_locatable(self.ax)

    def _add_ibound_mask(self, lay, zorder=20, alpha=0.5):
        """
        Add the ibound to plot
        :param lay: layer for selecting model ibound
        :param zorder: mpl plotting overlay order
        :param alpha: mpl transparency
        """
        array = np.ones((self.model.nrow, self.model.ncol))
        array = np.ma.masked_where(self.model.bas6.ibound.array[lay] != 0, array)
        self.ax.imshow(array, extent=self.extent, transform=self.mprj, cmap="Greys_r",
                       origin="upper", zorder=zorder, alpha=alpha)

    def _add_plotlayer(self, k, zorder=10, alpha=0.8, label=None, get_range=False):
        """
        Add image for head
        :param k: 2D numpy array
        :param zorder: mpl overlay order
        :param alpha: mpl transparency
        """
        if get_range:
            vmin, vmax = self._get_range(k)
        else:
            vmin, vmax = [None, None]
        if label is None:
            print("No label passed for colour bar")
            label = ""
        hax = self.ax.imshow(k, zorder=zorder, extent=self.extent, origin="upper", transform=self.mprj, alpha=alpha,
                             vmin=vmin, vmax=vmax)
        divider_props, props = self._get_cbar_props()
        cax = self.divider.append_axes("right", size="5%", axes_class=plt.Axes, **divider_props)
        cbar1 = self.fig.colorbar(hax, cax=cax)
        cbar1.set_label(label, **props)

    def _add_sfr(self, x, zorder=11, colorbar=True, cat_cmap=False,
                 cbar_label=None, cmap_txt='bwr_r',
                 points=None, points2=None):
        """
        Plot the array of surface water exchange (with SFR)
        :param x: 2D numpy array
        :param zorder: mpl overlay order
        """
        import seaborn as sns
        vmin = x.min()
        vmax = x.max()
        if cat_cmap:
            vals = np.ma.unique(x).compressed()
            bounds = np.append(np.sort(vals), vals.max() + 1)
            n = len(vals)
            cmap = colors.ListedColormap(
                sns.color_palette('Set2', n).as_hex())
            norm = colors.BoundaryNorm(bounds, cmap.N)
        else:
            cmap = cm.get_cmap(cmap_txt)
            norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
        imx = self.ax.imshow(x, extent=self.extent, transform=self.mprj,
                             cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
                             zorder=zorder, origin='upper')
        if points is not None:
            self.ax.scatter(self.model.modelgrid.xcellcenters[
                                points.i, points.j],
                            self.model.modelgrid.ycellcenters[
                                points.i, points.j],
                            marker='o', zorder=15, facecolors='none',
                            edgecolors='r')
        if points2 is not None:
            self.ax.scatter(self.model.modelgrid.xcellcenters[
                                points2.i, points2.j],
                            self.model.modelgrid.xcellcenters[
                                points2.i, points2.j],
                            marker='o', zorder=15, facecolors='none',
                            edgecolors='b')
        if colorbar:
            divider_props, props = self._get_cbar_props()
            cax = self.divider.append_axes("right", size="5%",
                                           axes_class=plt.Axes,
                                           **divider_props)
            cbar1 = self.fig.colorbar(imx, cax=cax)
            cbar1.set_label(cbar_label, **props)
            if cat_cmap:
                # imsfr.set_clim(1, n + 1)
                cbar1.set_ticks(bounds[:-1] + np.diff(bounds) / 2)
                cbar1.set_ticklabels(np.sort(vals).astype(int))




def sfr_rec_to_df(sfr):
    """
    Convert flopy rec arrays for ds2 and ds6 to pandas dataframes
    """
    d = sfr.segment_data
    # multi index
    reform = {(i, j): d[i][j] for i in d.keys() for j in d[i].dtype.names}
    segdatadf = pd.DataFrame.from_dict(reform)
    segdatadf.columns.names = ['kper', 'col']
    reachdatadf = pd.DataFrame.from_records(sfr.reach_data)
    return segdatadf, reachdatadf


def sfr_dfs_to_rec(model, segdatadf, reachdatadf, set_outreaches=False,
                   get_slopes=True, minslope=None):
    """
    Function to convert sfr ds6 (seg data) and ds2 (reach data) to model.sfr
    rec arrays option to update slopes from reachdata dataframes
    """
    if get_slopes:
        print('Getting slopes')
        if minslope is None:
            minslope = 1.0e-4
            print('using default minslope of {}'.format(minslope))
        else:
            print('using specified minslope of {}'.format(minslope))
    # segs ds6
    # multiindex
    g = segdatadf.groupby(level=0, axis=1)  # group multi index df by kper
    model.sfr.segment_data = g.apply(
        lambda k: k.xs(k.name, axis=1).to_records(index=False)).to_dict()
    # # reaches ds2
    model.sfr.reach_data = reachdatadf.to_records(index=False)
    if set_outreaches:
        # flopy method to set/fix outreaches from segment routing and reach number information
        model.sfr.set_outreaches()
    if get_slopes:
        model.sfr.get_slopes(minimum_slope=minslope)
    # as of 08/03/2018 flopy plotting of sfr plots whatever is in stress_period_data
    # add
    model.sfr.stress_period_data.data[0] = model.sfr.reach_data


def topnet2ts(nc_path, varname, log_level=logging.INFO):
    """Read TopNet data from a netCDF file into a pandas.DataFrame timeseries

    User may need to multiply DataFrame to convert units.

    Parameters
    ----------
    nc_path : str
        File path to netCDF file
    varname : str
        Variable name in netCDF file to read
    verbose : int, optional
        Level used by logging module; default is 20 (logging.INFO)

    Returns
    -------
    pandas.DataFrame
        Where columns is rchid and index is DatetimeIndex.
    """
    try:
        from netCDF4 import Dataset, num2date
    except ImportError:
        raise ImportError('this function requires netCDF4')
    logger = logging.getLogger('topnet2ts')
    logger.handlers = module_logger.handlers
    logger.setLevel(log_level)
    logger.info('reading file:%s', nc_path)
    with Dataset(nc_path, 'r') as nc:
        var = nc.variables[varname]
        logger.info('variable %s: %s', varname, var)
        assert len(var.shape) == 3
        assert var.shape[-1] == 1
        df = pd.DataFrame(var[:, :, 0])
        df.columns = nc.variables['rchid']
        time_v = nc.variables['time']
        df.index = pd.DatetimeIndex(num2date(time_v[:], time_v.units))
    logger.info('data successfully read')
    return df

