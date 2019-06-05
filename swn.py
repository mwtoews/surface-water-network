# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
import geopandas
try:
    from geopandas.tools import sjoin
except ImportError:
    sjoin = False
from fiona import crs as fiona_crs
from math import sqrt
from shapely.geometry import LineString, Point, Polygon, box
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
        formatter = logging.Formatter(logging.BASIC_FORMAT)
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


class SurfaceWaterNetwork(object):
    """Surface water network

    Attributes
    ----------
    segments : geopandas.GeoDataFrame
        Primary GeoDataFrame created from 'lines' input, containing
        attributes evaluated during initialisation. Index is treated as a
        segment number or ID.
    index : pandas.core.index.Int64Index
        Shortcut property to segments.index or segment number.
    END_SEGNUM : int
        Special segment number that indicates a line end, default is usually 0.
        This number is not part of the index.
    upstream_segnums : dict
        Key is downstream segment number, and values are a set of zero or more
        upstream segment numbers. END_SEGNUM and headwater segment numbers are
        not included.
    has_z : bool
        Property that indicates all segment lines have Z dimension coordinates.
    headwater : pandas.core.index.Int64Index
        Head water segment numbers at top of cachment.
    outlets : pandas.core.index.Int64Index
        Index segment numbers for each outlet.
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
    upstream_segnums = None
    logger = None
    warnings = None
    errors = None

    def __len__(self):
        return len(self.index)

    def __init__(self, lines, logger=None):
        """
        Initialise SurfaceWaterNetwork and evaluate segments

        Parameters
        ----------
        lines : geopandas.GeoSeries or geopandas.GeoDataFrame
            Input lines of surface water network. Geometries must be
            'LINESTRING' or 'LINESTRING Z'. Index is used for segment numbers.
            The geometry is copied to the segments property.
        logger : logging.Logger, optional
            Logger to show messages.
        """
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.handlers = module_logger.handlers
            self.logger.setLevel(module_logger.level)
        if isinstance(lines, geopandas.GeoSeries):
            lines = lines.copy()
        elif isinstance(lines, geopandas.GeoDataFrame):
            lines = lines.geometry.copy()
        else:
            raise ValueError('lines must be a GeoDataFrame or GeoSeries')
        if len(lines) == 0:
            raise ValueError('one or more lines are required')
        elif not (lines.geom_type == 'LineString').all():
            raise ValueError('lines must all be LineString types')
        # Create a new GeoDataFrame with a copy of line's geometry
        self.segments = geopandas.GeoDataFrame(geometry=lines)
        self.logger.info('creating network with %d segments', len(self))
        segments_sindex = get_sindex(self.segments)
        if self.index.min() > 0:
            self.END_SEGNUM = 0
        else:
            self.END_SEGNUM = self.index.min() - 1
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
                self.segments.loc[segnum1, 'to_segnum'] = to_segnums[0]

        outlets = self.outlets
        self.logger.debug('evaluating segments upstream from %d outlet%s',
                          len(outlets), 's' if len(outlets) != 1 else '')
        self.segments['cat_group'] = self.END_SEGNUM
        self.segments['num_to_outlet'] = 0
        self.segments['length_to_outlet'] = 0.0

        # Recursive function that accumulates information upstream
        def resurse_upstream(segnum, cat_group, num, length):
            self.segments.loc[segnum, 'cat_group'] = cat_group
            num += 1
            self.segments.loc[segnum, 'num_to_outlet'] = num
            length += self.segments.geometry[segnum].length
            self.segments.loc[segnum, 'length_to_outlet'] = length
            # Branch to zero or more upstream segments
            for upsegnum in self.index[self.segments['to_segnum'] == segnum]:
                resurse_upstream(upsegnum, cat_group, num, length)

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
        search_order = ['num_to_outlet', 'length_to_outlet']
        furthest_upstream = self.segments.loc[headwater]\
            .sort_values(search_order, ascending=False).index
        sequence = pd.Series(
            np.arange(len(furthest_upstream)) + 1, index=furthest_upstream)
        self.segments.loc[sequence.index, 'sequence'] = sequence
        self.segments.loc[sequence.index, 'stream_order'] = 1
        # Build a dict that describes downstream segs to one or more upstream
        self.upstream_segnums = {}
        for segnum in set(self.segments['to_segnum'])\
                .difference([self.END_SEGNUM]):
            self.upstream_segnums[segnum] = \
                set(self.index[self.segments['to_segnum'] == segnum])
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
                if self.upstream_segnums[segnum].issubset(completed):
                    sequence += 1
                    self.segments.loc[segnum, 'sequence'] = sequence
                    # self.segments.loc[segnum, 'numiter'] = numiter
                    up_ord = list(
                        self.segments.loc[
                            list(self.upstream_segnums[segnum]),
                            'stream_order'])
                    max_ord = max(up_ord)
                    if up_ord.count(max_ord) > 1:
                        self.segments.loc[segnum, 'stream_order'] = max_ord + 1
                    else:
                        self.segments.loc[segnum, 'stream_order'] = max_ord
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
    def has_z(self):
        """Returns True if all segment lines have Z dimension"""
        return bool(self.segments.geometry.apply(lambda x: x.has_z).all())

    @property
    def index(self):
        """Returns Int64Index pandas index from segments"""
        return self.segments.index

    @property
    def headwater(self):
        """Returns index of headwater segments"""
        return self.index[~self.index.isin(self.segments['to_segnum'])]

    @property
    def outlets(self):
        """Returns index of outlets"""
        return self.index[self.segments['to_segnum'] == self.END_SEGNUM]

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
        elif (len(values.index) != len(self.index) or
                not (values.index == self.index).all()):
            raise ValueError('index is different')
        accum = values.copy()
        try:
            accum.name = 'accumulated_' + accum.name
        except TypeError:
            pass
        for segnum in self.segments.sort_values('sequence').index:
            if segnum in self.upstream_segnums:
                upstream_segnums = list(self.upstream_segnums[segnum])
                if upstream_segnums:
                    accum[segnum] += accum[upstream_segnums].sum()
        return accum

    def segment_series(self, value):
        """Returns a pandas.Series along the segment index

        Parameters
        ----------
        value : float or pandas.Series
            If value is a float, it is repated for each segment, otherwise
            a Series is used, with a check to ensure it is the same index.

        Returns
        -------
        pandas.Series
        """
        if not isinstance(value, pd.Series):
            value = pd.Series(value, index=self.index)
        elif (len(value.index) != len(self.index) or
                not (value.index == self.index).all()):
            raise ValueError('index is different than for segments')
        return value

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
                self.segments.loc[segnum, geom_name] = LineString(coords)
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

    def process_flopy(self, m, ibound_action='freeze', min_slope=1./1000):
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
        self.segments['min_slope'] = self.segment_series(min_slope)
        if (self.segments['min_slope'] < 0.0).any():
            raise ValueError('min_slope must be greater than zero')
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
        crs = None
        if m.modelgrid.proj4 is not None:
            crs = fiona_crs.from_string(m.modelgrid.proj4)
        elif self.segments.geometry.crs is not None:
            crs = self.segments.geometry.crs
        self.grid_cells = grid_cells = geopandas.GeoDataFrame(
                grid_df, crs=crs,
                geometry=[Polygon(m.modelgrid.get_cell_vertices(row, col))
                          for row, col in grid_df.index])
        grid_sindex = get_sindex(grid_cells)
        # Make an empty DataFrame for reaches
        reach_df = pd.DataFrame(columns=['geometry'])
        reach_df.insert(1, column='segnum',
                        value=pd.Series(dtype=self.index.dtype))
        reach_df.insert(2, column='dist', value=pd.Series(dtype=float))
        reach_df.insert(3, column='row', value=pd.Series(dtype=int))
        reach_df.insert(4, column='col', value=pd.Series(dtype=int))

        def append_reach(segnum, row, col, line, reach_geom):
            if reach_geom.geom_type == 'LineString':
                if line.has_z:
                    # intersection(line) does not preserve Z coords,
                    # but line.interpolate(d) works as expected
                    reach_geom = LineString(line.interpolate(
                        line.project(Point(c))) for c in reach_geom.coords)
                reach_df.loc[len(reach_df) + 1] = {
                    'geometry': reach_geom,
                    'segnum': segnum,
                    'dist': line.project(Point(reach_geom.coords[0])),
                    'row': row,
                    'col': col,
                }
            elif reach_geom.geom_type.startswith('Multi'):
                for sub_reach_geom in reach_geom.geoms:  # recurse
                    append_reach(segnum, row, col, line, sub_reach_geom)
            else:
                raise NotImplementedError(reach_geom.geom_type)

        for segnum, line in self.segments.geometry.iteritems():
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
                    append_reach(segnum, row, col, line, reach_geom)
                    if ibound_action == 'modify' and ibound[row, col] == 0:
                        ibound_modified += 1
                        ibound[row, col] = 1
        # Some reaches match multiple cells if they share a border
        self.reaches = geopandas.GeoDataFrame(reach_df, geometry='geometry')
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
        # Add information from segments
        self.reaches = self.reaches.merge(
            self.segments[['sequence', 'min_slope']], 'left',
            left_on='segnum', right_index=True)
        self.reaches.sort_values(['sequence', 'dist'], inplace=True)
        del self.reaches['sequence']
        del self.reaches['dist']
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
            self.reaches.loc[idx, 'iseg'] = iseg
            self.reaches.loc[idx, 'ireach'] = ireach
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
                self.reaches.loc[reachID, 'slope'] = slope
                # Get strtop from LineString mid-point Z
                zm = geom.interpolate(0.5, normalized=True).z
                self.reaches.loc[reachID, 'strtop'] = zm
        else:
            r = self.reaches['row'].values
            c = self.reaches['col'].values
            # Estimate slope from top and grid spacing
            dc = np.median(m.dis.delr.array)
            if m.dis.delr.array.min() != m.dis.delr.array.max():
                self.logger.warning('assuming constant column spacing %s', dc)
            dr = np.median(m.dis.delc.array)
            if m.dis.delc.array.min() != m.dis.delc.array.max():
                self.logger.warning('assuming constant row spacing %s', dr)
            px, py = np.gradient(m.dis.top.array, dc, dr)
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
        if not hasattr(self.reaches.geometry, 'geom_type'):
            # workaround needed for reaches.to_file()
            self.reaches.geometry.geom_type = self.reaches.geom_type
        # Build reach_data for Data Set 2
        # See flopy.modflow.ModflowSfr2.get_default_reach_dtype()
        self.reach_data = pd.DataFrame(
            self.reaches[['row', 'col', 'iseg', 'ireach', 'strtop', 'slope']])
        self.reach_data.rename(columns={'row': 'i', 'col': 'j'}, inplace=True)
        self.reach_data.insert(0, column='k', value=0)  # only top layer
        self.reach_data.insert(5, column='rchlen', value=self.reaches.length)
        # Build segment_data for Data Set 5
        self.segment_data = self.reaches[['iseg', 'segnum']].drop_duplicates()
        self.segment_data.rename(columns={'iseg': 'nseg'}, inplace=True)
        self.segment_data.set_index('segnum', inplace=True)
        self.segment_data.index.name = self.segments.index.name
        # Translate 'to_segnum' to 'outseg' via 'nseg'
        self.segment_data['outseg'] = 0
        for insegnum, inseg in self.segment_data.nseg.iteritems():
            outsegnum = self.segments.loc[insegnum, 'to_segnum']
            if outsegnum in self.segment_data.index:
                self.segment_data.loc[insegnum, 'outseg'] = \
                    self.segment_data.loc[outsegnum, 'nseg']
        self.segment_data['icalc'] = 1  # assumption
        # Create flopy Sfr2 package
        sfr = flopy.modflow.mfsfr2.ModflowSfr2(
                model=m,
                reach_data=self.reach_data.to_records(index=True),
                segment_data={0: self.segment_data.to_records(index=False)})
