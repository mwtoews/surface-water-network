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
        for idx, (node, row) in enumerate(gdf.bounds.iterrows()):
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
        attributes evaluated during initialisation. Index is treated as node
        number, such as a segment ID.
    index : pandas.core.index.Int64Index
        Shortcut property to segments.index or node number.
    END_NODE : int
        Special node number that indicates a line end, default is usually 0.
        This number is not part of the index.
    upstream_nodes : dict
        Key is downstream node, and values are a set of zero or more upstream
        nodes. END_NODE and headwater nodes are not included.
    has_z : bool
        Property that indicates all segment lines have Z dimension coordinates.
    headwater : pandas.core.index.Int64Index
        Head water nodes at top of cachment.
    outlets : pandas.core.index.Int64Index
        Index nodes for each outlet.
    logger : logging.Logger
        Logger to show messages.
    warnings : list
        List of warning messages.
    errors : list
        List of error messages.
    """
    index = None
    END_NODE = None
    segments = None
    upstream_nodes = None
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
            'LINESTRING' or 'LINESTRING Z'. Index is used for node numbers.
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
            self.END_NODE = 0
        else:
            self.END_NODE = self.index.min() - 1
        self.segments['to_node'] = self.END_NODE
        self.errors = []
        self.warnings = []
        # Cartesian join of segments to find where ends connect to
        self.logger.debug('finding connections between pairs of segment lines')
        geom_name = self.segments.geometry.name
        for node, geom in self.segments.geometry.iteritems():
            end_coord = geom.coords[-1]  # downstream end
            end_pt = Point(*end_coord)
            if segments_sindex:
                subsel = segments_sindex.intersection(end_coord[0:2])
                sub = self.segments.iloc[list(subsel)]
            else:  # slow scan of all segments
                sub = self.segments
            to_nodes = []
            for node2, geom2 in sub.geometry.iteritems():
                if node2 == node:
                    continue
                start2_coord = geom2.coords[0]
                end2_coord = geom2.coords[-1]
                if start2_coord == end_coord:
                    # perfect 3D match from end of node to start of node2
                    to_nodes.append(node2)
                elif start2_coord[0:2] == end_coord[0:2]:
                    to_nodes.append(node2)
                    m = ('node %s matches %s in 2D, but not in Z dimension',
                         node, node2)
                    self.logger.warning(*m)
                    self.warnings.append(m[0] % m[1:])
                elif (geom2.distance(end_pt) < 1e-6 and
                      Point(*end2_coord).distance(end_pt) > 1e-6):
                    m = ('node %s connects to the middle of node %s',
                         node, node2)
                    self.logger.error(*m)
                    self.errors.append(m[0] % m[1:])
            if len(to_nodes) > 1:
                m = ('node %s has more than one downstream nodes: %s',
                     node, str(to_nodes))
                self.logger.error(*m)
                self.errors.append(m[0] % m[1:])
            if len(to_nodes) > 0:
                self.segments.loc[node, 'to_node'] = to_nodes[0]

        outlets = self.outlets
        self.logger.debug('evaluating segments upstream from %d outlet%s',
                          len(outlets), 's' if len(outlets) != 1 else '')
        self.segments['cat_group'] = self.END_NODE
        self.segments['num_to_outlet'] = 0
        self.segments['length_to_outlet'] = 0.0

        # Recursive function that accumulates information upstream
        def resurse_upstream(node, cat_group, num, length):
            self.segments.loc[node, 'cat_group'] = cat_group
            num += 1
            self.segments.loc[node, 'num_to_outlet'] = num
            length += self.segments.loc[node, geom_name].length
            self.segments.loc[node, 'length_to_outlet'] = length
            # Branch to zero or more upstream segments
            for upnode in self.index[self.segments['to_node'] == node]:
                resurse_upstream(upnode, cat_group, num, length)

        for node in self.segments.loc[outlets].index:
            resurse_upstream(node, node, 0, 0.0)

        # Check to see if headwater and outlets have common locations
        headwater = self.headwater
        self.logger.debug('checking %d headwater nodes and %d outlet nodes',
                          len(headwater), len(outlets))
        start_coords = {}  # key: 2D coord, value: list of nodes
        for node, geom in self.segments.loc[headwater].geometry.iteritems():
            start_coord = geom.coords[0]
            start_coord2d = start_coord[0:2]
            if segments_sindex:
                subsel = segments_sindex.intersection(start_coord2d)
                sub = self.segments.iloc[list(subsel)]
            else:  # slow scan of all segments
                sub = self.segments
            for node2, geom2 in sub.geometry.iteritems():
                if node2 == node:
                    continue
                start2_coord = geom2.coords[0]
                match = False
                if start2_coord == start_coord:
                    # perfect 3D match from end of node to start of node2
                    match = True
                elif start2_coord[0:2] == start_coord2d:
                    match = True
                    m = ('starting node %s matches %s in 2D, but not in '
                         'Z dimension', node, node2)
                    self.logger.warning(*m)
                    self.warnings.append(m[0] % m[1:])
                if match:
                    if start_coord2d in start_coords:
                        start_coords[start_coord2d].add(node2)
                    else:
                        start_coords[start_coord2d] = set([node2])
        for key in start_coords.keys():
            v = start_coords[key]
            m = ('starting coordinate %s matches start node%s: %s',
                 key, 's' if len(v) != 1 else '', v)
            self.logger.error(*m)
            self.errors.append(m[0] % m[1:])

        end_coords = {}  # key: 2D coord, value: list of nodes
        for node, geom in self.segments.loc[outlets].geometry.iteritems():
            end_coord = geom.coords[-1]
            end_coord2d = end_coord[0:2]
            if segments_sindex:
                subsel = segments_sindex.intersection(end_coord2d)
                sub = self.segments.iloc[list(subsel)]
            else:  # slow scan of all segments
                sub = self.segments
            for node2, geom2 in sub.geometry.iteritems():
                if node2 == node:
                    continue
                end2_coord = geom2.coords[-1]
                match = False
                if end2_coord == end_coord:
                    # perfect 3D match from end of node to start of node2
                    match = True
                elif end2_coord[0:2] == end_coord2d:
                    match = True
                    m = ('ending node %s matches %s in 2D, but not in '
                         'Z dimension', node, node2)
                    self.logger.warning(*m)
                    self.warnings.append(m[0] % m[1:])
                if match:
                    if end_coord2d in end_coords:
                        end_coords[end_coord2d].add(node2)
                    else:
                        end_coords[end_coord2d] = set([node2])
        for key in end_coords.keys():
            v = end_coords[key]
            m = ('ending coordinate %s matches end node%s: %s',
                 key, 's' if len(v) != 1 else '', v)
            self.logger.warning(*m)
            self.warnings.append(m[0] % m[1:])

        self.logger.debug('evaluating downstream sequence')
        self.segments['sequence'] = 0
        self.segments['stream_order'] = 0
        # self.segments['numiter'] = 0  # should be same as stream_order
        # Sort headwater nodes from the furthest from outlet to closest
        search_order = ['num_to_outlet', 'length_to_outlet']
        furthest_upstream = self.segments.loc[headwater]\
            .sort_values(search_order, ascending=False).index
        sequence = pd.Series(
            np.arange(len(furthest_upstream)) + 1, index=furthest_upstream)
        self.segments.loc[sequence.index, 'sequence'] = sequence
        self.segments.loc[sequence.index, 'stream_order'] = 1
        # Build a dict that describes downstream nodes to one or more upstream
        self.upstream_nodes = {}
        for node in set(self.segments['to_node']).difference([self.END_NODE]):
            self.upstream_nodes[node] = \
                set(self.index[self.segments['to_node'] == node])
        completed = set(headwater)
        sequence = int(sequence.max())
        for numiter in range(1, self.segments['num_to_outlet'].max() + 1):
            # Gather nodes downstream from completed upstream set
            downstream = set(
                self.segments.loc[completed, 'to_node'])\
                .difference(completed.union([self.END_NODE]))
            # Sort them to evaluate the furthest first
            downstream_sorted = self.segments.loc[downstream]\
                .sort_values(search_order, ascending=False).index
            for node in downstream_sorted:
                if self.upstream_nodes[node].issubset(completed):
                    sequence += 1
                    self.segments.loc[node, 'sequence'] = sequence
                    # self.segments.loc[node, 'numiter'] = numiter
                    up_ord = list(
                        self.segments.loc[
                            list(self.upstream_nodes[node]), 'stream_order'])
                    max_ord = max(up_ord)
                    if up_ord.count(max_ord) > 1:
                        self.segments.loc[node, 'stream_order'] = max_ord + 1
                    else:
                        self.segments.loc[node, 'stream_order'] = max_ord
                    completed.add(node)
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
        return self.index[~self.index.isin(self.segments['to_node'])]

    @property
    def outlets(self):
        """Returns index of outlets"""
        return self.index[self.segments['to_node'] == self.END_NODE]

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
        for node in self.segments.sort_values('sequence').index:
            if node in self.upstream_nodes:
                upstream_nodes = list(self.upstream_nodes[node])
                if upstream_nodes:
                    accum[node] += accum[upstream_nodes].sum()
        return accum

    def adjust_elevation_profile(self, min_slope=1./1000):
        """Check and adjust (if necessary) Z coordinates of elevation profiles

        Parameters
        ----------
        min_slope : float or pandas.Series, optional
            Minimum downwards slope imposed on segments. If float, then this is
            a global value, otherwise it is per-segment with a Series.
            Default 1./1000 (or 0.001).
        """
        if not isinstance(min_slope, pd.Series):
            min_slope = pd.Series(min_slope, index=self.index)
        elif (len(min_slope.index) != len(self.index) or
                not (min_slope.index == self.index).all()):
            raise ValueError('index for min_slope is different')
        if (min_slope < 0.0).any():
            raise ValueError('min_slope must be greater than zero')
        elif not self.has_z:
            raise AttributeError('line geometry does not have Z dimension')
        geom_name = self.segments.geometry.name
        # Build elevation profiles as 2D coordinates,
        # where X is 2D distance from downstream coordinate and Y is elevation
        profiles = []
        self.messages = []
        for node, geom in self.segments.geometry.iteritems():
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
                if slope < min_slope[node]:
                    modified += 1
                    z1 = z0 - dx * min_slope[node]
                    coords[idx] = (x1, y1, z1)
                profile_coords.append((dist, z1))
                x0, y0, z0 = x1, y1, z1
            if modified > 0:
                m = ('adjusting %d coordinate elevation%s in segment %s',
                     modified, 's' if modified != 1 else '', node)
                self.logger.debug(*m)
                self.messages.append(m[0] % m[1:])
            if modified:
                self.segments.loc[node, geom_name] = LineString(coords)
            profiles.append(LineString(profile_coords))
        self.profiles = geopandas.GeoSeries(profiles)
        return
        # TODO: adjust connected segments
        # Find minimum elevation, then force any nodes downstream to flow down
        self.profiles.geometry.bounds.miny.sort_values()

        # lines = self.segments
        numiter = 0
        while True:
            numiter += 1

    def process_flopy(self, m, ibound_action='freeze'):
        """Process MODFLOW groundwater model information from flopy

        Parameters
        ----------
        m : flopy.modflow.mf.Modflow
            Instance of a flopy MODFLOW model with DIS and BAS6 packages.
        ibound_action : str, optional
            Action to handle IBOUND:
                'freeze' : Freeze IBOUND, but clip streams to fit
                'modify' : Modify IBOUND to fit streams (not done)
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
        ibound = m.bas6.ibound[0].array
        grid_df = pd.DataFrame({
                'col': cols.flatten(),
                'row': rows.flatten(),
                'ibound': ibound.flatten()
        })
        if ibound_action == 'freeze' and (ibound == 0).any():
            # Remove any inactive grid cells from analysis
            grid_df = grid_df.loc[grid_df['ibound'] != 0]
        grid_list = [
            Polygon(m.modelgrid.get_cell_vertices(item.row, item.col))
            for _, item in grid_df.iterrows()]
        crs = None
        if m.modelgrid.proj4 is not None:
            crs = fiona_crs.from_string(m.modelgrid.proj4)
        elif self.segments.geometry.crs is not None:
            crs = self.segments.geometry.crs
        grid_cells = geopandas.GeoDataFrame(
                                        grid_df, geometry=grid_list, crs=crs)
        grid_sindex = get_sindex(grid_cells)
        self.grid_cells = grid_cells
        # sel_cols = [self.segments.geometry.name, 'sequence']
        # lines_gdf = self.segments.loc[:, sel_cols]
        # self.j = sjoin(grid_cells, lines_gdf).sort_values('sequence')
        seg_df = pd.DataFrame(
            columns=['lineseg', 'node', 'dist', 'row', 'col'])

        def append_seg(lineseg, node, dist, row, col):
            seg_df.loc[len(seg_df) + 1] = {
                'lineseg': lineseg,
                'node': node,
                'dist': dist,
                'row': grid.row,
                'col': grid.col,
            }

        for node, line in self.segments.geometry.iteritems():
            if grid_sindex:
                bbox_match = sorted(grid_sindex.intersection(line.bounds))
                if not bbox_match:
                    continue
            else:  # slow scan of all cells
                bbox_match = np.arange(len(grid_cells))
            for idx, grid in grid_cells.iloc[bbox_match].iterrows():
                lineseg = grid.geometry.intersection(line)
                if lineseg.length == 0.0:
                    pass  # GEOMETRYCOLLECTION EMPTY or POINT
                elif lineseg.geom_type == 'LineString':
                    append_seg(lineseg, node,
                               line.project(Point(lineseg.coords[0])),
                               grid.row, grid.col)
                elif lineseg.geom_type == 'MultiLineString':
                    for sublineseg in lineseg.geoms:
                        append_seg(sublineseg, node,
                                   line.project(Point(sublineseg.coords[0])),
                                   grid.row, grid.col)
                else:
                    raise NotImplementedError(lineseg.geom_type)

        self.seg_df = geopandas.GeoDataFrame(seg_df, geometry='lineseg')\
            .sort_values(['node', 'dist'])
