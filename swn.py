# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
import geopandas
from shapely.geometry import Point
try:
    from osgeo import gdal
except ImportError:
    gdal = False
try:
    import rtree
except ImportError:
    rtree = False

__version__ = '0.1'

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


class SurfaceWaterNetwork(object):
    """Surface water network

    Attributes
    ----------
    lines : geopandas.geodataframe.GeoDataFrame
        GeoDataFrame lines of surface water network. Index is treated as node
        number, such as a reach ID.
    index : pandas.core.index.Int64Index
        Shared index or node number.
    END_NODE : int
        Special node number that indicates a line end, default is usually 0.
        This number is not part of the index.
    reaches : pandas.core.frame.DataFrame
        DataFrame created by evaluate_reaches() and shares same index as lines.
    upstream_nodes : dict
        Key is downstream node, and values are a set of zero or more upstream
        nodes. END_NODE and headwater nodes are not included.
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
    lines = None
    index = None
    END_NODE = None
    reaches = None
    upstream_nodes = None
    logger = None
    warnings = None
    errors = None

    def __len__(self):
        return len(self.index)

    def __init__(self, lines, logger=None):
        """
        Initialise SurfaceWaterNetwork and evaluate reaches

        Parameters
        ----------
        lines : geopandas.geodataframe.GeoDataFrame
            Input GeoDataFrame lines of surface water network. Geometries
            must be 'LINESTRING Z'.
        logger : logging.Logger, optional
            Logger to show messages.
        """
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.handlers = module_logger.handlers
            self.logger.setLevel(module_logger.level)
        if not isinstance(lines, geopandas.geodataframe.GeoDataFrame):
            raise ValueError('lines must be a GeoDataFrame')
        elif len(lines) == 0:
            raise ValueError('one or more lines are required')
        elif not (lines.geom_type == 'LineString').all():
            raise ValueError('lines must all be LineString types')
        elif not lines.geometry.apply(lambda x: x.has_z).all():
            raise ValueError('lines must all have Z dimension')
        self.lines = lines
        self.index = self.lines.index
        self.logger.info('creating network with %d lines', len(self))
        # Populate a spatial index for speed
        if rtree and len(self) >= _rtree_threshold:
            self.logger.debug('building R-tree index of lines')
            self.lines_idx = rtree.Rtree()
            for node, row in self.lines.bounds.iterrows():
                self.lines_idx.add(node, row.tolist())
            assert self.lines_idx.valid()
        else:
            if len(self) >= _rtree_threshold:
                self.logger.debug(
                    'using slow sequence scanning; consider installing rtree')
            self.lines_idx = None
        if self.index.min() > 0:
            self.END_NODE = 0
        else:
            self.END_NODE = self.index.min() - 1
        self.reaches = pd.DataFrame(
                {'to_node': self.END_NODE}, index=self.index)
        self.errors = []
        self.warnings = []
        # Cartesian join of lines to find where ends connect to
        self.logger.debug('finding connections between pairs of lines')
        for node, row in self.lines.iterrows():
            end_coord = row.geometry.coords[-1]  # downstream end
            end_pt = Point(*end_coord)
            if self.lines_idx:
                # reduce number of rows to scan based on proximity in 2D
                subsel = self.lines_idx.intersection(end_coord[0:2])
                sub = self.lines.loc[list(subsel)]
            else:
                # slow scan of full table
                sub = self.lines
            to_nodes = []
            for node2, row2 in sub.iterrows():
                if node2 == node:
                    continue
                start2_coord = row2.geometry.coords[0]
                end2_coord = row2.geometry.coords[-1]
                if start2_coord == end_coord:
                    # perfect 3D match from end of node to start of node2
                    to_nodes.append(node2)
                elif start2_coord[0:2] == end_coord[0:2]:
                    to_nodes.append(node2)
                    m = ('node %s matches %s in 2D, but not in Z-dimension',
                         node, node2)
                    self.logger.warning(*m)
                    self.warnings.append(m[0] % m[1:])
                elif (row2.geometry.distance(end_pt) < 1e-6 and
                      Point(*end2_coord).distance(end_pt) > 1e-6):
                    m = ('node %s connects to the middle of node %s',
                         node, node2)
                    self.logger.error(*m)
                    self.errors.append(m[0] % m[1:])
            if len(to_nodes) > 1:
                m = ('node %s has more than one downstream nodes: %s',
                     node, tuple(to_nodes))
                self.logger.error(*m)
                self.errors.append(m[0] % m[1:])
            if len(to_nodes) > 0:
                self.reaches.loc[node, 'to_node'] = to_nodes[0]

        # Recursive function that accumulates information upstream
        def resurse_upstream(node, cat_group, num, length):
            self.reaches.loc[node, 'cat_group'] = cat_group
            num += 1
            self.reaches.loc[node, 'num_to_outlet'] = num
            length += self.lines.geometry[node].length
            self.reaches.loc[node, 'length_to_outlet'] = length
            # Branch to zero or more upstream reaches
            for upnode in self.reaches.index[self.reaches['to_node'] == node]:
                resurse_upstream(upnode, cat_group, num, length)

        outlets = self.outlets
        self.logger.debug('evaluating lines upstream from %d outlet%s',
                          len(outlets), 's' if len(outlets) != 1 else '')
        self.reaches['cat_group'] = self.END_NODE
        self.reaches['num_to_outlet'] = 0
        self.reaches['length_to_outlet'] = 0.0
        for node in self.reaches.loc[outlets].index:
            resurse_upstream(node, node, 0, 0.0)

        headwater = self.headwater
        self.logger.debug('checking %d headwater nodes', len(headwater))
        start_coords = {}  # key: 2D coord, value: list of nodes
        for node, row in self.lines.loc[headwater].iterrows():
            start_coord = row.geometry.coords[0]
            start_coord2d = start_coord[0:2]
            if self.lines_idx:
                subsel = self.lines_idx.intersection(start_coord2d)
                sub = self.lines.loc[list(subsel)]
            else:
                # slow scan of full table
                sub = self.lines
            for node2, row2 in sub.iterrows():
                if node2 == node:
                    continue
                start2_coord = row2.geometry.coords[0]
                match = False
                if start2_coord == start_coord:
                    # perfect 3D match from end of node to start of node2
                    match = True
                elif start2_coord[0:2] == start_coord[0:2]:
                    match = True
                    m = ('starting node %s matches %s in 2D, but not in '
                         'Z-dimension', node, node2)
                    self.logger.warning(*m)
                    self.warnings.append(m[0] % m[1:])
                if match:
                    if start_coord2d in start_coords:
                        start_coords[start_coord2d].add(node2)
                    else:
                        start_coords[start_coord2d] = set([node2])
        for key in start_coords.keys():
            m = ('starting coordinate %s matches start nodes %s',
                 key, start_coords[key])
            self.logger.error(*m)
            self.errors.append(m[0] % m[1:])

        self.logger.debug('evaluating downstream sequence')
        self.reaches['sequence'] = 0
        self.reaches['numiter'] = -1
        # Sort headwater nodes from the furthest to outlet to closest
        furthest_upstream = self.reaches.loc[headwater]\
            .sort_values('length_to_outlet', ascending=False).index
        sequence = pd.Series(
            np.arange(len(furthest_upstream)) + 1, index=furthest_upstream)
        self.reaches.loc[sequence.index, 'sequence'] = sequence
        self.reaches.loc[sequence.index, 'numiter'] = 0
        # Build a dict that describes downstream nodes to one or more upstream
        self.upstream_nodes = {}
        for node in set(self.reaches['to_node']).difference([self.END_NODE]):
            self.upstream_nodes[node] = \
                set(self.index[self.reaches['to_node'] == node])
        completed = set(headwater)
        sequence = int(sequence.max())
        for numiter in range(1, self.reaches['num_to_outlet'].max() + 1):
            # Gather nodes downstream from completed upstream set
            downstream = set(
                self.reaches.loc[completed, 'to_node'])\
                .difference(completed.union([self.END_NODE]))
            # Sort them to evaluate the furthest first
            downstream_sorted = self.reaches.loc[downstream]\
                .sort_values('length_to_outlet', ascending=False).index
            for node in downstream_sorted:
                if self.upstream_nodes[node].issubset(completed):
                    sequence += 1
                    self.reaches.loc[node, 'sequence'] = sequence
                    self.reaches.loc[node, 'numiter'] = numiter
                    completed.add(node)
            if self.reaches['sequence'].min() > 0:
                break
        self.logger.debug('sequence evaluated with %d iterations', numiter)

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
            then Z-dimension from lines used. Not implemented yet.
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
    def headwater(self):
        return self.index[
                ~self.reaches.index.isin(self.reaches['to_node'])]

    @property
    def outlets(self):
        return self.index[self.reaches['to_node'] == self.END_NODE]

    def accumulate_values(self, values):
        """Accumulate values down the stream network

        For example, calculate cumulative upstream catchment area for each
        reach.

        Parameters
        ----------
        values : pandas.core.series.Series
            Series of values that align with lines.index.

        Returns
        -------
        pandas.core.series.Series
            Accumulated values.
        """
        if not isinstance(values, pd.core.series.Series):
            raise ValueError('values must be a pandas Series')
        elif (len(values.index) != len(self.reaches.index) or
                not (values.index == self.reaches.index).all()):
            raise ValueError('index is different')
        accum = values.copy()
        if isinstance(accum.name, str):
            accum.name = 'accumulated_' + accum.name
        for node in self.reaches.sort_values('sequence').index:
            if node in self.upstream_nodes:
                upstream_nodes = list(self.upstream_nodes[node])
                if upstream_nodes:
                    accum[node] += accum[upstream_nodes].sum()
        return accum
