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
    END_NODE : int
        Node number that indicates a line end, default is usually 0.
    reaches : pandas.core.frame.DataFrame
        DataFrame created by evaluate_reaches() and shares same index as lines.
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
    END_NODE = None
    reaches = None
    logger = None
    warnings = None
    errors = None

    def __len__(self):
        return len(self.lines)

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
        self.logger.info('creating network with %d lines', len(self.lines))
        # Populate a spatial index for speed
        if rtree and len(self.lines) >= _rtree_threshold:
            self.logger.debug('building R-tree index of lines')
            self.lines_idx = rtree.Rtree()
            for node, row in self.lines.bounds.iterrows():
                self.lines_idx.add(node, row.tolist())
            assert self.lines_idx.valid()
        else:
            if len(self.lines) >= _rtree_threshold:
                self.logger.debug(
                    'using slow sequence scanning; consider installing rtree')
            self.lines_idx = None
        if self.lines.index.min() > 0:
            self.END_NODE = 0
        else:
            self.END_NODE = self.lines.index.min() - 1
        self.reaches = pd.DataFrame(
                {'to_node': self.END_NODE}, index=self.lines.index)
        self.errors = []
        self.warnings = []
        # Cartesian join of lines to find where ends connect to
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
                elif (row2.geometry.distance(end_pt) < 1e-6
                      and Point(*end2_coord).distance(end_pt) > 1e-6):
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
        # Check headwater nodes
        start_coords = {}  # key: 2D coord, value: list of nodes
        for node, row in self.lines.loc[self.headwater].iterrows():
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
        return self.lines.index[
                ~self.reaches.index.isin(self.reaches['to_node'])]

    @property
    def outlets(self):
        return self.lines.index[self.reaches['to_node'] == self.END_NODE]
