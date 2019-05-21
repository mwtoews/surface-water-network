# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import geopandas
from osgeo import gdal
from warnings import warn
try:
    import rtree
except ImportError:
    rtree = False

__version__ = '0.1'


class SurfaceWaterNetwork(object):
    """Surface water network

    Attributes
    ----------
    verbose : bool or int
        Show messages (or not).
    lines : geopandas.geodataframe.GeoDataFrame
        GeoDataFrame lines of surface water network. Index is treated as node
        number, such as a reach ID.
    END_NODE : int
        Node number that indicates a line end, default is usually 0.
    to_node : pandas.core.series.Series
        Node number that line flow to.
    """
    verbose = None
    lines = None
    END_NODE = None
    to_node = None

    def __len__(self):
        return len(self.lines)

    def __init__(self, lines, verbose=True):
        """
        Initialise SurfaceWaterNetwork

        Parameters
        ----------
        lines : geopandas.geodataframe.GeoDataFrame
            Input GeoDataFrame lines of surface water network. Geometries
            must be 'LINESTRING Z'.
        verbose : bool or int, optional
            Show messages, default True.
        """
        self.verbose = verbose
        if not isinstance(lines, geopandas.geodataframe.GeoDataFrame):
            raise ValueError('lines must be a GeoDataFrame')
        elif len(lines) == 0:
            raise ValueError('one or more lines are required')
        elif not (lines.geom_type == 'LineString').all():
            raise ValueError('lines must all be LineString types')
        elif not lines.has_z.all():
            raise ValueError('lines must all have Z dimension')
        self.lines = lines
        return

    @classmethod
    def init_from_gdal(cls, lines_srs, elevation_srs=None, verbose=True):
        """
        Initialise SurfaceWaterNetwork from GDAL source datasets

        Parameters
        ----------
        lines_srs : str
            Path to open vector GDAL dataset of stream network lines.
        elevation_srs : str, optional
            Path to open raster GDAL dataset of elevation. If not provided,
            then Z-dimension from lines used. Not implemented yet.
        verbose : bool, optional
            Show messages, default True.
        """
        if verbose:
            print('reading lines from: {}'.format(lines_srs))
        lines_ds = gdal.Open(lines_srs, gdal.GA_ReadOnly)
        if lines_ds is None:
            raise IOError('cannot open lines: {}'.format(lines_srs))
        projection = lines_ds.GetProjection()
        if elevation_srs is None:
            elevation_ds = None
        else:
            if verbose:
                print('reading elevation from: {}'.format(elevation_srs))
            elevation_ds = gdal.Open(elevation_srs, gdal.GA_ReadOnly)
            if elevation_ds is None:
                raise IOError('cannot open elevation: {}'.format(elevation_ds))
            elif elevation_ds.RasterCount != 1:
                warn('expected 1 raster band for elevation, found {}'
                     .format(elevation_ds.RasterCount))
            band = elevation_ds.GetRasterBand(1)
            elevation = np.ma.array(band.ReadAsArray(), np.float64, copy=True)
            nodata = band.GetNoDataValue()
            elevation_ds = band = None  # close raster
            if nodata is not None:
                elevation.mask = elevation == nodata
            raise NotImplementedError('nothing done with elevation yet')
        return cls(projection=projection, verbose=verbose)

    def process(self):
        """
        Process surface water network

        Parameters
        ----------
        """
        # Populate a spatial index for speed
        if rtree:
            self.lines_idx = rtree.Rtree()
            for node, row in self.lines.bounds.iterrows():
                self.lines_idx.add(node, row.tolist())
            assert self.lines_idx.valid()
        else:
            self.lines_idx = None
        if self.lines.index.min() > 0:
            self.END_NODE = 0
        else:
            self.END_NODE = self.lines.index.min() - 1
        self.to_node = pd.Series(self.END_NODE, index=self.lines.index)
        for node, row in self.lines.iterrows():
            end_coord = row.geometry.coords[-1]  # downstream end
            if self.lines_idx:
                # reduce number of rows to scan based on proximity in 2D
                sub = self.lines.loc[self.lines_idx.intersection(end_coord[0:2])]
            else:
                # slow scan of full table
                sub = self.lines
            for node2, row2 in sub.iterrows():
                if node2 == node:
                    continue
                start_coord = row2.geometry.coords[0]
                if start_coord == end_coord:
                    self.to_node.loc[node] = node2
                    break
                elif start_coord[0:2] == end_coord[0:2]:
                    # 2D match only
                    self.to_node.loc[node] = node2
                    break
        # raise NotImplementedError()
