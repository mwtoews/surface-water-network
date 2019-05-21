# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import geopandas
from osgeo import gdal
from warnings import warn

__version__ = '0.1'


class SurfaceWaterNetwork(object):
    """Surface water network

    Attributes
    ----------
    lines : geopandas.geodataframe.GeoDataFrame
        GeoDataFrame lines of surface water network.
    verbose : bool or int
        Show messages (or not).
    """
    lines = None
    verbose = None

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
        raise NotImplementedError()
