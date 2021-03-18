# -*- coding: utf-8 -*-
"""Abstract base class for a surface water network for MODFLOW."""

import pickle

from shapely.geometry import box

from swn.core import SurfaceWaterNetwork
from swn.spatial import compare_crs


class _SwnModflow(object):
    """Abstract class for a surface water network adaptor for MODFLOW.

    Attributes
    ----------
    segments : geopandas.GeoDataFrame
        Copied from swn.segments, but with additional columns added
    diversions :  geopandas.GeoDataFrame, pd.DataFrame or None
        Copied from swn.diversions, if set/defined.
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
        from importlib.util import find_spec
        if not find_spec('flopy'):
            raise ImportError(self.__class__.__name__ + ' requires flopy')
        from swn.logger import get_logger, logging
        if logger is None:
            self.logger = get_logger(self.__class__.__name__)
        elif isinstance(logger, logging.Logger):
            self.logger = logger
        else:
            raise ValueError(
                "expected 'logger' to be Logger; found " + str(type(logger)))
        self.logger.info('creating new %s object', self.__class__.__name__)
        self.segments = None
        self.diversions = None

    def __getstate__(self):
        """Serialize object attributes for pickle dumps."""
        return dict(self)

    @classmethod
    def from_pickle(cls, path, model):
        """Read a pickled format from a file.

        Parameters
        ----------
        path : str
            File path where the pickled object will be stored.
        model : flopy.modflow.mf.Modflow
            Instance of a flopy MODFLOW model.

        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        obj.model = model
        return obj

    @property
    def model(self):
        """Return flopy model object."""
        try:
            return getattr(self, '_model', None)
        except AttributeError:
            self.logger.error("'model' property not set")

    @classmethod
    def from_swn_flopy(
            cls, swn, model, ibound_action='freeze',
            reach_include_fraction=0.2, min_slope=1./1000):
        """Create a MODFLOW SFR structure from a surface water network.

        Parameters
        ----------
        swn : swn.SurfaceWaterNetwork
            Instance of a SurfaceWaterNetwork.
        model : flopy.modflow.mf.Modflow or flopy.mf6.ModflowGwf
            Instance of a flopy MODFLOW model.
        ibound_action : str, optional
            Action to handle IBOUND:
                - ``freeze`` : Freeze IBOUND, but clip streams to fit bounds.
                - ``modify`` : Modify IBOUND to fit streams, where possible.
        reach_include_fraction : float or pandas.Series, optional
            Fraction of cell size used as a threshold distance to determine if
            reaches outside the active grid should be included to a cell.
            Based on the furthest distance of the line and cell geometries.
            Default 0.2 (e.g. for a 100 m grid cell, this is 20 m).
        min_slope : float or pandas.Series, optional
            Minimum downwards slope imposed on segments. If float, then this is
            a global value, otherwise it is per-segment with a Series.
            Default 1./1000 (or 0.001).

        """
        obj = cls()
        if not isinstance(swn, SurfaceWaterNetwork):
            raise ValueError('swn must be a SurfaceWaterNetwork object')
        elif ibound_action not in ('freeze', 'modify'):
            raise ValueError('ibound_action must be one of freeze or modify')
        obj.model = model
        obj.segments = swn.segments.copy()
        # Make sure model CRS and segments CRS are the same (if defined)
        crs = None
        segments_crs = getattr(obj.segments.geometry, 'crs', None)
        modelgrid_crs = None
        modelgrid = obj.model.modelgrid
        epsg = modelgrid.epsg
        proj4_str = modelgrid.proj4
        if epsg is not None:
            segments_crs, modelgrid_crs, same = compare_crs(segments_crs, epsg)
        else:
            segments_crs, modelgrid_crs, same = compare_crs(segments_crs,
                                                            proj4_str)
        if (segments_crs is not None and modelgrid_crs is not None and
                not same):
            obj.logger.warning(
                'CRS for segments and modelgrid are different: {0} vs. {1}'
                .format(segments_crs, modelgrid_crs))
        crs = segments_crs or modelgrid_crs
        # Make sure their extents overlap
        minx, maxx, miny, maxy = modelgrid.extent
        model_bbox = box(minx, miny, maxx, maxy)
        rstats = obj.segments.bounds.describe()
        segments_bbox = box(
                rstats.loc['min', 'minx'], rstats.loc['min', 'miny'],
                rstats.loc['max', 'maxx'], rstats.loc['max', 'maxy'])
        if model_bbox.disjoint(segments_bbox):
            raise ValueError('modelgrid extent does not cover segments extent')
