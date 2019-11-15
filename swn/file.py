# -*- coding: utf-8 -*-
# File reading/writing helpers

import geopandas
import pandas as pd

from swn.logger import logging, get_logger


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
    logger = get_logger('topnet2ts', log_level)
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


def gdf_to_shapefile(gdf, shp_fname, **kwargs):
    """Write any GeoDataFrame to a shapefile

    This is a workaround to the to_file method, which cannot save
    GeoDataFrame objects with other data types, such as set.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame to export
    shp_fname : str
        File path for output shapefile
    kwargs : mapping
        Keyword arguments passed to to_file and to fiona.open

    Returns
    -------
    None
    """
    if not isinstance(gdf, geopandas.GeoDataFrame):
        raise ValueError('expected gdf to be a GeoDataFrame')
    gdf = gdf.copy()
    geom_name = gdf.geometry.name
    for col, dtype in gdf.dtypes.iteritems():
        if col == geom_name:
            continue
        is_none = gdf[col].map(lambda x: x is None)
        if dtype == object:
            gdf[col] = gdf[col].astype(str)
            gdf.loc[is_none, col] = ''
        elif dtype == bool:
            gdf[col] = gdf[col].astype(int)
    # potential names that need to be shortened to <= 10 characters for DBF
    colname10 = {
        'to_segnum': 'to_seg',
        'from_segnums': 'from_seg',
        'num_to_outlet': 'num_to_out',
        'dist_to_outlet': 'dst_to_out',
        'stream_order': 'strm_order',
        'upstream_length': 'upstr_len',
        'upstream_area': 'upstr_area',
        'inflow_segnums': 'inflow_seg',
    }
    for k, v in list(colname10.items()):
        assert len(v) <= 10, v
        if k == v or k not in gdf.columns:
            del colname10[k]
    gdf.rename(columns=colname10).reset_index(drop=False)\
        .to_file(str(shp_fname), **kwargs)
