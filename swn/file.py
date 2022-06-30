"""File reading/writing helpers."""

__all__ = ["topnet2ts", "gdf_to_shapefile"]

import geopandas
import pandas as pd

from .logger import get_logger, logging


def topnet2ts(nc_path, varname, *,
              mult=None, run=None, log_level=logging.INFO):
    """Read TopNet data from a netCDF file into a pandas.DataFrame timeseries.

    User may need to multiply DataFrame to convert units.

    Parameters
    ----------
    nc_path : str
        File path to netCDF file
    varname : str
        Variable name in netCDF file to read
    mult : float, optional
        Multiplier applied to dataset, which preserves dtype. For example,
        to convert from "meters3 second-1" to "meters3 day-1", use 86400.
    run : int, optional
        Files with an ensemble or uncertainty analysis may have more than one
        run, this option allows a run to be selected. Default behaviour is to
        take the first run (index 0). The last run can be selected with -1.
    log_level : int, optional
        Level used by logging module; default is 20 (logging.INFO)

    Returns
    -------
    pandas.DataFrame
        Where columns is rchid and index is DatetimeIndex.

    """
    try:
        from netCDF4 import Dataset
    except ImportError:
        raise ImportError('function requires netCDF4')
    try:
        from cftime import num2pydate as n2d
    except ImportError:
        from cftime import num2date as n2d
    logger = get_logger("topnet2ts", log_level)
    logger.info('reading file: "%s"', nc_path)
    with Dataset(nc_path, "r") as nc:
        nc.set_auto_mask(False)
        varnames = list(nc.variables.keys())
        if varname not in varnames:
            raise KeyError(
                f"{varname!r} not found in dataset; use one of {varnames}")
        var = nc.variables[varname]
        logger.info("variable %r:\n%s", varname, var)
        # Evaluate dimensions
        dim_has_time = False
        dim_has_nrch = False
        dim_ignore = []
        varslice = [Ellipsis]  # take first dimensions
        for name, size in zip(var.dimensions, var.shape):
            if name == "time":
                dim_has_time = True
            elif name == "nrch":
                dim_has_nrch = True
            elif name in ("nrun", "nens"):
                if run is None:
                    if size > 1:
                        logger.warning(
                            "no run specified; taking %s index 0 from dim "
                            "size %s", var.dimensions[2], var.shape[2])
                    run = 0
                varslice.append(run)
            elif size == 1:
                dim_ignore.append(name)
                varslice.append(0)
        if not dim_has_time:
            logger.error("no 'time' dimension found")
        if not dim_has_nrch:
            logger.error("no 'nrch' dimension found")
        if dim_ignore:
            logger.info("ignoring size 1 dimensions: %s", dim_ignore)
        varslice = tuple(varslice)
        logger.debug("indexing %r with %r", varname, varslice)
        dat = var[varslice]
        if len(dat.shape) != 2:
            logger.error("expected 2 dimensions, found shape %s", dat.shape)
        if dim_has_time and var.dimensions.index("time") == 1:
            dat = dat.T
        if mult is not None and mult != 1.0:
            dat *= mult
        df = pd.DataFrame(dat)
        df.columns = nc.variables["rchid"]
        time_v = nc.variables["time"]
        df.index = pd.DatetimeIndex(n2d(time_v[:], time_v.units))
    logger.info("data successfully read")
    return df


# potential names that need to be shortened to <= 10 characters for DBF
colname10 = {
    "to_segnum": "to_seg",
    "from_segnums": "from_seg",
    "num_to_outlet": "num_to_out",
    "dist_to_outlet": "dst_to_out",
    "stream_order": "strm_order",
    "upstream_length": "upstr_len",
    "upstream_area": "upstr_area",
    "inflow_segnums": "inflow_seg",
    "zcoord_count": "zcoord_num",
    "zcoord_first": "zcoordfrst",
    "zcoord_last": "zcoordlast",
    "strtop_incopt": "stpincopt",
    "prev_ibound": "previbound",
    "prev_idomain": "prevdomain",
    "dist_to_segnum": "dst_to_seg",
    "dist_to_seg": "dst_to_seg",
    "is_within_catchment": "within_cat",
    "div_from_rno": "divfromrno",
    "div_to_rnos": "divtornos",
}
assert all(len(x) <= 10 for x in colname10.values())


def gdf_to_shapefile(gdf, shp_fname, **kwargs):
    """Write any GeoDataFrame to a shapefile.

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
        raise ValueError("expected gdf to be a GeoDataFrame")
    gdf = gdf.copy()
    geom_name = gdf.geometry.name

    # Change data types, as necessary
    for col, dtype in gdf.dtypes.iteritems():
        if col == geom_name:
            continue
        if dtype == object:
            is_none = gdf[col].map(lambda x: x is None)
            gdf[col] = gdf[col].astype(str)
            gdf.loc[is_none, col] = ""
        elif dtype == bool:
            gdf[col] = gdf[col].astype(int)
    # Rename columns for shapefile so they are 10 characters or less
    rename = {}
    for key, value in colname10.items():
        if key in gdf.columns:
            rename[key] = value
    gdf.rename(columns=rename).reset_index(drop=False)\
        .to_file(str(shp_fname), **kwargs)
