"""File reading/writing helpers."""

__all__ = [
    "topnet2ts",
    "gdf_to_shapefile",
    "read_formatted_frame",
    "write_formatted_frame",
]

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
    "dist_to_reach": "dst_to_rch",
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
    for col, dtype in gdf.dtypes.items():
        if col == geom_name:
            continue
        if dtype == bool:
            gdf[col] = gdf[col].astype(int)
        else:
            is_none = gdf[col].map(lambda x: x is None).fillna(True)
            gdf[col] = gdf[col].astype(str)
            gdf.loc[is_none, col] = ""
    # Rename columns for shapefile so they are 10 characters or less
    rename = {}
    for key, value in colname10.items():
        if key in gdf.columns:
            rename[key] = value
    gdf.rename(columns=rename).reset_index(drop=False)\
        .to_file(str(shp_fname), **kwargs)


def read_formatted_frame(fname):
    r"""Write a data frame as a free formatted table.

    Parameters
    ----------
    fname : Path, str or file-like object
        Path to read file.
    comment_header : bool, default True
        If True, first line starts with '#' to make it a comment, otherwise
        it will be an uncommented header.

    Returns
    -------
    pandas.DataFrame

    See Also
    --------
    write_formatted_frame

    Examples
    --------
    >>> from io import StringIO
    >>> import pandas as pd
    >>> from swn.file import read_formatted_frame
    >>> f = StringIO('''\
    ... # rno        value1  value2  value3
    ... 1     -1.000000e+10       1  'first one'
    ... 12    -1.000000e-10      10   two
    ... 33     0.000000e+00     100   three
    ... 40     1.000000e-10    1000
    ... 450    1.000000e+00   10000   five
    ... 6267   1.000000e+03  100000   six
    ... ''')
    >>> df = read_formatted_frame(f).set_index("rno")
    >>> print(df)
                value1  value2     value3
    rno                                  
    1    -1.000000e+10       1  first one
    12   -1.000000e-10      10        two
    33    0.000000e+00     100      three
    40    1.000000e-10    1000        NaN
    450   1.000000e+00   10000       five
    6267  1.000000e+03  100000        six
    """  # noqa
    fname_is_filelike = hasattr(fname, "readline")
    try:
        if fname_is_filelike:
            f = fname
        else:
            f = open(fname)
        headers = f.readline().lstrip("#").strip().split()
        try:
            df = pd.read_csv(f, sep=r"\s+", quotechar="'", header=None)
            df.columns = headers
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=headers)
    finally:
        if not fname_is_filelike:
            f.close()
    return df


def write_formatted_frame(df, fname, index=True, comment_header=True):
    """Write a data frame as a free formatted table.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to write.
    fname : str, path-like or file-like object
        Path to write file.
    index : bool, default True
        Write row names (index).
    comment_header : bool, default True
        If True, first line starts with '#' to make it a comment, otherwise
        it will be an uncommented header.

    See Also
    --------
    read_formatted_frame

    Examples
    --------
    >>> from io import StringIO
    >>> import pandas as pd
    >>> from swn.file import write_formatted_frame
    >>> df = pd.DataFrame({
    ...     "value1": [-1e10, -1e-10, 0, 1e-10, 1, 1000],
    ...     "value2": [1, 10, 100, 1000, 10000, 100000],
    ...     "value3": ["first one", "two", "three", None, "five", "six"],
    ...     }, index=[1, 12, 33, 40, 450, 6267])
    >>> df.index.name = "rno"
    >>> print(df)
                value1  value2     value3
    rno                                  
    1    -1.000000e+10       1  first one
    12   -1.000000e-10      10        two
    33    0.000000e+00     100      three
    40    1.000000e-10    1000       None
    450   1.000000e+00   10000       five
    6267  1.000000e+03  100000        six
    >>> f = StringIO()
    >>> write_formatted_frame(df, f)
    >>> _ = f.seek(0)
    >>> print(f.read(), end="")
    # rno        value1  value2  value3
    1     -1.000000e+10       1  'first one'
    12    -1.000000e-10      10   two
    33     0.000000e+00     100   three
    40     1.000000e-10    1000
    450    1.000000e+00   10000   five
    6267   1.000000e+03  100000   six
    """  # noqa
    if not isinstance(df, pd.DataFrame):
        raise TypeError("expected df to be a pandas.DataFrame")
    fname_is_filelike = hasattr(fname, "write")

    if len(df) == 0:
        # Special case with no rows
        line = "# " if comment_header else ""
        if index:
            line += (df.index.name or "index") + " "
        line += (" ".join(df.columns)) + "\n"
        try:
            if fname_is_filelike:
                f = fname
            else:
                f = open(fname, "w")
            f.write(line)
        finally:
            if not fname_is_filelike:
                f.close()
        return

    df = df.copy()
    if comment_header:
        if index:
            if df.index.name is None:
                df.index.name = "index"
            elif not df.index.name.startswith("#"):
                df.index.name = "# " + df.index.name
        else:
            first = df.columns[0]
            df.rename(columns={first: "#" + first}, inplace=True)
    formatters = {}
    # scan for str type columns
    for name in list(df.columns):
        # add single quotes around items with space chars
        try:
            sel = df[name].str.contains(" ").fillna(False)
        except AttributeError:
            continue
        na = df[name].isna()
        if sel.any():
            df.loc[sel, name] = df.loc[sel, name].map(lambda x: f"'{x}'")
            df.loc[~sel, name] = df.loc[~sel, name].map(lambda x: f" {x} ")
        else:
            df[name] = df[name].astype(str)
        if na.any():
            df.loc[na, name] = ""
        # left-justify column
        ljust = max(df[name].str.len().max(), len(name))
        if len(name) < ljust:
            df.rename(columns={name: name.ljust(ljust)}, inplace=True)
            name = name.ljust(ljust)
        formatters[name] = f"{{:<{ljust}s}}".format
    # format the table to string
    out = df.to_string(header=True, index=index, formatters=formatters)
    lines = out.split("\n")
    if index:
        # combine the first two lines
        line = lines[1].rstrip()
        lines[0] = line + lines.pop(0)[len(line):]
    elif comment_header:
        # Move '#' to start of line
        line = lines[0]
        pos = line.index("#")
        if pos > 0:
            llist = list(line)
            llist[pos] = " "
            llist[0] = "#"
            line = "".join(llist)
            lines[0] = line
    try:
        if fname_is_filelike:
            f = fname
        else:
            f = open(fname, "w")
        for line in lines:
            f.write(line.rstrip() + "\n")
    finally:
        if not fname_is_filelike:
            f.close()
