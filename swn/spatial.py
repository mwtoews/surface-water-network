# -*- coding: utf-8 -*-
"""Spatial methods."""
import geopandas
import numpy as np
import pandas as pd
from shapely import wkt
import pyproj
try:
    from geopandas.tools import sjoin
except ImportError:
    sjoin = False

try:
    import rtree
    from rtree.index import Index
except ImportError:
    rtree = False

pyproj_CRS = getattr(pyproj, 'CRS', None)
if pyproj_CRS is None:
    pyproj_ver = 1
else:
    pyproj_ver = 2

# default threshold size of geometries when Rtree index is built
rtree_threshold = 100


def get_sindex(gdf):
    """Get or build an R-Tree spatial index.

    Particularly useful for geopandas<0.2.0;>0.7.0
    """
    sindex = None
    if (isinstance(gdf, geopandas.GeoDataFrame) and
            hasattr(gdf.geometry, 'sindex')):
        sindex = gdf.geometry.sindex
    elif isinstance(gdf, geopandas.GeoSeries) and hasattr(gdf, 'sindex'):
        sindex = gdf.sindex
    if sindex is not None:
        if hasattr(sindex, 'nearest'):
            # probably rtree.index.Index or future PyGEOSSTRTreeIndex method
            return sindex
        else:
            # probably PyGEOSSTRTreeIndex from geopandas>=0.7.0
            # but unfortunately, 'nearest' is required
            sindex = None
    if rtree and len(gdf) >= rtree_threshold:
        # Manually populate a 2D spatial index for speed
        sindex = Index()
        # slow, but reliable
        for idx, (segnum, row) in enumerate(gdf.bounds.iterrows()):
            sindex.add(idx, tuple(row))
    return sindex


def interp_2d_to_3d(gs, grid, gt):
    """Interpolate 2D vector to a 3D grid using a georeferenced grid.

    Parameters
    ----------
    gs : geopandas.GeoSeries
        Input geopandas GeoSeries
    grid : np.ndarray
        2D array of values, e.g. DEM
    gt : tuple
        GDAL-style geotransform coefficients for grid

    Returns
    -------
    geopandas.GeoSeries
        With 3rd dimension values interpolated from grid.

    """
    assert gt[1] > 0, gt[1]
    assert gt[2] == 0, gt[2]
    assert gt[4] == 0, gt[4]
    assert gt[5] < 0, gt[5]
    hx = gt[1] / 2.0
    hy = gt[5] / 2.0
    div = gt[1] * gt[5]
    ny, nx = grid.shape
    ar = np.pad(grid, 1, 'symmetric')

    def geom2dto3d(geom):
        x, y = geom.xy
        x = np.array(x)
        y = np.array(y)
        # Determine outside points
        outside = (
            (x < gt[0]) | (x > (gt[0] + nx * gt[1])) |
            (y > gt[3]) | (y < (gt[3] + ny * gt[5])))
        if outside.any():
            raise ValueError('{0} coordinates are outside grid'
                             .format(outside.sum()))
        # Use half raster cell widths for cell center values
        fx = (x - (gt[0] + hx)) / gt[1]
        fy = (y - (gt[3] + hy)) / gt[5]
        ix1 = np.floor(fx).astype(np.int32)
        iy1 = np.floor(fy).astype(np.int32)
        ix2 = ix1 + 1
        iy2 = iy1 + 1
        # Calculate differences from point to bounding raster midpoints
        dx1 = x - (gt[0] + ix1 * gt[1] + hx)
        dy1 = y - (gt[3] + iy1 * gt[5] + hy)
        dx2 = (gt[0] + ix2 * gt[1] + hx) - x
        dy2 = (gt[3] + iy2 * gt[5] + hy) - y
        # Use a 1-padded array to interpolate edges nicely, so add 1 to index
        ix1 += 1
        ix2 += 1
        iy1 += 1
        iy2 += 1
        # Use the differences to weigh the four raster values
        z = (ar[iy1, ix1] * dx2 * dy2 / div +
             ar[iy1, ix2] * dx1 * dy2 / div +
             ar[iy2, ix1] * dx2 * dy1 / div +
             ar[iy2, ix2] * dx1 * dy1 / div)
        return type(geom)(zip(x, y, z))

    return gs.apply(geom2dto3d)


def wkt_to_dataframe(wkt_list, geom_name='geometry'):
    """Convert list of WKT strings to a DataFrame."""
    df = pd.DataFrame({'wkt': wkt_list})
    df[geom_name] = df['wkt'].apply(wkt.loads)
    return df


def wkt_to_geodataframe(wkt_list, geom_name='geometry'):
    """Convert list of WKT strings to a GeoDataFrame."""
    return geopandas.GeoDataFrame(
            wkt_to_dataframe(wkt_list, geom_name), geometry=geom_name)


def wkt_to_geoseries(wkt_list, geom_name=None):
    """Convert list of WKT strings to a GeoSeries."""
    geom = geopandas.GeoSeries([wkt.loads(x) for x in wkt_list])
    if geom_name is not None:
        geom.name = geom_name
    return geom


def force_2d(gs):
    """Force any geometry GeoSeries as 2D geometries."""
    return wkt_to_geoseries(gs.apply(wkt.dumps, output_dimension=2))


def round_coords(gs, rounding_precision=3):
    """Round coordinate precision of a GeoSeries."""
    return wkt_to_geoseries(
            gs.apply(wkt.dumps, rounding_precision=rounding_precision))


def compare_crs(sr1, sr2):
    """Compare two crs, flexible on crs type."""
    crs1 = get_crs(sr1)
    crs2 = get_crs(sr2)
    try:
        are_eq = crs1.equals(crs2)
    except AttributeError:
        are_eq = crs1 == crs2
    return crs1, crs2, are_eq


def get_crs(s):
    """Get crs from variable argument type.

    Returns pyproj CRS instance (for pyproj>=2.0) or Proj instance for older.
    """
    if s is None:
        return None
    elif pyproj_CRS is not None and isinstance(s, pyproj_CRS):
        return s
    try:
        s = s['init']
    except TypeError:
        pass
    if pyproj_ver >= 2:
        try:
            s = int(s)
        except ValueError:
            pass
        if isinstance(s, int):
            return pyproj.CRS.from_epsg(s)
        if s.startswith("+init="):
            s = s[6:]
        if s.startswith('+') and hasattr(pyproj.CRS, 'from_proj4'):
            return pyproj.CRS.from_proj4(s)
        else:
            return pyproj.CRS.from_string(s)
    else:
        if isinstance(s, int):
            crs = pyproj.Proj(init='epsg:' + str(s))
        elif "init" not in s:
            crs = pyproj.Proj(init=s)
        else:
            crs = pyproj.Proj(s)
    return crs
