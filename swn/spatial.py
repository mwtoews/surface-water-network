"""Spatial methods."""

__all__ = [
    "get_sindex", "interp_2d_to_3d",
    "wkt_to_dataframe", "wkt_to_geodataframe", "wkt_to_geoseries",
    "force_2d", "round_coords", "compare_crs", "get_crs",
    "find_segnum_in_swn"
]

import geopandas
import numpy as np
import pandas as pd
import pyproj
from shapely import wkt
from shapely.geometry import Point

try:
    from geopandas.tools import sjoin
except ImportError:
    sjoin = False

try:
    import rtree
    from rtree.index import Index
except ImportError:
    rtree = False

# default threshold size of geometries when Rtree index is built
rtree_threshold = 100


def get_sindex(gdf):
    """Get or build an R-Tree spatial index.

    Particularly useful for geopandas<0.2.0;>0.7.0;0.9.0
    """
    sindex = None
    if (hasattr(gdf, '_rtree_sindex')):
        return getattr(gdf, '_rtree_sindex')
    if (isinstance(gdf, geopandas.GeoDataFrame) and
            hasattr(gdf.geometry, 'sindex')):
        sindex = gdf.geometry.sindex
    elif isinstance(gdf, geopandas.GeoSeries) and hasattr(gdf, 'sindex'):
        sindex = gdf.sindex
    if sindex is not None:
        if (hasattr(sindex, "nearest") and
                sindex.__class__.__name__ != "PyGEOSSTRTreeIndex"):
            # probably rtree.index.Index
            return sindex
        else:
            # probably PyGEOSSTRTreeIndex but unfortunately, 'nearest'
            # with 'num_results' is required
            sindex = None
    if rtree and len(gdf) >= rtree_threshold:
        # Manually populate a 2D spatial index for speed
        sindex = Index()
        # slow, but reliable
        for idx, item in enumerate(gdf.bounds.itertuples()):
            sindex.add(idx, item[1:])
        # cache the index for later
        setattr(gdf, '_rtree_sindex', sindex)
    return sindex


def interp_2d_to_3d(gs, grid, gt):
    """Interpolate 2D vector to a 3D grid using a georeferenced grid.

    Parameters
    ----------
    gs : geopandas.GeoSeries
        Input geopandas GeoSeries
    grid : array_like
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
            raise ValueError(f'{outside.sum()} coordinates are outside grid')
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

def visible_wkt(geom):
    """Re-generate geometry to only visible WKT, erase machine precision."""
    return wkt.loads(geom.wkt)

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

    Returns pyproj CRS instance.
    """
    if s is None:
        return None
    if isinstance(s, pyproj.CRS):
        return s
    if isinstance(s, int):
        return pyproj.CRS.from_epsg(s)
    # Remove +init
    if isinstance(s, str) and s.startswith("+init="):
        s = s[6:]
    elif isinstance(s, dict) and "init" in s.keys() and len(s) == 1:
        s = s["init"]
    return pyproj.CRS(s)


def find_segnum_in_swn(n, geom):
    """Return segnum within a catchment or nearest a segment to a geometry.

    If a catchment polygon is provided, this is used to find if the point
    is contained in the catchment, otherwise the closest segment is found.

    Parameters
    ----------
    n : SurfaceWaterNetwork
        A surface water network to evaluate.
    geom : various
        A geometry input, one of:
            - GeoSeries
            - tuple of coordinates (x, y)
            - list of coordinate tuples

    Returns
    -------
    pandas.DataFrame
        Resulting DataFrame has columns segnum, dist_to_segnum, and
        (if catchments are found) is_within_catchment.

    """
    from swn.core import SurfaceWaterNetwork

    if not isinstance(n, SurfaceWaterNetwork):
        raise TypeError('expected an instance of SurfaceWaterNetwork')
    elif isinstance(geom, geopandas.GeoSeries):
        pass
    elif isinstance(geom, tuple):
        geom = geopandas.GeoSeries([Point(geom)])
    elif isinstance(geom, list):
        geom = geopandas.GeoSeries([Point(v) for v in geom])

    has_catchments = n.catchments is not None
    if has_catchments:
        catchments_sindex = get_sindex(n.catchments)
    segments_sindex = False  # get this, only if needed

    # test disabling sindex:
    # segments_sindex = None
    # catchments_sindex = None

    num_results = 9
    segnums = []
    is_within_catchments = []
    dist_to_segnums = []
    for _, g in geom.iteritems():
        segnum = None
        if has_catchments:
            if catchments_sindex:
                subsel = sorted(catchments_sindex.intersection(g.bounds))
                contains = [n.catchments.index[idx] for idx in subsel
                            if n.catchments.iloc[idx].contains(g)]
            else:
                contains = [idx for (idx, p) in n.catchments.iteritems()
                            if p.contains(g)]
            if len(contains) == 0:
                n.logger.debug('geom %s not contained any catchment', g.wkt)
                is_within_catchment = False
            elif len(contains) > 1:
                n.logger.warning('geom %s contained in more than one of %s',
                                 g.wkt, contains)
                segnum = contains[0]  # take the first
                is_within_catchment = True
            else:
                segnum = contains[0]
                is_within_catchment = True
            is_within_catchments.append(is_within_catchment)
        if segnum is None:
            # Find nearest segment
            if segments_sindex is False:
                segments_sindex = get_sindex(n.segments)
            if segments_sindex:
                nearest = n.segments.iloc[
                    sorted(segments_sindex.nearest(g.bounds, num_results))]\
                    .distance(g).sort_values()
            else:
                nearest = n.segments.distance(g).sort_values()[:num_results]
            nearest_sel = nearest.iloc[0] == nearest
            segnum = nearest[nearest_sel].index[0]  # take the first
            if nearest_sel.sum() > 1:
                n.logger.warning('geom %s is nearest to more than one of %s',
                                 g.wkt, nearest[nearest_sel].index.to_list())
        segnums.append(segnum)
        dist_to_segnums.append(g.distance(n.segments.loc[segnum, "geometry"]))

    # Assemble results
    ret = pd.DataFrame({'segnum': segnums}, index=geom.index)
    ret['dist_to_segnum'] = dist_to_segnums
    if has_catchments:
        ret['is_within_catchment'] = is_within_catchments
    return ret
