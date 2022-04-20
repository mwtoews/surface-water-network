"""Spatial methods."""

__all__ = [
    "get_sindex", "interp_2d_to_3d",
    "wkt_to_dataframe", "wkt_to_geodataframe", "wkt_to_geoseries",
    "force_2d", "round_coords", "compare_crs", "get_crs",
    "find_segnum_in_swn", "find_geom_in_swn",
]

from warnings import warn

import geopandas
import numpy as np
import pandas as pd
import pyproj
from shapely import wkt
from shapely.geometry import LineString, Point

from swn.compat import ignore_shapely_warnings_for_object_array

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

    .. deprecated:: 0.5
        Use :py:meth:`find_geom_in_swn` instead.

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
    warn("Use spatial.find_geom_in_swn", DeprecationWarning, stacklevel=2)
    if isinstance(geom, tuple):
        geom = geopandas.GeoSeries([Point(geom)])
    elif isinstance(geom, list):
        geom = geopandas.GeoSeries([Point(v) for v in geom])
    if geom.crs is None and n.segments.crs is not None:
        geom.crs = n.segments.crs
    res = find_geom_in_swn(geom, n)
    res["is_within_catchment"] = res.method == "catchment"
    res.rename(columns={"dist_to_seg": "dist_to_segnum"}, inplace=True)
    return pd.DataFrame(
        res[["segnum", "dist_to_segnum", "is_within_catchment"]])


def find_geom_in_swn(geom, n, override={}):
    """Return GeoDataFrame of data associated in finding geometies in a swn.

    Parameters
    ----------
    geom : GeoSeries
        Geometry series input to process, e.g. stream gauge locations or
        or building footprints.
    n : SurfaceWaterNetwork
        A surface water network to evaluate.
    override : dict
        Override matches, where key is the index from geom, and the value is
        segnum. If value is None, no match is performed.

    Notes
    -----

    Seveal methods are used to pair the geometry with one segnum:

        1. override: explicit pairs are provided as a dict, with key for the
           geom index, and value with the segnum.
        2. catchment: if catchments are part of the surface water network,
           find the catchment polygons that contain the geometries. Input
           polygons that intersect with more than one catchment are matched
           with the catchment with the largest intersection polygon area.
        3. nearest: find the segment lines that are nearest to the input
           geometries. Input polygons that intersect with more than one
           segment are matched with the largest intersection line length.

    It is advised that outputs are checked in a GIS to ensure correct
    matches. Any error can be corrected using an override entry.

    The geometry returned by this method consists of two-coordinate line
    segments that represent the shortest distance between geom and the
    surface water network segment.

    Returns
    -------
    geopandas.GeoDataFrame
        Resulting GeoDataFrame has columns geometry (always LineString),
        method (override, catchment or nearest), segnum, seg_ndist (normalized
        distance along segment), and dist_to_seg (distance to segment).

    Examples
    --------
    >>> import swn
    >>> from swn.spatial import find_geom_in_swn, wkt_to_geoseries
    >>> lines = wkt_to_geoseries([
    ...    "LINESTRING (60 100, 60  80)",
    ...    "LINESTRING (40 130, 60 100)",
    ...    "LINESTRING (70 130, 60 100)"])
    >>> n = swn.SurfaceWaterNetwork.from_lines(lines)
    >>> obs_geom = wkt_to_geoseries([
    ...    "POINT (56 103)",
    ...    "LINESTRING (58 90, 62 90)",
    ...    "POLYGON ((60 107, 59 113, 61 113, 60 107))",
    ...    "POINT (55 130)"])
    >>> obs_geom.index += 101
    >>> obs_match = find_geom_in_swn(obs_geom, n, override={104: 2})
    >>> print(obs_match[["method", "segnum", "seg_ndist", "dist_to_seg"]])
           method  segnum  seg_ndist  dist_to_seg
    101   nearest       1   0.869231     1.664101
    102   nearest       0   0.500000     0.000000
    103   nearest       2   0.790000     2.213594
    104  override       2   0.150000    14.230249
    """
    from swn.core import SurfaceWaterNetwork

    if not isinstance(geom, geopandas.GeoSeries):
        raise TypeError("expected 'geom' as an instance of GeoSeries")
    elif not isinstance(n, SurfaceWaterNetwork):
        raise TypeError("expected 'n' as an instance of SurfaceWaterNetwork")

    # initialise return data frame
    res = geopandas.GeoDataFrame(geometry=geom)
    res["method"] = ""
    res["segnum"] = n.END_SEGNUM

    if not isinstance(override, dict):
        raise TypeError("expected 'override' as an instance of dict")
    if override:
        override = override.copy()
        override_keys_s = set(override.keys())
        missing_geom_idx_s = override_keys_s.difference(res.index)
        if missing_geom_idx_s:
            warn(
                f"{len(missing_geom_idx_s)} override keys don't match "
                f"point geom.index: {sorted(missing_geom_idx_s)}",
                UserWarning, stacklevel=2)
            for k in missing_geom_idx_s:
                del override[k]
        override_values_s = set(override.values())
        if None in override_values_s:
            override.update(
                {k: n.END_SEGNUM for k, v in override.items() if v is None})
            override_values_s.remove(None)
        missng_segnum_idx_s = override_values_s.difference(n.segments.index)
        if missng_segnum_idx_s:
            warn(
                f"{len(missng_segnum_idx_s)} override values don't match "
                f"segments.index: {sorted(missng_segnum_idx_s)}",
                UserWarning, stacklevel=2)
            for k, v in override.copy().items():
                if v in missng_segnum_idx_s:
                    del override[k]
        if override:
            res.segnum.update(override)
            res.loc[override.keys(), "method"] = "override"

    # First, look at geoms in catchments
    if n.catchments is not None:
        sel = (res.method == "")
        if sel.any():
            catchments_df = n.catchments.to_frame()
            if catchments_df.crs is None and n.segments.crs is not None:
                catchments_df.crs = n.segments.crs
            match_s = geopandas.sjoin(
                res[sel], catchments_df, "inner")["index_right"]
            match_s.name = "segnum"
            match_s.index.name = "gidx"
            match = match_s.reset_index()
            # geom may cover more than one catchment
            duplicated = match.gidx.duplicated(keep=False)
            if duplicated.any():
                for gidx, segnums in match[duplicated].groupby("gidx").segnum:
                    # find catchment with highest area of intersection
                    ca = segnums.to_frame()
                    ca["area"] = catchments_df.loc[segnums].intersection(
                        geom.loc[gidx]).area.values
                    ca.sort_values("area", ascending=False, inplace=True)
                    match.drop(index=ca.index[1:], inplace=True)
            res.loc[match.gidx, "segnum"] = match.segnum.values
            res.loc[match.gidx, "method"] = "catchment"

    # Match geometry to closest segment
    sel = res.method == ""
    if not sel.any():
        pass
    elif hasattr(geopandas, "sjoin_nearest"):
        # from geopandas 0.10.0 (October 3, 2021)
        match_s = geopandas.sjoin_nearest(
            res[sel], n.segments[["geometry"]], "inner")["index_right"]
        match_s.name = "segnum"
        match_s.index.name = "gidx"
        match = match_s.reset_index()
        duplicated = match.gidx.duplicated(keep=False)
        if duplicated.any():
            for gidx, segnums in match[duplicated].groupby("gidx").segnum:
                g = geom.loc[gidx]
                sg = n.segments.geometry[segnums]
                sl = segnums.to_frame()
                sl["length"] = sg.intersection(g).length.values
                sl["start"] = sg.interpolate(0.0).intersects(g).values
                if sl.length.max() > 0.0:
                    # find segment with highest length of intersection
                    sl.sort_values("length", ascending=False, inplace=True)
                elif sl.start.sum() == 1:
                    # find start segment at a junction
                    sl.sort_values("start", ascending=False, inplace=True)
                else:
                    sl.sort_values("segnum", inplace=True)
                match.drop(index=sl.index[1:], inplace=True)
        res.loc[match.gidx, "segnum"] = match.segnum.values
        res.loc[match.gidx, "method"] = "nearest"
    else:  # slower method
        for gidx, g in geom[sel].iteritems():
            if g.is_empty:
                continue
            dists = n.segments.distance(g).sort_values()
            segnums = dists.index[dists.iloc[0] == dists]
            if len(segnums) == 1:
                segnum = segnums[0]
            else:
                sg = n.segments.geometry[segnums]
                sl = pd.DataFrame(index=segnums)
                sl["length"] = sg.intersection(g).length
                sl["start"] = sg.interpolate(0.0).intersects(g)
                if sl.length.max() > 0.0:
                    # find segment with highest length of intersection
                    sl.sort_values("length", ascending=False, inplace=True)
                elif sl.start.sum() == 1:
                    # find start segment at a junction
                    sl.sort_values("start", ascending=False, inplace=True)
                else:
                    sl.sort_index(inplace=True)
                segnum = sl.index[0]
            res.loc[gidx, "segnum"] = segnum
            res.loc[gidx, "method"] = "nearest"

    # For non-point geometries, convert to point
    sel = (res.geom_type != "Point") & (res.segnum != n.END_SEGNUM)
    if sel.any():
        from shapely.ops import nearest_points
        with ignore_shapely_warnings_for_object_array():
            res.geometry.loc[sel] = res.loc[sel].apply(
                lambda f: nearest_points(
                    n.segments.geometry[f.segnum], f.geometry)[1], axis=1)

    # Add attributes for match
    sel = res.segnum != n.END_SEGNUM
    res["seg_ndist"] = res.loc[sel].apply(
        lambda f: n.segments.geometry[f.segnum].project(
            f.geometry, normalized=True), axis=1)
    # Line between geometry and line segment
    with ignore_shapely_warnings_for_object_array():
        res["link"] = res.loc[sel].apply(
            lambda f: LineString(
                [f.geometry, n.segments.geometry[f.segnum].interpolate(
                    f.seg_ndist, normalized=True)]), axis=1)
    if (~sel).any():
        linestring_empty = wkt.loads("LINESTRING EMPTY")
        for idx in res[~sel].index:
            res.at[idx, "link"] = linestring_empty
    res.set_geometry("link", drop=True, inplace=True)
    res["dist_to_seg"] = res[sel].length
    return res
