"""Spatial methods."""

__all__ = [
    "bias_substring",
    "compare_crs",
    "find_location_pairs",
    "force_2d",
    "get_crs",
    "get_z_coords",
    "interp_2d_to_3d",
    "location_pair_geoms",
    "round_coords",
]

from warnings import warn

import geopandas
import numpy as np
import pandas as pd
import pyproj
import shapely
from shapely import wkb, wkt
from shapely.geometry import MultiLineString, Point

try:
    from geopandas.tools import sjoin
except ImportError:
    sjoin = False

try:
    import rtree
    from rtree.index import Index
except ImportError:
    rtree = False

from .compat import SHAPELY_GE_20
from .util import is_location_frame

# default threshold size of geometries when Rtree index is built
rtree_threshold = 100


def get_sindex(gdf):
    """Get or build an R-Tree spatial index.

    .. deprecated:: 0.6
        This method is no longer used.
    """
    warn("get_sindex is no longer used", DeprecationWarning, stacklevel=2)
    sindex = None
    if hasattr(gdf, "_rtree_sindex"):
        return getattr(gdf, "_rtree_sindex")
    if isinstance(gdf, geopandas.GeoDataFrame) and hasattr(gdf.geometry, "sindex"):
        sindex = gdf.geometry.sindex
    elif isinstance(gdf, geopandas.GeoSeries) and hasattr(gdf, "sindex"):
        sindex = gdf.sindex
    if sindex is not None:
        if (
            hasattr(sindex, "nearest")
            and sindex.__class__.__name__ != "PyGEOSSTRTreeIndex"
        ):
            # probably rtree.index.Index
            return sindex
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
        setattr(gdf, "_rtree_sindex", sindex)
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
    ar = np.pad(grid, 1, "symmetric")

    def geom2dto3d(geom):
        if geom.geom_type == "MultiLineString":
            # Recursive call
            return MultiLineString([geom2dto3d(part) for part in geom.geoms])
        x, y = geom.xy
        x = np.array(x)
        y = np.array(y)
        # Determine outside points
        outside = (
            (x < gt[0])
            | (x > (gt[0] + nx * gt[1]))
            | (y > gt[3])
            | (y < (gt[3] + ny * gt[5]))
        )
        if outside.any():
            raise ValueError(f"{outside.sum()} coordinates are outside grid")
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
        z = (
            ar[iy1, ix1] * dx2 * dy2 / div
            + ar[iy1, ix2] * dx1 * dy2 / div
            + ar[iy2, ix1] * dx2 * dy1 / div
            + ar[iy2, ix2] * dx1 * dy1 / div
        )
        return type(geom)(zip(x, y, z))

    return gs.apply(geom2dto3d)


def get_z_coords(geom):
    """Get list of z-coordinates from geometry."""
    if geom.is_empty or not geom.has_z:
        return []
    if geom.geom_type == "LineString":
        return [c[2] for c in geom.coords[:]]
    if geom.geom_type == "Point":
        return [geom.z]
    if geom.geom_type.startswith("Multi") or geom.geom_type.startswith(
        "GeometryCollection"
    ):
        # recurse and flatten
        t = [get_z_coords(sg) for sg in geom.geoms]
        return [item for slist in t for item in slist]
    return []


def wkt_to_dataframe(wkt_list, geom_name="geometry"):
    """Convert list of WKT strings to a DataFrame.

    .. deprecated:: 0.6
        Use :meth:`geopandas.GeoSeries.from_wkt` with ``pandas.DataFrame``
    """
    warn(
        "wkt_to_dataframe: Use geopandas.GeoSeries.from_wkt with "
        "pandas.DataFrame instead",
        DeprecationWarning,
        stacklevel=2,
    )
    df = pd.DataFrame({"wkt": wkt_list})
    df[geom_name] = df["wkt"].apply(wkt.loads)
    return df.drop(columns="wkt")


def wkt_to_geodataframe(wkt_list, geom_name="geometry"):
    """Convert list of WKT strings to a GeoDataFrame.

    .. deprecated:: 0.6
        Use :meth:`geopandas.GeoSeries.from_wkt` with ``.to_frame("geometry")``
    """
    warn(
        "wkt_to_geodataframe: Use geopandas.GeoSeries.from_wkt with "
        '.to_frame("geometry") instead',
        DeprecationWarning,
        stacklevel=2,
    )
    gdf = geopandas.GeoSeries.from_wkt(wkt_list).to_frame(geom_name)
    return gdf.set_geometry(geom_name)


def wkt_to_geoseries(wkt_list, geom_name=None):
    """Convert list of WKT strings to a GeoSeries.

    .. deprecated:: 0.6
        Use :meth:`geopandas.GeoSeries.from_wkt`
    """
    warn(
        "wkt_to_geoseries: use geopandas.GeoSeries.from_wkt instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return geopandas.GeoSeries.from_wkt(wkt_list, name=geom_name)


def force_2d(gs):
    """Force any geometry GeoSeries as 2D geometries."""
    if SHAPELY_GE_20:
        return gs.apply(shapely.force_2d)
    return geopandas.GeoSeries.from_wkb(gs.apply(wkb.dumps, output_dimension=2))


def round_coords(gs, rounding_precision=3):
    """Round coordinate precision of a GeoSeries."""
    return geopandas.GeoSeries.from_wkt(
        gs.apply(wkt.dumps, rounding_precision=rounding_precision)
    )


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


def bias_substring(gs, downstream_bias, end_cut=1e-10):
    """Create substring of LineString GeoSeries with a bias value.

    Parameters
    ----------
    gs : geopandas.GeoSeries
        LineString GeoSeries to process.
    downstream_bias : float
        Bias value, where 0 is no bias on either end of linestring, negative
        removes downstream part of line to produce a distance bias towards
        the upstream end of lines, and a positive bias removes upstream part
        to bias the downstream end of lines. Valid range is -1 to +1.
    end_cut : float, default 1e-10
        The extra amount to remove on each end. Valid range is 0 to 0.5.

    Returns
    -------
    geopandas.GeoSeries

    """
    from shapely.ops import substring

    if not isinstance(gs, geopandas.GeoSeries):
        raise TypeError("expected 'gs' as an instance of GeoSeries")
    if not (-1.0 <= downstream_bias <= 1.0):
        raise ValueError("downstream_bias must be between -1 and 1")
    if not (0.0 <= end_cut <= 0.5):
        raise ValueError("end_cut must between 0 and 0.5")

    us = 0.0
    ds = 1.0
    if downstream_bias > 0.0:
        ds = 1.0 - downstream_bias
    else:
        us = float(-downstream_bias)
    # move start/end points so they are not exactly at start/end
    args = tuple(np.clip([us, ds], end_cut, 1.0 - end_cut))
    if SHAPELY_GE_20 and gs.has_z.any():
        gs = shapely.force_2d(gs)
    return gs.apply(substring, args=args, normalized=True)


def find_segnum_in_swn(n, geom):
    """Return segnum within a catchment or nearest a segment to a geometry.

    If a catchment polygon is provided, this is used to find if the point
    is contained in the catchment, otherwise the closest segment is found.

    .. deprecated:: 0.5
        Use :py:meth:`SurfaceWaterNetwork.locate_geoms` instead.

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
    warn("Use the n.locate_geoms method", DeprecationWarning, stacklevel=2)
    if isinstance(geom, tuple):
        geom = geopandas.GeoSeries([Point(geom)])
    elif isinstance(geom, list):
        geom = geopandas.GeoSeries([Point(v) for v in geom])
    if geom.crs is None and n.segments.crs is not None:
        geom.crs = n.segments.crs
    res = n.locate_geoms(geom)
    res["is_within_catchment"] = res.method == "catchment"
    res.rename(columns={"dist_to_seg": "dist_to_segnum"}, inplace=True)
    return pd.DataFrame(res[["segnum", "dist_to_segnum", "is_within_catchment"]])


def find_location_pairs(loc_df, n, *, all_pairs=False, exclude_branches=False):
    r"""Find pairs of locations in a surface water network that are connected.

    Parameters
    ----------
    loc_df : geopandas.GeoDataFrame or pandas.DataFrame
        Location [geo] dataframe, usually created by
        :py:meth:`SurfaceWaterNetwork.locate_geoms`.
    n : SurfaceWaterNetwork
        A surface water network to evaluate.
    all_pairs : bool, default False
        If True, find all combination pairs, including those that overlap.
        Default False will find shortest pair combinations only.
    exclude_branches : bool, default False
        If True, only pairs without branches between are permitted.

    Returns
    -------
    set
        tuple of location index (upstream, downstream).

    See Also
    --------
    location_pair_geoms : Extract linestring geometry for location pairs.

    Examples
    --------
    >>> import swn
    >>> import geopandas
    >>> from shapely import wkt
    >>> lines = geopandas.GeoSeries(list(wkt.loads('''\
    ... MULTILINESTRING(
    ...     (380 490, 370 420), (300 460, 370 420), (370 420, 420 330),
    ...     (190 250, 280 270), (225 180, 280 270), (280 270, 420 330),
    ...     (420 330, 584 250), (520 220, 584 250), (584 250, 710 160),
    ...     (740 270, 710 160), (735 350, 740 270), (880 320, 740 270),
    ...     (925 370, 880 320), (974 300, 880 320), (760 460, 735 350),
    ...     (650 430, 735 350), (710 160, 770 100), (700  90, 770 100),
    ...     (770 100, 820  40))''').geoms))
    >>> lines.index += 100
    >>> n = swn.SurfaceWaterNetwork.from_lines(lines)
    >>> obs_gs = geopandas.GeoSeries.from_wkt([
    ...    "POINT (370 400)",
    ...    "POINT (720 230)",
    ...    "POINT (780 70)",
    ...    "POINT (606 256)",
    ...    "POINT (690 170)"])
    >>> obs_gs.index += 11
    >>> obs_match = n.locate_geoms(obs_gs)
    >>> obs_match[["segnum", "seg_ndist"]]
        segnum  seg_ndist
    11     102   0.169811
    12     109   0.384615
    13     118   0.377049
    14     108   0.093093
    15     108   0.857357
    >>> sorted(swn.spatial.find_location_pairs(obs_match, n))
    [(11, 14), (12, 13), (14, 15), (15, 13)]
    >>> sorted(swn.spatial.find_location_pairs(obs_match, n, all_pairs=True))
    [(11, 13), (11, 14), (11, 15), (12, 13), (14, 13), (14, 15), (15, 13)]
    >>> swn.spatial.find_location_pairs(obs_match, n, exclude_branches=True)
    {(14, 15)}

    """
    from .core import SurfaceWaterNetwork

    # Check inputs
    if not isinstance(n, SurfaceWaterNetwork):
        raise TypeError("expected 'n' as an instance of SurfaceWaterNetwork")
    is_location_frame(loc_df, geom_required=False)
    segnum_is_in_index = loc_df.segnum.isin(n.segments.index)
    if not segnum_is_in_index.all():
        raise ValueError("loc_df has segnum values not found in surface water network")

    to_segnums_d = n.to_segnums.to_dict()
    if exclude_branches:
        from_segnums_d = n.from_segnums.to_dict()
    loc_df = loc_df[["segnum", "seg_ndist"]].assign(_="")  # also does .copy()
    loc_segnum_s = set(loc_df.segnum)
    loc_df["sequence"] = n.segments.sequence[loc_df.segnum].values
    loc_df.sort_values(["sequence", "seg_ndist"], inplace=True)
    # Start from upstream locations, querying downstream, except last
    pairs = set()
    for us_idx, us_segnum in loc_df.segnum.iloc[:-1].items():
        next_iloc = loc_df.index.get_loc(us_idx) + 1
        if all_pairs:
            # find downstream segnums in the same segnum
            sel = loc_df.iloc[next_iloc:].segnum == us_segnum
            for ds_idx in sel[sel].index:
                pairs.add((us_idx, ds_idx))
            # continue searching downstream
            cur_segnum = us_segnum
            while True:
                if cur_segnum in to_segnums_d:
                    next_segnum = to_segnums_d[cur_segnum]
                    if (
                        exclude_branches
                        and len(from_segnums_d.get(next_segnum, [])) > 1
                    ):
                        break  # stop searching due to branch
                    sel = loc_df["segnum"] == next_segnum
                    for ds_idx in sel[sel].index:
                        pairs.add((us_idx, ds_idx))
                else:
                    break  # stop searching due to no downstream location
                cur_segnum = next_segnum
        else:
            ds_idx = None
            # First case that the downstream segnum is in the same segnum
            next_loc = loc_df.iloc[next_iloc]
            if next_loc.segnum == us_segnum:
                ds_idx = next_loc.name.item()
            else:
                # otherwise search downstream
                cur_segnum = us_segnum
                while True:
                    if cur_segnum in to_segnums_d:
                        next_segnum = to_segnums_d[cur_segnum]
                        if (
                            exclude_branches
                            and len(from_segnums_d.get(next_segnum, [])) > 1
                        ):
                            break  # no pair due to branch
                        if next_segnum in loc_segnum_s:
                            ds_idx = (
                                loc_df.segnum[loc_df.segnum == next_segnum]
                                .index[0]
                                .item()
                            )
                            break  # found pair
                    else:
                        break  # no pair due to no downstream location
                    cur_segnum = next_segnum
            if ds_idx is not None:
                pairs.add((us_idx, ds_idx))
    return pairs


def location_pair_geoms(pairs, loc_df, n):
    r"""Extract linestring geometry for location pairs, that follow segments.

    Parameters
    ----------
    pairs : iterable
        Collection of location pair tuples (upstream, downstream), indexed by
        loc_df from :py:meth:`find_location_pairs`.
    loc_df : geopandas.GeoDataFrame or pandas.DataFrame
        Location [geo] dataframe, usually created by
        :py:meth:`SurfaceWaterNetwork.locate_geoms`.
    n : SurfaceWaterNetwork
        A surface water network.

    Returns
    -------
    dict
        keys are the pair tuple, and values are the linestring geometry.

    Examples
    --------
    >>> import geopandas
    >>> from shapely import wkt
    >>> import swn
    >>> lines = geopandas.GeoSeries(list(wkt.loads('''\
    ... MULTILINESTRING(
    ...     (380 490, 370 420), (300 460, 370 420), (370 420, 420 330),
    ...     (190 250, 280 270), (225 180, 280 270), (280 270, 420 330),
    ...     (420 330, 584 250), (520 220, 584 250), (584 250, 710 160),
    ...     (740 270, 710 160), (735 350, 740 270), (880 320, 740 270),
    ...     (925 370, 880 320), (974 300, 880 320), (760 460, 735 350),
    ...     (650 430, 735 350), (710 160, 770 100), (700  90, 770 100),
    ...     (770 100, 820  40))''').geoms))
    >>> lines.index += 100
    >>> n = swn.SurfaceWaterNetwork.from_lines(lines)
    >>> obs_gs = geopandas.GeoSeries.from_wkt([
    ...    "POINT (370 400)",
    ...    "POINT (720 230)",
    ...    "POINT (780 70)",
    ...    "POINT (606 256)",
    ...    "POINT (690 170)"])
    >>> obs_gs.index += 11
    >>> obs_match = n.locate_geoms(obs_gs)
    >>> pairs = swn.spatial.find_location_pairs(obs_match, n)
    >>> sorted(pairs)
    [(11, 14), (12, 13), (14, 15), (15, 13)]
    >>> pair_geoms = swn.spatial.location_pair_geoms(pairs, obs_match, n)
    >>> pair_gs = geopandas.GeoSeries(pair_geoms)
    >>> pair_gdf = geopandas.GeoDataFrame(geometry=pair_gs)
    >>> pair_gdf["length"] = pair_gdf.length
    >>> pair_gdf.sort_values("length", ascending=False, inplace=True)
    >>> pair_gdf
                                                    geometry      length
    11 14  LINESTRING (378.491 404.717, 420 330, 584 250,...  282.359779
    12 13  LINESTRING (728.462 227.692, 710 160, 770 100,...  184.465938
    15 13  LINESTRING (692.027 172.838, 710 160, 770 100,...  136.388347
    14 15       LINESTRING (595.73 241.622, 692.027 172.838)  118.340096

    """
    from shapely.ops import linemerge, substring, unary_union

    from .core import SurfaceWaterNetwork

    # Check inputs
    if not isinstance(n, SurfaceWaterNetwork):
        raise TypeError("expected 'n' as an instance of SurfaceWaterNetwork")
    is_location_frame(loc_df, geom_required=False)
    segnum_is_in_index = loc_df.segnum.isin(n.segments.index)
    if not segnum_is_in_index.all():
        raise ValueError("loc_df has segnum values not found in surface water network")

    seg_geom = force_2d(n.segments.geometry)
    geoms_d = {}
    for pair in pairs:
        upstream_idx, downstream_idx = pair
        upstream = loc_df.loc[upstream_idx]
        downstream = loc_df.loc[downstream_idx]
        if upstream.segnum == downstream.segnum:
            # pairs are in the same segnum
            geoms_d[pair] = substring(
                seg_geom[upstream.segnum],
                upstream.seg_ndist,
                downstream.seg_ndist,
                normalized=True,
            )
        else:
            try:
                segnums = n.route_segnums(upstream.segnum, downstream.segnum)
            except ConnectionError:
                geoms_d[pair] = None
                continue
            assert segnums[0] == upstream.segnum
            assert segnums[-1] == downstream.segnum
            geoms = [
                substring(
                    seg_geom[upstream.segnum], upstream.seg_ndist, 1.0, normalized=True
                )
            ]
            geoms.extend(list(seg_geom[segnums[1:-1]]))
            geoms.append(
                substring(
                    seg_geom[downstream.segnum],
                    0.0,
                    downstream.seg_ndist,
                    normalized=True,
                )
            )
            geom = unary_union(geoms)
            if geom.geom_type != "LineString":
                geom = linemerge(geom)
            geoms_d[pair] = geom
    return geoms_d
