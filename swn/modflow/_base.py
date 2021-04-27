# -*- coding: utf-8 -*-
"""Abstract base class for a surface water network for MODFLOW."""

import geopandas
import pickle
import numpy as np
import pandas as pd
from itertools import combinations
from shapely import wkt
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import linemerge

from swn.core import SurfaceWaterNetwork
from swn.spatial import get_sindex, compare_crs


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
        if not find_spec("flopy"):
            raise ImportError(self.__class__.__name__ + " requires flopy")
        from swn.logger import get_logger, logging
        if logger is None:
            self.logger = get_logger(self.__class__.__name__)
        elif isinstance(logger, logging.Logger):
            self.logger = logger
        else:
            raise ValueError(
                "expected 'logger' to be Logger; found " + str(type(logger)))
        self.logger.info("creating new %s object", self.__class__.__name__)
        self.segments = None
        self.diversions = None

    def __getstate__(self):
        """Serialize object attributes for pickle dumps."""
        return dict(self)

    def to_pickle(self, path, protocol=pickle.HIGHEST_PROTOCOL):
        """Pickle (serialize) non-flopy object data to file.

        Parameters
        ----------
        path : str
            File path where the pickled object will be stored.
        protocol : int
            Default is pickle.HIGHEST_PROTOCOL.

        """
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=protocol)

    @classmethod
    def from_pickle(cls, path, model):
        """Read a pickled format from a file.

        Parameters
        ----------
        path : str
            File path where the pickled object will be stored.
        model : flopy.modflow.Modflow or flopy.mf6.ModflowGwf
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
            return getattr(self, "_model", None)
        except AttributeError:
            self.logger.error("'model' property not set")

    @model.setter
    def model(self, model):
        raise NotImplementedError()

    @classmethod
    def from_swn_flopy(
            cls, swn, model, domain_action="freeze",
            reach_include_fraction=0.2):
        """Create a MODFLOW structure from a surface water network.

        Parameters
        ----------
        swn : swn.SurfaceWaterNetwork
            Instance of a SurfaceWaterNetwork.
        model : flopy.modflow.Modflow or flopy.mf6.ModflowGwf
            Instance of a flopy MODFLOW groundwater flow model.
        domain_action : str, optional
            Action to handle IBOUND or IDOMAIN:
                - ``freeze`` : Freeze domain, but clip streams to fit bounds.
                - ``modify`` : Modify domain to fit streams, where possible.
        reach_include_fraction : float or pandas.Series, optional
            Fraction of cell size used as a threshold distance to determine if
            reaches outside the active grid should be included to a cell.
            Based on the furthest distance of the line and cell geometries.
            Default 0.2 (e.g. for a 100 m grid cell, this is 20 m).

        Returns
        -------
        obj
        """
        if cls.__name__ == "SwnModflow":
            domain_label = "ibound"
        elif cls.__name__ == "SwnMf6":
            domain_label = "idomain"
        else:
            raise TypeError("unsupported subclass " + repr(cls))
        if not isinstance(swn, SurfaceWaterNetwork):
            raise ValueError("swn must be a SurfaceWaterNetwork object")
        elif domain_action not in ("freeze", "modify"):
            raise ValueError("domain_action must be one of freeze or modify")
        obj = cls()
        # Attach a few things to the fresh object
        obj.model = model
        obj.segments = swn.segments.copy()
        obj._swn = swn
        # Make sure model CRS and segments CRS are the same (if defined)
        crs = None
        segments_crs = getattr(obj.segments.geometry, "crs", None)
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
                "CRS for segments and modelgrid are different: {0} vs. {1}"
                .format(segments_crs, modelgrid_crs))
        crs = segments_crs or modelgrid_crs
        # Make sure their extents overlap
        minx, maxx, miny, maxy = modelgrid.extent
        model_bbox = box(minx, miny, maxx, maxy)
        rstats = obj.segments.bounds.describe()
        segments_bbox = box(
                rstats.loc["min", "minx"], rstats.loc["min", "miny"],
                rstats.loc["max", "maxx"], rstats.loc["max", "maxy"])
        if model_bbox.disjoint(segments_bbox):
            raise ValueError("modelgrid extent does not cover segments extent")
        # More careful check of overlap of lines with grid polygons
        obj.logger.debug("building model grid cell geometries")
        dis = obj.model.dis
        cols, rows = np.meshgrid(np.arange(dis.ncol.data),
                                 np.arange(dis.nrow.data))
        if domain_label == "IBOUND":
            domain = obj.model.bas6.ibound[0].array.copy()
        else:
            domain = dis.idomain.array[0].copy()
        num_domain_modified = 0
        grid_df = pd.DataFrame({"row": rows.flatten(), "col": cols.flatten()})
        grid_df.set_index(["row", "col"], inplace=True)
        grid_df[domain_label] = domain.flatten()
        if domain_action == "freeze" and (domain == 0).any():
            # Remove any inactive grid cells from analysis
            grid_df = grid_df.loc[grid_df[domain_label] != 0]
        # Determine grid cell size
        col_size = np.median(dis.delr.array)
        if dis.delr.array.min() != dis.delr.array.max():
            obj.logger.warning(
                "assuming constant column spacing %s", col_size)
        row_size = np.median(dis.delc.array)
        if dis.delc.array.min() != dis.delc.array.max():
            obj.logger.warning(
                "assuming constant row spacing %s", row_size)
        cell_size = (row_size + col_size) / 2.0
        # Note: modelgrid.get_cell_vertices(row, col) is slow!
        xv = modelgrid.xvertices
        yv = modelgrid.yvertices
        r, c = [np.array(s[1])
                for s in grid_df.reset_index()[["row", "col"]].iteritems()]
        cell_verts = zip(
            zip(xv[r, c], yv[r, c]),
            zip(xv[r, c + 1], yv[r, c + 1]),
            zip(xv[r + 1, c + 1], yv[r + 1, c + 1]),
            zip(xv[r + 1, c], yv[r + 1, c])
        )
        # Add dataframe of model grid cells to object
        obj.grid_cells = grid_cells = geopandas.GeoDataFrame(
            grid_df, geometry=[Polygon(r) for r in cell_verts], crs=crs)
        # Break up source segments according to the model grid definition
        obj.logger.debug("evaluating reach data on model grid")
        grid_sindex = get_sindex(grid_cells)
        reach_include = swn._segment_series(reach_include_fraction) * cell_size
        # Make an empty DataFrame for reaches
        obj.reaches = pd.DataFrame(columns=["geometry"])
        obj.reaches.insert(1, column="row", value=pd.Series(dtype=int))
        obj.reaches.insert(2, column="col", value=pd.Series(dtype=int))
        empty_reach_df = obj.reaches.copy()  # take this before more added
        obj.reaches.insert(
            1, column="segnum",
            value=pd.Series(dtype=obj.segments.index.dtype))
        obj.reaches.insert(2, column="segndist", value=pd.Series(dtype=float))
        empty_reach_df.insert(3, column="length", value=pd.Series(dtype=float))
        empty_reach_df.insert(4, column="moved", value=pd.Series(dtype=bool))

        # recusive helper function
        def append_reach_df(df, row, col, reach_geom, moved=False):
            if reach_geom.geom_type == "LineString":
                df.loc[len(df.index)] = {
                    "geometry": reach_geom,
                    "row": row,
                    "col": col,
                    "length": reach_geom.length,
                    "moved": moved,
                }
            elif reach_geom.geom_type.startswith("Multi"):
                for sub_reach_geom in reach_geom.geoms:  # recurse
                    append_reach_df(df, row, col, sub_reach_geom, moved)
            else:
                raise NotImplementedError(reach_geom.geom_type)

        # helper function that returns early, if necessary
        def assign_short_reach(reach_df, idx, segnum):
            reach = reach_df.loc[idx]
            reach_geom = reach["geometry"]
            threshold = reach_include[segnum]
            if reach_geom.length > threshold:
                return
            cell_lengths = reach_df.groupby(["row", "col"])["length"].sum()
            this_row_col = reach["row"], reach["col"]
            this_cell_length = cell_lengths[this_row_col]
            if this_cell_length > threshold:
                return
            grid_geom = grid_cells.at[(reach["row"], reach["col"]), "geometry"]
            # determine if it is crossing the grid once or twice
            grid_points = reach_geom.intersection(grid_geom.exterior)
            split_short = (
                grid_points.geom_type == "Point" or
                (grid_points.geom_type == "MultiPoint" and
                 len(grid_points) == 2))
            if not split_short:
                return
            matches = []
            # sequence scan on reach_df
            for oidx, orch in reach_df.iterrows():
                if oidx == idx or orch["moved"]:
                    continue
                other_row_col = orch["row"], orch["col"]
                other_cell_length = cell_lengths[other_row_col]
                if (orch["geometry"].distance(reach_geom) < 1e-6 and
                        this_cell_length < other_cell_length):
                    matches.append((oidx, orch["geometry"]))
            if len(matches) == 0:
                # don't merge, e.g. reach does not connect to adjacent cell
                pass
            elif len(matches) == 1:
                # short segment is in one other cell only
                # update new row and col values, keep geometry as it is
                row_col1 = tuple(reach_df.loc[matches[0][0], ["row", "col"]])
                reach_df.loc[idx, ["row", "col", "moved"]] = row_col1 + (True,)
                # self.logger.debug(
                #    "moved short segment of %s from %s to %s",
                #    segnum, this_row_col, row_col1)
            elif len(matches) == 2:
                assert grid_points.geom_type == "MultiPoint", grid_points.wkt
                if len(grid_points) != 2:
                    obj.logger.critical(
                        "expected 2 points, found %s", len(grid_points))
                # Build a tiny DataFrame of coordinates for this reach
                reach_c = pd.DataFrame({
                    "pt": [Point(c) for c in reach_geom.coords[:]]
                })
                if len(reach_c) == 2:
                    # If this is a simple line with two coords, split it
                    reach_c.index = [0, 2]
                    reach_c.loc[1] = {
                        "pt": reach_geom.interpolate(0.5, normalized=True)}
                    reach_c.sort_index(inplace=True)
                    reach_geom = LineString(list(reach_c["pt"]))  # rebuild
                # first match assumed to be touching the start of the line
                if reach_c.at[0, "pt"].distance(matches[1][1]) < 1e-6:
                    matches.reverse()
                reach_c["d1"] = reach_c["pt"].apply(
                                lambda p: p.distance(matches[0][1]))
                reach_c["d2"] = reach_c["pt"].apply(
                                lambda p: p.distance(matches[1][1]))
                reach_c["dm"] = reach_c[["d1", "d2"]].min(1)
                # try a simple split where distances switch
                ds = reach_c["d1"] < reach_c["d2"]
                cidx = ds[ds].index[-1]
                # ensure it's not the index of either end
                if cidx == 0:
                    cidx = 1
                elif cidx == len(reach_c) - 1:
                    cidx = len(reach_c) - 2
                row1, col1 = list(reach_df.loc[matches[0][0], ["row", "col"]])
                reach_geom1 = LineString(reach_geom.coords[:(cidx + 1)])
                row2, col2 = list(reach_df.loc[matches[1][0], ["row", "col"]])
                reach_geom2 = LineString(reach_geom.coords[cidx:])
                # update the first, append the second
                reach_df.loc[idx, ["row", "col", "length", "moved"]] = \
                    (row1, col1, reach_geom1.length, True)
                reach_df.at[idx, "geometry"] = reach_geom1
                append_reach_df(reach_df, row2, col2, reach_geom2, moved=True)
                # self.logger.debug(
                #   "split and moved short segment of %s from %s to %s and %s",
                #   segnum, this_row_col, (row1, col1), (row2, col2))
            else:
                obj.logger.critical(
                    "unhandled assign_short_reach case with %d matches: %s\n"
                    "%s\n%s", len(matches), matches, reach, grid_points.wkt)

        def assign_remaining_reach(reach_df, segnum, rem):
            if rem.geom_type == "LineString":
                threshold = cell_size * 2.0
                if rem.length > threshold:
                    obj.logger.debug(
                        "remaining line segment from %s too long to merge "
                        "(%.1f > %.1f)", segnum, rem.length, threshold)
                    return
                # search full grid for other cells that could match
                if grid_sindex:
                    bbox_match = sorted(grid_sindex.intersection(rem.bounds))
                    sub = grid_cells.geometry.iloc[bbox_match]
                else:  # slow scan of all cells
                    sub = grid_cells.geometry
                assert len(sub) > 0, len(sub)
                matches = []
                for (row, col), grid_geom in sub.iteritems():
                    if grid_geom.touches(rem):
                        matches.append((row, col, grid_geom))
                if len(matches) == 0:
                    return
                threshold = reach_include[segnum]
                # Build a tiny DataFrame for just the remaining coordinates
                rem_c = pd.DataFrame({
                    "pt": [Point(c) for c in rem.coords[:]]
                })
                if len(matches) == 1:  # merge it with adjacent cell
                    row, col, grid_geom = matches[0]
                    mdist = rem_c["pt"].apply(
                                    lambda p: grid_geom.distance(p)).max()
                    if mdist > threshold:
                        obj.logger.debug(
                            "remaining line segment from %s too far away to "
                            "merge (%.1f > %.1f)", segnum, mdist, threshold)
                        return
                    append_reach_df(reach_df, row, col, rem, moved=True)
                elif len(matches) == 2:  # complex: need to split it
                    if len(rem_c) == 2:
                        # If this is a simple line with two coords, split it
                        rem_c.index = [0, 2]
                        rem_c.loc[1] = {
                            "pt": rem.interpolate(0.5, normalized=True)}
                        rem_c.sort_index(inplace=True)
                        rem = LineString(list(rem_c["pt"]))  # rebuild
                    # first match assumed to be touching the start of the line
                    if rem_c.at[0, "pt"].touches(matches[1][2]):
                        matches.reverse()
                    rem_c["d1"] = rem_c["pt"].apply(
                                    lambda p: p.distance(matches[0][2]))
                    rem_c["d2"] = rem_c["pt"].apply(
                                    lambda p: p.distance(matches[1][2]))
                    rem_c["dm"] = rem_c[["d1", "d2"]].min(1)
                    mdist = rem_c["dm"].max()
                    if mdist > threshold:
                        obj.logger.debug(
                            "remaining line segment from %s too far away to "
                            "merge (%.1f > %.1f)", segnum, mdist, threshold)
                        return
                    # try a simple split where distances switch
                    ds = rem_c["d1"] < rem_c["d2"]
                    cidx = ds[ds].index[-1]
                    # ensure it's not the index of either end
                    if cidx == 0:
                        cidx = 1
                    elif cidx == len(rem_c) - 1:
                        cidx = len(rem_c) - 2
                    row, col = matches[0][0:2]
                    rem1 = LineString(rem.coords[:(cidx + 1)])
                    append_reach_df(reach_df, row, col, rem1, moved=True)
                    row, col = matches[1][0:2]
                    rem2 = LineString(rem.coords[cidx:])
                    append_reach_df(reach_df, row, col, rem2, moved=True)
                else:
                    obj.logger.critical(
                        "how does this happen? Segments from %d touching %d "
                        "grid cells", segnum, len(matches))
            elif rem.geom_type.startswith("Multi"):
                for sub_rem_geom in rem.geoms:  # recurse
                    assign_remaining_reach(reach_df, segnum, sub_rem_geom)
            else:
                raise NotImplementedError(rem.geom_type)

        # Looping over each segment breaking down into reaches
        for segnum, line in obj.segments.geometry.iteritems():
            remaining_line = line
            if grid_sindex:
                bbox_match = sorted(grid_sindex.intersection(line.bounds))
                if not bbox_match:
                    continue
                sub = grid_cells.geometry.iloc[bbox_match]
            else:  # slow scan of all cells
                sub = grid_cells.geometry
            # Find all intersections between segment and grid cells
            reach_df = empty_reach_df.copy()
            for (row, col), grid_geom in sub.iteritems():
                reach_geom = grid_geom.intersection(line)
                if reach_geom.is_empty or reach_geom.geom_type == "Point":
                    continue
                remaining_line = remaining_line.difference(grid_geom)
                append_reach_df(reach_df, row, col, reach_geom)
            # Determine if any remaining portions of the line can be used
            if line is not remaining_line and remaining_line.length > 0:
                assign_remaining_reach(reach_df, segnum, remaining_line)
            # Reassign short reaches to two or more adjacent grid cells
            # starting with the shortest reach
            reach_lengths = reach_df["length"].loc[
                reach_df["length"] < reach_include[segnum]]
            for idx in list(reach_lengths.sort_values().index):
                assign_short_reach(reach_df, idx, segnum)
            # Potentially merge a few reaches for each row/col of this segnum
            drop_reach_ids = []
            gb = reach_df.groupby(["row", "col"])["geometry"].apply(list)
            for (row, col), geoms in gb.copy().iteritems():
                row_col = row, col
                if len(geoms) > 1:
                    geom = linemerge(geoms)
                    if geom.geom_type == "MultiLineString":
                        # workaround for odd floating point issue
                        geom = linemerge([wkt.loads(g.wkt) for g in geoms])
                    if geom.geom_type == "LineString":
                        sel = ((reach_df["row"] == row) &
                               (reach_df["col"] == col))
                        drop_reach_ids += list(sel.index[sel])
                        obj.logger.debug(
                            "merging %d reaches for segnum %s at %s",
                            sel.sum(), segnum, row_col)
                        append_reach_df(reach_df, row, col, geom)
                    elif any(a.distance(b) < 1e-6
                             for a, b in combinations(geoms, 2)):
                        obj.logger.warning(
                            "failed to merge segnum %s at %s: %s",
                            segnum, row_col, geom.wkt)
                    # else: this is probably a meandering MultiLineString
            if drop_reach_ids:
                reach_df.drop(drop_reach_ids, axis=0, inplace=True)
            # TODO: Some reaches match multiple cells if they share a border
            # Add all reaches for this segment
            for _, reach in reach_df.iterrows():
                row, col, reach_geom = reach.loc[["row", "col", "geometry"]]
                if line.has_z:
                    # intersection(line) does not preserve Z coords,
                    # but line.interpolate(d) works as expected
                    reach_geom = LineString(line.interpolate(
                        line.project(Point(c))) for c in reach_geom.coords)
                # Get a point from the middle of the reach_geom
                reach_mid_pt = reach_geom.interpolate(0.5, normalized=True)
                reach_record = {
                    "geometry": reach_geom,
                    "segnum": segnum,  # TODO could use this in package data boundname to regroup reaches
                    "segndist": line.project(reach_mid_pt, normalized=True),
                    "row": row,
                    "col": col,
                }
                obj.reaches.loc[len(obj.reaches.index)] = reach_record
                if domain_action == "modify" and domain[row, col] == 0:
                    num_domain_modified += 1
                    domain[row, col] = 1

        if domain_action == "modify":
            if num_domain_modified:
                obj.logger.debug(
                    "updating %d cells from %s array for top layer",
                    num_domain_modified, domain_label.upper())
                if domain_label == "ibound":
                    obj.model.bas6.ibound[0] = domain
                elif domain_label == "idomain":
                    obj.model.dis.idomain.set_data(domain, layer=0)
                obj.reaches = obj.reaches.merge(
                    grid_df[[domain_label]],
                    left_on=["row", "col"], right_index=True)
                obj.reaches.rename(
                    columns={domain_label: "prev_" + domain_label},
                    inplace=True)
            else:
                obj.reaches["prev_" + domain_label] = 1

        # Now convert from DataFrame to GeoDataFrame
        obj.reaches = geopandas.GeoDataFrame(
                obj.reaches, geometry="geometry", crs=crs)

        if not hasattr(obj.reaches.geometry, "geom_type"):
            # workaround needed for reaches.to_file()
            obj.reaches.geometry.geom_type = obj.reaches.geom_type

        # Mark segments that are not used
        obj.segments["in_model"] = True
        outside_model = \
            set(swn.segments.index).difference(obj.reaches["segnum"])
        obj.segments.loc[list(outside_model), "in_model"] = False

        # Add information to reaches from segments
        obj.reaches = obj.reaches.merge(
            obj.segments[["sequence"]], "left",
            left_on="segnum", right_index=True)
        obj.reaches.sort_values(["sequence", "segndist"], inplace=True)
        del obj.reaches["sequence"]  # segment sequence not used anymore
        # keep "segndist" for interpolation from segment data
        return obj

        obj.reaches.reset_index(drop=True, inplace=True)
        obj.reaches.index += 1  # TODO need to check base here
        # if returning for flopy might need to be zero-based, there maybe some
        # funnyness going on in flopy. I am happy with one-base though as then
        # we can just dump the package out as an external file.
        # Might just need a user beware!
        obj.reaches.index.name = "rno"

        # Add model grid info to each reach
        r = obj.reaches["row"].values
        c = obj.reaches["col"].values
        obj.reaches["m_top"] = dis.top.array[r, c]
        # Estimate slope from top and grid spacing
        col_size = np.median(dis.delr.array)
        row_size = np.median(dis.delc.array)
        px, py = np.gradient(dis.top.array, col_size, row_size)
        grid_slope = np.sqrt(px ** 2 + py ** 2)
        obj.reaches["m_top_slope"] = grid_slope[r, c]
        obj.reaches["m_botm0"] = dis.botm.array[0, r, c]

        # Reach length is based on geometry property
        obj.reaches["rlen"] = obj.reaches.geometry.length

        if swn.has_z:
            # If using LineStringZ, use reach Z-coordinate data
            zcoords = obj.reaches.geometry.apply(
                lambda g: [c[2] for c in g.coords[:]])
            obj.reaches["lsz_min"] = zcoords.apply(lambda z: min(z))
            obj.reaches["lsz_avg"] = zcoords.apply(lambda z: sum(z) / len(z))
            obj.reaches["lsz_max"] = zcoords.apply(lambda z: max(z))
            obj.reaches["lsz_first"] = zcoords.apply(lambda z: z[0])
            obj.reaches["lsz_last"] = zcoords.apply(lambda z: z[-1])
            # Calculate gradient
            obj.reaches["rgrd"] = (
                (obj.reaches["lsz_first"] - obj.reaches["lsz_last"])
                / obj.reaches["rlen"]
            )
        else:
            # Otherwise assume gradient same as model top
            obj.reaches["rgrd"] = obj.reaches["m_top_slope"]
        rgrd_le_zero = obj.reaches["rgrd"] <= 0
        if (rgrd_le_zero).any():
            obj.logger.error(
                "there are {} reaches with 'rgrd' <= 0"
                .format(rgrd_le_zero.sum()))

        # Evaluate connections
        # Assume only converging network
        to_segnums_d = swn.to_segnums.to_dict()
        reaches_segnum_s = set(obj.reaches["segnum"])

        def find_next_rno(segnum):
            if segnum in to_segnums_d:
                to_segnum = to_segnums_d[segnum]
                if to_segnum in reaches_segnum_s:
                    sel = obj.reaches["segnum"] == to_segnum
                    return obj.reaches[sel].index[0]
                else:  # recurse downstream
                    return find_next_rno(to_segnum)
            else:
                return 0

        def get_to_rno():
            if segnum == next_segnum:
                return next_rno
            else:
                return find_next_rno(segnum)

        obj.reaches["to_rno"] = -1
        segnum_iter = obj.reaches["segnum"].iteritems()
        rno, segnum = next(segnum_iter)
        for next_rno, next_segnum in segnum_iter:
            obj.reaches.at[rno, "to_rno"] = get_to_rno()
            rno, segnum = next_rno, next_segnum
        next_segnum = swn.END_SEGNUM
        obj.reaches.at[rno, "to_rno"] = get_to_rno()
        assert obj.reaches.to_rno.min() >= 0

        # Populate from_rnos set
        obj.reaches["from_rnos"] = [set() for _ in range(len(obj.reaches))]
        to_rnos = obj.reaches.loc[obj.reaches["to_rno"] != 0, "to_rno"]
        for k, v in to_rnos.items():
            obj.reaches.at[v, "from_rnos"].add(k)

        # Diversions not handled (yet)
        obj.reaches["to_div"] = 0
        obj.reaches["ustrf"] = 1.

        if not hasattr(obj.reaches.geometry, "geom_type"):
            # workaround needed for reaches.to_file()
            obj.reaches.geometry.geom_type = obj.reaches.geom_type

        return obj

    def set_reach_data_from_series(
            self, name, value, value_out=None, log10=False):
        """Set reach data based on segment series (or scalar).

        Values are interpolated along lengths of each segment based on a
        'segndist' attribute for segment normalized distance, between 0 and 1.

        Parameters
        ----------
        name : str
            Name for reach dataset.
        value : float or pandas.Series
            Value to assign to the top of each segment. If a float, this value
            is a constant. If a pandas Series, then this is applied for
            each segment.
        value_out : None, float or pandas.Series, optional
            If None (default), the value used for the bottom of outlet segments
            is assumed to be the same as the top. Otherwise, a Series
            for each outlet can be specified.
        log10 : bool, optional
            If True, log-10 transformation applied to interpolation, otherwise
            a linear interpolation is used from start to end of each segment.
        """
        if not isinstance(name, str):
            raise ValueError("'name' must be a str type")
        segdat = self._swn._pair_segment_values(value, value_out, name)
        for segnum, (value1, value2) in segdat.iterrows():
            sel = self.reaches['segnum'] == segnum
            if value1 == value2:
                value = value1
            else:  # interpolate to mid points of each reach from segment data
                segndist = self.reaches.loc[sel, 'segndist']
                if log10:
                    lvalue1 = np.log10(value1)
                    lvalue2 = np.log10(value2)
                    value = 10 ** ((lvalue2 - lvalue1) * segndist + lvalue1)
                else:
                    value = (value2 - value1) * segndist + value1
            self.reaches.loc[sel, name] = value

    def set_reach_data_from_array(self, name, array):
        """Set reach data from an array that matches the model (nrow, ncol).

        Parameters
        ----------
        name : str
            Name for reach dataset.
        array : array_like
            2D array with dimensions (nrow, ncol).
        """
        if not isinstance(name, str):
            raise ValueError("'name' must be a str type")
        elif not hasattr(array, "ndim"):
            raise ValueError("'array' must be array-like")
        elif array.ndim != 2:
            raise ValueError("'array' must have two dimensions")
        dis = self.model.dis
        expected_shape = dis.nrow.data, dis.ncol.data
        if expected_shape != array.shape:
            raise ValueError("'array' must have shape (nrow, ncol)")
        self.reaches.loc[:, name] = array[self.reaches.row, self.reaches.col]
