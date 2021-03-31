# -*- coding: utf-8 -*-
"""Interface for flopy's implementation for MODFLOW 6."""

__all__ = ["SwnMf6"]

import geopandas
import numpy as np
import pandas as pd
import pickle
from itertools import combinations, zip_longest
from shapely import wkt
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import linemerge
from textwrap import dedent
import os

try:
    import matplotlib
except ImportError:
    matplotlib = False

from swn.core import SurfaceWaterNetwork
from swn.spatial import get_sindex, compare_crs
from swn.util import abbr_str
from swn.modflow._base import _SwnModflow
from swn.modflow._misc import set_outreaches


class SwnMf6(_SwnModflow):
    """Surface water network adaptor for MODFLOW 6.

    Attributes
    ----------
    model : flopy.mf6.ModflowGwf
        Instance of flopy.mf6.ModflowGwf
    segments : geopandas.GeoDataFrame
        Copied from swn.segments, but with additional columns added
    segment_data : pandas.DataFrame
        Simialr to structure in model.sfr.segment_data, but for one stress
        period. Transient data (where applicable) will show summary statistics.
        The index is 'nseg', ordered and starting from 1. An additional column
        'segnum' is used to identify segments, and if defined,
        abstraction/diversion identifiers, where iupseg != 0.
    reaches : geopandas.GeoDataFrame
        Similar to structure in model.sfr.reach_data with index 'reachID',
        ordered and starting from 1. Contains geometry and other columns
        not used by flopy. Use get_reach_data() for use with flopy.
    diversions :  geopandas.GeoDataFrame, pd.DataFrame or None
        Copied from swn.diversions, if set/defined.
    logger : logging.Logger
        Logger to show messages.

    """

    def __init__(self, logger=None):
        """Initialise SwnMf6.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger to show messages.
        """
        super().__init__(logger)
        # set empty properties for now
        self.segment_data = None
        self.reaches = None
        # all other properties added afterwards

    @classmethod
    def other_from_swn_flopy(
            cls, swn, model, idomain_action='freeze',
            reach_include_fraction=0.2, min_slope=1./1000,
            hyd_cond1=1., hyd_cond_out=None, thickness1=1., thickness_out=None,
            width1=10., width_out=None, roughch=0.024,
            abstraction={}, inflow={}, flow={}, runoff={}, etsw={}, pptsw={}
    ):
        """Create a MODFLOW 6 SFR structure from a surface water network.

        Parameters
        ----------
        swn : swn.SurfaceWaterNetwork
            Instance of a SurfaceWaterNetwork.
        model : flopy.mf6.ModflowGwf
            Instance of a flopy MODFLOW 6 groundwater flow model.
        idomain_action : str, optional
            Action to handle IDOMAIN:
                - ``freeze`` : Freeze IDOMAIN, but clip streams to fit bounds.
                - ``modify`` : Modify IDOMAIN to fit streams, where possible.
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
        import flopy
        if not isinstance(swn, SurfaceWaterNetwork):
            raise ValueError('swn must be a SurfaceWaterNetwork object')
        elif idomain_action not in ('freeze', 'modify'):
            raise ValueError('idomain_action must be one of freeze or modify')
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
        # More careful check of overlap of lines with grid polygons
        obj.logger.debug('building model grid cell geometries')
        dis = obj.model.dis
        cols, rows = np.meshgrid(np.arange(dis.ncol.data),
                                 np.arange(dis.nrow.data))
        idomain = dis.idomain[0].array.copy()
        idomain_modified = 0
        grid_df = pd.DataFrame({'row': rows.flatten(), 'col': cols.flatten()})
        grid_df.set_index(['row', 'col'], inplace=True)
        grid_df['idomain'] = idomain.flatten()
        if idomain_action == 'freeze' and (idomain == 0).any():
            # Remove any inactive grid cells from analysis
            grid_df = grid_df.loc[grid_df['idomain'] != 0]
        # Determine grid cell size
        col_size = np.median(dis.delr.array)
        if dis.delr.array.min() != dis.delr.array.max():
            obj.logger.warning(
                'assuming constant column spacing %s', col_size)
        row_size = np.median(dis.delc.array)
        if dis.delc.array.min() != dis.delc.array.max():
            obj.logger.warning(
                'assuming constant row spacing %s', row_size)
        cell_size = (row_size + col_size) / 2.0
        # Note: modelgrid.get_cell_vertices(row, col) is slow!
        xv = modelgrid.xvertices
        yv = modelgrid.yvertices
        r, c = [np.array(s[1])
                for s in grid_df.reset_index()[['row', 'col']].iteritems()]
        cell_verts = zip(
            zip(xv[r, c], yv[r, c]),
            zip(xv[r, c + 1], yv[r, c + 1]),
            zip(xv[r + 1, c + 1], yv[r + 1, c + 1]),
            zip(xv[r + 1, c], yv[r + 1, c])
        )
        obj.grid_cells = grid_cells = geopandas.GeoDataFrame(
            grid_df, geometry=[Polygon(r) for r in cell_verts], crs=crs)
        obj.logger.debug('evaluating reach data on model grid')
        grid_sindex = get_sindex(grid_cells)
        reach_include = swn._segment_series(reach_include_fraction) * cell_size
        # Make an empty DataFrame for reaches
        obj.reaches = pd.DataFrame(columns=['geometry'])
        obj.reaches.insert(1, column='row', value=pd.Series(dtype=int))
        obj.reaches.insert(2, column='col', value=pd.Series(dtype=int))
        empty_reach_df = obj.reaches.copy()  # take this before more added
        obj.reaches.insert(
            1, column='segnum',
            value=pd.Series(dtype=obj.segments.index.dtype))
        obj.reaches.insert(2, column='dist', value=pd.Series(dtype=float))
        empty_reach_df.insert(3, column='length', value=pd.Series(dtype=float))
        empty_reach_df.insert(4, column='moved', value=pd.Series(dtype=bool))

        # recusive helper function
        def append_reach_df(df, row, col, reach_geom, moved=False):
            if reach_geom.geom_type == 'LineString':
                df.loc[len(df.index)] = {
                    'geometry': reach_geom,
                    'row': row,
                    'col': col,
                    'length': reach_geom.length,
                    'moved': moved,
                }
            elif reach_geom.geom_type.startswith('Multi'):
                for sub_reach_geom in reach_geom.geoms:  # recurse
                    append_reach_df(df, row, col, sub_reach_geom, moved)
            else:
                raise NotImplementedError(reach_geom.geom_type)

        # helper function that returns early, if necessary
        def assign_short_reach(reach_df, idx, segnum):
            reach = reach_df.loc[idx]
            reach_geom = reach['geometry']
            threshold = reach_include[segnum]
            if reach_geom.length > threshold:
                return
            cell_lengths = reach_df.groupby(['row', 'col'])['length'].sum()
            this_row_col = reach['row'], reach['col']
            this_cell_length = cell_lengths[this_row_col]
            if this_cell_length > threshold:
                return
            grid_geom = grid_cells.at[(reach['row'], reach['col']), 'geometry']
            # determine if it is crossing the grid once or twice
            grid_points = reach_geom.intersection(grid_geom.exterior)
            split_short = (
                grid_points.geom_type == 'Point' or
                (grid_points.geom_type == 'MultiPoint' and
                 len(grid_points) == 2))
            if not split_short:
                return
            matches = []
            # sequence scan on reach_df
            for oidx, orch in reach_df.iterrows():
                if oidx == idx or orch['moved']:
                    continue
                other_row_col = orch['row'], orch['col']
                other_cell_length = cell_lengths[other_row_col]
                if (orch['geometry'].distance(reach_geom) < 1e-6 and
                        this_cell_length < other_cell_length):
                    matches.append((oidx, orch['geometry']))
            if len(matches) == 0:
                # don't merge, e.g. reach does not connect to adjacent cell
                pass
            elif len(matches) == 1:
                # short segment is in one other cell only
                # update new row and col values, keep geometry as it is
                row_col1 = tuple(reach_df.loc[matches[0][0], ['row', 'col']])
                reach_df.loc[idx, ['row', 'col', 'moved']] = row_col1 + (True,)
                # self.logger.debug(
                #    'moved short segment of %s from %s to %s',
                #    segnum, this_row_col, row_col1)
            elif len(matches) == 2:
                assert grid_points.geom_type == 'MultiPoint', grid_points.wkt
                if len(grid_points) != 2:
                    obj.logger.critical(
                        'expected 2 points, found %s', len(grid_points))
                # Build a tiny DataFrame of coordinates for this reach
                reach_c = pd.DataFrame({
                    'pt': [Point(c) for c in reach_geom.coords[:]]
                })
                if len(reach_c) == 2:
                    # If this is a simple line with two coords, split it
                    reach_c.index = [0, 2]
                    reach_c.loc[1] = {
                        'pt': reach_geom.interpolate(0.5, normalized=True)}
                    reach_c.sort_index(inplace=True)
                    reach_geom = LineString(list(reach_c['pt']))  # rebuild
                # first match assumed to be touching the start of the line
                if reach_c.at[0, 'pt'].distance(matches[1][1]) < 1e-6:
                    matches.reverse()
                reach_c['d1'] = reach_c['pt'].apply(
                                lambda p: p.distance(matches[0][1]))
                reach_c['d2'] = reach_c['pt'].apply(
                                lambda p: p.distance(matches[1][1]))
                reach_c['dm'] = reach_c[['d1', 'd2']].min(1)
                # try a simple split where distances switch
                ds = reach_c['d1'] < reach_c['d2']
                cidx = ds[ds].index[-1]
                # ensure it's not the index of either end
                if cidx == 0:
                    cidx = 1
                elif cidx == len(reach_c) - 1:
                    cidx = len(reach_c) - 2
                row1, col1 = list(reach_df.loc[matches[0][0], ['row', 'col']])
                reach_geom1 = LineString(reach_geom.coords[:(cidx + 1)])
                row2, col2 = list(reach_df.loc[matches[1][0], ['row', 'col']])
                reach_geom2 = LineString(reach_geom.coords[cidx:])
                # update the first, append the second
                reach_df.loc[idx, ['row', 'col', 'length', 'moved']] = \
                    (row1, col1, reach_geom1.length, True)
                reach_df.at[idx, 'geometry'] = reach_geom1
                append_reach_df(reach_df, row2, col2, reach_geom2, moved=True)
                # self.logger.debug(
                #   'split and moved short segment of %s from %s to %s and %s',
                #   segnum, this_row_col, (row1, col1), (row2, col2))
            else:
                obj.logger.critical(
                    'unhandled assign_short_reach case with %d matches: %s\n'
                    '%s\n%s', len(matches), matches, reach, grid_points.wkt)

        def assign_remaining_reach(reach_df, segnum, rem):
            if rem.geom_type == 'LineString':
                threshold = cell_size * 2.0
                if rem.length > threshold:
                    obj.logger.debug(
                        'remaining line segment from %s too long to merge '
                        '(%.1f > %.1f)', segnum, rem.length, threshold)
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
                    'pt': [Point(c) for c in rem.coords[:]]
                })
                if len(matches) == 1:  # merge it with adjacent cell
                    row, col, grid_geom = matches[0]
                    mdist = rem_c['pt'].apply(
                                    lambda p: grid_geom.distance(p)).max()
                    if mdist > threshold:
                        obj.logger.debug(
                            'remaining line segment from %s too far away to '
                            'merge (%.1f > %.1f)', segnum, mdist, threshold)
                        return
                    append_reach_df(reach_df, row, col, rem, moved=True)
                elif len(matches) == 2:  # complex: need to split it
                    if len(rem_c) == 2:
                        # If this is a simple line with two coords, split it
                        rem_c.index = [0, 2]
                        rem_c.loc[1] = {
                            'pt': rem.interpolate(0.5, normalized=True)}
                        rem_c.sort_index(inplace=True)
                        rem = LineString(list(rem_c['pt']))  # rebuild
                    # first match assumed to be touching the start of the line
                    if rem_c.at[0, 'pt'].touches(matches[1][2]):
                        matches.reverse()
                    rem_c['d1'] = rem_c['pt'].apply(
                                    lambda p: p.distance(matches[0][2]))
                    rem_c['d2'] = rem_c['pt'].apply(
                                    lambda p: p.distance(matches[1][2]))
                    rem_c['dm'] = rem_c[['d1', 'd2']].min(1)
                    mdist = rem_c['dm'].max()
                    if mdist > threshold:
                        obj.logger.debug(
                            'remaining line segment from %s too far away to '
                            'merge (%.1f > %.1f)', segnum, mdist, threshold)
                        return
                    # try a simple split where distances switch
                    ds = rem_c['d1'] < rem_c['d2']
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
                        'how does this happen? Segments from %d touching %d '
                        'grid cells', segnum, len(matches))
            elif rem.geom_type.startswith('Multi'):
                for sub_rem_geom in rem.geoms:  # recurse
                    assign_remaining_reach(reach_df, segnum, sub_rem_geom)
            else:
                raise NotImplementedError(rem.geom_type)

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
                if reach_geom.is_empty or reach_geom.geom_type == 'Point':
                    continue
                remaining_line = remaining_line.difference(grid_geom)
                append_reach_df(reach_df, row, col, reach_geom)
            # Determine if any remaining portions of the line can be used
            if line is not remaining_line and remaining_line.length > 0:
                assign_remaining_reach(reach_df, segnum, remaining_line)
            # Reassign short reaches to two or more adjacent grid cells
            # starting with the shortest reach
            reach_lengths = reach_df['length'].loc[
                reach_df['length'] < reach_include[segnum]]
            for idx in list(reach_lengths.sort_values().index):
                assign_short_reach(reach_df, idx, segnum)
            # Potentially merge a few reaches for each row/col of this segnum
            drop_reach_ids = []
            gb = reach_df.groupby(['row', 'col'])['geometry'].apply(list)
            for (row, col), geoms in gb.copy().iteritems():
                row_col = row, col
                if len(geoms) > 1:
                    geom = linemerge(geoms)
                    if geom.geom_type == 'MultiLineString':
                        # workaround for odd floating point issue
                        geom = linemerge([wkt.loads(g.wkt) for g in geoms])
                    if geom.geom_type == 'LineString':
                        sel = ((reach_df['row'] == row) &
                               (reach_df['col'] == col))
                        drop_reach_ids += list(sel.index[sel])
                        obj.logger.debug(
                            'merging %d reaches for segnum %s at %s',
                            sel.sum(), segnum, row_col)
                        append_reach_df(reach_df, row, col, geom)
                    elif any(a.distance(b) < 1e-6
                             for a, b in combinations(geoms, 2)):
                        obj.logger.warning(
                            'failed to merge segnum %s at %s: %s',
                            segnum, row_col, geom.wkt)
                    # else: this is probably a meandering MultiLineString
            if drop_reach_ids:
                reach_df.drop(drop_reach_ids, axis=0, inplace=True)
            # TODO: Some reaches match multiple cells if they share a border
            # Add all reaches for this segment
            for _, reach in reach_df.iterrows():
                row, col, reach_geom = reach.loc[['row', 'col', 'geometry']]
                if line.has_z:
                    # intersection(line) does not preserve Z coords,
                    # but line.interpolate(d) works as expected
                    reach_geom = LineString(line.interpolate(
                        line.project(Point(c))) for c in reach_geom.coords)
                # Get a point from the middle of the reach_geom
                reach_mid_pt = reach_geom.interpolate(0.5, normalized=True)
                reach_record = {
                    'geometry': reach_geom,
                    'segnum': segnum,
                    'dist': line.project(reach_mid_pt, normalized=True),
                    'row': row,
                    'col': col,
                }
                obj.reaches.loc[len(obj.reaches.index)] = reach_record
                if idomain_action == 'modify' and idomain[row, col] == 0:
                    idomain_modified += 1
                    idomain[row, col] = 1

        if idomain_action == 'modify':
            if idomain_modified:
                obj.logger.debug(
                    'updating %d cells from IDOMAIN array for top layer',
                    idomain_modified)
                obj.model.dis.idomain.set_data(idomain, layer=0)
                obj.reaches = obj.reaches.merge(
                    grid_df[['idomain']],
                    left_on=['row', 'col'], right_index=True)
                obj.reaches.rename(
                        columns={'idomain': 'prev_idomain'}, inplace=True)
            else:
                obj.reaches['prev_idomain'] = 1

        # Now convert from DataFrame to GeoDataFrame
        obj.reaches = geopandas.GeoDataFrame(
                obj.reaches, geometry='geometry', crs=crs)

        # Assign segment data
        # FOR MF6 dont really need this segment info garbage
        # -- but we are using it to help fill in packagedata (reach-based)
        obj.segments['min_slope'] = swn._segment_series(min_slope)
        if (obj.segments['min_slope'] < 0.0).any():
            raise ValueError('min_slope must be greater than zero')
        # Column names common to segments and segment_data
        segment_cols = [
            'roughch',
            'hcond1', 'thickm1', 'elevup', 'width1',
            'hcond2', 'thickm2', 'elevdn', 'width2']
        # Tidy any previous attempts
        for col in segment_cols:
            if col in obj.segments.columns:
                del obj.segments[col]
        # Combine pairs of series for each segment
        more_segment_columns = pd.concat([
            swn._pair_segment_values(hyd_cond1, hyd_cond_out, 'hcond'),
            swn._pair_segment_values(thickness1, thickness_out, 'thickm'),
            swn._pair_segment_values(width1, width_out, name='width')
        ], axis=1, copy=False)
        for name, series in more_segment_columns.iteritems():
            obj.segments[name] = series
        obj.segments['roughch'] = swn._segment_series(roughch)
        # Mark segments that are not used
        obj.segments['in_model'] = True
        outside_model = \
            set(swn.segments.index).difference(obj.reaches['segnum'])
        obj.segments.loc[list(outside_model), 'in_model'] = False

        # Add information to reaches from segments
        obj.reaches = obj.reaches.merge(
            obj.segments[['sequence', 'min_slope']], 'left',
            left_on='segnum', right_index=True)
        obj.reaches.sort_values(['sequence', 'dist'], inplace=True)
        # Interpolate segment properties to each reach
        obj.reaches['rbth'] = 0.0
        obj.reaches['rhk'] = 0.0
        for segnum, seg in obj.segments.iterrows():
            sel = obj.reaches['segnum'] == segnum
            if seg['thickm1'] == seg['thickm2']:
                val = seg['thickm1']
            else:  # linear interpolate to mid points
                tk1 = seg['thickm1']
                tk2 = seg['thickm2']
                dtk = tk2 - tk1
                val = dtk * obj.reaches.loc[sel, 'dist'] + tk1
            obj.reaches.loc[sel, 'rbth'] = val
            if seg['hcond1'] == seg['hcond2']:
                val = seg['hcond1']
            else:  # linear interpolate to mid points in log-10 space
                lhc1 = np.log10(seg['hcond1'])
                lhc2 = np.log10(seg['hcond2'])
                dlhc = lhc2 - lhc1
                val = 10 ** (dlhc * obj.reaches.loc[sel, 'dist'] + lhc1)
            obj.reaches.loc[sel, 'rhk'] = val
            if seg['width1'] == seg['width2']:
                val = seg['width1']
            else:  # linear interpolate to mid points in log-10 space
                lhc1 = np.log10(seg['width1'])
                lhc2 = np.log10(seg['width2'])
                dlhc = lhc2 - lhc1
                val = 10 ** (dlhc * obj.reaches.loc[sel, 'dist'] + lhc1)
            obj.reaches.loc[sel, 'rwid'] = val
            obj.reaches.loc[sel, 'man'] = seg['roughch']
        del obj.reaches['sequence']
        del obj.reaches['dist']
        # Use MODFLOW SFR dataset 2 terms ISEG and IREACH, counting from 1
        obj.reaches['iseg'] = 0
        obj.reaches['ireach'] = 0
        iseg = ireach = 0
        prev_segnum = None
        for idx, segnum in obj.reaches['segnum'].iteritems():
            if segnum != prev_segnum:
                iseg += 1
                ireach = 0
            ireach += 1
            obj.reaches.at[idx, 'iseg'] = iseg
            obj.reaches.at[idx, 'ireach'] = ireach
            prev_segnum = segnum
        obj.reaches.reset_index(inplace=True, drop=True)
        obj.reaches.index += 1  # flopy series starts at one
        obj.reaches.index.name = 'reachID'
        obj.reaches['rlen'] = obj.reaches.geometry.length
        obj.reaches['rtp'] = 0.0
        obj.reaches['rgrd'] = 0.0
        if swn.has_z:
            for reachID, item in obj.reaches.iterrows():
                geom = item.geometry
                # Get Z from each end
                z0 = geom.coords[0][2]
                z1 = geom.coords[-1][2]
                dz = z0 - z1
                dx = geom.length
                slope = dz / dx
                obj.reaches.at[reachID, 'rgrd'] = slope
                # Get strtop from LineString mid-point Z
                zm = geom.interpolate(0.5, normalized=True).z
                obj.reaches.at[reachID, 'rtp'] = zm
        else:
            r = obj.reaches['row'].values
            c = obj.reaches['col'].values
            # Estimate slope from top and grid spacing
            px, py = np.gradient(dis.top.array, col_size, row_size)
            grid_slope = np.sqrt(px ** 2 + py ** 2)
            obj.reaches['rgrd'] = grid_slope[r, c]
            # Get stream values from top of model
            obj.reaches['rtp'] = dis.top.array[r, c]
        # Enforce min_slope
        sel = obj.reaches['rgrd'] < obj.reaches['min_slope']
        if sel.any():
            obj.logger.warning(
                'enforcing min_slope for %d reaches (%.2f%%)',
                sel.sum(), 100.0 * sel.sum() / len(sel))
            obj.reaches.loc[sel, 'slope'] = obj.reaches.loc[sel, 'min_slope']
        if not hasattr(obj.reaches.geometry, 'geom_type'):
            # workaround needed for reaches.to_file()
            obj.reaches.geometry.geom_type = obj.reaches.geom_type
        # Finally, add/rename a few columns to align with reach_data
        obj.reaches.insert(2, column='k', value=0)
        obj.reaches.insert(3, column='outreach', value=pd.Series(dtype=int))
        obj.reaches.rename(columns={'row': 'i', 'col': 'j'}, inplace=True)
        obj.reaches['rno'] = obj.reaches.index
        # Build mf6 packagedata from reach info
        # get default columns for mf6 package data
        pkd_defcols = flopy.mf6.ModflowGwfsfr.packagedata.empty(model).dtype.descr
        defcols_names = []
        defcols_dtypes = []
        for n, t in pkd_defcols:
            defcols_names.append(n)
            defcols_dtypes.append(t)
        # use kij unstead of cellid
        if 'cellid' in defcols_names:
            i = defcols_names.index('cellid')
            defcols_names[i:i + 1] = 'k', 'i', 'j'
            defcols_dtypes[i:i + 1] = '<i8', '<i8', '<i8'
        # Add missing data cols
        for n, t in zip(defcols_names, defcols_dtypes):
            if n not in obj.reaches.columns:
                obj.reaches[n] = pd.Series(index=obj.reaches.index, dtype=t)
            # obj.reaches.loc[:, defcols_names]

        # Build some old style Data Set 6 segment data
        # -- again dont need this for mf6 but we can us it to fill in the
        # reach-based data sets.  # TODO would benefit from a tidy
        obj.segment_data = obj.reaches[['iseg', 'segnum']]\
            .drop_duplicates().rename(columns={'iseg': 'nseg'})
        # index changes from 'reachID', to 'segnum', to finally 'nseg'
        segnum2nseg_d = obj.segment_data.set_index('segnum')['nseg'].to_dict()
        # obj.segment_data[' icalc'] = 1  # assumption for all streams
        obj.segment_data['outseg'] = obj.segment_data['segnum'].map(
            lambda x: segnum2nseg_d.get(obj.segments.loc[x, 'to_segnum'], 0))

        # at this point we could try and build packagedata
        obj.reaches = set_outreaches(obj.reaches, obj.segment_data)
        outreach_dict = obj.reaches.outreach.to_dict()
        # get number of connections # TODO is this different if there are diversions/multiple downstream?
        obj.reaches['ncon'] = obj.reaches.rno.apply(
            lambda x: (outreach_dict[x] != 0) +
                      (obj.reaches.outreach == x).sum()
        )
        # add more cols... for package data
        obj.reaches['ustrf'] = 1.  # upstream flow fraction  (check manual for this...)
        # TODO: diversions
        obj.reaches['ndv'] = 0  # number of diversions - needs to be int

        # Package DATA
        obj.packagedata = obj.reaches.loc[:, defcols_names].sort_index()
        obj.packagedata.loc[:, ['k', 'i', 'j']] = obj.packagedata.loc[:, ['k', 'i', 'j']] + 1
        # save to external
        pkd_fname = 'hpm_sfr_pkgdata.txt'
        obj.packagedata.to_csv(os.path.join(model.model_ws, pkd_fname),
                               header=False, index=False, sep=' ')

        # Build mf6 sfr connections data from outreaches
        obj.connections = obj.packagedata.rno.apply(
            lambda x:
            obj.reaches.loc[obj.reaches.outreach == x].index.to_list() +
            [-outreach_dict[x]]
            if outreach_dict[x] != 0
            else obj.reaches.loc[obj.reaches.outreach == x].index.to_list()
        )
        # save to external
        conn_fname = 'hpm_sfr_connections.txt'
        with open(os.path.join(model.model_ws, conn_fname), 'w') as fp:
            for reach, conn in obj.connections.items():
                fp.write(f"{reach} " + ' '.join([str(c) for c in conn]) + '\n')


        # obj.segment_data['iupseg'] = 0  # handle diversions next
        # obj.segment_data['iprior'] = 0
        # obj.segment_data['flow'] = 0.0
        # obj.segment_data['runoff'] = 0.0
        # obj.segment_data['etsw'] = 0.0
        # obj.segment_data['pptsw'] = 0.0
        # upper elevation from the first and last reachID items from reaches
        # obj.segment_data['elevup'] = \
        #     obj.reaches.loc[obj.segment_data.index, 'rtop']
        # obj.segment_data['elevdn'] = obj.reaches.loc[
        #     obj.reaches.groupby(['iseg']).ireach.idxmax().values,
        #     'rtop'].values
        # obj.segment_data.set_index('segnum', drop=False, inplace=True)
        # copy several columns over (except 'elevup' and 'elevdn', for now)
        # segment_cols.remove('elevup')
        # segment_cols.remove('elevdn')
        # obj.segment_data[segment_cols] = obj.segments[segment_cols]
        # now use nseg as primary index, not reachID or segnum
        # obj.segment_data.set_index('nseg', inplace=True)
        # obj.segment_data.sort_index(inplace=True)
        # Add diversions (i.e. SW takes) # TODO bring diversions back for mf6
        # if swn.diversions is not None:
        #     obj.diversions = swn.diversions.copy()
        #     # Mark diversions that are not used / outside model
        #     obj.diversions['in_model'] = True
        #     outside_model = []
        #     # Add columns for ICALC=0
        #     obj.segment_data['depth1'] = 0.0
        #     obj.segment_data['depth2'] = 0.0
        #     # workaround for coercion issue
        #     obj.segment_data['foo'] = ''
        #     is_spatial = (
        #         isinstance(obj.diversions, geopandas.GeoDataFrame) and
        #         'geometry' in obj.diversions.columns and
        #         (~obj.diversions.is_empty).all())
        #     if swn.has_z:
        #         empty_geom = wkt.loads('linestring z empty')
        #     else:
        #         empty_geom = wkt.loads('linestring empty')
        #     for divid, divn in obj.diversions.iterrows():
        #         if divn.from_segnum not in segnum2nseg_d:
        #             # segnum does not exist -- segment is outside model
        #             outside_model.append(divid)
        #             continue
        #         iupseg = segnum2nseg_d[divn.from_segnum]
        #         assert iupseg != 0, iupseg
        #         nseg = len(obj.segment_data) + 1
        #         rchlen = 1.0  # length required
        #         thickm = 1.0  # thickness required
        #         hcond = 0.0  # don't allow GW exchange
        #         seg_d = dict(obj.segment_data.loc[iupseg])
        #         seg_d.update({  # index is nseg
        #             'segnum': divid,
        #             'icalc': 0,  # stream depth is specified
        #             'outseg': 0,
        #             'iupseg': iupseg,
        #             'iprior': 0,  # normal behaviour for SW takes
        #             'flow': 0.0,  # abstraction assigned later
        #             'runoff': 0.0,
        #             'etsw': 0.0,
        #             'pptsw': 0.0,
        #             'roughch': 0.0,  # not used
        #             'hcond1': hcond, 'hcond2': hcond,
        #             'thickm1': thickm, 'thickm2': thickm,
        #             'width1': 0.0, 'width2': 0.0,  # not used
        #         })
        #         # Use the last reach as a template to modify for new reach
        #         reach_d = dict(obj.reaches.loc[
        #             obj.reaches.iseg == iupseg].iloc[-1])
        #         reach_d.update({
        #             'segnum': divid,
        #             'iseg': nseg,
        #             'ireach': 1,
        #             'rchlen': rchlen,
        #             'min_slope': 0.0,
        #             'slope': 0.0,
        #             'strthick': thickm,
        #             'strhc1': hcond,
        #         })
        #         # Assign one reach at grid cell
        #         if is_spatial:
        #             # Find grid cell nearest to diversion
        #             if grid_sindex:
        #                 bbox_match = sorted(
        #                     grid_sindex.nearest(divn.geometry.bounds))
        #                 # more than one nearest can exist! just take one...
        #                 num_found = len(bbox_match)
        #                 grid_cell = grid_cells.iloc[bbox_match[0]]
        #             else:  # slow scan of all cells
        #                 sel = grid_cells.intersects(divn.geometry)
        #                 num_found = sel.sum()
        #                 grid_cell = grid_cells.loc[sel].iloc[0]
        #             if num_found > 1:
        #                 obj.logger.warning(
        #                     '%d grid cells are nearest to diversion %r, '
        #                     'but only taking the first %s',
        #                     num_found, divid, grid_cell)
        #             row, col = grid_cell.name
        #             strtop = dis.top[row, col]
        #             reach_d.update({
        #                 'geometry': empty_geom,  # divn.geometry,
        #                 'row': row,
        #                 'col': col,
        #                 'strtop': strtop,
        #             })
        #         else:
        #             strtop = dis.top[reach_d['row'], reach_d['col']]
        #             reach_d['strtop'] = strtop
        #             seg_d.update({
        #                 'geometry': empty_geom,
        #                 'elevup': strtop,
        #                 'elevdn': strtop,
        #             })
        #         depth = strtop + thickm
        #         seg_d.update({'depth1': depth, 'depth2': depth})
        #         obj.reaches.loc[len(obj.reaches) + 1] = reach_d
        #         obj.segment_data.loc[nseg] = seg_d
        #     if outside_model:
        #         obj.diversions.loc[list(outside_model), 'in_model'] = False
        #         obj.logger.debug(
        #             'added %d diversions, ignoring %d that did not connect to '
        #             'existing segments',
        #             obj.diversions['in_model'].sum(), len(outside_model))
        #     else:
        #         obj.logger.debug(
        #             'added all %d diversions', len(obj.diversions))
        #     # end of coercion workaround
        #     obj.segment_data.drop('foo', axis=1, inplace=True)
        # else:
        #     obj.diversions = None
        # Build stressperiod data
        sfr_spd = {}  # TODO this only works if flows etc are
                      #  expressed as segment (segnum) based.
                      #  Need to think more about the format that flows etc
                      #  are going to be passed to this constructor.
        for t in range(model.nper):
            # fname = f'hpm_sfr_perdata_{t}.txt'
            if t in flow.keys():
                persfr = pd.DataFrame()
                tdf = flow[t]
                withflow = tdf[tdf['flow'] != 0]
                persfr['rno'] = [obj.reaches.loc[
                                     (obj.reaches.iseg == seg.segnum) &
                                     (obj.reaches.ireach == 1)].index.values[0]
                                 for seg in withflow]
                persfr['settingtype'] = 'inflow'
                persfr['settingvalue'] = withflow.flow
            # if t in runoff.keys():  # TODO how is runoff to be divided up?
            #     tdf = runoff[t]
            #     withro = tdf[tdf['flow'] != 0]
            #     persfr['rno'] = [obj.reaches.loc[
            #                          (obj.reaches.iseg == seg.segnum) &
            #                          (obj.reaches.ireach == 1)].index.values[0]
            #                      for seg in withro]
            #     persfr['settingtype'] = 'runoff'
            #     persfr['settingvalue'] = withro.flow
                sfr_spd[t] = persfr
        obj.spd = sfr_spd

            # persfr.to_csv(os.path.join(gwf.model_ws, fname),
            #               index=False, header=False, sep=' ')
            # sfr_spd[t] = {'filename': fname, 'data': None}
        # Create flopy Sfr2 package
        # segment_data = obj.set_segment_data(
        #     abstraction=abstraction, inflow=inflow,
        #     flow=flow, runoff=runoff, etsw=etsw, pptsw=pptsw, return_dict=True)
        # reach_data = obj.get_reach_data()
        # flopy.modflow.mfsfr2.ModflowSfr2(
        #     model=obj.model, reach_data=reach_data, segment_data=segment_data)
        flopy.mf6.ModflowGwfsfr(
            model,
            save_flows=True,
            maximum_iterations=100,
            maximum_picard_iterations=10,
            nreaches=len(obj.packagedata),
            packagedata={'filename': pkd_fname, 'data': None},
            connectiondata={'filename': conn_fname, 'data': None},
            perioddata=obj.spd,
        )

        return obj

    def __repr__(self):
        """Return string representation of SwnModflow object."""
        is_diversion = self.segment_data['iupseg'] != 0
        segnum_l = list(self.segment_data.loc[~is_diversion, 'segnum'])
        segments_line = str(len(segnum_l)) + ' from segments'
        if set(segnum_l) != set(self.segments.index):
            segments_line += ' ({:.0%} used)'.format(
                len(segnum_l) / float(len(self.segments)))
        segments_line += ': ' + abbr_str(segnum_l, 4)
        if is_diversion.any() and self.diversions is not None:
            divid_l = list(self.segment_data.loc[is_diversion, 'segnum'])
            diversions_line = str(len(divid_l)) + ' from diversions'
            if set(divid_l) != set(self.diversions.index):
                diversions_line += ' ({:.0%} used)'.format(
                    len(divid_l) / float(len(self.diversions)))
            diversions_line += abbr_str(divid_l, 4)
        else:
            diversions_line = 'no diversions'
        nper = self.model.dis.nper
        return dedent('''\
            <{}: flopy {} {!r}
              {} in reaches ({}): {}
              {} in segment_data ({}): {}
                {}
                {}
              {} stress period{} with perlen: {} />'''.format(
            self.__class__.__name__, self.model.version, self.model.name,
            len(self.reaches), self.reaches.index.name,
            abbr_str(list(self.reaches.index), 4),
            len(self.segment_data), self.segment_data.index.name,
            abbr_str(list(self.segment_data.index), 4),
            segments_line,
            diversions_line,
            nper, '' if nper == 1 else 's',
            abbr_str(list(self.model.dis.perlen), 4)))

    def __eq__(self, other):
        """Return true if objects are equal."""
        import flopy
        try:
            for (ak, av), (bk, bv) in zip_longest(iter(self), iter(other)):
                if ak != bk:
                    return False
                is_none = (av is None, bv is None)
                if all(is_none):
                    continue
                elif any(is_none):
                    return False
                elif type(av) != type(bv):
                    return False
                elif isinstance(av, pd.DataFrame):
                    pd.testing.assert_frame_equal(av, bv)
                elif isinstance(av, pd.Series):
                    pd.testing.assert_series_equal(av, bv)
                elif isinstance(av, flopy.mf6.ModflowGwf):
                    # basic test
                    assert str(av) == str(bv)
                else:
                    assert av == bv
            return True
        except (AssertionError, TypeError, ValueError):
            return False

    def __iter__(self):
        """Return object datasets with an iterator."""
        yield "class", self.__class__.__name__
        yield "segments", self.segments
        yield "segment_data", self.segment_data
        yield "reaches", self.reaches
        yield "diversions", self.diversions
        yield "model", self.model

    def __getstate__(self):
        """Serialize object attributes for pickle dumps."""
        return dict(self)

    def __setstate__(self, state):
        """Set object attributes from pickle loads."""
        if not isinstance(state, dict):
            raise ValueError("expected 'dict'; found {!r}".format(type(state)))
        elif "class" not in state:
            raise KeyError("state does not have 'class' key")
        elif state["class"] != self.__class__.__name__:
            raise ValueError("expected state class {!r}; found {!r}"
                             .format(state["class"], self.__class__.__name__))
        self.__init__()
        self.segments = state["segments"]
        self.segment_data = state["segment_data"]
        self.reaches = state["reaches"]
        self.diversions = state["diversions"]
        # Note: model must be set outsie of this method

    @_SwnModflow.model.setter
    def model(self, model):
        """Set model property from flopy.mf6.ModflowGwf."""
        import flopy
        if not (isinstance(model, flopy.mf6.ModflowGwf)
                or isinstance(model, flopy.mf6.mfmodel.MFModel)):
            raise ValueError(
                "'model' must be a  flopy.mf6.ModflowGwf or flopy.mf6.mfmodel.MFModel object; found "
                + str(type(model)))
        elif 'dis' not in model.package_type_dict.keys():
            raise ValueError('DIS package required')
        if getattr(self, '_model', None) is not model:
            self.logger.info("swapping 'model' object")
        self._model = model
        # Build stress period DataFrame from modflow model
        stress_df = pd.DataFrame(
            {'perlen': self.model.simulation.tdis.perioddata.array.perlen})
        modeltime = self.model.modeltime
        stress_df['duration'] = pd.TimedeltaIndex(
            stress_df['perlen'].cumsum(), modeltime.time_units)
        stress_df['start'] = pd.to_datetime(modeltime.start_datetime)
        stress_df['end'] = stress_df['duration'] + stress_df.at[0, 'start']
        stress_df.loc[1:, 'start'] = stress_df['end'].iloc[:-1].values
        self._stress_df = stress_df  # keep this for debugging
        self.time_index = pd.DatetimeIndex(stress_df['start']).copy()
        self.time_index.name = None

    def plot(self, column='iseg',
             cmap='viridis_r', legend=False):
        """
        Show map of reaches with inflow segments in royalblue.

        Parameters
        ----------
        column : str
            Column from reaches to use with 'cmap'; default 'iseg'.
            See also 'legend' to help interpret values.
        cmap : str
            Matplotlib color map; default 'viridis_r',
        legend : bool
            Show legend for 'column'; default False.

        Returns
        -------
        AxesSubplot

        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        self.reaches[~self.reaches.is_empty].plot(
            column=column, label='reaches', legend=legend, ax=ax, cmap=cmap)

        self.grid_cells.plot(ax=ax, color='whitesmoke', edgecolor='gainsboro')
        # return ax

        is_diversion = self.segment_data['iupseg'] != 0
        outlet_sel = (self.segment_data['outseg'] == 0) & (~is_diversion)
        outlet_points = self.reaches.loc[self.reaches['iseg'].isin(
            self.segment_data.loc[outlet_sel].index), 'geometry']\
            .apply(lambda g: Point(g.coords[-1]))
        outlet_points.plot(
            ax=ax, label='outlet', marker='o', color='navy')
        if 'inflow_segnums' in self.segment_data.columns:
            inflow_sel = ~self.segment_data['inflow_segnums'].isnull()
            inflow_points = self.reaches.loc[self.reaches['iseg'].isin(
                self.segment_data.loc[inflow_sel].index), 'geometry']\
                .apply(lambda g: Point(g.coords[0]))
            inflow_points.plot(
                ax=ax, label='inflow points', marker='o', color='royalblue')

        return ax

    def get_reach_data(self):
        """Return numpy.recarray for flopy's ModflowSfr2 reach_data.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.recarray

        """
        from flopy.modflow.mfsfr2 import ModflowSfr2
        # Build reach_data for Data Set 2
        reach_data_names = []
        for name in ModflowSfr2.get_default_reach_dtype().names:
            if name in self.reaches.columns:
                reach_data_names.append(name)
        reach_data = pd.DataFrame(self.reaches[reach_data_names])
        return reach_data.to_records(index=True)

    def set_segment_data(self, abstraction={}, inflow={}, flow={}, runoff={},
                         etsw={}, pptsw={}, return_dict=False):
        """
        Set timeseries data in segment_data required for flopy's ModflowSfr2.

        This method does two things:

            1. Updates sfr.segment_data, which is a dict of rec.array
               for each stress period.
            2. Updates summary statistics in segment_data if there are more
               than one stress period, otherwise values are kept for one
               stress period.

        Other stationary data members that are part of segment_data
        (e.g. hcond1, elevup, etc.) are not modified.

        Parameters
        ----------
        abstraction : dict or pandas.DataFrame, optional
            Surface water abstraction from diversions. Default is {} (zero).
            Keys are matched to diversions index.
        inflow : dict or pandas.DataFrame, optional
            Streamflow at the bottom of each segment, which is used to to
            determine the streamflow entering the upstream end of a segment if
            it is not part of the SFR network. Internal flows are ignored.
            A dict can be used to provide constant values to segnum
            identifiers. If a DataFrame is passed for a model with more than
            one stress period, the index must be a DatetimeIndex aligned with
            the start of each model stress period.
            Default is {} (no outside inflow added to flow term).
        flow : dict or pandas.DataFrame, optional
            Flow to the top of each segment. This is added to any inflow,
            which is handled separately. This can be negative for withdrawls.
            Default is {} (zero).
        runoff : dict or pandas.DataFrame, optional
            Runoff to each segment. Default is {} (zero).
        etsw : dict or pandas.DataFrame, optional
            Evapotranspiration removed from each segment. Default is {} (zero).
        pptsw : dict or pandas.DataFrame, optional
            Precipitation added to each segment. Default is {} (zero).
        return_dict : bool, optional
            If True, return segment_data instead of setting the sfr object.
            Default False, which implies that an sfr object exists.

        Returns
        -------
        None or dict (if return_dict is True)

        """
        from flopy.modflow.mfsfr2 import ModflowSfr2
        # Build stress period DataFrame from modflow model
        dis = self.model.dis
        stress_df = pd.DataFrame({'perlen': dis.perlen.array})
        modeltime = self.model.modeltime
        stress_df['duration'] = pd.TimedeltaIndex(
                stress_df['perlen'].cumsum(), modeltime.time_units)
        stress_df['start'] = pd.to_datetime(modeltime.start_datetime)
        stress_df['end'] = stress_df['duration'] + stress_df.at[0, 'start']
        stress_df.loc[1:, 'start'] = stress_df['end'].iloc[:-1].values
        # Consider all IDs from segments/diversions
        segments_segnums = set(self.segments.index)
        has_diversions = self.diversions is not None
        if has_diversions:
            diversions_divids = set(self.diversions.index)
        else:
            diversions_divids = set()

        def check_ts(data, name):
            """Return DataFrame with index along nper.

            Columns are either segnum or divid (checked later).
            """
            if isinstance(data, dict):
                data = pd.DataFrame(data, index=stress_df['start'])
            elif not isinstance(data, pd.DataFrame):
                raise ValueError(
                    '{0} must be a dict or DataFrame'.format(name))
            data.index.name = name  # handy for debugging
            if len(data) != dis.nper:
                raise ValueError(
                    'length of {0} ({1}) is different than nper ({2})'
                    .format(name, len(data), dis.nper))
            if dis.nper > 1:  # check DatetimeIndex
                if not isinstance(data.index, pd.DatetimeIndex):
                    raise ValueError(
                        '{0}.index must be a pandas.DatetimeIndex'
                        .format(name))
                elif not (data.index == stress_df['start']).all():
                    try:
                        t = stress_df['start'].to_string(
                                index=False, max_rows=5).replace('\n', ', ')
                    except TypeError:
                        t = abbr_str(list(stress_df['start']))
                    raise ValueError(
                        '{0}.index does not match expected ({1})'
                        .format(name, t))
            # Also do basic check of column IDs against diversions/segments
            if name == 'abstraction':
                if not has_diversions:
                    if len(data.columns) > 0:
                        self.logger.error(
                            'abstraction provided, but diversions are not '
                            'defined for the surface water network')
                        data.drop(data.columns, axis=1, inplace=True)
                    return data
                parent = self.diversions
                parent_name = 'diversions'
                parent_s = diversions_divids
            else:
                parent = self.segments
                parent_name = 'segments'
                parent_s = segments_segnums
            try:
                data.columns = data.columns.astype(parent.index.dtype)
            except (ValueError, TypeError):
                raise ValueError(
                    '{0}.columns.dtype must be same as {1}.index.dtype'
                    .format(name, parent_name))
            data_id_s = set(data.columns)
            if len(data_id_s) > 0:
                if data_id_s.isdisjoint(parent_s):
                    msg = '{0}.columns (or keys) not found in {1}.index: {2}'\
                        .format(name, parent_name, abbr_str(data_id_s))
                    if name == 'inflow':
                        self.logger.warning(msg)
                    else:
                        raise ValueError(msg)
                if name != 'inflow':  # some segnums accumulate outside flow
                    not_found = data_id_s.difference(parent_s)
                    if not data_id_s.issubset(parent_s):
                        self.logger.warning(
                            'dropping %s of %s %s.columns, which are '
                            'not found in %s.index: %s',
                            len(not_found), len(data_id_s), name,
                            parent_name, abbr_str(data_id_s))
                        data.drop(not_found, axis=1, inplace=True)
            return data

        self.logger.debug('checking timeseries data against modflow model')
        abstraction = check_ts(abstraction, 'abstraction')
        inflow = check_ts(inflow, 'inflow')
        flow = check_ts(flow, 'flow')
        runoff = check_ts(runoff, 'runoff')
        etsw = check_ts(etsw, 'etsw')
        pptsw = check_ts(pptsw, 'pptsw')

        # Translate segnum/divid to nseg
        is_diversion = self.segment_data['iupseg'] != 0
        divid2nseg = self.segment_data[is_diversion]\
            .reset_index().set_index('segnum')['nseg']
        divid2nseg_d = divid2nseg.to_dict()
        segnum2nseg = self.segment_data[~is_diversion]\
            .reset_index().set_index('segnum')['nseg']
        segnum2nseg_d = segnum2nseg.to_dict()
        segnum_s = set(segnum2nseg_d.keys())

        def map_nseg(data, name):
            data_id_s = set(data.columns)
            if len(data_id_s) == 0:
                return data
            if name == 'abstraction':
                colid2nseg_d = divid2nseg_d
                parent_descr = 'diversions'
            else:
                colid2nseg_d = segnum2nseg_d
                parent_descr = 'regular segments'
            colid_s = set(colid2nseg_d.keys())
            not_found = data_id_s.difference(colid_s)
            if not data_id_s.issubset(colid_s):
                self.logger.warning(
                    'dropping %s of %s %s.columns, which are '
                    'not found in segment_data.index for %s',
                    len(not_found), len(data_id_s), name,
                    parent_descr)
                data.drop(not_found, axis=1, inplace=True)
            return data.rename(columns=colid2nseg_d)

        self.logger.debug('mapping segnum/divid to segment_data.index (nseg)')
        abstraction = map_nseg(abstraction, 'abstraction')
        flow = map_nseg(flow, 'flow')
        runoff = map_nseg(runoff, 'runoff')
        etsw = map_nseg(etsw, 'etsw')
        pptsw = map_nseg(pptsw, 'pptsw')

        self.logger.debug('accumulating inflow from outside network')
        # Create an 'inflows' DataFrame calculated from combining 'inflow'
        inflows = pd.DataFrame(index=inflow.index)
        has_inflow = len(inflow.columns) > 0
        missing_inflow_segnums = []
        if has_inflow:
            self.segment_data['inflow_segnums'] = None
        elif 'inflow_segnums' in self.segment_data:
            self.segment_data.drop('inflow_segnums', axis=1, inplace=True)
        # Determine upstream flows needed for each SFR segment
        for segnum in self.segment_data.loc[~is_diversion, 'segnum']:
            nseg = segnum2nseg_d[segnum]
            from_segnums = self.segments.at[segnum, 'from_segnums']
            if not from_segnums:
                continue
            # gather segments outside SFR network
            outside_segnums = from_segnums.difference(segnum_s)
            if not outside_segnums:
                continue
            if has_inflow:
                inflow_series = pd.Series(0.0, index=inflow.index)
                inflow_segnums = set()
                for from_segnum in outside_segnums:
                    try:
                        inflow_series += inflow[from_segnum]
                        inflow_segnums.add(from_segnum)
                    except KeyError:
                        self.logger.warning(
                            'flow from segment %s not provided by inflow term '
                            '(needed for %s)', from_segnum, segnum)
                        missing_inflow_segnums.append(from_segnum)
                if inflow_segnums:
                    inflows[nseg] = inflow_series
                    self.segment_data.at[nseg, 'inflow_segnums'] = \
                        inflow_segnums
            else:
                missing_inflow_segnums += outside_segnums
        if not has_inflow and len(missing_inflow_segnums) > 0:
            self.logger.warning(
                'inflow from %d segnums are needed to determine flow from '
                'outside SFR network: %s', len(missing_inflow_segnums),
                abbr_str(missing_inflow_segnums))
        # Append extra columns to segment_data that are used by flopy
        segment_column_names = []
        nss = len(self.segment_data)
        for name, dtype in ModflowSfr2.get_default_segment_dtype().descr:
            if name == 'nseg':  # skip adding the index
                continue
            segment_column_names.append(name)
            if name not in self.segment_data.columns:
                self.segment_data[name] = np.zeros(nss, dtype=dtype)
        # Re-assmble stress period dict for flopy, with iper keys
        segment_data = {}
        has_abstraction = len(abstraction.columns) > 0
        has_inflows = len(inflows.columns) > 0
        has_flow = len(flow.columns) > 0
        has_runoff = len(runoff.columns) > 0
        has_etsw = len(etsw.columns) > 0
        has_pptsw = len(pptsw.columns) > 0
        for iper in range(dis.nper):
            # Store data for each stress period
            self.segment_data['flow'] = 0.0
            self.segment_data['runoff'] = 0.0
            self.segment_data['etsw'] = 0.0
            self.segment_data['pptsw'] = 0.0
            if has_abstraction:
                item = abstraction.iloc[iper]
                self.segment_data.loc[item.index, 'flow'] = item
            if has_inflows:
                item = inflows.iloc[iper]
                self.segment_data.loc[item.index, 'flow'] += item
            if has_flow:
                item = flow.iloc[iper]
                self.segment_data.loc[item.index, 'flow'] += item
            if has_runoff:
                item = runoff.iloc[iper]
                self.segment_data.loc[item.index, 'runoff'] = item
            if has_etsw:
                item = etsw.iloc[iper]
                self.segment_data.loc[item.index, 'etsw'] = item
            if has_pptsw:
                item = pptsw.iloc[iper]
                self.segment_data.loc[item.index, 'pptsw'] = item
            segment_data[iper] = self.segment_data[segment_column_names]\
                .to_records(index=True)  # index is nseg

        # For models with more than one stress period, evaluate summary stats
        if dis.nper > 1:
            # Remove time-varying data from last stress period
            self.segment_data.drop(
                ['flow', 'runoff', 'etsw', 'pptsw'], axis=1, inplace=True)

            def add_summary_stats(name, df):
                if len(df.columns) == 0:
                    return
                self.segment_data[name + '_min'] = 0.0
                self.segment_data[name + '_mean'] = 0.0
                self.segment_data[name + '_max'] = 0.0
                min_v = df.min(0)
                mean_v = df.mean(0)
                max_v = df.max(0)
                self.segment_data.loc[min_v.index, name + '_min'] = min_v
                self.segment_data.loc[mean_v.index, name + '_mean'] = mean_v
                self.segment_data.loc[max_v.index, name + '_max'] = max_v

            add_summary_stats('abstraction', abstraction)
            add_summary_stats('inflow', inflows)
            add_summary_stats('flow', flow)
            add_summary_stats('runoff', runoff)
            add_summary_stats('etsw', etsw)
            add_summary_stats('pptsw', pptsw)

        if return_dict:
            return segment_data
        else:
            self.model.sfr.segment_data = segment_data

    def get_seg_ijk(self):
        """Get the upstream and downstream segment k,i,j."""
        topidx = self.reaches['ireach'] == 1
        kij_df = self.reaches[topidx][['iseg', 'k', 'i', 'j']].sort_values(
            'iseg')
        idx_name = self.segment_data.index.name or 'index'
        self.segment_data = self.segment_data.reset_index().merge(
            kij_df, left_on='nseg', right_on='iseg', how='left').drop(
            'iseg', axis=1).set_index(idx_name)
        self.segment_data.rename(
            columns={"k": "k_up", "i": "i_up", "j": "j_up"}, inplace=True)
        # seg bottoms
        btmidx = self.reaches.groupby('iseg')['ireach'].transform(max) == \
            self.reaches['ireach']
        kij_df = self.reaches[btmidx][['iseg', 'k', 'i', 'j']].sort_values(
            'iseg')

        self.segment_data = self.segment_data.reset_index().merge(
            kij_df, left_on='nseg', right_on='iseg', how='left').drop(
            'iseg', axis=1).set_index(idx_name)
        self.segment_data.rename(
            columns={"k": "k_dn", "i": "i_dn", "j": "j_dn"}, inplace=True)
        return self.segment_data[[
            "k_up", "i_up", "j_up", "k_dn", "i_dn", "j_dn"]]

    def get_top_elevs_at_segs(self, m=None):
        """
        Get topsurface elevations associated with segment up and dn elevations.

        Adds elevation of model top at
        upstream and downstream ends of each segment
        :param m: modeflow model with active dis package
        :return: Adds 'top_up' and 'top_dn' columns to segment data dataframe
        """
        if m is None:
            m = self.model
        assert m.sfr is not None, "need sfr package"
        self.segment_data['top_up'] = m.dis.top.array[
            tuple(self.segment_data[['i_up', 'j_up']].values.T)]
        self.segment_data['top_dn'] = m.dis.top.array[
            tuple(self.segment_data[['i_dn', 'j_dn']].values.T)]
        return self.segment_data[['top_up', 'top_dn']]

    def get_segment_incision(self):
        """
        Calculate the upstream and downstream incision of the segment.

        :return:
        """
        self.segment_data['diff_up'] = (self.segment_data['top_up'] -
                                        self.segment_data['elevup'])
        self.segment_data['diff_dn'] = (self.segment_data['top_dn'] -
                                        self.segment_data['elevdn'])
        return self.segment_data[['diff_up', 'diff_dn']]

    def set_seg_minincise(self, minincise=0.2, max_str_z=None):
        """
        Set segment elevation to have the minumum incision from the top.

        :param minincise: Desired minimum incision
        :param max_str_z: Optional parameter to prevent streams at
        high elevations (forces incision to max_str_z)
        :return: incisions at the upstream and downstream end of each segment
        """
        sel = self.segment_data['diff_up'] < minincise
        self.segment_data.loc[sel, 'elevup'] = (self.segment_data.loc[
                                                    sel, 'top_up'] - minincise)
        sel = self.segment_data['diff_dn'] < minincise
        self.segment_data.loc[sel, 'elevdn'] = (self.segment_data.loc[
                                                    sel, 'top_dn'] - minincise)
        if max_str_z is not None:
            sel = self.segment_data['elevup'] > max_str_z
            self.segment_data.loc[sel, 'elevup'] = max_str_z
            sel = self.segment_data['elevdn'] > max_str_z
            self.segment_data.loc[sel, 'elevdn'] = max_str_z
        # recalculate incisions
        updown_incision = self.get_segment_incision()
        return updown_incision

    def get_segment_length(self):
        """
        Get segment length from accumulated reach lengths.

        :return:
        """
        # extract segment length for calculating minimun drop later
        reaches = self.reaches[['geometry', 'iseg', 'rchlen']].copy()
        seglen = reaches.groupby('iseg')['rchlen'].sum()
        self.segment_data.loc[seglen.index, 'seglen'] = seglen
        return seglen

    def get_outseg_elev(self):
        """Get the max elevup from all downstream segments for each segment."""
        self.segment_data['outseg_elevup'] = self.segment_data.outseg.apply(
            lambda x: self.segment_data.loc[
                self.segment_data.index == x].elevup).max(axis=1)
        return self.segment_data['outseg_elevup']

    def set_outseg_elev_for_seg(self, seg):
        """Set outseg elevation for segment.

        Gets all the defined outseg_elevup associated with a specific segment
        (multiple upstream segements route to one segment)
        Returns a df with all the calculated outseg elevups for each segment.
        .min(axis=1) is a good way to collapse to a series
        :param seg: Pandas Series containing one row of seg_data dataframe
        :return: Returns a df of the outseg_elev up values
        where current segment is listed as an outseg
        """
        # downstreambuffer = 0.001 # 1mm
        # find where seg is listed as outseg
        outsegsel = self.segment_data['outseg'] == seg.name
        # set outseg elevup
        outseg_elevup = self.segment_data.loc[outsegsel, 'outseg_elevup']
        return outseg_elevup

    def minslope_seg(self, seg, *args):
        """
        Force segment to have minumim slope (check for backward flowing segs).

        Moves downstream end down (vertically, more incision)
        to acheive minimum slope.
        :param seg: Pandas Series containing one row of seg_data dataframe
        :param args: desired minumum slope
        :return: Pandas Series with new downstream elevation and
        associated outseg_elevup
        """
        # segdata_df = args[0]
        minslope = args[0]
        downstreambuffer = 0.001  # 1mm
        up = seg.elevup
        dn = np.nan
        outseg_up = np.nan
        # prefer slope derived from surface
        surfslope = (seg.top_up-seg.top_dn)/(10.*seg.seglen)
        prefslope = np.max([surfslope, minslope])
        if seg.outseg > 0.0:
            # select outflow segment for current seg and pull out elevup
            outsegsel = self.segment_data.index == seg.outseg
            outseg_elevup = self.segment_data.loc[outsegsel, 'elevup']
            down = outseg_elevup.values[0]
            if down >= up - (seg.seglen * prefslope):
                # downstream elevation too high
                dn = up - (seg.seglen * prefslope)  # set to minslope
                outseg_up = up - (seg.seglen * prefslope) - downstreambuffer
                print('Segment {}, outseg = {}, old outseg_elevup = {}, '
                      'new outseg_elevup = {}'
                      .format(seg.name, seg.outseg,
                              seg.outseg_elevup, outseg_up))
            else:
                dn = down
                outseg_up = down - downstreambuffer
        else:
            # must be an outflow segment
            down = seg.elevdn
            if down > up - (seg.seglen * prefslope):
                dn = up - (seg.seglen * prefslope)
                print('Outflow Segment {}, outseg = {}, old elevdn = {}, '
                      'new elevdn = {}'
                      .format(seg.name, seg.outseg, seg.elevdn, dn))
            else:
                dn = down
        # this returns a DF once the apply is done!
        return pd.Series({'nseg': seg.name, 'elevdn': dn,
                          'outseg_elevup': outseg_up})

    def set_forward_segs(self, min_slope=1.e-4):
        """Set minimum slope in forwards direction.

        Ensure slope of all segment is at least min_slope
        in the downstream direction.
        Moves down the network correcting downstream elevations if necessary
        :param min_slope: Desired minimum slope
        :return: and updated segment data df
        """
        # upper most segments (not referenced as outsegs)
        # segdata_df = self.segment_data.sort_index(axis=1)
        segsel = ~self.segment_data.index.isin(self.segment_data['outseg'])
        while segsel.sum() > 0:
            print('Checking elevdn and outseg_elevup for {} segments'
                  .format(segsel.sum()))
            # get elevdn and outseg_elevups with a minimum slope constraint
            # index should align with self.segment_data index
            # not applying directly allows us to filter out nans
            tmp = self.segment_data.assign(_='').loc[segsel].apply(
                self.minslope_seg, args=[min_slope], axis=1)
            ednsel = tmp[tmp['elevdn'].notna()].index
            oeupsel = tmp[tmp['outseg_elevup'].notna()].index
            # set elevdn and outseg_elevup
            self.segment_data.loc[ednsel, 'elevdn'] = tmp.loc[ednsel, 'elevdn']
            self.segment_data.loc[oeupsel, 'outseg_elevup'] = \
                tmp.loc[oeupsel, 'outseg_elevup']
            # get `elevups` for outflow segs from `outseg_elevups`
            # taking `min` ensures outseg elevup is below all inflow elevdns
            tmp2 = self.segment_data.apply(
                self.set_outseg_elev_for_seg, axis=1).min(axis=1)
            tmp2 = pd.DataFrame(tmp2, columns=['elevup'])
            # update `elevups`
            eupsel = tmp2[tmp2.loc[:, 'elevup'].notna()].index
            self.segment_data.loc[eupsel, 'elevup'] = \
                tmp2.loc[eupsel, 'elevup']
            # get list of next outsegs
            segsel = self.segment_data.index.isin(
                self.segment_data.loc[segsel, 'outseg'])
        return self.segment_data

    def fix_segment_elevs(self, min_incise=0.2, min_slope=1.e-4,
                          max_str_z=None):
        """
        Provide wrapper function for calculating SFR segment elevations.

        Calls series of functions to process and move sfr segment elevations,
        to try to ensure:
            0. Segments are below the model top
            1. Segments flow downstream
            2. Downstream segments are below upstream segments
        :param min_slope: desired minimum slope for segment
        :param min_incise: desired minimum incision (in model units)
        :return: segment data dataframe
        """
        kijcols = {"k_up", "i_up", "j_up", "k_dn", "i_dn", "j_dn"}
        dif = kijcols - set(self.segment_data.columns)
        if len(dif) > 1:
            # some missing
            # drop others
            others = kijcols - dif
            self.segment_data.drop(others, axis=0, inplace=True)
            # get model locations for segments ends
            _ = self.get_seg_ijk()
        # get model cell elevations at seg ends
        _ = self.get_top_elevs_at_segs()
        # get current segment incision at seg ends
        _ = self.get_segment_incision()
        # move segments end elevation down to achieve minimum incision
        _ = self.set_seg_minincise(minincise=min_incise, max_str_z=max_str_z)
        # get the elevations of downstream segments
        _ = self.get_outseg_elev()
        # get segment length from reach lengths
        _ = self.get_segment_length()
        # ensure downstream ends are below upstream ends
        # and reconcile upstream elevation of downstream segments
        self.set_forward_segs(min_slope=min_slope)
        # reassess segment incision after processing.
        self.get_segment_incision()
        return self.segment_data

    def reconcile_reach_strtop(self):
        """
        Recalculate reach strtop elevations after moving segment elevations.

        :return: None
        """
        def reach_elevs(seg):
            """Return reach properties.

            Calculate reach elevation from segment slope and
            reach length along segment.
            :param seg: one row of reach data dataframe grouped by segment
            :return: reaches by segment with strtop adjusted
            """
            segsel = self.segment_data.index == seg.name

            seg_elevup = self.segment_data.loc[segsel, 'elevup'].values[0]
            seg_slope = self.segment_data.loc[segsel, 'Zslope'].values[0]

            # interpolate reach lengths to cell centres
            cmids = seg.seglen.shift().fillna(0.0) + seg.rchlen.multiply(0.5)
            cmids.iat[0] = 0.0
            cmids.iat[-1] = seg.seglen.iloc[-1]
            seg['cmids'] = cmids  # cummod+(seg.rchlen *0.5)
            # calculate reach strtops
            seg['strtop'] = seg['cmids'].multiply(seg_slope) + seg_elevup
            # seg['slope']= #!!!!! use m.sfr.get_slopes() method
            return seg
        self.segment_data['Zslope'] = \
            ((self.segment_data['elevdn'] - self.segment_data['elevup']) /
             self.segment_data['seglen'])
        segs = self.reaches.groupby('iseg')
        self.reaches['seglen'] = segs.rchlen.cumsum()
        self.reaches = segs.apply(reach_elevs)
        return self.reaches

    def set_topbot_elevs_at_reaches(self, m=None):
        """
        Get top and bottom elevation of the cell containing a reach.

        :param m: Modflow model
        :return: dataframe with reach cell top and bottom elevations
        """
        if m is None:
            m = self.model
        self.reaches['top'] = m.dis.top.array[
            tuple(self.reaches[['i', 'j']].values.T)]
        self.reaches['bot'] = m.dis.botm[0].array[
            tuple(self.reaches[['i', 'j']].values.T)]
        return self.reaches[['top', 'bot']]

    def fix_reach_elevs(self, minslope=0.0001, fix_dis=True, minthick=0.5):
        """Fix reach elevations.

        Need to ensure reach elevation is:
            0. below the top
            1. below the upstream reach
            2. above the minimum slope to the bottom reach elevation
            3. above the base of layer 1
        segment by segment, reach by reach! Fun!

        :return:
        """
        def _check_reach_v_laybot(r, botms, buffer=1.0, rbed_elev=None):
            if rbed_elev is None:
                rbed_elev = r.strtop - r.strthick
            if (rbed_elev - buffer) < r.bot:
                # if new strtop is below layer one
                # drop bottom of layer one to accomodate stream
                # (top, bed thickness and buffer)
                new_elev = rbed_elev - buffer
                print('seg {} reach {} @ {} '
                      'is below layer 1 bottom @ {}'
                      .format(seg, r.ireach, rbed_elev,
                              r.bot))
                print('    dropping layer 1 bottom to {} '
                      'to accommodate stream @ i = {}, j = {}'
                      .format(new_elev, r.i, r.j))
                botms[0, r.i, r.j] = new_elev
            return botms

        buffer = 1.0  # 1 m (buffer to leave at the base of layer 1 -
        # also helps with precision issues)
        # make sure elevations are up-to-date
        # recalculate REACH strtop elevations
        self.reconcile_reach_strtop()
        _ = self.set_topbot_elevs_at_reaches()
        # top read from dis as float32 so comparison need to be with like
        reachsel = self.reaches['top'] <= self.reaches['strtop']
        reach_ij = tuple(self.reaches[['i', 'j']].values.T)
        print('{} segments with reaches above model top'.format(
            self.reaches[reachsel]['iseg'].unique().shape[0]))
        # get segments with reaches above the top surface
        segsabove = self.reaches[reachsel].groupby(
            'iseg').size().sort_values(ascending=False)
        # get incision gradient from segment elevups and elevdns
        # ('diff_up' and 'diff_dn' are the incisions of the top and
        # bottom reaches from the segment data)
        self.segment_data['incgrad'] = \
            ((self.segment_data['diff_up'] - self.segment_data['diff_dn']) /
             self.segment_data['seglen'])
        # copy of layer 1 bottom (for updating to fit in stream reaches)
        layerbots = self.model.dis.botm.array.copy()
        # loop over each segment
        for seg in self.segment_data.index:  # (all segs)
            # selection for segment in reachdata and seg data
            rsel = self.reaches['iseg'] == seg
            segsel = self.segment_data.index == seg

            if seg in segsabove.index:
                # check top and bottom reaches are above layer 1 bottom
                # (not adjusting elevations of reaches)
                for reach in self.reaches[rsel].iloc[[0, -1]].itertuples():
                    layerbots = _check_reach_v_laybot(reach, layerbots, buffer)
                # apparent optimised incision based
                # on the incision gradient for the segment
                self.reaches.loc[rsel, 'strtop_incopt'] = \
                    self.reaches.loc[rsel, 'top'].subtract(
                        self.segment_data.loc[segsel, 'diff_up'].values[0]) + \
                    (self.reaches.loc[rsel, 'cmids'].subtract(
                        self.reaches.loc[rsel, 'cmids'].values[0]) *
                     self.segment_data.loc[segsel, 'incgrad'].values[0])
                # falls apart when the top elevation is not monotonically
                # decreasing down the segment (/always!)

                # bottom reach elevation:
                botreach_strtop = self.reaches[rsel]['strtop'].values[-1]
                # total segment length
                seglen = self.reaches[rsel]['seglen'].values[-1]
                botreach_slope = minslope  # minimum slope of segment
                # top reach elevation and "length?":
                upreach_strtop = self.reaches[rsel]['strtop'].values[0]
                upreach_cmid = self.reaches[rsel]['cmids'].values[0]
                # use top reach as starting point

                # loop over reaches in segement from second to penultimate
                # (dont want to move elevup or elevdn)
                for reach in self.reaches[rsel][1:-1].itertuples():
                    # strtop that would result from minimum slope
                    # from upstream reach
                    strtop_withminslope = upreach_strtop - (
                            (reach.cmids - upreach_cmid) * minslope)
                    # strtop that would result from minimum slope
                    # from bottom reach
                    strtop_min2bot = botreach_strtop + (
                            (seglen - reach.cmids) * minslope)
                    # check 'optimum incision' is below upstream elevation
                    # and above the minimum slope to the bottom reach
                    if reach.strtop_incopt < strtop_min2bot:
                        # strtop would give too shallow a slope to
                        # the bottom reach (not moving bottom reach)
                        print('seg {} reach {}, incopt is \\/ below minimum '
                              'slope from bottom reach elevation'
                              .format(seg, reach.ireach))
                        print('    setting elevation to minslope from bottom')
                        # set to minimum slope from outreach
                        self.reaches.at[
                            reach.Index, 'strtop'] = strtop_min2bot
                        # update upreach for next iteration
                        upreach_strtop = strtop_min2bot
                    elif reach.strtop_incopt > strtop_withminslope:
                        # strtop would be above upstream or give
                        # too shallow a slope from upstream
                        print('seg {} reach {}, incopt /\\ above upstream'
                              .format(seg, reach.ireach))
                        print('    setting elevation to minslope from '
                              'upstream')
                        # set to minimum slope from upstream reach
                        self.reaches.at[
                            reach.Index, 'strtop'] = strtop_withminslope
                        # update upreach for next iteration
                        upreach_strtop = strtop_withminslope
                    else:
                        # strtop might be ok to set to 'optimum incision'
                        print('seg {} reach {}, incopt is -- below upstream '
                              'reach and above the bottom reach'
                              .format(seg, reach.ireach))
                        # CHECK FIRST:
                        # if optimium incision would place it
                        # below the bottom of layer 1
                        if reach.strtop_incopt - reach.strthick < \
                                reach.bot + buffer:
                            # opt - stream thickness lower than layer 1 bottom
                            # (with a buffer)
                            print('seg {} reach {}, incopt - bot is x\\/ '
                                  'below layer 1 bottom'
                                  .format(seg, reach.ireach))
                            if reach.bot + reach.strthick + buffer > \
                                    strtop_withminslope:
                                # if layer bottom would put reach above
                                # upstream reach we can only set to
                                # minimum slope from upstream
                                print('    setting elevation to minslope '
                                      'from upstream')
                                self.reaches.at[reach.Index, 'strtop'] = \
                                    strtop_withminslope
                                upreach_strtop = strtop_withminslope
                            else:
                                # otherwise we can move reach so that it
                                # fits into layer 1
                                new_elev = reach.bot + reach.strthick + buffer
                                print('    setting elevation to {}, above '
                                      'layer 1 bottom'.format(new_elev))
                                # set reach top so that it is above layer 1
                                # bottom with a buffer
                                # (allowing for bed thickness)
                                self.reaches.at[reach.Index, 'strtop'] = \
                                    reach.bot + buffer + reach.strthick
                                upreach_strtop = new_elev
                        else:
                            # strtop ok to set to 'optimum incision'
                            # set to "optimum incision"
                            print('    setting elevation to incopt')
                            self.reaches.at[
                                reach.Index, 'strtop'] = reach.strtop_incopt
                            upreach_strtop = reach.strtop_incopt
                    # check if new stream top is above layer 1 with a buffer
                    # (allowing for bed thickness)
                    reachbed_elev = upreach_strtop - reach.strthick
                    layerbots = _check_reach_v_laybot(reach, layerbots, buffer,
                                                      reachbed_elev)
                    upreach_cmid = reach.cmids
                    # upreach_slope=reach.slope
            else:
                # For segments that do not have reaches above top
                # check if reaches are below layer 1
                print('seg {} is always downstream and below the top'
                      .format(seg))
                for reach in self.reaches[rsel].itertuples():
                    reachbed_elev = reach.strtop - reach.strthick
                    layerbots = _check_reach_v_laybot(reach, layerbots, buffer,
                                                      reachbed_elev)
            # OH CRAP need to update dis bottoms in reach df!
            # self.reaches['top'] = layerbots[
            #     tuple(self.reaches[['i', 'j']].values.T)]
            self.reaches['bot'] = layerbots[0][reach_ij]
        if fix_dis:
            # fix dis for incised reaches
            for lay in range(self.model.dis.nlay - 1):
                laythick = layerbots[lay] - layerbots[
                    lay + 1]  # first one is layer 1 bottom - layer 2 bottom
                print('checking layer {} thicknesses'.format(lay + 2))
                thincells = laythick < minthick
                print('{} cells less than {}'
                      .format(thincells.sum(), minthick))
                laythick[thincells] = minthick
                layerbots[lay + 1] = layerbots[lay] - laythick
            self.model.dis.botm = layerbots

    def sfr_plot(self, model, sfrar, dem, points=None, points2=None,
                 label=None):
        """Plot sfr."""
        from swn.modflow._modelplot import ModelPlot
        p = ModelPlot(model)
        p._add_plotlayer(dem, label="Elevation (m)")
        p._add_sfr(sfrar, cat_cmap=False, cbar=True,
                   label=label)
        return p

    def plot_reaches_above(self, model, seg, dem=None,
                           plot_bottom=False, points2=None):
        """Plot sfr reaches above."""
        # ensure reach elevations are up-to-date
        _ = self.set_topbot_elevs_at_reaches()
        dis = model.dis
        sfr = model.sfr
        if dem is None:
            dem = np.ma.array(
                dis.top.array, mask=model.dis.idomain.array[0] == 0)
        sfrar = np.ma.zeros(dis.top.array.shape, 'f')
        sfrar.mask = np.ones(sfrar.shape)
        lay1reaches = self.reaches.loc[
            self.reaches.k.apply(lambda x: x == 1)]
        points = None
        if lay1reaches.shape[0] > 0:
            points = lay1reaches[['i', 'j']]
        # segsel=reachdata['iseg'].isin(segsabove.index)
        if seg == 'all':
            segsel = np.ones((self.reaches.shape[0]), dtype=bool)
        else:
            segsel = self.reaches['iseg'] == seg
        sfrar[tuple((self.reaches[segsel][['i', 'j']]
                     .values.T).tolist())] = \
            (self.reaches[segsel]['top'] -
             self.reaches[segsel]['strtop']).tolist()
        # .mask = np.ones(sfrar.shape)
        vtop = self.sfr_plot(model, sfrar, dem, points=points, points2=points2,
                             label="str below top (m)")
        if seg != 'all':
            sfr.plot_path(seg)
        if plot_bottom:
            dembot = np.ma.array(dis.botm.array[0],
                                 mask=model.dis.idomain.array[0] == 0)
            sfrarbot = np.ma.zeros(dis.botm.array[0].shape, 'f')
            sfrarbot.mask = np.ones(sfrarbot.shape)
            sfrarbot[tuple((self.reaches[segsel][['i', 'j']]
                            .values.T).tolist())] = \
                (self.reaches[segsel]['strtop'] -
                 self.reaches[segsel]['bot']).tolist()
            # .mask = np.ones(sfrar.shape)
            vbot = self.sfr_plot(model, sfrarbot, dembot, points=points,
                                 points2=points2, label="str above bottom (m)")
        else:
            vbot = None
        return vtop, vbot

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
        model : flopy.mf6.ModflowGwf
            Instance of flopy.mf6.ModflowGwf.

        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        obj.model = model
        return obj

    @classmethod
    def from_swn_mf6_sfrds(cls):
        pass
