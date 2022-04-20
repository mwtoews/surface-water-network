import numpy as np

from ..spatial import get_crs

try:
    import matplotlib
    from matplotlib import cm, colors
    from matplotlib import pyplot as plt
except ImportError:
    matplotlib = False


class ModelPlot:
    """Object for plotting array style results."""

    def __init__(self, model, domain_extent=None, fig=None, ax=None,
                 figsize=None):
        """
        Container to help with results plotting functions.

        :param model: flopy modflow model object
        :param fig: matplotlib figure handle
        :param ax: Cartopy GeoAxes with .projection attribute
        :param domain_extent: np.array of array(
                                    [long-left , long-right, lat-up, lat-dn])
        """
        import pyproj
        from matplotlib import pyplot as plt

        self.model = model
        self.fig = fig
        self.ax = ax
        # Use model projection, if defined by either epsg or proj4 attrs
        epsg = model.modelgrid.epsg
        self.pprj = get_crs(epsg)
        if self.pprj is None:
            self.pprj = get_crs(model.modelgrid.proj4)
        if domain_extent is None and self.pprj is not None:
            xmin, xmax, ymin, ymax = self.model.modelgrid.extent
            assert self.pprj.is_projected, self.pprj
            latlon = pyproj.CRS.from_epsg(4326)  # axis order: lat, lon
            tfm = pyproj.Transformer.from_crs(self.pprj, latlon)
            latmin, lonmin = tfm.transform(xmin, ymin)
            latmax, lonmax = tfm.transform(xmax, ymax)
            self.domain_extent = [lonmin, lonmax, latmin, latmax]
        else:
            self.domain_extent = domain_extent
        self.xg = self.model.modelgrid.xvertices
        self.yg = self.model.modelgrid.yvertices
        self.extent = self.model.modelgrid.extent

        if figsize is None:
            figsize = (8, 8)
        # if no figure or axes based initialise figure
        if fig is None or ax is None:
            plt.rc('font', size=10)
            # see https://github.com/SciTools/cartopy/issues/813
            self.mprj = None
            if epsg:
                try:
                    import cartopy.crs as ccrs
                    self.mprj = ccrs.epsg(epsg)
                    # empty figure container cartopy geoaxes
                    self.fig, self.ax = plt.subplots(figsize=figsize,
                                                     subplot_kw=dict(
                                                         projection=self.mprj))
                except ImportError:
                    self.fig, self.ax = plt.subplots(figsize=figsize)
            else:
                self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            try:
                self.mprj = self.ax.projection  # map projection
            except NameError:
                self.mprj = None
        # self._get_base_ax()
        if self.ax is not None:
            self._set_divider()

    # def _get_base_ax(self):
    #     """
    #     Define the plot axis based on the extent of the domain
    #     """
    #     import cartopy.io.img_tiles as cimgt
    #
    #     terrain = cimgt.Stamen('terrain-background')
    #     self.ax.set_extent(self.domain_extent)
    #
    #     # Add the Stamen data at zoom level 8.
    #     self.ax.add_image(terrain, 9, cmap="Greys")
    #
    #     # rivers = cartopy.feature.NaturalEarthFeature(
    #     #     category='physical', name='rivers_lake_centerlines',
    #     #     scale='10m', facecolor='none', edgecolor='b')
    #   # ax.add_feature(rivers, linewidth=0.5,)#,transform=ccrs.PlateCarree())
    #     self._label_utm_grid()

    # def _label_utm_grid(self):
    #     """
    #     Label axes as UTM
    #     Warning: should only use with small area UTM maps
    #     """
    #   for val, label in zip(self.ax.get_xticks(), self.ax.get_xticklabels()):
    #         label.set_text(str(val))
    #         label.set_position((val, 0))
    #         label.set_rotation(90)
    #
    #   for val, label in zip(self.ax.get_yticks(), self.ax.get_yticklabels()):
    #         label.set_text(str(val))
    #         label.set_position((0, val))
    #
    #     self.ax.tick_params(bottom=True, top=True, left=True, right=True,
    #                         labelbottom=True, labeltop=False, labelleft=True,
    #                         labelright=False)
    #
    #     self.ax.xaxis.set_visible(True)
    #     self.ax.yaxis.set_visible(True)
    #     self.ax.set_xlabel("Easting ($m$)")
    #     self.ax.set_ylabel("Northing ($m$)")
    #     self.ax.grid(True)

    @staticmethod
    def _get_range(k):
        kmin = np.min(k)
        kmax = np.max(k)
        krange = kmax - kmin
        if krange < np.abs(0.05 * ((kmin + kmax)/2)):
            vmin = ((kmin + kmax)/2) - np.abs(0.025 * ((kmin + kmax)/2))
            vmax = ((kmin + kmax)/2) + np.abs(0.025 * ((kmin + kmax)/2))
        else:
            vmin = kmin
            vmax = kmax
        return vmin, vmax

    def _get_cbar_props(self):
        """Get colorbar properties.

        Get properties for next colorbar and its label based on the number of
        axis already in plot

        :return: divider padding, label locations for use when appending
            divider axes and setting colorbar label.
        """
        numax = len(self.ax.figure.axes)
        if np.mod(numax, 2) == 1:
            # odd number of axes so even number of cbars
            if numax == 1:
                # no colorbar yet - small pad
                divider_props = dict(pad=0.2)
            else:
                divider_props = dict(pad=0.5)
            props = dict(labelpad=-40, y=1.01, rotation=0,
                         ha='center', va='bottom')
        else:
            divider_props = dict(pad=0.5)
            props = dict(labelpad=-40, y=-0.01, rotation=0,
                         ha='center', va='top')
        return divider_props, props

    def _set_divider(self):
        """Initiate a mpl divider for the colorbar location."""
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        self.divider = make_axes_locatable(self.ax)

    def _add_ibound_mask(self, lay, zorder=20, alpha=0.5):
        """
        Add the ibound to plot.

        :param lay: layer for selecting model ibound
        :param zorder: mpl plotting overlay order
        :param alpha: mpl transparency
        """
        array = np.ones((self.model.nrow, self.model.ncol))
        array = np.ma.masked_where(
            self.model.bas6.ibound.array[lay] != 0, array)
        self.ax.imshow(
            array, extent=self.extent, transform=self.mprj, cmap="Greys_r",
            origin="upper", zorder=zorder, alpha=alpha)

    def _add_plotlayer(self, ar, vmin=None, vmax=None, norm=None, cmap=None,
                       zorder=10, alpha=0.8, cbar=True, label=None):
        """
        Add image for layer array.

        :param ar: 2D numpy array to plot
        :param vmin: minimum value to clip
        :param vmax: maximum value to clip
        :param norm: mpl normalizer (default is None)
        :param cmap: mpl colormap (default is 'plasma')
        :param zorder: mpl overlay order
        :param alpha: mpl transparency
        :param cbar: flag to plot with colorbar for layer
        :param label: label for colorbar
        """
        if self.ax is None:
            return
        if cmap is None:
            cmap = cm.get_cmap('viridis')
        if self.mprj is None:
            transform = self.ax.transData
        else:
            transform = self.mprj
        if label is None:
            print("No label passed for colour bar")
            label = ""
        hax = self.ax.imshow(ar, zorder=zorder, vmin=vmin, vmax=vmax,
                             extent=self.extent, origin="upper",
                             transform=transform, norm=norm,
                             alpha=alpha, cmap=cmap, interpolation="none")
        if cbar:
            if label is None:
                print("No label passed for colour bar")
                label = ""
            divider_props, props = self._get_cbar_props()
            cax = self.divider.append_axes("right", size="5%",
                                           axes_class=plt.Axes,
                                           **divider_props)
            cbar1 = self.fig.colorbar(hax, cax=cax)
            cbar1.set_label(label, **props)
        return hax

    def _add_sfr(self, x, zorder=11, cbar=True, cat_cmap=False,
                 label=None, cmap_txt='bwr_r'):
        """
        Plot the array of surface water exchange (with SFR).

        :param x: 2D numpy array
        :param zorder: mpl overlay order
        """
        vmin = x.min()
        vmax = x.max()
        if cat_cmap:
            import seaborn as sns
            vals = np.ma.unique(x).compressed()
            bounds = np.append(np.sort(vals), vals.max() + 1)
            n = len(vals)
            cmap = colors.ListedColormap(
                sns.color_palette('Set2', n).as_hex())
            norm = colors.BoundaryNorm(bounds, cmap.N)
        else:
            cmap = cm.get_cmap(cmap_txt)
            norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
        self._add_plotlayer(x, cmap=cmap, norm=norm, zorder=zorder,
                            alpha=1, label=label, cbar=cbar)
        # if points is not None:
        #     self.ax.scatter(self.model.modelgrid.xcellcenters[
        #                         points.i, points.j],
        #                     self.model.modelgrid.ycellcenters[
        #                         points.i, points.j],
        #                     marker='o', zorder=15, facecolors='none',
        #                     edgecolors='r')
        # if points2 is not None:
        #     self.ax.scatter(self.model.modelgrid.xcellcenters[
        #                         points2.i, points2.j],
        #                     self.model.modelgrid.xcellcenters[
        #                         points2.i, points2.j],
        #                     marker='o', zorder=15, facecolors='none',
        #                     edgecolors='b')
        if cbar:
            if cat_cmap:
                last_cbar = self.fig.get_axes()[-1]
                last_cbar.set_yticks(bounds[:-1] + np.diff(bounds) / 2)
                last_cbar.set_yticklabels(np.sort(vals).astype(int))

    def _add_lines(self, lines, zorder=12, cmap_txt="RdGy"):
        import matplotlib.patheffects as pe
        col = [c for c in lines if c != 'geometry']
        if len(col) == 1:
            col = col[0]
            vmin = lines[col].min()
            vmax = lines[col].max()
            cmap = cm.get_cmap(cmap_txt)
            norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
            cb = True
        else:
            col = None
            cmap = None
            norm = None
            cb = False
        ls = lines.plot(
            col, ax=self.ax, zorder=zorder, cmap=cmap, norm=norm,
        )
        [
            li.set_path_effects([pe.withStroke(linewidth=5, foreground='0.5')])
            for li in ls.collections
        ]
        if cb:
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            divider_props, props = self._get_cbar_props()
            cax = self.divider.append_axes("right", size="5%",
                                           axes_class=plt.Axes,
                                           **divider_props)
            cbar1 = self.fig.colorbar(sm, cax=cax)
        # divider_props, props = vtop._get_cbar_props()
        # cax = vtop.divider.append_axes("right", size="5%",
        #                                axes_class=plt.Axes,
        #                                **divider_props)


def _profile_plot(
        sorted_df,
        lentag='rchlen',
        x='rchlen',
        cols=['strtop', 'top', 'bot'],
        ax=None,
):
    if not sorted_df[x].is_monotonic_increasing:
        print(f"Expected x column of `sorted_df` ({x}) to be "
              "monotonically increasing but it isn't, "
              "working on cumulative sum")
        sorted_df['csum'] = sorted_df[x].cumsum()
        x = 'csum'
    else:
        # will use this for seg dividers anyway
        sorted_df['csum'] = sorted_df[lentag].cumsum()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sorted_df[[x] + cols].append(  # make sure we include end of final reach
        sorted_df.iloc[[-1]][['csum']+cols].rename(
            columns={'csum': x})).plot(x=x, ax=ax)
    # adding some window dressing
    end_seg = sorted_df.segnum.diff(-1) != 0
    start_seg = sorted_df.segnum.diff() != 0
    for _, s in sorted_df[end_seg].iterrows():
        ax.axvline(s["csum"], c='0.5', alpha=0.5, linestyle='--')
    trans = ax.get_xaxis_transform()
    for _, s in sorted_df[start_seg].iterrows():
        ax.text(s[x], 1, s.segnum, transform=trans)


def sfr_plot(model, sfrar, dem, label=None, lines=None):
    """Plot sfr."""
    p = ModelPlot(model)
    p._add_plotlayer(dem, label="Elevation (m)")
    p._add_sfr(sfrar, cat_cmap=False, cbar=True,
               label=label)
    if lines is not None:
        p._add_lines(lines)
    return p


if matplotlib:
    class MidpointNormalize(matplotlib.colors.Normalize):
        """Mid-point normalize class."""

        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            """Initialize method."""
            self.midpoint = midpoint
            matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            """Call method."""
            # I'm ignoring masked values and all kinds of edge cases to make a
            # simple example...
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            mask = np.ma.getmask(value)
            return np.ma.masked_array(np.interp(value, x, y), mask=mask)
