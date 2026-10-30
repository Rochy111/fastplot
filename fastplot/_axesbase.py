"""What every axes has, maps and plain x/y axes alike: a place in the
figure, titles, a background and a frame, a list of artists drawn in
zorder -- and the plotting methods both share (pcolormesh, contourf,
contour, clabel, barbs, plot, scatter, text). Where those need to know
the axes' coordinates they ask it, through a handful of hooks:

    _src(transform)                       what coordinates the data is in
    _points_px(x, y, src)                 points -> canvas pixels
    _lines_px(x, y, offsets, src, parts)  polylines -> canvas pixels
    _vectors_px(x, y, u, v, src)          points and their directions on screen
    _draw_field(r, grid, z, mode, coloring, alpha)   a field's value raster
    _default_extent(src, shape)           imshow's extent when none is given

MapAxes (_axes) answers through a map projection, Axes (_cartesian) through
its scales and limits.
"""
import numpy as np

from . import _raster, _ticker
from . import transforms as _transforms
from ._style import _ignore, cycle_color, fontsize_pt, rc, rgba
from ._layout import _Box
from ._artists import Barbs, Line2D, Scatter, Text, _as_host, _text_kw
from ._fields import ContourSet, ContourfArtist, FieldArtist, _Grid, _field


class Background:
    """matplotlib's ax.patch / fig.patch, for what code does with it: show or
    hide the background, set its colour (the owner's facecolor) or alpha."""

    def __init__(self, owner, flag):
        self._owner, self._flag = owner, flag

    def set_visible(self, b):
        setattr(self._owner, self._flag, bool(b))

    def get_visible(self):
        return getattr(self._owner, self._flag)

    def set_facecolor(self, c):
        self._owner.set_facecolor(c)

    set_color = set_facecolor

    def get_facecolor(self):
        return self._owner.get_facecolor()

    def set_alpha(self, a):
        if a is not None:
            from ._color import to_rgba
            self._owner.set_facecolor(to_rgba(self._owner.get_facecolor(), a))

    def set_edgecolor(self, c):                         # no outline of its own
        pass

    def set_linewidth(self, w):
        pass


class _TitleText:
    """An axes title as matplotlib's Text (set_title's return, ax.title)."""

    def __init__(self, ax, loc):
        self.ax, self.loc = ax, loc

    def _d(self):
        return self.ax.titles.setdefault(self.loc, dict(text='', fontsize=None, weight='normal',
                                                        color=rc('text.color'), pad=None,
                                                        y=None, style='normal'))

    def get_text(self):
        return self._d()['text']

    def set_text(self, s):
        self._d()['text'] = s

    def set_fontsize(self, s):
        self._d()['fontsize'] = s

    set_size = set_fontsize

    def get_fontsize(self):
        return fontsize_pt(rc('axes.titlesize') if self._d()['fontsize'] is None
                           else self._d()['fontsize'])

    def set_color(self, c):
        self._d()['color'] = c

    def get_color(self):
        return self._d()['color']

    def set_fontweight(self, w):
        self._d()['weight'] = w

    set_weight = set_fontweight

    def set_fontstyle(self, s):
        self._d()['style'] = s

    def set_y(self, y):
        self._d()['y'] = y

    def set_position(self, xy):
        self._d()['y'] = xy[1]

    def set_visible(self, v):
        if not v:
            self._d()['text'] = ''

    def set_ha(self, ha):
        pass

    set_horizontalalignment = set_ha

    def set(self, **kw):
        for k, v in kw.items():
            getattr(self, 'set_' + {'size': 'fontsize', 'c': 'color'}.get(k, k))(v)

    def __repr__(self):
        return 'Text(%r)' % self.get_text()


class AxesBase:
    """An axes: its place (spec), titles, background, frame, artists."""

    _snap = False                   # pixel snapping of rectilinear paths (plain axes: 'auto')
    _agg = False                    # matplotlib's caps (plain axes)
    zorder = 0                      # among the figure's axes, colorbars, artists and texts

    def __init__(self, fig, spec, label=''):
        self.figure = fig
        self.spec = spec                                # SubplotSpec or _Box
        self.artists = []
        self.titles = {}
        self.legend_ = None
        self.containers = []                            # bar / errorbar / stem results
        self.anchor = 'C'
        self.box_aspect = None
        self.facecolor = rc('axes.facecolor')
        self.frame_on = True
        self.transAxes = _transforms.TransAxes(self)
        self.label = label
        self.patch_visible = True
        self._top_extent = None
        self._placed = None
        self._cycle = {}
        self._stale_bounds = []                         # removed artists' limits, until relim

    def get_figure(self):
        return self.figure

    @property
    def patch(self):
        return Background(self, 'patch_visible')

    def get_facecolor(self):
        return self.facecolor

    def set_title(self, label, loc='center', fontsize=None, fontweight='normal', color=None,
                  pad=None, y=None, fontdict=None, **kw):
        if color is None:                               # matplotlib's: axes.titlecolor, 'auto':
            color = rc('axes.titlecolor')               # text.color
            if color == 'auto':
                color = rc('text.color')
        if fontdict:
            fontsize = fontdict.get('fontsize', fontsize)
            fontweight = fontdict.get('fontweight', fontweight)
            color = fontdict.get('color', color)
        fontsize = kw.pop('size', fontsize)
        fontweight = kw.pop('weight', fontweight)
        loc = rc('axes.titlelocation') if loc is None else loc
        self.titles[loc] = dict(text=label, fontsize=fontsize, weight=fontweight, color=color,
                                pad=pad, y=y, style=kw.pop('fontstyle', 'normal'))
        return _TitleText(self, loc)

    @property
    def title(self):
        return _TitleText(self, 'center')

    def set(self, **kw):
        """matplotlib's Artist.set on an axes: set_<name>(value) for each."""
        out = []
        for k, v in kw.items():
            f = getattr(self, 'set_' + k, None)
            if f is None:
                raise AttributeError('%s.set() got an unexpected keyword argument %r'
                                     % (type(self).__name__, k))
            if k in ('xlim', 'ylim') and np.iterable(v):
                out.append(f(*v))
            else:
                out.append(f(v))
        return out

    def set_visible(self, v):
        self.visible = bool(v)

    def get_visible(self):
        return getattr(self, 'visible', True)

    def remove(self):
        fig = self.figure
        if self in fig.axes:
            fig.delaxes(self)

    def get_children(self):
        return list(self.artists)

    def cla(self):
        """matplotlib's cla: the axes as new, in place (shared axes stay so)."""
        if type(self).__name__ != 'Axes':               # maps, polar: their artists go
            self.artists.clear()
            self.titles.clear()
            self.legend_ = None
            self.containers = []
            return
        keep = dict(figure=self.figure, spec=self.spec, label=self.label)
        twin = getattr(self, '_twin_of', None)
        shares = {}
        for n in ('x', 'y'):
            axis = getattr(self, n + 'axis', None)
            if axis is not None:
                shares[n] = [a.axes for a in axis.shared if a.axes is not self]
        zorder = self.zorder
        self.__init__(keep['figure'], keep['spec'], keep['label'])
        self.zorder = zorder
        if twin is not None:
            self._twin_of = twin
        for n, others in shares.items():
            if others:
                getattr(self, 'share' + n)(others[0])

    clear = cla

    def set_prop_cycle(self, *args, **kw):
        """Colours from a cycler, a list, or color=[...]: the axes' own cycle."""
        from ._rc import _cycle
        if args and args[0] is None:
            self._prop_cycle = None
        else:
            v = args[0] if len(args) == 1 else (kw.get('color') if 'color' in kw else
                                                 (args[1] if len(args) == 2 else None))
            self._prop_cycle = list(_cycle(v)) if v is not None else None
        self._cycle = {}

    def get_title(self, loc='center'):
        return self.titles.get(loc, {}).get('text', '')

    def set_facecolor(self, c):
        self.facecolor = c

    def set_zorder(self, z):
        self.zorder = z

    def get_zorder(self):
        return self.zorder

    def set_anchor(self, a):
        self.anchor = a

    def set_frame_on(self, b):
        self.frame_on = bool(b)

    def set_box_aspect(self, a):
        self.box_aspect = a

    def get_position(self, original=False):
        from ._layout import spec_box
        pos = getattr(self, '_pos', None)
        if pos is None or original:
            pos = spec_box(self, self.figure)
        return pos

    def set_position(self, rect, which='both'):
        if hasattr(rect, 'bounds'):
            rect = rect.bounds
        self.spec = _Box(*rect)

    def axis(self, arg=None):
        if arg == 'off':
            self.frame_on = False
            self.patch_visible = False

    def set_axis_off(self):
        self.axis('off')

    def _add(self, artist):
        touch = getattr(self.figure, '_touch', None)
        if touch is not None:
            touch()
        self.artists.append(artist)
        return artist

    # -- artists made elsewhere (fastplot.patches, ...)
    def add_patch(self, p):
        """matplotlib's add_patch: a fastplot.patches shape, in data
        coordinates unless its transform says otherwise."""
        p.axes = self
        p.src = self._src(p.transform)
        return self._add(p)

    def add_collection(self, c, autolim=True):
        """matplotlib's add_collection: a fastplot.collections collection."""
        c.axes = self
        c.src = self._src(getattr(c, 'transform', None))
        c.requests_autoscale = bool(autolim)
        return self._add(c)

    def add_artist(self, a):
        if hasattr(a, 'get_path') or hasattr(a, 'posA'):
            return self.add_patch(a)
        if type(a).__name__ == 'Legend':
            self.legend_ = a
            return a
        a.axes = self
        return self._add(a)

    # -- legend
    def legend(self, *args, **kw):
        """matplotlib's: legend(), legend(labels), legend(handles, labels)."""
        from ._legend import Legend, parse_args
        handles, labels = parse_args(args, kw, [self])
        touch = getattr(self.figure, '_touch', None)
        if touch is not None:
            touch()
        self.legend_ = Legend(self, handles, labels, **kw)
        return self.legend_

    def get_legend(self):
        return self.legend_

    def get_legend_handles_labels(self):
        from ._legend import handles_and_labels
        return handles_and_labels([self])

    def _xy(self, x, y):
        """Data coordinates as given (plain axes: dates to day numbers)."""
        return x, y

    def pcolormesh(self, *args, cmap=None, norm=None, vmin=None, vmax=None, shading='auto',
                   transform=None, alpha=None, zorder=None, **kw):
        X, Y, C = _xyz(args)
        ec = kw.pop('edgecolors', kw.pop('edgecolor', kw.pop('ec', None)))
        elw = kw.pop('linewidth', kw.pop('linewidths', kw.pop('lw', None)))
        label = kw.pop('label', None)
        _ignore(kw, 'pcolormesh', ('antialiased', 'antialiaseds', 'rasterized', 'snap',
                                   'clip_on', 'linestyle', 'linestyles'))
        src = self._src(transform)
        C = _2d(C)
        X, Y = self._xy(_as_host(X), _as_host(Y))
        shading = (shading or 'auto').lower()
        flat_size = np.shape(X)[-1] == C.shape[1] + 1 and np.shape(Y)[0] == C.shape[0] + 1
        if shading == 'auto':
            shading = 'flat' if flat_size else 'nearest'
        if shading == 'flat' and not flat_size:
            shading = 'nearest'                         # matplotlib drops the last row/col; close enough
        if shading == 'gouraud' and src == 'data' and hasattr(self, 'xaxis'):
            return self._gouraud_mesh(X, Y, C, cmap, norm, vmin, vmax, alpha, zorder)
        if shading == 'gouraud':
            grid = _Grid(X, Y, C.shape, src)
            mode = _raster.BILINEAR
        elif shading == 'flat':
            grid = _Grid(X, Y, C.shape, src, edges=True)
            mode = _raster.NEAREST if grid.rect else _raster.FLAT
        else:
            grid = _Grid(X, Y, C.shape, src)
            mode = _raster.NEAREST
        fa = FieldArtist(self, grid, C, mode, cmap, norm, vmin, vmax, alpha, zorder)
        fa.label = label
        if ec is not None and not (isinstance(ec, str) and ec.lower() in ('none', 'face')):
            fa.edges = (ec, rc('patch.linewidth') if elw is None else float(np.ravel(elw)[0]),
                        shading == 'flat')
        return self._add(fa)

    def _gouraud_mesh(self, X, Y, C, cmap, norm, vmin, vmax, alpha, zorder):
        """matplotlib's gouraud QuadMesh: each quad four triangles about its
        centre (the corners' mean), the colour interpolated across each."""
        from ._tri import TriPcolor, Triangulation
        ny, nx = C.shape
        if np.ndim(X) == 1:
            X, Y = np.meshgrid(np.asarray(X, float), np.asarray(Y, float))
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        Z = np.ma.filled(np.ma.asarray(C, np.float64), np.nan)
        quad = lambda a: 0.25 * (a[:-1, :-1] + a[:-1, 1:] + a[1:, 1:] + a[1:, :-1])  # noqa: E731
        px = np.r_[X.ravel(), quad(X).ravel()]
        py = np.r_[Y.ravel(), quad(Y).ravel()]
        pz = np.r_[Z.ravel(), quad(Z).ravel()]
        i, j = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing='ij')
        a, b = (i * nx + j).ravel(), (i * nx + j + 1).ravel()
        c, d = ((i + 1) * nx + j + 1).ravel(), ((i + 1) * nx + j).ravel()
        m = ny * nx + (i * (nx - 1) + j).ravel()
        tris = np.concatenate([np.c_[a, b, m], np.c_[b, c, m], np.c_[c, d, m], np.c_[d, a, m]])
        tri = Triangulation(px, py, tris)
        tp = TriPcolor(self, tri, pz, 'data', 'gouraud', cmap, norm, vmin, vmax, alpha, zorder)
        tp.sticky = True
        tp.centres = (ny * nx, a, b, c, d)              # QuadMesh: the corners' colours' mean
        return self._add(tp)

    pcolor = pcolormesh

    def imshow(self, Z, extent=None, origin='upper', cmap=None, norm=None, vmin=None, vmax=None,
               transform=None, alpha=None, zorder=None, interpolation=None, **kw):
        aspect = kw.pop('aspect', None)
        _ignore(kw, 'imshow', ('resample', 'regrid_shape', 'interpolation_stage', 'filternorm',
                               'filterrad'))
        Z = _field(Z) if np.ndim(Z) == 2 else np.asarray(Z)
        src = self._src(transform)
        if extent is None:
            extent = self._default_extent(src, Z.shape)
        if hasattr(self, 'set_aspect') and src == 'data':   # plain axes: rc image.aspect,
            self.set_aspect(rc('image.aspect') if aspect is None else aspect)
            self._set_lim('x', extent[:2], auto=None)       # the limits the extent's (its
            self._set_lim('y', extent[2:], auto=None)       # order: origin 'upper' inverts)
        ny, nx = Z.shape[:2]
        x0, x1, y0, y1 = extent
        xe = np.linspace(x0, x1, nx + 1)
        ye = np.linspace(y0, y1, ny + 1) if origin == 'lower' else np.linspace(y1, y0, ny + 1)
        grid = _Grid(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), (ny, nx), src)
        if Z.ndim == 3:
            from ._fields import RGBImage
            return self._add(RGBImage(self, Z, extent, origin, alpha, zorder,
                                      grid if src != 'data' else None))
        if interpolation is None:
            interpolation = rc('image.interpolation')
        # the smoothing filters (bicubic, spline16, gaussian ...) as bilinear
        mode = _raster.NEAREST if interpolation in (None, 'none', 'nearest', 'antialiased',
                                                    'auto', 'data', 'rgba') \
            else _raster.IMAGE
        fa = FieldArtist(self, grid, Z, mode, cmap, norm, vmin, vmax, alpha, zorder,
                         rounding='floor')
        if src == 'data':
            fa.extent = tuple(extent)                   # its limits, sampled how it may be
        return self._add(fa)

    def contourf(self, *args, levels=None, cmap=None, colors=None, norm=None, vmin=None,
                 vmax=None, extend='neither', transform=None, alpha=None, zorder=None,
                 transform_first=None, **kw):
        hatches = kw.pop('hatches', None)
        origin, extent, locator = kw.pop('origin', None), kw.pop('extent', None), \
            kw.pop('locator', None)
        _ignore(kw, 'contourf', ('antialiased', 'algorithm', 'corner_mask', 'nchunk',
                                 'rasterized', 'label'))
        grid, Z, levels = self._contour_args(args, levels, transform, origin, extent, locator)
        if norm is None and locator is not None and 'Log' in type(locator).__name__:
            norm = 'log'                                # matplotlib: a log locator, a log norm
        cf = ContourfArtist(self, grid, Z, levels, cmap, colors, norm, vmin, vmax, extend, alpha,
                            zorder)
        cf.hatches = None if hatches is None else list(hatches)
        return self._add(cf)

    def contour(self, *args, levels=None, colors=None, cmap=None, norm=None, vmin=None,
                vmax=None, linewidths=None, linestyles=None, negative_linestyles=None,
                transform=None, alpha=None, zorder=None, transform_first=None, **kw):
        linewidths = kw.pop('linewidth', linewidths)
        linestyles = kw.pop('linestyle', linestyles)
        origin, extent, locator = kw.pop('origin', None), kw.pop('extent', None), \
            kw.pop('locator', None)
        _ignore(kw, 'contour', ('antialiased', 'algorithm', 'corner_mask', 'extend',
                                'nchunk', 'rasterized', 'label'))
        grid, Z, levels = self._contour_args(args, levels, transform, origin, extent, locator)
        if norm is None and locator is not None and 'Log' in type(locator).__name__:
            norm = 'log'
        return self._add(ContourSet(self, grid, Z, levels, colors, cmap, norm, vmin, vmax,
                                    linewidths, linestyles, negative_linestyles or
                                    rc('contour.negative_linestyle'), alpha, zorder))

    def _contour_args(self, args, levels, transform, origin=None, extent=None, locator=None):
        """contour/contourf's (Z), (Z, levels), (X, Y, Z) or (X, Y, Z, levels):
        the grid, the field and the levels. A contour set first stands for
        its grid, field and levels (matplotlib's contour(cs, ...)); Z alone
        is placed by origin / extent as an image is."""
        args = list(args)
        if args and hasattr(args[0], 'grid') and hasattr(args[0], 'levels'):
            cs = args.pop(0)
            lv = args[0] if args else None
            lv = cs.levels if lv is None and levels is None else lv
            return cs.grid, cs.z, lv if levels is None else levels
        X, Y, Z, lv = _xyz_levels(args)
        Z = _2d(Z)
        if len(args) in (1, 2) and (extent is not None or origin not in (None, 'lower')):
            ny, nx = Z.shape                            # matplotlib: Z as an image
            if extent is not None:
                x0, x1, y0, y1 = extent
                dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
                X = x0 + dx * (np.arange(nx) + 0.5)
                Y = y0 + dy * (np.arange(ny) + 0.5)
            if origin == 'upper':
                Y = np.asarray(Y, float)[::-1]
        X, Y = self._xy(_as_host(X), _as_host(Y))
        grid = _Grid(X, Y, Z.shape, self._src(transform))
        lv = lv if levels is None else levels
        if lv is None and locator is not None:          # contourf(locator=LogLocator())
            zf = np.asarray(_as_host(Z), float)
            fin = zf[np.isfinite(zf)]
            if hasattr(locator, 'tick_values') and fin.size:
                lo = fin[fin > 0].min() if 'Log' in type(locator).__name__ and \
                    (fin > 0).any() else fin.min()
                lv = np.asarray(locator.tick_values(lo, fin.max()), float)
                inside = (lv >= lo) & (lv <= fin.max())
                k = np.nonzero(inside)[0]
                if k.size:                              # matplotlib keeps one beyond each end
                    lv = lv[max(k[0] - 1, 0):min(k[-1] + 2, lv.size)]
        if lv is not None and np.ndim(lv) and len(lv) > 1 and np.any(np.diff(lv) <= 0):
            raise ValueError('Contour levels must be increasing')
        return grid, Z, lv

    def clabel(self, cs, levels=None, fontsize=None, inline=True, inline_spacing=5, fmt=None,
               colors=None, rightside_up=True, use_clabeltext=False, zorder=None, **kw):
        lv = cs.levels if levels is None else np.asarray(levels, float)
        idx = [int(np.argmin(np.abs(cs.levels - v))) for v in np.ravel(lv)]
        labels = _ticker.format_values(fmt, cs.levels[idx])
        texts = dict(zip(idx, labels))
        if colors is None:
            cols = {k: cs.line_colors[k] for k in range(cs.levels.size)}
        else:
            cl = [colors] if isinstance(colors, str) else list(colors)
            cols = {k: rgba(cl[n % len(cl)]) for n, k in enumerate(idx)}
        cs.labels = dict(levels=set(idx), texts=texts, fontsize=fontsize_pt(fontsize),
                         inline=bool(inline), spacing=float(inline_spacing), colors=cols,
                         rightside_up=rightside_up,
                         key=(tuple(idx), tuple(labels), fontsize, inline, inline_spacing))
        cs._cache = None
        return [texts[k] for k in idx]

    def barbs(self, *args, transform=None, **kw):
        if len(args) == 4:
            x, y, u, v = args
        elif len(args) == 5:
            x, y, u, v, _c = args
        else:
            raise TypeError('barbs(x, y, u, v)')
        src = self._src(transform)
        kw2 = {k: kw.pop(k) for k in list(kw) if k in (
            'length', 'pivot', 'barbcolor', 'flagcolor', 'color', 'sizes', 'fill_empty',
            'rounding', 'barb_increments', 'flip_barb', 'linewidth', 'alpha', 'zorder',
            'regrid_shape')}
        kw.pop('clip_on', None)
        if 'lw' in kw:
            kw2['linewidth'] = kw.pop('lw')
        if 'linewidths' in kw:
            kw2['linewidth'] = kw.pop('linewidths')
        _ignore(kw, 'barbs', ('label', 'rasterized', 'antialiased'))
        x, y = self._xy(x, y)
        return self._add(Barbs(self, x, y, u, v, src, **kw2))

    def quiver(self, *args, transform=None, **kw):
        """matplotlib's quiver([X, Y], U, V, [C]): arrows (cartopy's on maps:
        transform the vectors' crs, turned with the map)."""
        from ._quiver import Quiver, parse_args
        X, Y, U, V, C = parse_args(args)
        src = self._src(transform)
        if src == 'data':
            X, Y = (np.asarray(v, float) for v in self._xy(X, Y))
        if 'facecolor' in kw or 'fc' in kw:
            kw['color'] = kw.pop('facecolor', kw.pop('fc', None))
        for k in ('regrid_shape', 'target_extent', 'edgecolor', 'ec', 'linewidth', 'linewidths',
                  'lw', 'antialiased', 'rasterized'):     # accepted: drawn without an edge
            kw.pop(k, None)
        return self._add(Quiver(self, X, Y, U, V, C, src, **kw))

    # -- triangular grids (matplotlib.tri)
    def _tri_args(self, args, kw):
        from ._tri import Triangulation
        tri, args, kw = Triangulation.get_from_args_and_kwargs(*args, **kw)
        return tri, list(args), kw, self._src(kw.pop('transform', None))

    def tricontourf(self, *args, levels=None, cmap=None, colors=None, norm=None, vmin=None,
                    vmax=None, extend='neither', alpha=None, zorder=None, hatches=None, **kw):
        """matplotlib's tricontourf([triangulation | x, y[, triangles]], z[, levels])."""
        from ._tri import TriContourf
        tri, args, kw, src = self._tri_args(args, kw)
        _ignore(kw, 'tricontourf', ('antialiased', 'rasterized'))
        z = args.pop(0)
        if args:
            levels = args.pop(0)
        cf = TriContourf(self, tri, z, src, levels, cmap=cmap, colors=colors, norm=norm,
                         vmin=vmin, vmax=vmax, extend=extend, alpha=alpha, zorder=zorder)
        cf.hatches = None if hatches is None else list(hatches)
        return self._add(cf)

    def tricontour(self, *args, levels=None, colors=None, cmap=None, norm=None, vmin=None,
                   vmax=None, linewidths=None, linestyles=None, negative_linestyles=None,
                   alpha=None, zorder=None, **kw):
        """matplotlib's tricontour: contour lines on a triangulation."""
        from ._tri import TriContour
        tri, args, kw, src = self._tri_args(args, kw)
        _ignore(kw, 'tricontour', ('antialiased', 'rasterized', 'extend'))
        z = args.pop(0)
        if args:
            levels = args.pop(0)
        neg = rc('contour.negative_linestyle') if negative_linestyles is None \
            else negative_linestyles
        return self._add(TriContour(
            self, tri, z, src, levels, colors=colors, cmap=cmap, norm=norm, vmin=vmin,
            vmax=vmax, linewidths=linewidths, linestyles=linestyles, negative_linestyles=neg,
            alpha=alpha, zorder=zorder))

    def tripcolor(self, *args, shading='flat', cmap=None, norm=None, vmin=None, vmax=None,
                  alpha=None, zorder=None, facecolors=None, **kw):
        """matplotlib's tripcolor: flat or gouraud colours on a triangulation."""
        from ._tri import TriPcolor
        tri, args, kw, src = self._tri_args(args, kw)
        _ignore(kw, 'tripcolor', ('antialiased', 'rasterized', 'edgecolors', 'edgecolor',
                                  'linewidth', 'linewidths'))
        c = facecolors if facecolors is not None else args.pop(0)
        return self._add(TriPcolor(self, tri, c, src, shading, cmap, norm, vmin, vmax, alpha,
                                   zorder))

    def triplot(self, *args, **kw):
        """matplotlib's triplot: the triangles' edges (one line, zorder 1)
        and their points' markers: [lines, markers]."""
        from ._tri import Triangulation
        tri, args, kw = Triangulation.get_from_args_and_kwargs(*args, **kw)
        fmt = args[0] if args else ''
        c, m, ls = _parse_fmt(fmt) if fmt else (None, None, None)
        for key, val in (('linestyle', ls), ('marker', m), ('color', c)):
            if val is not None:
                kw.setdefault(key, val)
        linestyle = kw.get('linestyle', '-')
        marker = kw.get('marker', 'None')
        e = tri.edges
        kl = dict(kw, marker='None', zorder=kw.get('zorder', 1))
        if linestyle in (None, 'None', '', ' '):
            lines = self.plot([], [], **kl)
        else:
            xs = np.insert(tri.x[e], 2, np.nan, axis=1).ravel()
            ys = np.insert(tri.y[e], 2, np.nan, axis=1).ravel()
            lines = self.plot(xs, ys, **kl)
        km = dict(kw, linestyle='None')
        km.pop('label', None)
        if marker in (None, 'None', '', ' '):
            markers = self.plot([], [], **km)
        else:
            markers = self.plot(tri.x, tri.y, **km)
        return lines + markers

    def quiverkey(self, Q, X, Y, U, label, **kw):
        """matplotlib's quiverkey: a key arrow of length U for Quiver Q."""
        from ._quiver import QuiverKey
        return self._add(QuiverKey(Q, X, Y, U, label, **kw))

    def streamplot(self, x, y, u, v, **kw):
        """matplotlib's streamplot (fastplot._streamplot)."""
        from ._streamplot import streamplot
        return streamplot(self, x, y, u, v, **kw)

    def _next_color(self, which='lines'):
        """The axes' next default colour: matplotlib keeps one cycle for
        lines and one for patches (scatter, bars, fills)."""
        k = self._cycle.get(which, 0)
        self._cycle[which] = k + 1
        own = getattr(self, '_prop_cycle', None)
        if own:
            return own[k % len(own)]
        return cycle_color(k)

    def plot(self, *args, transform=None, color=None, linewidth=None, linestyle=None,
             marker=None, markersize=None, alpha=None, zorder=None, label=None, scalex=True,
             scaley=True, data=None, **kw):
        """matplotlib's plot: (x, y[, fmt]) or (y[, fmt]) groups, a 2-D y as
        a line per column; returns the list of lines."""
        color = kw.pop('c', color)
        linewidth = kw.pop('lw', linewidth)
        linestyle = kw.pop('ls', linestyle)
        markersize = kw.pop('ms', markersize)
        mfc = kw.pop('markerfacecolor', kw.pop('mfc', None))
        mec = kw.pop('markeredgecolor', kw.pop('mec', None))
        mew = kw.pop('markeredgewidth', kw.pop('mew', None))
        drawstyle = kw.pop('drawstyle', kw.pop('ds', None))
        markevery = kw.pop('markevery', None)
        fillstyle = kw.pop('fillstyle', None)
        mfcalt = kw.pop('markerfacecoloralt', kw.pop('mfcalt', 'none'))
        dashes = kw.pop('dashes', None)
        visible = kw.pop('visible', True)
        if dashes is not None and linestyle is None:
            linestyle = (0, tuple(dashes))
        _ignore(kw, 'plot', ('solid_capstyle', 'solid_joinstyle', 'path_effects', 'rasterized',
                             'dash_capstyle', 'dash_joinstyle', 'antialiased', 'aa', 'clip_on',
                             'gid', 'url', 'picker', 'pickradius', 'snap', 'animated',
                             'in_layout', 'sketch_params', 'clip_box', 'clip_path'))
        if data is not None:                            # plot('x', 'y', data=d)
            args = tuple(data[a] if isinstance(a, str) and a in data else a for a in args)
        src = self._src(transform)
        lines = []
        labels = label if isinstance(label, (list, tuple)) else None
        for x, y, fmt in _plot_groups(args):
            c, m, ls = _parse_fmt(fmt) if fmt else (None, None, None)
            ys = y if isinstance(y, np.ma.MaskedArray) else np.asarray(_as_host(y))
            cols = [ys] if ys.ndim < 2 else [ys[:, k] for k in range(ys.shape[1])]
            xs = None if x is None else (x if isinstance(x, np.ma.MaskedArray)
                                         else np.asarray(_as_host(x)) if not isinstance(x, (
                                             list, tuple)) or np.ndim(x) > 1 else x)
            if xs is not None and np.ndim(xs) == 2 and np.shape(xs)[1] > 1 and ys.ndim < 2:
                cols = [ys] * np.shape(xs)[1]
            for k, yk in enumerate(cols):
                if xs is not None and np.ndim(xs) == 2:
                    xk = xs[:, min(k, np.shape(xs)[1] - 1)]
                else:
                    xk = np.arange(len(yk)) if x is None else x
                if src == 'data':
                    xk, yk = self._xy(xk, yk)
                col = color or c or self._next_color('lines')
                mk = marker or m
                # a format's marker alone means no line; a marker keyword does not
                style = linestyle or ls or ('None' if m and not ls else rc('lines.linestyle'))
                lab = label if labels is None else (labels[len(lines)] if len(lines) < len(labels)
                                                    else None)
                ln = Line2D(self, xk, yk, src, col, rc('lines.linewidth') if linewidth is None
                            else linewidth, style, mk,
                            rc('lines.markersize') if markersize is None else markersize, mfc,
                            mec, rc('lines.markeredgewidth') if mew is None else mew, alpha,
                            zorder, lab, fillstyle or 'full', mfcalt)
                ln.drawstyle = drawstyle or 'default'
                ln.markevery = markevery
                ln.visible = bool(visible)
                lines.append(self._add(ln))
        return lines

    def add_line(self, line):
        """matplotlib's add_line: a Line2D (fastplot.lines) on this axes."""
        line.axes = self
        line.src = self._src(getattr(line, 'transform', None))
        if line.color is None:
            line.color = self._next_color('lines')
            line.mfc = line.mfc if line.mfc is not None else line.color
            line.mec = line.mec if line.mec is not None else line.color
        return self._add(line)

    def scatter(self, x, y, s=None, c=None, marker='o', cmap=None, norm=None, vmin=None,
                vmax=None, alpha=None, linewidths=None, edgecolors='face', transform=None,
                zorder=None, color=None, **kw):
        label = kw.pop('label', None)
        plotnonfinite = kw.pop('plotnonfinite', False)
        for k in ('facecolor', 'facecolors', 'fc'):     # matplotlib's aliases of c / color
            if k in kw:
                v = kw.pop(k)
                if c is None and color is None:
                    color = v
        for k in ('edgecolor', 'ec'):
            if k in kw:
                edgecolors = kw.pop(k)
        for k in ('linewidth', 'lw'):
            if k in kw:
                linewidths = kw.pop(k)
        _ignore(kw, 'scatter', ('rasterized', 'clip_on', 'antialiased', 'antialiaseds',
                                'snap', 'gid', 'url', 'picker'))
        if marker is None:
            marker = rc('scatter.marker')
        src = self._src(transform)
        x, y = (np.ma.filled(np.ma.asarray(v, float), np.nan)
                if isinstance(v, np.ma.MaskedArray) else v for v in (x, y))
        if src == 'data':
            x, y = self._xy(x, y)
        if c is None and color is None:
            color = self._next_color('patches')
        alphas = None
        if alpha is not None and np.ndim(alpha) > 0:   # matplotlib: an alpha per point
            alphas, alpha = np.asarray(alpha, float).ravel(), None
        sc = self._add(Scatter(self, x, y, src, rc('lines.markersize') ** 2 if s is None else s,
                               c, color, marker, cmap, norm, vmin, vmax, edgecolors,
                               rc('lines.linewidth') if linewidths is None else linewidths,
                               alpha, zorder, plotnonfinite))
        sc.alphas = alphas
        sc.label = label
        return sc

    def text(self, x, y, s, transform=None, **kw):
        src = self._src(transform)
        if src == 'data' and hasattr(self, '_xy'):         # dates, categories: as numbers
            x, y = (np.ravel(v)[0] if np.ndim(v) else v for v in self._xy(x, y))
        return self._add(Text(self, x, y, s, src, **_text_kw(kw)))

    def annotate(self, text, xy, xytext=None, xycoords='data', textcoords=None,
                 arrowprops=None, annotation_clip=None, **kw):
        """matplotlib's annotate: text at xytext, an arrow to xy."""
        from ._annotation import Annotation
        return self._add(Annotation(self, text, xy, xytext, xycoords, textcoords, arrowprops,
                                    annotation_clip, **_text_kw(kw)))

    def _draw_titles(self, r):
        x0, y0, x1, y1 = self.box_f
        top = y0 if self._top_extent is None else min(y0, self._top_extent)
        placed = []
        for loc, t in self.titles.items():
            if not t['text']:
                continue
            pad = r.px(rc('axes.titlepad') if t['pad'] is None else t['pad'])
            ty = t['y'] if t['y'] is not None else getattr(self, '_title_y', None)
            x, ha = {'center': (0.5 * (x0 + x1), 'center'), 'left': (x0, 'left'),
                     'right': (x1, 'right')}[loc]
            size = fontsize_pt(rc('axes.titlesize') if t['fontsize'] is None else t['fontsize'])
            if ty is not None:
                y = y1 - ty * (y1 - y0) - pad
            else:
                y = top - pad
                # matplotlib's _update_title_position: a title reaching below
                # the top (a deep mathtext) is raised until its bottom is on it
                e = r.measure(t['text'], x, y, size, ha=ha, va='baseline', weight=t['weight'])
                if e is not None and e[3] > top:
                    y -= e[3] - top
            placed.append((t, x, y, ha, size, ty is None))
        auto = [p[2] for p in placed if p[5]]
        for t, x, y, ha, size, is_auto in placed:          # and all three at the highest
            r.text(t['text'], x, min(auto) if is_auto else y, size, t['color'], ha=ha,
                   va='baseline', weight=t['weight'], style=t['style'])

    def work(self):
        return sum(a.work() for a in self.artists)


def _2d(C):
    C = _field(C)
    if C.ndim != 2:
        raise ValueError('the field must be 2-D, not %s' % (C.shape,))
    return C


def _xyz(args):
    if len(args) == 1:
        C = args[0]
        ny, nx = np.shape(C)
        return np.arange(nx + 1), np.arange(ny + 1), C
    if len(args) == 3:
        return args
    raise TypeError('expected (C) or (X, Y, C)')


def _xyz_levels(args):
    if len(args) in (1, 2):
        Z = args[0]
        ny, nx = np.shape(Z)
        return np.arange(nx), np.arange(ny), Z, (args[1] if len(args) == 2 else None)
    if len(args) in (3, 4):
        return args[0], args[1], args[2], (args[3] if len(args) == 4 else None)
    raise TypeError('expected (Z), (Z, levels), (X, Y, Z) or (X, Y, Z, levels)')


def _plot_groups(args):
    """plot's positional arguments as (x or None, y, fmt or None) groups."""
    out = []
    a = list(args)
    while a:
        first = a.pop(0)
        if a and not isinstance(a[0], str):
            x, y = first, a.pop(0)
        else:
            x, y = None, first
        fmt = a.pop(0) if a and isinstance(a[0], str) else None
        out.append((x, y, fmt))
    return out


def _parse_fmt(fmt):
    """matplotlib's _process_plot_format: (color, marker, linestyle) of a
    format string such as 'ro--', 'C1s', 'k:' or a whole colour ('0.5')."""
    from ._color import is_color_like
    from ._markers import MARKERS
    if fmt and len(fmt) > 1 and is_color_like(fmt) and not any(
            fmt.startswith(ls) for ls in ('--', '-.', '-', ':')):
        if not (len(fmt) == 2 and fmt[0] in 'bgrcmykw' and fmt[1] in MARKERS):
            return fmt, None, None
    c = m = ls = None
    i = 0
    while i < len(fmt):
        two = fmt[i:i + 2]
        if two in ('--', '-.'):
            ls, i = two, i + 2
        elif fmt[i] in '-:':
            ls, i = fmt[i], i + 1
        elif fmt[i] == 'C' and i + 1 < len(fmt) and fmt[i + 1].isdigit():
            j = i + 1
            while j < len(fmt) and fmt[j].isdigit():
                j += 1
            c, i = fmt[i:j], j
        elif fmt[i] in 'bgrcmykw' and c is None:
            c, i = fmt[i], i + 1
        elif fmt[i] in MARKERS:
            m, i = fmt[i], i + 1
        else:
            raise ValueError('unrecognized character %r in format string %r' % (fmt[i], fmt))
    return c, m, ls