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


class AxesBase:
    """An axes: its place (spec), titles, background, frame, artists."""

    _snap = False                   # pixel snapping of rectilinear paths (plain axes: 'auto')
    _agg = False                    # matplotlib's caps (plain axes)

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

    def get_figure(self):
        return self.figure

    def get_facecolor(self):
        return self.facecolor

    def set_title(self, label, loc='center', fontsize=None, fontweight='normal', color='black',
                  pad=None, y=None, fontdict=None, **kw):
        if fontdict:
            fontsize = fontdict.get('fontsize', fontsize)
            fontweight = fontdict.get('fontweight', fontweight)
            color = fontdict.get('color', color)
        fontsize = kw.pop('size', fontsize)
        fontweight = kw.pop('weight', fontweight)
        self.titles[loc] = dict(text=label, fontsize=fontsize, weight=fontweight, color=color,
                                pad=pad, y=y, style=kw.pop('fontstyle', 'normal'))

    def get_title(self, loc='center'):
        return self.titles.get(loc, {}).get('text', '')

    def set_facecolor(self, c):
        self.facecolor = c

    def set_anchor(self, a):
        self.anchor = a

    def set_frame_on(self, b):
        self.frame_on = bool(b)

    def set_box_aspect(self, a):
        self.box_aspect = a

    def get_position(self):
        return self._pos

    def set_position(self, rect):
        self.spec = _Box(*rect)

    def axis(self, arg=None):
        if arg == 'off':
            self.frame_on = False
            self.patch_visible = False

    def set_axis_off(self):
        self.axis('off')

    def _add(self, artist):
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
        _ignore(kw, 'pcolormesh', ('antialiased', 'rasterized', 'edgecolors', 'linewidth',
                                   'snap', 'label'))
        src = self._src(transform)
        C = _2d(C)
        X, Y = self._xy(_as_host(X), _as_host(Y))
        shading = (shading or 'auto').lower()
        flat_size = np.shape(X)[-1] == C.shape[1] + 1 and np.shape(Y)[0] == C.shape[0] + 1
        if shading == 'auto':
            shading = 'flat' if flat_size else 'nearest'
        if shading == 'flat' and not flat_size:
            shading = 'nearest'                         # matplotlib drops the last row/col; close enough
        if shading == 'gouraud':
            grid = _Grid(X, Y, C.shape, src)
            mode = _raster.BILINEAR
        elif shading == 'flat':
            grid = _Grid(X, Y, C.shape, src, edges=True)
            mode = _raster.NEAREST if grid.rect else _raster.FLAT
        else:
            grid = _Grid(X, Y, C.shape, src)
            mode = _raster.NEAREST
        return self._add(FieldArtist(self, grid, C, mode, cmap, norm, vmin, vmax, alpha, zorder))

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
        if Z.ndim == 3:
            from ._fields import RGBImage
            return self._add(RGBImage(self, Z, extent, origin, alpha, zorder))
        ny, nx = Z.shape[:2]
        x0, x1, y0, y1 = extent
        xe = np.linspace(x0, x1, nx + 1)
        ye = np.linspace(y0, y1, ny + 1) if origin == 'lower' else np.linspace(y1, y0, ny + 1)
        grid = _Grid(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), Z.shape, src)
        mode = _raster.BILINEAR if interpolation in ('bilinear',) else _raster.NEAREST
        return self._add(FieldArtist(self, grid, Z, mode, cmap, norm, vmin, vmax, alpha, zorder,
                                     rounding='floor'))

    def contourf(self, *args, levels=None, cmap=None, colors=None, norm=None, vmin=None,
                 vmax=None, extend='neither', transform=None, alpha=None, zorder=None,
                 transform_first=None, **kw):
        hatches = kw.pop('hatches', None)
        _ignore(kw, 'contourf', ('antialiased', 'algorithm', 'corner_mask', 'nchunk',
                                 'rasterized'))
        grid, Z, levels = self._contour_args(args, levels, transform)
        cf = ContourfArtist(self, grid, Z, levels, cmap, colors, norm, vmin, vmax, extend, alpha,
                            zorder)
        cf.hatches = None if hatches is None else list(hatches)
        return self._add(cf)

    def contour(self, *args, levels=None, colors=None, cmap=None, norm=None, vmin=None,
                vmax=None, linewidths=None, linestyles=None, negative_linestyles=None,
                transform=None, alpha=None, zorder=None, transform_first=None, **kw):
        linewidths = kw.pop('linewidth', linewidths)
        _ignore(kw, 'contour', ('antialiased', 'algorithm', 'corner_mask', 'extend',
                                'nchunk', 'rasterized'))
        grid, Z, levels = self._contour_args(args, levels, transform)
        return self._add(ContourSet(self, grid, Z, levels, colors, cmap, norm, vmin, vmax,
                                    linewidths, linestyles, negative_linestyles or 'dashed',
                                    alpha, zorder))

    def _contour_args(self, args, levels, transform):
        """contour/contourf's (Z), (Z, levels), (X, Y, Z) or (X, Y, Z, levels):
        the grid, the field and the levels."""
        X, Y, Z, lv = _xyz_levels(args)
        Z = _2d(Z)
        X, Y = self._xy(_as_host(X), _as_host(Y))
        grid = _Grid(X, Y, Z.shape, self._src(transform))
        return grid, Z, lv if levels is None else levels

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
        _ignore(kw, 'plot', ('solid_capstyle', 'solid_joinstyle', 'path_effects', 'rasterized',
                             'dash_capstyle', 'dash_joinstyle', 'antialiased', 'drawstyle',
                             'fillstyle', 'markevery', 'clip_on'))
        src = self._src(transform)
        lines = []
        for x, y, fmt in _plot_groups(args):
            c, m, ls = _parse_fmt(fmt) if fmt else (None, None, None)
            ys = np.asarray(_as_host(y))
            cols = [ys] if ys.ndim < 2 else [ys[:, k] for k in range(ys.shape[1])]
            for yk in cols:
                xk = np.arange(len(yk)) if x is None else x
                if src == 'data':
                    xk, yk = self._xy(xk, yk)
                col = color or c or self._next_color('lines')
                mk = marker or m
                style = linestyle or ls or ('None' if mk and not ls else '-')
                lines.append(self._add(Line2D(
                    self, xk, yk, src, col, rc('lines.linewidth') if linewidth is None
                    else linewidth, style, mk,
                    rc('lines.markersize') if markersize is None else markersize, mfc, mec,
                    rc('lines.markeredgewidth') if mew is None else mew, alpha, zorder, label)))
        return lines

    def scatter(self, x, y, s=None, c=None, marker='o', cmap=None, norm=None, vmin=None,
                vmax=None, alpha=None, linewidths=None, edgecolors='face', transform=None,
                zorder=None, color=None, **kw):
        label = kw.pop('label', None)
        _ignore(kw, 'scatter', ('rasterized', 'plotnonfinite'))
        src = self._src(transform)
        if src == 'data':
            x, y = self._xy(x, y)
        if c is None and color is None:
            color = self._next_color('patches')
        sc = self._add(Scatter(self, x, y, src, rc('lines.markersize') ** 2 if s is None else s,
                               c, color, marker, cmap, norm, vmin, vmax, edgecolors,
                               rc('lines.linewidth') if linewidths is None else linewidths,
                               alpha, zorder))
        sc.label = label
        return sc

    def text(self, x, y, s, transform=None, **kw):
        return self._add(Text(self, x, y, s, self._src(transform), **_text_kw(kw)))

    def annotate(self, text, xy, xytext=None, xycoords='data', textcoords=None,
                 arrowprops=None, annotation_clip=None, **kw):
        """matplotlib's annotate: text at xytext, an arrow to xy."""
        from ._annotation import Annotation
        return self._add(Annotation(self, text, xy, xytext, xycoords, textcoords, arrowprops,
                                    annotation_clip, **_text_kw(kw)))

    def _draw_titles(self, r):
        x0, y0, x1, y1 = self.box_f
        top = y0 if self._top_extent is None else min(y0, self._top_extent)
        for loc, t in self.titles.items():
            if not t['text']:
                continue
            pad = r.px(rc('axes.titlepad') if t['pad'] is None else t['pad'])
            ty = t['y'] if t['y'] is not None else getattr(self, '_title_y', None)
            if ty is not None:
                y = y1 - ty * (y1 - y0) - pad
            else:
                y = top - pad
            x, ha = {'center': (0.5 * (x0 + x1), 'center'), 'left': (x0, 'left'),
                     'right': (x1, 'right')}[loc]
            size = fontsize_pt(rc('axes.titlesize') if t['fontsize'] is None else t['fontsize'])
            r.text(t['text'], x, y, size, t['color'], ha=ha, va='baseline', weight=t['weight'],
                   style=t['style'])

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