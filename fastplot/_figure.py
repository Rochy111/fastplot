"""Figures, map axes and their artists: the matplotlib/cartopy API, drawn
by fastplot's kernels.

    fig = fastplot.figure(figsize=(12, 6.5), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection=fastplot.crs.PlateCarree())
    cf = ax.contourf(lon, lat, t2m, levels=..., cmap='turbo', extend='both',
                     transform=fastplot.crs.PlateCarree())
    ax.coastlines(); ax.gridlines(draw_labels=True)
    fig.colorbar(cf, ax=ax, orientation='horizontal', label='...')
    fig.savefig('t2m.png')

Nothing is drawn until savefig() / draw(): the axes keep their artists
(data, styles), and a draw lays the figure out, then paints each axes'
artists in zorder into one canvas, CPU or GPU (_device). Layout follows
matplotlib exactly -- subplot parameters, gridspecs, the shrink of a map
to its aspect, the room a colorbar takes (make_axes_gridspec) -- so a
script ported from matplotlib puts everything in the same place.

What is drawn and how (see _raster, _contour, _draw):
  pcolormesh, imshow   value raster (nearest), coloured by cmap + norm
  contourf             value raster (bilinear), coloured by level band
  contour, clabel      marching squares on the grid, projected, tiled lines
  barbs                matplotlib's barb polygons, filled and stroked
  features, gridlines  Natural Earth / graticule lines, projected, tiled
  plot, scatter, text  lines, markers, glyph masks
Projected geometry (features, gridlines, contour lines) is kept per artist
for the view it was made for, so redrawing a figure -- a new frame of an
animation after set_array() -- only redraws its fields.
"""
import os

import numpy as np

from . import _device, _draw, _hw, _jit, _rc, _text
from ._render import CpuCanvas, scratch
from ._style import _ignore, fontsize_pt, rc, rgba, subplotpars
from ._layout import GridSpec, SubplotSpec, _Box, tight_subplotpars, topmost
from ._artists import Text, _text_kw
from ._colorbar import Colorbar, ColorbarAxes, make_cax
from ._axes import MapAxes
from ._cartesian import Axes
from . import transforms as _transforms


def _nothing(*args, **kw):
    return None


class _NullCanvas:
    """A canvas that paints nothing: a draw that only measures where
    things go (tight_layout)."""
    device = 'cpu'

    def __init__(self, H, W):
        self.shape = (H, W, 4)

    def __getattr__(self, name):
        return _nothing


class _Renderer:
    """One draw: the canvas, the dpi, and the extents drawn (bbox_inches)."""

    def __init__(self, fig, canvas, dpi, W, H):
        self.fig, self.c, self.dpi, self.W, self.H = fig, canvas, dpi, W, H
        self.ox, self.oy = getattr(fig, '_org', (0.0, 0.0))   # the figure's place on the canvas
        self.extents = []
        self.measuring = isinstance(canvas, _NullCanvas)  # extents only, nothing painted

    def px(self, pt):
        return pt * self.dpi / 72.0

    def text(self, s, x, y, size_pt, color, ha='left', va='baseline', rotation=0.0,
             weight='normal', style='normal', rotation_mode='default', bbox=None, alpha=None,
             family=None, record=True, effects=None, ma=None, linespacing=None, clip=None):
        """A text at (x, y) px; clip: the box (x0, y0, x1, y1) it shows in."""
        if s is None or str(s) == '':
            return None
        size = self.px(size_pt)
        if self.measuring:                              # no glyphs
            mask = None
            ext = _text.text_extent(str(s), x, y, size, ha=ha, va=va, rotation=rotation,
                                    rotation_mode=rotation_mode, weight=weight, style=style,
                                    family=family, linespacing=linespacing, ma=ma)
        else:
            mask, x0, y0, ext = _text.place(str(s), x, y, size, ha=ha, va=va, rotation=rotation,
                                            rotation_mode=rotation_mode, weight=weight,
                                            style=style, family=family,
                                            linespacing=linespacing, ma=ma)
        if bbox is not None:
            ext = self._text_box(ext, bbox, size_pt)
        col = rgba(color, alpha)
        if mask is not None and effects:
            self._effects(mask.alpha, x0, y0, col, effects, alpha)
        elif col[3] and mask is not None:
            for a, dx, dy in _text.layers(mask):
                xa, ya = x0 + dx, y0 + dy
                if clip is not None:                    # the part of the mask inside the box
                    j0, i0 = max(clip[0] - xa, 0), max(clip[1] - ya, 0)
                    a = a[i0:max(clip[3] - ya, i0), j0:max(clip[2] - xa, j0)]
                    xa, ya = xa + j0, ya + i0
                if a.size:
                    self.c.mask(a, xa, ya, col)
        if clip is not None and ext is not None:        # what shows of it counts, if any does
            ext = (max(ext[0], clip[0]), max(ext[1], clip[1]), min(ext[2], clip[2]),
                   min(ext[3], clip[3]))
            record = record and ext[0] < ext[2] and ext[1] < ext[3]
        if record:
            self.extents.append(ext)
        return ext

    def _effects(self, alpha_mask, x0, y0, col, effects, alpha):
        """A text's path_effects (fastplot.patheffects), in order: strokes as
        its mask grown by half their width, Normal (and withStroke) the text."""
        from . import patheffects as pe
        for e in effects:
            dx, dy = (int(round(self.px(v))) for v in e.offset)
            if isinstance(e, pe.Stroke):
                grown, pad = _text.grow(alpha_mask, 0.5 * self.px(e.linewidth))
                c = rgba(e.foreground, e.alpha if e.alpha is not None else alpha)
                if c[3]:
                    self.c.mask(grown, x0 - pad + dx, y0 - pad - dy, c)
            if isinstance(e, (pe.Normal, pe.withStroke)) and col[3]:
                self.c.mask(alpha_mask, x0 + dx, y0 - dy, col)

    def _text_box(self, ext, bbox, size_pt, draw=True):
        """A text's bbox patch, as matplotlib's Text.set_bbox: a
        FancyBboxPatch round the text, its boxstyle ('square' by default,
        padded 4 pt; others 0.3 font sizes) scaled by the font size; face
        patch.facecolor, edge black, snapped. Returns its extent (draw
        False: only that)."""
        from ._patches import fancy_bbox
        from . import _path
        bbox = dict(bbox)
        style = bbox.pop('boxstyle', None)
        pad = bbox.pop('pad', None)
        if style is None:
            style = 'square'
            pad = (4.0 if pad is None else pad) / size_pt
        elif pad is None:
            pad = 0.3
        if isinstance(style, str) and 'pad' not in style:
            style += ',pad=%0.2f' % pad
        fs = self.px(size_pt)
        p = fancy_bbox((ext[0], ext[1]), ext[2] - ext[0], ext[3] - ext[1], style,
                       mutation_scale=fs, ydown=True)
        alpha = bbox.get('alpha')
        fill = bbox.get('fill', True)
        fc = rgba(bbox.get('facecolor', bbox.get('fc', bbox.get('color', rc('patch.facecolor')))),
                  alpha)
        ec = rgba(bbox.get('edgecolor', bbox.get('ec', bbox.get('color', rc('patch.edgecolor')))),
                  alpha)
        lw = self.px(bbox.get('linewidth', bbox.get('lw', rc('patch.linewidth'))))
        xy, offs, _ = _path.flatten(p.vertices, p.codes)
        xy = _draw.snap(xy, offs, lw if ec[3] else 0.0, None)
        if not draw:
            pass
        elif fc[3] and fill:
            self.c.polygons(xy, offs, fc, None, 'nonzero')
        if draw and ec[3] and lw > 0:
            self.lines(xy, offs, ec, lw, None, cap=_draw.BUTT)
        return (float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()),
                float(xy[:, 1].max()))

    def measure(self, s, x, y, size_pt, ha='left', va='baseline', rotation=0.0,
                weight='normal', rotation_mode='default'):
        if s is None or str(s) == '':
            return None
        return _text.text_extent(str(s), x, y, self.px(size_pt), ha=ha, va=va,
                                 rotation=rotation, rotation_mode=rotation_mode, weight=weight)

    def lines(self, xy, offsets, color, width_px, clip, dashes_px=None, snap=False,
              cap=_draw.BUTT, simplify=False, join=_draw.JOIN_ROUND, line=False, filled=False):
        """Polylines in one style; snap: matplotlib's pixel snapping (None:
        'auto', rectilinear parts only); cap, join: _draw's styles; simplify:
        matplotlib's path simplification of a path of 128 points or more
        (rcParams path.simplify[_threshold]), after the snapping, as Agg.
        line: the parts are one Line2D's, one open path; else each part is
        a path of its own, closed if it ends where it started. filled: the
        edge of a filled path.
        A measuring draw (tight_layout) paints nothing, so it skips the work."""
        if self.measuring:
            return
        xy = _draw.snap(xy, offsets, width_px, snap)
        if simplify and len(xy) >= 128 and rc('path.simplify'):
            xy, offsets = _draw.simplify(xy, offsets, rc('path.simplify_threshold'))
        styles = _draw.Styles(color, width_px, None if dashes_px is None else [dashes_px],
                              None, cap, join)
        segs = _draw.segments(xy, offsets, None, True if line else None, False if line else None,
                              filled)
        self.c.segments(segs, styles, clip)


class _FigureArea:
    """What a figure's own artists (add_artist) draw through in place of an
    axes: their coordinates straight to pixels by their transform, as Agg
    draws a figure's patches (snapped, butt caps); .artists is the figure's,
    so artist.remove() works."""

    _snap = None
    box_clip = None

    def __init__(self, fig):
        self.figure = fig

    @property
    def artists(self):
        return self.figure.artists

    def _src(self, transform):
        if transform is None:
            return self.figure.transFigure
        if not isinstance(transform, _transforms.Transform):
            raise TypeError('a figure artist\'s transform must be fig.transFigure or another '
                            'fastplot.transforms transform')
        return transform

    def _points_px(self, x, y, src):
        return src.to_px(x, y)

    def _shape_px(self, x, y, src):
        return [src.to_px(x, y)]


class Figure:
    """matplotlib's Figure, for maps."""

    def __init__(self, figsize=None, dpi=None, facecolor=None, edgecolor=None, **kw):
        layout = kw.pop('layout', None)
        cl, tl = kw.pop('constrained_layout', None), kw.pop('tight_layout', None)
        frameon = kw.pop('frameon', None)
        _ignore(kw, 'figure', ('num', 'clear', 'linewidth', 'subplotpars'))
        self._layout, self._cl_pads, self._tight_kw, self._cl_boxes = None, {}, {}, {}
        if layout is None:
            if cl:
                layout = 'constrained'
                if isinstance(cl, dict):
                    self._cl_pads = dict(cl)
            elif tl:
                layout = 'tight'
                if isinstance(tl, dict):
                    self._tight_kw = dict(tl)
            elif rc('figure.autolayout'):
                layout = 'tight'
            elif rc('figure.constrained_layout.use'):
                layout = 'constrained'
        self.set_layout_engine(layout)
        self.figsize = tuple(figsize) if figsize is not None else tuple(rc('figure.figsize'))
        self.dpi = float(dpi) if dpi is not None else float(rc('figure.dpi'))
        self.facecolor = rc('figure.facecolor') if facecolor is None else facecolor
        self.patch_visible = frameon is not False
        self.subplotpars = subplotpars()
        self.axes = []
        self.colorbars = []
        self.texts = []
        self.legends = []
        self.artists = []                               # add_artist's
        self.transFigure = _transforms.TransFigure(self)
        self.dpi_scale_trans = _transforms.DpiScaleTrans(self)
        self._area = _FigureArea(self)
        self._current_axes = None
        self._px_size = self._size_px(self.dpi)
        self._draw_dpi = self.dpi
        self._org = (0.0, 0.0)          # where the figure's top-left falls on the canvas, px
        self._touch()

    def _touch(self):
        """Keep the settings in force now: what the figure is drawn with
        (_rc.using), as matplotlib's artists keep those they were made with."""
        self._rc = dict(_rc.active())

    # -- sizes
    def set_size_inches(self, w, h=None, forward=True):
        if h is None:
            w, h = w
        self.figsize = (float(w), float(h))

    def set_figwidth(self, w, forward=True):
        self.figsize = (float(w), self.figsize[1])

    def set_figheight(self, h, forward=True):
        self.figsize = (self.figsize[0], float(h))

    def get_figwidth(self):
        return self.figsize[0]

    def get_figheight(self):
        return self.figsize[1]

    def get_dpi(self):
        return self.dpi

    def get_axes(self):
        return list(self.axes)

    @property
    def canvas(self):
        """matplotlib's fig.canvas, for what scripts ask of it: draw(),
        buffer_rgba(), get_width_height(), print_figure()."""
        return _Canvas(self)

    def align_labels(self, axs=None):
        pass                                            # labels already sit by their tick labels

    align_xlabels = align_ylabels = align_titles = align_labels

    def autofmt_xdate(self, bottom=0.2, rotation=30, ha='right', which='major'):
        """matplotlib's autofmt_xdate: the bottom row's x tick labels rotated
        and right-aligned, the other rows' hidden, room made below."""
        plain = [ax for ax in self.axes if hasattr(ax, 'xaxis')]
        if not plain:
            return
        if len(plain) == 1:
            plain[0].xaxis.tick_label_kw.update(rotation=rotation, ha=ha)
        else:
            for ax in plain:
                ss = ax.spec if isinstance(ax.spec, SubplotSpec) else None
                last = ss is None or ss.r1 == topmost(ss).gs.nrows - 1 if ss is not None \
                    else True
                if last:
                    ax.xaxis.tick_label_kw.update(rotation=rotation, ha=ha)
                else:
                    ax.xaxis.set_tick_params(which='both', labelbottom=False)
                    ax.xaxis.label_text = ''
        if self._layout is None:
            self.subplots_adjust(bottom=bottom)

    def get_size_inches(self):
        return np.array(self.figsize)

    def set_dpi(self, dpi):
        self.dpi = float(dpi)

    def set_facecolor(self, c):
        self.facecolor = c

    @property
    def patch(self):
        from ._axesbase import Background
        return Background(self, 'patch_visible')

    def get_facecolor(self):
        """The background as matplotlib gives it: an (r, g, b, a) of floats."""
        return tuple(float(v) / 255.0 for v in rgba(self.facecolor))

    # -- layout engines: matplotlib's layout='constrained' / 'tight'
    def set_layout_engine(self, layout=None, **kw):
        """'constrained' (or 'compressed'), 'tight', or 'none' / None; the
        constrained layout's w_pad, h_pad, wspace, hspace, the tight
        layout's pad, h_pad, w_pad, rect."""
        name = getattr(layout, 'name', layout)
        if name in ('compressed',):
            name = 'constrained'
        if name not in (None, 'none', 'constrained', 'tight'):
            raise ValueError("Invalid value for 'layout': %r" % (layout,))
        self._layout = None if name == 'none' else name
        params = dict(getattr(layout, 'params', {}), **kw)
        if self._layout == 'constrained':
            self._cl_pads.update({k: v for k, v in params.items() if k in
                                  ('w_pad', 'h_pad', 'wspace', 'hspace')})
        elif self._layout == 'tight':
            self._tight_kw.update({k: v for k, v in params.items() if k in
                                   ('pad', 'h_pad', 'w_pad', 'rect')})
        if self._layout != 'constrained':
            self._cl_boxes = {}

    def get_layout_engine(self):
        return None if self._layout is None else LayoutEngine(self)

    def set_constrained_layout(self, constrained):
        if constrained:
            self.set_layout_engine('constrained', **(constrained if isinstance(constrained, dict)
                                                     else {}))
        elif self._layout == 'constrained':
            self.set_layout_engine(None)

    def get_constrained_layout(self):
        return self._layout == 'constrained'

    def set_constrained_layout_pads(self, **kw):
        if self._layout == 'constrained':
            self._cl_pads.update(kw)

    def get_constrained_layout_pads(self, relative=False):
        from ._constrained import pads
        p = pads(self)
        if relative:
            p['w_pad'] /= self.figsize[0]
            p['h_pad'] /= self.figsize[1]
        return p['w_pad'], p['h_pad'], p['wspace'], p['hspace']

    def set_tight_layout(self, tight):
        if tight:
            self.set_layout_engine('tight', **(tight if isinstance(tight, dict) else {}))
        elif self._layout == 'tight':
            self.set_layout_engine(None)

    def get_tight_layout(self):
        return self._layout == 'tight'

    def _execute_layout(self):
        """The layout engine's work before a draw (matplotlib's: each draw)."""
        if self._layout == 'tight' and self.axes:
            self._tight_layout(self._tight_kw.get('pad', 1.08), self._tight_kw.get('h_pad'),
                               self._tight_kw.get('w_pad'), self._tight_kw.get('rect'))
        elif self._layout == 'constrained' and self.axes:
            from ._constrained import do_layout
            do_layout(self, Figure._measure_boxes)

    def _measure_boxes(self):
        """A draw that paints nothing: each axes', colorbar's and figure
        text's (box, box grown by what it draws), figure fractions, y up."""
        dpi = self.dpi
        W, H = self._size_f(dpi)
        r = _Renderer(self, _NullCanvas(*self._size_px(dpi)[::-1]), dpi, W, H)
        self._px_size, self._draw_dpi = (W, H), dpi

        def frac(e):
            return (e[0] / W, 1 - e[3] / H, e[2] / W, 1 - e[1] / H)

        def orig(ax):
            b = getattr(ax, '_orig_pos', None)
            return frac(ax.box_f) if b is None else (b.l, b.b, b.l + b.w, b.b + b.h)

        def grown(box, exts):
            e = np.asarray([box] + [x for x in exts if x is not None], float).reshape(-1, 4)
            return (e[:, 0].min(), e[:, 1].min(), e[:, 2].max(), e[:, 3].max())
        out = {}
        try:
            for ax in self.axes:
                ax._place(self, W, H)
            for cb in self.colorbars:
                cb.ax._place(self, W, H)
            for ax in self.axes:
                if getattr(ax, 'visible', True):
                    n = len(r.extents)
                    ax._draw(r)
                    out[id(ax)] = (orig(ax), frac(grown(ax.box_f, r.extents[n:])))
            for cb in self.colorbars:
                n = len(r.extents)
                cb._draw(r)
                out[id(cb.ax)] = (orig(cb.ax), frac(grown(cb.ax.box_f, r.extents[n:])))
            for t in list(self.texts) + list(self.legends):
                if t.visible:
                    n = len(r.extents)
                    t.render(r, None) if t in self.texts else t.render(r)
                    ext = [e for e in r.extents[n:] if e is not None]
                    if ext:
                        b = grown(ext[0], ext[1:])
                        out[id(t)] = (frac(b), frac(b))
        finally:
            for ax in self.axes:
                ax._placed = None
        return out

    def subplots_adjust(self, left=None, bottom=None, right=None, top=None, wspace=None,
                        hspace=None):
        if self._layout == 'constrained':               # matplotlib: warns, leaves it
            return
        for k, v in dict(left=left, bottom=bottom, right=right, top=top, wspace=wspace,
                         hspace=hspace).items():
            if v is not None:
                self.subplotpars[k] = v

    # -- axes
    def _new_axes(self, spec, projection, sharex=None, sharey=None, **kw):
        """Plain axes (no projection, or 'rectilinear') or a map (a crs)."""
        if projection is None or projection == 'rectilinear':
            ax = Axes(self, spec, sharex=sharex, sharey=sharey)
        elif projection == 'polar':
            from ._polar import PolarAxes
            ax = PolarAxes(self, spec, sharex=sharex, sharey=sharey,
                           **{k: kw.pop(k) for k in ('theta_offset', 'theta_direction',
                                                     'rlabel_position') if k in kw})
        elif projection == 'skewx':                     # MetPy's SkewXAxes
            ax = Axes(self, spec, sharex=sharex, sharey=sharey)
            ax.name = 'skewx'
            ax.rot = float(kw.pop('rotation', 30))
            ax._skew = float(np.tan(np.radians(ax.rot)))
        elif isinstance(projection, str):
            raise ValueError("unknown projection %r (None, 'rectilinear', 'skewx' or a "
                             'fastplot.crs projection)' % projection)
        else:
            ax = MapAxes(self, spec, projection)
        fc = kw.pop('facecolor', kw.pop('fc', None))
        if fc is not None:
            ax.set_facecolor(fc)
        if kw.pop('frameon', True) is False:
            ax.set_frame_on(False)
        for k in ('title', 'xlabel', 'ylabel', 'xlim', 'ylim', 'xscale', 'yscale', 'aspect'):
            if k in kw:
                v = kw.pop(k)
                getattr(ax, 'set_' + k)(*v) if k in ('xlim', 'ylim') else \
                    getattr(ax, 'set_' + k)(v)
        _ignore(kw, 'axes', ('label', 'polar'))
        return self._add_axes_object(ax)

    def _add_axes_object(self, ax):
        self._touch()
        self.axes.append(ax)
        self._current_axes = ax
        return ax

    def add_subplot(self, *args, projection=None, sharex=None, sharey=None, **kw):
        if kw.pop('polar', False):
            projection = 'polar'
        if not args:
            args = (1, 1, 1)
        if len(args) == 1 and isinstance(args[0], SubplotSpec):
            spec = args[0]
        else:
            if len(args) == 1:
                n = int(args[0])
                args = (n // 100, (n // 10) % 10, n % 10)
            nrows, ncols, index = args
            gs = self._gridspec_like(nrows, ncols)
            if isinstance(index, tuple):
                spec = SubplotSpec(gs, (index[0] - 1) // ncols, (index[1] - 1) // ncols,
                                   (index[0] - 1) % ncols, (index[1] - 1) % ncols)
            else:
                spec = gs[index - 1]
        return self._new_axes(spec, projection, sharex, sharey, **kw)

    def _gridspec_like(self, nrows, ncols):
        """matplotlib's GridSpec._check_gridspec_exists: the grid an earlier
        add_subplot(nrows, ncols, i) made, else a new one -- so subplot(121)
        and subplot(122) are cells of one grid (which layouts see)."""
        for ax in self.axes:
            s = getattr(ax, 'spec', None)
            if isinstance(s, SubplotSpec) and s.gs.parent is None and \
                    (s.gs.nrows, s.gs.ncols) == (nrows, ncols) and \
                    s.gs.width_ratios is None and s.gs.height_ratios is None and \
                    s.gs.wspace is None and s.gs.hspace is None and \
                    not any(v is not None for v in s.gs.pars.values()):
                return s.gs
        return GridSpec(nrows, ncols, figure=self)

    def add_axes(self, rect, projection=None, sharex=None, sharey=None, **kw):
        return self._new_axes(_Box(*rect), projection, sharex, sharey, **kw)

    def delaxes(self, ax):
        self.axes.remove(ax)
        self.colorbars = [cb for cb in self.colorbars if cb.mappable.axes is not ax]

    def add_gridspec(self, nrows=1, ncols=1, **kw):
        return GridSpec(nrows, ncols, figure=self, **kw)

    def subplots(self, nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True,
                 subplot_kw=None, gridspec_kw=None, width_ratios=None, height_ratios=None):
        """matplotlib's Figure.subplots: shared axes ('all'/True, 'row', 'col')
        keep only the outer subplots' tick labels."""
        gkw = dict(gridspec_kw or {})
        if width_ratios is not None:
            gkw['width_ratios'] = width_ratios
        if height_ratios is not None:
            gkw['height_ratios'] = height_ratios
        gs = GridSpec(nrows, ncols, figure=self, **gkw)
        return self._grid_subplots(gs, sharex, sharey, squeeze, subplot_kw)

    def _grid_subplots(self, gs, sharex, sharey, squeeze, subplot_kw):
        nrows, ncols = gs.nrows, gs.ncols
        subplot_kw = dict(subplot_kw or {})
        sx = {True: 'all', False: 'none'}.get(sharex, sharex)
        sy = {True: 'all', False: 'none'}.get(sharey, sharey)
        axs = np.empty((nrows, ncols), object)
        for i in range(nrows):
            for j in range(ncols):
                share = {'all': axs[0, 0], 'row': axs[i, 0], 'col': axs[0, j], 'none': None}
                kw = dict(subplot_kw)
                if subplot_kw.get('projection') is None:
                    kw['sharex'] = None if (i, j) == (0, 0) else share[sx]
                    kw['sharey'] = None if (i, j) == (0, 0) else share[sy]
                    if sx == 'row' and j == 0 or sx == 'col' and i == 0:
                        kw['sharex'] = None
                    if sy == 'row' and j == 0 or sy == 'col' and i == 0:
                        kw['sharey'] = None
                axs[i, j] = self.add_subplot(gs[i, j], **kw)
        for ax in axs.ravel():
            if not isinstance(ax, Axes):
                continue
            ss = ax.spec
            if sx in ('all', 'col') and ss.r1 != nrows - 1:
                ax.xaxis.set_tick_params(which='both', labelbottom=False)
            if sy in ('all', 'row') and ss.c0 != 0:
                ax.yaxis.set_tick_params(which='both', labelleft=False)
        if squeeze:
            return axs.item() if axs.size == 1 else axs.squeeze()
        return axs

    def subplot_mosaic(self, mosaic, *, sharex=False, sharey=False, width_ratios=None,
                       height_ratios=None, empty_sentinel='.', subplot_kw=None,
                       per_subplot_kw=None, gridspec_kw=None):
        """matplotlib's subplot_mosaic: axes laid out by a picture ('AB;CC',
        a multi-line string, or nested lists), {label: axes}. A label's cells
        must make a rectangle; a nested list is a mosaic inside its cell."""
        import inspect
        if isinstance(mosaic, str):
            lines = (mosaic.split(';') if '\n' not in mosaic else
                     inspect.cleandoc(mosaic).strip('\n').split('\n'))
            mosaic = [list(ln) for ln in lines]
        gkw = dict(gridspec_kw or {})
        if width_ratios is not None:
            gkw['width_ratios'] = width_ratios
        if height_ratios is not None:
            gkw['height_ratios'] = height_ratios
        per = {}
        for k, v in (per_subplot_kw or {}).items():
            for name in (k if isinstance(k, tuple) else (k,)):
                per[name] = v
        out = {}

        def layout(grid, gs):
            if not grid or not all(isinstance(r, (list, tuple, str)) for r in grid):
                raise ValueError('List mosaic specification must be 2D')
            rows = len(grid)
            cols = {len(r) for r in grid}
            if len(cols) != 1:
                raise ValueError('All of the rows in the mosaic must be the same length')
            ncols = cols.pop()
            cells = [[None] * ncols for _ in range(rows)]
            nested = {}
            names = []                                  # labels, in reading order
            for i, row in enumerate(grid):
                for j, v in enumerate(row):
                    if isinstance(v, list):
                        nested[(i, j)] = v
                        continue
                    cells[i][j] = (v,)                  # wrapped: any hashable label
                    if v != empty_sentinel and v not in names:
                        names.append(v)
            level = {}
            for name in names:
                idx = np.array([(i, j) for i in range(rows) for j in range(ncols)
                                if cells[i][j] == (name,)])
                r0, c0 = idx.min(axis=0)
                r1, c1 = idx.max(axis=0) + 1
                if not all(cells[a][b] == (name,) for a in range(r0, r1) for b in range(c0, c1)):
                    raise ValueError('While trying to layout\n%r\nwe found that the label %r '
                                     'specifies a non-rectangular or non-contiguous area.'
                                     % (grid, name))
                level[(r0, c0)] = (name, (r0, r1 - 1, c0, c1 - 1), 'axes')
            for (i, j), sub in nested.items():
                level[(i, j)] = (None, sub, 'nested')
            for key in sorted(level):
                name, arg, kind = level[key]
                if kind == 'nested':
                    inner = [[c for c in r] for r in arg]
                    layout(inner, gs[key[0], key[1]].subgridspec(len(inner), len(inner[0])))
                    continue
                r0, r1, c0, c1 = arg
                if name in out:
                    raise ValueError('There are duplicate keys %r in the layout' % (name,))
                kw = dict(subplot_kw or {}, **per.get(name, {}))
                first = next(iter(out.values()), None)
                if first is not None and kw.get('projection') is None:
                    kw.setdefault('sharex', first if sharex else None)
                    kw.setdefault('sharey', first if sharey else None)
                ax = self.add_subplot(SubplotSpec(gs, r0, r1, c0, c1), **kw)
                out[name] = ax
        layout(mosaic, GridSpec(len(mosaic), len(mosaic[0]), figure=self, **gkw))
        for ax in out.values():                         # matplotlib's _label_outer_*axis
            if not isinstance(ax, Axes):
                continue
            ss, top = ax.spec, ax.spec.gs
            if sharex and top.parent is None and ss.r1 != top.nrows - 1:
                ax.xaxis.set_tick_params(which='both', labelbottom=False)
            if sharey and top.parent is None and ss.c0 != 0:
                ax.yaxis.set_tick_params(which='both', labelleft=False)
        return out

    def gca(self):
        return self.axes[-1] if self.axes else self.add_subplot()

    def colorbar(self, mappable, cax=None, ax=None, orientation=None, location=None,
                 fraction=0.15, shrink=1.0, aspect=20, pad=None, anchor=None, panchor=None,
                 label='', ticks=None, format=None, extend=None, extendfrac=None,
                 drawedges=False, spacing='uniform', use_gridspec=True, **kw):
        _ignore(kw, 'colorbar', ('extendrect', 'boundaries', 'values', 'filled'))
        if location is None:
            location = 'bottom' if orientation == 'horizontal' else 'right'
        if orientation is None:
            orientation = 'horizontal' if location in ('top', 'bottom') else 'vertical'
        if cax is None:
            parents = ax if ax is not None else mappable.axes
            parents = list(np.ravel(parents)) if isinstance(parents, (list, tuple, np.ndarray)) \
                else [parents]
            cax = make_cax(self, parents, location, fraction, shrink, aspect, pad, anchor,
                           panchor, use_gridspec)
        elif not isinstance(cax, ColorbarAxes):          # any axes: the bar drawn in it (cb.ax
            self.axes.remove(cax)                        # is cax, as matplotlib's)
            if self._current_axes is cax:
                self._current_axes = self.axes[-1] if self.axes else None
            ColorbarAxes.adopt(cax)
        cb = Colorbar(self, mappable, cax, orientation, label, ticks, format, extend, extendfrac,
                      drawedges, spacing)
        self._touch()
        self.colorbars.append(cb)
        return cb

    def text(self, x, y, s, **kw):
        kw.pop('transform', None)
        t = Text(self, x, y, s, 'figure', **_text_kw(kw))
        self._touch()
        self.texts.append(t)
        return t

    def add_artist(self, artist, clip=False):
        """matplotlib's Figure.add_artist: a fastplot.patches shape drawn on
        the figure itself -- in fig.transFigure unless its transform says
        otherwise, unclipped unless clip -- among the axes by zorder."""
        area = self._area
        artist.axes = area
        artist.src = area._src(getattr(artist, 'transform', None))
        artist.clip = bool(clip)
        self.artists.append(artist)
        return artist

    def suptitle(self, t, x=None, y=None, ha=None, va=None, fontsize=None, **kw):
        return self._suplabel('_suptitle', t, x, y, ha, va, fontsize, kw,
                              (0.5, 0.98, 'center', 'top', 0.0), 'figure.title')

    def supxlabel(self, t, x=None, y=None, ha=None, va=None, fontsize=None, **kw):
        """matplotlib's supxlabel: a label centred below all the subplots."""
        return self._suplabel('_supxlabel', t, x, y, ha, va, fontsize, kw,
                              (0.5, 0.01, 'center', 'bottom', 0.0), 'figure.label')

    def supylabel(self, t, x=None, y=None, ha=None, va=None, fontsize=None, **kw):
        """matplotlib's supylabel: a label centred left of all the subplots."""
        return self._suplabel('_supylabel', t, x, y, ha, va, fontsize, kw,
                              (0.02, 0.5, 'left', 'center', 90.0), 'figure.label')

    def _suplabel(self, name, t, x, y, ha, va, fontsize, kw, info, rcp):
        x0, y0, ha0, va0, rot = info
        autopos = x is None and y is None
        ha = kw.pop('horizontalalignment', ha) or ha0
        va = kw.pop('verticalalignment', va) or va0
        size = fontsize if fontsize is not None else kw.pop('size', rc(rcp + 'size'))
        kw.setdefault('fontweight', kw.pop('weight', rc(rcp + 'weight')))
        kw.setdefault('rotation', rot)
        old = getattr(self, name, None)
        if old in self.texts:
            self.texts.remove(old)
        txt = self.text(x0 if x is None else x, y0 if y is None else y, t, ha=ha, va=va,
                        fontsize=size, **kw)
        txt._autopos = autopos
        setattr(self, name, txt)
        return txt

    def get_suptitle(self):
        t = getattr(self, '_suptitle', None)
        return '' if t is None else t.s

    def get_supxlabel(self):
        t = getattr(self, '_supxlabel', None)
        return '' if t is None else t.s

    def get_supylabel(self):
        t = getattr(self, '_supylabel', None)
        return '' if t is None else t.s

    def tight_layout(self, pad=1.08, h_pad=None, w_pad=None, rect=None):
        """matplotlib's tight_layout: the subplot parameters that fit every
        subplot's tick labels, axis labels, titles and colorbars in the
        figure, pad font sizes apart -- measured by a draw that paints
        nothing, then set (subplots_adjust)."""
        with _rc.using(self._rc):
            self.set_layout_engine(None)                # matplotlib's: once, then 'none'
            self._tight_layout(pad, h_pad, w_pad, rect)

    def _tight_layout(self, pad, h_pad, w_pad, rect):
        dpi = self.dpi
        W, H = self._size_f(dpi)
        r = _Renderer(self, _NullCanvas(*self._size_px(dpi)[::-1]), dpi, W, H)
        self._px_size, self._draw_dpi = (W, H), dpi
        for ax in self.axes:
            ax._place(self, W, H)
        for cb in self.colorbars:
            cb.ax._place(self, W, H)
        groups = {}                                     # topmost spec -> [boxes, extents]

        def measure(ax, draw):
            n = len(r.extents)
            draw(r)
            if not isinstance(ax.spec, SubplotSpec):
                return
            top = topmost(ax.spec)
            g = groups.setdefault((id(top.gs), top.r0, top.r1, top.c0, top.c1), [top, [], []])
            g[1].append(ax.spec.box(self))
            g[2] += [e for e in r.extents[n:] if e is not None]
        try:
            for ax in self.axes:
                if getattr(ax, 'visible', True):
                    measure(ax, ax._draw)
            for cb in self.colorbars:
                measure(cb.ax, cb._draw)
        finally:
            for ax in self.axes:
                ax._placed = None
        fit = []
        for top, boxes, exts in groups.values():
            l = min(b.l for b in boxes)
            b0 = min(b.b for b in boxes)
            box = _Box(l, b0, max(b.l + b.w for b in boxes) - l,
                       max(b.b + b.h for b in boxes) - b0)
            e = np.asarray(exts, float).reshape(-1, 4) if exts else \
                np.array([[box.l * W, (1 - box.b - box.h) * H, (box.l + box.w) * W,
                           (1 - box.b) * H]])
            fit.append((top, box, (e[:, 0].min() / W, 1 - e[:, 3].max() / H,
                                   e[:, 2].max() / W, 1 - e[:, 1].min() / H)))
        sups = []
        for name, k in (('_suptitle', 3), ('_supxlabel', 3), ('_supylabel', 2)):
            sup = getattr(self, name, None)
            ext = None
            if sup is not None and sup in self.texts and sup.visible:
                ext = r.measure(sup.s, 0, 0, fontsize_pt(sup.fontsize), rotation=sup.rotation,
                                weight=sup.weight)
            sups.append(0.0 if ext is None else
                        (ext[3] - ext[1]) / H if k == 3 else (ext[2] - ext[0]) / W)
        kw = tight_subplotpars(fit, self.figsize, rc('font.size'), pad, h_pad, w_pad, rect,
                               *sups)
        if kw:
            self.subplots_adjust(**kw)

    # -- drawing
    def _size_px(self, dpi):
        """The canvas in whole px: as matplotlib's get_width_height, with its
        allowance for a size just under a whole number (9.37 in at 100 dpi)."""
        return int(self.figsize[0] * dpi + 1e-8), int(self.figsize[1] * dpi + 1e-8)

    def _size_f(self, dpi):
        """The figure's exact size in px (matplotlib's bbox: figure fractions
        scale by it; the canvas is its whole part, the top's fraction cut)."""
        return float(self.figsize[0] * dpi), float(self.figsize[1] * dpi)

    def _work(self, dpi):
        W, H = self._size_px(dpi)
        on_gpu = any(getattr(a, 'on_gpu', False) for ax in self.axes for a in ax.artists)
        return W * H + sum(ax.work() for ax in self.axes), on_gpu

    def draw(self, dpi=None, device=None, facecolor=None, _transient=False, _page=None):
        """Draw the figure: RGBA uint8 (H, W, 4), numpy. (_transient: the
        image is used and dropped before the next draw on this thread --
        savefig's -- so its memory is reused.) Its kernels run on at most
        a draw's count of numba's threads (_hw.draw_threads), which are let
        go when it ends (_jit.park)."""
        _hw.fast_cores()
        dpi = float(dpi or self.dpi)
        W, H = self._size_px(dpi)
        work, on_gpu = self._work(dpi)
        dev = device or _device.pick(work, on_gpu)
        with _jit.threads(park=True), _rc.using(self._rc):
            self._execute_layout()
            if dev == 'gpu':
                try:
                    return self._draw_on('gpu', dpi, W, H, facecolor, _transient, _page)
                except Exception as e:                  # noqa: BLE001 -- the CPU draws it then
                    _device.gpu_failed(e)
            return self._draw_on('cpu', dpi, W, H, facecolor, _transient, _page)

    def _draw_on(self, dev, dpi, W, H, facecolor, transient=False, page=None):
        """page: (canvas width, height, x, y) -- a canvas larger than the
        figure, which is drawn with its top-left at (x, y) (savefig's
        bbox_inches='tight', when what is drawn reaches beyond the figure)."""
        if facecolor is None:
            facecolor = self.facecolor if self.patch_visible else (0, 0, 0, 0)
        bg = rgba(facecolor)
        CW, CH = (W, H) if page is None else page[:2]
        if dev == 'gpu':
            from . import _gpu
            canvas = _gpu.GpuCanvas(CH, CW, bg)
        else:
            canvas = CpuCanvas(CH, CW, bg, scratch(CH, CW) if transient else None)
        # matplotlib's display space: fractions of the exact size, y flipped
        # about the canvas' whole height (a fractional top is cut off)
        Wf, Hf = self._size_f(dpi)
        base = H - Hf
        self._org = (0.0, base) if page is None else (float(page[2]), float(page[3]) + base)
        try:
            return self._draw_children(dev, canvas, dpi, Wf, Hf, transient)
        finally:
            self._org = (0.0, 0.0)

    def _draw_children(self, dev, canvas, dpi, W, H, transient):
        r = _Renderer(self, canvas, dpi, W, H)
        self._px_size, self._draw_dpi = (W, H), dpi
        for ax in self.axes:
            ax._place(self, W, H)
        for cb in self.colorbars:
            cb.ax._place(self, W, H)
        for _z, _k, draw in self._children(r):
            draw()
        for ax in self.axes:
            ax._placed = None
        self._extents = r.extents
        return canvas.array() if dev == 'cpu' else canvas.array(copy=not transient)

    def _children(self, r):
        """What the figure draws, in matplotlib's order: by zorder (axes and
        colorbars 0, patches 1, texts 3, legends 5 unless set), then axes,
        colorbars, figure artists, texts, legends, each in the order added."""
        out = [(ax.zorder, 0, lambda ax=ax: ax._draw(r)) for ax in self.axes
               if getattr(ax, 'visible', True)]
        out += [(cb.ax.zorder, 1, lambda cb=cb: cb._draw(r)) for cb in self.colorbars]
        out += [(a.zorder, 2, lambda a=a: a.render(r, self._area))
                for a in self.artists if a.visible]
        out += [(t.zorder, 3, lambda t=t: t.render(r, None)) for t in self.texts]
        out += [(lg.zorder, 4, lambda lg=lg: lg.render(r)) for lg in self.legends]
        return sorted(out, key=lambda c: c[:2])         # stable: the order added within each

    def to_array(self, **kw):
        return self.draw(**kw)

    def savefig(self, fname, dpi=None, format=None, bbox_inches=None, pad_inches=None,
                facecolor=None, transparent=False, metadata=None, device=None, **kw):
        """Write the figure: PNG (fastplot's encoder), anything Pillow
        writes (jpg, webp, tiff, ...), or PDF / EPS / SVG holding the
        picture (fastplot draws pixels: dpi sets their resolution), by
        extension or format."""
        _ignore(kw, 'savefig', ('edgecolor', 'orientation', 'papertype', 'backend',
                                'pil_kwargs', 'bbox_extra_artists'))
        live = _rc.rcParams                             # savefig.* as they are when saving
        if dpi is None:
            dpi = live.get('savefig.dpi', 'figure')
        if dpi == 'figure' or dpi is None:
            dpi = self.dpi
        if bbox_inches is None:
            bbox_inches = live.get('savefig.bbox')
        if pad_inches is None or pad_inches == 'layout':
            pad_inches = live.get('savefig.pad_inches', 0.1)
        if format is None and not isinstance(fname, (str, os.PathLike)) and \
                not getattr(fname, 'name', ''):
            format = live.get('savefig.format', 'png')
        if not transparent:
            transparent = bool(live.get('savefig.transparent', False))
        if facecolor is None and live.get('savefig.facecolor', 'auto') != 'auto':
            facecolor = live['savefig.facecolor']
        fc = facecolor
        hidden = []
        if transparent:                                 # matplotlib's: the figure's background
            if facecolor is None:                       # (unless one is given) and every axes'
                fc = (0, 0, 0, 0)
            hidden = [ax for ax in self.axes if getattr(ax, 'patch_visible', False)]
        try:
            for ax in hidden:
                ax.patch_visible = False
            with _jit.threads(park=True):               # numba's workers let go once it is written
                self._save(fname, dpi, format, bbox_inches, pad_inches, fc, metadata, device)
        finally:
            for ax in hidden:
                ax.patch_visible = True

    def _save(self, fname, dpi, format, bbox_inches, pad_inches, fc, metadata, device):
        img = self.draw(dpi=dpi, device=device, facecolor=fc, _transient=True)
        if bbox_inches == 'tight':
            page = self._tight_page(img, float(dpi), pad_inches)
            if page is not None:                        # matplotlib's: drawn again on a canvas
                img = self.draw(dpi=dpi, device=device, facecolor=fc, _page=page)  # the bbox's
        if format is None:
            name = fname if isinstance(fname, (str, os.PathLike)) else getattr(fname, 'name', '')
            ext = os.path.splitext(str(name))[1].lower().lstrip('.')
            format = ext or _rc.rcParams.get('savefig.format', 'png')
        format = format.lower()
        meta = {'Software': 'fastplot'}
        meta.update(metadata or {})
        if format == 'png':
            from . import _png
            _png.save(fname, img, dpi=dpi, metadata=meta)
        elif format in ('raw', 'rgba'):
            data = np.ascontiguousarray(img).tobytes()
            if isinstance(fname, (str, os.PathLike)):
                with open(fname, 'wb') as fh:
                    fh.write(data)
            else:
                fname.write(data)
        elif format == 'svg':
            _save_svg(fname, img, dpi)
        elif format == 'pdf':
            _save_pdf(fname, img, dpi, meta)
        else:
            from PIL import Image
            im = Image.fromarray(img, 'RGBA')
            if format in ('jpg', 'jpeg'):
                im, format = im.convert('RGB'), 'jpeg'
            if format in ('eps', 'ps'):                 # the picture, flattened on white
                bg = Image.new('RGB', im.size, (255, 255, 255))
                bg.paste(im, mask=im.getchannel('A'))
                im, format = bg, 'eps'
            im.save(fname, format=format.upper(), dpi=(dpi, dpi))

    def _tight_page(self, img, dpi, pad_inches):
        """matplotlib's bbox_inches='tight' canvas: (width, height, x, y) --
        everything drawn and its pad, the size truncated to whole pixels,
        the figure drawn with its top-left at (x, y) (fractions of a pixel
        kept, as matplotlib's translated draw keeps them). None: nothing
        drawn."""
        ext = [e for e in getattr(self, '_extents', []) if e is not None]
        if not ext:
            return None
        e = np.asarray(ext, float)
        e = e[np.isfinite(e).all(axis=1)]
        if not len(e):
            return None
        pad = pad_inches * dpi
        H, W = img.shape[:2]
        x0, y0 = e[:, 0].min() - pad, e[:, 1].min() - pad
        x1, y1 = e[:, 2].max() + pad, e[:, 3].max() + pad
        cw, ch = int(x1 - x0), int(y1 - y0)
        if cw <= 0 or ch <= 0 or cw * ch > 16 * W * H:  # runaway extents: keep the figure
            return None
        return cw, ch, -x0, ch - y1                     # (y up as matplotlib's: from the bottom)

    def legend(self, *args, **kw):
        """A legend of the figure: of all its axes' labelled artists unless
        handles are given; placed in the figure."""
        from ._legend import Legend, parse_args
        handles, labels = parse_args(args, kw, self.axes)
        self._touch()
        lg = Legend(self, handles, labels, **kw)
        self.legends.append(lg)
        return lg

    def clear(self):
        self.axes, self.colorbars, self.texts, self.legends = [], [], [], []
        self.artists = []

    clf = clear


class _Canvas:
    """fig.canvas: draw() lays out and draws (the picture kept for
    buffer_rgba / tostring_argb / copy_from_bbox users); print_figure
    is savefig."""

    def __init__(self, fig):
        self.figure = fig

    def draw(self):
        self.figure._last_image = self.figure.draw()

    def draw_idle(self):
        self.draw()

    def flush_events(self):
        pass

    def get_renderer(self):
        return None

    def _image(self):
        img = getattr(self.figure, '_last_image', None)
        if img is None:
            self.draw()
            img = self.figure._last_image
        return img

    def buffer_rgba(self):
        return memoryview(np.ascontiguousarray(self._image()))

    def tostring_rgb(self):
        return np.ascontiguousarray(self._image()[..., :3]).tobytes()

    def tostring_argb(self):
        return np.ascontiguousarray(np.roll(self._image(), 1, axis=2)).tobytes()

    def get_width_height(self, physical=False):
        W, H = self.figure._size_px(self.figure.dpi)
        return W, H

    def print_figure(self, fname, **kw):
        self.figure.savefig(fname, **kw)

    def mpl_connect(self, *args, **kw):
        return 0

    def mpl_disconnect(self, cid):
        pass

    def set_window_title(self, t):
        pass

    @property
    def manager(self):
        return None


class LayoutEngine:
    """fig.get_layout_engine(): its name, and set() / get() of its pads."""

    def __init__(self, fig):
        self.fig = fig
        self.name = fig._layout

    def set(self, **kw):
        if self.name == 'constrained':
            self.fig._cl_pads.update(kw)
        else:
            self.fig._tight_kw.update(kw)

    def get(self):
        if self.name == 'constrained':
            from ._constrained import pads
            return pads(self.fig)
        return dict(self.fig._tight_kw)

    def execute(self, fig):
        fig._execute_layout()


def _save_pdf(fname, img, dpi, meta):
    """A one-page PDF of the picture: its pixels Flate-compressed (lossless),
    their alpha a soft mask, the page the figure's size in points."""
    import zlib
    H, W = img.shape[:2]
    rgb = zlib.compress(np.ascontiguousarray(img[..., :3]).tobytes(), 6)
    alpha = img[..., 3]
    smask = None if (alpha == 255).all() else zlib.compress(
        np.ascontiguousarray(alpha).tobytes(), 6)
    wpt, hpt = W * 72.0 / dpi, H * 72.0 / dpi
    content = b'q %.4f 0 0 %.4f 0 0 cm /Im0 Do Q' % (wpt, hpt)
    image = (b'<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB '
             b'/BitsPerComponent 8 /Filter /FlateDecode /Length %d%s >>' %
             (W, H, len(rgb), b' /SMask 6 0 R' if smask is not None else b''))
    info = ' '.join('/%s (%s)' % (k, str(v).replace('\\', '\\\\').replace('(', '\\(')
                                  .replace(')', '\\)'))
                    for k, v in (meta or {}).items() if k in ('Title', 'Author', 'Subject',
                                                              'Software', 'Creator'))
    if 'Software' in (meta or {}) and 'Creator' not in meta:
        info = info.replace('/Software', '/Creator')
    objs = [b'<< /Type /Catalog /Pages 2 0 R >>',
            b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
            b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.4f %.4f] '
            b'/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>' % (wpt, hpt),
            image + b'\nstream\n' + rgb + b'\nendstream',
            b'<< /Length %d >>\nstream\n' % len(content) + content + b'\nendstream']
    if smask is not None:
        objs.append(b'<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace '
                    b'/DeviceGray /BitsPerComponent 8 /Filter /FlateDecode /Length %d >>'
                    b'\nstream\n' % (W, H, len(smask)) + smask + b'\nendstream')
    objs.append(('<< %s >>' % info).encode('latin-1', 'replace'))
    out = bytearray(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
    where = []
    for k, o in enumerate(objs, 1):
        where.append(len(out))
        out += b'%d 0 obj\n' % k + o + b'\nendobj\n'
    xref = len(out)
    out += b'xref\n0 %d\n0000000000 65535 f \n' % (len(objs) + 1)
    for w in where:
        out += b'%010d 00000 n \n' % w
    out += (b'trailer\n<< /Size %d /Root 1 0 R /Info %d 0 R >>\nstartxref\n%d\n%%%%EOF\n'
            % (len(objs) + 1, len(objs), xref))
    if isinstance(fname, (str, os.PathLike)):
        with open(fname, 'wb') as fh:
            fh.write(out)
    else:
        fname.write(bytes(out))


def _save_svg(fname, img, dpi):
    """An SVG of the picture: the PNG inside an <image>, the page its size
    in points (dpi pixels an inch) -- fastplot draws pixels, so this is the
    raster in a vector container, as savefig(..., format='svg') of a
    rasterized matplotlib figure."""
    import base64
    import io
    from . import _png
    H, W = img.shape[:2]
    buf = io.BytesIO()
    _png.save(buf, img, dpi=dpi)
    data = base64.b64encode(buf.getvalue()).decode('ascii')
    wpt, hpt = W * 72.0 / dpi, H * 72.0 / dpi
    text = ('<?xml version="1.0" encoding="utf-8" standalone="no"?>\n'
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" '
            'width="%gpt" height="%gpt" viewBox="0 0 %d %d">\n'
            ' <image width="%d" height="%d" xlink:href="data:image/png;base64,%s"/>\n'
            '</svg>\n' % (wpt, hpt, W, H, W, H, data))
    if isinstance(fname, (str, os.PathLike)):
        with open(fname, 'w', encoding='utf-8') as fh:
            fh.write(text)
    elif isinstance(fname, io.TextIOBase):
        fname.write(text)
    else:
        fname.write(text.encode('utf-8'))


def figure(figsize=None, dpi=None, facecolor=None, **kw):
    return Figure(figsize=figsize, dpi=dpi, facecolor=facecolor, **kw)


def subplots(nrows=1, ncols=1, *, sharex=False, sharey=False, squeeze=True, figsize=None,
             dpi=None, subplot_kw=None, gridspec_kw=None, width_ratios=None, height_ratios=None,
             facecolor=None, **kw):
    fig = Figure(figsize=figsize, dpi=dpi, facecolor=facecolor, **kw)
    return fig, fig.subplots(nrows, ncols, sharex=sharex, sharey=sharey, squeeze=squeeze,
                             subplot_kw=subplot_kw, gridspec_kw=gridspec_kw,
                             width_ratios=width_ratios, height_ratios=height_ratios)


def close(fig=None):
    """Nothing to free (matplotlib's plt.close, for ported scripts)."""
    if isinstance(fig, Figure):
        fig.clear()
