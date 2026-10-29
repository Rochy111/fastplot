"""fig.colorbar: the bar (bands or a gradient), its extensions, outline,
ticks and label, laid out as matplotlib's Colorbar."""
import numpy as np

from . import _color, _draw, _ticker
from ._style import fontsize_pt, rc, rgba
from ._layout import SubplotSpec, _Box, _anchor, _fit
from ._fields import ContourfArtist


def _shade_1d(values, cmap, norm):
    """Colours of a few values, as a vector artist paints them."""
    out = np.empty((1, values.size, 4), np.uint8)
    _color.shade(values[None].astype(np.float32), cmap, norm, out=out, rounding='nearest')
    return out[0]


class Colorbar:
    """fig.colorbar: matplotlib's colorbar for a field or contourf."""

    def __init__(self, fig, mappable, cax, orientation, label, ticks, fmt, extend, extendfrac,
                 drawedges, spacing):
        self.figure, self.mappable, self.ax = fig, mappable, cax
        cax.colorbar = self
        # plain axes' colorbars draw as Agg (snapped, butt caps); the maps' as before
        self._agg = getattr(getattr(mappable, 'axes', None), '_agg', False)
        self.orientation = orientation
        self.label = label
        self.label_kw = {}
        self.ticks = ticks
        self.format = fmt
        self.drawedges = drawedges
        self.spacing = spacing
        self.extendfrac = 0.05 if extendfrac in (None, 'auto') else float(extendfrac)
        if extend is None:
            extend = getattr(mappable, 'extend', None) or getattr(
                getattr(mappable, 'norm', None), 'extend', None) or 'neither'
        self.extend = extend
        self.minorticks = False

    def set_label(self, label, fontsize=None, labelpad=None, **kw):
        self.label = label
        self.label_kw = dict(fontsize=fontsize, labelpad=labelpad, **kw)

    def set_ticks(self, ticks, labels=None, **kw):
        self.ticks = ticks
        if labels is not None:
            self.format = dict(zip(np.asarray(ticks, float), labels))

    def set_ticklabels(self, labels, **kw):
        if self.ticks is not None:
            self.format = dict(zip(np.asarray(self.ticks, float), labels))

    def minorticks_on(self):
        self.minorticks = True

    def minorticks_off(self):
        self.minorticks = False

    def update_normal(self, mappable):
        self.mappable = mappable

    # -- what to draw
    def _scale(self):
        """The bar's colour stops: boundaries b (n+1), colours (n, 4), stop
        positions t (0..1 along the bar), the tick locator, and the under
        and over colours (None without that extension)."""
        m = self.mappable
        lo_ext, hi_ext = self.extend in ('min', 'both'), self.extend in ('max', 'both')
        if isinstance(m, ContourfArtist):
            b, cols = m.levels, m.band_lut
            return (b, cols, np.linspace(0.0, 1.0, cols.shape[0] + 1),
                    _ticker.FixedLocator(b, nbins=10), m.under if lo_ext else None,
                    m.over if hi_ext else None)
        norm, cmap = m.norm_spec(), m.cmap
        kind = getattr(norm, 'kind', 'linear')
        under = cmap.under if lo_ext else None
        over = cmap.over if hi_ext else None
        if kind == 'boundary':
            b = np.asarray(norm.boundaries, float)
            cols = _shade_1d(0.5 * (b[1:] + b[:-1]), cmap, norm)
            t = np.linspace(0.0, 1.0, cols.shape[0] + 1) if self.spacing != 'proportional' \
                else (b - b[0]) / (b[-1] - b[0])
            return b, cols, t, _ticker.FixedLocator(b, nbins=10), under, over
        vmin, vmax = float(norm.vmin), float(norm.vmax)
        n = max(cmap.N, 2)
        if kind == 'log':
            b = vmin * (vmax / vmin) ** np.linspace(0, 1, n + 1)
            cols = _shade_1d(np.sqrt(b[1:] * b[:-1]), cmap, norm)
            t = np.log(b / b[0]) / np.log(b[-1] / b[0])
        else:
            b = np.linspace(vmin, vmax, n + 1)
            cols = _shade_1d(0.5 * (b[1:] + b[:-1]), cmap, norm)
            t = (b - b[0]) / (b[-1] - b[0])
        return b, cols, t, _ticker.auto_locator(), under, over

    @property
    def _cap(self):
        return _draw.BUTT if self._agg else _draw.LEGACY

    def _draw(self, r):
        x0, y0, x1, y1 = self.ax.box_f
        horiz = self.orientation == 'horizontal'
        b, cols, t, loc, under, over = self._scale()
        L = (x1 - x0) if horiz else (y1 - y0)
        ef = self.extendfrac * L                        # extension length, px
        r.c.image(_bar_image(cols, t, x1 - x0, y1 - y0, horiz), int(round(x0)), int(round(y0)))
        self._extensions(r, under, over, ef, horiz)
        self._outline(r, t, under, over, ef, horiz)
        exts = self._ticks(r, b, t, loc, L, horiz)
        self._label(r, exts, horiz)

    def _extensions(self, r, under, over, ef, horiz):
        """The triangles beyond the ends, in the under/over colours."""
        x0, y0, x1, y1 = self.ax.box_f
        for col, low in ((under, True), (over, False)):
            if col is None:
                continue
            if horiz:
                ym = (y0 + y1) / 2
                tri = [[x0, y0], [x0 - ef, ym], [x0, y1]] if low else \
                    [[x1, y0], [x1 + ef, ym], [x1, y1]]
            else:
                xm = (x0 + x1) / 2
                tri = [[x0, y1], [xm, y1 + ef], [x1, y1]] if low else \
                    [[x0, y0], [xm, y0 - ef], [x1, y0]]
            r.c.polygons(np.array(tri), np.array([0, 3]), np.asarray(col, np.uint8), None,
                         'nonzero')

    def _outline(self, r, t, under, over, ef, horiz):
        """Band dividers (drawedges) and the outline round bar and extensions."""
        x0, y0, x1, y1 = self.ax.box_f
        lw = r.px(rc('axes.linewidth'))
        if horiz:
            pts = [[x0, y0]] + ([[x0 - ef, (y0 + y1) / 2]] if under is not None else []) + \
                [[x0, y1], [x1, y1]] + ([[x1 + ef, (y0 + y1) / 2]] if over is not None else []) + \
                [[x1, y0], [x0, y0]]
        else:
            pts = [[x0, y1]] + ([[(x0 + x1) / 2, y1 + ef]] if under is not None else []) + \
                [[x1, y1], [x1, y0]] + ([[(x0 + x1) / 2, y0 - ef]] if over is not None else []) + \
                [[x0, y0], [x0, y1]]
        pts = np.asarray(pts, float)
        if self.drawedges and t.size > 2:
            tt = t[1:-1]
            if horiz:
                xx = x0 + tt * (x1 - x0)
                s = np.stack([np.c_[xx, np.full_like(xx, y0)], np.c_[xx, np.full_like(xx, y1)]], 1)
            else:
                yy = y1 - tt * (y1 - y0)
                s = np.stack([np.c_[np.full_like(yy, x0), yy], np.c_[np.full_like(yy, x1), yy]], 1)
            s = s.reshape(-1, 2)
            r.lines(s, np.arange(0, s.shape[0] + 1, 2), rgba('black'), 0.5 * lw, None,
                    snap=None if self._agg else False, cap=self._cap)
        r.lines(pts, np.array([0, pts.shape[0]]), rgba('black'), lw, None,
                snap=None if self._agg else False, cap=self._cap)
        r.extents.append((pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))

    def _ticks(self, r, b, t, loc, L, horiz):
        """Tick marks and their labels (matplotlib's locators and
        ScalarFormatter); returns the labels' extents."""
        x0, y0, x1, y1 = self.ax.box_f
        vmin, vmax = b[0], b[-1]
        tp = self.ax.tick_params_
        lsize = fontsize_pt(tp.get('labelsize'))
        if self.ticks is not None:
            ticks = np.asarray(self.ticks, float)
        else:
            tick_space = int(np.floor(L * 72.0 / r.dpi / (lsize * (3 if horiz else 2))))
            ticks = np.asarray(loc.tick_values(vmin, vmax, tick_space=tick_space), float)
        lo, hi = min(vmin, vmax), max(vmin, vmax)
        ticks = ticks[(ticks >= lo - 1e-10 * abs(hi - lo)) & (ticks <= hi + 1e-10 * abs(hi - lo))]
        if isinstance(self.format, dict):
            labels = [str(self.format.get(v, '')) for v in ticks]
        elif self.format is not None:
            labels = _ticker.format_values(self.format, ticks)
        else:
            labels = _ticker.ScalarFormatter().format(ticks, vmin, vmax)[0]
        pos = np.interp(ticks, b, t)
        ax = 'x' if horiz else 'y'                      # the long axis' tick rcParams
        tl = r.px(tp.get('length', rc(ax + 'tick.major.size')))
        tpad = r.px(tp.get('pad', rc(ax + 'tick.major.pad')))
        lcol = tp.get('labelcolor', 'black')
        marks, exts = [], []
        for p, lab in zip(pos, labels):
            if horiz:
                xx = x0 + p * (x1 - x0)
                marks += [[xx, y1], [xx, y1 + tl]]
                e = r.text(lab, xx, y1 + tl + tpad, lsize, lcol, ha='center', va='top')
            else:
                yy = y1 - p * (y1 - y0)
                marks += [[x1, yy], [x1 + tl, yy]]
                e = r.text(lab, x1 + tl + tpad, yy, lsize, lcol, ha='left', va='center_baseline')
            if e is not None:
                exts.append(e)
        if marks:
            s = np.asarray(marks, float)
            w = r.px(tp.get('width', rc(ax + 'tick.major.width')))
            if self._agg:                               # Agg's tick markers: whole pixels
                sv = float(_draw.snap_value(w))
                base = s[0::2].copy()
                along, across = (0, 1) if horiz else (1, 0)
                s[:, along] = np.floor(np.repeat(base[:, along], 2) + 0.5) + sv
                e = np.floor(base[0, across] + 0.5)
                s[0::2, across] = e + sv
                s[1::2, across] = e + np.floor(tl + 0.5) + sv
            r.lines(s, np.arange(0, s.shape[0] + 1, 2), rgba(tp.get('color', 'black')), w, None,
                    cap=self._cap)
        self._tick_end = (y1 if horiz else x1) + tl
        return exts

    def _label(self, r, exts, horiz):
        """The colorbar's label beyond its tick labels, and cb.ax's title."""
        x0, y0, x1, y1 = self.ax.box_f
        if self.label:
            lk = self.label_kw
            lpad = r.px(rc('axes.labelpad') if lk.get('labelpad') is None else lk['labelpad'])
            fs = fontsize_pt(lk.get('fontsize'))
            col = lk.get('color', 'black')
            weight = lk.get('fontweight', lk.get('weight', 'normal'))
            if horiz:
                ybot = max([e[3] for e in exts], default=self._tick_end)
                r.text(self.label, 0.5 * (x0 + x1), ybot + lpad, fs, col, ha='center', va='top',
                       weight=weight)
            else:
                xr = max([e[2] for e in exts], default=self._tick_end)
                r.text(self.label, xr + lpad, 0.5 * (y0 + y1), fs, col, ha='left', va='center',
                       rotation=90.0, weight=weight)
        title = self.ax.titles.get('center')
        if title and title['text']:
            r.text(title['text'], 0.5 * (x0 + x1), y0 - r.px(rc('axes.titlepad')),
                   fontsize_pt(title['fontsize'] if title['fontsize'] is not None
                               else rc('axes.titlesize')),
                   'black', ha='center', va='baseline')


def _bar_image(cols, t, w, h, horiz):
    """The bar's colours as an RGBA block, each pixel the colour of the
    stop its centre falls in."""
    W = max(int(round(w)), 1)
    H = max(int(round(h)), 1)
    c = (np.arange(W) + 0.5) / W if horiz else 1.0 - (np.arange(H) + 0.5) / H
    line = cols[np.clip(np.searchsorted(t, c, side='right') - 1, 0, cols.shape[0] - 1)]
    shape = (H, W, 4)
    return np.ascontiguousarray(np.broadcast_to(line[None] if horiz else line[:, None], shape))


class ColorbarAxes:
    """The axes a colorbar sits in (cb.ax)."""

    def __init__(self, fig, spec, anchor, box_aspect):
        self.figure, self.spec, self.anchor, self.box_aspect = fig, spec, anchor, box_aspect
        self.tick_params_ = {}
        self.colorbar = None
        self.titles = {}

    def tick_params(self, axis='both', **kw):
        self.tick_params_.update(kw)

    def set_title(self, label, fontsize=None, **kw):
        self.titles['center'] = dict(text=label, fontsize=fontsize)

    def set_xlabel(self, label, **kw):
        if self.colorbar is not None:
            self.colorbar.set_label(label, **kw)

    set_ylabel = set_xlabel

    def _place(self, fig, W, H):
        box = self.spec.box(fig) if isinstance(self.spec, SubplotSpec) else self.spec
        aspect = self.box_aspect
        cb = self.colorbar
        if cb is not None and cb.extend != 'neither':
            # matplotlib's _ColorbarAxesLocator: the bar shrinks along its
            # length so the extensions fit in the box, keeping its thickness
            e0 = cb.extendfrac if cb.extend in ('min', 'both') else 0.0
            e1 = cb.extendfrac if cb.extend in ('max', 'both') else 0.0
            shrink = 1.0 / (1.0 + e0 + e1)
            offset = e0 * shrink
            if cb.orientation == 'vertical':
                box = _Box(box.l, box.b + offset * box.h, box.w, box.h * shrink)
                aspect = None if aspect is None else aspect * shrink
            else:
                box = _Box(box.l + offset * box.w, box.b, box.w * shrink, box.h)
                aspect = None if aspect is None else aspect / shrink
        if aspect is not None:
            box = _fit(box, aspect, self.anchor, W, H)
        self._pos = box
        x0, x1 = box.l * W, (box.l + box.w) * W
        y0, y1 = (1 - box.b - box.h) * H, (1 - box.b) * H
        self.box_f = (x0, y0, x1, y1)

    def get_position(self):
        return self._pos


def make_cax(fig, parents, location, fraction, shrink, aspect, pad, anchor, panchor,
             use_gridspec):
    """The axes for a colorbar beside parents, room taken from them as
    matplotlib's make_axes_gridspec (one subplot) or make_axes (others) does."""
    vertical = location in ('left', 'right')
    if pad is None:
        pad = 0.05 if vertical else 0.15
    defaults = {'right': ((0.0, 0.5), (1.0, 0.5)), 'left': ((1.0, 0.5), (0.0, 0.5)),
                'bottom': ((0.5, 1.0), (0.5, 0.0)), 'top': ((0.5, 0.0), (0.5, 1.0))}
    anchor = defaults[location][0] if anchor is None else anchor
    panchor = defaults[location][1] if panchor is None else panchor
    box_aspect = aspect if vertical else 1.0 / aspect
    p = parents[0]
    if use_gridspec and len(parents) == 1 and isinstance(p.spec, SubplotSpec):
        # matplotlib's make_axes_gridspec
        whs = 2 * pad / (1 - pad)
        ss = p.spec
        if location == 'right':
            gs = ss.subgridspec(1, 2, wspace=whs, hspace=0,
                                width_ratios=[1 - fraction - pad, fraction])
            main, cbs = gs[0], gs[1]
        elif location == 'left':
            gs = ss.subgridspec(1, 2, wspace=whs, hspace=0,
                                width_ratios=[fraction, 1 - fraction - pad])
            main, cbs = gs[1], gs[0]
        elif location == 'bottom':
            gs = ss.subgridspec(2, 1, hspace=whs, wspace=0,
                                height_ratios=[1 - fraction - pad, fraction])
            main, cbs = gs[0], gs[1]
        else:
            gs = ss.subgridspec(2, 1, hspace=whs, wspace=0,
                                height_ratios=[fraction, 1 - fraction - pad])
            main, cbs = gs[1], gs[0]
        a = _anchor(anchor)
        if vertical:
            cbs = cbs.subgridspec(3, 1, hspace=0, wspace=0,
                                  height_ratios=[(1 - a[1]) * (1 - shrink), shrink,
                                                 a[1] * (1 - shrink)])[1]
        else:
            cbs = cbs.subgridspec(1, 3, wspace=0, hspace=0,
                                  width_ratios=[a[0] * (1 - shrink), shrink,
                                                (1 - a[0]) * (1 - shrink)])[1]
        p.spec = main
        p.anchor = panchor
        return ColorbarAxes(fig, cbs, anchor, box_aspect)
    # matplotlib's make_axes: shrink the parents' union box
    boxes = [q.spec.box(fig) if isinstance(q.spec, SubplotSpec) else q.spec for q in parents]
    l = min(bb.l for bb in boxes)
    b = min(bb.b for bb in boxes)
    rr = max(bb.l + bb.w for bb in boxes)
    t = max(bb.b + bb.h for bb in boxes)
    pw, ph = rr - l, t - b
    if location == 'right':
        pb1 = _Box(l, b, pw * (1 - fraction - pad), ph)
        cb = _Box(l + pw * (1 - fraction), b, pw * fraction, ph)
    elif location == 'left':
        cb = _Box(l, b, pw * fraction, ph)
        pb1 = _Box(l + pw * (fraction + pad), b, pw * (1 - fraction - pad), ph)
    elif location == 'bottom':
        cb = _Box(l, b, pw, ph * fraction)
        pb1 = _Box(l, b + ph * (fraction + pad), pw, ph * (1 - fraction - pad))
    else:
        pb1 = _Box(l, b, pw, ph * (1 - fraction - pad))
        cb = _Box(l, b + ph * (1 - fraction), pw, ph * fraction)
    a = _anchor(anchor)
    if vertical:
        nh = cb.h * shrink
        cb = _Box(cb.l, cb.b + (cb.h - nh) * a[1], cb.w, nh)
    else:
        nw = cb.w * shrink
        cb = _Box(cb.l + (cb.w - nw) * a[0], cb.b, nw, cb.h)
    for q, bb in zip(parents, boxes):
        nl = pb1.l + (bb.l - l) / pw * pb1.w
        nb = pb1.b + (bb.b - b) / ph * pb1.h
        q.spec = _Box(nl, nb, bb.w / pw * pb1.w, bb.h / ph * pb1.h)
        q.anchor = panchor
    return ColorbarAxes(fig, cb, anchor, box_aspect)

