"""fig.colorbar: the bar (bands or a gradient), its extensions, outline,
ticks and label, laid out as matplotlib's Colorbar."""
import numpy as np

from . import _color, _draw, _ticker
from ._style import fontsize_pt, rc, rgba
from ._layout import SubplotSpec, _Box, _anchor, _fit
from ._fields import ContourfArtist
from ._axesbase import AxesBase


def _shade_1d(values, cmap, norm):
    """Colours of a few values, as a vector artist paints them."""
    out = np.empty((1, values.size, 4), np.uint8)
    _color.shade(values[None].astype(np.float32), cmap, norm, out=out, rounding='nearest')
    return out[0]


class _Outline:
    """cb.outline: the frame round the bar (its colour, width, visibility)."""

    def __init__(self):
        self.visible, self.color, self.linewidth = True, None, None

    def set_visible(self, v):
        self.visible = bool(v)

    def set_edgecolor(self, c):
        self.color = c

    set_color = set_edgecolor

    def set_linewidth(self, w):
        self.linewidth = float(w)

    def set_linestyle(self, ls):
        pass

    def set(self, **kw):
        for k, v in kw.items():
            getattr(self, 'set_' + {'ec': 'edgecolor', 'lw': 'linewidth',
                                    'c': 'color'}.get(k, k))(v)


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
        self._lines = []
        self._minor = None
        self.outline = _Outline()
        if not hasattr(mappable, 'norm_spec') and not isinstance(mappable, ContourfArtist):
            self.extend = 'neither'                     # contour lines: no extensions
        elif hasattr(mappable, 'norm_spec') and not isinstance(mappable, ContourfArtist):
            n = mappable.norm_spec()                    # matplotlib's colorbar widens a
            if n.kind not in ('boundary', 'nonorm') and n.vmin is not None and \
                    n.vmin == n.vmax:                   # constant field's norm, in place
                n.vmin, n.vmax = _ticker._nonsingular(float(n.vmin), float(n.vmax),
                                                      expander=0.1)
                mappable._coloring = None

    @property
    def vmin(self):
        return self._scale()[0][0]

    @property
    def vmax(self):
        return self._scale()[0][-1]

    @property
    def cmap(self):
        return getattr(self.mappable, 'cmap', None)

    @property
    def norm(self):
        return getattr(self.mappable, 'norm', None)

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
        if not hasattr(m, 'norm_spec'):                 # contour lines: no fill, a line a level
            b = np.asarray(m.levels, float)
            if b.size < 2:
                b = np.r_[b, b + 1.0] if b.size else np.array([0.0, 1.0])
            return (b, None, np.linspace(0.0, 1.0, b.size), _ticker.FixedLocator(b, nbins=10),
                    None, None)
        norm, cmap = m.norm_spec(), m.cmap
        kind = getattr(norm, 'kind', 'linear')
        under = cmap.under if lo_ext else None
        over = cmap.over if hi_ext else None
        if kind != 'boundary' and not float(norm.vmin) < float(norm.vmax):
            # matplotlib's: a constant field's bar spans nonsingular(vmin, vmax, 0.1)
            norm = norm.copy()
            norm.vmin, norm.vmax = _ticker._nonsingular(float(norm.vmin), float(norm.vmax),
                                                        expander=0.1)
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
            return b, cols, t, _ticker.LogLocator(), under, over
        b = np.linspace(vmin, vmax, n + 1)
        cols = _shade_1d(0.5 * (b[1:] + b[:-1]), cmap, norm)
        t = (b - b[0]) / (b[-1] - b[0])
        if kind == 'func':                              # matplotlib: the norm as the bar's scale
            u = np.ma.filled(np.ma.asarray(norm.func(b), float), np.nan)
            if np.isfinite(u).all() and u[-1] != u[0]:
                t = (u - u[0]) / (u[-1] - u[0])
        if kind == 'twoslope':
            vc = float(norm.vcenter)
            t = np.where(b <= vc, 0.5 * (b - vmin) / (vc - vmin),
                         0.5 + 0.5 * (b - vc) / (vmax - vc))
        if kind == 'symlog':                            # positions on the norm's scale
            u = np.asarray(_color._symlog_t(norm, b), float)
            if np.isfinite(u).all() and u[-1] != u[0]:
                t = (u - u[0]) / (u[-1] - u[0])
            return b, cols, t, _ticker.SymmetricalLogLocator(
                linthresh=getattr(norm, 'linthresh', 1.0), base=getattr(norm, 'base', 10)), \
                under, over
        return b, cols, t, _ticker.auto_locator(), under, over

    @property
    def _cap(self):
        return _draw.BUTT if self._agg else _draw.LEGACY

    def _draw(self, r):
        x0, y0, x1, y1 = self.ax.box_f
        horiz = self.orientation == 'horizontal'
        b, cols, t, loc, under, over = self._scale()
        alpha = getattr(self.mappable, 'alpha', None)
        if alpha is not None and np.ndim(alpha) == 0:   # matplotlib's: the mappable's alpha
            def faded(c):
                if c is None:
                    return None
                c = np.array(c, np.uint8)
                c[..., 3] = np.rint(c[..., 3] * float(alpha)).astype(np.uint8)
                return c
            cols, under, over = faded(cols), faded(under), faded(over)
        L = (x1 - x0) if horiz else (y1 - y0)
        ef = self.extendfrac * L                        # extension length, px
        if cols is not None:
            img, j0, i0 = _bar_image(cols, t, x0, y0, x1, y1, horiz)
            r.c.image(img, j0, i0)
        self._extensions(r, under, over, ef, horiz)
        lines = list(self._lines)
        if cols is None:                                # a contour set's lines: a line a level
            m = self.mappable
            lines.insert(0, (np.asarray(m.levels, float), m.line_colors, m.linewidths))
        for levels, colors, widths in lines:
            pos = self._locate(levels, b, t)
            for k, p in enumerate(pos):
                if not (0.0 <= p <= 1.0):
                    continue
                c = colors[k % len(colors)]
                w = r.px(float(widths[k % len(widths)]))
                if horiz:
                    seg = np.array([[x0 + p * (x1 - x0), y0], [x0 + p * (x1 - x0), y1]])
                else:
                    seg = np.array([[x0, y1 - p * (y1 - y0)], [x1, y1 - p * (y1 - y0)]])
                r.lines(seg, np.array([0, 2]), rgba(c), w, None,
                        snap=None if self._agg else False, cap=self._cap)
        if self.outline.visible:
            self._outline(r, t, under, over, ef, horiz)
        exts = self._ticks(r, b, t, loc, L, horiz)
        self._label(r, exts, horiz)

    def add_lines(self, *args, colors=None, linewidths=None, erase=True):
        """matplotlib's add_lines(levels, colors, linewidths) or
        add_lines(contour_set): lines across the bar at those values."""
        if len(args) == 1 and hasattr(args[0], 'levels'):
            cs = args[0]
            levels = np.asarray(cs.levels, float)
            colors = getattr(cs, 'line_colors', None)
            if colors is None:
                colors = ['k']
            linewidths = getattr(cs, 'linewidths', [rc('lines.linewidth')])
        else:
            levels = np.atleast_1d(np.asarray(args[0], float))
            if len(args) > 1:
                colors = args[1]
            if len(args) > 2:
                linewidths = args[2]
        cl = [colors] if colors is None or isinstance(colors, str) or (
            np.ndim(colors) == 1 and len(colors) in (3, 4) and not isinstance(colors[0], str)
        ) else list(colors)
        cl = [rgba(c if c is not None else 'k') for c in cl]
        lw = np.atleast_1d(rc('lines.linewidth') if linewidths is None else linewidths)
        if erase:
            self._lines = []
        self._lines.append((levels, cl, lw))

    def remove(self):
        """Take the colorbar away; its parents get their room back."""
        if self in self.figure.colorbars:
            self.figure.colorbars.remove(self)
        for q, spec, anchor in getattr(self.ax, '_restore', []):
            q.spec, q.anchor = spec, anchor
            q._cl_pre = None

    def set_alpha(self, a):
        self.mappable.alpha = a

    def get_ticks(self, minor=False):
        return np.asarray(self.ticks if self.ticks is not None and
                          not hasattr(self.ticks, 'tick_values') else [], float)

    def set_ticklabels_kw(self, **kw):
        self.ax.tick_params_.update(kw)

    def ax_label(self):
        return self.label

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
        o = self.outline
        lw = r.px(rc('axes.linewidth') if o.linewidth is None else o.linewidth)
        ec = rgba(rc('axes.edgecolor') if o.color is None else o.color)
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
            r.lines(s, np.arange(0, s.shape[0] + 1, 2), ec, 0.5 * lw, None,
                    snap=None if self._agg else False, cap=self._cap)
        r.lines(pts, np.array([0, pts.shape[0]]), ec, lw, None,
                snap=None if self._agg else False, cap=self._cap)
        r.extents.append((pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))

    def _ticks(self, r, b, t, loc, L, horiz):
        """Tick marks and their labels (matplotlib's locators and
        ScalarFormatter); returns the labels' extents."""
        x0, y0, x1, y1 = self.ax.box_f
        vmin, vmax = b[0], b[-1]
        tp = self.ax.tick_params_
        lsize = fontsize_pt(tp.get('labelsize'))
        kind = self._kind()
        tick_space = int(np.floor(L * 72.0 / r.dpi / (lsize * (3 if horiz else 2))))
        if self.ticks is not None and hasattr(self.ticks, 'tick_values'):
            loc = _ticker.get_locator(self.ticks)       # a Locator given as ticks=
        if self.ticks is not None and not hasattr(self.ticks, 'tick_values'):
            ticks = np.asarray(self.ticks, float)
        else:
            from ._cartesian import tick_values
            ticks = tick_values(loc, vmin, vmax, space=tick_space)
        lo, hi = min(vmin, vmax), max(vmin, vmax)
        ticks = ticks[(ticks >= lo - 1e-10 * abs(hi - lo)) & (ticks <= hi + 1e-10 * abs(hi - lo))]
        if isinstance(self.format, dict):
            labels = [str(self.format.get(v, '')) for v in ticks]
        elif self.format is not None:
            from ._cartesian import format_ticks
            labels = format_ticks(_ticker.get_formatter(self.format), ticks, vmin, vmax)[0]
        elif kind == 'log':
            from ._cartesian import format_ticks
            labels = format_ticks(_ticker.LogFormatterSciNotation(), ticks, vmin, vmax)[0]
        else:
            labels = _ticker.ScalarFormatter().format(ticks, vmin, vmax)[0]
        pos = self._locate(ticks, b, t)
        self._minor = self._minor_pos(b, t, ticks) if kind == 'log' or self.minorticks else None
        ax = 'x' if horiz else 'y'                      # the long axis' tick rcParams
        tl = r.px(tp.get('length', rc(ax + 'tick.major.size')))
        tpad = r.px(tp.get('pad', rc(ax + 'tick.major.pad')))
        tcol = tp.get('color', rc(ax + 'tick.color'))
        lcol = tp.get('labelcolor', rc(ax + 'tick.labelcolor'))
        if lcol == 'inherit':
            lcol = tcol
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
            r.lines(s, np.arange(0, s.shape[0] + 1, 2), rgba(tcol), w, None, cap=self._cap)
        if self._minor is not None and len(self._minor):    # minor ticks (log bars)
            ml = r.px(rc(ax + 'tick.minor.size'))
            mw = r.px(rc(ax + 'tick.minor.width'))
            ms = []
            for p in self._minor:
                if horiz:
                    xx = x0 + p * (x1 - x0)
                    ms += [[xx, y1], [xx, y1 + ml]]
                else:
                    yy = y1 - p * (y1 - y0)
                    ms += [[x1, yy], [x1 + ml, yy]]
            s = np.asarray(ms, float)
            r.lines(_draw.snap(s, np.arange(0, s.shape[0] + 1, 2), mw, None if self._agg else
                               False), np.arange(0, s.shape[0] + 1, 2), rgba(tcol), mw, None,
                    cap=self._cap)
        self._tick_end = (y1 if horiz else x1) + tl
        return exts

    def _kind(self):
        m = self.mappable
        if isinstance(m, ContourfArtist) or not hasattr(m, 'norm_spec'):
            return 'levels'
        return getattr(m.norm_spec(), 'kind', 'linear')

    def _locate(self, v, b, t):
        """Values -> positions along the bar (0..1)."""
        v = np.asarray(v, float)
        kind = self._kind()
        if kind == 'log' and b[0] > 0:
            return np.log(v / b[0]) / np.log(b[-1] / b[0])
        if kind == 'symlog':
            u = np.asarray(_color._symlog_t(self.mappable.norm_spec(), np.r_[b[0], b[-1], v]))
            return (u[2:] - u[0]) / (u[1] - u[0])
        return np.interp(v, b, t)

    def _minor_pos(self, b, t, major):
        """Minor tick positions: a log bar's 2..9 of each decade; else
        matplotlib's AutoMinorLocator between the major ticks."""
        lo, hi = min(b[0], b[-1]), max(b[0], b[-1])
        if self._kind() == 'log' and lo > 0:
            e0, e1 = int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi)))
            v = np.array([k * 10.0 ** e for e in range(e0, e1 + 1) for k in range(2, 10)])
            if (e1 - e0) > 6:
                return None                             # matplotlib thins them out: none
        else:
            from ._cartesian import tick_values
            v = tick_values(_ticker.AutoMinorLocator(), lo, hi, major=major)
        v = v[(v >= lo) & (v <= hi)]
        return self._locate(v, b, t)

    def _label(self, r, exts, horiz):
        """The colorbar's label beyond its tick labels, and cb.ax's title."""
        x0, y0, x1, y1 = self.ax.box_f
        if self.label:
            lk = self.label_kw
            lpad = r.px(rc('axes.labelpad') if lk.get('labelpad') is None else lk['labelpad'])
            fs = fontsize_pt(lk.get('fontsize'))
            col = lk.get('color', rc('axes.labelcolor'))
            weight = lk.get('fontweight', lk.get('weight', 'normal'))
            if horiz:
                ybot = max([e[3] for e in exts], default=self._tick_end)
                r.text(self.label, 0.5 * (x0 + x1), ybot + lpad, fs, col, ha='center', va='top',
                       weight=weight)
            else:
                xr = max([e[2] for e in exts], default=self._tick_end)
                r.text(self.label, xr + lpad, 0.5 * (y0 + y1), fs, col, ha='left', va='center',
                       rotation=90.0, weight=weight)
        for loc, t in self.ax.titles.items():           # cb.ax.set_title's, as on any axes
            if not t['text']:
                continue
            x, ha = {'center': (0.5 * (x0 + x1), 'center'), 'left': (x0, 'left'),
                     'right': (x1, 'right')}[loc]
            r.text(t['text'], x, y0 - r.px(rc('axes.titlepad') if t['pad'] is None else t['pad']),
                   fontsize_pt(rc('axes.titlesize') if t['fontsize'] is None else t['fontsize']),
                   t['color'], ha=ha, va='baseline', weight=t['weight'], style=t['style'])


def _bar_image(cols, t, x0, y0, x1, y1, horiz):
    """The bar's colours as an RGBA block and its top-left pixel: the
    pixels from round(x0) to round(x1) (y likewise), as matplotlib's
    snapped mesh covers them, each the colour of the stop its centre is in."""
    j0, j1 = int(np.floor(x0 + 0.5)), int(np.floor(x1 + 0.5))
    i0, i1 = int(np.floor(y0 + 0.5)), int(np.floor(y1 + 0.5))
    W, H = max(j1 - j0, 1), max(i1 - i0, 1)
    if horiz:
        c = (j0 + np.arange(W) + 0.5 - x0) / max(x1 - x0, 1e-9)
    else:
        c = (y1 - (i0 + np.arange(H) + 0.5)) / max(y1 - y0, 1e-9)
    line = cols[np.clip(np.searchsorted(t, c, side='right') - 1, 0, cols.shape[0] - 1)]
    shape = (H, W, 4)
    return (np.ascontiguousarray(np.broadcast_to(line[None] if horiz else line[:, None], shape)),
            j0, i0)


def _tick_kw(kw):
    """Text keywords of set_yticklabels as tick_params' label ones."""
    out = {}
    for k, v in kw.items():
        k = {'fontsize': 'labelsize', 'size': 'labelsize', 'color': 'labelcolor',
             'rotation': 'labelrotation'}.get(k, k)
        if k in ('labelsize', 'labelcolor', 'labelrotation'):
            out[k] = v
    return out


class _CbAxis:
    """cb.ax.yaxis / xaxis: the colorbar's long axis -- its locator,
    formatter, label, tick parameters."""

    def __init__(self, cax):
        self.cax = cax

    @property
    def _cb(self):
        return self.cax.colorbar

    def set_major_locator(self, loc):
        self._cb.ticks = loc

    def set_major_formatter(self, fmt):
        self._cb.format = fmt

    def set_minor_locator(self, loc):
        self._cb.minorticks = True

    def set_ticks(self, ticks, labels=None, **kw):
        self._cb.set_ticks(ticks, labels)

    def set_ticklabels(self, labels, **kw):
        self._cb.set_ticklabels(labels)

    def set_label_text(self, s, **kw):
        self._cb.set_label(s, **kw)

    def set_tick_params(self, which='major', **kw):
        self.cax.tick_params(**kw)

    def set_ticks_position(self, pos):
        pass

    def set_label_position(self, pos):
        pass

    def get_ticklabels(self):
        return []

    def get_major_locator(self):
        return self._cb.ticks


class ColorbarAxes:
    """The axes a colorbar sits in (cb.ax)."""

    zorder = 0

    def __init__(self, fig, spec, anchor, box_aspect):
        self.figure, self.spec, self.anchor, self.box_aspect = fig, spec, anchor, box_aspect
        self.tick_params_ = {}
        self.colorbar = None
        self.titles = {}

    @classmethod
    def adopt(cls, ax):
        """Plain axes made a colorbar's axes, in place: matplotlib's
        fig.colorbar(cax=ax) draws the bar in ax itself (cb.ax is ax), so
        what is set on ax before or after -- title, zorder -- still counts."""
        titles, zorder = ax.titles, ax.zorder
        fig, spec = ax.figure, ax.spec
        ax.__dict__.clear()
        ax.__class__ = cls
        cls.__init__(ax, fig, spec, 'C', None)
        ax.titles, ax.zorder = titles, zorder
        return ax

    def tick_params(self, axis='both', **kw):
        if 'colors' in kw:                              # matplotlib's: ticks and labels
            c = kw.pop('colors')
            kw.setdefault('color', c)
            kw.setdefault('labelcolor', c)
        self.tick_params_.update(kw)

    set_title = AxesBase.set_title
    get_title = AxesBase.get_title
    set_zorder = AxesBase.set_zorder
    get_zorder = AxesBase.get_zorder

    def set_xlabel(self, label, **kw):
        if self.colorbar is not None:
            self.colorbar.set_label(label, **kw)

    set_ylabel = set_xlabel

    @property
    def xaxis(self):
        return _CbAxis(self)

    yaxis = xaxis

    def set_yticks(self, ticks, labels=None, **kw):
        if self.colorbar is not None:
            self.colorbar.set_ticks(ticks, labels)
        if kw:
            self.tick_params(**_tick_kw(kw))

    set_xticks = set_yticks

    def set_yticklabels(self, labels, **kw):
        if self.colorbar is not None:
            self.colorbar.set_ticklabels(labels)
        if kw:
            self.tick_params(**_tick_kw(kw))

    set_xticklabels = set_yticklabels

    def get_yticks(self):
        return self.colorbar.get_ticks() if self.colorbar is not None else np.array([])

    get_xticks = get_yticks

    def minorticks_on(self):
        if self.colorbar is not None:
            self.colorbar.minorticks_on()

    def minorticks_off(self):
        if self.colorbar is not None:
            self.colorbar.minorticks_off()

    def set_facecolor(self, c):
        pass

    def set_visible(self, v):
        if not v and self.colorbar is not None:
            self.colorbar.remove()

    def _place(self, fig, W, H):
        box = self.spec.box(fig) if isinstance(self.spec, SubplotSpec) else self.spec
        self._orig_pos = box                            # before the aspect (constrained layout)
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
        ox, oy = getattr(fig, '_org', (0.0, 0.0))     # the figure's place on the canvas
        x0, x1 = box.l * W + ox, (box.l + box.w) * W + ox
        y0, y1 = (1 - box.b - box.h) * H + oy, (1 - box.b) * H + oy
        self.box_f = (x0, y0, x1, y1)

    def get_position(self):
        return self._pos


def make_cax(fig, parents, location, fraction, shrink, aspect, pad, anchor, panchor,
             use_gridspec):
    """The axes for a colorbar beside parents, room taken from them as
    matplotlib's make_axes_gridspec (one subplot) or make_axes (others) does."""
    vertical = location in ('left', 'right')
    if pad is None:                                     # matplotlib's loc_settings
        pad = 0.05 if getattr(fig, '_layout', None) == 'constrained' else \
            {'left': 0.10, 'right': 0.05, 'top': 0.05, 'bottom': 0.15}[location]
    defaults = {'right': ((0.0, 0.5), (1.0, 0.5)), 'left': ((1.0, 0.5), (0.0, 0.5)),
                'bottom': ((0.5, 1.0), (0.5, 0.0)), 'top': ((0.5, 0.0), (0.5, 1.0))}
    anchor = defaults[location][0] if anchor is None else anchor
    panchor = defaults[location][1] if panchor is None else panchor
    box_aspect = aspect if vertical else 1.0 / aspect
    p = parents[0]
    if getattr(fig, '_layout', None) == 'constrained':
        # the constrained layout places it (_constrained): the parents keep
        # their room; until then, where make_axes would put it
        boxes = [q.spec.box(fig) if isinstance(q.spec, SubplotSpec) else q.spec
                 for q in parents]
        l, b = min(bb.l for bb in boxes), min(bb.b for bb in boxes)
        rr, t = max(bb.l + bb.w for bb in boxes), max(bb.b + bb.h for bb in boxes)
        pw, ph = rr - l, t - b
        # matplotlib's make_axes box, inside the parents' (which keep theirs)
        box0 = {'right': _Box(l + pw * (1 - fraction), b, pw * fraction, ph),
                'left': _Box(l, b, pw * fraction, ph),
                'bottom': _Box(l, b, pw, ph * fraction),
                'top': _Box(l, b + ph * (1 - fraction), pw, ph * fraction)}[location]
        a = _anchor(anchor)
        if vertical:
            nh = box0.h * shrink
            box0 = _Box(box0.l, box0.b + (box0.h - nh) * a[1], box0.w, nh)
        else:
            nw = box0.w * shrink
            box0 = _Box(box0.l + (box0.w - nw) * a[0], box0.b, nw, box0.h)
        if location == 'right':
            pb1 = _Box(l, b, pw * (1 - fraction - pad), ph)
        elif location == 'left':
            pb1 = _Box(l + pw * (fraction + pad), b, pw * (1 - fraction - pad), ph)
        elif location == 'bottom':
            pb1 = _Box(l, b + ph * (fraction + pad), pw, ph * (1 - fraction - pad))
        else:
            pb1 = _Box(l, b, pw, ph * (1 - fraction - pad))
        for q, bb in zip(parents, boxes):
            q._cl_pre = _Box(pb1.l + (bb.l - l) / pw * pb1.w, pb1.b + (bb.b - b) / ph * pb1.h,
                             bb.w / pw * pb1.w, bb.h / ph * pb1.h)
        cax = ColorbarAxes(fig, box0, anchor, box_aspect)
        cax._cl = dict(parents=list(parents), location=location, fraction=fraction,
                       shrink=shrink, pad=pad, anchor=anchor, box0=box0)
        return cax
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
        restore = [(p, p.spec, p.anchor)]
        p.spec = main
        p.anchor = panchor
        cax = ColorbarAxes(fig, cbs, anchor, box_aspect)
        cax._restore = restore
        return cax
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
    restore = [(q, q.spec, q.anchor) for q in parents]
    for q, bb in zip(parents, boxes):
        nl = pb1.l + (bb.l - l) / pw * pb1.w
        nb = pb1.b + (bb.b - b) / ph * pb1.h
        q.spec = _Box(nl, nb, bb.w / pw * pb1.w, bb.h / ph * pb1.h)
        q.anchor = panchor
    cax = ColorbarAxes(fig, cb, anchor, box_aspect)
    cax._restore = restore
    return cax

