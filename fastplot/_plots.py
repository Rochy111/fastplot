"""matplotlib's everyday plot types on plain axes -- bar, barh, hist,
fill_between(x), errorbar, step, stairs, axhline/axvline, axhspan/axvspan,
hlines/vlines, stem, stackplot, fill -- built on two collection artists:

    PolyCollection   filled polygons (bars, fills, spans): one fill per
                     colour, one tiled stroke for every edge
    LineCollection   segments (error bars, hlines, stems): one tiled draw

so a histogram of 500 bars costs what one does. Their colours, widths and
defaults are matplotlib's (rcParams patch.*, lines.*, errorbar.capsize,
hist.bins); default colours come from the axes' own cycles, one for lines
and one for patches, as matplotlib keeps them.
"""
import numpy as np

from . import _draw, _jit
from ._jit import capped, njit, prange
from ._style import _dashes, rc, rgba
from ._artists import Artist, _axis_bounds
from ._axesbase import _parse_fmt



def _colors(c, n, alpha=None, default=None):
    """n RGBA rows from one colour spec or a sequence of them (cycled)."""
    if c is None:
        c = default
    if c is None or (isinstance(c, str) and c.lower() in ('none', 'never')):
        return np.zeros((n, 4), np.uint8)
    if isinstance(c, str) or (np.ndim(c) == 1 and len(c) in (3, 4) and
                              not isinstance(c[0], str)):
        return np.repeat(rgba(c, alpha)[None], n, axis=0)
    cl = list(c)
    return np.array([rgba(cl[k % len(cl)], alpha) for k in range(n)], np.uint8)


def _broadcast(v, n):
    a = np.asarray(v, dtype=object if isinstance(v, (list, tuple)) and v and
                   isinstance(v[0], str) else float)
    return np.broadcast_to(a, (n,)) if a.ndim else np.resize(a, n)


HIST_TASKS = 8                      # parallel tasks of the histogram kernels, a thread


@njit(cache=True, nogil=True, parallel=True)
def _finite_range(v, out):
    """out[c] = (min, max) of the finite values of chunk c of len(out)
    (inf, -inf: none)."""
    n, m = v.size, out.shape[0]
    for c in prange(m):
        lo, hi = np.inf, -np.inf
        for i in range(c * n // m, (c + 1) * n // m):
            x = v[i]
            if np.isfinite(x):
                lo = min(lo, x)
                hi = max(hi, x)
        out[c, 0] = lo
        out[c, 1] = hi


@njit(cache=True, nogil=True, parallel=True)
def _bin_counts(v, edges, out):
    """np.histogram(v, edges)'s counts: bin k holds e[k] <= x < e[k+1], the
    last one x == e[-1] too; NaN and values outside are left out. Chunk c's
    in out[c]. The bin is guessed as if the edges were even, then walked to
    the one whose edges hold x -- exact for any edges, a step or none for
    even ones. Chunks: len(out)."""
    n, m = v.size, out.shape[0]
    nb = edges.size - 1
    lo, hi = edges[0], edges[nb]
    scale = nb / (hi - lo) if hi > lo else 0.0
    for c in prange(m):
        for i in range(c * n // m, (c + 1) * n // m):
            x = v[i]
            if not (x >= lo and x <= hi):
                continue
            if x == hi:
                k = nb - 1
            else:
                g = (x - lo) * scale
                if not g >= 0.0:                        # (NaN: an infinite edge)
                    k = 0
                elif g >= nb - 1:
                    k = nb - 1
                else:
                    k = np.int64(g)
                while k > 0 and edges[k] > x:           # edges[k] <= x < edges[k + 1]
                    k -= 1
                while k + 1 < nb and edges[k + 1] <= x:
                    k += 1
            out[c, k] += 1


@capped
def _histogram(v, edges, w=None):
    """np.histogram(v, edges, weights=w)[0] as floats (unweighted: the same
    integers, in parallel)."""
    if w is not None:
        m = np.isfinite(v)
        return np.histogram(v[m], edges, weights=w[m])[0].astype(float)
    out = np.zeros((HIST_TASKS * _jit.team(), edges.size - 1), np.int64)
    if v.size:
        _bin_counts(v, edges, out)
    return out.sum(axis=0).astype(float)


@capped
def _data_range(xs):
    """(min, max) of the finite values of all xs, or None."""
    lo, hi = np.inf, -np.inf
    out = np.empty((HIST_TASKS * _jit.team(), 2))
    for v in xs:
        if v.size:
            _finite_range(v, out)
            lo, hi = min(lo, out[:, 0].min()), max(hi, out[:, 1].max())
    return (lo, hi) if lo <= hi else None


class _Bounds:
    """Explicit autoscaling bounds: per axis (values) or None (no say)."""

    def __init__(self, xs=None, ys=None, stickyx=(), stickyy=()):
        self.x, self.y = xs, ys
        self.sx, self.sy = tuple(stickyx), tuple(stickyy)

    def get(self):
        nobody = (np.inf, -np.inf, np.inf, ())
        bx = _axis_bounds(self.x, self.sx) if self.x is not None else nobody[:3] + (self.sx,)
        by = _axis_bounds(self.y, self.sy) if self.y is not None else nobody[:3] + (self.sy,)
        return bx, by


class PolyCollection(Artist):
    """Filled polygons in a coordinate system: bars, fills, spans."""

    zorder = 1.0

    def __init__(self, ax, polys, src, facecolors, edgecolors, linewidths, linestyle='solid',
                 alpha=None, zorder=None, label=None, bounds=None, hatch=None):
        super().__init__(ax, zorder, alpha)
        self.polys = [np.asarray(p, np.float64).reshape(-1, 2) for p in polys]
        self.src = src
        n = len(self.polys)
        self.facecolors = _colors(facecolors, n, alpha)
        self.edgecolors = _colors(edgecolors, n, alpha)
        self.linewidths = np.broadcast_to(np.asarray(linewidths, float), (n,)).copy() if n else \
            np.zeros(0)
        self.linestyle = linestyle
        self.label = label
        self._bounds = bounds
        self.hatch = hatch
        self.open = False                               # an outline only (step, stairs)
        self.rectangles = False                         # matplotlib Rectangles (bar, spans)
        self.hatchcolor = None                          # None: 'edge' (rcParams hatch.color)

    def data_bounds(self):
        if self._bounds is not None:
            return self._bounds.get()
        if self.src != 'data' or not self.polys:
            return None
        v = np.concatenate(self.polys)
        return _axis_bounds(v[:, 0]), _axis_bounds(v[:, 1])

    def legend_style(self):
        return dict(kind='patch', facecolor=self.facecolors[0] if len(self.facecolors) else None,
                    edgecolor=self.edgecolors[0] if len(self.edgecolors) else None,
                    linewidth=self.linewidths[0] if len(self.linewidths) else 0.0,
                    linestyle=self.linestyle)

    def render(self, r, ax):
        if not self.polys:
            return
        polys = self.polys
        if self.rectangles and self.src == 'data' and hasattr(ax, '_interp_polys'):
            polys = ax._interp_polys(polys)             # polar bars: wedges, as matplotlib's
        lens = np.array([len(p) for p in polys])
        off = np.r_[0, np.cumsum(lens)].astype(np.int64)
        v = np.concatenate(polys)
        px, py = ax._points_px(v[:, 0], v[:, 1], self.src)
        xy = np.column_stack([px, py])
        if ax._snap is not False:                       # fill and edge on one snapped path
            edge = (self.edgecolors[:, 3] > 0) & (self.linewidths > 0)
            xy = _draw.snap(xy, off, np.where(edge, r.px(self.linewidths), 0.0), ax._snap)
        fc = self.facecolors
        if fc[:, 3].any():
            # one fill per distinct colour (polygons of a colour together, in order)
            keys = fc.view(np.uint32).ravel()
            for key in dict.fromkeys(keys.tolist()):
                sel = np.nonzero(keys == key)[0]
                if fc[sel[0], 3] == 0:
                    continue
                idx = np.concatenate([np.arange(off[k], off[k + 1]) for k in sel])
                r.c.polygons(xy[idx], np.r_[0, np.cumsum(lens[sel])], fc[sel[0]], ax.box_clip,
                             'nonzero')
        if self.hatch and not self.open:
            from ._hatch import color_for
            hc = color_for(self.edgecolors[0] if len(self.edgecolors) else None,
                           self.hatchcolor, self.alpha, collection=not self.rectangles)
            r.c.hatch(xy, off, self.hatch, hc, r.dpi, ax.box_clip)
        ec, lw = self.edgecolors, self.linewidths
        draw = (ec[:, 3] > 0) & (lw > 0)
        if draw.any():
            sel = np.nonzero(draw)[0]
            if self.open:
                ring = np.concatenate([np.arange(off[k], off[k + 1]) for k in sel])
                roff = np.r_[0, np.cumsum(lens[sel])].astype(np.int64)
            else:
                ring = np.concatenate([np.r_[np.arange(off[k], off[k + 1]), off[k]]
                                       for k in sel])
                roff = np.r_[0, np.cumsum(lens[sel] + 1)].astype(np.int64)
            d = _dashes(self.linestyle, float(lw[sel].max()))
            styles = _draw.Styles(ec[sel], r.px(lw[sel]),
                                  None if d is None else [tuple(r.px(x) for x in d)],
                                  caps=_draw.BUTT if ax._agg else _draw.LEGACY)
            segs = _draw.segments(xy[ring], roff, np.arange(sel.size, dtype=np.int32))
            r.c.segments(segs, styles, ax.box_clip)


class LineCollection(Artist):
    """Line segments (or polylines) in a coordinate system, each with its
    own colour and width."""

    zorder = 2.0

    def __init__(self, ax, lines, src, colors, linewidths, linestyle='solid', alpha=None,
                 zorder=None, label=None, bounds=None):
        super().__init__(ax, zorder, alpha)
        self.lines = [np.asarray(p, np.float64).reshape(-1, 2) for p in lines]
        self.src = src
        n = len(self.lines)
        self.colors = _colors(colors, n, alpha)
        self.linewidths = np.broadcast_to(np.asarray(linewidths, float), (n,)).copy() if n else \
            np.zeros(0)
        self.linestyle = linestyle
        self.label = label
        self._bounds = bounds

    def data_bounds(self):
        if self._bounds is not None:
            return self._bounds.get()
        if self.src != 'data' or not self.lines:
            return None
        v = np.concatenate(self.lines)
        return _axis_bounds(v[:, 0]), _axis_bounds(v[:, 1])

    def legend_style(self):
        return dict(kind='line', color=self.colors[0] if len(self.colors) else None,
                    linewidth=self.linewidths[0] if len(self.linewidths) else 0.0,
                    linestyle=self.linestyle)

    def render(self, r, ax):
        if not self.lines:
            return
        lens = np.array([len(p) for p in self.lines])
        off = np.r_[0, np.cumsum(lens)].astype(np.int64)
        v = np.concatenate(self.lines)
        px, py = ax._points_px(v[:, 0], v[:, 1], self.src)
        n = len(self.lines)
        d = _dashes(self.linestyle, float(self.linewidths.max()) if n else 1.0)
        styles = _draw.Styles(self.colors, r.px(self.linewidths),
                              None if d is None else [tuple(r.px(x) for x in d)],
                              caps=_draw.BUTT if ax._agg else _draw.LEGACY)
        xy = _draw.snap(np.column_stack([px, py]), off, r.px(self.linewidths), ax._snap)
        segs = _draw.segments(xy, off, np.arange(n, dtype=np.int32))
        r.c.segments(segs, styles, ax.box_clip)


class Handle:
    """One patch of a collection (a bar), for the calls made on matplotlib's
    Rectangles: colours, geometry."""

    def __init__(self, coll, k, x, y, w, h):
        self.coll, self.k = coll, k
        self._x, self._y, self._w, self._h = x, y, w, h

    def set_facecolor(self, c):
        self.coll.facecolors[self.k] = rgba(c, self.coll.alpha)

    def set_edgecolor(self, c):
        self.coll.edgecolors[self.k] = rgba(c, self.coll.alpha)

    def set_color(self, c):
        self.set_facecolor(c)
        self.set_edgecolor(c)

    def set_linewidth(self, w):
        self.coll.linewidths[self.k] = w

    def set_alpha(self, a):
        self.coll.facecolors[self.k, 3] = np.uint8(round(float(a) * 255))

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def get_width(self):
        return self._w

    def get_height(self):
        return self._h

    def get_xy(self):
        return (self._x, self._y)

    def get_facecolor(self):
        return tuple(self.coll.facecolors[self.k] / 255.0)


class _Container(tuple):
    """matplotlib's Container: a tuple of artists with a label (the
    legend's entry: containers come after the axes' own artists)."""
    visible = True
    label = None

    def get_label(self):
        return self.label

    def set_label(self, s):
        self.label = s


class BarContainer(_Container):
    """bar()'s result: its patches, as matplotlib's BarContainer."""

    def __new__(cls, patches, datavalues=None, orientation='vertical', errorbar=None, label=None):
        obj = super().__new__(cls, patches)
        obj.patches = list(patches)
        obj.datavalues = datavalues
        obj.orientation = orientation
        obj.errorbar = errorbar
        obj.label = label
        return obj

    def legend_style(self):
        return self.patches[0].coll.legend_style()


class ErrorbarContainer(_Container):
    """errorbar()'s result: (data_line, caplines, barlinecols)."""

    def __new__(cls, parts, has_xerr=False, has_yerr=False, label=None):
        obj = super().__new__(cls, parts)
        obj.lines = parts
        obj.has_xerr, obj.has_yerr = has_xerr, has_yerr
        obj.label = label
        return obj

    def legend_style(self):
        line, caps, bars = self
        b = bars[0]
        return dict(kind='errorbar', line=None if line is None else line.legend_style(),
                    bar=dict(color=b.colors[0], linewidth=b.linewidths[0]),
                    cap=caps[0].legend_style() if caps else None, xerr=self.has_xerr,
                    yerr=self.has_yerr)


class StemContainer(_Container):
    """stem()'s result: (markerline, stemlines, baseline)."""

    def __new__(cls, parts, label=None):
        obj = super().__new__(cls, parts)
        obj.markerline, obj.stemlines, obj.baseline = parts
        obj.label = label
        return obj

    def legend_style(self):
        s = self.stemlines
        return dict(kind='stem', marker=self.markerline.legend_style(),
                    stem=dict(color=s.colors[0], linewidth=s.linewidths[0],
                              linestyle=s.linestyle),
                    base=self.baseline.legend_style())


class PlotMixin:
    """The plot types above, as Axes methods (matplotlib's signatures)."""

    # -- bars
    def bar(self, x, height, width=0.8, bottom=None, *, align='center', color=None,
            edgecolor=None, linewidth=None, tick_label=None, label=None, xerr=None, yerr=None,
            ecolor=None, capsize=None, error_kw=None, log=False, alpha=None, zorder=None,
            orientation='vertical', hatch=None, **kw):
        lw = kw.pop('lw', linewidth)
        color = kw.pop('facecolor', kw.pop('fc', color))
        edgecolor = kw.pop('ec', edgecolor)
        kw.pop('antialiased', None)
        if kw:
            raise TypeError('bar: unsupported keyword(s): %s' % ', '.join(sorted(kw)))
        if orientation == 'vertical':
            x, height = self._xy(x, height)
        else:
            height, x = self._xy(height, x)
        x = np.atleast_1d(np.asarray(x, float))
        height = np.atleast_1d(np.asarray(height, float))
        n = max(x.size, height.size)
        x, height = np.broadcast_to(x, (n,)), np.broadcast_to(height, (n,))
        width = np.broadcast_to(np.asarray(width, float), (n,))
        bottom = np.zeros(n) if bottom is None else np.broadcast_to(np.asarray(bottom, float), (n,))
        if align == 'center':
            left = x - width / 2.0
        elif align == 'edge':
            left = x
        else:
            raise ValueError("align must be 'center' or 'edge'")
        if color is None:
            color = self._next_color('patches')
        if orientation == 'vertical':
            x0, x1, y0, y1 = left, left + width, bottom, bottom + height
            stick = (_Bounds(np.r_[x0, x1], np.r_[y0, y1], (), np.unique(bottom)))
        else:
            x0, x1, y0, y1 = bottom, bottom + height, left, left + width
            stick = _Bounds(np.r_[x0, x1], np.r_[y0, y1], np.unique(bottom), ())
        polys = [np.array([[a, c], [b, c], [b, d], [a, d]]) for a, b, c, d in zip(x0, x1, y0, y1)]
        if log:
            (self.set_yscale if orientation == 'vertical' else self.set_xscale)('log')
        coll = self._add(PolyCollection(self, polys, 'data', color, edgecolor or 'none',
                                        rc('patch.linewidth') if lw is None else lw, alpha=alpha,
                                        zorder=zorder, bounds=stick, hatch=hatch))
        coll.rectangles = True
        handles = [Handle(coll, k, *(x0[k], y0[k], x1[k] - x0[k], y1[k] - y0[k]))
                   for k in range(n)]
        err = None
        if xerr is not None or yerr is not None:
            ekw = dict(error_kw or {})
            ekw.setdefault('ecolor', ecolor)
            ekw.setdefault('capsize', capsize)
            cx = (x0 + x1) / 2 if orientation == 'vertical' else x1
            cy = y1 if orientation == 'vertical' else (y0 + y1) / 2
            err = self.errorbar(cx, cy, yerr=yerr, xerr=xerr, fmt='none', **ekw)
        if tick_label is not None:
            ticks = (x0 + x1) / 2 if orientation == 'vertical' else (y0 + y1) / 2
            (self.set_xticks if orientation == 'vertical' else self.set_yticks)(
                ticks, np.broadcast_to(np.asarray(tick_label, object), (n,)))
        c = BarContainer(handles, height, orientation, err, label)
        self.containers.append(c)
        return c

    def barh(self, y, width, height=0.8, left=None, *, align='center', **kw):
        return self.bar(y, width, height, left, align=align, orientation='horizontal', **kw)

    def bar_label(self, container, labels=None, *, fmt='%g', label_type='edge', padding=0,
                  **kw):
        """matplotlib's bar_label: each bar's value at its end (or centre)."""
        from ._ticker import format_values
        out = []
        vals = np.asarray(container.datavalues, float)
        texts = labels if labels is not None else format_values(fmt, vals)
        vert = container.orientation == 'vertical'
        for h, v, t in zip(container.patches, vals, texts):
            if vert:
                x = h.get_x() + h.get_width() / 2
                y = h.get_y() + (h.get_height() if label_type == 'edge' else h.get_height() / 2)
                va = ('bottom' if v >= 0 else 'top') if label_type == 'edge' else 'center'
                tr = self._offset((0, padding if v >= 0 else -padding))
                out.append(self.text(x, y, t, ha='center', va=va, transform=tr, **kw))
            else:
                x = h.get_x() + (h.get_width() if label_type == 'edge' else h.get_width() / 2)
                y = h.get_y() + h.get_height() / 2
                ha = ('left' if v >= 0 else 'right') if label_type == 'edge' else 'center'
                tr = self._offset((padding if v >= 0 else -padding, 0))
                out.append(self.text(x, y, t, ha=ha, va='center', transform=tr, **kw))
        return out

    def _offset(self, pts):
        from . import transforms
        return transforms.offset_copy(self.transData, self.figure, x=pts[0], y=pts[1],
                                      units='points')

    # -- histograms
    def hist(self, x, bins=None, range=None, density=False, weights=None, cumulative=False,
             bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None,
             log=False, color=None, label=None, stacked=False, alpha=None, zorder=None,
             edgecolor=None, linewidth=None, **kw):
        """matplotlib's hist: (n, bins, patches)."""
        lw = kw.pop('lw', linewidth)
        kw.pop('density_kw', None)
        if kw:
            raise TypeError('hist: unsupported keyword(s): %s' % ', '.join(sorted(kw)))
        xs = x if (isinstance(x, (list, tuple)) and x and np.ndim(x[0]) > 0) else [x]
        if isinstance(x, np.ndarray) and x.ndim == 2:
            xs = [x[:, k] for k in builtin_range(x.shape[1])]
        xs = [np.ascontiguousarray(np.asarray(v, float).ravel()) for v in xs]
        ws = [None] * len(xs) if weights is None else (
            [np.asarray(weights, float).ravel()] if len(xs) == 1 else
            [np.asarray(w, float).ravel() for w in weights])
        if range is None:
            range = _data_range(xs)
        bins = 10 if bins is None else bins
        if isinstance(bins, str):                       # 'auto', 'fd', ...: the data decide
            data = np.concatenate([v[np.isfinite(v)] for v in xs]) if xs else np.empty(0)
        else:
            data = np.empty(0)                          # a count or edges: only the range
        edges = np.asarray(np.histogram_bin_edges(data, bins, range), float)
        counts = [_histogram(v, edges, w) for v, w in zip(xs, ws)]
        if stacked:
            counts = list(np.cumsum(counts, axis=0))
        if density:
            widths = np.diff(edges)
            counts = [c / (c.sum() * widths) if not stacked else
                      c / (counts[-1].sum() * widths) for c in counts]
        if cumulative:
            counts = [np.cumsum(c[::-1])[::-1] if cumulative < 0 else np.cumsum(c)
                      for c in counts]
        ncol = len(xs)
        colors = color if (color is not None and ncol > 1) else [color] * ncol
        if color is None:                               # matplotlib's hist: the lines' cycle
            colors = [self._next_color('lines') for _ in [None] * ncol]
        labels = label if (isinstance(label, (list, tuple)) and ncol > 1) else \
            [label] + [None] * (ncol - 1)
        patches = []
        vert = orientation == 'vertical'
        base = np.zeros(edges.size - 1) if bottom is None else \
            np.broadcast_to(np.asarray(bottom, float), (edges.size - 1,))
        if histtype.startswith('bar'):
            totwidth = np.diff(edges)
            dr = 1.0 if rwidth is None else min(1.0, max(0.0, rwidth))
            if ncol > 1 and not stacked and histtype == 'bar':
                width = dr * totwidth / ncol
                boffset = -0.5 * dr * totwidth * (1 - 1 / ncol)
            else:
                width = dr * totwidth
                boffset = np.zeros_like(totwidth)
            prev = base
            for k, c in enumerate(counts):
                h = c - (counts[k - 1] if stacked and k else 0)
                center = edges[:-1] + 0.5 * totwidth + boffset + (width * k if ncol > 1 and
                                                                  not stacked else 0)
                kwb = dict(color=colors[k], alpha=alpha, zorder=zorder,
                           edgecolor=edgecolor, linewidth=lw, log=log)
                if vert:
                    bc = self.bar(center, h, width, prev if stacked else base, **kwb)
                else:
                    bc = self.barh(center, h, width, prev if stacked else base, **kwb)
                self.containers.remove(bc)
                bc.patches[0].coll.label = labels[k]
                patches.append(bc)
                prev = c + (base if stacked else 0)
        else:                                           # step, stepfilled
            filled = histtype == 'stepfilled'
            for k, c in enumerate(counts):
                xe = np.repeat(edges, 2)
                yv = np.r_[0.0, np.repeat(c, 2), 0.0] + np.r_[base[0], np.repeat(base, 2),
                                                              base[-1]]
                col = colors[k] if colors[k] is not None else self._next_color('patches')
                pts = np.c_[xe, yv] if vert else np.c_[yv, xe]
                lo = np.r_[np.repeat(base, 2)] if stacked or bottom is not None else None
                bnd = _Bounds(pts[:, 0], pts[:, 1], () if vert else (0.0,),
                              (0.0,) if vert else ())
                if filled:
                    poly = pts if lo is None else np.r_[pts[1:-1], (np.c_[xe[1:-1], lo] if vert
                                                                    else np.c_[lo, xe[1:-1]])[::-1]]
                    coll = PolyCollection(self, [poly], 'data', col, edgecolor or 'none',
                                          rc('patch.linewidth') if lw is None else lw, alpha=alpha,
                                          zorder=zorder, label=labels[k], bounds=bnd)
                else:
                    coll = PolyCollection(self, [pts], 'data', 'none', col,
                                          rc('patch.linewidth') if lw is None else lw, alpha=alpha,
                                          zorder=2.0 if zorder is None else zorder,  # a line's
                                          label=labels[k], bounds=bnd)
                    coll.open = True
                patches.append([self._add(coll)])
        if log:
            (self.set_yscale if vert else self.set_xscale)('log')
        n = counts[0] if ncol == 1 else counts
        return n, edges, (patches[0] if ncol == 1 else patches)

    # -- fills
    def fill_between(self, x, y1, y2=0, where=None, interpolate=False, step=None, *,
                     color=None, alpha=None, label=None, zorder=None, **kw):
        return self._fill_between(x, y1, y2, where, interpolate, step, 'x', color, alpha, label,
                                  zorder, kw)

    def fill_betweenx(self, y, x1, x2=0, where=None, step=None, interpolate=False, *,
                      color=None, alpha=None, label=None, zorder=None, **kw):
        return self._fill_between(y, x1, x2, where, interpolate, step, 'y', color, alpha, label,
                                  zorder, kw)

    def _fill_between(self, t, d1, d2, where, interpolate, step, ind, color, alpha, label,
                      zorder, kw):
        """matplotlib's _fill_between_x_or_y: a polygon per run of `where`."""
        fc = kw.pop('facecolor', kw.pop('fc', color))
        ec = kw.pop('edgecolor', kw.pop('ec', None))
        lw = kw.pop('linewidth', kw.pop('lw', None))
        ls = kw.pop('linestyle', kw.pop('ls', 'solid'))
        hatch = kw.pop('hatch', None)
        transform = kw.pop('transform', None)
        kw.pop('antialiased', None)
        if kw:
            raise TypeError('fill_between: unsupported keyword(s): %s' % ', '.join(sorted(kw)))
        if ind == 'x':
            t, d1 = self._xy(t, d1)
            _, d2 = self._xy(t, d2)
        else:
            d1, t = self._xy(d1, t)
            d2, _ = self._xy(d2, t)
        t = np.asarray(t, float).ravel()
        d1 = np.broadcast_to(np.asarray(d1, float), t.shape)
        d2 = np.broadcast_to(np.asarray(d2, float), t.shape)
        if fc is None and color is None:
            fc = self._next_color('patches')
        mask = np.isfinite(t) & np.isfinite(d1) & np.isfinite(d2)
        if where is not None:
            mask &= np.broadcast_to(np.asarray(where, bool), t.shape)
        polys = []
        idx = np.nonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))[0].reshape(-1, 2)
        for a, b in idx:
            tt, y1s, y2s = t[a:b], d1[a:b], d2[a:b]
            if step is not None:
                fn = {'pre': _steps_pre, 'post': _steps_post, 'mid': _steps_mid}[step]
                tt, y1s, y2s = fn(tt, y1s, y2s)
            if interpolate and where is not None:
                tt, y1s, y2s = _interp_ends(t, d1, d2, a, b, tt, y1s, y2s)
            pts = np.r_[np.c_[tt, y1s], np.c_[tt[::-1], y2s[::-1]]]
            polys.append(pts if ind == 'x' else pts[:, ::-1])
        src = self._src(transform)
        allp = np.concatenate(polys) if polys else np.empty((0, 2))
        bnd = _Bounds(allp[:, 0], allp[:, 1]) if src == 'data' else None
        coll = PolyCollection(self, polys, src, fc, ec if ec is not None else 'none',
                              rc('patch.linewidth') if lw is None else lw, ls, alpha=alpha,
                              zorder=zorder, label=label, bounds=bnd, hatch=hatch)
        if color is not None:                           # matplotlib 3.11: color sets the hatch's
            coll.hatchcolor = color
        return self._add(coll)

    def axhspan(self, ymin, ymax, xmin=0, xmax=1, **kw):
        return self._span('y', ymin, ymax, xmin, xmax, kw)

    def axvspan(self, xmin, xmax, ymin=0, ymax=1, **kw):
        return self._span('x', xmin, xmax, ymin, ymax, kw)

    def _span(self, along, a, b, c, d, kw):
        from . import transforms
        if along == 'y':
            _, (a, b) = self._xy(None, np.array([a, b]))
            tr = transforms.blended_transform_factory(self.transAxes, self.transData)
            poly = [[c, a], [d, a], [d, b], [c, b]]
            bnd = _Bounds(None, np.array([a, b], float))
        else:
            (a, b), _ = self._xy(np.array([a, b]), None)
            tr = transforms.blended_transform_factory(self.transData, self.transAxes)
            poly = [[a, c], [b, c], [b, d], [a, d]]
            bnd = _Bounds(np.array([a, b], float), None)
        fc = kw.pop('facecolor', kw.pop('fc', kw.pop('color', None)))
        if fc is None:
            fc = self._next_color('patches')
        ec = kw.pop('edgecolor', kw.pop('ec', 'none'))
        lw = kw.pop('linewidth', kw.pop('lw', rc('patch.linewidth')))
        coll = PolyCollection(self, [poly], tr, fc, ec, lw, kw.pop('linestyle', 'solid'),
                              alpha=kw.pop('alpha', None), zorder=kw.pop('zorder', None),
                              label=kw.pop('label', None), bounds=bnd,
                              hatch=kw.pop('hatch', None))
        if kw:
            raise TypeError('axspan: unsupported keyword(s): %s' % ', '.join(sorted(kw)))
        coll.rectangles = True
        return self._add(coll)

    def fill(self, *args, color=None, alpha=None, label=None, zorder=None, **kw):
        """matplotlib's fill: polygons from x, y[, color] groups."""
        out = []
        k = 0
        while k + 1 < len(args):
            x, y = self._xy(args[k], args[k + 1])
            c = color
            k += 2
            if k < len(args) and isinstance(args[k], str):
                c = args[k]
                k += 1
            if c is None:
                c = self._next_color('patches')
            pts = np.c_[np.asarray(x, float), np.asarray(y, float)]
            out.append(self._add(PolyCollection(
                self, [pts], 'data', kw.get('facecolor', c), kw.get('edgecolor', 'none'),
                kw.get('linewidth', rc('patch.linewidth')), alpha=alpha, zorder=zorder, label=label,
                hatch=kw.get('hatch'))))
        return out

    def stackplot(self, x, *ys, labels=(), colors=None, baseline='zero', alpha=None, **kw):
        y = np.vstack(ys) if len(ys) > 1 else np.atleast_2d(ys[0])
        stack = np.cumsum(y, axis=0)
        lo = np.zeros_like(stack[0])
        labels = list(labels)
        out = []
        for k in range(stack.shape[0]):                # matplotlib's: the lines' colours
            c = self._next_color('lines') if colors is None else colors[k % len(colors)]
            out.append(self.fill_between(x, lo, stack[k], color=c, alpha=alpha,
                                         label=labels[k] if k < len(labels) else None, **kw))
            lo = stack[k]
        if out and baseline == 'zero' and out[0]._bounds is not None:
            out[0]._bounds.sy = (0.0,)                  # sticky at zero, as matplotlib's
        return out

    # -- lines
    def axhline(self, y=0, xmin=0, xmax=1, **kw):
        return self._axline('y', y, xmin, xmax, kw)

    def axvline(self, x=0, ymin=0, ymax=1, **kw):
        return self._axline('x', x, ymin, ymax, kw)

    def _axline(self, along, v, a, b, kw):
        from . import transforms
        if along == 'y':
            _, v = self._xy(None, v)
            tr = transforms.blended_transform_factory(self.transAxes, self.transData)
            xs, ys = [a, b], [v, v]
            bnd = _Bounds(None, np.array([v], float))
        else:
            v, _ = self._xy(v, None)
            tr = transforms.blended_transform_factory(self.transData, self.transAxes)
            xs, ys = [v, v], [a, b]
            bnd = _Bounds(np.array([v], float), None)
        line = self.plot(xs, ys, transform=tr, **kw)[0]
        line._bounds = bnd
        return line

    def hlines(self, y, xmin, xmax, colors=None, linestyles='solid', label=None, **kw):
        return self._lines('y', y, xmin, xmax, colors, linestyles, label, kw)

    def vlines(self, x, ymin, ymax, colors=None, linestyles='solid', label=None, **kw):
        return self._lines('x', x, ymin, ymax, colors, linestyles, label, kw)

    def _lines(self, along, v, lo, hi, colors, ls, label, kw):
        if hasattr(self, '_to_num'):                    # dates, categories: as numbers
            pos, ext = (self.yaxis, self.xaxis) if along == 'y' else (self.xaxis, self.yaxis)
            v, lo, hi = self._to_num(pos, v), self._to_num(ext, lo), self._to_num(ext, hi)
        v = np.atleast_1d(np.asarray(v, float))
        n = v.size
        lo = np.broadcast_to(np.asarray(lo, float), (n,))
        hi = np.broadcast_to(np.asarray(hi, float), (n,))
        if along == 'y':
            segs = [[[a, y], [b, y]] for y, a, b in zip(v, lo, hi)]
        else:
            segs = [[[x, a], [x, b]] for x, a, b in zip(v, lo, hi)]
        lw = kw.pop('linewidth', kw.pop('linewidths', kw.pop('lw', rc('lines.linewidth'))))
        coll = LineCollection(self, segs, 'data', colors if colors is not None else
                              kw.pop('color', 'C0'), lw, ls, alpha=kw.pop('alpha', None),
                              zorder=kw.pop('zorder', None), label=label)
        if kw:
            raise TypeError('hlines/vlines: unsupported keyword(s): %s' % ', '.join(sorted(kw)))
        return self._add(coll)

    # -- log plots: matplotlib's semilogx / semilogy / loglog
    def _log_plot(self, axes, args, kw):
        for name in axes:
            sk = {k[:-1] if k.endswith(name) else k: kw.pop(k) for k in list(kw)
                  if k in ('base' + name, 'subs' + name, 'nonpositive' + name,
                           'base', 'subs', 'nonpositive')}
            getattr(self, 'set_%sscale' % name)('log', **sk)
        return self.plot(*args, **kw)

    def semilogx(self, *args, **kw):
        return self._log_plot('x', args, kw)

    def semilogy(self, *args, **kw):
        return self._log_plot('y', args, kw)

    def loglog(self, *args, **kw):
        return self._log_plot('xy', args, kw)

    def step(self, x, y, *args, where='pre', data=None, **kw):
        x, y = self._xy(x, y)
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        fn = {'pre': _steps_pre, 'post': _steps_post, 'mid': _steps_mid}[where]
        xs, ys = fn(x, y)
        return self.plot(xs, ys, *args, **kw)

    def stairs(self, values, edges=None, *, orientation='vertical', baseline=0, fill=False,
               color=None, label=None, alpha=None, zorder=None, linewidth=None, **kw):
        values = np.asarray(values, float)
        edges = np.arange(values.size + 1) if edges is None else np.asarray(edges, float)
        xe = np.repeat(edges, 2)
        yv = np.r_[baseline, np.repeat(values, 2), baseline]
        pts = np.c_[xe, yv] if orientation == 'vertical' else np.c_[yv, xe]
        if not fill:
            pts = pts[1:-1] if baseline is None else pts[1:-1]
        c = color if color is not None else self._next_color('patches')
        bnd = _Bounds(pts[:, 0], pts[:, 1], *(((), (baseline,)) if orientation == 'vertical'
                                             else ((baseline,), ())))
        coll = PolyCollection(self, [pts], 'data', c if fill else 'none', 'none' if fill else c,
                              rc('patch.linewidth') if linewidth is None else linewidth, alpha=alpha,
                              zorder=zorder, label=label, bounds=bnd)
        if not fill:
            coll.open = True
        return self._add(coll)

    def errorbar(self, x, y, yerr=None, xerr=None, fmt='', ecolor=None, elinewidth=None,
                 capsize=None, barsabove=False, lolims=False, uplims=False, xlolims=False,
                 xuplims=False, errorevery=1, capthick=None, *, label=None, **kw):
        """matplotlib's errorbar: (data line, caplines, barlinecols)."""
        x, y = self._xy(x, y)
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        data_line = None
        color = kw.pop('color', kw.pop('c', None))
        zorder = kw.pop('zorder', 2.0)
        if fmt != 'none':
            data_line = self.plot(x, y, *([fmt] if fmt else []), color=color, zorder=zorder + 0.1,
                                  **kw)
            data_line = data_line[0]
            color = data_line.color
        else:
            color = color or self._next_color('lines')
        ecolor = ecolor if ecolor is not None else color
        lw = elinewidth if elinewidth is not None else kw.get('linewidth', kw.get('lw',
                                                                                  rc('lines.linewidth')))
        capsize = 0.0 if capsize is None else float(capsize)
        sel = np.zeros(x.size, bool)
        sel[::errorevery] = True
        bars, caps = [], []
        for err, along in ((yerr, 'y'), (xerr, 'x')):
            if err is None:
                continue
            e = np.asarray(err, float)
            lo, hi = (e, e) if e.ndim < 2 else (e[0], e[1])
            lo = np.broadcast_to(lo, x.shape)
            hi = np.broadcast_to(hi, x.shape)
            if along == 'y':
                a, b = y - lo, y + hi
                segs = [[[xi, ai], [xi, bi]] for xi, ai, bi in zip(x[sel], a[sel], b[sel])]
                ends = (np.r_[x[sel], x[sel]], np.r_[a[sel], b[sel]], '_')
            else:
                a, b = x - lo, x + hi
                segs = [[[ai, yi], [bi, yi]] for yi, ai, bi in zip(y[sel], a[sel], b[sel])]
                ends = (np.r_[a[sel], b[sel]], np.r_[y[sel], y[sel]], '|')
            ez = zorder + (0.2 if barsabove else 0.0)
            bars.append(self._add(LineCollection(self, segs, 'data', ecolor, lw, zorder=ez)))
            if capsize > 0:
                caps += self.plot(ends[0], ends[1], linestyle='none', marker=ends[2],
                                  markersize=2.0 * capsize, color=ecolor, zorder=ez,
                                  markeredgewidth=capthick if capthick is not None else
                                  rc('lines.markeredgewidth'))
        c = ErrorbarContainer((data_line, tuple(caps), tuple(bars)), xerr is not None,
                              yerr is not None, label)
        self.containers.append(c)
        return c

    def stem(self, *args, linefmt=None, markerfmt=None, basefmt=None, bottom=0, label=None,
             orientation='vertical'):
        if len(args) == 1:
            y = np.asarray(args[0], float)
            x = np.arange(y.size)
        else:
            x, y = np.asarray(args[0], float), np.asarray(args[1], float)
        lc, _, ls = _parse_fmt(linefmt or '')
        mc, mm, _ = _parse_fmt(markerfmt or '')
        bc, _, bls = _parse_fmt(basefmt or '')
        vert = orientation == 'vertical'
        segs = [[[xi, bottom], [xi, yi]] if vert else [[bottom, xi], [yi, xi]]
                for xi, yi in zip(x, y)]
        stems = self._add(LineCollection(self, segs, 'data', lc or 'C0', rc('lines.linewidth'),
                                         linestyle=ls or '-'))
        mk = self.plot(x if vert else y, y if vert else x, marker=mm or 'o', color=mc or lc or 'C0',
                       linestyle='none')[0]
        lo, hi = (x.min(), x.max()) if x.size else (0, 1)
        base = self.plot([lo, hi] if vert else [bottom, bottom], [bottom, bottom] if vert
                         else [lo, hi], color=bc or 'C3', linestyle=bls or '-')[0]
        c = StemContainer((mk, stems, base), label)
        self.containers.append(c)
        return c


def _steps_pre(x, *ys):
    n = x.size
    xs = np.empty(2 * n - 1)
    xs[0::2], xs[1::2] = x, x[:-1]
    out = [xs]
    for y in ys:
        v = np.empty(2 * n - 1)
        v[0::2], v[1:-1:2] = y, y[1:]
        out.append(v)
    return tuple(out)


def _steps_post(x, *ys):
    n = x.size
    xs = np.empty(2 * n - 1)
    xs[::2], xs[1:-1:2] = x, x[1:]
    out = [xs]
    for y in ys:
        v = np.empty(2 * n - 1)
        v[0::2], v[1::2] = y, y[:-1]
        out.append(v)
    return tuple(out)


def _steps_mid(x, *ys):
    n = x.size
    xs = np.empty(2 * n)
    xs[1:-1:2] = xs[2::2] = (x[:-1] + x[1:]) / 2
    xs[0], xs[-1] = x[0], x[-1]
    out = [xs]
    for y in ys:
        v = np.empty(2 * n)
        v[0::2], v[1::2] = y, y
        out.append(v)
    return tuple(out)


def _interp_ends(t, d1, d2, a, b, tt, y1s, y2s):
    """fill_between(interpolate=True): extend a run to where the curves cross."""
    def cross(i, j):
        diff_i, diff_j = d1[i] - d2[i], d1[j] - d2[j]
        if diff_i == diff_j:
            return t[i], d1[i]
        f = diff_i / (diff_i - diff_j)
        return t[i] + f * (t[j] - t[i]), d1[i] + f * (d1[j] - d1[i])
    if a > 0:
        ct, cy = cross(a - 1, a)
        tt, y1s, y2s = np.r_[ct, tt], np.r_[cy, y1s], np.r_[cy, y2s]
    if b < t.size:
        ct, cy = cross(b - 1, b)
        tt, y1s, y2s = np.r_[tt, ct], np.r_[y1s, cy], np.r_[y2s, cy]
    return tt, y1s, y2s


builtin_range = range
