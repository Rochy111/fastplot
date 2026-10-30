"""matplotlib's statistical and special plot types on plain axes, built
from the artists the rest of fastplot draws (wedges, lines, fills,
collections), with matplotlib's statistics and its geometry:

    pie / pie_label     Wedge patches and their labels (PieContainer)
    boxplot / bxp       cbook.boxplot_stats, then lines (or patches) per box
    violinplot / violin cbook.violin_stats (mlab.GaussianKDE), fills, lines
    axline              an infinite line through a point or two
    broken_barh         rectangles along a row
    eventplot           a row of ticks per sequence (EventCollection)
    matshow             imshow of a matrix, its x ticks above
"""
import itertools
import math

import numpy as np

from ._style import _dashes, rc, rgba
from ._artists import Line2D
from ._plots import LineCollection, PolyCollection, _Bounds


class PieContainer:
    """pie()'s result (matplotlib 3.11's PieContainer): .wedges, .texts;
    unpacks as (wedges, texts[, autotexts])."""

    def __init__(self, wedges, values, normalize):
        self.wedges = wedges
        self._texts = []
        self._values = np.asarray(values)
        self._normalize = normalize

    @property
    def texts(self):
        return [t for t in self._texts if t]

    @property
    def values(self):
        return self._values.copy()

    @property
    def fracs(self):
        return self._values / self._values.sum() if self._normalize else self._values.copy()

    def add_texts(self, texts):
        self._texts.append(texts)

    def remove(self):
        for a in list(self.wedges) + [t for ts in self._texts for t in ts]:
            a.remove()

    def __getitem__(self, key):
        return (self.wedges, *self._texts)[key]

    def __iter__(self):
        return iter((self.wedges, *self._texts))

    def __len__(self):
        return 1 + len(self._texts)


def boxplot_stats(X, whis=1.5, bootstrap=None, labels=None, autorange=False):
    """matplotlib.cbook.boxplot_stats: per dataset the mean, quartiles,
    whisker ends, fliers and median confidence interval."""
    X = _reshape_2d(X)
    if labels is None:
        labels = itertools.repeat(None)
    elif len(labels) != len(X):
        raise ValueError('Dimensions of labels and X must be compatible')
    out = []
    for x, label in zip(X, labels):
        stats = {}
        if label is not None:
            stats['label'] = label
        out.append(stats)
        x = np.ma.asarray(x)
        x = x.data[~np.ma.getmaskarray(x)].ravel()
        if len(x) == 0:
            stats.update(fliers=np.array([]), mean=np.nan, med=np.nan, q1=np.nan, q3=np.nan,
                         iqr=np.nan, cilo=np.nan, cihi=np.nan, whislo=np.nan, whishi=np.nan)
            continue
        stats['mean'] = np.mean(x)
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        stats['iqr'] = q3 - q1
        w = whis
        if stats['iqr'] == 0 and autorange:
            w = (0, 100)
        if bootstrap is not None:
            rng = np.random.default_rng()
            meds = np.median(x[rng.integers(0, len(x), (int(bootstrap), len(x)))], axis=1)
            stats['cilo'], stats['cihi'] = np.percentile(meds, [2.5, 97.5])
        else:
            n = len(x)
            stats['cilo'] = med - 1.57 * stats['iqr'] / np.sqrt(n)
            stats['cihi'] = med + 1.57 * stats['iqr'] / np.sqrt(n)
        if np.iterable(w) and not isinstance(w, str):
            loval, hival = np.percentile(x, w)
        else:
            loval, hival = q1 - w * stats['iqr'], q3 + w * stats['iqr']
        hi = x[x <= hival]
        stats['whishi'] = q3 if len(hi) == 0 or np.max(hi) < q3 else np.max(hi)
        lo = x[x >= loval]
        stats['whislo'] = q1 if len(lo) == 0 or np.min(lo) > q1 else np.min(lo)
        stats['fliers'] = np.concatenate([x[x < stats['whislo']], x[x > stats['whishi']]])
        stats['q1'], stats['med'], stats['q3'] = q1, med, q3
    return out


def _reshape_2d(X):
    """matplotlib's cbook._reshape_2D: a list of 1-D datasets (a 2-D
    array's columns)."""
    if isinstance(X, np.ndarray) and X.ndim == 2 and not isinstance(X, np.ma.MaskedArray):
        return [X[:, k] for k in range(X.shape[1])]
    if isinstance(X, np.ma.MaskedArray) and X.ndim == 2:
        return [X[:, k] for k in range(X.shape[1])]
    if np.ndim(X) == 1 and len(X) and not np.iterable(X[0]):
        return [np.asarray(X)]
    if np.ndim(X) == 0:
        return [np.atleast_1d(X)]
    return [np.ravel(v) if not isinstance(v, np.ma.MaskedArray) else v.ravel() for v in X]


def _kde(x, coords, bw_method=None):
    """matplotlib.mlab.GaussianKDE(x, bw_method).evaluate(coords)."""
    x = np.asarray(x, float)
    if np.all(x[0] == x):
        return (x[0] == coords).astype(float)
    n = x.size
    if bw_method is None or bw_method == 'scott':
        factor = n ** (-1.0 / 5)
    elif bw_method == 'silverman':
        factor = (n * 3.0 / 4.0) ** (-1.0 / 5)
    elif callable(bw_method):
        class _K:
            dataset, dim, num_dp = np.atleast_2d(x), 1, n
        factor = bw_method(_K)
    else:
        factor = float(bw_method)
    var = np.cov(x, bias=False) * factor ** 2
    norm = np.sqrt(2 * np.pi * var) * n
    out = np.zeros(coords.size)
    for k in range(0, n, 4096):                         # bounded memory
        d = x[k:k + 4096, None] - coords[None, :]
        out += np.exp(-0.5 * d * d / var).sum(axis=0)
    return out / norm


def violin_stats(X, points=100, quantiles=None, bw_method=None):
    """matplotlib.cbook.violin_stats with its Gaussian KDE."""
    X = _reshape_2d(X)
    if quantiles is not None and len(quantiles) != 0:
        quantiles = _reshape_2d(quantiles)
    else:
        quantiles = [[]] * len(X)
    if len(X) != len(quantiles):
        raise ValueError('List of violinplot statistics and quantiles values must have the '
                         'same length')
    out = []
    for x, q in zip(X, quantiles):
        x = np.ma.asarray(x, float)
        x = x.data[~np.ma.getmaskarray(x) & np.isfinite(x.data)].ravel()
        if len(x) == 0:
            out.append(dict(vals=np.array([]), coords=np.array([]), mean=np.nan,
                            median=np.nan, min=np.nan, max=np.nan, quantiles=np.array([])))
            continue
        lo, hi = np.min(x), np.max(x)
        coords = np.linspace(lo, hi, points)
        out.append(dict(vals=_kde(x, coords, bw_method), coords=coords, mean=np.mean(x),
                        median=np.median(x), min=lo, max=hi,
                        quantiles=np.atleast_1d(np.percentile(x, 100 * np.asarray(q, float)))))
    return out


class AxLine(Line2D):
    """axline's line: through xy1 and xy2 (or xy1 at a slope), across the
    whole axes, straight on screen."""

    def __init__(self, ax, xy1, xy2, slope, **kw):
        super().__init__(ax, [xy1[0]], [xy1[1]], 'data', **kw)
        self.xy1, self.xy2, self.slope = tuple(map(float, xy1)), xy2, slope
        pts = [self.xy1] + ([tuple(map(float, xy2))] if xy2 is not None else [])
        self._bounds = _Bounds(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))

    def render(self, r, ax):
        x1, y1 = self.xy1
        if self.xy2 is not None:
            x2, y2 = (float(v) for v in self.xy2)
        else:
            x2, y2 = x1 + 1.0, y1 + float(self.slope)
        px, py = ax._points_px(np.array([x1, x2]), np.array([y1, y2]), 'data')
        dx, dy = px[1] - px[0], py[1] - py[0]
        if not (np.isfinite(px).all() and np.isfinite(py).all()) or (dx == 0 and dy == 0):
            return
        bx0, by0, bx1, by1 = ax.box_f
        reach = 2.0 * (abs(bx1 - bx0) + abs(by1 - by0)) / math.hypot(dx, dy)
        cx, cy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
        t0 = ((cx - px[0]) * dx + (cy - py[0]) * dy) / (dx * dx + dy * dy)
        xy = np.array([[px[0] + (t0 - reach) * dx, py[0] + (t0 - reach) * dy],
                       [px[0] + (t0 + reach) * dx, py[0] + (t0 + reach) * dy]])
        if self.linestyle in ('None', 'none', '', ' ') or self.linewidth <= 0:
            return
        from . import _draw
        d = _dashes(self.linestyle, self.linewidth)
        cap = (_draw.BUTT if d is not None else _draw.PROJECTING) if ax._agg else _draw.LEGACY
        r.lines(xy, np.array([0, 2]), rgba(self.color, self.alpha), r.px(self.linewidth),
                ax.box_clip, None if d is None else tuple(r.px(v) for v in d), ax._snap, cap,
                ax._agg)


class EventCollection(LineCollection):
    """eventplot's row of ticks."""

    def __init__(self, ax, positions, orientation, lineoffset, linelength, colors, linewidth,
                 linestyle, alpha=None, zorder=None, label=None):
        p = np.asarray(positions, float).ravel()
        lo, hi = lineoffset - 0.5 * linelength, lineoffset + 0.5 * linelength
        if orientation == 'horizontal':
            segs = [[[v, lo], [v, hi]] for v in p]
        else:
            segs = [[[lo, v], [hi, v]] for v in p]
        # matplotlib autoscales to the offset +- the whole line length
        lo2, hi2 = lineoffset - linelength, lineoffset + linelength
        pr = np.array([p.min(), p.max()]) if p.size else None
        bounds = (_Bounds(pr, np.array([lo2, hi2])) if orientation == 'horizontal'
                  else _Bounds(np.array([lo2, hi2]), pr))
        super().__init__(ax, segs, 'data', colors, linewidth, linestyle, alpha, zorder, label,
                         bounds=bounds)
        self.positions, self.orientation = p, orientation
        self.lineoffset, self.linelength = lineoffset, linelength

    def get_positions(self):
        return self.positions.tolist()

    def get_orientation(self):
        return self.orientation

    def get_lineoffset(self):
        return self.lineoffset

    def get_linelength(self):
        return self.linelength


class ColoredPolyCollection(PolyCollection):
    """Polygons coloured by values through a colormap and norm (hexbin's):
    a mappable, so a colorbar can show it; set_array to recolour."""

    def __init__(self, ax, polys, values, cmap, norm, vmin, vmax, edgecolors, linewidths,
                 alpha=None, zorder=None, bounds=None):
        from . import _color
        super().__init__(ax, polys, 'data', 'none', 'none', linewidths, alpha=alpha,
                         zorder=zorder, bounds=bounds)
        self.cmap = _color.get_cmap(cmap)
        self._norm_in, self._vmin, self._vmax = norm, vmin, vmax
        self._edge_spec = edgecolors
        self.set_array(values)

    def set_array(self, values):
        self.values = np.asarray(values, np.float32).ravel()
        self._norm = None

    def get_array(self):
        return self.values

    def norm_spec(self):
        from . import _color
        if self._norm is None:
            self._norm = _color.make_norm(self._norm_in, vmin=self._vmin, vmax=self._vmax,
                                          data=self.values, N=self.cmap.N)
        return self._norm

    @property
    def norm(self):
        return self.norm_spec()

    def get_cmap(self):
        return self.cmap

    def set_cmap(self, cmap):
        from . import _color
        self.cmap = _color.get_cmap(cmap)

    def set_clim(self, vmin=None, vmax=None):
        if isinstance(vmin, (tuple, list)):
            vmin, vmax = vmin
        self._vmin, self._vmax, self._norm = vmin, vmax, None

    def get_clim(self):
        n = self.norm_spec()
        return n.vmin, n.vmax

    def render(self, r, ax):
        from . import _color
        out = np.empty((1, self.values.size, 4), np.uint8)
        _color.shade(self.values[None, :], self.cmap, self.norm_spec(), out=out,
                     alpha=self.alpha, rounding='nearest')
        self.facecolors = out[0]
        e = self._edge_spec
        if isinstance(e, str) and e == 'face':
            self.edgecolors = out[0]
        else:
            from ._plots import _colors
            self.edgecolors = _colors(e, len(self.polys), self.alpha)
        super().render(r, ax)


def _line_kw(prefix, explicit):
    """bxp's merge_kw_rc: rcParams boxplot.<prefix>props.*, then the
    caller's (Line2D keywords, aliases allowed)."""
    d = {}
    for key in ('color', 'linewidth', 'linestyle', 'marker', 'markerfacecolor',
                'markeredgecolor', 'markeredgewidth', 'markersize'):
        k = 'boxplot.%sprops.%s' % (prefix, key)
        try:
            d[key] = rc(k)
        except KeyError:
            pass
    alias = {'c': 'color', 'lw': 'linewidth', 'ls': 'linestyle', 'mfc': 'markerfacecolor',
             'mec': 'markeredgecolor', 'mew': 'markeredgewidth', 'ms': 'markersize'}
    for k, v in (explicit or {}).items():
        d[alias.get(k, k)] = v
    return d


class StatsMixin:
    """The plot types above, as Axes methods (matplotlib's signatures)."""

    # -- pie
    def pie(self, x, explode=None, labels=None, colors=None, autopct=None, pctdistance=0.6,
            shadow=False, labeldistance=1.1, startangle=0, radius=1, counterclock=True,
            wedgeprops=None, textprops=None, center=(0, 0), frame=False, rotatelabels=False,
            *, normalize=True, hatch=None):
        """matplotlib's pie: Wedges and their labels; returns a PieContainer
        (unpacks as matplotlib's (wedges, texts[, autotexts]))."""
        from .patches import Wedge
        self.set_aspect('equal')
        x = np.asarray(x)
        if x.ndim > 1:
            raise ValueError('x must be 1D')
        if np.any(x < 0):
            raise ValueError("Wedge sizes 'x' must be non negative values")
        if not np.all(np.isfinite(x)):
            raise ValueError('Wedge sizes must be finite numbers')
        sx = x.sum()
        if sx == 0:
            raise ValueError('All wedge sizes are zero')
        if normalize:
            fracs = x / sx
        elif sx > 1:
            raise ValueError('Cannot plot an unnormalized pie with sum(x) > 1')
        else:
            fracs = x
        labels = [''] * len(x) if labels is None else list(labels)
        explode = [0] * len(x) if explode is None else list(explode)
        if len(x) != len(labels):
            raise ValueError("'labels' must be of length 'x', not %d" % len(labels))
        if len(x) != len(explode):
            raise ValueError("'explode' must be of length 'x', not %d" % len(explode))
        if colors is None:
            def next_color():
                return self._next_color('patches')
        else:
            cyc = itertools.cycle(colors if not isinstance(colors, str) else [colors])

            def next_color():
                return next(cyc)
        hatches = itertools.cycle(np.atleast_1d(hatch) if hatch is not None else [None])
        if radius <= 0:
            raise ValueError("'radius' must be a positive number, not %r" % (radius,))
        theta1 = startangle / 360
        wedges = []
        for frac, label, expl in zip(fracs, labels, explode):
            cx, cy = center
            theta2 = (theta1 + frac) if counterclock else (theta1 - frac)
            thetam = 2 * np.pi * 0.5 * (theta1 + theta2)
            cx += expl * math.cos(thetam)
            cy += expl * math.sin(thetam)
            w = Wedge((cx, cy), radius, 360. * min(theta1, theta2), 360. * max(theta1, theta2),
                      facecolor=next_color(), hatch=next(hatches), label=label)
            w.clip = False
            if wedgeprops:
                w.set(**wedgeprops)
            wedges.append(w)
            if shadow:                                  # matplotlib's Shadow: just behind
                sd = {'ox': -0.02, 'oy': -0.02}
                if isinstance(shadow, dict):
                    sd.update(shadow)
                sh = Wedge((cx, cy), radius, 360. * min(theta1, theta2),
                           360. * max(theta1, theta2), width=w.width,
                           facecolor=0.3 * np.asarray(rgba(w.facecolor)[:3]) / 255.0,
                           edgecolor='none', alpha=0.5, zorder=np.nextafter(w.zorder, -np.inf))
                sh.clip = False
                self.add_patch(sh)
            self.add_patch(w)
            theta1 = theta2
        pc = PieContainer(wedges, x, normalize)
        if labeldistance is None:
            pc.add_texts([])
        else:
            tp = {'fontsize': rc('xtick.labelsize')}
            tp.update(textprops or {})
            self.pie_label(pc, labels, distance=labeldistance, alignment='outer',
                           rotate=rotatelabels, textprops=tp)
        if autopct is not None:
            auto = []
            for frac in fracs:
                if isinstance(autopct, str):
                    auto.append(autopct % (100. * frac))
                elif callable(autopct):
                    auto.append(autopct(100. * frac))
                else:
                    raise TypeError('autopct must be callable or a format string')
            self.pie_label(pc, auto, distance=pctdistance, alignment='center',
                           textprops=textprops)
        if not frame:
            self.set_frame_on(False)
            self.set_xticks([])
            self.set_yticks([])
            self.set_xlim(-1.25 + center[0], 1.25 + center[0])
            self.set_ylim(-1.25 + center[1], 1.25 + center[1])
        else:
            for w in wedges:
                w.requests_autoscale = True
        return pc

    def pie_label(self, container, labels, *, distance=0.6, textprops=None, rotate=False,
                  alignment='auto'):
        """matplotlib 3.11's pie_label: a label per wedge, distance radii out."""
        if alignment == 'auto':
            alignment = 'outer' if distance > 1 else 'center'
        textprops = dict(textprops or {})
        if isinstance(labels, str):
            labels = [labels.format(absval=v, frac=f)
                      for v, f in zip(container.values, container.fracs)]
        elif len(container.wedges) != len(labels):
            raise ValueError('The number of labels (%d) must match the number of wedges (%d)'
                             % (len(labels), len(container.wedges)))
        texts = []
        for w, label in zip(container.wedges, labels):
            thetam = 2 * np.pi * 0.5 * (w.theta1 + w.theta2) / 360
            xt = w.center[0] + distance * w.r * math.cos(thetam)
            yt = w.center[1] + distance * w.r * math.sin(thetam)
            ha = ('left' if xt > 0 else 'right') if alignment == 'outer' else 'center'
            va, rot = 'center', 0.0
            if rotate:
                if alignment == 'outer':
                    va = 'bottom' if yt > 0 else 'top'
                rot = np.rad2deg(thetam) + (0 if xt > 0 else 180)
            kw = dict(clip_on=False, rotation=rot, ha=ha, va=va)
            kw.update(textprops)
            texts.append(self.text(xt, yt, label, **kw))
        container.add_texts(texts)
        return texts

    # -- box plots
    def boxplot(self, x, notch=None, sym=None, vert=None, orientation='vertical', whis=None,
                positions=None, widths=None, patch_artist=None, bootstrap=None,
                usermedians=None, conf_intervals=None, meanline=None, showmeans=None,
                showcaps=None, showbox=None, showfliers=None, boxprops=None, tick_labels=None,
                flierprops=None, medianprops=None, meanprops=None, capprops=None,
                whiskerprops=None, manage_ticks=True, autorange=False, zorder=None,
                capwidths=None, label=None, labels=None):
        """matplotlib's boxplot: boxplot_stats, then bxp's lines; returns
        its dict (whiskers, caps, boxes, medians, fliers, means)."""
        whis = 1.5 if whis is None else whis
        bootstrap = None if bootstrap is None else bootstrap
        if tick_labels is None:
            tick_labels = labels
        stats = boxplot_stats(x, whis=whis, bootstrap=bootstrap, labels=tick_labels,
                              autorange=autorange)
        notch = False if notch is None else notch
        patch_artist = False if patch_artist is None else patch_artist
        meanline = False if meanline is None else meanline
        showmeans = False if showmeans is None else showmeans
        showcaps = True if showcaps is None else showcaps
        showbox = True if showbox is None else showbox
        showfliers = True if showfliers is None else showfliers
        flierprops = dict(flierprops or {})
        if sym is not None:
            if sym == '':                               # no fliers
                flierprops['marker'] = ''
            else:
                from ._axesbase import _parse_fmt
                c, m, _ = _parse_fmt(sym)
                if m is not None:
                    flierprops['marker'] = m
                if c is not None:
                    flierprops['color'] = c
                    flierprops['markerfacecolor'] = c
                    flierprops['markeredgecolor'] = c
        if usermedians is not None:
            for st, m in zip(stats, usermedians):
                if m is not None:
                    st['med'] = m
        if conf_intervals is not None:
            for st, ci in zip(stats, conf_intervals):
                if ci is not None:
                    st['cilo'], st['cihi'] = ci
        if vert is False:
            orientation = 'horizontal'
        return self.bxp(stats, positions=positions, widths=widths, orientation=orientation,
                        patch_artist=patch_artist, shownotches=notch, showmeans=showmeans,
                        showcaps=showcaps, showbox=showbox, boxprops=boxprops,
                        flierprops=flierprops, medianprops=medianprops, meanprops=meanprops,
                        meanline=meanline, showfliers=showfliers, capprops=capprops,
                        whiskerprops=whiskerprops, manage_ticks=manage_ticks, zorder=zorder,
                        capwidths=capwidths, label=label)

    def bxp(self, bxpstats, positions=None, widths=None, vert=None, orientation='vertical',
            patch_artist=False, shownotches=False, showmeans=False, showcaps=True,
            showbox=True, showfliers=True, boxprops=None, whiskerprops=None, flierprops=None,
            medianprops=None, meanprops=None, capprops=None, meanline=False,
            manage_ticks=True, zorder=None, capwidths=None, label=None):
        """matplotlib's bxp: boxes, whiskers, caps, medians, means, fliers."""
        from .patches import Polygon
        if vert is False:
            orientation = 'horizontal'
        z = 2.0 if zorder is None else zorder

        def kw(prefix, explicit, zd=0.0, marker=True):
            d = _line_kw(prefix, explicit)
            d['zorder'] = z + zd if 'zorder' not in (explicit or {}) else explicit['zorder']
            if not marker and 'marker' not in (explicit or {}):
                d['marker'] = ''
            return d
        whisker_kw = kw('whisker', whiskerprops, marker=False)
        cap_kw = kw('cap', capprops, marker=False)
        flier_kw = kw('flier', flierprops)
        median_kw = kw('median', medianprops, 0.1, marker=False)
        mean_kw = kw('mean', meanprops, 0.1)
        if meanprops is None or ('marker' if meanline else 'linestyle') not in meanprops:
            mean_kw['marker' if meanline else 'linestyle'] = '' if meanline else 'None'
        if patch_artist:
            bp = dict(boxprops or {})
            box_kw = dict(linestyle=rc('boxplot.boxprops.linestyle'),
                          linewidth=rc('boxplot.boxprops.linewidth'),
                          edgecolor=rc('boxplot.boxprops.color'),
                          facecolor=rc('patch.facecolor'), zorder=z)
            for k, v in bp.items():
                box_kw[{'c': 'color', 'lw': 'linewidth', 'ls': 'linestyle', 'fc': 'facecolor',
                        'ec': 'edgecolor'}.get(k, k)] = v
            if 'color' in box_kw:
                c = box_kw.pop('color')
                box_kw.setdefault('facecolor', c)
                box_kw['edgecolor'] = c if 'edgecolor' not in bp and 'ec' not in bp else \
                    box_kw['edgecolor']
                box_kw['facecolor'] = c if 'facecolor' not in bp and 'fc' not in bp else \
                    box_kw['facecolor']
        else:
            box_kw = kw('box', boxprops, marker=False)
        vertical = orientation == 'vertical'

        def swap(xs, ys):
            return (xs, ys) if vertical else (ys, xs)

        def line(xs, ys, d):
            d = dict(d)
            d.setdefault('label', '_nolegend_')
            for k in ('solid_capstyle', 'dash_capstyle', 'solid_joinstyle', 'dash_joinstyle'):
                d.pop(k, None)
            mk = d.pop('marker', None)
            if mk in ('', None):
                mk = 'None'
            return self.plot(*swap(np.asarray(xs, float), np.asarray(ys, float)), marker=mk,
                             **d)[0]

        def patch(xs, ys, d):
            p = Polygon(np.column_stack(swap(np.asarray(xs, float), np.asarray(ys, float))),
                        closed=True, **d)
            return self.add_patch(p)
        N = len(bxpstats)
        if positions is None:
            positions = list(range(1, N + 1))
        elif len(positions) != N:
            raise ValueError('List of boxplot statistics and `positions` values must have '
                             'same the length')
        positions = np.array(positions)
        if widths is None:
            widths = [np.clip(0.15 * np.ptp(positions), 0.15, 0.5)] * N if N else []
        elif np.isscalar(widths):
            widths = [widths] * N
        if capwidths is None:
            capwidths = 0.5 * np.array(widths)
        elif np.isscalar(capwidths):
            capwidths = [capwidths] * N
        out = dict(whiskers=[], caps=[], boxes=[], medians=[], fliers=[], means=[])
        datalabels = []
        for pos, width, st, cw in zip(positions, widths, bxpstats, capwidths):
            datalabels.append(st.get('label', pos))
            cl, cr = pos - cw * 0.5, pos + cw * 0.5
            bl, br = pos - width * 0.5, pos + width * 0.5
            if shownotches:
                nl, nr = pos - width * 0.25, pos + width * 0.25
                box_x = [bl, br, br, nr, br, br, bl, bl, nl, bl, bl]
                box_y = [st['q1'], st['q1'], st['cilo'], st['med'], st['cihi'], st['q3'],
                         st['q3'], st['cihi'], st['med'], st['cilo'], st['q1']]
                med_x = [nl, nr]
            else:
                box_x = [bl, br, br, bl, bl]
                box_y = [st['q1'], st['q1'], st['q3'], st['q3'], st['q1']]
                med_x = [bl, br]
            if showbox:
                out['boxes'].append(patch(box_x[:-1], box_y[:-1], box_kw) if patch_artist
                                    else line(box_x, box_y, box_kw))
            out['whiskers'].append(line([pos, pos], [st['q1'], st['whislo']], whisker_kw))
            out['whiskers'].append(line([pos, pos], [st['q3'], st['whishi']], whisker_kw))
            if showcaps:
                out['caps'].append(line([cl, cr], [st['whislo']] * 2, cap_kw))
                out['caps'].append(line([cl, cr], [st['whishi']] * 2, cap_kw))
            med = line(med_x, [st['med']] * 2, median_kw)
            if manage_ticks:                            # sticky at pos +- 0.5, as matplotlib's
                lo, hi = pos - 0.5, pos + 0.5
                mx, my = swap(np.array([lo, hi], float), np.array([st['med']] * 2, float))
                med._bounds = _Bounds(mx, my, (lo, hi) if vertical else (),
                                      () if vertical else (lo, hi))
            out['medians'].append(med)
            if showmeans:
                if meanline:
                    out['means'].append(line([bl, br], [st['mean']] * 2, mean_kw))
                else:
                    out['means'].append(line([pos], [st['mean']], mean_kw))
            if showfliers:
                f = np.asarray(st['fliers'], float)
                out['fliers'].append(line(np.full(len(f), pos, float), f, flier_kw))
        if label:
            targets = out['boxes'] if showbox and patch_artist else out['medians']
            labs = [label] if isinstance(label, str) else list(label)
            for a, lab in zip(targets, labs):
                a.label = lab
        if manage_ticks and N:
            labs = [str(v) if not isinstance(v, str) else v for v in datalabels]
            (self.set_xticks if vertical else self.set_yticks)(positions, labs)
        return out

    # -- violins
    def violinplot(self, dataset, positions=None, vert=None, orientation='vertical',
                   widths=0.5, showmeans=False, showextrema=True, showmedians=False,
                   quantiles=None, points=100, bw_method=None, side='both', facecolor=None,
                   linecolor=None):
        """matplotlib's violinplot: Gaussian KDE bodies and their lines."""
        stats = violin_stats(dataset, points=points, quantiles=quantiles, bw_method=bw_method)
        return self.violin(stats, positions=positions, vert=vert, orientation=orientation,
                           widths=widths, showmeans=showmeans, showextrema=showextrema,
                           showmedians=showmedians, side=side, facecolor=facecolor,
                           linecolor=linecolor)

    def violin(self, vpstats, positions=None, vert=None, orientation='vertical', widths=0.5,
               showmeans=False, showextrema=True, showmedians=False, side='both',
               facecolor=None, linecolor=None):
        from ._color import to_rgba_array
        if vert is not None:
            orientation = 'vertical' if vert else 'horizontal'
        N = len(vpstats)
        positions = list(range(1, N + 1)) if positions is None else list(positions)
        widths = [widths] * N if np.isscalar(widths) else list(widths)
        lo_k = -0.25 if side in ('both', 'low') else 0.0
        hi_k = 0.25 if side in ('both', 'high') else 0.0
        line_ends = np.array([[lo_k], [hi_k]]) * np.array(widths) + np.array(positions)

        def cycle(color):
            cols = to_rgba_array(color)
            return [tuple(cols[k]) if k < len(cols) or len(cols) else 'none'
                    for k in range(N)] if len(cols) >= N else \
                [tuple(cols[k % len(cols)]) for k in range(N)]
        if facecolor is None or linecolor is None:
            nxt = self._next_color('lines')
        if facecolor is not None:
            fcs, body_alpha = cycle(facecolor), None
        else:
            fcs, body_alpha = cycle(nxt), 0.3
        lcs = cycle(linecolor) if linecolor is not None else cycle(nxt)
        vertical = orientation == 'vertical'
        fill = self.fill_betweenx if vertical else self.fill_between
        perp = self.hlines if vertical else self.vlines
        par = self.vlines if vertical else self.hlines
        bodies = []
        means, mins, maxes, medians, quants, qlens = [], [], [], [], [], []
        for st, pos, width, fc in zip(vpstats, positions, widths, fcs):
            vals = np.array(st['vals'], float)
            if len(vals):
                vals = 0.5 * width * vals / vals.max()
            bodies.append(fill(st['coords'], -vals + pos if side in ('both', 'low') else pos,
                               vals + pos if side in ('both', 'high') else pos,
                               facecolor=fc, alpha=body_alpha))
            means.append(st['mean'])
            mins.append(st['min'])
            maxes.append(st['max'])
            medians.append(st['median'])
            q = st.get('quantiles')
            q = [] if q is None else list(q)
            quants.extend(q)
            qlens.append(len(q))
        out = {'bodies': bodies}
        if showmeans:
            out['cmeans'] = perp(means, *line_ends, colors=lcs)
        if showextrema:
            out['cmaxes'] = perp(maxes, *line_ends, colors=lcs)
            out['cmins'] = perp(mins, *line_ends, colors=lcs)
            out['cbars'] = par(positions, mins, maxes, colors=lcs)
        if showmedians:
            out['cmedians'] = perp(medians, *line_ends, colors=lcs)
        if quants:
            ends = np.repeat(line_ends, qlens, axis=1)
            out['cquantiles'] = perp(quants, *ends,
                                     colors=[c for c, n in zip(lcs, qlens) for _ in range(n)])
        return out

    # -- lines and rows
    def axline(self, xy1, xy2=None, *, slope=None, **kw):
        """matplotlib's axline: an infinite straight line through xy1 and
        xy2, or through xy1 at slope (linear scales)."""
        if (xy2 is None) == (slope is None):
            raise TypeError("Exactly one of 'xy2' and 'slope' must be given")
        if slope is not None and (self.get_xscale() != 'linear' or
                                  self.get_yscale() != 'linear'):
            raise TypeError("'slope' cannot be used with non-linear scales")
        color = kw.pop('color', kw.pop('c', None))
        lw = kw.pop('linewidth', kw.pop('lw', None))
        ls = kw.pop('linestyle', kw.pop('ls', '-'))
        alpha, zorder, label = kw.pop('alpha', None), kw.pop('zorder', None), kw.pop('label', None)
        for k in ('transform', 'clip_on', 'antialiased', 'solid_capstyle', 'dash_capstyle'):
            kw.pop(k, None)
        if kw:
            raise TypeError('axline: unsupported keyword(s): %s' % ', '.join(sorted(kw)))
        if color is None:
            color = self._next_color('lines')
        x1, y1 = (float(np.ravel(v)[0]) for v in self._xy(*xy1))
        if xy2 is not None:
            xy2 = tuple(float(np.ravel(v)[0]) for v in self._xy(*xy2))
        line = AxLine(self, (x1, y1), xy2, slope, color=color,
                      linewidth=rc('lines.linewidth') if lw is None else lw, linestyle=ls,
                      alpha=alpha, zorder=zorder, label=label)
        return self._add(line)

    def broken_barh(self, xranges, yrange, align='bottom', **kw):
        """matplotlib's broken_barh: rectangles (x, width) along y .. y + height."""
        fc = kw.pop('facecolors', kw.pop('facecolor', kw.pop('fc', kw.pop('color', None))))
        ec = kw.pop('edgecolors', kw.pop('edgecolor', kw.pop('ec', 'none')))
        lw = kw.pop('linewidths', kw.pop('linewidth', kw.pop('lw', rc('patch.linewidth'))))
        ls = kw.pop('linestyle', kw.pop('ls', 'solid'))
        alpha, zorder, label = kw.pop('alpha', None), kw.pop('zorder', None), kw.pop('label', None)
        hatch = kw.pop('hatch', None)
        kw.pop('antialiased', None)
        if kw:
            raise TypeError('broken_barh: unsupported keyword(s): %s' % ', '.join(sorted(kw)))
        y, h = yrange
        if align == 'center':
            y = y - h / 2
        elif align == 'top':
            y = y - h
        polys = []
        for x0, w in xranges:
            polys.append([[x0, y], [x0 + w, y], [x0 + w, y + h], [x0, y + h]])
        if fc is None:
            fc = self._next_color('patches')
        coll = PolyCollection(self, polys, 'data', fc, ec, lw, ls, alpha=alpha, zorder=zorder,
                              label=label, hatch=hatch)
        coll.rectangles = True
        return self._add(coll)

    def eventplot(self, positions, orientation='horizontal', lineoffsets=1, linelengths=1,
                  linewidths=None, colors=None, alpha=None, linestyles='solid', **kw):
        """matplotlib's eventplot: a row of ticks per sequence; a list of
        EventCollections."""
        zorder, label = kw.pop('zorder', None), kw.pop('label', None)
        for k in ('antialiaseds', 'antialiased'):
            kw.pop(k, None)
        if kw:
            raise TypeError('eventplot: unsupported keyword(s): %s' % ', '.join(sorted(kw)))
        if not np.iterable(positions) or (len(positions) and
                                          not np.iterable(positions[0])):
            positions = [positions]
        n = len(positions)
        lineoffsets = np.atleast_1d(lineoffsets)
        linelengths = np.atleast_1d(linelengths)
        if lineoffsets.size == 1 and n > 1:             # matplotlib's: 0, d, 2d, ...
            lineoffsets = np.cumsum(np.r_[0.0, np.full(n - 1, float(lineoffsets[0]))])
        lineoffsets = np.broadcast_to(lineoffsets, (n,))
        linelengths = np.broadcast_to(linelengths, (n,))
        lws = np.broadcast_to(np.atleast_1d(rc('lines.linewidth') if linewidths is None
                                            else linewidths), (n,))
        if colors is None:
            cols = [rc('lines.color')] * n
        elif isinstance(colors, str) or (np.ndim(colors) == 1 and len(colors) in (3, 4) and
                                         not isinstance(colors[0], str)):
            cols = [colors] * n
        else:
            cols = list(colors)
            if len(cols) == 1:
                cols = cols * n
        lss = [linestyles] * n if isinstance(linestyles, str) or linestyles is None or \
            isinstance(linestyles, tuple) else list(linestyles)
        if len(lss) == 1:
            lss = lss * n
        orient = 'horizontal' if orientation in ('horizontal', None) else 'vertical'
        out = []
        for k in range(n):
            ec = EventCollection(self, np.ravel(positions[k]), orient, float(lineoffsets[k]),
                                 float(linelengths[k]), cols[k], float(lws[k]), lss[k],
                                 alpha=alpha, zorder=zorder, label=label if k == 0 else None)
            out.append(self._add(ec))
        return out

    def hexbin(self, x, y, C=None, gridsize=100, bins=None, xscale='linear', yscale='linear',
               extent=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None,
               linewidths=None, edgecolors='face', reduce_C_function=np.mean, mincnt=None,
               marginals=False, zorder=None, **kw):
        """matplotlib's hexbin: points counted (or C reduced) in hexagons,
        coloured by cmap and norm; a mappable collection."""
        from . import colors as _colors_mod
        kw.pop('label', None)
        for k in ('antialiased', 'rasterized'):
            kw.pop(k, None)
        if kw:
            raise TypeError('hexbin: unsupported keyword(s): %s' % ', '.join(sorted(kw)))
        x, y = np.ma.asarray(x, float), np.ma.asarray(y, float)
        keep = ~(np.ma.getmaskarray(x) | np.ma.getmaskarray(y))
        if C is not None:
            C = np.ma.asarray(C, float)
            keep &= ~np.ma.getmaskarray(C)
            C = np.asarray(C.data)[keep]
        x, y = np.asarray(x.data)[keep], np.asarray(y.data)[keep]
        if np.iterable(gridsize):
            nx, ny = gridsize
        else:
            nx, ny = gridsize, int(gridsize / math.sqrt(3))
        tx = np.log10(x) if xscale == 'log' else x
        ty = np.log10(y) if yscale == 'log' else y
        if extent is not None:
            xmin, xmax, ymin, ymax = extent
        else:
            from ._ticker import _nonsingular
            xmin, xmax = (tx.min(), tx.max()) if len(x) else (0, 1)
            ymin, ymax = (ty.min(), ty.max()) if len(y) else (0, 1)
            xmin, xmax = _nonsingular(xmin, xmax, expander=0.1)
            ymin, ymax = _nonsingular(ymin, ymax, expander=0.1)
        nx1, ny1, nx2, ny2 = nx + 1, ny + 1, nx, ny
        n = nx1 * ny1 + nx2 * ny2
        padding = 1.e-9 * (xmax - xmin)
        xmin, xmax = xmin - padding, xmax + padding
        sx, sy = (xmax - xmin) / nx, (ymax - ymin) / ny
        ix, iy = (tx - xmin) / sx, (ty - ymin) / sy
        ix1, iy1 = np.round(ix).astype(int), np.round(iy).astype(int)
        ix2, iy2 = np.floor(ix).astype(int), np.floor(iy).astype(int)
        i1 = np.where((0 <= ix1) & (ix1 < nx1) & (0 <= iy1) & (iy1 < ny1), ix1 * ny1 + iy1 + 1, 0)
        i2 = np.where((0 <= ix2) & (ix2 < nx2) & (0 <= iy2) & (iy2 < ny2), ix2 * ny2 + iy2 + 1, 0)
        d1 = (ix - ix1) ** 2 + 3.0 * (iy - iy1) ** 2
        d2 = (ix - ix2 - 0.5) ** 2 + 3.0 * (iy - iy2 - 0.5) ** 2
        bdist = d1 < d2
        if C is None:
            c1 = np.bincount(i1[bdist], minlength=1 + nx1 * ny1)[1:]
            c2 = np.bincount(i2[~bdist], minlength=1 + nx2 * ny2)[1:]
            accum = np.concatenate([c1, c2]).astype(float)
            if mincnt is not None:
                accum[accum < mincnt] = np.nan
        else:
            mincnt = 1 if mincnt is None else mincnt
            idx = np.where(bdist, i1, nx1 * ny1 + 1 + i2)     # one index space; 0: outside
            valid = np.where(bdist, i1 > 0, i2 > 0)
            accum = np.full(n, np.nan)
            order = np.argsort(idx[valid], kind='stable')
            iv, cv = idx[valid][order], C[valid][order]
            uniq, start = np.unique(iv, return_index=True)
            stops = np.r_[start[1:], iv.size]
            for u, a, b in zip(uniq, start, stops):
                if b - a >= mincnt:
                    k = u - 1 if u <= nx1 * ny1 else u - 2
                    accum[k] = reduce_C_function(cv[a:b])
        good = ~np.isnan(accum)
        offsets = np.zeros((n, 2), float)
        offsets[:nx1 * ny1, 0] = np.repeat(np.arange(nx1), ny1)
        offsets[:nx1 * ny1, 1] = np.tile(np.arange(ny1), nx1)
        offsets[nx1 * ny1:, 0] = np.repeat(np.arange(nx2) + 0.5, ny2)
        offsets[nx1 * ny1:, 1] = np.tile(np.arange(ny2), nx2) + 0.5
        offsets = offsets * [sx, sy] + [xmin, ymin]
        offsets, accum = offsets[good], accum[good]
        hexagon = [sx, sy / 3] * np.array([[.5, -.5], [.5, .5], [0., 1.], [-.5, .5],
                                           [-.5, -.5], [0., -1.]])
        polys = offsets[:, None, :] + hexagon[None]
        if xscale == 'log':
            polys[:, :, 0] = 10.0 ** polys[:, :, 0]
            xmin, xmax = 10.0 ** xmin, 10.0 ** xmax
            self.set_xscale('log')
        if yscale == 'log':
            polys[:, :, 1] = 10.0 ** polys[:, :, 1]
            ymin, ymax = 10.0 ** ymin, 10.0 ** ymax
            self.set_yscale('log')
        if isinstance(bins, str) and bins == 'log':
            if norm is None:
                norm = _colors_mod.LogNorm(vmin=vmin, vmax=vmax)
                vmin = vmax = None
            bins = None
        if bins is not None:
            if not np.iterable(bins):
                lo, hi = accum.min(), accum.max()
                b = bins - 1
                bins = lo + (hi - lo) * np.arange(b) / b
            accum = np.sort(bins).searchsorted(accum).astype(float)
        coll = ColoredPolyCollection(
            self, list(polys), accum, cmap, norm, vmin, vmax, edgecolors,
            rc('patch.linewidth') if linewidths is None else np.ravel(linewidths)[0],
            alpha=alpha, zorder=zorder,
            bounds=_Bounds(np.array([xmin, xmax]), np.array([ymin, ymax])))
        self._tight = True                              # matplotlib: autoscale_view(tight=True)
        return self._add(coll)

    def hist2d(self, x, y, bins=10, range=None, density=False, weights=None, cmin=None,
               cmax=None, **kw):
        """matplotlib's hist2d: (counts, xedges, yedges, the pcolormesh)."""
        h, xedges, yedges = np.histogram2d(np.ravel(x), np.ravel(y), bins=bins, range=range,
                                           density=density, weights=weights)
        if cmin is not None:
            h[h < cmin] = np.nan
        if cmax is not None:
            h[h > cmax] = np.nan
        pc = self.pcolormesh(xedges, yedges, np.ma.masked_invalid(h.T), **kw)
        self.set_xlim(xedges[0], xedges[-1])
        self.set_ylim(yedges[0], yedges[-1])
        return h, xedges, yedges, pc

    def matshow(self, Z, **kw):
        """matplotlib's matshow: imshow (nearest, equal aspect, origin upper),
        the x ticks above, integer ticks."""
        from . import ticker
        Z = np.asanyarray(Z)
        kw = {'origin': 'upper', 'interpolation': 'nearest', 'aspect': 'equal', **kw}
        im = self.imshow(Z, **kw)
        self.xaxis.tick_top()
        self.xaxis.set_ticks_position('both')
        for axis in (self.xaxis, self.yaxis):
            axis.set_major_locator(ticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10],
                                                      integer=True))
        return im
