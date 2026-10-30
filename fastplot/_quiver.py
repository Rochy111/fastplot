"""matplotlib's quiver: arrows sized and placed as its Quiver makes them.

    q = ax.quiver(x, y, u, v)                    # auto width and scale
    ax.quiver(x, y, u, v, speed, cmap='viridis', scale=300, width=0.003)

An arrow is matplotlib's polygon (Quiver._h_arrows) in arrow-width units:
shaft and head (headwidth, headlength, headaxislength), shrunk to a head
alone below minshaft head lengths, a dot below minlength; turned by the
vector's angle on the screen ('uv', the axes' own turning on maps) or
along the data ('xy'), and scaled to pixels by `units` ('width': the axes
width, 'x', 'y', 'xy', 'height', 'dots', 'inches'). With no width, it is
0.06 axes widths over clip(sqrt(n), 8, 25); with no scale, matplotlib's
1.8 * mean length * max(10, sqrt(n)) / span. Filled, no edge (linewidth
0), as matplotlib draws them.
"""
import math

import numpy as np

from . import _color
from ._artists import Artist, Coloring, _Mappable, _PointsArtist, _as_host, _axis_bounds
from ._style import rgba

_PIVOTS = {'tail': 'tail', 'mid': 'middle', 'middle': 'middle', 'tip': 'tip'}


def parse_args(args):
    """quiver([X, Y], U, V, [C]): X, Y, U, V, C (C None) as matplotlib reads them."""
    if len(args) not in (2, 3, 4, 5):
        raise TypeError('quiver(X, Y, U, V[, C]) or quiver(U, V[, C])')
    a = [np.asarray(_as_host(v)) for v in args]
    if len(a) in (2, 3):
        U, V = a[0], a[1]
        C = a[2] if len(a) == 3 else None
        nr, nc = (1, U.shape[0]) if U.ndim == 1 else U.shape
        X, Y = np.meshgrid(np.arange(nc), np.arange(nr))
    else:
        X, Y, U, V = a[:4]
        C = a[4] if len(a) == 5 else None
        if X.ndim == 1 and Y.ndim == 1 and U.ndim == 2 and U.shape == (Y.size, X.size):
            X, Y = np.meshgrid(X, Y)
    X, Y = np.asarray(X, float).ravel(), np.asarray(Y, float).ravel()
    n = X.size

    def flat(v):
        v = np.ma.masked_invalid(np.ma.asarray(v, float)).ravel()
        if v.size not in (n, 1):
            raise ValueError('quiver: an argument has %d values for %d arrows' % (v.size, n))
        return np.ma.resize(v, n) if v.size == 1 else v
    return X, Y, flat(U), flat(V), None if C is None else flat(C)


class Quiver(_Mappable, _PointsArtist):
    zorder = 1.0

    def __init__(self, ax, X, Y, U, V, C, src, scale=None, headwidth=3, headlength=5,
                 headaxislength=4.5, minshaft=1, minlength=1, units='width', scale_units=None,
                 angles='uv', width=None, color='k', pivot='tail', cmap=None, norm=None,
                 vmin=None, vmax=None, alpha=None, zorder=None, label=None):
        super().__init__(ax, zorder, alpha)
        self.x, self.y, self.src = X, Y, src
        mask = np.ma.getmaskarray(U) | np.ma.getmaskarray(V)
        if C is not None:
            mask |= np.ma.getmaskarray(C)
        self.mask = mask
        self.U, self.V = U.filled(1.0), V.filled(1.0)
        self.values = None if C is None else C.filled(np.nan).astype(np.float32)
        self.N = X.size
        self.scale, self.width = scale, width
        self.headwidth, self.headlength = headwidth, float(headlength)
        self.headaxislength, self.minshaft, self.minlength = headaxislength, minshaft, minlength
        self.units, self.scale_units, self.angles = units, scale_units, angles
        if pivot.lower() not in _PIVOTS:
            raise ValueError('pivot must be tail, middle or tip')
        self.pivot = _PIVOTS[pivot.lower()]
        self.color = color
        self.cmap = _color.get_cmap(cmap)
        self._norm, self._norm_in, self._vmin, self._vmax = None, norm, vmin, vmax
        self.label = label

    # -- the colour scale (a C argument)
    def norm_spec(self):
        if self._norm is None:
            self._norm = _color.make_norm(self._norm_in, vmin=self._vmin, vmax=self._vmax,
                                          data=self.values, N=self.cmap.N)
        return self._norm

    @property
    def norm(self):
        return self.norm_spec()

    def coloring(self):
        return Coloring(cmap=self.cmap, norm=self.norm_spec())

    def set_UVC(self, U, V, C=None):
        _, _, U, V, C = parse_args((self.x, self.y, U, V) + (() if C is None else (C,)))
        self.__init__(self.axes, self.x, self.y, U, V, C, self.src, self.scale, self.headwidth,
                      self.headlength, self.headaxislength, self.minshaft, self.minlength,
                      self.units, self.scale_units, self.angles, self.width, self.color,
                      self.pivot, self.cmap, self._norm_in, self._vmin, self._vmax, self.alpha,
                      self.zorder, self.label)

    def data_bounds(self):                              # matplotlib's: the arrows' positions
        if self.src != 'data':
            return None
        return _axis_bounds(self.x), _axis_bounds(self.y)

    def legend_style(self):
        from ._plots import _one_color
        c = self.color if _one_color(self.color) else list(self.color)[0]
        return dict(kind='patch', facecolor=rgba(c, self.alpha), edgecolor=None,
                    linewidth=0.0, linestyle='solid')

    # -- geometry (matplotlib's Quiver)
    def _dots_per_unit(self, ax, units, dpi):
        x0, y0, x1, y1 = ax.box_f
        bw, bh = x1 - x0, y1 - y0
        e = ax.extent if hasattr(ax, 'extent') else (0.0, 1.0, 0.0, 1.0)
        vw, vh = abs(e[1] - e[0]), abs(e[3] - e[2])
        table = {'x': bw / vw, 'y': bh / vh, 'xy': math.hypot(bw, bh) / math.hypot(vw, vh),
                 'width': bw, 'height': bh, 'dots': 1.0, 'inches': dpi}
        if units not in table:
            raise ValueError('quiver units must be one of %s' % ', '.join(table))
        return table[units]

    def _h_arrows(self, length):
        """matplotlib's Quiver._h_arrows: (X, Y) (n, 8), arrow-width units."""
        minsh = self.minshaft * self.headlength
        n = len(length)
        length = np.clip(length.reshape(n, 1), 0, 2 ** 16)
        x = np.array([0, -self.headaxislength, -self.headlength, 0], np.float64)
        x = x + np.array([0, 1, 1, 1]) * length
        y = 0.5 * np.array([1, 1, self.headwidth, 0], np.float64)
        y = np.repeat(y[np.newaxis, :], n, axis=0)
        x0 = np.array([0, minsh - self.headaxislength, minsh - self.headlength, minsh],
                      np.float64)
        y0 = 0.5 * np.array([1, 1, self.headwidth, 0], np.float64)
        ii = [0, 1, 2, 3, 2, 1, 0, 0]
        X, Y = x[:, ii], y[:, ii]
        Y[:, 3:-1] *= -1
        X0, Y0 = x0[ii], y0[ii]
        Y0[3:-1] *= -1
        shrink = length / minsh if minsh != 0. else 0.
        X0 = shrink * X0[np.newaxis, :]
        Y0 = shrink * Y0[np.newaxis, :]
        short = np.repeat(length < minsh, 8, axis=1)
        np.copyto(X, X0, where=short)
        np.copyto(Y, Y0, where=short)
        if self.pivot == 'middle':
            X -= 0.5 * X[:, 3, np.newaxis]
        elif self.pivot == 'tip':
            X = X - X[:, 3, np.newaxis]
        tooshort = length < self.minlength
        if tooshort.any():                              # a heptagonal dot
            th = np.arange(0, 8, 1, np.float64) * (np.pi / 3.0)
            X1 = np.repeat((np.cos(th) * self.minlength * 0.5)[np.newaxis, :], n, axis=0)
            Y1 = np.repeat((np.sin(th) * self.minlength * 0.5)[np.newaxis, :], n, axis=0)
            tooshort = np.repeat(tooshort, 8, 1)
            np.copyto(X, X1, where=tooshort)
            np.copyto(Y, Y1, where=tooshort)
        return X, Y

    def _eps(self):
        """matplotlib's step along the data for 'xy' angles and lengths."""
        if self.angles == 'xy' and self.scale_units == 'xy':
            return 1.0
        v = np.r_[self.x, self.y]
        v = np.abs(v[np.isfinite(v)])
        return (v.max() * 0.001 if v.size else 0.0) or 1.0

    def _along_data(self, ax, x, y, u, v):
        """Screen angles (y up) and px lengths of the data vectors (u, v) at
        (x, y): matplotlib's _angles_lengths."""
        eps = self._eps()
        px, py = ax._points_px(x, y, self.src)
        qx, qy = ax._points_px(x + eps * u, y + eps * v, self.src)
        return np.arctan2(-(qy - py), qx - px), np.hypot(qx - px, qy - py) / eps

    def _scale(self, r, ax):
        """(px per arrow-width unit, width, scale, width units per length
        unit) as matplotlib's Quiver sets them, the auto scale from these
        vectors."""
        dx = self._dots_per_unit(ax, self.units, r.dpi)
        span = (ax.box_f[2] - ax.box_f[0]) / dx
        width = self.width
        if width is None:
            width = 0.06 * span / np.clip(math.sqrt(self.N), 8, 25)
        scale = self.scale
        if self.scale_units is None:
            per = 1.0
        else:
            per = (1.0 if self.scale_units == 'xy' else
                   self._dots_per_unit(ax, self.scale_units, r.dpi)) / dx
        if scale is None:                               # matplotlib's crude autoscaling
            if isinstance(self.angles, str) and self.scale_units == 'xy':
                a = self._along_data(ax, self.x, self.y, self.U, self.V)[1]
            else:
                a = np.hypot(self.U, self.V)
            sn = max(10, math.sqrt(self.N))
            good = a[~self.mask]
            amean = good.mean() if good.size else 1.0
            scale = 1.8 * amean * sn / span * per
        return dx, width, scale, per

    def render(self, r, ax):
        if not self.N:
            return
        dx, width, scale, per = self._scale(r, ax)
        self.scale_used = scale
        px, py, sx, sy, _ = ax._vectors_px(self.x, self.y, self.U, self.V, self.src)
        str_angles = self.angles if isinstance(self.angles, str) else ''
        if str_angles == 'xy' or self.scale_units == 'xy':
            xy_angles, xy_lengths = self._along_data(ax, self.x, self.y, self.U, self.V)
        a = xy_lengths if (str_angles and self.scale_units == 'xy') else np.hypot(self.U, self.V)
        length = a * (per / (scale * width))
        X, Y = self._h_arrows(length)
        if str_angles == 'xy':
            theta = xy_angles
        elif str_angles == 'uv':
            theta = np.arctan2(-sy, sx)                 # the screen's turn, y up
        else:
            theta = np.where(np.isfinite(self.angles), np.deg2rad(self.angles), 0.0) * \
                np.ones(self.N)
        theta = theta.reshape(-1, 1)
        c, s = np.cos(theta), np.sin(theta)
        ox = (X * c - Y * s) * width * dx
        oy = (X * s + Y * c) * width * dx
        ok = ~self.mask & np.isfinite(px) & np.isfinite(py) & np.isfinite(theta[:, 0])
        if not ok.any():
            return
        vx = (px[ok, None] + ox[ok]).ravel()
        vy = (py[ok, None] - oy[ok]).ravel()             # y down
        xy = np.column_stack([vx, vy])
        n = int(ok.sum())
        offs = np.arange(n + 1) * 8
        from ._plots import _one_color
        per_arrow = self.values is None and self.color is not None and \
            not _one_color(self.color) and np.ndim(self.color) >= 1
        if self.values is None and not per_arrow:
            fc = rgba(self.color, self.alpha)
            if fc[3]:
                r.c.polygons(xy, offs, fc, ax.box_clip, 'nonzero')
            return
        if per_arrow:                                   # color=[...]: one per arrow, cycled
            cl = list(self.color)
            allc = np.array([rgba(cl[k % len(cl)], self.alpha) for k in range(ok.size)],
                            np.uint8)
            cols = allc[ok]
        else:
            cols = np.empty((1, n, 4), np.uint8)
            _color.shade(self.values[ok][None, :], self.cmap, self.norm_spec(), out=cols,
                         alpha=self.alpha, rounding='nearest')
            cols = cols[0]
        uniq, group = np.unique(cols.view(np.uint32).ravel(), return_inverse=True)
        uniq = uniq.view(np.uint8).reshape(-1, 4)
        v = xy.reshape(n, 8, 2)
        for k, col in enumerate(uniq):
            sel = np.nonzero(group == k)[0]
            if col[3]:
                r.c.polygons(v[sel].reshape(-1, 2), np.arange(sel.size + 1) * 8, col,
                             ax.box_clip, 'nonzero')


class QuiverKey(Artist):
    """matplotlib's quiverkey: an arrow of length U (the Quiver's scale) at
    (X, Y) in `coordinates`, its label labelsep inches off at labelpos."""

    _HALIGN = {'N': 'center', 'S': 'center', 'E': 'left', 'W': 'right'}
    _VALIGN = {'N': 'bottom', 'S': 'top', 'E': 'center', 'W': 'center'}
    _PIVOT = {'N': 'middle', 'S': 'middle', 'E': 'tip', 'W': 'tail'}

    def __init__(self, Q, X, Y, U, label, *, angle=0, coordinates='axes', color=None,
                 labelsep=0.1, labelpos='N', labelcolor=None, fontproperties=None,
                 zorder=None, **kw):
        super().__init__(Q.axes, Q.zorder + 0.1 if zorder is None else zorder)
        if labelpos not in self._PIVOT:
            raise ValueError("labelpos must be 'N', 'S', 'E' or 'W'")
        if coordinates not in ('data', 'axes', 'figure', 'inches'):
            raise ValueError("coordinates must be 'data', 'axes', 'figure' or 'inches'")
        self.Q, self.X, self.Y, self.U, self.label = Q, X, Y, U, label
        self.angle, self.coord, self.color = angle, coordinates, color
        self.labelsep, self.labelpos, self.labelcolor = labelsep, labelpos, labelcolor
        fp = fontproperties or {}
        self.fontsize = fp.get('size', kw.pop('fontsize', None))

    def _xy(self, r, ax):
        if self.coord == 'axes':
            x0, y0, x1, y1 = ax.box_f
            return x0 + self.X * (x1 - x0), y1 - self.Y * (y1 - y0)
        if self.coord == 'figure':
            return self.X * r.W + r.ox, (1 - self.Y) * r.H + r.oy
        if self.coord == 'inches':
            return self.X * r.dpi + r.ox, r.H - self.Y * r.dpi + r.oy
        px, py = ax._points_px(np.array([self.X], float), np.array([self.Y], float),
                               self.Q.src)
        return px[0], py[0]

    def render(self, r, ax):
        from ._style import fontsize_pt, rc
        Q = self.Q
        dx, width, scale, per = Q._scale(r, ax)
        th = np.radians(self.angle)
        u, v = self.U * np.cos(th), self.U * np.sin(th)
        a = np.array([math.hypot(u, v)])
        if Q.scale_units == 'xy':                       # a length along the data at (0, 0)
            a = Q._along_data(ax, np.zeros(1), np.zeros(1), np.array([u]), np.array([v]))[1]
        pivot, Q.pivot = Q.pivot, self._PIVOT[self.labelpos]
        try:
            X, Y = Q._h_arrows(a * (per / (scale * width)))
        finally:
            Q.pivot = pivot
        theta = math.atan2(v, u)
        c, s = math.cos(theta), math.sin(theta)
        x0, y0 = self._xy(r, ax)
        vx = x0 + (X[0] * c - Y[0] * s) * width * dx
        vy = y0 - (X[0] * s + Y[0] * c) * width * dx
        from ._plots import _one_color
        qc = Q.color if _one_color(Q.color) else list(Q.color)[0]
        col = rgba(self.color if self.color is not None else qc, Q.alpha)
        if col[3]:
            r.c.polygons(np.column_stack([vx, vy]), np.array([0, 8]), col, None, 'nonzero')
        sep = self.labelsep * r.dpi
        tx, ty = {'N': (0, -sep), 'S': (0, sep), 'E': (sep, 0), 'W': (-sep, 0)}[self.labelpos]
        r.text(self.label, x0 + tx, y0 + ty, fontsize_pt(self.fontsize),
               self.labelcolor if self.labelcolor is not None else rc('text.color'),
               ha=self._HALIGN[self.labelpos], va=self._VALIGN[self.labelpos])
