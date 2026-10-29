"""What axes hold, besides fields: the Artist base, projecting points and
lines onto a map (seam-safe), Natural Earth features, lines and markers,
scatter, barbs, text."""
import threading
from collections import OrderedDict

import numpy as np

from . import _barbs, _color, _draw, _features, _path, _proj
from . import _markers as _mk
from ._jit import njit, prange
from ._render import Coloring
from ._style import _dashes, fontsize_pt, rc, rgba


class Artist:
    zorder = 1.0

    def __init__(self, ax, zorder=None, alpha=None, visible=True):
        self.axes = ax
        if zorder is not None:
            self.zorder = zorder
        self.alpha = alpha
        self.visible = visible

    def set_zorder(self, z):
        self.zorder = z

    def set_alpha(self, a):
        self.alpha = a

    def set_visible(self, v):
        self.visible = bool(v)

    def remove(self):
        if self in self.axes.artists:
            self.axes.artists.remove(self)

    def datalim(self, proj):
        return None

    def data_bounds(self):
        """Plain axes' autoscaling: per axis (lo, hi, smallest positive,
        sticky edges), or None for an artist that does not count."""
        return None

    def work(self):
        return 0


def _axis_bounds(v, sticky=()):
    """(lo, hi, minpos, sticky) of finite values."""
    v = np.asarray(v, np.float64).ravel()
    v = v[np.isfinite(v)]
    if not v.size:
        return (np.inf, -np.inf, np.inf, tuple(sticky))
    pos = v[v > 0]
    return (v.min(), v.max(), pos.min() if pos.size else np.inf, tuple(sticky))


def _as_host(a):
    """numpy from numpy or cupy (fastgrib's gpu=True) arrays."""
    if hasattr(a, '__cuda_array_interface__') and hasattr(a, 'get'):
        return a.get()
    return np.asarray(a)


def _on_gpu(a):
    return hasattr(a, '__cuda_array_interface__')


def _wrap_x(X, proj, x0=None):
    """Cylindrical maps: x into one period from x0 (the map's left edge,
    or the projection's), as a point at 354 deg E lands at 6 deg W."""
    if not proj.cylindrical or not proj.period:
        return X
    x0 = proj.x_limits[0] if x0 is None else x0
    return x0 + np.mod(np.asarray(X, np.float64) - x0, proj.period)


def _lims(X, Y, proj):
    """Data limits of points in proj (autoscaling)."""
    X = _wrap_x(X, proj)
    ok = np.isfinite(X) & np.isfinite(Y)
    if not ok.any():
        return None
    return X[ok].min(), X[ok].max(), Y[ok].min(), Y[ok].max()


_GEOM = OrderedDict()                   # key -> projected geometry, least recent first
_GEOM_BYTES = [0]
_GEOM_MAX = 256 << 20                   # bytes kept
_GEOM_LOCK = threading.Lock()


def _nbytes(v):
    if isinstance(v, np.ndarray):
        return v.nbytes
    if isinstance(v, (tuple, list)):
        return sum(_nbytes(u) for u in v)
    if isinstance(v, dict):
        return sum(_nbytes(u) for u in v.values())
    return 0


def cached_geometry(key, make):
    """Geometry that depends only on key (a feature, a projection, an extent,
    a pixel box ...) made once and kept across figures: a new figure of the
    same map -- an animation's next frame -- reuses its coastlines, borders
    and graticule as projected, segmented and binned. Least recently used
    first out past _GEOM_MAX bytes."""
    with _GEOM_LOCK:
        if key in _GEOM:
            _GEOM.move_to_end(key)
            return _GEOM[key]
    value = make()
    size = _nbytes(value)
    with _GEOM_LOCK:
        if key not in _GEOM:
            _GEOM[key] = value
            _GEOM_BYTES[0] += size
            while _GEOM_BYTES[0] > _GEOM_MAX and len(_GEOM) > 1:
                _, old = _GEOM.popitem(last=False)
                _GEOM_BYTES[0] -= _nbytes(old)
        return _GEOM[key]


def _points(a):
    """Coordinates as a flat float64 array (numpy or cupy in)."""
    return np.asarray(_as_host(a), np.float64).ravel()


def _gather(offsets, parts):
    """The indices of parts (polylines, rings) [offsets[k], offsets[k+1])
    for k in parts, concatenated."""
    starts = offsets[parts]
    lens = offsets[parts + 1] - starts
    ramp = np.arange(int(lens.sum())) - np.repeat(np.cumsum(lens) - lens, lens)
    return np.repeat(starts, lens) + ramp


class _PointsArtist(Artist):
    """An artist at points (x, y) given in a crs."""

    def datalim(self, proj):
        X, Y = _proj.transform_points(self.src, proj, self.x, self.y)
        return _lims(X, Y, proj)

    def _screen(self, ax):
        """The points in canvas pixels."""
        return ax._points_px(self.x, self.y, self.src)

    def data_bounds(self):
        if getattr(self, '_bounds', None) is not None:     # (axhline: its own say)
            return self._bounds.get()
        if self.src != 'data':
            return None
        return _axis_bounds(self.x), _axis_bounds(self.y)


class _Mappable(Artist):
    """Something a colorbar can show."""

    def get_cmap(self):
        return self.cmap

    def set_cmap(self, cmap):
        self.cmap = _color.get_cmap(cmap)
        self._coloring = None

    def set_clim(self, vmin=None, vmax=None):
        if isinstance(vmin, (tuple, list)):
            vmin, vmax = vmin
        self._vmin, self._vmax = vmin, vmax
        self._coloring = None

    def get_clim(self):
        n = self.norm_spec()
        return n.vmin, n.vmax


@njit(cache=True, nogil=True)
def _unwrap(X, offsets, period, xlo, xhi, shift, need):
    """_unwrap_copies' first part, in place on X (a copy): each polyline made
    continuous -- one running sum of the rounded jumps over the whole array,
    less its value at the polyline's start, as np.cumsum would -- then
    shifted a whole number of periods to sit over [xlo, xhi]. need[k] gets
    1 (copy a period right), 2 (left), 3 (both) or 0."""
    n = X.size
    nparts = offsets.size - 1
    run = np.empty(n)
    run[0] = 0.0
    first = np.zeros(n, np.bool_)
    for k in range(1, nparts):
        if offsets[k] < n:
            first[offsets[k]] = True
    c = 0.0
    for k in range(n - 1):
        adj = 0.0 if first[k + 1] else -period * np.round((X[k + 1] - X[k]) / period)
        if not np.isfinite(adj):
            adj = 0.0
        c = adj if k == 0 else c + adj                  # np.cumsum: the first as it is
        run[k + 1] = c
    centre = 0.5 * (xlo + xhi)
    for p in range(nparts):
        a, b = offsets[p], offsets[p + 1]
        base = run[a] if a < n else 0.0
        mn, mx = np.inf, -np.inf
        for k in range(a, b):
            X[k] += run[k] - base
            v = X[k]
            if np.isfinite(v):
                mn = min(mn, v)
                mx = max(mx, v)
        if a == b:                                      # as np.minimum.reduceat for an empty part
            mn = mx = X[a] if (a < n and np.isfinite(X[a])) else np.nan
        mid = 0.5 * (mn + mx)
        sh = -period * np.round((mid - centre) / period) if np.isfinite(mid) else 0.0
        shift[p] = sh
        for k in range(a, b):
            X[k] += sh
        need[p] = 0
        if np.isfinite(mid):
            need[p] = (1 if mn + sh < xlo else 0) + (2 if mx + sh > xhi else 0)


def _unwrap_copies(X, Y, offsets, period, xlo, xhi):
    """Cylindrical maps: each polyline made continuous in x (no jumps of
    more than half a period), then copied a period left or right where it
    runs off the map -- so lines crossing the map's seam are drawn whole.
    Also returns, per polyline out, the polyline in it came from."""
    nparts = offsets.size - 1
    if period <= 0 or X.size == 0:
        return X, Y, offsets, np.arange(nparts)
    X = X.copy()
    shift = np.zeros(nparts)
    need = np.zeros(nparts, np.int64)
    _unwrap(X, offsets, float(period), float(xlo), float(xhi), shift, need)
    lens = np.diff(offsets)
    xs, ys, offs, parts = [X], [Y], [offsets], [np.arange(nparts)]
    base = X.size
    for sgn, bit in ((1.0, 1), (-1.0, 2)):
        idx = np.nonzero(need & bit)[0]
        if not idx.size:
            continue
        sel = _gather(offsets, idx)
        xs.append(X[sel] + sgn * period)
        ys.append(Y[sel])
        offs.append(base + np.cumsum(lens[idx]))
        parts.append(idx)
        base += sel.size
    if len(xs) == 1:
        return X, Y, offsets, parts[0]
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(offs), np.concatenate(parts)


@njit(cache=True, nogil=True, parallel=True)
def _to_px(X, Y, x0, e0, sx, y0, e3, sy, out):
    """MapAxes.to_px, point by point, into out (n, 2)."""
    for k in prange(X.size):
        out[k, 0] = x0 + (X[k] - e0) * sx
        out[k, 1] = y0 + (e3 - Y[k]) * sy


def _project_lines(ax, x, y, offsets, src, parts=False):
    """Polylines (in src) -> canvas pixels, seam-safe; parts=True also
    gives each output polyline's input polyline."""
    X, Y = _proj.transform_points(src, ax.projection, np.asarray(x, np.float64),
                                  np.asarray(y, np.float64))
    offsets = np.asarray(offsets, np.int64)
    which = np.arange(offsets.size - 1)
    if ax.projection.cylindrical:
        X, Y, offsets, which = _unwrap_copies(X, Y, offsets, ax.projection.period,
                                              ax.extent[0], ax.extent[1])
    e = ax.extent
    xy = np.empty((X.size, 2), np.float64)
    if X.size:
        _to_px(np.ascontiguousarray(X, np.float64), np.ascontiguousarray(Y, np.float64),
               float(ax.box_f[0]), float(e[0]), float(ax._sx), float(ax.box_f[1]), float(e[3]),
               float(ax._sy), xy)
    out = (xy, np.asarray(offsets, np.int64))
    return out + (which,) if parts else out


class FeatureArtist(Artist):
    """A Natural Earth feature (cartopy's FeatureArtist): zorder 1.5."""

    zorder = 1.5

    def __init__(self, ax, feature, **kw):
        kw = {**feature.kwargs, **kw}
        super().__init__(ax, kw.pop('zorder', None), kw.pop('alpha', None))
        self.feature = feature
        c = kw.pop('color', None)
        self.edgecolor = kw.pop('edgecolor', kw.pop('ec', c if c is not None else 'black'))
        self.facecolor = kw.pop('facecolor', kw.pop('fc', 'none' if c is None else c))
        self.linewidth = kw.pop('linewidth', kw.pop('lw', 1.0))
        self.linestyle = kw.pop('linestyle', kw.pop('ls', 'solid'))

    def _scale(self, ax):
        s = self.feature.scale
        return _features.scale_for(ax.lonlat_extent()) if s == 'auto' else s

    def render(self, r, ax):
        name, scale = self.feature.name, self._scale(ax)
        bbox = ax.lonlat_bbox()
        pc = _proj.PlateCarree()
        if _features.kind(name) == 'polygons' and rgba(self.facecolor)[3]:
            def rings():
                p = _features.polygons(name, scale, bbox=bbox)
                return _project_lines(ax, p.xy[:, 0], p.xy[:, 1], p.ring_offsets, pc)
            xy, off = cached_geometry(('feature-rings', name, scale, ax.view_key), rings)
            r.c.polygons(xy, off, rgba(self.facecolor, self.alpha), ax.box_clip)
        if rgba(self.edgecolor)[3] and self.linewidth > 0:
            def lines():
                ln = _features.lines(name, scale, bbox=bbox)
                return _draw.segments(*_project_lines(ax, ln.xy[:, 0], ln.xy[:, 1], ln.offsets,
                                                      pc)), {}
            segs, memo = cached_geometry(('feature-lines', name, scale, ax.view_key), lines)
            if segs.shape[0]:
                d = _dashes(self.linestyle, self.linewidth)
                styles = _draw.Styles(rgba(self.edgecolor, self.alpha), r.px(self.linewidth),
                                      None if d is None else [tuple(r.px(v) for v in d)])
                r.c.segments(segs, styles, ax.box_clip, memo)


class Line2D(_PointsArtist):
    """ax.plot: a line (and markers) through points in a crs."""

    zorder = 2.0

    def __init__(self, ax, x, y, src, color='C0', linewidth=None, linestyle='-',
                 marker=None, markersize=6.0, markerfacecolor=None, markeredgecolor=None,
                 markeredgewidth=1.0, alpha=None, zorder=None, label=None, fillstyle='full',
                 markerfacecoloralt='none'):
        super().__init__(ax, zorder, alpha)
        self.x, self.y, self.src = _points(x), _points(y), src
        self.color, self.linestyle = color, linestyle
        self.linewidth = rc('lines.linewidth') if linewidth is None else linewidth
        self.marker, self.markersize = marker, markersize
        self.mfc = markerfacecolor if markerfacecolor is not None else color
        self.mec = markeredgecolor if markeredgecolor is not None else color
        self.mew = markeredgewidth
        self.fillstyle, self.mfcalt = fillstyle, markerfacecoloralt
        self.label = label

    def legend_style(self):
        return dict(kind='line', color=rgba(self.color, self.alpha), linewidth=self.linewidth,
                    linestyle=self.linestyle, marker=self.marker, markersize=self.markersize,
                    mfc=rgba(self.mfc, self.alpha), mec=rgba(self.mec, self.alpha),
                    mew=self.mew, fillstyle=self.fillstyle, mfcalt=rgba(self.mfcalt, self.alpha))

    def render(self, r, ax):
        xy, off = ax._lines_px(self.x, self.y, np.array([0, self.x.size]), self.src)
        if self.linestyle not in ('None', 'none', '', ' ') and self.linewidth > 0:
            d = _dashes(self.linestyle, self.linewidth)
            cap = (_draw.BUTT if d is not None else _draw.PROJECTING) if ax._agg else \
                _draw.LEGACY
            r.lines(xy, off, rgba(self.color, self.alpha), r.px(self.linewidth), ax.box_clip,
                    None if d is None else tuple(r.px(v) for v in d), ax._snap, cap)
        if self.marker not in (None, 'None', 'none', ''):
            _markers(r, xy[:self.x.size], self.marker, r.px(self.markersize),
                     rgba(self.mfc, self.alpha), rgba(self.mec, self.alpha), r.px(self.mew),
                     ax.box_clip, snap=ax._snap, fillstyle=self.fillstyle,
                     fc_alt=rgba(self.mfcalt, self.alpha))


def _marker_paths(marker, size, ew, fillstyle, agg):
    """A marker's flattened (fill, fill_alt, stroke) at size px, relative
    to its centre -- each (xy, offsets) or None -- and its edge's cap. agg:
    as Agg's draw_markers, a rectilinear marker snapped to the pixel grid
    (for its edge width ew); the caps matplotlib's (else the maps' own)."""
    geom = _mk.marker_geom(marker, size, fillstyle)
    flats = [None if f is None else (f[0], f[1]) for f in geom.flat(0.1)]
    cap = _draw.CAPS.get(geom.capstyle, _draw.BUTT) if agg else _draw.LEGACY
    if agg and flats[2] is not None and _draw.rectilinear(*flats[2]).all():
        sv = float(_draw.snap_value(geom.edge_width(ew)))
        flats = [None if f is None else (np.floor(f[0] + 0.5) + sv, f[1]) for f in flats]
    return geom, flats, cap


def _per_marker(xy, offs, n, sel):
    """The parts of markers sel of n placed copies (translate_many's)."""
    m, k = xy.shape[0] // n, (offs.size - 1) // n
    sub = xy.reshape(n, m, 2)[sel].reshape(-1, 2)
    base = offs[:k] - offs[0]
    o = np.empty(sel.size * k + 1, np.int64)
    o[:-1] = (base[None, :] + m * np.arange(sel.size)[:, None]).ravel()
    o[-1] = sel.size * m
    return sub, o


def _markers(r, pts, marker, size, fc, ec, ew, clip, colors=None, snap=False,
             fillstyle='full', fc_alt=None, scatter=False):
    """matplotlib markers (_markers.marker_geom) at pts (px): size px across
    (markersize, or sqrt(s) for scatter, in px), faces fc -- or colors, one
    per marker -- edges ec (None: each its face colour) ew px wide.
    snap (not False) draws as Agg: one colour (draw_markers), each marker
    placed at whole pixels; matplotlib's caps. scatter: an unfilled
    marker's face is not filled, its edge takes the face colour."""
    agg = snap is not False
    geom, (fill, alt, stroke), cap = _marker_paths(marker, size, ew, fillstyle, agg)
    if not geom:
        return
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    ok = np.isfinite(pts).all(axis=1)
    if not ok.all():
        pts = pts[ok]
        colors = None if colors is None else colors[ok]
    if agg and colors is None:
        pts = np.floor(pts + 0.5)
    n = pts.shape[0]
    if not n:
        return
    if scatter and not geom.filled:
        fill = None
        ec = None
    face = fill is not None and fillstyle != 'none'
    w = geom.edge_width(ew)
    if agg and alt is None:                             # Agg: each marker in turn
        own = colors if colors is not None else fc
        face_col = own if face and (colors is not None or fc[3]) else None
        edge_col = None if stroke is None or w <= 0 else (own if ec is None else ec)
        if face_col is not None or edge_col is not None:
            r.c.stamp(pts, fill if face_col is not None else None,
                      stroke if edge_col is not None else None, w, face_col, edge_col, clip, cap)
        return
    if colors is not None:                              # one fill per distinct colour
        uniq, group = np.unique(colors.view(np.uint32).ravel(), return_inverse=True)
        uniq = uniq.view(np.uint8).reshape(-1, 4)
    if face:
        xy, offs = _path.translate_many(fill[0], fill[1], pts)
        if colors is None:
            if fc[3]:
                r.c.polygons(xy, offs, fc, clip, 'nonzero')
        else:
            for k, col in enumerate(uniq):
                if col[3]:
                    r.c.polygons(*_per_marker(xy, offs, n, np.nonzero(group == k)[0]), col,
                                 clip, 'nonzero')
    if alt is not None and fc_alt is not None and fc_alt[3]:
        r.c.polygons(*_path.translate_many(alt[0], alt[1], pts), fc_alt, clip, 'nonzero')
    if stroke is None or w <= 0:
        return
    xy, offs = _path.translate_many(stroke[0], stroke[1], pts)
    if ec is None and colors is not None:               # edges each in its marker's colour
        parts = (stroke[1].size - 1)
        segs = _draw.segments(xy, offs, np.repeat(group, parts).astype(np.int32))
        r.c.segments(segs, _draw.Styles(uniq, w, caps=cap), clip)
    else:
        col = fc if ec is None else ec
        if col[3]:
            r.lines(xy, offs, col, w, clip, cap=cap)


class Scatter(_Mappable, _PointsArtist):
    """ax.scatter: markers, one colour or coloured by values."""

    zorder = 1.0

    def __init__(self, ax, x, y, src, s=20.0, c=None, color=None, marker='o', cmap=None,
                 norm=None, vmin=None, vmax=None, edgecolors='face', linewidths=1.0,
                 alpha=None, zorder=None):
        super().__init__(ax, zorder, alpha)
        self.x, self.y, self.src = _points(x), _points(y), src
        self.s = float(np.mean(s)) if np.ndim(s) else float(s)
        self.marker = marker
        self.cmap = _color.get_cmap(cmap)
        self.values = None
        self.color = color if color is not None else 'C0'
        if c is not None:
            ca = np.asarray(c)
            if ca.dtype.kind in 'fiu' and ca.size == self.x.size and ca.ndim == 1:
                self.values = ca.astype(np.float32)
            else:
                self.color = c
        self._norm = None
        self._norm_in, self._vmin, self._vmax = norm, vmin, vmax
        self.edgecolors, self.linewidths = edgecolors, linewidths

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

    def _colors(self):
        if self.values is None:
            return None
        out = np.empty((1, self.values.size, 4), np.uint8)
        _color.shade(self.values[None, :], self.cmap, self.norm_spec(), out=out,
                     alpha=self.alpha, rounding='nearest')
        return out[0]

    def _edge(self, fc):
        """(edge colour, edge width in pt)."""
        face = isinstance(self.edgecolors, str) and self.edgecolors == 'face'
        none = isinstance(self.edgecolors, str) and self.edgecolors == 'none'
        return (fc if face else rgba(self.edgecolors, self.alpha)), \
            (0.0 if none else self.linewidths)

    def legend_style(self):
        colors = self._colors()
        fc = colors[0] if colors is not None and len(colors) else rgba(self.color, self.alpha)
        ec, ew = self._edge(fc)
        return dict(kind='scatter', marker=self.marker, s=self.s, facecolor=fc, edgecolor=ec,
                    ew=ew)

    def render(self, r, ax):
        pts = np.column_stack(self._screen(ax))
        size = np.sqrt(self.s) * r.dpi / 72.0           # s is the marker's area in pt^2
        fc = rgba(self.color, self.alpha)
        colors = self._colors()
        ec, ew = self._edge(None if colors is not None else fc)
        _markers(r, pts, self.marker, size, fc, ec, r.px(ew), ax.box_clip, colors, ax._snap,
                 scatter=True)


class Barbs(_PointsArtist):
    """ax.barbs, matplotlib's barbs (cartopy rotating the vectors)."""

    zorder = 1.0

    def __init__(self, ax, x, y, u, v, src, length=7, pivot='tip', barbcolor=None,
                 flagcolor=None, color='k', sizes=None, fill_empty=False, rounding=True,
                 barb_increments=None, flip_barb=False, linewidth=1.0, alpha=None, zorder=None,
                 regrid_shape=None):
        super().__init__(ax, zorder, alpha)
        u = np.asarray(_as_host(u), np.float64)
        x = np.asarray(_as_host(x), np.float64)
        y = np.asarray(_as_host(y), np.float64)
        if x.ndim == 1 and y.ndim == 1 and u.ndim == 2 and u.shape == (y.size, x.size):
            x, y = np.meshgrid(x, y)
        self.x, self.y, self.src = x.ravel(), y.ravel(), src
        self.u, self.v = u.ravel(), _points(v)
        self.length, self.pivot = length, pivot
        self.barbcolor = barbcolor if barbcolor is not None else color
        self.flagcolor = flagcolor if flagcolor is not None else color
        sz = dict(spacing=0.125, height=0.4, width=0.25, emptybarb=0.15)
        sz.update(sizes or {})
        self.sizes = sz
        inc = dict(half=5, full=10, flag=50)
        inc.update(barb_increments or {})
        self.inc = inc
        self.fill_empty, self.rounding, self.flip = fill_empty, rounding, flip_barb
        self.linewidth = linewidth
        self.regrid_shape = regrid_shape                # accepted, unused: subsample by slicing

    def render(self, r, ax):
        x = self.x
        px, py, dx, dy, mag = ax._vectors_px(self.x, self.y, self.u, self.v, self.src)
        ok = np.isfinite(px) & np.isfinite(py) & np.isfinite(dx) & np.isfinite(dy) & \
            np.isfinite(mag)
        b = ax.box
        reach = self.length * self.length / 2.0 * r.dpi / 72.0 * 1.1
        ok &= (px > b[0] - reach) & (px < b[2] + reach) & (py > b[1] - reach) & (py < b[3] + reach)
        if not ok.any():
            return
        # matplotlib draws the barb's vertices (length units) through a size
        # transform of sqrt(length^2 / 4) * dpi / 72: the staff is length^2/2 points
        L = self.length * self.length / 2.0 * r.dpi / 72.0
        xy, offs, empty = _barbs.barbs(px[ok], py[ok], dx[ok], dy[ok], mag[ok], length=L,
                                       fill_empty=self.fill_empty,
                                       pivot=self.pivot, rounding=self.rounding,
                                       flip=np.broadcast_to(np.asarray(self.flip), x.shape)[ok],
                                       **self.sizes, **self.inc)
        fc = rgba(self.flagcolor, self.alpha)
        ec = rgba(self.barbcolor, self.alpha)
        lens = np.diff(offs)
        full = np.nonzero(~empty)[0]
        if full.size and fc[3]:
            r.c.polygons(xy[_gather(offs, full)], np.r_[0, np.cumsum(lens[full])], fc,
                         ax.box_clip, 'nonzero')
        if self.fill_empty and (~empty).sum() != empty.size and fc[3]:
            emp = np.nonzero(empty)[0]
            r.c.polygons(xy[_gather(offs, emp)], np.r_[0, np.cumsum(lens[emp])], fc, ax.box_clip)
        if ec[3] and self.linewidth > 0:                # outlines: every ring closed
            roff = np.r_[0, np.cumsum(lens + 1)]
            ring = np.empty(roff[-1], np.int64)
            last = roff[1:] - 1
            body = np.ones(ring.size, bool)
            body[last] = False
            ring[body] = np.arange(offs[-1])
            ring[last] = offs[:-1]
            r.lines(xy[ring], roff, ec, r.px(self.linewidth), ax.box_clip)


class Text(Artist):
    """ax.text / fig.text."""

    zorder = 3.0

    def __init__(self, owner, x, y, s, coords, fontsize=None, color='black', ha='left',
                 va='baseline', rotation=0.0, fontweight='normal', fontstyle='normal',
                 bbox=None, alpha=None, zorder=None, rotation_mode='default', clip_on=False,
                 family=None):
        super().__init__(owner, zorder, alpha)
        self.x, self.y, self.s = x, y, s
        self.coords = coords                            # 'data' crs, 'axes', 'figure'
        self.fontsize, self.color = fontsize, color
        self.ha, self.va, self.rotation = ha, va, rotation
        self.weight, self.style = fontweight, fontstyle
        self.bbox, self.rotation_mode, self.clip_on = bbox, rotation_mode, clip_on
        self.family = family

    def set_text(self, s):
        self.s = s

    def set_position(self, xy):
        self.x, self.y = xy

    def _xy(self, r, ax):
        if self.coords == 'figure':
            return self.x * r.W, (1 - self.y) * r.H
        if self.coords == 'axes':
            b = ax.box_f
            return b[0] + self.x * (b[2] - b[0]), b[3] - self.y * (b[3] - b[1])
        x, y = np.array([self.x], float), np.array([self.y], float)
        if hasattr(self.coords, 'to_px'):               # transforms: transFigure, blended, ...
            px, py = self.coords.to_px(x, y)
        else:
            px, py = ax._points_px(x, y, self.coords)
        return px[0], py[0]

    def render(self, r, ax):
        x, y = self._xy(r, ax)
        if not (np.isfinite(x) and np.isfinite(y)):
            return
        r.text(self.s, x, y, fontsize_pt(self.fontsize), self.color, ha=self.ha, va=self.va,
               rotation=self.rotation, weight=self.weight, style=self.style,
               rotation_mode=self.rotation_mode, bbox=self.bbox, alpha=self.alpha,
               family=self.family)


def _text_kw(kw):
    """matplotlib's Text keyword aliases."""
    out = {}
    out['fontsize'] = kw.pop('fontsize', kw.pop('size', None))
    out['color'] = kw.pop('color', kw.pop('c', 'black'))
    out['ha'] = kw.pop('ha', kw.pop('horizontalalignment', 'left'))
    out['va'] = kw.pop('va', kw.pop('verticalalignment', 'baseline'))
    out['rotation'] = kw.pop('rotation', 0.0)
    if out['rotation'] in ('horizontal', None):
        out['rotation'] = 0.0
    elif out['rotation'] == 'vertical':
        out['rotation'] = 90.0
    out['fontweight'] = kw.pop('fontweight', kw.pop('weight', 'normal'))
    out['fontstyle'] = kw.pop('fontstyle', kw.pop('style', 'normal'))
    out['bbox'] = kw.pop('bbox', None)
    out['alpha'] = kw.pop('alpha', None)
    out['zorder'] = kw.pop('zorder', None)
    out['rotation_mode'] = kw.pop('rotation_mode', 'default')
    out['clip_on'] = kw.pop('clip_on', False)
    out['family'] = kw.pop('family', kw.pop('fontfamily', None))
    kw.pop('transform', None)
    for k in ('fontname', 'wrap', 'zorder', 'path_effects', 'usetex', 'label'):
        kw.pop(k, None)
    if kw:
        raise TypeError('unsupported text keywords: %s' % ', '.join(kw))
    return out
