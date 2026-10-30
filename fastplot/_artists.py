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


_ALIASES = {'fc': 'facecolor', 'ec': 'edgecolor', 'lw': 'linewidth', 'ls': 'linestyle',
            'c': 'color'}                               # matplotlib's, for Artist.set


class Artist:
    zorder = 1.0

    def __init__(self, ax, zorder=None, alpha=None, visible=True):
        self.axes = ax
        if zorder is not None:
            self.zorder = zorder
        self.alpha = alpha
        self.visible = visible

    def set(self, **kw):
        """matplotlib's Artist.set: each keyword (or its short alias: fc, ec,
        lw, ls, c) through the artist's set_<name>."""
        out = []
        for k, v in kw.items():
            f = getattr(self, 'set_' + _ALIASES.get(k, k), None)
            if f is None:
                raise AttributeError('%s.set() got an unexpected keyword argument %r'
                                     % (type(self).__name__, k))
            out.append(f(v))
        return out

    def set_zorder(self, z):
        self.zorder = z

    def get_zorder(self):
        return self.zorder

    def set_alpha(self, a):
        self.alpha = a

    def set_visible(self, v):
        self.visible = bool(v)

    def get_visible(self):
        return self.visible

    def remove(self):
        if self in self.axes.artists:
            self.axes.artists.remove(self)
            b = self.data_bounds() if getattr(self.axes, '_data_limits', None) else None
            if b is not None:                           # matplotlib's data limits keep it
                self.axes.__dict__.setdefault('_stale_bounds', []).append(b)   # until relim()

    def datalim(self, proj):
        return None

    def data_bounds(self):
        """Plain axes' autoscaling: per axis (lo, hi, smallest positive,
        sticky edges), or None for an artist that does not count."""
        return None

    def work(self):
        return 0


@njit(cache=True, nogil=True)
def _bounds(v):
    """(min, max, smallest positive) of v's finite values, in one pass
    (inf, -inf, inf: none) -- exact, as numpy's in any order."""
    lo, hi, pos = np.inf, -np.inf, np.inf
    for i in range(v.size):
        x = v[i]
        if np.isfinite(x):
            if x < lo:
                lo = x
            if x > hi:
                hi = x
            if x > 0.0 and x < pos:
                pos = x
    return lo, hi, pos


def _axis_bounds(v, sticky=()):
    """(lo, hi, minpos, sticky) of finite values."""
    v = np.asarray(v, np.float64).ravel()
    if v.size < 64:                                     # (not worth a kernel call)
        v = v[np.isfinite(v)]
        if not v.size:
            return (np.inf, -np.inf, np.inf, tuple(sticky))
        minpos = np.min(v, where=v > 0, initial=np.inf)
        return (v.min(), v.max(), minpos, tuple(sticky))
    lo, hi, pos = _bounds(v)
    return (np.float64(lo), np.float64(hi), np.float64(pos), tuple(sticky))


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
    """Coordinates as a flat float64 array (numpy or cupy in; masked
    values NaN, which breaks a line there, as matplotlib's masks do)."""
    if isinstance(a, np.ma.MaskedArray):
        return np.ma.filled(a.astype(np.float64), np.nan).ravel()
    return np.array(_as_host(a), np.float64).ravel()       # a copy: the caller's may change


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
        if not hasattr(self.src, 'kind'):               # axes / figure coordinates: not data
            return None
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
        x, y = self.x, self.y
        if x.size == y.size:                            # a point counts only with both finite
            ok = np.isfinite(x) & np.isfinite(y)
            if not ok.all():
                x, y = x[ok], y[ok]
        return _axis_bounds(x), _axis_bounds(y)


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
        n = self.norm_spec() if hasattr(self, 'norm_spec') else None
        if n is not None:                               # the artist's own norm: its limits
            if vmin is not None:                        # (matplotlib: the other one stays)
                n.vmin = float(vmin)
            if vmax is not None:
                n.vmax = float(vmax)
        else:
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


SEAM_EPS = 1e-7                 # degrees: a cut line's ends, inside the seam
SEAM_STEP = 0.5                 # degrees between points along the seam and the pole


@njit(cache=True, nogil=True)
def _rel(lon, c):
    """lon relative to c, in [-180, 180)."""
    return (lon - c + 180.0) % 360.0 - 180.0


@njit(cache=True, nogil=True)
def _on_seam(d):
    return abs(abs(d) - 180.0) < 1e-9


@njit(cache=True, nogil=True)
def _seam_split(lon, lat, offsets, c, fill, olon, olat, ooff, owhich, oside):
    """Polylines cut where they cross the meridian c + 180 (a step of more
    than 180 degrees in longitude relative to c), each side's end placed on
    the seam at the crossing's interpolated latitude. A point on the seam
    itself (Natural Earth's, cut at +-180) goes to the side its line comes
    from (or, at its start, goes to). fill=False: just the sizes (points,
    parts). oside[q] = (the side, +1 east or -1 west, part q starts on the
    seam at, the side it ends on there; 0 where it does not)."""
    npts = 0
    nq = 0
    for p in range(offsets.size - 1):
        a, b = offsets[p], offsets[p + 1]
        if fill:
            ooff[nq] = npts
            owhich[nq] = p
            oside[nq, 0] = 0.0
            oside[nq, 1] = 0.0
        prev, plat = np.nan, np.nan
        for k in range(a, b):
            d = _rel(lon[k], c)
            lo = lon[k]
            if _on_seam(d):
                side = 0.0
                if np.isfinite(prev):
                    side = 1.0 if prev > 0.0 else -1.0
                else:
                    for j in range(k + 1, b):
                        dj = _rel(lon[j], c)
                        if np.isfinite(dj) and not _on_seam(dj):
                            side = 1.0 if dj > 0.0 else -1.0
                            break
                if side != 0.0:
                    d = side * (180.0 - SEAM_EPS)
                    lo = c + d
            if np.isfinite(prev) and np.isfinite(d) and abs(d - prev) > 180.0:
                s = 1.0 if prev > 0.0 else -1.0
                t = (180.0 * s - prev) / (d + 360.0 * s - prev)
                la = plat + t * (lat[k] - plat)
                if fill:
                    olon[npts] = c + s * (180.0 - SEAM_EPS)
                    olat[npts] = la
                    oside[nq, 1] = s
                    ooff[nq + 1] = npts + 1
                    owhich[nq + 1] = p
                    oside[nq + 1, 0] = -s
                    oside[nq + 1, 1] = 0.0
                    olon[npts + 1] = c - s * (180.0 - SEAM_EPS)
                    olat[npts + 1] = la
                npts += 2
                nq += 1
            if fill:
                olon[npts] = lo
                olat[npts] = lat[k]
            npts += 1
            prev, plat = d, lat[k]
        nq += 1
    if fill:
        ooff[nq] = npts
    return npts, nq


def _seam_path(c, s0, lat0, s1, lat1, pole):
    """Points along the seam from (side s0, lat0) to (side s1, lat1), both
    ends left out: along one side, or, where the sides differ, down to the
    pole (its latitude), along it and up the other."""
    def side(s, a, b):
        t = np.linspace(0.0, 1.0, int(np.ceil(abs(b - a) / SEAM_STEP)) + 1)
        return np.full(t.size, c + s * (180.0 - SEAM_EPS)), a + t * (b - a)
    if s0 == s1:
        x, y = side(s0, lat0, lat1)
        return x[1:-1], y[1:-1]
    xa, ya = side(s0, lat0, pole)
    across = c + np.linspace(s0 * (180.0 - SEAM_EPS), s1 * (180.0 - SEAM_EPS),
                             int(np.ceil(360.0 / SEAM_STEP)) + 1)[1:-1]
    xb, yb = side(s1, pole, lat1)
    return (np.r_[xa[1:], across, xb[:-1]],
            np.r_[ya[1:], np.full(across.size, float(pole)), yb[:-1]])


def _split_seam(lon, lat, offsets, c, rings=False, lat_limits=(-90.0, 90.0)):
    """_seam_split's polylines, and (rings) each ring cut into pieces,
    closed again along the seam -- a straight chord would cut across the
    map's curved edge: on the side both its ends are on, or, a ring around
    a pole (its ends on either side), through that pole (lat_limits: the
    latitudes the map has an image of, the pole's taken where they stop)."""
    lon = np.ascontiguousarray(lon, np.float64)
    lat = np.ascontiguousarray(lat, np.float64)
    nparts = offsets.size - 1
    e1, e2, e3 = np.empty(0), np.empty(0, np.int64), np.empty((0, 2))
    npts, nq = _seam_split(lon, lat, offsets, float(c), False, e1, e1, e2, e2, e3)
    olon, olat = np.empty(npts), np.empty(npts)
    ooff, owhich, oside = np.empty(nq + 1, np.int64), np.empty(nq, np.int64), np.empty((nq, 2))
    _seam_split(lon, lat, offsets, float(c), True, olon, olat, ooff, owhich, oside)
    if nq == nparts or not rings:
        return olon, olat, ooff, owhich
    count = np.bincount(owhich, minlength=nparts)
    keep = np.nonzero(count[owhich] == 1)[0]
    sel = _gather(ooff, keep)
    xs, ys, lens, which = [olon[sel]], [olat[sel]], [np.diff(ooff)[keep]], [owhich[keep]]
    for p in np.nonzero(count > 1)[0]:
        q = np.nonzero(owhich == p)[0]
        pieces = [[olon[ooff[i]:ooff[i + 1]], olat[ooff[i]:ooff[i + 1]], oside[i, 0], oside[i, 1]]
                  for i in q]
        last, first = pieces[-1], pieces[0]             # the ring's start is no cut: rejoined
        pieces = ([[np.r_[last[0], first[0]], np.r_[last[1], first[1]], last[2], first[3]]]
                  + pieces[1:-1])
        pole = lat_limits[0] if np.nanmean(lat[offsets[p]:offsets[p + 1]]) < 0 else lat_limits[1]
        for x, y, s0, s1 in pieces:
            if s0 != 0 and s1 != 0 and x.size:
                sx, sy = _seam_path(c, s1, y[-1], s0, y[0], pole)
                x, y = np.r_[x, sx], np.r_[y, sy]
            xs.append(x)
            ys.append(y)
            lens.append([x.size])
            which.append([p])
    lens = np.concatenate(lens)
    return (np.concatenate(xs), np.concatenate(ys), np.r_[0, np.cumsum(lens)].astype(np.int64),
            np.concatenate(which).astype(np.int64))


def _project_lines(ax, x, y, offsets, src, parts=False, rings=False):
    """Polylines (in src) -> canvas pixels, seam-safe; parts=True also
    gives each output polyline's input polyline; rings: closed rings, to be
    filled (cut at a seam, each piece closed along it)."""
    P = ax.projection
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    offsets = np.asarray(offsets, np.int64)
    which = np.arange(offsets.size - 1)
    if P.seam:
        g = P.as_geodetic()
        lon, lat = _proj.transform_points(src, g, x, y)
        if rings and P.lat_limits != (-90.0, 90.0):     # every point of a ring placed: past
            lat = np.clip(lat, *P.lat_limits)           # where the map stops, on its edge
        lon, lat, offsets, which = _split_seam(lon, lat, offsets, P.central_longitude, rings,
                                               P.lat_limits)
        X, Y = _proj.transform_points(g, P, lon, lat)
    else:
        X, Y = _proj.transform_points(src, P, x, y)
    if P.cylindrical:
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
                return _project_lines(ax, p.xy[:, 0], p.xy[:, 1], p.ring_offsets, pc,
                                      rings=True)
            xy, off = cached_geometry(('feature-rings', name, scale, ax.view_key), rings)
            # nonzero, not even-odd: polygons of one feature overlap (Natural Earth's
            # Lake Huron and its Georgian Bay), and each is filled, as cartopy fills
            # them one by one; holes stay open, wound against their exterior
            r.c.polygons(xy, off, rgba(self.facecolor, self.alpha), ax.box_clip, 'nonzero')
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
        self.drawstyle = 'default'
        self.markevery = None
        self.dashes = None

    # matplotlib's Line2D setters and getters
    def set_data(self, *args):
        x, y = args[0] if len(args) == 1 else args
        self.set_xdata(x)
        self.set_ydata(y)

    def set_xdata(self, x):
        x = self.axes._xy(x, None)[0] if self.src == 'data' and self.axes is not None and \
            hasattr(self.axes, '_xy') else x
        self.x = _points(x)

    def set_ydata(self, y):
        y = self.axes._xy(None, y)[1] if self.src == 'data' and self.axes is not None and \
            hasattr(self.axes, '_xy') else y
        self.y = _points(y)

    def get_xdata(self, orig=True):
        return self.x.copy()

    def get_ydata(self, orig=True):
        return self.y.copy()

    def get_data(self, orig=True):
        return self.get_xdata(), self.get_ydata()

    def get_xydata(self):
        return np.column_stack([self.x, self.y])

    def set_color(self, c):
        if self.mfc is self.color or (isinstance(self.mfc, str) and self.mfc == self.color):
            self.mfc = c
        if self.mec is self.color or (isinstance(self.mec, str) and self.mec == self.color):
            self.mec = c
        self.color = c

    def get_color(self):
        return self.color

    def set_linewidth(self, w):
        self.linewidth = float(w)

    def get_linewidth(self):
        return self.linewidth

    def set_linestyle(self, ls):
        self.linestyle = ls

    def get_linestyle(self):
        return self.linestyle

    def set_marker(self, m):
        self.marker = m

    def get_marker(self):
        return self.marker

    def set_markersize(self, s):
        self.markersize = float(s)

    def get_markersize(self):
        return self.markersize

    def set_markerfacecolor(self, c):
        self.mfc = c

    def get_markerfacecolor(self):
        return self.mfc

    def set_markeredgecolor(self, c):
        self.mec = c

    def get_markeredgecolor(self):
        return self.mec

    def set_markeredgewidth(self, w):
        self.mew = float(w)

    def set_fillstyle(self, s):
        self.fillstyle = s

    def set_drawstyle(self, s):
        self.drawstyle = s or 'default'

    def set_markevery(self, m):
        self.markevery = m

    def set_dashes(self, seq):
        self.linestyle = (0, tuple(seq)) if seq is not None and len(seq) else '-'

    def set_label(self, s):
        self.label = s

    def get_label(self):
        return self.label

    def get_alpha(self):
        return self.alpha

    def set_solid_capstyle(self, s):
        pass

    set_dash_capstyle = set_solid_joinstyle = set_dash_joinstyle = set_solid_capstyle
    set_antialiased = set_clip_on = set_rasterized = set_solid_capstyle

    def set_transform(self, t):
        self.src = self.axes._src(t) if self.axes is not None else t

    def _stepped(self):
        """drawstyle steps: the line's vertices as matplotlib's STEP_LOOKUP_MAP."""
        ds = self.drawstyle
        if ds in (None, 'default') or self.x.size < 2:
            return self.x, self.y
        from ._plots import _steps_mid, _steps_post, _steps_pre
        fn = {'steps': _steps_pre, 'steps-pre': _steps_pre, 'steps-mid': _steps_mid,
              'steps-post': _steps_post}.get(ds)
        return (self.x, self.y) if fn is None else fn(self.x, self.y)

    def _marker_index(self, n):
        """markevery: which points get markers."""
        m = self.markevery
        if m is None or m is False:
            return slice(None)
        if isinstance(m, (int, np.integer)) and not isinstance(m, bool):
            return slice(None, None, int(m))
        if isinstance(m, tuple) and len(m) == 2 and all(isinstance(v, (int, np.integer))
                                                       for v in m):
            return slice(int(m[0]), None, int(m[1]))
        if isinstance(m, slice):
            return m
        a = np.asarray(m)
        if a.dtype == bool:
            return np.nonzero(a)[0]
        if a.dtype.kind in 'iu':
            return a
        return slice(None)

    def legend_style(self):
        return dict(kind='line', color=rgba(self.color, self.alpha), linewidth=self.linewidth,
                    linestyle=self.linestyle, marker=self.marker, markersize=self.markersize,
                    mfc=rgba(self.mfc, self.alpha), mec=rgba(self.mec, self.alpha),
                    mew=self.mew, fillstyle=self.fillstyle, mfcalt=rgba(self.mfcalt, self.alpha))

    def render(self, r, ax):
        xs, ys = self._stepped()
        if xs is not self.x:                            # drawstyle steps: the line stepped,
            sxy, soff = ax._lines_px(xs, ys, np.array([0, xs.size]), self.src)  # markers not
            xy, _ = ax._lines_px(self.x, self.y, np.array([0, self.x.size]), self.src)
        else:
            xy, off = ax._lines_px(self.x, self.y, np.array([0, self.x.size]), self.src)
            sxy, soff = xy, off
        self._render(r, ax, xy, sxy, soff)

    def _render(self, r, ax, pts, xy, off):
        if self.linestyle not in ('None', 'none', '', ' ') and self.linewidth > 0:
            d = _dashes(self.linestyle, self.linewidth)
            cap = (_draw.BUTT if d is not None else _draw.PROJECTING) if ax._agg else \
                _draw.LEGACY
            r.lines(xy, off, rgba(self.color, self.alpha), r.px(self.linewidth), ax.box_clip,
                    None if d is None else tuple(r.px(v) for v in d), ax._snap, cap, ax._agg)
        if self.marker not in (None, 'None', 'none', ''):
            _markers(r, pts[:self.x.size][self._marker_index(self.x.size)], self.marker,
                     r.px(self.markersize),
                     rgba(self.mfc, self.alpha), rgba(self.mec, self.alpha), r.px(self.mew),
                     ax.box_clip, snap=ax._snap, fillstyle=self.fillstyle,
                     fc_alt=rgba(self.mfcalt, self.alpha))


def _marker_paths(marker, size, ew, fillstyle, agg, centred=True):
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
    elif agg and centred and isinstance(marker, str):
        # Agg's draw_markers (lines' markers, not scatter's path collection)
        # places the others half a pixel on, to pixel centres, as it does a
        # stroke of odd width (circles and hexagons: always).
        sv = 0.5 if marker in _CENTRED else float(_draw.snap_value(geom.edge_width(ew)))
        if sv:
            flats = [None if f is None else (f[0] + sv, f[1]) for f in flats]
    return geom, flats, cap


_CENTRED = ('o', '.', 'h', 'H')


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
    (markersize, or sqrt(s) for scatter, in px; or an array, one per
    marker), faces fc -- or colors, one per marker -- edges ec (None: each
    its face colour; or one per marker) ew px wide.
    snap (not False) draws as Agg: one colour (draw_markers), each marker
    placed at whole pixels; matplotlib's caps. scatter: an unfilled
    marker's face is not filled, its edge takes the face colour."""
    agg = snap is not False
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if isinstance(marker, str) and len(marker) > 2 and marker[0] == marker[-1] == '$':
        _text_markers(r, pts, marker, size, fc if colors is None else colors, ec, ew, clip)
        return
    sizes = None
    if np.ndim(size):
        sizes = np.broadcast_to(np.asarray(size, np.float64), (pts.shape[0],))
        if sizes.size and (sizes == sizes[0]).all():
            size, sizes = float(sizes[0]), None
    ok = np.isfinite(pts).all(axis=1)
    if sizes is not None:
        ok &= np.isfinite(sizes) & (sizes > 0)
    if not ok.all():
        pts = pts[ok]
        colors = None if colors is None else colors[ok]
        ec = ec[ok] if ec is not None and np.ndim(ec) == 2 else ec
        sizes = None if sizes is None else sizes[ok]
    if sizes is not None:
        _sized_markers(r, pts, marker, sizes, fc, ec, ew, clip, colors, agg, fillstyle, fc_alt,
                       scatter)
        return
    if np.ndim(ec) == 2 and not agg:                    # (maps) one edge colour
        ec = ec[0] if len(ec) else None
    geom, (fill, alt, stroke), cap = _marker_paths(marker, size, ew, fillstyle, agg,
                                                   centred=not scatter)
    if not geom:
        return
    if agg and colors is None and np.ndim(ec) < 2:
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
        if edge_col is not None and np.ndim(edge_col) == 1 and not edge_col[3]:
            edge_col = None
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


def _text_markers(r, pts, marker, size, fc, ec, ew, clip):
    """'$...$' markers: the mathtext's glyphs, its ink box's larger side
    `size` px and centred on each point (matplotlib's TextPath marker,
    normalised so), filled in the face colour and grown by half the edge
    width in the edge's (its outline's stroke)."""
    from . import _text
    ref = 100.0
    tm = _text.mask(marker, ref)
    a = tm.alpha
    if not a.size or not (a > 0).any():
        return
    rows, cols = np.nonzero(a > 0)
    dim = max(rows.max() - rows.min() + 1, cols.max() - cols.min() + 1)
    sizes = np.broadcast_to(np.asarray(size, float), (pts.shape[0],))
    cache = {}
    for k, (x, y) in enumerate(pts):
        if not (np.isfinite(x) and np.isfinite(y)) or sizes[k] <= 0:
            continue
        if clip is not None and not (clip[0] <= x <= clip[2] and clip[1] <= y <= clip[3]):
            continue                                    # (outside the axes: not drawn)
        key = round(float(sizes[k]) * 8)
        face = fc[k] if np.ndim(fc) == 2 else fc
        edge = (ec[k] if np.ndim(ec) == 2 else ec) if ec is not None else face
        stroked = edge is not None and edge[3] and ew > 0
        if (key, stroked) not in cache:
            m = _text.mask(marker, ref * (key / 8.0) / dim)
            al = m.alpha
            if stroked:                                 # the outline's stroke: half outside
                al, _pad = _text.grow(al, ew / 2.0)
            rr, cc = np.nonzero(al > 0)
            cy = 0.5 * (rr.min() + rr.max() + 1) if rr.size else 0.0
            cx = 0.5 * (cc.min() + cc.max() + 1) if cc.size else 0.0
            cache[key, stroked] = (np.ascontiguousarray(al), cx, cy)
        al, cx, cy = cache[key, stroked]
        col = edge if stroked else face
        if col is not None and col[3]:
            r.c.mask(al, int(np.floor(x - cx + 0.5)), int(np.floor(y - cy + 0.5)), col)


def _sized_markers(r, pts, marker, sizes, fc, ec, ew, clip, colors, agg, fillstyle, fc_alt,
                   scatter):
    """_markers of several sizes (scatter's s per point), to 1/8 px: on
    plain axes stamped in one pass, in order, as Agg draws a path
    collection; on maps a size at a time."""
    q = np.rint(sizes * 8.0).astype(np.int64)
    uq, which = np.unique(q, return_inverse=True)
    if not agg or fc_alt is not None and fillstyle not in ('full', 'none'):
        for k, v in enumerate(uq.tolist()):
            sel = which == k
            _markers(r, pts[sel], marker, v / 8.0, fc, ec[sel] if np.ndim(ec) == 2 else ec, ew,
                     clip, None if colors is None else colors[sel], agg, fillstyle, fc_alt,
                     scatter)
        return
    shapes = [_marker_paths(marker, v / 8.0, ew, fillstyle, agg, centred=not scatter)
              for v in uq.tolist()]
    geom, cap = shapes[0][0], shapes[0][2]
    if not geom:
        return
    fills = [s[1][0] for s in shapes]
    strokes = [s[1][2] for s in shapes]
    if scatter and not geom.filled:
        fills = [None] * len(shapes)
        ec = None
    face = fills[0] is not None and fillstyle != 'none'
    w = geom.edge_width(ew)
    own = colors if colors is not None else fc
    face_col = own if face and (colors is not None or fc[3]) else None
    edge_col = None if strokes[0] is None or w <= 0 else (own if ec is None else ec)
    if edge_col is not None and np.ndim(edge_col) == 1 and not edge_col[3]:
        edge_col = None
    if face_col is None and edge_col is None:
        return
    r.c.stamp(pts, fills if face_col is not None else [None] * len(shapes),
              strokes if edge_col is not None else [None] * len(shapes), w, face_col, edge_col,
              clip, cap, which)


class Scatter(_Mappable, _PointsArtist):
    """ax.scatter: markers, one colour or coloured by values."""

    zorder = 1.0

    def __init__(self, ax, x, y, src, s=20.0, c=None, color=None, marker='o', cmap=None,
                 norm=None, vmin=None, vmax=None, edgecolors='face', linewidths=1.0,
                 alpha=None, zorder=None, plotnonfinite=False):
        super().__init__(ax, zorder, alpha)
        self.x, self.y, self.src = _points(x), _points(y), src
        n = self.x.size
        keep = np.ones(n, bool)
        for v in (x, y):                                # masked points are not drawn
            m = np.ma.getmask(v)
            if m is not np.ma.nomask:
                keep &= ~np.ravel(m)
        self.sizes = None
        if np.ndim(s) and np.size(s) > 1:
            self.sizes = np.ma.filled(np.ma.asarray(s, np.float64).ravel(), np.nan)
            if self.sizes.size != n:
                raise ValueError('s must be a scalar, or float array-like with the same size '
                                 'as x and y')
            fin = np.isfinite(self.sizes)
            self.s = float(self.sizes[fin].mean()) if fin.any() else 0.0
        else:
            self.s = float(np.ravel(s)[0]) if np.ndim(s) else float(s)
        self.marker = marker
        self.cmap = _color.get_cmap(cmap)
        self.values = None
        self._rgba_list = None                          # one colour per point
        self.color = color if color is not None else 'C0'
        if c is not None:
            ca = None if isinstance(c, str) else np.ma.asarray(c)
            rgb_rows = ca is not None and ca.ndim == 2 and ca.shape[1] in (3, 4) and \
                ca.shape[0] == n and ca.size != n
            if ca is not None and ca.dtype.kind in 'fiub' and ca.size == n and not rgb_rows:
                vals = np.ma.filled(ca.astype(np.float64).ravel(), np.nan)
                if not plotnonfinite:
                    keep &= np.isfinite(vals)
                self.values = vals.astype(np.float32)
            elif ca is None or (ca.ndim == 1 and ca.dtype.kind in 'fiu' and ca.size in (3, 4)):
                self.color = c
            else:
                cols = _color.to_rgba_array(c)
                if len(cols) == 1:
                    self.color = tuple(cols[0])
                elif len(cols) != n:
                    raise ValueError("'c' argument has %d elements, which is inconsistent with "
                                     "'x' and 'y' with size %d." % (len(cols), n))
                else:
                    self._rgba_list = cols
                    self.color = tuple(cols[0])
        elif color is not None and not isinstance(color, str) and np.ndim(color) == 2 or \
                isinstance(color, (list, tuple)) and len(color) == n > 1 and \
                all(isinstance(v, str) for v in color):
            cols = _color.to_rgba_array(color)
            if len(cols) == n:
                self._rgba_list = cols
                self.color = tuple(cols[0])
        self.keep = None if keep.all() else keep
        self._norm = None
        self._norm_in, self._vmin, self._vmax = norm, vmin, vmax
        self.edgecolors = edgecolors
        self.linewidths = float(np.mean(linewidths)) if np.ndim(linewidths) else linewidths

    def get_offsets(self):
        return np.column_stack([self.x, self.y])

    def set_offsets(self, xy):
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        self.x, self.y = xy[:, 0].copy(), xy[:, 1].copy()

    def get_sizes(self):
        return np.array([self.s]) if self.sizes is None else self.sizes

    def set_sizes(self, s):
        if np.ndim(s) and np.size(s) > 1:
            self.sizes = np.asarray(s, np.float64).ravel()
            self.s = float(np.nanmean(self.sizes))
        else:
            self.sizes, self.s = None, float(np.ravel(s)[0]) if np.ndim(s) else float(s)

    def set_array(self, a):
        self.values = np.asarray(a, np.float32).ravel()
        self._norm = None

    def get_array(self):
        return self.values

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
            if self._rgba_list is None:
                return None
            return np.array([rgba(tuple(c), self.alpha) for c in self._rgba_list], np.uint8)
        out = np.empty((1, self.values.size, 4), np.uint8)
        _color.shade(self.values[None, :], self.cmap, self.norm_spec(), out=out,
                     alpha=self.alpha, rounding='nearest')
        return out[0]

    def _edge(self, fc):
        """(edge colour -- one, or one per point -- edge width in pt)."""
        e = self.edgecolors
        face = isinstance(e, str) and e == 'face'
        none = isinstance(e, str) and e.lower() == 'none'
        if face or none or e is None:
            return fc, (0.0 if none else self.linewidths)
        if not isinstance(e, str) and (np.ndim(e) == 2 or isinstance(e, (list, tuple)) and
                                       len(e) > 1 and all(isinstance(v, str) for v in e)):
            cols = _color.to_rgba_array(e)
            if len(cols) == self.x.size and len(cols) > 1:
                return np.array([rgba(tuple(c), self.alpha) for c in cols], np.uint8), \
                    self.linewidths
            e = tuple(cols[0])
        return rgba(e, self.alpha), self.linewidths

    def legend_style(self):
        colors = self._colors()
        fc = colors[0] if colors is not None and len(colors) else rgba(self.color, self.alpha)
        ec, ew = self._edge(fc)
        if np.ndim(ec) == 2:
            ec = ec[0]
        s = self.s if self.sizes is None or not self.sizes.size else float(self.sizes[0])
        return dict(kind='scatter', marker=self.marker, s=s, facecolor=fc, edgecolor=ec,
                    ew=ew)

    def render(self, r, ax):
        pts = np.column_stack(self._screen(ax))
        if self.sizes is not None:                      # s is the marker's area in pt^2
            size = np.sqrt(np.abs(self.sizes)) * r.dpi / 72.0
        else:
            size = np.sqrt(self.s) * r.dpi / 72.0
        fc = rgba(self.color, self.alpha)
        colors = self._colors()
        alphas = getattr(self, 'alphas', None)
        if alphas is not None and alphas.size:          # an alpha per point
            n = self.x.size
            if colors is None:
                colors = np.repeat(fc[None], n, axis=0)
            colors = colors.copy()
            colors[:, 3] = np.rint(np.clip(np.resize(alphas, n), 0, 1) * 255).astype(np.uint8)
        ec, ew = self._edge(None if colors is not None else fc)
        if alphas is not None and alphas.size and np.ndim(ec) == 1 and ec is not None:
            ec = np.repeat(ec[None], self.x.size, axis=0)
            ec[:, 3] = colors[:, 3]
        if self.keep is not None:
            pts = pts[self.keep]
            colors = None if colors is None else colors[self.keep]
            ec = ec[self.keep] if np.ndim(ec) == 2 else ec
            size = size[self.keep] if np.ndim(size) else size
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
        self.clip_axes = None                           # (x0, y0, x1, y1) axes fractions, or the box

    def _clip(self, ax):
        if self.clip_axes is None:
            return ax.box_clip
        a0, b0, a1, b1 = self.clip_axes
        x0, y0, x1, y1 = ax.box_f
        w, h = x1 - x0, y1 - y0
        return tuple(int(np.floor(v + 0.5)) for v in (x0 + a0 * w, y1 - b1 * h, x0 + a1 * w,
                                                       y1 - b0 * h))

    def render(self, r, ax):
        x = self.x
        px, py, dx, dy, mag = ax._vectors_px(self.x, self.y, self.u, self.v, self.src)
        ok = np.isfinite(px) & np.isfinite(py) & np.isfinite(dx) & np.isfinite(dy) & \
            np.isfinite(mag)
        b = clip = self._clip(ax)
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
            r.c.polygons(xy[_gather(offs, full)], np.r_[0, np.cumsum(lens[full])], fc, clip,
                         'nonzero')
        if self.fill_empty and (~empty).sum() != empty.size and fc[3]:
            emp = np.nonzero(empty)[0]
            r.c.polygons(xy[_gather(offs, emp)], np.r_[0, np.cumsum(lens[emp])], fc, clip)
        if ec[3] and self.linewidth > 0:                # outlines: every ring closed
            roff = np.r_[0, np.cumsum(lens + 1)]
            ring = np.empty(roff[-1], np.int64)
            last = roff[1:] - 1
            body = np.ones(ring.size, bool)
            body[last] = False
            ring[body] = np.arange(offs[-1])
            ring[last] = offs[:-1]
            r.lines(xy[ring], roff, ec, r.px(self.linewidth), clip)


class Text(Artist):
    """ax.text / fig.text."""

    zorder = 3.0

    def __init__(self, owner, x, y, s, coords, fontsize=None, color='black', ha='left',
                 va='baseline', rotation=0.0, fontweight='normal', fontstyle='normal',
                 bbox=None, alpha=None, zorder=None, rotation_mode='default', clip_on=False,
                 family=None, path_effects=None, multialignment=None, linespacing=None,
                 visible=True, label=None):
        super().__init__(owner, zorder, alpha, visible)
        self.x, self.y, self.s = x, y, s
        self.ma, self.linespacing, self.label = multialignment, linespacing, label
        self.path_effects = list(path_effects) if path_effects else None
        self.coords = coords                            # 'data' crs, 'axes', 'figure'
        self.fontsize, self.color = fontsize, color
        self.ha, self.va, self.rotation = ha, va, rotation
        self.weight, self.style = fontweight, fontstyle
        self.bbox, self.rotation_mode, self.clip_on = bbox, rotation_mode, clip_on
        self.family = family

    def set_text(self, s):
        self.s = s

    def get_text(self):
        return self.s

    def get_position(self):
        return (self.x, self.y)

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y

    def set_color(self, c):
        self.color = c

    def get_color(self):
        return self.color

    def set_fontsize(self, s):
        self.fontsize = s

    set_size = set_fontsize

    def get_fontsize(self):
        return fontsize_pt(self.fontsize)

    def set_fontweight(self, w):
        self.weight = w

    set_weight = set_fontweight

    def set_fontstyle(self, s):
        self.style = s

    set_style = set_fontstyle

    def set_family(self, f):
        self.family = f

    set_fontfamily = set_family

    def set_ha(self, ha):
        self.ha = ha

    set_horizontalalignment = set_ha

    def set_va(self, va):
        self.va = va

    set_verticalalignment = set_va

    def set_ma(self, ma):
        self.ma = ma

    set_multialignment = set_ma

    def set_rotation(self, r):
        self.rotation = _rotation_value(r)

    def get_rotation(self):
        return self.rotation

    def set_rotation_mode(self, m):
        self.rotation_mode = m

    def set_bbox(self, b):
        self.bbox = b

    def set_backgroundcolor(self, c):
        self.bbox = dict(self.bbox or {}, facecolor=c, edgecolor='none')

    def set_linespacing(self, s):
        self.linespacing = s

    def set_clip_on(self, b):
        self.clip_on = bool(b)

    def set_label(self, s):
        self.label = s

    def set_path_effects(self, effects):
        self.path_effects = list(effects) if effects else None

    def set_position(self, xy):
        self.x, self.y = xy

    def _xy(self, r, ax):
        if self.coords == 'figure':
            return self.x * r.W + r.ox, (1 - self.y) * r.H + r.oy
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
               family=self.family, effects=self.path_effects, ma=self.ma,
               linespacing=self.linespacing)


def _rotation_value(r):
    if r in ('horizontal', None):
        return 0.0
    if r == 'vertical':
        return 90.0
    return float(r)


def _text_kw(kw):
    """matplotlib's Text keyword aliases."""
    out = {}
    fp = kw.pop('fontproperties', kw.pop('font_properties', kw.pop('font', None)))
    if fp is not None:
        for k, name in (('size', 'fontsize'), ('weight', 'fontweight'), ('style', 'fontstyle'),
                        ('family', 'family')):
            v = fp.get(k) if isinstance(fp, dict) else getattr(fp, 'get_' + k, lambda: None)()
            if v is not None:
                kw.setdefault(name, v)
    for k, v in (kw.pop('fontdict', None) or {}).items():
        kw.setdefault(k, v)
    out['fontsize'] = kw.pop('fontsize', kw.pop('size', None))
    out['color'] = kw.pop('color', kw.pop('c', None))
    if out['color'] is None:                            # matplotlib's: rcParams text.color
        out['color'] = rc('text.color')
    out['ha'] = kw.pop('ha', kw.pop('horizontalalignment', 'left'))
    out['va'] = kw.pop('va', kw.pop('verticalalignment', 'baseline'))
    out['rotation'] = _rotation_value(kw.pop('rotation', 0.0))
    out['multialignment'] = kw.pop('multialignment', kw.pop('ma', None))
    out['linespacing'] = kw.pop('linespacing', None)
    out['visible'] = kw.pop('visible', True)
    if 'backgroundcolor' in kw:
        bc = kw.pop('backgroundcolor')
        if kw.get('bbox') is None:
            kw['bbox'] = dict(facecolor=bc, edgecolor='none')
    out['fontweight'] = kw.pop('fontweight', kw.pop('weight', 'normal'))
    out['fontstyle'] = kw.pop('fontstyle', kw.pop('style', 'normal'))
    out['bbox'] = kw.pop('bbox', None)
    out['alpha'] = kw.pop('alpha', None)
    out['zorder'] = kw.pop('zorder', None)
    out['rotation_mode'] = kw.pop('rotation_mode', 'default')
    out['clip_on'] = kw.pop('clip_on', False)
    out['family'] = kw.pop('family', kw.pop('fontfamily', None))
    out['path_effects'] = kw.pop('path_effects', None)
    out['label'] = kw.pop('label', None)
    kw.pop('transform', None)
    for k in ('fontname', 'wrap', 'zorder', 'usetex', 'gid', 'url', 'picker', 'snap',
              'rasterized', 'parse_math', 'fontstretch', 'stretch', 'fontvariant', 'variant',
              'animated', 'in_layout', 'math_fontfamily', 'antialiased', 'clip_box'):
        kw.pop(k, None)
    if kw:
        raise TypeError('unsupported text keywords: %s' % ', '.join(kw))
    return out
