"""Unstructured triangular grids: matplotlib's tricontourf, tricontour,
tripcolor and triplot.

    t = fastplot.tri.Triangulation(lon, lat)            # Delaunay, or triangles=
    ax.tricontourf(t, temp, levels=12, cmap='turbo')
    ax.tricontour(lon, lat, temp, levels=6, colors='k')

The triangulation is matplotlib's: Qhull's Delaunay (through scipy, called
as matplotlib calls it -- points shifted by their sequentially summed mean,
options Qt Qbb Qz -- so even a regular grid's ties come out the same), or,
without scipy, fastplot's own Bowyer-Watson (the same triangles wherever
the Delaunay triangulation is unique). Fields are rasterized per pixel:
each pixel centre inside a triangle (a top-left rule on shared edges)
takes the linear interpolation of the corner values (tricontourf,
gouraud tripcolor) or the triangle's value (flat tripcolor), so
tricontourf's bands are matplotlib's exactly where its polygons are.
Contour lines are marching triangles on the same linear surfaces, joined
into polylines through shared edges.
"""
import numpy as np

from ._jit import njit, prange

_F0 = 0.0


# ------------------------------------------------------------ triangulation

class Triangulation:
    """matplotlib.tri.Triangulation: points and triangles (Delaunay when
    not given), an optional mask of triangles."""

    def __init__(self, x, y, triangles=None, mask=None):
        self.x = np.asarray(x, np.float64).ravel()
        self.y = np.asarray(y, np.float64).ravel()
        if self.x.shape != self.y.shape:
            raise ValueError('x and y must be equal-length 1D arrays')
        if triangles is None:
            self.triangles = delaunay(self.x, self.y)
            self.is_delaunay = True
        else:
            t = np.asarray(triangles, np.int32)
            if t.ndim != 2 or t.shape[1] != 3:
                raise ValueError('triangles must be a (n, 3) array')
            if t.size and (t.max() >= self.x.size or t.min() < 0):
                raise ValueError('triangles are indices into the points')
            self.triangles = t
            self.is_delaunay = False
        self.mask = None
        self.set_mask(mask)

    def set_mask(self, mask):
        self.mask = None if mask is None else np.asarray(mask, bool).ravel()
        if self.mask is not None and self.mask.size != len(self.triangles):
            raise ValueError('mask must have one value per triangle')

    def get_masked_triangles(self):
        return self.triangles if self.mask is None else self.triangles[~self.mask]

    @property
    def edges(self):
        """The unmasked triangles' edges, each once: (n, 2)."""
        t = self.get_masked_triangles()
        e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
        return np.unique(e, axis=0)

    @classmethod
    def get_from_args_and_kwargs(cls, *args, **kwargs):
        """matplotlib's: (Triangulation, the remaining args, kwargs)."""
        if args and isinstance(args[0], Triangulation):
            return args[0], args[1:], kwargs
        if 'triangulation' in kwargs:
            return kwargs.pop('triangulation'), args, kwargs
        x, y, args = args[0], args[1], args[2:]
        triangles = kwargs.pop('triangles', None)
        mask = kwargs.pop('mask', None)
        if triangles is None and args and np.ndim(args[0]) == 2 and np.shape(args[0])[-1] == 3 \
                and np.issubdtype(np.asarray(args[0]).dtype, np.integer):
            triangles, args = args[0], args[1:]
        return cls(x, y, triangles, mask), args, kwargs

    def get_trifinder(self):
        """matplotlib's: the triangulation's TrapezoidMapTriFinder, made once
        and shared (it follows later set_mask calls)."""
        if getattr(self, '_trifinder', None) is None:
            from ._triinterp import TrapezoidMapTriFinder
            self._trifinder = TrapezoidMapTriFinder(self)
        return self._trifinder

    def calculate_plane_coefficients(self, z):
        """matplotlib's: (ntri, 3) planes z = a x + b y + c through the
        triangles' corner values (zero for masked triangles)."""
        from ._triinterp import plane_coefficients
        return plane_coefficients(self, z)


def delaunay(x, y):
    """matplotlib's Delaunay triangles of the points (int32 (n, 3))."""
    n = x.size
    if n < 3:
        raise ValueError('x and y must hold at least 3 points')
    xm = np.cumsum(x)[-1] / n                           # matplotlib's _qhull: a running sum
    ym = np.cumsum(y)[-1] / n
    try:
        from scipy.spatial import Delaunay
    except ImportError:
        return _bowyer_watson(x - xm, y - ym)
    return Delaunay(np.column_stack([x - xm, y - ym]),
                    qhull_options='Qt Qbb Qz').simplices.astype(np.int32)


@njit(cache=True, nogil=True)
def _incircle(ax, ay, bx, by, cx, cy, px, py):
    """> 0 when p is inside the circumcircle of the anticlockwise abc."""
    adx, ady = ax - px, ay - py
    bdx, bdy = bx - px, by - py
    cdx, cdy = cx - px, cy - py
    return ((adx * adx + ady * ady) * (bdx * cdy - cdx * bdy) -
            (bdx * bdx + bdy * bdy) * (adx * cdy - cdx * ady) +
            (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady))


@njit(cache=True, nogil=True)
def _bw(px, py, order):
    """Bowyer-Watson over the points in `order` (a super triangle's three
    extra points last): the triangles (anticlockwise), alive flags."""
    n = px.size - 3
    cap = 2 * n + 16
    tri = np.empty((cap, 3), np.int64)
    alive = np.zeros(cap, np.bool_)
    nt = 1
    tri[0] = n, n + 1, n + 2
    alive[0] = True
    bad = np.empty(cap, np.int64)
    edges = np.empty((3 * cap, 2), np.int64)
    for p in order:
        x, y = px[p], py[p]
        nb = 0
        for t in range(nt):
            if alive[t]:
                a, b, c = tri[t]
                if _incircle(px[a], py[a], px[b], py[b], px[c], py[c], x, y) > 0:
                    bad[nb] = t
                    nb += 1
        ne = 0
        for k in range(nb):                             # the cavity's boundary edges
            t = bad[k]
            for e in range(3):
                u, v = tri[t, e], tri[t, (e + 1) % 3]
                shared = False
                for m in range(nb):
                    if m != k:
                        s = bad[m]
                        for f in range(3):
                            if tri[s, f] == v and tri[s, (f + 1) % 3] == u:
                                shared = True
                if not shared:
                    edges[ne, 0], edges[ne, 1] = u, v
                    ne += 1
            alive[t] = False
        for k in range(ne):
            slot = bad[k] if k < nb else nt
            if k >= nb:
                if nt >= cap:
                    grow = np.empty((2 * cap, 3), np.int64)
                    grow[:nt] = tri[:nt]
                    tri = grow
                    ga = np.zeros(2 * cap, np.bool_)
                    ga[:nt] = alive[:nt]
                    alive = ga
                    bad = np.empty(2 * cap, np.int64)
                    edges2 = np.empty((6 * cap, 2), np.int64)
                    edges2[:ne] = edges[:ne]
                    edges = edges2
                    cap *= 2
                nt += 1
            tri[slot, 0], tri[slot, 1], tri[slot, 2] = edges[k, 0], edges[k, 1], p
            alive[slot] = True
    return tri[:nt], alive[:nt]


def _bowyer_watson(x, y):
    n = x.size
    span = max(np.ptp(x), np.ptp(y), 1e-300) * 1e3
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    px = np.r_[x, cx - 2 * span, cx + 2 * span, cx]
    py = np.r_[y, cy - span, cy - span, cy + 2 * span]
    order = np.lexsort((y, x)).astype(np.int64)
    tri, alive = _bw(px, py, order)
    tri = tri[alive & (tri < n).all(axis=1)]
    return tri.astype(np.int32)


# --------------------------------------------------------------- raster

@njit(cache=True, nogil=True, parallel=True)
def raster_triangles(values, sx, sy, tri, zt, flat):
    """Each triangle (corners sx, sy: raster px) into values: at every
    pixel centre inside it (top-left rule on its edges) the linear
    interpolation of its corners' zt (flat: zt[t, 0])."""
    H, W = values.shape
    for t in prange(tri.shape[0]):
        a, b, c = tri[t, 0], tri[t, 1], tri[t, 2]
        x0, y0, x1, y1, x2, y2 = sx[a], sy[a], sx[b], sy[b], sx[c], sy[c]
        z0, z1, z2 = zt[t, 0], zt[t, 1], zt[t, 2]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if not (area != 0.0) or not np.isfinite(area):
            continue
        if area < 0.0:                                  # one orientation for the edge rule
            x1, y1, x2, y2 = x2, y2, x1, y1
            z1, z2 = z2, z1
            area = -area
        i0 = max(0, int(np.floor(min(y0, y1, y2))))
        i1 = min(H - 1, int(np.ceil(max(y0, y1, y2))))
        j0 = max(0, int(np.floor(min(x0, x1, x2))))
        j1 = min(W - 1, int(np.ceil(max(x0, x1, x2))))
        # edges (from -> to) and their top-left flags (y down, area > 0)
        for i in range(i0, i1 + 1):
            py = i + 0.5
            for j in range(j0, j1 + 1):
                px = j + 0.5
                w0 = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
                w1 = (x0 - x2) * (py - y2) - (y0 - y2) * (px - x2)
                w2 = (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0)
                if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                    continue
                if w0 == 0.0 and not _top_left(x1, y1, x2, y2):
                    continue
                if w1 == 0.0 and not _top_left(x2, y2, x0, y0):
                    continue
                if w2 == 0.0 and not _top_left(x0, y0, x1, y1):
                    continue
                if flat:
                    values[i, j] = z0
                else:
                    values[i, j] = (w0 * z0 + w1 * z1 + w2 * z2) / area


@njit(cache=True, nogil=True, parallel=True)
def raster_triangles_rgba(layer, sx, sy, tri, cols):
    """raster_triangles in colour: each triangle's corners' RGBA (cols,
    per point, 0..255) interpolated linearly to every pixel centre inside
    it, rounded -- Agg's gouraud triangles (draw_gouraud_triangles)."""
    H, W = layer.shape[0], layer.shape[1]
    for t in prange(tri.shape[0]):
        a, b, c = tri[t, 0], tri[t, 1], tri[t, 2]
        x0, y0, x1, y1, x2, y2 = sx[a], sy[a], sx[b], sy[b], sx[c], sy[c]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if not (area != 0.0) or not np.isfinite(area):
            continue
        if area < 0.0:
            x1, y1, x2, y2 = x2, y2, x1, y1
            b, c = c, b
            area = -area
        i0 = max(0, int(np.floor(min(y0, y1, y2))))
        i1 = min(H - 1, int(np.ceil(max(y0, y1, y2))))
        j0 = max(0, int(np.floor(min(x0, x1, x2))))
        j1 = min(W - 1, int(np.ceil(max(x0, x1, x2))))
        for i in range(i0, i1 + 1):
            py = i + 0.5
            for j in range(j0, j1 + 1):
                px = j + 0.5
                w0 = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
                w1 = (x0 - x2) * (py - y2) - (y0 - y2) * (px - x2)
                w2 = (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0)
                if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                    continue
                if w0 == 0.0 and not _top_left(x1, y1, x2, y2):
                    continue
                if w1 == 0.0 and not _top_left(x2, y2, x0, y0):
                    continue
                if w2 == 0.0 and not _top_left(x0, y0, x1, y1):
                    continue
                for k in range(4):
                    v = (w0 * cols[a, k] + w1 * cols[b, k] + w2 * cols[c, k]) / area
                    layer[i, j, k] = np.uint8(min(255.0, max(0.0, np.floor(v + 0.5))))


@njit(cache=True, nogil=True)
def _top_left(ax, ay, bx, by):
    """Is the edge a -> b a top or left edge (y down, positive area)?"""
    return (ay == by and bx < ax) or (by < ay)


# ---------------------------------------------------------- contour lines

@njit(cache=True, nogil=True)
def _crossings(x, y, z, tri, level, out, keys):
    """Marching triangles at one level: segments (x0, y0, x1, y1) into out,
    the edges they end on (as point-pair keys) into keys; returns how many."""
    n = 0
    npts = x.size
    for t in range(tri.shape[0]):
        v = tri[t]
        above = 0
        for k in range(3):
            if z[v[k]] > level:
                above |= 1 << k
        if above == 0 or above == 7:
            continue
        m = 0
        for e in range(3):
            a, b = v[e], v[(e + 1) % 3]
            if (z[a] > level) != (z[b] > level):
                f = (z[b] - level) / (z[b] - z[a])      # matplotlib's interp
                out[n, 2 * m] = f * x[a] + (1 - f) * x[b]
                out[n, 2 * m + 1] = f * y[a] + (1 - f) * y[b]
                keys[n, m] = min(a, b) * npts + max(a, b)
                m += 1
        if m == 2:
            n += 1
    return n


def contour_lines(tri_obj, z, levels):
    """Polylines of z's levels on the triangulation: (x, y, offsets, level
    index of each polyline), NaN-free, joined through shared edges."""
    t = tri_obj.get_masked_triangles().astype(np.int64)
    x, y, z = tri_obj.x, tri_obj.y, np.asarray(z, np.float64).ravel()
    keep = np.isfinite(z[t]).all(axis=1)
    t = np.ascontiguousarray(t[keep])
    xs, ys, offs, lof = [], [], [0], []
    seg = np.empty((len(t), 4))
    keys = np.empty((len(t), 2), np.int64)
    for li, level in enumerate(levels):
        n = _crossings(x, y, z, t, float(level), seg, keys)
        if not n:
            continue
        s, k = seg[:n], keys[:n]
        ends = {}                                       # edge key -> [(segment, end)]
        for i in range(n):
            ends.setdefault(k[i, 0], []).append((i, 0))
            ends.setdefault(k[i, 1], []).append((i, 1))
        used = np.zeros(n, bool)

        def walk(i, end):
            """Points from segment i's other end onward."""
            pts = []
            while True:
                used[i] = True
                out = 1 - end
                pts.append((s[i, 2 * out], s[i, 2 * out + 1]))
                nxt = [p for p in ends[k[i, out]] if p[0] != i and not used[p[0]]]
                if not nxt:
                    return pts
                i, end = nxt[0]
        opens = [i for i in range(n) if len(ends[k[i, 0]]) == 1 or len(ends[k[i, 1]]) == 1]
        for i in opens + list(range(n)):                # open lines from a free end, then loops
            if used[i]:
                continue
            free = [e for e in (0, 1) if len(ends[k[i, e]]) == 1]
            start = free[0] if free else 0
            pts = [(s[i, 2 * start], s[i, 2 * start + 1])] + walk(i, start)
            if not free:                                # a closed loop: back to its start
                pts.append(pts[0])
            arr = np.asarray(pts)
            xs.append(arr[:, 0])
            ys.append(arr[:, 1])
            offs.append(offs[-1] + len(arr))
            lof.append(li)
    if not xs:
        return np.empty(0), np.empty(0), np.zeros(1, np.int64), np.empty(0, np.int64)
    return (np.concatenate(xs), np.concatenate(ys), np.asarray(offs, np.int64),
            np.asarray(lof, np.int64))


# ---------------------------------------------------------------- artists

from ._artists import _Mappable, _axis_bounds                            # noqa: E402
from ._fields import ContourSet, ContourfArtist                          # noqa: E402
from ._render import Coloring                                            # noqa: E402
from . import _color                                                     # noqa: E402


class _TriGrid:
    """What a contour set reads of its grid: the source crs, the size."""
    rect = False

    def __init__(self, src, n):
        self.src = src
        self.shape = (n,)


def _bounds(tri_obj):
    """Tight autoscaling, as matplotlib's contour sets (sticky at the data)."""
    x, y = tri_obj.x, tri_obj.y
    return (_axis_bounds(x, (x.min(), x.max())), _axis_bounds(y, (y.min(), y.max())))


def _raster(r, ax, tri_obj, src, zt, flat):
    """The triangles' values at the axes box's pixels (NaN off them)."""
    box = ax.box
    h, w = box[3] - box[1], box[2] - box[0]
    values = np.full((max(h, 0), max(w, 0)), np.nan, np.float32)
    if h <= 0 or w <= 0:
        return box, values
    px, py = ax._points_px(tri_obj.x, tri_obj.y, src)
    t = tri_obj.get_masked_triangles().astype(np.int64)
    keep = np.isfinite(zt).all(axis=1)
    raster_triangles(values, np.ascontiguousarray(px - box[0]),
                     np.ascontiguousarray(py - box[1]), np.ascontiguousarray(t[keep]),
                     np.ascontiguousarray(zt[keep], np.float64), flat)
    return box, values


class TriContourf(ContourfArtist):
    """tricontourf: contourf's bands of the linear surface on the triangles."""

    def __init__(self, ax, tri_obj, z, src, levels, **kw):
        self.tri, self.src = tri_obj, src
        super().__init__(ax, _TriGrid(src, tri_obj.x.size), np.asarray(z, np.float64).ravel(),
                         levels, **kw)

    def data_bounds(self):
        return _bounds(self.tri) if self.src == 'data' else None

    def _draw_field(self, r, ax, mode, alpha):
        z = np.asarray(self.z, np.float64)
        box, values = _raster(r, ax, self.tri, self.src, z[self.tri.get_masked_triangles()],
                              False)
        r.c.field_values(box, values, self.coloring(), alpha)


class TriContour(ContourSet):
    """tricontour: contour's lines (colours, dashes, clabel) on the triangles."""

    def __init__(self, ax, tri_obj, z, src, levels, **kw):
        self.tri, self.src = tri_obj, src
        super().__init__(ax, _TriGrid(src, tri_obj.x.size), np.asarray(z, np.float64).ravel(),
                         levels, **kw)

    def data_bounds(self):
        return _bounds(self.tri) if self.src == 'data' else None

    def _grid_lines(self):
        if self._lines is None:
            self._lines = contour_lines(self.tri, self.z, self.levels)
        return self._lines


class TriPcolor(_Mappable):
    """tripcolor: each triangle its colour (flat: its corners' mean, or a
    value per triangle), or interpolated between its corners (gouraud)."""

    zorder = 1.0

    def __init__(self, ax, tri_obj, c, src, shading='flat', cmap=None, norm=None, vmin=None,
                 vmax=None, alpha=None, zorder=None):
        super().__init__(ax, zorder, alpha)
        self.tri, self.src = tri_obj, src
        c = np.ma.filled(np.ma.asarray(c, np.float64), np.nan).ravel()
        ntri = len(tri_obj.triangles)
        if shading not in ('flat', 'gouraud'):
            raise ValueError("shading must be 'flat' or 'gouraud'")
        if c.size == tri_obj.x.size:                    # matplotlib: points first
            self.at_points = True
        elif c.size == ntri:
            if shading == 'gouraud':
                raise ValueError('gouraud shading needs a value at each point')
            self.at_points = False
        else:
            raise ValueError('C must have one value per point or per triangle')
        self.c, self.shading = c, shading
        self.cmap = _color.get_cmap(cmap)
        self._norm, self._norm_in, self._vmin, self._vmax = None, norm, vmin, vmax

    def _zt(self):
        t = self.tri.get_masked_triangles()
        if not self.at_points:
            v = self.c if self.tri.mask is None else self.c[~self.tri.mask]
            return np.repeat(v[:, None], 3, axis=1)
        zt = self.c[t]
        if self.shading == 'flat':                      # matplotlib: the corners' mean
            return np.repeat(zt.mean(axis=1)[:, None], 3, axis=1)
        return zt

    @property
    def values(self):
        return self._zt()[:, 0] if self.shading == 'flat' else self.c

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

    def data_bounds(self):
        if self.src != 'data':
            return None
        bx, by = _axis_bounds(self.tri.x), _axis_bounds(self.tri.y)
        if getattr(self, 'sticky', False):              # a gouraud pcolormesh: QuadMesh's edges
            return bx[:3] + ((bx[0], bx[1]),), by[:3] + ((by[0], by[1]),)
        return bx, by

    # a gouraud pcolormesh's quad centres: (first centre point, the corner
    # points a, b, c, d of each), their colours the corners' mean
    centres = None

    def render(self, r, ax):
        if self.shading == 'gouraud' and self.at_points:
            return self._render_gouraud(r, ax)
        box, values = _raster(r, ax, self.tri, self.src, self._zt(), self.shading == 'flat')
        r.c.field_values(box, values, self.coloring(), self.alpha)

    def _render_gouraud(self, r, ax):
        """matplotlib's: the points coloured, the colours interpolated
        across each triangle (not the values)."""
        box = ax.box
        h, w = box[3] - box[1], box[2] - box[0]
        if h <= 0 or w <= 0:
            return
        n = self.c.size
        rgba = np.empty((1, n, 4), np.uint8)
        _color.shade(self.c.astype(np.float32)[None], self.cmap, self.norm_spec(), out=rgba,
                     alpha=self.alpha, rounding='nearest')
        cols = rgba[0].astype(np.float64)
        if self.centres is not None:
            m0, qa, qb, qc, qd = self.centres
            cols[m0:m0 + qa.size] = 0.25 * (cols[qa] + cols[qb] + cols[qc] + cols[qd])
        px, py = ax._points_px(self.tri.x, self.tri.y, self.src)
        t = self.tri.get_masked_triangles().astype(np.int64)
        t = t[np.isfinite(self.c[t]).all(axis=1)]
        layer = np.zeros((h, w, 4), np.uint8)
        raster_triangles_rgba(layer, np.ascontiguousarray(px - box[0], np.float64),
                              np.ascontiguousarray(py - box[1], np.float64),
                              np.ascontiguousarray(t), cols)
        r.c.image(layer, box[0], box[1])
