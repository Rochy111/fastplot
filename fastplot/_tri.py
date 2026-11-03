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

from . import _agg
from ._jit import njit, prange
from ._raster import square_cover

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
        if not e.size:
            return np.unique(e, axis=0)
        n = int(e.max()) + 1                            # (a pair as one number sorts as the rows do:
        u = np.unique(e[:, 0].astype(np.int64) * n + e[:, 1])   # np.unique by rows is ~10x slower)
        out = np.empty((u.size, 2), e.dtype)
        out[:, 0], out[:, 1] = u // n, u % n
        return out

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


@njit(cache=True, nogil=True)
def _meet(x0, y0, x1, y1, x2, y2, x3, y3):
    """Where the lines (x0, y0)-(x1, y1) and (x2, y2)-(x3, y3) cross."""
    den = (x1 - x0) * (y3 - y2) - (y1 - y0) * (x3 - x2)
    if abs(den) < 1e-30:
        return x1, y1
    t = ((x2 - x0) * (y3 - y2) - (y2 - y0) * (x3 - x2)) / den
    return x0 + t * (x1 - x0), y0 + t * (y1 - y0)


@njit(cache=True, nogil=True, parallel=True)
def draw_gouraud(canvas, bx0, by0, bx1, by1, sx, sy, tri, cols, nb):
    """Agg's gouraud triangles (draw_gouraud_triangles), each over the
    last, inside the box (bx0, by0, bx1, by1) of canvas pixels: the
    triangle's sides moved out half a pixel with its corners cut off
    (agg::dilate_triangle), anti-aliased, its corners' RGBA (cols, per
    point, 0..255) spread over the moved triangle and read at pixel
    centres. Each triangle's shape is worked out once (_gouraud_shapes);
    then nb bands of rows are drawn at once, each taking the triangles in
    order."""
    side, corner, box, abc, ok = _gouraud_shapes(bx0, by0, bx1, by1, sx, sy, tri)
    rows = max(1, -(-(by1 - by0) // max(nb, 1)))
    for k in prange(max(nb, 1)):
        r0 = by0 + k * rows
        _gouraud_rows(canvas, r0, min(by1, r0 + rows) - 1, side, corner, box, abc, ok, cols)


@njit(cache=True, nogil=True)
def _gouraud_shapes(bx0, by0, bx1, by1, sx, sy, tri):
    """Each triangle of draw_gouraud moved out: its sides (x, y, dx, dy,
    length), where they meet and the area between, its pixels, its corners
    (turned to positive area), and whether it is drawn at all. (Not in the
    parallel kernel itself, where numba would make each array expression a
    parallel region of its own.)"""
    nt = tri.shape[0]
    side = np.empty((nt, 6, 5))                 # per side of the moved triangle: x, y, dx, dy, length
    corner = np.empty((nt, 7))                  # its corners where the moved sides meet, its area
    box = np.empty((nt, 4), np.int64)           # its pixels (j0, j1, i0, i1)
    abc = np.empty((nt, 3), np.int64)
    ok = np.zeros(nt, np.bool_)
    from_x, from_y = np.empty(6), np.empty(6)
    px3, py3 = np.empty(3), np.empty(3)
    ox, oy = np.empty(3), np.empty(3)
    for t in range(nt):
        a, b, c = tri[t, 0], tri[t, 1], tri[t, 2]
        area = (sx[b] - sx[a]) * (sy[c] - sy[a]) - (sx[c] - sx[a]) * (sy[b] - sy[a])
        if not (area != 0.0) or not np.isfinite(area):
            continue
        if area < 0.0:
            b, c = c, b
        px3[0], py3[0], px3[1], py3[1], px3[2], py3[2] = sx[a], sy[a], sx[b], sy[b], sx[c], sy[c]
        for e in range(3):                              # each side's way out, half a pixel
            f = (e + 1) % 3
            ex, ey = px3[f] - px3[e], py3[f] - py3[e]
            ln = np.hypot(ex, ey)
            ox[e], oy[e] = 0.5 * ey / ln, -0.5 * ex / ln
        for e in range(3):
            f = (e + 1) % 3
            from_x[2 * e], from_y[2 * e] = px3[e] + ox[e], py3[e] + oy[e]
            from_x[2 * e + 1], from_y[2 * e + 1] = px3[f] + ox[e], py3[f] + oy[e]
        dx0, dy0 = _meet(from_x[4], from_y[4], from_x[5], from_y[5],
                         from_x[0], from_y[0], from_x[1], from_y[1])
        dx1, dy1 = _meet(from_x[0], from_y[0], from_x[1], from_y[1],
                         from_x[2], from_y[2], from_x[3], from_y[3])
        dx2, dy2 = _meet(from_x[2], from_y[2], from_x[3], from_y[3],
                         from_x[4], from_y[4], from_x[5], from_y[5])
        darea = (dx1 - dx0) * (dy2 - dy0) - (dx2 - dx0) * (dy1 - dy0)
        if darea == 0.0:
            continue
        for e in range(6):
            f = (e + 1) % 6
            ex, ey = from_x[f] - from_x[e], from_y[f] - from_y[e]
            side[t, e, 0], side[t, e, 1] = from_x[e], from_y[e]
            side[t, e, 2], side[t, e, 3], side[t, e, 4] = ex, ey, np.hypot(ex, ey)
        corner[t, 0], corner[t, 1], corner[t, 2], corner[t, 3] = dx0, dy0, dx1, dy1
        corner[t, 4], corner[t, 5], corner[t, 6] = dx2, dy2, darea
        box[t, 0] = max(bx0, int(np.floor(from_x.min())))
        box[t, 1] = min(bx1 - 1, int(np.ceil(from_x.max())))
        box[t, 2] = max(by0, int(np.floor(from_y.min())))
        box[t, 3] = min(by1 - 1, int(np.ceil(from_y.max())))
        abc[t, 0], abc[t, 1], abc[t, 2] = a, b, c
        ok[t] = True
    return side, corner, box, abc, ok


@njit(cache=True, nogil=True)
def _gouraud_rows(canvas, r0, r1, side, corner, box, abc, ok, cols):
    """draw_gouraud's triangles in order, rows r0..r1 of them."""
    fx, fy = np.empty(6), np.empty(6)
    sdx, sdy, sln = np.empty(6), np.empty(6), np.empty(6)
    ax, ay, bx, by = np.empty(16), np.empty(16), np.empty(16), np.empty(16)
    for t in range(ok.size):
        if not ok[t]:
            continue
        i0, i1 = max(box[t, 2], r0), min(box[t, 3], r1)
        if i0 > i1:
            continue
        for e in range(6):
            fx[e], fy[e] = side[t, e, 0], side[t, e, 1]
            sdx[e], sdy[e], sln[e] = side[t, e, 2], side[t, e, 3], side[t, e, 4]
        a, b, c = abc[t, 0], abc[t, 1], abc[t, 2]
        dx0, dy0, dx1, dy1 = corner[t, 0], corner[t, 1], corner[t, 2], corner[t, 3]
        dx2, dy2, darea = corner[t, 4], corner[t, 5], corner[t, 6]
        j0, j1 = box[t, 0], box[t, 1]
        for i in range(i0, i1 + 1):
            yc = i + 0.5
            for j in range(j0, j1 + 1):
                xc = j + 0.5
                inside = 1e30                           # the nearest side, from within
                for e in range(6):
                    ln = sln[e]
                    if ln > 0.0:
                        d = (sdx[e] * (yc - fy[e]) - sdy[e] * (xc - fx[e])) / ln
                        inside = min(inside, d)
                if inside <= -0.7072:
                    continue
                cover = 1.0 if inside >= 0.7072 else \
                    square_cover(fx, fy, 6, float(j), float(i), ax, ay, bx, by)
                if cover <= 0.0:
                    continue
                w0 = ((dx2 - dx1) * (yc - dy1) - (dy2 - dy1) * (xc - dx1)) / darea
                w1 = ((dx0 - dx2) * (yc - dy2) - (dy0 - dy2) * (xc - dx2)) / darea
                w2 = 1.0 - w0 - w1
                r = min(255.0, max(0.0, w0 * cols[a, 0] + w1 * cols[b, 0] + w2 * cols[c, 0]))
                g = min(255.0, max(0.0, w0 * cols[a, 1] + w1 * cols[b, 1] + w2 * cols[c, 1]))
                bl = min(255.0, max(0.0, w0 * cols[a, 2] + w1 * cols[b, 2] + w2 * cols[c, 2]))
                al = min(255.0, max(0.0, w0 * cols[a, 3] + w1 * cols[b, 3] + w2 * cols[c, 3]))
                _draw.blend_px(canvas, i, j, np.float32(r), np.float32(g), np.float32(bl),
                               np.float32(al), np.float32(cover))


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


@njit(cache=True, nogil=True)
def _join(s, k, ox, oy, ooff):
    """contour_lines' joining of one level's n segments (s, k) into
    polylines, as its dict of ends does it, when every edge holds at most
    two ends: open lines from their free end, segments in order, then the
    loops, each closed on its first point. Points into ox, oy (3 n room),
    line starts into ooff (n + 1): (points, lines), or (-1, -1) if an edge
    holds more than two ends (the dicts then)."""
    n = s.shape[0]
    flat = k.ravel()                                    # end q = 2 i + e on edge flat[q]
    order = np.argsort(flat, kind='mergesort')
    other = np.full(2 * n, -1, np.int64)                # the end sharing q's edge, if any
    r = 0
    while r < 2 * n:
        r2 = r + 1
        while r2 < 2 * n and flat[order[r2]] == flat[order[r]]:
            r2 += 1
        if r2 - r > 2:
            return -1, -1
        if r2 - r == 2:
            other[order[r]], other[order[r + 1]] = order[r + 1], order[r]
        r = r2
    used = np.zeros(n, np.bool_)
    p = 0
    m = 0
    for sweep in range(2):                              # open lines from a free end, then loops
        for i in range(n):
            if used[i]:
                continue
            free0, free1 = other[2 * i] < 0, other[2 * i + 1] < 0
            if sweep == 0 and not (free0 or free1):
                continue
            start = 0 if free0 else (1 if free1 else 0)
            ooff[m] = p
            ox[p], oy[p] = s[i, 2 * start], s[i, 2 * start + 1]
            p += 1
            j, end = i, start
            while True:
                used[j] = True
                out = 1 - end
                ox[p], oy[p] = s[j, 2 * out], s[j, 2 * out + 1]
                p += 1
                q = other[2 * j + out]
                if q < 0 or used[q // 2]:
                    break
                j, end = q // 2, q % 2
            if not (free0 or free1):                    # a closed loop: back to its start
                ox[p], oy[p] = ox[ooff[m]], oy[ooff[m]]
                p += 1
            m += 1
    return p, m


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
        ox, oy, ooff = np.empty(3 * n), np.empty(3 * n), np.empty(n + 1, np.int64)
        np_, nl = _join(np.ascontiguousarray(s), np.ascontiguousarray(k), ox, oy, ooff)
        if np_ >= 0:
            ooff[nl] = np_
            xs.append(ox[:np_])
            ys.append(oy[:np_])
            offs.extend((offs[-1] + ooff[1:nl + 1]).tolist())
            lof.extend([li] * nl)
            continue
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
from . import _color, _draw, _jit                                        # noqa: E402


class _TriGrid:
    """What a contour set reads of its grid: the source crs, the size."""
    rect = False

    def __init__(self, src, n):
        self.src = src
        self.shape = (n,)


@njit(cache=True, nogil=True, parallel=True)
def _take_neighbours(values, touched):
    """Each pixel touched and without a value takes its neighbours' largest:
    the bands show there, highest first."""
    H, W = values.shape
    src = values.copy()
    for i in prange(H):
        for j in range(W):
            if touched[i, j, 3] and not np.isfinite(src[i, j]):
                for a in range(max(i - 1, 0), min(i + 2, H)):
                    for b in range(max(j - 1, 0), min(j + 2, W)):
                        v = src[a, b]
                        if np.isfinite(v) and not v <= values[i, j]:
                            values[i, j] = v


def _touch_hull(values, px, py, box, tri):
    """Pixel centres miss what a filled contour does not: Agg fills, not
    antialiased, every pixel a polygon touches, so one past the hull that
    it reaches into. The triangles as one path give those pixels exactly."""
    x, y = px - box[0], py - box[1]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    turn = (x[b] - x[a]) * (y[c] - y[a]) - (x[c] - x[a]) * (y[b] - y[a]) < 0.0
    b, c = np.where(turn, c, b), np.where(turn, b, c)   # all one way round: inner edges cancel
    pts = np.column_stack([x[np.c_[a, b, c, a]].ravel(), y[np.c_[a, b, c, a]].ravel()])
    keep = np.isfinite(pts).all(axis=1).reshape(-1, 4).all(axis=1)
    h, w = values.shape
    tile = np.zeros((h, w, 4), np.uint8)
    n = int(keep.sum())
    _agg.fill(tile, np.ascontiguousarray(pts.reshape(-1, 4, 2)[keep].reshape(-1, 2)),
              np.arange(n + 1, dtype=np.int64) * 4, np.zeros(n, np.int64),
              np.array([[255, 255, 255, 255]], np.uint8), (0, 0, w, h))
    _take_neighbours(values, tile)


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
        if values.size:
            _touch_hull(values, *ax._points_px(self.tri.x, self.tri.y, self.src), box,
                        self.tri.get_masked_triangles().astype(np.int64))
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
            self._open = np.zeros(self._lines[2].size - 1, bool)
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
        if r.measuring or box[3] <= box[1] or box[2] <= box[0]:
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
        c = r.c._to_cpu() if hasattr(r.c, '_to_cpu') else r.c
        draw_gouraud(c.a, int(box[0]), int(box[1]), int(box[2]), int(box[3]),
                     np.ascontiguousarray(px, np.float64), np.ascontiguousarray(py, np.float64),
                     np.ascontiguousarray(t), cols, _jit.team())
