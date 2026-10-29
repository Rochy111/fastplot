"""matplotlib.tri's finder, interpolators and analyzer on fastplot's
Triangulation.

    finder = t.get_trifinder();  finder(x, y)           # triangle index, -1 off the mesh
    LinearTriInterpolator(t, z)(xi, yi)                 # masked off the mesh
    CubicTriInterpolator(t, z, kind='min_E').gradient(xi, yi)
    TriAnalyzer(t).get_flat_tri_mask(0.01)

matplotlib finds triangles with a trapezoid map (C++). Its answers are a
function of the point alone wherever the map is deterministic, so they are
found here by a grid-bucketed point-in-triangle search (numba) that makes
the same orientation tests on the same edges, with matplotlib's rules for
the ties: a point exactly on an edge belongs to the triangle above it
(lexicographic left-to-right edges; to the left of a vertical one), or to
the one below on the mesh's lower border; a point exactly on a vertex to
the lowest-numbered unmasked triangle at that vertex. NaN and infinite
points are off the mesh (matplotlib's own answer for some of them depends
on its tree).

The interpolators are matplotlib's: plane coefficients as its C++ computes
them (so linear values come out bit for bit), the reduced Hsieh-Clough-
Tocher element evaluated per point in numba, and the nodal derivatives
(geom, min_E with its Jacobi-preconditioned conjugate gradient) by
matplotlib's own numpy algebra, so the iteration sees the same numbers and
stops at the same step.

matplotlib's C++ triangulation turns a given triangulation's clockwise
triangles anticlockwise in place the first time it is needed; so does
this module (the Triangulation's triangles become a corrected copy).
"""
import warnings

import numpy as np

from ._jit import njit, prange
from ._tri import Triangulation


# ------------------------------------------------------- the C++ triangulation

def _cpp_triangles(tri):
    """The triangles as matplotlib's C++ triangulation holds them: a given
    (not Delaunay) triangulation's clockwise triangles made anticlockwise
    by swapping their last two points, once."""
    t = tri.triangles
    if tri.is_delaunay or getattr(tri, '_ccw', None) is t:
        return t
    x, y = tri.x, tri.y
    p0, p1, p2 = t[:, 0], t[:, 1], t[:, 2]
    cw = (x[p1] - x[p0]) * (y[p2] - y[p0]) - (y[p1] - y[p0]) * (x[p2] - x[p0]) < 0.0
    if cw.any():
        t = t.copy()                                    # not the caller's array
        t[cw, 1], t[cw, 2] = p2[cw], p1[cw]
        tri.triangles = t
    tri._ccw = t
    return t


@njit(cache=True, nogil=True)
def _neighbors_kernel(tri, masked, npts):
    """matplotlib's calculate_neighbors: each unmasked triangle edge paired
    with the reversed edge of another, in triangle order."""
    ntri = tri.shape[0]
    nb = np.full((ntri, 3), -1, np.int32)
    open_edges = {np.int64(-1): np.int64(-1)}           # (typed by a key never met)
    for t in range(ntri):
        if masked[t]:
            continue
        for e in range(3):
            a, b = np.int64(tri[t, e]), np.int64(tri[t, (e + 1) % 3])
            back = b * npts + a
            if back in open_edges:
                te = open_edges.pop(back)
                nb[t, e] = te // 3
                nb[te // 3, te % 3] = t
            else:
                open_edges[a * npts + b] = np.int64(3 * t + e)
    return nb


def neighbors(tri):
    """matplotlib's Triangulation.neighbors: (ntri, 3), the triangle across
    the edge from point triangles[i, j] to triangles[i, (j+1) % 3], -1 for
    none (masked triangles neither have nor are neighbours)."""
    t = _cpp_triangles(tri)
    key = (t, tri.mask)
    cached = getattr(tri, '_nb', None)
    if cached is not None and cached[0][0] is key[0] and cached[0][1] is key[1]:
        return cached[1]
    masked = np.zeros(len(t), bool) if tri.mask is None else tri.mask
    nb = _neighbors_kernel(np.ascontiguousarray(t), masked, tri.x.size)
    tri._nb = (key, nb)
    return nb


@njit(cache=True, nogil=True)
def _planes_kernel(x, y, z, tri, masked):
    """matplotlib's calculate_plane_coefficients, operation for operation."""
    ntri = tri.shape[0]
    planes = np.zeros((ntri, 3))
    for t in range(ntri):
        if masked[t]:
            continue
        a, b, c = tri[t, 0], tri[t, 1], tri[t, 2]
        x0, y0, z0 = x[a], y[a], z[a]
        ax, ay, az = x[b] - x0, y[b] - y0, z[b] - z0
        bx, by, bz = x[c] - x0, y[c] - y0, z[c] - z0
        nx = ay * bz - az * by
        ny = az * bx - ax * bz
        nz = ax * by - ay * bx
        if nz == 0.0:                                   # colinear: the pseudo-inverse
            sum2 = ax * ax + ay * ay + bx * bx + by * by
            pa = (ax * az + bx * bz) / sum2
            pb = (ay * az + by * bz) / sum2
            planes[t, 0] = pa
            planes[t, 1] = pb
            planes[t, 2] = z0 - pa * x0 - pb * y0
        else:
            planes[t, 0] = -nx / nz
            planes[t, 1] = -ny / nz
            planes[t, 2] = (nx * x0 + ny * y0 + nz * z0) / nz
    return planes


def plane_coefficients(tri, z):
    """(ntri, 3) planes z = a x + b y + c of the triangles (zero where masked)."""
    z = np.asarray(z, np.float64).ravel()
    if z.size != tri.x.size:
        raise ValueError('z must have one value per point')
    t = np.ascontiguousarray(_cpp_triangles(tri))
    masked = np.zeros(len(t), bool) if tri.mask is None else tri.mask
    return _planes_kernel(tri.x, tri.y, z, t, masked)


# ----------------------------------------------------------------- finders

class TriFinder:
    """matplotlib.tri.TriFinder: finds the triangle of a Triangulation each
    (x, y) point lies in. Usually had from Triangulation.get_trifinder."""

    def __init__(self, triangulation):
        if not isinstance(triangulation, Triangulation):
            raise TypeError('triangulation must be a Triangulation')
        self._triangulation = triangulation

    def __call__(self, x, y):
        raise NotImplementedError


@njit(cache=True, nogil=True)
def _cell(v, v0, s, n):
    """The grid column (row) of v: monotonic in v, so a box's cells hold
    every point inside it."""
    return min(int((v - v0) * s), n - 1)


@njit(cache=True, nogil=True)
def _tri_cells(px, py, tri, t, x0, y0, sx, sy, nx, ny):
    """The grid cells a triangle's bounding box touches: i0, i1, j0, j1."""
    a, b, c = tri[t, 0], tri[t, 1], tri[t, 2]
    return (_cell(min(px[a], px[b], px[c]), x0, sx, nx),
            _cell(max(px[a], px[b], px[c]), x0, sx, nx),
            _cell(min(py[a], py[b], py[c]), y0, sy, ny),
            _cell(max(py[a], py[b], py[c]), y0, sy, ny))


@njit(cache=True, nogil=True)
def _bucket(px, py, tri, live, x0, y0, sx, sy, nx, ny):
    """The live triangles into the grid cells their bounding boxes touch:
    (cell offsets, triangle indices), CSR."""
    start = np.zeros(nx * ny + 1, np.int64)
    for t in live:
        i0, i1, j0, j1 = _tri_cells(px, py, tri, t, x0, y0, sx, sy, nx, ny)
        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                start[j * nx + i + 1] += 1
    for c in range(nx * ny):
        start[c + 1] += start[c]
    items = np.empty(start[-1], np.int32)
    fill = start[:-1].copy()
    for t in live:
        i0, i1, j0, j1 = _tri_cells(px, py, tri, t, x0, y0, sx, sy, nx, ny)
        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                items[fill[j * nx + i]] = t
                fill[j * nx + i] += 1
    return start, items


@njit(cache=True, nogil=True)
def _find_one(X, Y, px, py, tri, vtri, start, items, box, sx, sy, nx, ny):
    """The triangle holding (X, Y) as matplotlib's trapezoid map finds it."""
    if not (X >= box[0] and X <= box[1] and Y >= box[2] and Y <= box[3]):
        return -1                                       # (NaN too)
    c = _cell(Y, box[2], sy, ny) * nx + _cell(X, box[0], sx, nx)
    below = -1
    for m in range(start[c], start[c + 1]):
        t = items[m]
        on = -1
        on_above = False
        inside = True
        for e in range(3):
            s, f = tri[t, e], tri[t, (e + 1) % 3]
            above = px[f] > px[s] or (px[f] == px[s] and py[f] > py[s])
            if above:                                   # edge s -> f, t above it
                lx, ly, rx, ry = px[s], py[s], px[f], py[f]
            else:
                lx, ly, rx, ry = px[f], py[f], px[s], py[s]
            cz = (X - lx) * (ry - ly) - (Y - ly) * (rx - lx)   # > 0: below the edge
            if (cz > 0.0) if above else (cz < 0.0):
                inside = False
                break
            if cz == 0.0:
                on, on_above = e, above
        if not inside:
            continue
        if on < 0:
            return t
        for v in range(3):                              # a vertex: its first triangle
            p = tri[t, v]
            if X == px[p] and Y == py[p]:
                return vtri[p]
        if on_above:                                    # an edge: the triangle above it
            return t
        if below < 0:
            below = t
    return below


@njit(cache=True, nogil=True, parallel=True)
def _find_many(qx, qy, px, py, tri, vtri, start, items, box, sx, sy, nx, ny, out):
    for k in prange(qx.size):
        out[k] = _find_one(qx[k], qy[k], px, py, tri, vtri, start, items, box, sx, sy,
                           nx, ny)


class TrapezoidMapTriFinder(TriFinder):
    """matplotlib.tri.TrapezoidMapTriFinder: the triangle of each point, or
    -1. matplotlib's answers (its trapezoid map's tie rules) from a
    grid-bucketed search; rebuilt when the triangulation's mask or
    triangles are replaced."""

    def __init__(self, triangulation):
        super().__init__(triangulation)
        self._initialize()

    def _initialize(self):
        tri = self._triangulation
        t = np.ascontiguousarray(_cpp_triangles(tri), np.int32)
        self._state = (tri.triangles, tri.mask, tri.x, tri.y)
        live = np.arange(len(t), dtype=np.int32)
        if tri.mask is not None:
            live = live[~tri.mask]
        px, py = tri.x + 0.0, tri.y + 0.0               # matplotlib's: no -0.0
        self._px, self._py, self._t = px, py, t
        vtri = np.full(px.size, np.iinfo(np.int32).max, np.int32)
        np.minimum.at(vtri, t[live].ravel(), np.repeat(live, 3))
        self._vtri = vtri
        if live.size == 0:
            self._box = None
            return
        used = np.unique(t[live])
        x0, x1, y0, y1 = px[used].min(), px[used].max(), py[used].min(), py[used].max()
        w, h = x1 - x0, y1 - y0
        n = live.size
        if w > 0 and h > 0:
            nx = int(min(max(np.sqrt(n * w / h), 1), n))
            ny = int(min(max(n // nx, 1), n))
        else:
            nx = n if w > 0 else 1
            ny = n if h > 0 else 1
        self._grid = (nx / w if w > 0 else 0.0, ny / h if h > 0 else 0.0, nx, ny)
        self._box = np.array([x0, x1, y0, y1])
        sx, sy = self._grid[:2]
        self._start, self._items = _bucket(px, py, t, live, x0, y0, sx, sy, nx, ny)

    def __call__(self, x, y):
        """The index of the triangle each (x, y) lies in, -1 for none
        (int32, the shape of x and y)."""
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        if x.shape != y.shape:
            raise ValueError('x and y must be array-like with the same shape')
        tri = self._triangulation
        state = (tri.triangles, tri.mask, tri.x, tri.y)
        if any(a is not b for a, b in zip(state, self._state)):
            self._initialize()
        out = np.full(x.size, -1, np.int32)
        if self._box is not None and x.size:
            sx, sy, nx, ny = self._grid
            _find_many(np.ascontiguousarray(x.ravel()), np.ascontiguousarray(y.ravel()),
                       self._px, self._py, self._t, self._vtri, self._start, self._items,
                       self._box, sx, sy, nx, ny, out)
        return out.reshape(x.shape)


# ------------------------------------------------------------ interpolators

class TriInterpolator:
    """matplotlib.tri.TriInterpolator: the base of the interpolators on a
    triangular grid, called as interp(x, y) and interp.gradient(x, y)."""

    def __init__(self, triangulation, z, trifinder=None):
        if not isinstance(triangulation, Triangulation):
            raise TypeError('triangulation must be a Triangulation')
        self._triangulation = triangulation
        self._z = np.asarray(z)
        if self._z.shape != triangulation.x.shape:
            raise ValueError('z array must have same length as triangulation x and y arrays')
        if trifinder is not None and not isinstance(trifinder, TriFinder):
            raise TypeError('trifinder must be a TriFinder')
        self._trifinder = trifinder or triangulation.get_trifinder()
        self._unit_x = 1.0                              # coordinate scaling (cubic)
        self._unit_y = 1.0
        self._tri_renum = None                          # triangle renumbering (cubic)

    def __call__(self, x, y):
        """The interpolated values at (x, y): a masked array of their shape,
        masked off the triangulation."""
        return self._interpolate_multikeys(x, y, tri_index=None, return_keys=('z',))[0]

    def gradient(self, x, y):
        """[dz/dx, dz/dy] of the interpolant at (x, y): masked arrays of
        their shape, masked off the triangulation."""
        return self._interpolate_multikeys(x, y, tri_index=None,
                                           return_keys=('dzdx', 'dzdy'))

    def _interpolate_multikeys(self, x, y, tri_index=None, return_keys=('z',)):
        """matplotlib's: the requested keys ('z', 'dzdx', 'dzdy') at (x, y),
        in triangles tri_index (found when None, -1 off the mesh)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        sh_ret = x.shape
        if x.shape != y.shape:
            raise ValueError('x and y shall have same shapes. Given: %s and %s'
                             % (x.shape, y.shape))
        x, y = np.ravel(x), np.ravel(y)
        x_scaled, y_scaled = x / self._unit_x, y / self._unit_y
        if tri_index is None:
            tri_index = self._trifinder(x, y)
        else:
            tri_index = np.asarray(tri_index)
            if tri_index.shape != sh_ret:
                raise ValueError('tri_index array is provided and shall have same shape as '
                                 'x and y. Given: %s and %s' % (tri_index.shape, sh_ret))
        tri_index = np.ravel(tri_index).astype(np.int64)
        idx = np.full(tri_index.size, -1, np.int64)
        mask_in = tri_index != -1
        valid = tri_index[mask_in]
        if self._tri_renum is not None:
            valid = self._tri_renum[valid].astype(np.int64)
        n = self._ntri()
        idx[mask_in] = np.where(valid < 0, valid + n, valid)   # numpy's wrap-around
        keys = []
        for key in return_keys:
            if key not in ('z', 'dzdx', 'dzdy'):
                raise ValueError("return_keys items shall take values in {'z', 'dzdx', 'dzdy'}")
            keys.append(key)
        values = self._evaluate(set(keys), idx, np.ascontiguousarray(x_scaled),
                                np.ascontiguousarray(y_scaled))
        scale = {'z': 1., 'dzdx': 1. / self._unit_x, 'dzdy': 1. / self._unit_y}
        ret = []
        for key in keys:
            v = values[key]
            if scale[key] != 1.:
                v = v * scale[key]
            ret.append(np.ma.masked_invalid(v.reshape(sh_ret), copy=False))
        return ret


@njit(cache=True, nogil=True, parallel=True)
def _linear_kernel(planes, idx, x, y, z, dzdx, dzdy, want_z, want_d):
    for k in prange(idx.size):
        t = idx[k]
        if t < 0:
            if want_z:
                z[k] = np.nan
            if want_d:
                dzdx[k] = dzdy[k] = np.nan
            continue
        if want_z:
            z[k] = planes[t, 0] * x[k] + planes[t, 1] * y[k] + planes[t, 2]
        if want_d:
            dzdx[k] = planes[t, 0]
            dzdy[k] = planes[t, 1]


class LinearTriInterpolator(TriInterpolator):
    """matplotlib.tri.LinearTriInterpolator: each triangle a plane through
    its corners' values -- continuous, with a gradient constant per
    triangle."""

    def __init__(self, triangulation, z, trifinder=None):
        super().__init__(triangulation, z, trifinder)
        self._plane_coefficients = plane_coefficients(triangulation, self._z)

    def _ntri(self):
        return len(self._plane_coefficients)

    def _evaluate(self, keys, idx, x, y):
        n = idx.size
        z, dx, dy = np.empty(n), np.empty(n), np.empty(n)
        _linear_kernel(self._plane_coefficients, idx, x, y, z, dx, dy, 'z' in keys,
                       'dzdx' in keys or 'dzdy' in keys)
        return {'z': z, 'dzdx': dx, 'dzdy': dy}


# The reduced HCT element (Bernadou & Hassan, Int. J. Numer. Meth. Eng.
# 17(5), 1981, p. 11): shape-function coefficients as functions of the
# triangle eccentricities, as matplotlib tabulates them.
_M = np.array([
    [0.00, 0.00, 0.00, 4.50, 4.50, 0.00, 0.00, 0.00, 0.00, 0.00],
    [-0.25, 0.00, 0.00, 0.50, 1.25, 0.00, 0.00, 0.00, 0.00, 0.00],
    [-0.25, 0.00, 0.00, 1.25, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.50, 1.00, 0.00, -1.50, 0.00, 3.00, 3.00, 0.00, 0.00, 3.00],
    [0.00, 0.00, 0.00, -0.25, 0.25, 0.00, 1.00, 0.00, 0.00, 0.50],
    [0.25, 0.00, 0.00, -0.50, -0.25, 1.00, 0.00, 0.00, 0.00, 1.00],
    [0.50, 0.00, 1.00, 0.00, -1.50, 0.00, 0.00, 3.00, 3.00, 3.00],
    [0.25, 0.00, 0.00, -0.25, -0.50, 0.00, 0.00, 0.00, 1.00, 1.00],
    [0.00, 0.00, 0.00, 0.25, -0.25, 0.00, 0.00, 1.00, 0.00, 0.50]])
_M0 = np.array([
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [-1.00, 0.00, 0.00, 1.50, 1.50, 0.00, 0.00, 0.00, 0.00, -3.00],
    [-0.50, 0.00, 0.00, 0.75, 0.75, 0.00, 0.00, 0.00, 0.00, -1.50],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [1.00, 0.00, 0.00, -1.50, -1.50, 0.00, 0.00, 0.00, 0.00, 3.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.50, 0.00, 0.00, -0.75, -0.75, 0.00, 0.00, 0.00, 0.00, 1.50]])
_M1 = np.array([
    [-0.50, 0.00, 0.00, 1.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [-0.25, 0.00, 0.00, 0.75, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.50, 0.00, 0.00, -1.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.25, 0.00, 0.00, -0.75, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]])
_M2 = np.array([
    [0.50, 0.00, 0.00, 0.00, -1.50, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.25, 0.00, 0.00, 0.00, -0.75, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [-0.50, 0.00, 0.00, 0.00, 1.50, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [-0.25, 0.00, 0.00, 0.00, 0.75, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]])
_MS = np.stack([_M, _M0, _M1, _M2])
_ROTATE_DV = np.array([[1., 0.], [0., 1.], [0., 1.], [-1., -1.], [-1., -1.], [1., 0.]])
_ROTATE_D2V = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                        [0., 1., 0.], [1., 1., 1.], [0., -2., -1.],
                        [1., 1., 1.], [1., 0., 0.], [-2., 0., -1.]])
_GAUSS_PTS = np.array([[13., 4., 1.], [4., 13., 1.], [7., 7., 4.], [1., 13., 4.],
                       [1., 4., 13.], [4., 7., 7.], [4., 1., 13.], [13., 1., 4.],
                       [7., 4., 7.]]) / 18.     # 3 per subtriangle: P2-exact
_GAUSS_W = np.ones(9) / 9.
_E = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 2.]])   # curvature energy
_J0_TO_J1 = np.array([[-1., 1.], [-1., 0.]])
_J0_TO_J2 = np.array([[0., -1.], [1., -1.]])


@njit(cache=True, nogil=True)
def _hct_point(tp, ecc, dof, X, Y, ms, rot, want_z, want_d):
    """The reduced HCT element at (X, Y) in the triangle tp (3, 2): its
    value and global derivatives, as matplotlib's vectorized algebra."""
    ax, ay = tp[1, 0] - tp[0, 0], tp[1, 1] - tp[0, 1]
    bx, by = tp[2, 0] - tp[0, 0], tp[2, 1] - tp[0, 1]
    ox, oy = X - tp[0, 0], Y - tp[0, 1]
    m00, m01 = ax * ax + ay * ay, ax * bx + ay * by     # the metric, pseudo-inverted
    m10, m11 = bx * ax + by * ay, bx * bx + by * by
    prod1 = m00 * m11
    delta = prod1 - m01 * m10
    if abs(delta) > 1e-8 * abs(prod1):
        i00, i01, i10, i11 = m11 / delta, -m01 / delta, -m10 / delta, m00 / delta
    else:
        tr = m00 + m11
        tz = 1.0 if abs(tr) < 1.e-8 else 0.0
        sq = (1. - tz) / (tr ** 2 + tz)
        i00, i01, i10, i11 = m00 * sq, m01 * sq, m10 * sq, m11 * sq
    c0, c1 = ax * ox + ay * oy, bx * ox + by * oy
    k0, k1 = i00 * c0 + i01 * c1, i10 * c0 + i11 * c1
    a0, a1, a2 = 1 - k0 - k1, k0, k1
    sub = 0                                             # the subtriangle: argmin alpha
    if a1 < a0:
        sub = 1
    if a2 < (a1 if sub == 1 else a0):
        sub = 2
    alpha = (a0, a1, a2)
    x, y, z = alpha[sub], alpha[(sub + 1) % 3], alpha[(sub + 2) % 3]
    e0, e1, e2 = ecc[sub], ecc[(sub + 1) % 3], ecc[(sub + 2) % 3]
    xs, ys, zs = x * x, y * y, z * z
    val = np.nan
    gx = gy = np.nan
    if want_z:
        V = (xs * x, ys * y, zs * z, xs * z, xs * y, ys * x, ys * z, zs * y, zs * x, x * y * z)
        val = 0.0
        for i in range(9):
            p0 = p1 = p2 = p3 = 0.0
            for k in range(10):
                p0 += ms[0, i, k] * V[k]
                p1 += ms[1, i, k] * V[k]
                p2 += ms[2, i, k] * V[k]
                p3 += ms[3, i, k] * V[k]
            val += dof[(i + 3 * sub) % 9] * (p0 + e0 * p1 + e1 * p2 + e2 * p3)
    if want_d:
        u = (-3. * xs, 3. * ys, 0., -2. * x * z, -2. * x * y + xs, 2. * x * y - ys, 2. * y * z,
             zs, -zs, x * z - y * z)
        v = (-3. * xs, 0., 3. * zs, -2. * x * z + xs, -2. * x * y, -ys, ys, 2. * y * z,
             2. * x * z - zs, x * y - y * z)
        r00, r01 = rot[2 * sub, 0], rot[2 * sub, 1]     # back to the first apex basis
        r10, r11 = rot[2 * sub + 1, 0], rot[2 * sub + 1, 1]
        w0 = (u[0] * r00 + v[0] * r10, u[1] * r00 + v[1] * r10, u[2] * r00 + v[2] * r10,
              u[3] * r00 + v[3] * r10, u[4] * r00 + v[4] * r10, u[5] * r00 + v[5] * r10,
              u[6] * r00 + v[6] * r10, u[7] * r00 + v[7] * r10, u[8] * r00 + v[8] * r10,
              u[9] * r00 + v[9] * r10)
        w1 = (u[0] * r01 + v[0] * r11, u[1] * r01 + v[1] * r11, u[2] * r01 + v[2] * r11,
              u[3] * r01 + v[3] * r11, u[4] * r01 + v[4] * r11, u[5] * r01 + v[5] * r11,
              u[6] * r01 + v[6] * r11, u[7] * r01 + v[7] * r11, u[8] * r01 + v[8] * r11,
              u[9] * r01 + v[9] * r11)
        d0 = d1 = 0.0
        for i in range(9):
            p0 = p1 = p2 = p3 = q0 = q1 = q2 = q3 = 0.0
            for k in range(10):
                p0 += ms[0, i, k] * w0[k]
                p1 += ms[1, i, k] * w0[k]
                p2 += ms[2, i, k] * w0[k]
                p3 += ms[3, i, k] * w0[k]
                q0 += ms[0, i, k] * w1[k]
                q1 += ms[1, i, k] * w1[k]
                q2 += ms[2, i, k] * w1[k]
                q3 += ms[3, i, k] * w1[k]
            f = dof[(i + 3 * sub) % 9]
            d0 += f * (p0 + e0 * p1 + e1 * p2 + e2 * p3)
            d1 += f * (q0 + e0 * q1 + e1 * q2 + e2 * q3)
        # J = [[ax, ay], [bx, by]], safely inverted (0 when rank-deficient)
        prod1 = ax * by
        delta = prod1 - ay * bx
        dinv = 1. / delta if abs(delta) > 1e-8 * abs(prod1) else 0.0
        j00, j01, j10, j11 = by * dinv, -ay * dinv, -bx * dinv, ax * dinv
        gx = j00 * d0 + j01 * d1
        gy = j10 * d0 + j11 * d1
    return val, gx, gy


@njit(cache=True, nogil=True, parallel=True)
def _cubic_kernel(tris_pts, eccs, dofs, idx, x, y, ms, rot, z, dzdx, dzdy, want_z, want_d):
    for k in prange(idx.size):
        t = idx[k]
        if t < 0:
            z[k] = dzdx[k] = dzdy[k] = np.nan
            continue
        z[k], dzdx[k], dzdy[k] = _hct_point(tris_pts[t], eccs[t], dofs[t], x[k], y[k], ms,
                                            rot, want_z, want_d)


class CubicTriInterpolator(TriInterpolator):
    """matplotlib.tri.CubicTriInterpolator: a C1 cubic on a Clough-Tocher
    split of each triangle (the reduced HCT element), with the nodal
    derivatives minimizing the bending energy (kind='min_E'), averaged from
    the triangles' planes (kind='geom'), or given (kind='user', dz=(dzdx,
    dzdy))."""

    def __init__(self, triangulation, z, kind='min_E', trifinder=None, dz=None):
        super().__init__(triangulation, z, trifinder)
        _cpp_triangles(triangulation)                   # anticlockwise, as matplotlib
        self._z = np.array(self._z)                     # (matplotlib renumbers z in place)
        (compressed_triangles, compressed_x, compressed_y, tri_renum,
         node_renum) = TriAnalyzer(triangulation)._get_compressed_triangulation()
        self._triangles = compressed_triangles
        self._tri_renum = tri_renum
        valid_node = node_renum != -1
        self._z[node_renum[valid_node]] = self._z[valid_node]
        self._unit_x = np.ptp(compressed_x)
        self._unit_y = np.ptp(compressed_y)
        self._pts = np.column_stack([compressed_x / self._unit_x,
                                     compressed_y / self._unit_y])
        self._tris_pts = self._pts[self._triangles]
        self._eccs = _tri_eccentricities(self._tris_pts)
        if kind not in ('user', 'geom', 'min_E'):
            raise ValueError("kind must be 'user', 'geom' or 'min_E'")
        self._dof = self._compute_dof(kind, dz=dz)

    def _ntri(self):
        return len(self._triangles)

    def _compute_dof(self, kind, dz=None):
        """The nodal degrees of freedom of the HCT elements (ntri, 9)."""
        if kind == 'user':
            if dz is None:
                raise ValueError("For a CubicTriInterpolator with *kind*='user', a valid *dz* "
                                 "argument is expected.")
            TE = _DOF_estimator_user(self, dz=dz)
        elif kind == 'geom':
            TE = _DOF_estimator_geom(self)
        else:
            TE = _DOF_estimator_min_E(self)
        return TE.compute_dof_from_df()

    def _evaluate(self, keys, idx, x, y):
        n = idx.size
        z, dx, dy = np.empty(n), np.empty(n), np.empty(n)
        if len(self._triangles):
            _cubic_kernel(np.ascontiguousarray(self._tris_pts), np.ascontiguousarray(self._eccs),
                          np.ascontiguousarray(self._dof, np.float64), idx, x, y, _MS,
                          _ROTATE_DV, z, dx, dy, 'z' in keys, 'dzdx' in keys or 'dzdy' in keys)
        else:
            z[:] = dx[:] = dy[:] = np.nan
        return {'z': z, 'dzdx': dx, 'dzdy': dy}


def _jacobian(tris_pts):
    """J at each triangle's first apex, [[a], [b]] of its sides (ntri, 2, 2)."""
    a = np.array(tris_pts[:, 1, :] - tris_pts[:, 0, :])
    b = np.array(tris_pts[:, 2, :] - tris_pts[:, 0, :])
    return _to_matrix_vectorized([[a[:, 0], a[:, 1]], [b[:, 0], b[:, 1]]])


def _tri_eccentricities(tris_pts):
    """The HCT eccentricity parameters of each triangle (ntri, 3)."""
    a = np.expand_dims(tris_pts[:, 2, :] - tris_pts[:, 1, :], axis=2)
    b = np.expand_dims(tris_pts[:, 0, :] - tris_pts[:, 2, :], axis=2)
    c = np.expand_dims(tris_pts[:, 1, :] - tris_pts[:, 0, :], axis=2)
    dot_a = (_transpose_vectorized(a) @ a)[:, 0, 0]
    dot_b = (_transpose_vectorized(b) @ b)[:, 0, 0]
    dot_c = (_transpose_vectorized(c) @ c)[:, 0, 0]
    with np.errstate(divide='ignore', invalid='ignore'):   # (duplicate points)
        return _to_matrix_vectorized([[(dot_c - dot_b) / dot_a], [(dot_a - dot_c) / dot_b],
                                      [(dot_b - dot_a) / dot_c]])[:, :, 0]


# ---------------------------------------------------- nodal derivatives (dof)
# matplotlib's numpy algebra, the same products on the same operands. The
# min_E conjugate gradient on a mesh with flat border triangles runs
# hundreds of steps to its relative tolerance, where an ulp's difference in
# the stiffness grows to ~1e-5 in the answer: agreeing with matplotlib to
# ~1e-12 takes its numbers bit for bit (numpy's matmul goes through BLAS).

def _d2_parts(g):
    """At Gauss point g (3 per subtriangle, exact for the P2 second
    derivatives): its subtriangle and M_m @ d2V for the 4 shape matrices
    ((1, 9, 3) each), d2V the cubic monomials' second derivatives in the
    first apex basis -- what matplotlib's get_d2Sidksij2 computes alike
    for every triangle."""
    alpha = np.expand_dims(_GAUSS_PTS[g:g + 1], 2)
    subtri = np.argmin(alpha, axis=1)[:, 0]
    ksi = _roll_vectorized(alpha, -subtri, axis=0)
    x = ksi[:, 0, 0]
    y = ksi[:, 1, 0]
    z = ksi[:, 2, 0]
    d2V = _to_matrix_vectorized([
        [6. * x, 6. * x, 6. * x],
        [6. * y, 0., 0.],
        [0., 6. * z, 0.],
        [2. * z, 2. * z - 4. * x, 2. * z - 2. * x],
        [2. * y - 4. * x, 2. * y, 2. * y - 2. * x],
        [2. * x - 4. * y, 0., -2. * y],
        [2. * z, 0., 2. * y],
        [0., 2. * y, 2. * z],
        [0., 2. * x - 4. * z, -2. * z],
        [-2. * z, -2. * y, x - y - z]])
    d2V = d2V @ _extract_submatrices(_ROTATE_D2V, subtri, block_size=3, axis=0)
    return int(subtri[0]), [_M @ d2V, _M0 @ d2V, _M1 @ d2V, _M2 @ d2V]


_D2 = []                                                # (made on first use)


def _bending_matrices(J, ecc):
    """matplotlib's get_bending_matrices: each element's bending stiffness
    in global nodal dofs, the 9-point integral of d2S E d2S^T rotated by
    the dofs' Jacobians, times the area (ntri, 9, 9)."""
    n = np.size(ecc, 0)
    J1 = _J0_TO_J1 @ J
    J2 = _J0_TO_J2 @ J
    DOF_rot = np.zeros([n, 9, 9], dtype=np.float64)
    DOF_rot[:, 0, 0] = 1
    DOF_rot[:, 3, 3] = 1
    DOF_rot[:, 6, 6] = 1
    DOF_rot[:, 1:3, 1:3] = J
    DOF_rot[:, 4:6, 4:6] = J1
    DOF_rot[:, 7:9, 7:9] = J2
    J_inv = _safe_inv22_vectorized(J)
    Ji00 = J_inv[:, 0, 0]
    Ji11 = J_inv[:, 1, 1]
    Ji10 = J_inv[:, 1, 0]
    Ji01 = J_inv[:, 0, 1]
    H_rot = _to_matrix_vectorized([
        [Ji00 * Ji00, Ji10 * Ji10, Ji00 * Ji10],
        [Ji01 * Ji01, Ji11 * Ji11, Ji01 * Ji11],
        [2 * Ji00 * Ji01, 2 * Ji11 * Ji10, Ji00 * Ji11 + Ji10 * Ji01]])
    area = 0.5 * (J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0])
    K = np.zeros([n, 9, 9], dtype=np.float64)
    if not _D2:
        _D2.extend(_d2_parts(g) for g in range(9))
    for g in range(9):
        sub, (P, P0, P1, P2) = _D2[g]
        prod = np.repeat(P, n, axis=0)
        prod += ecc[:, sub, None, None] * P0            # (the eccentricities rolled by sub)
        prod += ecc[:, (sub + 1) % 3, None, None] * P1
        prod += ecc[:, (sub + 2) % 3, None, None] * P2
        d2Skdksi2 = prod[:, (np.arange(9) - 3 * sub) % 9, :]
        d2Skdx2 = d2Skdksi2 @ H_rot
        K += _GAUSS_W[g] * (d2Skdx2 @ _E @ _transpose_vectorized(d2Skdx2))
    K = _transpose_vectorized(DOF_rot) @ K @ DOF_rot
    return _scalar_vectorized(area, K)


def _kff_and_ff(J, ecc, triangles, Uc):
    """matplotlib's get_Kff_and_Ff: the free-dof (nodal gradient) stiffness
    Kff in COO form, duplicates to be summed, in matplotlib's entry order,
    and the force Ff = -Kfc Uc of the values imposed at the nodes."""
    ntri = len(triangles)
    f_dof = [1, 2, 4, 5, 7, 8]
    c_dof = [0, 3, 6]
    t = triangles.astype(np.int32)
    f_idx = np.column_stack([t[:, 0] * 2, t[:, 0] * 2 + 1, t[:, 1] * 2, t[:, 1] * 2 + 1,
                             t[:, 2] * 2, t[:, 2] * 2 + 1])          # (ntri, 6)
    K_elem = _bending_matrices(J, ecc)
    vec_range = np.arange(ntri)
    Kff_vals = np.ravel(K_elem[np.ix_(vec_range, f_dof, f_dof)])
    Kff_rows = np.ravel(np.repeat(f_idx[:, :, None], 6, axis=2))
    Kff_cols = np.ravel(np.repeat(f_idx[:, None, :], 6, axis=1))
    Kfc_elem = K_elem[np.ix_(vec_range, f_dof, c_dof)]
    Ff_elem = -(Kfc_elem @ np.expand_dims(Uc, axis=2))[:, :, 0]
    Ff = np.bincount(np.ravel(f_idx), weights=np.ravel(Ff_elem))
    return Kff_rows, Kff_cols, Kff_vals, Ff


class _DOF_estimator:
    """The nodal gradient estimate (compute_dz) and the HCT dofs from it."""

    def __init__(self, interpolator, **kwargs):
        self._pts = interpolator._pts
        self._tris_pts = interpolator._tris_pts
        self.z = interpolator._z
        self._triangles = interpolator._triangles
        self._unit_x, self._unit_y = interpolator._unit_x, interpolator._unit_y
        self.dz = self.compute_dz(**kwargs)

    def compute_dof_from_df(self):
        J = _jacobian(self._tris_pts)
        tri_z = self.z[self._triangles]
        tri_dz = self.dz[self._triangles]
        return self.get_dof_vec(tri_z, tri_dz, J)

    @staticmethod
    def get_dof_vec(tri_z, tri_dz, J):
        """dof[3 i] = f(Ai), dof[3 i + 1] = df(Ai).AiAi+, dof[3 i + 2] =
        df(Ai).AiAi- for each apex i."""
        npt = tri_z.shape[0]
        dof = np.zeros([npt, 9], dtype=np.float64)
        J1 = _J0_TO_J1 @ J
        J2 = _J0_TO_J2 @ J
        col0 = J @ np.expand_dims(tri_dz[:, 0, :], axis=2)
        col1 = J1 @ np.expand_dims(tri_dz[:, 1, :], axis=2)
        col2 = J2 @ np.expand_dims(tri_dz[:, 2, :], axis=2)
        dfdksi = _to_matrix_vectorized([
            [col0[:, 0, 0], col1[:, 0, 0], col2[:, 0, 0]],
            [col0[:, 1, 0], col1[:, 1, 0], col2[:, 1, 0]]])
        dof[:, 0:7:3] = tri_z
        dof[:, 1:8:3] = dfdksi[:, 0]
        dof[:, 2:9:3] = dfdksi[:, 1]
        return dof


class _DOF_estimator_user(_DOF_estimator):
    """dz given by the caller (scaled to the unit box)."""

    def compute_dz(self, dz):
        dzdx, dzdy = dz
        dzdx = dzdx * self._unit_x
        dzdy = dzdy * self._unit_y
        return np.vstack([dzdx, dzdy]).T


class _DOF_estimator_geom(_DOF_estimator):
    """Each node's gradient as the angle-weighted mean of its triangles'
    plane gradients."""

    def compute_dz(self):
        el_geom_w = self.compute_geom_weights()
        el_geom_grad = self.compute_geom_grads()
        w_node_sum = np.bincount(np.ravel(self._triangles), weights=np.ravel(el_geom_w))
        dfx_el_w = np.empty_like(el_geom_w)
        dfy_el_w = np.empty_like(el_geom_w)
        for iapex in range(3):
            dfx_el_w[:, iapex] = el_geom_w[:, iapex] * el_geom_grad[:, 0]
            dfy_el_w[:, iapex] = el_geom_w[:, iapex] * el_geom_grad[:, 1]
        dfx_node_sum = np.bincount(np.ravel(self._triangles), weights=np.ravel(dfx_el_w))
        dfy_node_sum = np.bincount(np.ravel(self._triangles), weights=np.ravel(dfy_el_w))
        dfx_estim = dfx_node_sum / w_node_sum
        dfy_estim = dfy_node_sum / w_node_sum
        return np.vstack([dfx_estim, dfy_estim]).T

    def compute_geom_weights(self):
        """(ntri, 3): each apex angle / pi, folded at pi/2 (flat: 0)."""
        weights = np.zeros([np.size(self._triangles, 0), 3])
        tris_pts = self._tris_pts
        for ipt in range(3):
            p0 = tris_pts[:, ipt % 3, :]
            p1 = tris_pts[:, (ipt + 1) % 3, :]
            p2 = tris_pts[:, (ipt - 1) % 3, :]
            alpha1 = np.arctan2(p1[:, 1] - p0[:, 1], p1[:, 0] - p0[:, 0])
            alpha2 = np.arctan2(p2[:, 1] - p0[:, 1], p2[:, 0] - p0[:, 0])
            angle = np.abs(((alpha2 - alpha1) / np.pi) % 1)
            weights[:, ipt] = 0.5 - np.abs(angle - 0.5)
        return weights

    def compute_geom_grads(self):
        """(ntri, 2): each triangle's plane gradient (0 when flat)."""
        tris_pts = self._tris_pts
        tris_f = self.z[self._triangles]
        dM1 = tris_pts[:, 1, :] - tris_pts[:, 0, :]
        dM2 = tris_pts[:, 2, :] - tris_pts[:, 0, :]
        dM = np.dstack([dM1, dM2])
        dM_inv = _safe_inv22_vectorized(dM)
        dZ1 = tris_f[:, 1] - tris_f[:, 0]
        dZ2 = tris_f[:, 2] - tris_f[:, 0]
        dZ = np.vstack([dZ1, dZ2]).T
        df = np.empty_like(dZ)
        df[:, 0] = dZ[:, 0] * dM_inv[:, 0, 0] + dZ[:, 1] * dM_inv[:, 1, 0]
        df[:, 1] = dZ[:, 0] * dM_inv[:, 0, 1] + dZ[:, 1] * dM_inv[:, 1, 1]
        return df


class _DOF_estimator_min_E(_DOF_estimator_geom):
    """The gradients minimizing the bending energy, by a Jacobi-
    preconditioned conjugate gradient from the geom estimate."""

    def __init__(self, interpolator):
        self._eccs = interpolator._eccs
        super().__init__(interpolator)

    def compute_dz(self):
        dz_init = super().compute_dz()
        Uf0 = np.ravel(dz_init)
        J = _jacobian(self._tris_pts)
        Uc = self.z[self._triangles]
        Kff_rows, Kff_cols, Kff_vals, Ff = _kff_and_ff(J, self._eccs, self._triangles, Uc)
        tol = 1.e-10
        n_dof = Ff.shape[0]
        Kff_coo = _Sparse_Matrix_coo(Kff_vals, Kff_rows, Kff_cols, shape=(n_dof, n_dof))
        Kff_coo.compress_csc()
        Uf, err = _cg(A=Kff_coo, b=Ff, x0=Uf0, tol=tol)
        err0 = np.linalg.norm(Kff_coo.dot(Uf0) - Ff)
        if err0 < err:
            warnings.warn('In TriCubicInterpolator initialization, PCG sparse solver did not '
                          'converge after 1000 iterations. `geom` approximation is used '
                          'instead of `min_E`', stacklevel=4)
            Uf = Uf0
        dz = np.empty([self._pts.shape[0], 2], dtype=np.float64)
        dz[:, 0] = Uf[::2]
        dz[:, 1] = Uf[1::2]
        return dz


@njit(cache=True, nogil=True)
def _coo_dot(rows, cols, vals, V, m):
    """np.bincount(rows, weights=vals * V[cols], minlength=m), the same
    sums in the same order."""
    out = np.zeros(max(m, rows.max() + 1 if rows.size else 0))
    for i in range(rows.size):
        out[rows[i]] += vals[i] * V[cols[i]]
    return out


class _Sparse_Matrix_coo:
    """A COO sparse matrix: its product with a vector, its diagonal."""

    def __init__(self, vals, rows, cols, shape):
        self.n, self.m = shape
        self.vals = np.asarray(vals, dtype=np.float64)
        self.rows = np.asarray(rows, dtype=np.int32)
        self.cols = np.asarray(cols, dtype=np.int32)

    def dot(self, V):
        return _coo_dot(self.rows, self.cols, self.vals, V, self.m)

    def compress_csc(self):
        """Duplicates summed, entries sorted by column."""
        _, unique, indices = np.unique(self.rows + self.n * self.cols, return_index=True,
                                       return_inverse=True)
        self.rows = self.rows[unique]
        self.cols = self.cols[unique]
        self.vals = np.bincount(indices, weights=self.vals)

    @property
    def diag(self):
        in_diag = self.rows == self.cols
        diag = np.zeros(self.n, dtype=np.float64)
        diag[self.rows[in_diag]] = self.vals[in_diag]
        return diag


def _cg(A, b, x0=None, tol=1.e-10, maxiter=1000):
    """matplotlib's Jacobi-preconditioned conjugate gradient (after C. T.
    Kelley): x, and the residual norm |A x - b|."""
    n = b.size
    b_norm = np.linalg.norm(b)
    kvec = np.maximum(A.diag, 1e-6)
    x = np.zeros(n) if x0 is None else x0
    r = b - A.dot(x)
    w = r / kvec
    p = np.zeros(n)
    beta = 0.0
    rho = np.dot(r, w)
    k = 0
    while (np.sqrt(abs(rho)) > tol * b_norm) and (k < maxiter):
        p = w + beta * p
        z = A.dot(p)
        alpha = rho / np.dot(p, z)
        r = r - alpha * z
        w = r / kvec
        rhoold = rho
        rho = np.dot(r, w)
        x = x + alpha * p
        beta = rho / rhoold
        k += 1
    err = np.linalg.norm(A.dot(x) - b)
    return x, err


# matplotlib's (2, 2)-matrix helpers on (n, rows, cols) stacks. A flat
# triangle's rank-deficient matrices invert to 0 (no energy, no weight), or
# to the pseudo-inverse for barycentric coordinates.

def _safe_inv22_vectorized(M):
    M_inv = np.empty_like(M)
    prod1 = M[:, 0, 0] * M[:, 1, 1]
    delta = prod1 - M[:, 0, 1] * M[:, 1, 0]
    rank2 = np.abs(delta) > 1e-8 * np.abs(prod1)
    if np.all(rank2):
        delta_inv = 1. / delta
    else:
        delta_inv = np.zeros(M.shape[0])
        delta_inv[rank2] = 1. / delta[rank2]
    M_inv[:, 0, 0] = M[:, 1, 1] * delta_inv
    M_inv[:, 0, 1] = -M[:, 0, 1] * delta_inv
    M_inv[:, 1, 0] = -M[:, 1, 0] * delta_inv
    M_inv[:, 1, 1] = M[:, 0, 0] * delta_inv
    return M_inv


def _scalar_vectorized(scalar, M):
    return scalar[:, np.newaxis, np.newaxis] * M


def _transpose_vectorized(M):
    return np.transpose(M, [0, 2, 1])


def _roll_vectorized(M, roll_indices, axis):
    """Each matrix of M rolled along axis (0: rows) by its own index."""
    sh = M.shape
    r, c = sh[-2:]
    vec_indices = np.arange(sh[0], dtype=np.int32)
    M_roll = np.empty_like(M)
    for ir in range(r):
        for ic in range(c):
            if axis == 0:
                M_roll[:, ir, ic] = M[vec_indices, (-roll_indices + ir) % r, ic]
            else:
                M_roll[:, ir, ic] = M[vec_indices, ir, (-roll_indices + ic) % c]
    return M_roll


def _to_matrix_vectorized(M):
    """The stack M_res[..., i, j] = M[i][j] from nested lists of arrays."""
    r, c = len(M), len(M[0])
    M00 = np.asarray(M[0][0])
    M_ret = np.empty([M00.shape[0], r, c], dtype=M00.dtype)
    for irow in range(r):
        for icol in range(c):
            M_ret[:, irow, icol] = np.asarray(M[irow][icol])
    return M_ret


def _extract_submatrices(M, block_indices, block_size, axis):
    """M_res[..., ir, :] = M[block_indices * block_size + ir, :] (axis 0)."""
    r, c = M.shape
    sh = [block_indices.shape[0], block_size, c] if axis == 0 else \
        [block_indices.shape[0], r, block_size]
    M_res = np.empty(sh, dtype=M.dtype)
    for i in range(block_size):
        if axis == 0:
            M_res[:, i, :] = M[(block_indices * block_size + i), :]
        else:
            M_res[:, :, i] = M[:, (block_indices * block_size + i)]
    return M_res


# ---------------------------------------------------------------- analyzer

class TriAnalyzer:
    """matplotlib.tri.TriAnalyzer: mesh quality (circle ratios) and the
    mask that trims a Delaunay mesh's flat border triangles."""

    def __init__(self, triangulation):
        if not isinstance(triangulation, Triangulation):
            raise TypeError('triangulation must be a Triangulation')
        self._triangulation = triangulation

    @property
    def scale_factors(self):
        """(kx, ky) that fit the unmasked triangles into a unit square."""
        compressed_triangles = self._triangulation.get_masked_triangles()
        node_used = np.bincount(np.ravel(compressed_triangles),
                                minlength=self._triangulation.x.size) != 0
        return (1 / np.ptp(self._triangulation.x[node_used]),
                1 / np.ptp(self._triangulation.y[node_used]))

    def circle_ratios(self, rescale=True):
        """Each triangle's incircle over circumcircle radius (0.5 for an
        equilateral one, < 0.01 very flat), after the unit-square rescaling
        unless rescale=False; masked where the triangles are."""
        kx, ky = self.scale_factors if rescale else (1.0, 1.0)
        pts = np.vstack([self._triangulation.x * kx, self._triangulation.y * ky]).T
        tri_pts = pts[self._triangulation.triangles]
        a = tri_pts[:, 1, :] - tri_pts[:, 0, :]
        b = tri_pts[:, 2, :] - tri_pts[:, 1, :]
        c = tri_pts[:, 0, :] - tri_pts[:, 2, :]
        a = np.hypot(a[:, 0], a[:, 1])
        b = np.hypot(b[:, 0], b[:, 1])
        c = np.hypot(c[:, 0], c[:, 1])
        s = (a + b + c) * 0.5
        prod = s * (a + b - s) * (a + c - s) * (b + c - s)
        bool_flat = prod == 0.
        if np.any(bool_flat):                           # flat: an infinite circumcircle
            circum_radius = np.empty(tri_pts.shape[0], dtype=np.float64)
            circum_radius[bool_flat] = np.inf
            abc = a * b * c
            circum_radius[~bool_flat] = abc[~bool_flat] / (4.0 * np.sqrt(prod[~bool_flat]))
        else:
            circum_radius = (a * b * c) / (4.0 * np.sqrt(prod))
        in_radius = (a * b * c) / (4.0 * circum_radius * s)
        circle_ratio = in_radius / circum_radius
        mask = self._triangulation.mask
        if mask is None:
            return circle_ratio
        return np.ma.array(circle_ratio, mask=mask)

    def get_flat_tri_mask(self, min_circle_ratio=0.01, rescale=True):
        """The triangulation's mask extended by its flat border triangles
        (circle ratio < min_circle_ratio), peeled from the border inward so
        that no holes open: for Triangulation.set_mask."""
        ntri = self._triangulation.triangles.shape[0]
        mask_bad_ratio = self.circle_ratios(rescale) < min_circle_ratio
        current_mask = self._triangulation.mask
        if current_mask is None:
            current_mask = np.zeros(ntri, dtype=bool)
        valid_neighbors = np.copy(neighbors(self._triangulation))
        renum_neighbors = np.arange(ntri, dtype=np.int32)
        nadd = -1
        while nadd != 0:
            wavefront = (np.min(valid_neighbors, axis=1) == -1) & ~current_mask
            added_mask = wavefront & mask_bad_ratio
            current_mask = added_mask | current_mask
            nadd = np.sum(added_mask)
            valid_neighbors[added_mask, :] = -1
            renum_neighbors[added_mask] = -1
            valid_neighbors = np.where(valid_neighbors == -1, -1,
                                       renum_neighbors[valid_neighbors])
        return np.ma.filled(current_mask, True)

    def _get_compressed_triangulation(self):
        """The unmasked triangles over the points they use, renumbered:
        (triangles, x, y, triangle renumbering, node renumbering; -1 for
        the dropped)."""
        tri_mask = self._triangulation.mask
        compressed_triangles = self._triangulation.get_masked_triangles()
        ntri = self._triangulation.triangles.shape[0]
        if tri_mask is not None:
            tri_renum = self._total_to_compress_renum(~tri_mask)
        else:
            tri_renum = np.arange(ntri, dtype=np.int32)
        valid_node = np.bincount(np.ravel(compressed_triangles),
                                 minlength=self._triangulation.x.size) != 0
        compressed_x = self._triangulation.x[valid_node]
        compressed_y = self._triangulation.y[valid_node]
        node_renum = self._total_to_compress_renum(valid_node)
        compressed_triangles = node_renum[compressed_triangles]
        return compressed_triangles, compressed_x, compressed_y, tri_renum, node_renum

    @staticmethod
    def _total_to_compress_renum(valid):
        renum = np.full(np.size(valid), -1, dtype=np.int32)
        n_valid = np.sum(valid)
        renum[valid] = np.arange(n_valid, dtype=np.int32)
        return renum
