"""Value rasters: a gridded field resampled onto the map's pixels, float32,
NaN off the data (the CPU side; _gpu has the same kernels in CUDA).

    ax = Axis(coords, periodic=..., angular=..., edges=...)    # a 1-D grid axis
    sample_rect(out, view, src, xaxis, yaxis, z, mode)          # 1-D axes: per pixel
    forward_nodes(view, src, x2d, y2d) -> sx, sy                # 2-D coords: nodes to px
    raster_cells(out, sx, sy, z, mode, period_px)               # ... and their cells

`view` is the map: its projection (kind, params) and the affine from
pixels of the raster to projected x/y, x = x0 + (j + 1/2) dx, y = y0 +
(i + 1/2) dy (dy < 0: row 0 at the top). `src` is the projection the
grid's coordinates are in (PlateCarree for lon/lat).

Grids with 1-D coordinate axes (GFS, ERA5, any lat-lon or projected
rectilinear grid) are sampled per pixel: the pixel's x/y back to lon/lat
(the map's inverse), into the grid's coordinates (the source's forward),
to a fractional index along each axis -- a division on a uniform axis, a
binary search otherwise -- and the value: the nearest cell (pcolormesh,
imshow) or bilinear between the four around it (contourf: the bands are
then the level bins of a continuous surface, as matplotlib's marching
squares draws them). Angular axes (longitude) wrap, so a 0..360 grid
lands on a -180..180 map and a global one joins across its seam.

On a map that is not separable (Lambert, polar, Robinson ...) the two
projections are most of the cost -- ~4 ms for an 1800x975 Lambert map --
and depend on the map alone, not on the field. A map drawn again (the
frames of a forecast: one map, the next hour's field) gets them from a
table instead: per pixel the grid-coordinate point it lands on, made the
second time the map is seen (TABLE_KEEP maps kept, at most TABLE_BYTES
in all) and read from then on -- the same float64 numbers, the same
image.

Grids with 2-D coordinates (WRF, HRRR, rotated or curvilinear grids) are
drawn forward: every node projected to the raster once (kept_nodes: a
large grid's kept from the second time it is drawn, for as long as its
coordinates stay the same), every cell split
into two triangles rasterized with their values interpolated (bilinear
mode), the value of the node nearest in the cell's own parameter space
(nearest mode: the cell around each node, its sides half-way to the next
nodes, as matplotlib's pcolormesh(shading='nearest') builds them), or the cell's own value (flat mode: the nodes are cell corners,
pcolormesh(shading='flat')). On a cylindrical map a cell crossing the seam is unwrapped and drawn on both
sides.
"""
import threading
import weakref
from collections import OrderedDict

import numpy as np

from . import _arena, _jit
from . import _proj as P
from ._jit import njit, prange

TABLE_KEEP = 2                  # maps whose pixel -> source tables are kept
TABLE_BYTES = 128 << 20         # and at most this much memory in all
_TABLES = OrderedDict()         # map key -> (sx, sy, ok): the newest last
_SEEN = OrderedDict()           # map keys drawn once (a table is made the second time)
_TABLE_LOCK = threading.Lock()

CELL_BANDS = 4                  # bands of raster rows a thread (_raster_cells)
NEAREST, BILINEAR, FLAT, IMAGE = 0, 1, 2, 3  # FLAT: 2-D cells given by their corners;
# IMAGE: bilinear out to the outer cells' edges, held there (matplotlib's images)


class Axis:
    """A grid axis prepared for the kernels: increasing coordinates, and
    whether the data has to be flipped to match. edges=True when coords
    are the n+1 cell edges (pcolormesh 'flat') rather than n centres."""

    def __init__(self, coords, angular=False, edges=False):
        c = np.asarray(coords, np.float64).ravel()
        if angular and c.size > 1:                      # each step the short way round: an
            step = (np.diff(c) + 180.0) % 360.0 - 180.0  # axis through 360 (ECMWF's 180..359.75,
            self.flip = bool(step.sum() < 0)            # 0..179.75) still increases
        else:
            self.flip = c.size > 1 and c[-1] < c[0]
        if self.flip:
            c = c[::-1]
        if angular:                                     # make longitudes increase through 360
            c = c[0] + np.concatenate([[0.0], np.cumsum(np.mod(np.diff(c), 360.0))])
        self.coords = np.ascontiguousarray(c)
        self.edges = bool(edges)
        self.angular = bool(angular)
        d = np.diff(c)
        self.uniform = c.size < 3 or bool(np.allclose(d, d[0], rtol=1e-5, atol=0.0))
        self.x0 = c[0]
        self.dx = d[0] if c.size > 1 else 1.0
        n = c.size - 1 if edges else c.size
        self.n = n
        # a full circle of longitude: the last cell joins the first
        span = (c[-1] - c[0]) if edges else (c[-1] - c[0] + (d.mean() if d.size else 0))
        self.periodic = angular and abs(span - 360.0) < 1e-3 * max(abs(self.dx), 1e-9) + 1e-6
        self.lo = self.x0 - (0.0 if edges else 0.5 * self.dx)

    def params(self):
        """float64 [x0, dx, n, uniform, edges, periodic, angular, lo]."""
        return np.array([self.x0, self.dx, self.n, self.uniform, self.edges,
                         self.periodic, self.angular, self.lo], np.float64)


@njit(cache=True, nogil=True, inline='always')
def _frac(u, ap, coords):
    """Fractional index of coordinate u on an axis (params ap, coords):
    centres at 0..n-1 (edges at 0..n); NaN off the axis."""
    x0, dx, n = ap[0], ap[1], ap[2]
    uniform, edges, periodic, angular, lo = ap[3] != 0, ap[4] != 0, ap[5] != 0, ap[6] != 0, ap[7]
    if angular:                                         # into [lo, lo + 360)
        u = lo + ((u - lo) % 360.0)
    if uniform:
        f = (u - x0) / dx
    else:
        m = coords.size
        if u <= coords[0]:
            f = (u - coords[0]) / (coords[1] - coords[0])
        elif u >= coords[m - 1]:
            f = (m - 1) + (u - coords[m - 1]) / (coords[m - 1] - coords[m - 2])
        else:
            k = np.searchsorted(coords, u, side='right') - 1
            f = k + (u - coords[k]) / (coords[k + 1] - coords[k])
    if periodic:
        return f
    if edges:
        if f < 0.0 or f >= n:
            return np.nan
    elif f < -0.5 or f >= n - 0.5:
        return np.nan
    return f


@njit(cache=True, nogil=True, inline='always')
def _wrap(k, n):
    k = k % n
    return k + n if k < 0 else k


@njit(cache=True, nogil=True, inline='always')
def _axis(f, n, per, edge, mode):
    """A fractional index f on an axis of n nodes -> the nodes (k0, k1)
    to read, the weight t of k1, and whether there is a value at all."""
    if not np.isfinite(f):
        return np.int64(0), np.int64(0), 0.0, False
    if mode == NEAREST:
        k = np.int64(np.floor(f)) if edge else np.int64(np.floor(f + 0.5))
        if per:
            k = _wrap(k, n)
        return k, k, 0.0, True
    # bilinear: between the first and last nodes only, as contourf fills;
    # an image's to its edges, half a cell further, its end values held
    if not per and mode != IMAGE and (f < 0.0 or f > n - 1):
        return np.int64(0), np.int64(0), 0.0, False
    k0 = np.int64(np.floor(f))
    t = f - k0
    k1 = k0 + 1
    if per:
        k0, k1 = _wrap(k0, n), _wrap(k1, n)
    else:                                               # clamp the half cell at either end
        if k0 < 0:
            k0, t = 0, 0.0
        if k1 > n - 1:
            k1, t = n - 1, 0.0
            k0 = min(k0, n - 1)
    return k0, k1, t, True


@njit(cache=True, nogil=True, inline='always')
def _value(z, ky0, ky1, ty, kx0, kx1, tx, mode):
    if mode == NEAREST:
        return z[ky0, kx0]
    v00, v01 = np.float64(z[ky0, kx0]), np.float64(z[ky0, kx1])
    v10, v11 = np.float64(z[ky1, kx0]), np.float64(z[ky1, kx1])
    v = (v00 * (1.0 - tx) + v01 * tx) * (1.0 - ty) + (v10 * (1.0 - tx) + v11 * tx) * ty
    return np.float32(v)                                # NaN if any corner is


@njit(cache=True, nogil=True, parallel=True)
def _sample_rect(out, mkind, mp, x0, dx, y0, dy, skind, sp, xp, xc, yp, yc, z, mode):
    H, W = out.shape
    nxv, nyv = np.int64(xp[2]), np.int64(yp[2])
    xper, yper = xp[5] != 0, yp[5] != 0
    xedge, yedge = xp[4] != 0, yp[4] != 0
    nan = np.float32(np.nan)
    for i in prange(H):
        Y = y0 + (i + 0.5) * dy
        for j in range(W):
            X = x0 + (j + 0.5) * dx
            lon, lat = P.inv(mkind, mp, X, Y)
            if not (np.isfinite(lon) and np.isfinite(lat)):
                out[i, j] = nan
                continue
            sx, sy = P.fwd(skind, sp, lon, lat)
            kx0, kx1, tx, okx = _axis(_frac(sx, xp, xc), nxv, xper, xedge, mode)
            ky0, ky1, ty, oky = _axis(_frac(sy, yp, yc), nyv, yper, yedge, mode)
            out[i, j] = _value(z, ky0, ky1, ty, kx0, kx1, tx, mode) if (okx and oky) else nan


@njit(cache=True, nogil=True, parallel=True)
def _pixel_src(mkind, mp, x0, dx, y0, dy, skind, sp, sx, sy, ok):
    """_sample_rect's projections alone: each pixel's point in the
    source's coordinates (sx, sy), ok where the map's inverse has one."""
    H, W = sx.shape
    for i in prange(H):
        Y = y0 + (i + 0.5) * dy
        for j in range(W):
            X = x0 + (j + 0.5) * dx
            lon, lat = P.inv(mkind, mp, X, Y)
            if not (np.isfinite(lon) and np.isfinite(lat)):
                ok[i, j] = False
                continue
            ok[i, j] = True
            sx[i, j], sy[i, j] = P.fwd(skind, sp, lon, lat)


@njit(cache=True, nogil=True, parallel=True)
def _sample_table(out, sx, sy, ok, xp, xc, yp, yc, z, mode):
    """_sample_rect from a _pixel_src table: the rest of it, as it is."""
    H, W = out.shape
    nxv, nyv = np.int64(xp[2]), np.int64(yp[2])
    xper, yper = xp[5] != 0, yp[5] != 0
    xedge, yedge = xp[4] != 0, yp[4] != 0
    nan = np.float32(np.nan)
    for i in prange(H):
        for j in range(W):
            if not ok[i, j]:
                out[i, j] = nan
                continue
            kx0, kx1, tx, okx = _axis(_frac(sx[i, j], xp, xc), nxv, xper, xedge, mode)
            ky0, ky1, ty, oky = _axis(_frac(sy[i, j], yp, yc), nyv, yper, yedge, mode)
            out[i, j] = _value(z, ky0, ky1, ty, kx0, kx1, tx, mode) if (okx and oky) else nan


def _table(view, src, H, W):
    """The map's pixel -> source table, if it has one (made now if the
    map was drawn once before), else None."""
    key = (view.kind, np.asarray(view.params, np.float64).tobytes(), float(view.x0),
           float(view.dx), float(view.y0), float(view.dy), src.kind,
           np.asarray(src.params, np.float64).tobytes(), H, W)
    size = H * W * 17
    with _TABLE_LOCK:
        t = _TABLES.get(key)
        if t is not None:
            _TABLES.move_to_end(key)
            return t
        if key not in _SEEN or size > TABLE_BYTES:
            _SEEN[key] = True
            _SEEN.move_to_end(key)
            while len(_SEEN) > 16:
                _SEEN.popitem(last=False)
            return None
    sx, sy = np.empty((H, W)), np.empty((H, W))
    ok = np.empty((H, W), np.bool_)
    _pixel_src(view.kind, view.params, view.x0, view.dx, view.y0, view.dy, src.kind, src.params,
               sx, sy, ok)
    with _TABLE_LOCK:
        _TABLES[key] = (sx, sy, ok)
        while len(_TABLES) > TABLE_KEEP or (
                len(_TABLES) > 1 and sum(v[0].size for v in _TABLES.values()) * 17 > TABLE_BYTES):
            _TABLES.popitem(last=False)
    return sx, sy, ok


def clear_tables():
    """Every kept pixel -> source table and grid's nodes let go."""
    with _TABLE_LOCK:
        _TABLES.clear()
        _SEEN.clear()
    with _KEPT_LOCK:
        _KEPT.clear()
    with _CELL_LOCK:
        _CELL_TABLES.clear()


@njit(cache=True, nogil=True)
def _sep_axes(mkind, mp, x0, dx, y0, dy, skind, sp, xp, xc, yp, yc, mode, W, H, cols, rows):
    """Per column and per row of a separable map (see sample_rect): the
    nodes, weight and validity _axis gives, as (k0, k1, t, ok) in cols
    (W, 4) and rows (H, 4) float64. False if no probed pixel lies inside
    the map's domain (the general kernel then draws it, pixel by pixel)."""
    jv = -1
    iv = -1
    for i in range(H):                                  # a row and a column inside the map
        Y = y0 + (i + 0.5) * dy
        for j in (W // 2, 0, W - 1):
            lon, lat = P.inv(mkind, mp, x0 + (j + 0.5) * dx, Y)
            if np.isfinite(lon) and np.isfinite(lat):
                iv, jv = i, j
                break
        if iv >= 0:
            break
    if iv < 0:
        return False
    Xv = x0 + (jv + 0.5) * dx
    Yv = y0 + (iv + 0.5) * dy
    nxv, nyv = np.int64(xp[2]), np.int64(yp[2])
    for j in range(W):
        lon, lat = P.inv(mkind, mp, x0 + (j + 0.5) * dx, Yv)
        ok = np.isfinite(lon) and np.isfinite(lat)
        k0, k1, t, okx = _axis(_frac(P.fwd(skind, sp, lon, lat)[0], xp, xc), nxv, xp[5] != 0,
                               xp[4] != 0, mode) if ok else (np.int64(0), np.int64(0), 0.0, False)
        cols[j, 0], cols[j, 1], cols[j, 2], cols[j, 3] = k0, k1, t, okx
    for i in range(H):
        lon, lat = P.inv(mkind, mp, Xv, y0 + (i + 0.5) * dy)
        ok = np.isfinite(lon) and np.isfinite(lat)
        k0, k1, t, oky = _axis(_frac(P.fwd(skind, sp, lon, lat)[1], yp, yc), nyv, yp[5] != 0,
                               yp[4] != 0, mode) if ok else (np.int64(0), np.int64(0), 0.0, False)
        rows[i, 0], rows[i, 1], rows[i, 2], rows[i, 3] = k0, k1, t, oky
    return True


@njit(cache=True, nogil=True, parallel=True)
def _sample_sep(out, cols, rows, z, mode):
    H, W = out.shape
    nan = np.float32(np.nan)
    for i in prange(H):
        if rows[i, 3] == 0.0:
            for j in range(W):
                out[i, j] = nan
            continue
        ky0, ky1, ty = np.int64(rows[i, 0]), np.int64(rows[i, 1]), rows[i, 2]
        for j in range(W):
            if cols[j, 3] == 0.0:
                out[i, j] = nan
            else:
                out[i, j] = _value(z, ky0, ky1, ty, np.int64(cols[j, 0]), np.int64(cols[j, 1]),
                                   cols[j, 2], mode)


@njit(cache=True, nogil=True)
def _axis_table(values, ap, coords, mode, out):
    for k in range(values.size):
        k0, k1, t, ok = _axis(_frac(values[k], ap, coords), np.int64(ap[2]), ap[5] != 0,
                              ap[4] != 0, mode)
        out[k, 0], out[k, 1], out[k, 2], out[k, 3] = k0, k1, t, ok


def axis_table(values, axis, mode):
    """Per data coordinate along a grid axis: (k0, k1, t, ok) as _axis gives
    them -- a column or row table for _sample_sep (plain axes, where each
    pixel column has one data x)."""
    v = np.ascontiguousarray(values, dtype=np.float64)
    out = np.empty((v.size, 4), np.float64)
    if v.size:
        _axis_table(v, axis.params(), axis.coords, mode, out)
    return out


def sample_sep(out, cols, rows, z, xaxis, yaxis, mode):
    """Fill out (H, W) from per-column and per-row tables (axis_table)."""
    zz = z
    if yaxis.flip:
        zz = zz[::-1]
    if xaxis.flip:
        zz = zz[:, ::-1]
    _sample_sep(out, cols, rows, zz, mode)


# map kinds whose inverse gives lon from x alone and lat from y alone (and
# whose domain is a rectangle), and source kinds whose forward gives x from
# lon alone and y from lat alone: a pixel's grid index is then its column's
# along x and its row's along y -- computed W + H times, not W * H
_SEP_MAP = (P.GEODETIC, P.PLATE_CARREE, P.MERCATOR, P.MILLER)
_SEP_SRC = (P.GEODETIC, P.PLATE_CARREE)


def sample_rect(out, view, src, xaxis, yaxis, z, mode):
    """Fill the raster out (H, W) float32 from z (ny, nx) on 1-D axes."""
    zz = z
    if yaxis.flip:
        zz = zz[::-1]
    if xaxis.flip:
        zz = zz[:, ::-1]
    args = (view.kind, view.params, view.x0, view.dx, view.y0, view.dy, src.kind, src.params,
            xaxis.params(), xaxis.coords, yaxis.params(), yaxis.coords)
    if view.kind in _SEP_MAP and src.kind in _SEP_SRC:
        H, W = out.shape
        cols = np.empty((W, 4), np.float64)
        rows = np.empty((H, 4), np.float64)
        if _sep_axes(*args, mode, W, H, cols, rows):
            _sample_sep(out, cols, rows, zz, mode)
            return
    t = _table(view, src, *out.shape)
    if t is not None:
        _sample_table(out, *t, xaxis.params(), xaxis.coords, yaxis.params(), yaxis.coords, zz,
                      mode)
        return
    _sample_rect(out, *args, zz, mode)


# ---------------------------------------------------------- 2-D coordinates

@njit(cache=True, nogil=True, parallel=True)
def _forward_nodes(mkind, mp, x0, dx, y0, dy, skind, sp, xs, ys, sx, sy):
    ny, nx = xs.shape
    for i in prange(ny):
        for j in range(nx):
            lon, lat = P.inv(skind, sp, xs[i, j], ys[i, j])
            X, Y = P.fwd(mkind, mp, lon, lat)
            sx[i, j] = (X - x0) / dx - 0.5
            sy[i, j] = (Y - y0) / dy - 0.5


def forward_nodes(view, src, x2d, y2d):
    """Nodes with 2-D coordinates (in src) -> raster pixel coordinates
    (column, row: pixel (i, j)'s centre is (j, i) here)."""
    xs = np.ascontiguousarray(x2d, dtype=np.float64)
    ys = np.ascontiguousarray(y2d, dtype=np.float64)
    sx = np.empty(xs.shape, np.float64)
    sy = np.empty(xs.shape, np.float64)
    _forward_nodes(view.kind, view.params, view.x0, view.dx, view.y0, view.dy,
                   src.kind, src.params, xs, ys, sx, sy)
    return sx, sy


# A forecast's frames draw new values on one grid in one view, and an HRRR
# grid's projection is ~20 ms of ~60 of a frame. A large grid's nodes are
# kept once it has come twice, and used when the coordinates are the same to
# the bit, which costs ~1 ms: nothing is assumed about arrays that look
# alike, and a one-off figure neither copies nor keeps anything.
KEEP_NODES = 1 << 12            # a grid of fewer nodes is not worth comparing
KEEP_BYTES = 512 << 20          # at most this much is kept in all (32 bytes a node)
_KEPT = OrderedDict()           # view and grid: None (seen once) or (xs, ys, sx, sy)
_KEPT_LOCK = threading.Lock()
_KEPT_MAX = 4


@njit(cache=True, nogil=True, parallel=True)
def _bits_differ(a, b):
    """How many rows of two (uint64 views of) arrays hold different bits."""
    n = 0
    for i in prange(a.shape[0]):
        d = np.uint64(0)
        for j in range(a.shape[1]):
            d |= a[i, j] ^ b[i, j]
        n += 1 if d != 0 else 0
    return n


def kept_nodes(view, src, x2d, y2d):
    """forward_nodes, of arrays only to read, for a large grid seen before
    the arrays kept."""
    xs = np.ascontiguousarray(x2d, dtype=np.float64)
    ys = np.ascontiguousarray(y2d, dtype=np.float64)
    if xs.size < KEEP_NODES or xs.size * 32 > KEEP_BYTES:
        return forward_nodes(view, src, xs, ys)
    key = (view.kind, view.params.tobytes(), view.x0, view.dx, view.y0, view.dy,
           src.kind, src.params.tobytes(), xs.shape)
    with _KEPT_LOCK:
        seen = key in _KEPT
        old = _KEPT.get(key)
        if seen:
            _KEPT.move_to_end(key)
    if old is not None and _bits_differ(xs.view(np.uint64), old[0].view(np.uint64)) == 0 and \
            _bits_differ(ys.view(np.uint64), old[1].view(np.uint64)) == 0:
        return old[2], old[3]
    sx, sy = forward_nodes(view, src, xs, ys)
    keep = (xs.copy(), ys.copy(), sx, sy) if seen and old is None else None
    with _KEPT_LOCK:
        _KEPT[key] = keep
        while len(_KEPT) > _KEPT_MAX or \
                sum(v[0].size for v in _KEPT.values() if v) * 32 > KEEP_BYTES:
            _KEPT.popitem(last=False)
    return sx, sy


@njit(cache=True, nogil=True)
def _tri(out, ax, ay, bx, by, cx, cy, za, zb, zc, mode, kind, zn, r0, r1, W):
    """One triangle into the raster, rows r0..r1. Bilinear mode: values za,
    zb, zc interpolated. Nearest mode: zn holds the cell's four node values
    (z00, z01, z11, z10) and kind 0/1 says which half of the cell this is."""
    area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if area == 0.0 or not np.isfinite(area):
        return
    j0 = max(0, np.int64(np.ceil(min(ax, min(bx, cx)))))
    j1 = min(W - 1, np.int64(np.floor(max(ax, max(bx, cx)))))
    i0 = max(r0, np.int64(np.ceil(min(ay, min(by, cy)))))
    i1 = min(r1 - 1, np.int64(np.floor(max(ay, max(by, cy)))))
    inv = 1.0 / area
    for i in range(i0, i1 + 1):
        py = np.float64(i)
        for j in range(j0, j1 + 1):
            px = np.float64(j)
            w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) * inv
            w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) * inv
            w2 = 1.0 - w0 - w1
            if w0 < -1e-12 or w1 < -1e-12 or w2 < -1e-12:
                continue
            if mode == BILINEAR:
                out[i, j] = np.float32(w0 * za + w1 * zb + w2 * zc)
            else:
                if kind == 0:                           # a=00, b=01, c=11
                    s, t = w1 + w2, w2
                else:                                   # a=00, b=11, c=10
                    s, t = w1, w1 + w2
                q = (1 if s >= 0.5 else 0) + (2 if t >= 0.5 else 0)
                out[i, j] = zn[(0, 1, 3, 2)[q]]         # 00, 01, 10, 11


@njit(cache=True, nogil=True, parallel=True)
def _raster_cells(out, sx, sy, z, mode, period, tasks, bands):
    """Rows of the raster in bands, each band on one core taking the cells
    in order: a pixel two cells share (on their common edge, or where the
    mesh folds) ends up the later cell's however many threads draw it, as
    Agg paints them. Rows of cells on cores of their own raced for it."""
    H = out.shape[0]
    ny = sx.shape[0]
    lo, hi = np.empty(ny), np.empty(ny)
    for i in prange(ny - 1):
        a, b = np.inf, -np.inf
        for j in range(sx.shape[1]):
            for y in (sy[i, j], sy[i + 1, j]):
                if y < a:
                    a = y
                if y > b:
                    b = y
        lo[i], hi[i] = a, b
    for t in prange(tasks):
        for k in range(t, bands, tasks):
            _cells_band(out, sx, sy, z, mode, period, lo, hi, H * k // bands, H * (k + 1) // bands)


@njit(cache=True, nogil=True)
def _cells_band(out, sx, sy, z, mode, period, lo, hi, r0, r1):
    """The rows r0..r1 of the raster, from every cell in order."""
    W = out.shape[1]
    ny, nx = sx.shape
    zn = np.empty(4, np.float32)
    for i in range(ny - 1):
        if hi[i] < r0 - 1.0 or lo[i] > r1:              # (no cell of the row: nodes of NaN too)
            continue
        for j in range(nx - 1):
            x00, y00 = sx[i, j], sy[i, j]
            x01, y01 = sx[i, j + 1], sy[i, j + 1]
            x11, y11 = sx[i + 1, j + 1], sy[i + 1, j + 1]
            x10, y10 = sx[i + 1, j], sy[i + 1, j]
            if not (np.isfinite(x00) and np.isfinite(x01) and np.isfinite(x11)
                    and np.isfinite(x10)):
                continue
            if period > 0.0:                            # unwrap the cell around its first node
                h = 0.5 * period
                if x01 - x00 > h:
                    x01 -= period
                elif x01 - x00 < -h:
                    x01 += period
                if x11 - x00 > h:
                    x11 -= period
                elif x11 - x00 < -h:
                    x11 += period
                if x10 - x00 > h:
                    x10 -= period
                elif x10 - x00 < -h:
                    x10 += period
            left = min(min(x00, x01), min(x11, x10))
            right = max(max(x00, x01), max(x11, x10))
            if right < -1.0 and period == 0.0 or left > W and period == 0.0:
                continue
            top = min(min(y00, y01), min(y11, y10))
            bot = max(max(y00, y01), max(y11, y10))
            if bot < r0 - 1.0 or top > r1:
                continue
            if period == 0.0 and (right - left < 1.0 and np.ceil(left) > np.floor(right) or
                                  bot - top < 1.0 and np.ceil(top) > np.floor(bot)):
                continue                                # no pixel centre (a cell under a pixel)
            if mode == FLAT:                            # z holds the cells' own values
                z00 = z01 = z11 = z10 = z[i, j]
            else:
                z00, z01 = z[i, j], z[i, j + 1]
                z11, z10 = z[i + 1, j + 1], z[i + 1, j]
            zn[0], zn[1], zn[2], zn[3] = z00, z01, z11, z10
            nshift = 1
            shifts = (0.0, 0.0, 0.0)
            if period > 0.0:
                if left < 0.0:
                    shifts = (0.0, period, 0.0)
                    nshift = 2
                elif right > W - 1:
                    shifts = (0.0, -period, 0.0)
                    nshift = 2
            for q in range(nshift):
                o = shifts[q]
                tm = BILINEAR if mode == FLAT or mode == IMAGE else mode
                _tri(out, x00 + o, y00, x01 + o, y01, x11 + o, y11, z00, z01, z11, tm, 0, zn,
                     r0, r1, W)
                _tri(out, x00 + o, y00, x11 + o, y11, x10 + o, y10, z00, z11, z10, tm, 1, zn,
                     r0, r1, W)


def _run_cells(out, sx, sy, z, mode, period):
    team = _jit.team()
    bands = min(CELL_BANDS * team, max(out.shape[0], 1))
    _raster_cells(out, sx, sy, z, mode, period, min(team, bands), bands)


def raster_cells(out, sx, sy, z, mode, period_px=0.0):
    """Cells of a grid whose nodes are at raster pixels (sx, sy) into out;
    period_px > 0 on a cylindrical map (the width of 360 degrees)."""
    _run_cells(out, sx, sy, np.ascontiguousarray(z, dtype=np.float32), mode, float(period_px))


@njit(cache=True, nogil=True)
def square_cover(hx, hy, n, px, py, ax, ay, bx, by):
    """The area of the unit square at (px, py) inside the convex polygon
    (hx, hy) of n corners, positive area (y down): the square clipped by
    each side in turn. ax, ay, bx, by: working room."""
    ax[0], ay[0], ax[1], ay[1] = px, py, px + 1.0, py
    ax[2], ay[2], ax[3], ay[3] = px + 1.0, py + 1.0, px, py + 1.0
    m = 4
    for e in range(n):
        x0, y0 = hx[e], hy[e]
        ex, ey = hx[(e + 1) % n] - x0, hy[(e + 1) % n] - y0
        k = 0
        for q in range(m):
            r = (q + 1) % m
            d0 = ex * (ay[q] - y0) - ey * (ax[q] - x0)
            d1 = ex * (ay[r] - y0) - ey * (ax[r] - x0)
            if d0 >= 0.0:
                bx[k], by[k] = ax[q], ay[q]
                k += 1
            if (d0 >= 0.0) != (d1 >= 0.0):
                t = d0 / (d0 - d1)
                bx[k], by[k] = ax[q] + t * (ax[r] - ax[q]), ay[q] + t * (ay[r] - ay[q])
                k += 1
        m = k
        if m == 0:
            return 0.0
        for q in range(m):
            ax[q], ay[q] = bx[q], by[q]
    s = 0.0
    for q in range(m):
        r = (q + 1) % m
        s += ax[q] * ay[r] - ax[r] * ay[q]
    return 0.5 * abs(s)


TOUCHED_LEAN = 0.05                                     # about 3 degrees off the pixel grid


def _leans(bx, by):
    """Whether a cell's two sides (its 2 x 2 corners) run off the pixel
    grid: more of their length across it than TOUCHED_LEAN."""
    ax, ay = abs(bx[0, 1] - bx[0, 0]), abs(by[0, 1] - by[0, 0])
    dx, dy = abs(bx[1, 0] - bx[0, 0]), abs(by[1, 0] - by[0, 0])
    if not (np.isfinite(ax + ay + dx + dy) and ax + ay > 0.0 and dx + dy > 0.0):
        return False
    return min(ax, ay) > TOUCHED_LEAN * (ax + ay) or min(dx, dy) > TOUCHED_LEAN * (dx + dy)


def _sample_cells(ny, nx):
    """Where to look at a grid's cells: its middle and the quarters' middles."""
    return [(int(a * (ny - 2)), int(b * (nx - 2)))
            for a, b in ((0.5, 0.5), (0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75))]


def _margin_shows(x, y, xin, yin, shape):
    """Whether the half cells past a grid's outermost nodes (x, y; the nodes
    next inside xin, yin) reach a raster of shape: each stretch of them, by
    its box, so that a slanted edge is at worst taken for one that shows."""
    H, W = shape
    ox, oy = 1.5 * x - 0.5 * xin, 1.5 * y - 0.5 * yin
    x0, x1 = np.minimum(x, ox), np.maximum(x, ox)
    y0, y1 = np.minimum(y, oy), np.maximum(y, oy)
    return bool(((np.minimum(x0[:-1], x0[1:]) <= W) & (np.maximum(x1[:-1], x1[1:]) >= -1.0) &
                 (np.minimum(y0[:-1], y0[1:]) <= H) & (np.maximum(y1[:-1], y1[1:]) >= -1.0)).any())


def wants_touched(sx, sy, mode, period_px, ncells, shape):
    """Whether raster_cells_touched changes a mesh's picture on a raster of
    shape. Agg rounds every cell's corners to whole pixels: on cells along
    the pixel grid that is what sampling pixel centres gives, so only leaning
    cells, at any density, need the extra pass (a grid on its own projection
    pays for it only where its edge shows: sampling stops at the outermost
    nodes, and shading='nearest' has half a cell past them). Read off five
    cells across the grid."""
    if mode not in (NEAREST, FLAT) or period_px > 0.0 or ncells >= 1 << 24:
        return False
    ny, nx = sx.shape
    if ny < 2 or nx < 2:
        return False
    if any(_leans(sx[i:i + 2, j:j + 2], sy[i:i + 2, j:j + 2]) for i, j in _sample_cells(ny, nx)):
        return True
    return mode == NEAREST and (
        _margin_shows(sx[0], sy[0], sx[1], sy[1], shape) or
        _margin_shows(sx[-1], sy[-1], sx[-2], sy[-2], shape) or
        _margin_shows(sx[:, 0], sy[:, 0], sx[:, 1], sy[:, 1], shape) or
        _margin_shows(sx[:, -1], sy[:, -1], sx[:, -2], sy[:, -2], shape))


def wants_touched_nodes(view, src, x2d, y2d, mode, period_px, ncells, shape):
    """wants_touched of nodes not projected yet (a GPU canvas's): the five
    cells and the grid's two outer rows and columns projected on the host."""
    if mode not in (NEAREST, FLAT) or period_px > 0.0 or ncells >= 1 << 24:
        return False
    ny, nx = x2d.shape
    if ny < 2 or nx < 2:
        return False

    def nodes(i, j):
        bx, by = x2d[i, j], y2d[i, j]
        return forward_nodes(view, src, bx.get() if hasattr(bx, 'get') else bx,
                             by.get() if hasattr(by, 'get') else by)

    for i, j in _sample_cells(ny, nx):
        if _leans(*nodes(slice(i, i + 2), slice(j, j + 2))):
            return True
    if mode != NEAREST:
        return False
    all_of = slice(None)
    for inner, outer in ((1, 0), (-2, -1)):
        sx, sy = nodes(np.array([outer, inner]), all_of)
        if _margin_shows(sx[0], sy[0], sx[1], sy[1], shape):
            return True
        sx, sy = nodes(all_of, np.array([outer, inner]))
        if _margin_shows(sx[:, 0], sy[:, 0], sx[:, 1], sy[:, 1], shape):
            return True
    return False


def axes_lean(view, src, cx, cy):
    """_leans of the cells of a rectilinear grid (corners cx, cy along each
    axis in src): five cells, projected."""
    for i, j in _sample_cells(cy.size - 1, cx.size - 1):
        bx, by = np.meshgrid(cx[j:j + 2], cy[i:i + 2])
        sx, sy = forward_nodes(view, src, bx, by)
        if _leans(sx, sy):
            return True
    return False


def _cell_corners(a):
    """A node-centred grid's cell corners as shading='nearest' makes them
    (matplotlib's _interp_grid): half way between nodes, half a cell past
    the outermost, along each axis in turn."""
    def half(b):
        d = 0.5 * np.diff(b, axis=1)
        return np.hstack((b[:, [0]] - d[:, [0]], b[:, :-1] + d, b[:, [-1]] + d[:, [-1]]))
    return half(half(a).T).T


@njit(cache=True, nogil=True, inline='always')
def _touches(hx, hy, px, py, ax, ay, bx, by):
    """Whether a quad covers more than Agg's 1/256 of the unit square at
    (px, py). A side the whole square lies beyond separates them; a square
    well inside every side is covered; only a graze is measured."""
    if not np.isfinite(hx[0] + hx[1] + hx[2] + hx[3] + hy[0] + hy[1] + hy[2] + hy[3]):
        return False
    if max(max(hx[0], hx[1]), max(hx[2], hx[3])) <= px or \
            min(min(hx[0], hx[1]), min(hx[2], hx[3])) >= px + 1.0 or \
            max(max(hy[0], hy[1]), max(hy[2], hy[3])) <= py or \
            min(min(hy[0], hy[1]), min(hy[2], hy[3])) >= py + 1.0:
        return False
    s = (hx[0] * hy[1] - hx[1] * hy[0]) + (hx[1] * hy[2] - hx[2] * hy[1]) + \
        (hx[2] * hy[3] - hx[3] * hy[2]) + (hx[3] * hy[0] - hx[0] * hy[3])
    if s == 0.0:
        return False
    if s < 0.0:
        hx[1], hy[1], hx[3], hy[3] = hx[3], hy[3], hx[1], hy[1]
    cx, cy = px + 0.5, py + 0.5
    depth = 1.0
    for e in range(4):
        f = (e + 1) % 4
        ex, ey = hx[f] - hx[e], hy[f] - hy[e]
        reach = abs(ex) + abs(ey)
        if reach > 0.0:
            d = (ex * (cy - hy[e]) - ey * (cx - hx[e]) + 0.5 * reach) / reach
            if d <= 0.0:
                return False
            depth = min(depth, d)
    if depth >= 0.1:
        return True
    return square_cover(hx, hy, 4, px, py, ax, ay, bx, by) > 1.0 / 256.0


@njit(cache=True, nogil=True, inline='always')
def _higher(a, b):
    """The larger, NaN only if both are."""
    return b if b > a or a != a else a


@njit(cache=True, nogil=True, parallel=True)
def _touched_cells(cells, mx, my, z, out):
    """out: each pixel the value of the last cell (highest index) that
    covers more than Agg's 1/256 of it, of those at it and around it in
    cells (cell indices, NaN none). Pixel (i, j) spans j +- 1/2, i +- 1/2
    in (mx, my), the cells' corners."""
    H, W = cells.shape
    nc = mx.shape[1] - 1
    nan = np.float32(np.nan)
    for i in prange(H):
        hx, hy = np.empty(4), np.empty(4)
        ax, ay, bx, by = np.empty(16), np.empty(16), np.empty(16), np.empty(16)
        near = np.empty(9, np.float32)
        tops = np.empty(W, np.float32)                  # per column, its three rows' highest
        above, below = max(i - 1, 0), min(i + 1, H - 1)
        for j in range(W):
            tops[j] = _higher(_higher(cells[i, j], cells[above, j]), cells[below, j])
        for j in range(W):
            best = cells[i, j]
            if best != best:
                best = np.float32(-1.0)
            n = 0
            top = _higher(_higher(tops[j], tops[max(j - 1, 0)]), tops[min(j + 1, W - 1)])
            if top > best:                              # most pixels: no cell above their own
                for ni in range(max(i - 1, 0), min(i + 2, H)):
                    for nj in range(max(j - 1, 0), min(j + 2, W)):
                        q = cells[ni, nj]
                        if q > best:                    # (not NaN)
                            near[n] = q
                            n += 1
            while n > 0:                                # latest first: the first that touches wins
                q = near[0]
                for m in range(1, n):
                    q = max(q, near[m])
                k = np.int64(q)
                r, c = k // nc, k % nc
                hx[0], hy[0], hx[1], hy[1] = mx[r, c], my[r, c], mx[r, c + 1], my[r, c + 1]
                hx[2], hy[2] = mx[r + 1, c + 1], my[r + 1, c + 1]
                hx[3], hy[3] = mx[r + 1, c], my[r + 1, c]
                if _touches(hx, hy, j - 0.5, i - 0.5, ax, ay, bx, by):
                    best = q
                    break
                left = 0
                for m in range(n):
                    if near[m] != q:
                        near[left] = near[m]
                        left += 1
                n = left
            out[i, j] = z[np.int64(best)] if best >= 0.0 else nan


@njit(cache=True, nogil=True, inline='always')
def _along(a, r, k, nx):
    """_cell_corners' first step, one value: corner k of row r."""
    if k == 0:
        return a[r, 0] - 0.5 * (a[r, 1] - a[r, 0])
    if k == nx:
        return a[r, nx - 1] + 0.5 * (a[r, nx - 1] - a[r, nx - 2])
    return a[r, k - 1] + 0.5 * (a[r, k] - a[r, k - 1])


@njit(cache=True, nogil=True, parallel=True)
def _rounded_corners(a, out, nodes):
    """out: the cells' corners rounded to whole pixels (pixel centres are
    the integers here). nodes: a holds node-centred values, _cell_corners
    first; else a holds the corners."""
    ny, nx = a.shape
    if not nodes:
        for i in prange(ny):
            for j in range(nx):
                out[i, j] = np.floor(a[i, j] + 1.0) - 0.5
        return
    for q in prange(ny + 1):
        for k in range(nx + 1):
            if q == 0:
                b0, b1 = _along(a, 0, k, nx), _along(a, 1, k, nx)
                v = b0 - 0.5 * (b1 - b0)
            elif q == ny:
                b0, b1 = _along(a, ny - 2, k, nx), _along(a, ny - 1, k, nx)
                v = b1 + 0.5 * (b1 - b0)
            else:
                b0, b1 = _along(a, q - 1, k, nx), _along(a, q, k, nx)
                v = b0 + 0.5 * (b1 - b0)
            out[q, k] = np.floor(v + 1.0) - 0.5


@njit(cache=True, nogil=True, parallel=True)
def _cell_indices(z, idx):
    """idx: each cell's place in z as float32, NaN where z has no value."""
    ny, nx = z.shape
    nan = np.float32(np.nan)
    for i in prange(ny):
        for j in range(nx):
            idx[i, j] = np.float32(i * nx + j) if np.isfinite(z[i, j]) else nan


def raster_cells_touched(out, cells, sx, sy, z, mode):
    """raster_cells as Agg fills matplotlib's QuadMesh where wants_touched
    says so: every cell's corners rounded to whole pixels, then each pixel
    it covers at all painted, in order. cells: room for the cell indices,
    the shape of out."""
    zf = np.ascontiguousarray(z, dtype=np.float32)
    idx = _arena.take('raster.idx', zf.shape, np.float32)
    _cell_indices(zf, idx)
    nodes = mode != FLAT
    shape = (sx.shape[0] + 1, sx.shape[1] + 1) if nodes else sx.shape
    mx = _arena.take('raster.mx', shape, np.float64)
    my = _arena.take('raster.my', shape, np.float64)
    _rounded_corners(np.ascontiguousarray(sx, dtype=np.float64), mx, nodes)
    _rounded_corners(np.ascontiguousarray(sy, dtype=np.float64), my, nodes)
    cells.fill(np.nan)
    _run_cells(cells, mx, my, idx, FLAT, 0.0)
    _touched_cells(cells, mx, my, zf.ravel(), out)


# In a mesh of nearest or flat cells, which cell a pixel shows follows from
# the geometry: rasterizing and the search for the last cell that touches it
# (15-20 ms of a mesh of 1.9M cells) never look at a value. So from its second
# frame a mesh keeps, per pixel, the cell shown, and a frame is z taken by that
# table, ~1 ms. Made by the kernels themselves, given each cell's index for
# its value: what they would paint, the table says. By the nodes' arrays
# (kept_nodes' or cached_geometry's, which live as long as the view does),
# held weakly: new arrays, new table.
_CELL_TABLES = OrderedDict()    # sx, sy and the raster: [their weakrefs, the table or None]
_CELL_LOCK = threading.Lock()
_CELL_TABLE_MAX = 8


@njit(cache=True, nogil=True, parallel=True)
def _not_finite(z):
    """(NaNs, infinities) in z."""
    nan, inf = 0, 0
    for i in prange(z.size):
        v = z[i]
        nan += 1 if v != v else 0
        inf += 1 if v == np.inf or v == -np.inf else 0
    return nan, inf


@njit(cache=True, nogil=True, parallel=True)
def _take_cells(z, table, out):
    """out: z at each pixel's cell, NaN at a pixel of none (-1)."""
    H, W = out.shape
    nan = np.float32(np.nan)
    for i in prange(H):
        for j in range(W):
            k = table[i, j]
            out[i, j] = z[k] if k >= 0 else nan


def _cell_table(shape, sx, sy, zshape, mode, period, touched):
    """Each pixel's cell, as an index into the values raveled (-1: none)."""
    n = int(np.prod(zshape))
    index = np.arange(n, dtype=np.float32).reshape(zshape)
    t = np.full(shape, np.nan, np.float32)
    if touched:
        cells = _arena.take('canvas.cells', shape, np.float32)
        raster_cells_touched(t, cells, sx, sy, index, mode)
    else:
        raster_cells(t, sx, sy, index, mode, period)
    return np.where(np.isnan(t), -1, t).astype(np.int32)


def cells_by_table(out, sx, sy, z, mode, period, touched):
    """raster_cells (touched: raster_cells_touched) into a raster filled
    with NaN, by this mesh's table: True if done. Values that would take
    another way do not use one: an infinity weighs 0 into NaN in a flat cell,
    and the touched cells pass over a NaN one for the cell beneath."""
    if mode not in (NEAREST, FLAT) or z.size >= 1 << 24:
        return False
    key = (id(sx), id(sy), out.shape, mode, period)
    with _CELL_LOCK:
        e = _CELL_TABLES.get(key)
        if e is None or e[0]() is not sx or e[1]() is not sy:
            _CELL_TABLES[key] = [weakref.ref(sx), weakref.ref(sy), None]
            while len(_CELL_TABLES) > _CELL_TABLE_MAX:
                _CELL_TABLES.popitem(last=False)
            return False
        _CELL_TABLES.move_to_end(key)
    zf = np.ascontiguousarray(z, dtype=np.float32).ravel()
    nan, inf = _not_finite(zf)
    if inf or (touched and nan):
        return False
    if e[2] is None:
        e[2] = _cell_table(out.shape, sx, sy, np.shape(z), mode, period, touched)
    _take_cells(zf, e[2], out)
    return True

