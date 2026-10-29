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
drawn forward: every node projected to the raster once, every cell split
into two triangles rasterized with their values interpolated (bilinear
mode), the value of the node nearest in the cell's own parameter space
(nearest mode: the cell around each node, its sides half-way to the next
nodes, as matplotlib's pcolormesh(shading='nearest') builds them), or the cell's own value (flat mode: the nodes are cell corners,
pcolormesh(shading='flat')). On a cylindrical map a cell crossing the seam is unwrapped and drawn on both
sides.
"""
import threading
from collections import OrderedDict

import numpy as np

from . import _proj as P
from ._jit import njit, prange

TABLE_KEEP = 2                  # maps whose pixel -> source tables are kept
TABLE_BYTES = 128 << 20         # and at most this much memory in all
_TABLES = OrderedDict()         # map key -> (sx, sy, ok): the newest last
_SEEN = OrderedDict()           # map keys drawn once (a table is made the second time)
_TABLE_LOCK = threading.Lock()

NEAREST, BILINEAR, FLAT = 0, 1, 2           # FLAT: 2-D cells given by their corners


class Axis:
    """A grid axis prepared for the kernels: increasing coordinates, and
    whether the data has to be flipped to match. edges=True when coords
    are the n+1 cell edges (pcolormesh 'flat') rather than n centres."""

    def __init__(self, coords, angular=False, edges=False):
        c = np.asarray(coords, np.float64).ravel()
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
    # bilinear: between the first and last nodes only, as contourf fills
    if not per and (f < 0.0 or f > n - 1):
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
    """Every kept pixel -> source table let go."""
    with _TABLE_LOCK:
        _TABLES.clear()
        _SEEN.clear()


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


@njit(cache=True, nogil=True)
def _tri(out, ax, ay, bx, by, cx, cy, za, zb, zc, mode, kind, zn, H, W):
    """One triangle into the raster. Bilinear mode: values za, zb, zc
    interpolated. Nearest mode: zn holds the cell's four node values
    (z00, z01, z11, z10) and kind 0/1 says which half of the cell this is."""
    area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if area == 0.0 or not np.isfinite(area):
        return
    j0 = max(0, np.int64(np.ceil(min(ax, min(bx, cx)))))
    j1 = min(W - 1, np.int64(np.floor(max(ax, max(bx, cx)))))
    i0 = max(0, np.int64(np.ceil(min(ay, min(by, cy)))))
    i1 = min(H - 1, np.int64(np.floor(max(ay, max(by, cy)))))
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
def _raster_cells(out, sx, sy, z, mode, period):
    H, W = out.shape
    ny, nx = sx.shape
    for i in prange(ny - 1):
        zn = np.empty(4, np.float32)
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
            lo = min(min(x00, x01), min(x11, x10))
            hi = max(max(x00, x01), max(x11, x10))
            if hi < -1.0 and period == 0.0 or lo > W and period == 0.0:
                continue
            if max(max(y00, y01), max(y11, y10)) < -1.0 or min(min(y00, y01), min(y11, y10)) > H:
                continue
            if mode == FLAT:                            # z holds the cells' own values
                z00 = z01 = z11 = z10 = z[i, j]
            else:
                z00, z01 = z[i, j], z[i, j + 1]
                z11, z10 = z[i + 1, j + 1], z[i + 1, j]
            zn[0], zn[1], zn[2], zn[3] = z00, z01, z11, z10
            nshift = 1
            shifts = (0.0, 0.0, 0.0)
            if period > 0.0:
                if lo < 0.0:
                    shifts = (0.0, period, 0.0)
                    nshift = 2
                elif hi > W - 1:
                    shifts = (0.0, -period, 0.0)
                    nshift = 2
            for q in range(nshift):
                o = shifts[q]
                tm = BILINEAR if mode == FLAT else mode
                _tri(out, x00 + o, y00, x01 + o, y01, x11 + o, y11, z00, z01, z11, tm, 0, zn,
                     H, W)
                _tri(out, x00 + o, y00, x11 + o, y11, x10 + o, y10, z00, z11, z10, tm, 1, zn,
                     H, W)


def raster_cells(out, sx, sy, z, mode, period_px=0.0):
    """Cells of a grid whose nodes are at raster pixels (sx, sy) into out;
    period_px > 0 on a cylindrical map (the width of 360 degrees)."""
    _raster_cells(out, sx, sy, np.ascontiguousarray(z, dtype=np.float32), mode,
                  float(period_px))
