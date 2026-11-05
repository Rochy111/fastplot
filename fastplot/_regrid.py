"""cartopy's regrid_shape for barbs and quiver on a map: the vectors (and a
colour array) interpolated onto a regular grid of the map's own x and y,
regrid_shape points across its extent (an int for the shorter side, the
longer in proportion, as GeoAxes._regrid_shape_aspect; or (nx, ny)).

Each new point is a weighted sum of a few of the data's points, and which
ones, with what weights, depends only on the map, the extent, the shape and
the data's coordinates: that plan is made once and kept (_artists.drawn,
told by the coordinates' bits), so a forecast's later frames each cost a
gather of a few hundred values. Making it:

- 1-D axes: each new point taken back into the data's crs and read
  bilinearly between the four nodes round it (a whole circle of longitude
  wrapping), as contourf reads a field. Under a millisecond for a 0.25
  degree global grid.
- 2-D coordinates (WRF, HRRR): the grid's own cells, projected onto the
  map and each split into two triangles, the new points found in them
  linearly. cartopy triangulates the projected points (Delaunay) instead:
  the same surface inside the grid, without the convex hull's bridging of
  a curved edge; a 400 x 600 grid takes ~40 ms here against ~1 s.
- scattered points (1-D x, y and u): Delaunay, as cartopy.

The vectors stay in the data's crs, turned onto the map where each new
point is, as any other barbs. Off the data the new points are NaN, which
draw nothing."""
import numpy as np

from . import _proj
from ._artists import _as_host, _masked

SEAM_SIZE = 20.0                # a cell this many times the median's size crosses the map's seam


def shape_of(regrid_shape, extent):
    """(nx, ny) of regrid_shape over extent (x0, x1, y0, y1)."""
    if np.ndim(regrid_shape) == 0:
        n = int(regrid_shape)
        xr, yr = extent[1] - extent[0], extent[3] - extent[2]
        aspect = xr / yr
        return (int(n * aspect), n) if xr >= yr else (n, int(n / aspect))
    nx, ny = regrid_shape
    return int(nx), int(ny)


def _values(a):
    """float64 or float32, NaN where masked: flat."""
    if _masked(a):
        return np.ma.filled(np.ma.asarray(a, np.float64), np.nan).ravel()
    a = np.asarray(_as_host(a))
    if a.dtype.kind != 'f':
        a = a.astype(np.float64)
    return a.ravel()


def _nodes(axis, u):
    """For coordinates u on a grid axis (_raster.Axis): the nodes either side
    (k0, k1), the weight of k1, and whether u is on the axis at all."""
    c = axis.coords
    n = c.size
    with np.errstate(invalid='ignore'):
        if axis.angular:
            u = axis.lo + np.mod(u - axis.lo, 360.0)
        if axis.periodic:                               # the cell from the last back to the first
            ce = np.r_[c[-1] - 360.0, c, c[0] + 360.0]
            k = np.clip(np.searchsorted(ce, u, 'right') - 1, 0, ce.size - 2)
            t = (u - ce[k]) / (ce[k + 1] - ce[k])
            return (k - 1) % n, k % n, t, np.isfinite(u)
        k = np.clip(np.searchsorted(c, u, 'right') - 1, 0, n - 2)
        t = (u - c[k]) / (c[k + 1] - c[k])
        return k, k + 1, t, (u >= c[0]) & (u <= c[-1])


def _rect_plan(g, sx, sy, width):
    """Bilinear: four nodes a point, as flat indices into the field as
    given (width columns: a cyclic grid's dropped column, flips undone)."""
    xa, ya = g.xaxis, g.yaxis
    i0, i1, ti, oki = _nodes(xa, sx)
    j0, j1, tj, okj = _nodes(ya, sy)
    nxa, nya = xa.coords.size, ya.coords.size
    shift = 1 if g.cyclic > 0 else 0
    gi = [(nxa - 1 - i if xa.flip else i) + shift for i in (i0, i1)]
    gj = [nya - 1 - j if ya.flip else j for j in (j0, j1)]
    idx = np.stack([gj[0] * width + gi[0], gj[0] * width + gi[1],
                    gj[1] * width + gi[0], gj[1] * width + gi[1]], axis=1)
    with np.errstate(invalid='ignore'):
        w = np.stack([(1 - ti) * (1 - tj), ti * (1 - tj), (1 - ti) * tj, ti * tj], axis=1)
    w[~(oki & okj)] = np.nan
    return idx, w


def _cell_triangles(px, py, x0, dx, nx, y0, dy, ny):
    """A structured grid's cells (px, py: its nodes on the map, 2-D) that
    can hold a point of the regular grid, each as two triangles (sw, se,
    ne), (sw, ne, nw) of flat node indices. Told by each cell's bounding
    box, whole arrays at a time: a fine grid's cells mostly hold none.
    Cells far larger than the rest (one projected across the map's seam)
    are left out."""
    gx, gy = (px - x0) / dx, (py - y0) / dy             # in grid points
    with np.errstate(invalid='ignore'):
        q = [(gx[:-1, :-1], gy[:-1, :-1]), (gx[:-1, 1:], gy[:-1, 1:]), (gx[1:, 1:], gy[1:, 1:]),
             (gx[1:, :-1], gy[1:, :-1])]
        xlo = np.minimum(np.minimum(q[0][0], q[1][0]), np.minimum(q[2][0], q[3][0]))
        xhi = np.maximum(np.maximum(q[0][0], q[1][0]), np.maximum(q[2][0], q[3][0]))
        ylo = np.minimum(np.minimum(q[0][1], q[1][1]), np.minimum(q[2][1], q[3][1]))
        yhi = np.maximum(np.maximum(q[0][1], q[1][1]), np.maximum(q[2][1], q[3][1]))
        ok = np.isfinite(xlo) & np.isfinite(xhi) & np.isfinite(ylo) & np.isfinite(yhi)
        ok &= (np.ceil(np.maximum(xlo, 0)) <= np.floor(np.minimum(xhi, nx - 1))) & \
            (np.ceil(np.maximum(ylo, 0)) <= np.floor(np.minimum(yhi, ny - 1)))
        size = np.maximum(xhi - xlo, yhi - ylo)
        fin = np.isfinite(size)
        if ok.any() and fin.any():
            typical = np.median(size[fin][::max(1, fin.sum() // 4096)])     # (a sample's)
            ok &= (size <= SEAM_SIZE * max(typical, 1e-12)) | (size <= 2.0)
    cj, ci = np.nonzero(ok)
    W = px.shape[1]
    sw = cj * W + ci
    se, ne, nw = sw + 1, sw + W + 1, sw + W
    return np.concatenate([np.stack([sw, se, ne], 1), np.stack([sw, ne, nw], 1)])


def _locate(px, py, T, x0, dx, nx, y0, dy, ny):
    """The regular grid's points (x0 + i dx, y0 + j dy) found in triangles T
    of points (px, py, flat): (idx (m, 3), w (m, 3)) of the m = nx ny
    points, row by row, NaN weights for a point in none. Each triangle is
    tried only at the grid points its bounding box holds."""
    m = nx * ny
    idx = np.zeros((m, 3), np.int64)
    w = np.full((m, 3), np.nan)
    gx, gy = (px - x0) / dx, (py - y0) / dy             # in grid points
    tx, ty = gx[T], gy[T]
    with np.errstate(invalid='ignore'):
        lo_i, hi_i = np.ceil(tx.min(1)), np.floor(tx.max(1))
        lo_j, hi_j = np.ceil(ty.min(1)), np.floor(ty.max(1))
        ok = np.isfinite(lo_i) & np.isfinite(lo_j) & np.isfinite(hi_i) & np.isfinite(hi_j)
        lo_i, hi_i = np.maximum(lo_i, 0), np.minimum(hi_i, nx - 1)
        lo_j, hi_j = np.maximum(lo_j, 0), np.minimum(hi_j, ny - 1)
        ok &= (lo_i <= hi_i) & (lo_j <= hi_j)
    t = np.flatnonzero(ok)
    if not t.size:
        return idx, w
    lo_i, lo_j = lo_i[t].astype(np.int64), lo_j[t].astype(np.int64)
    ni = hi_i[t].astype(np.int64) - lo_i + 1
    nj = hi_j[t].astype(np.int64) - lo_j + 1
    count = ni * nj
    tt = np.repeat(np.arange(t.size), count)            # (triangle, grid point) pairs
    k = np.arange(tt.size) - np.repeat(np.cumsum(count) - count, count)
    i = lo_i[tt] + k % ni[tt]
    j = lo_j[tt] + k // ni[tt]
    v = T[t[tt]]
    x = np.stack([px[v[:, 0]], px[v[:, 1]], px[v[:, 2]]], 1)
    y = np.stack([py[v[:, 0]], py[v[:, 1]], py[v[:, 2]]], 1)
    qx, qy = x0 + i * dx, y0 + j * dy
    with np.errstate(invalid='ignore', divide='ignore'):
        det = (y[:, 1] - y[:, 2]) * (x[:, 0] - x[:, 2]) + (x[:, 2] - x[:, 1]) * (y[:, 0] - y[:, 2])
        l0 = ((y[:, 1] - y[:, 2]) * (qx - x[:, 2]) + (x[:, 2] - x[:, 1]) * (qy - y[:, 2])) / det
        l1 = ((y[:, 2] - y[:, 0]) * (qx - x[:, 2]) + (x[:, 0] - x[:, 2]) * (qy - y[:, 2])) / det
    l2 = 1.0 - l0 - l1
    eps = -1e-9
    inside = (det != 0) & (l0 >= eps) & (l1 >= eps) & (l2 >= eps)
    p = j[inside] * nx + i[inside]
    _, first = np.unique(p, return_index=True)          # (a point on a shared edge: once)
    sel = np.flatnonzero(inside)[first]
    p = p[first]
    idx[p] = v[sel]
    w[p] = np.stack([l0[sel], l1[sel], l2[sel]], 1)
    return idx, w


def _plan(P, src, extent, nx, ny, x, y, shape):
    """(sx, sy, idx, w): the new points in src, and each one's nodes and
    weights in the field as given (flat). None where it cannot be made."""
    from ._fields import _Grid
    x0, x1, y0, y1 = extent
    X, Y = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    sx, sy = _proj.transform_points(P, src, X.ravel(), Y.ravel())
    sx, sy = np.asarray(sx, np.float64), np.asarray(sy, np.float64)
    dx = (x1 - x0) / (nx - 1) if nx > 1 else 1.0
    dy = (y1 - y0) / (ny - 1) if ny > 1 else 1.0
    g = _Grid(x, y, shape, src) if len(shape) == 2 else None
    if g is not None and g.rect:
        if g.xaxis.coords.size < 2 or g.yaxis.coords.size < 2:
            return None
        idx, w = _rect_plan(g, sx, sy, shape[1])
    elif len(shape) == 2 and np.shape(x) == shape and np.shape(y) == shape:
        if min(shape) < 2:
            return None
        px, py = _proj.transform_points(src, P, np.asarray(x, np.float64).ravel(),
                                        np.asarray(y, np.float64).ravel())
        px = np.asarray(px, np.float64).reshape(shape)
        py = np.asarray(py, np.float64).reshape(shape)
        T = _cell_triangles(px, py, x0, dx, nx, y0, dy, ny)
        idx, w = _locate(px.ravel(), py.ravel(), T, x0, dx, nx, y0, dy, ny)
    else:                                               # scattered points: as cartopy, Delaunay
        from ._tri import delaunay
        xs = np.broadcast_to(np.asarray(x, np.float64), shape).ravel()
        ys = np.broadcast_to(np.asarray(y, np.float64), shape).ravel()
        px, py = _proj.transform_points(src, P, xs, ys)
        px, py = np.asarray(px, np.float64), np.asarray(py, np.float64)
        keep = np.flatnonzero(np.isfinite(px) & np.isfinite(py))
        _, first = np.unique(np.c_[px[keep], py[keep]], axis=0, return_index=True)
        keep = keep[np.sort(first)]                     # (a point given twice: once)
        if keep.size < 3:
            return None
        T = keep[delaunay(px[keep], py[keep]).astype(np.int64)]
        idx, w = _locate(px, py, T, x0, dx, nx, y0, dy, ny)
    return sx, sy, idx, w


def regrid(P, src, regrid_shape, extent, x, y, fields):
    """(x, y, fields): fields (on points x, y in src) interpolated onto the
    regular grid of map P, as points in src. None where they cannot be
    (a single row or column)."""
    from ._artists import _src_key, drawn
    if extent is None:
        return None
    x0, x1, y0, y1 = (float(v) for v in extent)
    nx, ny = shape_of(regrid_shape, (x0, x1, y0, y1))
    if nx < 1 or ny < 1:
        return None
    shape = np.shape(fields[0])
    x, y = np.asarray(_as_host(x)), np.asarray(_as_host(y))

    def make():
        return _plan(P, src, (x0, x1, y0, y1), nx, ny, x, y, shape)
    skey = _src_key(src)
    if skey is None:
        plan = make()
    else:
        plan = drawn(('regrid', skey, _proj.value_key(P), (x0, x1, y0, y1), nx, ny, shape),
                     (x, y), make)
    if plan is None:
        return None
    sx, sy, idx, w = plan
    out = []
    for f in fields:
        v = _values(f)
        with np.errstate(invalid='ignore'):
            out.append((v[idx] * w).sum(axis=1).reshape(ny, nx))
    return sx.reshape(ny, nx).copy(), sy.reshape(ny, nx).copy(), out    # (the plan's: kept)
