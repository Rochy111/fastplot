"""Contour lines: marching squares on the grid, joined into polylines.

    fi, fj, offsets, level_of = lines(z, levels)

z (ny, nx) float32, NaN where there is no value; levels any order. Every
polyline k runs through fractional grid indices (fi, fj)[offsets[k]:
offsets[k+1]] (row, column) at height levels[level_of[k]]; closed ones end
on their first point. The caller maps indices to coordinates (np.interp on
1-D axes, bilinear on 2-D lon/lat) and projects them -- so contouring is
done once on the grid whatever the map, never on projected polygons.

Three passes, all numba:
  1. per row of cells, in parallel: how many segments each level makes
     there. A cell only visits the levels between its corners' min and max
     (a binary search in the sorted levels), so 40 levels cost little more
     than one;
  2. per row again: the segments, each with its two end points and the ids
     of the grid edges they lie on (2 (i nx + j) for the edge from (i, j)
     to (i, j+1), +1 for the edge to (i+1, j)). Segments land grouped by
     level, rows in order;
  3. per level, in parallel: the segments' end edges hashed, so the two
     segments meeting on an edge find each other; then walks along those
     links -- first from every loose end (lines leaving the grid or a NaN
     hole), then around what is left (closed rings).
Saddles (two opposite corners above) are split by the cell's mean, as
matplotlib's (contourpy's) default does. Cells with a NaN corner are left
out (matplotlib's corner_mask=True would keep their unmasked triangle).
"""
import numpy as np

from ._jit import njit, prange

# per marching-squares case, the edges its segments join: up to two pairs;
# edges 0 bottom (z00-z01), 1 right (z01-z11), 2 top (z10-z11), 3 left (z00-z10)
_CASES = np.array([
    [-1, -1, -1, -1],       # 0
    [3, 0, -1, -1],         # 1  z00
    [0, 1, -1, -1],         # 2  z01
    [3, 1, -1, -1],         # 3  z00 z01
    [1, 2, -1, -1],         # 4  z11
    [3, 0, 1, 2],           # 5  z00 z11 (saddle: high corners apart)
    [0, 2, -1, -1],         # 6  z01 z11
    [3, 2, -1, -1],         # 7  z00 z01 z11
    [2, 3, -1, -1],         # 8  z10
    [0, 2, -1, -1],         # 9  z00 z10
    [0, 1, 2, 3],           # 10 z01 z10 (saddle: high corners apart)
    [1, 2, -1, -1],         # 11 z00 z01 z10
    [3, 1, -1, -1],         # 12 z11 z10
    [0, 1, -1, -1],         # 13 z00 z11 z10
    [3, 0, -1, -1],         # 14 z01 z11 z10
    [-1, -1, -1, -1],       # 15
], np.int64)
# saddles whose centre is high: the high corners join, the low ones are cut off
_SADDLE_HIGH = np.array([[0, 1, 2, 3],       # 5:  around z01, around z10
                         [3, 0, 1, 2]],      # 10: around z00, around z11
                        np.int64)


@njit(cache=True, nogil=True)
def _span(levels, lo, hi):
    """Indices [a, b) of the sorted levels with lo < level <= hi."""
    a = np.searchsorted(levels, lo, side='right')
    b = np.searchsorted(levels, hi, side='right')
    return a, b


@njit(cache=True, nogil=True)
def _corners(z, i, j, j1):
    return z[i, j], z[i, j1], z[i + 1, j1], z[i + 1, j]


@njit(cache=True, nogil=True, parallel=True)
def _count(z, levels, per, counts):
    ny, nx = z.shape
    for i in prange(ny - 1):
        for j in range(nx - 1 + per):
            j1 = j + 1 if j + 1 < nx else 0             # (periodic: the last cell wraps)
            z00, z01, z11, z10 = _corners(z, i, j, j1)
            if not (np.isfinite(z00) and np.isfinite(z01) and np.isfinite(z11)
                    and np.isfinite(z10)):
                continue
            lo = min(min(z00, z01), min(z11, z10))
            hi = max(max(z00, z01), max(z11, z10))
            if lo == hi:
                continue
            a, b = _span(levels, lo, hi)
            for lv in range(a, b):
                v = levels[lv]
                c = ((z00 >= v) * 1) | ((z01 >= v) * 2) | ((z11 >= v) * 4) | ((z10 >= v) * 8)
                counts[lv, i] += 2 if (c == 5 or c == 10) else 1


@njit(cache=True, nogil=True)
def _edge_point(z, nx, i, j, j1, e, v):
    """The level-v crossing on edge e of cell (i, j): (row, col, edge id);
    j1 the column read as j + 1, nx the row length of the edge ids."""
    if e == 0:                                          # bottom: (i, j) - (i, j+1)
        za, zb = z[i, j], z[i, j1]
        t = (v - za) / (zb - za)
        return np.float64(i), j + t, 2 * (i * nx + j)
    if e == 1:                                          # right: (i, j+1) - (i+1, j+1)
        za, zb = z[i, j1], z[i + 1, j1]
        t = (v - za) / (zb - za)
        return i + t, np.float64(j + 1), 2 * (i * nx + j + 1) + 1
    if e == 2:                                          # top: (i+1, j) - (i+1, j+1)
        za, zb = z[i + 1, j], z[i + 1, j1]
        t = (v - za) / (zb - za)
        return np.float64(i + 1), j + t, 2 * ((i + 1) * nx + j)
    za, zb = z[i, j], z[i + 1, j]                       # left: (i, j) - (i+1, j)
    t = (v - za) / (zb - za)
    return i + t, np.float64(j), 2 * (i * nx + j) + 1


@njit(cache=True, nogil=True, parallel=True)
def _fill(z, levels, per, starts, cases, saddle_high, pts, eids):
    ny, nx = z.shape
    nxe = nx + per                                      # the row length edge ids count in
    for i in prange(ny - 1):
        pos = np.empty(levels.size, np.int64)
        for lv in range(levels.size):
            pos[lv] = starts[lv, i]
        for j in range(nx - 1 + per):
            j1 = j + 1 if j + 1 < nx else 0
            z00, z01, z11, z10 = _corners(z, i, j, j1)
            if not (np.isfinite(z00) and np.isfinite(z01) and np.isfinite(z11)
                    and np.isfinite(z10)):
                continue
            lo = min(min(z00, z01), min(z11, z10))
            hi = max(max(z00, z01), max(z11, z10))
            if lo == hi:
                continue
            a, b = _span(levels, lo, hi)
            for lv in range(a, b):
                v = levels[lv]
                c = ((z00 >= v) * 1) | ((z01 >= v) * 2) | ((z11 >= v) * 4) | ((z10 >= v) * 8)
                row = cases[c]
                if c == 5 or c == 10:
                    centre = 0.25 * (np.float64(z00) + z01 + z11 + z10)
                    if centre >= v:
                        row = saddle_high[0 if c == 5 else 1]
                for k in range(0, 4, 2):
                    if row[k] < 0:
                        break
                    s = pos[lv]
                    pos[lv] += 1
                    r0, c0, e0 = _edge_point(z, nxe, i, j, j1, row[k], v)
                    r1, c1, e1 = _edge_point(z, nxe, i, j, j1, row[k + 1], v)
                    pts[s, 0], pts[s, 1], pts[s, 2], pts[s, 3] = r0, c0, r1, c1
                    eids[s, 0], eids[s, 1] = e0, e1


@njit(cache=True, nogil=True, parallel=True)
def _chain(pts, eids, lstart, out, ooff, nverts, npoly):
    """Per level: segments [lstart[l], lstart[l+1]) joined into polylines,
    written at out[2 lstart[l]:] (vertices) and ooff[lstart[l] + l:]
    (polyline starts, relative); nverts/npoly the counts written."""
    for lv in prange(lstart.size - 1):
        s0, s1 = lstart[lv], lstart[lv + 1]
        n = s1 - s0
        if n == 0:
            continue
        link = np.full(2 * n, -1, np.int64)             # endpoint -> the endpoint it meets
        size = 1 << max(4, int(np.ceil(np.log2(8 * n))))
        mask = size - 1
        tkey = np.full(size, -1, np.int64)              # open addressing: edge id -> endpoint
        tval = np.empty(size, np.int64)
        for q in range(2 * n):
            key = eids[s0 + q // 2, q % 2]
            h = (key * 2654435761) & mask              # (an edge holds at most two endpoints)
            while tkey[h] != -1 and tkey[h] != key:
                h = (h + 1) & mask
            if tkey[h] == key:
                link[q] = tval[h]
                link[tval[h]] = q
            else:
                tkey[h] = key
                tval[h] = q
        seen = np.zeros(n, np.bool_)
        vbase = 2 * s0
        pbase = s0 + lv
        nv = 0
        npl = 0
        for sweep in range(2):                          # loose ends first, then rings
            for start in range(n):
                if seen[start]:
                    continue
                if sweep == 0:
                    if link[2 * start] == -1:
                        end = 0
                    elif link[2 * start + 1] == -1:
                        end = 1
                    else:
                        continue
                else:
                    end = 0
                ooff[pbase + npl] = nv
                npl += 1
                s = start
                out[vbase + nv, 0] = pts[s0 + s, 2 * end]
                out[vbase + nv, 1] = pts[s0 + s, 2 * end + 1]
                nv += 1
                while True:
                    seen[s] = True
                    other = 1 - end
                    out[vbase + nv, 0] = pts[s0 + s, 2 * other]
                    out[vbase + nv, 1] = pts[s0 + s, 2 * other + 1]
                    nv += 1
                    nxt = link[2 * s + other]
                    if nxt == -1:
                        break
                    s2 = nxt // 2
                    if seen[s2]:                        # back at the start of a ring
                        break
                    s = s2
                    end = nxt % 2
        nverts[lv] = nv
        npoly[lv] = npl


@njit(cache=True, nogil=True, parallel=True)
def _interp_index(f, fp, out):
    """np.interp(f, np.arange(fp.size), fp), value for value: numpy's
    arr_interp on an axis of whole numbers (its slope divides by exactly 1)."""
    n = fp.size
    last = np.float64(n - 1)
    for i in prange(f.size):
        x = f[i]
        if n == 1:                                      # numpy's one-point case: NaN too
            out[i] = fp[0]
        elif np.isnan(x):
            out[i] = x
        elif x < 0.0:
            out[i] = fp[0]
        elif x >= last:
            out[i] = fp[n - 1]
        else:
            j = np.int64(np.floor(x))
            if np.float64(j) == x:
                out[i] = fp[j]
            else:
                slope = (fp[j + 1] - fp[j]) / (np.float64(j + 1) - np.float64(j))
                r = slope * (x - np.float64(j)) + fp[j]
                if np.isnan(r):
                    r = slope * (x - np.float64(j + 1)) + fp[j + 1]
                    if np.isnan(r) and fp[j] == fp[j + 1]:
                        r = fp[j]
                out[i] = r


def interp_index(f, fp):
    """np.interp(f, np.arange(len(fp)), fp), bit for bit, on every core."""
    f = np.ascontiguousarray(f, dtype=np.float64)
    out = np.empty(f.size, np.float64)
    if f.size:
        _interp_index(f, np.ascontiguousarray(fp, dtype=np.float64), out)
    return out


def lines(z, levels, periodic=False):
    """Contour lines of z at levels: (fi, fj, offsets, level_of) -- see the
    module docstring. periodic: the last column joins the first (a global
    grid), as if z had its first column again after its last."""
    z = np.ascontiguousarray(z, dtype=np.float32)
    levels = np.asarray(levels, np.float64).ravel()
    order = np.argsort(levels, kind='stable')
    lv_sorted = np.ascontiguousarray(levels[order])
    ny, nx = z.shape
    L = lv_sorted.size
    empty = (np.empty(0), np.empty(0), np.zeros(1, np.int64), np.empty(0, np.int64))
    if L == 0 or ny < 2 or nx < 2:
        return empty
    per = int(bool(periodic))
    counts = np.zeros((L, ny - 1), np.int64)
    _count(z, lv_sorted, per, counts)
    flat = counts.ravel()
    starts = np.zeros(flat.size + 1, np.int64)
    np.cumsum(flat, out=starts[1:])
    nseg = int(starts[-1])
    if nseg == 0:
        return empty
    pts = np.empty((nseg, 4), np.float64)
    eids = np.empty((nseg, 2), np.int64)
    _fill(z, lv_sorted, per, starts[:-1].reshape(L, ny - 1), _CASES, _SADDLE_HIGH, pts, eids)
    lstart = np.ascontiguousarray(starts[::ny - 1])     # each level's first segment
    out = np.empty((2 * nseg, 2), np.float64)
    ooff = np.empty(nseg + L, np.int64)
    nverts = np.zeros(L, np.int64)
    npoly = np.zeros(L, np.int64)
    _chain(pts, eids, lstart, out, ooff, nverts, npoly)
    # compact the per-level blocks
    vparts, oparts, lof = [], [], []
    vtot = 0
    for lv in range(L):
        nv, npl = nverts[lv], npoly[lv]
        if not npl:
            continue
        vb = 2 * lstart[lv]
        pb = lstart[lv] + lv
        vparts.append(out[vb:vb + nv])
        oparts.append(ooff[pb:pb + npl] + vtot)
        lof.append(np.full(npl, order[lv], np.int64))
        vtot += nv
    v = np.concatenate(vparts)
    offsets = np.concatenate(oparts + [np.array([vtot], np.int64)])
    return v[:, 0].copy(), v[:, 1].copy(), offsets, np.concatenate(lof)
