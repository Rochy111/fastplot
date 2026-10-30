"""Contour lines: marching squares on the grid, joined into polylines.

    fi, fj, offsets, level_of = lines(z, levels)

z (ny, nx) float32, NaN where there is no value; levels any order. Every
polyline k runs through fractional grid indices (fi, fj)[offsets[k]:
offsets[k+1]] (row, column) at height levels[level_of[k]]; closed ones end
on their first point. The caller maps indices to coordinates (np.interp on
1-D axes, bilinear on 2-D lon/lat) and projects them -- so contouring is
done once on the grid whatever the map, never on projected polygons.

Three passes, all numba:
  0. per node, in parallel: its band, how many of the sorted levels are at
     or below it (a binary search, once a node rather than twice a cell);
  1. per row of cells, in parallel: how many segments each level makes
     there. A cell only visits the levels between its corners' lowest and
     highest bands -- the levels above its min and at or below its max --
     so 40 levels cost little more than one, and a cell inside one band
     (most of them) costs four compares;
  2. per row again: the segments, each with its two end points and the ids
     of the grid edges they lie on (2 (i nx + j) for the edge from (i, j)
     to (i, j+1), +1 for the edge to (i+1, j)). Segments land grouped by
     level, rows in order;
  3. per level, in parallel: the two segments meeting on an edge linked,
     in one sweep down the level's segments (row by row, left to right, as
     pass 2 laid them): an edge is shared with the cell to the right (a
     vertical edge, met later in the same row) or the one below (a
     horizontal edge, met in the next row), so a row of pending ends per
     kind, by column, finds every pair -- no hash table; then walks along
     those links -- first from every loose end (lines leaving the grid or
     a NaN hole), then around what is left (closed rings).
Saddles (two opposite corners above) are split by the cell's mean, as
matplotlib's (contourpy's) default does. Cells with a NaN corner are left
out (matplotlib's corner_mask=True would keep their unmasked triangle).
"""
import numpy as np

from . import _arena
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


@njit(cache=True, nogil=True, parallel=True)
def _bands(z, levels, band):
    """band[i, j]: the sorted levels at or below z[i, j] (searchsorted
    'right'), so a cell's levels lo < level <= hi are [its corners' least
    band, their greatest)."""
    ny, nx = z.shape
    for i in prange(ny):
        for j in range(nx):
            band[i, j] = np.searchsorted(levels, z[i, j], side='right')


@njit(cache=True, nogil=True)
def _span(band, i, j, j1):
    """Indices [a, b) of the sorted levels crossing cell (i, j)."""
    b00, b01, b11, b10 = band[i, j], band[i, j1], band[i + 1, j1], band[i + 1, j]
    return min(min(b00, b01), min(b11, b10)), max(max(b00, b01), max(b11, b10))


@njit(cache=True, nogil=True)
def _corners(z, i, j, j1):
    return z[i, j], z[i, j1], z[i + 1, j1], z[i + 1, j]


@njit(cache=True, nogil=True, parallel=True)
def _count(z, band, levels, per, counts):
    ny, nx = z.shape
    for i in prange(ny - 1):
        for j in range(nx - 1 + per):
            j1 = j + 1 if j + 1 < nx else 0             # (periodic: the last cell wraps)
            a, b = _span(band, i, j, j1)
            if a == b:                                  # no level crosses it
                continue
            z00, z01, z11, z10 = _corners(z, i, j, j1)
            if not (np.isfinite(z00) and np.isfinite(z01) and np.isfinite(z11)
                    and np.isfinite(z10)):
                continue
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
def _fill(z, band, levels, per, starts, cases, saddle_high, pts, eids):
    ny, nx = z.shape
    nxe = nx + per                                      # the row length edge ids count in
    for i in prange(ny - 1):
        pos = np.empty(levels.size, np.int64)
        for lv in range(levels.size):
            pos[lv] = starts[lv, i]
        for j in range(nx - 1 + per):
            j1 = j + 1 if j + 1 < nx else 0
            a, b = _span(band, i, j, j1)
            if a == b:
                continue
            z00, z01, z11, z10 = _corners(z, i, j, j1)
            if not (np.isfinite(z00) and np.isfinite(z01) and np.isfinite(z11)
                    and np.isfinite(z10)):
                continue
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


@njit(cache=True, nogil=True)
def _meet(link, pend, stamp, slot, tag, q):
    """Endpoint q on the edge in slot: linked to the one waiting there
    (stamped tag), or left waiting."""
    if stamp[slot] == tag:
        p = pend[slot]
        link[q] = p
        link[p] = q
        stamp[slot] = -1
    else:
        pend[slot] = q
        stamp[slot] = tag


@njit(cache=True, nogil=True, parallel=True)
def _chain(pts, eids, lstart, nxe, out, ooff, nverts, npoly, links, seens):
    """Per level: segments [lstart[l], lstart[l+1]) joined into polylines,
    written at out[2 lstart[l]:] (vertices) and ooff[lstart[l] + l:]
    (polyline starts, relative); nverts/npoly the counts written. nxe:
    the row length of the edge ids (_fill's). links (2 nseg) and seens
    (nseg): working memory, the level's share at its segments'."""
    for lv in prange(lstart.size - 1):
        s0, s1 = lstart[lv], lstart[lv + 1]
        n = s1 - s0
        if n == 0:
            continue
        link = links[2 * s0:2 * s1]                     # endpoint -> the endpoint it meets
        link[:] = -1
        # an edge holds at most two endpoints, of the two cells beside it:
        # horizontal edges of grid row r wait in hpend[r % 2, c] (stamped r:
        # a cell row stores its top edges' while matching its bottom ones'),
        # the vertical ones of cell row r in vpend[c] (stamped r)
        hpend = np.empty(2 * (nxe + 1), np.int64)
        hstamp = np.full(2 * (nxe + 1), -1, np.int64)
        vpend = np.empty(nxe + 1, np.int64)
        vstamp = np.full(nxe + 1, -1, np.int64)
        for q in range(2 * n):
            key = eids[s0 + q // 2, q % 2]
            r, c = (key >> 1) // nxe, (key >> 1) % nxe
            if key & 1:
                _meet(link, vpend, vstamp, c, r, q)
            else:
                _meet(link, hpend, hstamp, (r & 1) * (nxe + 1) + c, r, q)
        seen = seens[s0:s1]
        seen[:] = False
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


@njit(cache=True, nogil=True)
def _higher_left(out, a, b, z, level, nx):
    """Whether polyline out[a:b] has z >= level on its left (x the column,
    y the row): read off its first segment that leaves a grid edge
    between a node at or above the level and one below."""
    for s in range(a, b - 1):
        r, c = out[s, 0], out[s, 1]
        fr, fc = np.floor(r), np.floor(c)
        if r == fr and c != fc:                         # on a row: nodes (r, fc), (r, fc + 1)
            ra, ca, rb, cb = r, fc, r, fc + 1.0
        elif c == fc and r != fr:                       # on a column
            ra, ca, rb, cb = fr, c, fr + 1.0, c
        else:
            continue
        za = z[np.int64(ra), np.int64(ca) % nx]
        zb = z[np.int64(rb), np.int64(cb) % nx]
        if (za >= level) == (zb >= level):
            continue
        hr, hc = (ra, ca) if za >= level else (rb, cb)
        dx, dy = out[s + 1, 1] - c, out[s + 1, 0] - r
        cross = dx * (hr - r) - dy * (hc - c)
        if cross != 0.0:
            return cross > 0.0
    return True


@njit(cache=True, nogil=True)
def _seg_quad(out, s):
    """The cell (row, column) segment out[s] -> out[s + 1] crosses."""
    return (np.int64(np.floor(0.5 * (out[s, 0] + out[s + 1, 0]))),
            np.int64(np.floor(0.5 * (out[s, 1] + out[s + 1, 1]))))


@njit(cache=True, nogil=True, parallel=True)
def _canonical(out, ooff, lstart, nverts, npoly, z, levels, nx, out2, ooff2):
    """_chain's polylines as matplotlib's contour generator (contourpy's
    mpl2014) gives them, which clabel's label places and dash phases
    follow: each with z above the level on its left; lines from the
    grid's edges first, by the cell they start in (rows, then columns
    across a row; a cell's south, west, north, east edge in that order),
    then the rings, by their lowest row's first cell, each starting on
    that cell's east edge. Written to out2 / ooff2, laid out as out / ooff."""
    for lv in prange(nverts.size):
        npl = npoly[lv]
        if npl == 0:
            continue
        vb, pb = 2 * lstart[lv], lstart[lv] + lv
        level = levels[lv]
        keys = np.empty(npl, np.int64)
        starts = np.empty(npl, np.int64)                # each ring's first vertex
        span = np.int64(nx + 2) * 4
        for k in range(npl):
            a = vb + ooff[pb + k]
            b = vb + (ooff[pb + k + 1] if k + 1 < npl else nverts[lv])
            if b - a < 2:
                keys[k] = 0
                starts[k] = 0
                continue
            if not _higher_left(out, a, b, z, level, nx):
                for q in range((b - a) // 2):
                    for d in range(2):
                        t = out[a + q, d]
                        out[a + q, d] = out[b - 1 - q, d]
                        out[b - 1 - q, d] = t
            ring = out[a, 0] == out[b - 1, 0] and out[a, 1] == out[b - 1, 1] and b - a > 2
            if ring:
                best, bs = np.int64(-1), a
                for s in range(a, b - 1):
                    qr, qc = _seg_quad(out, s)
                    kk = qr * span + qc * 4
                    if best < 0 or kk < best:
                        best, bs = kk, s
                qc = np.float64(_seg_quad(out, bs)[1] + 1)
                starts[k] = bs - a + 1 if out[bs + 1, 1] == qc and out[bs, 1] != qc else bs - a
                keys[k] = (np.int64(1) << 60) + best
            else:
                qr, qc = _seg_quad(out, a)
                r, c = out[a, 0], out[a, 1]
                side = 0 if r == qr else (1 if c == qc else (2 if r == qr + 1 else 3))
                keys[k] = qr * span + qc * 4 + side
                starts[k] = 0
        order = np.argsort(keys, kind='mergesort')
        nv = 0
        for m in range(npl):
            k = order[m]
            a = vb + ooff[pb + k]
            b = vb + (ooff[pb + k + 1] if k + 1 < npl else nverts[lv])
            ooff2[pb + m] = nv
            n = b - a
            t = starts[k]
            if t == 0 or n < 3:
                for q in range(n):
                    out2[vb + nv + q, 0] = out[a + q, 0]
                    out2[vb + nv + q, 1] = out[a + q, 1]
            else:                                       # a ring turned to start at vertex t
                u = n - 1
                for q in range(u):
                    out2[vb + nv + q, 0] = out[a + (t + q) % u, 0]
                    out2[vb + nv + q, 1] = out[a + (t + q) % u, 1]
                out2[vb + nv + u, 0] = out[a + t, 0]
                out2[vb + nv + u, 1] = out[a + t, 1]
            nv += n


@njit(cache=True, nogil=True)
def _interp1(x, fp):
    """np.interp(x, np.arange(fp.size), fp) of one x, value for value:
    numpy's arr_interp on an axis of whole numbers (its slope divides by
    exactly 1)."""
    n = fp.size
    if n == 1:                                          # numpy's one-point case: NaN too
        return fp[0]
    if np.isnan(x):
        return x
    if x < 0.0:
        return fp[0]
    if x >= np.float64(n - 1):
        return fp[n - 1]
    j = np.int64(np.floor(x))
    if np.float64(j) == x:
        return fp[j]
    slope = (fp[j + 1] - fp[j]) / (np.float64(j + 1) - np.float64(j))
    r = slope * (x - np.float64(j)) + fp[j]
    if np.isnan(r):
        r = slope * (x - np.float64(j + 1)) + fp[j + 1]
        if np.isnan(r) and fp[j] == fp[j + 1]:
            r = fp[j]
    return r


@njit(cache=True, nogil=True, parallel=True)
def _interp_index(f, fp, out):
    """np.interp(f, np.arange(fp.size), fp), value for value."""
    for i in prange(f.size):
        out[i] = _interp1(f[i], fp)


@njit(cache=True, nogil=True, parallel=True)
def _compact(out, ooff, lstart, nverts, npoly, order, vstart, pstart, rows, cols, axes,
             fa, fb, offsets, lof):
    """_chain's per-level blocks packed: level l's vertices from vstart[l]
    -- (row, column) into (fa, fb), or with axes the coordinates they stand
    for, np.interp along cols into fa and along rows into fb -- its
    polylines' starts from pstart[l], each of level order[l]."""
    for lv in prange(nverts.size):
        vb, pb = 2 * lstart[lv], lstart[lv] + lv
        v0, p0 = vstart[lv], pstart[lv]
        for k in range(nverts[lv]):
            r, c = out[vb + k, 0], out[vb + k, 1]
            if axes:
                fa[v0 + k] = _interp1(c, cols)
                fb[v0 + k] = _interp1(r, rows)
            else:
                fa[v0 + k] = r
                fb[v0 + k] = c
        for k in range(npoly[lv]):
            offsets[p0 + k] = ooff[pb + k] + v0
            lof[p0 + k] = order[lv]


def interp_index(f, fp):
    """np.interp(f, np.arange(len(fp)), fp), bit for bit, on every core."""
    f = np.ascontiguousarray(f, dtype=np.float64)
    out = np.empty(f.size, np.float64)
    if f.size:
        _interp_index(f, np.ascontiguousarray(fp, dtype=np.float64), out)
    return out


def lines(z, levels, periodic=False, cols=None, rows=None):
    """Contour lines of z at levels: (fi, fj, offsets, level_of) -- see the
    module docstring. periodic: the last column joins the first (a global
    grid), as if z had its first column again after its last. With cols
    and rows (the grid's 1-D axes): (x, y, offsets, level_of), x =
    interp_index(fj, cols), y = interp_index(fi, rows), made as the lines
    are packed. The working arrays are this thread's _arena buffers."""
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
    band = _arena.take('contour.band', (ny, nx), np.int16 if L < 32767 else np.int32)
    _bands(z, lv_sorted, band)
    counts = _arena.take('contour.counts', (L, ny - 1), np.int64)
    counts[:] = 0
    _count(z, band, lv_sorted, per, counts)
    flat = counts.ravel()
    starts = np.zeros(flat.size + 1, np.int64)
    np.cumsum(flat, out=starts[1:])
    nseg = int(starts[-1])
    if nseg == 0:
        return empty
    pts = _arena.take('contour.pts', (nseg, 4), np.float64)
    eids = _arena.take('contour.eids', (nseg, 2), np.int64)
    _fill(z, band, lv_sorted, per, starts[:-1].reshape(L, ny - 1), _CASES, _SADDLE_HIGH, pts,
          eids)
    lstart = np.ascontiguousarray(starts[::ny - 1])     # each level's first segment
    out = _arena.take('contour.out', (2 * nseg, 2), np.float64)
    ooff = _arena.take('contour.ooff', nseg + L, np.int64)
    nverts = np.zeros(L, np.int64)
    npoly = np.zeros(L, np.int64)
    _chain(pts, eids, lstart, nx + per, out, ooff, nverts, npoly,
           _arena.take('contour.links', 2 * nseg, np.int64),
           _arena.take('contour.seen', nseg, np.bool_))
    out2 = _arena.take('contour.out2', (2 * nseg, 2), np.float64)
    ooff2 = _arena.take('contour.ooff2', nseg + L, np.int64)
    _canonical(out, ooff, lstart, nverts, npoly, z, lv_sorted, nx, out2, ooff2)
    out, ooff = out2, ooff2
    # the per-level blocks packed into the results
    vstart = np.zeros(L + 1, np.int64)
    np.cumsum(nverts, out=vstart[1:])
    pstart = np.zeros(L + 1, np.int64)
    np.cumsum(npoly, out=pstart[1:])
    nv, npl = int(vstart[-1]), int(pstart[-1])
    fa, fb = np.empty(nv), np.empty(nv)
    offsets = np.empty(npl + 1, np.int64)
    lof = np.empty(npl, np.int64)
    axes = cols is not None
    dummy = np.zeros(1)
    _compact(out, ooff, lstart, nverts, npoly, order.astype(np.int64), vstart, pstart,
             np.ascontiguousarray(rows, np.float64) if axes else dummy,
             np.ascontiguousarray(cols, np.float64) if axes else dummy, axes, fa, fb, offsets,
             lof)
    offsets[npl] = nv
    return fa, fb, offsets, lof
