"""Contour labels as matplotlib's ContourLabeler places them: on each
line long enough for its level's label, at the middle of the straightest
block of the line's points (on a map, its visible ones), not too close to a label already
placed; the line cut around it (inline) and the label turned along it.

    xy, offsets, levels, texts = place(r, ax, cs, xy, offsets, levels)

The choices are matplotlib's, the numbers computed as numpy computes
them: the per-line statistics, the blocks' straightness (a row sum in
numpy's pairwise order) and np.interp along the line are numba kernels
written to the same arithmetic (tests/test_clabel.py holds them to
numpy), and every labelled line is placed and cut in one kernel (_place).
Two things can differ from numpy on some CPU in the last bit or in a
tie: libm's hypot and atan2 against numpy's vector loops, and the blocks
sorted by straightness -- a stable sort here, where numpy's quicksort
orders equal blocks as its CPU's sort code does.
"""
import numpy as np

from ._artists import _gather
from ._jit import njit, prange


@njit(cache=True, nogil=True, parallel=True)
def _line_stats(xy, poff, x0, y0, x1, y1, nvis, ptp):
    """Per line: its points inside the box (x0..x1, y0..y1) and their
    extent in x and y (0 when none), as the reduceats over the masked
    points give them."""
    for p in prange(poff.size - 1):
        n = 0
        mnx = mny = np.inf
        mxx = mxy = -np.inf
        for k in range(poff[p], poff[p + 1]):
            x, y = xy[k, 0], xy[k, 1]
            if np.isfinite(x) and np.isfinite(y) and x >= x0 and x <= x1 and y >= y0 and \
                    y <= y1:
                n += 1
                mnx, mxx = min(mnx, x), max(mxx, x)
                mny, mxy = min(mny, y), max(mxy, y)
        nvis[p] = n
        dx, dy = mxx - mnx, mxy - mny
        ptp[p, 0] = dx if np.isfinite(dx) else 0.0
        ptp[p, 1] = dy if np.isfinite(dy) else 0.0


@njit(cache=True, nogil=True)
def _visible(pts, x0, y0, x1, y1):
    """A line's points inside the box (x0..x1, y0..y1), finite: their
    indices and their x and y (pts[ins], as the boolean mask gives them)."""
    n = pts.shape[0]
    idx = np.empty(n, np.int64)
    m = 0
    for k in range(n):
        x, y = pts[k, 0], pts[k, 1]
        if np.isfinite(x) and np.isfinite(y) and x >= x0 and x <= x1 and y >= y0 and y <= y1:
            idx[m] = k
            m += 1
    vx = np.empty(m)
    vy = np.empty(m)
    for q in range(m):
        vx[q] = pts[idx[q], 0]
        vy[q] = pts[idx[q], 1]
    return idx[:m], vx, vy


@njit(cache=True, nogil=True)
def _arc(seg, arc):
    """arc[0] 0, then the running sum of seg with its non-finite lengths 0
    (np.cumsum's order: one add after another)."""
    arc[0] = 0.0
    acc = 0.0
    for k in range(seg.size):
        v = seg[k]
        if not np.isfinite(v):
            v = 0.0
        acc = v if k == 0 else acc + v
        arc[k + 1] = acc


@njit(cache=True, nogil=True)
def _leaf(a, i0, n):
    """numpy's pairwise_sum for n <= 128: eight running sums, then the rest."""
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[i0 + i]
        return res
    r0, r1, r2, r3 = a[i0], a[i0 + 1], a[i0 + 2], a[i0 + 3]
    r4, r5, r6, r7 = a[i0 + 4], a[i0 + 5], a[i0 + 6], a[i0 + 7]
    i = 8
    while i < n - (n % 8):
        r0 += a[i0 + i]
        r1 += a[i0 + i + 1]
        r2 += a[i0 + i + 2]
        r3 += a[i0 + i + 3]
        r4 += a[i0 + i + 4]
        r5 += a[i0 + i + 5]
        r6 += a[i0 + i + 6]
        r7 += a[i0 + i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[i0 + i]
        i += 1
    return res


@njit(cache=True, nogil=True)
def _pairwise(a, i0, n):
    """numpy's pairwise_sum of a[i0:i0+n]: halves (cut at a multiple of 8)
    summed apart and added, left + right, down to blocks of 128 -- the
    recursion unrolled onto a stack (numba's recursive functions do not
    survive its cache)."""
    if n <= 128:
        return _leaf(a, i0, n)
    si = np.empty(64, np.int64)
    sn = np.empty(64, np.int64)
    sv = np.empty(64)
    ss = np.zeros(64, np.int64)                         # 0 new, 1 left half pending, 2 right
    top = 0
    si[0], sn[0] = i0, n
    while True:
        m = sn[top]
        if ss[top] == 0 and m > 128:                    # split: the left half first
            ss[top] = 1
            h = m // 2
            h -= h % 8
            top += 1
            si[top], sn[top], ss[top] = si[top - 1], h, 0
            continue
        val = _leaf(a, si[top], m)
        top -= 1
        while True:                                     # hand val up the tree
            if top < 0:
                return val
            if ss[top] == 1:                            # the left half: now the right
                sv[top] = val
                ss[top] = 2
                h = sn[top] // 2
                h -= h % 8
                top += 1
                si[top], sn[top], ss[top] = si[top - 1] + h, sn[top - 1] - h, 0
                break
            val = sv[top] + val                         # both halves: left + right
            top -= 1


@njit(cache=True, nogil=True)
def _block_ends(vx, vy, nb, bs, dx, dy):
    """The blocks of np.resize(vis, (nb, bs)): their last point less their
    first, (xl - xf, yl - yf)."""
    n = vx.size
    for r in range(nb):
        f, l = (r * bs) % n, (r * bs + bs - 1) % n
        dx[r] = vx[l] - vx[f]
        dy[r] = vy[l] - vy[f]


@njit(cache=True, nogil=True, error_model='numpy')
def _straightness(vx, vy, nb, bs, ln, out):
    """matplotlib's per-block distance of the points from the chord,
    (abs(s) / ln).sum(axis=-1), summed in numpy's order."""
    n = vx.size
    row = np.empty(bs)
    for r in range(nb):
        f, l = (r * bs) % n, (r * bs + bs - 1) % n
        xf, xl, yf, yl = vx[f], vx[l], vy[f], vy[l]
        for c in range(bs):
            m = (r * bs + c) % n
            s = (yf - vy[m]) * (xl - xf) - (xf - vx[m]) * (yl - yf)
            row[c] = abs(s) / ln[r]
        out[r] = 0.0 + _pairwise(row, 0, bs)


@njit(cache=True, nogil=True, error_model='numpy')
def _interp(x, xp, fp):
    """np.interp(x, xp, fp) of one x (numpy's arr_interp; xp increasing,
    repeats allowed)."""
    n = xp.size
    if n == 1:
        return fp[0]
    if np.isnan(x):
        return x
    if x < xp[0]:
        return fp[0]
    if x > xp[n - 1]:
        return fp[n - 1]
    lo, hi = 0, n                                       # the last j with xp[j] <= x
    while lo < hi:
        mid = (lo + hi) >> 1
        if xp[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    j = lo - 1
    if j == n - 1 or xp[j] == x:
        return fp[j]
    slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
    r = slope * (x - xp[j]) + fp[j]
    if np.isnan(r):
        r = slope * (x - xp[j + 1]) + fp[j + 1]
        if np.isnan(r) and fp[j] == fp[j + 1]:
            r = fp[j]
    return r


@njit(cache=True, nogil=True, error_model='numpy')
def _at(arc, px, py, a):
    """The point at arc length a along a line."""
    return _interp(a, arc, px), _interp(a, arc, py)


@njit(cache=True, nogil=True)
def _before(a, b):
    """a sorts before b: a < b, NaN last (numpy's order)."""
    return a < b or (np.isnan(b) and not np.isnan(a))


@njit(cache=True, nogil=True)
def _argsort(a, out):
    """np.argsort(a) into out, stable: a bottom-up merge sort, equal keys
    in index order (numpy's for rows of 16 or fewer, its insertion sort)."""
    n = a.size
    for i in range(n):
        out[i] = i
    src, dst = out, np.empty(n, np.int64)
    width, flipped = 1, False
    while width < n:
        for lo in range(0, n, 2 * width):
            mid, hi = min(lo + width, n), min(lo + 2 * width, n)
            i, j, k = lo, mid, lo
            while i < mid and j < hi:
                if _before(a[src[j]], a[src[i]]):
                    dst[k] = src[j]
                    j += 1
                else:
                    dst[k] = src[i]
                    i += 1
                k += 1
            while i < mid:
                dst[k] = src[i]
                i += 1
                k += 1
            while j < hi:
                dst[k] = src[j]
                j += 1
                k += 1
        src, dst = dst, src
        width *= 2
        flipped = not flipped
    if flipped:
        out[:] = src


@njit(cache=True, nogil=True)
def _close(x0, y0, x1, y1):
    """np.allclose((x0, y0), (x1, y1)) of two finite points."""
    return abs(x0 - x1) <= 1e-08 + 1e-05 * abs(x1) and abs(y0 - y1) <= 1e-08 + 1e-05 * abs(y1)


@njit(cache=True, nogil=True, error_model='numpy')
def _rank(pts, L, bx0, by0, bx1, by1, ox, oy, ok):
    """locate_label's candidates on one line, label width L: the middles
    of its blocks of visible points, the straightest first -- their x, y
    and index into the line (ox, oy, ok). Returns how many."""
    ins, vx, vy = _visible(pts, bx0, by0, bx1, by1)
    n = vx.size
    nb = int(np.ceil(n / L)) if L > 1 else 1
    bs = n if nb == 1 else int(L)
    dx = np.empty(nb)
    dy = np.empty(nb)
    _block_ends(vx, vy, nb, bs, dx, dy)
    ln = np.empty(nb)
    for i in range(nb):
        ln[i] = np.hypot(dx[i], dy[i])
    dist = np.empty(nb)
    _straightness(vx, vy, nb, bs, ln, dist)
    order = np.empty(nb, np.int64)
    _argsort(dist, order)
    hb = bs // 2
    for i in range(nb):
        m = (order[i] * bs + hb) % n
        ox[i], oy[i], ok[i] = vx[m], vy[m], ins[m]
    return nb


@njit(cache=True, nogil=True, error_model='numpy')
def _cut(pts, k, L, spacing, inline, upright, out):
    """calc_label_rot_and_inline for a label at point k of a line: its
    angle, and the line left round it in out (at most len(pts) + 2
    points): (angle, e0, e1), the pieces out[:e0] and out[e0:e1]."""
    npt = pts.shape[0]
    seg = np.empty(npt - 1)
    for i in range(npt - 1):
        seg[i] = np.hypot(pts[i + 1, 0] - pts[i, 0], pts[i + 1, 1] - pts[i, 1])
    arc = np.empty(npt)
    _arc(seg, arc)
    px, py = pts[:, 0].copy(), pts[:, 1].copy()
    a0 = arc[k]
    half = L / 2.0
    xa, ya = _at(arc, px, py, a0 - half)
    xb, yb = _at(arc, px, py, a0 + half)
    ang = np.degrees(np.arctan2(-(yb - ya), xb - xa))
    if upright:
        ang = (ang + 90) % 180 - 90
    if not inline:
        out[:npt] = pts
        return ang, npt, npt
    gap = half + spacing
    lo, hi = a0 - gap, a0 + gap
    total = arc[-1]
    no = 0
    if _close(pts[0, 0], pts[0, 1], pts[-1, 0], pts[-1, 1]) and total > 2 * gap:
        # a ring: one open line from the gap's far side round to its near side
        out[no, 0], out[no, 1] = _at(arc, px, py, hi % total)
        no += 1
        if hi <= total and lo >= 0:
            last = -1
            for i in range(npt):
                if arc[i] > hi:
                    last = i
            for i in range(npt):
                if arc[i] > hi and i != last:
                    out[no] = pts[i]
                    no += 1
            first = True
            for i in range(npt):
                if arc[i] < lo:
                    if not first:
                        out[no] = pts[i]
                        no += 1
                    first = False
        else:
            for i in range(npt):
                if (arc[i] > hi - total and arc[i] < lo) if hi > total else \
                        (arc[i] > hi and arc[i] < lo + total):
                    out[no] = pts[i]
                    no += 1
        out[no, 0], out[no, 1] = _at(arc, px, py, lo % total)
        no += 1
        return ang, no, no
    if lo > 0:
        for i in range(npt):
            if arc[i] < lo:
                out[no] = pts[i]
                no += 1
        out[no, 0], out[no, 1] = _at(arc, px, py, lo)
        no += 1
    e0 = no
    if hi < total:
        out[no, 0], out[no, 1] = _at(arc, px, py, hi)
        no += 1
        for i in range(npt):
            if arc[i] > hi:
                out[no] = pts[i]
                no += 1
    return ang, e0, no


@njit(cache=True, nogil=True, parallel=True, error_model='numpy')
def _place(xy, poff, cand, lw, thresh, spacing, bx0, by0, bx1, by1, inline, upright,
           out_k, out_ang, out_xy, out_off, out_of):
    """matplotlib's locate_label, too_close and calc_label_rot_and_inline
    for the labelled lines cand, in order: each one's label point (out_k,
    an index into the line) and angle (out_ang), and the pieces of 2 or
    more points left of it (out_xy, starting at out_off, of the line
    out_of). The lines are ranked and cut in parallel, the labels chosen
    one after another (each kept from those before). Returns (points,
    pieces) out."""
    nc = cand.size
    base = np.zeros(nc + 1, np.int64)
    for c in range(nc):
        base[c + 1] = base[c] + poff[cand[c] + 1] - poff[cand[c]] + 2
    ox, oy = np.empty(base[nc]), np.empty(base[nc])
    ok = np.empty(base[nc], np.int64)
    nbs = np.empty(nc, np.int64)
    for c in prange(nc):
        b = base[c]
        nbs[c] = _rank(xy[poff[cand[c]]:poff[cand[c] + 1]], lw[c], bx0, by0, bx1, by1,
                       ox[b:], oy[b:], ok[b:])
    placed = np.empty((nc, 2))
    for c in range(nc):
        b = base[c]
        pick = b                                        # all too close: the straightest
        for i in range(b, b + nbs[c]):
            close = False
            for q in range(c):
                ex, ey = ox[i] - placed[q, 0], oy[i] - placed[q, 1]
                if ex ** 2.0 + ey ** 2.0 < thresh[c]:
                    close = True
                    break
            if not close:
                pick = i
                break
        placed[c, 0], placed[c, 1] = ox[pick], oy[pick]
        out_k[c] = ok[pick]
    buf = np.empty((base[nc], 2))
    ends = np.empty((nc, 2), np.int64)
    for c in prange(nc):
        ang, e0, e1 = _cut(xy[poff[cand[c]]:poff[cand[c] + 1]], out_k[c], lw[c], spacing,
                           inline, upright, buf[base[c]:])
        out_ang[c], ends[c, 0], ends[c, 1] = ang, e0, e1
    no = npc = 0
    for c in range(nc):
        s = 0
        for j in range(2):
            e = ends[c, j]
            if e - s >= 2:
                out_off[npc], out_of[npc] = no, c
                npc += 1
                out_xy[no:no + e - s] = buf[base[c] + s:base[c] + e]
                no += e - s
            s = e
    return no, npc


def place(r, ax, cs, xy, poff, lof):
    """Labels for the contour set cs on its projected lines (xy, poff, the
    level of each in lof): (xy, offsets, levels) of the lines left after
    cutting them round the labels, and the texts to draw, as r.text
    arguments."""
    cfg = cs.labels
    texts = []
    nparts = poff.size - 1
    if nparts == 0:
        return xy, poff, lof, texts
    size_pt = cfg['fontsize']
    # matplotlib's print_label, for every line at once: long enough (in
    # visible vertices, or in extent) for its level's label
    widths = {}
    for lv in cfg['levels']:
        ext = r.measure(cfg['texts'][lv], 0, 0, size_pt)
        widths[lv] = (ext[2] - ext[0]) if ext else 0.0
    lw_part = np.array([widths.get(int(v), 0.0) for v in range(int(lof.max()) + 1)])[lof]
    # matplotlib 3.11 ranks every point of a line, inside the axes or not
    # (a label off the axes is clipped away); on a map, whose lines are not
    # cartopy's point for point anyway, only the points on the map count
    from ._axes import MapAxes
    b = ax.box if isinstance(ax, MapAxes) else (-np.inf, -np.inf, np.inf, np.inf)
    xy = np.ascontiguousarray(xy, np.float64)
    nvis = np.empty(nparts, np.int64)
    ptp = np.empty((nparts, 2))
    _line_stats(xy, poff, float(b[0]), float(b[1]), float(b[2]), float(b[3]), nvis, ptp)
    cand = np.isin(lof, list(cfg['levels'])) & (lw_part > 0) & (nvis >= 2) & \
        ((nvis > 10 * lw_part) | (ptp > 1.2 * lw_part[:, None]).any(axis=1))
    # lines without a label, kept whole in one go
    rest = np.nonzero(~cand)[0]
    cand = np.nonzero(cand)[0]
    rest_xy = xy[_gather(poff, rest)]
    lv = lof[cand]
    lw = lw_part[cand]
    thresh = np.array([(1.2 * widths[v]) ** 2 for v in lv.tolist()])
    nc = cand.size
    out_k, out_ang = np.empty(nc, np.int64), np.empty(nc)
    out_xy = np.empty((int((np.diff(poff)[cand] + 2).sum()), 2))
    out_off, out_of = np.empty(2 * nc + 1, np.int64), np.empty(2 * nc, np.int64)
    no, npc = _place(xy, poff, cand, lw, thresh, float(cfg['spacing']), float(b[0]),
                     float(b[1]), float(b[2]), float(b[3]), bool(cfg['inline']),
                     bool(cfg['rightside_up']), out_k, out_ang, out_xy, out_off, out_of)
    at = (poff[cand] + out_k).tolist()
    for c, v in enumerate(lv.tolist()):
        i = at[c]
        texts.append(((cfg['texts'][v], xy[i, 0], xy[i, 1], size_pt, cfg['colors'][v]),
                      dict(ha='center', va='center', rotation=out_ang[c],
                           rotation_mode='anchor')))
    out_off[npc] = no
    keep_off = np.concatenate([[0], np.cumsum(np.diff(poff)[rest]),
                               rest_xy.shape[0] + out_off[1:npc + 1]])
    return (np.concatenate([rest_xy, out_xy[:no]]), keep_off.astype(np.int64),
            np.concatenate([lof[rest], lv[out_of[:npc]]]).astype(np.int64), texts)
