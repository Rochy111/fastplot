"""Contour labels as matplotlib's ContourLabeler places them: on each
line long enough for its level's label, at the middle of the straightest
block of the line's visible points, not too close to a label already
placed; the line cut around it (inline) and the label turned along it.

    xy, offsets, levels, texts = place(r, ax, cs, xy, offsets, levels)

The choices are matplotlib's to the bit, so the numbers are computed as
numpy computes them: the per-line statistics, the blocks' straightness
(a row sum in numpy's pairwise order) and np.interp along the line are
numba kernels written to the same arithmetic (tests/test_clabel.py holds
them to numpy); hypot, arctan2, argsort and the distance test between
labels stay numpy's own calls, whose results can depend on the CPU's
vector code path or sort order.
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


def _close(a, b):
    """np.allclose(a, b) of two finite points, without its overhead."""
    return (abs(a[0] - b[0]) <= 1e-08 + 1e-05 * abs(b[0]) and
            abs(a[1] - b[1]) <= 1e-08 + 1e-05 * abs(b[1]))


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
    b = ax.box
    xy = np.ascontiguousarray(xy, np.float64)
    nvis = np.empty(nparts, np.int64)
    ptp = np.empty((nparts, 2))
    _line_stats(xy, poff, float(b[0]), float(b[1]), float(b[2]), float(b[3]), nvis, ptp)
    cand = np.isin(lof, list(cfg['levels'])) & (lw_part > 0) & (nvis >= 2) & \
        ((nvis > 10 * lw_part) | (ptp > 1.2 * lw_part[:, None]).any(axis=1))
    # lines without a label, kept whole in one go
    rest = np.nonzero(~cand)[0]
    keep_xy, keep_off, keep_lof = [xy[_gather(poff, rest)]], [0], []
    keep_off.extend(np.cumsum(np.diff(poff)[rest]).tolist())
    keep_lof.extend(lof[rest].tolist())
    placed = np.empty((int(cand.sum()), 2))             # labels so far (too_close)
    nplaced = 0
    for p in np.nonzero(cand)[0]:
        pts = xy[poff[p]:poff[p + 1]]
        lv = lof[p]
        lw = widths[lv]
        ins = np.isfinite(pts).all(axis=1) & (pts[:, 0] >= b[0]) & (pts[:, 0] <= b[2]) & \
            (pts[:, 1] >= b[1]) & (pts[:, 1] <= b[3])
        vis = pts[ins]
        vx, vy = vis[:, 0].copy(), vis[:, 1].copy()
        # matplotlib's locate_label on the visible points
        n = vis.shape[0]
        nb = int(np.ceil(n / lw)) if lw > 1 else 1
        bs = n if nb == 1 else int(lw)
        dx = np.empty(nb)
        dy = np.empty(nb)
        _block_ends(vx, vy, nb, bs, dx, dy)
        ln = np.hypot(dx, dy)
        dist = np.empty(nb)
        _straightness(vx, vy, nb, bs, ln, dist)
        hb = bs // 2
        order = np.argsort(dist)
        thresh = (1.2 * lw) ** 2
        q = placed[:nplaced]
        two = np.full(nplaced, 2.0)                     # an array exponent: libm pow, as the
        for idx in order:                               # scalar (x0 - qx) ** 2 it stands for
            m = (idx * bs + hb) % n
            x0, y0 = vx[m], vy[m]
            if not (np.power(x0 - q[:, 0], two) + np.power(y0 - q[:, 1], two) < thresh).any():
                break
        else:                                           # all too close: the straightest
            idx = order[0]
            m = (idx * bs + hb) % n
            x0, y0 = vx[m], vy[m]
        placed[nplaced] = x0, y0
        nplaced += 1
        # the label's position and angle along the whole line
        k = int(np.nonzero(ins)[0][m])
        seg = np.hypot(*np.diff(pts, axis=0).T)
        seg[~np.isfinite(seg)] = 0.0
        arc = np.empty(seg.size + 1)
        arc[0] = 0.0
        np.cumsum(seg, out=arc[1:])
        px, py = pts[:, 0].copy(), pts[:, 1].copy()
        a0 = arc[k]
        half = lw / 2.0
        (xa, ya), (xb, yb) = _at(arc, px, py, a0 - half), _at(arc, px, py, a0 + half)
        ang = np.degrees(np.arctan2(np.float64(-(yb - ya)), np.float64(xb - xa)))
        if cfg['rightside_up']:
            ang = (ang + 90) % 180 - 90
        texts.append(((cfg['texts'][lv], pts[k, 0], pts[k, 1], size_pt, cfg['colors'][lv]),
                      dict(ha='center', va='center', rotation=ang, rotation_mode='anchor')))
        pieces = [pts]
        if cfg['inline']:
            gap = half + cfg['spacing']
            lo, hi = a0 - gap, a0 + gap
            total = arc[-1]
            if _close(pts[0], pts[-1]) and total > 2 * gap:
                # a ring: one open line from the gap's far side round to its near side
                if hi <= total and lo >= 0:
                    mid = np.concatenate([np.nonzero(arc > hi)[0][:-1],
                                          np.nonzero(arc < lo)[0][1:]])
                elif hi > total:
                    mid = np.nonzero((arc > hi - total) & (arc < lo))[0]
                else:
                    mid = np.nonzero((arc > hi) & (arc < lo + total))[0]
                pieces = [np.concatenate([[_at(arc, px, py, hi % total)], pts[mid],
                                          [_at(arc, px, py, lo % total)]])]
            else:
                pieces = []
                if lo > 0:
                    pieces.append(np.concatenate([pts[arc < lo], [_at(arc, px, py, lo)]]))
                if hi < total:
                    pieces.append(np.concatenate([[_at(arc, px, py, hi)], pts[arc > hi]]))
        for pc in pieces:
            if pc.shape[0] >= 2:
                keep_xy.append(pc)
                keep_off.append(keep_off[-1] + pc.shape[0])
                keep_lof.append(lv)
    return (np.concatenate(keep_xy), np.asarray(keep_off, np.int64),
            np.asarray(keep_lof, np.int64), texts)
