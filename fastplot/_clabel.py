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
libm's hypot and atan2 can differ from numpy's vector loops on some CPU in
the last bit. Blocks that all but tie, as symmetric data gives, are told
apart by that last bit of matplotlib's numbers: only then are its lines
fetched from contourpy and ranked by numpy's own sort (_matplotlib_lines).
"""
import numpy as np

from ._artists import _gather
from ._jit import njit, prange

_TIE = 1e-9                         # keys this close: the interpolation's rounding can turn them


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


@njit(cache=True, nogil=True, error_model='numpy')
def _blocks(pts, L, bx0, by0, bx1, by1):
    """locate_label's blocks of visible points on one line, label width L:
    (the points' indices, x, y, the block size, each block's distances)."""
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
    return ins, vx, vy, bs, dist


@njit(cache=True, nogil=True)
def _middles(order, ins, vx, vy, bs, ox, oy, ok):
    """The blocks' middles in the given order: x, y and index into the line
    (ox, oy, ok)."""
    n, hb = vx.size, bs // 2
    for i in range(order.size):
        m = (order[i] * bs + hb) % n
        ox[i], oy[i], ok[i] = vx[m], vy[m], ins[m]


@njit(cache=True, nogil=True, error_model='numpy')
def _rank(pts, L, bx0, by0, bx1, by1, ox, oy, ok):
    """locate_label's candidates on one line: the middles of its blocks,
    the straightest first (_middles). Returns how many, and whether any
    two keys all but tie."""
    ins, vx, vy, bs, dist = _blocks(pts, L, bx0, by0, bx1, by1)
    order = np.empty(dist.size, np.int64)
    _argsort(dist, order)
    _middles(order, ins, vx, vy, bs, ox, oy, ok)
    tied = False
    for i in range(dist.size - 1):
        a, b = dist[order[i]], dist[order[i + 1]]
        if a == b or b - a <= _TIE * (b + 1.0) or (np.isnan(a) and np.isnan(b)):
            tied = True
            break
    return dist.size, tied


@njit(cache=True, nogil=True, error_model='numpy')
def _distances(pts, L, bx0, by0, bx1, by1):
    return _blocks(pts, L, bx0, by0, bx1, by1)[4]


@njit(cache=True, nogil=True, error_model='numpy')
def _rerank(pts, L, bx0, by0, bx1, by1, order, ox, oy, ok):
    """_rank's candidates in an order given."""
    ins, vx, vy, bs, _ = _blocks(pts, L, bx0, by0, bx1, by1)
    _middles(order, ins, vx, vy, bs, ox, oy, ok)


@njit(cache=True, nogil=True, error_model='numpy')
def _cut(pts, k, L, spacing, inline, upright, out, noise=1e-9):
    """_split_path_and_get_label_rotation for a label at point k of a line:
    its angle, and the line left round it in out (at most len(pts) + 2
    points): (angle, e0, e1), the pieces out[:e0] and out[e0:e1]."""
    npt = pts.shape[0]
    closed = pts[0, 0] == pts[-1, 0] and pts[0, 1] == pts[-1, 1]
    cc = np.empty((npt, 2))
    if closed:                                          # a ring starts at the label
        for i in range(npt):
            cc[i] = pts[(k + i) % (npt - 1)]
        idx = 0
    else:
        cc[:] = pts
        idx = k
    seg = np.empty(npt - 1)
    for i in range(npt - 1):
        seg[i] = np.hypot(cc[i + 1, 0] - cc[i, 0], cc[i + 1, 1] - cc[i, 1])
    arc = np.empty(npt)
    _arc(seg, arc)
    arc -= arc[idx]
    px, py = cc[:, 0].copy(), cc[:, 1].copy()
    half = L / 2.0
    t0, t1 = -half, half
    if closed:                                          # from the far end of the ring
        t0 += arc[-1] - arc[0]
    xa, ya = _at(arc, px, py, t0)
    xb, yb = _at(arc, px, py, t1)
    dx = xb - xa
    if abs(dx) < noise:                                 # upright: a sign of noise turns it over
        dx = 0.0
    ang = np.degrees(np.arctan2(-(yb - ya), dx))
    if upright:
        ang = (ang + 90) % 180 - 90
    if not inline:
        out[:npt] = pts
        return ang, npt, npt
    t0 -= spacing
    t1 += spacing
    index = np.arange(npt, dtype=np.float64)
    i0 = i1 = -1
    if arc[0] <= t0 <= arc[-1]:
        i0 = np.int64(np.floor(_interp(t0, arc, index)))
    if arc[0] <= t1 <= arc[-1]:
        i1 = np.int64(np.ceil(_interp(t1, arc, index)))
    x0, y0 = _at(arc, px, py, t0)
    x1, y1 = _at(arc, px, py, t1)
    no = 0
    if closed:
        if i0 != -1 and i1 != -1:
            out[no, 0], out[no, 1] = x1, y1
            no += 1
            for i in range(i1, i0 + 1):
                out[no] = cc[i]
                no += 1
            out[no, 0], out[no, 1] = x0, y0
            no += 1
        return ang, no, no
    if i0 != -1:
        for i in range(i0 + 1):
            out[no] = cc[i]
            no += 1
        out[no, 0], out[no, 1] = x0, y0
        no += 1
    e0 = no
    if i1 != -1:
        out[no, 0], out[no, 1] = x1, y1
        no += 1
        for i in range(i1, npt):
            out[no] = cc[i]
            no += 1
    return ang, e0, no


@njit(cache=True, nogil=True, error_model='numpy')
def _exact_angle(pts, k, L, upright):
    """_cut's angle on a line whose every bit counts, y down."""
    return _cut(pts, k, L, 0.0, False, upright, np.empty((pts.shape[0] + 2, 2)), 0.0)[0]


@njit(cache=True, nogil=True, parallel=True, error_model='numpy')
def _rank_lines(xy, poff, cand, lw, bx0, by0, bx1, by1, base, ox, oy, ok, nbs, tied):
    """_rank for the labelled lines cand, in parallel, line c's candidates
    from base[c] on."""
    for c in prange(cand.size):
        b = base[c]
        nbs[c], tied[c] = _rank(xy[poff[cand[c]]:poff[cand[c] + 1]], lw[c], bx0, by0, bx1, by1,
                                ox[b:], oy[b:], ok[b:])


@njit(cache=True, nogil=True, parallel=True, error_model='numpy')
def _place(xy, poff, cand, lw, thresh, spacing, inline, upright, base, ox, oy, ok, nbs,
           out_k, out_ang, out_xy, out_off, out_of):
    """matplotlib's too_close and calc_label_rot_and_inline for the ranked
    lines cand, in order: each one's label point (out_k, an index into the
    line) and angle (out_ang), and the pieces of 2 or more points left of
    it (out_xy, starting at out_off, of the line out_of). The labels are
    chosen one after another (each kept from those before), the lines cut
    in parallel. Returns (points, pieces) out."""
    nc = cand.size
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


def _before_aspect(r, ax):
    """clabel works on the screen as it is when called, which on an axes of
    fixed aspect is before the first draw has fitted the box to it: labels
    are chosen, and their angles kept, in that stretched box. The drawn
    pixels' origin, the scale to the stretched box and its origin, or None
    where the box did not change."""
    o, p = getattr(ax, '_orig_pos', None), getattr(ax, '_pos', None)
    if o is None or p is None or (o.w == p.w and o.h == p.h):
        return None
    ox, oy = r.ox + o.l * r.W, r.oy + (1 - o.b - o.h) * r.H
    return (np.array(ax.box_f[:2]), np.array([o.w / p.w, o.h / p.h]), np.array([ox, oy]))


def _matplotlib_lines(ax, cs, level):
    """The lines of a level as matplotlib holds them on the screen, to the
    last bit: contourpy's vertices through transData's matrices (y up), or
    None without contourpy, or on axes other than plain linear ones."""
    try:
        import contourpy
    except ImportError:
        return None
    g = cs.grid
    if not hasattr(ax, 'get_xscale') or ax.get_xscale() != 'linear' or \
            ax.get_yscale() != 'linear':
        return None
    if g.rect:
        if g.xaxis.flip or g.yaxis.flip or g.xaxis.periodic:
            return None
        x, y = np.meshgrid(g.xaxis.coords, g.yaxis.coords)
    else:
        x, y = g.x2d, g.y2d
    z = np.ma.masked_invalid(cs.z.astype(np.float64))
    lines = contourpy.contour_generator(x, y, z, name='mpl2014', corner_mask=True,
                                        line_type=contourpy.LineType.SeparateCode).lines(level)[0]
    fig = ax.figure
    w, h = fig.get_size_inches()
    W, H = w * fig.dpi, h * fig.dpi
    b = getattr(ax, '_orig_pos', ax._pos)
    x0, x1, y0, y1 = b.l * W, (b.l + b.w) * W, b.b * H, (b.b + b.h) * H
    (l, r), (o, t) = ax.get_xlim(), ax.get_ylim()
    sx, sy = 1.0 / (r - l), 1.0 / (t - o)
    box = np.array([[x1 - x0, 0.0, x0], [0.0, y1 - y0, y0], [0.0, 0.0, 1.0]])
    lim = np.array([[sx, 0.0, -l * sx], [0.0, sy, -o * sy], [0.0, 0.0, 1.0]])
    m = np.dot(np.dot(box, lim), np.eye(3))             # transData, as it multiplies them
    return [np.c_[m[0, 0] * v[:, 0] + m[0, 2], m[1, 1] * v[:, 1] + m[1, 2]] for v in lines]


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
    # (a label off the axes is clipped away), and so does cartopy's clabel
    b = (-np.inf, -np.inf, np.inf, np.inf)
    xy = np.ascontiguousarray(xy, np.float64)
    view = _before_aspect(r, ax)
    screen = xy if view is None else (xy - view[0]) * view[1] + view[2]
    nvis = np.empty(nparts, np.int64)
    ptp = np.empty((nparts, 2))
    _line_stats(screen, poff, float(b[0]), float(b[1]), float(b[2]), float(b[3]), nvis, ptp)
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
    base = np.r_[0, np.cumsum(np.diff(poff)[cand] + 2)]
    ox, oy, ok = np.empty(base[nc]), np.empty(base[nc]), np.empty(base[nc], np.int64)
    nbs, tied = np.empty(nc, np.int64), np.empty(nc, np.bool_)
    box = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    _rank_lines(screen, poff, cand, lw, *box, base, ox, oy, ok, nbs, tied)
    theirs = {}

    def their_line(c):
        """Line c as matplotlib has it, or fastplot's where it can't be had."""
        k, v = int(cand[c]), int(lof[cand[c]])
        pts = screen[poff[k]:poff[k + 1]]
        if v not in theirs:
            theirs[v] = _matplotlib_lines(ax, cs, float(cs.levels[v])) or []
        nth = int(np.count_nonzero(lof[:k] == v))
        line = theirs[v][nth] if nth < len(theirs[v]) else pts
        if line.shape != pts.shape or np.abs(line[:, 0] - pts[:, 0]).max() > 1e-3:
            return pts, pts
        return pts, np.ascontiguousarray(line)

    for c in np.flatnonzero(tied).tolist():             # numpy's sort, on matplotlib's numbers
        pts, line = their_line(c)
        order = np.argsort(_distances(line, lw[c], *box))
        _rerank(pts, lw[c], *box, order, ox[base[c]:], oy[base[c]:], ok[base[c]:])
    no, npc = _place(screen, poff, cand, lw, thresh, float(cfg['spacing']), bool(cfg['inline']),
                     bool(cfg['rightside_up']), base, ox, oy, ok, nbs, out_k, out_ang, out_xy,
                     out_off, out_of)
    for c in np.flatnonzero(np.abs(np.abs(out_ang) - 90.0) < 1e-6).tolist():    # which way up
        pts, line = their_line(c)
        if line is not pts:
            out_ang[c] = _exact_angle(np.c_[line[:, 0], -line[:, 1]], out_k[c], lw[c],
                                      bool(cfg['rightside_up']))
    if view is not None:
        out_xy[:no] = (out_xy[:no] - view[2]) / view[1] + view[0]
    at = (poff[cand] + out_k).tolist()
    for c, v in enumerate(lv.tolist()):
        i = at[c]
        texts.append(((cfg['texts'][v], xy[i, 0], xy[i, 1], size_pt, cfg['colors'][v]),
                      dict(ha='center', va='center', rotation=out_ang[c],
                           rotation_mode='anchor', clip=ax.box_clip)))
    out_off[npc] = no
    keep_off = np.concatenate([[0], np.cumsum(np.diff(poff)[rest]),
                               rest_xy.shape[0] + out_off[1:npc + 1]])
    return (np.concatenate([rest_xy, out_xy[:no]]), keep_off.astype(np.int64),
            np.concatenate([lof[rest], lv[out_of[:npc]]]).astype(np.int64), texts)
