"""Agg's rasterizer, as matplotlib draws with it: strokes become outlines
(math_stroke, vcgen_stroke, vcgen_dash) and outlines become coverage by
signed cell areas (rasterizer_cells_aa, rasterizer_sl_clip,
rasterizer_scanline_aa), in Agg's own 24.8 fixed point, so a pixel gets what
Agg gives it to the unit. The results it depends on:

- an outline crossing itself adds up: where a stroke meets itself inside a
  pixel, as dense coastlines do at every vertex, the signed areas sum, which
  is why dense lines look darker than one stroke's coverage would;
- a stroke's inner joins fold back, and the folds cancel by their sign.

    outline = stroke(xy, offsets, closed, styles)      # polylines -> rings
    fill(canvas, *outline, ...)                        # rings -> pixels

Agg's cells are sorted lists. Here they are one dense array of cover and area
side by side, because a pixel row's cells are summed by x anyway: the same
sums, no sort, and a bit per cell so that a sweep finds what was touched
without walking the row. Rows go in bands, each on its own core. An edge's cells
on a row depend only on where it crosses that row, which Agg's stepping
reaches by whole divisions, so a band can start part way down an edge and
get the very cells Agg would (x at crossing k = x1 + floor((p + k P) / dy)).
Within a band paths run in order, each swept and blended before the next,
so overlapping paths stack as they were drawn.
"""
import threading

import numpy as np

from . import _arena, _jit
from ._jit import njit, prange
from ._path import _walk as _path_walk

BUTT, PROJECTING, ROUND = 1, 2, 3                  # as _draw's cap styles
JOIN_MITER, JOIN_ROUND, JOIN_BEVEL = 0, 1, 2        # miter: Agg's miter_join_revert
JOINS = {'miter': JOIN_MITER, 'round': JOIN_ROUND, 'bevel': JOIN_BEVEL}

_EPS = 1e-14                        # vertex_dist_epsilon
_ISECT = 1e-30                      # intersection_epsilon
_INNER_MITER = 1.01                 # m_inner_miter_limit
BAND_ROWS = 8                       # rows a band at least
BAND_TASKS = 8                      # bands a thread
_OUT = np.int32(-(1 << 31))                         # a point off the clip (prepare)
_DEBRUIJN = np.int64(0x03f79d71b4cb0a89)            # lowest set bit -> its number, by table
_LOG2 = np.zeros(64, np.int64)
for _k in range(64):
    _LOG2[(((1 << _k) * 0x03f79d71b4cb0a89) & ((1 << 64) - 1)) >> 58] = _k


# ---------------------------------------------------------------- stroking

@njit(cache=True, nogil=True, error_model='numpy')
def _dist(src, i, j):
    return np.sqrt((src[j, 0] - src[i, 0]) ** 2 + (src[j, 1] - src[i, 1]) ** 2)


@njit(cache=True, nogil=True, error_model='numpy')
def _keep(src, i, j):
    """vertex_dist::operator(): v[i].dist to v[j], and whether they differ."""
    d = _dist(src, i, j)
    ok = d > _EPS
    src[i, 2] = d if ok else 1.0 / _EPS
    return ok


@njit(cache=True, nogil=True, error_model='numpy')
def _seq_add(src, a, n, x, y):
    """vertex_sequence::add, the sequence being src[a:a + n]."""
    if n > 1 and not _keep(src, a + n - 2, a + n - 1):
        n -= 1
    src[a + n, 0], src[a + n, 1], src[a + n, 2] = x, y, 0.0
    return n + 1


@njit(cache=True, nogil=True, error_model='numpy')
def _sequence(xy, a, b, closed, src):
    """The polyline xy[a:b] as Agg's vertex_sequence holds it, in src[a:]
    (x, y, distance to the next): add() of every vertex, close(). Returns
    the count."""
    n = 0
    for k in range(a, b):
        n = _seq_add(src, a, n, xy[k, 0], xy[k, 1])
    while n > 1:
        if _keep(src, a + n - 2, a + n - 1):
            break
        x, y = src[a + n - 1, 0], src[a + n - 1, 1]
        n -= 2
        n = _seq_add(src, a, n, x, y)
    if closed:
        while n > 1:
            if _keep(src, a + n - 1, a):
                break
            n -= 1
    return n


@njit(cache=True, nogil=True, error_model='numpy')
def _put(out, p, write, x, y):
    if write:
        out[p, 0], out[p, 1] = x, y
    return p + 1


@njit(cache=True, nogil=True, error_model='numpy')
def _intersection(ax, ay, bx, by, cx, cy, dx, dy):
    """calc_intersection: (found, x, y)."""
    num = (ay - cy) * (dx - cx) - (ax - cx) * (dy - cy)
    den = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx)
    if abs(den) < _ISECT:
        return False, 0.0, 0.0
    u = num / den
    return True, ax + u * (bx - ax), ay + u * (by - ay)


@njit(cache=True, nogil=True, error_model='numpy')
def _cross(x1, y1, x2, y2, x, y):
    return (x - x2) * (y2 - y1) - (y - y2) * (x2 - x1)


@njit(cache=True, nogil=True, error_model='numpy')
def _arc(out, p, write, x, y, dx1, dy1, dx2, dy2, w):
    """calc_arc, of a positive width. Most joins of a thin line turn by less
    than one step of the arc, and then only its ends are written, whatever
    the angles are: the cosine of the turn says so without them. (Its margin
    is far wider than the rounding of the angles that Agg compares.)"""
    c = w / (w + 0.125)
    cross, dot = dx1 * dy2 - dy1 * dx2, dx1 * dx2 + dy1 * dy2
    ww = w * w
    if dot > 0.0 and cross > 1e-9 * ww and dot > (2.0 * c * c - 1.0 + 1e-6) * ww:
        p = _put(out, p, write, x + dx1, y + dy1)
        return _put(out, p, write, x + dx2, y + dy2)
    a1 = np.arctan2(dy1, dx1)
    a2 = np.arctan2(dy2, dx2)
    da = np.arccos(w / (w + 0.125)) * 2.0
    p = _put(out, p, write, x + dx1, y + dy1)
    if a1 > a2:
        a2 += 2.0 * np.pi
    n = np.int64((a2 - a1) / da)
    da = (a2 - a1) / (n + 1)
    a1 += da
    for _ in range(n):
        p = _put(out, p, write, x + np.cos(a1) * w, y + np.sin(a1) * w)
        a1 += da
    return _put(out, p, write, x + dx2, y + dy2)


@njit(cache=True, nogil=True, error_model='numpy')
def _miter(out, p, write, x0, y0, x1, y1, x2, y2, dx1, dy1, dx2, dy2, mlimit, w):
    """calc_miter of a miter_join_revert (matplotlib's, and inner joins')."""
    lim = w * mlimit
    exceeded = True
    found, xi, yi = _intersection(x0 + dx1, y0 - dy1, x1 + dx1, y1 - dy1, x1 + dx2, y1 - dy2,
                                  x2 + dx2, y2 - dy2)
    if found:
        di = np.sqrt((xi - x1) ** 2 + (yi - y1) ** 2)
        if di <= lim:
            p = _put(out, p, write, xi, yi)
            exceeded = False
    else:                                               # the three in one straight line
        xq, yq = x1 + dx1, y1 - dy1
        if (_cross(x0, y0, x1, y1, xq, yq) < 0.0) == (_cross(x1, y1, x2, y2, xq, yq) < 0.0):
            p = _put(out, p, write, xq, yq)             # it goes straight on
            exceeded = False
    if exceeded:
        p = _put(out, p, write, x1 + dx1, y1 - dy1)
        p = _put(out, p, write, x1 + dx2, y1 - dy2)
    return p


@njit(cache=True, nogil=True, error_model='numpy')
def _join(out, p, write, src, i0, i1, i2, len1, len2, w, join, mlimit):
    """calc_join: the outline at v1, between v0-v1 and v1-v2 (lengths len1
    and len2), on this side (inner_miter, Agg's default inner join)."""
    x0, y0, x1, y1, x2, y2 = (src[i0, 0], src[i0, 1], src[i1, 0], src[i1, 1], src[i2, 0],
                              src[i2, 1])
    dx1, dy1 = w * (y1 - y0) / len1, w * (x1 - x0) / len1
    dx2, dy2 = w * (y2 - y1) / len2, w * (x2 - x1) / len2
    cp = _cross(x0, y0, x1, y1, x2, y2)
    if cp != 0 and cp > 0:                              # inner join
        limit = min(len1, len2) / w
        if limit < _INNER_MITER:
            limit = _INNER_MITER
        return _miter(out, p, write, x0, y0, x1, y1, x2, y2, dx1, dy1, dx2, dy2, limit, w)
    dx, dy = 0.5 * (dx1 + dx2), 0.5 * (dy1 + dy2)
    if join == JOIN_ROUND or join == JOIN_BEVEL:
        # nearly straight: one point stands for the bevel's two
        if w - np.sqrt(dx * dx + dy * dy) < w / 1024.0:
            found, xi, yi = _intersection(x0 + dx1, y0 - dy1, x1 + dx1, y1 - dy1, x1 + dx2,
                                          y1 - dy2, x2 + dx2, y2 - dy2)
            if found:
                return _put(out, p, write, xi, yi)
            return _put(out, p, write, x1 + dx1, y1 - dy1)
    if join == JOIN_MITER:
        return _miter(out, p, write, x0, y0, x1, y1, x2, y2, dx1, dy1, dx2, dy2, mlimit, w)
    if join == JOIN_ROUND:
        return _arc(out, p, write, x1, y1, dx1, -dy1, dx2, -dy2, w)
    p = _put(out, p, write, x1 + dx1, y1 - dy1)
    return _put(out, p, write, x1 + dx2, y1 - dy2)


@njit(cache=True, nogil=True, error_model='numpy')
def _cap(out, p, write, src, i0, i1, ln, w, cap):
    """calc_cap: the end at v0 of the line to v1."""
    x0, y0 = src[i0, 0], src[i0, 1]
    dx1, dy1 = w * (src[i1, 1] - y0) / ln, w * (src[i1, 0] - x0) / ln
    if cap != ROUND:
        dx2, dy2 = (dy1, dx1) if cap == PROJECTING else (0.0, 0.0)
        p = _put(out, p, write, x0 - dx1 - dx2, y0 + dy1 - dy2)
        return _put(out, p, write, x0 + dx1 - dx2, y0 - dy1 - dy2)
    da = np.arccos(w / (w + 0.125)) * 2.0
    n = np.int64(np.pi / da)
    da = np.pi / (n + 1)
    p = _put(out, p, write, x0 - dx1, y0 + dy1)
    a1 = np.arctan2(dy1, -dx1) + da
    for _ in range(n):
        p = _put(out, p, write, x0 + np.cos(a1) * w, y0 + np.sin(a1) * w)
        a1 += da
    return _put(out, p, write, x0 + dx1, y0 - dy1)


@njit(cache=True, nogil=True, error_model='numpy')
def _stroke_steps(src, offsets, counts, closed, part_style, widths, caps, joins, limits, q, s, q1,
                  s1, out, rings, ring_part, p, k, write):
    """vcgen_stroke, as steps -- a part's caps and joins in the order they
    are written round it -- from step s of part q up to step s1 of part q1:
    points from out[p], rings from rings[k]. An open part has 2n - 2 steps
    (cap, joins forward, cap, joins back), one ring; a closed part 2n, two
    rings of n. Returns (p, k)."""
    while q < q1 or (q == q1 and s < s1):
        n = counts[q]
        if s >= (2 * n if closed[q] else 2 * n - 2):
            q += 1
            s = 0
            continue
        st = part_style[q]
        w = 0.5 * widths[st]
        cap, join, ml = caps[st], joins[st], limits[st]
        a = offsets[q]
        if s == 0 or (closed[q] and s == n):
            if write:
                rings[k] = p
                ring_part[k] = q
            k += 1
        if closed[q]:
            if s < n:
                i = a + s
                ia, ib = a + (s + n - 1) % n, a + (s + 1) % n
                p = _join(out, p, write, src, ia, i, ib, src[ia, 2], src[i, 2], w, join, ml)
            else:
                m = 2 * n - 1 - s
                i = a + m
                ia, ib = a + (m + n - 1) % n, a + (m + 1) % n
                p = _join(out, p, write, src, ib, i, ia, src[i, 2], src[ia, 2], w, join, ml)
        elif s == 0:
            p = _cap(out, p, write, src, a, a + 1, src[a, 2], w, cap)
        elif s < n - 1:
            i = a + s
            p = _join(out, p, write, src, i - 1, i, i + 1, src[i - 1, 2], src[i, 2], w, join, ml)
        elif s == n - 1:
            p = _cap(out, p, write, src, a + n - 1, a + n - 2, src[a + n - 2, 2], w, cap)
        else:
            i = a + 2 * n - 2 - s
            p = _join(out, p, write, src, i + 1, i, i - 1, src[i, 2], src[i - 1, 2], w, join, ml)
        s += 1
    return p, k


@njit(cache=True, nogil=True, error_model='numpy')
def _sequence_parts(xy, offsets, closed, src, counts, q):
    """Part q as Agg holds it, its count, and whether it is closed: a part
    of fewer than three points is not, and one point is not a line."""
    n = _sequence(xy, offsets[q], offsets[q + 1], closed[q], src)
    counts[q] = n if n >= 2 else 0
    closed[q] = closed[q] and n >= 3


@njit(cache=True, nogil=True, parallel=True, error_model='numpy')
def _sequences(xy, offsets, closed, src, counts):
    for q in prange(offsets.size - 1):
        _sequence_parts(xy, offsets, closed, src, counts, q)


@njit(cache=True, nogil=True, error_model='numpy')
def _sequences_serial(xy, offsets, closed, src, counts):
    for q in range(offsets.size - 1):
        _sequence_parts(xy, offsets, closed, src, counts, q)


@njit(cache=True, nogil=True, parallel=True, error_model='numpy')
def _stroke_count(src, offsets, counts, closed, part_style, widths, caps, joins, limits, cq, cs,
                  npts, nrings):
    d2 = np.empty((0, 2), np.float64)
    d1 = np.empty(0, np.int64)
    for c in prange(cq.size - 1):
        p, k = _stroke_steps(src, offsets, counts, closed, part_style, widths, caps, joins, limits,
                             cq[c], cs[c], cq[c + 1], cs[c + 1], d2, d1, d1, 0, 0, False)
        npts[c + 1], nrings[c + 1] = p, k


@njit(cache=True, nogil=True, parallel=True, error_model='numpy')
def _stroke_fill(src, offsets, counts, closed, part_style, widths, caps, joins, limits, cq, cs,
                 npts, nrings, out, rings, ring_part):
    for c in prange(cq.size - 1):
        _stroke_steps(src, offsets, counts, closed, part_style, widths, caps, joins, limits,
                      cq[c], cs[c], cq[c + 1], cs[c + 1], out, rings, ring_part, npts[c],
                      nrings[c], True)


SERIAL_STROKE = 4096                # vertices below which stroking stays on one thread
CHUNK = 2048                        # steps a task (stroking)
PART_TASKS = 4                      # stretches of parts a thread (clipping, dashing)


def stroke(xy, offsets, closed, part_style, widths, caps, joins, limits):
    """Polylines (xy (n, 2) px, parts by offsets, closed (m,) bool, part_style
    (m,)) as Agg strokes them, by style (widths px, caps, joins, miter
    limits): (points (N, 2), ring offsets (R + 1), the part of each ring).
    The tasks cut the steps of all parts evenly, so a single long line runs
    on every core too."""
    m = offsets.size - 1
    n = xy.shape[0]
    src = np.empty((n, 3))
    counts = np.empty(m, np.int64)
    closed = np.array(closed, np.bool_)
    d2 = np.empty((0, 2), np.float64)
    d1 = np.empty(0, np.int64)
    if n < SERIAL_STROKE or m < 1:
        _sequences_serial(xy, offsets, closed, src, counts)
        counts[widths[part_style] <= 0] = 0
        p, k = _stroke_steps(src, offsets, counts, closed, part_style, widths, caps, joins, limits,
                             0, 0, m, 0, d2, d1, d1, 0, 0, False)
        out = np.empty((p, 2))
        rings = np.empty(k + 1, np.int64)
        ring_part = np.empty(k, np.int64)
        _stroke_steps(src, offsets, counts, closed, part_style, widths, caps, joins, limits, 0, 0,
                      m, 0, out, rings, ring_part, 0, 0, True)
        rings[k] = p
        return out, rings, ring_part
    _sequences(xy, offsets, closed, src, counts)
    counts[widths[part_style] <= 0] = 0
    steps = np.where(closed, 2 * counts, np.maximum(2 * counts - 2, 0))
    before = np.cumsum(steps)
    total = int(before[-1])
    at = np.arange(0, total, CHUNK, dtype=np.int64)
    cq = np.searchsorted(before, at, 'right')
    cs = at - np.where(cq > 0, before[np.maximum(cq - 1, 0)], 0)
    cq, cs = np.concatenate((cq, [m])), np.concatenate((cs, [0]))
    npts = np.zeros(cq.size, np.int64)
    nrings = np.zeros(cq.size, np.int64)
    _stroke_count(src, offsets, counts, closed, part_style, widths, caps, joins, limits, cq, cs,
                  npts, nrings)
    np.cumsum(npts, out=npts)
    np.cumsum(nrings, out=nrings)
    out = np.empty((npts[-1], 2))
    rings = np.empty(nrings[-1] + 1, np.int64)
    ring_part = np.empty(nrings[-1], np.int64)
    _stroke_fill(src, offsets, counts, closed, part_style, widths, caps, joins, limits, cq, cs,
                 npts, nrings, out, rings, ring_part)
    rings[-1] = npts[-1]
    return out, rings, ring_part


# ---------------------------------------------------------------- clipping

@njit(cache=True, nogil=True, error_model='numpy')
def _flags(x, y, W, H):
    return (1 if x > W + 1.0 else 0) | (2 if y > H + 1.0 else 0) | (4 if x < -1.0 else 0) | \
        (8 if y < -1.0 else 0)


@njit(cache=True, nogil=True, error_model='numpy')
def _move_point(x, y, x1, y1, x2, y2, W, H, flags):
    """agg::clip_move_point: (ok, x, y), (x, y) the end of the segment
    whose flags these are, moved to the box."""
    if flags & 5:
        if x1 == x2:
            return False, 0.0, 0.0
        bound = -1.0 if flags & 4 else W + 1.0
        y = (bound - x1) * (y2 - y1) / (x2 - x1) + y1
        x = bound
    flags = _flags(x, y, W, H)
    if flags & 10:
        if y1 == y2:
            return False, 0.0, 0.0
        bound = -1.0 if flags & 8 else H + 1.0
        x = (bound - y1) * (x2 - x1) / (y2 - y1) + x1
        y = bound
    return True, x, y


@njit(cache=True, nogil=True, error_model='numpy')
def _clip_segment(x1, y1, x2, y2, W, H):
    """agg::clip_line_segment in matplotlib's box, a pixel past the canvas:
    (moved, x1, y1, x2, y2), moved 0 unchanged, bit 1 the first point
    moved, bit 2 the second, 4 nothing left."""
    f1, f2 = _flags(x1, y1, W, H), _flags(x2, y2, W, H)
    if f1 == 0 and f2 == 0:
        return 0, x1, y1, x2, y2
    if ((f1 & 5) != 0 and (f1 & 5) == (f2 & 5)) or ((f1 & 10) != 0 and (f1 & 10) == (f2 & 10)):
        return 4, x1, y1, x2, y2
    ax, ay, bx, by = x1, y1, x2, y2
    moved = 0
    if f1:
        ok, ax, ay = _move_point(x1, y1, x1, y1, x2, y2, W, H, f1)
        if not ok or (ax == x2 and ay == y2):
            return 4, x1, y1, x2, y2
        moved |= 1
    if f2:
        ok, bx, by = _move_point(x2, y2, x1, y1, x2, y2, W, H, f2)
        if not ok or (ax == bx and ay == by):
            return 4, x1, y1, x2, y2
        moved |= 2
    return moved, ax, ay, bx, by


@njit(cache=True, nogil=True, error_model='numpy')
def _clip_all(xy, offsets, closed, W, H, q0, q1, out, off, part, cl, p, k, write):
    """matplotlib's PathClipper over parts q0..q1: what is left of each, a
    new part wherever the line comes back in. Closing a ring it never cut
    leaves it a ring. Written (write) or counted, from points p and parts k:
    (points, parts)."""
    for q in range(q0, q1):
        a, b = offsets[q], offsets[q + 1]
        moveto = True
        cut = False
        k0 = k
        end = b - 1 if closed[q] else b                 # (a ring ends on its first point)
        for i in range(a + 1, end + (1 if closed[q] else 0)):
            j = a if closed[q] and i == end else i
            moved, x0, y0, x1, y1 = _clip_segment(xy[i - 1, 0], xy[i - 1, 1], xy[j, 0], xy[j, 1],
                                                  W, H)
            cut = cut or moved != 0
            if moved < 4:
                if (moved & 1) != 0 or moveto:
                    if write:
                        off[k] = p
                        part[k] = q
                        cl[k] = False
                        out[p, 0], out[p, 1] = x0, y0
                    p += 1
                    k += 1
                if write:
                    out[p, 0], out[p, 1] = x1, y1
                p += 1
                moveto = False
        if closed[q] and not cut and k == k0 + 1 and write:
            cl[k0] = True
    return p, k


@njit(cache=True, nogil=True, parallel=True, error_model='numpy')
def _clip_tasks(xy, offsets, closed, W, H, bounds, out, off, part, cl, ps, ks, write):
    """_clip_all of the parts between bounds, each stretch on a core:
    counting, ps and ks get what each stretch comes to (from their [t + 1]);
    writing, they say where each starts."""
    for t in prange(bounds.size - 1):
        p, k = _clip_all(xy, offsets, closed, W, H, bounds[t], bounds[t + 1], out, off, part, cl,
                         ps[t] if write else 0, ks[t] if write else 0, write)
        if not write:
            ps[t + 1], ks[t + 1] = p, k


def _part_tasks(offsets):
    """Parts cut into stretches of about equal points, one stretch a task:
    (the bounds, as part numbers; or None, if it is not worth a parallel
    region)."""
    n = int(offsets[-1])
    team = _jit.team()
    if n < SERIAL_STROKE or team < 2 or offsets.size < 3:
        return None
    at = np.linspace(0, n, PART_TASKS * team + 1).astype(np.int64)
    return np.unique(np.searchsorted(offsets, at))


def clip_paths(xy, offsets, closed, W, H):
    """Polylines cut at a pixel past the canvas (W, H), as matplotlib cuts
    a line with no fill before stroking: its joins and dashes start there.
    (xy, offsets, closed, the source part of each)."""
    d2 = np.empty((0, 2), np.float64)
    d1 = np.empty(0, np.int64)
    db = np.empty(0, np.bool_)
    W, H = float(W), float(H)
    m = offsets.size - 1
    bounds = _part_tasks(offsets)
    if bounds is None:
        p, k = _clip_all(xy, offsets, closed, W, H, 0, m, d2, d1, d1, db, 0, 0, False)
        out = np.empty((p, 2))
        off = np.empty(k + 1, np.int64)
        part = np.empty(k, np.int64)
        cl = np.empty(k, np.bool_)
        _clip_all(xy, offsets, closed, W, H, 0, m, out, off, part, cl, 0, 0, True)
    else:
        ps, ks = np.zeros(bounds.size, np.int64), np.zeros(bounds.size, np.int64)
        _clip_tasks(xy, offsets, closed, W, H, bounds, d2, d1, d1, db, ps, ks, False)
        np.cumsum(ps, out=ps)
        np.cumsum(ks, out=ks)
        p, k = int(ps[-1]), int(ks[-1])
        out = np.empty((p, 2))
        off = np.empty(k + 1, np.int64)
        part = np.empty(k, np.int64)
        cl = np.empty(k, np.bool_)
        _clip_tasks(xy, offsets, closed, W, H, bounds, out, off, part, cl, ps, ks, True)
    off[k] = p
    return out, off, cl, part


# ----------------------------------------------------------------- dashing

@njit(cache=True, nogil=True, error_model='numpy')
def _dash_start(dashes, da, nd, ds):
    """calc_dash_start: (the dash, how far into it)."""
    cur, into = 0, 0.0
    while ds > 0.0:
        if ds > dashes[da + cur]:
            ds -= dashes[da + cur]
            cur += 1
            into = 0.0
            if cur >= nd:
                cur = 0
        else:
            into = ds
            ds = 0.0
    return cur, into


@njit(cache=True, nogil=True, error_model='numpy')
def _dash_part(src, n, closed, dashes, da, nd, start, out, off, p, k, write):
    """vcgen_dash over src[:n]: the dashes as polylines from out[p], parts
    from off[k]. Returns (p, k)."""
    if n < 2:
        return p, k
    cur, into = _dash_start(dashes, da, nd, abs(start))
    v1, v2 = 0, 1
    rest = src[0, 2]
    vertex = 1
    x, y = src[0, 0], src[0, 1]
    open_ = False                                       # a part opens at its first line_to
    px, py = x, y
    run = True
    while run:
        dash_rest = dashes[da + cur] - into
        on = (cur & 1) == 0
        if rest > dash_rest:
            rest -= dash_rest
            cur += 1
            if cur >= nd:
                cur = 0
            into = 0.0
            x = src[v2, 0] - (src[v2, 0] - src[v1, 0]) * rest / src[v1, 2]
            y = src[v2, 1] - (src[v2, 1] - src[v1, 1]) * rest / src[v1, 2]
        else:
            into += rest
            x, y = src[v2, 0], src[v2, 1]
            vertex += 1
            v1 = v2
            rest = src[v1, 2]
            if closed:
                if vertex > n:
                    run = False
                else:
                    v2 = 0 if vertex >= n else vertex
            elif vertex >= n:
                run = False
            else:
                v2 = vertex
        if on:
            if not open_:
                if write:
                    off[k] = p
                    out[p, 0], out[p, 1] = px, py
                k += 1
                p += 1
                open_ = True
            if write:
                out[p, 0], out[p, 1] = x, y
            p += 1
        else:
            open_ = False
        px, py = x, y
    return p, k


@njit(cache=True, nogil=True, error_model='numpy')
def _dash_all(xy, offsets, closed, part_style, dashes, dash_starts, dash_offsets, q0, q1, src, ns,
              out, off, part, p, k, write):
    """vcgen_dash of parts q0..q1, from points p and parts k: counting, each
    part's vertex_sequence goes in src and ns, for writing to use."""
    for q in range(q0, q1):
        st = part_style[q]
        da, db = dash_starts[st], dash_starts[st + 1]
        a = offsets[q]
        if write:
            n = ns[q]
        else:
            n = ns[q] = _sequence(xy, a, offsets[q + 1], closed[q], src)
        k0 = k
        if db - da >= 2:
            p, k = _dash_part(src[a:], n, closed[q], dashes, da, db - da, dash_offsets[st], out,
                              off, p, k, write)
        elif n >= 2:                                    # solid, among dashed
            if write:
                off[k] = p
                out[p:p + n, :] = src[a:a + n, :2]
            p += n
            k += 1
        if write:
            for r in range(k0, k):
                part[r] = q
    return p, k


@njit(cache=True, nogil=True, parallel=True, error_model='numpy')
def _dash_tasks(xy, offsets, closed, part_style, dashes, dash_starts, dash_offsets, bounds, src, ns,
                out, off, part, ps, ks, write):
    """_dash_all of the stretches of parts between bounds, as _clip_tasks."""
    for t in prange(bounds.size - 1):
        p, k = _dash_all(xy, offsets, closed, part_style, dashes, dash_starts, dash_offsets,
                         bounds[t], bounds[t + 1], src, ns, out, off, part,
                         ps[t] if write else 0, ks[t] if write else 0, write)
        if not write:
            ps[t + 1], ks[t + 1] = p, k


def dash(xy, offsets, closed, part_style, dashes, dash_starts, dash_offsets):
    """The dashes of polylines, as vcgen_dash cuts them: (xy, offsets, the
    source part of each)."""
    d2 = np.empty((0, 2), np.float64)
    d1 = np.empty(0, np.int64)
    m = offsets.size - 1
    src = np.empty((xy.shape[0], 3))
    ns = np.empty(m, np.int64)
    style = (dashes, dash_starts, dash_offsets)
    bounds = _part_tasks(offsets)
    if bounds is None:
        p, k = _dash_all(xy, offsets, closed, part_style, *style, 0, m, src, ns, d2, d1, d1, 0, 0,
                         False)
        out = np.empty((p, 2))
        off = np.empty(k + 1, np.int64)
        part = np.empty(k, np.int64)
        _dash_all(xy, offsets, closed, part_style, *style, 0, m, src, ns, out, off, part, 0, 0, True)
    else:
        ps, ks = np.zeros(bounds.size, np.int64), np.zeros(bounds.size, np.int64)
        _dash_tasks(xy, offsets, closed, part_style, *style, bounds, src, ns, d2, d1, d1, ps, ks,
                    False)
        np.cumsum(ps, out=ps)
        np.cumsum(ks, out=ks)
        p, k = int(ps[-1]), int(ks[-1])
        out = np.empty((p, 2))
        off = np.empty(k + 1, np.int64)
        part = np.empty(k, np.int64)
        _dash_tasks(xy, offsets, closed, part_style, *style, bounds, src, ns, out, off, part, ps, ks,
                    True)
    off[k] = p
    return out, off, part


# ------------------------------------------------------------- rasterizing

@njit(cache=True, nogil=True, error_model='numpy', inline='always')
def _add(cell, bits, at, wat, x, c, a):
    """A cell's cover and area (cell[at + 2x], side by side as they are
    used), and the bit that says it has been touched."""
    i = at + 2 * x
    cell[i] += c
    cell[i + 1] += a
    bits[wat + (x >> 6)] |= np.int64(1) << (x & 63)


@njit(cache=True, nogil=True, error_model='numpy', inline='always')
def _hline(cell, bits, at, wat, x1, y1, x2, y2):
    """render_hline: the part of a line on one row, from (x1, y1) to
    (x2, y2), x whole 24.8, y in 0..256."""
    if y1 == y2:
        return
    ex1, ex2 = x1 >> 8, x2 >> 8
    fx1, fx2 = x1 & 255, x2 & 255
    if ex1 == ex2:
        d = y2 - y1
        _add(cell, bits, at, wat, ex1, d, (fx1 + fx2) * d)
        return
    dx = x2 - x1
    if dx < 0:
        p, first, incr, dx = fx1 * (y2 - y1), 0, -1, -dx
    else:
        p, first, incr = (256 - fx1) * (y2 - y1), 256, 1
    delta = p // dx
    mod = p - delta * dx
    _add(cell, bits, at, wat, ex1, delta, (fx1 + first) * delta)
    ex1 += incr
    y1 += delta
    if ex1 != ex2:
        p = 256 * (y2 - y1 + delta)
        lift = p // dx
        rem = p - lift * dx
        mod -= dx
        while ex1 != ex2:
            delta = lift
            mod += rem
            if mod >= 0:
                mod -= dx
                delta += 1
            _add(cell, bits, at, wat, ex1, delta, 256 * delta)
            y1 += delta
            ex1 += incr
    delta = y2 - y1
    _add(cell, bits, at, wat, ex1, delta, (fx2 + 256 - first) * delta)


@njit(cache=True, nogil=True, error_model='numpy', inline='always')
def _mark(span, row, a, b):
    """A row's first and last columns used (span[2 row], span[2 row + 1])."""
    if a > b:
        a, b = b, a
    if a < span[2 * row]:
        span[2 * row] = a
    if b > span[2 * row + 1]:
        span[2 * row + 1] = b


@njit(cache=True, nogil=True, error_model='numpy', inline='always')
def _line(cell, bits, span, stride, words, x1, y1, x2, y2, r0, r1, st):
    """rasterizer_cells_aa::line, of the rows r0..r1 only (whole 24.8 x and
    y, x from the first column, y from the first row of the cells; a row is
    stride entries of cell, words of bits): st[0..1] the rows it touched,
    widened. The buffers go as arguments, not as a tuple: taking one apart
    counts references on each, atomically, for every edge."""
    ey1, ey2 = y1 >> 8, y2 >> 8
    fy1, fy2 = y1 & 255, y2 & 255
    if ey1 == ey2:
        if r0 <= ey1 < r1:
            _hline(cell, bits, ey1 * stride, ey1 * words, x1, fy1, x2, fy2)
            _mark(span, ey1, x1 >> 8, x2 >> 8)
            if ey1 < st[0]:
                st[0] = ey1
            if ey1 > st[1]:
                st[1] = ey1
        return
    dx, dy = x2 - x1, y2 - y1
    if dx == 0:
        ex = x1 >> 8
        two = (x1 & 255) << 1
        first, incr = (256, 1) if dy > 0 else (0, -1)
        ra, rb = (ey1, ey2) if dy > 0 else (ey2, ey1)
        for row in range(max(ra, r0), min(rb + 1, r1)):
            if row == ey1:
                d = first - fy1
            elif row == ey2:
                d = fy2 - 256 + first
            else:
                d = first + first - 256
            _add(cell, bits, row * stride, row * words, ex, d, two * d)
            _mark(span, row, ex, ex)
        ra, rb = max(ra, r0), min(rb, r1 - 1)
        st[0], st[1] = min(st[0], ra), max(st[1], rb)
        return
    if dy > 0:
        p, first, incr = (256 - fy1) * dx, 256, 1
    else:
        p, first, incr, dy = fy1 * dx, 0, -1, -dy
    P = 256 * dx
    ra, rb = (ey1, ey2) if incr > 0 else (ey2, ey1)
    ra, rb = max(ra, r0), min(rb, r1 - 1)
    if ra > rb:
        return
    st[0], st[1] = min(st[0], ra), max(st[1], rb)
    k0, k1 = (ra - ey1, rb - ey1) if incr > 0 else (ey1 - rb, ey1 - ra)
    last = (ey2 - ey1) * incr
    # x at the crossing after step k: x1 + floor((p + k P) / dy), as Agg steps to it
    xa = x1 if k0 == 0 else x1 + (p + (k0 - 1) * P) // dy
    for k in range(k0, k1 + 1):
        row = ey1 + k * incr
        if k == last:
            xb, yb = x2, fy2
        else:
            xb, yb = x1 + (p + k * P) // dy, first
        ya = fy1 if k == 0 else 256 - first
        _hline(cell, bits, row * stride, row * words, xa, ya, xb, yb)
        _mark(span, row, xa >> 8, xb >> 8)
        xa = xb


@njit(cache=True, nogil=True, error_model='numpy')
def _iround(v):
    return np.int64(v - 0.5) if v < 0.0 else np.int64(v + 0.5)


@njit(cache=True, nogil=True, error_model='numpy')
def _line_y(cells, x1, y1, x2, y2, f1, f2, cy0, cy1, ox, oy, r0, r1, st):
    """line_clip_y: an edge on the rows of the clip, in 24.8 from (ox, oy)."""
    cell, bits, span, stride, words = cells
    f1 &= 10
    f2 &= 10
    if f1 != 0 or f2 != 0:
        if f1 == f2:
            return
        tx1, ty1, tx2, ty2 = x1, y1, x2, y2
        if f1 & 8:
            tx1, ty1 = x1 + (cy0 - y1) * (x2 - x1) / (y2 - y1), cy0
        if f1 & 2:
            tx1, ty1 = x1 + (cy1 - y1) * (x2 - x1) / (y2 - y1), cy1
        if f2 & 8:
            tx2, ty2 = x1 + (cy0 - y1) * (x2 - x1) / (y2 - y1), cy0
        if f2 & 2:
            tx2, ty2 = x1 + (cy1 - y1) * (x2 - x1) / (y2 - y1), cy1
        x1, y1, x2, y2 = tx1, ty1, tx2, ty2
    _line(cell, bits, span, stride, words, _iround(x1 * 256.0) - ox, _iround(y1 * 256.0) - oy,
          _iround(x2 * 256.0) - ox, _iround(y2 * 256.0) - oy, r0, r1, st)


@njit(cache=True, nogil=True, error_model='numpy')
def _flags_y(y, cy0, cy1):
    return (2 if y > cy1 else 0) | (8 if y < cy0 else 0)


@njit(cache=True, nogil=True, error_model='numpy')
def _edge(cells, x1, y1, x2, y2, cx0, cy0, cx1, cy1, ox, oy, r0, r1, st):
    """rasterizer_sl_clip::line_to: an edge clipped to the box (cx0, cy0,
    cx1, cy1); what lies left or right of it runs along its side, so the
    winding stays right."""
    f1 = (1 if x1 > cx1 else 0) | (2 if y1 > cy1 else 0) | (4 if x1 < cx0 else 0) | \
        (8 if y1 < cy0 else 0)
    f2 = (1 if x2 > cx1 else 0) | (2 if y2 > cy1 else 0) | (4 if x2 < cx0 else 0) | \
        (8 if y2 < cy0 else 0)
    if (f1 & 10) == (f2 & 10) and (f1 & 10) != 0:
        return
    case = ((f1 & 5) << 1) | (f2 & 5)
    if case == 0:
        _line_y(cells, x1, y1, x2, y2, f1, f2, cy0, cy1, ox, oy, r0, r1, st)
    elif case == 1:
        y3 = y1 + (cx1 - x1) * (y2 - y1) / (x2 - x1)
        f3 = _flags_y(y3, cy0, cy1)
        _line_y(cells, x1, y1, cx1, y3, f1, f3, cy0, cy1, ox, oy, r0, r1, st)
        _line_y(cells, cx1, y3, cx1, y2, f3, f2, cy0, cy1, ox, oy, r0, r1, st)
    elif case == 2:
        y3 = y1 + (cx1 - x1) * (y2 - y1) / (x2 - x1)
        f3 = _flags_y(y3, cy0, cy1)
        _line_y(cells, cx1, y1, cx1, y3, f1, f3, cy0, cy1, ox, oy, r0, r1, st)
        _line_y(cells, cx1, y3, x2, y2, f3, f2, cy0, cy1, ox, oy, r0, r1, st)
    elif case == 3:
        _line_y(cells, cx1, y1, cx1, y2, f1, f2, cy0, cy1, ox, oy, r0, r1, st)
    elif case == 4:
        y3 = y1 + (cx0 - x1) * (y2 - y1) / (x2 - x1)
        f3 = _flags_y(y3, cy0, cy1)
        _line_y(cells, x1, y1, cx0, y3, f1, f3, cy0, cy1, ox, oy, r0, r1, st)
        _line_y(cells, cx0, y3, cx0, y2, f3, f2, cy0, cy1, ox, oy, r0, r1, st)
    elif case == 6:
        y3 = y1 + (cx1 - x1) * (y2 - y1) / (x2 - x1)
        y4 = y1 + (cx0 - x1) * (y2 - y1) / (x2 - x1)
        f3, f4 = _flags_y(y3, cy0, cy1), _flags_y(y4, cy0, cy1)
        _line_y(cells, cx1, y1, cx1, y3, f1, f3, cy0, cy1, ox, oy, r0, r1, st)
        _line_y(cells, cx1, y3, cx0, y4, f3, f4, cy0, cy1, ox, oy, r0, r1, st)
        _line_y(cells, cx0, y4, cx0, y2, f4, f2, cy0, cy1, ox, oy, r0, r1, st)
    elif case == 8:
        y3 = y1 + (cx0 - x1) * (y2 - y1) / (x2 - x1)
        f3 = _flags_y(y3, cy0, cy1)
        _line_y(cells, cx0, y1, cx0, y3, f1, f3, cy0, cy1, ox, oy, r0, r1, st)
        _line_y(cells, cx0, y3, x2, y2, f3, f2, cy0, cy1, ox, oy, r0, r1, st)
    elif case == 9:
        y3 = y1 + (cx0 - x1) * (y2 - y1) / (x2 - x1)
        y4 = y1 + (cx1 - x1) * (y2 - y1) / (x2 - x1)
        f3, f4 = _flags_y(y3, cy0, cy1), _flags_y(y4, cy0, cy1)
        _line_y(cells, cx0, y1, cx0, y3, f1, f3, cy0, cy1, ox, oy, r0, r1, st)
        _line_y(cells, cx0, y3, cx1, y4, f3, f4, cy0, cy1, ox, oy, r0, r1, st)
        _line_y(cells, cx1, y4, cx1, y2, f4, f2, cy0, cy1, ox, oy, r0, r1, st)
    else:                                               # case 12
        _line_y(cells, cx0, y1, cx0, y2, f1, f2, cy0, cy1, ox, oy, r0, r1, st)


@njit(cache=True, nogil=True, error_model='numpy')
def _alpha(a, evenodd):
    """calculate_alpha."""
    c = a >> 9
    if c < 0:
        c = -c
    if evenodd:
        c &= 511
        if c > 256:
            c = 512 - c
    return 255 if c > 255 else c


@njit(cache=True, nogil=True, error_model='numpy')
def _blend(canvas, i, j, cr, cg, cb, ca, cover):
    """matplotlib's fixed_blender_rgba_plain, on 8 bits."""
    if ca == 255 and cover == 255:
        canvas[i, j, 0], canvas[i, j, 1], canvas[i, j, 2], canvas[i, j, 3] = cr, cg, cb, 255
        return
    t = ca * cover + 128
    alpha = ((t >> 8) + t) >> 8
    if alpha == 0:
        return
    a = np.int64(canvas[i, j, 3])
    r = np.int64(canvas[i, j, 0]) * a
    g = np.int64(canvas[i, j, 1]) * a
    b = np.int64(canvas[i, j, 2]) * a
    a = ((alpha + a) << 8) - alpha * a
    canvas[i, j, 3] = a >> 8
    canvas[i, j, 0] = (((cr << 8) - r) * alpha + (r << 8)) // a
    canvas[i, j, 1] = (((cg << 8) - g) * alpha + (g << 8)) // a
    canvas[i, j, 2] = (((cb << 8) - b) * alpha + (b << 8)) // a


@njit(cache=True, nogil=True, error_model='numpy')
def _sweep(canvas, cells, ra, rb, x0, y0, w, cr, cg, cb, ca, evenodd):
    """sweep_scanline over rows ra..rb of the cells, blending, and the cells
    used cleared for the next path. The touched cells come off the row's
    bits, so a row with little on it costs little however wide it is."""
    cell, bits, span, stride, words = cells
    for row in range(ra, rb + 1):
        a, b = span[2 * row], span[2 * row + 1]
        span[2 * row], span[2 * row + 1] = 1 << 30, -1
        c = 0
        i = y0 + row
        at, wat = row * stride, row * words
        nxt = a                                         # the first pixel not yet drawn
        for word in range(a >> 6, (b >> 6) + 1):
            v = bits[wat + word]
            if v == 0:
                continue
            bits[wat + word] = 0
            while v != 0:
                low = v & -v
                v ^= low
                x = (word << 6) + _LOG2[((low * _DEBRUIJN) >> 58) & 63]
                if c != 0 and nxt < x:                  # the span since the last cell
                    m = _alpha(c << 9, evenodd)
                    if m != 0:
                        for j in range(nxt, min(x, w)):
                            _blend(canvas, i, x0 + j, cr, cg, cb, ca, m)
                q = at + 2 * x
                cc, aa = cell[q], cell[q + 1]
                cell[q] = 0
                cell[q + 1] = 0
                c += cc
                m = _alpha((c << 9) - aa, evenodd)
                if m != 0 and x < w:
                    _blend(canvas, i, x0 + x, cr, cg, cb, ca, m)
                nxt = x + 1


@njit(cache=True, nogil=True, error_model='numpy')
def _band(canvas, pts, ipts, nxt, a, b, index, e_path, path_rgba, cx0, cy0, cx1, cy1, r0, r1,
          cells, evenodd):
    """The rows r0..r1 of the clip (the cells' rows, from cy0), edges
    index[a:b]: each path's edges summed into cells, swept, blended, in
    order."""
    w = cx1 - cx0
    ox, oy = cx0 * 256, cy0 * 256
    fx0, fy0, fx1, fy1 = np.float64(cx0), np.float64(cy0), np.float64(cx1), np.float64(cy1)
    st = np.empty(2, np.int64)
    st[0], st[1] = 1 << 30, -1
    cell, bits, span, stride, words = cells
    cur = -1
    for q in range(a, b):
        e = index[q]
        pth = e_path[e]
        if pth != cur:
            if cur >= 0 and st[1] >= 0:
                _sweep(canvas, cells, st[0], st[1], cx0, cy0, w,
                       np.int64(path_rgba[cur, 0]), np.int64(path_rgba[cur, 1]),
                       np.int64(path_rgba[cur, 2]), np.int64(path_rgba[cur, 3]), evenodd)
                st[0], st[1] = 1 << 30, -1
            cur = pth
        f = nxt[e]
        x1, y1, x2, y2 = ipts[e, 0], ipts[e, 1], ipts[f, 0], ipts[f, 1]
        if x1 != _OUT and x2 != _OUT:                   # both on the canvas: no clipping to do
            _line(cell, bits, span, stride, words, np.int64(x1) - ox, np.int64(y1) - oy,
                  np.int64(x2) - ox, np.int64(y2) - oy, r0, r1, st)
        else:
            _edge(cells, pts[e, 0], pts[e, 1], pts[f, 0], pts[f, 1], fx0, fy0, fx1, fy1,
                  ox, oy, r0, r1, st)
    if cur >= 0 and st[1] >= 0:
        _sweep(canvas, cells, st[0], st[1], cx0, cy0, w, np.int64(path_rgba[cur, 0]),
               np.int64(path_rgba[cur, 1]), np.int64(path_rgba[cur, 2]),
               np.int64(path_rgba[cur, 3]), evenodd)


@njit(cache=True, nogil=True, error_model='numpy')
def _bands_serial(canvas, pts, ipts, nxt, index, e_path, path_rgba, cx0, cy0, cx1, cy1, cells,
                  evenodd):
    _band(canvas, pts, ipts, nxt, 0, index.size, index, e_path, path_rgba, cx0, cy0, cx1, cy1, 0,
          cy1 - cy0, cells, evenodd)


@njit(cache=True, nogil=True, parallel=True, error_model='numpy')
def _bands(canvas, pts, ipts, nxt, starts, index, e_path, path_rgba, cx0, cy0, cx1, cy1, band_h,
           cells, evenodd, ntask):
    nb = starts.size - 1
    for c in prange(ntask):                             # bands interleaved: a dense middle shared
        for b in range(c, nb, ntask):
            r0 = b * band_h
            _band(canvas, pts, ipts, nxt, starts[b], starts[b + 1], index, e_path, path_rgba, cx0,
                  cy0, cx1, cy1, r0, min(r0 + band_h, cy1 - cy0), cells, evenodd)


@njit(cache=True, nogil=True, error_model='numpy')
def _bin(pts, nxt, e0, e1, cy0, cy1, band_h, nb, counts, index, fill):
    """Which bands edges e0..e1 reach: counts (fill False), then the lists
    (fill, counts each band's next free slot). Returns how many edges."""
    n = 0
    for e in range(e0, e1):
        f = nxt[e]
        if f < 0:
            continue
        n += 1
        y1, y2 = pts[e, 1], pts[f, 1]
        ya, yb = (y1, y2) if y1 < y2 else (y2, y1)
        if not (yb >= cy0 and ya <= cy1):
            continue
        ba = (np.int64(max(ya, cy0)) - cy0) // band_h
        bb = min(nb - 1, (np.int64(min(yb, cy1)) - cy0) // band_h)
        for b in range(ba, bb + 1):
            if fill:
                index[counts[b]] = e
            counts[b] += 1
    return n


@njit(cache=True, nogil=True, parallel=True, error_model='numpy')
def _bin_chunks(pts, nxt, bounds, cy0, cy1, band_h, nb, counts, index, fill):
    for c in prange(bounds.size - 1):
        _bin(pts, nxt, bounds[c], bounds[c + 1], cy0, cy1, band_h, nb, counts[c], index, fill)


@njit(cache=True, nogil=True, error_model='numpy')
def _slots(counts, starts):
    """Band starts (prefix of the chunks' counts) and, in counts, each
    chunk's first slot in each band: a band lists its edges in order."""
    nc, nb = counts.shape
    pos = 0
    for b in range(nb):
        starts[b] = pos
        for c in range(nc):
            n = counts[c, b]
            counts[c, b] = pos
            pos += n
    starts[nb] = pos


@njit(cache=True, nogil=True, error_model='numpy')
def _ring_links(rings, ring_path, nxt, e_path, r):
    a, b = rings[r], rings[r + 1]
    pth = ring_path[r]
    for e in range(a, b):
        nxt[e] = e + 1
        e_path[e] = pth
    if b > a:
        nxt[b - 1] = a


@njit(cache=True, nogil=True, error_model='numpy')
def _edges_ok(pts, ipts, nxt, a, b, box, clip):
    """Edges a..b from or to a non-finite point dropped; the rest widen
    box (x0, y0, x1, y1). The point an edge runs to is some edge's start,
    or a finite ring's, so starts alone make the box. ipts: each point in
    Agg's 24.8, _OUT off the clip (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
    for e in range(a, b):
        x, y = pts[e, 0], pts[e, 1]
        ok = np.isfinite(x) and np.isfinite(y)
        f = nxt[e]
        if f >= 0 and not (ok and np.isfinite(pts[f, 0]) and np.isfinite(pts[f, 1])):
            nxt[e] = -1
        ipts[e, 0] = _OUT
        if ok:
            if clip[0] <= x <= clip[2] and clip[1] <= y <= clip[3]:
                ipts[e, 0], ipts[e, 1] = _iround(x * 256.0), _iround(y * 256.0)
            if x < x0:
                x0 = x
            if x > x1:
                x1 = x
            if y < y0:
                y0 = y
            if y > y1:
                y1 = y
    box[0], box[1], box[2], box[3] = x0, y0, x1, y1


@njit(cache=True, nogil=True, parallel=True, error_model='numpy')
def _links(rings, ring_path, nxt, e_path, pts, ipts, bounds, boxes, clip):
    """nxt[e]: the point edge e runs to (-1: none), e_path[e] its path."""
    for r in prange(rings.size - 1):
        _ring_links(rings, ring_path, nxt, e_path, r)
    for c in prange(bounds.size - 1):
        _edges_ok(pts, ipts, nxt, bounds[c], bounds[c + 1], boxes[c], clip)


@njit(cache=True, nogil=True, error_model='numpy')
def _links_serial(rings, ring_path, nxt, e_path, pts, ipts, box, clip):
    for r in range(rings.size - 1):
        _ring_links(rings, ring_path, nxt, e_path, r)
    _edges_ok(pts, ipts, nxt, 0, pts.shape[0], box, clip)


SERIAL_EDGES = 2048                 # edges below which a draw stays on one thread
LINK_CHUNK = 16384                  # edges a task (links, binning)


_LOCAL = threading.local()


def _new_cells(rows, cols):
    """(cover and area, side by side, by row and column; the touched bits;
    each row's first and last column; a row's entries in the first and in
    the second)."""
    words = (cols >> 6) + 1
    span = np.empty(2 * rows, np.int32)
    span[0::2], span[1::2] = 1 << 30, -1
    return (np.zeros(2 * rows * cols, np.int32), np.zeros(rows * words, np.int64), span, 2 * cols,
            words)


def _cells(h, w):
    """This thread's cells for a box of h rows and w columns, all clear: the
    sweep clears what it reads, so one set serves every draw, and no draw
    pays for fresh pages."""
    c = getattr(_LOCAL, 'cells', None)
    if c is None or c[2].size < 2 * h or c[3] < 2 * (w + 2):
        rows = max(h, c[2].size // 2 if c is not None else 0)
        cols = max(w + 2, c[3] // 2 if c is not None else 0)
        c = _LOCAL.cells = _new_cells(rows, cols)
    return c


def fill(canvas, pts, rings, ring_path, path_rgba, clip, evenodd=False):
    """Rings (pts (n, 2) px, ring offsets, each closing back to its first
    point) rasterized as Agg does: the rings of one path summed together,
    paths blended in order. ring_path: each ring's path, rising; path_rgba
    (paths, 4) uint8: their colours; clip: (x0, y0, x1, y1) whole pixels
    on the canvas."""
    if clip is not None:
        run = prepare(pts, rings, ring_path, clip)
        if run is not None:
            rasterize(canvas, run, path_rgba, evenodd)


def prepare(pts, rings, ring_path, clip):
    """What rasterize() needs of rings that does not change from frame to
    frame: the edges, by band; None if nothing shows."""
    n = pts.shape[0]
    if not n:
        return None
    team = _jit.team()
    nxt = np.empty(n, np.int64)
    e_path = np.empty(n, np.int32)
    ipts = np.empty((n, 2), np.int32)
    fclip = np.array(clip, np.float64)
    serial = n <= SERIAL_EDGES or team < 2
    if serial:
        box = np.array([1e300, 1e300, -1e300, -1e300])
        _links_serial(rings, ring_path, nxt, e_path, pts, ipts, box, fclip)
    else:
        bounds = np.arange(0, n + LINK_CHUNK, LINK_CHUNK, dtype=np.int64)
        bounds[-1] = n
        boxes = np.empty((bounds.size - 1, 4))
        boxes[:, :2], boxes[:, 2:] = 1e300, -1e300
        _links(rings, ring_path, nxt, e_path, pts, ipts, bounds, boxes, fclip)
        box = np.concatenate((boxes[:, :2].min(axis=0), boxes[:, 2:].max(axis=0)))
    if box[0] > box[2]:
        return None
    # the cell buffers need only reach the outline: its box within the clip
    # (a pixel to spare, for an edge rounding up onto the next pixel)
    x0, y0 = max(clip[0], int(np.floor(max(box[0], -1e9)))), max(clip[1],
                                                                 int(np.floor(max(box[1], -1e9))))
    x1, y1 = min(clip[2], int(np.floor(min(box[2], 1e9))) + 2), min(
        clip[3], int(np.floor(min(box[3], 1e9))) + 2)
    if x1 <= x0 or y1 <= y0:
        return None
    h = y1 - y0
    if serial or h < 2 * BAND_ROWS:
        return (pts, ipts, nxt, e_path, None, np.flatnonzero(nxt >= 0).astype(np.int32), 0,
                (x0, y0, x1, y1), 1)
    nb = min(BAND_TASKS * team, h // BAND_ROWS)
    band_h = -(-h // nb)
    nb = -(-h // band_h)
    bounds = np.arange(0, n + LINK_CHUNK, LINK_CHUNK, dtype=np.int64)
    bounds[-1] = n
    nc = bounds.size - 1
    counts = np.zeros((nc, nb), np.int64)
    starts = np.zeros(nb + 1, np.int64)
    d1 = np.empty(0, np.int32)
    _bin_chunks(pts, nxt, bounds, y0, y1, band_h, nb, counts, d1, False)
    _slots(counts, starts)
    index = np.empty(starts[-1], np.int32)
    _bin_chunks(pts, nxt, bounds, y0, y1, band_h, nb, counts, index, True)
    return pts, ipts, nxt, e_path, starts, index, band_h, (x0, y0, x1, y1), min(team, nb)


def rasterize(canvas, run, path_rgba, evenodd=False):
    """prepare()'s edges into the canvas, path_rgba (paths, 4) uint8 their
    paths' colours."""
    pts, ipts, nxt, e_path, starts, index, band_h, (x0, y0, x1, y1), ntask = run
    cells = _cells(y1 - y0, x1 - x0)
    if starts is None:
        _bands_serial(canvas, pts, ipts, nxt, index, e_path, path_rgba, x0, y0, x1, y1, cells,
                      evenodd)
    else:
        _bands(canvas, pts, ipts, nxt, starts, index, e_path, path_rgba, x0, y0, x1, y1, band_h,
               cells, evenodd, ntask)


@njit(cache=True, nogil=True, error_model='numpy')
def _strokes_small(canvas, xy, offsets, closed, part_style, part_path, widths, caps, joins, limits,
                   colors, clip, filled, cells):
    """stroke(), prepare() and rasterize() of a few short lines, on this
    thread, in one call: a plot's lines are mostly the numpy between those
    three. False, with nothing drawn, if the lines leave the canvas and are
    to be cut there first (clip_paths)."""
    H, W = canvas.shape[0], canvas.shape[1]
    m, n = offsets.size - 1, xy.shape[0]
    if not filled:
        for i in range(n):
            if xy[i, 0] < -1.0 or xy[i, 1] < -1.0 or xy[i, 0] > W + 1.0 or xy[i, 1] > H + 1.0:
                return False
    src = np.empty((n, 3))
    counts = np.empty(m, np.int64)
    shut = closed.copy()
    _sequences_serial(xy, offsets, shut, src, counts)
    for q in range(m):
        if widths[part_style[q]] <= 0:
            counts[q] = 0
    d2 = xy[:0]
    d1 = offsets[:0]
    p, k = _stroke_steps(src, offsets, counts, shut, part_style, widths, caps, joins, limits, 0, 0,
                         m, 0, d2, d1, d1, 0, 0, False)
    if k == 0:
        return True
    pts = np.empty((p, 2))
    rings = np.empty(k + 1, np.int64)
    ring_part = np.empty(k, np.int64)
    _stroke_steps(src, offsets, counts, shut, part_style, widths, caps, joins, limits, 0, 0, m, 0,
                  pts, rings, ring_part, 0, 0, True)
    rings[k] = p
    ring_path = np.empty(k, np.int64)
    path_rgba = np.empty((k, 4), np.uint8)
    paths = 0
    for r in range(k):                                  # ids from 0; a path's colour its first ring's
        q = ring_part[r]
        if r == 0 or part_path[q] != part_path[ring_part[r - 1]]:
            c = part_style[q]
            path_rgba[paths, 0], path_rgba[paths, 1] = colors[c, 0], colors[c, 1]
            path_rgba[paths, 2], path_rgba[paths, 3] = colors[c, 2], colors[c, 3]
            paths += 1
        ring_path[r] = paths - 1
    nxt = np.empty(p, np.int64)
    e_path = np.empty(p, np.int32)
    ipts = np.empty((p, 2), np.int32)
    box = np.empty(4)
    box[0], box[1], box[2], box[3] = 1e300, 1e300, -1e300, -1e300
    _links_serial(rings, ring_path, nxt, e_path, pts, ipts, box, clip)
    if box[0] > box[2]:
        return True
    x0 = max(np.int64(clip[0]), np.int64(np.floor(max(box[0], -1e9))))
    y0 = max(np.int64(clip[1]), np.int64(np.floor(max(box[1], -1e9))))
    x1 = min(np.int64(clip[2]), np.int64(np.floor(min(box[2], 1e9))) + 2)
    y1 = min(np.int64(clip[3]), np.int64(np.floor(min(box[3], 1e9))) + 2)
    if x1 <= x0 or y1 <= y0:
        return True
    index = np.empty(p, np.int32)
    e = 0
    for i in range(p):
        if nxt[i] >= 0:
            index[e] = i
            e += 1
    _band(canvas, pts, ipts, nxt, 0, e, index, e_path, path_rgba, x0, y0, x1, y1, 0, y1 - y0, cells,
          False)
    return True


# ----------------------------------------------------------------- markers
#
# A path collection is many small paths, each filled and stroked, blended in
# order. What a path puts down depends on its outline alone, only the blend
# on what came before: so each marker's coverage is worked out on its own, on
# any core, into a mask a few pixels across, by the very steps of a path
# above (_walk, _stroke_steps, _line, _alpha), and the masks blended in order
# afterwards by row bands. The cells are a marker's own, so nothing is shared
# and no band cuts a marker's outline: in place of one prepare() and
# rasterize() of everything, about twice as soon.

@njit(cache=True, nogil=True, error_model='numpy')
def _sweep_mask(cells, ra, rb, w, mask, at):
    """_sweep's coverage, into mask from at (rows of w), not blended; the
    cells used cleared."""
    cell, bits, span, stride, words = cells
    for row in range(ra, rb + 1):
        a, b = span[2 * row], span[2 * row + 1]
        span[2 * row], span[2 * row + 1] = 1 << 30, -1
        c = 0
        base = at + row * w
        cat, wat = row * stride, row * words
        nxt = a
        for word in range(a >> 6, (b >> 6) + 1):
            v = bits[wat + word]
            if v == 0:
                continue
            bits[wat + word] = 0
            while v != 0:
                low = v & -v
                v ^= low
                x = (word << 6) + _LOG2[((low * _DEBRUIJN) >> 58) & 63]
                if c != 0 and nxt < x:
                    m = _alpha(c << 9, False)
                    if m != 0:
                        for j in range(nxt, min(x, w)):
                            mask[base + j] = m
                q = cat + 2 * x
                cc, aa = cell[q], cell[q + 1]
                cell[q] = 0
                cell[q + 1] = 0
                c += cc
                m = _alpha((c << 9) - aa, False)
                if m != 0 and x < w:
                    mask[base + x] = m
                nxt = x + 1


@njit(cache=True, nogil=True, error_model='numpy')
def _ring_cells(cells, pts, a, b, clip, inside, cx0, cy0, cx1, cy1, r1, st):
    """A ring pts[a:b], closing back to its first point, into the cells of
    the box (cx0, cy0, cx1, cy1): _band's edges. inside: all of it on clip."""
    ox, oy = cx0 * 256, cy0 * 256
    fx0, fy0, fx1, fy1 = np.float64(cx0), np.float64(cy0), np.float64(cx1), np.float64(cy1)
    cell, bits, span, stride, words = cells
    x2, y2 = pts[a, 0], pts[a, 1]
    in2 = inside or (clip[0] <= x2 <= clip[2] and clip[1] <= y2 <= clip[3])
    ix2, iy2 = _iround(x2 * 256.0) - ox, _iround(y2 * 256.0) - oy
    for e in range(a, b):
        x1, y1, in1, ix1, iy1 = x2, y2, in2, ix2, iy2
        f = e + 1 if e + 1 < b else a
        x2, y2 = pts[f, 0], pts[f, 1]
        in2 = inside or (clip[0] <= x2 <= clip[2] and clip[1] <= y2 <= clip[3])
        ix2, iy2 = _iround(x2 * 256.0) - ox, _iround(y2 * 256.0) - oy
        if in1 and in2:
            _line(cell, bits, span, stride, words, ix1, iy1, ix2, iy2, 0, r1, st)
        else:
            _edge(cells, x1, y1, x2, y2, fx0, fy0, fx1, fy1, ox, oy, 0, r1, st)


@njit(cache=True, nogil=True, error_model='numpy')
def _within(pts, n, x0, y0, x1, y1, clip):
    """Whether pts[:n] all lie in the box, its sides on the clip open: what
    is off the clip is clipped, and what is on it must stay in the cells."""
    lx, ly = -np.inf if x0 <= clip[0] else x0, -np.inf if y0 <= clip[1] else y0
    hx, hy = np.inf if x1 >= clip[2] else x1, np.inf if y1 >= clip[3] else y1
    for i in range(n):
        if not (lx <= pts[i, 0] <= hx and ly <= pts[i, 1] <= hy):
            return False
    return True


@njit(cache=True, nogil=True, error_model='numpy')
def _rectilinear(xy, a, b):
    """Whether part xy[a:b] of a marker has no slanted segment, as _draw.snap
    asks."""
    if b - a > 1024:
        return False
    for k in range(a, b - 1):
        if abs(xy[k + 1, 0] - xy[k, 0]) >= 1e-4 and abs(xy[k + 1, 1] - xy[k, 1]) >= 1e-4:
            return False
    return True


@njit(cache=True, nogil=True, error_model='numpy')
def _snap_flat(xy, off, parts, sv):
    """_draw.snap of a marker's parts in place, each if rectilinear."""
    for q in range(parts):
        a, b = off[q], off[q + 1]
        if _rectilinear(xy, a, b):
            for k in range(a, b):
                xy[k, 0] = np.floor(xy[k, 0] + 0.5) + sv
                xy[k, 1] = np.floor(xy[k, 1] + 0.5) + sv


@njit(cache=True, nogil=True, error_model='numpy')
def _marker_flat(v, codes, s, px, py, placed, xy, off, closed, last_s):
    """A marker's path at size s and (px, py), flattened as _path.flatten
    does: (xy holding the points, or a bigger array if it had to be; off,
    closed, its parts; points, parts). off, closed must hold a part per
    vertex, as a path cannot have more. At the size of the marker before,
    it needs no counting first: only rounding sets one place's curves apart
    from another's, and a test that rounding turns halves a curve once more,
    so that xy, kept twice as long as the count, has room."""
    m = v.shape[0]
    for i in range(m):
        placed[i, 0], placed[i, 1] = v[i, 0] * s + px, v[i, 1] * s + py
    if s != last_s:
        n = _path_walk(placed, codes, 0, m, 0.5, xy[:0], off[:0], closed[:0], 0, 0, False, 0)[0]
        if 2 * n + 64 > xy.shape[0]:
            xy = np.empty((2 * (2 * n + 64), 2))
    n, parts = _path_walk(placed, codes, 0, m, 0.5, xy, off, closed, 0, 0, True, xy.shape[0])
    off[parts] = n
    return xy, n, parts


@njit(cache=True, nogil=True, error_model='numpy')
def _marker_chunk(pts, sizes, fv, fc, sv, sc, same, sv_px, w_arr, cap_arr, join_arr, lim_arr, clip,
                  box, mask_at, masks, k0, k1, hmax, wmax, room, bad):
    """Markers k0..k1 into their masks (the fill's, then the stroke's, rows
    of their box): see the comment above. bad[0]: an outline left its box,
    so that the masks are not to be used. Any kernel outside touches nothing
    but its own marker's mask."""
    has_fill, has_stroke = fv.shape[0] > 0, sv.shape[0] > 0
    nv = max(fv.shape[0], sv.shape[0])
    placed = np.empty((nv, 2))
    xy, xy2 = np.empty((256, 2)), np.empty((256, 2))
    off, off2 = np.empty(nv + 1, np.int64), np.empty(nv + 1, np.int64)
    closed, closed2 = np.empty(nv, np.bool_), np.empty(nv, np.bool_)
    src = np.empty((256, 3))
    counts = np.empty(nv, np.int64)
    style = np.zeros(nv, np.int64)
    out = np.empty((4096, 2))
    rings = np.empty(2 * nv + 1, np.int64)
    ring_part = np.empty(2 * nv, np.int64)
    words = ((wmax + 2) >> 6) + 1
    span = np.empty(2 * hmax, np.int32)
    span[0::2] = 1 << 30
    span[1::2] = -1
    cells = (np.zeros(2 * hmax * (wmax + 2), np.int32), np.zeros(hmax * words, np.int64), span,
             2 * (wmax + 2), words)
    st = np.empty(2, np.int64)
    fx0, fy0, fx1, fy1 = clip[0], clip[1], clip[2], clip[3]
    last_s = np.nan
    for k in range(k0, k1):
        x0, y0, x1, y1 = box[k, 0], box[k, 1], box[k, 2], box[k, 3]
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            continue
        at = mask_at[k]
        masks[at:mask_at[k + 1]] = 0
        s, px, py = sizes[k], pts[k, 0], pts[k, 1]
        inside = np.float64(x0) - 1.0 >= fx0 and np.float64(y0) - 1.0 >= fy0 and \
            np.float64(x1) + 1.0 <= fx1 and np.float64(y1) + 1.0 <= fy1
        n, parts = 0, 0
        if has_fill:
            xy, n, parts = _marker_flat(fv, fc, s, px, py, placed, xy, off, closed, last_s)
            _snap_flat(xy, off, parts, sv_px)
            if not _within(xy, n, x0, y0, x1, y1, clip):
                bad[0] = 1
                return
            st[0], st[1] = 1 << 30, -1
            for q in range(parts):
                _ring_cells(cells, xy, off[q], off[q + 1], clip, inside, x0, y0, x1, y1, h, st)
            if st[1] >= 0:
                _sweep_mask(cells, st[0], st[1], w, masks, at)
            at += w * h
        if has_stroke:
            if not same:
                xy2, n, parts = _marker_flat(sv, sc, s, px, py, placed, xy2, off2, closed2, last_s)
                _snap_flat(xy2, off2, parts, sv_px)
                line, loff, lclosed = xy2, off2, closed2
            else:
                line, loff, lclosed = xy, off, closed
            if n > src.shape[0]:
                src = np.empty((2 * n, 3))
            for q in range(parts):
                _sequence_parts(line, loff, lclosed, src, counts, q)
            if (2 * n + 4) * room > out.shape[0]:
                out = np.empty((2 * (2 * n + 4) * room, 2))
            p, r = _stroke_steps(src, loff, counts, lclosed, style, w_arr, cap_arr, join_arr,
                                 lim_arr, 0, 0, parts, 0, out, rings, ring_part, 0, 0, True)
            rings[r] = p
            if not _within(out, p, x0, y0, x1, y1, clip):
                bad[0] = 1
                return
            st[0], st[1] = 1 << 30, -1
            for q in range(r):
                _ring_cells(cells, out, rings[q], rings[q + 1], clip, inside, x0, y0, x1, y1, h, st)
            if st[1] >= 0:
                _sweep_mask(cells, st[0], st[1], w, masks, at)
        last_s = s


@njit(cache=True, nogil=True, parallel=True, error_model='numpy')
def _marker_masks(pts, sizes, fv, fc, sv, sc, same, sv_px, w_arr, cap_arr, join_arr, lim_arr, clip,
                  box, mask_at, masks, bounds, hmax, wmax, room, bad):
    for c in prange(bounds.size - 1):
        _marker_chunk(pts, sizes, fv, fc, sv, sc, same, sv_px, w_arr, cap_arr, join_arr, lim_arr,
                      clip, box, mask_at, masks, bounds[c], bounds[c + 1], hmax, wmax, room,
                      bad[c:c + 1])


@njit(cache=True, nogil=True, error_model='numpy')
def _blit_mask(canvas, masks, at, w, x0, y0, ra, rb, ca, cb, r, g, b, a):
    """A marker's mask (rows y0.., w wide, from at) over rows ra..rb and
    columns ca..cb."""
    for i in range(ra, rb):
        base = at + (i - y0) * w - x0
        for j in range(ca, cb):
            m = masks[base + j]
            if m != 0:
                _blend(canvas, i, j, r, g, b, a, np.int64(m))


@njit(cache=True, nogil=True, error_model='numpy')
def _blit_band(canvas, masks, mask_at, box, face, edge, has_fill, has_stroke, order, a, b, r0, r1,
               c0, c1):
    """Rows r0..r1 and columns c0..c1: markers order[a:b] in turn, face then
    edge."""
    nf, ne = face.shape[0], edge.shape[0]
    for q in range(a, b):
        k = order[q]
        x0, y0, x1, y1 = box[k, 0], box[k, 1], box[k, 2], box[k, 3]
        w, h = x1 - x0, y1 - y0
        ra, rb = max(y0, r0), min(y1, r1)
        ca, cb = max(x0, c0), min(x1, c1)
        at = mask_at[k]
        if has_fill:
            c = k if nf > 1 else 0
            _blit_mask(canvas, masks, at, w, x0, y0, ra, rb, ca, cb, np.int64(face[c, 0]),
                       np.int64(face[c, 1]), np.int64(face[c, 2]), np.int64(face[c, 3]))
            at += w * h
        if has_stroke:
            c = k if ne > 1 else 0
            _blit_mask(canvas, masks, at, w, x0, y0, ra, rb, ca, cb, np.int64(edge[c, 0]),
                       np.int64(edge[c, 1]), np.int64(edge[c, 2]), np.int64(edge[c, 3]))


@njit(cache=True, nogil=True, error_model='numpy')
def _bin_markers(box, k0, k1, cy0, band_h, nb, starts, order, fill):
    """Which bands markers k0..k1 reach: their counts (fill False), then the
    lists in order (fill, starts each band's next free slot)."""
    for k in range(k0, k1):
        if box[k, 2] <= box[k, 0] or box[k, 3] <= box[k, 1]:
            continue
        ba = max(0, (box[k, 1] - cy0) // band_h)        # (a stamp can start above the clip)
        bb = min(nb - 1, (box[k, 3] - 1 - cy0) // band_h)
        for b in range(ba, bb + 1):
            if fill:
                order[starts[b]] = k
            starts[b] += 1


@njit(cache=True, nogil=True, parallel=True, error_model='numpy')
def _blit_bands(canvas, masks, mask_at, box, face, edge, has_fill, has_stroke, order, ends, clip,
                band_h, ntask):
    nb = ends.size - 1
    c0, cy0, c1, cy1 = clip[0], clip[1], clip[2], clip[3]
    for c in prange(ntask):                             # bands interleaved: a dense middle shared
        for b in range(c, nb, ntask):
            r0 = cy0 + b * band_h
            _blit_band(canvas, masks, mask_at, box, face, edge, has_fill, has_stroke, order,
                       ends[b], ends[b + 1], r0, min(r0 + band_h, cy1), c0, c1)


# A big scatter's markers, stamped: where exactness costs more than the eye can
# see. Curved and slanted outlines at the same size differ between places by
# which 1/256 of a pixel they start on, so each place has its own mask (above,
# ~7 us of a thread). Rounded to 1/8 of a pixel there are 64 places per size
# (one size in 1/8 of a pixel's steps, when sizes vary), and each mask, drawn
# by the same kernels, serves every marker there: blended in order as before,
# with the same colours, so what differs is coverage at an edge, a mean of
# 0.2 to 1.2 levels of 255 across a figure, 38 at most. Rectilinear markers
# stay exact: snapping turns a shift of 1/16 of a pixel into a whole one, and
# they cost less. rcParams['fastplot.exact'] turns stamps off.

@njit(cache=True, nogil=True, error_model='numpy')
def _has_rectilinear(v, codes, s):
    """Whether a marker at size s has a part that would be snapped."""
    m = v.shape[0]
    off = np.empty(m + 1, np.int64)
    xy, n, parts = _marker_flat(v, codes, s, 0.0, 0.0, np.empty((m, 2)), np.empty((256, 2)), off,
                                np.empty(m, np.bool_), np.nan)
    for q in range(parts):
        if _rectilinear(xy, off[q], off[q + 1]):
            return True
    return False


@njit(cache=True, nogil=True, parallel=True, error_model='numpy')
def _stamp_place(pts, which, first, step, reach, clip, at, box):
    """Each marker's mask (at) and box: at its place rounded to 1/8 of a
    pixel, which sets the whole pixel its size's reach is added to. The box
    is empty for one off the clip (or not finite), else it is left whole for
    the blit to cut, as the mask's rows run across it."""
    cx0, cy0, cx1, cy1 = clip[0], clip[1], clip[2], clip[3]
    for k in prange(pts.shape[0]):
        box[k, 0], box[k, 1], box[k, 2], box[k, 3] = 0, 0, 0, 0
        px, py = pts[k, 0], pts[k, 1]
        if abs(px) < 1e9 and abs(py) < 1e9:
            qx, qy = np.int64(np.floor(px * 8.0 + 0.5)), np.int64(np.floor(py * 8.0 + 0.5))
            ix, iy, s = qx >> 3, qy >> 3, which[k]
            x0, y0 = ix + reach[s, 0], iy + reach[s, 1]
            x1, y1 = ix + reach[s, 2], iy + reach[s, 3]
            if x0 < cx1 and y0 < cy1 and x1 > cx0 and y1 > cy0:
                at[k] = first[s] + step[s] * ((qy & 7) * 8 + (qx & 7))
                box[k, 0], box[k, 1], box[k, 2], box[k, 3] = x0, y0, x1, y1


MARKER_MASKS = 96 << 20             # bytes of masks at once: more markers go in turns
MARKER_TASKS = 8                    # tasks a thread
WIDE_MARKERS = 8192                 # markers from which masks take the cores' second threads
STAMP_MIN = 2000                    # markers from which they can be stamped: fewer stay exact
STAMP_GAIN = 2                      # and at least this many markers a mask
_LOCAL_MASKS = threading.local()


def _mask_buffer(n):
    b = getattr(_LOCAL_MASKS, 'masks', None)
    if b is None or b.size < n:
        b = _LOCAL_MASKS.masks = np.empty(max(n, 1 << 16), np.uint8)
    return b


def _stamped(canvas, pts, sizes, shapes, reach, extent, args, room, face, edge, clip, team):
    """Markers stamped (see above): False, with nothing drawn, if they are
    too few, or small or rectilinear, or too many masks a marker to pay."""
    n = pts.shape[0]
    lo, hi = float(sizes.min()), float(sizes.max())
    if n < STAMP_MIN or not lo >= 1.0 or not hi < 4096.0:
        return False
    if any(v.shape[0] and _has_rectilinear(v, c, lo) for v, c in shapes):
        return False
    layers = sum(v.shape[0] > 0 for v, _ in shapes)
    if lo == hi:
        kept, which = np.array([lo]), np.zeros(n, np.int64)
    else:                                               # the sizes there are, in steps of 1/8 px
        step = np.floor(sizes * 8.0 + 0.5).astype(np.int64)
        seen = np.zeros(int(step.max()) + 1, np.int64)
        seen[step] = 1
        kept = np.flatnonzero(seen) / 8.0
        which = (np.cumsum(seen) - 1)[step]
    many = kept.size * 64
    if STAMP_GAIN * many > n:
        return False
    vx0, vy0, vx1, vy1 = extent
    org = int(np.ceil(max(-vx0, -vy0, vx1, vy1, 0.0) * hi + reach)) + 3
    sub = np.arange(8) / 8.0
    x, y = np.meshgrid(org + sub, org + sub)
    mpts = np.tile(np.stack((x.ravel(), y.ravel()), 1), (kept.size, 1))
    msizes = np.repeat(kept, 64)
    b = np.stack((np.floor(org + vx0 * kept - reach), np.floor(org + vy0 * kept - reach),
                  np.floor(org + 1.0 + vx1 * kept + reach) + 2.0,
                  np.floor(org + 1.0 + vy1 * kept + reach) + 2.0), 1).astype(np.int32)
    area = (b[:, 2].astype(np.int64) - b[:, 0]) * (b[:, 3].astype(np.int64) - b[:, 1]) * layers
    if int(area.sum()) * 64 > MARKER_MASKS:
        return False
    mbox = np.repeat(b, 64, axis=0)
    mat = np.empty(many + 1, np.int64)
    mat[0], mat[1:] = 0, np.cumsum(np.repeat(area, 64))
    masks = _mask_buffer(int(mat[-1]))
    hmax, wmax = int((b[:, 3] - b[:, 1]).max()), int((b[:, 2] - b[:, 0]).max())
    tasks = min(MARKER_TASKS * team, many // 16 + 1)
    bad = np.zeros(tasks, np.int64)
    big = np.array([0.0, 0.0, 2.0 * org + 8.0, 2.0 * org + 8.0])
    all_args = (mpts, msizes) + args + (big, mbox, mat, masks)
    with _jit.wider(many >= WIDE_MARKERS):
        _marker_masks(*all_args, np.linspace(0, many, tasks + 1).astype(np.int64), hmax, wmax, room,
                      bad)
    if bad.any():
        return False
    at = _arena.take('markers.at', n, np.int64)
    box = _arena.take('markers.box', (n, 4), np.int32)
    _stamp_place(pts, which, mat[:-1:64].copy(), area, b - org, clip, at, box)
    _blit_markers(canvas, masks, at, box, face, edge, 0, n, shapes[0][0].shape[0] > 0,
                  shapes[1][0].shape[0] > 0, clip, team)
    return True


def draw_markers(canvas, pts, sizes, fill, stroke, width, snap_v, face, edge, clip, stamp=False):
    """Markers as a path collection draws them, each exactly at its place
    and size, face then edge (round joins, butt caps): fill and stroke
    (vertices, codes) of a marker of size 1, or None; sizes (n,); snap_v
    the rectilinear parts' snap_value; face, edge uint8 (n, 4) or (1, 4);
    clip (x0, y0, x1, y1) whole pixels. stamp: many may be drawn from masks
    placed to 1/8 of a pixel (_stamped). False if an outline went where its
    box did not expect, with nothing drawn."""
    n = pts.shape[0]
    d2, d1 = np.empty((0, 2)), np.empty(0, np.uint8)
    fv, fc = fill if fill is not None else (d2, d1)
    sv, sc = stroke if stroke is not None else (d2, d1)
    same = fill is not None and stroke is not None and fv is sv
    all_v = np.concatenate((fv, sv)) if not same else fv
    reach = 0.5 * width + 1.5                           # the stroke's half width, a snap's pixel,
    vx0, vy0 = all_v.min(axis=0)                        # room for rounding: and it is checked
    vx1, vy1 = all_v.max(axis=0)
    # arcs of a round join: _arc's points, a cap's or a join's at most
    w = 0.5 * width
    room = int(2.0 * np.pi / (np.arccos(w / (w + 0.125)) * 2.0)) + 4 if w > 0 else 4
    team = _jit.team()
    arrs = (np.array([width]), np.array([BUTT], np.int64), np.array([JOIN_ROUND], np.int64),
            np.array([width]))
    if stamp and _stamped(canvas, pts, sizes, ((fv, fc), (sv, sc)), reach, (vx0, vy0, vx1, vy1),
                          (fv, fc, sv, sc, same, snap_v) + arrs, room, face, edge, clip, team):
        return True
    box = np.empty((n, 4), np.int32)
    box[:, 0] = np.maximum(np.floor(np.minimum(pts[:, 0] + vx0 * sizes, 1e9) - reach), clip[0])
    box[:, 1] = np.maximum(np.floor(np.minimum(pts[:, 1] + vy0 * sizes, 1e9) - reach), clip[1])
    box[:, 2] = np.minimum(np.floor(np.maximum(pts[:, 0] + vx1 * sizes, -1e9) + reach) + 2, clip[2])
    box[:, 3] = np.minimum(np.floor(np.maximum(pts[:, 1] + vy1 * sizes, -1e9) + reach) + 2, clip[3])
    bw = np.maximum(box[:, 2].astype(np.int64) - box[:, 0], 0)
    bh = np.maximum(box[:, 3].astype(np.int64) - box[:, 1], 0)
    layers = (fill is not None) + (stroke is not None)
    area = bw * bh * layers
    used = np.cumsum(area)
    fclip = np.array(clip, np.float64)
    k0 = 0
    done = 0
    while k0 < n:
        k1 = max(int(np.searchsorted(used, done + MARKER_MASKS, 'right')), k0 + 1)
        size = int(used[k1 - 1]) - done
        masks = _mask_buffer(size)
        at = np.empty(k1 - k0 + 1, np.int64)
        at[0], at[1:] = 0, used[k0:k1] - done
        hmax, wmax = int(bh[k0:k1].max()), int(bw[k0:k1].max())
        if hmax and wmax:
            tasks = min(MARKER_TASKS * team, (k1 - k0) // 16 + 1)
            bounds = np.linspace(0, k1 - k0, tasks + 1).astype(np.int64)
            bad = np.zeros(tasks, np.int64)
            args = (pts[k0:k1], sizes[k0:k1], fv, fc, sv, sc, same, snap_v, *arrs, fclip,
                    box[k0:k1], at, masks)
            if tasks == 1:
                _marker_chunk(*args, 0, k1 - k0, hmax, wmax, room, bad)
            else:
                with _jit.wider(k1 - k0 >= WIDE_MARKERS):
                    _marker_masks(*args, bounds, hmax, wmax, room, bad)
            if bad.any():
                return False
            _blit_markers(canvas, masks, at, box[k0:k1], face, edge, k0, k1, fill is not None,
                          stroke is not None, clip, team)
        done = int(used[k1 - 1])
        k0 = k1
    return True


def _blit_markers(canvas, masks, at, box, face, edge, k0, k1, has_fill, has_stroke, clip, team):
    face = face if face.shape[0] == 1 else face[k0:k1]
    edge = edge if edge.shape[0] == 1 else edge[k0:k1]
    y0, y1 = clip[1], clip[3]
    h = y1 - y0
    nb = min(BAND_TASKS * team, max(h // BAND_ROWS, 1), max((k1 - k0) // 4, 1))
    band_h = -(-h // nb)
    nb = -(-h // band_h)
    n = k1 - k0
    starts = np.zeros(nb + 1, np.int64)
    d1 = np.empty(0, np.int32)
    _bin_markers(box, 0, n, y0, band_h, nb, starts, d1, False)
    pos = np.empty(nb + 1, np.int64)
    pos[0], pos[1:] = 0, np.cumsum(starts[:nb])
    order = np.empty(pos[nb], np.int32)
    fillp = pos.copy()
    _bin_markers(box, 0, n, y0, band_h, nb, fillp, order, True)
    if nb == 1 or n < 16:
        _blit_band(canvas, masks, at, box, face, edge, has_fill, has_stroke, order, 0, order.size,
                   y0, y1, clip[0], clip[2])
    else:
        _blit_bands(canvas, masks, at, box, face, edge, has_fill, has_stroke, order, pos,
                    np.array(clip, np.int64), band_h, min(team, nb))


def draw_strokes(canvas, xy, offsets, closed, part_style, part_path, colors, widths, caps, joins,
                 limits, clip, dashes=None, dash_starts=None, dash_offsets=None):
    """Polylines stroked and rasterized as Agg does: part_style and
    part_path (m,) give each its style and the path it is part of."""
    if dashes is not None and dashes.size:
        xy, offsets, part = dash(xy, offsets, closed, part_style, dashes, dash_starts,
                                 dash_offsets)
        closed = np.zeros(offsets.size - 1, np.bool_)
        part_style, part_path = part_style[part], part_path[part]
    pts, rings, ring_part = stroke(xy, offsets, closed, part_style, widths, caps, joins, limits)
    if rings.size < 2:
        return
    path_style = np.zeros(int(part_path.max()) + 1, np.int64)
    path_style[part_path[::-1]] = part_style[::-1]
    fill(canvas, pts, rings, part_path[ring_part], colors[path_style], clip)
