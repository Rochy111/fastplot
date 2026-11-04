"""Small solid lines drawn together: what _draw.small_lines draws one
kernel call each (a plot line, its path effects' strokes, a patch's edge)
kept in order and drawn by one call when the axes that drew them end --
a run of axes that draw only their artists, a hodograph map's insets,
the figure draws one after another (Axes._draw_artists) -- or something
else is to be drawn first.

    s = Strokes(canvas)
    r2 = Deferred(r, s)                 # a renderer whose lines() wait in s
    artist.render(r2, ax) ...           # (anything else drawn: s first)
    r2.flush()

The pixels are those of the calls one by one. Each line is checked,
snapped and stroked as _draw._small_lines does it; consecutive lines of
one clip box go to _agg._strokes_small as one call (a run) -- which sums
each path's edges and blends it before the next, in order, and decides
what is off the clip by the clip alone, so lines passed together put
down what they put down apart. A line small_lines would hand to the
steps (a NaN, a line leaving the canvas) is drawn by them in its turn.

A flush draws its runs on this thread (_strokes) or on a team of threads
(_strokes_par: the runs stroked at once; then rasterized level by level,
the runs of a level, which touch no pixel in common, at once), as this
machine has drawn as many points faster (CORES, 'strokes'). A hodograph
map's 64 insets x 14 strokes: 900 kernel calls and their Python, then
one, the insets' rasterizing on every thread of the draw.
"""
import threading
from array import array

import numpy as np

from . import _agg, _draw, _jit, _patches, _path
from ._jit import njit, prange
from ._style import rgba

# a group (one small_lines call) in Strokes.groups: its parts, how it is
# snapped (0 not, 1 rectilinear parts, 2 all) by what, one path or a path a
# part, the edge of a fill, its width, cap, join, clip box, colour
P0, P1, MODE, SV, LINE, FILLED, WIDTH, CAP, JOIN, X0, Y0, X1, Y1, R, G, B, A = range(17)
NG = 17

_LOCAL = threading.local()              # this thread's Strokes (its memory kept between axes)


@njit(cache=True, nogil=True)
def _collect(xy, offsets, groups, ng, g0, H, W):
    """Groups g0.. of a Strokes laid out for _strokes_small, up to the
    first one to be drawn the slow way (as _small_lines returns False:
    a NaN or inf; off an H x W canvas, unless the edge of a fill): each
    checked and snapped as _small_lines does, its parts' paths and closed
    rings as it makes them; consecutive groups of one clip box a run, a
    call. Returns (that first group or ng, the runs' first row, part and
    style (each nruns + 1), their clip boxes, then the runs' points, part
    ends (rows, each run's from 0), closed, part styles and paths (each
    run's from 0), styles' widths, caps, joins, colours)."""
    nrow, npart = 0, 0
    for q in range(g0, ng):
        p0, p1 = np.int64(groups[q, P0]), np.int64(groups[q, P1])
        nrow += offsets[p1] - offsets[p0]
        npart += p1 - p0
    k = ng - g0
    sx = np.empty((nrow, 2))
    off = np.empty(npart + 1, np.int64)
    closed = np.zeros(npart, np.bool_)
    style = np.empty(npart, np.int64)
    paths = np.empty(npart, np.int64)
    widths = np.empty(k)
    caps = np.empty(k, np.int64)
    joins = np.empty(k, np.int64)
    colors = np.empty((k, 4), np.uint8)
    rr = np.empty(k + 1, np.int64)
    rp = np.empty(k + 1, np.int64)
    rs = np.empty(k + 1, np.int64)
    clips = np.empty((k, 4))
    off[0] = 0
    r, p, s, nr = 0, 0, 0, 0
    f = g0
    while f < ng:
        if nr == 0 or groups[f, X0] != clips[nr - 1, 0] or groups[f, Y0] != clips[nr - 1, 1] or \
                groups[f, X1] != clips[nr - 1, 2] or groups[f, Y1] != clips[nr - 1, 3]:
            new = True                                  # (a run begins: another clip box)
        else:
            new = False
        p0, p1 = np.int64(groups[f, P0]), np.int64(groups[f, P1])
        ra, rb = offsets[p0], offsets[p1]
        ok = True
        for i in range(ra, rb):                         # (_small_lines: a NaN or inf: the steps)
            if not (abs(xy[i, 0]) < 1e300 and abs(xy[i, 1]) < 1e300):
                ok = False
        if not ok:
            break
        gx = xy[ra:rb]
        mode = np.int64(groups[f, MODE])
        if mode:
            go = offsets[p0:p1 + 1] - ra
            gx = _draw._snap_parts(gx, go, np.full(p1 - p0, groups[f, SV]), mode == 1)[0]
        if groups[f, FILLED] == 0.0:                    # (_strokes_small: off the canvas, cut first)
            for i in range(rb - ra):
                if gx[i, 0] < -1.0 or gx[i, 1] < -1.0 or gx[i, 0] > W + 1.0 or gx[i, 1] > H + 1.0:
                    ok = False
            if not ok:
                break
        if new:
            rr[nr], rp[nr], rs[nr] = r, p, s
            for c in range(4):
                clips[nr, c] = groups[f, X0 + c]
            nr += 1
            pid = 0
        base = r - rr[nr - 1]                           # (rows, parts, styles: the run's own)
        line = groups[f, LINE] != 0.0
        for i in range(rb - ra):
            sx[r + i, 0], sx[r + i, 1] = gx[i, 0], gx[i, 1]
        for q in range(p1 - p0):                        # (_small_lines' paths, ids rising)
            a, b = offsets[p0 + q] - ra, offsets[p0 + q + 1] - ra
            off[p + q + 1] = base + b
            style[p + q] = s - rs[nr - 1]
            if line:
                paths[p + q] = pid
            else:
                paths[p + q] = pid + q
                closed[p + q] = b - 1 - a >= 3 and gx[a, 0] == gx[b - 1, 0] and \
                    gx[a, 1] == gx[b - 1, 1]
        pid += 1 if line else p1 - p0
        widths[s] = groups[f, WIDTH]
        caps[s], joins[s] = np.int64(groups[f, CAP]), np.int64(groups[f, JOIN])
        for c in range(4):
            colors[s, c] = np.uint8(groups[f, R + c])
        r += rb - ra
        p += p1 - p0
        s += 1
        f += 1
    rr[nr], rp[nr], rs[nr] = r, p, s
    return (f, rr[:nr + 1], rp[:nr + 1], rs[:nr + 1], clips[:nr], sx, off, closed, style, paths,
            widths, caps, joins, colors)


@njit(cache=True, nogil=True)
def _run_offsets(off, pa, pb):
    """A run's part ends, from 0 (its first part's start 0)."""
    o = np.empty(pb - pa + 1, np.int64)
    o[0] = 0
    for q in range(pb - pa):
        o[q + 1] = off[pa + q + 1]
    return o


@njit(cache=True, nogil=True)
def _strokes(canvas, xy, offsets, groups, ng, g0, cells):
    """Groups g0.. of a Strokes in order, each run of one clip box as one
    _strokes_small call: the index of the first group to be drawn the
    slow way (as _small_lines returns False), or ng, all drawn."""
    f, rr, rp, rs, clips, sx, off, closed, style, paths, widths, caps, joins, colors = \
        _collect(xy, offsets, groups, ng, g0, canvas.shape[0], canvas.shape[1])
    for i in range(rr.size - 1):
        ra, rb, pa, pb, sa, sb = rr[i], rr[i + 1], rp[i], rp[i + 1], rs[i], rs[i + 1]
        _agg._strokes_small(canvas, sx[ra:rb], _run_offsets(off, pa, pb), closed[pa:pb],
                            style[pa:pb], paths[pa:pb], widths[sa:sb], caps[sa:sb], joins[sa:sb],
                            widths[sa:sb], colors[sa:sb], clips[i], True, cells)
    return f


@njit(cache=True, nogil=True, parallel=True, error_model='numpy')
def _strokes_par(canvas, xy, offsets, groups, ng, g0, team):
    """_strokes on a team of threads: the runs (each what _strokes_small
    does with it) stroked on all of them; then rasterized by levels -- a
    run a level above every earlier run whose pixel box it meets -- the
    runs of a level at once, each on one thread with its own cells. Runs
    of a level touch no pixel in common, and the runs meeting a pixel go
    in order, so every pixel is blended as one by one."""
    H, W = canvas.shape[0], canvas.shape[1]
    f, rr, rp, rs, clips, sx, off, closed, style, paths, widths, caps, joins, colors = \
        _collect(xy, offsets, groups, ng, g0, H, W)
    nr = rr.size - 1
    if nr == 0:
        return f
    src = np.empty((sx.shape[0], 3))
    counts = np.empty(closed.size, np.int64)
    shut = closed.copy()
    pk = np.zeros((nr, 2), np.int64)
    for i in prange(nr):                                # _strokes_small's counts, a run a task
        ra, rb, pa, pb, sa, sb = rr[i], rr[i + 1], rp[i], rp[i + 1], rs[i], rs[i + 1]
        lo = _run_offsets(off, pa, pb)
        _agg._sequences_serial(sx[ra:rb], lo, shut[pa:pb], src[ra:rb], counts[pa:pb])
        for q in range(pb - pa):
            if widths[sa + style[pa + q]] <= 0:
                counts[pa + q] = 0
        d2 = sx[:0]
        d1 = off[:0]
        p, k = _agg._stroke_steps(src[ra:rb], lo, counts[pa:pb], shut[pa:pb], style[pa:pb],
                                  widths[sa:sb], caps[sa:sb], joins[sa:sb], widths[sa:sb], 0, 0,
                                  pb - pa, 0, d2, d1, d1, 0, 0, False)
        pk[i, 0], pk[i, 1] = p, k
    po = np.zeros(nr + 1, np.int64)
    ko = np.zeros(nr + 1, np.int64)
    for i in range(nr):
        po[i + 1] = po[i] + pk[i, 0]
        ko[i + 1] = ko[i] + pk[i, 1]
    pts = np.empty((po[nr], 2))
    ipts = np.empty((po[nr], 2), np.int32)
    nxt = np.empty(po[nr], np.int64)
    e_path = np.empty(po[nr], np.int32)
    index = np.empty(po[nr], np.int32)
    rings = np.empty(ko[nr] + nr, np.int64)
    ring_part = np.empty(ko[nr], np.int64)
    ring_path = np.empty(ko[nr], np.int64)
    path_rgba = np.empty((ko[nr], 4), np.uint8)
    boxes = np.zeros((nr, 4), np.int64)
    ne = np.full(nr, -1, np.int64)                      # edges a run (-1: it shows nothing)
    for i in prange(nr):                                # its rings, paths, edges, box
        p, k = pk[i, 0], pk[i, 1]
        if k == 0:
            continue
        ra, rb, pa, pb, sa, sb = rr[i], rr[i + 1], rp[i], rp[i + 1], rs[i], rs[i + 1]
        a, b, ka, kb = po[i], po[i + 1], ko[i], ko[i + 1]
        lo = _run_offsets(off, pa, pb)
        rg = rings[ka + i:kb + i + 1]
        _agg._stroke_steps(src[ra:rb], lo, counts[pa:pb], shut[pa:pb], style[pa:pb],
                           widths[sa:sb], caps[sa:sb], joins[sa:sb], widths[sa:sb], 0, 0, pb - pa,
                           0, pts[a:b], rg, ring_part[ka:kb], 0, 0, True)
        rg[k] = p
        n = 0
        for j in range(k):                              # ids from 0; a path's colour its first ring's
            q = ring_part[ka + j]
            if j == 0 or paths[pa + q] != paths[pa + ring_part[ka + j - 1]]:
                c = sa + style[pa + q]
                for t in range(4):
                    path_rgba[ka + n, t] = colors[c, t]
                n += 1
            ring_path[ka + j] = n - 1
        box = np.empty(4)
        box[0], box[1], box[2], box[3] = 1e300, 1e300, -1e300, -1e300
        clip = clips[i]
        _agg._links_serial(rg, ring_path[ka:kb], nxt[a:b], e_path[a:b], pts[a:b], ipts[a:b], box,
                           clip)
        if box[0] > box[2]:
            continue
        x0 = max(np.int64(clip[0]), np.int64(np.floor(max(box[0], -1e9))))
        y0 = max(np.int64(clip[1]), np.int64(np.floor(max(box[1], -1e9))))
        x1 = min(np.int64(clip[2]), np.int64(np.floor(min(box[2], 1e9))) + 2)
        y1 = min(np.int64(clip[3]), np.int64(np.floor(min(box[3], 1e9))) + 2)
        if x1 <= x0 or y1 <= y0:
            continue
        e = 0
        for j in range(p):
            if nxt[a + j] >= 0:
                index[a + e] = j
                e += 1
        boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3] = x0, y0, x1, y1
        ne[i] = e
    level = np.full(nr, -1, np.int64)                   # the levels
    nl, hmax, wmax = 0, 0, 0
    for i in range(nr):
        if ne[i] < 0:
            continue
        m = -1
        for q in range(i):
            if ne[q] >= 0 and level[q] > m and boxes[q, 0] < boxes[i, 2] and \
                    boxes[i, 0] < boxes[q, 2] and boxes[q, 1] < boxes[i, 3] and \
                    boxes[i, 1] < boxes[q, 3]:
                m = level[q]
        level[i] = m + 1
        nl = max(nl, m + 2)
        hmax = max(hmax, boxes[i, 3] - boxes[i, 1])
        wmax = max(wmax, boxes[i, 2] - boxes[i, 0])
    at = np.zeros(nl + 1, np.int64)
    for i in range(nr):
        if level[i] >= 0:
            at[level[i] + 1] += 1
    for L in range(nl):
        at[L + 1] += at[L]
    order = np.empty(at[nl], np.int64)
    fill = at[:nl].copy()
    for i in range(nr):
        if level[i] >= 0:
            order[fill[level[i]]] = i
            fill[level[i]] += 1
    cols = wmax + 2                                     # (_agg._cells' for the largest box)
    words = (cols >> 6) + 1
    cell = np.zeros((team, 2 * hmax * cols), np.int32)
    bits = np.zeros((team, hmax * words), np.int64)
    span = np.empty((team, 2 * hmax), np.int32)
    for t in range(team):
        for j in range(hmax):
            span[t, 2 * j], span[t, 2 * j + 1] = 1 << 30, -1
    for L in range(nl):
        a0, b0 = at[L], at[L + 1]
        nt = min(team, b0 - a0)
        for t in prange(nt):
            cells = (cell[t], bits[t], span[t], 2 * cols, words)
            for q in range(a0 + t, b0, nt):
                i = order[q]
                a, b, ka, kb = po[i], po[i + 1], ko[i], ko[i + 1]
                x0, y0, x1, y1 = boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3]
                _agg._band(canvas, pts[a:b], ipts[a:b], nxt[a:b], 0, ne[i], index[a:b],
                           e_path[a:b], path_rgba[ka:kb], x0, y0, x1, y1, 0, y1 - y0, cells,
                           False)
    return f


@njit(cache=True, nogil=True)
def _ellipse_into(ux, uy, codes, sx, sy, cx, cy, x0, ex, kx, w, y1, ey, ky, h, tol, xy, off,
                  closed, n, p):
    """_patches.ellipse (angle 0: sx * unit + c) mapped by _linear_px's
    numbers, then _path.flatten, its points written from xy[n], its parts
    from off[p]: (points end, parts end), or (-1, -1) when they do not
    fit (nothing written)."""
    m = ux.size
    v = np.empty((m, 2))
    for i in range(m):
        v[i, 0] = x0 + (((sx * ux[i] + cx) - ex) * kx) * w
        v[i, 1] = y1 - (((sy * uy[i] + cy) - ey) * ky) * h
    n2, p2 = _path._walk(v, codes, 0, m, tol, xy, off, closed, n, p, False, 0)
    if n2 > xy.shape[0] or p2 >= off.size or p2 > closed.size:
        return -1, -1
    _path._walk(v, codes, 0, m, tol, xy, off, closed, n, p, True, n2)
    off[p2] = n2
    return n2, p2


class Strokes:
    """Small solid lines waiting, in order, for one kernel call (flush):
    points (canvas px) by parts, and groups -- what one small_lines call
    draws -- over consecutive parts (several groups may stroke the same
    ones: a line and its path effects)."""

    def __init__(self, canvas):
        self.xy = np.empty((1024, 2))
        self.off = np.zeros(257, np.int64)
        self.closed = np.empty(257, np.bool_)           # (flatten's, unused)
        self.busy = False
        self.reset(canvas)

    def reset(self, canvas):
        """Empty, for canvas (the memory kept)."""
        self.canvas = canvas
        self.a = canvas.a
        self.H, self.W = self.a.shape[:2]
        self.groups = array('d')                        # rows of NG numbers, one after another
        self._boxes = []                                # their clip boxes, each once
        self.n = self.p = 0
        self.low = _draw.SMALL_MAX                      # points: fewer, small_lines' own call
        self.box_all = self.box(None)
        # the axes' (its _draw sets them): small_lines' box for its box_clip;
        # data -> px, (to_px, its numbers after x, y) for plain linear axes
        self.box_clip = self.box_all
        self.to_px = self.linear = None
        self._colors = {}                               # (colour, alpha) -> rgba, this draw's
        return self

    def rgba(self, c, alpha=None):
        """_style.rgba(c, alpha) as 4 ints, for stroke() (kept for the
        axes' run: the settings do not change within a draw); None for
        no colour."""
        try:
            return self._colors[c, alpha]
        except KeyError:
            v = rgba(c, alpha)
            v = self._colors[c, alpha] = None if v is None else tuple(v.tolist())
        except TypeError:                               # (unhashable: as it comes)
            v = rgba(c, alpha)
            v = None if v is None else tuple(v.tolist())
        return v

    @classmethod
    def for_canvas(cls, canvas):
        """This thread's Strokes, emptied for canvas (a new one while it
        is in use)."""
        s = getattr(_LOCAL, 'strokes', None)
        if s is None or s.busy:
            s = cls(canvas)
            if getattr(_LOCAL, 'strokes', None) is None:
                _LOCAL.strokes = s
            return s
        return s.reset(canvas)

    def release(self):
        """Done with: its canvas let go, free for the next axes."""
        self.canvas = self.a = None
        self.busy = False

    def box(self, clip):
        """small_lines' clip box on this canvas (None: nothing shows)."""
        return _draw._clip_to(self.a, clip)

    def points(self, k):
        """Room for one part of k points: (its index, the rows to fill)."""
        n, p = self.n, self.p
        if n + k > self.xy.shape[0]:
            self.xy = np.concatenate([self.xy[:n], np.empty((max(n + k, 2 * n), 2))])
        if p + 2 > self.off.size:
            self.off = np.concatenate([self.off[:p + 1], np.empty(p + 2, np.int64)])
        self.n = n + k
        self.p = p + 1
        self.off[p + 1] = n + k
        return p, self.xy[n:n + k]

    def parts(self, xy, offsets):
        """Parts xy (n, 2) by offsets (starting 0, ending n), copied in:
        (first part, last part + 1)."""
        n, p, k, m = self.n, self.p, xy.shape[0], offsets.size - 1
        if n + k > self.xy.shape[0]:
            self.xy = np.concatenate([self.xy[:n], np.empty((max(n + k, 2 * n), 2))])
        if p + m + 1 > self.off.size:
            self.off = np.concatenate([self.off[:p + 1], np.empty(p + m + 1, np.int64)])
        self.xy[n:n + k] = xy
        self.off[p + 1:p + m + 1] = offsets[1:] + n
        self.n, self.p = n + k, p + m
        return p, p + m

    def ellipse(self, sx, sy, cx, cy, ahead=256):
        """_patches.ellipse(angle 0)'s Beziers, mapped by self.linear and
        flattened into the points, not yet taken: (first point, end, first
        part, end) -- take() keeps them."""
        n, p = self.n, self.p
        while True:
            if n + ahead > self.xy.shape[0]:
                self.xy = np.concatenate([self.xy[:n], np.empty((n + ahead, 2))])
            if p + 65 > self.off.size:
                self.off = np.concatenate([self.off[:p + 1], np.empty(p + 65, np.int64)])
            if self.closed.size < self.off.size:
                self.closed = np.empty(self.off.size, np.bool_)
            n2, p2 = _ellipse_into(_patches._CIRCLE_X, _patches._CIRCLE_Y, _patches._CIRCLE_CODES,
                                   sx, sy, cx, cy, *self.linear, 0.5, self.xy, self.off,
                                   self.closed, n, p)
            if n2 >= 0:
                return n, n2, p, p2
            ahead *= 4                                  # (a curve of many points: room, again)

    def take(self, n, p):
        """The points and parts written up to n, p kept."""
        self.n, self.p = n, p

    def stroke(self, p0, p1, snap, sv, line, filled, width, cap, join, box, color):
        """A group: parts p0..p1-1 as small_lines(snap, width ...) strokes
        them, clipped to box (small_lines' own), color uint8 (4,)."""
        c = color if type(color) is tuple else color.tolist()   # (uint8 (4,), or 4 ints)
        self.groups.extend((p0, p1, 0 if snap is False else (1 if snap is None else 2), sv, line,
                            filled, width, cap, join, box[0], box[1], box[2], box[3],
                            c[0], c[1], c[2], c[3]))
        if box not in self._boxes:
            self._boxes.append(box)

    def lines(self, xy, offsets, color, width_px, clip, snap, cap, join, line, filled):
        """small_lines(canvas, ...) put off: True if it is to draw them (or
        nothing shows), False for what it would hand back -- a style or
        shape it does not draw, SMALL_MAX points or more (_draw.small_lines'
        first checks, in its order; the way of what it draws from
        CORES.low points up is the flush's to choose)."""
        caps, joins = _draw._SOLID_CAPS.get(cap), _draw._SOLID_JOINS.get(join)
        if caps is None or joins is None:
            return False
        if type(width_px) is not float and np.ndim(width_px) != 0:
            return False
        if type(xy) is not np.ndarray:
            xy = np.asarray(xy)
        if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] >= _draw.SMALL_MAX:
            return False
        colors = color if type(color) is np.ndarray else np.asarray(color)
        if colors.shape != (4,):
            return False
        box = self.box(clip)
        if box is None:
            return True
        offsets = _draw._rw(offsets, np.int64)
        m, n = offsets.size - 1, xy.shape[0]
        if m <= 0 or n == 0:
            return True
        if offsets[0] != 0 or offsets[m] != n:
            return False
        w = float(width_px)
        col = _draw._rw(colors[None], np.uint8)[0]
        p0, p1 = self.parts(_draw._rw(xy, np.float64), offsets)
        self.stroke(p0, p1, snap, float(_draw.snap_value(w)), bool(line), bool(filled), w,
                    int(caps[0]), int(joins[0]), box, col)
        return True

    def flush(self, r, way=None):
        """Everything waiting drawn, in order; r: the renderer whose
        lines() draw what the kernel leaves (as small_lines' False). On
        this thread or the runs on a team of threads (_strokes_par), as
        this machine has drawn as many points faster (way: that one)."""
        if not self.groups:
            return
        groups = np.frombuffer(self.groups, np.float64).reshape(-1, NG)    # (its own memory now)
        self.groups = array('d')
        ng = groups.shape[0]
        boxes, self._boxes = self._boxes, []
        xy, off = self.xy, self.off
        way, tick = _agg.CORES.pick('strokes', self.n) if way is None else (way, None)
        team = _jit.team() if way != 1 else max(2, _jit.team())
        if way and team > 1:
            cells = None
        else:
            way = 0
            cells = _agg._cells(max(b[3] - b[1] for b in boxes), max(b[2] - b[0] for b in boxes))
        g, slow = 0, False
        while g < ng:
            g = _strokes_par(self.a, xy, off, groups, ng, g, team) if way else \
                _strokes(self.a, xy, off, groups, ng, g, cells)
            if g < ng:                                  # the steps, in its turn
                slow = True
                self._slow(r, groups[g])
                g += 1
        _agg.CORES.took(None if slow else tick, way)    # (timed: the kernel's work alone)
        self.n = self.p = 0

    def _slow(self, r, G):
        """Group G by r.lines (whose small_lines hands it to the steps)."""
        p0, p1 = int(G[P0]), int(G[P1])
        a = self.off[p0]
        xy = self.xy[a:self.off[p1]].copy()
        offs = self.off[p0:p1 + 1] - a
        mode = int(G[MODE])
        r.lines(xy, offs, G[R:A + 1].astype(np.uint8), float(G[WIDTH]),
                tuple(int(v) for v in G[X0:Y1 + 1]), None,
                (False, None, True)[mode], int(G[CAP]), False, int(G[JOIN]), line=bool(G[LINE]),
                filled=bool(G[FILLED]))


class _Gate:
    """The canvas, seen through a Deferred: what is asked of it is asked
    after the waiting lines are drawn."""

    def __init__(self, owner, canvas):
        self._owner, self._canvas = owner, canvas

    def __getattr__(self, name):
        self._owner.flush()
        return getattr(self._canvas, name)


class Deferred:
    """A renderer (_figure._Renderer) for one axes' draw whose small solid
    lines wait in a Strokes: lines() as the renderer's, put off where
    small_lines would draw them; anything else of it or of its canvas
    asked for draws them first."""

    def __init__(self, r, strokes):
        d = self.__dict__
        d['_r'], d['strokes'] = r, strokes
        d['_canvas'] = r.c
        d['_gate'] = _Gate(self, r.c)
        for k in ('fig', 'dpi', 'W', 'H', 'ox', 'oy', 'extents', 'measuring', 'px'):
            d[k] = getattr(r, k)

    @property
    def c(self):
        return self._gate if self._r.c is self._canvas else self._r.c

    @c.setter
    def c(self, v):                                     # (a field's canvas swapped and back)
        self.flush()
        self._r.c = self._canvas if v is self._gate else v

    def flush(self):
        self.strokes.flush(self._r)

    def lines_now(self, *args, **kw):
        """The renderer's lines() at once (what small_lines leaves to
        CORES, or will not draw), after what waits."""
        self.flush()
        self._r.lines(*args, **kw)

    def lines(self, xy, offsets, color, width_px, clip, dashes_px=None, snap=False,
              cap=_draw.BUTT, simplify=False, join=_draw.JOIN_ROUND, line=False, filled=False):
        if dashes_px is None and self._r.c is self._canvas and \
                self.strokes.lines(xy, offsets, color, width_px, clip, snap, cap, join, line,
                                   filled):
            return
        self.flush()
        self._r.lines(xy, offsets, color, width_px, clip, dashes_px, snap, cap, simplify, join,
                      line, filled)

    def __getattr__(self, name):
        self.flush()
        return getattr(self._r, name)

    def __setattr__(self, name, v):
        if name == 'c':
            object.__setattr__(self, name, v)
        else:
            self.flush()
            setattr(self._r, name, v)


def usable(r):
    """Can r's lines be put off (a CPU canvas drawn on, untimed choices)?"""
    c = r.c
    return not r.measuring and getattr(c, 'device', None) == 'cpu' and \
        getattr(c, 'a', None) is not None and _agg.CORES.force is None
