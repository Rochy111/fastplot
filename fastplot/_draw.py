"""Anti-aliased lines and filled polygons, drawn straight into an RGBA
canvas by tiles (the CPU side; _gpu runs the same tile and row kernels).

    segs = segments(xy, offsets, style, dashes)       # polylines -> segments
    bins = bin_segments(segs, width, clip)            # into 32x32 px tiles
    draw_tiles(canvas, segs, bins, colors, widths, dash_tables)
    draw_lines(canvas, xy, offsets, color, width)      # the three above
    fill_polygons(canvas, xy, ring_offsets, rgba, clip)

Lines. A polyline (screen pixels, NaN breaking it) becomes segments, each
carrying its style (colour, width, dashes) and the arc length at its start
(dashes run on along a polyline). Segments are binned into the 32x32
pixel tiles their stroke can reach -- a tile is kept only if its centre
lies within (tile half-diagonal + half width + 1) of the segment, so a
long diagonal does not claim its whole bounding box. Each tile then runs
on its own core: every pixel takes, from each segment, the box-filtered
coverage of a stroke of width w at distance d from its centre,

    cov = max(0, min(w/2, d + 1/2) - max(-w/2, d - 1/2))

(1 inside a wide line, w for a hairline, a ramp across the edge) -- the
maximum over the segments of one style, so joints of a polyline are not
drawn twice (a thin translucent line would darken at every vertex), then
blended once. Runs of segments of one style are flushed in draw order, so
overlapping styles stack as they were drawn. Round caps and joins come
from the distance to a segment being clamped to its ends (cap LEGACY, the
map styles'); the matplotlib caps BUTT / PROJECTING / ROUND take, at a
polyline's two ends, the coverage across the line times the box filter
along it over the stroke's extent -- and along a dashed line over the
dashes, so dash ends are anti-aliased butt ends, as Agg draws them.
Interior vertices join round.

Polygons are filled by scanlines, four sub-rows a pixel row: every edge is
listed under the rows it crosses; a row gathers its edges' crossings at
each sub-row, sorts them and covers the spans between (even-odd, or
nonzero winding), with fractional coverage at span ends -- so edges are
anti-aliased across, and to a quarter pixel along. Rings of a polygon and
its holes go in one call.
"""
import numpy as np

from . import _jit
from ._jit import njit, prange

TILE = 32                           # tile side, px
SUB = 4                             # sub-rows a pixel row (polygons)

_F0 = np.float32(0.0)
_F1 = np.float32(1.0)
_FH = np.float32(0.5)
_F255 = np.float32(255.0)
_FBIG = np.float32(1e30)

LEGACY, BUTT, PROJECTING, ROUND = 0, 1, 2, 3        # cap styles (LEGACY: the maps')
CAPS = {'butt': BUTT, 'projecting': PROJECTING, 'round': ROUND}


@njit(cache=True, nogil=True)
def blend_px(canvas, i, j, r, g, b, a, m):
    """The internals.md blend: colour (r, g, b, a) 0..255 at coverage m,
    "over" in straight alpha. Over an opaque pixel (every pixel of a figure
    with a background) src f + dst (1 - f); over a partly transparent one
    the destination's colour weighs by its alpha too, and the sum is
    divided by the new alpha -- else a transparent page's colour bleeds
    into anti-aliased edges (a black line on transparent white, lighter)."""
    f = (a / _F255) * m
    k = _F1 - f
    da = np.float32(canvas[i, j, 3])
    if da == _F255:
        canvas[i, j, 0] = np.uint8(r * f + np.float32(canvas[i, j, 0]) * k + _FH)
        canvas[i, j, 1] = np.uint8(g * f + np.float32(canvas[i, j, 1]) * k + _FH)
        canvas[i, j, 2] = np.uint8(b * f + np.float32(canvas[i, j, 2]) * k + _FH)
        canvas[i, j, 3] = np.uint8(_F255 * f + da * k + _FH)
        return
    w = (da / _F255) * k
    oa = f + w
    if oa <= _F0:
        return
    canvas[i, j, 0] = np.uint8((r * f + np.float32(canvas[i, j, 0]) * w) / oa + _FH)
    canvas[i, j, 1] = np.uint8((g * f + np.float32(canvas[i, j, 1]) * w) / oa + _FH)
    canvas[i, j, 2] = np.uint8((b * f + np.float32(canvas[i, j, 2]) * w) / oa + _FH)
    canvas[i, j, 3] = np.uint8(_F255 * oa + _FH)


# ---------------------------------------------------------------- segments

@njit(cache=True, nogil=True)
def _count_part(xy, a, b):
    """The drawable segments of the polyline xy[a:b]."""
    n = 0
    for k in range(a, b - 1):
        if (np.isfinite(xy[k, 0]) and np.isfinite(xy[k, 1])
                and np.isfinite(xy[k + 1, 0]) and np.isfinite(xy[k + 1, 1])):
            n += 1
    return n


@njit(cache=True, nogil=True)
def _fill_part(xy, a, b, st, n, out):
    """The segments of the polyline xy[a:b] (style st) into out from row n:
    x0, y0, x1, y1, style, arc length at the start, ends (1: the start ends
    a polyline, 2: the end does; a closed ring -- first point repeated last
    -- has none). Returns the row after them."""
    s = 0.0
    b -= 1
    closed = (b - a >= 2 and xy[a, 0] == xy[b, 0] and xy[a, 1] == xy[b, 1])
    for k in range(a, b):
        x0, y0, x1, y1 = xy[k, 0], xy[k, 1], xy[k + 1, 0], xy[k + 1, 1]
        if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1)):
            s = 0.0                                     # a break restarts the dashes
            continue
        ends = 0
        if k == a or not (np.isfinite(xy[k - 1, 0]) and np.isfinite(xy[k - 1, 1])):
            ends |= 1
        if k + 1 == b or not (np.isfinite(xy[k + 2, 0]) and np.isfinite(xy[k + 2, 1])):
            ends |= 2
        if closed:
            ends &= ~((1 if k == a else 0) | (2 if k + 1 == b else 0))
        out[n, 0] = x0
        out[n, 1] = y0
        out[n, 2] = x1
        out[n, 3] = y1
        out[n, 4] = st
        out[n, 5] = s
        out[n, 6] = ends
        s += np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        n += 1
    return n


@njit(cache=True, nogil=True, parallel=True)
def _count_segments(xy, offsets, counts):
    """counts[p + 1]: the drawable segments of polyline p."""
    for p in prange(offsets.size - 1):
        counts[p + 1] = _count_part(xy, offsets[p], offsets[p + 1])


@njit(cache=True, nogil=True, parallel=True)
def _fill_segments(xy, offsets, part_style, starts, out):
    """out (n, 7) float32: polyline p's segments (_fill_part) from starts[p]."""
    for p in prange(offsets.size - 1):
        _fill_part(xy, offsets[p], offsets[p + 1], part_style[p], starts[p], out)


@njit(cache=True, nogil=True)
def _segments_serial(xy, offsets, part_style):
    """segments() of a small draw in one call on this thread: a parallel
    region's start costs more than the work."""
    n = 0
    for p in range(offsets.size - 1):
        n += _count_part(xy, offsets[p], offsets[p + 1])
    out = np.empty((n, 7), np.float32)
    n = 0
    for p in range(offsets.size - 1):
        n = _fill_part(xy, offsets[p], offsets[p + 1], part_style[p], n, out)
    return out


SERIAL_POINTS = 4096                # vertices (segments) below which a draw stays on one thread


def _rw(a, dtype):
    """a as a C-contiguous, writable array of dtype (a copy only when it
    is not one). numba types a read-only array apart, compiling every
    kernel it reaches again (seconds each, on a new machine): the kernels
    are given writable ones throughout."""
    a = np.ascontiguousarray(a, dtype=dtype)
    return a if a.flags.writeable else a.copy()


def segments(xy, offsets, part_style=None):
    """Polylines (xy (n, 2) screen px, offsets (m+1)) -> segments (k, 7)
    float32; part_style (m,) gives each polyline's style index (0)."""
    xy = _rw(xy, np.float64)
    offsets = _rw(offsets, np.int64)
    if part_style is None:
        part_style = np.zeros(max(offsets.size - 1, 0), np.int32)
    if offsets.size < 2:
        return np.empty((0, 7), np.float32)
    if xy.shape[0] < SERIAL_POINTS:
        return _segments_serial(xy, offsets, np.ascontiguousarray(part_style, dtype=np.int32))
    starts = np.zeros(offsets.size, np.int64)
    _count_segments(xy, offsets, starts)
    np.cumsum(starts, out=starts)
    out = np.empty((starts[-1], 7), np.float32)
    if out.shape[0]:
        _fill_segments(xy, offsets, np.ascontiguousarray(part_style, dtype=np.int32), starts,
                       out)
    return out


@njit(cache=True, nogil=True)
def _seg_tile_near(x0, y0, x1, y1, cx, cy, reach):
    """Is the tile centre (cx, cy) within reach of the segment?"""
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    t = 0.0
    if L2 > 0.0:
        t = ((cx - x0) * dx + (cy - y0) * dy) / L2
        t = min(1.0, max(0.0, t))
    ex, ey = x0 + t * dx - cx, y0 + t * dy - cy
    return ex * ex + ey * ey <= reach * reach


@njit(cache=True, nogil=True)
def _bin(segs, widths, cx0, cy0, cx1, cy1, ntx, nty, s0, s1, counts, fill, index):
    """One pass of binning segments [s0, s1): counts (fill=False) or the
    tile lists (fill, counts holding each tile's next free slot)."""
    half = TILE * 0.7072                                # half diagonal
    for s in range(s0, s1):
        x0, y0, x1, y1 = segs[s, 0], segs[s, 1], segs[s, 2], segs[s, 3]
        hw = widths[np.int64(segs[s, 4])] * 0.5 + 1.0
        bx0, bx1 = min(x0, x1) - hw, max(x0, x1) + hw
        by0, by1 = min(y0, y1) - hw, max(y0, y1) + hw
        if bx1 < cx0 or bx0 > cx1 or by1 < cy0 or by0 > cy1:
            continue
        tx0 = max(0, np.int64(np.floor((bx0 - cx0) / TILE)))
        tx1 = min(ntx - 1, np.int64(np.floor((bx1 - cx0) / TILE)))
        ty0 = max(0, np.int64(np.floor((by0 - cy0) / TILE)))
        ty1 = min(nty - 1, np.int64(np.floor((by1 - cy0) / TILE)))
        big = (tx1 - tx0) + (ty1 - ty0) > 2
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                if big and not _seg_tile_near(x0, y0, x1, y1, cx0 + (tx + 0.5) * TILE,
                                              cy0 + (ty + 0.5) * TILE, half + hw):
                    continue
                t = ty * ntx + tx
                if fill:
                    index[counts[t]] = s
                counts[t] += 1


@njit(cache=True, nogil=True, parallel=True)
def _bin_chunks(segs, widths, cx0, cy0, cx1, cy1, ntx, nty, bounds, counts, fill, index):
    """_bin over chunks of segments at once, chunk c in counts[c]."""
    for c in prange(bounds.size - 1):
        _bin(segs, widths, cx0, cy0, cx1, cy1, ntx, nty, bounds[c], bounds[c + 1], counts[c],
             fill, index)


@njit(cache=True, nogil=True)
def _chunk_slots(counts, starts):
    """Tile starts (prefix of the chunks' counts) and, in counts, each
    chunk's first slot in each tile: chunk after chunk, so every tile lists
    its segments in increasing order, as one pass in order would."""
    nc, nt = counts.shape
    pos = 0
    for t in range(nt):
        starts[t] = pos
        for c in range(nc):
            n = counts[c, t]
            counts[c, t] = pos
            pos += n
    starts[nt] = pos


@njit(cache=True, nogil=True)
def _bin_serial(segs, widths, cx0, cy0, cx1, cy1, ntx, nty):
    """bin_segments of one chunk, both passes in one call on this thread."""
    nt = ntx * nty
    counts = np.zeros(nt, np.int64)
    starts = np.zeros(nt + 1, np.int64)
    dummy = np.empty(0, np.int32)
    _bin(segs, widths, cx0, cy0, cx1, cy1, ntx, nty, 0, segs.shape[0], counts, False, dummy)
    pos = 0
    for t in range(nt):
        starts[t] = pos
        n = counts[t]
        counts[t] = pos
        pos += n
    starts[nt] = pos
    index = np.empty(pos, np.int32)
    if pos:
        _bin(segs, widths, cx0, cy0, cx1, cy1, ntx, nty, 0, segs.shape[0], counts, True, index)
    return starts, index


BIN_CHUNK = 4096                    # segments a chunk (binning in parallel)


def bin_segments(segs, widths, clip):
    """Segments into tiles of the clip rectangle (x0, y0, x1, y1), whole
    pixels: (starts (ntiles+1) int64, index int32 of segments, ntx, nty)."""
    cx0, cy0, cx1, cy1 = clip
    ntx = max(1, -(-(cx1 - cx0) // TILE))
    nty = max(1, -(-(cy1 - cy0) // TILE))
    widths = np.ascontiguousarray(widths, dtype=np.float64)
    n = segs.shape[0]
    nc = max(1, min(64, -(-n // BIN_CHUNK)))
    if nc == 1:                                         # one chunk: no parallel region
        starts, index = _bin_serial(segs, widths, cx0, cy0, cx1, cy1, ntx, nty)
        return starts, index, ntx, nty
    bounds = np.linspace(0, n, nc + 1).astype(np.int64)
    counts = np.zeros((nc, ntx * nty), np.int64)
    starts = np.zeros(ntx * nty + 1, np.int64)
    dummy = np.empty(0, np.int32)
    if n:
        _bin_chunks(segs, widths, cx0, cy0, cx1, cy1, ntx, nty, bounds, counts, False, dummy)
    _chunk_slots(counts, starts)
    index = np.empty(starts[-1], np.int32)
    if index.size:
        _bin_chunks(segs, widths, cx0, cy0, cx1, cy1, ntx, nty, bounds, counts, True, index)
    return starts, index, ntx, nty


# ------------------------------------------------------------------ tiles

@njit(cache=True, nogil=True)
def _dash_on(x, dashes, da, ndash, period):
    """The length of the dash pattern that is on over [0, x)."""
    k = np.floor(x / period)
    ph = x - period * k
    on = _F0
    total = _F0
    acc = _F0
    for q in range(ndash):
        L = dashes[da + q]
        if q % 2 == 0:
            total += L
            on += min(max(ph - acc, _F0), L)
        acc += L
    return k * total + on


@njit(cache=True, nogil=True)
def _coverage(px, py, x0, y0, dx, dy, inv2, ln, hw, lim2, s0, dashes, da, ndash, period, doff,
              cap, ends):
    """A segment's box-filtered stroke coverage at the pixel centre (px, py):
    the segment from (x0, y0) along (dx, dy), inv2 = 1/|d|^2 (0 for a
    point), ln = |d|; lim2: the squared distance from it beyond which
    nothing is covered; dashes[da:da+ndash] its style's dash pattern
    (ndash 0: solid); cap its style's cap, ends its polyline-end flags."""
    tr = ((px - x0) * dx + (py - y0) * dy) * inv2      # float32 throughout (as on the GPU)
    t = min(_F1, max(_F0, tr))
    ex, ey = x0 + t * dx - px, y0 + t * dy - py
    d2 = ex * ex + ey * ey
    if d2 >= lim2:
        return _F0
    if cap != LEGACY:
        c0 = (ends & 1) != 0 and cap != ROUND
        c1 = (ends & 2) != 0 and cap != ROUND
        if inv2 == _F0:                                 # a point: a square, or nothing
            if cap == BUTT:
                return _F0
            if cap == PROJECTING:
                ax_ = min(hw, abs(ex) + _FH) - max(-hw, abs(ex) - _FH)
                ay_ = min(hw, abs(ey) + _FH) - max(-hw, abs(ey) - _FH)
                return min(max(ax_, _F0) * max(ay_, _F0), _F1)
        elif not ((tr < _F0 and not c0) or (tr > _F1 and not c1)):
            # across: from the line; along: the box over the stroke (and dashes)
            dp = abs((px - x0) * dy - (py - y0) * dx) / ln
            c = min(hw, dp + _FH) - max(-hw, dp - _FH)
            if c <= _F0:
                return _F0
            u = tr * ln
            lo = -_FBIG if not c0 else (_F0 if cap == BUTT else -hw)
            hi = _FBIG if not c1 else (ln if cap == BUTT else ln + hw)
            a = max(lo, u - _FH)
            b = min(hi, u + _FH)
            if b <= a:
                return _F0
            if ndash:
                base = s0 + doff
                b = _dash_on(base + b, dashes, da, ndash, period) - \
                    _dash_on(base + a, dashes, da, ndash, period)
                a = _F0
            return min(c * (b - a), _F1)
    d = np.sqrt(d2)
    c = min(hw, d + _FH) - max(-hw, d - _FH)
    if c <= _F0:
        return _F0
    if ndash:
        s = s0 + t * ln + doff
        ph = s - period * np.floor(s / period)
        on = False
        acc = _F0
        for q in range(ndash):
            nxt = acc + dashes[da + q]
            if ph < nxt:
                on = (q % 2) == 0
                break
            acc = nxt
        if not on:
            return _F0
    return min(c, _F1)


@njit(cache=True, nogil=True)
def _top(hw, cap):
    """The most a pixel can take of a style's stroke (_coverage never
    returns more): min(hw + hw, 1) for LEGACY -- min(hw, .) - max(-hw, .)
    rounds to at most hw - -hw -- and 1 for the caps, whose box product
    can round a hair over c."""
    return min(hw + hw, _F1) if cap == LEGACY else _F1


@njit(cache=True, nogil=True)
def _style(st, widths, dashes, dash_starts, dash_offsets, caps):
    """A style's half width, dash pattern (start, count, period), offset,
    cap and the reach of its stroke from a segment."""
    da, db = dash_starts[st], dash_starts[st + 1]
    period = _F0
    for q in range(da, db):
        period += dashes[q]
    ndash = db - da if period > _F0 else 0
    hw = np.float32(widths[st] * 0.5)
    cap = caps[st]
    lim = hw + _FH if cap == LEGACY else hw * np.float32(1.4143) + np.float32(0.75)
    return hw, da, ndash, period, np.float32(dash_offsets[st]), cap, lim


TILE_TASKS = 12                     # tile-drawing tasks a thread: tile t goes to task t % tasks


@njit(cache=True, nogil=True, parallel=True)
def _draw_tiles(canvas, segs, starts, index, ntx, nty, cx0, cy0, cx1, cy1,
                colors, widths, dashes, dash_starts, dash_offsets, caps, ntask):
    nt = ntx * nty
    busy = np.empty(nt, np.int64)                       # the tiles with segments, in order
    nb = 0
    for t in range(nt):
        if starts[t] != starts[t + 1]:
            busy[nb] = t
            nb += 1
    nch = min(ntask, nb)
    for c in prange(nch):                               # tiles interleaved: the busy ones spread
        buf = np.zeros((TILE, TILE), np.float32)        # left all zero by every tile's flushes
        for k in range(c, nb, nch):
            _draw_tile(canvas, segs, starts, index, ntx, busy[k], cx0, cy0, cx1, cy1, colors,
                       widths, dashes, dash_starts, dash_offsets, caps, buf)


@njit(cache=True, nogil=True)
def _draw_tiles_serial(canvas, segs, starts, index, ntx, nty, cx0, cy0, cx1, cy1,
                       colors, widths, dashes, dash_starts, dash_offsets, caps):
    """_draw_tiles on this thread (the tiles are apart: the same pixels)."""
    buf = np.zeros((TILE, TILE), np.float32)
    for t in range(ntx * nty):
        if starts[t] == starts[t + 1]:
            continue
        _draw_tile(canvas, segs, starts, index, ntx, t, cx0, cy0, cx1, cy1, colors, widths,
                   dashes, dash_starts, dash_offsets, caps, buf)


SERIAL_TILE_PX = 4096               # pixels (about) under which tiles are drawn on one thread
SERIAL_TILE_PAIRS = 256             # segments in tiles past which they never are


@njit(cache=True, nogil=True)
def _tile_work(segs, index, reach):
    """About how many pixels drawing the binned segments looks at: each
    segment's box in a tile, a slanted one's rows narrowed to its stroke,
    and some for the segment itself."""
    work = 0.0
    r = 2.0 * reach + 1.0
    for q in range(index.size):
        s = index[q]
        w = min(np.float64(TILE), abs(np.float64(segs[s, 2]) - segs[s, 0]) + r)
        h = min(np.float64(TILE), abs(np.float64(segs[s, 3]) - segs[s, 1]) + r)
        work += min(w * h, max(w, h) * (r + 2.0)) + 48.0
    return work


@njit(cache=True, nogil=True)
def _draw_tile(canvas, segs, starts, index, ntx, t, cx0, cy0, cx1, cy1, colors, widths,
               dashes, dash_starts, dash_offsets, caps, buf):
    """One tile: each run of segments of a style takes, per pixel, its
    largest coverage in buf, then blends it once. Only pixels a segment can
    reach are looked at: per row of its box, those within its reach of the
    part of it that row's band of y crosses (a slanted line: a few pixels,
    not the tile's width), and a run's flush only over the box its
    segments covered -- every other pixel's coverage is 0."""
    a, b = starts[t], starts[t + 1]
    if a != b:
        tx, ty = t % ntx, t // ntx
        j0 = cx0 + tx * TILE
        i0 = cy0 + ty * TILE
        j1 = min(j0 + TILE, cx1)
        i1 = min(i0 + TILE, cy1)
        ri0, ri1, rj0, rj1 = i1, i0, j1, j0             # the run's box so far (empty)
        cur = np.int64(segs[index[a], 4])
        hw, da, ndash, period, doff, cap, lim = _style(cur, widths, dashes, dash_starts,
                                                       dash_offsets, caps)
        top = _top(hw, cap)
        for q in range(a, b + 1):
            st = -1 if q == b else np.int64(segs[index[q], 4])
            if st != cur:                               # flush the run of one style
                r, g, bl, al = (np.float32(colors[cur, 0]), np.float32(colors[cur, 1]),
                                np.float32(colors[cur, 2]), np.float32(colors[cur, 3]))
                for i in range(ri0, ri1):
                    for j in range(rj0, rj1):
                        m = buf[i - i0, j - j0]
                        if m > _F0:
                            blend_px(canvas, i, j, r, g, bl, al, m)
                            buf[i - i0, j - j0] = _F0
                if q == b:
                    break
                ri0, ri1, rj0, rj1 = i1, i0, j1, j0
                cur = st
                hw, da, ndash, period, doff, cap, lim = _style(cur, widths, dashes, dash_starts,
                                                               dash_offsets, caps)
                top = _top(hw, cap)
            s = index[q]
            x0, y0, x1, y1 = segs[s, 0], segs[s, 1], segs[s, 2], segs[s, 3]
            dx, dy = x1 - x0, y1 - y0
            L2 = dx * dx + dy * dy
            inv2 = _F1 / L2 if L2 > _F0 else _F0
            ln = np.sqrt(L2)
            lim2 = lim * lim
            reach = lim + _FH
            ends = np.int64(segs[s, 6])
            # the pixels whose centres (j + 0.5) can lie within lim of it: from
            # floor(lo - lim - 0.5) to ceil(hi + lim - 0.5), each end a pixel to
            # spare (one past is a pixel or more beyond lim: coverage 0 however
            # float32 rounds)
            far = lim - _FH
            bx0 = max(j0, np.int64(np.floor(min(x0, x1) - reach)))
            bx1 = min(j1, np.int64(np.ceil(max(x0, x1) + far)) + 1)
            by0 = max(i0, np.int64(np.floor(min(y0, y1) - reach)))
            by1 = min(i1, np.int64(np.ceil(max(y0, y1) + far)) + 1)
            if bx1 <= bx0 or by1 <= by0:
                continue
            ri0, ri1, rj0, rj1 = min(ri0, by0), max(ri1, by1), min(rj0, bx0), max(rj1, bx1)
            # rows narrowed to the segment's reach (in float64, a pixel to spare)
            narrow = bx1 - bx0 > 8 and abs(dy) > abs(dx) * 0.0625
            slope = np.float64(dx) / np.float64(dy) if narrow else 0.0
            ylo, yhi = np.float64(min(y0, y1)), np.float64(max(y0, y1))
            rr = np.float64(reach) + 1.0
            for i in range(by0, by1):
                py = np.float32(i) + _FH
                ja, jb = bx0, bx1
                if narrow:
                    yc = np.float64(i) + 0.5
                    ya = min(max(yc - rr, ylo), yhi)
                    yb = min(max(yc + rr, ylo), yhi)
                    xa = np.float64(x0) + (ya - np.float64(y0)) * slope
                    xb = np.float64(x0) + (yb - np.float64(y0)) * slope
                    ja = max(bx0, np.int64(np.floor(min(xa, xb) - rr)))
                    jb = min(bx1, np.int64(np.ceil(max(xa, xb) + rr)) + 1)
                for j in range(ja, jb):
                    if buf[i - i0, j - j0] >= top:      # as covered as this style gets
                        continue
                    c = _coverage(np.float32(j) + _FH, py, x0, y0, dx, dy, inv2, ln, hw, lim2,
                                  segs[s, 5], dashes, da, ndash, period, doff, cap, ends)
                    if c > buf[i - i0, j - j0]:
                        buf[i - i0, j - j0] = c


class Styles:
    """Line styles of one draw: colours (k, 4) uint8, widths (k,) px, and
    per style a dash pattern (on, off, ... px; empty = solid), offset and
    cap (LEGACY, BUTT, PROJECTING, ROUND or a name)."""

    def __init__(self, colors, widths, dashes=None, offsets=None, caps=LEGACY):
        self.colors = np.ascontiguousarray(np.atleast_2d(colors), dtype=np.uint8)
        k = self.colors.shape[0]
        if k == 1 and offsets is None and isinstance(caps, (int, np.integer)) and \
                np.ndim(widths) == 0 and (dashes is None or (
                    len(dashes) == 1 and (dashes[0] is None or len(dashes[0]) == 0))):
            self._solid(float(widths), int(caps))       # one solid style (most draws)
            return
        self.widths = np.array(np.broadcast_to(np.asarray(widths, np.float64), (k,)))
        dashes = [()] * k if dashes is None else list(dashes)
        if len(dashes) == 1 and k > 1:
            dashes = dashes * k
        lens = [0 if d is None else len(d) for d in dashes]
        self.dash_starts = np.zeros(k + 1, np.int64)
        np.cumsum(lens, out=self.dash_starts[1:])
        flat = [float(v) for d in dashes if d is not None for v in d]
        self.dashes = np.asarray(flat, np.float32)
        self.dash_offsets = np.zeros(k, np.float32) if offsets is None else \
            np.array(np.broadcast_to(np.asarray(offsets, np.float32), (k,)))
        caps = [CAPS.get(c, c) for c in caps] if isinstance(caps, (list, tuple)) else \
            CAPS.get(caps, caps)
        self.caps = np.array(np.broadcast_to(np.asarray(caps, np.int64), (k,)))
        # the reach of a stroke (binning): a projecting corner is sqrt(2) hw out
        self.reach_widths = self.widths if not self.caps.any() else \
            np.where(self.caps == LEGACY, self.widths, self.widths * 1.4143 + 0.5)

    def _solid(self, width, cap):
        """__init__'s arrays for one solid style, made directly."""
        self.widths = np.array([width])
        self.dash_starts = np.zeros(2, np.int64)
        self.dashes = np.empty(0, np.float32)
        self.dash_offsets = np.zeros(1, np.float32)
        self.caps = np.array([cap], np.int64)
        self.reach_widths = self.widths if cap == LEGACY else self.widths * 1.4143 + 0.5


def snap_value(width_px):
    """matplotlib's PathSnapper offset: vertices go to pixel centres for a
    stroke of odd whole width, to pixel edges otherwise (a fill: width 0)."""
    w = np.floor(np.asarray(width_px, np.float64) + 0.5).astype(np.int64)
    return np.where(w % 2 != 0, 0.5, 0.0)


def rectilinear(xy, offsets):
    """Per part: has it at most 1024 vertices, every segment horizontal or
    vertical (matplotlib's should_snap for 'auto'; NaN is a move)?"""
    xy = np.asarray(xy, np.float64)
    offsets = np.asarray(offsets, np.int64)
    ok = np.diff(offsets) <= 1024
    if xy.shape[0] > 1 and ok.any():
        d = np.abs(np.diff(xy, axis=0))
        with np.errstate(invalid='ignore'):
            slanted = (d[:, 0] >= 1e-4) & (d[:, 1] >= 1e-4)
        k = np.nonzero(slanted)[0]
        part = np.searchsorted(offsets, k, 'right') - 1
        inside = k + 1 < offsets[part + 1]                  # not across two parts
        ok[part[inside]] = False
    return ok


@njit(cache=True, nogil=True)
def _snap_parts(xy, offsets, sv, check):
    """snap() in one pass: each part (all, or with check the rectilinear()
    ones) snapped by its sv, in a copy of xy made at the first one --
    (the copy, True), or (xy, False) when none is."""
    snapped = False
    out = xy
    for p in range(offsets.size - 1):
        a, b = offsets[p], offsets[p + 1]
        ok = True
        if check:
            ok = b - a <= 1024
            k = a
            while ok and k < b - 1:
                if abs(xy[k + 1, 0] - xy[k, 0]) >= 1e-4 and abs(xy[k + 1, 1] - xy[k, 1]) >= 1e-4:
                    ok = False                          # a slanted segment (NaN: a move)
                k += 1
        if ok:
            if not snapped:
                out = xy.copy()
                snapped = True
            for k in range(a, b):
                out[k, 0] = np.floor(xy[k, 0] + 0.5) + sv[p]
                out[k, 1] = np.floor(xy[k, 1] + 0.5) + sv[p]
    return out, snapped


def snap(xy, offsets, width_px, mode=None):
    """matplotlib Agg's pixel snapping of paths (xy px, parts by offsets):
    mode None ('auto') snaps the rectilinear parts, True every part, False
    none; a snapped vertex goes to floor(v + 0.5) + snap_value(width)
    (width_px one per call or one per part). Returns xy, a copy when
    changed."""
    if mode is False:
        return xy
    xy = np.asarray(xy, np.float64)
    offsets = np.asarray(offsets, np.int64)
    lens = np.diff(offsets)
    if lens.size and offsets[0] == 0 and offsets[-1] == xy.shape[0] and xy.ndim == 2:
        sv = np.array(np.broadcast_to(snap_value(width_px), (lens.size,)))
        return _snap_parts(_rw(xy, np.float64), _rw(offsets, np.int64), sv, mode is None)[0]
    ok = rectilinear(xy, offsets) if mode is None else np.ones(lens.size, bool)
    if not ok.any():
        return xy
    sv = np.broadcast_to(snap_value(width_px), (lens.size,))
    out = xy.copy()
    sel = np.repeat(ok, lens)
    out[sel] = np.floor(xy[sel] + 0.5) + np.repeat(sv, lens)[sel][:, None]
    return out


def draw_segments(canvas, segs, styles, clip, memo=None):
    """Segments (from segments()) of the given Styles onto the canvas,
    clipped to clip = (x0, y0, x1, y1) whole pixels. memo: a dict kept
    with segs, where their binning (by widths and clip) is remembered."""
    if not segs.shape[0]:
        return
    clip = _clip_to(canvas, clip)
    if clip is None:
        return
    if memo is None:
        starts, index, ntx, nty = bin_segments(segs, styles.reach_widths, clip)
    else:
        key = (styles.reach_widths.tobytes(), clip)
        if key not in memo:
            memo[key] = bin_segments(segs, styles.reach_widths, clip)
        starts, index, ntx, nty = memo[key]
    if not index.size:
        return
    # little to draw (ticks, a spine): one thread draws it sooner than a parallel region starts
    args = (canvas, segs, starts, index, ntx, nty, clip[0], clip[1], clip[2], clip[3],
            styles.colors, styles.widths, styles.dashes, styles.dash_starts, styles.dash_offsets,
            styles.caps)
    if index.size <= SERIAL_TILE_PAIRS and _tile_work(
            segs, index, 0.5 * float(styles.reach_widths.max()) + 1.0) <= SERIAL_TILE_PX:
        _draw_tiles_serial(*args)
    else:
        _draw_tiles(*args, TILE_TASKS * _jit.team())


def draw_lines(canvas, xy, offsets, color, width, clip=None, dashes=None, dash_offset=0.0,
               cap=LEGACY):
    """Polylines onto the canvas in one style: color uint8[4], width px,
    dashes (on, off, ... px) or None."""
    draw_styled(canvas, xy, offsets, Styles(color, width, None if dashes is None else [dashes],
                                            dash_offset, cap), None, clip)


def draw_styled(canvas, xy, offsets, styles, part_style, clip=None):
    """Polylines onto the canvas, each in its style of a Styles."""
    draw_segments(canvas, segments(xy, offsets, part_style), styles, clip)


def _clip_to(canvas, clip):
    H, W = canvas.shape[:2]
    if clip is None:
        return (0, 0, W, H)
    x0, y0, x1, y1 = (int(np.floor(clip[0])), int(np.floor(clip[1])),
                      int(np.ceil(clip[2])), int(np.ceil(clip[3])))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


# --------------------------------------------------------------- polygons

@njit(cache=True, nogil=True)
def _edges(xy, ring_offsets, y0, y1):
    """Polygon edges (x0, y0, x1, y1, dir) of rings, horizontal ones and
    those outside rows [y0, y1) dropped; closes each ring."""
    n = 0
    for r in range(ring_offsets.size - 1):
        n += ring_offsets[r + 1] - ring_offsets[r]
    out = np.empty((n, 5), np.float64)
    k = 0
    for r in range(ring_offsets.size - 1):
        a, b = ring_offsets[r], ring_offsets[r + 1]
        m = b - a
        if m < 2:
            continue
        for q in range(m):
            p0 = a + q
            p1 = a + (q + 1) % m
            xa, ya, xb, yb = xy[p0, 0], xy[p0, 1], xy[p1, 0], xy[p1, 1]
            if not (np.isfinite(xa) and np.isfinite(ya) and np.isfinite(xb) and np.isfinite(yb)):
                continue
            if ya == yb or max(ya, yb) < y0 or min(ya, yb) >= y1:
                continue
            if ya < yb:
                out[k, 0], out[k, 1], out[k, 2], out[k, 3], out[k, 4] = xa, ya, xb, yb, 1.0
            else:
                out[k, 0], out[k, 1], out[k, 2], out[k, 3], out[k, 4] = xb, yb, xa, ya, -1.0
            k += 1
    return out[:k]


@njit(cache=True, nogil=True)
def _row_lists(edges, y0, y1):
    nrow = y1 - y0
    counts = np.zeros(nrow + 1, np.int64)
    for e in range(edges.shape[0]):
        r0 = max(y0, np.int64(np.floor(edges[e, 1]))) - y0
        r1 = min(y1 - 1, np.int64(np.floor(edges[e, 3]))) - y0
        for r in range(r0, r1 + 1):
            counts[r + 1] += 1
    for r in range(nrow):
        counts[r + 1] += counts[r]
    pos = counts[:-1].copy()
    idx = np.empty(counts[-1], np.int32)
    for e in range(edges.shape[0]):
        r0 = max(y0, np.int64(np.floor(edges[e, 1]))) - y0
        r1 = min(y1 - 1, np.int64(np.floor(edges[e, 3]))) - y0
        for r in range(r0, r1 + 1):
            idx[pos[r]] = e
            pos[r] += 1
    return counts, idx


@njit(cache=True, nogil=True)
def _stride(n):
    """A step through 0..n-1 (i * step % n) visiting every one, about n / 8
    apart: consecutive iterations far apart."""
    s = max(1, n // 8 + 1)
    while True:
        a, b = s, n
        while b:
            a, b = b, a % b
        if a == 1:
            return s
        s += 1


@njit(cache=True, nogil=True, parallel=True)
def _fill_rows(canvas, edges, starts, idx, x0, y0, x1, y1, r, g, b, a, nonzero):
    nrow = y1 - y0
    W = x1 - x0
    step = _stride(nrow)
    for nth in prange(nrow):
        rr = nth * step % nrow                  # a thread's rows spread down the polygons
        e0, e1 = starts[rr], starts[rr + 1]
        if e0 == e1:
            continue
        i = y0 + rr
        cov = np.zeros(W + 1, np.float32)
        xs = np.empty(e1 - e0, np.float64)
        ds = np.empty(e1 - e0, np.float64)
        for sr in range(SUB):
            ys = i + (sr + 0.5) / SUB
            n = 0
            for q in range(e0, e1):
                e = idx[q]
                if edges[e, 1] <= ys < edges[e, 3]:
                    t = (ys - edges[e, 1]) / (edges[e, 3] - edges[e, 1])
                    x = edges[e, 0] + t * (edges[e, 2] - edges[e, 0])
                    # insertion into the sorted crossings
                    k = n
                    while k > 0 and xs[k - 1] > x:
                        xs[k] = xs[k - 1]
                        ds[k] = ds[k - 1]
                        k -= 1
                    xs[k] = x
                    ds[k] = edges[e, 4]
                    n += 1
            wind = 0.0
            for k in range(n - 1):
                if nonzero:
                    wind += ds[k]
                    inside = wind != 0.0
                else:
                    inside = (k % 2) == 0
                if not inside:
                    continue
                xa = min(max(xs[k], x0), x1) - x0
                xb = min(max(xs[k + 1], x0), x1) - x0
                if xb <= xa:
                    continue
                ja, jb = np.int64(np.floor(xa)), np.int64(np.floor(xb))
                w = np.float32(1.0 / SUB)
                if ja == jb:
                    cov[ja] += w * np.float32(xb - xa)
                else:
                    cov[ja] += w * np.float32(ja + 1 - xa)
                    for j in range(ja + 1, jb):
                        cov[j] += w
                    if jb < W:
                        cov[jb] += w * np.float32(xb - jb)
        for j in range(W):
            m = min(cov[j], _F1)
            if m > _F0:
                blend_px(canvas, i, x0 + j, r, g, b, a, m)


def fill_polygons(canvas, xy, ring_offsets, color, clip=None, rule='evenodd'):
    """Rings (xy (n, 2) screen px; ring_offsets (r+1)) filled with one
    colour uint8[4], clipped to clip (x0, y0, x1, y1)."""
    clip = _clip_to(canvas, clip)
    if clip is None or len(ring_offsets) < 2:
        return
    x0, y0, x1, y1 = clip
    xy = np.ascontiguousarray(xy, dtype=np.float64)
    edges = _edges(_rw(xy, np.float64), _rw(ring_offsets, np.int64), float(y0), float(y1))
    if not edges.shape[0]:
        return
    starts, idx = _row_lists(edges, y0, y1)
    c = np.asarray(color, np.float32)
    _fill_rows(canvas, edges, starts, idx, x0, y0, x1, y1, c[0], c[1], c[2], c[3],
               rule == 'nonzero')


# ---------------------------------------------------------------- markers

STAMP_Q = 8                         # a marker's coverage at 1/8-px sub-positions


STAMP_BANDS = 8                     # row bands stamped in parallel, a thread


@njit(cache=True, nogil=True, parallel=True)
def _stamp(canvas, ix, iy, q, face_masks, face_cols, edge_masks, edge_cols, one_face,
           one_edge, has_face, has_edge, cx0, cy0, cx1, cy1, nbands):
    """Markers in order, each its face coverage then its edge's (masks
    [q[k]] with top-left at (ix[k], iy[k])), as Agg composites them. Row
    bands run in parallel; each takes every marker in order, so every
    pixel sees them in order."""
    mh, mw = face_masks.shape[1], face_masks.shape[2]
    nb = min(nbands, cy1 - cy0)
    step = _stride(nb)
    for i in prange(nb):
        band = i * step % nb                    # a thread's run of bands spread down the
        b0 = cy0 + (cy1 - cy0) * band // nb     # canvas: a dense middle shared out
        b1 = cy0 + (cy1 - cy0) * (band + 1) // nb
        _stamp_band(canvas, ix, iy, q, face_masks, face_cols, edge_masks, edge_cols, one_face,
                    one_edge, has_face, has_edge, cx0, b0, cx1, b1, mh, mw)


@njit(cache=True, nogil=True)
def _stamp_band(canvas, ix, iy, q, face_masks, face_cols, edge_masks, edge_cols, one_face,
                one_edge, has_face, has_edge, cx0, cy0, cx1, cy1, mh, mw):
    for k in range(ix.size):
        if iy[k] >= cy1 or iy[k] + mh <= cy0 or ix[k] >= cx1 or ix[k] + mw <= cx0:
            continue
        for layer in range(2):
            if layer == 0:
                if not has_face:
                    continue
                m = face_masks[q[k]]
                c = 0 if one_face else k
                r, g, b, a = (np.float32(face_cols[c, 0]), np.float32(face_cols[c, 1]),
                              np.float32(face_cols[c, 2]), np.float32(face_cols[c, 3]))
            else:
                if not has_edge:
                    continue
                m = edge_masks[q[k]]
                c = 0 if one_edge else k
                r, g, b, a = (np.float32(edge_cols[c, 0]), np.float32(edge_cols[c, 1]),
                              np.float32(edge_cols[c, 2]), np.float32(edge_cols[c, 3]))
            if a == _F0:
                continue
            for i in range(mh):
                ci = iy[k] + i
                if ci < cy0 or ci >= cy1:
                    continue
                for j in range(mw):
                    cj = ix[k] + j
                    if cj < cx0 or cj >= cx1:
                        continue
                    v = m[i, j]
                    if v > _F0:
                        blend_px(canvas, ci, cj, r, g, b, a, v)


def _coverage_masks(fill, stroke, width, cap, subs, ox, oy, mh, mw):
    """Each sub-position (fx, fy) in subs: the coverage of the marker's
    fill rings and its stroke drawn at (fx - ox, fy - oy) in an (mh, mw)
    box -- float32 (len(subs), mh, mw) each (None when absent). All the
    boxes are drawn in one call, stacked (mh keeps them 2 px apart)."""
    white = np.array([255, 255, 255, 255], np.uint8)
    k = len(subs)
    shift = np.array([[fx - ox, fy - oy + n * mh] for n, (fx, fy) in enumerate(subs)])
    out = []
    for geom, stroked in ((fill, False), (stroke, True)):
        if geom is None or (stroked and width <= 0):
            out.append(None)
            continue
        tile = np.zeros((k * mh, mw, 4), np.uint8)
        xy = (geom[0][None, :, :] + shift[:, None, :]).reshape(-1, 2)
        m, p = geom[0].shape[0], geom[1].size - 1
        offs = np.empty(k * p + 1, np.int64)
        offs[:-1] = (geom[1][:-1][None, :] + m * np.arange(k)[:, None]).ravel()
        offs[-1] = k * m
        if stroked:
            draw_lines(tile, xy, offs, white, width, cap=cap)
        else:
            fill_polygons(tile, xy, offs, white, None, 'nonzero')
        out.append((tile[..., 3] * np.float32(1.0 / 255.0)).reshape(k, mh, mw))
    return out


def stamp_markers(canvas, pts, fill, stroke, width, face, edge, clip, cap=BUTT):
    """Markers at pts (n, 2 px; NaN-free) composited one after another --
    face then edge -- as matplotlib's draw_markers / path collections do:
    fill (xy, ring offsets) and stroke (xy, offsets) relative to a marker's
    centre, width its edge px; face, edge uint8 (4,) or (n, 4) (None:
    none). The coverage is made once per 1/8-px sub-position in use."""
    clip = _clip_to(canvas, clip)
    n = pts.shape[0]
    if clip is None or not n:
        return
    parts = [g[0] for g in (fill, stroke) if g is not None]
    if not parts:
        return
    allv = np.concatenate(parts)
    reach = width / 2.0 + 2.0
    ox = np.floor(allv[:, 0].min() - reach)
    oy = np.floor(allv[:, 1].min() - reach)
    mw = int(np.ceil(allv[:, 0].max() + reach - ox)) + 1
    mh = int(np.ceil(allv[:, 1].max() + reach - oy)) + 1
    base = np.floor(pts)
    sub = np.rint((pts - base) * STAMP_Q).astype(np.int64)
    carry = sub == STAMP_Q
    base = base + carry
    sub[carry] = 0
    code = sub[:, 0] * STAMP_Q + sub[:, 1]
    used, q = np.unique(code, return_inverse=True)
    subs = [((c // STAMP_Q) / STAMP_Q, (c % STAMP_Q) / STAMP_Q) for c in used.tolist()]
    fm, em = _coverage_masks(fill if face is not None else None,
                             stroke if edge is not None else None, width, cap, subs, ox, oy,
                             mh, mw)
    dummy = np.zeros((1, mh, mw), np.float32)
    nocol = np.zeros((1, 4), np.uint8)
    fc = nocol if face is None else np.ascontiguousarray(np.atleast_2d(face), np.uint8)
    ec = nocol if edge is None else np.ascontiguousarray(np.atleast_2d(edge), np.uint8)
    _stamp(canvas, (base[:, 0] + ox).astype(np.int64), (base[:, 1] + oy).astype(np.int64),
           q.astype(np.int64), dummy if fm is None else fm, fc, dummy if em is None else em,
           ec, fc.shape[0] == 1, ec.shape[0] == 1, fm is not None, em is not None,
           clip[0], clip[1], clip[2], clip[3], STAMP_BANDS * _jit.team())


# ---------------------------------------------------------- simplification

@njit(cache=True, nogil=True)
def _simplify(xy, move, thr2, out, kind):
    """Agg's PathSimplifier (matplotlib's path_converters.h), vertex for
    vertex: runs of points within sqrt(thr2) px of the line they extend
    merge into its farthest points forward and back. xy finite, move[k]
    True where a sub-path starts; out / kind (1 move, 2 line) get the
    vertices. Returns how many."""
    n = 0
    moveto, after_moveto, clipped = True, False, False
    lastx = lasty = 0.0
    odx = ody = on2 = 0.0
    fmax = bmax = 0.0
    lastF = lastB = False
    nx = ny = nbx = nby = 0.0
    cvx = cvy = 0.0
    for k in range(xy.shape[0]):
        x, y = xy[k, 0], xy[k, 1]
        if moveto or move[k]:
            if on2 != 0.0 and not after_moveto:
                n, odx, ody, on2, fmax, lastF, cvx, cvy, lastx, lasty, nx, ny, bmax, lastB, \
                    clipped = _simplify_push(out, kind, n, x, y, bmax, lastF, lastB, nx, ny,
                                             nbx, nby, clipped, lastx, lasty)
            after_moveto = True
            lastx, lasty = x, y
            moveto = False
            on2 = 0.0
            bmax = 0.0
            clipped = True
            continue
        after_moveto = False
        if on2 == 0.0:                                  # the vector the line is built on
            if clipped:
                out[n, 0], out[n, 1], kind[n] = lastx, lasty, 1
                n += 1
                clipped = False
            odx, ody = x - lastx, y - lasty
            on2 = odx * odx + ody * ody
            fmax, bmax = on2, 0.0
            lastF, lastB = True, False
            cvx, cvy = lastx, lasty
            nx = lastx = x
            ny = lasty = y
            continue
        tdx, tdy = x - cvx, y - cvy
        tdot = odx * tdx + ody * tdy
        pdx, pdy = tdot * odx / on2, tdot * ody / on2
        qx, qy = tdx - pdx, tdy - pdy
        if qx * qx + qy * qy < thr2:                    # close enough: merge it
            pn2 = pdx * pdx + pdy * pdy
            lastF = lastB = False
            if tdot > 0.0:
                if pn2 > fmax:
                    lastF, fmax, nx, ny = True, pn2, x, y
            elif pn2 > bmax:
                lastB, bmax, nbx, nby = True, pn2, x, y
            lastx, lasty = x, y
            continue
        n, odx, ody, on2, fmax, lastF, cvx, cvy, lastx, lasty, nx, ny, bmax, lastB, clipped = \
            _simplify_push(out, kind, n, x, y, bmax, lastF, lastB, nx, ny, nbx, nby, clipped,
                           lastx, lasty)
    c = 1 if (moveto or after_moveto) else 2
    if on2 != 0.0:
        out[n, 0], out[n, 1], kind[n] = nx, ny, c
        n += 1
        if bmax > 0.0:
            out[n, 0], out[n, 1], kind[n] = nbx, nby, c
            n += 1
        moveto = False
        c = 1 if after_moveto else 2
    out[n, 0], out[n, 1], kind[n] = lastx, lasty, c
    return n + 1


@njit(cache=True, nogil=True)
def _simplify_push(out, kind, n, x, y, bmax, lastF, lastB, nx, ny, nbx, nby, clipped, lastx,
                   lasty):
    """PathSimplifier::_push: the line built so far out, a new one from it."""
    if bmax > 0.0:
        if lastF:
            out[n, 0], out[n, 1], kind[n] = nbx, nby, 2
            out[n + 1, 0], out[n + 1, 1], kind[n + 1] = nx, ny, 2
        else:
            out[n, 0], out[n, 1], kind[n] = nx, ny, 2
            out[n + 1, 0], out[n + 1, 1], kind[n + 1] = nbx, nby, 2
        n += 2
    else:
        out[n, 0], out[n, 1], kind[n] = nx, ny, 2
        n += 1
    if clipped:
        out[n, 0], out[n, 1], kind[n] = lastx, lasty, 1
        n += 1
    elif not lastF and not lastB:
        out[n, 0], out[n, 1], kind[n] = lastx, lasty, 2
        n += 1
    odx, ody = x - lastx, y - lasty
    on2 = odx * odx + ody * ody
    cvx, cvy = out[n - 1, 0], out[n - 1, 1]
    return (n, odx, ody, on2, on2, True, cvx, cvy, x, y, x, y, 0.0, False, False)


@njit(cache=True, nogil=True)
def _finite_moves(xy, offsets, v, mv):
    """The finite points of polylines (parts by offsets, NaN breaking
    them) into v, mv[k] whether v[k] moves (the first, a part's start or
    the first after a NaN since the last kept): how many."""
    n = xy.shape[0]
    start = np.zeros(n, np.bool_)
    for j in range(offsets.size - 1):
        if offsets[j + 1] > offsets[j]:
            start[offsets[j]] = True
    m = 0
    pending = True
    for i in range(n):
        x, y = xy[i, 0], xy[i, 1]
        pending |= start[i]
        if np.isfinite(x) and np.isfinite(y):
            v[m, 0], v[m, 1] = x, y
            mv[m] = pending
            pending = False
            m += 1
        else:
            pending = True                              # a NaN: the next point moves
    return m


def simplify(xy, offsets, threshold=1.0 / 9.0):
    """Polylines (xy px, NaN breaking them, parts by offsets) as Agg draws
    them after matplotlib's path simplification: (xy, offsets)."""
    xy = np.ascontiguousarray(xy, np.float64)
    v = np.empty_like(xy)
    mv = np.empty(xy.shape[0], np.bool_)
    m = _finite_moves(xy, np.ascontiguousarray(offsets, np.int64), v, mv)
    if not m:
        return np.empty((0, 2)), np.zeros(1, np.int64)
    out = np.empty((2 * m + 8, 2))
    kind = np.empty(2 * m + 8, np.int8)
    n = _simplify(v[:m], mv[:m], threshold * threshold, out, kind)
    starts = np.nonzero(kind[:n] == 1)[0]
    return out[:n], np.r_[starts, n].astype(np.int64)
