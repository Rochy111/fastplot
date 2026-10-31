"""Anti-aliased lines, filled polygons and markers, drawn straight into an
RGBA canvas with Agg's rasterizer (_agg), the CPU side of every figure.

    segs = segments(xy, offsets, part_style)           # polylines to draw
    draw_segments(canvas, segs, Styles(...), clip)
    draw_lines(canvas, xy, offsets, color, width)      # the two above
    fill_polygons(canvas, xy, ring_offsets, rgba, clip)

Lines. A polyline (screen pixels, NaN breaking it) is stroked as Agg strokes
it, with matplotlib's caps and joins, and dashed as Agg dashes it. A style
gives colour, width, dashes, cap, join and mitre limit; parts of a compound
path sum their coverage, separate paths blend one after another, as Agg's
draw_path does.
"""
import math

import numpy as np

from . import _agg, _arena, _jit, _path
from ._agg import BUTT, JOIN_BEVEL, JOIN_MITER, JOIN_ROUND, JOINS, PROJECTING, ROUND
from ._features import _ranges
from ._jit import njit, prange
from ._rc import active

_F0 = np.float32(0.0)
_F1 = np.float32(1.0)
_FH = np.float32(0.5)
_F255 = np.float32(255.0)

CAPS = {'butt': BUTT, 'projecting': PROJECTING, 'round': ROUND}
# Styles' arrays for one solid style, shared: only kernels read them
_SOLID = (np.zeros(2, np.int64), np.empty(0, np.float64), np.zeros(1, np.float64))
_SOLID_CAPS = {c: np.array([c], np.int64) for c in (BUTT, PROJECTING, ROUND)}
_SOLID_JOINS = {j: np.array([j], np.int64) for j in (JOIN_MITER, JOIN_ROUND, JOIN_BEVEL)}


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


# ---------------------------------------------------------------- polylines

def _rw(a, dtype):
    """a as a C-contiguous, writable array of dtype (a copy only when it
    is not one). numba types a read-only array apart, compiling every
    kernel it reaches again (seconds each, on a new machine): the kernels
    are given writable ones throughout."""
    a = np.ascontiguousarray(a, dtype=dtype)
    return a if a.flags.writeable else a.copy()


@njit(cache=True, nogil=True)
def _breaks(xy, offsets, out, off, part, write):
    """Polylines cut at non-finite points into parts of 2 points or more,
    written (write) to out, off and part (the source of each) or counted:
    (points, parts)."""
    p, k = 0, 0
    for q in range(offsets.size - 1):
        a = offsets[q]
        b = offsets[q + 1]
        i = a
        while i < b:
            while i < b and not (np.isfinite(xy[i, 0]) and np.isfinite(xy[i, 1])):
                i += 1
            s = i
            while i < b and np.isfinite(xy[i, 0]) and np.isfinite(xy[i, 1]):
                i += 1
            if i - s >= 2:
                if write:
                    off[k] = p
                    part[k] = q
                    out[p:p + i - s, :] = xy[s:i, :]
                p += i - s
                k += 1
    return p, k


class Polylines:
    """What segments() makes: finite parts (xy, offsets), and per part its
    style, its path and whether it closes on itself."""

    __slots__ = ('xy', 'offsets', 'part_style', 'part_path', 'closed', 'filled')

    def __init__(self, xy, offsets, part_style, part_path, closed, filled=False):
        self.xy, self.offsets = xy, offsets
        self.part_style, self.part_path, self.closed = part_style, part_path, closed
        self.filled = filled

    @property
    def shape(self):
        return self.xy.shape


def segments(xy, offsets, part_style=None, paths=None, closed=None, filled=False):
    """Polylines (xy (n, 2) screen px, offsets (m+1), NaN breaking them) to
    draw; part_style (m,) gives each its style (0). paths (m,): the path
    each is part of, rising (each its own, or True for one path of them
    all). closed (m,): those that close on themselves (default: those
    whose last point is their first). filled: the edge of a filled path,
    which matplotlib does not cut at the canvas."""
    out = _polylines(xy, offsets, part_style, paths, closed)
    out.filled = filled
    return out


def _polylines(xy, offsets, part_style, paths, closed):
    xy = _rw(xy, np.float64)
    offsets = _rw(offsets, np.int64)
    m = max(offsets.size - 1, 0)
    st = np.zeros(m, np.int64) if part_style is None else _rw(part_style, np.int64)
    if paths is None:
        pth = np.arange(m, dtype=np.int64)
    elif paths is True:
        pth = np.zeros(m, np.int64)
    else:
        pth = _rw(paths, np.int64)
    if closed is False:
        closed = np.zeros(m, np.bool_)
    if not m or not xy.shape[0]:
        return Polylines(np.empty((0, 2)), np.zeros(1, np.int64), st[:0], pth[:0],
                         np.zeros(0, np.bool_))
    if not np.isfinite(xy.sum()):                       # (a NaN or inf: the sum shows it)
        most = xy.shape[0]                              # (a part is 2 points or more)
        out, off, part = np.empty((most, 2)), np.empty(most // 2 + 2, np.int64), \
            np.empty(most // 2 + 1, np.int64)
        n, k = _breaks(xy, offsets, out, off, part, True)
        off[k] = n
        part = part[:k]
        xy, offsets, st, pth = out[:n], off[:k + 1], st[part], pth[part]
        closed = None if closed is None else closed[part] & False
        m = k
        if not m:
            return Polylines(np.empty((0, 2)), np.zeros(1, np.int64), st, pth,
                             np.zeros(0, np.bool_))
    if closed is None:
        s, e = offsets[:-1], offsets[1:] - 1
        closed = (e - s >= 3) & (xy[s, 0] == xy[e, 0]) & (xy[s, 1] == xy[e, 1])
    else:
        closed = _rw(closed, np.bool_)
    return Polylines(xy, offsets, st, pth, closed)


class Styles:
    """Line styles of one draw: colours (k, 4) uint8, widths (k,) px, and
    per style a dash pattern (on, off, ... px; empty = solid), offset, cap
    (BUTT, PROJECTING, ROUND or a name), join (JOIN_ROUND, JOIN_MITER,
    JOIN_BEVEL or a name) and mitre limit (default the width, as matplotlib
    sets it)."""

    def __init__(self, colors, widths, dashes=None, offsets=None, caps=BUTT, joins=JOIN_ROUND,
                 limits=None):
        self.colors = np.ascontiguousarray(np.atleast_2d(colors), dtype=np.uint8)
        k = self.colors.shape[0]
        if k == 1 and offsets is None and limits is None and np.ndim(widths) == 0 and \
                (dashes is None or len(dashes) == 1 and not dashes[0]) and \
                isinstance(caps, (int, np.integer)) and isinstance(joins, (int, np.integer)) and \
                caps in _SOLID_CAPS and joins in _SOLID_JOINS:
            self.widths = self.limits = np.array([float(widths)])       # most draws: one solid style
            self.dash_starts, self.dashes, self.dash_offsets = _SOLID
            self.caps, self.joins = _SOLID_CAPS[caps], _SOLID_JOINS[joins]
            return
        self.widths = np.array(np.broadcast_to(np.asarray(widths, np.float64), (k,)))
        dashes = [()] * k if dashes is None else list(dashes)
        if len(dashes) == 1 and k > 1:
            dashes = dashes * k
        lens = [0 if d is None else len(d) for d in dashes]
        self.dash_starts = np.zeros(k + 1, np.int64)
        np.cumsum(lens, out=self.dash_starts[1:])
        flat = [float(v) for d in dashes if d is not None for v in d]
        self.dashes = np.asarray(flat, np.float64)
        self.dash_offsets = np.zeros(k, np.float64) if offsets is None else \
            np.array(np.broadcast_to(np.asarray(offsets, np.float64), (k,)))
        caps = [CAPS.get(c, c) for c in caps] if isinstance(caps, (list, tuple)) else \
            CAPS.get(caps, caps)
        self.caps = np.array(np.broadcast_to(np.asarray(caps, np.int64), (k,)))
        joins = [JOINS.get(c, c) for c in joins] if isinstance(joins, (list, tuple)) else \
            JOINS.get(joins, joins)
        self.joins = np.array(np.broadcast_to(np.asarray(joins, np.int64), (k,)))
        self.limits = self.widths if limits is None else \
            np.array(np.broadcast_to(np.asarray(limits, np.float64), (k,)))


def snap_value(width_px):
    """matplotlib's PathSnapper offset: vertices go to pixel centres for a
    stroke of odd whole width, to pixel edges otherwise (a fill: width 0)."""
    if isinstance(width_px, float) and 0.0 <= width_px < 1e15:
        return 0.5 if math.floor(width_px + 0.5) % 2 else 0.0   # (most calls: one width)
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
    lens = offsets[1:] - offsets[:-1]
    if lens.size and offsets[0] == 0 and offsets[-1] == xy.shape[0] and xy.ndim == 2:
        sv = snap_value(width_px)
        sv = np.full(lens.size, sv) if isinstance(sv, float) else \
            np.array(np.broadcast_to(sv, (lens.size,)))
        return _snap_parts(_rw(xy, np.float64), _rw(offsets, np.int64), sv, mode is None)[0]
    ok = rectilinear(xy, offsets) if mode is None else np.ones(lens.size, bool)
    if not ok.any():
        return xy
    sv = np.broadcast_to(snap_value(width_px), (lens.size,))
    out = xy.copy()
    sel = np.repeat(ok, lens)
    out[sel] = np.floor(xy[sel] + 0.5) + np.repeat(sv, lens)[sel][:, None]
    return out


SMALL_LINE = 1024                   # points a draw takes in one kernel, on one thread


def draw_segments(canvas, segs, styles, clip, memo=None):
    """Polylines (from segments()) of the given Styles onto the canvas,
    clipped to clip = (x0, y0, x1, y1) whole pixels. memo: a dict kept
    with segs, where their outlines (by style) are remembered."""
    if not segs.shape[0]:
        return
    clip = _clip_to(canvas, clip)
    if clip is None:
        return
    sx, so, cl, st, pth = segs.xy, segs.offsets, segs.closed, segs.part_style, segs.part_path
    if memo is None and sx.shape[0] <= SMALL_LINE and not styles.dashes.size and _agg._strokes_small(
            canvas, sx, so, cl, st, pth, styles.widths, styles.caps, styles.joins, styles.limits,
            styles.colors, np.array(clip, np.float64), segs.filled,
            _agg._cells(clip[3] - clip[1], clip[2] - clip[0])):
        return
    key = None
    if memo is not None:
        key = (styles.widths.tobytes(), styles.caps.tobytes(), styles.joins.tobytes(),
               styles.dashes.tobytes(), styles.dash_offsets.tobytes(), styles.limits.tobytes())
    outline = None if key is None else memo.get(key)
    if outline is None:
        H, W = canvas.shape[:2]
        if not segs.filled and (sx.min() < -1.0 or sx[:, 0].max() > W + 1.0 or
                                sx[:, 1].max() > H + 1.0):
            sx, so, cl, part = _agg.clip_paths(sx, so, cl, W, H)
            st, pth = st[part], pth[part]
        if styles.dashes.size:
            sx, so, part = _agg.dash(sx, so, cl, st, styles.dashes, styles.dash_starts,
                                     styles.dash_offsets)
            cl, st, pth = np.zeros(so.size - 1, np.bool_), st[part], pth[part]
        pts, rings, ring_part = _agg.stroke(sx, so, cl, st, styles.widths, styles.caps,
                                            styles.joins, styles.limits)
        ring_style, ring_path = st[ring_part], pth[ring_part]
        outline = (pts, rings, ring_style, ring_path)
        if key is not None:
            memo[key] = outline
    pts, rings, ring_style, ring_path = outline
    if rings.size < 2:
        return
    # a path's ids count up from 0; its colour is the style of its first ring
    new = np.concatenate(([True], ring_path[1:] != ring_path[:-1]))
    colors = styles.colors[ring_style[np.flatnonzero(new)]]
    run = False if key is None else memo.get((key, clip, 'run'), False)
    if run is False:
        run = _agg.prepare(pts, rings, np.cumsum(new) - 1, clip)
        if key is not None:
            memo[key, clip, 'run'] = run
    if run is not None:
        _agg.rasterize(canvas, run, colors)


def draw_lines(canvas, xy, offsets, color, width, clip=None, dashes=None, dash_offset=0.0,
               cap=BUTT, join=JOIN_ROUND):
    """Polylines onto the canvas in one style: color uint8[4], width px,
    dashes (on, off, ... px) or None."""
    draw_styled(canvas, xy, offsets, Styles(color, width, None if dashes is None else [dashes],
                                            dash_offset, cap, join), None, clip)


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


def fill_polygons(canvas, xy, ring_offsets, color, clip=None, rule='evenodd', ring_path=None,
                  memo=None):
    """Rings (xy (n, 2) screen px; ring_offsets (r+1)) filled with one
    colour uint8[4], clipped to clip (x0, y0, x1, y1). ring_path (r,): the
    path each ring is part of, rising, the paths blended one after another
    (default: all one path). memo: a dict kept with the rings, where their
    edges are remembered."""
    clip = _clip_to(canvas, clip)
    if clip is None or len(ring_offsets) < 2:
        return
    ro = _rw(ring_offsets, np.int64)
    color = np.ascontiguousarray(color, dtype=np.uint8)
    if ring_path is None:
        rp = np.zeros(ro.size - 1, np.int64)
        cols = color.reshape(1, 4)
    else:
        rp = _rw(ring_path, np.int64)
        cols = np.broadcast_to(color, (int(rp.max()) + 1, 4)).copy()
    run = False if memo is None else memo.get(clip, False)
    if run is False:
        run = _agg.prepare(_rw(xy, np.float64), ro, rp, clip)
        if memo is not None:
            memo[clip] = run
    if run is not None:
        _agg.rasterize(canvas, run, cols, rule != 'nonzero')


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


def _coverage_masks(fills, strokes, width, cap, join, shape, subs, ox, oy, mh, mw):
    """Each box k: the coverage of fills[shape[k]] and of strokes[shape[k]]
    drawn at subs[k] - (ox, oy) in an (mh, mw) box -- float32 (k, mh, mw)
    each (None when absent). All the boxes are drawn in one call, across
    the rows of a square atlas (mh, mw keep them 2 px apart), a box's parts
    one path."""
    white = np.array([255, 255, 255, 255], np.uint8)
    k = shape.size
    across = int(np.ceil(np.sqrt(k)))
    down = (k + across - 1) // across
    place = np.arange(k)
    shift = subs - (ox, oy) + np.stack((place % across * mw, place // across * mh), axis=1)
    out = []
    for geoms, stroked in ((fills, False), (strokes, True)):
        if any(g is None for g in geoms) or (stroked and width <= 0):
            out.append(None)
            continue
        xy, lens, paths = [], [], []
        for s in np.unique(shape).tolist():
            box = np.flatnonzero(shape == s)
            v, off = geoms[s]
            xy.append((v[None, :, :] + shift[box, None, :]).reshape(-1, 2))
            lens.append(np.tile(np.diff(off), box.size))
            paths.append(np.repeat(box, off.size - 1))
        xy = np.concatenate(xy)
        offs = np.r_[0, np.cumsum(np.concatenate(lens))]
        paths = np.concatenate(paths)
        tile = np.zeros((down * mh, across * mw, 4), np.uint8)
        if stroked:
            draw_segments(tile, segments(xy, offs, None, paths, filled=True),
                          Styles(white, width, caps=cap, joins=join), None)
        else:
            fill_polygons(tile, xy, offs, white, None, 'nonzero', paths)
        masks = (tile[..., 3] * np.float32(1.0 / 255.0)).reshape(down, mh, across, mw)
        out.append(masks.transpose(0, 2, 1, 3).reshape(down * across, mh, mw)[:k])
    return out


MASK_MARKERS = 1                    # markers from which each is worked out on its own (_agg)


def _flat_markers(path, pts, sizes):
    """A marker's path at size 1, one at each place and size, flattened
    where it lands, as Agg flattens a collection's: (xy, offsets, closed,
    the marker of each part)."""
    v = path.vertices
    n, m = pts.shape[0], v.shape[0]
    placed = (v[None, :, :] * sizes[:, None, None] + pts[:, None, :]).reshape(-1, 2)
    xy, off, closed, parts = _path.flatten_many(placed, np.tile(path.full_codes(), n),
                                                np.arange(n + 1) * m)
    return xy, off, closed, np.repeat(np.arange(n), np.diff(parts))


def exact_markers(canvas, pts, sizes, fill, stroke, width, face, edge, clip, cap=BUTT,
                  join=JOIN_ROUND):
    """Markers as Agg draws a path collection: each its own face path, then
    edge path, exactly at its place and size, snapped there if rectilinear
    (many of them to 1/8 of a pixel, unless rcParams['fastplot.exact']: see
    _agg._stamped). fill, stroke: Paths of a marker of size 1 (None: none);
    face, edge uint8 (4,) or (n, 4); sizes (n,)."""
    clip = _clip_to(canvas, clip)
    n = pts.shape[0]
    if clip is None or not n:
        return
    wf = width if edge is not None else 0.0
    if cap == BUTT and join == JOIN_ROUND and n >= MASK_MARKERS:
        none = np.zeros((1, 4), np.uint8)
        pair = [None if p is None else (np.ascontiguousarray(p.vertices, np.float64),
                                        np.ascontiguousarray(p.full_codes(), np.uint8))
                for p in (fill, stroke)]
        if stroke is fill:
            pair[1] = pair[0]
        fc = none if face is None else np.ascontiguousarray(np.atleast_2d(face), np.uint8)
        ec = none if edge is None else np.ascontiguousarray(np.atleast_2d(edge), np.uint8)
        if _agg.draw_markers(canvas, _rw(pts, np.float64), _rw(sizes, np.float64), pair[0], pair[1],
                             float(width), float(snap_value(float(wf))), fc, ec, clip,
                             not active().get('fastplot.exact', False)):
            return
    pxy, plen, ppath = [], [], []
    flat = None
    if fill is not None:
        flat = _flat_markers(fill, pts, sizes)
        xy, off, _, mk = flat
        pxy.append(snap(xy, off, wf))
        plen.append(np.diff(off))
        ppath.append(2 * mk)
    if stroke is not None:
        xy, off, closed, mk = flat if stroke is fill else _flat_markers(stroke, pts, sizes)
        seg = segments(snap(xy, off, wf), off, None, mk, closed, filled=True)
        o, r, part = _agg.stroke(seg.xy, seg.offsets, seg.closed, seg.part_style,
                                 np.array([width]), np.array([cap], np.int64),
                                 np.array([join], np.int64), np.array([width]))
        pxy.append(o)
        plen.append(np.diff(r))
        ppath.append(2 * seg.part_path[part] + 1)
    cols = np.empty((2 * n, 4), np.uint8)
    for j, c in ((0, face), (1, edge)):
        if c is not None:
            cols[j::2] = np.asarray(c, np.uint8)
    _fill_in_order(canvas, pxy, plen, ppath, cols, clip)


@njit(cache=True, nogil=True, parallel=True)
def _merge_rings(axy, alen, apath, bxy, blen, bpath):
    """Two lists of rings, each with its paths rising, as one: (points,
    ring offsets, each ring's path)."""
    na, nb = alen.size, blen.size
    n = na + nb
    rings = np.empty(n + 1, np.int64)
    path = np.empty(n, np.int64)
    src = np.empty(n, np.int64)                         # where a ring's points are
    i = j = sa = sb = 0
    rings[0] = 0
    for r in range(n):
        if j >= nb or (i < na and apath[i] <= bpath[j]):
            path[r], rings[r + 1], src[r] = apath[i], rings[r] + alen[i], sa
            sa += alen[i]
            i += 1
        else:
            path[r], rings[r + 1], src[r] = bpath[j], rings[r] + blen[j], ~sb
            sb += blen[j]
            j += 1
    xy = np.empty((rings[n], 2), np.float64)
    for r in prange(n):
        s = src[r]
        for k in range(rings[r + 1] - rings[r]):
            at = rings[r] + k
            if s >= 0:
                xy[at, 0], xy[at, 1] = axy[s + k, 0], axy[s + k, 1]
            else:
                xy[at, 0], xy[at, 1] = bxy[~s + k, 0], bxy[~s + k, 1]
    return xy, rings, path


def _fill_in_order(canvas, pxy, plen, ppath, cols, clip):
    """Rings of one or two lists (points, ring lengths, each ring's path,
    rising) as one draw: path by path, cols[p] the colour of path p."""
    if not pxy:
        return
    xy, lens, path = [[np.ascontiguousarray(a, t) for a in part]
                      for part, t in ((pxy, np.float64), (plen, np.int64), (ppath, np.int64))]
    if len(xy) == 2:
        xy, rings, path = _merge_rings(xy[0], lens[0], path[0], xy[1], lens[1], path[1])
    else:
        xy, rings, path = xy[0], np.concatenate(([0], np.cumsum(lens[0]))), path[0]
    _agg.fill(canvas, xy, rings, path, cols, clip)


def draw_paths(canvas, xy, offsets, faces, part_style, styles, clip):
    """Parts in order, as Agg's draw_path one after another: each filled
    (faces (m, 4) uint8, alpha 0: not filled), then stroked in its style of
    a Styles (part_style (m,), -1: not stroked)."""
    clip = _clip_to(canvas, clip)
    if clip is None:
        return
    offsets = np.asarray(offsets, np.int64)
    m = offsets.size - 1
    faces = np.asarray(faces, np.uint8)
    part_style = np.asarray(part_style, np.int64)
    pxy, plen, ppath = [], [], []
    f = np.flatnonzero(faces[:, 3] > 0)
    if f.size:
        pxy.append(xy[_ranges(offsets[f], offsets[f + 1])])
        plen.append(offsets[f + 1] - offsets[f])
        ppath.append(2 * f)
    s = np.flatnonzero(part_style >= 0)
    cols = np.empty((2 * m, 4), np.uint8)
    cols[0::2] = faces
    if s.size:
        lens = offsets[s + 1] - offsets[s]
        seg = segments(xy[_ranges(offsets[s], offsets[s + 1])],
                       np.concatenate(([0], np.cumsum(lens))), part_style[s], s, filled=True)
        o, r, part = _agg.stroke(seg.xy, seg.offsets, seg.closed, seg.part_style, styles.widths,
                                 styles.caps, styles.joins, styles.limits)
        pxy.append(o)
        plen.append(np.diff(r))
        ppath.append(2 * seg.part_path[part] + 1)
        cols[2 * seg.part_path + 1] = styles.colors[seg.part_style]
    _fill_in_order(canvas, pxy, plen, ppath, cols, clip)


def stamp_markers(canvas, pts, fill, stroke, width, face, edge, clip, cap=BUTT, which=None,
                  join=JOIN_ROUND):
    """Markers at pts (n, 2 px; NaN-free) composited one after another --
    face then edge -- as matplotlib's draw_markers / path collections do:
    fill (xy, ring offsets) and stroke (xy, offsets) relative to a marker's
    centre, width its edge px; face, edge uint8 (4,) or (n, 4) (None:
    none). The coverage is made once per 1/8-px sub-position in use.
    which (n,): markers of several shapes (scatter's sizes) -- fill and
    stroke are then lists, which[k] the one of marker k."""
    clip = _clip_to(canvas, clip)
    n = pts.shape[0]
    if clip is None or not n:
        return
    join = JOIN_ROUND if join is None else join
    if which is None:
        fills, strokes, which = [fill], [stroke], np.zeros(n, np.int64)
    else:
        fills, strokes = list(fill), list(stroke)
        which = np.asarray(which, np.int64)
    parts = [g[0] for gs in (fills, strokes) for g in gs if g is not None]
    if not parts:
        return
    allv = np.concatenate(parts)
    reach = width / 2.0 * (max(width, 1.5) if JOIN_MITER == join or cap == PROJECTING else 1.0) + 2.0
    ox = np.floor(allv[:, 0].min() - reach)
    oy = np.floor(allv[:, 1].min() - reach)
    mw = int(np.ceil(allv[:, 0].max() + reach - ox)) + 1
    mh = int(np.ceil(allv[:, 1].max() + reach - oy)) + 1
    base = np.floor(pts)
    sub = np.rint((pts - base) * STAMP_Q).astype(np.int64)
    carry = sub == STAMP_Q
    base = base + carry
    sub[carry] = 0
    QQ = STAMP_Q * STAMP_Q
    code = which * QQ + sub[:, 0] * STAMP_Q + sub[:, 1]
    used, q = np.unique(code, return_inverse=True)
    subs = np.stack((used % QQ // STAMP_Q, used % STAMP_Q), axis=1) / STAMP_Q
    fm, em = _coverage_masks(fills if face is not None else [None],
                             strokes if edge is not None else [None], width, cap, join,
                             used // QQ, subs, ox, oy, mh, mw)
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
    v = _arena.take('simplify.v', xy.shape, np.float64)
    mv = _arena.take('simplify.mv', xy.shape[0], np.bool_)
    m = _finite_moves(xy, np.ascontiguousarray(offsets, np.int64), v, mv)
    if not m:
        return np.empty((0, 2)), np.zeros(1, np.int64)
    out = np.empty((2 * m + 8, 2))
    kind = _arena.take('simplify.kind', 2 * m + 8, np.int8)
    n = _simplify(v[:m], mv[:m], threshold * threshold, out, kind)
    starts = np.nonzero(kind[:n] == 1)[0]
    return out[:n], np.concatenate((starts, [n])).astype(np.int64)
