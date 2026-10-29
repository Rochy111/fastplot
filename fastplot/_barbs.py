"""Wind barbs as matplotlib builds them, made in numba.

    xy, offsets, empty = barbs(px, py, dx, dy, mag, length, ...)

Each barb is matplotlib's polygon (quiver.Barbs._make_barbs): a staff of
`length` px from the station, flags (triangles), full and half barbs
(degenerate triangles that only the outline draws) at its far end, rotated
so the staff points the way the wind comes from. Like matplotlib the
polygon is then both filled and stroked -- _figure draws the rings with
_draw.fill_polygons and _draw.draw_lines. A calm barb (nothing to draw)
is matplotlib's empty circle, unrotated (doubled back over itself unless
fill_empty, as matplotlib makes it); `empty` marks those rings.

Magnitudes are rounded and split as matplotlib's _find_tails does:
rounded to the nearest half increment (rounding=True), then flags of
`flag`, full barbs of `full`, a half barb if what is left is >= `half`.

Screen coordinates are y down; (dx, dy) is the wind's direction on the
screen (only its angle is used), mag its speed.
"""
import numpy as np

from ._jit import njit

NCIRCLE = 21                                    # matplotlib's CirclePolygon: 20 sides, closed


@njit(cache=True, nogil=True)
def _tails(mag, rounding, half, full, flag):
    if rounding:
        mag = half * np.floor(mag / half + 0.5)
    nflags = np.int64(np.floor(mag / flag))
    mag = mag % flag
    nbarbs = np.int64(np.floor(mag / full))
    mag = mag % full
    hb = mag >= half
    empty = not (hb or nflags > 0 or nbarbs > 0)
    return nflags, nbarbs, hb, empty


@njit(cache=True, nogil=True)
def _count(mag, rounding, half, full, flag, fill_empty):
    n = 0
    for k in range(mag.size):
        if not np.isfinite(mag[k]):
            continue
        nf, nb, hb, empty = _tails(mag[k], rounding, half, full, flag)
        if empty:
            n += NCIRCLE if fill_empty else 2 * NCIRCLE
        else:
            n += 1 + 3 * nf + 3 * nb + (4 if hb else 0)
    return n


@njit(cache=True, nogil=True)
def _make(px, py, dx, dy, mag, flip, length, spacing, height, width, emptyrad, endy,
          rounding, half, full, flag, fill_empty, xy, offsets, empty):
    v = 0
    p = 0
    for k in range(mag.size):
        if not np.isfinite(mag[k]):
            continue
        nf, nb, hb, emp = _tails(mag[k], rounding, half, full, flag)
        offsets[p] = v
        empty[p] = emp
        # matplotlib's angle: rotate the barb, built along +y (y up), by
        # atan2(v, u) + pi/2; our dy points down, so v = -dy
        th = np.arctan2(-dy[k], dx[k]) + np.pi / 2
        c, s = np.cos(th), np.sin(th)
        x0, y0 = px[k], py[k]
        if emp:                                         # matplotlib's CirclePolygon, unrotated:
            for q in range(NCIRCLE):                    # from the top, anticlockwise, closed
                a = np.pi / 2 + 2 * np.pi * q / (NCIRCLE - 1)
                xy[v, 0] = x0 + emptyrad * np.cos(a)
                xy[v, 1] = y0 - emptyrad * np.sin(a)
                v += 1
            if not fill_empty:                          # and back over itself: never filled
                for q in range(NCIRCLE):
                    xy[v, 0] = xy[v - 1 - 2 * q, 0]
                    xy[v, 1] = xy[v - 1 - 2 * q, 1]
                    v += 1
            p += 1
            continue
        bh = -height if flip[k] else height
        # vertices in barb space (bx, by), y up; written rotated, y flipped
        start = v
        tmp = np.empty((1 + 3 * nf + 3 * nb + 4, 2), np.float64)
        n = 0
        tmp[n, 0], tmp[n, 1] = 0.0, endy
        n += 1
        off = length
        for q in range(nf):
            if off != length:
                off += spacing / 2.0
            tmp[n, 0], tmp[n, 1] = 0.0, endy + off
            tmp[n + 1, 0], tmp[n + 1, 1] = bh, endy - width / 2 + off
            tmp[n + 2, 0], tmp[n + 2, 1] = 0.0, endy - width + off
            n += 3
            off -= width + spacing
        for q in range(nb):
            tmp[n, 0], tmp[n, 1] = 0.0, endy + off
            tmp[n + 1, 0], tmp[n + 1, 1] = bh, endy + off + width / 2
            tmp[n + 2, 0], tmp[n + 2, 1] = 0.0, endy + off
            n += 3
            off -= spacing
        if hb:
            if off == length:
                tmp[n, 0], tmp[n, 1] = 0.0, endy + off
                n += 1
                off -= 1.5 * spacing
            tmp[n, 0], tmp[n, 1] = 0.0, endy + off
            tmp[n + 1, 0], tmp[n + 1, 1] = bh / 2, endy + off + width / 4
            tmp[n + 2, 0], tmp[n + 2, 1] = 0.0, endy + off
            n += 3
        for q in range(n):
            bx, by = tmp[q, 0], tmp[q, 1]
            xy[start + q, 0] = x0 + (c * bx - s * by)
            xy[start + q, 1] = y0 - (s * bx + c * by)
        v = start + n
        p += 1
    offsets[p] = v
    return v, p


def barbs(px, py, dx, dy, mag, length=7.0, pivot='tip', spacing=0.125, height=0.4,
          width=0.25, emptybarb=0.15, rounding=True, half=5, full=10, flag=50, flip=False,
          fill_empty=False):
    """Barb polygons at screen points (px, py): (xy (n, 2), offsets (m+1),
    empty (m,) bool). length in px; sizes as fractions of it (matplotlib's
    `sizes`); pivot 'tip' or 'middle' (or a number, px)."""
    px, py, dx, dy, mag = (np.ascontiguousarray(np.ravel(a), dtype=np.float64)
                           for a in (px, py, dx, dy, mag))
    flip = np.ascontiguousarray(np.broadcast_to(np.asarray(flip, bool), mag.shape))
    endy = {'tip': 0.0, 'middle': -length / 2.0}.get(pivot) if isinstance(pivot, str) else \
        float(pivot)
    n = _count(mag, bool(rounding), float(half), float(full), float(flag), bool(fill_empty))
    m = int(np.isfinite(mag).sum())
    xy = np.empty((n + 8, 2), np.float64)               # (+ room a half barb may skip)
    offsets = np.empty(m + 1, np.int64)
    empty = np.empty(m, np.bool_)
    v, p = _make(px, py, dx, dy, mag, flip, float(length), length * spacing, length * height,
                 length * width, length * emptybarb, endy, bool(rounding), float(half),
                 float(full), float(flag), bool(fill_empty), xy, offsets, empty)
    return xy[:v], offsets[:p + 1], empty[:p]
