"""Hatching, as matplotlib's Agg backend draws it.

    _hatch.fill(canvas, xy, ring_offsets, '//', color, dpi, clip)

matplotlib.hatch makes the pattern a path in the unit square (density 6:
'/' six lines a square, '//' twelve; '-', '|', '+', '\\\\', 'x', 'o', 'O',
'.', '*'); Agg fills it (nonzero: solid dots and stars, rings for 'o')
and strokes it (square caps, hatch.linewidth wide) once into a tile `dpi`
pixels square, y up, then fills the shape with that tile
repeated from the canvas' top-left corner, its coverage times the
shape's. The same here: the tile drawn by fastplot's line kernel (cached
by pattern, size and width), the shape's coverage by its polygon fill,
the two multiplied and blended by one numba pass.
"""
from functools import lru_cache

import numpy as np

from . import _draw, _path
from ._jit import njit, prange

DENSITY = 6
_F0 = np.float32(0.0)


def _lines(n, vertical):
    steps = np.linspace(0.0, 1.0, n, False)
    steps += (1.0 / n) / 2.0 if n else 0.0
    v = np.empty((2 * n, 2))
    if vertical:
        v[0::2] = np.c_[steps, np.zeros(n)]
        v[1::2] = np.c_[steps, np.ones(n)]
    else:
        v[0::2] = np.c_[np.zeros(n), steps]
        v[1::2] = np.c_[np.ones(n), steps]
    return v


def _diagonals(n, north_east):
    steps = np.linspace(-0.5, 0.5, n + 1)
    v = np.empty((2 * (n + 1), 2))
    if north_east:
        v[0::2] = np.c_[steps, -steps]
        v[1::2] = np.c_[1.0 + steps, 1.0 - steps]
    else:
        v[0::2] = np.c_[steps, 1.0 + steps]
        v[1::2] = np.c_[1.0 + steps, steps]
    return v


def _shapes(rows, size, shape, codes, filled):
    """matplotlib's hatch Shapes: rows of shapes, alternate rows offset."""
    offset = 1.0 / rows
    sv = shape * offset * size
    sc = codes
    if not filled:                                      # forward, then back at 0.9
        sv = np.concatenate([sv, sv[::-1] * 0.9])
        sc = np.concatenate([sc, sc])
    vs, cs = [], []
    for row in range(rows + 1):
        cols = np.linspace(0, 1, rows + 1) if row % 2 == 0 else \
            np.linspace(offset / 2, 1 - offset / 2, rows)
        for col in cols:
            vs.append(sv + [col, row * offset])
            cs.append(sc)
    return np.concatenate(vs), np.concatenate(cs)


def path(pattern, density=DENSITY):
    """matplotlib.hatch.get_path: (vertices (n, 2), codes) in the unit square."""
    count = pattern.count
    parts = []
    move_line = lambda v: (v, np.tile([_path.MOVETO, _path.LINETO], len(v) // 2))  # noqa
    n = int((count('-') + count('+')) * density)
    if n:
        parts.append(move_line(_lines(n, False)))
    n = int((count('|') + count('+')) * density)
    if n:
        parts.append(move_line(_lines(n, True)))
    n = int((count('/') + count('x') + count('X')) * density)
    if n:
        parts.append(move_line(_diagonals(n, True)))
    n = int((count('\\') + count('x') + count('X')) * density)
    if n:
        parts.append(move_line(_diagonals(n, False)))
    circle = _path.unit_circle()
    for ch, size, filled in (('o', 0.2, False), ('O', 0.35, False), ('.', 0.1, True)):
        rows = count(ch) * density
        if rows:
            parts.append(_shapes(rows, size, circle.vertices, circle.full_codes(), filled))
    rows = count('*') * density
    if rows:
        star = _path.unit_regular_star(5)
        codes = np.full(len(star.vertices), _path.LINETO, np.uint8)
        codes[0], codes[-1] = _path.MOVETO, _path.CLOSEPOLY
        parts.append(_shapes(rows, 1.0 / 3.0, star.vertices, codes, True))
    if not parts:
        return np.empty((0, 2)), np.empty(0, np.uint8)
    return (np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]).astype(np.uint8))


@lru_cache(maxsize=64)
def tile(pattern, size, width_px):
    """The pattern stroked into a size x size tile (Agg's hatch buffer):
    coverage, float32, row 0 at the top."""
    v, codes = path(pattern)
    out = np.zeros((size, size), np.float32)
    if not len(v):
        return out
    px = np.column_stack([v[:, 0] * size, (1.0 - v[:, 1]) * size])
    xy, offs, _ = _path.flatten(px, codes)
    buf = np.zeros((size, size, 4), np.uint8)
    white = np.array([255, 255, 255, 255], np.uint8)
    _draw.fill_polygons(buf, xy, offs, white, None, 'nonzero')
    _draw.draw_lines(buf, xy, offs, white, width_px, cap=_draw.PROJECTING)
    out[:] = buf[..., 3] * np.float32(1.0 / 255.0)
    out.setflags(write=False)
    return out


@njit(cache=True, nogil=True, parallel=True)
def _blend(canvas, cov, x0, y0, tl, r, g, b, a):
    """The tile, repeated from the canvas origin, over the canvas where cov
    (its box at x0, y0) covers: coverage tile * cov."""
    hs = tl.shape[0]
    h, w = cov.shape
    for i in prange(h):
        ti = (y0 + i) % hs
        for j in range(w):
            c = cov[i, j]
            if c > _F0:
                m = tl[ti, (x0 + j) % hs] * c
                if m > _F0:
                    _draw.blend_px(canvas, y0 + i, x0 + j, r, g, b, a, m)


def fill(canvas, xy, ring_offsets, pattern, color, dpi, clip=None, width_px=None,
         rule='nonzero'):
    """Hatch the rings (px) with pattern in color (uint8[4]): Agg's hatch,
    a tile of int(dpi) px, its lines hatch.linewidth wide."""
    if not pattern or color is None or not color[3] or not len(xy):
        return
    from ._style import rc
    size = int(dpi)
    if width_px is None:
        width_px = rc('hatch.linewidth') * dpi / 72.0
    tl = tile(pattern, size, float(width_px))
    clip = _draw._clip_to(canvas, clip)
    if clip is None:
        return
    ok = np.isfinite(xy).all(axis=1)
    if not ok.any():
        return
    x0 = max(clip[0], int(np.floor(xy[ok, 0].min())))
    y0 = max(clip[1], int(np.floor(xy[ok, 1].min())))
    x1 = min(clip[2], int(np.ceil(xy[ok, 0].max())) + 1)
    y1 = min(clip[3], int(np.ceil(xy[ok, 1].max())) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    buf = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
    _draw.fill_polygons(buf, xy - (x0, y0), ring_offsets,
                        np.array([255, 255, 255, 255], np.uint8), None, rule)
    cov = np.ascontiguousarray(buf[..., 3]) * np.float32(1.0 / 255.0)
    c = np.asarray(color, np.float32)
    _blend(canvas, cov, x0, y0, tl, c[0], c[1], c[2], c[3])


def fill_mask(canvas, mask, x0, y0, pattern, color, dpi, width_px=None):
    """Hatch where a coverage mask (float32, its box at x0, y0) covers --
    a contourf band."""
    if not pattern or color is None or not color[3]:
        return
    from ._style import rc
    if width_px is None:
        width_px = rc('hatch.linewidth') * dpi / 72.0
    tl = tile(pattern, int(dpi), float(width_px))
    c = np.asarray(color, np.float32)
    _blend(canvas, np.ascontiguousarray(mask, np.float32), int(x0), int(y0), tl, c[0], c[1],
           c[2], c[3])


def color_for(edgecolor, hatchcolor=None, alpha=None, collection=False):
    """matplotlib's hatch colour: 'edge' (rcParams hatch.color) is the edge
    colour, or patch.edgecolor when the edge is invisible -- with the
    artist's alpha for a collection (as Collection.get_hatchcolor), without
    for a patch."""
    from ._style import rc, rgba
    hc = rc('hatch.color') if hatchcolor is None else hatchcolor
    if isinstance(hc, str) and hc == 'edge':
        if edgecolor is None or edgecolor[3] == 0:
            return rgba(rc('patch.edgecolor'), alpha if collection else None)
        return np.asarray(edgecolor, np.uint8)
    return rgba(hc, alpha)
