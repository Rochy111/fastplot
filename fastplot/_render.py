"""The canvas a figure is drawn into: the same short list of operations
on the CPU (numpy + numba) and the GPU (cupy, _gpu.GpuCanvas), so the
artists in _figure never know which one they are on.

    c = CpuCanvas(H, W, rgba)
    c.fill_rect(x0, y0, x1, y1, rgba)               # whole pixels, blended
    c.field_rect(box, view, src, xaxis, yaxis, z, mode, coloring, alpha)
    c.field_cells(box, view, src, x2d, y2d, z, mode, coloring, alpha, period)
    c.segments(segs, styles, clip)                  # _draw's tiles
    c.polygons(xy, ring_offsets, rgba, clip, rule)
    c.mask(alpha_mask, x0, y0, rgba)                # text
    c.image(rgba, x0, y0)                           # a ready RGBA block
    c.array() -> numpy (H, W, 4) uint8

A field is a value raster over `box` (x0, y0, x1, y1: the canvas pixels
of the map), coloured by a Coloring and blended over what is there.
"""
import threading

import numpy as np

from . import _arena, _color, _draw, _jit, _raster
from ._jit import njit, prange

_F1 = np.float32(1.0)
_FH = np.float32(0.5)
_F255 = np.float32(255.0)


class View:
    """A map's projection and the affine from raster pixels to its x/y:
    x = x0 + (j + 1/2) dx, y = y0 + (i + 1/2) dy for raster pixel (i, j)."""

    def __init__(self, proj, x0, dx, y0, dy):
        self.proj = proj
        self.kind = proj.kind
        self.params = proj.params
        self.x0, self.dx, self.y0, self.dy = float(x0), float(dx), float(y0), float(dy)


class Coloring:
    """How values become colours: a colormap and norm (pcolormesh, imshow),
    or level bands (contourf)."""

    def __init__(self, cmap=None, norm=None, levels=None, band_lut=None, under=None, over=None,
                 rounding='nearest'):
        self.cmap, self.norm = cmap, norm
        self.rounding = rounding                        # 'nearest': as Agg paints a vector
        # artist (pcolormesh, contourf); 'floor': as matplotlib's images
        self.levels = None if levels is None else np.ascontiguousarray(levels, np.float64)
        self.band_lut, self.under, self.over = band_lut, under, over

    @property
    def bands(self):
        return self.levels is not None

    def shade(self, values, out, alpha=None):
        if self.bands:
            _color.shade_bands(values, self.levels, self.band_lut, self.under, self.over, out=out)
        else:
            _color.shade(values, self.cmap, self.norm, out=out, alpha=alpha,
                         rounding=self.rounding)
        return out

    def plan(self, alpha=None):
        """_color's ShadePlan: the kernel and its arguments (CPU and GPU alike)."""
        if self.bands:
            return _color.bands_plan(self.levels, self.band_lut, self.under, self.over)
        return _color.shade_plan(self.norm, self.cmap, alpha, self.rounding)

    def fused(self):
        """(_render's _bands_* arguments of these bands: top, then the
        'shade_bands' plan's levels, m, table (uint32 rows), eq0, levels[0],
        1 / step and uniform, then bands_guess's; whether every colour is
        opaque or transparent)."""
        f = getattr(self, '_fused', None)
        if f is None:
            lev, m, tab, eq0, log, l0, inv_d, uniform = _color._cpu_args(self.plan().args)
            assert not log
            a = tab >> 24
            f = self._fused = ((self.top, lev, m, tab, eq0, l0, inv_d, uniform) +
                               bands_guess(lev), bool(np.all((a == 0) | (a == 255))))
        return f

    @property
    def top(self):
        """The highest value a band paints: levels[-1], or inf when the
        band above it (over) is filled."""
        return np.float32(np.inf if self.over is not None else self.levels[-1])

    @property
    def translucent(self):
        """Do bands show through each other, where they meet?"""
        if not self.bands:
            return False
        return any(c is not None and np.min(np.asarray(c)[..., 3]) < 255
                   for c in (self.band_lut, self.under, self.over))


class IndexColoring(Coloring):
    """An RGBA pcolormesh's colouring: each value is a cell's own number,
    lut ((N + 3, 4) uint8: the N cells' colours, then under, over, bad) its
    colour, alpha already in it."""

    def __init__(self, lut):
        super().__init__()
        self.lut = np.ascontiguousarray(lut, np.uint8)
        self.N = self.lut.shape[0] - 3

    def plan(self, alpha=None):
        return _color.ShadePlan('shade_index', (self.lut, np.int32(self.N)))

    def shade(self, values, out, alpha=None):
        v, v2 = _color._prep_values(values)
        out, o32, back = _color._prep_out(out, v.shape, v2.shape)
        _color._run_cpu(self.plan(), v2, o32)
        if back is not None:
            back[...] = o32.view(np.uint8).reshape(back.shape)
        return out


@njit(cache=True, nogil=True, inline='always')
def _rise(c, n, rise, fall):
    """rise and fall from c to a neighbour n, added in if it has a value
    (off the edge, n is c itself: nothing either way). Selects of bitwise
    conditions, not branches (`and` is one), so the row loop vectorizes."""
    d = n - c
    ok = (n - n) == np.float32(0.0)                     # (not NaN, not infinite)
    rise = d if (ok & (d > rise)) else rise
    nd = -d
    fall = nd if (ok & (nd > fall)) else fall
    return rise, fall


@njit(cache=True, nogil=True, inline='always')
def _touched_px(c, right, left, down, up, top):
    z = np.float32(0.0)
    rx, fx = _rise(c, right, z, z)
    rx, fx = _rise(c, left, rx, fx)
    ry, fy = _rise(c, down, z, z)
    ry, fy = _rise(c, up, ry, fy)
    hi = c + np.float32(0.5) * (rx + ry)
    lo = c - np.float32(0.5) * (fx + fy)
    hi = top if ((hi > top) & (lo <= top)) else hi
    return hi if ((c - c) == np.float32(0.0)) else c   # (c finite)


@njit(cache=True, nogil=True)
def _touched_row(a, b, c, out, top):
    """touched_bands' row b (a above it, c below: b itself at an edge) into
    out. The first and last columns go apart, so the inner loop has no
    clamped indices and vectorizes."""
    W = b.shape[0]
    if W == 0:
        return
    out[0] = _touched_px(b[0], b[min(1, W - 1)], b[0], c[0], a[0], top)
    for j in range(1, W - 1):
        out[j] = _touched_px(b[j], b[j + 1], b[j - 1], c[j], a[j], top)
    if W > 1:
        out[W - 1] = _touched_px(b[W - 1], b[W - 1], b[W - 2], c[W - 1], a[W - 1], top)


@njit(cache=True, nogil=True, parallel=True)
def touched_bands(src, out, top):
    """contourf's polygons as Agg fills them, not antialiased, lowest band
    first: a pixel any polygon touches gets its colour, so the highest
    band touching each pixel shows. Each pixel's value becomes the largest
    in its square -- the centre plus half the largest rise to a neighbour
    across and down (a plane's corner) -- held to `top` when the square
    reaches down into the painted bands."""
    H = src.shape[0]
    for i in prange(H):
        _touched_row(src[max(i - 1, 0)], src[i], src[min(i + 1, H - 1)], out[i], top)


# -- contourf's bands straight onto the canvas, a raster's rows one at a time
# (its values, made there -- three rows kept, a pixel's touched value reads
# the rows either side -- or read; touched_bands' row; each pixel's band
# colour; blended), so that none of the three passes' full-size rasters
# (values, touched values, colours) is written and read again.
#
# The rows are cleaned first, every infinity made NaN: a neighbour that is
# not finite then drops out of _rise by its NaN difference alone, with no
# test of its own (and a pixel not finite has no colour either way).
#
# A pixel's band row (_color._band_row: the levels below its value) comes
# from a guess in float32 along the whole row, k = floor((x - l0) / d) + 1 on
# the levels' mean step d, taken where it cannot be wrong: x further from
# every level than the levels lie from the even steps (E) plus the guess's
# own rounding (_bands_guess); _band_row for the rest (x at a level, x not
# finite, levels far from even).

ROW_TASKS = 4                   # row bands a thread (load balance, not a cache size)


@njit(cache=True, nogil=True, inline='always')
def _clean(v):
    return v if (v - v) == np.float32(0.0) else np.float32(np.nan)


@njit(cache=True, nogil=True)
def _clean_row(src, out):
    for j in range(src.shape[0]):
        out[j] = _clean(src[j])


@njit(cache=True, nogil=True, inline='always')
def _rise_c(c, n, rise, fall):
    """_rise of cleaned values: a NaN neighbour's difference is no rise."""
    d = n - c
    rise = d if d > rise else rise
    nd = -d
    fall = nd if nd > fall else fall
    return rise, fall


@njit(cache=True, nogil=True, inline='always')
def _touched_c(c, right, left, down, up, top):
    """_touched_px of cleaned values (NaN where it is not finite)."""
    z = np.float32(0.0)
    rx, fx = _rise_c(c, right, z, z)
    rx, fx = _rise_c(c, left, rx, fx)
    ry, fy = _rise_c(c, down, z, z)
    ry, fy = _rise_c(c, up, ry, fy)
    hi = c + np.float32(0.5) * (rx + ry)
    lo = c - np.float32(0.5) * (fx + fy)
    hi = top if ((hi > top) & (lo <= top)) else hi
    return hi if ((c - c) == np.float32(0.0)) else c


@njit(cache=True, nogil=True, inline='always')
def _guess(x, L0, D, mu, mf):
    """x's band row by the even steps, or -1 where that may not be it."""
    g = (x - L0) * D + np.float32(1.0)
    fl = np.floor(g)
    fr = g - fl
    finite = (x - x) == np.float32(0.0)
    k = np.float32(0.0) if fl < np.float32(0.0) else fl  # (g < -1/2: under every level, 0;
    k = mf if k > mf else k                              # g > m + 1/2: over them all, m)
    k = k if finite else mf + np.float32(1.0)           # (not finite: row m + 1)
    near = (fr <= mu) | (fr >= np.float32(1.0) - mu)    # (an infinite g's fr is NaN: not near)
    unsure = near & (g >= np.float32(-0.5)) & (g <= mf + np.float32(0.5)) & finite
    return np.int32(-1) if unsure else np.int32(k)


@njit(cache=True, nogil=True)
def _touched_guess(a, b, c, s, e, top, L0, D, mu, mf, vec, row, kk):
    """Cleaned row b's touched values (a above it, c below) in columns s..e
    into row, and their band rows by _guess into kk (-1: none; all of them
    unless vec). Returns whether any is -1 (a min: a count would not
    vectorize)."""
    W = b.shape[0]
    if s == 0:
        row[0] = _touched_c(b[0], b[min(1, W - 1)], b[0], c[0], a[0], top)
    lo, hi = max(s, 1), min(e, W - 1)
    if hi > lo:                                         # (loops from 0 over views: from s
        bb, aa, cc, rr = b[lo - 1:hi + 1], a[lo:hi], c[lo:hi], row[lo:hi]  # they would not
        for q in range(hi - lo):                                           # vectorize)
            rr[q] = _touched_c(bb[q + 1], bb[q + 2], bb[q], cc[q], aa[q], top)
    if e == W and W > 1 and W - 1 >= lo:
        row[W - 1] = _touched_c(b[W - 1], b[W - 1], b[W - 2], c[W - 1], a[W - 1], top)
    if not vec:
        kk[s:e] = -1
        return True
    least = np.int32(0)
    rs, ks = row[s:e], kk[s:e]
    for q in range(e - s):
        k = _guess(rs[q], L0, D, mu, mf)
        ks[q] = k
        least = min(least, k)
    return least < 0


@njit(cache=True, nogil=True)
def _fix_bands(row, kk, j0, j1, levels, m, eq0, l0, inv_d, uniform):
    """kk's band rows _guess left (-1) from _band_row (a pass of its own: the
    call in _put_bands' loop costs it registers)."""
    for j in range(j0, j1):
        if kk[j] < 0:
            kk[j] = _color._band_row(np.float64(row[j]), levels, m, eq0, False, l0, inv_d,
                                     uniform)


@njit(cache=True, nogil=True)
def _put_bands(kk, j0, j1, tab, crow, canvas, ci, x0, alpha, fast):
    """Row's pixels j0..j1 (band rows kk) in their bands' colours over
    canvas row ci (at x0 + j): _k_bands' colour, _blend_image's blend.
    fast: alpha 1 and every colour opaque or transparent."""
    if fast:
        for j in range(j0, j1):
            w = tab[kk[j]]
            if (w >> 24) != 0:
                crow[x0 + j] = w
        return
    for j in range(j0, j1):
        w = tab[kk[j]]
        a = w >> 24
        if a == 0:
            continue
        if a == 255 and alpha == _F1:
            crow[x0 + j] = w
        else:
            _draw.blend_px(canvas, ci, x0 + j, np.float32(w & 255), np.float32((w >> 8) & 255),
                           np.float32((w >> 16) & 255), np.float32(a), alpha)


@njit(cache=True, nogil=True, inline='always')
def _put_bounds(h, w, H, W, y0, x0, t, tasks):
    """Task t's raster rows (r0, r1) and the columns on the canvas (j0, j1)."""
    return h * t // tasks, h * (t + 1) // tasks, max(0, -x0), min(w, W - x0)


@njit(cache=True, nogil=True)
def _row_bands(ring, up, i, down, row, kk, j0, j1, canvas, c32, ci, x0, top, levels, m, tab,
               eq0, l0, inv_d, uniform, L0, D, mu, mf, vec, fast, alpha):
    """Raster row i (its cleaned rows i - 1, i, i + 1 in ring) onto canvas row
    ci -- its pixels from the first with a value to the last: a pixel of NaN
    is NaN touched, band row m + 1, which is transparent."""
    b = ring[i % 3]
    s, e = j0, j1
    while s < e and b[s] != b[s]:
        s += 1
    while e > s and b[e - 1] != b[e - 1]:
        e -= 1
    if s == e:
        return
    if _touched_guess(ring[up % 3], b, ring[down % 3], s, e, top, L0, D, mu, mf, vec, row, kk):
        _fix_bands(row, kk, s, e, levels, m, eq0, l0, inv_d, uniform)
    _put_bands(kk, s, e, tab, c32[ci], canvas, ci, x0, alpha, fast)


@njit(cache=True, nogil=True, parallel=True)
def _bands_raster(values, canvas, c32, y0, x0, top, levels, m, tab, eq0, l0, inv_d, uniform, L0,
                  D, mu, mf, vec, fast, alpha, tasks):
    """A value raster's bands onto the canvas at (y0, x0)."""
    h, w = values.shape
    H, W = canvas.shape[0], canvas.shape[1]
    for t in prange(tasks):
        r0, r1, j0, j1 = _put_bounds(h, w, H, W, y0, x0, t, tasks)
        ring = np.empty((3, w), np.float32)
        row = np.empty(w, np.float32)
        kk = np.empty(w, np.int32)
        done = -1
        for i in range(r0, r1):
            ci = y0 + i
            if ci < 0 or ci >= H or j1 <= j0:
                continue
            up, down = max(i - 1, 0), min(i + 1, h - 1)
            for r in range(max(up, done + 1), down + 1):
                _clean_row(values[r], ring[r % 3])
                done = r
            _row_bands(ring, up, i, down, row, kk, j0, j1, canvas, c32, ci, x0, top, levels, m,
                       tab, eq0, l0, inv_d, uniform, L0, D, mu, mf, vec, fast, alpha)


@njit(cache=True, nogil=True, parallel=True)
def _bands_sep(X, slot, colok, rows, keep, canvas, c32, y0, x0, top, levels, m, tab, eq0, l0,
               inv_d, uniform, L0, D, mu, mf, vec, fast, alpha, tasks):
    """_raster._sep_yrows' rows, made three at a time, their bands onto the
    canvas (the rows kept in keep too, unless it is (0, 0))."""
    h, w = rows.shape[0], colok.shape[0]
    H, W = canvas.shape[0], canvas.shape[1]
    for t in prange(tasks):
        r0, r1, j0, j1 = _put_bounds(h, w, H, W, y0, x0, t, tasks)
        ring = np.empty((3, w), np.float32)
        row = np.empty(w, np.float32)
        kk = np.empty(w, np.int32)
        done = -1
        for i in range(r0, r1):
            ci = y0 + i
            shows = ci >= 0 and ci < H and j1 > j0
            if not (shows or keep.shape[0] != 0):       # (rows kept: all of them)
                continue
            up, down = max(i - 1, 0), min(i + 1, h - 1)
            for r in range(max(up, done + 1), down + 1):
                dst = keep[r] if keep.shape[0] != 0 else ring[r % 3]
                _raster._sep_row(dst, X, slot, colok, rows, r)
                _clean_row(dst, ring[r % 3])
                done = r
            if not shows:
                continue
            _row_bands(ring, up, i, down, row, kk, j0, j1, canvas, c32, ci, x0, top, levels, m,
                       tab, eq0, l0, inv_d, uniform, L0, D, mu, mf, vec, fast, alpha)


@njit(cache=True, nogil=True, parallel=True)
def _bands_tris(table, spans, xs, ys, zs, nx, period, keep, canvas, c32, y0, x0, top, levels, m,
                tab, eq0, l0, inv_d, uniform, L0, D, mu, mf, vec, fast, alpha, tasks):
    """_raster's cells, by the mesh's table of each pixel's triangle, three
    rows at a time, their bands onto the canvas (the rows kept in keep too,
    unless it is (0, 0))."""
    h, w = table.shape
    H, W = canvas.shape[0], canvas.shape[1]
    for t in prange(tasks):
        r0, r1, j0, j1 = _put_bounds(h, w, H, W, y0, x0, t, tasks)
        ring = np.empty((3, w), np.float32)
        row = np.empty(w, np.float32)
        kk = np.empty(w, np.int32)
        done = -1
        for i in range(r0, r1):
            ci = y0 + i
            shows = ci >= 0 and ci < H and j1 > j0
            if not (shows or keep.shape[0] != 0):       # (rows kept: all of them)
                continue
            up, down = max(i - 1, 0), min(i + 1, h - 1)
            for r in range(max(up, done + 1), down + 1):
                dst = keep[r] if keep.shape[0] != 0 else ring[r % 3]
                _raster._tri_row(dst, table[r], np.float64(r), xs, ys, zs, nx, period,
                                 spans[r, 0], spans[r, 1])
                _clean_row(dst, ring[r % 3])
                done = r
            if not shows:
                continue
            _row_bands(ring, up, i, down, row, kk, j0, j1, canvas, c32, ci, x0, top, levels, m,
                       tab, eq0, l0, inv_d, uniform, L0, D, mu, mf, vec, fast, alpha)


_NO_KEEP = np.zeros((0, 0), np.float32)


# -- a field of cells (pcolormesh, shading 'nearest' or 'flat') whose colours
# are the cells' own: each cell's value shaded once, every pixel of it taken
# that colour straight onto the canvas (the pixel's value is its cell's, the
# same colour) -- no value raster, no colour layer.

@njit(cache=True, nogil=True, inline='always')
def _put_word(crow, canvas, ci, cj, w, fast):
    """_blend_image's pixel, alpha 1 (fast: w opaque or transparent)."""
    a = w >> 24
    if a == 0:
        return
    if fast or a == 255:
        crow[cj] = w
    else:
        _draw.blend_px(canvas, ci, cj, np.float32(w & 255), np.float32((w >> 8) & 255),
                       np.float32((w >> 16) & 255), np.float32(a), _F1)


@njit(cache=True, nogil=True, parallel=True)
def _cells_sep(colors, kx, colok, ky, rowok, bad, canvas, c32, y0, x0, fast, tasks):
    """_raster._sample_sep's nearest cells (column j's cell kx[j], row i's
    ky[i]; none where not ok), their colours onto the canvas at (y0, x0)."""
    h, w = ky.shape[0], kx.shape[0]
    H, W = canvas.shape[0], canvas.shape[1]
    for t in prange(tasks):
        r0, r1, j0, j1 = _put_bounds(h, w, H, W, y0, x0, t, tasks)
        for i in range(r0, r1):
            ci = y0 + i
            if ci < 0 or ci >= H or j1 <= j0:
                continue
            crow = c32[ci]
            if rowok[i] == 0:
                for j in range(j0, j1):
                    _put_word(crow, canvas, ci, x0 + j, bad, fast)
                continue
            src = colors[ky[i]]
            for j in range(j0, j1):
                _put_word(crow, canvas, ci, x0 + j, src[kx[j]] if colok[j] != 0 else bad, fast)


@njit(cache=True, nogil=True, parallel=True)
def _pixels_linear(z, kx, colok, ky, rowok, vmin, vmax, clip, lut, N, bad, canvas, c32, y0, x0,
                   fast, tasks):
    """_cells_sep of float32 cells z shaded 'shade_linear', each pixel's
    colour made from its cell's value as it is put -- _k_linear's, on the
    value the sampled raster would hold -- rather than every cell's first:
    for more cells in view than pixels, with no raster nor layer between."""
    h, w = ky.shape[0], kx.shape[0]
    H, W = canvas.shape[0], canvas.shape[1]
    for t in prange(tasks):
        r0, r1, j0, j1 = _put_bounds(h, w, H, W, y0, x0, t, tasks)
        n = max(j1 - j0, 0)
        vals = np.empty(n, np.float32)
        rows_ = np.empty(n, np.int64)
        for i in range(r0, r1):
            ci = y0 + i
            if ci < 0 or ci >= H or j1 <= j0:
                continue
            crow = c32[ci]
            if rowok[i] == 0:
                for j in range(j0, j1):
                    _put_word(crow, canvas, ci, x0 + j, bad, fast)
                continue
            src = z[ky[i]]
            for q in range(n):
                vals[q] = src[kx[j0 + q]]
            _linear_rows(vals, vmin, vmax, clip, N, rows_)
            for q in range(n):
                word = lut[rows_[q]] if colok[j0 + q] != 0 else bad
                _put_word(crow, canvas, ci, x0 + j0 + q, word, fast)


@njit(cache=True, nogil=True)
def _linear_rows(v, vmin, vmax, clip, N, out):
    """out[q]: _color._row_linear(float64(v[q]), True, vmin, vmax, clip, N)
    for float32 v -- the same arithmetic, in the same order, its choices
    of row made as selects (_row_scaled's: NaN 0, N the last row, under 0
    row N, over N row N + 1; not finite N + 2), so that the loop is one
    the compiler can vectorize."""
    n = v.size
    F = np.float64(N)
    last = F - 1.0
    if vmin == vmax:
        for q in range(n):
            x = np.float64(v[q])
            out[q] = 0 if x - x == 0.0 else N + 2
        return
    if clip:
        for q in range(n):
            x = np.float64(v[q])
            d = min(max(x, vmin), vmax)
            out[q] = _scaled_row((d - vmin) / (vmax - vmin) * N, F, last, N, x)
        return
    for q in range(n):
        x = np.float64(v[q])
        s = np.float64(np.float32(x - vmin))
        s = np.float64(np.float32(s / (vmax - vmin)))
        out[q] = _scaled_row(np.float64(np.float32(s * N)), F, last, N, x)


@njit(cache=True, nogil=True, inline='always')
def _scaled_row(tN, F, last, N, x):
    """_color._row_scaled(tN, N) by selects, N + 2 for a value x not finite."""
    c = tN if tN > 0.0 else 0.0                         # (NaN: 0)
    c = c if c < last else last                         # (N itself: the last row)
    k = np.int64(c)
    k = N if tN < 0.0 else k
    k = N + 1 if tN > F else k
    return k if x - x == 0.0 else N + 2


@njit(cache=True, nogil=True, parallel=True)
def _cells_table(colors, table, spans, bad, canvas, c32, y0, x0, fast, tasks):
    """_raster.take_colors (each pixel its cell's colour, bad for none) onto
    the canvas at (y0, x0); a transparent bad not put round the row's span
    of cells (spans: _raster.table_spans)."""
    h, w = table.shape
    H, W = canvas.shape[0], canvas.shape[1]
    clear = (bad >> 24) == 0
    for t in prange(tasks):
        r0, r1, j0, j1 = _put_bounds(h, w, H, W, y0, x0, t, tasks)
        for i in range(r0, r1):
            ci = y0 + i
            if ci < 0 or ci >= H or j1 <= j0:
                continue
            crow = c32[ci]
            trow = table[i]
            s, e = j0, j1
            if clear:
                s, e = max(s, spans[i, 0]), min(e, spans[i, 1])
            for j in range(s, e):
                k = trow[j]
                _put_word(crow, canvas, ci, x0 + j, colors[k] if k >= 0 else bad, fast)


def _fixed(coloring):
    """Are a value's colours its own, whatever else is shown (a norm already
    scaled, or a cell's own colour)?"""
    norm = getattr(coloring, 'norm', None)
    return isinstance(coloring, IndexColoring) or (isinstance(norm, _color.Norm) and
                                                   norm.scaled())


def _cell_words(coloring, z, alpha):
    """The cells z (float32, 2-D: shaded by rows on every core) shaded:
    (their colours as uint32 raveled, the colour of a pixel of no cell (a
    NaN's), whether all are opaque or transparent)."""
    col = np.empty(z.shape + (4,), np.uint8)
    coloring.shade(z, col, alpha)
    nan = np.empty((1, 1, 4), np.uint8)
    coloring.shade(np.full((1, 1), np.nan, np.float32), nan, alpha)
    a = np.append(col[..., 3].reshape(-1), nan[0, 0, 3])
    return (col.view(np.uint32).reshape(-1), nan.view(np.uint32).reshape(-1)[0],
            bool(np.all((a == 0) | (a == 255))))


def bands_guess(levels):
    """_guess's (L0, D, mu, mf, vec) for these levels: levels[0] and 1 / d in
    float32, the margin mu, m as float32; vec False where no guess is sure.
    mu bounds, in steps, how far a level lies from its even step (E / d, as
    computed, and its rounding) plus how far the float32 g can lie from
    (x - l0) / d + 1 for |g| up to m + 2 (each rounding at most 2^-24 of its
    result, levels[0]'s at most 2^-24 |l0| / d in steps) -- times 4 -- and
    fr's rounding below 0: a guess further than mu from every integer is
    exactly the count _band_row makes. Beyond the ends (g < -1/2, g > m + 1/2)
    it is 0 or m on any levels."""
    lev = np.asarray(levels, np.float64)
    m = len(lev)
    d = (lev[-1] - lev[0]) / (m - 1)
    E = np.abs(lev - (lev[0] + d * np.arange(m))).max()
    l0 = lev[0]
    D = np.float32(1.0 / d)
    err = 2.0 ** -21 * (abs(l0) / d + 4 * (m + 3)) + 2.0 ** -24
    ok = bool(np.isfinite(D) and D > 0 and err < 0.25 and np.abs(lev).max() < 1e37)  # (and
    # x - l0 overflows float32 only for an x past every level)
    mu = E / d + 4 * 2.0 ** -53 * np.abs(lev).max() / d + err
    return (np.float32(l0), D, np.float32(min(mu, 1.0)), np.float32(m), ok)


def touched_bands_xp(xp, src, top, low=False):
    """touched_bands in array operations, for cupy (or numpy) arrays. low:
    with the smallest values of the squares too."""
    pad = xp.pad(src, 1, constant_values=xp.nan)
    zero = xp.float32(0.0)

    def step(n):                                        # the rise to a neighbour, 0 if none
        with np.errstate(invalid='ignore'):
            d = n - src
        return xp.where(xp.isfinite(d), d, zero)
    dr, dl = step(pad[1:-1, 2:]), step(pad[1:-1, :-2])
    dd, du = step(pad[2:, 1:-1]), step(pad[:-2, 1:-1])
    rx = xp.maximum(xp.maximum(dr, dl), zero)
    ry = xp.maximum(xp.maximum(dd, du), zero)
    fx = xp.maximum(xp.maximum(-dr, -dl), zero)
    fy = xp.maximum(xp.maximum(-dd, -du), zero)
    half = xp.float32(0.5)
    hi = src + half * (rx + ry)
    lo = src - half * (fx + fy)
    top = xp.float32(top)
    out = xp.ascontiguousarray(xp.where((hi > top) & (lo <= top), top, hi), dtype=xp.float32)
    return (out, lo) if low else out


def band_seams(xp, src, coloring):
    """Translucent bands: where two touch a pixel Agg paints it with both,
    so seams show darker between them. The lower band's values where a
    pixel has two (NaN elsewhere), to paint before the raster itself."""
    hi, lo = touched_bands_xp(xp, src, coloring.top, low=True)
    lv = xp.asarray(coloring.levels.astype(np.float32))
    return xp.where(xp.searchsorted(lv, hi) != xp.searchsorted(lv, lo), lo, xp.float32(xp.nan))


@njit(cache=True, nogil=True, parallel=True)
def _blend_nearest(canvas, img, rows, cols, y0, x0, alpha):
    """img's pixel rows[i], cols[j] (-1: none) over canvas pixel (y0 + i,
    x0 + j): a nearest-sampled image, gathered and blended in one pass."""
    H, W = canvas.shape[0], canvas.shape[1]
    for i in prange(rows.size):
        ci, r = y0 + i, rows[i]
        if r < 0 or ci < 0 or ci >= H:
            continue
        for j in range(cols.size):
            cj, c = x0 + j, cols[j]
            if c < 0 or cj < 0 or cj >= W:
                continue
            a = img[r, c, 3]
            if a == 0:
                continue
            if a == 255 and alpha == _F1:
                canvas[ci, cj, 0] = img[r, c, 0]
                canvas[ci, cj, 1] = img[r, c, 1]
                canvas[ci, cj, 2] = img[r, c, 2]
                canvas[ci, cj, 3] = 255
                continue
            _draw.blend_px(canvas, ci, cj, np.float32(img[r, c, 0]), np.float32(img[r, c, 1]),
                           np.float32(img[r, c, 2]), np.float32(a), alpha)


@njit(cache=True, nogil=True, parallel=True)
def rgba_bytes(Z, out):
    """An RGB(A) float image as matplotlib's bytes: NaN 0, clipped to
    0..1, times 255, truncated; alpha 255 for RGB."""
    ny, nx, nc = Z.shape
    for i in prange(ny):
        for j in range(nx):
            for c in range(4):
                if c == 3 and nc == 3:
                    out[i, j, 3] = 255
                    continue
                x = Z[i, j, c]
                if x != x:
                    x = 0.0
                x = min(max(x, 0.0), 1.0)
                out[i, j, c] = np.uint8(x * 255.0)


@njit(cache=True, nogil=True, inline='always')
def _byte(img, r, c, k, floats):
    """Channel k of img's pixel (r, c) as rgba_bytes makes it a byte (floats),
    or the byte it is."""
    if not floats:
        return np.uint8(img[r, c, k])
    x = np.float64(img[r, c, k])
    if x != x:
        x = 0.0
    x = min(max(x, 0.0), 1.0)
    return np.uint8(x * 255.0)


@njit(cache=True, nogil=True, parallel=True)
def _blend_nearest_any(canvas, img, rows, cols, y0, x0, alpha, floats):
    """_blend_nearest of an image as it was given -- uint8 RGB, or float
    RGB(A) (floats) -- each pixel drawn made bytes as rgba_bytes makes
    them (alpha 255 for RGB), so that none other is converted."""
    H, W = canvas.shape[0], canvas.shape[1]
    four = img.shape[2] == 4
    for i in prange(rows.size):
        ci, r = y0 + i, rows[i]
        if r < 0 or ci < 0 or ci >= H:
            continue
        for j in range(cols.size):
            cj, c = x0 + j, cols[j]
            if c < 0 or cj < 0 or cj >= W:
                continue
            a = _byte(img, r, c, 3, floats) if four else np.uint8(255)
            if a == 0:
                continue
            red, green, blue = (_byte(img, r, c, 0, floats), _byte(img, r, c, 1, floats),
                                _byte(img, r, c, 2, floats))
            if a == 255 and alpha == _F1:
                canvas[ci, cj, 0] = red
                canvas[ci, cj, 1] = green
                canvas[ci, cj, 2] = blue
                canvas[ci, cj, 3] = 255
                continue
            _draw.blend_px(canvas, ci, cj, np.float32(red), np.float32(green), np.float32(blue),
                           np.float32(a), alpha)


@njit(cache=True, nogil=True, parallel=True)
def _blend_image(canvas, img, y0, x0, alpha):
    """img (h, w, 4) straight RGBA over the canvas at (y0, x0), clipped;
    alpha scales img's own alpha. The internals.md blend, per pixel."""
    H, W = canvas.shape[0], canvas.shape[1]
    h, w = img.shape[0], img.shape[1]
    for i in prange(h):
        ci = y0 + i
        if ci < 0 or ci >= H:
            continue
        for j in range(w):
            cj = x0 + j
            if cj < 0 or cj >= W:
                continue
            a = img[i, j, 3]
            if a == 0:
                continue
            if a == 255 and alpha == _F1:
                canvas[ci, cj, 0] = img[i, j, 0]
                canvas[ci, cj, 1] = img[i, j, 1]
                canvas[ci, cj, 2] = img[i, j, 2]
                canvas[ci, cj, 3] = 255
                continue
            _draw.blend_px(canvas, ci, cj, np.float32(img[i, j, 0]), np.float32(img[i, j, 1]),
                           np.float32(img[i, j, 2]), np.float32(a), alpha)


_NAN32 = np.array([np.nan], np.float32).view(np.uint32)[0]


@njit(cache=True, nogil=True, parallel=True)
def _fill32(a, v):
    """Every pixel of a new canvas (as uint32) set, rows on every core (the
    first touch of fresh memory is the cost, so it is spread too)."""
    for i in prange(a.shape[0]):
        a[i, :] = v


@njit(cache=True, nogil=True, parallel=True)
def _fill_rect(canvas, c32, x0, y0, x1, y1, r, g, b, a, word):
    """An opaque rectangle as whole pixels (word: its RGBA bytes as the
    canvas's uint32, c32 the canvas so viewed); a translucent one blended."""
    if a == _F255:
        for i in prange(y0, y1):
            c32[i, x0:x1] = word
        return
    for i in prange(y0, y1):
        for j in range(x0, x1):
            _draw.blend_px(canvas, i, j, r, g, b, a, _F1)


_SCRATCH = threading.local()
WINDOW_A = 'window.a'           # this thread's window buffers (_arena), shared by what never
WINDOW_B = 'window.b'           # holds them at once: last_touched, contourf's hole edges


def scratch(H, W):
    """This thread's reusable (H, W, 4) canvas memory, for a draw whose
    image does not outlive the call (savefig): no fresh pages to fault in."""
    a = getattr(_SCRATCH, 'a', None)
    if a is None or a.shape != (H, W, 4):
        a = _SCRATCH.a = np.empty((H, W, 4), np.uint8)
    return a


class CpuCanvas:
    """The canvas in host memory, drawn by numba (into buf if given)."""

    device = 'cpu'

    def __init__(self, H, W, rgba, buf=None):
        self.H, self.W = H, W
        self.a = np.empty((H, W, 4), np.uint8) if buf is None else buf
        _fill32(self.a.view(np.uint32).reshape(H, W), np.asarray(rgba, np.uint8).view(np.uint32)[0])

    def _clip(self, x0, y0, x1, y1):
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(self.W, int(x1)), min(self.H, int(y1))
        return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None

    def fill_rect(self, x0, y0, x1, y1, rgba):
        r = self._clip(x0, y0, x1, y1)
        if r is None or rgba[3] == 0:
            return
        c = np.asarray(rgba, np.float32)
        word = np.array([c[0], c[1], c[2], 255], np.float32).astype(np.uint8).view(np.uint32)[0]
        _fill_rect(self.a, self.a.view(np.uint32).reshape(self.H, self.W), r[0], r[1], r[2], r[3],
                   c[0], c[1], c[2], c[3], word)

    # A field's value raster and colour layer are this thread's _arena
    # buffers: ~7 MB each for an 1800x975 map, fresh pages every frame
    # otherwise. last_values gives the touched raster of the last field
    # until the next (a hatched contourf reads its bands from it, one with
    # holes the values round them, straight after drawing): made when asked
    # for, from the value raster kept -- made again where the field went
    # onto the canvas without one. keep_values (a contourf that will ask)
    # keeps it as the field is drawn.

    keep_values = False
    _lv = None

    def _values(self, h, w, nan=False):
        v = _arena.take('canvas.values', (h, w), np.float32)
        if nan and v.size:
            _fill32(v.view(np.uint32), _NAN32)          # (v.fill(np.nan), rows on every core)
        return v

    @property
    def last_values(self):
        box, values, top, remake = self._lv
        if values is None and remake is not None:
            values = remake(self)
        if top is None:
            self._lv = (box, values, None, None)
            return box, values
        touched = _arena.take('canvas.touched', values.shape, np.float32)
        touched_bands(values, touched, top)
        self._lv = (box, touched, None, None)
        return box, touched

    def last_box(self):
        """The last field's box."""
        return self._lv[0]

    def last_touched(self, y0, y1, x0, x1):
        """last_values' touched raster in rows y0..y1, columns x0..x1 alone
        (made from the value raster there and a pixel round it)."""
        box, values, top, remake = self._lv
        if top is None or (values is None and remake is None):
            return self.last_values[1][y0:y1, x0:x1]
        if values is None:
            values = remake(self)
            self._lv = (box, values, top, None)
        h, w = values.shape
        a0, a1, b0, b1 = max(y0 - 1, 0), min(y1 + 1, h), max(x0 - 1, 0), min(x1 + 1, w)
        sub = values[a0:a1, b0:b1]                      # (this thread's buffers, frame to frame:
        if not sub.flags.c_contiguous:                  # WINDOW_A free again once this returns,
            buf = _arena.take(WINDOW_A, sub.shape, sub.dtype)   # WINDOW_B what it returns)
            np.copyto(buf, sub)
            sub = buf
        t = _arena.take(WINDOW_B, sub.shape, sub.dtype)
        if sub.size:
            touched_bands(sub, t, top)
        return t[y0 - a0:y1 - a0, x0 - b0:x1 - b0]

    def _direct(self, coloring):
        """Can the field's bands go onto the canvas as its rows are made?
        (Translucent ones paint their seams from the whole raster first.)"""
        return coloring.bands and not coloring.translucent

    def _bands_args(self, box, coloring, alpha):
        """_bands_*'s arguments after the source's: the canvas, the plan."""
        a = np.float32(1.0 if alpha is None else alpha)
        tasks = max(1, min(ROW_TASKS * _jit.team(), box[3] - box[1]))
        args, opaque = coloring.fused()
        return (self.a, self.a.view(np.uint32).reshape(self.H, self.W), int(box[1]),
                int(box[0])) + args + (bool(opaque and a == _F1), a, tasks)

    def _keep(self, h, w):
        """The value raster a field made row by row keeps (keep_values), and
        _bands_*'s array for it ((0, 0): none)."""
        if not self.keep_values:
            return None, _NO_KEEP
        v = self._values(h, w)
        return v, v

    def _blend_field(self, box, values, coloring, alpha):
        h, w = values.shape
        if coloring.bands:
            a = np.float32(1.0 if alpha is None else alpha)
            if coloring.translucent:
                layer = _arena.take('canvas.layer', (h, w, 4), np.uint8)
                coloring.shade(np.ascontiguousarray(band_seams(np, values, coloring)), layer, None)
                _blend_image(self.a, layer, box[1], box[0], a)
            self._lv = (box, values, coloring.top, None)
            if h and w:
                _bands_raster(values, *self._bands_args(box, coloring, alpha))
            return
        layer = _arena.take('canvas.layer', (h, w, 4), np.uint8)
        self._lv = (box, values, None, None)
        coloring.shade(values, layer, alpha)
        _blend_image(self.a, layer, box[1], box[0], _F1)

    def _sep_direct(self, box, cols, rows, zz, mode, coloring, alpha):
        """A separable field's rows made as its bands go onto the canvas
        (_raster._sep's rows, interpolated along x once): True if done."""
        if not self._direct(coloring):
            return False
        got = _raster.sep_xrows(cols, rows, zz, mode, box[3] - box[1])
        if got is None:
            return False
        X, slot, colok = got

        def remake(c):                                  # (the canvas passed, not held: a
            values = c._values(box[3] - box[1], box[2] - box[0])   # canvas in a cycle of its
                                                                    # own waits for the collector)
            _raster._sep(values, cols, rows, zz, mode)
            return values
        kept, keep = self._keep(box[3] - box[1], box[2] - box[0])
        self._lv = (box, kept, coloring.top, remake)
        _bands_sep(X, slot, colok, rows, keep, *self._bands_args(box, coloring, alpha))
        return True

    def _cells_direct(self, box, cols, rows, zz, mode, coloring, alpha):
        """A separable field of nearest cells coloured by themselves, onto the
        canvas: True if done. Only the cells in view are coloured, and only
        when they are no more than the pixels they go to: a big image drawn
        small (a 60 M-pixel Landsat band in a 2 M-pixel figure) is cheaper
        coloured a pixel at a time, after sampling (the same colours: a
        cell's colour is its value's)."""
        if coloring.bands or mode != _raster.NEAREST or not _fixed(coloring) or \
                zz.ndim != 2 or zz.size == 0:
            return False
        h, w = box[3] - box[1], box[2] - box[0]
        kx, colok = cols[:, 0].astype(np.int64), (cols[:, 3] != 0.0).astype(np.uint8)
        ky, rowok = rows[:, 0].astype(np.int64), (rows[:, 3] != 0.0).astype(np.uint8)
        xs, ys = kx[colok != 0], ky[rowok != 0]
        kx0, kx1 = (int(xs.min()), int(xs.max()) + 1) if xs.size else (0, 1)
        ky0, ky1 = (int(ys.min()), int(ys.max()) + 1) if ys.size else (0, 1)
        if (kx1 - kx0) * (ky1 - ky0) > h * w:
            return self._pixels_direct(box, cols, rows, zz, coloring, alpha, kx, colok, ky, rowok)
        zc = np.ascontiguousarray(zz[ky0:ky1, kx0:kx1], dtype=np.float32)
        colors, bad, fast = _cell_words(coloring, zc, alpha)

        def remake(c):
            values = c._values(h, w)
            _raster._sep(values, cols, rows, zz, mode)
            return values
        self._lv = (box, None, None, remake)
        _cells_sep(colors.reshape(zc.shape), kx - kx0, colok, ky - ky0, rowok, bad, self.a,
                   self.a.view(np.uint32).reshape(self.H, self.W), int(box[1]), int(box[0]), fast,
                   max(1, min(ROW_TASKS * _jit.team(), h)))
        return True

    def _pixels_direct(self, box, cols, rows, zz, coloring, alpha, kx, colok, ky, rowok):
        """_cells_direct's pixels when the cells in view outnumber them: each
        shaded from its cell's value as it goes onto the canvas
        (_pixels_linear: a linear norm's), not sampled into a raster, shaded
        into a layer and blended. True if done."""
        if type(coloring) is not Coloring or zz.dtype != np.float32:
            return False
        plan = _color.shade_plan(coloring.norm, _color.get_cmap(coloring.cmap), alpha=alpha,
                                 rounding=coloring.rounding)
        if plan.kernel != 'shade_linear':
            return False
        flipy, flipx = zz.strides[0] < 0, zz.strides[1] < 0     # (sep_data's views: z itself
        z = zz[::-1] if flipy else zz                           # with the indices turned)
        z = z[:, ::-1] if flipx else z
        if not z.flags.c_contiguous:
            return False
        if flipy:
            ky = np.where(rowok != 0, z.shape[0] - 1 - ky, 0)
        if flipx:
            kx = np.where(colok != 0, z.shape[1] - 1 - kx, 0)
        vmin, vmax, clip, lut, N = _color._cpu_args(plan.args)
        nan = np.empty((1, 1, 4), np.uint8)             # a pixel of no cell: a NaN's colour
        coloring.shade(np.full((1, 1), np.nan, np.float32), nan, alpha)
        bad = nan.view(np.uint32).reshape(-1)[0]
        a = np.append(lut >> 24, bad >> 24)
        h = box[3] - box[1]

        def remake(c):
            values = c._values(h, box[2] - box[0])
            _raster._sep(values, cols, rows, zz, _raster.NEAREST)
            return values
        self._lv = (box, None, None, remake)
        _pixels_linear(z, kx, colok, ky, rowok, vmin, vmax, clip, lut, N, bad, self.a,
                       self.a.view(np.uint32).reshape(self.H, self.W), int(box[1]), int(box[0]),
                       bool(np.all((a == 0) | (a == 255))), max(1, min(ROW_TASKS * _jit.team(), h)))
        return True

    def field_rect(self, box, view, src, xaxis, yaxis, z, mode, coloring, alpha=None):
        h, w = box[3] - box[1], box[2] - box[0]
        if h <= 0 or w <= 0:
            return
        z = np.asarray(z, np.float32)
        if self._direct(coloring) or (mode == _raster.NEAREST and _fixed(coloring)):
            got = _raster.rect_sep(view, src, xaxis, yaxis, z, mode, h, w)
            if got is not None:
                cols, rows, zz = got
                if self._sep_direct(box, cols, rows, zz, mode, coloring, alpha) or \
                        self._cells_direct(box, cols, rows, zz, mode, coloring, alpha):
                    return
                values = self._values(h, w)
                _raster._sep(values, cols, rows, zz, mode)
                self._blend_field(box, values, coloring, alpha)
                return
        values = self._values(h, w)
        _raster.sample_rect(values, view, src, xaxis, yaxis, z, mode)
        self._blend_field(box, values, coloring, alpha)

    def field_sep(self, box, cols, rows, z, grid, mode, coloring, alpha=None):
        """A field from per-column and per-row tables (plain axes)."""
        h, w = box[3] - box[1], box[2] - box[0]
        if h <= 0 or w <= 0:
            return
        z = np.asarray(z, np.float32)
        zz = _raster.sep_data(z, grid.xaxis, grid.yaxis)
        if self._sep_direct(box, cols, rows, zz, mode, coloring, alpha) or \
                self._cells_direct(box, cols, rows, zz, mode, coloring, alpha):
            return
        values = self._values(h, w)
        _raster.sample_sep(values, cols, rows, z, grid.xaxis, grid.yaxis, mode)
        self._blend_field(box, values, coloring, alpha)

    def field_cells_px(self, box, sx, sy, z, mode, coloring, alpha=None):
        """A field's cells, nodes already in raster pixels (plain axes)."""
        h, w = box[3] - box[1], box[2] - box[0]
        if h <= 0 or w <= 0:
            return
        self._mesh(box, h, w, sx, sy, z, mode, 0.0, coloring, alpha)

    def field_cells(self, box, view, src, x2d, y2d, z, mode, coloring, alpha=None, period=0.0):
        h, w = box[3] - box[1], box[2] - box[0]
        if h <= 0 or w <= 0:
            return
        sx, sy = _raster.kept_nodes(view, src, x2d, y2d)
        self._mesh(box, h, w, sx, sy, z, mode, period, coloring, alpha)

    def _mesh(self, box, h, w, sx, sy, z, mode, period, coloring, alpha):
        """A field's cells: by the mesh's table of each pixel's cell, once
        it has one, its cells shaded and their colours taken to the pixels
        (the pixels' values are the cells' own: the same colours, for a
        tenth of the shading on a map of fine pixels); a band colouring
        (contourf) or the first frame by the value raster."""
        touched = _raster.wants_touched(sx, sy, mode, period, np.size(z), (h, w))
        if not coloring.bands and _fixed(coloring):     # (else: scaled by what is shown)
            got = _raster.cell_table((h, w), sx, sy, z, mode, period, touched)
            if got is not None:
                table, zf, spans = got
                colors, bad, fast = _cell_words(coloring, zf.reshape(np.shape(z)), alpha)
                self._lv = (box, None, None, None)      # (no value raster: no contourf's)
                _cells_table(colors, table, spans, bad, self.a,
                             self.a.view(np.uint32).reshape(self.H, self.W), int(box[1]),
                             int(box[0]), fast, max(1, min(ROW_TASKS * _jit.team(), h)))
                return
        if not touched and self._direct(coloring):     # contourf's: by the mesh's triangles
            got = _raster.tri_table((h, w), sx, sy, z, mode, period)
            if got is not None:
                table, zf, spans = got

                def remake(c):
                    values = c._values(h, w)
                    _raster.take_tris(table, sx, sy, zf, period, values)
                    return values
                kept, keep = self._keep(h, w)
                self._lv = (box, kept, coloring.top, remake)
                _bands_tris(table, spans, sx.reshape(-1), sy.reshape(-1), zf, sx.shape[1],
                            float(period), keep, *self._bands_args(box, coloring, alpha))
                return
        values = self._values(h, w, nan=True)
        self._cells(values, sx, sy, z, mode, period, touched)
        self._blend_field(box, values, coloring, alpha)

    def _cells(self, values, sx, sy, z, mode, period, touched=None):
        if touched is None:
            touched = _raster.wants_touched(sx, sy, mode, period, np.size(z), values.shape)
        if _raster.cells_by_table(values, sx, sy, z, mode, period, touched):
            return
        if touched:
            cells = _arena.take('canvas.cells', values.shape, np.float32)
            _raster.raster_cells_touched(values, cells, sx, sy, z, mode)
        else:
            _raster.raster_cells(values, sx, sy, z, mode, period)

    def segments(self, segs, styles, clip, memo=None):
        _draw.draw_segments(self.a, segs, styles, clip, memo)

    def polygons(self, xy, ring_offsets, rgba, clip, rule='evenodd', ring_path=None, memo=None):
        _draw.fill_polygons(self.a, xy, ring_offsets, rgba, clip, rule, ring_path, memo)

    def paths(self, xy, offsets, faces, part_style, styles, clip):
        _draw.draw_paths(self.a, xy, offsets, faces, part_style, styles, clip)

    def mask(self, alpha_mask, x0, y0, rgba):
        from . import _text
        _text.blend_mask(self.a, alpha_mask, int(x0), int(y0), rgba)

    def field_values(self, box, values, coloring, alpha=None):
        """A value raster already made (the axes box's), coloured and blended."""
        if values.size:
            self._blend_field(box, values, coloring, alpha)

    def hatch(self, xy, ring_offsets, pattern, color, dpi, clip):
        from . import _hatch
        _hatch.fill(self.a, xy, ring_offsets, pattern, color, dpi, clip)

    def stamp(self, pts, fill, stroke, width, face, edge, clip, cap, which=None, join=None):
        _draw.stamp_markers(self.a, pts, fill, stroke, width, face, edge, clip, cap, which, join)

    def exact_markers(self, pts, sizes, fill, stroke, width, face, edge, clip):
        _draw.exact_markers(self.a, pts, sizes, fill, stroke, width, face, edge, clip)

    def image_nearest(self, rgba, rows, cols, x0, y0, alpha=1.0):
        """rgba: uint8 RGBA, or (C-contiguous) uint8 RGB or float32 /
        float64 RGB(A) as imshow takes them, made bytes as they are drawn
        (_blend_nearest_any)."""
        if rgba.dtype == np.uint8 and rgba.shape[2] == 4:
            _blend_nearest(self.a, rgba, rows, cols, int(y0), int(x0), np.float32(alpha))
        else:
            _blend_nearest_any(self.a, rgba, rows, cols, int(y0), int(x0), np.float32(alpha),
                               rgba.dtype != np.uint8)

    def image(self, rgba, x0, y0, alpha=1.0):
        _blend_image(self.a, np.ascontiguousarray(rgba, np.uint8), int(y0), int(x0),
                     np.float32(alpha))

    def array(self):
        return self.a
