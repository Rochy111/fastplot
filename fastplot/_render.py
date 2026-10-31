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

from . import _arena, _color, _draw, _raster
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


@njit(cache=True, nogil=True, inline='always')
def _rise(c, n, rise, fall):
    """rise and fall from c to a neighbour n, added in if it has a value
    (off the edge, n is c itself: nothing either way)."""
    if n - n == 0.0:                                    # (not NaN, not infinite)
        d = n - c
        if d > rise:
            rise = d
        if -d > fall:
            fall = -d
    return rise, fall


@njit(cache=True, nogil=True, parallel=True)
def touched_bands(src, out, top):
    """contourf's polygons as Agg fills them, not antialiased, lowest band
    first: a pixel any polygon touches gets its colour, so the highest
    band touching each pixel shows. Each pixel's value becomes the largest
    in its square -- the centre plus half the largest rise to a neighbour
    across and down (a plane's corner) -- held to `top` when the square
    reaches down into the painted bands."""
    H, W = src.shape
    for i in prange(H):
        up, down = max(i - 1, 0), min(i + 1, H - 1)
        for j in range(W):
            c = src[i, j]
            if not np.isfinite(c):
                out[i, j] = c
                continue
            rx, fx = _rise(c, src[i, min(j + 1, W - 1)], np.float32(0.0), np.float32(0.0))
            rx, fx = _rise(c, src[i, max(j - 1, 0)], rx, fx)
            ry, fy = _rise(c, src[down, j], np.float32(0.0), np.float32(0.0))
            ry, fy = _rise(c, src[up, j], ry, fy)
            hi = c + np.float32(0.5) * (rx + ry)
            if hi > top and c - np.float32(0.5) * (fx + fy) <= top:
                hi = top
            out[i, j] = hi


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


@njit(cache=True, nogil=True, parallel=True)
def _fill32(a, v):
    """Every pixel of a new canvas (as uint32) set, rows on every core (the
    first touch of fresh memory is the cost, so it is spread too)."""
    for i in prange(a.shape[0]):
        a[i, :] = v


@njit(cache=True, nogil=True, parallel=True)
def _fill_rect(canvas, x0, y0, x1, y1, r, g, b, a):
    for i in prange(y0, y1):
        for j in range(x0, x1):
            if a == _F255:
                canvas[i, j, 0] = np.uint8(r)
                canvas[i, j, 1] = np.uint8(g)
                canvas[i, j, 2] = np.uint8(b)
                canvas[i, j, 3] = 255
            else:
                _draw.blend_px(canvas, i, j, r, g, b, a, _F1)


_SCRATCH = threading.local()


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
        _fill_rect(self.a, r[0], r[1], r[2], r[3], c[0], c[1], c[2], c[3])

    # A field's value raster and colour layer are this thread's _arena
    # buffers: ~7 MB each for an 1800x975 map, fresh pages every frame
    # otherwise. last_values holds the raster until the next field (a
    # hatched contourf reads its bands from it straight after drawing).

    def _values(self, h, w, nan=False):
        v = _arena.take('canvas.values', (h, w), np.float32)
        if nan:
            v.fill(np.nan)
        return v

    def _blend_field(self, box, values, coloring, alpha):
        h, w = values.shape
        layer = _arena.take('canvas.layer', (h, w, 4), np.uint8)
        a = np.float32(1.0 if (alpha is None or not coloring.bands) else alpha)
        if coloring.bands:
            if coloring.translucent:
                coloring.shade(np.ascontiguousarray(band_seams(np, values, coloring)), layer, None)
                _blend_image(self.a, layer, box[1], box[0], a)
            touched = _arena.take('canvas.touched', values.shape, np.float32)
            touched_bands(values, touched, coloring.top)
            values = touched
        self.last_values = (box, values)                 # a hatched contourf reads its bands
        coloring.shade(values, layer, None if coloring.bands else alpha)
        _blend_image(self.a, layer, box[1], box[0], a)

    def field_rect(self, box, view, src, xaxis, yaxis, z, mode, coloring, alpha=None):
        h, w = box[3] - box[1], box[2] - box[0]
        if h <= 0 or w <= 0:
            return
        values = self._values(h, w)
        _raster.sample_rect(values, view, src, xaxis, yaxis, np.asarray(z, np.float32), mode)
        self._blend_field(box, values, coloring, alpha)

    def field_sep(self, box, cols, rows, z, grid, mode, coloring, alpha=None):
        """A field from per-column and per-row tables (plain axes)."""
        h, w = box[3] - box[1], box[2] - box[0]
        if h <= 0 or w <= 0:
            return
        values = self._values(h, w)
        _raster.sample_sep(values, cols, rows, np.asarray(z, np.float32), grid.xaxis, grid.yaxis,
                           mode)
        self._blend_field(box, values, coloring, alpha)

    def field_cells_px(self, box, sx, sy, z, mode, coloring, alpha=None):
        """A field's cells, nodes already in raster pixels (plain axes)."""
        h, w = box[3] - box[1], box[2] - box[0]
        if h <= 0 or w <= 0:
            return
        values = self._values(h, w, nan=True)
        self._cells(values, sx, sy, z, mode, 0.0)
        self._blend_field(box, values, coloring, alpha)

    def field_cells(self, box, view, src, x2d, y2d, z, mode, coloring, alpha=None, period=0.0):
        h, w = box[3] - box[1], box[2] - box[0]
        if h <= 0 or w <= 0:
            return
        values = self._values(h, w, nan=True)
        sx, sy = _raster.kept_nodes(view, src, x2d, y2d)
        self._cells(values, sx, sy, z, mode, period)
        self._blend_field(box, values, coloring, alpha)

    def _cells(self, values, sx, sy, z, mode, period):
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
        _blend_nearest(self.a, rgba, rows, cols, int(y0), int(x0), np.float32(alpha))

    def image(self, rgba, x0, y0, alpha=1.0):
        _blend_image(self.a, np.ascontiguousarray(rgba, np.uint8), int(y0), int(x0),
                     np.float32(alpha))

    def array(self):
        return self.a
