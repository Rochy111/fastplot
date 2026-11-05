"""Text as matplotlib's Agg drew it before 3.9 (3.8 and earlier), for a
figure that should look like those versions' -- a run of frames of which
matplotlib 3.8 drew the earlier ones.

Those versions hint glyphs at text.hinting_factor (8) times the horizontal
resolution and scale x back, with text.hinting's flags ('force_autohint'),
lay a string out as one run of FreeType pens (FT2Font.set_text: hinted
advances, the font's 'kern' table) and draw it as one bitmap: upright,
blitted at whole pixels (draw_text_image's blend_solid_hspan); turned, the
bitmap's rectangle rasterized and each pixel taken from the bitmap through
Agg's spline36 filter. Lines stack 1.2 font heights apart (Text's
linespacing then). matplotlib 3.9 lays text out with harfbuzz and draws
each glyph at its own position instead; fastplot's _text draws that, the
default.

The FreeType of those versions (2.6.1, built into matplotlib) is not the
one freetype-py brings: its sizes round and its autohinter differ, a
pixel's worth on a 13 pt header. So the glyphs and metrics come from
matplotlib's own ft2font, found where matplotlib is installed: what is
drawn here is what that matplotlib draws. Without matplotlib, text stays
fastplot's (3.11's).

It applies when rcParams['text.hinting_factor'] is a number (None:
matplotlib 3.11's, the default); text.hinting names the FreeType flags
('force_autohint' then, 'default', 'no_autohint', 'no_hinting', and the
older True / False / 'auto' / 'native' / 'either' / 'none') and
text.kerning_factor the kerning's (0). Mathtext is drawn as fastplot draws
it either way.
"""
import math
import threading
import warnings
from functools import lru_cache

import numpy as np

from . import _agg
from ._jit import njit

_LOCK = threading.Lock()
_FT = []                        # matplotlib.ft2font, once looked for (None: not there)
_WARNED = []


def _ft2font():
    if not _FT:
        try:
            from matplotlib import ft2font
            _FT.append(ft2font)
        except Exception:                               # noqa: BLE001 -- no (working) matplotlib
            _FT.append(None)
    return _FT[0]


_FLAGS = {'default': 'LOAD_DEFAULT', 'no_autohint': 'LOAD_NO_AUTOHINT',
          'force_autohint': 'LOAD_FORCE_AUTOHINT', 'no_hinting': 'LOAD_NO_HINTING',
          True: 'LOAD_FORCE_AUTOHINT', False: 'LOAD_NO_HINTING', 'either': 'LOAD_DEFAULT',
          'native': 'LOAD_NO_AUTOHINT', 'auto': 'LOAD_FORCE_AUTOHINT', 'none': 'LOAD_NO_HINTING'}


def _flag(ft, name):
    flags = getattr(ft, 'LoadFlags', None)              # (matplotlib 3.10+: an enum, the
    if flags is not None:                               # LOAD_ names deprecated)
        return getattr(flags, name[5:])
    return getattr(ft, name)


def mode():
    """(load flags, hinting factor, kerning factor) when text is to be
    drawn as matplotlib before 3.9 drew it, else None."""
    from ._rc import active
    rc = active()
    hf = rc.get('text.hinting_factor')
    if hf is None:
        return None
    ft = _ft2font()
    key = (ft, hf, rc.get('text.kerning_factor'), rc.get('text.hinting', 'force_autohint'))
    try:
        return _MODES[key]                              # (asked for every text drawn)
    except (KeyError, TypeError):
        pass
    m = _mode(ft, hf, rc)
    if m is not None:
        try:
            _MODES[key] = m
        except TypeError:                               # (a setting that cannot be a key)
            pass
    return m


_MODES = {}


def _mode(ft, hf, rc):
    kf = int(rc.get('text.kerning_factor') or 0)
    if ft is None or not _usable(int(hf), kf):
        if not _WARNED:
            _WARNED.append(1)
            warnings.warn('text.hinting_factor is set, but no matplotlib whose ft2font takes '
                          'it (matplotlib before 3.13) is installed: drawing text as fastplot '
                          'does', stacklevel=4)
        return None
    h = rc.get('text.hinting', 'force_autohint')
    name = _FLAGS.get(h.lower() if isinstance(h, str) else h, 'LOAD_FORCE_AUTOHINT')
    return _flag(ft, name), int(hf), kf                 # (the flag as ft2font takes it: an
                                                        # int, or 3.10's LoadFlags)


@lru_cache(maxsize=64)
def _font(path, hf, kf):
    """matplotlib's FT2Font of a file at a hinting (and kerning) factor;
    matplotlib 3.11 deprecates the factors (what 3.8 drew with), so that
    warning is fastplot's business, not the caller's."""
    ft = _ft2font()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        try:
            return ft.FT2Font(path, hinting_factor=hf, _kerning_factor=kf)
        except TypeError:                               # (no _kerning_factor)
            return ft.FT2Font(path, hinting_factor=hf)


def _usable(hf, kf):
    """Does this matplotlib's FT2Font take a hinting factor? (Not after its
    removal, announced for 3.13.)"""
    import os
    try:
        _font(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'fonts',
                           'DejaVuSans.ttf'), hf, kf)
        return True
    except Exception:                                   # noqa: BLE001
        return False


def _set(path, pt, dpi, m):
    f = _font(path, m[1], m[2])
    f.clear()
    f.set_size(pt, dpi)
    return f


@lru_cache(maxsize=8192)
def metrics(path, pt, dpi, m, s):
    """RendererAgg.get_text_width_height_descent: (w, h, d) px."""
    with _LOCK:
        f = _set(path, pt, dpi, m)
        f.set_text(s, 0.0, flags=m[0])
        w, h = f.get_width_height()
        d = f.get_descent()
    return w / 64.0, h / 64.0, d / 64.0


@lru_cache(maxsize=4096)
def bitmap(path, pt, dpi, m, s):
    """draw_text's bitmap of one line: (coverage uint8 (h, w), the bbox's
    xMin px, the descent px)."""
    with _LOCK:
        f = _set(path, pt, dpi, m)
        f.set_text(s, 0, flags=m[0])
        f.draw_glyphs_to_bitmap(antialiased=True)
        img = np.array(f.get_image(), np.uint8)
        xo, _ = f.get_bitmap_offset()
        d = f.get_descent()
    img.setflags(write=False)
    return img, xo / 64.0, d / 64.0


def _rotate(angle, pts):
    """Affine2D().rotate_deg(angle) on (x, y) pairs."""
    a = math.radians(angle)
    c, s = math.cos(a), math.sin(a)
    return [(c * x - s * y, s * x + c * y) for x, y in pts]


@lru_cache(maxsize=4096)
def layout(text, path, pt, dpi, m, linespacing, ma, ha, va, rotation, rotation_mode):
    """matplotlib 3.8's Text._get_layout: the box (x0, y0, x1, y1) about
    the text's position, y up, each line's (string, x, y), and the turned
    box's four corners (as box_corners gives them)."""
    ls = 1.2 if linespacing is None or linespacing == 'normal' else float(linespacing)
    lines = text.split('\n')
    _, lp_h, lp_d = metrics(path, pt, dpi, m, 'lp')
    min_dy = (lp_h - lp_d) * ls
    ws, xs, ys = [], [], []
    thisy = 0.0
    baseline = d = 0.0
    for i, line in enumerate(lines):
        w, h, d = metrics(path, pt, dpi, m, line) if line else (0.0, 0.0, 0.0)
        h, d = max(h, lp_h), max(d, lp_d)
        ws.append(w)
        baseline = (h - d) - thisy
        if i == 0:
            thisy = -(h - d)
        else:
            thisy -= max(min_dy, (h - d) * ls)
        xs.append(0.0)
        ys.append(thisy)
        thisy -= d
    descent = d
    width = max(ws)
    xmin, xmax, ymax, ymin = 0.0, width, 0.0, ys[-1] - descent
    if ma == 'center':
        offs = [(x + width / 2 - w / 2, y) for x, y, w in zip(xs, ys, ws)]
    elif ma == 'right':
        offs = [(x + width - w, y) for x, y, w in zip(xs, ys, ws)]
    else:
        offs = list(zip(xs, ys))
    horiz = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
    rot = _rotate(rotation, horiz)
    xmin, xmax = min(p[0] for p in rot), max(p[0] for p in rot)
    ymin, ymax = min(p[1] for p in rot), max(p[1] for p in rot)
    width, height = xmax - xmin, ymax - ymin
    if rotation_mode != 'anchor':
        ox = (xmin + xmax) / 2 if ha == 'center' else xmax if ha == 'right' else xmin
        oy = ((ymin + ymax) / 2 if va == 'center' else ymax if va == 'top' else
              ymin + descent if va == 'baseline' else
              ymin + height - baseline / 2.0 if va == 'center_baseline' else ymin)
    else:
        (x1, y1), (x2, y2) = horiz[0], horiz[2]
        ox = (x1 + x2) / 2.0 if ha == 'center' else x2 if ha == 'right' else x1
        oy = ((y1 + y2) / 2.0 if va == 'center' else y2 if va == 'top' else
              y2 - baseline if va == 'baseline' else
              y2 - baseline / 2.0 if va == 'center_baseline' else y1)
        ox, oy = _rotate(rotation, [(ox, oy)])[0]
    box = (xmin - ox, ymin - oy, xmin - ox + width, ymin - oy + height)
    pos = [(s, x - ox, y - oy) for s, (x, y) in zip(lines, _rotate(rotation, offs))]
    corners = tuple((rot[k][0] - ox, rot[k][1] - oy) for k in (0, 3, 2, 1))   # (_text's order)
    return box, tuple(pos), corners


def extent(box, x, y):
    """The layout's box at canvas (x, y), y down: (x0, y0, x1, y1)."""
    return (x + box[0], y - box[3], x + box[2], y - box[1])


def lines_at(pos, path, pt, dpi, m, x, y, rotation):
    """Each line as draw_text puts it: (bitmap, its bitmap() key, its left
    column and its bottom row + 1 on the canvas -- draw_text_image's x, y)."""
    out = []
    a = math.radians(rotation)
    for s, lx, ly in pos:
        if not s:
            continue
        key = (path, pt, dpi, m, s)
        img, xo, d = bitmap(*key)
        px, py = x + lx, y - ly                         # (canvas y down, as Text.draw flips)
        px = round(px + xo + d * math.sin(a))           # (Python's round, as matplotlib's)
        py = round(py + d * math.cos(a))
        out.append((img, key, int(px), int(py) + 1))
    return out


# ------------------------------------------------- turned: Agg's resampling

_SHIFT = 8                                              # image_subpixel_shift
_SCALE = 1 << _SHIFT
_MASK = _SCALE - 1
_FSHIFT = 14                                            # image_filter_shift
_FSCALE = 1 << _FSHIFT


def _spline36(x):
    if x < 1.0:
        return ((13.0 / 11.0 * x - 453.0 / 209.0) * x - 3.0 / 209.0) * x + 1.0
    if x < 2.0:
        x -= 1
        return ((-6.0 / 11.0 * x + 270.0 / 209.0) * x - 156.0 / 209.0) * x
    x -= 2
    return ((1.0 / 11.0 * x - 45.0 / 209.0) * x + 26.0 / 209.0) * x


def _iround(v):
    return int(v - 0.5) if v < 0 else int(v + 0.5)


def _lut():
    """agg::image_filter_lut of image_filter_spline36, normalized: (int16
    weights, diameter, start)."""
    radius = 3.0
    diameter = int(math.ceil(radius)) * 2
    start = -(diameter // 2 - 1)
    w = [0] * (diameter << _SHIFT)
    pivot = diameter << (_SHIFT - 1)
    for i in range(pivot):
        y = _spline36(i / _SCALE)
        w[pivot + i] = w[pivot - i] = _iround(y * _FSCALE)
    end = (diameter << _SHIFT) - 1
    w[0] = w[end]
    flip = 1                                            # image_filter_lut::normalize
    for i in range(_SCALE):
        while True:
            s = sum(w[j * _SCALE + i] for j in range(diameter))
            if s == _FSCALE:
                break
            k = _FSCALE / s
            s = 0
            for j in range(diameter):
                w[j * _SCALE + i] = _iround(w[j * _SCALE + i] * k)
                s += w[j * _SCALE + i]
            s -= _FSCALE
            inc = -1 if s > 0 else 1
            j = 0
            while j < diameter and s:
                flip ^= 1
                idx = diameter // 2 + j // 2 if flip else diameter // 2 - j // 2
                if w[idx * _SCALE + i] < _FSCALE:
                    w[idx * _SCALE + i] += inc
                    s += inc
                j += 1
    for i in range(pivot):
        w[pivot + i] = w[pivot - i]
    w[0] = w[end]
    return np.array(w, np.int64), diameter, start


_LUT = []


def lut():
    if not _LUT:
        _LUT.append(_lut())
    return _LUT[0]


@njit(cache=True, nogil=True, inline='always')
def _ir(v):
    return np.int64(v - 0.5) if v < 0 else np.int64(v + 0.5)


@njit(cache=True, nogil=True, inline='always')
def _cdiv(a, b):
    """C's integer division (toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


@njit(cache=True, nogil=True)
def _span(rec, img, inv, weights, diameter, start, x, y, n, covers, solid, ox, oy):
    """One scanline span of the turned bitmap: span_interpolator_linear over
    n pixels from x, span_image_filter_gray (spline36) of the bitmap
    (image_accessor_clip: 0 outside) -- each pixel's value and its cover
    (solid: one cover, covers[0]) recorded in rec (rows, cols, 2) at
    (y - oy, x - ox): font_to_rgba and Agg's blend make them a colour's."""
    a0, a1, a2, a3, a4, a5 = inv[0], inv[1], inv[2], inv[3], inv[4], inv[5]
    fx, fy = np.float64(x) + 0.5, np.float64(y) + 0.5
    tx, ty = a0 * fx + a2 * fy + a4, a1 * fx + a3 * fy + a5
    x1, y1 = _ir(tx * 256.0), _ir(ty * 256.0)
    fx2 = fx + n
    tx, ty = a0 * fx2 + a2 * fy + a4, a1 * fx2 + a3 * fy + a5
    x2, y2 = _ir(tx * 256.0), _ir(ty * 256.0)
    # dda2_line_interpolator, x and y
    cnt = n if n > 0 else 1
    lx, rx = _cdiv(x2 - x1, cnt), (x2 - x1) - _cdiv(x2 - x1, cnt) * cnt
    mx = rx
    vx = x1
    if mx <= 0:
        mx += n
        rx += n
        lx -= 1
    mx -= n
    ly, ry = _cdiv(y2 - y1, cnt), (y2 - y1) - _cdiv(y2 - y1, cnt) * cnt
    my = ry
    vy = y1
    if my <= 0:
        my += n
        ry += n
        ly -= 1
    my -= n
    h, w = img.shape
    half = 128                                          # filter_dx_int: image_subpixel_scale / 2
    for k in range(n):
        sx = vx - half
        sy = vy - half
        x_lr, y_lr = sx >> 8, sy >> 8
        x_fract = sx & 255
        y_hr = 255 - (sy & 255)
        fg = np.int64(0)
        for jy in range(diameter):
            wy = weights[y_hr]
            yy = y_lr + start + jy
            x_hr = 255 - x_fract
            for jx in range(diameter):
                xx = x_lr + start + jx
                if 0 <= yy < h and 0 <= xx < w:
                    v = np.int64(img[yy, xx])
                    if v:
                        fg += v * ((wy * weights[x_hr] + 8192) >> 14)
                x_hr += 256
            y_hr += 256
        fg >>= 14
        if fg < 0:
            fg = 0
        if fg > 255:
            fg = 255
        px, py = x + k - ox, y - oy
        if fg and 0 <= px < rec.shape[1] and 0 <= py < rec.shape[0]:
            rec[py, px, 0] = fg
            rec[py, px, 1] = covers[0] if solid else covers[k]
        mx += rx                                        # ++interpolator
        vx += lx
        if mx > 0:
            mx -= n
            vx += 1
        my += ry
        vy += ly
        if my > 0:
            my -= n
            vy += 1


@njit(cache=True, nogil=True, error_model='numpy')
def _turned(rec, img, inv, weights, diameter, start, cells, ra, rb, cx0, cy0, w, ox, oy):
    """sweep_scanline of the bitmap rectangle's cells (as _agg._sweep), each
    row's spans as scanline_p8 makes them -- runs of cells with ink of
    their own, solid runs between -- sampled by _span into rec."""
    cell, bits, span, stride, words = cells
    covs = np.zeros(w + 2, np.int64)
    one = np.zeros(1, np.int64)
    for row in range(ra, rb + 1):
        a, b = span[2 * row], span[2 * row + 1]
        span[2 * row], span[2 * row + 1] = 1 << 30, -1
        if b < a:
            continue
        # the row's cells, in x order, summed per x (they come off the bits so)
        c = 0
        i = cy0 + row
        at, wat = row * stride, row * words
        # scanline_p8's spans, emitted when they can grow no more (they never
        # overlap, so the order they are drawn in does not matter): a run of
        # cells (each its cover), or a solid run (one cover; one just after
        # another of the same cover joins it, as add_span joins them)
        run0, run_n = 0, 0                              # the open run of cells
        s0, s_n, s_m = 0, 0, 0                          # the open solid run
        last = -(1 << 30)                               # the last x a span covered + 1
        nxt = -(1 << 30)
        for word in range(a >> 6, (b >> 6) + 1):
            v = bits[wat + word]
            if v == 0:
                continue
            bits[wat + word] = 0
            while v != 0:
                low = v & -v
                v ^= low
                x = (word << 6) + _agg._LOG2[((low * _agg._DEBRUIJN) >> 58) & 63]
                q = at + 2 * x
                cc, aa = cell[q], cell[q + 1]
                cell[q] = 0
                cell[q + 1] = 0
                if c != 0 and nxt < x and nxt > -(1 << 29):     # add_span(nxt, x - nxt)
                    m = _agg._alpha(c << 9, False)
                    if m != 0:
                        if run_n:
                            _span(rec, img, inv, weights, diameter, start, cx0 + run0, i,
                                  run_n, covs, False, ox, oy)
                            run_n = 0
                        if s_n and nxt == last and m == s_m:
                            s_n += x - nxt
                        else:
                            if s_n:
                                one[0] = s_m
                                _span(rec, img, inv, weights, diameter, start, cx0 + s0, i,
                                      s_n, one, True, ox, oy)
                            s0, s_n, s_m = nxt, x - nxt, m
                        last = x
                c += cc
                if aa != 0:                             # add_cell(x), and x + 1
                    m = _agg._alpha((c << 9) - aa, False)
                    if m != 0:
                        if s_n:
                            one[0] = s_m
                            _span(rec, img, inv, weights, diameter, start, cx0 + s0, i,
                                  s_n, one, True, ox, oy)
                            s_n = 0
                        if run_n and x == last:
                            covs[run_n] = m
                            run_n += 1
                        else:
                            if run_n:
                                _span(rec, img, inv, weights, diameter, start, cx0 + run0,
                                      i, run_n, covs, False, ox, oy)
                            run0, run_n = x, 1
                            covs[0] = m
                        last = x + 1
                    nxt = x + 1
                else:                                   # cover only: a solid run starts here
                    nxt = x
        if run_n:
            _span(rec, img, inv, weights, diameter, start, cx0 + run0, i, run_n, covs,
                  False, ox, oy)
        if s_n:
            one[0] = s_m
            _span(rec, img, inv, weights, diameter, start, cx0 + s0, i, s_n, one, True,
                  ox, oy)


def _mul(a, b):
    """agg::trans_affine::multiply: a, then b; (sx, shy, shx, sy, tx, ty)."""
    sx, shy, shx, sy, tx, ty = a
    t0 = sx * b[0] + shy * b[2]
    t2 = shx * b[0] + sy * b[2]
    t4 = tx * b[0] + ty * b[2] + b[4]
    shy = sx * b[1] + shy * b[3]
    sy = shx * b[1] + sy * b[3]
    ty = tx * b[1] + ty * b[3] + b[5]
    return (t0, shy, t2, sy, t4, ty)


def _mtx(h, angle, x, y):
    """draw_text_image's: translation(0, -h), rotation(-angle), translation(x, y)."""
    a = -angle * (math.pi / 180.0)
    m = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    m = _mul(m, (1.0, 0.0, 0.0, 1.0, 0.0, float(-h)))
    m = _mul(m, (math.cos(a), math.sin(a), -math.sin(a), math.cos(a), 0.0, 0.0))
    return _mul(m, (1.0, 0.0, 0.0, 1.0, float(x), float(y)))


def _invert(m):
    """agg::trans_affine::invert."""
    sx, shy, shx, sy, tx, ty = m
    d = 1.0 / (sx * sy - shy * shx)
    t0 = sy * d
    sy = sx * d
    shy = -shy * d
    shx = -shx * d
    t4 = -tx * t0 - ty * shx
    ty = -tx * shy - ty * sy
    return (t0, shy, shx, sy, t4, ty)


@lru_cache(maxsize=2048)
def turned(img_key, x, y, angle, W, H):
    """draw_text_image's turned branch, all but the colour: the bitmap
    (bitmap()'s, by its key) placed with its bottom-left at canvas (x, y)
    and turned angle degrees counter-clockwise about it, on a W x H canvas
    -- its rectangle rasterized, each pixel's value through the spline36
    filter: (values and covers, uint8 (rows, cols, 2), their top-left
    pixel), or None where nothing shows."""
    return turned_image(bitmap(*img_key)[0], x, y, angle, W, H)


def turned_image(img, x, y, angle, W, H):
    """turned() of a bitmap given."""
    h, w = img.shape
    m = _mtx(h, angle, x, y)
    sx, shy, shx, sy, tx, ty = m

    def fwd(u, v):                                      # conv_transform: trans_affine::transform
        return u * sx + v * shx + tx, u * shy + v * sy + ty
    pts = np.array([fwd(0.0, 0.0), fwd(float(w), 0.0), fwd(float(w), float(h)),
                    fwd(0.0, float(h))], np.float64)
    inv = np.array(_invert(m), np.float64)
    run = _agg.prepare(pts, np.array([0, 4], np.int64), np.zeros(1, np.int64), (0, 0, W, H))
    if run is None:                                     # (the rasterizer's clip box: the canvas)
        return None
    p, ipts, nxt, e_path, (x0, y0, x1, y1), _ = run
    cells = _agg._cells(y1 - y0, x1 - x0)
    weights, diameter, start = lut()
    st = np.empty(2, np.int64)
    st[0], st[1] = 1 << 30, -1
    _edges(p, ipts, nxt, x0, y0, x1, y1, cells, st)
    if st[1] < 0:
        return None
    rec = np.zeros((y1 - y0, x1 - x0, 2), np.uint8)
    _turned(rec, np.ascontiguousarray(img), inv, weights, diameter, start, cells, st[0], st[1],
            x0, y0, x1 - x0, x0, y0)
    rec.setflags(write=False)
    return rec, x0, y0


def alpha_mask(rec, ca):
    """The coverage mask that blends a colour of alpha ca (0..255) as Agg
    blends the turned bitmap's pixels (font_to_rgba's alpha, ca * v >> 8,
    then the blend with each pixel's cover): for each, the least m with
    multiply(ca, m) equal to multiply(ca * v >> 8, cover) -- there always
    is one, and it is below 255, so Agg's opaque copy never comes in where
    it did not."""
    v = rec[..., 0].astype(np.int64)
    cov = rec[..., 1].astype(np.int64)
    a = (int(ca) * v) >> 8
    t = a * cov + 128
    want = ((t >> 8) + t) >> 8                          # the blend's multiply(a, cover)
    m = np.arange(256, dtype=np.int64)
    t = int(ca) * m + 128
    table = ((t >> 8) + t) >> 8                         # multiply(ca, m), rising by 0 or 1
    out = np.searchsorted(table, want, side='left').astype(np.uint8)
    out[want == 0] = 0
    out.setflags(write=False)
    return out


@njit(cache=True, nogil=True, error_model='numpy')
def _edges(pts, ipts, nxt, cx0, cy0, cx1, cy1, cells, st):
    """The rectangle's edges into the cells, as _agg._band puts a path's."""
    ox, oy = cx0 * 256, cy0 * 256
    fx0, fy0, fx1, fy1 = np.float64(cx0), np.float64(cy0), np.float64(cx1), np.float64(cy1)
    cell, bits, span, stride, words = cells
    r0, r1 = 0, cy1 - cy0
    for e in range(pts.shape[0]):
        f = nxt[e]
        if f < 0:
            continue
        x1, y1, x2, y2 = ipts[e, 0], ipts[e, 1], ipts[f, 0], ipts[f, 1]
        if x1 != _agg._OUT and x2 != _agg._OUT:
            _agg._line(cell, bits, span, stride, words, np.int64(x1) - ox, np.int64(y1) - oy,
                       np.int64(x2) - ox, np.int64(y2) - oy, r0, r1, st)
        else:
            _agg._edge(cells, pts[e, 0], pts[e, 1], pts[f, 0], pts[f, 1], fx0, fy0, fx1, fy1,
                       ox, oy, r0, r1, st)
