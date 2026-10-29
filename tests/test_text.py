"""fastplot._text against matplotlib's Agg text: extents, glyph positions
and pixels; masks, rotation, clipping, the blend formula, CPU == GPU."""
import os

import numpy as np
import pytest

from fastplot import _text as T

mpl = pytest.importorskip('matplotlib')
mpl.use('agg')
import matplotlib.pyplot as plt                        # noqa: E402

W, H = 600, 300

PLAIN = ['Hello World', 'To', 'AV', 'Temperature (K)', '-10', '1000 hPa', 'gjy',
         'Valid 2026-10-28 12:00 UTC', 'Specific humidity flux', '30\u00b0N', 'Wind speed',
         '500 hPa geopotential height (dam)']
MATH = [r'T$_{2m}$ ($^\circ$C)', r'$10^{-3}$ kg m$^{-2}$', r'$\alpha + \beta = \gamma$',
        r'Wind (m s$^{-1}$)', r'$x_i^2$', r'$\mathbf{Bold}$ $\mathrm{rm}$', r'$\theta_e$ (K)',
        r'$\pm 5\%$', r'$2 \times 10^{5}$ Pa']
MULTI = ['a\nlonger line\nb', 'Title line\nsecond', 'x\n\ny']
SIZES = [(10, 100), (12, 100), (8, 72), (14, 150)]
ALIGN = [dict(ha='left', va='baseline'), dict(ha='center', va='center'),
         dict(ha='right', va='top', rotation=30), dict(ha='center', va='bottom', rotation=90),
         dict(ha='left', va='center_baseline', rotation=45, rotation_mode='anchor'),
         dict(ha='right', va='baseline', rotation=270, rotation_mode='anchor'),
         dict(ha='center', va='top', rotation=180)]


def _fig(dpi):
    return plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)


def mpl_extent(s, x, y, pt, dpi, **kw):
    fig = _fig(dpi)
    try:
        t = fig.text(x / W, 1 - y / H, s, fontsize=pt, **kw)
        b = t.get_window_extent(renderer=fig.canvas.get_renderer())
        return (b.x0, H - b.y1, b.x1, H - b.y0)
    finally:
        plt.close(fig)


def mpl_pixels(s, x, y, pt, dpi, **kw):
    fig = _fig(dpi)
    try:
        fig.text(x / W, 1 - y / H, s, fontsize=pt, **kw)
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., 0].astype(int)
    finally:
        plt.close(fig)


def our_pixels(s, x, y, pt, dpi, **kw):
    c = np.full((H, W, 4), 255, np.uint8)
    T.draw_text(c, s, x, y, pt * dpi / 72, (0, 0, 0, 255), **kw)
    return c[..., 0].astype(int)


# ------------------------------------------------------------- extents

@pytest.mark.parametrize('pt,dpi', SIZES)
@pytest.mark.parametrize('kw', ALIGN, ids=lambda k: '-'.join(str(v) for v in k.values()))
def test_extent_plain(pt, dpi, kw):
    for s in PLAIN:
        e = mpl_extent(s, 300.3, 150.6, pt, dpi, **kw)
        m = T.text_extent(s, 300.3, 150.6, pt * dpi / 72, **kw)
        assert np.allclose(e, m, atol=0.76), (s, e, m)


@pytest.mark.parametrize('pt,dpi', SIZES)
@pytest.mark.parametrize('kw', ALIGN, ids=lambda k: '-'.join(str(v) for v in k.values()))
def test_extent_math(pt, dpi, kw):
    for s in MATH:
        e = mpl_extent(s, 300.3, 150.6, pt, dpi, **kw)
        m = T.text_extent(s, 300.3, 150.6, pt * dpi / 72, **kw)
        assert np.allclose(e, m, atol=1.01), (s, e, m)      # mathtext ceils its box


@pytest.mark.parametrize('pt,dpi', SIZES[:2])
@pytest.mark.parametrize('ls', [None, 1.2, 2.0])
@pytest.mark.parametrize('kw', ALIGN[:5], ids=lambda k: '-'.join(str(v) for v in k.values()))
def test_extent_multiline(pt, dpi, ls, kw):
    for s in MULTI + ['T$_{2m}$\n(K)']:
        mk = dict(kw) if ls is None else dict(kw, linespacing=ls)
        e = mpl_extent(s, 300.3, 150.6, pt, dpi, **mk)
        m = T.text_extent(s, 300.3, 150.6, pt * dpi / 72, **mk)
        assert np.allclose(e, m, atol=1.01), (s, ls, e, m)


def test_extent_weights_and_empty():
    for kw in (dict(weight='bold'), dict(style='italic'), dict(weight='bold', style='italic')):
        for s in ('Temperature', 'AV To', '-10'):
            fig = _fig(100)
            t = fig.text(0.5, 0.5, s, fontsize=11, fontweight=kw.get('weight', 'normal'),
                         fontstyle=kw.get('style', 'normal'))
            b = t.get_window_extent(renderer=fig.canvas.get_renderer())
            plt.close(fig)
            m = T.text_extent(s, 300, 150, 11 * 100 / 72, **kw)
            assert np.allclose((b.x0, H - b.y1, b.x1, H - b.y0), m, atol=0.76), (s, kw)
    assert T.text_extent('', 10, 20, 12) == (10.0, 20.0, 10.0, 20.0)


def test_glyph_positions_match_raqm():
    """Advances, GPOS kerning and ligatures as matplotlib's libraqm layout."""
    from matplotlib.backends.backend_agg import get_hinting_flag
    from matplotlib.ft2font import FT2Font
    for pt, dpi in SIZES:
        f = FT2Font(T.font(10).path)
        f.set_size(pt, dpi)
        ours = T.font(pt * dpi / 72)
        for s in PLAIN + ['fluffy office', 'AVAWAY']:
            items = f._layout(s, flags=get_hinting_flag())
            line = T._plain_line(ours, s)
            assert len(items) == len(line.glyphs), s
            for it, (_, c, x, _) in zip(items, line.glyphs):
                assert abs(it.x - x) < 1e-6, (s, it.char, c, it.x, x)


def test_mathtext_glyph_positions():
    """Glyph by glyph as matplotlib's mathtext ('path' parse): sizes and
    offsets, sub- and superscripts included."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import MathTextParser
    P = MathTextParser('path')
    for pt, dpi in SIZES:
        for s in MATH:
            r = P.parse(s, dpi, FontProperties(size=pt))
            ln = T._math_line(T.font(pt * dpi / 72), s, pt * dpi / 72)
            assert len(r.glyphs) == len(ln.glyphs), s
            assert ln.width == r.width and ln.descent == r.depth
            assert ln.ascent == r.height - r.depth, s
            for g, (f, c, x, y) in zip(r.glyphs, ln.glyphs):
                assert ord(c) == g[2], s
                assert abs(f.size_px - g[1] * dpi / 72) < 0.02, s
                assert abs(g[4] - x) < 0.02 and abs(g[5] + y) < 0.02, (s, c, g[4:], (x, -y))


def test_mathtext_sanity():
    sz = 20
    base = T._math_line(T.font(sz), '$x$')
    sup = T._math_line(T.font(sz), '$x^2$')
    sub = T._math_line(T.font(sz), '$x_2$')
    both = T._math_line(T.font(sz), '$x_i^2$')
    (fx, _, xx, yx), (f2, _, x2, y2) = sup.glyphs
    assert abs(f2.size_px - 0.7 * fx.size_px) < 0.02            # shrunk
    assert x2 > xx and y2 < yx - 0.3 * sz                       # raised
    (_, _, _, ys), (_, _, _, y2s) = sub.glyphs[0], sub.glyphs[1]
    assert y2s > ys + 0.1 * sz                                  # lowered
    ysup, ysub = both.glyphs[1][3], both.glyphs[2][3]         # ship order: x, sup, sub
    assert ysup < base.glyphs[0][3] < ysub
    assert sup.ascent > base.ascent and sub.descent > base.descent
    # spaces, Greek, degree, escapes, unknown commands
    assert T._math_line(T.font(sz), r'$a\quad b$').width > T._math_line(T.font(sz), r'$a\,b$').width
    cs = [g[1] for g in T._math_line(T.font(sz), r'$\alpha\Omega\degree\times\pm\cdot\mu\%\$$').glyphs]
    assert cs == list('\u03b1\u03a9\u00b0\u00d7\u00b1\u22c5\u03bc%$')
    ln = T._math_line(T.font(sz), r'$\foo{x}$')
    assert ''.join(g[1] for g in ln.glyphs) == '\\foox'
    assert T.text_extent(r'a \$5 b', 0, 0, sz)[2] > 0
    assert [g[1] for g in T._layout(r'cost \$5', float(sz), False, False, None,
                                    'normal', 'left').lines[0].glyphs][-2:] == ['$', '5']
    # \mathbf is bold, letters in math italic
    fonts = [os.path.basename(g[0].path) for g in T._math_line(T.font(sz), r'$\mathbf{B}x\mathrm{d}$').glyphs]
    assert fonts == ['DejaVuSans-Bold.ttf', 'DejaVuSans-Oblique.ttf', 'DejaVuSans.ttf']


# ------------------------------------------------------------- pixels

@pytest.mark.parametrize('pt,dpi', SIZES)
def test_pixels_like_matplotlib(pt, dpi):
    """Plain and math text at a whole-pixel position: matplotlib's pixels
    (FreeType's own hinted glyphs) to within subpixel resampling."""
    for s in ['Hello World', 'Temperature (K) fluffy', '1000 hPa', r'T$_{2m}$ ($^\circ$C) $10^{-3}$']:
        a = mpl_pixels(s, 100, 150, pt, dpi)
        b = our_pixels(s, 100, 150, pt, dpi)
        ink = (a < 255) | (b < 255)
        err = np.abs(a - b)[ink].mean()
        lim = (6 if pt * dpi / 72 > 12 else 15) if '$' not in s else 25   # subpixel resampling
        assert err < lim, (s, err)
        best = min(((np.abs(a - np.roll(np.roll(b, dy, 0), dx, 1)).sum(), dx, dy)
                    for dx in (-1, 0, 1) for dy in (-1, 0, 1)))
        assert best[1:] == (0, 0), (s, best)


@pytest.mark.parametrize('kw', [dict(rotation=90), dict(rotation=30), dict(rotation=-45, ha='center', va='center')])
def test_rotated_pixels_in_place(kw):
    """Rotated text lands where matplotlib's does (its glyphs are drawn
    rotated by FreeType at fractional positions, ours are the upright mask
    turned and put on whole pixels)."""
    for pt, dpi in SIZES[:2]:
        a = mpl_pixels('Rotated text 25', 300, 150, pt, dpi, **kw)
        b = our_pixels('Rotated text 25', 300, 150, pt, dpi, **kw)
        best = min(((np.abs(a - np.roll(np.roll(b, dy, 0), dx, 1)).sum(), dx, dy)
                    for dx in (-1, 0, 1) for dy in (-1, 0, 1)))
        assert best[1:] == (0, 0), best
        yy, xx = np.mgrid[:H, :W]

        def centre(z):
            return np.array([((255 - z) * yy).sum(), ((255 - z) * xx).sum()]) / (255 - z).sum()

        assert np.abs(centre(a) - centre(b)).max() < (0.5 if kw['rotation'] == 90 else 1.0)


# ------------------------------------------------------------- masks

def test_exact_right_angles():
    for s in ('Hello gW', 'T$_{2m}$\n(K)'):
        m0 = T.mask(s, 17.3)
        h, w = m0.alpha.shape
        for rot, k in ((90, 1), (180, 2), (270, 3), (-90, 3), (450, 1)):
            m = T.mask(s, 17.3, rotation=rot)
            assert np.array_equal(m.alpha, np.rot90(m0.alpha, k))
        assert T.mask(s, 17.3, rotation=90)[1:] == (m0.ay, w - m0.ax)
        assert T.mask(s, 17.3, rotation=180)[1:] == (w - m0.ax, h - m0.ay)
        assert T.mask(s, 17.3, rotation=270)[1:] == (h - m0.ay, m0.ax)


def test_general_rotation_tracks_anchor():
    tm = T.mask('Hello gW', 20)

    def centre(m):
        a = m.alpha.astype(float)
        yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
        return np.array([((xx + 0.5) * a).sum() / a.sum() - m.ax,
                         ((yy + 0.5) * a).sum() / a.sum() - m.ay])

    c0 = centre(tm)
    for rot in (90.0, 180.0, 270.0):
        assert np.abs(centre(T._rotate(tm, rot, exact=False)) - centre(T._rotate(tm, rot))).max() < 0.1
    for rot in (15, 30, 45, 135, 200, 333):
        t = np.radians(rot)
        want = np.array([np.cos(t) * c0[0] + np.sin(t) * c0[1], -np.sin(t) * c0[0] + np.cos(t) * c0[1]])
        assert np.abs(centre(T.mask('Hello gW', 20, rotation=rot)) - want).max() < 0.25


def test_mask_anchor_and_cache():
    T.cache_clear()
    m = T.mask('Hg', 30)
    assert m.alpha.dtype == np.uint8 and not m.alpha.flags.writeable
    f = T.font(30)
    # the anchor is the pen origin on the baseline: H's stem starts right of it
    # (the mask begins after the anchor), g's descender below
    assert -4 < m.ax < 0 and m.alpha.shape[0] - m.ay > 0.15 * 30
    assert m.ay == pytest.approx(f.bbox('H')[1] * -1, abs=1)
    hits = T.cache_info()[1].hits
    assert T.mask('Hg', 30) is m
    assert T.cache_info()[1].hits == hits + 1
    tm, x0, y0, e = T.place('Hg', 100, 50, 30)
    assert tm is m and T.cache_info()[1].hits == hits + 2
    assert (x0, y0) == (100 - m.ax, 50 - m.ay)
    assert T.font(30) is f and T.font(30, 'bold') is not f
    assert T.font(30, family=T.font(30).path) is f
    assert T.mask('', 12).alpha.shape == (0, 0)
    assert T.mask('   ', 12).alpha.shape == (0, 0)


def test_place_is_draw_text():
    for kw in (dict(), dict(ha='center', va='top', rotation=33),
               dict(ha='right', va='center_baseline', rotation=90, rotation_mode='anchor'),
               dict(ha='center', va='bottom', linespacing=1.2)):
        s = 'Mean sea level\npressure (hPa)'
        a = np.zeros((200, 300, 4), np.uint8)
        e1 = T.draw_text(a, s, 150.4, 99.6, 15.2, (0.2, 0.4, 0.6, 0.9), **kw)
        tm, x0, y0, e2 = T.place(s, 150.4, 99.6, 15.2, **kw)
        b = np.zeros_like(a)
        T.blend_mask(b, tm.alpha, x0, y0, (0.2, 0.4, 0.6, 0.9))
        assert e1 == e2 == T.text_extent(s, 150.4, 99.6, 15.2, **kw)
        assert np.array_equal(a, b) and a[..., 3].any()


def test_clipping():
    s = 'Clipped text here'
    big = np.full((140, 340, 4), 200, np.uint8)
    T.draw_text(big, s, 100 - 30, 70, 20, (10, 20, 30, 255), ha='left', va='center')
    for ox, oy, w, h in ((100, 70, 60, 50), (100, 50, 40, 30), (150, 80, 200, 15), (95, 55, 300, 200)):
        c = np.full((h, w, 4), 200, np.uint8)
        T.draw_text(c, s, 100 - 30 - ox, 70 - oy, 20, (10, 20, 30, 255), ha='left', va='center')
        ref = big[oy:oy + h, ox:ox + w]
        assert np.array_equal(c[:ref.shape[0], :ref.shape[1]], ref)
    # entirely outside: nothing, no error
    c = np.full((20, 20, 4), 7, np.uint8)
    T.draw_text(c, s, -500, 10, 20, (0, 0, 0, 255))
    T.draw_text(c, s, 10, 500, 20, (0, 0, 0, 255))
    assert (c == 7).all()


def _blend_ref(canvas, m, x0, y0, rgba):
    out = canvas.copy()
    r, g, b, a = [np.float32(v) for v in rgba]
    mh, mw = m.shape
    for i in range(mh):
        for j in range(mw):
            y, x = y0 + i, x0 + j
            if 0 <= y < out.shape[0] and 0 <= x < out.shape[1] and m[i, j]:
                mm = np.float32(m[i, j]) / np.float32(255)
                f = (a / np.float32(255)) * mm
                k = np.float32(1) - f
                da = np.float32(out[y, x, 3])
                if da == np.float32(255):           # over an opaque pixel (internals.md)
                    for c, v in enumerate((r, g, b, np.float32(255))):
                        out[y, x, c] = np.uint8(np.float32(v * f)
                                                + np.float32(np.float32(out[y, x, c]) * k)
                                                + np.float32(0.5))
                    continue
                w = np.float32((da / np.float32(255)) * k)  # over a partly transparent one
                oa = np.float32(f + w)
                if oa <= 0:
                    continue
                for c, v in enumerate((r, g, b)):
                    out[y, x, c] = np.uint8(np.float32(np.float32(np.float32(v * f)
                                                       + np.float32(np.float32(out[y, x, c]) * w))
                                            / oa) + np.float32(0.5))
                out[y, x, 3] = np.uint8(np.float32(np.float32(255) * oa) + np.float32(0.5))
    return out


def test_blend_formula_exact():
    rng = np.random.default_rng(1)
    for _ in range(5):
        canvas = rng.integers(0, 256, (23, 31, 4), dtype=np.uint8)
        canvas[::2, :, 3] = 255                         # opaque rows and translucent ones
        m = rng.integers(0, 256, (9, 14), dtype=np.uint8)
        m[rng.random(m.shape) < 0.2] = 0
        m[0, 0], m[1, 1] = 255, 1
        rgba = rng.integers(0, 256, 4)
        x0, y0 = int(rng.integers(-8, 28)), int(rng.integers(-6, 20))
        want = _blend_ref(canvas, m, x0, y0, rgba)
        got = canvas.copy()
        T.blend_mask(got, m, x0, y0, np.asarray(rgba, np.uint8))
        assert np.array_equal(got, want)
    # opaque colour through full coverage replaces the pixel
    c = np.zeros((2, 2, 4), np.uint8)
    T.blend_mask(c, np.full((2, 2), 255, np.uint8), 0, 0, (1.0, 0.5, 0.0, 1.0))
    assert (c == [255, 128, 0, 255]).all()


def test_colours():
    s = 'Ab'
    a = np.zeros((40, 60, 4), np.uint8)
    b = np.zeros_like(a)
    T.draw_text(a, s, 5, 30, 20, (1.0, 0.0, 0.5, 1.0))
    T.draw_text(b, s, 5, 30, 20, np.array([255, 0, 128, 255], np.uint8))
    assert np.array_equal(a, b)
    c = np.zeros_like(a)
    d = np.zeros_like(a)
    T.draw_text(c, s, 5, 30, 20, (255, 0, 128, 255), alpha=0.5)
    T.draw_text(d, s, 5, 30, 20, (1.0, 0.0, 0.5, 0.5))
    assert np.array_equal(c, d)
    e = np.zeros_like(a)
    T.draw_text(e, s, 5, 30, 20, (255, 0, 128))
    assert np.array_equal(e, b)


def test_gpu_blend_identical():
    cp = pytest.importorskip('cupy')
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip('no CUDA device')
    mod = cp.RawModule(code=T.CUDA, options=('--std=c++14', '--fmad=false'))
    k = mod.get_function('blend_mask')
    rng = np.random.default_rng(7)
    for s, rot, col in (('Temperature (K)', 0, (0.1, 0.2, 0.3, 0.77)),
                        ('500 hPa\n$T_{2m}$', 35, (250, 3, 128, 200)),
                        ('edge', 90, (0, 0, 0, 255))):
        canvas = rng.integers(0, 256, (120, 160, 4), dtype=np.uint8)
        for x, y in ((80, 60), (-5, 10), (150, 115)):
            tm, x0, y0, _ = T.place(s, x, y, 21.3, rotation=rot, ha='center', va='center')
            cpu = canvas.copy()
            T.blend_mask(cpu, tm.alpha, x0, y0, col)
            r, g, b, a = (np.float32(v) for v in T._rgba(col))
            gc = cp.asarray(canvas)
            gm = cp.asarray(tm.alpha)
            mh, mw = tm.alpha.shape
            k(((mw + 15) // 16, (mh + 15) // 16), (16, 16),
              (gc, np.int32(120), np.int32(160), gm, np.int32(mh), np.int32(mw),
               np.int32(x0), np.int32(y0), r, g, b, a))
            assert np.array_equal(cp.asnumpy(gc), cpu)
