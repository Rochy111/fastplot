"""Text drawn as matplotlib's Agg drew it before 3.9 (fastplot._classic),
when rcParams['text.hinting_factor'] is set: the same pixels as such a
matplotlib draws (where one is installed), the turned lines' masks
blending as Agg's resampled pixels blend, and fastplot's own text
untouched otherwise."""
import warnings

import numpy as np
import pytest

import fastplot
from fastplot import _agg, _classic, _text

CLASSIC = {'text.hinting_factor': 8, 'text.hinting': 'force_autohint', 'text.kerning_factor': 0}


def _old_matplotlib():
    try:
        import matplotlib
        from matplotlib import ft2font                  # noqa: F401
    except Exception:                                   # noqa: BLE001
        return None
    v = tuple(int(p) for p in matplotlib.__version__.split('.')[:2])
    return matplotlib if v < (3, 9) else None


OLD = _old_matplotlib()
needs_old = pytest.mark.skipif(OLD is None, reason='needs matplotlib < 3.9 (its ft2font, its Agg)')
needs_ft2font = pytest.mark.skipif(_classic._ft2font() is None, reason='needs matplotlib')


@pytest.fixture
def classic():
    with fastplot.rc_context(CLASSIC):
        yield


def test_off_by_default():
    assert fastplot.rcParams['text.hinting_factor'] is None
    assert _classic.mode() is None


@needs_ft2font
def test_on_when_asked(classic):
    m = _classic.mode()
    assert m is not None and m[1] == 8 and m[2] == 0


def test_without_matplotlib_warns_and_draws_as_fastplot(monkeypatch):
    monkeypatch.setattr(_classic, '_FT', [None])
    monkeypatch.setattr(_classic, '_WARNED', [])
    with fastplot.rc_context(CLASSIC):
        with pytest.warns(UserWarning, match='matplotlib'):
            assert _classic.mode() is None
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert _classic.mode() is None              # (once)


def test_lut_rows_sum_to_one():
    """(normalize() makes each sum 16384; its mirroring after can leave one
    off by one, as Agg's own does)"""
    w, diameter, start = _classic.lut()
    assert diameter == 6 and start == -2
    for i in range(256):
        assert abs(sum(int(w[j * 256 + i]) for j in range(diameter)) - (1 << 14)) <= 1


@pytest.mark.parametrize('ca', [255, 254, 200, 153, 128, 64, 1])
def test_turned_masks_blend_as_agg_blends_resampled_pixels(ca):
    rng = np.random.default_rng(ca)
    rec = rng.integers(0, 256, (17, 23, 2)).astype(np.uint8)
    rec[..., 1][rng.random((17, 23)) < 0.6] = 255       # (mostly solid spans)
    base = rng.integers(0, 256, (17, 23, 4)).astype(np.uint8)
    base[..., 3][rng.random((17, 23)) < 0.7] = 255
    want, got = base.copy(), base.copy()
    r, g, b = 30, 120, 210
    for i in range(17):
        for j in range(23):
            a = (ca * int(rec[i, j, 0])) >> 8           # font_to_rgba, then the blend
            if a:
                _agg._blend(want, i, j, r, g, b, a, int(rec[i, j, 1]))
    _text.blend_mask(got, _classic.alpha_mask(rec, ca), 0, 0, np.array([r, g, b, ca], np.uint8))
    assert np.array_equal(got, want)


@needs_ft2font
def test_turned_line_drawn(classic):
    fig = fastplot.figure(figsize=(3, 2), dpi=80)
    fig.text(0.5, 0.5, '1008 hPa', rotation=33, ha='center', va='center')
    a = fig.draw()
    assert (a[..., :3] < 128).any()


def _both(build, size=(6, 4), dpi=100):
    import matplotlib.pyplot as plt
    out = []
    for lib in (plt, fastplot):
        fig = lib.figure(figsize=size, dpi=dpi)
        build(fig)
        fig.canvas.draw()
        out.append(np.asarray(fig.canvas.buffer_rgba()).copy())
        lib.close(fig)
    return out


CASES = [(0, 'FORECAST', 13, 'bold', 'left', 'baseline'),
         (0, 'Surface Temperature (°F), MSLP (mb)\nValid 14 Mar 2025 18:00 UTC', 13, 'bold',
          'left', 'top'),
         (90, 'Wind (kt)', 10, 'normal', 'center', 'center'),
         (270, '1008', 8, 'normal', 'right', 'bottom'),
         (180, 'two lines\nof text', 9, 'bold', 'center', 'center_baseline'),
         (33.5, '1004', 8, 'normal', 'center', 'center'),
         (-20, 'Valid 2025-03-14', 14.5, 'normal', 'right', 'top')]


@needs_old
@pytest.mark.parametrize('angle,s,size,weight,ha,va', CASES)
def test_fig_text_as_matplotlib_before_3_9(classic, angle, s, size, weight, ha, va):
    OLD.use('Agg')

    def build(fig):
        fig.text(0.45, 0.55, s, fontsize=size, weight=weight, ha=ha, va=va, rotation=angle,
                 color=(0.1, 0.25, 0.6, 0.7))
    a, b = _both(build)
    assert np.array_equal(a, b)


@needs_old
def test_a_plots_text_as_matplotlib_before_3_9(classic):
    OLD.use('Agg')

    def build(fig):
        ax = fig.add_subplot(111)
        ax.plot(np.arange(10), np.arange(10) ** 1.5)
        ax.set_title('Bold title', fontweight='bold')
        ax.set_xlabel('time (h)')
        ax.set_ylabel('value (units)')
    a, b = _both(build)
    assert np.array_equal(a, b)


@needs_ft2font
def test_extent_follows_the_classic_layout(classic):
    """A wider header (8x-hinted advances) and its two lines 1.2 apart."""
    from fastplot._figure import Figure                 # noqa: F401
    s = 'Surface Temperature (°F), MSLP (mb), 10m AGL Wind (kt)'
    path = _text._font_path(True, False, None)
    m = _classic.mode()
    w, h, d = _classic.metrics(path, 13.0, 125.0, m, s)
    with fastplot.rc_context({'text.hinting_factor': None}):
        e = _text.text_extent(s, 0, 0, 13 * 125 / 72, weight='bold')
    e2 = _text.text_extent(s, 0, 0, 13 * 125 / 72, weight='bold', pt_dpi=(13, 125))
    assert e2[2] - e2[0] == pytest.approx(w)
    assert e2 != e


@needs_ft2font
def test_a_text_drawn_again_reuses_its_mask(classic):
    """A text placed again where it was (a frame's tick labels and header)
    is the mask it was, and the next frame's pixels are the first's; a
    change of hinting is a new one."""
    kw = dict(ha='center', va='top', rotation=90, pt_dpi=(9.0, 125.0), canvas_wh=(800, 600))
    a = _text.place('850 hPa', 300.4, 200.2, 15.6, **kw)
    b = _text.place('850 hPa', 300.4, 200.2, 15.6, **kw)
    assert a is b
    with fastplot.rc_context({'text.hinting': 'no_hinting'}):
        c = _text.place('850 hPa', 300.4, 200.2, 15.6, **kw)
    assert c is not a

    def frame():
        fig = fastplot.figure(figsize=(3, 2), dpi=100)
        fig.text(0.5, 0.5, 'Valid 12Z', rotation=30, ha='center')
        fig.text(0.1, 0.9, 'Header', fontsize=13, weight='bold')
        return np.array(fig.canvas.buffer_rgba())
    assert np.array_equal(frame(), frame())
