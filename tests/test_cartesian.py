"""Plain axes against matplotlib: where the axes go, the limits autoscaling
picks, the ticks and labels, legends, tight_layout, the picture itself
(pixel by pixel, text aside), the GPU matching the CPU -- and the Agg
behaviours underneath (pixel snapping, line caps, box-filtered dashes)."""
import datetime as dt
import io

import numpy as np
import pytest

import fastplot as fp
import fastplot.pyplot as fplt
from fastplot import _device, _draw
from fastplot._ticker import strip_mathdefault


def _mpl():
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


@pytest.fixture(autouse=True)
def _fresh():
    fp.rcdefaults()
    yield
    fplt.close('all')


X = np.linspace(0, 10, 60)
RNG = np.random.default_rng(3)
DATA = RNG.normal(size=400)
IMG = RNG.random((12, 16))

# each: draws into ax (the same calls for both libraries)
SCENES = {
    'plot': lambda ax: (ax.plot(X, np.sin(X), label='s'), ax.plot(X, 2 * np.cos(X), 'o--')),
    'scatter': lambda ax: ax.scatter(X, X ** 2, c=X, s=12),
    'bar': lambda ax: ax.bar(np.arange(6), [3, 1, 4, 1, 5, 9]),
    'barh_cat': lambda ax: ax.barh(['a', 'b', 'c'], [2, 7, 1]),
    'hist': lambda ax: ax.hist(DATA, bins=20),
    'hist_step': lambda ax: ax.hist(DATA, bins=15, histtype='step', density=True),
    'semilogy': lambda ax: ax.semilogy(X, np.exp(X)),
    'loglog': lambda ax: ax.loglog(X[1:], X[1:] ** 3),
    'fill_between': lambda ax: ax.fill_between(X, np.sin(X), 2 + np.cos(X), alpha=0.4),
    'stackplot': lambda ax: ax.stackplot(X, np.abs(np.sin(X)), np.abs(np.cos(X))),
    'errorbar': lambda ax: ax.errorbar(np.arange(5), [1, 3, 2, 4, 3], yerr=0.5, capsize=3),
    'step_stem': lambda ax: (ax.step(np.arange(6), [1, 3, 2, 5, 4, 6]),
                             ax.stem(np.arange(6), [2, 1, 3, 1, 2, 1])),
    'inverted': lambda ax: (ax.plot(X, X ** 0.5), ax.invert_yaxis()),
    'set_lims': lambda ax: (ax.plot(X, X), ax.set_xlim(2, 7), ax.set_ylim(10, -1)),
    'axlines': lambda ax: (ax.plot(X, X), ax.axhline(3), ax.axvspan(2, 4, alpha=0.3)),
    'imshow': lambda ax: ax.imshow(IMG),
    'dates': lambda ax: ax.plot([dt.datetime(2026, 1, 1) + dt.timedelta(hours=7 * k)
                                 for k in range(30)], np.arange(30)),
    'fill_betweenx_logp': lambda ax: (ax.fill_betweenx(np.linspace(1000, 100, 30),
                                                       np.linspace(20, -60, 30),
                                                       np.linspace(10, -70, 30)),
                                      ax.set_yscale('log'), ax.set_ylim(1050, 100)),
}


def _both(scene, figsize=(6, 4), dpi=100):
    plt = _mpl()
    out = []
    for lib in (plt, fplt):
        fig, ax = lib.subplots(figsize=figsize, dpi=dpi)
        SCENES[scene](ax)
        out.append((fig, ax))
    return out


@pytest.mark.parametrize('scene', sorted(SCENES))
def test_limits_and_ticks(scene):
    """The view limits autoscaling gives, and the major ticks and their
    labels, as matplotlib's."""
    (mfig, max_), (ffig, fax) = _both(scene)
    mfig.canvas.draw()
    ffig.draw(device='cpu')
    np.testing.assert_allclose(fax.get_xlim(), max_.get_xlim(), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(fax.get_ylim(), max_.get_ylim(), rtol=1e-9, atol=1e-12)
    ffig.draw(device='cpu')                             # draw keeps the ticks drawn
    for name in ('x', 'y'):
        maxis = getattr(max_, name + 'axis')
        vmin, vmax = sorted(maxis.get_view_interval())
        mt = [(t.get_loc(), t.label1.get_text()) for t in maxis.get_major_ticks()
              if vmin - 1e-9 * abs(vmax - vmin) <= t.get_loc() <= vmax + 1e-9 * abs(vmax - vmin)]
        locs, labels, _ = fax._ticks_cache[(name, 'major')]
        assert len(locs) == len(mt), (name, locs, mt)
        np.testing.assert_allclose(locs, [m[0] for m in mt], rtol=1e-9, atol=1e-12)
        assert [s.replace('−', '-') for s in labels] == \
            [strip_mathdefault(m[1]).replace('−', '-') for m in mt]


def test_subplot_positions():
    plt = _mpl()
    for nr, nc in ((1, 1), (2, 3), (3, 1)):
        mfig, maxs = plt.subplots(nr, nc, figsize=(9, 6))
        ffig, faxs = fplt.subplots(nr, nc, figsize=(9, 6))
        ffig.draw(device='cpu')
        W, H = ffig._px_size
        for ma, fa in zip(np.ravel(maxs), np.ravel(faxs)):
            b = ma.get_position()
            x0, y0, x1, y1 = fa.box_f
            np.testing.assert_allclose([x0 / W, 1 - y1 / H, x1 / W, 1 - y0 / H],
                                       [b.x0, b.y0, b.x1, b.y1], atol=1e-12)


def test_tight_layout_matches():
    """tight_layout's subplot parameters within a pixel of matplotlib's
    (they rest on text extents)."""
    plt = _mpl()
    pars = []
    for lib in (plt, fplt):
        fig, axs = lib.subplots(2, 2, figsize=(8, 6), dpi=100)
        axs[0, 0].plot(X, 1e4 * np.sin(X))
        axs[0, 0].set_ylabel('a long y label')
        axs[0, 1].set_title('title')
        axs[1, 0].set_xlabel('x')
        m = axs[1, 1].pcolormesh(RNG.random((5, 6)))
        fig.colorbar(m, ax=axs[1, 1])
        fig.suptitle('sup')
        fig.tight_layout()
        sp = fig.subplotpars
        pars.append([getattr(sp, k) if hasattr(sp, k) else sp[k]
                     for k in ('left', 'right', 'bottom', 'top', 'wspace', 'hspace')])
    np.testing.assert_allclose(pars[1][:4], pars[0][:4], atol=1.5 / 600)
    np.testing.assert_allclose(pars[1][4:], pars[0][4:], atol=0.01)


@pytest.mark.parametrize('loc', ['best', 'upper right', 'lower left', 'center', (0.3, 0.2)])
def test_legend_frame(loc):
    plt = _mpl()
    frames = []
    for lib in (plt, fplt):
        fig, ax = lib.subplots(figsize=(6, 4), dpi=100)
        ax.plot(X, np.sin(X), label='sin')
        ax.plot(X, np.cos(X), '--', label='cos')
        ax.scatter(X[::6], np.sin(X[::6]), label='pts')
        ax.bar([2, 4], [0.5, -0.5], label='bars')
        leg = ax.legend(loc=loc, title='T')
        if lib is plt:
            fig.canvas.draw()
            b = leg.get_frame().get_window_extent()
            frames.append((b.x0, 400 - b.y1, b.width, b.height))
        else:
            from fastplot import _legend
            got = []
            orig = _legend.Legend.render

            def spy(self, r, ax=None):
                L = self._layout(r)
                box = self._parent_box(r)
                pad = L['P']['borderaxespad']
                if self.loc == 0:
                    left, top = self._best(r, L['W'], L['H'], box, pad)
                elif isinstance(self.loc, tuple):
                    left = box[0] + self.loc[0] * (box[2] - box[0])
                    top = box[3] - self.loc[1] * (box[3] - box[1]) - L['H']
                else:
                    left, top = self._anchored(self.loc, L['W'], L['H'], box, pad)
                got.append((left, top, L['W'], L['H']))
                return orig(self, r, ax)
            _legend.Legend.render = spy
            try:
                fig.draw(device='cpu')
            finally:
                _legend.Legend.render = orig
            frames.append(got[0])
    np.testing.assert_allclose(frames[1], frames[0], atol=0.01)


SIMILAR = {                         # measured share of pixels > 32 levels off (text included)
    'plot': 0.0122, 'bar': 0.0017, 'hist': 0.0028, 'semilogy': 0.0029, 'fill_between': 0.0043,
    'errorbar': 0.0057, 'imshow': 0.0050, 'stackplot': 0.0036,
}


@pytest.mark.parametrize('scene', sorted(SIMILAR))
def test_pixels_close_to_matplotlib(scene):
    """The picture: the share of pixels more than 32 levels off
    matplotlib's (glyph rasterization, which differs, included)."""
    from PIL import Image
    (mfig, _), (ffig, _) = _both(scene)
    buf = io.BytesIO()
    mfig.savefig(buf, format='png')
    a = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert('RGB')).astype(int)
    b = ffig.draw(device='cpu')[..., :3].astype(int)
    assert a.shape == b.shape
    off = (np.abs(a - b).max(axis=2) > 32).mean()
    assert off < SIMILAR[scene] * 1.5, off           # a margin over what it is today


@pytest.mark.skipif(not _device.gpu_usable(), reason='no usable GPU')
@pytest.mark.parametrize('scene', ['plot', 'imshow', 'fill_between', 'scatter'])
def test_gpu_matches_cpu(scene):
    fig, ax = fplt.subplots(figsize=(5, 4), dpi=100)
    SCENES[scene](ax)
    ax.legend(['x']) if scene == 'plot' else None
    cpu = fig.draw(device='cpu')
    gpu = fig.draw(device='gpu')
    assert np.array_equal(cpu, gpu)


# ------------------------------------------------------- Agg's behaviours

def test_snap_rectilinear_only():
    xy = np.array([[10.2, 5.7], [30.6, 5.7], [0.3, 0.3], [9.7, 7.2]])
    off = np.array([0, 2, 4])
    out = _draw.snap(xy, off, 1.11)                     # odd width: pixel centres
    np.testing.assert_array_equal(out[:2], [[10.5, 6.5], [31.5, 6.5]])
    np.testing.assert_array_equal(out[2:], xy[2:])      # slanted: as it was
    out = _draw.snap(xy[:2], off[:2], 2.08)             # even: pixel edges
    np.testing.assert_array_equal(out, [[10.0, 6.0], [31.0, 6.0]])
    assert _draw.snap(xy, off, 1.0, False) is xy


def _line(cap, x0, x1, w=2.0, dashes=None):
    canvas = np.zeros((20, 40, 4), np.uint8)
    _draw.draw_lines(canvas, np.array([[x0, 10.0], [x1, 10.0]]), np.array([0, 2]),
                     np.array([255, 255, 255, 255], np.uint8), w, dashes=dashes, cap=cap)
    return canvas[..., 3].astype(float) / 255


def test_caps():
    """Butt ends where the line ends, projecting ones half a width on,
    both box-filtered along the line."""
    butt = _line(_draw.BUTT, 5.0, 25.0)
    assert butt[9:11, 5:25].min() == 1.0 and butt[:, :5].max() == 0 and butt[:, 25:].max() == 0
    half = _line(_draw.BUTT, 5.5, 25.0)
    np.testing.assert_allclose(half[9:11, 5], 0.5, atol=1 / 255)
    proj = _line(_draw.PROJECTING, 5.0, 25.0)
    assert proj[9:11, 4].min() == 1.0 and proj[:, 3].max() == 0 and proj[9:11, 25].min() == 1.0


def test_dashes_box_filtered():
    """A dash pattern's coverage: each pixel the dashes' share of it."""
    a = _line(_draw.BUTT, 0.0, 40.0, dashes=(2.5, 1.5))
    row = a[9]
    s = np.arange(40)
    on = lambda u: np.floor(u / 4.0) * 2.5 + np.clip(u - np.floor(u / 4.0) * 4.0, 0, 2.5)  # noqa
    np.testing.assert_allclose(row, on(s + 1.0) - on(s), atol=1.5 / 255)


def test_style_sheet_and_rc():
    with fp.style.context('ggplot'):
        fig, ax = fplt.subplots()
        assert fp.rcParams['axes.facecolor'] == ax.facecolor
    assert fp.rcParams['axes.facecolor'] == 'white'
    with fp.rc_context({'lines.linewidth': 3.0}):
        fig, ax = fplt.subplots()
        line, = ax.plot([0, 1], [0, 1])
        assert line.linewidth == 3.0


def test_categorical_and_legend_order():
    fig, ax = fplt.subplots()
    ax.bar(['x', 'y', 'z'], [1, 2, 3], label='bars')
    ax.plot([0, 2], [1, 3], label='line')
    h, labels = ax.get_legend_handles_labels()
    assert labels == ['line', 'bars']                   # containers after the axes' artists
    fig.draw(device='cpu')
    assert fax_labels(ax) == ['x', 'y', 'z']


def fax_labels(ax):
    return list(ax._ticks_cache[('x', 'major')][1])


def test_log_autoscale_from_zero_baseline():
    """Bars from 0 on a log axis autoscale from the smallest positive value
    (matplotlib's LogLocator.nonsingular reads it from the axis), not from
    1e-300: matplotlib 3.10's limits."""
    import fastplot.pyplot as fplt
    rng = np.random.default_rng(0)
    fig, ax = fplt.subplots()
    ax.hist(rng.normal(size=10000), bins=40)
    ax.set_yscale('log')
    np.testing.assert_allclose(ax.get_ylim(), (0.7184427827662687, 1036.9649718401747),
                               rtol=1e-12)
    fig, ax = fplt.subplots()
    ax.stem(np.arange(10), rng.random(10))
    ax.set_yscale('log')
    lo, hi = ax.get_ylim()
    assert 1e-3 < lo < 1 < hi < 10
    fplt.close('all')
