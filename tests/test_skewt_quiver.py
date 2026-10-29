"""Skew-T (MetPy's SkewT as the reference), quiver (matplotlib's), and the
exact ports underneath: Agg's path simplification, hist's counts."""
import io

import numpy as np
import pytest

import fastplot as fp
import fastplot.pyplot as fplt
from fastplot import _draw
from fastplot._plots import _data_range, _histogram


@pytest.fixture(autouse=True)
def _fresh():
    fp.rcdefaults()
    yield
    fplt.close('all')


def _mpl():
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _pixels_off(mfig, ffig):
    from PIL import Image
    buf = io.BytesIO()
    mfig.savefig(buf, format='png')
    a = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert('RGB')).astype(int)
    b = ffig.draw(device='cpu')[..., :3].astype(int)
    assert a.shape == b.shape
    return (np.abs(a - b).max(axis=2) > 32).mean()


# ------------------------------------------------------------- the ports

def test_simplify_is_matplotlibs():
    """Vertex for vertex what Path.cleaned(simplify=True) gives."""
    Path = pytest.importorskip('matplotlib.path').Path
    rng = np.random.default_rng(1)
    for trial in range(120):
        n = int(rng.integers(2, 2000))
        t = np.linspace(0, 1, n)
        y = [rng.normal(size=n).cumsum(), np.sin(20 * t) * 50 + rng.normal(size=n) * 0.05,
             np.round(rng.normal(size=n) * 3)][trial % 3]
        xy = np.c_[t * rng.uniform(10, 2000), y * rng.uniform(0.1, 20)]
        if trial % 4 == 0:
            xy[rng.integers(0, n, 3)] = np.nan
        c = Path(xy).cleaned(simplify=True, remove_nans=True)
        got, offs = _draw.simplify(xy, np.array([0, n]))
        kinds = np.full(len(got), 2)
        kinds[offs[:-1]] = 1
        np.testing.assert_array_equal(got, c.vertices[:-1])
        np.testing.assert_array_equal(kinds, c.codes[:-1])


def test_histogram_counts_exact():
    rng = np.random.default_rng(5)
    for trial in range(100):
        v = rng.normal(size=rng.integers(0, 4000)) * rng.uniform(0.1, 1e6)
        if v.size and trial % 3 == 0:
            v[rng.integers(0, v.size, 10)] = np.nan
        r = _data_range([v])
        for bins in (1, 7, 100):
            e = np.histogram_bin_edges(v[np.isfinite(v)], bins, r)
            np.testing.assert_array_equal(_histogram(v, e),
                                          np.histogram(v[np.isfinite(v)], e)[0])
    w = np.repeat(np.arange(11.0), 3)                   # values on the edges
    e = np.linspace(0, 10, 6)
    np.testing.assert_array_equal(_histogram(w, e), np.histogram(w, e)[0])


# ---------------------------------------------------------------- Skew-T

P = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100], float)
T = np.array([28, 23, 18, 7.5, 0, -8.5, -19.5, -35, -45, -55, -61, -63], float)
TD = np.array([22, 19, 13, -4, -16, -28, -40, -52, -60, -68, -75, -80], float)


def test_thermo_is_metpys():
    calc = pytest.importorskip('metpy.calc')
    units = pytest.importorskip('metpy.units').units
    from fastplot import skewt
    t = np.linspace(200, 320, 40)
    np.testing.assert_allclose(skewt.saturation_vapor_pressure(t),
                               calc.saturation_vapor_pressure(t * units.K).m_as('hPa'),
                               rtol=1e-12)
    p = np.linspace(1050, 100, 40)
    t0 = np.arange(-40, 51, 10.0)
    ref = calc.moist_lapse(p * units.hPa, t0 * units.degC, 1000 * units.hPa).m_as('K')
    np.testing.assert_allclose(skewt.moist_lapse(p, t0 + skewt.ZERO_C), ref, atol=5e-4)
    np.testing.assert_allclose(skewt.dry_lapse(p, 300.0),
                               calc.dry_lapse(p * units.hPa, 300 * units.K,
                                              1000 * units.hPa).m_as('K'), rtol=1e-12)


def test_skewt_like_metpy():
    plt = _mpl()
    MSkewT = pytest.importorskip('metpy.plots').SkewT
    from fastplot.skewt import SkewT
    figs = []
    for lib, cls in ((plt, MSkewT), (fplt, SkewT)):
        fig = lib.figure(figsize=(7, 7), dpi=100)
        skew = cls(fig, rotation=45)
        skew.plot(P, T, 'r')
        skew.plot(P, TD, 'g')
        skew.plot_barbs(P, np.linspace(0, 50, P.size), np.linspace(5, -5, P.size))
        skew.shade_cape(P, T, T + 3)
        skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)
        skew.plot_dry_adiabats()
        skew.plot_moist_adiabats()
        skew.plot_mixing_lines()
        skew.ax.set_xlim(-40, 50)
        figs.append(fig)
    assert _pixels_off(*figs) < 0.015                   # today ~1.0%, text included


def test_skewed_ticks_each_edge():
    """MetPy's SkewXTick: the bottom ticks where the bottom's x range
    reaches, the grid over both edges' ranges."""
    from fastplot.skewt import SkewT
    skew = SkewT(rotation=45)
    skew.ax.figure.draw(device='cpu')
    ax = skew.ax
    lo = sorted(ax._edge_xlim(0))
    hi = sorted(ax._edge_xlim(1))
    assert lo == pytest.approx([-40, 50])
    assert hi[0] < lo[0] and hi[1] < lo[1]              # the top reaches further left
    locs = ax._ticks_cache[('x', 'major')][0]
    assert locs.min() <= hi[0] + 10 and locs.max() >= 50


# ---------------------------------------------------------------- quiver

@pytest.mark.parametrize('kw', [dict(), dict(pivot='mid', scale=20),
                                dict(angles='xy', scale_units='xy', scale=2, width=0.005),
                                dict(units='dots', width=2, headwidth=4)])
def test_quiver_like_matplotlib(kw):
    plt = _mpl()
    x, y = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-1.5, 1.5, 11))
    figs = []
    for lib in (plt, fplt):
        fig, ax = lib.subplots(figsize=(5, 4), dpi=100)
        ax.quiver(x, y, -y, 0.7 * x, **kw)
        figs.append(fig)
    assert _pixels_off(*figs) < 0.01


# ------------------------------------------------------------ streamplot

Y, X = np.mgrid[-3:3:100j, -3:3:120j]
STREAM_CASES = {
    'vortex': (-1 - X ** 2 + Y, 1 + X - Y ** 2, {}),
    'masked': (np.where(X ** 2 + Y ** 2 < 1, np.nan, -Y), X, dict(density=(1.5, 0.8))),
    'forward': (np.cos(X) * 3, np.sin(Y), dict(integration_direction='forward',
                                               minlength=0.3)),
    'unbroken': (-Y, X, dict(broken_streamlines=False, density=0.6)),
    'starts': (-Y, X + 0.3 * Y, dict(start_points=[[0.5, 0.2], [-2, 1], [2.9, -2.9]])),
}


@pytest.mark.parametrize('case', sorted(STREAM_CASES))
def test_streamlines_are_matplotlibs(case):
    """The trajectories bit for bit (matplotlib's integrator, in numba)."""
    plt = _mpl()
    from fastplot._streamplot import trajectories
    u, v, kw = STREAM_CASES[case]
    fig, ax = plt.subplots()
    ref = [p.vertices for p in ax.streamplot(X, Y, u, v, **kw).lines.get_paths()]
    data, _, offs, _ = trajectories(X, Y, u, v, **kw)
    ours = [data[offs[k]:offs[k + 1]] for k in range(offs.size - 1)]
    assert len(ours) == len(ref)
    for a, b in zip(ours, ref):
        np.testing.assert_array_equal(a, b)


def test_streamplot_picture():
    plt = _mpl()
    u, v, _ = STREAM_CASES['vortex']
    figs = []
    for lib in (plt, fplt):
        fig, axs = lib.subplots(1, 2, figsize=(8, 4), dpi=100)
        axs[0].streamplot(X, Y, u, v)
        axs[1].streamplot(X, Y, u, v, color=np.hypot(u, v), linewidth=2 * np.hypot(u, v) / 17)
        figs.append(fig)
    assert _pixels_off(*figs) < 0.008


# ------------------------------------------------------------ hodograph

def test_hodograph_like_metpy():
    plt = _mpl()
    MHodo = pytest.importorskip('metpy.plots').Hodograph
    from fastplot.skewt import Hodograph
    z = np.linspace(0, 10000, 30)
    u, v = 30 * np.sin(z / 4000) + 5, 25 * np.cos(z / 5000) - 10
    figs = []
    for lib, cls in ((plt, MHodo), (fplt, Hodograph)):
        fig, axs = lib.subplots(1, 2, figsize=(10, 5), dpi=100)
        h = cls(axs[0], component_range=60)
        h.add_grid(increment=20)
        h.plot_colormapped(u, v, np.hypot(u, v))
        h.wind_vectors(u[::6], v[::6])
        h2 = cls(axs[1], component_range=50)
        h2.add_grid(increment=10, linewidth=0.5)
        h2.plot_colormapped(u, v, z, intervals=np.array([0, 1000, 3000, 6000, 10000.]),
                            colors=['magenta', 'red', 'green', 'blue'])
        figs.append(fig)
    assert _pixels_off(*figs) < 0.012


# ---------------------------------------------------------------- hatches

def test_hatches_like_matplotlib():
    """Every pattern, on bars, a patch and contourf bands: Agg's tile, its
    phase and colour."""
    plt = _mpl()
    import matplotlib.patches as mpatches
    import fastplot.patches as fpatches
    y, x = np.mgrid[-2:2:120j, -3:3:160j]
    z = np.exp(-x ** 2 - y ** 2) - 0.5 * np.exp(-(x - 1.5) ** 2 - (y + 0.5) ** 2)
    figs = []
    for lib, P in ((plt, mpatches), (fplt, fpatches)):
        fig, axs = lib.subplots(1, 3, figsize=(12, 4), dpi=100)
        for k, h in enumerate(['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']):
            axs[0].bar(k, 1 + k % 3, color='w', edgecolor='C%d' % k, hatch=h)
        axs[1].add_patch(P.Circle((0.5, 0.5), 0.3, facecolor='none', edgecolor='k',
                                  hatch='xx'))
        axs[1].add_patch(P.Rectangle((0.05, 0.05), 0.3, 0.3, fc='yellow', ec='none',
                                     hatch='//'))
        axs[2].contourf(x, y, z, levels=[-0.5, -0.2, 0.2, 0.6, 1.0],
                        hatches=['////', None, '..', 'xx'], cmap='coolwarm', extend='both',
                        alpha=0.6)
        figs.append(fig)
    assert _pixels_off(*figs) < 0.018


def test_thin_dashes_scale_down():
    """matplotlib scales a dash pattern by the line width, below 1 too."""
    from fastplot._style import _dashes
    assert _dashes('--', 0.4) == pytest.approx((3.7 * 0.4, 1.6 * 0.4))
