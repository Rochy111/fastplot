"""subplot_mosaic, quiverkey and the triangular-grid plots, against
matplotlib's own."""
import io

import numpy as np
import pytest

import fastplot as fp
import fastplot.pyplot as fplt


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


@pytest.mark.parametrize('mosaic, kw', [
    ('AB;CC', {}),
    ('\n    AAB\n    C.B\n    CDD\n    ', dict(width_ratios=[1, 2, 1])),
    ([['a', [['x', 'y'], ['z', 'z']]], ['b', 'b']], {}),
    ('AB;CD', dict(sharex=True, sharey=True, height_ratios=[2, 1])),
])
def test_subplot_mosaic_layout(mosaic, kw):
    """matplotlib's axes, in its order, where it puts them."""
    plt = _mpl()
    mfig, mout = plt.subplot_mosaic(mosaic, figsize=(8, 6), **kw)
    ffig, fout = fplt.subplot_mosaic(mosaic, figsize=(8, 6), **kw)
    assert list(fout) == list(mout)
    ffig.draw(device='cpu')
    W, H = ffig._px_size
    for k, ax in fout.items():
        x0, y0, x1, y1 = ax.box_f
        np.testing.assert_allclose([x0 / W, 1 - y1 / H, x1 / W, 1 - y0 / H],
                                   mout[k].get_position().extents, atol=1e-9)


def test_subplot_mosaic_errors():
    with pytest.raises(ValueError):
        fplt.subplot_mosaic('AB;BA')                    # not a rectangle


def test_quiverkey_like_matplotlib():
    plt = _mpl()
    x, y = np.meshgrid(np.linspace(-2, 2, 12), np.linspace(-1.5, 1.5, 9))
    figs = []
    for lib in (plt, fplt):
        fig, axs = lib.subplots(1, 2, figsize=(10, 4), dpi=100)
        q = axs[0].quiver(x, y, -10 * y, 7 * x)
        axs[0].quiverkey(q, 0.9, 1.05, 10, '10 m/s', labelpos='E')
        axs[0].quiverkey(q, 0.1, 1.05, 20, '20 m/s', labelpos='W', color='r')
        q2 = axs[1].quiver(x, y, -10 * y, 7 * x, scale=200, width=0.006, color='C0')
        axs[1].quiverkey(q2, 0.0, -1.2, 15, '15 kt', labelpos='N', coordinates='data')
        axs[1].quiverkey(q2, 0.8, 0.5, 15, 'S', labelpos='S', coordinates='figure', angle=30)
        figs.append(fig)
    assert _pixels_off(*figs) < 0.007                   # today ~0.47%, text included


# -------------------------------------------------------------- tri

def _canon(t):
    t = np.sort(t, axis=1)
    return t[np.lexsort(t.T[::-1])]


@pytest.mark.parametrize('kind', ['random', 'grid', 'grid2', 'hex'])
def test_triangulation_is_matplotlibs(kind):
    """The same triangles as matplotlib's Qhull call -- a regular grid's
    ties included (through scipy, called as matplotlib calls Qhull)."""
    mtri = pytest.importorskip('matplotlib.tri')
    pytest.importorskip('scipy')
    from fastplot.tri import Triangulation
    rng = np.random.default_rng(0)
    x, y = {'random': rng.random((2, 300)),
            'grid': np.meshgrid(np.arange(20.), np.arange(15.)),
            'grid2': np.meshgrid(np.linspace(-3, 3, 31), np.linspace(0, 2, 9)),
            'hex': (np.r_[np.arange(10.), np.arange(10.) + .5],
                    np.r_[np.zeros(10), np.ones(10) * .866])}[kind]
    x, y = np.ravel(x).astype(float), np.ravel(y).astype(float)
    np.testing.assert_array_equal(_canon(Triangulation(x, y).triangles),
                                  _canon(mtri.Triangulation(x, y).triangles))


def test_fallback_delaunay():
    """Without scipy: Bowyer-Watson, Qhull's triangles for points in
    general position."""
    pytest.importorskip('scipy')
    from fastplot._tri import _bowyer_watson, delaunay
    rng = np.random.default_rng(2)
    for n in (5, 50, 800):
        x, y = rng.normal(size=(2, n)) * rng.uniform(0.01, 100)
        np.testing.assert_array_equal(_canon(_bowyer_watson(x - x.mean(), y - y.mean())),
                                      _canon(delaunay(x, y)))


def test_tri_plots_like_matplotlib():
    plt = _mpl()
    import matplotlib.tri as mtri
    import fastplot.tri as ftri
    rng = np.random.default_rng(4)
    x, y = rng.uniform(-3, 3, 400), rng.uniform(-2, 2, 400)
    z = np.exp(-x ** 2 - y ** 2) - 0.6 * np.exp(-(x - 1.5) ** 2 - (y + 0.5) ** 2)
    figs = []
    for lib, tri in ((plt, mtri), (fplt, ftri)):
        t = tri.Triangulation(x, y)
        fig, axs = lib.subplots(2, 2, figsize=(10, 8), dpi=100)
        cf = axs[0, 0].tricontourf(t, z, levels=10, cmap='RdBu_r')
        fig.colorbar(cf, ax=axs[0, 0])
        axs[0, 1].tricontour(t, z, levels=8, colors='k', linewidths=0.8)
        axs[1, 0].tripcolor(t, z, cmap='viridis')
        axs[1, 1].tripcolor(t, z, cmap='magma')
        axs[1, 1].triplot(t, 'w-', lw=0.3)
        figs.append(fig)
    assert _pixels_off(*figs) < 0.02


def test_tricontour_lines_join():
    """A closed level makes one closed polyline (through shared edges)."""
    from fastplot._tri import contour_lines
    from fastplot.tri import Triangulation
    x, y = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21))
    t = Triangulation(x.ravel(), y.ravel())
    lx, ly, offs, lof = contour_lines(t, (x ** 2 + y ** 2).ravel(), [0.25])
    assert offs.size == 2                               # one line
    assert (lx[0], ly[0]) == (lx[-1], ly[-1])           # closed
    np.testing.assert_allclose(np.hypot(lx, ly), 0.5, atol=0.02)
