"""Details of matplotlib's output that are easy to lose: contour line order,
how Agg fills leaning meshes, gouraud outlines, mathtext's boxes."""
import numpy as np
import pytest

import fastplot.pyplot as fplt
from fastplot import _raster, _text

matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')
import matplotlib.pyplot as mplt                                         # noqa: E402

needs_311 = pytest.mark.skipif(tuple(int(v) for v in matplotlib.__version__.split('.')[:2]) < (3, 11),
                               reason='matplotlib 3.11 lays text out differently')


def _smooth(ny=40, nx=50, seed=1):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:1:ny * 1j, 0:1:nx * 1j]
    z = np.zeros((ny, nx))
    for _ in range(8):
        a, b, c, d = rng.uniform(1, 9), rng.uniform(1, 9), *rng.uniform(0, 6, 2)
        z += np.sin(a * x + c) * np.cos(b * y + d)
    return z


def _pixels(fig):
    if hasattr(fig, 'draw') and fig.__class__.__module__.startswith('fastplot'):
        return np.asarray(fig.draw(), np.int16)[..., :3]
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba(), np.int16)[..., :3]


def _bad(a, b):
    assert a.shape == b.shape
    return (np.abs(a - b).max(axis=2) > 64).mean()


def test_contour_lines_in_matplotlibs_order_and_direction():
    z = _smooth()
    a = mplt.contour(z, levels=10)
    b = fplt.contour(z, levels=10)
    try:
        for la, lb in zip(a.allsegs, b.allsegs):
            assert len(la) == len(lb)
            for p, q in zip(la, lb):
                assert p.shape == q.shape
                np.testing.assert_allclose(p, q, atol=1e-4)
    finally:
        mplt.close('all')
        fplt.close('all')


def test_contour_paths_one_a_level():
    cs = fplt.contour(_smooth(), levels=[-1.0, 0.0, 1e9])
    paths = cs.get_paths()
    assert len(paths) == 3 and len(paths[2].vertices) == 0
    assert cs.allsegs[2][0].shape == (0, 2) and cs.allkinds[2] == [None]
    fplt.close('all')


def _lean_mesh(plt, shading):
    fig, ax = plt.subplots(figsize=(4, 3.2), dpi=100)
    i, j = np.meshgrid(np.arange(15.0), np.arange(12.0))
    x, y = i * 0.93 - j * 0.33, i * 0.33 + j * 0.93
    shape = x.shape if shading == 'nearest' else (x.shape[0] - 1, x.shape[1] - 1)
    z = np.random.default_rng(1).random(shape)
    ax.pcolormesh(x, y, np.ma.masked_greater(z, 0.8), shading=shading, cmap='viridis')
    return fig


@pytest.mark.parametrize('shading', ['flat', 'nearest'])
def test_leaning_mesh_filled_as_agg_fills_it(shading):
    bad = _bad(_pixels(_lean_mesh(mplt, shading)), _pixels(_lean_mesh(fplt, shading)))
    mplt.close('all')
    fplt.close('all')
    assert bad < 2e-3


def test_only_leaning_cells_take_the_exact_path():
    j, i = np.meshgrid(np.arange(30.0), np.arange(20.0))
    shape = (100, 100)
    assert not _raster.wants_touched(j * 2, i * 2, _raster.FLAT, 0.0, 551, shape)
    assert _raster.wants_touched(j * 2 + i * 0.5, i * 2, _raster.FLAT, 0.0, 551, shape)
    assert not _raster.wants_touched(j * 2 + i * 0.5, i * 2, _raster.BILINEAR, 0.0, 551, shape)
    assert not _raster.wants_touched(j * 2 + i * 0.5, i * 2, _raster.FLAT, 360.0, 551, shape)


@pytest.mark.parametrize('dx, dy, want', [
    (0.0, 0.0, True),                                   # all of it in view
    (-20.0, -20.0, False), (-20.0, 0.0, True),          # its edge past the view, or a side of it in
    (-200.0, 0.0, False), (0.0, -200.0, False),         # beyond the view
    (-58.5, -38.5, True)])                              # only a half cell past the last nodes
def test_upright_nearest_cells_take_it_where_the_grids_edge_shows(dx, dy, want):
    j, i = np.meshgrid(np.arange(30.0), np.arange(20.0))
    x, y = 2.0 * j + 0.5 + dx, 2.0 * i + 0.5 + dy
    assert _raster.wants_touched(x, y, _raster.NEAREST, 0.0, 600, (5, 5)) is want


@pytest.mark.parametrize('ny, nx', [(106, 180), (40, 60)])
@pytest.mark.parametrize('holes', [False, True])
def test_grid_on_its_own_projection_reaches_its_outer_cells(ny, nx, holes):
    """shading='nearest' has half a cell past the last nodes, which show where
    the grid ends inside the map."""
    ccrs = pytest.importorskip('cartopy.crs')
    import fastplot.crs as fcrs

    def draw(plt, crs):
        p = crs.LambertConformal(-97.5, 38.5, standard_parallels=(38.5, 38.5))
        fig = plt.figure(figsize=(8, 5), dpi=100)
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.96], projection=p)
        ax.set_extent([-135, -60, 15, 58], crs=crs.PlateCarree())
        ax.pcolormesh(lon, lat, z, transform=crs.PlateCarree(), cmap='turbo', shading='nearest')
        ax.spines['geo'].set_visible(False)
        fig.canvas.draw() if plt is mplt else None
        px = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.int16) if plt is mplt \
            else np.asarray(fig.draw())[..., :3].astype(np.int16)
        plt.close(fig)
        return px

    x, y = np.meshgrid(np.linspace(-2.2e6, 2.2e6, nx), np.linspace(-1.5e6, 1.3e6, ny))
    q = ccrs.PlateCarree().transform_points(
        ccrs.LambertConformal(-97.5, 38.5, standard_parallels=(38.5, 38.5)), x, y)
    lon, lat = q[..., 0], q[..., 1]
    z = np.sin(lon / 7.0) + np.cos(lat / 5.0)
    if holes:
        z[np.sin(lon * 0.3) + np.cos(lat * 0.4) > 0.6] = np.nan
    a, b = draw(mplt, ccrs), draw(fplt, fcrs)
    assert (np.abs(a - b).max(axis=2) > 8).sum() == 0


def _gouraud(plt):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    x = np.array([0.1, 0.9, 0.5, 0.2, 0.8])
    y = np.array([0.1, 0.2, 0.9, 0.6, 0.65])
    tri = np.array([[0, 1, 4], [0, 4, 3], [3, 4, 2]])
    ax.tripcolor(x, y, tri, np.array([0.0, 1.0, 0.5, 0.2, 0.7]), shading='gouraud')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig


def test_gouraud_outline():
    bad = _bad(_pixels(_gouraud(mplt)), _pixels(_gouraud(fplt)))
    mplt.close('all')
    fplt.close('all')
    assert bad < 2e-3


@needs_311
@pytest.mark.parametrize('s', [
    r'$\left[x^2\right]$', r'$\left|\frac{a}{b}\right|$', r'$\left.\frac{a}{b}\right|_0$',
    r'$\sqrt{x}$', r'$\mathbb{R}\mathcal{L}\mathfrak{g}\mathscr{F}$',
    r'$\mathtt{code}+\mathcal{L}_x$', r'$\sum_{i=0}^{n} x_i$', r'$\lim_{x \to 0} y$',
    r'$\log_{10}(x)$'])
def test_mathtext_boxes(s):
    from matplotlib import _mathtext as M
    from matplotlib.font_manager import FontProperties
    from matplotlib.ft2font import LoadFlags
    a = M.Parser().parse(s, M.DejaVuSansFonts(FontProperties(size=13), LoadFlags.NO_HINTING),
                         13, 100)
    size = 13 * 100 / 72
    fnt = _text.font(size)
    b = _text._HList(_text._MathParser(s.strip('$'), size, fnt).expr('normal'), user=fnt)
    assert (b.width, b.height, b.depth) == pytest.approx((a.width, a.height, a.depth), abs=0.01)


def test_bundled_families():
    assert _text._font_path(False, False, 'DejaVu Serif').endswith('DejaVuSerif.ttf')
    assert _text._font_path(True, True, 'DejaVu Sans Mono').endswith('DejaVuSansMono-BoldOblique.ttf')
