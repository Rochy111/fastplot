"""pcolormesh cells as Agg fills a QuadMesh: a pixel whose centre lies on
a cell edge goes to the cell before it on the canvas (left, or above),
with the edges made and placed by matplotlib's (and cartopy's) own
arithmetic -- the coordinates' dtype, apply_aspect's corners -- so the
edges that land on pixel centres are the ones that do in matplotlib."""
import numpy as np
import pytest

import fastplot.pyplot as fplt
from fastplot import _raster
from fastplot._fields import _Grid

matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')
import matplotlib.pyplot as mplt                                         # noqa: E402


def _red(fig, plt):
    if plt is mplt:
        fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., 0].astype(np.int16).copy()
    plt.close('all')
    return a


# ------------------------------------------------------------- the table

def test_mesh_table_gives_a_tied_pixel_to_the_cell_before_it():
    # cells (0.5, 2.5] and (2.5, 4.5]: centres 0.5 .. 5.5
    t = _raster.mesh_table([0.5, 2.5, 4.5], 6, 0.5)
    assert t[:, 3].tolist() == [0, 1, 1, 1, 1, 0]
    assert t[1:5, 0].tolist() == [0, 0, 1, 1]
    # the same edges the other way round: cell 0 is (2.5, 4.5] on the canvas
    t = _raster.mesh_table([4.5, 2.5, 0.5], 6, 0.5)
    assert t[:, 3].tolist() == [0, 1, 1, 1, 1, 0]
    assert t[1:5, 0].tolist() == [1, 1, 0, 0]


def test_mesh_table_refuses_edges_it_cannot_place():
    assert _raster.mesh_table([0.0, np.nan, 2.0], 4) is None
    assert _raster.mesh_table([0.0, 3.0, 2.0], 4) is None


# ------------------------------------------------------------- the edges

def test_nearest_edges_in_the_coordinates_dtype():
    """matplotlib's _interp_grid works in the coordinates' dtype: float32
    centres give float32 mid-points, not float64 ones."""
    c = np.sort(np.random.default_rng(0).random(500) * 100).astype(np.float32)
    g = _Grid(c, np.arange(3.0), (3, c.size), 'data')
    cx, _ = g.mesh_corners()
    d = np.diff(c) * 0.5
    want = np.hstack((c[:1] - d[:1], c[:-1] + d, c[-1:] + d[-1:])).astype(np.float64)
    assert np.array_equal(cx, want)
    c64 = c.astype(np.float64)
    d64 = np.diff(c64) * 0.5
    assert not np.array_equal(cx[1:-1], c64[:-1] + d64)        # (float64 would differ)


def test_map_edges_as_cartopy_makes_them():
    """cartopy's _wrap_args: each step taken the short way round the source
    crs' x range, then halved, in the coordinates' dtype."""
    import fastplot.crs as fcrs
    c = np.array([178.0, 179.5, -179.0], np.float32)
    g = _Grid(c, np.arange(2.0), (2, 3), fcrs.PlateCarree())
    cx, _ = g.mesh_corners(360.0)
    d = np.diff(c)
    d = (d + 180.0) % 360.0 - 180.0
    d = d / 2
    want = np.hstack((c[:1] - d[:1], c[:-1] + d, c[-1:] + d[-1:])).astype(np.float64)
    assert np.array_equal(cx, want)


def test_one_centre_is_a_cell_of_no_width():
    g = _Grid(np.array([3.0]), np.arange(4.0), (4, 1), 'data')
    cx, cy = g.mesh_corners()
    assert cx.tolist() == [3.0, 3.0] and cy.size == 5


# -------------------------------------------- pixels, against matplotlib

def _tie_mesh(plt, axis, inv, shading, delta=0.0, even=False):
    """Two cells (even: many) whose edge lies at data 100.5 + delta, on
    axes mapping data 0..200 to 200 px: the edge on pixel 100's centre."""
    e = 100.5 + delta
    if even:
        n = 402
        step = e / 201.0
        coords = np.arange(n + 1) * step if shading == 'flat' else (np.arange(n) + 0.5) * step
        z1 = (np.arange(n) >= 201) + 1.0
    elif shading == 'flat':
        coords, z1 = np.array([0.0, e, 200.0]), np.array([1.0, 2.0])
    else:
        coords, z1 = np.array([e - 50.0, e + 50.0]), np.array([1.0, 2.0])
    other = np.array([0.0, 200.0]) if shading == 'flat' else np.array([50.0, 150.0])
    k = other.size - (1 if shading == 'flat' else 0)
    Z = np.tile(z1, (k, 1)) if axis == 'x' else np.tile(z1[:, None], (1, k))
    X, Y = (coords, other) if axis == 'x' else (other, coords)
    fig = plt.figure(figsize=(2, 2), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.pcolormesh(X, Y, Z, cmap='gray', vmin=0, vmax=2, shading=shading)
    lims = (200.0, 0.0) if inv else (0.0, 200.0)
    ax.set_xlim(*(lims if axis == 'x' else (0.0, 200.0)))
    ax.set_ylim(*(lims if axis == 'y' else (0.0, 200.0)))
    return _red(fig, plt)


@pytest.mark.parametrize('even', [False, True])
@pytest.mark.parametrize('shading', ['flat', 'nearest'])
@pytest.mark.parametrize('axis,inv', [('x', False), ('x', True), ('y', False), ('y', True)])
@pytest.mark.parametrize('delta', [-2.0 ** -30, 0.0, 2.0 ** -30])
def test_edges_on_pixel_centres_as_matplotlib(axis, inv, shading, even, delta):
    m = _tie_mesh(mplt, axis, inv, shading, delta, even)
    f = _tie_mesh(fplt, axis, inv, shading, delta, even)
    assert np.abs(m - f).max() <= 2


@pytest.mark.parametrize('shading', ['flat', 'nearest'])
def test_cells_over_half_pixels_as_matplotlib(shading):
    """8 cells over 180 px: every other edge on a pixel centre."""
    rng = np.random.default_rng(4)
    Z = rng.random((6, 8))
    out = []
    for plt in (mplt, fplt):
        fig = plt.figure(figsize=(3, 2), dpi=60)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        n = 1 if shading == 'flat' else 0
        ax.pcolormesh(np.arange(9 - 1 + n), np.arange(7 - 1 + n), Z, shading=shading, cmap='gray')
        out.append(_red(fig, plt))
    assert np.abs(out[0] - out[1]).max() <= 2


def test_one_row_of_centres_draws_as_matplotlib():
    """A single row of centres: matplotlib's cells of no height, nothing shown."""
    Z = np.arange(8.0)[None, :]
    out = []
    for plt in (mplt, fplt):
        fig = plt.figure(figsize=(2, 1), dpi=64)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.pcolormesh(np.arange(8.0), np.array([0.0]), Z, shading='nearest', cmap='gray')
        ax.set_ylim(-1, 1)
        out.append(_red(fig, plt))
    assert np.abs(out[0] - out[1]).max() <= 2


def test_aspect_box_has_matplotlibs_corners():
    """apply_aspect's Bbox.shrunk_to_aspect then anchored, bit for bit."""
    pos = []
    for plt in (mplt, fplt):
        fig = plt.figure(figsize=(15, 9), dpi=125)
        ax = fig.add_axes([0.0366666666666667, 0.0388888888888889, 0.9, 0.9])
        ax.set_xlim(-91.96359634399414, -57.33640670776367)
        ax.set_ylim(5.212242126464844, 25.988555908203125)
        ax.set_aspect('equal')
        if plt is mplt:
            fig.canvas.draw()
            pos.append(tuple(ax.get_position().extents))
        else:
            fig.draw()
            pos.append(tuple(ax.get_position().extents))
        plt.close('all')
    assert pos[0] == pos[1]


# -------------------------------------------------- maps, against cartopy

def _map_mesh(lib, axis, shading, delta):
    ccrs = pytest.importorskip('cartopy.crs')
    import fastplot.crs as fcrs
    plt, crs = (mplt, ccrs) if lib == 'mpl' else (fplt, fcrs)
    e = (80.5 if axis == 'lon' else 40.5) + delta
    c = np.array([e - 30.0, e, e + 30.0]) if shading == 'flat' else np.array([e - 15.0, e + 15.0])
    if axis == 'lon':
        o = np.array([20.0, 60.0]) if shading == 'flat' else np.array([30.0, 50.0])
        X, Y = c, o
        Z = np.tile([1.0, 2.0], ((o.size - 1) if shading == 'flat' else o.size, 1))
    else:
        o = np.array([40.0, 120.0]) if shading == 'flat' else np.array([60.0, 100.0])
        X, Y = o, c
        Z = np.tile([[1.0], [2.0]], (1, (o.size - 1) if shading == 'flat' else o.size))
    fig = plt.figure(figsize=(1.6, 0.8), dpi=100)          # a degree a pixel
    ax = fig.add_axes([0, 0, 1, 1], projection=crs.PlateCarree())
    ax.set_extent([0, 160, 0, 80], crs=crs.PlateCarree())
    ax.pcolormesh(X, Y, Z, shading=shading, cmap='gray', vmin=0, vmax=3,
                  transform=crs.PlateCarree())
    ax.spines['geo'].set_visible(False)
    return _red(fig, plt)


@pytest.mark.parametrize('shading', ['flat', 'nearest'])
@pytest.mark.parametrize('axis', ['lon', 'lat'])
@pytest.mark.parametrize('delta', [-2.0 ** -12, 0.0, 2.0 ** -12])
def test_map_edges_on_pixel_centres_as_cartopy(axis, shading, delta):
    m = _map_mesh('mpl', axis, shading, delta)
    f = _map_mesh('fp', axis, shading, delta)
    assert np.abs(m - f).max() <= 2


def test_map_box_rounds_half_a_pixel_up():
    """Agg's clip box: an axes edge at x.5 px rounds up, as plain axes' do."""
    import fastplot.crs as fcrs
    fig = fplt.figure(figsize=(2, 1), dpi=100)
    ax = fig.add_axes([0.125, 0.125, 0.75, 0.75], projection=fcrs.PlateCarree())
    ax.set_extent([0, 32, 0, 16], crs=fcrs.PlateCarree())
    fig.draw()
    assert ax.box_f[1] == 12.5 and ax.box[1] == 13
    fplt.close('all')
