"""The core kernels against references: contour lines against contourpy
(matplotlib's), tick choices against matplotlib's and cartopy's locators,
barb polygons against matplotlib's, line and polygon coverage by hand."""
import numpy as np
import pytest

from fastplot import _barbs, _contour, _draw, _ticker


# ----------------------------------------------------------------- contour

def _field(ny=181, nx=360, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:ny, 0:nx]
    z = (np.sin(x / 17.0) * np.cos(y / 11.0) * 10 + rng.normal(0, 0.3, (ny, nx))).astype(np.float32)
    return z


def _length(lines):
    return sum(np.hypot(*np.diff(p, axis=0).T).sum() for p in lines)


@pytest.mark.parametrize('nan_hole', [False, True])
def test_contour_matches_contourpy(nan_hole):
    contourpy = pytest.importorskip('contourpy')
    z = _field()
    if nan_hole:
        z[50:70, 100:140] = np.nan
        z[0, :] = np.nan
    levels = np.linspace(-9, 9, 13)
    fi, fj, off, lof = _contour.lines(z, levels)
    gen = contourpy.contour_generator(z=np.ma.masked_invalid(z), corner_mask=False,
                                      line_type='SeparateCode')
    for k, lv in enumerate(levels):
        mine = [np.c_[fj[off[p]:off[p + 1]], fi[off[p]:off[p + 1]]] for p in np.nonzero(lof == k)[0]]
        ref = gen.lines(lv)[0]
        assert len(mine) == len(ref), lv
        assert _length(mine) == pytest.approx(_length(ref), rel=1e-9, abs=1e-9)
        # every vertex lies on the level (linear along its grid edge)
        for p in mine:
            for c, r in p:
                if r == int(r):
                    a, b = z[int(r), int(np.floor(c))], z[int(r), min(int(np.floor(c)) + 1, z.shape[1] - 1)]
                    t = c - np.floor(c)
                else:
                    a, b = z[int(np.floor(r)), int(c)], z[min(int(np.floor(r)) + 1, z.shape[0] - 1), int(c)]
                    t = r - np.floor(r)
                assert a + t * (b - a) == pytest.approx(lv, abs=1e-4)


def test_contour_closed_rings_and_levels_any_order():
    y, x = np.mgrid[-20:21, -20:21]
    z = np.hypot(x, y).astype(np.float32)
    fi, fj, off, lof = _contour.lines(z, [15.0, 5.0, 10.0])
    assert off.size - 1 == 3
    for p in range(3):
        a, b = off[p], off[p + 1]
        assert fi[a] == fi[b - 1] and fj[a] == fj[b - 1]          # closed
        r = np.hypot(fi[a:b] - 20, fj[a:b] - 20)
        assert np.allclose(r, [15.0, 5.0, 10.0][lof[p]], atol=0.3)


def test_contour_empty():
    fi, fj, off, lof = _contour.lines(np.zeros((5, 5), np.float32), [1.0])
    assert fi.size == 0 and list(off) == [0] and lof.size == 0


# ------------------------------------------------------------------ ticker

def test_maxn_matches_matplotlib():
    mt = pytest.importorskip('matplotlib.ticker')
    rng = np.random.default_rng(1)
    for _ in range(300):
        a, b = sorted(rng.uniform(-1e4, 1e4, 2) * 10.0 ** rng.integers(-4, 3))
        nb = int(rng.integers(1, 10))
        ref = mt.MaxNLocator(nbins=nb, steps=[1, 2, 2.5, 5, 10]).tick_values(a, b)
        got = _ticker.MaxNLocator(nbins=nb, steps=[1, 2, 2.5, 5, 10]).tick_values(a, b)
        assert np.allclose(ref, got)


def test_fixed_locator_and_formatter_match_matplotlib():
    mt = pytest.importorskip('matplotlib.ticker')
    locs = np.arange(-40, 46, 2.0)
    assert np.array_equal(mt.FixedLocator(locs, nbins=10).tick_values(-40, 44),
                          _ticker.FixedLocator(locs, nbins=10).tick_values())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for ticks, lim in [([-40, -20, 0, 20, 40], (-40, 44)), ([0.1, 0.2, 0.3], (0.05, 0.35)),
                       ([1000.5, 1001.0, 1001.5], (1000.2, 1001.7)),
                       ([0, 2.5e-6, 5e-6], (0, 5e-6)), ([250, 500, 750], (200, 800))]:
        fig, ax = plt.subplots()
        ax.set_xlim(*lim)
        ax.set_xticks(ticks)
        fig.canvas.draw()
        ref = [t.get_text() for t in ax.get_xticklabels()]
        got, off = _ticker.ScalarFormatter().format(np.array(ticks, float), *lim)
        assert got == ref, (ticks, got, ref)
        plt.close(fig)


def test_lonlat_locators_match_cartopy():
    ct = pytest.importorskip('cartopy.mpl.ticker')
    for lo, hi in [(-180, 180), (-130, -60), (100, 140), (-10, 3), (0.2, 1.1), (-170, 170)]:
        assert np.allclose(ct.LongitudeLocator().tick_values(lo, hi),
                           _ticker.LongitudeLocator().tick_values(lo, hi))
    for lo, hi in [(-90, 90), (20, 55), (-5, 5), (60, 90)]:
        assert np.allclose(ct.LatitudeLocator().tick_values(lo, hi),
                           _ticker.LatitudeLocator().tick_values(lo, hi))
    assert _ticker.lon_label(-120) == '120°W' and _ticker.lon_label(180) == '180°'
    assert _ticker.lat_label(-30) == '30°S' and _ticker.lat_label(0) == '0°'


# ------------------------------------------------------------------- barbs

def test_barbs_match_matplotlib():
    pytest.importorskip('matplotlib.quiver')
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(3)
    u = rng.uniform(-80, 80, 40)
    v = rng.uniform(-80, 80, 40)
    u[:3] = 0.0
    v[:3] = [0.0, 2.0, 7.0]
    mag = np.hypot(u, v)
    fig, ax = plt.subplots()
    b = ax.barbs(np.zeros(40), np.zeros(40), u, v, length=7)
    nf, nb, hb, em = b._find_tails(mag, b.rounding, **b.barb_increments)
    ref = b._make_barbs(u, v, nf, nb, hb, em, b._length, b._pivot, b.sizes, b.fill_empty,
                        np.zeros(40, bool))
    plt.close(fig)
    # ours on a y-down screen with the station at the origin
    xy, off, empty = _barbs.barbs(np.zeros(40), np.zeros(40), u, -v, mag, length=7.0)
    assert off.size - 1 == 40
    for k in range(40):
        p = xy[off[k]:off[k + 1]].copy()
        p[:, 1] = -p[:, 1]
        r = np.asarray(ref[k])
        assert empty[k] == bool(em[k])
        assert p.shape == r.shape
        assert np.allclose(p, r, atol=1e-9)


# -------------------------------------------------------------------- draw

def test_line_coverage():
    c = np.full((20, 40, 4), 255, np.uint8)
    _draw.draw_lines(c, np.array([[2.0, 10.0], [38.0, 10.0]]), [0, 2], (0, 0, 0, 255), 2.0)
    assert (c[9, 5:35, 0] == 0).all() and (c[10, 5:35, 0] == 0).all()     # 2 px, centred on y=10
    assert (c[8, 5:35, 0] == 255).all() and (c[11, 5:35, 0] == 255).all()
    c = np.full((20, 40, 4), 255, np.uint8)
    _draw.draw_lines(c, np.array([[2.0, 10.5], [38.0, 10.5]]), [0, 2], (0, 0, 0, 255), 0.5)
    assert (c[10, 5:35, 0] == 128).all()                                  # a half-pixel line


def test_joints_not_drawn_twice():
    c = np.full((40, 40, 4), 255, np.uint8)
    pts = np.array([[5, 20.5], [20, 20.5], [35, 20.5]], float)
    _draw.draw_lines(c, pts, [0, 3], (0, 0, 0, 255), 0.5)
    assert c[20, 20, 0] == c[20, 10, 0] == c[20, 30, 0]


def test_dashes():
    c = np.full((10, 100, 4), 255, np.uint8)
    _draw.draw_lines(c, np.array([[0.0, 5.0], [100.0, 5.0]]), [0, 2], (0, 0, 0, 255), 2.0,
                     dashes=(10.0, 10.0))
    row = c[5, :, 0]
    assert (row[1:9] == 0).all() and (row[11:19] == 255).all() and (row[21:29] == 0).all()


def test_polygon_fill_with_hole():
    c = np.full((50, 50, 4), 255, np.uint8)
    outer = [[5, 5], [45, 5], [45, 45], [5, 45]]
    hole = [[20, 20], [30, 20], [30, 30], [20, 30]]
    _draw.fill_polygons(c, np.array(outer + hole, float), [0, 4, 8], (0, 0, 255, 255))
    assert (c[10, 10] == [0, 0, 255, 255]).all()
    assert (c[25, 25] == [255, 255, 255, 255]).all()
    assert (c[2, 2] == [255, 255, 255, 255]).all()
    assert c[5:45, 5:45, 0].sum() == 255 * 100                              # exactly the hole


def test_blend_formula():
    c = np.zeros((1, 1, 4), np.uint8)
    c[...] = (200, 100, 50, 255)
    _draw.blend_px(c, 0, 0, np.float32(0), np.float32(0), np.float32(255), np.float32(128),
                   np.float32(0.5))
    f = np.float32(128) / np.float32(255) * np.float32(0.5)
    k = np.float32(1) - f
    exp = [np.uint8(np.float32(0) * f + np.float32(v) * k + np.float32(0.5)) for v in (200, 100)]
    assert c[0, 0, 0] == exp[0] and c[0, 0, 1] == exp[1]


def _bits(u):
    u = np.asarray(u)
    return u.view(np.uint64) if u.dtype == np.float64 else u


def test_contour_periodic_is_the_wrapped_grid():
    """periodic=True gives exactly the lines of z with its first column again
    after its last (what a global grid's seam needs), without copying z."""
    rng = np.random.default_rng(5)
    for trial in range(40):
        ny, nx = int(rng.integers(2, 80)), int(rng.integers(2, 120))
        y, x = np.mgrid[0:ny, 0:nx]
        z = (np.sin(x / rng.uniform(2, 30)) * np.cos(y / rng.uniform(2, 30)) * 10
             + rng.normal(0, 2, (ny, nx))).astype(np.float32)
        if trial % 3 == 0:
            z[rng.random(z.shape) < 0.05] = np.nan
        lv = np.sort(rng.uniform(-12, 12, int(rng.integers(1, 12))))
        a = _contour.lines(z, lv, periodic=True)
        b = _contour.lines(np.concatenate([z, z[:, :1]], axis=1), lv)
        assert all(np.array_equal(_bits(u), _bits(v)) for u, v in zip(a, b))


def test_interp_index_is_np_interp():
    rng = np.random.default_rng(6)
    for n in (1, 2, 3, 17, 500):
        fp = np.cumsum(rng.uniform(0.1, 2, n))
        f = rng.uniform(-2, n + 2, 2000)
        f[::7] = np.round(f[::7])
        f[::11] = np.nan
        f[::13] = n - 1
        assert np.array_equal(_bits(_contour.interp_index(f, fp)),
                              _bits(np.interp(f, np.arange(n), fp)))
