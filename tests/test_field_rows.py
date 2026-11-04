"""Fields put onto the canvas row by row (_render's _bands_*, _cells_*)
against the way through full rasters -- value raster, touched_bands,
shade, _blend_image: the same canvas bytes."""
import types

import numpy as np
import pytest

from fastplot import _color, _jit, _raster, _render


def _canvas(H=90, W=130, seed=0):
    """A canvas of random colours, some pixels translucent or transparent
    (the blend's other ways)."""
    rng = np.random.default_rng(seed)
    c = _render.CpuCanvas(H, W, (255, 255, 255, 255))
    c.a[...] = rng.integers(0, 256, c.a.shape, dtype=np.uint8)
    c.a[..., 3] = np.where(rng.random((H, W)) < 0.8, 255, c.a[..., 3])
    return c


def _bands(levels, under=True, over=True, alpha=None, lut_alpha=255):
    lut = np.zeros((len(levels) - 1, 4), np.uint8)
    lut[:, 0] = np.arange(len(levels) - 1) * 7 % 256
    lut[:, 1] = np.arange(len(levels) - 1) * 13 % 256
    lut[:, 2] = 200
    lut[:, 3] = lut_alpha
    return _render.Coloring(levels=levels, band_lut=lut,
                            under=(1, 2, 3, 255) if under else None,
                            over=(250, 251, 252, 255) if over else None)


def _old_bands(canvas, box, values, coloring, alpha=None):
    """_blend_field as it was: through the touched raster and a layer."""
    h, w = values.shape
    layer = np.empty((h, w, 4), np.uint8)
    a = np.float32(1.0 if alpha is None else alpha)
    if coloring.translucent:
        coloring.shade(np.ascontiguousarray(_render.band_seams(np, values, coloring)), layer, None)
        _render._blend_image(canvas, layer, box[1], box[0], a)
    t = np.empty_like(values)
    _render.touched_bands(values, t, coloring.top)
    coloring.shade(t, layer, None)
    _render._blend_image(canvas, layer, box[1], box[0], a)


def _field(shape, levels, seed=1):
    """Values over the levels' range and past it, at levels exactly, NaN and
    infinities among them."""
    rng = np.random.default_rng(seed)
    lo, hi = levels[0], levels[-1]
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    v = (lo + (hi - lo) * (0.5 + 0.7 * np.sin(x / 9.0) * np.cos(y / 7.0))).astype(np.float32)
    v += rng.normal(0, (hi - lo) * 0.01, shape).astype(np.float32)
    pick = rng.random(shape)
    v[pick < 0.03] = rng.choice(np.asarray(levels, np.float32), (pick < 0.03).sum())
    v[(pick > 0.03) & (pick < 0.05)] = np.nan
    v[:, :7] = np.nan                                  # (a margin of no field)
    v[5, 9], v[6, 20], v[30, 40] = np.inf, -np.inf, np.float32(3e38)
    return v


LEVELS = [np.linspace(-10.0, 12.0, 41), np.linspace(0, 75, 76), np.array([0.0, 1.0, 5.0, 6.0, 30.0]),
          np.linspace(1000.0, 1000.5, 11), np.linspace(-1.0, 1.0, 3)]


@pytest.mark.parametrize('levels', LEVELS)
@pytest.mark.parametrize('under,over', [(True, True), (False, True), (False, False)])
@pytest.mark.parametrize('box', [(10, 5, 120, 80), (-15, -9, 100, 60), (40, 30, 160, 110)])
def test_bands_raster_as_the_full_rasters(levels, under, over, box):
    coloring = _bands(levels, under, over)
    values = _field((box[3] - box[1], box[2] - box[0]), levels)
    a, b = _canvas(), _canvas()
    _old_bands(a.a, box, values, coloring)
    b.field_values(box, values, coloring)
    assert a.a.tobytes() == b.a.tobytes()


@pytest.mark.parametrize('alpha,lut_alpha', [(0.6, 255), (None, 140)])
def test_bands_raster_translucent_as_the_full_rasters(alpha, lut_alpha):
    levels = LEVELS[0]
    coloring = _bands(levels, alpha=alpha, lut_alpha=lut_alpha)
    box = (3, 4, 110, 70)
    values = _field((66, 107), levels)
    a, b = _canvas(), _canvas()
    _old_bands(a.a, box, values, coloring, alpha)
    b.field_values(box, values, coloring, alpha)
    assert a.a.tobytes() == b.a.tobytes()


def test_band_guess_is_band_row_or_none():
    """_guess's rows are _band_row's, wherever it gives one: values at the
    levels, a float32 step either side, between them, far out."""
    rng = np.random.default_rng(4)
    for levels in LEVELS + [np.cumsum(rng.uniform(0.9, 1.1, 60)) - 30.0,
                            np.linspace(-3e5, 3e5, 90), np.linspace(1e-6, 2e-6, 33)]:
        c = _bands(levels)
        (top, lev, m, tab, eq0, l0, inv_d, uniform, L0, D, mu, mf, vec), _ = c.fused()
        l32 = lev.astype(np.float32)
        x = np.concatenate([l32, np.nextafter(l32, np.float32(np.inf)),
                            np.nextafter(l32, np.float32(-np.inf)),
                            rng.uniform(lev[0] - 3 * (lev[-1] - lev[0]), lev[-1] + 3 * (lev[-1] - lev[0]),
                                        5000).astype(np.float32),
                            np.array([np.nan, np.inf, -np.inf, 3e38, -3e38], np.float32)])
        x = np.repeat(x, 2)                             # (pairs: some pixels rise by nothing)
        row = np.empty_like(x)
        kk = np.empty(x.size, np.int32)
        _render._touched_guess(x, x, x, 0, x.size, np.float32(np.inf), L0, D, mu, mf, vec, row, kk)
        want = np.array([_color._band_row(np.float64(v), lev, m, eq0, False, l0, inv_d, uniform)
                         for v in row])
        sure = kk >= 0
        assert vec == bool(sure.any())
        assert np.array_equal(kk[sure], want[sure])
        assert sure.mean() > 0.5 or not vec


def _axis_grid(nx, ny, flip_y=False):
    x = np.linspace(-120.0, -80.0, nx)
    y = np.linspace(20.0, 50.0, ny)
    if flip_y:
        y = y[::-1]
    xa, ya = _raster.Axis(x), _raster.Axis(y)
    return types.SimpleNamespace(xaxis=xa, yaxis=ya), xa, ya


@pytest.mark.parametrize('flip', [False, True])
@pytest.mark.parametrize('keep', [False, True])
@pytest.mark.parametrize('box', [(6, 2, 126, 88), (6, -20, 126, 66), (-30, 30, 90, 116)])
def test_bands_sep_as_the_full_rasters(flip, keep, box):
    """A plain axes' field drawn larger than its grid: its rows made as they
    go onto the canvas (and kept, for a contourf with holes, those off the
    canvas too)."""
    grid, xa, ya = _axis_grid(23, 17, flip)
    levels = np.linspace(-1.0, 1.0, 31)
    yy, xx = np.mgrid[0:17, 0:23]
    z = (np.sin(xx / 3.0) * np.cos(yy / 4.0)).astype(np.float32)
    z[4, 5], z[9, 9] = np.nan, np.inf
    h, w = box[3] - box[1], box[2] - box[0]
    cols = _raster.axis_table(np.linspace(-121.0, -79.0, w), xa, _raster.BILINEAR)
    rows = _raster.axis_table(np.linspace(51.0, 19.0, h), ya, _raster.BILINEAR)
    values = np.empty((h, w), np.float32)
    _raster.sample_sep(values, cols, rows, z, xa, ya, _raster.BILINEAR)
    coloring = _bands(levels)
    a, b = _canvas(), _canvas()
    _old_bands(a.a, box, values, coloring)
    b.keep_values = keep
    b.field_sep(box, cols, rows, z, grid, _raster.BILINEAR, coloring)
    assert a.a.tobytes() == b.a.tobytes()
    t = np.empty_like(values)
    _render.touched_bands(values, t, coloring.top)
    assert b.last_values[1].tobytes() == t.tobytes()   # kept, or made again
    assert b.last_touched(3, 40, 0, 50).tobytes() == t[3:40, 0:50].tobytes()


def _mesh(seed=0, period=0.0, fold=False):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:31, 0:41].astype(np.float64)
    sx = x * 3.4 + 0.02 * y + 5.0 + rng.normal(0, 0.05, x.shape)
    sy = y * 2.8 - 0.01 * x + 3.0
    if fold:
        sy = np.abs(y - 15.0) * 2.9 + 1.5 + x * 0.1
    if period:                                          # across a cylindrical map's seam
        sx = (sx + 60.0) % period - 10.0
    return np.ascontiguousarray(sx), np.ascontiguousarray(sy)


@pytest.mark.parametrize('period,fold', [(0.0, False), (0.0, True), (150.0, False)])
@pytest.mark.parametrize('keep', [False, True])
@pytest.mark.parametrize('at', [(0, 0), (-12, 20)])
def test_bands_by_triangle_table_as_the_cells(period, fold, keep, at):
    """A contourf's mesh drawn again by its table of triangles: the raster
    raster_cells makes, and the same canvas."""
    sx, sy = _mesh(period=period, fold=fold)
    rng = np.random.default_rng(5)
    levels = np.linspace(-2.0, 2.0, 25)
    h, w = 95, 140
    box = (at[0], at[1], at[0] + w, at[1] + h)
    coloring = _bands(levels)
    _raster._TRI_TABLES.clear()
    for frame in range(3):
        z = rng.normal(0, 0.8, sx.shape).astype(np.float32)
        z[3, 4] = np.nan
        values = np.full((h, w), np.nan, np.float32)
        _raster.raster_cells(values, sx, sy, z, _raster.BILINEAR, period)
        a, b = _canvas(seed=frame), _canvas(seed=frame)
        _old_bands(a.a, box, values, coloring)
        b.keep_values = keep
        b._mesh(box, h, w, sx, sy, z, _raster.BILINEAR, period, coloring, None)
        assert a.a.tobytes() == b.a.tobytes()
        t = np.empty_like(values)
        _render.touched_bands(values, t, coloring.top)
        assert b.last_values[1].tobytes() == t.tobytes()
        got = _raster.tri_table((h, w), sx, sy, z, _raster.BILINEAR, period)
        if frame:                                       # the table itself: raster_cells' raster
            out = np.empty((h, w), np.float32)
            _raster.take_tris(got[0], sx, sy, got[1], period, out)
            assert out.tobytes() == values.tobytes()


def test_triangle_table_on_any_number_of_threads_as_on_one():
    sx, sy = _mesh(fold=True)
    _raster._TRI_TABLES.clear()
    z = np.ones(sx.shape, np.float32)
    _raster.tri_table((95, 140), sx, sy, z, _raster.BILINEAR, 0.0)
    with _jit.threads(1):
        one = _raster.tri_table((95, 140), sx, sy, z, _raster.BILINEAR, 0.0)[0].copy()
    for _ in range(10):
        _raster._TRI_TABLES.clear()
        _raster.tri_table((95, 140), sx, sy, z, _raster.BILINEAR, 0.0)
        with _jit.threads(8):
            assert _raster.tri_table((95, 140), sx, sy, z, _raster.BILINEAR,
                                     0.0)[0].tobytes() == one.tobytes()


def _old_cells(canvas, box, values, coloring, alpha):
    layer = np.empty(values.shape + (4,), np.uint8)
    coloring.shade(values, layer, alpha)
    _render._blend_image(canvas, layer, box[1], box[0], np.float32(1.0))


@pytest.mark.parametrize('alpha', [None, 0.5])
@pytest.mark.parametrize('bad', [None, (0.2, 0.3, 0.4, 1.0)])
def test_cells_sep_as_the_value_raster(alpha, bad):
    """pcolormesh's nearest cells by their own colours: as shading the
    pixels' values."""
    grid, xa, ya = _axis_grid(23, 17, True)
    rng = np.random.default_rng(6)
    z = rng.normal(0, 1, (17, 23)).astype(np.float32)
    z[2, 3] = np.nan
    box = (-5, 4, 120, 85)
    h, w = box[3] - box[1], box[2] - box[0]
    cols = _raster.axis_table(np.linspace(-123.0, -79.0, w), xa, _raster.NEAREST)
    rows = _raster.axis_table(np.linspace(52.0, 19.0, h), ya, _raster.NEAREST)
    values = np.empty((h, w), np.float32)
    _raster.sample_sep(values, cols, rows, z, xa, ya, _raster.NEAREST)
    cmap = _color.get_cmap('viridis')
    if bad is not None:
        cmap = cmap.with_extremes(bad=bad)
    coloring = _render.Coloring(cmap=cmap, norm=_color.make_norm(None, vmin=-1.5, vmax=1.5))
    a, b = _canvas(), _canvas()
    _old_cells(a.a, box, values, coloring, alpha)
    b.field_sep(box, cols, rows, z, grid, _raster.NEAREST, coloring, alpha)
    assert a.a.tobytes() == b.a.tobytes()


@pytest.mark.parametrize('bad', [None, (0.2, 0.3, 0.4, 1.0)])
def test_cells_table_as_take_colors(bad):
    y, x = np.mgrid[0:41, 0:61].astype(np.float64)
    sx, sy = x * 9.0 + 7.0 * y * 0.3 + 3.0, y * 9.0 - x * 2.0 + 90.0   # leaning cells
    rng = np.random.default_rng(7)
    cmap = _color.get_cmap('magma')
    if bad is not None:
        cmap = cmap.with_extremes(bad=bad)
    coloring = _render.Coloring(cmap=cmap, norm=_color.make_norm(None, vmin=-2, vmax=2))
    box = (0, 0, 620, 240)
    _raster.clear_tables()
    for frame in range(3):
        z = rng.normal(size=sy.shape).astype(np.float32)
        values = np.full((240, 620), np.nan, np.float32)
        a, b = _canvas(240, 620, frame), _canvas(240, 620, frame)
        touched = _raster.wants_touched(sx, sy, _raster.NEAREST, 0.0, z.size, values.shape)
        if touched:                                     # (leaning cells: Agg's corners too)
            _raster.raster_cells_touched(values, np.empty(values.shape, np.float32), sx, sy, z,
                                         _raster.NEAREST)
        else:
            _raster.raster_cells(values, sx, sy, z, _raster.NEAREST, 0.0)
        _old_cells(a.a, box, values, coloring, None)
        b._mesh(box, 240, 620, sx, sy, z, _raster.NEAREST, 0.0, coloring, None)
        assert a.a.tobytes() == b.a.tobytes()
