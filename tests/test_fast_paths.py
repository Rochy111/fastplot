"""Each short cut against the long way round: the same numbers to the bit,
and nothing kept that should not be."""
import numpy as np
import pytest

import fastplot
import fastplot.crs as ccrs
from fastplot import _agg, _artists, _cartesian, _contour, _draw, _fields, _jit, _legend, _markers
from fastplot import _proj, _raster, _render
from fastplot._render import View


def _lambert_nodes(ny=500, nx=400, jitter=0.0):
    p = _proj.LambertConformal(central_longitude=-97.5, central_latitude=38.5)
    lon, lat = np.meshgrid(np.linspace(-125.0, -70.0, nx), np.linspace(22.0, 50.0, ny))
    view = View(p, -3e6, 6e3, 2e6, -6e3)
    return view, _proj.PlateCarree(), lon + jitter, lat


@pytest.fixture
def kept(monkeypatch):
    monkeypatch.setattr(_raster, '_KEPT', type(_raster._KEPT)())
    return _raster._KEPT


def test_kept_nodes_are_forward_nodes(kept):
    view, src, x, y = _lambert_nodes()
    want = _raster.forward_nodes(view, src, x, y)
    first = _raster.kept_nodes(view, src, x, y)         # seen: nothing kept
    assert next(iter(kept.values())) is None
    second = _raster.kept_nodes(view, src, x, y)        # seen twice: kept
    third = _raster.kept_nodes(view, src, x.copy(), y.copy())   # the same numbers, new arrays
    for got in (first, second, third):
        assert got[0].tobytes() == want[0].tobytes() and got[1].tobytes() == want[1].tobytes()
    assert third[0] is second[0]
    assert np.shares_memory(next(iter(kept.values()))[0], x) is False   # a copy: x is the user's


def test_kept_nodes_notice_one_bit(kept):
    view, src, x, y = _lambert_nodes()
    for _ in range(3):
        _raster.kept_nodes(view, src, x, y)
    moved = x.copy()
    moved[250, 200] = np.nextafter(moved[250, 200], 0.0)
    got = _raster.kept_nodes(view, src, moved, y)
    want = _raster.forward_nodes(view, src, moved, y)
    assert got[0].tobytes() == want[0].tobytes()
    # changed in place, the arrays the caller gave are not trusted either
    x2, y2 = x.copy(), y.copy()
    for _ in range(3):
        _raster.kept_nodes(view, src, x2, y2)
    y2[3, 3] += 0.25
    got = _raster.kept_nodes(view, src, x2, y2)
    assert got[1].tobytes() == _raster.forward_nodes(view, src, x2, y2)[1].tobytes()


def test_kept_nodes_tell_negative_zero(kept):
    assert _raster._bits_differ(np.zeros((2, 2)).view(np.uint64),
                                np.array([[0.0, -0.0], [0.0, 0.0]]).view(np.uint64)) == 1


def test_kept_nodes_leave_small_grids_and_stay_few(kept):
    view, src, x, y = _lambert_nodes(50, 40)
    _raster.kept_nodes(view, src, x, y)
    assert not kept
    view, src, x, y = _lambert_nodes()
    for k in range(_raster._KEPT_MAX + 3):
        v = View(view.proj, view.x0 + k, view.dx, view.y0, view.dy)
        _raster.kept_nodes(v, src, x, y)
    assert len(kept) == _raster._KEPT_MAX


def test_kept_nodes_stay_within_their_memory(kept, monkeypatch):
    view, src, x, y = _lambert_nodes()
    monkeypatch.setattr(_raster, 'KEEP_BYTES', x.size * 32 + 1)     # room for one grid
    views = [View(view.proj, view.x0 + k, view.dx, view.y0, view.dy) for k in range(2)]
    for v in views + views:                             # each of them seen twice
        _raster.kept_nodes(v, src, x, y)
    assert sum(1 for v in kept.values() if v) == 1
    monkeypatch.setattr(_raster, 'KEEP_BYTES', x.size * 32 - 1)     # too big to keep at all
    kept.clear()
    for _ in range(3):
        _raster.kept_nodes(view, src, x, y)
    assert not kept


def _touched_slow(src, top):
    H, W = src.shape
    out = np.empty((H, W), np.float32)
    for i in range(H):
        for j in range(W):
            c = src[i, j]
            if not np.isfinite(c):
                out[i, j] = c
                continue
            rx = fx = ry = fy = np.float32(0.0)
            for k, (ok, n) in enumerate(((j + 1 < W, src[i, min(j + 1, W - 1)]),
                                         (j > 0, src[i, max(j - 1, 0)]),
                                         (i + 1 < H, src[min(i + 1, H - 1), j]),
                                         (i > 0, src[max(i - 1, 0), j]))):
                if not ok or not np.isfinite(n):
                    continue
                d = n - c
                if k < 2:
                    rx, fx = max(rx, d), max(fx, -d)
                else:
                    ry, fy = max(ry, d), max(fy, -d)
            hi = c + np.float32(0.5) * (rx + ry)
            if hi > top and c - np.float32(0.5) * (fx + fy) <= top:
                hi = top
            out[i, j] = hi
    return out


def test_touched_bands_with_holes_and_infinities():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(40, 60)).astype(np.float32)
    src[rng.random(src.shape) < 0.1] = np.nan
    src[3, 4], src[20, 30], src[0, 0], src[39, 59] = np.inf, -np.inf, 3.3e38, -3.3e38
    src[10, 10], src[10, 11] = 3.3e38, -3.3e38          # their difference overflows
    out = np.empty_like(src)
    with np.errstate(over='ignore', invalid='ignore'):
        _render.touched_bands(src, out, np.float32(0.5))
        want = _touched_slow(src, np.float32(0.5))
    assert np.array_equal(out, want, equal_nan=True)


def test_rectilinear_as_before():
    x, y = np.meshgrid(np.arange(4.0), np.arange(3.0))
    a, b = _fields._rectilinear(x, y)
    assert a.tolist() == [0, 1, 2, 3] and b.tolist() == [0, 1, 2]
    bent = x.copy()
    bent[1, 2] += 0.5                                   # first and last rows alike, the middle not
    assert _fields._rectilinear(bent, y) is None
    bent = x.copy()
    bent[-1, 0] += 0.5
    assert _fields._rectilinear(bent, y) is None


def test_bilinear_as_the_formula():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(30, 40))
    fi, fj = rng.uniform(-0.5, 29.5, 5000), rng.uniform(-0.5, 39.5, 5000)
    ny, nx = a.shape
    i0 = np.clip(np.floor(fi).astype(np.int64), 0, ny - 2)
    j0 = np.clip(np.floor(fj).astype(np.int64), 0, nx - 2)
    ti, tj = fi - i0, fj - j0
    want = ((a[i0, j0] * (1 - tj) + a[i0, j0 + 1] * tj) * (1 - ti)
            + (a[i0 + 1, j0] * (1 - tj) + a[i0 + 1, j0 + 1] * tj) * ti)
    assert _fields._bilinear(a, fi, fj).tobytes() == want.tobytes()


def test_inside_count_as_numpy():
    p = np.random.default_rng(2).uniform(0, 10, (1000, 2))
    p[::50, 0] = np.nan
    l, t, r, b = 2.0, 3.0, 7.0, 8.0
    want = int(((p[:, 0] > l) & (p[:, 0] < r) & (p[:, 1] > t) & (p[:, 1] < b)).sum())
    assert _legend._inside_count(p, l, t, r, b) == want
    assert _legend._inside_count(np.empty((0, 2)), l, t, r, b) == 0


def test_across_as_c_():
    pos = np.array([1.5, 2.5, 9.0])
    want = np.stack([np.c_[pos, np.full_like(pos, 4.0)], np.c_[pos, np.full_like(pos, 7.0)]], 1)
    assert np.array_equal(_cartesian._across(pos, 4.0, 7.0, 0), want)
    want = np.stack([np.c_[np.full_like(pos, 4.0), pos], np.c_[np.full_like(pos, 7.0), pos]], 1)
    assert np.array_equal(_cartesian._across(pos, 4.0, 7.0, 1), want)


def _canvas():
    return np.full((300, 400, 4), 255, np.uint8)


@pytest.mark.parametrize('marker', ['o', 's', '^', '*', 'x', '+', 'D', '|', '8'])
@pytest.mark.parametrize('clip', [None, (60.5, 40.25, 320.75, 250.5)])
@pytest.mark.parametrize('width', [0.0, 0.5, 1.0, 4.0])
def test_markers_by_masks_as_the_general_path(marker, clip, width, monkeypatch):
    """Coverage worked out marker by marker and blended after: the same bytes."""
    g = _markers.marker_geom(marker, 1.0, 'full')
    rng = np.random.default_rng(0)
    n = 500
    pts = rng.uniform((-20, -20), (420, 320), (n, 2))
    pts[::3] = np.floor(pts[::3] * 2) / 2               # ties: half pixels
    sizes = rng.uniform(1, 30, n)
    cols = rng.integers(0, 256, (n, 4), dtype=np.uint8)
    face = cols if g.filled else None
    edge = cols[::-1] if width > 0 else None
    if face is None and edge is None:
        return
    fill = g.fill if g.filled else None
    stroke = g.stroke if width > 0 else None
    a, b = _canvas(), _canvas()
    monkeypatch.setattr(_draw, 'MASK_MARKERS', 1 << 62)
    _draw.exact_markers(a, pts, sizes, fill, stroke, width, face, edge, clip)
    monkeypatch.setattr(_draw, 'MASK_MARKERS', 1)
    drew = []
    real = _agg.draw_markers
    monkeypatch.setattr(_agg, 'draw_markers', lambda *k: drew.append(real(*k)) or drew[-1])
    _draw.exact_markers(b, pts, sizes, fill, stroke, width, face, edge, clip)
    assert drew == [True] and a.tobytes() == b.tobytes()


def test_outline_out_of_its_box_is_noticed():
    clip = np.array([0.0, 0.0, 100.0, 100.0])
    pts = np.array([[10.0, 10.0], [20.0, 30.0]])
    assert _agg._within(pts, 2, 5, 5, 25, 35, clip)
    assert not _agg._within(pts, 2, 11, 5, 25, 35, clip)
    assert not _agg._within(pts, 2, 5, 5, 19, 35, clip)
    # a side on the clip is open: what is past it is clipped
    assert _agg._within(np.array([[-9.0, 10.0]]), 1, 0, 5, 25, 35, clip)


@pytest.mark.parametrize('filled', [False, True])
@pytest.mark.parametrize('cap', [_agg.BUTT, _agg.PROJECTING, _agg.ROUND])
@pytest.mark.parametrize('join', [_agg.JOIN_ROUND, _agg.JOIN_MITER, _agg.JOIN_BEVEL])
def test_small_lines_in_one_kernel_as_the_steps_apart(cap, join, filled, monkeypatch):
    rng = np.random.default_rng(1)
    xy = rng.uniform((-30, -30), (430, 330), (120, 2)).cumsum(0) % (460, 360) - (30, 30)
    off = np.array([0, 40, 41, 90, 120])
    st = _draw.Styles((200, 30, 10, 180), 2.5, None, None, cap, join)
    segs = _draw.segments(xy, off, filled=filled)
    a, b = _canvas(), _canvas()
    monkeypatch.setattr(_draw, 'SMALL_LINE', -1)
    _draw.draw_segments(a, segs, st, (20, 20, 380, 280))
    monkeypatch.setattr(_draw, 'SMALL_LINE', 1 << 40)
    _draw.draw_segments(b, segs, st, (20, 20, 380, 280))
    assert a.tobytes() == b.tobytes()


def _arc_by_angles(x, y, dx1, dy1, dx2, dy2, w):
    """calc_arc as Agg writes it, angles and all."""
    a1, a2 = np.arctan2(dy1, dx1), np.arctan2(dy2, dx2)
    da = np.arccos(w / (w + 0.125)) * 2.0
    pts = [(x + dx1, y + dy1)]
    if a1 > a2:
        a2 += 2.0 * np.pi
    n = int((a2 - a1) / da)
    da = (a2 - a1) / (n + 1)
    a1 += da
    for _ in range(n):
        pts.append((x + np.cos(a1) * w, y + np.sin(a1) * w))
        a1 += da
    pts.append((x + dx2, y + dy2))
    return pts


def test_arc_without_angles_as_with_them():
    rng = np.random.default_rng(2)
    out = np.empty((64, 2))
    turns = np.r_[rng.uniform(-3.2, 3.2, 3000), 10.0 ** rng.uniform(-14, 0.5, 3000),
                  -(10.0 ** rng.uniform(-14, 0.5, 1000)), 0.0]
    for t in turns:
        w = 10.0 ** rng.uniform(-2, 1.5)
        a = rng.uniform(-np.pi, np.pi)
        dx1, dy1 = w * np.cos(a), w * np.sin(a)
        dx2, dy2 = w * np.cos(a + t), w * np.sin(a + t)
        n = _agg._arc(out, 0, True, 3.0, 4.0, dx1, dy1, dx2, dy2, w)
        want = _arc_by_angles(3.0, 4.0, dx1, dy1, dx2, dy2, w)
        assert n == len(want) and np.array_equal(out[:n], np.array(want))


def test_styles_take_lists_of_caps():
    a = _draw.Styles((0, 0, 0, 255), 2.0, None, None, [_draw.ROUND], [_draw.JOIN_MITER])
    b = _draw.Styles((0, 0, 0, 255), 2.0, None, None, _draw.ROUND, _draw.JOIN_MITER)
    for k in ('widths', 'limits', 'dash_starts', 'dashes', 'dash_offsets', 'caps', 'joins'):
        x, y = getattr(a, k), getattr(b, k)
        assert x.dtype == y.dtype and np.array_equal(x, y)


def _mesh_cells(mode, shape=(40, 60)):
    ny, nx = shape
    y, x = np.mgrid[0:ny + 1, 0:nx + 1].astype(np.float64)
    return x * 9.0 + 7.0 * y * 0.3 + 3.0, y * 9.0 - x * 2.0 + 90.0   # leaning cells


@pytest.mark.parametrize('mode', [_raster.NEAREST, _raster.FLAT])
@pytest.mark.parametrize('touched', [False, True])
def test_mesh_by_its_table_as_the_full_way(mode, touched):
    sx, sy = _mesh_cells(mode)
    shape = sy.shape if mode == _raster.NEAREST else (sy.shape[0] - 1, sy.shape[1] - 1)
    rng = np.random.default_rng(3)

    def full(z):
        out = np.full((240, 620), np.nan, np.float32)
        if touched:
            _raster.raster_cells_touched(out, np.empty(out.shape, np.float32), sx, sy, z, mode)
        else:
            _raster.raster_cells(out, sx, sy, z, mode, 0.0)
        return out

    def table(z):
        out = np.full((240, 620), np.nan, np.float32)
        return out, _raster.cells_by_table(out, sx, sy, z, mode, 0.0, touched)
    _raster.clear_tables()
    z = rng.normal(size=shape).astype(np.float32)
    assert not table(z)[1]                              # seen once: nothing kept
    for k in range(4):
        z = rng.normal(size=shape).astype(np.float32)
        got, used = table(z)
        assert used and got.tobytes() == full(z).tobytes()
    z[3, 5] = np.nan
    got, used = table(z)
    assert used != touched                              # the cell beneath shows: the full way
    if used:
        assert got.tobytes() == full(z).tobytes()
    z[3, 5] = np.inf
    assert not table(z)[1]


def test_cells_on_any_number_of_threads_as_on_one():
    """Pixel centres on the edge two rows of cells share, and a mesh folded
    over itself: the later cell wins, not the faster thread."""
    y, x = np.mgrid[0:41, 0:61].astype(np.float64)
    grids = [(x * 9.0 + 2.1 * y + 3.0, y * 9.0 - x * 2.0 + 90.0),       # ties
             (x * 10.0, y * 5.0),                                       # along the pixels, ties
             (x * 9.5, np.abs(y - 20.0) * 11.0 + 1.5 + x * 0.25)]       # folded
    for sx, sy in grids:
        z = np.random.default_rng(0).normal(size=(40, 60)).astype(np.float32)

        def raster(mode, zz):
            out = np.full((240, 620), np.nan, np.float32)
            _raster.raster_cells(out, sx, sy, zz, mode, 0.0)
            return out.tobytes()
        with _jit.threads(1):
            one = raster(_raster.FLAT, z)
        with _jit.threads(8):
            assert all(raster(_raster.FLAT, z) == one for _ in range(30))


def test_mesh_table_goes_with_its_arrays():
    sx, sy = _mesh_cells(_raster.FLAT)
    z = np.ones((sy.shape[0] - 1, sy.shape[1] - 1), np.float32)
    out = np.full((240, 620), np.nan, np.float32)
    _raster.clear_tables()
    _raster.cells_by_table(out, sx, sy, z, _raster.FLAT, 0.0, False)
    assert _raster.cells_by_table(out, sx, sy, z, _raster.FLAT, 0.0, False)
    assert not _raster.cells_by_table(out, sx.copy(), sy, z, _raster.FLAT, 0.0, False)


def _lines_many(n=800, k=9):
    """Polylines of several lengths, leaving the canvas, some of them rings."""
    rng = np.random.default_rng(4)
    lens = rng.integers(2, k, n)
    off = np.r_[0, np.cumsum(lens)].astype(np.int64)
    xy = rng.uniform((-50, -50), (450, 350), (off[-1], 2))
    return xy, off, (lens > 3) & (np.arange(n) % 3 == 0)


def test_clip_on_every_core_as_on_one(monkeypatch):
    xy, off, closed = _lines_many()
    many = _agg.clip_paths(xy, off, closed, 400, 300)
    monkeypatch.setattr(_agg, 'SERIAL_STROKE', 1 << 40)
    one = _agg.clip_paths(xy, off, closed, 400, 300)
    for a, b in zip(many, one):
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes()


def test_dash_on_every_core_as_on_one(monkeypatch):
    xy, off, closed = _lines_many()
    st = np.zeros(off.size - 1, np.int64)
    st[::2] = 1
    args = (np.array([5.0, 3.0, 1.0, 2.0, 7.0, 1.5]), np.array([0, 2, 6], np.int64),
            np.array([0.0, 1.25]))
    many = _agg.dash(xy, off, closed, st, *args)
    monkeypatch.setattr(_agg, 'SERIAL_STROKE', 1 << 40)
    one = _agg.dash(xy, off, closed, st, *args)
    for a, b in zip(many, one):
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes()


def test_seam_split_on_every_core_as_on_one(monkeypatch):
    rng = np.random.default_rng(5)
    lens = rng.integers(2, 12, 900)
    off = np.r_[0, np.cumsum(lens)].astype(np.int64)
    lon = rng.uniform(-180, 180, off[-1])
    lon[::37] = 180.0
    lat = rng.uniform(-80, 80, off[-1])
    many = _artists._split_seam(lon, lat, off, -20.0)
    monkeypatch.setattr(_agg, 'SERIAL_STROKE', 1 << 40)
    one = _artists._split_seam(lon, lat, off, -20.0)
    for a, b in zip(many, one):
        assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize('fy', [False, True])
@pytest.mark.parametrize('fx', [False, True])
def test_field64_as_flipped_copies(fy, fx):
    z = np.random.default_rng(6).normal(size=(5, 7)).astype(np.float32)
    want = z[::-1 if fy else 1, ::-1 if fx else 1].astype(np.float64)
    assert np.array_equal(_contour.field64(z, fy, fx), want)
    assert np.array_equal(_contour.field64(z.astype(np.float64), fy, fx), want)


def _grid_map(**kw):
    fig = fastplot.figure(figsize=(5, 3), dpi=80)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
    ax.set_extent([-120, -75, 25, 50], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True)
    for k, v in kw.items():
        setattr(gl, k, v)
    return fig


@pytest.mark.parametrize('change', [
    dict(xlabel_style={'size': 14}), dict(ylabel_style={'color': 'red'}), dict(top_labels=False),
    dict(xformatter='{:.1f}'), dict(rotate_labels=True), dict(xpadding=12),
    dict(xlabel_style={'weight': 'bold'})])
def test_grid_labels_kept_only_for_the_same_labels(change):
    """What a label's place depends on is in its key: each of these drawn
    after the plain map comes out as it does on an empty cache."""
    plain = _grid_map().draw().tobytes()
    assert _grid_map().draw().tobytes() == plain        # kept: the same
    after = _grid_map(**change).draw().tobytes()
    assert after != plain
    _artists._GEOM.clear()
    _artists._GEOM_BYTES[0] = 0
    assert _grid_map(**change).draw().tobytes() == after


def test_grid_labels_follow_the_font():
    plain = _grid_map().draw().tobytes()
    with fastplot.rc_context({'font.family': 'DejaVu Serif'}):
        serif = _grid_map().draw().tobytes()
    assert serif != plain and _grid_map().draw().tobytes() == plain


def test_wider_gives_the_threads_back():
    numba = pytest.importorskip('numba')
    from fastplot import _hw
    before = numba.get_num_threads()
    if before != _hw.draw_threads():
        pytest.skip('a count set by hand is left alone')
    with _jit.wider(False):
        assert numba.get_num_threads() == before
    with pytest.raises(ValueError):
        with _jit.wider():
            assert numba.get_num_threads() == min(_hw.wide_threads(),
                                                  numba.config.NUMBA_NUM_THREADS)
            raise ValueError
    assert numba.get_num_threads() == before


def test_wider_leaves_a_count_set_by_hand():
    numba = pytest.importorskip('numba')
    before = numba.get_num_threads()
    if before < 2:
        pytest.skip('one thread')
    numba.set_num_threads(before - 1)
    try:
        with _jit.wider():
            assert numba.get_num_threads() == before - 1
    finally:
        numba.set_num_threads(before)


def test_moist_lapse_kept_is_not_handed_out():
    from fastplot.skewt import _adiabats, moist_lapse
    p, t0 = np.linspace(1000.0, 100.0, 50), np.array([[260.0, 280.0], [290.0, 300.0]])
    _adiabats.cache_clear()
    first = moist_lapse(p, t0)
    want = first.copy()
    assert first.shape == (2, 2, 50)
    first[:] = 0.0                                      # theirs to change
    second = moist_lapse(p.copy(), t0.copy())
    assert second.tobytes() == want.tobytes()
    assert _adiabats.cache_info().hits == 1
    assert moist_lapse(p, t0 + 1.0).tobytes() != want.tobytes()
    assert moist_lapse(p, 280.0).shape == (50,)
    assert moist_lapse(p[:0], t0).shape == (2, 2, 0)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_contour_lines_are_of_the_field_as_given(dtype):
    """They are traced late, and z may have changed by then."""
    y, x = np.mgrid[0:40, 0:50]
    z = np.sin(x / 7.0).astype(dtype) * np.cos(y / 5.0).astype(dtype)
    def lines(field):
        fig, ax = fastplot.subplots(figsize=(3, 2), dpi=60)
        ax.contour(field, levels=[-0.5, 0.0, 0.5])
        return fig
    want = lines(z.copy()).draw().tobytes()
    fig = lines(z)
    z[:] = 0.0
    assert fig.draw().tobytes() == want


def test_traced_bounds_kept_as_worked_out():
    p = _proj.LambertConformal(central_longitude=-97.5, central_latitude=38.5)
    src, args = _proj.PlateCarree(), (-125.0, -70.0, 22.0, 50.0)
    assert _proj._traced_bounds(p, src, *args) == _proj._traced_bounds.__wrapped__(p, src, *args)
    assert _proj._traced_bounds(p, src, -120.0, -70.0, 22.0, 50.0) \
        != _proj._traced_bounds(p, src, *args)
    q = _proj.LambertConformal(central_longitude=-90.0, central_latitude=38.5)
    assert _proj._traced_bounds(q, src, *args) != _proj._traced_bounds(p, src, *args)


def _scatter(n=4000, marker='o', s=12.0, edge='face', exact=False, lim=4.0, alpha=None, seed=0,
             face=None):
    r = np.random.default_rng(seed)
    x, y = r.normal(size=(2, n))
    with fastplot.rc_context({'fastplot.exact': exact}):
        fig, ax = fastplot.subplots(figsize=(4, 3), dpi=100)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_axis_off()
        fig.subplots_adjust(0, 0, 1, 1)
        kw = {'c': x * y} if face is None else {'facecolors': face}
        ax.scatter(x, y, s=r.uniform(4, 40, n) if s is None else s, marker=marker,
                   linewidths=0.5, edgecolors=edge, alpha=alpha, **kw)
        return np.asarray(fig.draw()).astype(np.int16)


@pytest.mark.parametrize('kw', [
    dict(), dict(marker='^'), dict(marker='D', edge='k'), dict(alpha=0.5), dict(lim=1.0),
    dict(n=20000, s=None), dict(edge='none'),
    dict(edge=np.c_[np.random.default_rng(1).random((4000, 3)), np.ones(4000)], face='none')])
def test_stamps_stay_close_to_exact(kw):
    """Coverage at edges differs; no marker moves or goes missing."""
    stamped, exact = _scatter(**kw), _scatter(exact=True, **kw)
    d = np.abs(stamped - exact).max(axis=-1)
    assert d.any()                                      # (stamped indeed)
    assert d.mean() < 2.5 and d.max() <= 96
    assert (d > 64).mean() < 1e-3


@pytest.mark.parametrize('kw', [dict(n=1999), dict(marker='s'), dict(marker='+'), dict(s=0.2),
                                dict(n=2100, s=None)])
def test_markers_that_stay_exact(kw):
    assert _scatter(**kw).tobytes() == _scatter(exact=True, **kw).tobytes()


def test_exact_setting_is_the_exact_path(monkeypatch):
    want = _scatter(exact=True)
    monkeypatch.setattr(_agg, 'STAMP_MIN', 1 << 40)
    assert _scatter().tobytes() == want.tobytes()


def test_stamps_keep_the_order_markers_came_in():
    x, y = np.zeros(4000), np.zeros(4000)
    c = np.full((4000, 4), [1.0, 0.0, 0.0, 1.0])
    c[-1] = [0.0, 0.0, 1.0, 1.0]                        # the last on top of all the rest
    fig, ax = fastplot.subplots(figsize=(2, 2), dpi=100)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    ax.scatter(x, y, s=400.0, c=c, edgecolors='none')
    px = np.asarray(fig.draw())
    assert px[100, 100].tolist()[:3] == [0, 0, 255]


def test_stamps_off_the_canvas_and_not_finite():
    x = np.r_[np.full(3000, 50.0), np.linspace(-1, 1, 3000)]
    y = np.r_[np.full(3000, -50.0), np.linspace(-1, 1, 3000)]
    x[10], y[20] = np.nan, np.inf
    fig, ax = fastplot.subplots(figsize=(2, 2), dpi=100)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.scatter(x, y, s=4.0, color='k')
    assert (np.asarray(fig.draw())[:, :, :3] < 128).any()


@pytest.mark.parametrize('at', [[0], [5], [0, 0, 3, 3, 5, 5], [1, 2, 4], []])
def test_insert_points_as_numpy_inserts(at):
    at = np.array(at, np.int64)
    X, Y = np.arange(5.0), np.arange(5.0) * 2
    px, py = 100.0 + np.arange(at.size), 200.0 + np.arange(at.size)
    outx, outy = np.empty(X.size + at.size), np.empty(Y.size + at.size)
    _proj._insert_points(X, Y, at, px, py, outx, outy)
    assert outx.tolist() == np.insert(X, at, px).tolist()
    assert outy.tolist() == np.insert(Y, at, py).tolist()


def test_long_segments_as_numpy_finds_them():
    r = np.random.default_rng(3)
    X, Y = r.normal(size=(2, 4001))
    X[r.integers(0, X.size, 40)] = np.nan
    Y[r.integers(0, Y.size, 40)] = np.inf
    th2 = 1.5
    got = np.empty(X.size - 1, np.bool_)
    _proj._long_segments(X, Y, th2, got)
    with np.errstate(invalid='ignore'):
        assert (got == (np.diff(X) ** 2 + np.diff(Y) ** 2 > th2)).all()


@pytest.mark.parametrize('w', [0.0, 0.4, 0.5, 0.75, 1.0, 1.49, 1.5, 2.0, 2.5, 3.0, 1e14])
def test_snap_value_of_one_width_as_of_many(w):
    assert float(_draw.snap_value(float(w))) == float(_draw.snap_value(np.array([w]))[0])
    assert float(_draw.snap_value(np.float64(w))) == float(_draw.snap_value(float(w)))
