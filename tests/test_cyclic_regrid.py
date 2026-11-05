"""Global grids with a cyclic column (0..360 with 360 repeating 0, as
cartopy's add_cyclic_point makes them), contour lines crossing a map's seam
through a grid longitude, and barbs / quiver regridded onto the map
(cartopy's regrid_shape)."""
import warnings

import numpy as np
import pytest

import fastplot as fp
import fastplot.crs as ccrs
from fastplot import _artists

D = 1.40625
LON = np.arange(0, 360, D)
LAT = np.linspace(-89, 89, 128)


def _field(lon, lat):
    X, Y = np.meshgrid(lon, lat)
    return np.cos(np.deg2rad(Y)) * (1 + 0.5 * np.sin(np.deg2rad(3 * X))) * 30


def _cyclic(lon, z):
    return np.append(lon, lon[0] + 360.0), np.concatenate([z, z[:, :1]], axis=1)


def _map(proj):
    fig = fp.figure(figsize=(6, 3.4), dpi=80)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96], projection=proj)
    ax.set_global()
    return fig, ax


@pytest.mark.parametrize('proj', [ccrs.Robinson(), ccrs.PlateCarree(),
                                  ccrs.Robinson(central_longitude=180)],
                         ids=['robinson', 'platecarree', 'robinson180'])
@pytest.mark.parametrize('start', [0.0, -180.0])
def test_cyclic_column_draws_as_the_grid_without_it(proj, start):
    lon = LON + start
    z = _field(lon, LAT)
    lonc, zc = _cyclic(lon, z)
    out = []
    for x, f in ((lon, z), (lonc, zc)):
        fig, ax = _map(proj)
        ax.contourf(x, LAT, f, levels=np.arange(0, 50, 2), cmap='viridis',
                    transform=ccrs.PlateCarree())
        ax.contour(x, LAT, f, levels=np.arange(0, 50, 6), colors='k', linewidths=0.7,
                   transform=ccrs.PlateCarree())
        out.append(np.asarray(fig.draw()).copy())
    assert np.array_equal(out[0], out[1])


def test_cyclic_column_no_blank_line():
    z = _field(LON, LAT)
    lonc, zc = _cyclic(LON, z)
    fig, ax = _map(ccrs.Robinson())
    ax.contourf(lonc, LAT, zc, levels=np.arange(0, 50, 2), cmap='viridis',
                transform=ccrs.PlateCarree())
    a = np.asarray(fig.draw())
    h = a.shape[0]
    mid = a[h // 2 - 20:h // 2 + 20]                    # rows round the equator, all on the map
    x0, x1 = int(ax.box_f[0]) + 3, int(ax.box_f[2]) - 3
    white = (mid[:, x0:x1, :3] == 255).all(axis=2)
    assert not white.any()


def test_cyclic_grid_takes_full_width_arrays_again():
    z = _field(LON, LAT)
    lonc, zc = _cyclic(LON, z)
    fig, ax = _map(ccrs.Robinson())
    cf = ax.contourf(lonc, LAT, zc, levels=np.arange(0, 50, 2), transform=ccrs.PlateCarree())
    assert cf.grid.cyclic and cf.z.shape == z.shape
    cf.set_array(zc * 0.5)
    assert cf.z.shape == z.shape
    pm = ax.pcolormesh(lonc, LAT, zc, transform=ccrs.PlateCarree())
    assert pm.z.shape == z.shape
    fig.draw()


def test_cyclic_column_flipped_longitudes():
    z = _field(LON, LAT)
    lonc, zc = _cyclic(LON, z)
    out = []
    for x, f in ((LON, z), (lonc[::-1], zc[:, ::-1])):
        fig, ax = _map(ccrs.Robinson())
        ax.contourf(x, LAT, f, levels=np.arange(0, 50, 2), transform=ccrs.PlateCarree())
        out.append(np.asarray(fig.draw()).copy())
    assert np.array_equal(out[0], out[1])


def test_regional_grid_keeps_every_column():
    lon = np.arange(0, 300, D)
    fig, ax = _map(ccrs.Robinson())
    cf = ax.contourf(lon, LAT, _field(lon, LAT), transform=ccrs.PlateCarree())
    assert not cf.grid.cyclic and cf.z.shape == (LAT.size, lon.size)


@pytest.mark.parametrize('direction', [1, -1], ids=['east', 'west'])
def test_line_through_a_seam_point_is_cut_there(direction):
    lon = np.arange(0, 360 + D / 2, D)[::direction]     # 180 E is a point of the line
    lat = np.full(lon.size, 78.0)
    off = np.array([0, lon.size])
    lo, la, o, w = _artists._split_seam(lon, lat, off, 0.0, False, (-90.0, 90.0), True)
    assert o.size - 1 == 2
    for a, b in zip(o[:-1], o[1:]):
        assert np.abs(np.diff(lo[a:b])).max() < 2 * D
    fig, ax = _map(ccrs.Robinson())
    fig.draw()
    xy, poff = ax._lines_px(lon, lat, off, ccrs.PlateCarree())
    wpx = ax.box_f[2] - ax.box_f[0]
    for a, b in zip(poff[:-1], poff[1:]):
        assert np.hypot(*np.diff(xy[a:b], axis=0).T).max() < 0.1 * wpx


def test_contour_rings_round_the_poles_no_streaks():
    z = _field(LON, LAT)
    fig, ax = _map(ccrs.Robinson())
    cs = ax.contour(LON, LAT, z, levels=np.arange(0, 50, 6), colors='k',
                    transform=ccrs.PlateCarree())
    fig.draw()
    xy, poff, lof, _, _ = cs._px_lines(None, ax, None)
    wpx = ax.box_f[2] - ax.box_f[0]
    for a, b in zip(poff[:-1], poff[1:]):
        assert np.hypot(*np.diff(xy[a:b], axis=0).T).max() < 0.1 * wpx


# -- regrid_shape

def _winds(lon, lat):
    X, Y = np.meshgrid(lon, lat)
    return 20 * np.cos(np.deg2rad(Y)), 5 * np.sin(np.deg2rad(X))


def test_barbs_regrid_shape_as_cartopy():
    u, v = _winds(LON, LAT)
    fig, ax = _map(ccrs.Robinson())
    b = ax.barbs(LON, LAT, u, v, transform=ccrs.PlateCarree(), regrid_shape=20, length=4)
    x0, x1, y0, y1 = ax.get_extent()
    assert (x1 - x0) > (y1 - y0)
    nx = int(20 * (x1 - x0) / (y1 - y0))
    assert b.x.size == nx * 20 == 780                   # cartopy's 39 x 20
    ok = np.isfinite(b.u)
    assert 0.6 * b.x.size < ok.sum() < b.x.size         # (the corners off the oval: NaN)
    fig.draw()


def test_barbs_regrid_tuple_and_target_extent():
    u, v = _winds(LON, LAT)
    fig, ax = _map(ccrs.PlateCarree())
    b = ax.barbs(LON, LAT, u, v, transform=ccrs.PlateCarree(), regrid_shape=(12, 7),
                 target_extent=(-60, 60, -30, 30))
    assert b.x.size == 84
    X, Y = np.meshgrid(np.linspace(-60, 60, 12), np.linspace(-30, 30, 7))
    assert np.allclose(b.x, X.ravel()) and np.allclose(b.y, Y.ravel())


def test_regrid_interpolates_the_field_and_wraps():
    lon = LON
    X, Y = np.meshgrid(lon, LAT)
    u = Y * 0.5 + 3.0                                   # linear in latitude: bilinear exactly
    v = np.cos(np.deg2rad(X))                           # smooth across 0 / 360
    fig, ax = _map(ccrs.PlateCarree())
    b = ax.barbs(lon, LAT, u, v, transform=ccrs.PlateCarree(), regrid_shape=(73, 31),
                 target_extent=(-180, 180, -60, 60))
    ok = np.isfinite(b.u)
    assert ok.all()                                     # every longitude, -0.7 .. 0 as well
    assert np.allclose(b.u, b.y * 0.5 + 3.0)
    assert np.allclose(b.v, np.cos(np.deg2rad(b.x)), atol=2e-3)


def test_regrid_off_a_regional_grid_is_nan():
    lon = np.arange(-120, -60, 0.5)
    lat = np.arange(20, 50, 0.5)
    X, Y = np.meshgrid(lon, lat)
    fig, ax = _map(ccrs.PlateCarree())
    b = ax.barbs(lon, lat, X * 0 + 10, Y * 0, transform=ccrs.PlateCarree(), regrid_shape=10)
    inside = (b.x >= lon[0]) & (b.x <= lon[-1]) & (b.y >= lat[0]) & (b.y <= lat[-1])
    assert np.array_equal(np.isfinite(b.u), inside)
    assert np.allclose(b.u[inside], 10.0)


def test_regrid_2d_coordinates():
    ii, jj = np.meshgrid(np.arange(60), np.arange(40))
    lon2 = -125 + ii * 1.0 + jj * 0.2
    lat2 = 20 + jj * 0.9 - ii * 0.05
    fig, ax = _map(ccrs.LambertConformal())
    ax.set_extent([-120, -70, 22, 52], ccrs.PlateCarree())
    b = ax.barbs(lon2, lat2, lon2 * 0 + 10, lat2 * 0 + 5, transform=ccrs.PlateCarree(),
                 regrid_shape=12)
    ok = np.isfinite(b.u)
    assert ok.sum() > 0.5 * b.x.size
    assert np.allclose(b.u[ok], 10.0) and np.allclose(b.v[ok], 5.0)
    fig.draw()


def test_regrid_2d_matches_linear_interpolation_in_the_map():
    """The grid's own cells, triangulated, give what cartopy's Delaunay of
    the projected points gives inside the grid."""
    from fastplot import _proj, _regrid
    from fastplot._tri import Triangulation
    from fastplot._triinterp import LinearTriInterpolator
    ii, jj = np.meshgrid(np.arange(120), np.arange(80))
    lon2 = -125 + ii * 0.5 + jj * 0.05
    lat2 = 20 + jj * 0.375 - ii * 0.025
    f = np.sin(np.deg2rad(lon2) * 3) * 20 + lat2
    L, pc = ccrs.LambertConformal(), ccrs.PlateCarree()
    ext = (-2.5e6, 2.5e6, -1.5e6, 1.5e6)
    _, _, (new,) = _regrid.regrid(L, pc, (31, 19), ext, lon2, lat2, [f])
    X, Y = np.meshgrid(np.linspace(ext[0], ext[1], 31), np.linspace(ext[2], ext[3], 19))
    px, py = _proj.transform_points(pc, L, lon2.ravel(), lat2.ravel())
    old = np.ma.filled(LinearTriInterpolator(Triangulation(px, py), f.ravel())(X, Y)
                       .astype(float), np.nan)
    both = np.isfinite(new) & np.isfinite(old)
    assert both.sum() > 0.5 * new.size
    assert np.isfinite(old[np.isfinite(new)]).all()     # (the hull holds the grid)
    assert np.abs(new - old)[both].max() < 1e-3 * np.ptp(f)


def test_regrid_plan_is_made_once(monkeypatch):
    from fastplot import _regrid
    made = []
    real = _regrid._plan
    monkeypatch.setattr(_regrid, '_plan', lambda *a: made.append(1) or real(*a))
    ii, jj = np.meshgrid(np.arange(50), np.arange(40))
    lon2 = -110 + ii * 0.5 + jj * 0.05
    lat2 = 30 + jj * 0.4 - ii * 0.02
    fig, ax = _map(ccrs.LambertConformal())
    ax.set_extent([-110, -80, 28, 48], ccrs.PlateCarree())
    for k in range(3):
        b = ax.barbs(lon2, lat2, lon2 * 0 + k, lat2 * 0 + 1.0, transform=ccrs.PlateCarree(),
                     regrid_shape=10)
        ok = np.isfinite(b.u)
        assert ok.any() and np.allclose(b.u[ok], k)
    assert len(made) == 1
    b = ax.barbs(lon2 + 1e-3, lat2, lon2 * 0, lat2 * 0, transform=ccrs.PlateCarree(),
                 regrid_shape=10)                       # other coordinates: made again
    assert len(made) == 2


def test_regrid_2d_global_grid_across_the_seam():
    """A global curvilinear grid: its cells across the map's seam span the
    map, and are left out, not read across it."""
    lon = np.arange(-180, 180, 2.0)
    lat = np.arange(-80, 81, 2.0)
    LON, LAT = np.meshgrid(lon, lat)
    lon2 = LON + 0.01 * LAT                             # (not 1-D axes: the cells' path)
    f = np.sin(np.deg2rad(lon2) / 2)                    # -1 .. 1 jumping at the seam
    fig, ax = _map(ccrs.Robinson())
    b = ax.barbs(lon2, LAT, f, f * 0, transform=ccrs.PlateCarree(), regrid_shape=25)
    ok = np.isfinite(b.u)
    assert ok.sum() > 0.5 * b.x.size
    exact = np.sin(np.deg2rad(((b.x + 180) % 360) - 180) / 2)
    assert np.abs(b.u[ok] - exact[ok]).max() < 0.02


def test_quiver_regrid_with_colours():
    u, v = _winds(LON, LAT)
    fig, ax = _map(ccrs.Robinson())
    q = ax.quiver(LON, LAT, u, v, np.hypot(u, v), transform=ccrs.PlateCarree(),
                  regrid_shape=15)
    x0, x1, y0, y1 = ax.get_extent()
    assert q.N == int(15 * (x1 - x0) / (y1 - y0)) * 15
    assert q.values is not None and q.mask.any() and not q.mask.all()
    fig.draw()


def test_regrid_shape_on_plain_axes_warns():
    fig = fp.figure()
    ax = fig.add_subplot()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        b = ax.barbs(np.arange(3.0), np.arange(3.0), np.ones(3), np.ones(3), regrid_shape=5)
    assert b.x.size == 3
    assert any('regrid_shape' in str(m.message) for m in w)
