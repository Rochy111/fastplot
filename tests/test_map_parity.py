"""Maps pixel for pixel against matplotlib with cartopy: meshes on leaning
cells, labelled contours, gridlines and their labels, Natural Earth lines."""
import io
import os

import numpy as np
import pytest

import fastplot.crs as fcrs
import fastplot.feature as ffeature
import fastplot.pyplot as fplt

matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')
ccrs = pytest.importorskip('cartopy.crs')
cfeature = pytest.importorskip('cartopy.feature')
from PIL import Image                                                    # noqa: E402
import matplotlib.pyplot as mplt                                         # noqa: E402

needs_311 = pytest.mark.skipif(tuple(int(v) for v in matplotlib.__version__.split('.')[:2]) < (3, 11),
                               reason='matplotlib 3.11 lays text out differently')


def lambert(c):
    return c.LambertConformal(-97.5, 38.5, standard_parallels=(38.5, 38.5))


@pytest.mark.parametrize('proj, extent, src, xs, ys', [
    (lambert, [-125, -70, 22, 50], 'PlateCarree', [-120, -75, -80, -118], [30, 32, 47, 45]),
    (lambda c: c.Mercator(), [-130, 10, 20, 70], 'Geodetic', [-120, 0], [35, 52]),
    (lambda c: c.PlateCarree(), [-130, 10, 20, 70], 'Geodetic', [-120, 0], [35, 52]),
    (lambda c: c.NorthPolarStereo(), [-180, 180, 40, 90], 'PlateCarree', [-120, 60, 100],
     [50, 60, 50]),
    (lambda c: c.Robinson(), [-170, 170, -60, 80], 'PlateCarree', [-150, 150, 100, -100],
     [60, 60, -40, -40]),
    (lambda c: c.PlateCarree(), [-180, 180, -60, 80], 'PlateCarree', [170, -170, -170, 170],
     [10, 10, 40, 50]),
    (lambda c: c.Mercator(), [-180, 180, -60, 70], 'PlateCarree', [170, -170, 150], [10, 10, 40]),
], ids=['lambert', 'mercator-geodetic', 'plate-geodetic', 'polar', 'robinson', 'dateline',
        'mercator-dateline'])
def test_long_lines_bend_as_cartopy_bends_them(proj, extent, src, xs, ys):
    """project_linear: chords as long as the threshold lets them be, on
    straight lines of the source or geodesics of a Geodetic; degrees of a
    PlateCarree run on, whatever the shorter way round."""
    out = []
    for plt, crs in ((mplt, ccrs), (fplt, fcrs)):
        fig = plt.figure(figsize=(6, 4), dpi=100)
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection=proj(crs))
        ax.set_extent(extent, crs=crs.PlateCarree())
        ax.plot(xs, ys, transform=getattr(crs, src)(), color='k', lw=1)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close('all')
        buf.seek(0)
        out.append(np.asarray(Image.open(buf).convert('RGB')).astype(np.int16))
    d = np.abs(out[0] - out[1]).max(axis=2)
    assert (d > 64).mean() == 0.0 and d.mean() < 0.02, ((d > 64).mean(), d.mean())


def test_a_patch_bends_too():
    import fastplot.patches as fpatches
    import matplotlib.patches as mpatches
    out = []
    for plt, crs, patches in ((mplt, ccrs, mpatches), (fplt, fcrs, fpatches)):
        fig = plt.figure(figsize=(6, 4), dpi=100)
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection=lambert(crs))
        ax.set_extent([-125, -70, 22, 50], crs=crs.PlateCarree())
        ax.add_patch(patches.Rectangle((-110, 30), 30, 12, facecolor='C1', edgecolor='k',
                                       transform=crs.PlateCarree()))
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close('all')
        buf.seek(0)
        out.append(np.asarray(Image.open(buf).convert('RGB')).astype(np.int16))
    d = np.abs(out[0] - out[1]).max(axis=2)
    assert (d > 64).mean() < 0.0005 and d.mean() < 0.1, ((d > 64).mean(), d.mean())


def test_geodesics_are_vincenty():
    pyproj = pytest.importorskip('pyproj')
    from fastplot import _proj
    g = pyproj.Geod(a=6378137.0, f=1 / 298.257223563)
    r = np.random.default_rng(0)
    lo0, la0 = r.uniform(-180, 180, 500), r.uniform(-80, 80, 500)
    lo1, la1 = lo0 + r.uniform(-120, 120, 500), r.uniform(-80, 80, 500)
    azim, _, s12 = g.inv(lo0, la0, lo1, la1)
    a, s = _proj.geodesic_inverse(lo0, la0, lo1, la1)
    assert np.abs(s - s12).max() < 1e-3
    lon, lat, _ = g.fwd(lo0, la0, azim, s12 * 0.37)
    x, y = _proj.geodesic_direct(lo0, la0, a, s * 0.37)
    assert np.abs((x - lon + 180) % 360 - 180).max() < 1e-8 and np.abs(y - lat).max() < 1e-8


def albers(c):
    return c.AlbersEqualArea(-96, 37.5, standard_parallels=(29.5, 45.5))


def polar(c):
    return c.NorthPolarStereo(central_longitude=-100)


def mercator(c):
    return c.Mercator()


def plate(c):
    return c.PlateCarree()


MAPS = {'lambert': (lambert, [-112, -83, 30, 46]), 'wide': (lambert, [-125, -66, 22, 52]),
        'albers': (albers, [-120, -75, 25, 50]), 'polar': (polar, [-140, -60, 30, 60]),
        'mercator': (mercator, [-120, -70, 22, 50]), 'plate': (plate, [-120, -70, 22, 50])}

LON, LAT = np.meshgrid(np.linspace(-130, -60, 141), np.linspace(20, 55, 71))


def _field():
    z = (np.sin(np.radians(LON) * 7.0) * np.cos(np.radians(LAT) * 9.0) +
         0.5 * np.cos(np.radians(LON) * 13.0 + np.radians(LAT) * 5.0))
    return (z - z.min()) / np.ptp(z)


def _both(name, draw, pos=(0.05, 0.05, 0.9, 0.9)):
    proj, extent = MAPS[name]
    out = []
    for plt, crs, feature in ((mplt, ccrs, cfeature), (fplt, fcrs, ffeature)):
        fig = plt.figure(figsize=(8, 5), dpi=100)
        ax = fig.add_axes(list(pos), projection=proj(crs))
        ax.set_extent(extent, crs=crs.PlateCarree())
        draw(ax, crs, feature)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close('all')
        buf.seek(0)
        out.append(np.asarray(Image.open(buf).convert('RGB')).astype(np.int16))
    return np.abs(out[0] - out[1]).max(axis=2)


@pytest.mark.parametrize('name', ['lambert', 'wide', 'albers', 'polar', 'mercator'])
def test_rectilinear_mesh_on_a_map(name):
    """A lon/lat grid on 1-D axes draws as a QuadMesh does, by cells, on
    every projection that leans them."""
    z = _field()

    def draw(ax, crs, _):
        ax.pcolormesh(LON[0], LAT[:, 0], z, transform=crs.PlateCarree(), cmap='viridis', vmin=0,
                      vmax=1)
    d = _both(name, draw)
    assert (d > 64).mean() == 0 and d.mean() < 0.02


def test_flipped_rectilinear_mesh_on_a_map():
    """Latitudes running north to south, as GRIB files keep them."""
    z = _field()

    def draw(ax, crs, _):
        ax.pcolormesh(LON[0], LAT[::-1, 0], z[::-1], transform=crs.PlateCarree(), cmap='viridis')
    d = _both('lambert', draw)
    assert (d > 64).mean() == 0 and d.mean() < 0.02


@needs_311
@pytest.mark.parametrize('name', list(MAPS))
def test_contour_labels_on_a_map(name):
    """Labels are chosen in the box as it is before the aspect fits it, and
    those off the map are clipped."""
    z = _field()

    def draw(ax, crs, _):
        cs = ax.contour(LON, LAT, z, levels=np.linspace(0.1, 0.9, 9), colors='k', linewidths=0.8,
                        transform=crs.PlateCarree())
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
    d = _both(name, draw)
    assert (d > 64).mean() < 0.0005 and d.mean() < 0.1


@needs_311
@pytest.mark.parametrize('name', list(MAPS))
def test_gridline_labels_on_a_map(name):
    """Labels along curved lines, moved clear of the frame, inline on
    Lambert and polar maps; lines straight on screen snapped."""
    def draw(ax, crs, _):
        ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
    d = _both(name, draw, (0.1, 0.1, 0.8, 0.8))
    assert (d > 64).mean() == 0 and d.mean() < 0.05


@needs_311
def test_gridline_on_the_edge_of_a_regional_map():
    """A meridian on the map's edge, and the labels either side of it."""
    MAPS['edge'] = (plate, [-95, -80, 37.5, 48])

    def draw(ax, crs, _):
        ax.gridlines(draw_labels=True, xlocs=np.arange(-94, -80 + 1e-6, 2), linewidth=0.5)
    assert (_both('edge', draw, (0.1, 0.1, 0.8, 0.8)) > 64).mean() == 0


@needs_311
def test_gridline_labels_chosen_sides():
    def draw(ax, crs, _):
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
        gl.top_labels = gl.right_labels = False
    assert (_both('plate', draw, (0.1, 0.1, 0.8, 0.8)) > 64).mean() == 0


@needs_311
def test_gridline_labels_by_list_and_not_inline():
    def draw(ax, crs, _):
        ax.gridlines(draw_labels=['bottom', 'left'], x_inline=False, y_inline=False,
                     rotate_labels=False, linewidth=0.5, color='gray')
    assert (_both('lambert', draw, (0.1, 0.1, 0.8, 0.8)) > 64).mean() == 0


def _have_50m():
    base = os.path.join(os.path.expanduser('~'), '.local', 'share', 'cartopy', 'shapefiles',
                        'natural_earth')
    return os.path.exists(os.path.join(base, 'physical', 'ne_50m_coastline.shp')) and \
        os.path.exists(os.path.join(base, 'cultural',
                                    'ne_50m_admin_1_states_provinces_lakes.shp'))


@pytest.mark.skipif(not _have_50m(), reason="cartopy's Natural Earth files are not here")
def test_coastlines_and_states():
    """Lines dense enough to meet themselves in a pixel, on a Lambert map."""
    def draw(ax, _, feature):
        ax.add_feature(feature.COASTLINE.with_scale('50m'), linewidth=0.8)
        ax.add_feature(feature.STATES.with_scale('50m'), linewidth=0.4)
    d = _both('lambert', draw)
    assert (d > 64).mean() == 0 and d.mean() < 0.01
