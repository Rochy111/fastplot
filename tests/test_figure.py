"""Whole figures: the layout matplotlib + cartopy give (axes and colorbar
positions), the pixels they give (field colours), and the GPU drawing the
CPU's image byte for byte."""
import io

import numpy as np
import pytest

import fastplot as fp
import fastplot.crs as fcrs
import fastplot.feature as ffeature
from fastplot import _device


def _gpu_ok():
    return _device.gpu_usable()


def _field(ny=181, nx=360):
    lat = np.linspace(-90, 90, ny)
    lon = np.linspace(0, 359, nx)
    z = (30 * np.cos(np.radians(lat))[:, None] * (1 + 0.3 * np.sin(np.radians(3 * lon)))[None]
         - 15).astype(np.float32)
    return lon, lat, z


def _mpl():
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ccrs = pytest.importorskip('cartopy.crs')
    return plt, ccrs


# ------------------------------------------------------------------ layout

LAYOUTS = [
    dict(figsize=(12, 6.5), dpi=150, cbar=dict(orientation='horizontal', pad=0.06, shrink=0.8)),
    dict(figsize=(10, 6), dpi=100, cbar=dict()),
    dict(figsize=(8, 8), dpi=100, cbar=dict(orientation='vertical', shrink=0.7, aspect=30)),
    dict(figsize=(9, 5), dpi=120, cbar=None),
    dict(figsize=(10, 7), dpi=100, cbar=dict(location='left', pad=0.1)),
]


@pytest.mark.parametrize('cfg', LAYOUTS)
@pytest.mark.parametrize('proj', ['pc', 'lcc'])
@pytest.mark.parametrize('extend', ['neither', 'both'])
def test_layout_matches_cartopy(cfg, proj, extend):
    plt, ccrs = _mpl()
    lon, lat, z = _field()
    out = []
    for lib in ('mpl', 'fp'):
        crs = ccrs if lib == 'mpl' else fcrs
        fig = (plt.figure if lib == 'mpl' else fp.figure)(figsize=cfg['figsize'], dpi=cfg['dpi'])
        p = crs.PlateCarree() if proj == 'pc' else crs.LambertConformal(central_longitude=-97,
                                                                         central_latitude=38)
        ax = fig.add_subplot(1, 1, 1, projection=p)
        if proj == 'lcc':
            ax.set_extent([-125, -66, 22, 51], crs=crs.PlateCarree())
        else:
            ax.set_global()
        m = ax.contourf(lon, lat, z, levels=np.arange(-15, 20, 2.5), extend=extend,
                        transform=crs.PlateCarree())
        cb = fig.colorbar(m, ax=ax, **cfg['cbar']) if cfg['cbar'] is not None else None
        if lib == 'mpl':
            fig.canvas.draw()
            pos = [ax.get_position().bounds] + ([cb.ax.get_position().bounds] if cb else [])
            plt.close(fig)
        else:
            fig.draw(device='cpu')
            pos = [tuple(ax.get_position())] + ([tuple(cb.ax.get_position())] if cb else [])
        out.append(np.asarray(pos))
    # cartopy's set_extent bounds a curved (LCC) outline from a coarse
    # interpolation that cuts its edges short by ~6e-4 of the span; fastplot's
    # are exact, so a conic map's aspect differs by ~0.3% (a few pixels)
    tol = 2e-4 if proj == 'pc' else 3e-3
    assert np.allclose(out[0], out[1], atol=tol), (out[0], out[1])


def test_subplots_grid_matches_matplotlib():
    plt, ccrs = _mpl()
    res = []
    for lib in ('mpl', 'fp'):
        crs = ccrs if lib == 'mpl' else fcrs
        sub = plt.subplots if lib == 'mpl' else fp.subplots
        fig, axs = sub(2, 3, figsize=(12, 7), dpi=100,
                       subplot_kw={'projection': crs.PlateCarree()})
        for ax in axs.ravel():
            ax.set_extent([-130, -60, 20, 55], crs=crs.PlateCarree())
        if lib == 'mpl':
            fig.canvas.draw()
            res.append(np.array([a.get_position().bounds for a in axs.ravel()]))
            plt.close(fig)
        else:
            fig.draw(device='cpu')
            res.append(np.array([tuple(a.get_position()) for a in axs.ravel()]))
    assert np.allclose(res[0], res[1], atol=2e-4)


# ------------------------------------------------------------------ pixels

def _render_both(kind):
    plt, ccrs = _mpl()
    lon, lat, z = _field()
    imgs = []
    for lib in ('mpl', 'fp'):
        crs = ccrs if lib == 'mpl' else fcrs
        fig = (plt.figure if lib == 'mpl' else fp.figure)(figsize=(8, 4.5), dpi=100)
        ax = fig.add_subplot(1, 1, 1, projection=crs.PlateCarree())
        ax.set_global()
        if kind == 'contourf':
            ax.contourf(lon, lat, z, levels=np.arange(-15, 20, 2.5), cmap='viridis',
                        extend='both', transform=crs.PlateCarree())
        else:
            ax.pcolormesh(lon, lat, z, cmap='viridis', vmin=-15, vmax=16, shading='auto',
                          transform=crs.PlateCarree())
        buf = io.BytesIO()
        fig.savefig(buf, format='png') if lib == 'mpl' else None
        if lib == 'mpl':
            from PIL import Image
            imgs.append(np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert('RGBA')))
            box = ax.get_position().bounds
            plt.close(fig)
        else:
            imgs.append(fig.draw(device='cpu'))
    return imgs, box


@pytest.mark.parametrize('kind', ['contourf', 'pcolormesh'])
def test_field_pixels_match_cartopy(kind):
    (a, b), box = _render_both(kind)
    assert a.shape == b.shape
    H, W = a.shape[:2]
    l, bt, w, h = box
    x0, x1 = int(l * W) + 3, int((l + w) * W) - 3
    y0, y1 = int((1 - bt - h) * H) + 3, int((1 - bt) * H) - 3
    da = a[y0:y1, x0:x1, :3].astype(int)
    db = b[y0:y1, x0:x1, :3].astype(int)
    same = (np.abs(da - db).max(axis=2) == 0).mean()
    # identical colours away from band edges and cell edges (anti-aliasing,
    # sub-pixel geometry), so most of the map
    assert same > 0.93, same


# -------------------------------------------------------------- CPU == GPU

def _full_figure(proj='pc', curvi=False):
    lon, lat, z = _field()
    fig = fp.figure(figsize=(9, 5), dpi=100)
    p = fcrs.PlateCarree() if proj == 'pc' else fcrs.LambertConformal()
    ax = fig.add_subplot(1, 1, 1, projection=p)
    if proj != 'pc':
        ax.set_extent([-125, -66, 22, 51], crs=fcrs.PlateCarree())
    pc = fcrs.PlateCarree()
    if curvi:
        LO, LA = np.meshgrid(lon, lat)
        m = ax.pcolormesh(LO + 0.01 * LA, LA, z, transform=pc, cmap='turbo')
    else:
        m = ax.contourf(lon, lat, z, levels=12, cmap='turbo', extend='both', transform=pc)
    ax.pcolormesh(lon, lat, z, transform=pc, alpha=0.3, cmap='gray', norm='log')
    cs = ax.contour(lon, lat, z, levels=6, colors='k', transform=pc)
    ax.clabel(cs, fontsize=8)
    s = slice(None, None, 15)
    ax.barbs(lon[s], lat[s], z[s, s], -z[s, s], transform=pc, length=5)
    ax.coastlines()
    ax.add_feature(ffeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True)
    ax.text(-100, 40, 'H', transform=pc, fontsize=14, color='b', fontweight='bold')
    ax.set_title('parity', loc='left')
    fig.colorbar(m, ax=ax, label='$T$ ($^{\\circ}$C)')
    fig.text(0.99, 0.01, 'fastplot', ha='right', fontsize=8)
    return fig


@pytest.mark.parametrize('proj,curvi', [('pc', False), ('pc', True), ('lcc', False),
                                        ('lcc', True)])
def test_gpu_draws_the_cpu_image(proj, curvi):
    if not _gpu_ok():
        pytest.skip('no usable GPU')
    fig = _full_figure(proj, curvi)
    a = fig.draw(device='cpu')
    b = fig.draw(device='gpu')
    assert np.array_equal(a, b)


def test_gpu_failure_falls_back(monkeypatch):
    """A GPU that fails mid-draw (here: always) gives the CPU's image, a
    warning once, and the CPU from then on."""
    pytest.importorskip('cupy')
    fig = _full_figure()
    ref = fig.draw(device='cpu')
    from fastplot import _gpu

    def broken(self, *a, **k):
        raise RuntimeError('simulated GPU failure')

    monkeypatch.setattr(_gpu.GpuCanvas, '__init__', broken)
    monkeypatch.setattr(_device, '_usable', True)
    monkeypatch.setattr(_device, '_device', 'gpu')
    with pytest.warns(RuntimeWarning, match='drawing on the CPU instead'):
        img = fig.draw()
    assert np.array_equal(img, ref)
    assert _device.gpu_usable() is False


# ------------------------------------------------------------------ saving

def test_savefig_formats_and_tight(tmp_path):
    fig = _full_figure()
    from PIL import Image
    for ext in ('png', 'jpg', 'webp'):
        p = tmp_path / ('f.' + ext)
        fig.savefig(p, device='cpu')
        im = Image.open(p)
        assert im.size == (900, 500)
    p = tmp_path / 'tight.png'
    fig.savefig(p, bbox_inches='tight', device='cpu')
    w, h = Image.open(p).size
    assert w < 900 and h < 500
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=50, device='cpu')
    assert Image.open(io.BytesIO(buf.getvalue())).size == (450, 250)


def test_animation_frames_reuse(tmp_path):
    lon, lat, z = _field()
    fig = fp.figure(figsize=(6, 4), dpi=80)
    ax = fig.add_subplot(projection=fcrs.PlateCarree())
    m = ax.pcolormesh(lon, lat, z, vmin=-15, vmax=20, transform=fcrs.PlateCarree())
    ax.coastlines()
    a = fig.draw(device='cpu')
    m.set_array(z + 5)
    b = fig.draw(device='cpu')
    assert not np.array_equal(a, b)
    m.set_array(z)
    assert np.array_equal(fig.draw(device='cpu'), a)


def test_global_autoscale_and_wrapped_points():
    """A 0..360 global grid with no set_extent fills the whole map, and
    points east of 180 (barbs, scatter) land west of it, as in cartopy."""
    lon, lat, z = _field()
    pc = fcrs.PlateCarree()
    fig = fp.figure(figsize=(6, 3), dpi=100)
    ax = fig.add_subplot(projection=pc)
    ax.pcolormesh(lon, lat, z, transform=pc)
    s = slice(None, None, 30)
    ax.barbs(lon[s], lat[s], np.full((lat[s].size, lon[s].size), 30.0),
             np.zeros((lat[s].size, lon[s].size)), transform=pc)
    ax.scatter([350.0], [0.0], s=400, color='red', transform=pc)
    img = fig.draw(device='cpu')
    assert ax.get_extent() == (-180.0, 180.0, -90.0, 90.0)
    x0, y0, x1, y1 = ax.box
    # the red marker at 350 E = 10 W: just left of the map's centre
    cx = x0 + (x1 - x0) * (170.0 / 360.0)
    cy = (y0 + y1) / 2
    px = img[int(cy), int(round(cx))]
    assert px[0] > 200 and px[1] < 60 and px[2] < 60
