"""What forecast maps ask of fastplot (maps with hodograph insets, RGBA
meshes, lon/lat ticks, shapefile counties): each against matplotlib and
cartopy, or against the workaround it replaces."""
import os
import struct
import tempfile

import numpy as np
import pytest

import fastplot
import fastplot.crs as fcrs
import fastplot.feature as ffeature
import fastplot.patheffects as fpe
import fastplot.pyplot as fplt
from fastplot import _agg, _artists

matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')
import matplotlib.patheffects as mpe                                     # noqa: E402
import matplotlib.pyplot as mplt                                         # noqa: E402


def _rgb(fig):
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.int16).copy()


def _mpl_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.int16).copy()


# ------------------------------------------------------------- limits

def test_set_limits_with_both_sides_does_not_autoscale(monkeypatch):
    """Both limits given: the old ones are not worked out (that went
    through every artist's data bounds)."""
    fig = fastplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot([0, 5], [0, 7])
    called = []
    orig = type(ax)._autoscaled
    monkeypatch.setattr(type(ax), '_autoscaled', lambda self, n: called.append(n) or orig(self, n))
    assert ax.set_xlim(-40, 40) == (-40, 40)
    assert ax.set_ylim(-40, 40) == (-40, 40)
    assert called == []
    assert ax.set_xlim(None, 10) == (-40, 10)           # a side missing: the old one kept
    ax.set_yscale('log')
    assert ax.set_ylim(-1, 100)[1] == 100               # (log: a bound <= 0 ignored, as before)


# ---------------------------------------------------- hodograph insets

def _hodographs(plt, Circle, withStroke):
    fig = plt.figure(figsize=(6, 4), dpi=100)
    fig.patch.set_facecolor('#88aacc')
    rng = np.random.default_rng(1)
    cols = ['#D77A61', '#D8B4A0', '#DBD3D8', '#EFF1F3', '#223843']
    for k in range(4):
        u, v = np.cumsum(rng.normal(3, 4, 40)) - 20, np.cumsum(rng.normal(1, 4, 40)) - 10
        ax = fig.add_axes([0.1 + 0.22 * k, 0.4, 0.18, 0.27])
        ax.patch.set_alpha(0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.plot([-40, 40], [0, 0], color='#666666', linewidth=0.5, alpha=0.8, zorder=1)
        for r in (20, 40):
            ax.add_artist(Circle((0, 0), r, fill=False, color='#666666', alpha=0.8,
                                 linewidth=0.5, zorder=2))
        ax.set_clip_on(False)
        for i in range(5):
            ax.plot(u[i * 8:i * 8 + 9], v[i * 8:i * 8 + 9], color=cols[i], linewidth=1.8,
                    solid_joinstyle='round', solid_capstyle='round', zorder=20 + i,
                    clip_on=False,
                    path_effects=[withStroke(linewidth=2.5, foreground='white', alpha=0.7)])
        ax.set_aspect('equal')
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        ax.set_xticks([])
        ax.set_yticks([])
    return fig


def test_hodograph_insets_like_matplotlib():
    """Round caps and joins, a white withStroke halo under each trace,
    traces past the inset's box (clip_on=False), plt.Circle."""
    m = _mpl_rgb(_hodographs(mplt, mplt.Circle, mpe.withStroke))
    mplt.close('all')
    f = _rgb(_hodographs(fplt, fplt.Circle, fpe.withStroke))
    d = np.abs(m - f).max(axis=2)
    assert d.mean() < 0.02 and (d > 64).mean() < 1e-4, (d.mean(), (d > 64).mean())


def test_line_path_effects_order_and_styles():
    fig = fastplot.figure(figsize=(2, 2), dpi=50)
    ax = fig.add_axes([0, 0, 1, 1])
    ln = ax.plot([0, 1], [0, 1], path_effects=[fpe.Stroke(linewidth=6, foreground='k'),
                                               fpe.Normal()], solid_capstyle='butt')[0]
    assert ln.get_solid_capstyle() == 'butt' and len(ln.get_path_effects()) == 2
    with pytest.raises(ValueError):
        ln.set_solid_joinstyle('pointy')
    only_stroke = fastplot.figure(figsize=(2, 2), dpi=50)
    ax2 = only_stroke.add_axes([0, 0, 1, 1])
    ax2.plot([0, 1], [0, 1], color='r', path_effects=[fpe.Stroke(linewidth=6, foreground='k')])
    img = _rgb(only_stroke)
    assert not ((img[..., 0] > 200) & (img[..., 1] < 50)).any()   # (no Normal: no red line)


# ------------------------------------------------------------ RGBA mesh

def _curvilinear(ny=40, nx=50):
    j, i = np.meshgrid(np.arange(nx), np.arange(ny))
    return -100 + 0.2 * j + 0.03 * i, 30 + 0.15 * i - 0.02 * j


def _mesh_img(C=None, palette=None, **kw):
    lon, lat = _curvilinear(*C.shape[:2]) if C is not None else _curvilinear(*palette[0].shape)
    fig = fastplot.figure(figsize=(4, 3), dpi=80)
    pc = fcrs.PlateCarree()
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection=pc)
    ax.set_extent([-101, -89, 29, 37], crs=pc)
    if palette is not None:
        ax.pcolormesh(lon, lat, palette[0], cmap=palette[1], norm=palette[2],
                      shading='nearest', transform=pc)
    else:
        ax.pcolormesh(lon, lat, C, shading='nearest', transform=pc, **kw)
    return _rgb(fig)


def _palette(c8):
    """The palette workaround: the colours as an index into a ListedColormap."""
    from fastplot.colors import BoundaryNorm, ListedColormap
    packed = np.ascontiguousarray(c8).view('<u4')[..., 0]
    shown = c8[..., 3] > 0
    uniq, inv = np.unique(packed[shown], return_inverse=True)
    idx = np.zeros(packed.shape)
    idx[shown] = inv
    n = len(uniq)
    return (np.ma.masked_array(idx, mask=~shown),
            ListedColormap(uniq.astype('<u4').view(np.uint8).reshape(-1, 4) / 255.0),
            BoundaryNorm(np.arange(n + 1) - 0.5, n))


def test_rgba_pcolormesh_paints_each_cell_its_colour():
    rng = np.random.default_rng(3)
    pal = rng.integers(0, 256, (20, 4)).astype(np.uint8)
    pal[:, 3] = 255
    pal[2, 3], pal[5, 3] = 0, 140                       # unpainted, and translucent
    c8 = pal[rng.integers(0, 20, (40, 50))]
    ref = _mesh_img(palette=_palette(c8))
    assert np.array_equal(_mesh_img(c8), ref)
    f = c8 / 255.0
    assert np.array_equal(_mesh_img(f), ref)
    assert np.array_equal(_mesh_img(f.astype(np.float32)), ref)
    bad = f.copy()
    bad[c8[..., 3] == 0, 0] = np.nan                    # NaN: not painted either
    bad[c8[..., 3] == 0, 3] = 1.0
    assert np.array_equal(_mesh_img(bad), ref)
    masked = np.ma.masked_array(f, mask=np.broadcast_to((c8[..., 3] == 0)[..., None], f.shape))
    masked.data[..., 3][c8[..., 3] == 0] = 1.0
    assert np.array_equal(_mesh_img(masked), ref)


def test_rgb_pcolormesh_alpha_as_matplotlib():
    """alpha replaces an RGB array's (opaque by default) and leaves an
    RGBA one's, as matplotlib's to_rgba does."""
    from fastplot._fields import _cell_rgba
    rgb = np.full((2, 3, 3), 0.5)
    assert (_cell_rgba(rgb, None)[2][..., 3] == 255).all()
    assert (_cell_rgba(rgb, 0.5)[2][..., 3] == 128).all()
    rgba = np.full((2, 3, 4), 0.6)
    assert (_cell_rgba(rgba, 0.5)[2][..., 3] == 153).all()
    assert (_cell_rgba((rgba * 255).astype(np.uint8), 0.5)[2][..., 3] == 153).all()


needs_311 = pytest.mark.skipif(
    tuple(int(v) for v in matplotlib.__version__.split('.')[:2]) < (3, 11),
    reason="matplotlib before 3.11 puts a mesh's cell edges elsewhere (any mesh's)")


def _plain_mesh(plt, C, **kw):
    fig = plt.figure(figsize=(3, 2), dpi=60)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.pcolormesh(np.arange(C.shape[1]), np.arange(C.shape[0]), C, shading='nearest', **kw)
    return _mpl_rgb(fig) if plt is mplt else _rgb(fig)


def _translucent_cells():
    rng = np.random.default_rng(4)
    C = rng.random((6, 8, 4))
    C[..., 3] = np.where(rng.random((6, 8)) < 0.2, 0.0, 1.0)
    return C


def test_rgba_pcolormesh_on_plain_axes():
    C = _translucent_cells()
    c8 = (C * 255 + 0.5).astype(np.uint8)
    z, cm, norm = _palette(c8)
    assert np.array_equal(_plain_mesh(fplt, C), _plain_mesh(fplt, z, cmap=cm, norm=norm))


@needs_311
def test_rgba_pcolormesh_on_plain_axes_matches_matplotlib():
    C = _translucent_cells()
    d = np.abs(_plain_mesh(mplt, C) - _plain_mesh(fplt, C)).max(axis=2)
    mplt.close('all')
    assert (d > 2).mean() < 0.001, (d > 2).mean()


# ----------------------------------------------------------- map ticks

E = (-105.0, -93.0, 28.0, 35.5)                         # (the boxes' 0.625: the map fills them)


def _ticked(fig, pc, FF):
    ax = fig.add_axes([0.06, 0.08, 0.8, 0.8], projection=pc)
    ax.set_extent(E, crs=pc)
    ax.set_xticks(np.linspace(E[0], E[1], 8), crs=pc)
    ax.set_yticks(np.linspace(E[2], E[3], 6), crs=pc)
    ax.xaxis.set_major_formatter(FF(lambda v, p: '%.1fW' % abs(v)))
    ax.yaxis.set_major_formatter(FF(lambda v, p: '%.1fN' % abs(v)))
    ax.tick_params(axis='both', labelsize=8, pad=3, width=0.5)
    ax.grid(linestyle=':', alpha=0.4, color='gray', linewidth=0.5)
    return ax


def test_map_ticks_as_the_overlay_axes_drew_them():
    """MapAxes.set_xticks(crs=), formatters, tick_params, grid: the picture
    plain axes laid over the map gave (bar the graticule, which
    gridlines drew there)."""
    fig = fastplot.figure(figsize=(8, 5), dpi=80)
    ax = _ticked(fig, fcrs.PlateCarree(), fplt.FuncFormatter)
    ax.spines['geo'].set_visible(False)                 # (the overlay's grid went over the frame;
    new = _rgb(fig)                                     # a map's goes under it, as matplotlib's)
    fig2 = fastplot.figure(figsize=(8, 5), dpi=80)
    m = fig2.add_axes([0.06, 0.08, 0.8, 0.8], projection=fcrs.PlateCarree())
    m.set_extent(E, crs=fcrs.PlateCarree())
    m.spines['geo'].set_visible(False)
    t = fig2.add_axes([0.06, 0.08, 0.8, 0.8])
    t.patch.set_visible(False)
    for s in t.spines.values():
        s.set_visible(False)
    t.set_xlim(*E[:2])
    t.set_ylim(*E[2:])
    t.set_xticks(np.linspace(E[0], E[1], 8))
    t.set_yticks(np.linspace(E[2], E[3], 6))
    t.xaxis.set_major_formatter(fplt.FuncFormatter(lambda v, p: '%.1fW' % abs(v)))
    t.yaxis.set_major_formatter(fplt.FuncFormatter(lambda v, p: '%.1fN' % abs(v)))
    t.tick_params(axis='both', labelsize=8, pad=3, width=0.5)
    t.grid(linestyle=':', alpha=0.4, color='gray', linewidth=0.5)
    assert np.array_equal(new, _rgb(fig2))
    assert np.allclose(ax.get_xticks(), np.linspace(E[0], E[1], 8))


def test_map_without_ticks_draws_none():
    fig = fastplot.figure(figsize=(4, 3), dpi=60)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=fcrs.PlateCarree())
    ax.set_extent(E, crs=fcrs.PlateCarree())
    before = _rgb(fig)
    fig2 = fastplot.figure(figsize=(4, 3), dpi=60)
    ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8], projection=fcrs.PlateCarree())
    ax2.set_extent(E, crs=fcrs.PlateCarree())
    ax2.tick_params(labelsize=8)                        # (no ticks set: still none, as cartopy)
    ax2.grid(True)
    assert np.array_equal(before, _rgb(fig2))


def test_map_ticks_need_rectangular_crs():
    fig = fastplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=fcrs.LambertConformal())
    with pytest.raises(RuntimeError):
        ax.set_xticks([-100, -90], crs=fcrs.PlateCarree())


# -------------------------------------------------- shapefile features

def _write_shp(path, parts, polygon):
    """A tiny shapefile (no .shx / .dbf: fastplot's reader needs neither)."""
    recs = []
    for k, ring in enumerate(parts):
        ring = np.asarray(ring, float)
        body = struct.pack('<i4d2i', 5 if polygon else 3, ring[:, 0].min(), ring[:, 1].min(),
                           ring[:, 0].max(), ring[:, 1].max(), 1, len(ring))
        body += struct.pack('<i', 0) + ring.astype('<f8').tobytes()
        recs.append(struct.pack('>2i', k + 1, len(body) // 2) + body)
    data = b''.join(recs)
    head = struct.pack('>7i', 9994, 0, 0, 0, 0, 0, (100 + len(data)) // 2)
    head += struct.pack('<2i4d4d', 1000, 5 if polygon else 3, -180, -90, 180, 90, 0, 0, 0, 0)
    with open(path, 'wb') as f:
        f.write(head + data)


def _squares():
    out = []
    for i in range(3):
        for j in range(3):
            x0, y0 = -104 + 3 * j, 29 + 2 * i
            out.append([(x0, y0), (x0, y0 + 2), (x0 + 3, y0 + 2), (x0 + 3, y0), (x0, y0)])
    return out


def _feature_img(add):
    fig = fastplot.figure(figsize=(5, 3), dpi=80)
    pc = fcrs.PlateCarree()
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection=pc)
    ax.set_extent(E, crs=pc)
    add(ax, pc)
    return _rgb(fig)


def test_shapefile_feature_draws_its_rings_as_shapely_feature_does():
    shapely = pytest.importorskip('shapely.geometry')
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'squares.shp')
        _write_shp(path, _squares(), True)
        style = dict(facecolor='none', edgecolor='gray', linewidth=0.8)
        a = _feature_img(lambda ax, pc: ax.add_feature(ffeature.ShapefileFeature(path), **style))
        geoms = [shapely.Polygon(r) for r in _squares()]
        b = _feature_img(lambda ax, pc: ax.add_feature(ffeature.ShapelyFeature(geoms, pc),
                                                       **style))
        assert np.array_equal(a, b)
        assert (a.min(axis=2) < 200).sum() > 500        # (something drawn)
        filled = _feature_img(lambda ax, pc: ax.add_feature(
            ffeature.ShapefileFeature(path), facecolor='tab:green', edgecolor='none'))
        assert (np.abs(filled - a).max(axis=2) > 64).mean() > 0.2


def test_shapefile_feature_lines():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'roads.shp')
        lines = [[(-104, 29), (-95, 34)], [(-104, 34), (-95, 29), (-94, 30)]]
        _write_shp(path, lines, False)
        a = _feature_img(lambda ax, pc: ax.add_feature(ffeature.ShapefileFeature(path),
                                                       edgecolor='k', linewidth=1))
        assert ffeature.ShapefileFeature(path)._kind() == 'lines'
        assert len(ffeature.ShapefileFeature(path).geometries()) == 2
        assert (a.min(axis=2) < 100).sum() > 300


# ---------------------------------------------------------------- ink

def test_static_strokes_drawn_again_from_their_ink_are_the_same():
    """A map's features drawn by later figures from the coverage kept on
    the second draw: the same pixels as rasterizing them each time."""
    lon, lat = _curvilinear()

    def frames():
        out = []
        for k in range(4):
            fig = fastplot.figure(figsize=(5, 3), dpi=80)
            pc = fcrs.PlateCarree()
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection=pc)
            ax.set_extent([-101, -89, 29, 37], crs=pc)
            z = np.sin(lon / 3 + k) + 0.01 * np.random.default_rng(k).normal(size=lon.shape)
            ax.contourf(lon, lat, z, levels=30, transform=pc)
            ax.coastlines(color='k', linewidth=0.7)
            ax.add_feature(ffeature.STATES, edgecolor='#333333', linewidth=0.6, alpha=0.7)
            ax.add_feature(ffeature.LAKES, alpha=0.5)
            ax.gridlines(linestyle=':', alpha=0.4, color='gray')
            out.append(_rgb(fig))
        return out

    def fresh():
        _artists._GEOM.clear()

    keep = _agg.INK_MAX
    try:
        fresh()
        _agg.INK_MAX = 0
        plain = frames()
        fresh()
        _agg.INK_MAX = keep
        inked = frames()
    finally:
        _agg.INK_MAX = keep
        fresh()
    for a, b in zip(plain, inked):
        assert np.array_equal(a, b)


# ----------------------------------------------- frames of the same map

def test_raster_triangles_the_same_on_any_number_of_cores():
    """A pixel two triangles claim (a centre on their shared edge) goes to
    the later one, as drawn in order: by rows, not by racing threads."""
    from fastplot import _tri
    j, i = np.meshgrid(np.arange(41, dtype=float), np.arange(31, dtype=float))
    sx, sy = (2.0 * j).ravel(), (2.0 * i).ravel()        # pixel centres on the cells' edges
    k = (i[:-1, :-1] * 41 + j[:-1, :-1]).ravel().astype(np.int64)
    tri = np.concatenate([np.c_[k, k + 1, k + 42], np.c_[k, k + 42, k + 41]])
    zt = np.random.default_rng(0).random(tri.shape)
    out = []
    for ntask in (1, 2, 3, 8, 16):
        v = np.full((62, 82), np.nan, np.float32)
        _tri._raster_triangles(v, sx, sy, tri, zt, False, ntask)
        out.append(v)
    for v in out[1:]:
        assert np.array_equal(v, out[0], equal_nan=True)


def _frames(draw, n=3):
    out = []
    for k in range(n):
        fig = fastplot.figure(figsize=(5, 3), dpi=80)
        pc = fcrs.PlateCarree()
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection=pc)
        ax.set_extent([-101, -89, 29, 37], crs=pc)
        draw(ax, pc)
        out.append(_rgb(fig))
    return out


def test_mesh_drawn_again_by_its_cells_colours():
    """From a mesh's second frame its cells are shaded and their colours
    taken to the pixels: the first frame's picture (values by pixel)."""
    lon, lat = _curvilinear()
    z = np.sin(lon) * np.cos(lat) * 10
    rgba = np.random.default_rng(2).random(lon.shape + (4,))
    for kw in (dict(C=z, cmap='turbo', vmin=-8, vmax=8), dict(C=z), dict(C=rgba)):
        C = kw.pop('C')
        frames = _frames(lambda ax, pc: ax.pcolormesh(lon, lat, C, shading='nearest',
                                                      transform=pc, **kw))
        assert all(np.array_equal(frames[0], f) for f in frames[1:])


def test_contourf_holes_drawn_again_from_the_same_pattern():
    """A field masked the same way frame after frame (the ground at 850
    hPa): the holes' triangles kept, the same picture."""
    lon, lat = _curvilinear()
    z = np.sin(lon / 2) * np.cos(lat / 3) * 10
    z[5:15, 10:25] = np.nan
    z[:4, :6] = np.nan                                  # (at the map's edge)
    frames = _frames(lambda ax, pc: ax.contourf(lon, lat, z, levels=20, transform=pc))
    assert all(np.array_equal(frames[0], f) for f in frames[1:])
    moved = z.copy()
    moved[20:30, 30:40] = np.nan                        # another pattern: made again
    f2 = _frames(lambda ax, pc: ax.contourf(lon, lat, moved, levels=20, transform=pc), 1)[0]
    assert not np.array_equal(frames[0], f2)


def _without_kept(fn):
    from fastplot import _artists
    _artists._DRAWN.clear()
    return fn()


def test_overlays_kept_across_figures_draw_as_new():
    """A frame's products share their MSLP contours, labels and barbs: the
    next figure drawing the same arrays takes the kept drawing -- every
    pixel as drawn afresh -- and an array changed in place since, or other
    levels, labels or colours, are drawn anew."""
    lon, lat = _curvilinear()
    slp = 1010 + 8 * np.sin(lon / 3) * np.cos(lat / 2)
    u, v = 20 * np.cos(lat / 4), 15 * np.sin(lon / 5)

    def draw(ax, pc, levels=np.arange(990, 1030, 2), colors='k', fmt='%d'):
        cs = ax.contour(lon, lat, slp, levels=levels, colors=colors, linewidths=1.2,
                        transform=pc)
        ax.clabel(cs, fmt=fmt, fontsize=7)
        ax.barbs(lon[::6, ::6], lat[::6, ::6], u[::6, ::6], v[::6, ::6], length=5,
                 transform=pc)

    fresh = _without_kept(lambda: _frames(draw, 1)[0])
    again = _frames(draw, 4)
    assert all(np.array_equal(fresh, f) for f in again)
    slp[10:20, 10:30] += 3.0                            # the same array object, changed in place
    u[::6, ::6][2, 2] += 25.0
    changed = _without_kept(lambda: _frames(draw, 1)[0])
    assert not np.array_equal(fresh, changed)
    assert all(np.array_equal(changed, f) for f in _frames(draw, 2))
    for kw in (dict(levels=np.arange(991, 1031, 2)), dict(colors='r'), dict(fmt='%.1f')):
        want = _without_kept(lambda: _frames(lambda ax, pc: draw(ax, pc, **kw), 1)[0])
        _frames(draw, 1)                                # (the plain drawing kept again)
        assert np.array_equal(want, _frames(lambda ax, pc: draw(ax, pc, **kw), 1)[0])
