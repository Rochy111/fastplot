"""What porting a cartopy map pipeline turned up: overlapping feature
polygons, patches across a map's edge, BoundaryNorm's indices, gridline
labels on the edge, and matplotlib API code relies on (cax of any axes,
set_over, ax.patch, fig.patch, set_aspect on maps, text path effects)."""
import numpy as np
import pytest

import fastplot as fp
import fastplot.colors as fc
import fastplot.crs as ccrs
import fastplot.feature as cfeature
import fastplot.patheffects as pe
from fastplot import _proj
from fastplot.patches import Polygon

PC = ccrs.PlateCarree()


def _map(extent=(-95, -80, 37.5, 48), size=(6, 4.5), dpi=60):
    fig = fp.figure(figsize=size, dpi=dpi)
    ax = fig.add_subplot(projection=PC)
    ax.set_extent(list(extent))
    return fig, ax


def _at(ax, img, lon, lat):
    px, py = ax.to_px(np.array([lon]), np.array([lat]))
    return img[int(py[0]), int(px[0]), :3]


def test_overlapping_feature_polygons_filled():
    """Natural Earth's 10 m lakes: Lake Huron and separate pieces of it
    (Saginaw Bay, the North Channel) overlap; each is filled, as cartopy
    fills them, and islands stay open."""
    pytest.importorskip('shapefile')
    fig, ax = _map()
    try:
        ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='blue')
        img = fig.draw()
    except OSError:
        pytest.skip('Natural Earth 10 m lakes not available')
    for lon, lat in ((-83.6, 43.9), (-83.0, 46.1), (-82.3, 44.5)):     # Saginaw Bay, North
        assert tuple(_at(ax, img, lon, lat)) == (0, 0, 255)              # Channel, Huron


def test_patch_across_the_west_edge_is_not_torn():
    fig, ax = _map()
    ax.add_patch(Polygon([[-96, 40], [-94, 40], [-94, 42], [-96, 42]], facecolor='red',
                         transform=PC))
    img = fig.draw()
    assert tuple(_at(ax, img, -94.5, 41)) == (255, 0, 0)               # its inside part
    assert tuple(_at(ax, img, -87, 41)) != (255, 0, 0)                 # not a band across
    assert tuple(_at(ax, img, -81, 41)) != (255, 0, 0)


def test_patch_across_the_seam_of_a_global_map():
    fig, ax = _map(extent=(-180, 180, -90, 90), size=(6, 3))
    ax.add_patch(Polygon([[175, 10], [185, 10], [185, 20], [175, 20]], facecolor='red',
                         transform=PC))
    img = fig.draw()
    assert tuple(_at(ax, img, 177, 15)) == (255, 0, 0)                 # both sides of 180
    assert tuple(_at(ax, img, -177, 15)) == (255, 0, 0)
    assert tuple(_at(ax, img, 0, 15)) != (255, 0, 0)


@pytest.mark.parametrize('levels, ncolors, extend, clip', [
    (np.linspace(0, 31, 32), 31, 'neither', True),
    ([0, 1, 2, 3], 256, 'neither', False),
    ([0, 1, 2, 3], 256, 'both', False),
    ([0, 1], 8, 'neither', False),
    (np.linspace(-5, 5, 11), 12, 'both', False),
])
def test_boundary_norm_indices_are_matplotlibs(levels, ncolors, extend, clip):
    mc = pytest.importorskip('matplotlib.colors')
    x = np.r_[np.random.default_rng(1).uniform(np.min(levels) - 2, np.max(levels) + 2, 300),
              levels, np.nan]
    ours = fc.BoundaryNorm(levels, ncolors, clip=clip, extend=extend)(x)
    ref = mc.BoundaryNorm(levels, ncolors, clip=clip, extend=extend)(x)
    np.testing.assert_array_equal(np.ma.getmaskarray(ours), np.ma.getmaskarray(ref))
    np.testing.assert_array_equal(ours.filled(-99), ref.filled(-99))
    colours = ['C%d' % (k % 10) for k in range(ncolors)]
    np.testing.assert_allclose(fc.ListedColormap(colours)(ours),
                               mc.ListedColormap(colours)(ref))
    assert fc.BoundaryNorm([0, 1, 2], 2)(1.5) == 1


def test_colormap_set_extremes_in_place():
    cm = fc.ListedColormap(['red', 'green'])
    cm.set_under('blue')
    cm.set_over('black')
    cm.set_bad('white', 0.5)
    np.testing.assert_allclose(cm(np.array([-1.0, 2.0, np.nan])),
                               [[0, 0, 1, 1], [0, 0, 0, 1], [1, 1, 1, 0.5]])


def test_axes_and_figure_patch():
    fig = fp.figure(figsize=(3, 2), dpi=40)
    ax = fig.add_subplot()
    ax.patch.set_visible(False)
    fig.patch.set_visible(False)
    assert fig.draw()[1, 1, 3] == 0                                   # transparent
    fig = fp.figure(figsize=(3, 2), dpi=40)
    fig.patch.set_facecolor('black')
    assert tuple(fig.draw()[1, 1, :3]) == (0, 0, 0)
    assert fig.patch.get_visible()


def test_map_set_aspect_auto_fills_the_box():
    sizes = {}
    for aspect in ('equal', 'auto'):
        fig, ax = _map(size=(6, 3))
        ax.set_aspect(aspect)
        fig.draw()
        sizes[aspect] = ax.box_f[2] - ax.box_f[0]
    assert sizes['auto'] > sizes['equal'] * 1.3


def test_colorbar_cax_of_plain_axes():
    fig, ax = _map()
    mesh = ax.pcolormesh(np.arange(-95, -79.0), np.arange(37, 49.0),
                         np.random.default_rng(2).random((12, 16)), transform=PC)
    cax = fig.add_axes((0.92, 0.2, 0.02, 0.6))
    fig.colorbar(mesh, cax=cax, label='x')
    fig.draw()


def test_text_stroke_outline():
    """withStroke draws an outline round the text, then the text; a lone
    Stroke draws only the outline."""
    def label(effects):
        fig = fp.figure(figsize=(2, 1), dpi=100, facecolor='blue')
        fig.text(0.5, 0.5, 'Hm', color='white', fontsize=20, ha='center', va='center',
                 path_effects=effects)
        return fig.draw()[..., :3].astype(int)

    plain, stroked = label(None), label([pe.withStroke(linewidth=3, foreground='black')])
    black = (stroked.sum(-1) < 60)
    assert black.sum() > 50 and (plain.sum(-1) < 60).sum() == 0        # an outline appeared
    assert ((stroked == 255).all(-1)).sum() == ((plain == 255).all(-1)).sum()   # text on top
    only = label([pe.Stroke(linewidth=3, foreground='black')])
    assert ((only == 255).all(-1)).sum() == 0                          # no white text


def _line_figure():
    fig = fp.figure(figsize=(3, 2), dpi=80)
    ax = fig.add_subplot()
    x = np.linspace(0, 6, 200)
    ax.plot(x, np.sin(3 * x), color='black', linewidth=1.3)
    ax.text(1, 0.3, 'Label', color='black', fontsize=12)
    return fig, ax


def test_transparent_edges_keep_their_colour():
    """Anti-aliased black over a transparent white page: the partly covered
    pixels are black, part transparent -- not grey (the page's white mixed
    in), which composited over anything comes out lighter."""
    fig, ax = _line_figure()
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    img = fig.draw()
    edge = (img[..., 3] > 10) & (img[..., 3] < 245)
    assert edge.sum() > 100
    assert img[..., :3][edge].max() <= 2


def test_savefig_transparent_clears_axes_backgrounds():
    import io
    from PIL import Image
    fig, ax = _line_figure()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=True)
    img = np.asarray(Image.open(buf).convert('RGBA'))
    assert img[2, 2, 3] == 0                                # the figure's background gone
    inside = img[30:-30, 60:-30]
    assert (inside[..., 3] == 0).mean() > 0.5              # the axes' white is gone too
    assert ax.patch.get_visible()                           # and back afterwards


def test_opaque_figures_unchanged_by_transparent_blending():
    """Over an opaque pixel the blend is the one fastplot always used: the
    golden images (tests/golden) pin it; here one value, by hand."""
    from fastplot import _draw
    c = np.zeros((1, 1, 4), np.uint8)
    c[0, 0] = (255, 255, 255, 255)
    _draw.blend_px(c, 0, 0, np.float32(0), np.float32(0), np.float32(0), np.float32(255),
                   np.float32(0.3))
    f = np.float32(0.3)
    want = np.uint8(np.float32(255) * (np.float32(1) - f) + np.float32(0.5))
    assert tuple(c[0, 0]) == (want, want, want, 255)


def _rgb(ny=90, nx=160, seed=3):
    return np.random.default_rng(seed).integers(0, 256, (ny, nx, 3), dtype=np.uint8)


def test_rgb_imshow_on_a_map_samples_as_a_field():
    """An RGB image on a map: each channel lands where imshow of that
    channel alone (as a field, a grey for each byte) puts it."""
    img = _rgb()
    extent = (-100, -75, 30, 50)
    grey = fc.ListedColormap(np.repeat(np.arange(256)[:, None] / 255, 3, 1))
    norm = fc.BoundaryNorm(np.arange(257) - 0.5, 256)
    out = []
    for z in (img, img[..., 1]):
        fig = fp.figure(figsize=(5, 4), dpi=60)
        ax = fig.add_subplot(projection=ccrs.LambertConformal(-90, 40))
        ax.set_extent([-98, -78, 32, 48], crs=PC)
        ax.imshow(z, extent=extent, transform=PC, origin='upper',
                  **({} if z.ndim == 3 else dict(cmap=grey, norm=norm)))
        out.append(fig.draw())
    np.testing.assert_array_equal(out[0][..., 1], out[1][..., 1])
    assert len(np.unique(out[0][..., :3].reshape(-1, 3), axis=0)) > 1000    # not one colour


def test_rgb_imshow_on_a_map_pixel_by_pixel():
    """Each canvas pixel holds the image pixel under its centre, found here
    by inverting the projection independently."""
    img = _rgb()
    x0, x1, y0, y1 = -100.0, -75.0, 30.0, 50.0
    proj = ccrs.LambertConformal(-90, 40)
    drawn = []
    for image in (None, img):
        fig = fp.figure(figsize=(5, 4), dpi=60)
        ax = fig.add_subplot(projection=proj)
        ax.set_extent([-98, -78, 32, 48], crs=PC)
        if image is not None:
            ax.imshow(image, extent=(x0, x1, y0, y1), transform=PC)
        drawn.append(fig.draw())
    bare, canvas = drawn
    v = ax.view
    bx0, by0, bx1, by1 = ax.box
    j, i = np.meshgrid(np.arange(bx1 - bx0), np.arange(by1 - by0))
    lon, lat = _proj.transform_points(proj, PC, v.x0 + (j + 0.5) * v.dx,
                                      v.y0 + (i + 0.5) * v.dy)
    fc_ = (lon - x0) / (x1 - x0) * img.shape[1]
    fr = (y1 - lat) / (y1 - y0) * img.shape[0]
    clear = ((fc_ % 1 > 1e-6) & (fc_ % 1 < 1 - 1e-6) & (fr % 1 > 1e-6) & (fr % 1 < 1 - 1e-6)
             & (fc_ >= 0) & (fc_ < img.shape[1]) & (fr >= 0) & (fr < img.shape[0])
             & (bare[by0:by1, bx0:bx1, :3] == 255).all(-1))     # not under the map's outline
    want = img[fr[clear].astype(int), fc_[clear].astype(int)]
    got = canvas[by0:by1, bx0:bx1, :3][clear]
    assert clear.sum() > 10000
    np.testing.assert_array_equal(got, want)


def test_rgba_imshow_on_a_global_map():
    """A whole-world RGBA image on Robinson: drawn across the dateline, its
    alpha kept, and the map's extent taken from it."""
    img = np.concatenate([_rgb(60, 120), np.full((60, 120, 1), 255, np.uint8)], 2)
    img[:, :60, 3] = 0                                  # the western half see-through
    fig = fp.figure(figsize=(6, 3), dpi=60, facecolor='white')
    ax = fig.add_subplot(projection=ccrs.Robinson())
    ax.imshow(img, extent=(-180, 180, -90, 90), transform=PC, origin='upper')
    canvas = fig.draw()
    def at(lon, lat):
        x, y = _proj.transform_points(PC, ax.projection, np.array([lon]), np.array([lat]))
        return tuple(_at(ax, canvas, x[0], y[0]))

    assert at(-90.5, 10.5) == (255, 255, 255)                        # alpha 0: the page
    for lon, lat in ((90.5, 10.5), (160.5, -40.5), (1.5, 70.5)):
        assert at(lon, lat) == tuple(img[int((90 - lat) / 3), int((lon + 180) / 3), :3])
    e = ax.get_extent()
    assert e[1] - e[0] > 3.3e7                                       # the whole world


def _mpl():
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _ink_rows(img, x0, x1, y1):
    """The rows with dark pixels in columns x0:x1 above row y1."""
    dark = img[:y1, x0:x1, :3].astype(int).sum(-1) < 300
    return np.where(dark.any(1))[0]


def test_colorbar_cax_is_cb_ax():
    """matplotlib's fig.colorbar(cax=ax) draws in ax itself: cb.ax is cax,
    so its title -- set before or after -- is drawn, and the current axes
    stays a plotting one."""
    fig, ax = _map()
    mesh = ax.pcolormesh(np.arange(-95, -79.0), np.arange(37, 49.0),
                         np.random.default_rng(2).random((12, 16)), transform=PC)
    cax = fig.add_axes((0.1, 0.05, 0.8, 0.04))
    cax.set_title('before')
    cb = fig.colorbar(mesh, cax=cax, orientation='horizontal')
    assert cb.ax is cax and cax.get_title() == 'before'
    assert fig.gca() is ax
    img = fig.draw()
    x0, y0, x1, _ = cax.box_f
    assert _ink_rows(img, int(x0), int(x1), int(y0)).size        # the title, above the bar
    cax.set_title('after', pad=12, color='red')
    assert cb.ax.get_title() == 'after'
    img = fig.draw()
    red = (img[:int(y0), int(x0):int(x1), 0] > 200) & (img[:int(y0), int(x0):int(x1), 1] < 60)
    assert red.any()


@pytest.mark.parametrize('pad', [None, 2, 10])
def test_colorbar_title_pad_like_matplotlib(pad):
    plt = _mpl()
    import matplotlib.cm as mcm
    import matplotlib.colors as mco
    import fastplot.cm as fcm
    kw = {} if pad is None else dict(pad=pad)
    tops = []
    for lib, cm, co in ((plt, mcm, mco), (fp, fcm, fc)):
        fig = lib.figure(figsize=(6, 2), dpi=100)
        cax = fig.add_axes([0.1, 0.3, 0.8, 0.1])
        fig.colorbar(cm.ScalarMappable(co.Normalize(0, 1), 'viridis'), cax=cax,
                     orientation='horizontal')
        cax.set_title('Title Hg', fontsize=12, **kw)
        if lib is plt:
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())
        else:
            img = fig.draw()
        tops.append(_ink_rows(img, 260, 340, 135).min())
        lib.close(fig)
    assert abs(int(tops[0]) - int(tops[1])) <= 1, tops


def test_figure_add_artist_and_artist_set():
    """matplotlib code for strips on a figure: rect.set(...), fig.add_artist;
    in fig.transFigure, over or under the axes by zorder, removable."""
    from fastplot.patches import Rectangle
    fig = fp.figure(figsize=(4, 3), dpi=50)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('white')
    top = Rectangle((0, 0.8), 1, 0.2)
    top.set(transform=fig.transFigure, fc='red', edgecolor='none', zorder=10)
    under = fig.add_artist(Rectangle((0, 0), 1, 0.2, transform=fig.transFigure,
                                     facecolor='blue', zorder=-1))
    fig.add_artist(top)
    img = fig.draw()
    assert tuple(img[5, 100, :3]) == (255, 0, 0)                    # above the axes
    assert tuple(img[145, 100, :3]) == (255, 255, 255)              # under its background
    under.remove()
    assert under not in fig.artists
    with pytest.raises(AttributeError):
        top.set(nonsense=1)


def test_figure_artist_pixels_like_matplotlib():
    plt = _mpl()
    from matplotlib.patches import Rectangle as MRect
    from fastplot.patches import Rectangle
    imgs = []
    for lib, R in ((plt, MRect), (fp, Rectangle)):
        fig = lib.figure(figsize=(4, 3), dpi=50)
        fig.add_artist(R((0.1, 0.25), 0.5, 0.3, transform=fig.transFigure, facecolor='red',
                         edgecolor='none'))
        if lib is plt:
            fig.canvas.draw()
            imgs.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        else:
            imgs.append(fig.draw()[..., :3])
        lib.close(fig)
    np.testing.assert_array_equal(imgs[1], imgs[0])


def test_axes_and_figure_coordinates_on_a_map():
    """Patches and lines in ax.transAxes or fig.transFigure on a map: drawn
    where they say, not projected, and not counted in the map's extent."""
    from fastplot.patches import Rectangle
    fig, ax = _map()
    ax.add_patch(Rectangle((0, 0), 1, 0.25, transform=ax.transAxes, facecolor='blue'))
    ax.add_patch(Rectangle((0, 0.6), 1, 0.1, transform=fig.transFigure, facecolor='red',
                           edgecolor='none', zorder=10))            # clipped to the map, as there
    ax.plot([0, 1], [0.5, 0.5], transform=ax.transAxes, color='lime', linewidth=4)
    img = fig.draw()
    x0, y0, x1, y1 = ax.box_f
    mx = int(0.5 * (x0 + x1))
    assert tuple(img[int(y1 - 0.1 * (y1 - y0)), mx, :3]) == (0, 0, 255)
    assert tuple(img[int(0.5 * (y0 + y1)), mx, :3]) == (0, 255, 0)
    assert tuple(img[int(0.35 * img.shape[0]), mx, :3]) == (255, 0, 0)
    fig = fp.figure(figsize=(4, 3), dpi=50)
    ax = fig.add_subplot(projection=ccrs.LambertConformal())       # no extent: autoscaled
    ax.plot([-100, -90], [35, 45], transform=PC)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    fig.draw()
    lon0, lon1, lat0, lat1 = ax.get_extent(PC)
    assert -101 < lon0 < -99 and -91 < lon1 < -89


def test_axes_zorder_orders_overlapping_axes_like_matplotlib():
    plt = _mpl()
    got = []
    for lib in (plt, fp):
        fig = lib.figure(figsize=(3, 3), dpi=40)
        a = fig.add_axes([0.1, 0.1, 0.6, 0.6], facecolor='red')
        b = fig.add_axes([0.3, 0.3, 0.6, 0.6], facecolor='blue')
        a.set_zorder(2)
        assert b.get_zorder() == 0 and a.get_zorder() == 2
        fig.text(0.5, 0.5, 'x', zorder=-1)                         # under both backgrounds
        if lib is plt:
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())
        else:
            img = fig.draw()
        got.append(tuple(int(v) for v in img[60, 60, :3]))
        lib.close(fig)
    assert got[0] == got[1] == (255, 0, 0)
