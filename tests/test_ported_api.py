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


def test_gridline_on_the_edge_of_a_regional_map_has_no_label():
    """As cartopy: 80W on a 95W-80W map is not labelled; a global map labels
    180 on both sides."""
    fig, ax = _map()
    gl = ax.gridlines(draw_labels=True, xlocs=np.arange(-94, -80 + 1e-6, 2))
    fig.draw()
    from fastplot._gridliner import _edge_crossings
    assert _edge_crossings(ax, PC, 'bottom', lon=-80.0) == []
    assert len(_edge_crossings(ax, PC, 'bottom', lon=-82.0)) == 1
    fig, ax = _map(extent=(-180, 180, -90, 90), size=(6, 3))
    ax.gridlines(draw_labels=True)
    fig.draw()
    assert len(_edge_crossings(ax, PC, 'bottom', lon=180.0)) == 2
    assert gl is not None


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
