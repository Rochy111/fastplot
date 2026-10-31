"""Plain axes pixel for pixel against matplotlib: what each figure here got
wrong once. (bad: the share of pixels off by more than a quarter of the range.)"""
import io

import numpy as np
import pytest

import fastplot
import fastplot.colors as fcolors
import fastplot.patches as fpatches
import fastplot.pyplot as fplt

matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')
import matplotlib.colors as mcolors                                      # noqa: E402
import matplotlib.patches as mpatches                                    # noqa: E402
import matplotlib.pyplot as mplt                                         # noqa: E402
from PIL import Image                                                    # noqa: E402

needs_311 = pytest.mark.skipif(tuple(int(v) for v in matplotlib.__version__.split('.')[:2]) < (3, 11),
                               reason='matplotlib 3.11 lays text out differently')


class Lib:
    def __init__(self, plt, patches, colors):
        self.plt, self.patches, self.colors = plt, patches, colors


def _both(draw, size=(5, 4), dpi=80):
    """draw(lib, fig, ax) in each library: the pixels' differences."""
    out = []
    for lib in (Lib(mplt, mpatches, mcolors), Lib(fplt, fpatches, fcolors)):
        lib.plt.close('all')
        fig, ax = lib.plt.subplots(figsize=size, dpi=dpi)
        draw(lib, fig, ax)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        lib.plt.close('all')
        buf.seek(0)
        out.append(np.asarray(Image.open(buf).convert('RGB')).astype(np.int16))
    return np.abs(out[0] - out[1]).max(axis=2)


def _same(d, bad=0.0, mean=0.05):
    assert (d > 64).mean() <= bad and d.mean() <= mean, ((d > 64).mean(), d.mean())


@needs_311
@pytest.mark.parametrize('marker', ['o', 's', '^', 'D', '*', 'h', 'x', '+', 'v', 'p', '8', 'P',
                                    'X', '1', '|', '_', 3, 4])
def test_line_markers(marker):
    """Each marker's snap threshold, cap and join."""
    def draw(lib, fig, ax):
        ax.plot(np.arange(5) + 0.37, np.arange(5) * 0.77, ls='none', marker=marker, ms=9)
        ax.set_xlim(0, 6)
        ax.set_ylim(-1, 4.5)
    _same(_both(draw), mean=0.02)


@needs_311
@pytest.mark.parametrize('marker', ['o', 's', '^', 'x', '+', '*', 'D', 'h'])
def test_scatter_of_one_style_goes_through_draw_markers(marker):
    """Whole pixels and a centred marker, and a filled marker's edge is a
    collection's line width, not a line's."""
    def draw(lib, fig, ax):
        ax.scatter(np.arange(5) + 0.31, np.arange(5) * 0.77, marker=marker, s=60)
    _same(_both(draw), mean=0.03)


@needs_311
@pytest.mark.parametrize('marker', ['o', 's', '^', 'x'])
def test_scatter_of_many_styles_is_placed_exactly(marker):
    """A path collection: each marker where it is, not on a grid of 1/8 px."""
    rng = np.random.default_rng(1)
    x, y, c, s = rng.uniform(size=(4, 40))

    def draw(lib, fig, ax):
        ax.scatter(x, y, c=c, s=20 + 100 * s, marker=marker, cmap='viridis', alpha=0.7)
    _same(_both(draw), bad=0.0002, mean=0.05)


@needs_311
@pytest.mark.parametrize('marker', ['o', 's'])
def test_scatter_of_thousands_is_exact_too(marker):
    """Enough edges for the rasterizer's parallel bands: still in order."""
    rng = np.random.default_rng(2)
    x, y, c, s = rng.uniform(size=(4, 7000))

    def draw(lib, fig, ax):
        ax.scatter(x, y, c=c, s=5 + 40 * s, marker=marker)
    with fplt.rc_context({'fastplot.exact': True}):
        _same(_both(draw), mean=0.001)


@needs_311
@pytest.mark.parametrize('marker', ['o', 's'])
def test_scatter_of_any_size_is_exact(marker):
    """With fastplot.exact, however many stamps would serve, not a pixel apart."""
    rng = np.random.default_rng(3)
    x, y, c, s = rng.uniform(size=(4, 60000))

    def draw(lib, fig, ax):
        ax.scatter(x, y, c=c, s=5 + 40 * s, marker=marker, edgecolors='k', linewidths=0.5)
    with fplt.rc_context({'fastplot.exact': True}):
        assert not _both(draw).any()


@needs_311
@pytest.mark.parametrize('size', [(9.37, 6.11), (8.27, 11.69), (3.33, 2.22), (1.1, 1.3), (10.01, 7.77)])
@pytest.mark.parametrize('dpi', [72, 100, 101, 150])
@pytest.mark.parametrize('kw', [{}, {'bbox_inches': 'tight'}])
def test_canvas_size_of_any_figsize(size, dpi, kw):
    """A size just under a whole pixel (9.37 in at 100 dpi) counts as it."""
    def px(plt):
        fig, ax = plt.subplots(figsize=size, dpi=dpi)
        ax.plot([0, 1], [0, 1])
        b = io.BytesIO()
        fig.savefig(b, format='png', **kw)
        plt.close(fig)
        b.seek(0)
        return Image.open(b).size
    assert px(fplt) == px(mplt)


@needs_311
@pytest.mark.parametrize('kw', [dict(s=6.0), dict(s=12.0, marker='^'), dict(s=30.0, alpha=0.5),
                                dict(s=12.0, edgecolors='k', linewidths=0.5)])
def test_stamped_scatter_differs_only_at_edges(kw):
    """As it is by default: a shade at an edge, no marker out of place."""
    rng = np.random.default_rng(4)
    x, y, c = rng.uniform(size=(3, 20000))

    def draw(lib, fig, ax):
        ax.scatter(x, y, c=c, **kw)
    d = _both(draw)
    assert d.mean() < 2.0 and d.max() <= 64


@needs_311
def test_thick_mitred_marker_keeps_its_points():
    """A mitre reaches the limit (the width) times the half width: the box
    a stamp is made in holds it."""
    def draw(lib, fig, ax):
        ax.plot(np.arange(5) + 0.37, np.arange(5) * 0.77, ls='none', marker='^', ms=12, mew=7,
                mfc='w', mec='k')
        ax.set_xlim(-1, 6)
        ax.set_ylim(-1, 5)
    _same(_both(draw), mean=0.03)


@needs_311
def test_scatter_with_sizes_all_alike_is_a_collection_of_paths():
    def draw(lib, fig, ax):
        ax.scatter(np.arange(5) + 0.31, np.arange(5) * 0.77, s=np.full(5, 60.0))
    _same(_both(draw), mean=0.03)


@pytest.mark.parametrize('yscale', ['log', 'symlog', 'linear'])
def test_a_scale_change_settles_the_other_axis(yscale):
    """log has a linear axis's default limits no more, so set_xscale
    autoscales at once and settles y as linear; symlog then keeps those."""
    lims = []
    for plt in (mplt, fplt):
        fig, ax = plt.subplots()
        ax.plot([3e3, 7e4], [3e3, 7e4])
        ax.set_xscale('log')
        ax.set_yscale(yscale)
        lims.append(ax.get_ylim())
        ax.plot([1e5, 2e5], [1e5, 2e5])                 # and a new artist asks again
        lims.append(ax.get_ylim())
        plt.close('all')
    assert np.allclose(lims[:2], lims[2:], rtol=1e-9)


@needs_311
@pytest.mark.parametrize('kw', [dict(shadow=True), dict(shadow=True, fancybox=False),
                                dict(title='lower left', title_fontsize=7, fontsize=7)])
def test_legend_frame_and_rows(kw):
    """Rows as tall as their own text's descent, the frame a snapped patch
    with curved corners, its Shadow 2 points off and not snapped."""
    def draw(lib, fig, ax):
        ax.plot([0, 1], [0, 1], label='one')
        ax.plot([1, 0], [0, 1], label='three')
        ax.legend(loc='center', framealpha=1, **kw)
    _same(_both(draw), mean=0.03)


@pytest.mark.parametrize('n, m', [(300, 200), (60, 40), (20, 14)])
def test_contourf_round_masked_points(n, m):
    """contourpy's corner_mask, and Agg's every touched pixel."""
    x, y = np.meshgrid(np.linspace(-3, 3, n), np.linspace(-2, 2, m))
    z = np.ma.masked_where((x ** 2 + y ** 2 < 1.2) | (x > 2.4),
                           np.sin(x * 1.3) * np.cos(y * 1.7) + 0.2 * x)

    def draw(lib, fig, ax):
        ax.contourf(x, y, z, 10)
    _same(_both(draw, dpi=100), bad=0.0003, mean=0.2)


def test_translucent_bands_show_their_seams():
    """Agg paints a pixel two bands touch with both."""
    x, y = np.meshgrid(np.linspace(-3, 3, 60), np.linspace(-2, 2, 40))
    z = np.sin(x * 1.3) * np.cos(y * 1.7) + 0.2 * x

    def draw(lib, fig, ax):
        ax.contourf(x, y, z, 8, alpha=0.5)
    _same(_both(draw, dpi=100), bad=0.0001, mean=1.0)


def test_polar_artists_stop_at_the_circle():
    out = []
    for plt in (mplt, fplt):
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(4, 4), dpi=100)
        t = np.linspace(0, 8, 400)
        ax.plot(t, t, lw=2)
        ax.scatter(t[::40], t[::40], s=40, c='C1')
        ax.set_rmax(5)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close('all')
        buf.seek(0)
        out.append(np.asarray(Image.open(buf).convert('RGB')).astype(np.int16))
    _same(np.abs(out[0] - out[1]).max(axis=2), bad=0.0003, mean=0.15)


def test_tricontourf_fills_what_touches_its_hull():
    rng = np.random.default_rng(4)
    x, y = rng.random((2, 60))

    def draw(lib, fig, ax):
        ax.tricontourf(x, y, np.sin(4 * x) * y, 8)
    _same(_both(draw), bad=0.0008, mean=0.15)


def test_top_axis_label_sits_on_its_baseline():
    def draw(lib, fig, ax):
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.set_xlabel('period, y')
        ax.set_title('over it')
    _same(_both(draw))


def test_axhspan_color_is_the_edge_too():
    def draw(lib, fig, ax):
        ax.axhspan(1, 2, color='y', alpha=0.3)
        ax.axvspan(0.2, 0.4, facecolor='C1', alpha=0.2, edgecolor='k')
        ax.set_ylim(0, 3)
    _same(_both(draw), mean=0.1)


def test_fill_between_color_is_the_edge_too():
    def draw(lib, fig, ax):
        ax.fill_betweenx(np.linspace(0, 2, 50), -1.5, -1.2, color='C0')
    _same(_both(draw), mean=0.1)


def test_bar_error_bars_are_black():
    def draw(lib, fig, ax):
        ax.bar(np.arange(3), [3.0, 2.0, 4.0], yerr=[0.5, 0.2, 0.4])
    _same(_both(draw), mean=0.1)


def test_fancy_patches_have_an_edge():
    def draw(lib, fig, ax):
        ax.add_patch(lib.patches.FancyBboxPatch((0.2, 0.6), 0.3, 0.2, boxstyle='round,pad=0.02',
                                                fc='C5'))
        ax.add_patch(lib.patches.FancyArrowPatch((0.1, 0.3), (0.8, 0.3), arrowstyle='-|>',
                                                 mutation_scale=20))
    _same(_both(draw), mean=0.1)


def test_patch_corners_are_mitred_and_circles_flattened_as_agg_does():
    def draw(lib, fig, ax):
        ax.add_patch(lib.patches.Polygon([[0.1, 0.1], [0.8, 0.2], [0.3, 0.9]], fc='C0', ec='k',
                                         lw=4))
        ax.add_patch(lib.patches.Circle((0.7, 0.7), 0.2, fc='none', ec='r', lw=2))
    _same(_both(draw), mean=0.05)


@needs_311
def test_supylabel_is_not_anchored():
    def draw(lib, fig, ax):
        fig.supylabel('common y')
        fig.supxlabel('common x')
    _same(_both(draw), mean=0.1)


@needs_311
def test_inset_indicator():
    """The box unclipped; box and connectors one path, unsnapped."""
    def draw(lib, fig, ax):
        ax.plot(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)))
        ins = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
        ins.set_xlim(1.0, 2.5)
        ins.set_ylim(0.5, 1.5)
        ins.set_xticks([])
        ins.set_yticks([])
        ax.indicate_inset_zoom(ins)
    _same(_both(draw), mean=0.1)


@needs_311
def test_contour_label_near_the_start_of_a_ring():
    """A ring starts at its label before the label's angle is measured."""
    y, x = np.mgrid[-2:2:60j, -2:2:60j]
    z = np.exp(-x ** 2 - y ** 2) + 0.3 * np.exp(-(x - 1) ** 2 - (y + 1) ** 2)

    def draw(lib, fig, ax):
        ax.clabel(ax.contour(x, y, z, levels=[0.2, 0.4, 0.6, 0.8], colors='k'), fontsize=8)
    _same(_both(draw), mean=0.05)


@needs_311
@pytest.mark.parametrize('z', [lambda x, y: np.abs(x) + np.abs(y),
                               lambda x, y: np.maximum(np.abs(x), np.abs(y)),
                               lambda x, y: np.sin(x * 2) * np.cos(y * 2),
                               lambda x, y: np.cos(x * 3) + np.cos(y * 3),
                               lambda x, y: np.exp(-x ** 2 - y ** 2) -
                               np.exp(-(x - 1) ** 2 - (y - 1) ** 2)],
                         ids=['diamond', 'square', 'sincos', 'lattice', 'peaks'])
def test_contours_of_symmetric_data(z):
    """Levels of rectilinear lines are snapped, saddles turn on the last bit
    of the double, blocks that all but tie go by numpy's sort of matplotlib's
    numbers, and a ring whose ends differ in the last bit stays open."""
    x, y = np.meshgrid(np.linspace(-2, 2, 41), np.linspace(-2, 2, 41))
    v = z(x, y)

    def draw(lib, fig, ax):
        ax.clabel(ax.contour(x, y, v, levels=8, colors='k'), fontsize=9)
    _same(_both(draw, dpi=100), mean=0.03)


def test_rings_contourpy_leaves_open():
    from scipy.ndimage import gaussian_filter
    pytest.importorskip('contourpy')
    import contourpy
    from fastplot import _contour
    z = gaussian_filter(np.random.default_rng(3).normal(size=(60, 80)), 2.0) * 10
    xc, yc = np.arange(80.0) * 0.37, np.arange(60.0) * 0.13 + 1.1
    levels = np.linspace(-2, 2, 9)
    x, y, off, lof = _contour.lines(z, levels, cols=xc, rows=yc)
    opened = _contour.open_rings(z, levels, x, y, off, lof, xc, yc)
    g = contourpy.contour_generator(xc, yc, z, name='mpl2014', line_type='SeparateCode')
    theirs = []
    for level in levels:
        for v, c in zip(*g.lines(level)):
            theirs.append(c[-1] != 79 and np.abs(v[0] - v[-1]).max() < 1e-9)
    assert opened.tolist() == theirs and opened.any()


def test_contour_label_off_the_axes_is_clipped():
    y, x = np.mgrid[-3:3:60j, -3:3:60j]
    z = np.sin(x) * np.cos(y)

    def draw(lib, fig, ax):
        ax.clabel(ax.contour(x, y, z, colors='k'), fontsize=8)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
    d = _both(draw)
    assert d[:30].mean() < 1.0                          # (nothing above the axes)


def test_symlog_colorbar_ticks():
    z = np.random.default_rng(0).lognormal(size=(20, 20)) * 100 - 100

    def draw(lib, fig, ax):
        im = ax.imshow(z, norm=lib.colors.SymLogNorm(linthresh=10))
        fig.colorbar(im, ax=ax)
    assert (_both(draw) > 64).mean() < 0.002


@needs_311
@pytest.mark.parametrize('fontset', ['dejavusans', 'dejavuserif', 'stix', 'stixsans', 'cm'])
def test_mathtext_fontset(fontset):
    """Each fontset's fonts, its glyph rules and constants: integrals' limits
    (dropsub), primes, the Computer Modern extension font's offset."""
    def draw(lib, fig, ax):
        ax.text(0.05, 0.3, r'$\int_0^\infty \frac{x^2}{y}\,dx + \sqrt{a_i} \sum_{k=1}^n '
                           r'\overline{ab}$', fontsize=16)
        ax.text(0.05, 0.7, r"$f'(x) = \alpha\beta\Gamma \mathbf{v} \mathit{w} \mathrm{R} "
                           r"\mathcal{L} \partial$", fontsize=16)
    with mplt.rc_context({'mathtext.fontset': fontset}), \
            fplt.rc_context({'mathtext.fontset': fontset}):
        d = _both(draw)
    _same(d, bad=0.002, mean=0.3)


@needs_311
@pytest.mark.parametrize('default', ['regular', 'rm', 'bf', 'it', 'sf', 'tt'])
def test_mathtext_default(default):
    def draw(lib, fig, ax):
        ax.set_title(r'$xy^2 + \alpha\sin(z) + 3.5$ and more')
        ax.set_xlabel(r'wind (m s$^{-1}$)')
    with mplt.rc_context({'mathtext.default': default}), \
            fplt.rc_context({'mathtext.default': default}):
        d = _both(draw)
    _same(d, bad=0.0003, mean=0.1)


@needs_311
def test_rotated_mathtext_turns_its_rules():
    def draw(lib, fig, ax):
        ax.text(0.2, 0.3, r'$\frac{a}{b}+\sqrt{xy}+\overline{ab}$', fontsize=18, rotation=30)
    _same(_both(draw), bad=0.0005, mean=0.12)


@needs_311
def test_text_is_every_byte_matplotlibs():
    """Agg's own blend on 8 bits, and overlapping glyphs blended in turn."""
    def draw(lib, fig, ax):
        ax.axis('off')
        for k in range(6):
            ax.text(0.02, k * 0.16 + 0.02, 'Wavy AVATAR, fjord ffi Tyrol rj',
                    fontsize=8 + k, style='italic' if k % 2 else 'normal', color=('k', 'C3')[k % 2])
        ax.text(0.5, 0.97, r'$\int\!\!\!\int f\!\!f\,dx \sum\!\!\!\!\sum$', fontsize=14)
    assert _both(draw).max() == 0


def test_overlapping_glyphs_are_layers():
    from fastplot import _text
    tm = _text.place(r'$f\!\!\!f\!\!\!f$', 0, 0, 30.0)[0]
    assert tm.layers is not None and len(tm.layers) >= 3
    assert sum(np.count_nonzero(a) for a, _, _ in tm.layers) > np.count_nonzero(tm.alpha)
    assert _text.place('plain', 0, 0, 12.0)[0].layers is None


def test_fontsets_on_threads_at_once():
    """Each thread its own fontset, and the caches keep them apart."""
    import threading

    def png(fontset):
        with fplt.rc_context({'mathtext.fontset': fontset}):
            fig, ax = fplt.subplots(figsize=(3, 2), dpi=60)
            ax.set_title(r'$\alpha\sum_i x^2$')
            ax.set_xlabel('$y$')
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        return buf.getvalue()

    want = {f: png(f) for f in ('cm', 'stix', 'dejavusans')}
    assert len(set(want.values())) == 3
    got = []
    figs = {}                                           # (rcParams are one: figures keep theirs)
    for f in want:
        with fplt.rc_context({'mathtext.fontset': f}):
            fig, ax = fplt.subplots(figsize=(3, 2), dpi=60)
            ax.set_title(r'$\alpha\sum_i x^2$')
            ax.set_xlabel('$y$')
            figs[f] = fig

    def save(f):
        for _ in range(10):
            buf = io.BytesIO()
            figs[f].savefig(buf, format='png')
            got.append(buf.getvalue() == want[f])

    threads = [threading.Thread(target=save, args=(f,)) for f in want]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(got) == 30 and all(got)


def test_fontset_comes_back_with_the_rc():
    from fastplot import _text
    with fplt.rc_context({'mathtext.fontset': 'cm'}):
        _text.text_path('$x$')
        assert _text._FS.name == 'cm'
    _text.text_path('$x$')
    assert _text._FS.name == 'dejavusans'


def test_fastplot_has_no_legacy_cap():
    from fastplot import _draw
    assert not hasattr(_draw, 'LEGACY') and fastplot is not None
