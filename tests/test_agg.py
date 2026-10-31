"""The Agg port pixel for pixel against matplotlib's own Agg: strokes of every
width, cap and join, dashes, fills by either rule, clipping, and the blending
of one path over another."""
import numpy as np
import pytest

from fastplot import _agg, _draw, _jit

matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.patches import PathPatch                                 # noqa: E402
from matplotlib.path import Path                                         # noqa: E402

W, H, DPI = 200, 150, 100.0
PX = 72.0 / DPI

rng = np.random.default_rng(0)
LINES = {
    'corner': (np.array([[20.0, 20.0], [100.0, 120.0], [180.0, 30.5]]), False),
    'polyline': (np.array([[20.3, 20.1], [100.7, 40.0], [60.0, 120.0], [150.0, 130.2],
                           [180.0, 30.5]]), False),
    'ring': (np.array([[20.3, 20.1], [100.7, 40.0], [60.0, 120.0], [150.0, 130.2]]), True),
    'hairpin': (np.array([[20.0, 75.0], [180.0, 75.2], [20.0, 75.4]]), False),
    # as dense as a 10m coastline: strokes overlap inside a pixel
    'dense': (np.c_[np.linspace(10, 190, 800),
                    75 + 40 * np.sin(np.linspace(0, 40, 800)) + rng.normal(0, 0.3, 800)], False),
    'jagged': (np.c_[np.linspace(10, 190, 300), 75 + rng.normal(0, 3, 300)], False),
    'repeats': (np.array([[20.0, 20.0], [20.0, 20.0], [100.0, 90.0], [100.0, 90.0],
                          [170.0, 20.0]]), False),
    'off the canvas': (np.array([[-40.0, 70.0], [100.0, 20.0], [260.0, 90.0], [100.0, 200.0],
                                 [-30.0, 130.0]]), False),
}


def _axes():
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    return fig, ax


def _pixels(fig):
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return a.astype(np.int16)


def _mpl_stroke(xy, closed, lw, cap, join, dashes=None):
    fig, ax = _axes()
    v, codes = xy, None
    if closed:
        v, codes = np.vstack([xy, xy[:1]]), [1] + [2] * (len(xy) - 1) + [79]
    ls = '-' if dashes is None else (0, tuple(d / lw for d in dashes))
    with plt.rc_context({'path.simplify': False}):      # (read when the Path is made)
        ax.add_patch(PathPatch(Path(v, codes), fc='none', ec='k', lw=lw * PX, capstyle=cap,
                               joinstyle=join, snap=False, linestyle=ls))
    return _pixels(fig)


def _fp_stroke(xy, closed, lw, cap, join, dashes=None):
    c = np.full((H, W, 4), 255, np.uint8)
    m = len(xy)
    if closed:
        xy = np.vstack([xy, xy[:1]])
        m += 1
    with _jit.threads():
        _draw.draw_segments(c, _draw.segments(xy, np.array([0, m]), closed=np.array([closed])),
                            _draw.Styles((0, 0, 0, 255), lw, None if dashes is None else [dashes],
                                         caps=cap, joins=join), None)
    return c.astype(np.int16)


@pytest.mark.parametrize('name', list(LINES))
@pytest.mark.parametrize('lw', [0.4, 1.0, 1.5, 3.0, 8.0])
@pytest.mark.parametrize('cap,join', [('butt', 'round'), ('projecting', 'miter'),
                                      ('round', 'bevel')])
def test_stroke_is_agg_to_the_pixel(name, lw, cap, join):
    xy, closed = LINES[name]
    a = _mpl_stroke(xy, closed, lw, cap, join)
    assert (a == _fp_stroke(xy, closed, lw, cap, join)).all()


@pytest.mark.parametrize('name', ['polyline', 'ring', 'dense'])
@pytest.mark.parametrize('dashes', [(6.0, 3.0), (2.0, 2.0, 8.0, 3.0)])
def test_dashes_are_agg_to_the_pixel(name, dashes):
    xy, closed = LINES[name]
    a = _mpl_stroke(xy, closed, 1.5, 'butt', 'round', dashes)
    assert (a == _fp_stroke(xy, closed, 1.5, 'butt', 'round', dashes)).all()


def test_a_dense_line_is_as_dark_as_agg_makes_it():
    """Where a stroke meets itself inside a pixel the signed areas add:
    the ink of a dense line, which one stroke's coverage falls short of."""
    xy, closed = LINES['jagged']
    ink = 255 - _mpl_stroke(xy, closed, 1.0, 'butt', 'round')[..., 0]
    assert ink.sum() == (255 - _fp_stroke(xy, closed, 1.0, 'butt', 'round')[..., 0]).sum()


def test_many_threads_draw_what_one_does():
    n = 60000
    t = np.linspace(0, 2000, n)
    xy = np.c_[100 + 90 * np.sin(t * 0.0137) * np.cos(t * 0.31),
               75 + 70 * np.cos(t * 0.0171) * np.sin(t * 0.27)]
    seg = _draw.segments(xy, np.arange(0, n + 1, 500), closed=False)
    st = _draw.Styles((200, 0, 0, 128), 1.2)
    a, b = np.full((H, W, 4), 255, np.uint8), np.full((H, W, 4), 255, np.uint8)
    with _jit.threads(1):
        _draw.draw_segments(a, seg, st, None)
    with _jit.threads(8):
        _draw.draw_segments(b, seg, st, None)
    assert (a == b).all()


@pytest.mark.parametrize('rule', ['nonzero', 'evenodd'])
def test_fill_is_agg_to_the_pixel(rule):
    """Two rings of one path, wound both ways, and one that crosses them."""
    rings = [np.array([[20.2, 20.7], [170.3, 30.1], [150.9, 130.4], [30.5, 120.6]]),
             np.array([[60.1, 50.3], [120.4, 55.8], [110.2, 100.6], [70.9, 90.1]]),
             np.array([[100.0, 10.0], [190.0, 90.0], [40.0, 140.0]])]
    v = np.vstack([np.vstack([r, r[:1]]) for r in rings])
    codes = np.concatenate([[1, 2, 2, 2, 79], [1, 2, 2, 2, 79], [1, 2, 2, 79]])
    if rule == 'evenodd':                               # Agg's even-odd: mathtext's and hatches'
        c = np.full((H, W, 4), 255, np.uint8)
        _draw.fill_polygons(c, np.vstack(rings), np.array([0, 4, 8, 11]), (10, 90, 200, 180), None,
                            'evenodd')
        d = np.full((H, W, 4), 255, np.uint8)
        _draw.fill_polygons(d, v, np.array([0, 5, 10, 14]), (10, 90, 200, 180), None, 'evenodd')
        assert (c == d).all()                           # a repeated first point changes nothing
        return
    fig, ax = _axes()
    ax.add_patch(PathPatch(Path(v, codes), fc=(10 / 255, 90 / 255, 200 / 255, 180 / 255),
                           ec='none', lw=0, snap=False))
    a = _pixels(fig)
    c = np.full((H, W, 4), 255, np.uint8)
    _draw.fill_polygons(c, np.vstack(rings), np.array([0, 4, 8, 11]), (10, 90, 200, 180), None,
                        rule)
    assert (a == c).all()


def test_paths_blend_one_after_another():
    """Two strokes crossing, translucent: separate paths, as two patches,
    and the blending matplotlib's to the level."""
    fig, ax = _axes()
    p = [np.array([[20.0, 20.0], [180.0, 130.0]]), np.array([[20.0, 130.0], [180.0, 20.0]])]
    for q in p:
        ax.add_patch(PathPatch(Path(q), fc='none', ec=(0.8, 0.1, 0.1, 0.5), lw=4 * PX, snap=False))
    a = _pixels(fig)
    c = np.full((H, W, 4), 255, np.uint8)
    xy = np.vstack(p)
    _draw.draw_segments(c, _draw.segments(xy, np.array([0, 2, 4])),
                        _draw.Styles(np.array([204, 26, 26, 128], np.uint8), 4.0), None)
    assert np.abs(a - c).max() <= 1


def test_clip_box_keeps_the_winding():
    """A fill reaching past the clip on the left stays filled up to it."""
    c = np.zeros((20, 40, 4), np.uint8)
    _draw.fill_polygons(c, np.array([[-30.0, 2.0], [60.0, 2.0], [60.0, 18.0], [-30.0, 18.0]]),
                        np.array([0, 4]), (255, 255, 255, 255), (10, 0, 30, 20), 'nonzero')
    assert (c[2:18, 10:30, 3] == 255).all() and c[:, :10].max() == 0 and c[:, 30:].max() == 0


def test_stroke_outline_rings():
    """An open line has one ring; a closed one two."""
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 0.0]])
    args = (np.zeros(1, np.int64), np.array([2.0]), np.array([1], np.int64),
            np.array([1], np.int64), np.array([2.0]))
    _, rings, _ = _agg.stroke(xy, np.array([0, 4]), np.array([False]), *args)
    assert rings.size == 2
    _, rings, _ = _agg.stroke(xy, np.array([0, 4]), np.array([True]), *args)
    assert rings.size == 3
