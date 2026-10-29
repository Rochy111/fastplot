"""Markers, patches, box styles and fancy arrows against matplotlib's own
geometry (vertices and codes to ~1e-9), and the flattening of curves to
polylines within tolerance."""
import time

import numpy as np
import pytest

from fastplot import _markers, _patches, _path
from fastplot._path import Path

mpl = pytest.importorskip('matplotlib')
from matplotlib import patches as mpatches  # noqa: E402
from matplotlib.markers import MarkerStyle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.transforms import Affine2D, IdentityTransform  # noqa: E402

ATOL = 1e-9


def _codes(p):
    """A path's codes, None made explicit (MOVETO, LINETO ...)."""
    if p.codes is not None:
        return np.asarray(p.codes, np.uint8)
    c = np.full(len(p.vertices), 2, np.uint8)
    if len(c):
        c[0] = 1
    return c


def assert_same_path(ours, theirs, atol=ATOL):
    v1, v2 = np.asarray(ours.vertices), np.asarray(theirs.vertices)
    assert v1.shape == v2.shape, (v1.shape, v2.shape)
    np.testing.assert_allclose(v1, v2, rtol=0, atol=atol)
    np.testing.assert_array_equal(_codes(ours), _codes(theirs))


# ------------------------------------------------------------------ markers

ALL_MARKERS = [m for m in MarkerStyle.markers]
EXTRA_MARKERS = [(5, 0), (6, 1), (7, 2), (5, 0, 30.0), (4, 1, 45), (8, 2, 10),
                 [[-1, -1], [1, 0], [0, 2]],
                 np.array([[0.0, 0.0], [2.0, 1.0], [1.0, 3.0], [-1.0, 1.0]])]
SIZE = 13.7                                     # px


def _mpl_marker_paths(ms, size):
    flip = Affine2D().scale(1.0, -1.0)
    pixel = isinstance(ms.get_marker(), str) and ms.get_marker() == ','
    scale = Affine2D() if pixel else Affine2D().scale(size)
    p = ms.get_path().transformed(ms.get_transform() + scale + flip)
    alt = ms.get_alt_path()
    if alt is not None:
        alt = alt.transformed(ms.get_alt_transform() + scale + flip)
    return p, alt


def _marker_id(m):
    return repr(m) if not isinstance(m, np.ndarray) else 'ndarray'


@pytest.mark.parametrize('fillstyle', _markers.FILLSTYLES)
@pytest.mark.parametrize('marker', ALL_MARKERS + EXTRA_MARKERS, ids=_marker_id)
def test_marker_matches_matplotlib(marker, fillstyle):
    ms = MarkerStyle(marker, fillstyle)
    g = _markers.marker_geom(marker, SIZE, fillstyle)
    p, alt = _mpl_marker_paths(ms, SIZE)
    assert_same_path(g.fill, p)
    if alt is None:
        assert g.fill_alt is None
    else:
        assert_same_path(g.fill_alt, alt)
    assert g.filled == ms.is_filled()
    assert g.joinstyle == ms.get_joinstyle()
    assert g.capstyle == ms.get_capstyle()
    assert bool(g) == bool(ms)
    if len(p.vertices):                         # stroked: the path, then the alt one
        want = p if alt is None else MPath.make_compound_path(p, alt)
        assert_same_path(g.stroke, want)
    else:
        assert g.stroke is None


def test_marker_path_marker_and_none():
    verts = np.array([[0.0, 0.0], [1.0, 0.5], [0.3, 1.2], [0.0, 0.0]])
    codes = [1, 3, 3, 79]
    ms = MarkerStyle(MPath(verts, codes))
    g = _markers.marker_geom(Path(verts, codes), SIZE)
    assert_same_path(g.fill, _mpl_marker_paths(ms, SIZE)[0])
    for m in (None, 'none', 'None', ' ', ''):
        g = _markers.marker_geom(m, SIZE)
        assert not g and g.stroke is None and not g.filled
    with pytest.raises(ValueError):
        _markers.marker_geom('$x$', SIZE)
    with pytest.raises(ValueError):
        _markers.marker_geom('Q', SIZE)


@pytest.mark.parametrize('marker', ['^', 'o', 's', 'D', '*', '1', 'P'])
@pytest.mark.parametrize('fillstyle', ['full', 'left', 'top'])
def test_marker_user_transform_and_styles(marker, fillstyle):
    ms = MarkerStyle(marker, fillstyle, transform=Affine2D().rotate_deg(30).scale(1.5, 0.7),
                     capstyle='round', joinstyle='bevel')
    U = _path.chain(_path.rotation(30), _path.scaling(1.5, 0.7))
    g = _markers.marker_geom(marker, SIZE, fillstyle, transform=U, capstyle='round',
                             joinstyle='bevel')
    p, alt = _mpl_marker_paths(ms, SIZE)
    assert_same_path(g.fill, p)
    if alt is not None:
        assert_same_path(g.fill_alt, alt)
    assert g.joinstyle == ms.get_joinstyle() and g.capstyle == ms.get_capstyle()


def test_marker_edge_width():
    assert _markers.marker_geom('o', 5).edge_width(1.5) == 1.5
    assert _markers.marker_geom(',', 5).edge_width(1.5) == 0.0


def test_place_markers():
    g = _markers.marker_geom('o', 9.0, 'left')
    centers = np.array([[10.0, 20.0], [np.nan, 1.0], [300.5, 40.25], [55.0, 66.0]])
    pl = _markers.place(g, centers)
    np.testing.assert_array_equal(pl.index, [0, 2, 3])
    for part, path, nper in ((pl.fill, g.fill, pl.rings), (pl.fill_alt, g.fill_alt, pl.alt_rings),
                             (pl.stroke, g.stroke, pl.lines)):
        xy, offs = part
        fx, fo, _ = _path.flatten(path.vertices, path.codes)
        assert nper == fo.size - 1
        assert offs.size == 3 * nper + 1 and xy.shape[0] == 3 * fx.shape[0]
        for j, c in enumerate(centers[pl.index]):
            seg = xy[offs[j * nper]:offs[(j + 1) * nper]]
            np.testing.assert_allclose(seg, fx + c, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(offs[j * nper:(j + 1) * nper + 1] - offs[j * nper], fo)
    empty = _markers.place(_markers.marker_geom('none', 5), centers)
    assert empty.fill is None and empty.stroke is None


# --------------------------------------------------------------- flattening

def _bezier_samples(p0, ctrl, n=2000):
    """Dense points of the Bezier from p0 through ctrl (control points and end)."""
    pts = np.vstack([p0, ctrl])
    t = np.linspace(0, 1, n)[:, None]
    if len(pts) == 3:
        return (1 - t) ** 2 * pts[0] + 2 * (1 - t) * t * pts[1] + t ** 2 * pts[2]
    if len(pts) == 4:
        return ((1 - t) ** 3 * pts[0] + 3 * (1 - t) ** 2 * t * pts[1]
                + 3 * (1 - t) * t ** 2 * pts[2] + t ** 3 * pts[3])
    return (1 - t) * pts[0] + t * pts[1]


def _dist_to_polyline(pts, poly):
    a, b = poly[:-1], poly[1:]
    d = b - a
    L2 = (d ** 2).sum(1)
    L2[L2 == 0] = 1
    t = np.clip(((pts[:, None, :] - a[None]) * d[None]).sum(2) / L2[None], 0, 1)
    q = a[None] + t[..., None] * d[None]
    return np.sqrt(((pts[:, None, :] - q) ** 2).sum(2)).min(1)


def _true_curve(vertices, codes):
    """Dense points along each part of a path (for the distance checks)."""
    parts, cur, start, pts = [], None, None, []
    i = 0
    while i < len(vertices):
        c = codes[i]
        if c == 1:
            if len(pts) > 1:
                parts.append(np.vstack(pts))
            cur = start = vertices[i]
            pts = [cur[None]]
            i += 1
        elif c == 2:
            s = _bezier_samples(cur, vertices[i:i + 1], 50)
            pts.append(s)
            cur = vertices[i]
            i += 1
        elif c in (3, 4):
            m = c - 1
            pts.append(_bezier_samples(cur, vertices[i:i + m]))
            cur = vertices[i + m - 1]
            i += m
        elif c == 79:
            pts.append(_bezier_samples(cur, start[None], 50))
            cur = start
            i += 1
    if len(pts) > 1:
        parts.append(np.vstack(pts))
    return parts


def _random_path(rng, scale):
    v, c = [], []
    for part in range(rng.integers(1, 4)):
        v.append(rng.normal(size=2) * scale)
        c.append(1)
        for seg in range(rng.integers(1, 5)):
            kind = rng.choice([2, 3, 4])
            m = {2: 1, 3: 2, 4: 3}[kind]
            v.extend(rng.normal(size=(m, 2)) * scale)
            c.extend([kind] * m)
        if rng.random() < 0.5:
            v.append([0.0, 0.0])
            c.append(79)
    return np.array(v), np.array(c, np.uint8)


@pytest.mark.parametrize('tol', [0.05, 0.1, 0.5])
def test_flatten_within_tolerance(tol):
    rng = np.random.default_rng(1)
    for k in range(40):
        v, c = _random_path(rng, [1.0, 20.0, 300.0][k % 3])
        xy, offs, closed = _path.flatten(v, c, tol)
        truth = _true_curve(v, c)
        assert len(truth) == offs.size - 1 == closed.size
        assert offs.size - 1 == np.count_nonzero(c == 1)       # every MOVETO a part
        for j, curve in enumerate(truth):
            poly = xy[offs[j]:offs[j + 1]]
            assert _dist_to_polyline(curve, poly).max() <= tol * (1 + 1e-6)
            mids = np.vstack([poly, 0.5 * (poly[1:] + poly[:-1])])
            assert _dist_to_polyline(mids, curve).max() <= tol + 1e-3 * (1 + np.ptp(curve))
            if closed[j]:
                np.testing.assert_array_equal(poly[0], poly[-1])


def test_flatten_closes_moves_and_breaks():
    xy, offs, closed = _path.circle((5, 5), 30).flatten(0.1)
    assert offs.size == 2 and closed[0] and (xy[0] == xy[-1]).all()
    assert np.abs(np.hypot(*(xy - 5).T) - 30).max() < 0.1     # (the Beziers are near circular)
    # polygons: CLOSEPOLY adds the first point once; MOVETO splits
    v = [[0, 0], [10, 0], [10, 10], [99, 99], [20, 0], [30, 0], [30, 5], [20, 0], [0, 0]]
    c = [1, 2, 2, 79, 1, 2, 2, 2, 79]
    xy, offs, closed = _path.flatten(v, c)
    np.testing.assert_array_equal(offs, [0, 4, 8])
    np.testing.assert_array_equal(closed, [True, True])
    np.testing.assert_array_equal(xy[:4], [[0, 0], [10, 0], [10, 10], [0, 0]])
    np.testing.assert_array_equal(xy[4:], [[20, 0], [30, 0], [30, 5], [20, 0]])
    # a polyline (codes None), NaN breaking it, single points dropped
    v = [[0, 0], [1, 1], [np.nan, 0], [2, 2], [3, 3], [4, 4], [np.nan, np.nan], [9, 9]]
    xy, offs, closed = _path.flatten(v, None)
    np.testing.assert_array_equal(offs, [0, 2, 5])
    assert not closed.any()
    # LINETO after CLOSEPOLY starts from the closed part's start
    xy, offs, closed = _path.flatten([[0, 0], [5, 0], [5, 5], [0, 0], [0, 7]], [1, 2, 2, 79, 2])
    np.testing.assert_array_equal(offs, [0, 4, 6])
    np.testing.assert_array_equal(xy[4:], [[0, 0], [0, 7]])


def test_flatten_many_matches_flatten():
    rng = np.random.default_rng(7)
    paths = [_random_path(rng, 50.0) for _ in range(60)]
    v = np.concatenate([p[0] for p in paths])
    c = np.concatenate([p[1] for p in paths])
    po = np.cumsum([0] + [len(p[0]) for p in paths])
    xy, offs, closed, pp = _path.flatten_many(v, c, po, 0.2)
    for j, (vj, cj) in enumerate(paths):
        x1, o1, c1 = _path.flatten(vj, cj, 0.2)
        a, b = pp[j], pp[j + 1]
        np.testing.assert_array_equal(xy[offs[a]:offs[b]], x1)
        np.testing.assert_array_equal(offs[a:b + 1] - offs[a], o1)
        np.testing.assert_array_equal(closed[a:b], c1)
    # polylines (codes None)
    xy, offs, closed, pp = _path.flatten_many(np.arange(20.0).reshape(10, 2), None, [0, 3, 10])
    np.testing.assert_array_equal(offs, [0, 3, 10])
    np.testing.assert_array_equal(pp, [0, 1, 2])
    # paths ending on a lone (dropped) point: nothing written past their own points
    v = np.array([[0, 0], [1, 0], [5, 5], [2, 2], [3, 2], [7, 7]], float)
    c = np.array([1, 2, 1, 1, 2, 1], np.uint8)
    xy, offs, closed, pp = _path.flatten_many(v, c, [0, 3, 6])
    np.testing.assert_array_equal(xy, [[0, 0], [1, 0], [2, 2], [3, 2]])
    np.testing.assert_array_equal(offs, [0, 2, 4])
    np.testing.assert_array_equal(pp, [0, 1, 2])


def test_unit_shapes_match_matplotlib():
    for ours, theirs in [
            (_path.unit_circle(), MPath.unit_circle()),
            (_path.circle((3, -2), 7.5), MPath.circle((3, -2), 7.5)),
            (_path.unit_circle_righthalf(), MPath.unit_circle_righthalf()),
            (_path.unit_rectangle(), MPath.unit_rectangle()),
            (_path.unit_regular_polygon(7), MPath.unit_regular_polygon(7)),
            (_path.unit_regular_star(5, 0.3), MPath.unit_regular_star(5, 0.3)),
            (_path.unit_regular_asterisk(6), MPath.unit_regular_asterisk(6)),
            (_path.arc(10, 250), MPath.arc(10, 250)),
            (_path.arc(-30, 400), MPath.arc(-30, 400)),
            (_path.arc(0, 360), MPath.arc(0, 360)),
            (_path.arc(20, 50, n=3), MPath.arc(20, 50, n=3)),
            (_path.wedge(45, 300), MPath.wedge(45, 300))]:
        assert_same_path(ours, theirs)
    M = _path.affine((2.0, 3.0), 30.0, (5.0, -1.0))
    A = Affine2D().scale(2.0, 3.0).rotate_deg(30.0).translate(5.0, -1.0).get_matrix()
    np.testing.assert_allclose(M, A, atol=1e-15)
    xy = np.random.default_rng(0).normal(size=(10, 2))
    np.testing.assert_allclose(_path.apply(M, xy), Affine2D(A).transform(xy), atol=1e-12)


# ------------------------------------------------------------------ patches

def _mpl_patch_path(p):
    return p.get_path().transformed(p.get_patch_transform())


@pytest.mark.parametrize('ours,theirs', [
    (lambda: _patches.rectangle((1, 2), 3, 4),
     lambda: mpatches.Rectangle((1, 2), 3, 4)),
    (lambda: _patches.rectangle((1, 2), 3, -4, angle=35),
     lambda: mpatches.Rectangle((1, 2), 3, -4, angle=35)),
    (lambda: _patches.rectangle((1, 2), 3, 4, angle=-20, rotation_point='center'),
     lambda: mpatches.Rectangle((1, 2), 3, 4, angle=-20, rotation_point='center')),
    (lambda: _patches.rectangle((1, 2), 3, 4, angle=70, rotation_point=(0.5, -1.0)),
     lambda: mpatches.Rectangle((1, 2), 3, 4, angle=70, rotation_point=(0.5, -1.0))),
    (lambda: _patches.circle((3, 4), 2.5), lambda: mpatches.Circle((3, 4), 2.5)),
    (lambda: _patches.ellipse((3, 4), 5, 2, angle=33),
     lambda: mpatches.Ellipse((3, 4), 5, 2, angle=33)),
    (lambda: _patches.polygon([[0, 0], [3, 1], [1, 4]]),
     lambda: mpatches.Polygon([[0, 0], [3, 1], [1, 4]])),
    (lambda: _patches.polygon([[0, 0], [3, 1], [1, 4], [0, 0]]),
     lambda: mpatches.Polygon([[0, 0], [3, 1], [1, 4], [0, 0]])),
    (lambda: _patches.polygon([[0, 0], [3, 1], [1, 4], [0, 0]], closed=False),
     lambda: mpatches.Polygon([[0, 0], [3, 1], [1, 4], [0, 0]], closed=False)),
    (lambda: _patches.polygon([[0, 0], [3, 1], [1, 4]], closed=False),
     lambda: mpatches.Polygon([[0, 0], [3, 1], [1, 4]], closed=False)),
    (lambda: _patches.regular_polygon((1, 1), 6, radius=3, orientation=0.4),
     lambda: mpatches.RegularPolygon((1, 1), 6, radius=3, orientation=0.4)),
    (lambda: _patches.wedge((1, 2), 3, 20, 145),
     lambda: mpatches.Wedge((1, 2), 3, 20, 145)),
    (lambda: _patches.wedge((1, 2), 3, 20, 290, width=1.2),
     lambda: mpatches.Wedge((1, 2), 3, 20, 290, width=1.2)),
    (lambda: _patches.wedge((0, 0), 3, 0, 360, width=1),
     lambda: mpatches.Wedge((0, 0), 3, 0, 360, width=1)),
    (lambda: _patches.wedge((0, 0), 3, 10, 370),
     lambda: mpatches.Wedge((0, 0), 3, 10, 370)),
    (lambda: _patches.arc((1, 2), 4, 4, theta1=30, theta2=200),
     lambda: mpatches.Arc((1, 2), 4, 4, theta1=30, theta2=200)),
    (lambda: _patches.arc((1, 2), 6, 2, angle=25, theta1=30, theta2=300),
     lambda: mpatches.Arc((1, 2), 6, 2, angle=25, theta1=30, theta2=300)),
    (lambda: _patches.arc((1, 2), 6, 2, angle=25),
     lambda: mpatches.Arc((1, 2), 6, 2, angle=25)),
], ids=lambda f: '')
def test_patch_matches_matplotlib(ours, theirs):
    assert_same_path(ours(), _mpl_patch_path(theirs()))


def test_shadow_offset():
    r = mpatches.Rectangle((1, 2), 3, 4)
    s = mpatches.Shadow(r, 5, -3)
    s._update_transform(type('R', (), {'points_to_pixels': staticmethod(lambda v: v)})())
    assert_same_path(_patches.shadow(_patches.rectangle((1, 2), 3, 4), 5, -3),
                     _mpl_patch_path(s))


BOXES = [('square', {}), ('square', {'pad': 0.5}), ('circle', {}), ('ellipse', {'pad': 0.1}),
         ('round', {}), ('round', {'pad': 0.2, 'rounding_size': 0.8}), ('round4', {}),
         ('round4', {'rounding_size': 0.3}), ('sawtooth', {}), ('sawtooth', {'tooth_size': 0.4}),
         ('roundtooth', {}), ('roundtooth', {'pad': 0.5}), ('larrow', {}), ('rarrow', {}),
         ('darrow', {}), ('rarrow', {'head_width': 0.6, 'head_angle': 60}),
         ('rarrow', {'head_width': 2.5, 'head_angle': 250}),
         ('darrow', {'head_width': 2.0, 'head_angle': 300}), ('larrow', {'head_angle': 120})]


@pytest.mark.parametrize('name,kw', BOXES)
@pytest.mark.parametrize('box', [(10.0, 20.0, 80.0, 25.0), (-3.0, 4.0, 12.5, 40.0)])
def test_boxstyle_matches_matplotlib(name, kw, box):
    ms = 14.0
    theirs = mpatches.BoxStyle(name, **kw)(*box, ms)
    assert_same_path(_patches.boxstyle_path(name, *box, ms, **kw), theirs)
    spec = ','.join([name] + ['%s=%r' % i for i in kw.items()])  # 'name,key=value'
    assert_same_path(_patches.boxstyle_path(spec, *box, ms), theirs)
    p = mpatches.FancyBboxPatch(box[:2], box[2], box[3], boxstyle=mpatches.BoxStyle(name, **kw),
                                mutation_scale=ms, mutation_aspect=0.7)
    ours = _patches.fancy_bbox(box[:2], box[2], box[3], name, mutation_scale=ms,
                               mutation_aspect=0.7, **kw)
    assert_same_path(ours, p.get_path())
    # on a y-down screen: matplotlib's outline mirrored
    down = _patches.fancy_bbox((box[0], -(box[1] + box[3])), box[2], box[3], name,
                               mutation_scale=ms, ydown=True, **kw)
    up = _patches.fancy_bbox(box[:2], box[2], box[3], name, mutation_scale=ms, **kw)
    np.testing.assert_allclose(down.vertices * [1, -1], up.vertices, atol=1e-9)


# ------------------------------------------------------------------- arrows

ARROWSTYLES = ['-', '->', '<-', '<->', '-|>', '<|-', '<|-|>', '-[', ']-', ']-[', '|-|', ']->',
               '<-[', 'simple', 'fancy', 'wedge',
               '->,head_length=0.8,head_width=0.5', ']-[,widthA=2,lengthA=0.5,angleA=20',
               '-[,widthB=0.5,lengthB=1,angleB=-30', '|-|,widthA=0.5,widthB=1.5',
               'simple,head_length=1,head_width=1.2,tail_width=0.4',
               'fancy,head_length=0.6,head_width=0.7,tail_width=0.2',
               'wedge,tail_width=1.2,shrink_factor=0.3']
QUADRATIC = ('simple', 'fancy', 'wedge')
CONNECTIONS = ['arc3', 'arc3,rad=0.3', 'arc3,rad=-0.5', 'angle3', 'angle3,angleA=0,angleB=90',
               'angle', 'angle,angleA=-90,angleB=180,rad=8', 'arc',
               'arc,angleA=0,angleB=90,armA=30,armB=20,rad=5', 'arc,armA=40,rad=10',
               'arc,armB=40,angleB=45', 'bar', 'bar,fraction=-0.2', 'bar,armA=10,angle=30']
POSITIONS = [((10.0, 15.0), (180.0, 95.0)), ((200.0, 50.0), (40.0, 160.0)),
             ((0.0, 0.0), (0.5, 70.0))]


def _mpl_arrow(posA, posB, style, conn, **kw):
    p = mpatches.FancyArrowPatch(posA, posB, arrowstyle=style, connectionstyle=conn,
                                 transform=IdentityTransform(), **kw)
    return p.get_path(), p._get_path_in_displaycoord()[1]


def _ours(posA, posB, style, conn, **kw):
    paths, fills = _patches.arrow_paths(posA, posB, style, conn, ydown=False, **kw)
    return _path.make_compound_path(*paths), fills


def _compare_arrow(posA, posB, style, conn, kw, ours_kw=None):
    try:
        theirs, tfill = _mpl_arrow(posA, posB, style, conn, **kw)
    except ValueError:
        with pytest.raises(ValueError):
            _ours(posA, posB, style, conn, **(ours_kw or kw))
        return
    ours, ofill = _ours(posA, posB, style, conn, **(ours_kw or kw))
    assert_same_path(ours, theirs, atol=1e-8)
    assert list(np.atleast_1d(ofill)) == list(np.atleast_1d(tfill))


@pytest.mark.parametrize('style', ARROWSTYLES)
@pytest.mark.parametrize('conn', CONNECTIONS)
def test_fancy_arrow_matches_matplotlib(style, conn):
    if style.split(',')[0] in QUADRATIC and conn.split(',')[0] not in ('arc3', 'angle3'):
        with pytest.raises(ValueError):
            _ours(*POSITIONS[0], style, conn)
        return
    for k, (posA, posB) in enumerate(POSITIONS):
        kw = dict(mutation_scale=[10.0, 23.0, 7.5][k], shrinkA=[2.0, 0.0, 6.0][k],
                  shrinkB=[2.0, 9.0, 0.0][k], linewidth=[1.0, 2.5, 0.5][k])
        _compare_arrow(posA, posB, style, conn, kw)


@pytest.mark.parametrize('style', ['->', '-|>', 'simple', 'fancy', ']-[', '<|-|>'])
def test_fancy_arrow_clipped_at_boxes(style):
    posA, posB = (50.0, 40.0), (220.0, 160.0)
    boxA, boxB = (20.0, 25.0, 90.0, 55.0), (200.0, 140.0, 260.0, 175.0)
    pA = mpatches.Rectangle(boxA[:2], boxA[2] - boxA[0], boxA[3] - boxA[1],
                            transform=IdentityTransform())
    pB = mpatches.Rectangle(boxB[:2], boxB[2] - boxB[0], boxB[3] - boxB[1],
                            transform=IdentityTransform())
    for conn in ('arc3,rad=0.25', 'angle3', 'arc3'):
        kw = dict(mutation_scale=12.0, shrinkA=3.0, shrinkB=4.0, linewidth=1.5)
        _compare_arrow(posA, posB, style, conn, dict(kw, patchA=pA, patchB=pB),
                       dict(kw, patchA=boxA, patchB=boxB))
    # a FancyBboxPatch frame as patchA, given as its path
    fb = mpatches.FancyBboxPatch(boxA[:2], boxA[2] - boxA[0], boxA[3] - boxA[1],
                                 boxstyle='round,pad=0.3', mutation_scale=10,
                                 transform=IdentityTransform(), edgecolor='none')
    frame = _patches.fancy_bbox(boxA[:2], boxA[2] - boxA[0], boxA[3] - boxA[1], 'round,pad=0.3',
                                mutation_scale=10)
    kw = dict(mutation_scale=12.0, shrinkA=2.0, shrinkB=2.0)
    theirs, _ = _mpl_arrow(posA, posB, style, 'arc3,rad=0.2', patchA=fb, **kw)
    ours, _ = _ours(posA, posB, style, 'arc3,rad=0.2', patchA=frame, **kw)
    assert_same_path(ours, theirs, atol=1e-6)


def test_fancy_arrow_ydown_and_split():
    posA, posB = (30.0, 200.0), (250.0, 40.0)
    box = (10.0, 180.0, 60.0, 215.0)
    down, fd = _patches.arrow_paths(posA, posB, '-|>', 'arc3,rad=0.3', mutation_scale=15,
                                    patchA=box)
    up, fu = _patches.arrow_paths((30.0, -200.0), (250.0, -40.0), '-|>', 'arc3,rad=0.3',
                                  mutation_scale=15, patchA=(10.0, -215.0, 60.0, -180.0),
                                  ydown=False)
    assert fd == fu == [False, True]
    for a, b in zip(down, up):
        np.testing.assert_allclose(a.vertices * [1, -1], b.vertices, atol=1e-9)
        np.testing.assert_array_equal(_codes(a), _codes(b))
    fill, stroke = _patches.fancy_arrow(posA, posB, '-|>', 'arc3,rad=0.3', mutation_scale=15)
    assert len(fill) == 1 and len(stroke) == 2
    fill, stroke = _patches.fancy_arrow(posA, posB, 'fancy', mutation_scale=15)
    assert len(fill) == 1 and len(stroke) == 1 and fill[0] is stroke[0]
    with pytest.raises(ValueError):
        _patches.fancy_arrow(posA, posB, 'nope')
    with pytest.raises(TypeError):
        _patches.fancy_arrow(posA, posB, '->,tail_width=1')


def test_fancy_arrow_with_path_and_aspect():
    v = np.array([[10.0, 10.0], [80.0, 120.0], [200.0, 30.0]])
    mp = MPath(v, [1, 3, 3])
    for style in ('->', 'simple', '<|-|>'):
        p = mpatches.FancyArrowPatch(path=mp, arrowstyle=style, mutation_scale=9,
                                     mutation_aspect=0.6, transform=IdentityTransform())
        paths, _ = _patches.arrow_paths(path=Path(v, [1, 3, 3]), arrowstyle=style,
                                        mutation_scale=9, mutation_aspect=0.6, ydown=False)
        assert_same_path(_path.make_compound_path(*paths), p.get_path())


# -------------------------------------------------------------- performance

def test_place_100k_circles_speed():
    g = _markers.marker_geom('o', 6 * 100 / 72)
    centers = np.random.default_rng(0).uniform(0, 1000, (100_000, 2))
    _markers.place(g, centers[:10])                     # (compiled)
    g._flat = None
    t = time.perf_counter()
    pl = _markers.place(g, centers)
    ms = (time.perf_counter() - t) * 1e3
    print('\nflatten + place 100k circle markers: %.1f ms (%d fill points)' %
          (ms, pl.fill[0].shape[0]))
    assert pl.fill[1].size == 100_000 * pl.rings + 1
