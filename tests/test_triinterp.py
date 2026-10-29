"""fastplot.tri's finder, interpolators, refiner and analyzer against
matplotlib.tri's own, on the same triangles (random Delaunay meshes, a
regular grid with its ties, masked meshes, clockwise given triangles).
To the bit, but where matplotlib's C++ was built with fused multiply-adds
(see _reference)'s rounding."""
import warnings

import numpy as np
import pytest
from _reference import FUSED, assert_same, ties

import fastplot.tri as ftri

mtri = pytest.importorskip('matplotlib.tri')


def _points(kind, rng):
    if kind == 'grid':
        gx, gy = np.meshgrid(np.arange(11.), np.arange(8.))
        return gx.ravel(), gy.ravel()
    n = {'random': 200, 'masked': 300, 'clockwise': 120}[kind]
    return rng.random(n) * 3 - 1, rng.random(n) * 2


def _pair(kind, seed=0):
    """(matplotlib's, fastplot's) Triangulation on the same points, triangles
    (matplotlib's Delaunay) and mask."""
    rng = np.random.default_rng(seed)
    x, y = _points(kind, rng)
    t = mtri.Triangulation(x, y).triangles.copy()
    mask = None
    if kind == 'masked':
        mask = rng.random(len(t)) < 0.2
    if kind == 'clockwise':
        flip = rng.random(len(t)) < 0.5
        t[flip] = t[flip][:, ::-1]
    return mtri.Triangulation(x, y, t.copy(), mask), ftri.Triangulation(x, y, t.copy(), mask)


def _probes(m, rng, n=4000):
    """Random points in and around the mesh, its vertices, its edges'
    midpoints and (on the grid) a lattice of points on edges and vertices."""
    x, y = m.x, m.y
    t = m.triangles
    e = np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    lo, hi = min(x.min(), y.min()) - 0.5, max(x.max(), y.max()) + 0.5
    q = np.arange(lo, hi, 0.25)
    gx, gy = np.meshgrid(q, q)
    return (np.r_[rng.uniform(lo, hi, n), x, (x[e[:, 0]] + x[e[:, 1]]) / 2, gx.ravel()],
            np.r_[rng.uniform(lo, hi, n), y, (y[e[:, 0]] + y[e[:, 1]]) / 2, gy.ravel()])


def _same_masked(a, b, rtol=0.0, well=None, tie=None):
    """The same mask and values: equal, or within rtol of the largest value
    (where `well`, and within 1e-6 elsewhere: see _well_shaped). FUSED: the
    masks but where the finders tie (`tie`, from ties()), the values where
    both have one, to 1e-12 at least."""
    a, b = np.ma.asarray(a), np.ma.asarray(b)
    ma, mb = np.ma.getmaskarray(a).ravel(), np.ma.getmaskarray(b).ravel()
    keep = np.ones(ma.size, bool) if tie is None else ~tie
    np.testing.assert_array_equal(ma[keep], mb[keep])
    fa, fb = a.filled(np.nan).ravel(), b.filled(np.nan).ravel()
    if FUSED:
        both = ~(ma | mb)
        fa, fb = fa[both], fb[both]
        well = None if well is None else well[both]
        rtol = max(rtol, 1e-12)
    if rtol == 0.0:
        np.testing.assert_array_equal(fa, fb)
        return
    scale = max(np.nanmax(np.abs(fa)), 1e-300) if np.isfinite(fa).any() else 1.0
    if well is None:
        well = np.ones(fa.size, bool)
    np.testing.assert_allclose(fa[well], fb[well], rtol=0, atol=rtol * scale)
    np.testing.assert_allclose(fa, fb, rtol=0, atol=max(rtol, 1e-6) * scale)


def _well_shaped(m, qx, qy, tri_index=None):
    """The points in triangles of circle ratio >= 1e-3. In flatter ones (a
    random Delaunay mesh's border slivers, ratios to 1e-6) the cubic
    element's barycentric coordinates come from a near-singular metric, and
    matplotlib's BLAS-ordered sums and fastplot's differ there by the
    rounding times the condition number (~1e-7 at worst in gradients)."""
    t = m.get_trifinder()(qx, qy) if tri_index is None else tri_index
    cr = np.ma.filled(mtri.TriAnalyzer(m).circle_ratios(rescale=False), 0.0)
    return np.ravel((t >= 0) & (cr[t] >= 1e-3))


KINDS = ['random', 'grid', 'masked', 'clockwise']


@pytest.mark.parametrize('kind', KINDS)
def test_trifinder(kind):
    m, f = _pair(kind)
    qx, qy = _probes(m, np.random.default_rng(1))
    b = f.get_trifinder()(qx, qy)
    assert b.dtype == np.int32 and b.shape == qx.shape
    ties(m, f, qx, qy)
    assert (b == -1).any() and (b >= 0).any()
    qx2, qy2 = qx[:qx.size // 2 * 2].reshape(-1, 2), qy[:qy.size // 2 * 2].reshape(-1, 2)
    assert f.get_trifinder()(qx2, qy2).shape == qx2.shape
    ties(m, f, qx2, qy2)
    assert f.get_trifinder() is f.get_trifinder()


@pytest.mark.parametrize('seed', range(12))
def test_trifinder_lattice_ties(seed):
    """Meshes of lattice points (colinear runs, exact ties everywhere) probed
    on a finer lattice: every edge, vertex and border tie as matplotlib."""
    rng = np.random.default_rng(seed)
    pts = np.unique(rng.integers(0, 16, (60, 2)), axis=0).astype(float)
    if seed % 3 == 0:
        pts = pts * 0.1 + 0.3                           # not dyadic
    x, y = pts[:, 0], pts[:, 1]
    t = mtri.Triangulation(x, y).triangles
    mask = rng.random(len(t)) < 0.15 if seed % 2 else None
    m, f = mtri.Triangulation(x, y, t, mask), ftri.Triangulation(x, y, t, mask)
    q = np.arange(-2, 70) / 4.0
    if seed % 3 == 0:
        q = q * 0.1 + 0.3
    qx, qy = np.meshgrid(q, q)
    ties(m, f, qx, qy)


def test_trifinder_follows_set_mask():
    m, f = _pair('random')
    qx, qy = _probes(m, np.random.default_rng(2))
    f.get_trifinder()(qx, qy)
    mask = np.random.default_rng(3).random(len(m.triangles)) < 0.3
    m.set_mask(mask)
    f.set_mask(mask)
    ties(m, f, qx, qy)
    f.set_mask(np.ones(len(m.triangles), bool))
    assert (f.get_trifinder()(qx, qy) == -1).all()


def test_trifinder_nonfinite_and_explicit():
    m, f = _pair('random')
    finder = ftri.TrapezoidMapTriFinder(f)
    assert isinstance(finder, ftri.TriFinder)
    out = finder(np.array([np.nan, np.inf, 0.2, -np.inf]), np.array([0.5, 0.5, np.nan, 1.0]))
    np.testing.assert_array_equal(out, -1)
    with pytest.raises(ValueError):
        finder([0, 1], [0])


def test_clockwise_reoriented_as_matplotlib():
    m, f = _pair('clockwise')
    given = f.triangles
    before = given.copy()
    m.get_trifinder()
    f.get_trifinder()
    np.testing.assert_array_equal(m.triangles, f.triangles)
    np.testing.assert_array_equal(given, before)       # the given array left alone


@pytest.mark.parametrize('kind', KINDS)
def test_plane_coefficients(kind):
    m, f = _pair(kind)
    z = np.sin(2 * m.x) * np.cos(3 * m.y) + m.x
    assert_same(m.calculate_plane_coefficients(z), f.calculate_plane_coefficients(z))


@pytest.mark.parametrize('kind', KINDS)
def test_linear(kind):
    m, f = _pair(kind)
    z = np.sin(2 * m.x) * np.cos(3 * m.y) + m.x
    qx, qy = _probes(m, np.random.default_rng(4))
    im, fi = mtri.LinearTriInterpolator(m, z), ftri.LinearTriInterpolator(f, z)
    tie = ties(m, f, qx, qy)
    _same_masked(im(qx, qy), fi(qx, qy), tie=tie)
    for a, b in zip(im.gradient(qx, qy), fi.gradient(qx, qy)):
        _same_masked(a, b, tie=tie)
    g = np.linspace(-1, 2, 37)
    gx, gy = np.meshgrid(g, g)
    out = fi(gx, gy)
    assert isinstance(out, np.ma.MaskedArray) and out.shape == gx.shape
    _same_masked(im(gx, gy), out, tie=ties(m, f, gx, gy))


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('cubic', ['min_E', 'geom', 'user'])
def test_cubic(kind, cubic):
    m, f = _pair(kind)
    z = np.sin(2 * m.x) * np.cos(3 * m.y) + m.x
    dz = None
    if cubic == 'user':
        dz = (2 * np.cos(2 * m.x) * np.cos(3 * m.y) + 1, -3 * np.sin(2 * m.x) * np.sin(3 * m.y))
    qx, qy = _probes(m, np.random.default_rng(5))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        im = mtri.CubicTriInterpolator(m, z, kind=cubic, dz=dz)
    fi = ftri.CubicTriInterpolator(f, z, kind=cubic, dz=dz)
    well = _well_shaped(m, qx, qy)
    assert well.mean() > 0.3
    tie = ties(m, f, qx, qy)
    _same_masked(im(qx, qy), fi(qx, qy), rtol=1e-10, well=well, tie=tie)
    for a, b in zip(im.gradient(qx, qy), fi.gradient(qx, qy)):
        _same_masked(a, b, rtol=1e-10, well=well, tie=tie)


def test_cubic_leaves_z_and_reproduces_planes():
    m, f = _pair('masked')
    z = 3 * f.x - 2 * f.y + 1
    keep = z.copy()
    fi = ftri.CubicTriInterpolator(f, z)
    np.testing.assert_array_equal(z, keep)
    qx, qy = _probes(m, np.random.default_rng(6))
    v = fi(qx, qy)
    inside = ~np.ma.getmaskarray(v)
    np.testing.assert_allclose(v[inside], (3 * qx - 2 * qy + 1)[inside], atol=1e-10)
    gx, gy = fi.gradient(qx, qy)
    np.testing.assert_allclose(gx[inside], 3, atol=1e-8)
    np.testing.assert_allclose(gy[inside], -2, atol=1e-8)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('subdiv', [0, 1, 2, 3])
def test_refine_triangulation(kind, subdiv):
    m, f = _pair(kind)
    rm, im = mtri.UniformTriRefiner(m).refine_triangulation(return_tri_index=True, subdiv=subdiv)
    rf, if_ = ftri.UniformTriRefiner(f).refine_triangulation(return_tri_index=True,
                                                            subdiv=subdiv)
    np.testing.assert_array_equal(rm.x, rf.x)
    np.testing.assert_array_equal(rm.y, rf.y)
    np.testing.assert_array_equal(rm.triangles, rf.triangles)
    np.testing.assert_array_equal(im, if_)
    assert (rm.mask is None) == (rf.mask is None)
    if rm.mask is not None:
        np.testing.assert_array_equal(rm.mask, rf.mask)
    assert isinstance(ftri.UniformTriRefiner(f).refine_triangulation(subdiv=subdiv),
                      ftri.Triangulation)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('subdiv', [1, 2, 3])
def test_refine_field(kind, subdiv):
    m, f = _pair(kind)
    z = np.sin(2 * m.x) * np.cos(3 * m.y) + m.x
    tm, zm = mtri.UniformTriRefiner(m).refine_field(z, subdiv=subdiv)
    tf, zf = ftri.UniformTriRefiner(f).refine_field(z, subdiv=subdiv)
    np.testing.assert_array_equal(tm.triangles, tf.triangles)
    _, found = mtri.UniformTriRefiner(m).refine_triangulation(True, subdiv)
    _same_masked(zm, zf, rtol=1e-10, well=_well_shaped(m, tm.x, tm.y, found))
    lin_m, lin_f = mtri.LinearTriInterpolator(m, z), ftri.LinearTriInterpolator(f, z)
    _, zm = mtri.UniformTriRefiner(m).refine_field(z, lin_m, subdiv=subdiv)
    _, zf = ftri.UniformTriRefiner(f).refine_field(z, lin_f, subdiv=subdiv)
    _same_masked(zm, zf)


@pytest.mark.parametrize('kind', KINDS)
def test_analyzer(kind):
    m, f = _pair(kind)
    am, af = mtri.TriAnalyzer(m), ftri.TriAnalyzer(f)
    assert am.scale_factors == af.scale_factors
    for rescale in (True, False):
        _same_masked(am.circle_ratios(rescale), af.circle_ratios(rescale))
        for ratio in (0.01, 0.1, 0.3):
            np.testing.assert_array_equal(am.get_flat_tri_mask(ratio, rescale),
                                          af.get_flat_tri_mask(ratio, rescale))


def test_fastplot_delaunay():
    """On fastplot's own Delaunay: with scipy (Qhull, as matplotlib calls
    it) the same triangles in the same order, so the same finder answers;
    without it (Bowyer-Watson) the same triangles in another order -- a
    probe on an edge then lands in either of its two -- so the same
    values. Linear values to rounding either way."""
    rng = np.random.default_rng(7)
    x, y = rng.random(500), rng.random(500)
    m, f = mtri.Triangulation(x, y), ftri.Triangulation(x, y)
    qx, qy = _probes(m, rng)
    try:
        import scipy.spatial  # noqa: F401
        tie = ties(m, f, qx, qy)
    except ImportError:
        assert ({frozenset(t) for t in m.triangles.tolist()}
                == {frozenset(t) for t in f.triangles.tolist()})
        tie = ties(m, f, qx, qy, order=False)
    z = np.hypot(x - 0.3, y - 0.6)
    _same_masked(mtri.LinearTriInterpolator(m, z)(qx, qy),
                 ftri.LinearTriInterpolator(f, z)(qx, qy), rtol=1e-13, tie=tie)
    _same_masked(mtri.CubicTriInterpolator(m, z)(qx, qy),
                 ftri.CubicTriInterpolator(f, z)(qx, qy), rtol=1e-10, well=_well_shaped(m, qx, qy),
                 tie=tie)


def test_api():
    for name in ('TriFinder', 'TrapezoidMapTriFinder', 'TriInterpolator',
                 'LinearTriInterpolator', 'CubicTriInterpolator', 'TriRefiner',
                 'UniformTriRefiner', 'TriAnalyzer', 'Triangulation'):
        assert name in ftri.__all__ and hasattr(ftri, name)
    m, f = _pair('random')
    with pytest.raises(ValueError):
        ftri.LinearTriInterpolator(f, np.zeros(3))
    with pytest.raises(ValueError):
        ftri.CubicTriInterpolator(f, f.x, kind='bogus')
    with pytest.raises(ValueError):
        ftri.CubicTriInterpolator(f, f.x, kind='user')
    with pytest.raises(TypeError):
        ftri.UniformTriRefiner(m)
    fi = ftri.LinearTriInterpolator(f, f.x, trifinder=f.get_trifinder())
    assert float(fi(0.5, 0.5)) == pytest.approx(0.5)
