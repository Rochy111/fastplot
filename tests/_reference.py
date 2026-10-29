"""Whether the compiled references -- numpy's C, matplotlib's C++ -- were
built with a * b + c contracted to one fused multiply-add: clang's default
where FMA is native (arm64: Apple silicon), not on x86-64. fastplot's
kernels are not (numba keeps each operation's rounding), so fastplot's
numbers are the same on every machine; on such a build the references'
last bits are not, and the tests that check fastplot to the bit against
them check it to the rounding there instead: values within ~1 ulp, and a
point on an edge found in either of the triangles that share it."""
from fractions import Fraction

import numpy as np


def _fused():
    """np.interp(x, xp, fp) = slope * (x - xp[j]) + fp[j]: fused or not, on a
    case where the two differ."""
    rng = np.random.default_rng(0)
    for _ in range(10000):
        xp = np.array([0.0, rng.uniform(0.5, 3.0)])
        fp = rng.normal(0, 100, 2)
        x = float(rng.uniform(0.01, 0.99) * xp[1])
        s = float((fp[1] - fp[0]) / (xp[1] - xp[0]))
        d, c = x - float(xp[0]), float(fp[0])
        plain = s * d + c
        fused = float(Fraction(s) * Fraction(d) + Fraction(c))
        if plain != fused:
            return float(np.interp(x, xp, fp)) == fused
    return False


FUSED = _fused()


def same_bits(a, b):
    """a's float64s b's to the bit, or (FUSED) within 1 ulp."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if not FUSED:
        return np.array_equal(a.view(np.uint64), b.view(np.uint64))
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return False
    ok = ~np.isnan(a)
    ia, ib = a[ok].view(np.int64), b[ok].view(np.int64)
    return bool((np.abs(ia - ib) <= 1).all() or np.array_equal(a[ok], b[ok]))


def assert_same(a, b, rtol=1e-12):
    """Equal, or (FUSED) within rtol of the largest value."""
    a, b = np.asarray(a), np.asarray(b)
    if not FUSED:
        np.testing.assert_array_equal(a, b)
        return
    fa = a[np.isfinite(a)]
    scale = max(np.abs(fa).max(), 1e-300) if fa.size else 1.0
    np.testing.assert_allclose(a, b, rtol=0, atol=rtol * scale)


def _on_edge(tri, t, qx, qy, eps=1e-9):
    """Whether each (qx, qy) lies on the boundary of triangle t of tri."""
    v = tri.triangles[t]
    x, y = tri.x[v], tri.y[v]
    ex = x[:, [1, 2, 0]] - x
    ey = y[:, [1, 2, 0]] - y
    side = ex * (qy[:, None] - y) - ey * (qx[:, None] - x)     # 2 x the signed areas
    size = np.hypot(ex, ey).max(axis=1) ** 2
    return np.abs(side).min(axis=1) <= eps * size


def ties(m, f, qx, qy, order=True):
    """Where matplotlib's and fastplot's finders answer otherwise (none on
    an unfused build), after checking each such point is on an edge of the
    triangle found. order=False: only inside / outside compared (fastplot's
    triangles numbered its own way)."""
    qx, qy = np.ravel(qx), np.ravel(qy)
    a, b = np.ravel(m.get_trifinder()(qx, qy)), np.ravel(f.get_trifinder()(qx, qy))
    differ = (a < 0) != (b < 0) if not order else a != b
    if not FUSED:
        assert not differ.any(), '%d of %d points found otherwise' % (differ.sum(), differ.size)
        return differ
    for tri, t in ((m, a), (f, b)):
        k = differ & (t >= 0)
        assert _on_edge(tri, t[k], qx[k], qy[k]).all(), 'a point off the edges found otherwise'
    return differ
