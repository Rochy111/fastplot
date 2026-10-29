"""The value rasters: the separable path (cylindrical maps of lon/lat
grids, a grid index per column and per row) against the general kernel
(every pixel projected), bit for bit."""
import numpy as np
import pytest

from fastplot import _proj as P
from fastplot import _raster as R
from fastplot._render import View

MAPS = [P.PlateCarree(), P.PlateCarree(central_longitude=180),
        P.PlateCarree(central_longitude=-73.5), P.Mercator(),
        P.Mercator(central_longitude=100, min_latitude=10, max_latitude=70), P.Miller(),
        P.Geodetic()]


@pytest.mark.parametrize('seed', range(4))
def test_separable_equals_general(seed):
    rng = np.random.default_rng(seed)
    for trial in range(60):
        m = MAPS[trial % len(MAPS)]
        src = [P.PlateCarree(), P.Geodetic(), P.PlateCarree(central_longitude=40)][trial % 3]
        nx, ny = rng.integers(3, 60), rng.integers(3, 40)
        if rng.random() < 0.4:                          # global, periodic
            lon = np.linspace(0, 360, nx, endpoint=False) + rng.uniform(-5, 5)
        else:
            lon = np.sort(rng.uniform(-200, 200) + rng.uniform(0, 120)
                          * np.linspace(0, 1, nx) ** rng.choice([1, 1.3]))
        lat = np.linspace(-90, 90, ny) if rng.random() < 0.5 else np.sort(rng.uniform(-80, 80, ny))
        if rng.random() < 0.3:
            lat = lat[::-1]
        if rng.random() < 0.2:
            lon = lon[::-1]
        edges = bool(rng.random() < 0.2)
        z = rng.normal(size=(ny - edges, nx - edges)).astype(np.float32)
        z[rng.random(z.shape) < 0.05] = np.nan
        xa, ya = R.Axis(lon, angular=True, edges=edges), R.Axis(lat, edges=edges)
        xl, yl = m.x_limits, m.y_limits
        W, H = int(rng.integers(5, 300)), int(rng.integers(5, 200))
        cx, hw = rng.uniform(*xl), rng.uniform(0.01, 1.3) * (xl[1] - xl[0]) / 2
        cy, hh = rng.uniform(*yl), rng.uniform(0.01, 1.3) * (yl[1] - yl[0]) / 2
        view = View(m, cx - hw, 2 * hw / W, cy + hh, -2 * hh / H)
        zz = z[::-1] if ya.flip else z
        zz = zz[:, ::-1] if xa.flip else zz
        for mode in (R.NEAREST, R.BILINEAR):
            a = np.empty((H, W), np.float32)
            b = np.empty((H, W), np.float32)
            R.sample_rect(a, view, src, xa, ya, z, mode)
            R._sample_rect(b, view.kind, view.params, view.x0, view.dx, view.y0, view.dy,
                           src.kind, src.params, xa.params(), xa.coords, ya.params(),
                           ya.coords, zz, mode)
            assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


def _unwrap_copies_numpy(X, Y, offsets, period, xlo, xhi):
    """The seam handling as first written, in numpy: the reference."""
    nparts = offsets.size - 1
    X = X.copy()
    d = np.diff(X)
    adj = -period * np.round(d / period)
    adj[offsets[1:-1] - 1] = 0.0
    adj[~np.isfinite(adj)] = 0.0
    corr = np.concatenate([[0.0], np.cumsum(adj)])
    starts = offsets[:-1]
    lens = np.diff(offsets)
    corr -= np.repeat(corr[starts], lens)
    X += corr
    with np.errstate(invalid='ignore'):
        mn = np.minimum.reduceat(np.where(np.isfinite(X), X, np.inf), starts)
        mx = np.maximum.reduceat(np.where(np.isfinite(X), X, -np.inf), starts)
        mid = 0.5 * (mn + mx)
    shift = np.zeros(starts.size)
    ok = np.isfinite(mid)
    shift[ok] = -period * np.round((mid[ok] - 0.5 * (xlo + xhi)) / period)
    X += np.repeat(shift, lens)
    mn, mx = mn + shift, mx + shift
    xs, ys, offs, parts = [X], [Y], [offsets], [np.arange(nparts)]
    base = X.size
    for sgn, need in ((1.0, mn < xlo), (-1.0, mx > xhi)):
        idx = np.nonzero(need & ok)[0]
        if not idx.size:
            continue
        sel = np.concatenate([np.arange(offsets[k], offsets[k + 1]) for k in idx])
        xs.append(X[sel] + sgn * period)
        ys.append(Y[sel])
        offs.append(base + np.cumsum(np.r_[0, lens[idx]])[1:])
        parts.append(idx)
        base += sel.size
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(offs), np.concatenate(parts)


def test_unwrap_equals_numpy_reference():
    from fastplot._artists import _unwrap_copies
    rng = np.random.default_rng(2)

    def bits(u):
        u = np.asarray(u)
        return u.view(np.uint64) if u.dtype == np.float64 else u

    for trial in range(200):
        lens = rng.integers(1, 300, int(rng.integers(1, 40)))
        offs = np.r_[0, np.cumsum(lens)].astype(np.int64)
        n = offs[-1]
        period = [360.0, 2 * np.pi * 6370997.0, 40031000.123][trial % 3]
        X = np.cumsum(rng.normal(0, period * rng.uniform(0.001, 0.3), n)) + \
            rng.uniform(-2, 2) * period
        if trial % 5 == 0:
            X[rng.random(n) < 0.05] = np.nan
        if trial % 11 == 0:
            X[rng.random(n) < 0.05] = -0.0
        Y = rng.normal(size=n)
        xlo = rng.uniform(-0.6, 0.4) * period
        xhi = xlo + rng.uniform(0.05, 1.0) * period
        a = _unwrap_copies(X, Y, offs, period, xlo, xhi)
        b = _unwrap_copies_numpy(X, Y, offs, period, xlo, xhi)
        assert all(np.array_equal(bits(u), bits(v)) for u, v in zip(a, b))
