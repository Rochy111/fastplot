"""The contour-label kernels against the numpy expressions they stand for
(matplotlib's locate_label and print_label, np.interp), bit for bit."""
import numpy as np

from fastplot import _clabel as L


def _bits(a):
    return np.asarray(a, np.float64).view(np.uint64)


def test_straightness_is_matplotlibs_expression():
    rng = np.random.default_rng(0)
    for bs in list(range(1, 140)) + [255, 256, 257, 513, 1000, 4097]:
        for trial in range(4):
            nb = int(rng.integers(1, 6))
            n = nb * bs - int(rng.integers(0, bs))      # np.resize wraps the last block
            vx = rng.normal(0, 10 ** rng.uniform(-2, 3), n)
            vy = rng.normal(0, 10 ** rng.uniform(-2, 3), n)
            if trial == 0:
                vx[rng.random(n) < 0.02] = np.nan
            xx, yy = np.resize(vx, (nb, bs)), np.resize(vy, (nb, bs))
            yf, yl, xf, xl = yy[:, :1], yy[:, -1:], xx[:, :1], xx[:, -1:]
            s = (yf - yy) * (xl - xf) - (xf - xx) * (yl - yf)
            with np.errstate(divide='ignore', invalid='ignore'):
                ref = (abs(s) / np.hypot(xl - xf, yl - yf)).sum(axis=-1)
            dx, dy = np.empty(nb), np.empty(nb)
            L._block_ends(vx, vy, nb, bs, dx, dy)
            out = np.empty(nb)
            L._straightness(vx, vy, nb, bs, np.hypot(dx, dy), out)
            assert np.array_equal(_bits(out), _bits(ref))


def test_interp_is_np_interp():
    rng = np.random.default_rng(3)
    for trial in range(400):
        n = int(rng.integers(1, 60))
        seg = rng.uniform(0, 3, n - 1)
        seg[rng.random(seg.size) < 0.2] = 0.0             # repeated arc lengths
        xp = np.concatenate([[0.0], np.cumsum(seg)])
        fp = rng.normal(0, 100, n)
        if trial % 9 == 0:
            fp[rng.random(n) < 0.2] = np.nan
        for x in np.concatenate([rng.uniform(-1, xp[-1] + 1, 10), xp, [np.nan, np.inf]]):
            assert _bits(L._interp(x, xp, fp)) == _bits(np.interp(x, xp, fp))


def test_line_stats_are_the_masked_reduceats():
    rng = np.random.default_rng(4)
    for trial in range(50):
        lens = rng.integers(0, 50, int(rng.integers(1, 30)))
        poff = np.r_[0, np.cumsum(lens)].astype(np.int64)
        xy = rng.uniform(-50, 250, (poff[-1], 2))
        xy[rng.random(poff[-1]) < 0.05] = np.nan
        b = (0, 0, 200, 150)
        inside = np.isfinite(xy).all(axis=1) & (xy[:, 0] >= b[0]) & (xy[:, 0] <= b[2]) & \
            (xy[:, 1] >= b[1]) & (xy[:, 1] <= b[3])
        nvis, ptp = np.empty(lens.size, np.int64), np.empty((lens.size, 2))
        L._line_stats(xy, poff, *map(float, b), nvis, ptp)
        for p in range(lens.size):
            v = xy[poff[p]:poff[p + 1]][inside[poff[p]:poff[p + 1]]]
            assert nvis[p] == len(v)
            assert np.array_equal(ptp[p], np.ptp(v, axis=0) if len(v) else np.zeros(2))
