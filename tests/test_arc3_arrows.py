"""_patches.arc3_arrows_flat: many arc3 arrows at once, the numbers
arrow_paths and flatten give one at a time."""
import numpy as np
import pytest

from fastplot import _patches, _path


@pytest.mark.parametrize('style', sorted(_patches.HEAD_STYLES) + ['-|>,head_width=0.4'])
@pytest.mark.parametrize('rad', [0.0, 0.3, -0.2])
def test_matches_one_at_a_time(style, rad):
    rng = np.random.default_rng(3)
    n = 60
    A = rng.uniform(0, 400, (n, 2))
    L = rng.choice([0.0, 0.5, 3.0, 4.5, 8.0, 40.0, 150.0], n)
    ang = rng.uniform(0, 2 * np.pi, n)
    B = A + np.stack([L * np.cos(ang), L * np.sin(ang)], 1)
    ms, sa, sb, lw = (rng.uniform(5, 20, n), rng.choice([0.0, 2.0, 2.7], n),
                      rng.choice([0.0, 2.0, 5.0], n), rng.uniform(0.5, 3, n))
    xy, offs, arrow, fill = _patches.arc3_arrows_flat(
        A, B, style, rad, mutation_scale=ms, shrinkA=sa, shrinkB=sb, linewidth=lw)
    parts = []
    for i in range(n):
        paths, fills = _patches.arrow_paths(A[i], B[i], style, 'arc3,rad=%r' % rad,
                                            mutation_scale=ms[i], shrinkA=sa[i],
                                            shrinkB=sb[i], linewidth=lw[i])
        for p, f in zip(paths, fills):
            pxy, po, _ = _path.flatten(p.vertices, p.codes)
            parts += [(i, f, pxy[po[j]:po[j + 1]]) for j in range(po.size - 1)]
    assert len(parts) == offs.size - 1
    for j, (i, f, pxy) in enumerate(parts):
        assert arrow[j] == i and fill[j] == f
        np.testing.assert_allclose(xy[offs[j]:offs[j + 1]], pxy, rtol=0, atol=1e-9)
