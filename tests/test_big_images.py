"""Images much bigger than the figure (a Landsat band in a few megapixels):
the nearest cells in view coloured once where they are no more than the
pixels, else a pixel at a time after sampling -- the same picture either
way as colouring every pixel."""
import numpy as np
import pytest

import fastplot.pyplot as plt
from fastplot import _render


def _draw(z, lim=None, alpha=None, size=(4, 3), dpi=50):
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(z, cmap='viridis', interpolation='nearest', alpha=alpha)
    if lim:
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
    ax.axis('off')
    out = fig.draw(dpi=dpi)
    plt.close('all')
    return out


def _field():
    rng = np.random.default_rng(3)
    z = rng.random((900, 1200)).astype(np.float32)
    z[rng.random(z.shape) < 0.05] = np.nan
    return z


@pytest.mark.parametrize('lim,alpha', [
    (None, None),                                   # 1.08 M cells into 30 k pixels
    (None, 0.5),
    ((300, 360, 500, 460), None),                   # 60 x 40 cells in view: the cell path
    ((300, 700, 800, 500), 0.7),                    # 400 x 300 in view: more cells than px
])
def test_big_image_as_pixel_by_pixel(monkeypatch, lim, alpha):
    z = _field()
    got = _draw(z, lim, alpha)
    monkeypatch.setattr(_render.CpuCanvas, '_cells_direct', lambda self, *a, **k: False)
    want = _draw(z, lim, alpha)
    assert np.array_equal(got, want)


def test_cells_in_view_only_are_coloured(monkeypatch):
    """A zoom into a big image colours its cells in view, not all of them."""
    z = _field()
    seen = []
    orig = _render._cell_words

    def spy(coloring, zc, alpha):
        seen.append(zc.size)
        return orig(coloring, zc, alpha)
    monkeypatch.setattr(_render, '_cell_words', spy)
    _draw(z, (300, 360, 500, 460))
    assert seen and max(seen) <= 62 * 42
    seen.clear()
    _draw(z)                                        # drawn small: not cell by cell
    assert not seen
