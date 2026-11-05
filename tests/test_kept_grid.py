"""Dashed grid lines (ax.grid, and a map's grid at its set_xticks) kept
across figures as a map's graticule is: the same pixels as drawn anew."""
import numpy as np
import pytest

import fastplot
import fastplot.crs as ccrs
from fastplot import _artists
from fastplot._style import _dashes, rc, rgba
from fastplot import _draw


def _old_draw_grid(self, r, axis, locs, kw):
    """_draw_grid as it was: every figure through r.lines."""
    from fastplot._cartesian import _across
    x0, y0, x1, y1 = self.box_f
    color = kw.get('color', kw.get('c', rc('grid.color')))
    lw = kw.get('linewidth', kw.get('lw', rc('grid.linewidth')))
    ls = kw.get('linestyle', kw.get('ls', rc('grid.linestyle')))
    alpha = kw.get('alpha', rc('grid.alpha'))
    if self._skew and axis.axis_name == 'x':
        ylo, yhi = self._lims['y']
        xs, ys = np.repeat(locs, 2), np.tile([ylo, yhi], locs.size)
        px, py = self._data_px(xs, ys)
        xy = np.c_[px, py]
    else:
        pos = self._tick_px(axis, locs)
        xy = (_across(pos, y0, y1, 0) if axis.axis_name == 'x' else
              _across(pos, x0, x1, 1)).reshape(-1, 2)
    d = _dashes(ls, lw)
    r.lines(xy, np.arange(0, xy.shape[0] + 1, 2), rgba(color, alpha), r.px(lw),
            self.box_clip, None if d is None else tuple(r.px(v) for v in d), self._snap,
            _draw.PROJECTING if d is None else _draw.BUTT)


def _plain(ls, i):
    fig = fastplot.figure(figsize=(6, 4), dpi=90)
    ax = fig.add_subplot(111)
    ax.plot(np.arange(10), np.sin(np.arange(10) + i))
    ax.grid(True, linestyle=ls, alpha=0.4, color='gray', linewidth=0.5 + 0.25 * (i % 2))
    out = fig.draw()
    fastplot.close(fig)
    return out


def _map(ls, i):
    pc = ccrs.PlateCarree()
    fig = fastplot.figure(figsize=(7, 4.2), dpi=80)
    ax = fig.add_axes([0.06, 0.08, 0.84, 0.84], projection=pc)
    ax.set_extent([-90, -60, 10, 28], crs=pc)
    ax.set_xticks(np.linspace(-90, -60, 8), crs=pc)
    ax.set_yticks(np.linspace(10, 28, 6), crs=pc)
    ax.tick_params(axis='both', labelsize=8, pad=3, width=0.5)
    ax.grid(True, linestyle=ls, alpha=0.4, color='gray', linewidth=0.5)
    ax.coastlines(resolution='110m', linewidth=0.7)
    out = fig.draw()
    fastplot.close(fig)
    return out


@pytest.mark.parametrize('make', [_plain, _map])
@pytest.mark.parametrize('ls', [':', '--', '-.', (0, (3, 1, 1, 1)), '-'])
def test_kept_grid_draws_as_new(monkeypatch, make, ls):
    _artists._GEOM.clear()
    got = [make(ls, i) for i in range(4)]               # kept from the second on, ink later
    kept = [k for k in _artists._GEOM.keys() if k[0] == 'axis-grid']
    assert bool(kept) == (ls != '-')                    # (solid: drawn as before)
    from fastplot import _cartesian
    cls = next(c for c in vars(_cartesian).values()
               if isinstance(c, type) and '_draw_grid' in vars(c))
    monkeypatch.setattr(cls, '_draw_grid', _old_draw_grid)
    want = [make(ls, i) for i in range(4)]
    for a, b in zip(got, want):
        assert a.tobytes() == b.tobytes()
