"""fastplot.frames: figures drawn on threads at once, as one at a time."""
import io

import numpy as np
import pytest

import fastplot as fp
import fastplot.crs as ccrs


def _frame(k):
    lon = np.linspace(0, 359, 90)
    lat = np.linspace(-90, 90, 46)
    z = np.cos(np.radians(lat))[:, None] * np.sin(np.radians(lon + 10 * k))[None, :]
    fig = fp.figure(figsize=(4, 2.5), dpi=80)
    ax = fig.add_subplot(projection=ccrs.Robinson())
    cf = ax.contourf(lon, lat, z, levels=10, transform=ccrs.PlateCarree())
    ax.contour(lon, lat, z, levels=5, colors='k', transform=ccrs.PlateCarree())
    ax.coastlines()
    fig.colorbar(cf, ax=ax)
    ax.set_title('frame %d' % k)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', device='cpu')
    return buf.getvalue()


def test_frames_equal_one_at_a_time():
    one = [_frame(k) for k in range(6)]
    assert fp.frames(_frame, range(6), workers=3, threads=2) == one
    assert fp.frames(_frame, range(6)) == one


def test_frames_learn_a_split_and_draw_the_same(monkeypatch):
    from fastplot import _batch
    monkeypatch.setattr(_batch, '_LEARNT', {})
    monkeypatch.setattr(_batch, '_splits', lambda n: [(2, 2), (3, 1), (4, 1)])
    n = 20                                              # (enough to try all three: 2 + 3 + 4 + 2)
    one = [_frame(k % 4) for k in range(n)]
    assert fp.frames(lambda k: _frame(k % 4), range(n)) == one
    split, = _batch._LEARNT.values()
    assert split in [(2, 2), (3, 1), (4, 1)]
    assert fp.frames(_frame, range(3)) == one[:3]       # (too few to try them: the first)


def test_frames_first_error_raised():
    def f(k):
        if k in (3, 5):
            raise ValueError(k)
        return k
    with pytest.raises(ValueError, match='3'):
        fp.frames(f, range(8), workers=1)
    assert fp.frames(f, []) == []
    assert fp.frames(lambda k: k * 2, range(5), workers=2) == [0, 2, 4, 6, 8]
