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


def _learns():
    """frames() tries splits here: not on numba's workqueue layer (macOS on
    Apple silicon), where frames are drawn one at a time."""
    from fastplot import _jit
    if _jit.layer() == 'workqueue':
        pytest.skip('frames are drawn one at a time on the workqueue layer')


def test_frames_learn_a_split_and_draw_the_same(monkeypatch):
    _learns()
    from fastplot import _batch
    monkeypatch.setattr(_batch, '_LEARNT', {})
    monkeypatch.setattr(_batch, '_splits', lambda n: [(2, 2), (3, 1), (4, 1)])
    n = 20                                              # (enough to try all three: 2 + 3 + 4 + 2)
    one = [_frame(k % 4) for k in range(n)]
    assert fp.frames(lambda k: _frame(k % 4), range(n)) == one
    split, = _batch._LEARNT.values()
    assert split in [(2, 2), (3, 1), (4, 1)]
    assert fp.frames(_frame, range(3)) == one[:3]       # (too few to try them: the first)


def _square(k):
    import time
    time.sleep(0.003)                                   # (some work: rates over thread start-up)
    return k * k


def _new_process(monkeypatch, _batch):
    """This process's learning forgotten, frames.json read again (as a new process would)."""
    monkeypatch.setattr(_batch, '_LEARNT', {})
    monkeypatch.setattr(_batch, '_store', {})


def test_frames_split_kept_for_the_next_process(monkeypatch):
    _learns()
    from fastplot import _batch, _hw
    monkeypatch.setattr(_batch, '_LEARNT', {})
    monkeypatch.setattr(_batch, '_splits', lambda n: [(2, 1), (3, 1)])
    assert fp.frames(_square, range(20)) == [k * k for k in range(20)]
    split = _batch._LEARNT[_square.__code__]
    _new_process(monkeypatch, _batch)
    monkeypatch.setattr(_batch, '_splits', lambda n: pytest.fail('learnt again'))
    assert fp.frames(_square, range(20)) == [k * k for k in range(20)]
    assert _batch._LEARNT[_square.__code__] == split      # (the stored split, used and kept)
    _new_process(monkeypatch, _batch)                      # another machine: its own learning
    monkeypatch.setattr(_hw, 'machine_key', lambda gpu_name='': 'another machine')
    monkeypatch.setattr(_batch, '_splits', lambda n: [(2, 1), (4, 1)])
    fp.frames(_square, range(20))
    assert _batch._LEARNT[_square.__code__] in [(2, 1), (4, 1)]


def test_frames_stored_split_forgotten_when_slower(monkeypatch):
    _learns()
    from fastplot import _batch
    ckey = _batch._code_key(_square)
    _batch._keep(ckey, [2, 1, 1e12])                    # (a rate no batch reaches)
    _new_process(monkeypatch, _batch)
    fp.frames(_square, range(20))
    _new_process(monkeypatch, _batch)
    assert ckey not in _batch._stored()


def test_frames_store_keeps_the_latest(monkeypatch):
    from fastplot import _batch
    for k in range(_batch.STORE_MOST + 10):
        _batch._keep('fn%d' % k, [2, 1, 100.0])
    _new_process(monkeypatch, _batch)
    kept = _batch._stored()
    assert len(kept) == _batch.STORE_MOST and 'fn%d' % (_batch.STORE_MOST + 9) in kept and 'fn0' not in kept


def test_frames_first_error_raised():
    def f(k):
        if k in (3, 5):
            raise ValueError(k)
        return k
    with pytest.raises(ValueError, match='3'):
        fp.frames(f, range(8), workers=1)
    assert fp.frames(f, []) == []
    assert fp.frames(lambda k: k * 2, range(5), workers=2) == [0, 2, 4, 6, 8]
