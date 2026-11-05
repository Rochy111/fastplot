"""fastplot._budget: one memory budget for what is kept across figures --
least recently used first across caches, weighted; kept copies shared by
their bits; the budget worked out per machine and per pool worker; and
the same pictures whatever the budget."""
import multiprocessing

import numpy as np
import pytest

import fastplot
import fastplot.crs as ccrs
from fastplot import _budget


@pytest.fixture
def budget(monkeypatch):
    """Set the budget (bytes) for a test; the caches emptied before and after."""
    _budget.clear()

    def set_(b):
        monkeypatch.setattr(_budget, '_BUDGET', [int(b)])
    yield set_
    _budget.clear()


@pytest.fixture
def caches():
    made = []

    def make(*a, **k):
        c = _budget.Cache(*a, **k)
        made.append(c)
        return c
    yield make
    for c in made:
        _budget._CACHES.remove(c)


def test_lru_across_caches(budget, caches):
    budget(1000)
    a, b = caches('a'), caches('b')
    a.put(1, 'a1', 240)
    b.put(1, 'b1', 240)
    a.put(2, 'a2', 240)
    a.put(3, 'a3', 240)
    assert a.get(1) == 'a1'                             # a1 used: b1 is now the oldest
    b.put(2, 'b2', 240)                                 # 1200 > 1000: b1 goes
    assert 1 not in b and all(k in a for k in (1, 2, 3)) and 2 in b
    assert _budget.held() <= 1000


def test_weights_keep_a_maps_entries_longer(budget, caches):
    budget(1000)
    frame, grid = caches('frame', weight=1), caches('map', weight=16)
    grid.put('table', 't', 240)                         # older, but a map's
    for k in range(3):
        frame.put(k, k, 240)
    frame.put(3, 3, 240)                                # over: the oldest frame entry goes
    assert 'table' in grid and 0 not in frame


def test_too_big_not_kept_and_just_put_never_dropped(budget, caches):
    budget(1000)
    a = caches('a')
    assert a.put('big', 1, 251) is False and 'big' not in a
    assert a.put('fits', 1, 250) is True
    a.put('fits2', 2, 250)
    a.put('fits3', 3, 250)
    a.put('fits4', 4, 250)
    a.put('fits5', 5, 250)                              # its own entry is never the one dropped
    assert 'fits5' in a and _budget.held() <= 1000


def test_resize_counts_a_filled_entry(budget, caches):
    budget(1000)
    a = caches('a')
    a.put('x', [None], 0)
    a.put('y', [None], 0)
    assert _budget.held() == 0
    a.resize('x', 200)
    a.resize('y', 200)
    assert a.nbytes == 400


def test_keep_all_never_dropped_nor_counted(budget, caches):
    budget(1000)
    feats, a = caches('feats', keep_all=True), caches('a')
    feats.put('coast', 'c', 5000)                       # bigger than the budget: kept anyway
    a.put(1, 1, 200)
    assert 'coast' in feats and 1 in a and _budget.held() == 200


def test_max_n(budget, caches):
    budget(10 ** 9)
    a = caches('a', 3)
    for k in range(5):
        a.put(k, k, 10)
    assert a.keys() == [2, 3, 4] and a.nbytes == 30


def test_kept_copy_shared_by_bits_only():
    x = np.linspace(0, 1, 5000).reshape(50, 100)
    c1 = _budget.kept_copy(x)
    c2 = _budget.kept_copy(x.copy())
    assert c1 is c2 and not np.shares_memory(c1, x)
    y = x.copy()
    y[49, 99] = np.nextafter(y[49, 99], 0)              # one bit: not alike
    assert _budget.kept_copy(y) is not c1
    z = np.zeros(4096)
    zn = z.copy()
    zn[7] = -0.0                                        # -0.0 is not 0.0 here
    assert _budget.kept_copy(z) is not _budget.kept_copy(zn)
    f32 = _budget.kept_copy(x.astype(np.float32))
    assert f32.dtype == np.float32 and f32 is not c1


def test_kept_copies_counted_while_held():
    before = _budget._COPY_BYTES[0]
    c = _budget.kept_copy(np.arange(1 << 16, dtype=np.float64) + 0.5)
    assert _budget._COPY_BYTES[0] == before + c.nbytes
    del c
    import gc
    gc.collect()
    assert _budget._COPY_BYTES[0] == before


def test_budget_worked_out(monkeypatch):
    from fastplot import _hw
    monkeypatch.delenv('FASTPLOT_CACHE_MB', raising=False)
    monkeypatch.setattr(_hw, 'memory', lambda: 64 << 30)
    monkeypatch.setattr(_budget, '_sharers', lambda: 1)
    assert _budget._work_out() == int((64 << 30) * _budget.CACHE_SHARE)
    monkeypatch.setattr(_budget, '_sharers', lambda: 8)
    assert _budget._work_out() == int((64 << 30) * _budget.CACHE_SHARE / 8)
    monkeypatch.setattr(_hw, 'memory', lambda: 1 << 30)         # small machine: the floor
    assert _budget._work_out() == _budget.CACHE_MIN
    monkeypatch.setattr(_hw, 'memory', lambda: None)
    assert _budget._work_out() == _budget.CACHE_MIN
    monkeypatch.setenv('FASTPLOT_CACHE_MB', '12.5')
    assert _budget._work_out() == int(12.5 * (1 << 20))
    monkeypatch.setenv('FASTPLOT_CACHE_MB', '0')
    assert _budget._work_out() == 0


def test_a_pool_worker_shares_the_machine(monkeypatch):
    from fastplot import _hw

    class Parent:
        pass
    monkeypatch.setattr(multiprocessing, 'parent_process', lambda: Parent())
    monkeypatch.setattr(_hw, 'cpus', lambda: 24)
    monkeypatch.setattr(_hw, 'draw_threads', lambda: 3)
    assert _budget._sharers() == 8
    monkeypatch.setattr(multiprocessing, 'parent_process', lambda: None)
    assert _budget._sharers() == 1


def test_set_budget_and_info(monkeypatch):
    keep = _budget._BUDGET[0]
    try:
        fastplot.set_cache_budget(20)
        info = fastplot.cache_info()
        assert info['budget_mb'] == 20 and 'drawings' in info['caches']
        assert info['held_mb'] <= 20
    finally:
        _budget._BUDGET[0] = keep


def _frame(i):
    rng = np.random.default_rng(i // 2)
    lon, lat = np.meshgrid(np.linspace(-110, -80, 120), np.linspace(25, 45, 90))
    lat = lat + 0.01 * (lon + 95)                       # 2-D coordinates
    pc = ccrs.PlateCarree()
    fig = fastplot.figure(figsize=(5, 3.5), dpi=80)
    ax = fig.add_axes([0.05, 0.05, 0.8, 0.9], projection=pc)
    ax.set_extent([-110, -80, 25, 45], crs=pc)
    z = np.cumsum(rng.normal(size=lon.shape), 1)
    p = np.cumsum(rng.normal(size=lon.shape), 0)
    if i % 2:
        ax.contourf(lon, lat, z, levels=20, transform=pc)
    else:
        ax.pcolormesh(lon, lat, z, shading='nearest', transform=pc)
    cs = ax.contour(lon, lat, p, levels=8, colors='k', transform=pc)
    ax.clabel(cs, fmt='%d', fontsize=7)
    ax.barbs(lon[::15, ::15], lat[::15, ::15], z[::15, ::15], p[::15, ::15], length=5,
             transform=pc)
    ax.coastlines()
    out = fig.draw()
    fastplot.close(fig)
    return out


def test_idle_entries_go_after_their_frames(budget, caches):
    """A cache made with idle drops what no frame used for that many
    frames, whatever the budget; what frames use stays."""
    budget(1 << 30)
    c = caches('test idle', keep_all=True, idle=2)
    try:
        c.put('read', np.zeros(10), 80)
        c.put('left', np.zeros(10), 80)
        for _ in range(3):
            c.get('read')
            _budget.frame_done()
        assert 'read' in c and 'left' not in c
        for _ in range(3):
            _budget.frame_done()
        assert 'read' not in c and c.nbytes == 0
    finally:
        _budget._IDLE.remove(c)


def test_whole_features_go_once_the_map_is_kept(budget):
    """A map's Natural Earth features are read to make its geometry; frames
    of it after that draw the kept geometry, and the features go."""
    from fastplot import _features
    budget(1 << 30)
    _features._CACHE.clear()
    for i in range(_features.IDLE_FRAMES + 3):
        _frame(i)
        if i == 0:
            assert len(_features._CACHE)
    assert not len(_features._CACHE)


@pytest.mark.parametrize('mb', [0, 0.25, 2])
def test_same_pictures_whatever_the_budget(budget, mb):
    budget(1 << 30)
    want = [_frame(i) for i in range(8)]
    budget(mb * (1 << 20))
    got = [_frame(i) for i in range(8)]
    for a, b in zip(want, got):
        assert a.tobytes() == b.tobytes()
