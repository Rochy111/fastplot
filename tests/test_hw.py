"""_hw: the machine looked at, the settings drawn from it, tune()'s file;
_jit.park."""
import json
import os
import sys

import numpy as np
import pytest

from fastplot import _hw, _jit


@pytest.fixture
def fresh(monkeypatch, tmp_path):
    """_hw's findings forgotten (and restored after), its cache in tmp_path,
    no FASTPLOT_THREADS / FASTPLOT_GPU_WORK."""
    saved = dict(_hw._found)
    for k in ('draw_threads', '_tune_file'):
        _hw._found.pop(k, None)
    monkeypatch.setenv('FASTPLOT_CACHE_DIR', str(tmp_path))
    monkeypatch.delenv('FASTPLOT_THREADS', raising=False)
    monkeypatch.delenv('FASTPLOT_GPU_WORK', raising=False)
    yield tmp_path
    _hw._found.clear()
    _hw._found.update(saved)


def test_cpus_and_topology_consistent():
    n = _hw.cpus()
    assert 1 <= n <= (os.cpu_count() or 1)
    t = _hw.topology()
    assert 1 <= t['perf_cores'] <= t['cores'] <= t['logical']
    assert t['perf_cores'] <= t['perf_logical'] <= t['logical']
    assert t['perf_cores'] <= n
    assert t['hybrid'] == (t['perf_cores'] != t['cores'])


def test_cpu_list():
    assert _hw._cpu_list('0-3,8,10-11') == {0, 1, 2, 3, 8, 10, 11}
    assert _hw._cpu_list('') == set()


@pytest.mark.parametrize('files,want', [
    ({'/sys/fs/cgroup/cpu.max': '250000 100000'}, 2.5),
    ({'/sys/fs/cgroup/cpu.max': 'max 100000'}, None),
    ({'/sys/fs/cgroup/cpu/cpu.cfs_quota_us': '400000',
      '/sys/fs/cgroup/cpu/cpu.cfs_period_us': '100000'}, 4.0),
    ({'/sys/fs/cgroup/cpu/cpu.cfs_quota_us': '-1',
      '/sys/fs/cgroup/cpu/cpu.cfs_period_us': '100000'}, None),
    ({}, None),
])
def test_container_quota(monkeypatch, files, want):
    monkeypatch.setattr(_hw, '_read', files.get)
    assert _hw._quota() == want


def test_draw_threads_rule(fresh, monkeypatch):
    t = _hw.topology()
    want = t['perf_cores']
    if want < _hw.MIN_THREADS:
        want = min(_hw.MIN_THREADS, t['perf_logical'])
    want = max(1, min(want, _hw.MAX_THREADS, _hw.cpus()))
    assert _hw.draw_threads() == want
    _hw._found.pop('draw_threads')
    monkeypatch.setenv('FASTPLOT_THREADS', '3')
    assert _hw.draw_threads() == min(3, _hw.cpus())


def test_tune_file_is_this_machines(fresh):
    _hw.save_tuned(cpu={'threads': 1})
    assert _hw.tuned() == {'threads': 1}
    assert _hw.draw_threads() == 1
    path = os.path.join(str(fresh), 'tune.json')
    d = json.load(open(path))
    d['cpu_key'] = 'another CPU|64|' + d['cpu_key'].split('|', 2)[2]
    json.dump(d, open(path, 'w'))
    _hw._found.pop('_tune_file', None)
    _hw._found.pop('draw_threads', None)
    assert _hw.tuned() == {}                            # another machine's: not read
    assert _hw.draw_threads() != 1 or _hw.topology()['perf_cores'] == 1


def test_tune_file_bad(fresh):
    open(os.path.join(str(fresh), 'tune.json'), 'w').write('{not json')
    assert _hw.tuned() == {}


def test_gpu_work_env_and_tuned(fresh, monkeypatch):
    monkeypatch.setenv('FASTPLOT_GPU_WORK', '1234')
    assert _hw.gpu_work() == 1234
    monkeypatch.delenv('FASTPLOT_GPU_WORK')
    monkeypatch.setitem(_hw._found, 'gpu_props',
                        dict(name='Card X', sms=35, memory=1, integrated=False, cc='8.6'))
    w = _hw.gpu_work()                                  # half the reference card's SMs: twice
    assert w == int(_hw.AUTO_GPU_WORK * _hw.draw_threads() / _hw.REF_THREADS * 2)
    _hw.save_tuned(gpu={'key': _hw.machine_key('Card X'), 'work': 777})
    assert _hw.gpu_work() == 777
    monkeypatch.setitem(_hw._found, 'gpu_props',
                        dict(name='Card Y', sms=35, memory=1, integrated=True, cc='8.6'))
    assert _hw.gpu_work() == int(w * _hw.AUTO_GPU_IGP)   # another card: not its tuning


def test_info_keys():
    i = _hw.info()
    for k in ('cpu', 'cpus_allowed', 'topology', 'draw_threads', 'threads_from'):
        assert k in i


def test_fast_cores_asked_once_and_can_be_left(fresh, monkeypatch):
    _hw._found.pop('fast_cores', None)
    monkeypatch.setenv('FASTPLOT_ECOQOS', '1')
    assert _hw.fast_cores() is None
    monkeypatch.delenv('FASTPLOT_ECOQOS')
    assert _hw.fast_cores() is None                     # once a process, as everything here
    _hw._found.pop('fast_cores')
    want = True if sys.platform == 'win32' and _hw.topology()['hybrid'] else None
    assert _hw.fast_cores() is want


def test_park_keeps_the_thread_count():
    _jit.load()
    nb = _jit._numba
    from fastplot import _draw
    cv = np.zeros((64, 64, 4), np.uint8)
    _draw.fill_polygons(cv, np.array([[0, 0], [60, 0], [60, 60.]]), np.array([0, 3]),
                        np.array([255, 0, 0, 255], np.uint8))
    before = nb.get_num_threads()
    _jit.park()
    assert nb.get_num_threads() == before
    with _jit.threads(2, park=True):
        assert nb.get_num_threads() == min(2, before)
    assert nb.get_num_threads() == before


def test_team_counts():
    _jit.load()
    with _jit.threads(3):
        assert _jit.team() == min(3, _jit._numba.config.NUMBA_NUM_THREADS)


@pytest.fixture
def auto(fresh, monkeypatch):
    """_device as a new process under 'auto' finds it, a card 'Card X' that
    can draw from W (returned, with the looks made) found by a look that
    takes 0.2 s and imports nothing."""
    from fastplot import _device
    _hw._found.pop('gpu_seen', None)
    monkeypatch.setitem(_hw._found, 'gpu_props', dict(name='Card X', sms=70, memory=1, integrated=False, cc='8.6'))
    for name, value in (('_device', 'auto'), ('_usable', None), ('_name', ''), ('_rent', [0.0]),
                        ('_look', [0.0]), ('_first', [True]), ('_rates', {'cpu': [], 'gpu': []})):
        monkeypatch.setattr(_device, name, value)
    looks = []

    def look():
        if _device._usable is None:
            looks.append(1)
            _device._usable, _device._name, _device._look[0] = True, 'Card X', 0.2
            _device._looked()
        return _device._usable
    monkeypatch.setattr(_device, 'gpu_usable', look)
    return _device, _hw.gpu_work(env=False), looks


def test_auto_rents_before_it_looks(auto):
    d, W, looks = auto
    assert d.pick(2 * W) == 'cpu' and d.pick(100) == 'cpu' and not looks     # never looked here: renting
    d.drew('cpu', 2 * W, 0.3)
    d.drew('cpu', 100, 5.0)                             # (too small to count)
    assert d.pick(2 * W) == 'cpu' and not looks
    d.drew('cpu', 2 * W, 0.3)                           # 0.6 s paid: more than the guess
    assert d.pick(2 * W) == 'gpu' and looks == [1]
    assert _hw.gpu_seen() == (True, W, 0.4, 'Card X')   # twice the look, until a figure is timed
    assert d.pick(W // 2) == 'cpu'
    d.drew('gpu', 2 * W, 0.3)
    assert _hw.gpu_seen()[2] == pytest.approx(0.5)      # the look and its first figure


def test_auto_believes_the_last_process(auto, monkeypatch):
    d, W, looks = auto
    _hw.see_gpu(True, W, 1.0, 'Card X')
    _hw._found.pop('gpu_seen')                          # (read back from the file)
    assert d.pick(W // 2) == 'cpu'                      # too small for the card
    assert d.pick(2 * W) == 'cpu'                       # big enough, but 1 s not yet paid
    assert not looks
    _hw.see_gpu(False)
    assert d.pick(10 * W) == 'cpu' and not looks        # no card
    with open(_hw._gpu_seen_path(), encoding='utf-8') as f:
        line = f.read()
    with open(_hw._gpu_seen_path(), 'w', encoding='utf-8') as f:
        f.write(line.replace('|', '|another ', 1))      # another machine's (or cupy's, fastplot's)
    _hw._found.pop('gpu_seen')
    assert _hw.gpu_seen() is None


@pytest.mark.parametrize('gpu_seconds, kept', [(0.002, True), (0.0095, False)])
def test_auto_judges_the_card(auto, gpu_seconds, kept):
    d, W, looks = auto
    for _ in range(3):
        d.drew('cpu', 2 * W, 0.01)
    d.drew('cpu', 2 * W, 0.5)                           # (the rent paid)
    assert d.pick(2 * W) == 'gpu'
    d.drew('gpu', 2 * W, 0.4)                           # its first: starting it
    for _ in range(d.GPU_TRIES):
        d.drew('gpu', 2 * W, gpu_seconds)
    assert _hw.gpu_seen()[1] == (W if kept else 4 * W)  # not 10% faster: from past them
    assert d.pick(2 * W) == ('gpu' if kept else 'cpu')
    assert d.pick(5 * W) == 'gpu'


def test_tune_runs(fresh):
    import fastplot
    r = fastplot.tune(save=True, rounds=1, gpu=False)
    assert r['threads'] in [int(k) for k in r['threads_measured']]
    assert os.path.exists(r['saved_to'])
    assert _hw.draw_threads() == r['threads']
    h = fastplot.hardware()
    assert h['threads_from'] == 'tune()'
