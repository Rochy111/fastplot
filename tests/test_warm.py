"""warm(): its recipes, the manifest and how it is shared out, and what
starts it when installing did not."""
import os
import pickle
import subprocess
import sys
from types import SimpleNamespace

import pytest

from fastplot import _warm


def test_recipes_run():
    _warm.run()


def test_shares_balanced_and_complete():
    entries = [('m', 'k%d' % i, [()], float(t)) for i, t in enumerate([9, 7, 5, 4, 3, 3, 2, 1, 1, 0])]
    shares = _warm._shares(entries, 3)
    got = sorted(e[1] for s in shares for e in s)
    assert got == sorted(e[1] for e in entries)
    loads = [sum(max(e[3], 0.01) for e in s) for s in shares]
    assert max(loads) - min(loads) <= 9                  # LPT: within the biggest job


def test_manifest_matches_numba_and_names_real_kernels():
    if not os.path.exists(_warm.MANIFEST):
        return
    entries = _warm._manifest()
    d = pickle.load(open(_warm.MANIFEST, 'rb'))
    if entries is None:                                 # another numba: warm() uses the recipes
        assert 'numba' in d
        return
    import importlib
    missing = [(m, n) for m, n, _, _ in entries
               if not hasattr(importlib.import_module(m), n)]
    assert not missing, 'stale manifest (tools/make_warm_manifest.py): %s' % missing[:5]


def test_warm_in_process():
    assert _warm.warm(processes=1) > 0
    assert not _warm.stale()


def _kernel_with(index):
    """A kernel whose cache index numba reads as `index`."""
    return lambda: SimpleNamespace(_cache=SimpleNamespace(
        _cache_file=SimpleNamespace(_load_index=lambda: index)))


def _done_here(tmp_path, monkeypatch):
    """warm()'s file in tmp_path, and keep_up() on, starting nothing."""
    done = str(tmp_path / 'warmed.x')
    started = []
    monkeypatch.delenv('FASTPLOT_AUTOWARM')
    monkeypatch.setattr(_warm, '_done_file', lambda: done)
    monkeypatch.setattr(_warm, '_start', started.append)
    monkeypatch.setattr(_warm, '_kernel', _kernel_with({'a signature': 'its data'}))
    return done, started


def test_stale_until_this_source_was_warmed(tmp_path, monkeypatch):
    done, _ = _done_here(tmp_path, monkeypatch)
    assert _warm.stale()                                # never warmed
    with open(done, 'w') as f:
        f.write('another version of the sources')
    assert _warm.stale()
    with open(done, 'w') as f:
        f.write(_warm._sources())
    assert not _warm.stale()


def test_stale_when_numba_disowns_the_cache(tmp_path, monkeypatch):
    """Before numba 0.68 pip's copying of the sources makes a built cache
    worthless, though warm()'s file came along with it."""
    done, _ = _done_here(tmp_path, monkeypatch)
    with open(done, 'w') as f:
        f.write(_warm._sources())
    monkeypatch.setattr(_warm, '_kernel', _kernel_with({}))
    assert _warm.stale()


def test_the_kernel_asked_about_is_compiled_by_the_recipes():
    """Or stale() would be true for good, and every program would start a warm()."""
    from fastplot import _jit
    _jit.load()
    _warm.run('plain_lines')
    assert _warm._kernel()._cache._cache_file._load_index()


def test_keep_up_starts_one_warm(tmp_path, monkeypatch):
    done, started = _done_here(tmp_path, monkeypatch)
    _warm.keep_up()
    assert started == [done + '.lock']


def test_keep_up_nothing_to_do(tmp_path, monkeypatch):
    done, started = _done_here(tmp_path, monkeypatch)
    with open(done, 'w') as f:
        f.write(_warm._sources())
    _warm.keep_up()
    assert not started


def test_keep_up_turned_off(tmp_path, monkeypatch):
    _, started = _done_here(tmp_path, monkeypatch)
    monkeypatch.setenv('FASTPLOT_AUTOWARM', '0')
    _warm.keep_up()
    assert not started


def test_keep_up_leaves_a_warm_running(tmp_path, monkeypatch):
    done, started = _done_here(tmp_path, monkeypatch)
    lock = done + '.lock'
    open(lock, 'w').close()
    _warm.keep_up()
    assert not started
    os.utime(lock, (0, 0))                              # one that died long ago
    _warm.keep_up()
    assert started == [lock] and not os.path.exists(lock)


def test_keep_up_never_raises(tmp_path, monkeypatch):
    _done_here(tmp_path, monkeypatch)

    def fail(lock):
        raise OSError('no')
    monkeypatch.setattr(_warm, '_start', fail)
    _warm.keep_up()


def test_background_runs_warm_once(tmp_path, monkeypatch):
    lock = str(tmp_path / 'warmed.x.lock')
    ran = []
    monkeypatch.setattr(_warm, 'warm', lambda: ran.append(os.path.exists(lock)))
    monkeypatch.setattr(os, 'nice', lambda n: 0, raising=False)
    _warm.background(lock)
    assert ran == [True] and not os.path.exists(lock)   # held meanwhile, then let go
    open(lock, 'w').close()
    _warm.background(lock)
    assert ran == [True]                                # held by another: left to it


def test_background_failure_keeps_the_lock(tmp_path, monkeypatch):
    lock = str(tmp_path / 'warmed.x.lock')

    def fail():
        raise RuntimeError('no')
    monkeypatch.setattr(_warm, 'warm', fail)
    monkeypatch.setattr(os, 'nice', lambda n: 0, raising=False)
    with pytest.raises(RuntimeError):
        _warm.background(lock)
    assert os.path.exists(lock)                         # so not every program tries again


def test_preload_loads_only_this_machines_kernels():
    """A cache carried from another CPU is not compiled on the preload thread."""
    code = ('import fastplot\n'
            'from fastplot import _jit, _warm\n'
            'calls = []\n'
            '_jit.load()\n'
            'class D:\n'
            '    overloads = {}\n'
            '    def compile(self, sig):\n'
            '        calls.append(sig)\n'
            '    class _cache:\n'
            '        class _cache_file:\n'
            '            @staticmethod\n'
            '            def _load_index():\n'
            '                return {("elsewhere", ("a", "b", "c"), ()): 0,\n'
            '                        ("here", _warm._machine(), ()): 0}\n'
            '_jit._MADE[:] = [("m", D())]\n'
            '_jit._preload_run()\n'
            'print(calls)\n')
    env = dict(os.environ, FASTPLOT_PRELOAD='0')
    out = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
    assert out.stdout.strip() == "['here']", out.stderr
