"""warm(): its recipes, the manifest and how it is shared out."""
import os
import pickle

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
