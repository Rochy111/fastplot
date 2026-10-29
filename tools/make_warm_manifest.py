"""fastplot/data/warm_manifest.pkl: every kernel warm()'s recipes compile,
with its argument types and how long it took, so that warm() can share
the compiling out evenly among processes.

    python tools/make_warm_manifest.py

Run it after changing kernels (a stale manifest is harmless: warm()
compiles from it what it still can, and the recipes do the rest). It
compiles everything in a fresh, temporary numba cache, one process.
"""
import os
import pickle
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'fastplot', 'data', 'warm_manifest.pkl')

CHILD = r'''
import collections, pickle, sys
import numba
from numba.core import event
from fastplot import _jit, _warm
with event.install_recorder('numba:compile') as rec:
    _warm.run()
start, spent = {}, collections.Counter()
for ts, ev in rec.buffer:
    d = ev.data.get('dispatcher') if ev.data else None
    if d is None:
        continue
    k = (id(d), str(ev.data.get('args')))
    if ev.is_start:
        start[k] = ts
    elif k in start:
        spent[id(d)] += ts - start.pop(k)
entries = []
for mod, disp in _jit._MADE:
    sigs = list(disp.signatures)
    if sigs:
        entries.append((mod, disp.py_func.__name__, sigs, float(spent.get(id(disp), 0.0))))
pickle.dump({'numba': numba.__version__, 'entries': entries}, open(sys.argv[1], 'wb'))
'''


def main():
    with tempfile.TemporaryDirectory() as cache:
        env = dict(os.environ, NUMBA_CACHE_DIR=cache, FASTPLOT_PRELOAD='0',
                   FASTPLOT_DEVICE='cpu')
        env.pop('PYTHONDONTWRITEBYTECODE', None)
        subprocess.run([sys.executable, '-c', CHILD, OUT], env=env, cwd=ROOT, check=True)
    d = pickle.load(open(OUT, 'rb'))
    n = sum(len(e[2]) for e in d['entries'])
    print('%s: %d kernels, %d signatures, %.0f s of compiling (numba %s)'
          % (OUT, len(d['entries']), n, sum(e[3] for e in d['entries']), d['numba']))


if __name__ == '__main__':
    main()
