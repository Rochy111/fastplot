"""The deferred numba import: once the first kernel has loaded numba,
every fastplot module -- those imported later too -- sees numba's own
prange (a stand-in left behind only fails when its kernel compiles from
scratch, i.e. on a fresh install, with no cache to hide it)."""
import os
import subprocess
import sys


def test_prange_replaced_everywhere_after_load():
    code = '''
import numpy as np
import numba
import fastplot
from fastplot import _draw, _jit
_draw.snap(np.zeros((2, 2)), np.array([0, 2]), 1.0)         # the first kernel: numba loads
assert _jit.prange is numba.prange, _jit.prange
from fastplot import _png                                     # imported after the load
assert _png.prange is numba.prange, _png.prange
print('ok')
'''
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0 and out.stdout.strip().endswith('ok'), out.stderr[-2000:]


def test_cached_kernels_leave_numba_whole():
    """A kernel from the cache sets up nothing of numba's compiler, and
    the user's own numba code, BLAS and dicts and all, compiles after it."""
    code = '''
import sys
import numba, numpy as np
import fastplot
from fastplot import _jit
import fastplot.crs as ccrs
fig = fastplot.figure(figsize=(2, 2), dpi=50)
ax = fig.add_subplot(projection=ccrs.PlateCarree())
ax.contourf(np.arange(9.), np.arange(9.), np.arange(81.).reshape(9, 9), transform=ccrs.PlateCarree())
ax.set_title('t')
fig.savefig(__import__('io').BytesIO(), format='png')
if tuple(numba.__version__.split('.')[:2]) in _jit.CACHE_ONLY_NUMBA:
    assert 'scipy.linalg' not in sys.modules
@numba.njit
def mine(a):
    d = {1: np.dot(a, a) + np.linalg.norm(a)}
    return d[1], hash('x') != 0
assert mine(np.arange(3.0))[1]
assert 'scipy.linalg' in sys.modules
print('ok')
'''
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0 and out.stdout.strip().endswith('ok'), out.stderr[-2000:]


def test_numba_release_not_tried_is_left_alone():
    code = '''
import numpy as np
import fastplot
from fastplot import _draw, _jit
_jit.CACHE_ONLY_NUMBA.clear()
_draw.snap(np.zeros((2, 2)), np.array([0, 2]), 1.0)
disp = next(d for _, d in _jit._MADE if getattr(d, '_cache', None) is not None)
assert 'load_overload' not in vars(disp._cache)
from numba.core import caching
assert caching._CacheLocator.ensure_cache_path.__qualname__.startswith('_CacheLocator')
print('ok')
'''
    env = dict(os.environ, FASTPLOT_PRELOAD='0')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env,
                         timeout=600)
    assert out.returncode == 0 and out.stdout.strip().endswith('ok'), out.stderr[-2000:]


def test_kernel_not_in_the_cache_compiles(tmp_path):
    code = '''
import numpy as np
import fastplot
from fastplot import _draw
print(_draw.snap(np.zeros((2, 2)), np.array([0, 2]), 1.0) is not None, 'ok')
'''
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path), FASTPLOT_PRELOAD='0')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env,
                         timeout=600)
    assert out.returncode == 0 and out.stdout.strip().endswith('ok'), out.stderr[-2000:]
