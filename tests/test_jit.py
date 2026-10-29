"""The deferred numba import: once the first kernel has loaded numba,
every fastplot module -- those imported later too -- sees numba's own
prange (a stand-in left behind only fails when its kernel compiles from
scratch, i.e. on a fresh install, with no cache to hide it)."""
import subprocess
import sys


def test_prange_replaced_everywhere_after_load():
    code = '''
import numpy as np
import numba
import fastplot
from fastplot import _draw, _jit
_draw.segments(np.zeros((2, 2)), np.array([0, 2]))          # the first kernel: numba loads
assert _jit.prange is numba.prange, _jit.prange
from fastplot import _png                                     # imported after the load
assert _png.prange is numba.prange, _png.prange
print('ok')
'''
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0 and out.stdout.strip().endswith('ok'), out.stderr[-2000:]
