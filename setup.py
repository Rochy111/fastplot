"""The wheel carries its kernels compiled: pip builds a wheel from source on
the machine installing it, so `pip install` can compile for that very CPU
(~30 s more) and the first figure then finds them all in numba's cache. The
cache's stamps are hashes of the sources, so it stays good where pip puts
it; a wheel carried to another machine just misses (see _warm.keep_up).

Never the reason an install fails: without compiled kernels fastplot still
works, and compiles them itself. FASTPLOT_BUILD_KERNELS=0 skips this, and
an editable install has no build to keep them in. The metadata is all in
pyproject.toml.
"""
import os
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_py import build_py

KEEP = ('.nbi', '.nbc')         # numba's cache


def installing_into(lib, env):
    """env, with pip's build environment taken away if what is installed
    into has fastplot's dependencies: its numba is the one that will run
    the kernels, and numba's cache is good for its own version only -- an
    upgrade of fastplot beside an older numba would otherwise compile with
    the newest there is, for nothing. pip's isolation is a PYTHONPATH on
    this same interpreter."""
    target = {k: v for k, v in env.items() if k not in ('PYTHONPATH', 'PYTHONNOUSERSITE')}
    has = subprocess.run([sys.executable, '-c', 'import numba, numpy, PIL, freetype'], cwd=lib,
                         env=target, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return target if has.returncode == 0 else env


def compile_kernels(lib):
    """Every kernel of the package in `lib` compiled into its __pycache__."""
    env = dict(os.environ, FASTPLOT_PRELOAD='0', FASTPLOT_DEVICE='cpu',
               PYTHONDONTWRITEBYTECODE='1')
    env.pop('NUMBA_CACHE_DIR', None)                    # (the wheel would be without them)
    try:
        env = installing_into(lib, env)
        subprocess.run([sys.executable, '-c', 'from fastplot import _warm; _warm.warm(verbose=True)'],
                       cwd=lib, env=env, check=True, timeout=1800)
    except (OSError, subprocess.SubprocessError) as e:
        print('fastplot: kernels not compiled while building (%s): it will do so itself' % e)
    cache = os.path.join(lib, 'fastplot', '__pycache__')
    if os.path.isdir(cache):
        for f in os.listdir(cache):                     # the import bundle names this folder
            if not f.endswith(KEEP) and not f.startswith('warmed.'):
                os.remove(os.path.join(cache, f))


class BuildPy(build_py):
    def run(self):
        super().run()
        if not self.editable_mode and os.environ.get('FASTPLOT_BUILD_KERNELS', '1') != '0':
            compile_kernels(self.build_lib)


setup(cmdclass={'build_py': BuildPy})
