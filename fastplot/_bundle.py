"""Imports without most of their trips to the file system: the compiled
modules of fastplot and what it imports, in one file. `import numba` is
~320 ms, and much of that is finding and opening ~250 .pyc files (a few
stats and an open each, ~1 ms on Windows, more with a virus scanner); read
from one file it is ~220 ms, and a fresh process's first figure ~230 ms
sooner, the numba modules its kernels bring in with them counted.

The bundle is a .pyc file for many modules: each entry is what the
module's own .pyc holds, with the size and time of the source it came
from, checked as Python checks a .pyc. A module not in it, or whose source
has changed, imports the usual way, as does every module once fastplot is
gone. Loaders are Python's own SourceFileLoader but for where the code
comes from, so __file__, resources and tracebacks are what they always
were. A package another copy of which comes first on sys.path is not used
(the standard library is left out for the same reason: a script's own
random.py wins, as it should).

install() runs first thing in fastplot/__init__.py. write() makes the
bundle: by warm(), and at the exit of a process that missed it, once per
install or upgrade of numba (~0.8 s). It sits in fastplot's __pycache__,
trusted as the .pyc files beside it are. FASTPLOT_BUNDLE=0 turns it off.
"""
import importlib.machinery
import importlib.util
import marshal
import os
import sys

PACKAGES = ('numba', 'llvmlite', 'freetype', 'PIL', 'fastplot')
_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '__pycache__', 'imports.%s.opt%d.bin'
                     % (sys.implementation.cache_tag, sys.flags.optimize))
_TABLE = {}                     # module name: (source path, its mtime, its size, is it a package, code)
_STATE = {'wanted': False}      # numba's modules are not in the bundle as they are now


class _Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        return marshal.loads(_TABLE[fullname][4])


class _Finder:
    """On sys.meta_path: the bundle's modules whose sources have not changed."""

    @staticmethod
    def find_spec(name, path=None, target=None):
        e = _TABLE.get(name)
        if e is None:
            return None
        try:
            st = os.stat(e[0])
        except OSError:
            return None
        if int(st.st_mtime) != e[1] or st.st_size != e[2]:
            return None
        spec = importlib.machinery.ModuleSpec(name, _Loader(name, e[0]), origin=e[0], is_package=e[3])
        spec.has_location = True
        if e[3]:
            spec.submodule_search_locations = [os.path.dirname(e[0])]
        return spec


def _origin(pkg):
    """The __init__.py a package imports from, by the path search."""
    mod = sys.modules.get(pkg)
    if mod is not None:
        return getattr(mod, '__file__', None)
    spec = importlib.machinery.PathFinder.find_spec(pkg)
    return None if spec is None else spec.origin


def _usable(table):
    """table without the packages that would import from somewhere else."""
    out = dict(table)
    for pkg in PACKAGES:
        e = table.get(pkg)
        if e is not None and pkg != 'fastplot' and _origin(pkg) != e[0]:
            out = {k: v for k, v in out.items() if k != pkg and not k.startswith(pkg + '.')}
    return out


def install():
    """The bundle read, if there is one, and put in front of the path search."""
    if os.environ.get('FASTPLOT_BUNDLE', '1') == '0':
        return
    try:
        with open(_FILE, 'rb') as fh:
            magic, table = marshal.load(fh)
        if magic != importlib.util.MAGIC_NUMBER:
            table = {}
    except (OSError, ValueError, EOFError, TypeError):
        table = {}
    e = table.get('numba')
    if e is not None:
        try:
            st = os.stat(e[0])
            fresh = int(st.st_mtime) == e[1] and st.st_size == e[2]
        except OSError:
            fresh = False
    else:
        fresh = False
    _STATE['wanted'] = not fresh
    if fresh:
        _TABLE.update(_usable(table))
        sys.meta_path.insert(0, _Finder)


def _entry(name, path, is_pkg):
    st = os.stat(path)
    code = importlib.machinery.SourceFileLoader(name, path).get_code(name)
    return path, int(st.st_mtime), st.st_size, is_pkg, marshal.dumps(code)


def _walk(pkg, root):
    """(module name, file, is it a package) of a package's Python files, tests left out."""
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ('__pycache__', 'tests', 'test', 'testing')]
        rel = os.path.relpath(folder, root)
        parts = [pkg] + ([] if rel == '.' else rel.split(os.sep))
        for f in files:
            if not f.endswith('.py'):
                continue
            if f == '__init__.py':
                yield '.'.join(parts), os.path.join(folder, f), True
            else:
                yield '.'.join(parts + [f[:-3]]), os.path.join(folder, f), False


def write():
    """The bundle written (whole, under a new name, then renamed over the old)."""
    table = {}
    for pkg in PACKAGES:
        origin = _origin(pkg)
        if origin is None or not origin.endswith('__init__.py'):
            continue
        for name, path, is_pkg in _walk(pkg, os.path.dirname(origin)):
            try:
                table[name] = _entry(name, path, is_pkg)
            except (OSError, SyntaxError, ValueError):
                pass
    tmp = '%s.%d' % (_FILE, os.getpid())
    os.makedirs(os.path.dirname(_FILE), exist_ok=True)
    with open(tmp, 'wb') as fh:
        marshal.dump((importlib.util.MAGIC_NUMBER, table), fh)
    os.replace(tmp, _FILE)
    _STATE['wanted'] = False


def write_if_wanted():
    """At exit: the bundle made if this process missed it, as Python would
    have written its .pyc files. Not a failure of the program if it cannot."""
    if _STATE['wanted'] and not sys.dont_write_bytecode and 'numba' in sys.modules \
            and os.environ.get('FASTPLOT_BUNDLE', '1') != '0':
        try:
            write()
        except Exception:                       # noqa: BLE001 -- a head start only
            pass
