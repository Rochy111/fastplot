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

There are two such files: fastplot's own with freetype's and Pillow's, read
by install() first thing in fastplot/__init__.py, and numba's with
llvmlite's (two thirds of the bytes), read by install_numba() only when
numba is about to be imported -- with the kernel image it never is (~4 ms
a fresh process does not spend). write() makes them: by warm(), and at the
exit of a process that missed one, once per install or upgrade (~0.8 s).
They sit in fastplot's __pycache__, trusted as the .pyc files beside them
are. FASTPLOT_BUNDLE=0 turns them off.
"""
import _thread
import importlib.machinery
import importlib.util
import marshal
import os
import sys

_LOCK = _thread.allocate_lock()

GROUPS = {'imports': ('freetype', 'PIL', 'fastplot'), 'imports-numba': ('numba', 'llvmlite')}
PACKAGES = GROUPS['imports-numba'] + GROUPS['imports']
_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '__pycache__')
_TAG = '%s.opt%d.bin' % (sys.implementation.cache_tag, sys.flags.optimize)
_TABLE = {}                     # module name: (source path, its mtime, its size, is it a package, code)
_STATE = {'wanted': set(), 'read': set()}   # files not as their packages are now; files read


def _file(group):
    return os.path.join(_DIR, '%s.%s' % (group, _TAG))


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


def install(group='imports'):
    """A bundle file read, if there is one and its packages are as they were
    (each package's __init__ unchanged), and put in front of the path search."""
    if os.environ.get('FASTPLOT_BUNDLE', '1') == '0':
        return
    with _LOCK:                                 # (a second thread waits until it is in place)
        if group not in _STATE['read']:
            _install(group)
            _STATE['read'].add(group)


def _install(group):
    try:
        with open(_file(group), 'rb') as fh:
            magic, table = marshal.load(fh)
        if magic != importlib.util.MAGIC_NUMBER:
            table = {}
    except (OSError, ValueError, EOFError, TypeError):
        table = {}
    fresh = bool(table)
    for pkg in GROUPS[group]:
        e = table.get(pkg)
        if e is not None:
            try:
                st = os.stat(e[0])
                fresh = fresh and int(st.st_mtime) == e[1] and st.st_size == e[2]
            except OSError:
                fresh = False
    if not fresh:
        _STATE['wanted'].add(group)
        return
    _TABLE.update(_usable(table))
    if _Finder not in sys.meta_path:
        sys.meta_path.insert(0, _Finder)


def install_numba():
    """numba's and llvmlite's bundle, as numba is about to be imported."""
    install('imports-numba')


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


def write(groups=tuple(GROUPS)):
    """The bundle files written (each whole, under a new name, then renamed
    over the old)."""
    for group in groups:
        table = {}
        for pkg in GROUPS[group]:
            origin = _origin(pkg)
            if origin is None or not origin.endswith('__init__.py'):
                continue
            for name, path, is_pkg in _walk(pkg, os.path.dirname(origin)):
                try:
                    table[name] = _entry(name, path, is_pkg)
                except (OSError, SyntaxError, ValueError):
                    pass
        target = _file(group)
        tmp = '%s.%d' % (target, os.getpid())
        os.makedirs(_DIR, exist_ok=True)
        with open(tmp, 'wb') as fh:
            marshal.dump((importlib.util.MAGIC_NUMBER, table), fh)
        os.replace(tmp, target)
        _STATE['wanted'].discard(group)


def write_if_wanted():
    """At exit: the bundle files this process missed, as Python would have
    written its .pyc files: fastplot's after it drew, numba's if it used
    numba. Not a failure of the program if it cannot."""
    from . import _jit
    drew = 'numba' in sys.modules or _jit.image() is not None
    groups = [g for g in GROUPS if g in _STATE['wanted'] and drew
              and (g != 'imports-numba' or 'numba' in sys.modules)]
    if groups and not sys.dont_write_bytecode and os.environ.get('FASTPLOT_BUNDLE', '1') != '0':
        try:
            write(groups)
        except Exception:                       # noqa: BLE001 -- a head start only
            pass
