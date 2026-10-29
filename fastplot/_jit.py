"""numba, imported when the first kernel runs rather than with fastplot:
`njit`, `intrinsic` and `prange` as the modules use them, standing in
until then. Importing fastplot, building a figure, measuring text -- the
work that runs no kernel -- then never pays for numba's import (~240 ms),
and a figure drawn on the GPU never pays for it at all.

The first call of any kernel imports numba, makes every kernel known so
far (each from its cache as before: the functions are the same), and puts
them in place of their stand-ins in every fastplot module, so kernels
calling kernels see numba's dispatchers. Kernels defined after that are
numba's from the start, and so is `prange` imported from here; one whose
module took a stand-in while numba was loading (on the preload thread)
has its module's stand-ins replaced first. A stand-in called from Python
works throughout.

`preload()` (called by `import fastplot`) gets numba ready on a daemon
thread while the program does other work -- reads its data, builds its
figure: numba imported, its compiler set up (~0.3 s, once a process) and
every kernel's compiled code loaded from numba's cache, those a draw uses
first, until the first draw begins (which then loads what it still needs
itself, never waiting behind kernels it does not). Only loaded, never
run, so nothing races the program's own kernels. Forking waits for it;
at exit it stops after the kernel it is loading.
FASTPLOT_PRELOAD=0 turns it off.

`threads(n)` runs a block's kernels on at most n of numba's threads (n
None: a draw's, from the machine's cores -- _hw.draw_threads). A draw is
many parallel regions, most of them small (a spine, a legend, a tile
list), and each one's start costs more the more threads it wakes: on a
24-thread i7-13700K a draw was 10-40% faster on its 8 performance cores
than on 24, a map at 3600x1950 px no slower. threads(park=True) lets
numba's workers go when the outermost such block ends (park()). The count is best kept the same from one
parallel region to the next (numba resizing its team of threads costs
~0.8 ms), so the kernels run outside a draw take it too (`capped`), and so
do the PNG encoder's. The image is the same on any number.
"""
import atexit
import functools
import os
import sys
import threading

_PENDING = []                   # stand-ins not yet made
_LOCK = threading.RLock()
_numba = None                   # the numba module, once imported
_PARK = sys.platform == 'win32' and os.environ.get('FASTPLOT_PARK', '1') != '0'   # see park()
_LOCAL = threading.local()      # this thread's threads() blocks waiting for numba
_MADE = []                      # every njit kernel made: (its module, numba's dispatcher)
_PRELOAD = {'thread': None, 'stop': False}
PRELOAD_FIRST = ('_render', '_draw', '_text', '_png', '_color', '_artists', '_raster',
                 '_cartesian', '_plots', '_contour', '_fields', '_clabel', '_path', '_markers',
                 '_legend', '_proj')    # a draw's modules: their kernels load first


class _Kernel:
    """A stand-in for numba.njit(**opts)(fn) or numba.extending.intrinsic(fn)."""

    def __init__(self, fn, opts, kind):
        self.fn, self.opts, self.kind, self.real = fn, opts, kind, None
        self.__wrapped__, self.__name__ = fn, fn.__name__
        self.__doc__ = fn.__doc__

    def make(self, numba):
        if self.kind == 'intrinsic':
            from numba.extending import intrinsic as make
            return make(self.fn)
        return numba.njit(**self.opts)(self.fn)

    def __call__(self, *args, **kwargs):
        if self.real is None:
            load()
        return self.real(*args, **kwargs)

    def __getattr__(self, name):                # (signatures, inspect_asm, py_func ...)
        if name == 'real':
            raise AttributeError(name)
        if self.real is None:
            load()
        return getattr(self.real, name)

    def __repr__(self):
        return '<kernel %s%s>' % (self.__name__, '' if self.real is not None
                                  else ' (numba not loaded yet)')


class _Prange:
    """numba.prange's stand-in (a plain range from Python)."""

    def __call__(self, *args):
        return range(*args)


_STAND_IN = _Prange()
prange = _STAND_IN


def njit(*args, **opts):
    """numba.njit, deferred (as @njit(cache=True, nogil=True), or bare)."""
    if args and callable(args[0]):
        return _defer(args[0], {}, 'njit')
    return lambda fn: _defer(fn, opts, 'njit')


def intrinsic(fn):
    """numba.extending.intrinsic, deferred."""
    return _defer(fn, {}, 'intrinsic')


def _defer(fn, opts, kind):
    with _LOCK:
        k = _Kernel(fn, opts, kind)
        if _numba is not None:                  # numba up already: made at once
            _swap(fn.__globals__)               # (stand-ins its module took meanwhile)
            real = k.make(_numba)
            if kind == 'njit':
                _MADE.append((fn.__module__, real))
            return real
        _PENDING.append(k)
        return k


class threads:
    """A block whose kernels run on at most n (a draw's) of numba's threads,
    numba's count as it was after it. numba's set_num_threads is the
    calling thread's own, so other threads are not touched; before numba
    is imported the cap waits, and is set when the first kernel imports it."""

    def __init__(self, n=None, park=False):
        self.n, self.old, self.park = None if n is None else int(n), None, park

    def apply(self):
        n = self.n
        if n is None:
            from . import _hw
            n = _hw.draw_threads()
        self.old = _numba.get_num_threads()
        _numba.set_num_threads(max(1, min(self.old, n)))
        _LOCAL.__dict__.setdefault('active', []).append(self)

    def __enter__(self):
        _PRELOAD['stop'] = True                 # a draw: it loads what it needs from here
        if _numba is not None:
            self.apply()
        else:
            _LOCAL.__dict__.setdefault('waiting', []).append(self)
        return self

    def __exit__(self, *exc):
        waiting = getattr(_LOCAL, 'waiting', None)
        if waiting and self in waiting:
            waiting.remove(self)
        if self.old is not None:
            _numba.set_num_threads(self.old)
            self.old = None
            _LOCAL.active.remove(self)
            if self.park and not _LOCAL.active:
                park()
        return False


def _size_pool():
    """numba's pool sized to the CPUs this process may use, when that is
    fewer than the machine's (an affinity mask, a container's quota): numba
    counts the machine's (os.cpu_count), and would start a thread for each
    -- 64 in a 4-CPU container on a 64-core host. Not when numba is already
    imported, nor over the user's NUMBA_NUM_THREADS."""
    if 'numba' in sys.modules or 'NUMBA_NUM_THREADS' in os.environ:
        return
    from . import _hw
    n = _hw.cpus()
    if n < (os.cpu_count() or n):
        os.environ['NUMBA_NUM_THREADS'] = str(n)


def team():
    """The threads this thread's next parallel region runs on (numba's
    count now; before numba is up, a draw's)."""
    if _numba is None:
        from . import _hw
        return _hw.draw_threads()
    return _numba.get_num_threads()


def park():
    """numba's idle worker threads sent to sleep now, rather than after
    spinning: MSVC's OpenMP (numba's omp layer on Windows) keeps every
    worker of the last parallel region busy-waiting ~100 ms after it, for a
    region that might follow -- 7 cores x 100 ms after every frame, more
    than the frame's own work, and cores that deflate's threads and other
    processes then share. A region of one thread lets them go at once
    (~3 us); the next region wakes them again (~0.3 ms, once a frame).
    Called on the thread that ran the kernels (the pool is its own) when a
    draw or a savefig ends and before deflate; a no-op elsewhere (other
    OpenMP runtimes, numba's other layers, numba not up)."""
    if not _PARK or _numba is None:
        return
    try:
        if _numba.threading_layer() != 'omp':
            return
    except ValueError:                          # no parallel region has run: nothing spins
        return
    old = _numba.get_num_threads()
    _numba.set_num_threads(1)
    try:
        _park_region(_PARK_BUF)
    finally:
        _numba.set_num_threads(old)


def capped(fn):
    """fn run inside threads(): for the kernels fastplot runs outside a
    draw (colours, a histogram's counts), so that they and the draw's run
    on as many threads -- numba resizing its team of threads between two
    parallel regions costs ~0.8 ms."""
    @functools.wraps(fn)
    def run(*args, **kwargs):
        with threads():
            return fn(*args, **kwargs)
    return run


def load():
    """numba imported, every pending kernel made, and each fastplot module's
    stand-ins replaced by them."""
    global _numba
    with _LOCK:
        if _numba is None:
            _size_pool()
            import numba
            for k in _PENDING:
                k.real = k.make(numba)
                if k.kind == 'njit':
                    _MADE.append((k.fn.__module__, k.real))
            _numba = numba
        for t in getattr(_LOCAL, 'waiting', ()):        # this thread's threads() blocks
            t.apply()
        _LOCAL.waiting = []
        if not _PENDING:
            return
        # the kernels' own modules (whatever sys.modules says now), and every
        # fastplot module (those holding a kernel imported from another)
        pkg = __name__.rpartition('.')[0]
        spaces = {id(k.fn.__globals__): k.fn.__globals__ for k in _PENDING}
        for name, mod in list(sys.modules.items()):
            if mod is not None and (name == pkg or name.startswith(pkg + '.')):
                spaces[id(vars(mod))] = vars(mod)
        for d in spaces.values():
            _swap(d)
        _PENDING.clear()


def _swap(d):
    """A namespace's stand-ins replaced by numba's: kernels made, prange."""
    # (the sentinel held in a local: replacing this module's own _STAND_IN
    # global mid-loop would hide every stand-in after it -- this module's
    # `prange` among them, which a module imported later then picks up)
    stand_in = _STAND_IN
    for key, val in list(d.items()):
        if isinstance(val, _Kernel):
            if val.real is not None:
                d[key] = val.real
        elif val is stand_in and not (d is globals() and key == '_STAND_IN'):
            d[key] = _numba.prange


def _preload_run():
    try:
        load()
        pkg = __name__.rpartition('.')[0] + '.'
        first = {pkg + m: k for k, m in enumerate(PRELOAD_FIRST)}
        order = sorted(range(len(_MADE)), key=lambda i: (first.get(_MADE[i][0], len(first)), i))
        for i in order:
            disp = _MADE[i][1]
            cache = getattr(disp, '_cache', None)
            try:                                # numba's own cache index (internal API)
                sigs = list(cache._cache_file._load_index()) if cache is not None else []
            except Exception:                   # noqa: BLE001 -- then it loads when called
                sigs = []
            for key in sigs:
                if _PRELOAD['stop']:
                    return
                if key[0] not in disp.overloads:
                    disp.compile(key[0])
    except Exception:                           # noqa: BLE001 -- a head start only
        pass


def _stop_preload():
    _PRELOAD['stop'] = True
    _join_preload()


def _join_preload():
    t = _PRELOAD['thread']
    if t is not None and t is not threading.current_thread():
        t.join()


def preload():
    """numba imported and the kernels' compiled code loaded on a daemon
    thread, now (once a process; FASTPLOT_PRELOAD=0: never)."""
    if _PRELOAD['thread'] is not None or os.environ.get('FASTPLOT_PRELOAD', '1') == '0':
        return
    t = threading.Thread(target=_preload_run, name='fastplot-preload', daemon=True)
    _PRELOAD['thread'] = t
    atexit.register(_stop_preload)
    if hasattr(os, 'register_at_fork'):
        os.register_at_fork(before=_join_preload)
    t.start()


@njit(cache=True, nogil=True, parallel=True)
def _park_region(a):
    """A parallel region (park() runs it on one thread)."""
    for i in prange(a.size):
        a[i] = 0.0


_PARK_BUF = __import__('numpy').zeros(1)
