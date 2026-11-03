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
figure: numba imported (its compiler is not set up: only a kernel that
is not in the cache needs it) and
every kernel's compiled code loaded from numba's cache, those a draw uses
first, until the first draw begins (which then loads what it still needs
itself, never waiting behind kernels it does not). Only loaded, never
run, so nothing races the program's own kernels. Forking waits for it;
at exit it stops after the kernel it is loading.
FASTPLOT_PRELOAD=0 turns it off. It is also what notices when the cache
was not filled for this machine (_warm.keep_up).

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

Before any of that, this machine's kernel image (_image.py) is looked for,
on the preload thread or by the first kernel: it runs the kernels with no
numba at all, the parallel ones on numba's own threading layer, loaded
alone. Then numba is not imported, and the thread counts above are that
layer's. A kernel or argument types the image does not hold, or no image
to be had, bring numba up as above (an image is then built for later
processes, in the background: _image.build_later).
"""
import atexit
import functools
import os
import sys
import threading
import time

_PENDING = []                   # stand-ins not yet made
_LOCK = threading.RLock()
_numba = None                   # the numba module, once imported
_PARK = sys.platform == 'win32' and os.environ.get('FASTPLOT_PARK', '1') != '0'   # see park()
_LOCAL = threading.local()      # this thread's threads() blocks waiting for numba
_MADE = []                      # every njit kernel made: (its module, numba's dispatcher)
_PRELOAD = {'thread': None, 'stop': False}
_LAYER = []                     # numba's threading layer, once known
_LAYER_LOCK = threading.Lock()
_SERIAL = threading.RLock()     # on the workqueue layer: one thread's parallel regions at a time
_LOOK = threading.Lock()
_looked = False                 # this process has looked for a kernel image
_IMAGE = None                   # _image.Image the stand-ins call through, if any
_NO = None                      # _image.NO
_MISS = type('Miss', (), {})()  # an image's function of Python's: not arguments it takes
PRELOAD_FIRST = ('_render', '_draw', '_text', '_png', '_color', '_artists', '_raster',
                 '_cartesian', '_plots', '_contour', '_fields', '_clabel', '_path', '_markers',
                 '_legend', '_proj')    # a draw's modules: their kernels load first
CACHE_ONLY_NUMBA = {('0', '68')}        # see _cache_needs_no_compiler


class _Kernel:
    """A stand-in for numba.njit(**opts)(fn) or numba.extending.intrinsic(fn)."""

    def __init__(self, fn, opts, kind):
        self.fn, self.opts, self.kind, self.real = fn, opts, kind, None
        self.fast = {}                          # (the image) Python types of the arguments -> unit
        self.entries = []                       # (the image) its units as functions of Python's
        self.__wrapped__, self.__name__ = fn, fn.__name__
        self.__doc__ = fn.__doc__

    def make(self, numba):
        if self.kind == 'intrinsic':
            from numba.extending import intrinsic as make
            return make(self.fn)
        real = numba.njit(**self.opts)(self.fn)
        if self.opts.get('cache'):
            _cache_needs_no_compiler(real, numba)
        return real

    def __call__(self, *args, **kwargs):
        if not kwargs:
            for f in self.entries:
                r = f(*args)
                if r is not _MISS:
                    return r
        if not _looked:
            _look()
        if _IMAGE is not None and not kwargs:
            r = _IMAGE.call(self, args)
            if r is not _NO:
                return r
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


def _cache_needs_no_compiler(disp, numba):
    """A kernel of `disp`'s found in numba's cache, without numba first
    setting up the compiler it will not use. numba refreshes its target
    context before looking in the cache (every implementation module,
    scipy.linalg with them: ~0.4 s once a process) and reads every
    installed package's metadata for extensions (~40 ms); compiled code
    needs only the runtime. A kernel not in the cache compiles, and numba's
    compiler does all of that itself. Only on numba releases tried
    (CACHE_ONLY_NUMBA): compiled code missing a symbol stops the process."""
    if tuple(numba.__version__.split('.')[:2]) not in CACHE_ONLY_NUMBA:
        return
    cache = disp._cache

    def load_overload(sig, target_context):
        from numba.core.runtime import rtsys
        rtsys.initialize(target_context)
        with cache._guard_against_spurious_io_errors():
            return cache._load_overload(sig, target_context)
    cache.load_overload = load_overload
    disp._compilation_chain_init_hook = _nothing


def _nothing():
    pass


def _blas_check_no_scipy_linalg():
    """Cached kernels using numba's own array functions import
    numba.np.arraymath, which at import asks whether scipy has BLAS by
    importing scipy.linalg (~200 ms). That one question is answered by
    scipy being there (numba has imported it); anything asked after the
    import, by code being compiled, goes to numba's own check."""
    from numba.np import linalg
    real = linalg.ensure_blas

    def ensure_blas():
        arraymath = sys.modules.get('numba.np.arraymath')
        if arraymath is None or hasattr(arraymath, '_HAVE_BLAS') or 'scipy' not in sys.modules:
            real()
    linalg.ensure_blas = ensure_blas


def _check_folder_once():
    """numba proves a cache folder writable, by making a file in it, once
    per kernel (~0.3 ms each on Windows): once per folder is enough."""
    from numba.core import caching
    real = caching._CacheLocator.ensure_cache_path
    done = set()

    def ensure_cache_path(self):
        path = self.get_cache_path()
        if path not in done:
            real(self)
            done.add(path)
    caching._CacheLocator.ensure_cache_path = ensure_cache_path


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
        self.held = False

    def apply(self):
        n = self.n
        if n is None:
            from . import _hw
            n = _hw.draw_threads()
        active = _LOCAL.__dict__.setdefault('active', [])
        p = self.pool = _pool()
        if not active and layer() == 'workqueue':       # see layer()
            _SERIAL.acquire()
            self.held = True
        self.old = p.get()
        p.set(max(1, min(self.old, n)))
        active.append(self)

    def __enter__(self):
        _PRELOAD['stop'] = True                 # a draw: it loads what it needs from here
        if not _looked:
            _look()
        if _ready():
            self.apply()
        else:
            _LOCAL.__dict__.setdefault('waiting', []).append(self)
        return self

    def __exit__(self, *exc):
        waiting = getattr(_LOCAL, 'waiting', None)
        if waiting and self in waiting:
            waiting.remove(self)
        if self.old is not None:
            self.pool.set(self.old)
            self.old = None
            _LOCAL.active.remove(self)
            if self.park and not _LOCAL.active:
                park()
        if self.held:
            self.held = False
            _SERIAL.release()
        return False


class wider:
    """A block of even, independent work (markers' masks), on the performance
    cores' second threads too, where a draw would not use them: each part of
    a draw is short and starting them costs it, but here 16 threads did the
    work of 8 in 2/3 of the time. Nothing if asked not to, or if the count of
    threads is not a draw's own (frames(), FASTPLOT_THREADS)."""

    def __init__(self, on=True):
        self.on, self.old, self.pool = on, None, None

    def __enter__(self):
        from . import _hw
        n = _hw.wide_threads()
        p = _current_pool()
        if self.on and p is not None and n > p.get() == _hw.draw_threads() and n <= p.size:
            self.old, self.pool = p.get(), p
            p.set(n)
        return self

    def __exit__(self, *exc):
        if self.old is not None:
            self.pool.set(self.old)
        return False


class kept:
    """A block (a draw, a savefig) whose numpy arrays come from the kernel
    image's kept memory (_image._kept_object): the large ones a frame lets go
    of are handed to the next rather than back to the system, whose new
    pages each cost a fault. numpy's handler is per thread (a context
    variable): other threads are not touched, and an array made here goes
    back to the kept memory whenever it is let go. Nothing without an image."""
    __slots__ = ('old',)

    def __enter__(self):
        self.old = None
        im = _IMAGE
        if im is not None:
            h = im.numpy_handler()
            if h is not None:
                from . import _image
                self.old = _image.set_numpy_handler(h)
        return self

    def __exit__(self, *exc):
        if self.old is not None:
            from . import _image
            _image.set_numpy_handler(self.old)
        return False


class _NumbaPool:
    """numba's thread counts, as _image.Pool has them."""

    @staticmethod
    def get():
        return _numba.get_num_threads()

    @staticmethod
    def set(n):
        _numba.set_num_threads(n)

    @property
    def size(self):
        return _numba.config.NUMBA_NUM_THREADS


_NUMBA_POOL = _NumbaPool()


def _ready():
    """Is there a pool to set thread counts on (or one to be had at once)?"""
    return _numba is not None or _IMAGE is not None


def _pool():
    """What runs the parallel kernels' threads: numba's threading layer as
    the image's kernels use it, else numba (loaded, if it is not)."""
    if not _looked:
        _look()
    if _IMAGE is not None:
        p = _IMAGE.pool()
        if p is not None:
            return p
    load()
    return _NUMBA_POOL


def _current_pool():
    """_pool(), if there is one yet (nothing is started for it); else None."""
    if _IMAGE is not None:
        from . import _image
        p = _image.pool()
        if p is not None:
            return p
    return _NUMBA_POOL if _numba is not None else None


def get_num_threads():
    """numba.get_num_threads(), from whichever runs the kernels."""
    return _pool().get()


def set_num_threads(n):
    """numba.set_num_threads(n), on whichever runs the kernels."""
    _pool().set(n)


def pool_size():
    """numba.config.NUMBA_NUM_THREADS, from whichever runs the kernels."""
    return _pool().size


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


class Choice:
    """Serial or parallel: of two kernels computing the very same thing (to
    the bit), the one this machine runs faster for work of a size, found by
    timing them as they are used -- no count fitted to one machine. Sizes
    go by powers of two, per key (what the work is) and team(). At a size
    not yet timed both ways, parallel goes first, then serial if parallel
    was quick (_way); a parallel win at a size stands for the sizes above
    it, a serial one for those below; a size's
    loser is tried again every RETRY calls of it, so one slow run (a pool
    waking, another process) is not kept; each way's first run of a key is
    not counted (a kernel loaded or compiled, threads started). Below LOW
    points serial, as no launch of threads can pay; from SURE up parallel,
    untimed."""

    RETRY = 64
    EXPLORE = 1e-4                      # s: parallel this quick, serial is worth a try

    def __init__(self, low=64, sure=1 << 16):
        self.low, self.sure = low, sure
        self.best = {}                  # (key, team) -> {size bit: [serial s/pt, parallel s/pt, calls]}
        self.ran = set()                # (key, way): run once already

    force = None                        # 0 or 1: that way, untimed (tests)

    def run(self, key, n, ser, par, *args):
        """ser(*args) or par(*args), timed."""
        way, tick = self.pick(key, n)
        out = (par if way else ser)(*args)
        self.took(tick, way)
        return out

    def pick(self, key, n):
        """(way, tick): way 1 (parallel) or 0 for n points of key, starting
        now; tick goes to took() when that way is done (None: not timed)."""
        if self.force is not None:
            return self.force, None
        if n < self.low:
            return 0, None
        k = team()
        if k < 2:
            return 0, None
        if n >= self.sure:
            return 1, None
        sizes = self.best.setdefault((key, k), {})
        bit = int(n).bit_length()
        rec = sizes.get(bit)
        if rec is None:
            rec = sizes[bit] = [None, None, 0]
        return self._way(sizes, bit, rec, n), (key, n, rec, time.perf_counter())

    def took(self, tick, way):
        """The work tick was picked for is done, the way it went."""
        if tick is None:
            return
        key, n, rec, t = tick
        per = (time.perf_counter() - t) / n
        if (key, way) not in self.ran:
            self.ran.add((key, way))
        elif rec[way] is None or per < rec[way]:
            rec[way] = per

    def _way(self, sizes, bit, rec, n):
        """1 (parallel) or 0 for a call of n points, this size."""
        s, p = rec[0], rec[1]
        if s is not None and p is not None:
            rec[2] += 1
            win = 1 if p < s else 0
            return 1 - win if rec[2] % self.RETRY == 0 else win
        for b, (bs, bp, _) in sizes.items():            # what the sizes around it found
            if bs is not None and bp is not None:
                if b <= bit and bp < bs:
                    return 1
                if b >= bit and bs <= bp:
                    return 0
        # not yet timed here: parallel first (a wrong try costs a launch of
        # threads, a wrong serial one all the work), and serial only where
        # parallel was quick enough for a launch to matter -- past a few
        # launches' time the work is most of it, and one thread does it slower
        if p is None:
            return 1
        return 0 if p * n < self.EXPLORE else 1


def team():
    """The threads this thread's next parallel region runs on (numba's
    count now; before numba or the image's pool is up, a draw's)."""
    p = _current_pool()
    if p is None:
        from . import _hw
        return _hw.draw_threads()
    return p.get()


def layer():
    """numba's threading layer: 'tbb', 'omp' or 'workqueue' (a parallel
    region run first, on one thread, if none has: numba picks it then).
    'workqueue' -- what pip's numba has on Apple silicon, where it ships
    neither TBB nor an OpenMP runtime -- aborts the process when two
    threads start parallel regions at once: there fastplot's run one
    thread at a time (a draw, a savefig, a kernel outside them: serial()),
    and frames() draws one frame at a time."""
    if _LAYER:
        return _LAYER[0]
    p = _pool()
    if p is not _NUMBA_POOL:                    # (the image's: started, so known)
        return p.name
    with _LAYER_LOCK:
        if not _LAYER:
            try:
                name = _numba.threading_layer()
            except ValueError:                  # no parallel region has run
                old = _numba.get_num_threads()
                _numba.set_num_threads(1)
                try:
                    _park_region(_PARK_BUF)
                finally:
                    _numba.set_num_threads(old)
                name = _numba.threading_layer()
            _LAYER.append(name)
    return _LAYER[0]


class serial:
    """A block whose parallel regions no other thread's overlap, on the
    workqueue layer (see layer()); elsewhere nothing."""

    def __enter__(self):
        self.held = layer() == 'workqueue'
        if self.held:
            _SERIAL.acquire()
        return self

    def __exit__(self, *exc):
        if self.held:
            _SERIAL.release()
        return False


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
    OpenMP runtimes, numba's other layers, neither numba nor the image's
    pool up)."""
    p = _current_pool() if _PARK else None
    if p is None:
        return
    if p is _NUMBA_POOL:
        try:
            if _numba.threading_layer() != 'omp':
                return
        except ValueError:                      # no parallel region has run: nothing spins
            return
    elif p.name != 'omp':
        return
    old = p.get()
    p.set(1)
    try:
        _park_region(_PARK_BUF)
    finally:
        p.set(old)


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
    if _load():
        from . import _image
        _image.build_later()
    _settle()


def _settle():
    """This thread's threads() blocks applied, once there is a pool for them."""
    waiting = getattr(_LOCAL, 'waiting', None)
    if waiting and _ready():
        _LOCAL.waiting = []
        for t in waiting:
            t.apply()


def _load():
    """Whether this brought numba up (an image looked for and not enough)."""
    global _numba
    first = False
    with _LOCK:
        if _numba is None:
            _size_pool()
            from . import _bundle
            _bundle.install_numba()
            import numba
            if tuple(numba.__version__.split('.')[:2]) in CACHE_ONLY_NUMBA:
                _check_folder_once()
                _blas_check_no_scipy_linalg()
            for k in _PENDING:
                k.real = k.make(numba)
                if k.kind == 'njit':
                    _MADE.append((k.fn.__module__, k.real))
            _numba = numba
            _keep_count(numba)
            first = _looked and not _image_off()
        if not _PENDING:
            return first
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
    return first


def _keep_count(numba):
    """numba, come up after the image's pool did (the same threading layer:
    the same DLL), sets this thread's count to the pool's size as it starts
    the layer. Started now, and this thread's count put back."""
    if _IMAGE is None:
        return
    from . import _image
    p = _image.pool()
    if p is not None:
        n = p.get()
        from numba.np.ufunc import parallel
        parallel._launch_threads()
        p.set(n)


def _image_off():
    return os.environ.get('FASTPLOT_IMAGE', '').strip() == '0'


def _look():
    """The image looked for, once a process: every thread here first waits for
    that, then each goes by what it found."""
    global _looked
    with _LOOK:
        if not _looked:
            if not _image_off():
                try:
                    from . import _image
                    _image.open_image()
                except Exception:                       # noqa: BLE001 -- numba, then
                    pass
            _looked = True
    _settle()


def image():
    """The kernel image the stand-ins call through, or None."""
    return _IMAGE


def use_image(im):
    """Kernels not yet numba's go through im (an _image.Image) where it can."""
    global _IMAGE, _NO
    from . import _image
    _NO = _image.NO
    _IMAGE = im


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
        _look()
        if _IMAGE is not None:                  # (nothing of numba's wanted: what a first
            from . import _image                # figure waits for, done meanwhile)
            _image.prewarm(_IMAGE)
            return
        load()
        from . import _warm
        _warm.keep_up()
        here = _warm._machine()
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
                if key[1] == here and key[0] not in disp.overloads:   # (another CPU's: not here)
                    disp.compile(key[0])
    except Exception:                           # noqa: BLE001 -- a head start only
        pass


def _stop_preload():
    _PRELOAD['stop'] = _PRELOAD['exiting'] = True
    _join_preload()
    from . import _bundle
    _bundle.write_if_wanted()


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
