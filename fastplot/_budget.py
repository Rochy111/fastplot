"""One memory budget for everything fastplot keeps from one figure for the
next: a map's projected features and graticule, drawings kept for the
arrays they were made from, pixel and cell tables, kept regions, the
triangles along a field's NaN holes, recorded ink.

Each of those is a Cache: least recently used first, its entries' bytes
counted. When a cache takes an entry and all of them together hold more
than budget(), the entry longest unused for what it is worth -- in
whichever cache it is, its age divided by its cache's weight and by
1 + the times it was used since it was kept -- goes, until they fit
(never the entry just taken; an entry of over a quarter of the budget is
not kept at all). The weights say how long a cache's entries stay worth
keeping:
a frame's drawings 1 (the next forecast hour has new data), a grid's
tables 4, a map's pixel tables and geometry 16. So a map drawn frame
after frame keeps what it uses, and an earlier frame's drawings go
first. The Natural Earth features themselves are kept whole outside the
budget (keep_all), while frames read them: once no frame has for a few
(idle), they go -- a map's projected geometry is what its frames draw.

The budget is the machine's: CACHE_SHARE of its memory, divided among the
processes likely to share it -- a multiprocessing worker counts the CPUs
over its own draw threads as its siblings, since a pool fills the
machine -- within CACHE_MIN..CACHE_MAX. FASTPLOT_CACHE_MB (or
set_budget) sets it in megabytes; 0 keeps nothing across figures.
(forecast-map render workers on a 64 GB, 24-thread machine at 3 threads:
~80 MB each, against ~300 MB the same caches held uncapped in 0.5.3; a
worker drawing 80 frames of new data then settled at ~433 MB rather
than ~607, as fast.)
"""
import os
import sys
import threading
import weakref
from collections import OrderedDict

CACHE_SHARE = 0.01              # of the machine's memory, among the processes sharing it
CACHE_MIN = 32 << 20            # bytes: a budget at least this
CACHE_MAX = 1 << 30             # and at most this

_CACHES = []                    # every Cache made
_EXTRA = []                     # functions giving bytes kept elsewhere (counted, not dropped)
_FIT = threading.Lock()
_TICK = [0]                     # uses so far, of any cache: entries' ages are told in them
_BUDGET = [None]                # bytes, once worked out (or set)


def _tick():
    _TICK[0] += 1               # (two threads alike now and then: an age off by one)
    return _TICK[0]


def _sharers():
    """Processes likely to share this machine's memory with this one: in a
    multiprocessing worker, the CPUs over this process' draw threads."""
    mp = sys.modules.get('multiprocessing')
    if mp is None:
        return 1
    try:
        if mp.parent_process() is None:
            return 1
    except Exception:                                   # noqa: BLE001 -- an odd start method
        return 1
    from . import _hw
    return max(1, _hw.cpus() // max(1, _hw.draw_threads()))


def _work_out():
    v = os.environ.get('FASTPLOT_CACHE_MB', '').strip()
    if v:
        try:
            return max(0, int(float(v) * (1 << 20)))
        except ValueError:
            pass
    from . import _hw
    mem = _hw.memory()
    if not mem:
        return CACHE_MIN
    return int(min(CACHE_MAX, max(CACHE_MIN, mem * CACHE_SHARE / _sharers())))


def budget():
    """Bytes the caches may hold in all."""
    b = _BUDGET[0]
    if b is None:
        b = _BUDGET[0] = _work_out()
    return b


def set_budget(mb=None):
    """The caches' budget in megabytes from now on (None: worked out again
    from the machine and FASTPLOT_CACHE_MB); what is over it goes now."""
    _BUDGET[0] = None if mb is None else max(0, int(float(mb) * (1 << 20)))
    fit()


def held():
    """Bytes the caches hold now, in all (but keep_all ones)."""
    return sum(c.nbytes for c in _CACHES if not c.keep_all) + sum(f() for f in _EXTRA)


def report():
    """{cache name: (entries, bytes)} (what count() was given: (None,
    bytes)), and 'budget'."""
    out = {c.name: (len(c), c.nbytes) for c in _CACHES}
    for name, fn in _NAMED:
        out[name] = (None, fn())
    out['budget'] = budget()
    return out


_NAMED = []


def info():
    """fastplot.cache_info(): the budget, what is held against it, and
    each cache's entries and megabytes."""
    rep = report()
    b = rep.pop('budget')
    return {'budget_mb': b / 2**20, 'held_mb': held() / 2**20,
            'caches': {k: (n, v / 2**20) for k, (n, v) in rep.items()}}


def count(fn, name=None):
    """fn() (bytes kept somewhere other than a Cache) counted in the total."""
    _EXTRA.append(fn)
    _NAMED.append((name or getattr(fn, '__name__', 'other'), fn))


def fit(protect=None):
    """Drop least recently used entries, across caches, until they fit the
    budget; protect: (cache, key) not to drop."""
    with _FIT:
        limit = budget()
        while held() > limit:
            best, bkey, score, now = None, None, -1.0, _TICK[0] + 1
            for c in _CACHES:
                got = c._victim(protect, now)
                if got is not None and got[0] > score:
                    best, bkey, score = c, got[1], got[0]
            if best is None or not best._drop(bkey):
                return


def clear():
    """Every cache emptied."""
    for c in _CACHES:
        c.clear()


_FRAME = [0]                    # frames drawn (frame_done), what an entry's idleness is told in
_IDLE = []                      # caches whose entries go once idle (Cache's idle)


def frame_done():
    """A frame's end (_jit.kept, the outermost draw or savefig): the entries
    of caches made with `idle` that no frame used for that many go."""
    _FRAME[0] += 1
    for c in _IDLE:
        c._drop_idle(_FRAME[0])


class Cache:
    """An LRU of key -> value within the budget (see the module). get()
    marks an entry used; put() keeps one of nbytes; resize() changes an
    entry's bytes (one filled in later). max_n: entries at most; weight:
    how many times longer than a frame's drawing an entry stays worth
    keeping (see the module); counted: False for one outside the budget's
    count (a test's stand-in); keep_all: its entries never dropped nor
    refused, nor counted against the budget (what every figure of a map
    needs and reading it again costs more than the memory: the Natural
    Earth features, kept so since 0.5.0), only reported; idle: frames an
    entry no frame used is kept for, whatever the budget (None: as long
    as the budget lets it) -- for what is cheap to make again for its size
    and wanted only now and then (a map's whole Natural Earth features,
    once its projected geometry is kept)."""

    def __init__(self, name, max_n=None, weight=1, counted=True, keep_all=False,
                 remeasure=False, idle=None):
        self.name, self.max_n, self.weight = name, max_n, float(weight)
        self.keep_all = keep_all
        self.idle = idle
        if idle is not None:
            _IDLE.append(self)
        self.remeasure = remeasure      # values that grow after they are kept (a drawing's memo
                                        # of outlines and prepared edges, filled as it is drawn):
        self._fresh = []                # measured again each use, and the next time one is kept
        self._d = OrderedDict()                 # key -> [value, nbytes, tick]
        self._bytes = 0
        self._lock = threading.RLock()
        if counted:
            _CACHES.append(self)

    @property
    def nbytes(self):
        return self._bytes

    def __len__(self):
        return len(self._d)

    def __contains__(self, key):
        return key in self._d

    def get(self, key, default=None):
        with self._lock:
            e = self._d.get(key)
            if e is None:
                return default
            self._d.move_to_end(key)
            e[2] = _tick()
            e[3] += 1
            e[4] = _FRAME[0]
            v = e[0]
            grew = False
            if self.remeasure and e[3] <= 4:            # (a memo fills in its first draws: ink
                                                        # from the second; then it is what it is)
                n = self.remeasure(v) if callable(self.remeasure) else nbytes(v)
                if n != e[1]:
                    self._bytes += n - e[1]
                    e[1] = n
                    grew = True
        if grew:
            fit((self, key))
        return v

    def peek(self, key, default=None):
        """The value without marking it used."""
        e = self._d.get(key)
        return default if e is None else e[0]

    def put(self, key, value, nbytes=0):
        """Keep value (nbytes) under key; False if it is too big to keep."""
        nbytes = int(nbytes)
        if nbytes > budget() // 4 and not self.keep_all:
            self.pop(key)
            return False
        with self._lock:
            if self.remeasure:                          # (what the last ones kept learnt since)
                for k in self._fresh:
                    e = self._d.get(k)
                    if e is not None:
                        n = self.remeasure(e[0]) if callable(self.remeasure) else \
                            _measure(e[0])              # (put's nbytes is the argument)
                        self._bytes += n - e[1]
                        e[1] = n
                self._fresh = [key]
            old = self._d.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            self._d[key] = [value, nbytes, _tick(), 0, _FRAME[0]]   # (value, bytes, last use,
                                                                    # uses, frame last used)
            self._bytes += nbytes
            while self.max_n is not None and len(self._d) > self.max_n:
                _, gone = self._d.popitem(last=False)
                self._bytes -= gone[1]
        if nbytes:
            fit((self, key))
        return True

    def resize(self, key, nbytes):
        """The entry under key holds nbytes now (False: dropped, too big)."""
        nbytes = int(nbytes)
        if nbytes > budget() // 4:
            self.pop(key)
            return False
        with self._lock:
            e = self._d.get(key)
            if e is None:
                return False
            self._bytes += nbytes - e[1]
            e[1] = nbytes
        fit((self, key))
        return True

    def pop(self, key, default=None):
        with self._lock:
            e = self._d.pop(key, None)
            if e is None:
                return default
            self._bytes -= e[1]
            return e[0]

    def clear(self):
        with self._lock:
            self._d.clear()
            self._bytes = 0
            self._fresh = []

    def keys(self):
        with self._lock:
            return list(self._d)

    def values(self):
        with self._lock:
            return [e[0] for e in self._d.values()]

    def _victim(self, protect, now):
        """(score, key) of the entry to drop first: the longest unused for
        what it is worth -- its age over the cache's weight and over
        1 + its uses since it was kept (up to 8: a table every frame reads
        outweighs a drawing no other figure took) -- or None."""
        if self.keep_all:
            return None
        best = None
        with self._lock:
            for k, e in self._d.items():
                if e[1] and (protect is None or protect[0] is not self or protect[1] != k):
                    s = (now - e[2]) / (self.weight * (1 + min(e[3], 8)))
                    if best is None or s > best[0]:
                        best = (s, k)
        return best

    def _drop_idle(self, frame):
        """The entries no frame used for more than `idle` frames, gone."""
        idle = self.idle
        with self._lock:
            old = [k for k, e in self._d.items() if frame - e[4] > idle]
            for k in old:
                self._bytes -= self._d.pop(k)[1]

    def _drop(self, key):
        with self._lock:
            e = self._d.pop(key, None)
            if e is None:
                return False
            self._bytes -= e[1]
            return True


_COPIES = {}                    # (shape, dtype, a sample's bytes) -> weakrefs of kept copies
_COPIES_LOCK = threading.Lock()
_COPY_BYTES = [0]               # bytes of the copies alive


def _copy_gone(key, ref, n):
    with _COPIES_LOCK:
        refs = _COPIES.get(key)
        if refs is not None:
            try:
                refs.remove(ref)
            except ValueError:
                pass
            if not refs:
                del _COPIES[key]
        _COPY_BYTES[0] -= n


def kept_copy(a):
    """A C-contiguous copy of array a that a cache keeps to tell a's bits
    by: one copy for every array of the same bits kept anywhere (a map's
    lon/lat under each of a frame's drawings), counted in the budget while
    anything holds it. Compared to the bit, never assumed alike."""
    import numpy as np
    a = np.asarray(a)
    flat = a.reshape(-1)
    step = max(1, flat.size // 64)
    sample = np.ascontiguousarray(flat[::step][:64])
    key = (a.shape, a.dtype.str, sample.tobytes())
    k = a.dtype.itemsize
    bits = (lambda v: v.view('u%d' % k)) if k in (1, 2, 4, 8) and a.dtype.kind != 'O' else \
        (lambda v: v.view(np.uint8))
    with _COPIES_LOCK:
        refs = list(_COPIES.get(key, ()))
    mine = bits(np.ascontiguousarray(a))
    for ref in refs:
        c = ref()
        if c is not None and np.array_equal(bits(c), mine):
            return c
    c = np.array(a, copy=True, order='C')               # (writeable, as the kernels' image
    ref = weakref.ref(c)                                # types it: only read, never written)
    with _COPIES_LOCK:
        _COPIES.setdefault(key, []).append(ref)
        _COPY_BYTES[0] += c.nbytes
    weakref.finalize(c, _copy_gone, key, ref, c.nbytes)
    return c


count(lambda: _COPY_BYTES[0], 'kept copies')


_ATOMS = {str, bytes, int, float, bool, complex, type(None)}
_NP = []


def nbytes(v):
    """Bytes of the arrays in v: an array, a tuple / list / dict of them,
    or an object holding them (fastplot's Lines, Polygons, Polylines ...),
    six levels down at most. (A kept drawing is measured each time one is
    kept: a loop that passes over its labels' numbers and strings cheaply.)"""
    if not _NP:
        import numpy as np
        _NP.append(np.ndarray)
    nd = _NP[0]
    atoms = _ATOMS
    total = 0
    stack = [(v, 0)]
    while stack:
        o, d = stack.pop()
        if isinstance(o, nd):
            b = o
            while isinstance(b.base, nd):
                b = b.base
            total += b.nbytes
            continue
        if d > 6 or type(o) in atoms:
            continue
        if isinstance(o, (tuple, list)):
            items = o
        elif isinstance(o, dict):
            items = o.values()
        else:
            dd = getattr(o, '__dict__', None)
            items = dd.values() if dd is not None else \
                [getattr(o, s, None) for s in getattr(type(o), '__slots__', ())]
        d += 1
        for u in items:
            if type(u) not in atoms:
                stack.append((u, d))
    return total


_measure = nbytes
