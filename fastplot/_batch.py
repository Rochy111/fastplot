"""fastplot.frames(): many figures drawn at once, in one process.

    def hour(h):
        fig = fastplot.figure(figsize=(12, 6.5), dpi=150)
        ...                                        # fastplot.figure, not pyplot (its state is global)
        fig.savefig('t2m_%03d.png' % h)

    fastplot.frames(hour, range(0, 385, 3))         # the results of hour, in order

A frame is Python (building the figure, laying out text) between numba
kernels, and zlib's deflate at the end; the kernels and deflate let go of
the GIL. Several frames drawn on threads at once fill each other's gaps:
on an 8-P-core i7-13700K, a Lambert map of an HRRR field went from 44 to
26 ms a frame and a GFS map from 13.5 to 12.4 (4 frames at once, 2 numba
threads each), no process started, nothing copied, the images byte for
byte those drawn one at a time. A figure that is mostly Python between
small kernels has few gaps to fill: six panels of lines and text take 14.2
ms in a plain loop and 15.4 four at a time, so they are drawn one at a
time once that is found. Threads, not processes: fastplot's data
(features, fonts, colormaps, the map tables) is loaded once and shared.

How many frames at once, on how many numba threads each, is found as the
batch goes (_splits, frames), unless given; numba's workers are let go
when the batch ends.

On numba's workqueue layer (pip's numba on Apple silicon has no other)
two threads' parallel regions at once abort the process: there the
frames are drawn one at a time, each on a draw's threads (install tbb,
or numba from conda-forge, which brings an OpenMP runtime, for more).
"""
import threading
import time

from . import _hw, _jit


def _sizes(workers, threads):
    perf = _hw.topology()['perf_cores']
    w = max(2, perf // 2) if workers is None else max(1, int(workers))
    w = min(w, _hw.cpus())
    t = max(2, round(perf / w)) if threads is None else max(1, int(threads))
    return w, max(1, min(t, _hw.draw_threads()))


_LEARNT = {}                    # fn's code -> the split a batch of it drew fastest
STORE_FORMAT = 1                # frames.json's layout: another is ignored
SLOWER = 0.6                    # a stored split drawing below this share of its rate is learnt again
                                # (its rate the best of a few tries': a changed machine or data
                                # shows as far less)
STORE_MOST = 64                 # functions frames.json keeps, the latest learnt
_store = {}                     # frames.json as read: {code key: [workers, threads, frames a second]}
_store_lock = threading.Lock()


def _code_key(fn):
    """fn as frames.json knows it from one process to the next: its file,
    name, first line and a hash of its code (changed code learns again);
    None for a callable with no code of its own."""
    code = getattr(fn, '__code__', None)
    if code is None:
        return None
    import hashlib
    h = hashlib.sha1(code.co_code)
    h.update(repr((code.co_names, code.co_varnames)).encode())
    return '%s:%s:%d:%s' % (code.co_filename, getattr(fn, '__qualname__', code.co_name),
                            code.co_firstlineno, h.hexdigest()[:16])


def _store_path():
    import os
    return os.path.join(_hw.cache_dir(), 'frames.json')


def _stored():
    """This machine's splits learnt by earlier processes (read once)."""
    with _store_lock:
        if 'splits' not in _store:
            _store['splits'] = {}
            try:
                import json
                with open(_store_path()) as f:
                    d = json.load(f)
                if d.get('format') == STORE_FORMAT and d.get('key') == _hw.machine_key():
                    _store['splits'] = dict(d.get('splits', {}))
            except (OSError, ValueError, AttributeError):
                pass
        return _store['splits']


def _keep(ckey, entry):
    """frames.json with ckey's split set (entry) or dropped (None), written
    whole and atomically; nothing if the cache cannot be written."""
    import json
    import os
    splits = _stored()
    with _store_lock:
        splits.pop(ckey, None)
        if entry is not None:
            splits[ckey] = entry                        # (last: the most recently learnt)
            for old in list(splits)[:-STORE_MOST]:
                del splits[old]
        path = _store_path()
        tmp = '%s.%d.tmp' % (path, os.getpid())
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(tmp, 'w') as f:
                json.dump({'format': STORE_FORMAT, 'key': _hw.machine_key(), 'splits': splits}, f)
            os.replace(tmp, path)
        except OSError:
            pass


def _TRY(w):
    """Frames a split of w workers is tried on."""
    return max(4, w)


def _splits(n):
    """The splits (workers, numba threads each) a batch of n tries: the
    default, then a thread a worker on the performance cores, on their
    second threads too, on every CPU allowed, and last one at a time, as a
    loop draws them. Which is fastest depends on the machine and the
    figure: on an i7-13700K a GFS map drew 16% faster on 16 workers of one
    thread than on the default 4 of two, and a six-panel plot of lines and
    text, mostly Python, fastest one at a time."""
    t = _hw.topology()
    out = []
    for w, th in (_sizes(None, None), (t['perf_cores'], 1), (t['perf_logical'], 1), (_hw.cpus(), 1),
                  (1, _hw.draw_threads())):
        w = max(1, min(w, _hw.cpus(), n))
        if (w, th) not in out:
            out.append((w, th))
    return out


def frames(fn, items, workers=None, threads=None):
    """fn(item) for every item, several at once on threads (workers of
    them, each on `threads` of numba's threads: see the module). Returns
    the results in the items' order; the first exception raised stops the
    rest from starting and is raised here. fn should make its figures
    with fastplot.figure / subplots (not pyplot) and save or return them.
    With neither count given, the first items are drawn a few ways
    (_splits), each until it has drawn as many frames as it has workers,
    and the rest the way that drew most a second -- for this process's
    later batches of fn too."""
    items = list(items)
    n = len(items)
    if not n:
        return []
    key = getattr(fn, '__code__', fn)
    ckey = stored = None
    if workers is None and threads is None:
        if key in _LEARNT:
            tries = [_LEARNT[key]]
        else:
            ckey = _code_key(fn)
            stored = _stored().get(ckey) if ckey is not None else None
            tries = [(int(stored[0]), int(stored[1]))] if stored else _splits(n)   # (an earlier process's)
        if len(tries) > 1 and n < sum(_TRY(w) for w, _ in tries) + tries[0][0]:
            tries = tries[:1]                           # (too few items to try them on)
    else:
        tries = [_sizes(workers, threads)]
    if _jit.layer() == 'workqueue':
        tries = [(1, None)]                             # see the module
    results = [None] * n
    errors = []
    cv = threading.Condition(threading.Lock())
    st = dict(next=0, at=0, w=min(tries[0][0], n), t=tries[0][1], done=0, start=None, rates=[],
              busy=0, phase=0, drain=False)

    def finished(phase):
        """A frame of the try `phase` drawn (holding cv). A try counts its
        own frames only: once it has drawn as many as it has workers (4 at
        least), no more are begun until those in hand are done, and the next
        try starts from an idle batch -- frames of the last one would run on
        into it and blur both."""
        if st['at'] >= len(tries):
            return
        if phase == st['phase'] and not st['drain']:
            st['done'] += 1
            if st['done'] >= _TRY(st['w']):
                st['rates'].append(st['done'] / max(time.perf_counter() - st['start'], 1e-9))
                if len(tries) == 1:                     # (its rate known; nothing after it to try)
                    st['at'] = 1
                    return
                st['drain'] = True
        if st['drain'] and not st['busy']:
            st['at'] += 1
            if st['at'] < len(tries):
                w, t = tries[st['at']]
            else:
                w, t = tries[max(range(len(tries)), key=lambda i: st['rates'][i])]
                if len(tries) > 1:
                    _LEARNT[key] = (w, t)
            st.update(w=min(w, n), t=t, done=0, start=None, drain=False, phase=st['phase'] + 1)
            cv.notify_all()

    def work(j):
        try:
            while True:
                with cv:
                    while not errors and st['next'] < n and (j >= st['w'] or st['drain']):
                        cv.wait()                       # (a worker this split does not use)
                    k = st['next']
                    if errors or k >= n:
                        cv.notify_all()
                        return
                    st['next'] = k + 1
                    t, phase = st['t'], st['phase']
                    st['busy'] += 1
                    if st['start'] is None:
                        st['start'] = time.perf_counter()
                try:
                    with _jit.threads(t):
                        results[k] = fn(items[k])
                except BaseException as e:              # noqa: BLE001 -- raised by frames()
                    with cv:
                        st['busy'] -= 1
                        errors.append((k, e))
                        cv.notify_all()
                    return
                with cv:
                    st['busy'] -= 1
                    finished(phase)
        finally:
            _jit.park()                                 # this thread's numba workers let go

    most = min(max(w for w, _ in tries), n)
    if most == 1:
        work(0)
    else:
        ts = [threading.Thread(target=work, args=(i,), name='fastplot-frame-%d' % i, daemon=True)
              for i in range(most)]
        for th in ts:
            th.start()
        for th in ts:
            th.join()
    if errors:
        raise min(errors, key=lambda e: e[0])[1]
    if ckey is not None and st['rates']:                # for this machine's next processes
        if len(tries) > 1 and key in _LEARNT:
            _keep(ckey, list(_LEARNT[key]) + [max(st['rates'])])
        elif stored is not None:
            if st['rates'][0] < SLOWER * stored[2]:     # (the data or the machine has changed)
                _keep(ckey, None)
            else:
                _LEARNT[key] = tries[0]
    return results
