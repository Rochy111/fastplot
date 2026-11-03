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
small kernels has few gaps to fill: six panels of lines and text take 17
ms in a plain loop and 22 here. Threads, not processes: fastplot's data
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


def _splits(n):
    """The splits (workers, numba threads each) a batch of n tries: the
    default, then a thread a worker on the performance cores, on their
    second threads too, on every CPU allowed. Which is fastest depends on
    the machine and the figure: on an i7-13700K a GFS map drew 16% faster
    on 16 workers of one thread than on the default 4 of two."""
    t = _hw.topology()
    out = []
    for w, th in (_sizes(None, None), (t['perf_cores'], 1), (t['perf_logical'], 1), (_hw.cpus(), 1)):
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
    if workers is None and threads is None:
        tries = [_LEARNT[key]] if key in _LEARNT else _splits(n)
        if len(tries) > 1 and n < sum(w for w, _ in tries) + tries[0][0]:
            tries = tries[:1]                           # (too few items to try them on)
    else:
        tries = [_sizes(workers, threads)]
    if _jit.layer() == 'workqueue':
        tries = [(1, None)]                             # see the module
    results = [None] * n
    errors = []
    cv = threading.Condition(threading.Lock())
    st = dict(next=0, at=0, w=min(tries[0][0], n), t=tries[0][1], done=0, start=time.perf_counter(),
              rates=[])

    def finished():
        """One more frame drawn (holding cv): the try of a split ends when
        it has drawn as many frames as it has workers, and the next begins."""
        st['done'] += 1
        if st['at'] >= len(tries) or st['done'] < st['w']:
            return
        now = time.perf_counter()
        st['rates'].append(st['done'] / max(now - st['start'], 1e-9))
        st['at'] += 1
        if st['at'] < len(tries):
            w, t = tries[st['at']]
        else:
            w, t = tries[max(range(len(tries)), key=lambda i: st['rates'][i])]
            if len(tries) > 1:
                _LEARNT[key] = (w, t)
        st.update(w=min(w, n), t=t, done=0, start=now)
        cv.notify_all()

    def work(j):
        try:
            while True:
                with cv:
                    while not errors and st['next'] < n and j >= st['w']:
                        cv.wait()                       # (a worker this split does not use)
                    k = st['next']
                    if errors or k >= n:
                        cv.notify_all()
                        return
                    st['next'] = k + 1
                    t = st['t']
                try:
                    with _jit.threads(t):
                        results[k] = fn(items[k])
                except BaseException as e:              # noqa: BLE001 -- raised by frames()
                    with cv:
                        errors.append((k, e))
                        cv.notify_all()
                    return
                with cv:
                    finished()
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
    return results
