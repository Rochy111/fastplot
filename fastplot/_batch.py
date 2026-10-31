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

WORKERS frames at once (half the performance cores, at least 2) on
THREADS numba threads each (together the performance cores), unless
given; numba's workers are let go when the batch ends.

On numba's workqueue layer (pip's numba on Apple silicon has no other)
two threads' parallel regions at once abort the process: there the
frames are drawn one at a time, each on a draw's threads (install tbb,
or numba from conda-forge, which brings an OpenMP runtime, for more).
"""
import threading

from . import _hw, _jit


def _sizes(workers, threads):
    perf = _hw.topology()['perf_cores']
    w = max(2, perf // 2) if workers is None else max(1, int(workers))
    w = min(w, _hw.cpus())
    t = max(2, round(perf / w)) if threads is None else max(1, int(threads))
    return w, max(1, min(t, _hw.draw_threads()))


def frames(fn, items, workers=None, threads=None):
    """fn(item) for every item, several at once on threads (workers of
    them, each on `threads` of numba's threads: see the module). Returns
    the results in the items' order; the first exception raised stops the
    rest from starting and is raised here. fn should make its figures
    with fastplot.figure / subplots (not pyplot) and save or return them."""
    items = list(items)
    if not items:
        return []
    w, t = _sizes(workers, threads)
    w = min(w, len(items))
    if w > 1 and _jit.layer() == 'workqueue':
        w, t = 1, None                                  # see the module
    results = [None] * len(items)
    errors = []
    lock = threading.Lock()
    nxt = [0]

    def work():
        try:
            with _jit.threads(t):
                while True:
                    with lock:
                        k = nxt[0]
                        nxt[0] = k + 1
                    if k >= len(items) or errors:
                        return
                    try:
                        results[k] = fn(items[k])
                    except BaseException as e:          # noqa: BLE001 -- raised by frames()
                        with lock:
                            errors.append((k, e))
                        return
        finally:
            _jit.park()                                 # this thread's numba workers let go

    if w == 1:
        work()
    else:
        ts = [threading.Thread(target=work, name='fastplot-frame-%d' % i, daemon=True)
              for i in range(w)]
        for th in ts:
            th.start()
        for th in ts:
            th.join()
    if errors:
        raise min(errors, key=lambda e: e[0])[1]
    return results
