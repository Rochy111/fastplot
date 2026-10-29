"""fastplot.tune(): this machine's settings measured rather than guessed.

    fastplot.tune()             # ~2-5 s: times draws, keeps what it found
    fastplot.hardware()         # what fastplot found and uses

Two things depend on the machine more than a rule can tell (_hw):

The draw threads. A plot (a line, a scatter, bars: 800x500 px) and a map
(a 1440x721 field filled and contoured, coastlines, a colorbar: 1800x975
px) are written as PNG at a few thread counts, in rounds (each round
every count once, so a slow moment of the machine falls on all of them).
Each scene's median is set against its best; the count kept is the
fewest threads within TOLERANCE of the best mean -- a few percent of
speed is not worth more cores busy.

The GPU's threshold ('auto'). With a usable GPU the map is drawn on both
at a few sizes; the threshold is where the GPU starts to win (by WIN, and
at every bigger size), interpolated between the sizes tried, in work
(grid points + pixels) as _device counts it. A GPU that never wins: never
('auto' keeps to the CPU); one that always does: from the smallest size.

The results go to fastplot's cache directory (_hw.cache_dir), keyed by
the machine: another CPU, a different count of CPUs allowed (a container
resized) or another GPU read none of them. FASTPLOT_THREADS and
FASTPLOT_GPU_WORK still come first.
"""
import io
import math
import time

import numpy as np

from . import _hw, _jit

TOLERANCE = 0.03                # threads: the fewest within this of the best
WIN = 0.05                      # GPU: faster by this to count as winning
NEVER = 10 ** 15                # GPU work threshold: never


def _scenes():
    from . import crs, figure
    rng = np.random.default_rng(0)
    x = np.linspace(0, 100, 50_000)
    y = np.sin(x) + 0.1 * rng.normal(size=x.size)
    sx, sy = rng.normal(size=(2, 10_000))
    lon = np.linspace(-180, 179.75, 1440)
    lat = np.linspace(-90, 90, 721)
    z = (30 * np.cos(np.radians(lat))[:, None] - 10
         + 8 * np.sin(np.radians(3 * lon))[None, :] * np.cos(np.radians(2 * lat))[:, None]
         + rng.normal(size=(721, 1440)).cumsum(1) * 0.05).astype(np.float32)

    def plot(dpi=100, device='cpu'):
        fig = figure(figsize=(8, 5), dpi=dpi)
        ax = fig.add_subplot()
        ax.plot(x, y, lw=0.8)
        ax.scatter(sx * 15 + 50, sy, c=sx * sy, s=6)
        ax.bar(np.arange(0, 100, 4), np.arange(25) % 5 * 0.3 + 0.2, width=2, alpha=0.5)
        ax.set_title('tune')
        fig.savefig(io.BytesIO(), format='png', device=device)
        return fig._work(dpi)[0]

    def map_(dpi=150, device='cpu'):
        pc = crs.PlateCarree()
        fig = figure(figsize=(12, 6.5), dpi=dpi)
        ax = fig.add_subplot(projection=pc)
        cf = ax.contourf(lon, lat, z, levels=np.arange(-40, 41, 4), cmap='turbo', transform=pc)
        ax.contour(lon, lat, z, levels=np.arange(-40, 41, 20), colors='k', linewidths=0.4,
                   transform=pc)
        ax.coastlines(linewidth=0.5)
        fig.colorbar(cf, ax=ax, orientation='horizontal', shrink=0.8)
        ax.set_title('tune')
        fig.savefig(io.BytesIO(), format='png', device=device)
        return fig._work(dpi)[0]                    # the work _device.pick weighs

    return {'plot': plot, 'map': map_}


def _candidates():
    n = _hw.cpus()
    t = _hw.topology()
    c = {1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, t['perf_cores'], t['perf_logical'], t['cores'], n}
    return sorted(k for k in c if 1 <= k <= n)


def _timed(f, **kw):
    t = time.perf_counter()
    f(**kw)
    return time.perf_counter() - t


def _threads(scenes, rounds):
    counts = _candidates()
    times = {(s, n): [] for s in scenes for n in counts}
    cpu = {n: 0.0 for n in counts}
    saved = _hw._found.get('draw_threads')
    try:
        for n in counts:                                # every count's first frame, untimed
            _hw._found['draw_threads'] = n
            for f in scenes.values():
                f()
        for _ in range(rounds):
            for n in counts:
                _hw._found['draw_threads'] = n
                for s, f in scenes.items():
                    c0 = time.process_time()
                    times[s, n].append(_timed(f))
                    cpu[n] += time.process_time() - c0
    finally:
        if saved is None:
            _hw._found.pop('draw_threads', None)
        else:
            _hw._found['draw_threads'] = saved
    med = {k: float(np.median(v)) for k, v in times.items()}
    best = {s: min(med[s, n] for n in counts) for s in scenes}
    score = {n: float(np.mean([med[s, n] / best[s] for s in scenes])) for n in counts}
    top = min(score.values())
    pick = min(n for n in counts if score[n] <= top * (1 + TOLERANCE))
    table = {str(n): {'ms': {s: round(1e3 * med[s, n], 2) for s in scenes},
                      'cpu_ms_per_frame': round(1e3 * cpu[n] / (rounds * len(scenes)), 1),
                      'score': round(score[n], 3)} for n in counts}
    return pick, table


def _gpu(map_, rounds):
    from . import _device
    if not _device.gpu_usable():
        return None
    p = _hw.gpu_props() or {}
    rows = []
    for dpi in (50, 80, 120, 160, 220):
        for dev in ('cpu', 'gpu'):
            w = map_(dpi=dpi, device=dev)               # first frame at this size, untimed
        tc = float(np.median([_timed(map_, dpi=dpi, device='cpu') for _ in range(rounds)]))
        tg = float(np.median([_timed(map_, dpi=dpi, device='gpu') for _ in range(rounds)]))
        rows.append((w, tc, tg))
    wins = [tg < tc * (1 - WIN) for _, tc, tg in rows]
    if not any(wins):
        w = NEVER
    elif all(wins):
        w = rows[0][0]
    else:
        k = max(i for i, v in enumerate(wins) if not v) + 1   # wins from here on
        if k >= len(rows):
            w = NEVER
        else:
            (w0, c0, g0), (w1, c1, g1) = rows[k - 1], rows[k]
            r0, r1 = g0 / c0, g1 / c1                   # GPU / CPU time: where it crosses 1 - WIN
            f = 1.0 if r0 == r1 else min(1.0, max(0.0, (r0 - (1 - WIN)) / (r0 - r1)))
            w = int(math.exp(math.log(w0) + f * (math.log(w1) - math.log(w0))))
    return {'key': _hw.machine_key(p.get('name', '')), 'work': int(w),
            'measured': [{'work': a, 'cpu_ms': round(1e3 * b, 2), 'gpu_ms': round(1e3 * c, 2)}
                         for a, b, c in rows]}


def tune(save=True, rounds=5, gpu=True):
    """Time draws on this machine -- a plot and a map at a few thread
    counts, and (gpu, with a usable GPU) the map on CPU and GPU at a few
    sizes -- and keep the best settings for it (save): the draw threads
    and the work from which 'auto' draws on the GPU. Returns what it
    measured and chose. A few seconds; run it once a machine."""
    scenes = _scenes()
    t0 = time.perf_counter()
    for f in scenes.values():                           # kernels compiled / loaded first
        f()
    n, table = _threads(scenes, rounds)                 # (no threads() block round it: each
    _hw._found['draw_threads'] = n                      # draw sets its own count)
    out = {'threads': n, 'threads_measured': table}
    g = _gpu(scenes['map'], rounds) if gpu else None
    if g is not None:
        out['gpu_work'] = g['work']
        out['gpu_measured'] = g['measured']
    _hw._found.pop('draw_threads', None)
    _jit.park()
    if save:
        out['saved_to'] = _hw.save_tuned(
            cpu={'threads': n, 'when': time.strftime('%Y-%m-%d %H:%M:%S')}, gpu=g)
    out['seconds'] = round(time.perf_counter() - t0, 1)
    return out


def hardware():
    """What fastplot found on this machine and what it does with it: the
    CPU, the CPUs allowed, cores, the draw threads (and where that count
    came from), the GPU and 'auto''s threshold for it."""
    from . import _device
    out = _hw.info()
    g = _device.gpu_info() if _device.get_device() != 'cpu' else {'usable': None}
    if g.get('usable'):
        out['gpu'] = dict(_hw.gpu_props() or {}, auto_work=_hw.gpu_work())
    else:
        out['gpu'] = {'usable': g.get('usable'), 'reason': g.get('reason', '')}
    return out
