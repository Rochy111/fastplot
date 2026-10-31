"""fastplot.warm(): every kernel compiled once a machine, several at a time.

    fastplot.warm()                 # or: python -m fastplot warm

numba compiles a kernel the first time it runs with given argument types
and keeps the machine code in its cache, so a new machine's first map
takes ~15 s and the rest of the kernels ~20 s more, on whatever figure
first needs them. warm() draws small figures of every kind (RECIPES)
so that nothing is left to compile later -- and since numba compiles
holding the GIL, one kernel at a time a process, it compiles in several
processes at once, each writing what it compiled to the shared cache
(numba writes each file whole, under a new name, then renames it over
the old: two processes compiling one kernel at worst lose one's entry,
compiled again later).

What each process compiles: data/warm_manifest.pkl lists every kernel
the recipes reach, its argument types and how long it took to compile
(tools/make_warm_manifest.py writes it); the kernels are dealt out to
the processes longest first, each to the least loaded, so they finish
together -- ~4x sooner than one process on 8 cores. Without a manifest
for this numba (or with one gone stale) the processes run the recipes
instead, one each. Then this process runs the recipes itself, loading
everything from the cache and compiling whatever is still missing. On a
cache already full, warm() takes about two seconds.
"""
import importlib
import pickle
import io
import os
import subprocess
import sys
import time

import numpy as np

# each recipe: figures of one kind -- kernels that kind uses (the common
# ones -- text, lines, fills, PNG -- come with every recipe)


def _save(fig, **kw):
    fig.savefig(io.BytesIO(), format='png', **kw)


def map_fields():
    from . import crs, figure
    pc = crs.PlateCarree()
    lon = np.linspace(0, 359, 72)
    lat = np.linspace(-90, 90, 37)
    z = (np.cos(np.radians(lat))[:, None] * np.sin(np.radians(lon))[None, :]).astype(np.float32)
    for proj in (pc, crs.LambertConformal(central_longitude=-97, central_latitude=38),
                 crs.Robinson(), crs.Orthographic(-40, 40), crs.NorthPolarStereo()):
        fig = figure(figsize=(3, 2), dpi=60)
        ax = fig.add_subplot(projection=proj)
        cf = ax.contourf(lon, lat, z, levels=8, extend='both', transform=pc)
        ax.pcolormesh(lon, lat, z, transform=pc, alpha=0.5)
        cs = ax.contour(lon, lat, z, levels=5, colors='k', transform=pc)
        ax.clabel(cs)
        ax.imshow(z, extent=(0, 359, -90, 90), transform=pc, alpha=0.3)
        fig.colorbar(cf, ax=ax)
        _save(fig)
        fig.draw(device='cpu')


def map_vectors():
    from . import crs, feature, figure
    pc = crs.PlateCarree()
    lon = np.linspace(-120, -60, 25)
    lat = np.linspace(20, 55, 15)
    u = 15 * np.cos(np.radians(lat))[:, None] * np.ones(lon.size)
    v = 10 * np.sin(np.radians(lon))[None, :] * np.ones((lat.size, 1))
    fig = figure(figsize=(3, 2), dpi=60)
    ax = fig.add_subplot(projection=crs.LambertConformal(central_longitude=-97,
                                                          central_latitude=38))
    ax.set_extent([-120, -70, 22, 52], crs=pc)
    ax.barbs(lon[::3], lat[::3], u[::3, ::3], v[::3, ::3], transform=pc)
    ax.quiver(lon[::4], lat[::4], u[::4, ::4], v[::4, ::4], transform=pc)
    ax.streamplot(lon, lat, u, v, transform=pc)
    ax.scatter(lon[::5], np.full(5, 40.0), c=lon[::5], transform=pc)
    ax.plot(lon, 30 + 5 * np.sin(lon / 10), transform=pc)
    ax.text(-100, 45, 'warm', transform=pc)
    ax.coastlines()
    ax.add_feature(feature.BORDERS)
    ax.add_feature(feature.STATES)
    ax.add_feature(feature.LAND, facecolor='0.9')
    ax.gridlines(draw_labels=True)
    ax.set_title('warm')
    _save(fig)


def plain_lines():
    from . import pyplot as plt
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 9000)
    fig, axs = plt.subplots(2, 2, figsize=(4, 3), dpi=60)
    a = axs[0, 0]
    a.plot(x, np.sin(x), 'o-', markevery=500, label='a')
    a.plot(x, np.cos(x), '--', lw=2, label='b')
    a.scatter(x[::300], np.sin(x[::300]), c=x[::300], s=20, marker='^')
    a.fill_between(x, np.sin(x), 0, alpha=0.3, hatch='//')
    a.legend(loc='best')
    a.annotate('peak', (1.5, 1), (3, 0.5), arrowprops=dict(arrowstyle='->'))
    a.text(5, 0, 'box', bbox=dict(boxstyle='round'))
    b = axs[0, 1]
    b.bar(np.arange(10), rng.random(10), yerr=0.1, capsize=2, hatch='x')
    b.barh(np.arange(5), rng.random(5), left=1)
    b.errorbar(np.arange(10), rng.random(10), xerr=0.2, fmt='s')
    b.axhline(0.5)
    b.axvspan(2, 3, alpha=0.2)
    c = axs[1, 0]
    c.hist(rng.normal(size=10000), bins=40)
    c.hist(rng.normal(size=(3000, 2)), bins=20, histtype='step', stacked=True)
    c.step(np.arange(10), rng.random(10))
    c.stem(np.arange(10), rng.random(10))
    c.set_yscale('log')
    d = axs[1, 1]
    d.stackplot(np.arange(10), rng.random((3, 10)))
    d.semilogx(np.logspace(0, 3, 50), np.arange(50))
    d.grid(True)
    fig.suptitle('warm')
    fig.tight_layout()
    _save(fig)
    _save(fig, bbox_inches='tight', transparent=True)
    plt.close('all')


def plain_fields():
    from . import pyplot as plt
    rng = np.random.default_rng(1)
    xx, yy = np.meshgrid(np.linspace(-3, 3, 60), np.linspace(-2, 2, 40))
    zz = np.exp(-xx ** 2 - yy ** 2)
    fig, axs = plt.subplots(2, 2, figsize=(4, 3), dpi=60)
    axs[0, 0].pcolormesh(zz, cmap='RdBu_r')
    axs[0, 0].pcolormesh(xx, yy, zz, shading='gouraud', alpha=0.5)
    cf = axs[0, 1].contourf(xx, yy, zz, levels=8, hatches=['', '/'])
    cs = axs[0, 1].contour(xx, yy, zz, levels=4, colors='k')
    axs[0, 1].clabel(cs)
    fig.colorbar(cf, ax=axs[0, 1])
    axs[1, 0].imshow(rng.random((20, 30, 3)))
    axs[1, 0].imshow(zz, alpha=0.5, interpolation='bilinear')
    axs[1, 1].fill_betweenx(np.linspace(1000, 100, 50), np.linspace(20, -60, 50),
                            np.linspace(10, -70, 50))
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_ylim(1050, 100)
    _save(fig)
    plt.close('all')


def special():
    from . import pyplot as plt
    from . import tri
    from .skewt import Hodograph, SkewT
    rng = np.random.default_rng(2)
    x, y = rng.uniform(-3, 3, 200), rng.uniform(-2, 2, 200)
    z = np.exp(-x ** 2 - y ** 2)
    t = tri.Triangulation(x, y)
    fig, axs = plt.subplots(1, 3, figsize=(5, 2), dpi=60)
    axs[0].tricontourf(t, z, levels=6)
    cs = axs[0].tricontour(t, z, levels=4, colors='k')
    axs[0].clabel(cs)
    axs[1].tripcolor(t, z, shading='gouraud')
    axs[1].triplot(t, lw=0.3)
    h = Hodograph(axs[2], component_range=40)
    h.add_grid(increment=10)
    u, v = 20 * np.sin(np.linspace(0, 3, 20)), 15 * np.cos(np.linspace(0, 3, 20))
    h.plot_colormapped(u, v, np.hypot(u, v))
    _save(fig)
    fig = plt.figure(figsize=(3, 3), dpi=60)
    skew = SkewT(fig, rotation=45)
    p = np.linspace(1000, 100, 20)
    T = np.linspace(25, -60, 20)
    skew.plot(p, T, 'r')
    skew.plot_barbs(p, np.linspace(0, 30, 20), np.linspace(5, -5, 20))
    skew.shade_cape(p, T, T - 2)
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()
    _save(fig)
    fig = plt.figure(figsize=(3, 3), dpi=60)
    ax = fig.add_subplot(projection='polar')
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(th, 1 + 0.5 * np.sin(3 * th))
    ax.bar(th[::10], np.ones(10), width=0.3, alpha=0.4)
    _save(fig)
    for fmt in ('pdf', 'svg'):
        fig.savefig(io.BytesIO(), format=fmt)
    plt.close('all')


def large_grids():
    """What only what is too large for a recipe runs, and markers by number."""
    from . import _raster, figure
    a = np.zeros((2, 2), np.uint64)
    _raster._bits_differ(a, a)
    z = np.zeros(4, np.float32)
    _raster._not_finite(z)
    _raster._take_cells(z, np.zeros((2, 2), np.int32), np.empty((2, 2), np.float32))
    for n in (3, 40):                                   # one thread, and several
        fig = figure(figsize=(2, 2), dpi=60)
        ax = fig.add_subplot()
        ax.scatter(np.arange(n), np.arange(n), s=np.arange(n) + 4.0, c=np.arange(n))
        ax.scatter(np.arange(n), np.arange(n), s=np.arange(n) + 4.0, marker='x', c=np.arange(n))
        _save(fig)


RECIPES = (map_fields, map_vectors, plain_lines, plain_fields, special, large_grids)


def run(name=None):
    """The recipes (or the one named) in this process."""
    for r in RECIPES:
        if name is None or r.__name__ == name:
            r()


MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'warm_manifest.pkl')


def _manifest():
    """The manifest's entries (module, name, signatures, seconds), or None
    (none, or written by another numba)."""
    try:
        import numba
        with open(MANIFEST, 'rb') as f:
            d = pickle.load(f)
        return d['entries'] if d.get('numba') == numba.__version__ else None
    except Exception:                                   # noqa: BLE001 -- the recipes, then
        return None


def _shares(entries, n):
    """entries dealt into n lists, longest first, each to the least loaded."""
    load, out = [0.0] * n, [[] for _ in range(n)]
    for e in sorted(entries, key=lambda e: -max(e[3], 0.01)):
        k = load.index(min(load))
        out[k].append(e)
        load[k] += max(e[3], 0.01)
    return out


def compile_share(k, n):
    """Share k of n of the manifest's kernels compiled (into the cache)."""
    from . import _jit
    entries = _manifest()
    if entries is None:
        return
    _jit.load()
    for mod, name, sigs, _ in _shares(entries, n)[k]:
        try:
            disp = getattr(importlib.import_module(mod), name)
        except Exception:                               # noqa: BLE001 -- gone: skipped
            continue
        for sig in sigs:
            try:
                disp.compile(sig)
            except Exception:                           # noqa: BLE001 -- the recipes make it
                pass


def warm(processes=None, verbose=False):
    """Compile (or load from numba's cache) every CPU kernel, by drawing
    small figures of every kind -- for a job's set-up, or once after
    installing, so that no real frame is the one that pays. processes:
    how many recipes compile at once in their own processes (None: as
    many as there are recipes and CPUs; 1: all here, one after another).
    Returns the seconds it took."""
    from . import _bundle, _hw, _jit
    t0 = time.perf_counter()
    manifest = _manifest() is not None
    most = _hw.topology()['perf_cores'] if manifest else len(RECIPES)
    n = max(1, min(most, _hw.cpus())) if processes is None else max(1, int(processes))
    if n > 1:
        env = dict(os.environ, FASTPLOT_PRELOAD='0', FASTPLOT_DEVICE='cpu')
        env.setdefault('NUMBA_NUM_THREADS', str(max(1, min(4, _hw.cpus() // n))))
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if manifest:
            pending = ['_warm.compile_share(%d, %d)' % (k, n) for k in range(n)]
        else:
            pending = ['_warm.run(%r)' % r.__name__ for r in RECIPES]
        running = []
        while pending or running:
            while pending and len(running) < n:
                name = pending.pop(0)
                code = 'from fastplot import _warm; ' + name
                running.append((name, subprocess.Popen(
                    [sys.executable, '-c', code], env=env, cwd=root,
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)))
            name, p = running.pop(0)
            err = p.communicate()[1]
            if p.returncode and verbose:                # this process runs it again below
                print('fastplot.warm: %s failed in its process:\n%s' % (name, err.decode()))
            elif verbose:
                print('fastplot.warm: %s compiled (%.1f s)' % (name, time.perf_counter() - t0))
    with _jit.threads(park=True):
        run()                                           # here: loaded from the cache
    try:
        _bundle.write()
    except OSError:                                     # a read-only install: imports as usual
        pass
    if verbose:
        print('fastplot.warm: done in %.1f s' % (time.perf_counter() - t0))
    return time.perf_counter() - t0
