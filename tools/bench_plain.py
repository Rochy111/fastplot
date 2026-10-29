"""Plain-axes plots, matplotlib vs fastplot: warm frames (draw + PNG to memory),
median of 7, ms.  python tools/bench_plain.py mpl|fp"""
import io
import sys
import time

import numpy as np

rng = np.random.default_rng(0)
N = 100_000
x = np.linspace(0, 100, N)
y = np.sin(x) + 0.1 * rng.normal(size=N)
sx, sy = rng.normal(size=(2, 20_000))
grid = rng.normal(size=(500, 500)).cumsum(0).cumsum(1)
xx, yy = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-2, 2, 200))
zz = np.exp(-xx ** 2 - yy ** 2) + 0.4 * np.sin(3 * xx) * np.cos(2 * yy)
data = rng.normal(size=1_000_000)
img = rng.random((400, 600, 3))

CASES = {
    'line 100k pts': lambda ax: ax.plot(x, y, lw=0.8),
    'scatter 20k (cmap)': lambda ax: ax.scatter(sx, sy, c=sx * sy, s=6),
    'hist 1M, 100 bins': lambda ax: ax.hist(data, bins=100),
    'bar 50 + errorbar': lambda ax: ax.bar(np.arange(50), np.arange(50) % 7 + 1,
                                           yerr=0.5, capsize=2),
    'pcolormesh 500x500': lambda ax: ax.pcolormesh(grid, cmap='RdBu_r'),
    'contourf+contour 300x200': lambda ax: (ax.contourf(xx, yy, zz, levels=14),
                                            ax.contour(xx, yy, zz, levels=7, colors='k')),
    'imshow rgb 600x400': lambda ax: ax.imshow(img),
    'log-p fill_betweenx': lambda ax: (ax.fill_betweenx(np.linspace(1000, 100, 200),
                                                        np.linspace(20, -60, 200),
                                                        np.linspace(10, -70, 200)),
                                       ax.set_yscale('log'), ax.set_ylim(1050, 100)),
}


def frame(plt, fn):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    fn(ax)
    ax.set_title('t')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)


def bench(plt, fn, n=7):
    frame(plt, fn)                                      # warm: imports, kernels, caches
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        frame(plt, fn)
        ts.append(time.perf_counter() - t)
    return 1e3 * float(np.median(ts))


if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'mpl':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    else:
        import fastplot.pyplot as plt
    for name, fn in CASES.items():
        print('%-26s %8.1f' % (name, bench(plt, fn)), flush=True)
