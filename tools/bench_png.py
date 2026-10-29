"""fastplot._png against Pillow (matplotlib's PNG writer since 3.3: savefig
hands Agg's RGBA buffer to PIL with compress_level=6 unless told otherwise),
on maps rendered by matplotlib at 1800x975 with text and a colorbar:

    contourf      filled contours + contour lines, anti-aliased text: RGB
    contourf_pal  the same with nothing anti-aliased: 20 colours, indexed
    imshow        a smooth field through 'turbo', bilinear: RGB
    pcolormesh    nearest cells, viridis: RGB
    relief        shaded relief under a translucent field: continuous RGB

    python tools/bench_png.py               # the comparison table
    python tools/bench_png.py --levels      # filters x levels, size and time
    python tools/bench_png.py -n 50 --only imshow,contourf

Times are the median of -n runs back to back, after one to warm up
(--interleave: of the cases in turn); 'threads=1' is one thread.
"""
import argparse
import io
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fastplot import _png  # noqa: E402

NAMES = ('contourf', 'contourf_pal', 'imshow', 'pcolormesh', 'relief')


def field(seed=0, ny=181, nx=360):
    """A smooth random field, a sum of 40 waves."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:ny, 0:nx]
    f = np.zeros((ny, nx))
    for _ in range(40):
        kx, ky = rng.normal(0, 4, 2)
        f += rng.normal() * np.cos(2 * np.pi * (kx * x / nx + ky * y / ny) + rng.uniform(0, 6.3))
    return f


def render(kind):
    """An 1800x975 RGBA map, drawn by matplotlib (Agg)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f, g = field(0), field(3)
    aa = kind != 'contourf_pal'
    with plt.rc_context({'text.antialiased': aa, 'lines.antialiased': aa,
                         'patch.antialiased': aa}):
        fig = plt.figure(figsize=(18, 9.75), dpi=100)
        ax = fig.add_axes([0.04, 0.06, 0.84, 0.86])
        if kind.startswith('contourf'):
            m = ax.contourf(f, levels=np.linspace(f.min(), f.max(), 16), cmap='turbo',
                            antialiased=False)
            ax.contour(f, levels=m.levels[::2], colors='k', linewidths=0.8)
        elif kind == 'imshow':
            m = ax.imshow(f, cmap='turbo', origin='lower', aspect='auto', interpolation='bilinear')
        elif kind == 'pcolormesh':
            m = ax.pcolormesh(f, cmap='viridis')
        else:
            from matplotlib.colors import LightSource
            rgb = LightSource(315, 45).shade(g, plt.get_cmap('gist_earth'), vert_exag=30,
                                             blend_mode='soft')
            ax.imshow(rgb, origin='lower', aspect='auto', interpolation='bilinear')
            m = ax.imshow(f, cmap='turbo', alpha=0.45, origin='lower', aspect='auto',
                          interpolation='bilinear')
        ax.set_title('2 m temperature, GFS 0.25 deg  2026-10-28 00Z +024h', fontsize=16)
        ax.set_xlabel('longitude')
        ax.set_ylabel('latitude')
        fig.colorbar(m, cax=fig.add_axes([0.90, 0.06, 0.02, 0.86]), label='K')
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba()).copy()
        plt.close(fig)
    return img


def pil(img, level):
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(img).save(b, 'PNG', compress_level=level)
    return b.getvalue()


def timed(cases, n, interleave=False):
    """{name: median ms} of each fn() after a warm-up: n runs back to back
    (warm), or, interleaved, one run of each case in turn n times (so each
    encode follows 100-200 ms of the others' single-threaded work, its
    pool's threads idle meanwhile)."""
    for fn in cases.values():
        fn()
    ts = {k: [] for k in cases}
    for k, fn in cases.items() if not interleave else ():
        for _ in range(n):
            t = time.perf_counter()
            fn()
            ts[k].append(time.perf_counter() - t)
    for _ in range(n if interleave else 0):
        for k, fn in cases.items():
            t = time.perf_counter()
            fn()
            ts[k].append(time.perf_counter() - t)
    return {k: 1e3 * float(np.median(v)) for k, v in ts.items()}


def check(img, data):
    from PIL import Image
    back = np.asarray(Image.open(io.BytesIO(data)).convert('RGBA'))
    assert np.array_equal(back, img), 'round trip differs'


def table(imgs, n, interleave=False):
    print('%-13s %-26s %9s %9s  %s' % ('image', 'encoder', 'ms', 'bytes', 'png'))
    for name, img in imgs.items():
        ncol = len(np.unique(img.reshape(-1, 4).view(np.uint32)))
        cases = {
            'fastplot encode()': lambda: _png.encode(img),
            'fastplot encode(level=1)': lambda: _png.encode(img, level=1),
            'fastplot encode(threads=1)': lambda: _png.encode(img, threads=1),
            'PIL compress_level=6': lambda: pil(img, 6),
            'PIL compress_level=1': lambda: pil(img, 1),
        }
        if ncol <= 256:
            cases['fastplot palette=False'] = lambda: _png.encode(img, palette=False)
        ms = timed(cases, n, interleave)
        for k, fn in cases.items():
            data = fn()
            if k.startswith('fastplot'):
                check(img, data)
            kind = {0: 'gray', 2: 'RGB', 3: 'P', 4: 'GA', 6: 'RGBA'}[data[25]]
            print('%-13s %-26s %9.2f %9d  %s' % (name, k, ms[k], len(data), kind))
        print('%-13s (%d colours)' % ('', ncol))


def levels(imgs, n, interleave=False):
    print('%-13s %-9s %5s %9s %9s %9s' % ('image', 'filter', 'level', 'bytes', 'ms', 't1 ms'))
    for name, img in imgs.items():
        fts = ('none', 'sub', 'up', 'average', 'paeth', 'adaptive', 'auto')
        cases, single = {}, {}
        for ft in fts:
            for lv in (1, 2, 3, 6):
                cases[ft, lv] = (lambda ft=ft, lv=lv: _png._encode(img, level=lv, filter=ft))
                single[ft, lv] = (lambda ft=ft, lv=lv:
                                  _png._encode(img, level=lv, filter=ft, threads=1))
        ms = timed(cases, n, interleave)
        ms1 = timed(single, 3)
        for (ft, lv), fn in cases.items():
            print('%-13s %-9s %5d %9d %9.2f %9.1f' % (name, ft, lv, len(fn()), ms[ft, lv],
                                                      ms1[ft, lv]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('-n', type=int, default=30, help='runs per case (median)')
    ap.add_argument('--levels', action='store_true', help='filters x levels instead')
    ap.add_argument('--interleave', action='store_true', help='time the cases in turn')
    ap.add_argument('--only', help='images, comma-separated (of %s)' % ', '.join(NAMES))
    a = ap.parse_args()
    names = a.only.split(',') if a.only else NAMES
    imgs = {k: render(k) for k in names}
    print('%d logical CPUs; numba first call (compile or cache load) ...' % os.cpu_count())
    t = time.perf_counter()
    _png.encode(imgs[names[0]])
    print('  %.0f ms' % (1e3 * (time.perf_counter() - t)))
    (levels if a.levels else table)(imgs, a.n, a.interleave)


if __name__ == '__main__':
    main()
