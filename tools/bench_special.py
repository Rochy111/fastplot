"""The speed page's Skew-T, streamplot, hodograph and triangulation rows,
matplotlib (with MetPy) vs fastplot: warm frames (figure made, drawn, PNG to
memory), median of 9, ms.  python tools/bench_special.py mpl|fp"""
import io
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastplot.skewt import dry_lapse, moist_lapse  # noqa: E402  (the data, not timed)

MPL = len(sys.argv) > 1 and sys.argv[1] == 'mpl'
if MPL:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from metpy.plots import Hodograph, SkewT
    from metpy.units import units

    def q(a, unit):
        return a * units(unit)
else:
    import fastplot.pyplot as plt
    import fastplot.tri as mtri
    from fastplot.skewt import Hodograph, SkewT

    def q(a, unit):
        return a

p = np.linspace(1000, 100, 60)
T = 25 - 0.06 * (1000 - p)
Td = T - np.linspace(3, 25, p.size)
parcel = np.where(p >= 850, dry_lapse(p, 298.15, 1000.0), moist_lapse(p, 291.0, 850.0)) - 273.15
u = np.linspace(0, 60, p.size)
v = 20 * np.sin(np.linspace(0, 3, p.size))
gy, gx = np.mgrid[-2:2:60j, -3:3:90j]
su, sv = -1 - gx ** 2 + gy, 1 + gx - gy ** 2
rng = np.random.default_rng(3)
tx, ty = rng.uniform(-1, 1, (2, 400))
tz = np.cos(3 * tx) * np.sin(2 * ty)


def skewt():
    fig = plt.figure(figsize=(7, 7), dpi=100)
    skew = SkewT(fig, rotation=45)
    P, TT, TD, PA = q(p, 'hPa'), q(T, 'degC'), q(Td, 'degC'), q(parcel, 'degC')
    skew.plot(P, TT, 'r')
    skew.plot(P, TD, 'g')
    skew.plot(P, PA, 'k--')
    skew.plot_barbs(P[::3], q(u[::3], 'knot'), q(v[::3], 'knot'))
    skew.shade_cape(P, TT, PA)
    skew.shade_cin(P, TT, PA, TD)
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()
    skew.ax.set_xlim(-40, 50)
    return fig


def streams():
    fig, axs = plt.subplots(1, 3, figsize=(8, 5), dpi=100)
    axs[0].streamplot(gx, gy, su, sv)
    axs[1].streamplot(gx, gy, su, sv, color=np.hypot(su, sv), cmap='viridis')
    axs[2].streamplot(gx, gy, su, sv, density=2, linewidth=0.7)
    return fig


def hodographs():
    fig, axs = plt.subplots(1, 2, figsize=(8, 5), dpi=100)
    for ax in axs:
        h = Hodograph(ax, component_range=80)
        h.add_grid(increment=20)
        h.plot_colormapped(u, v, p, cmap='viridis')
        h.wind_vectors(u[::6], v[::6])
    return fig


def tri():
    t = mtri.Triangulation(tx, ty)
    fig, axs = plt.subplots(2, 3, figsize=(8, 5), dpi=100)
    axs[0, 0].tricontourf(t, tz, levels=10)
    cs = axs[0, 1].tricontour(t, tz, levels=6, colors='k')
    axs[0, 1].clabel(cs, fontsize=6)
    axs[0, 2].tripcolor(t, tz)
    axs[1, 0].tripcolor(t, tz, shading='gouraud')
    axs[1, 1].triplot(t, lw=0.4)
    return fig


def frame(make):
    fig = make()
    fig.savefig(io.BytesIO(), format='png')
    plt.close(fig)


if __name__ == '__main__':
    for name, make in (('Skew-T', skewt), ('streamplot x3', streams), ('hodographs x2', hodographs),
                       ('tri x5', tri)):
        frame(make)                                     # warm: imports, kernels, caches
        ts = []
        for _ in range(9):
            t = time.perf_counter()
            frame(make)
            ts.append(time.perf_counter() - t)
        print('%-16s %8.1f' % (name, 1e3 * float(np.median(ts))), flush=True)
