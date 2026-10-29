"""fastplot against matplotlib + cartopy: the same map, timed both ways.

    python tools/bench.py GRIB_OR_NPY [--reps 7] [--out DIR] [--only NAME,...] [--device cpu|gpu]

The field is 2 m temperature from a GRIB2 file (read with fastgrib) or a
.npy beside lat.npy/lon.npy. Every case is drawn by matplotlib+cartopy
(Agg) and by fastplot at 1800x975 px and written as PNG; times are per
frame, warm (median of the reps after the first), and each tool's first
frame (cold: numba / cartopy caches loaded, nothing compiled) separately.
Both images are saved to --out for a side by side look.
"""
import argparse
import io
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CASES = ['pcolormesh', 'contourf', 'contour_clabel', 'map_pcolormesh', 'map_contourf',
         'map_full', 'lcc_conus', 'barbs']


def load(path):
    if path.endswith('.npy'):
        d = os.path.dirname(path)
        z = np.load(path)
        lat = np.load(os.path.join(d, 'lat.npy'))
        lon = np.load(os.path.join(d, 'lon.npy'))
    else:
        import fastgrib
        f = fastgrib.open(path)
        z = f.read(':TMP:2 m above ground:')
        la, lo = f.latlon()
        lat, lon = la[:, 0], lo[0]
    return lon.astype(np.float64), lat.astype(np.float64), (z - 273.15).astype(np.float32)


def draw(lib, case, lon, lat, z):
    """One frame of a case, to PNG bytes, by 'mpl' or 'fp'."""
    if lib == 'mpl':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        new = plt.figure
    else:
        import fastplot as plt                          # noqa: N813 -- the same calls
        import fastplot.crs as ccrs
        import fastplot.feature as cfeature
        new = plt.figure
    pc = ccrs.PlateCarree()
    levels = np.arange(-40, 46, 2)
    fig = new(figsize=(12, 6.5), dpi=150)
    if case == 'lcc_conus':
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(central_longitude=-97,
                                                                       central_latitude=38))
        ax.set_extent([-125, -66, 22, 51], crs=pc)
    else:
        ax = fig.add_subplot(1, 1, 1, projection=pc)
    kw = dict(transform=pc)
    if case in ('pcolormesh', 'map_pcolormesh'):
        m = ax.pcolormesh(lon, lat, z, cmap='turbo', vmin=-40, vmax=45, shading='auto', **kw)
    elif case in ('contourf', 'map_contourf', 'map_full', 'lcc_conus'):
        m = ax.contourf(lon, lat, z, levels=levels, cmap='turbo', extend='both', **kw)
    if case == 'contour_clabel':
        m = ax.contour(lon, lat, z, levels=levels[::5], colors='k', linewidths=0.6, **kw)
        ax.clabel(m, fontsize=7)
    if case in ('map_full', 'lcc_conus'):
        ax.contour(lon, lat, z, levels=levels[::5], colors='k', linewidths=0.4, **kw)
    if case == 'barbs':
        m = ax.pcolormesh(lon, lat, z, cmap='turbo', vmin=-40, vmax=45, shading='auto', **kw)
        s = slice(None, None, 24)
        u = 15 * np.cos(np.radians(lat[s]))[:, None] * np.ones(lon[s].size)
        v = 10 * np.sin(np.radians(lon[s]))[None, :] * np.ones((lat[s].size, 1))
        ax.barbs(lon[s], lat[s], u, v, length=5, linewidth=0.5, **kw)
    if case.startswith('map') or case in ('lcc_conus', 'barbs'):
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.gridlines(draw_labels=True, linewidth=0.2)
    if case == 'lcc_conus':
        ax.add_feature(cfeature.STATES, linewidth=0.2)
    if case != 'contour_clabel':
        fig.colorbar(m, ax=ax, orientation='horizontal', pad=0.06, shrink=0.8,
                     label='2 m temperature (C)')
    ax.set_title('GFS 0.25 deg  TMP 2 m  (%s)' % case)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    if lib == 'mpl':
        plt.close(fig)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('--reps', type=int, default=7)
    ap.add_argument('--out', default=None)
    ap.add_argument('--only', default=None)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--libs', default='mpl,fp')
    a = ap.parse_args()
    lon, lat, z = load(a.data)
    cases = a.only.split(',') if a.only else CASES
    import fastplot
    fastplot.set_device(a.device)
    rows = []
    for case in cases:
        row = [case]
        for lib in a.libs.split(','):
            reps = a.reps if not (lib == 'mpl' and case.startswith('lcc')) else max(2, a.reps // 3)
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                png = draw(lib, case, lon, lat, z)
                ts.append(time.perf_counter() - t0)
            if a.out:
                os.makedirs(a.out, exist_ok=True)
                with open(os.path.join(a.out, '%s_%s.png' % (case, lib)), 'wb') as fh:
                    fh.write(png)
            row += [ts[0] * 1e3, statistics.median(ts[1:]) * 1e3]
        rows.append(row)
        print(' '.join('%-16s' % c if isinstance(c, str) else '%9.1f' % c for c in row),
              flush=True)
    if 'mpl' in a.libs and 'fp' in a.libs:
        print('\n| case | matplotlib+cartopy (ms) | fastplot (ms) | speed-up |')
        print('|---|---:|---:|---:|')
        for r in rows:
            print('| %s | %.0f | %.1f | %.0fx |' % (r[0], r[2], r[4], r[2] / r[4]))


if __name__ == '__main__':
    main()
