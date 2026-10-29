"""The dynamic tropopause: potential temperature and wind on the 2 PVU surface,
with cyclonic vorticity in the 925-850 hPa layer.

Ertel potential vorticity is computed on each of 23 pressure levels from the
GFS temperature and wind (isobaric form, hydrostatic):

    PV = -g [ (f + zeta) dtheta/dp - (dv/dp dtheta/dx - du/dp dtheta/dy) ]

Each column is then searched from the top down for the first level where PV
falls below 2 PVU, and potential temperature and wind are interpolated
linearly in PV to that crossing. Searching from the top skips the low-level
PV made by latent heating. Low potential temperature on the tropopause marks
troughs, where stratospheric air reaches down; high values mark ridges.
"""
import os

import numpy as np

import fastplot.crs as ccrs
import fastplot.pyplot as plt
import gfs3d

grib = gfs3d.open_gfs()
lat, lon = gfs3d.axes(grib)
p = np.array(gfs3d.LEVELS, np.float64)
t, u, v = (gfs3d.levels(grib, n) for n in ('TMP', 'UGRD', 'VGRD'))

th = gfs3d.theta(t, p)
zeta = gfs3d.vorticity(u, v, lat, lon)
pv = -gfs3d.G * ((gfs3d.coriolis(lat) + zeta) * gfs3d.ddp(th, p)
                 - (gfs3d.ddp(v, p) * gfs3d.ddx(th, lat, lon)
                    - gfs3d.ddp(u, p) * gfs3d.ddy(th, lat))) * 1e6          # PVU

# Top down: the first level below 2 PVU, and the one above it.
pv_top, th_top, u_top, v_top = (a[::-1] for a in (gfs3d.smooth(pv, 2), th, u, v))
below = pv_top < 2.0
k = np.argmax(below, axis=0)
found = below.any(axis=0) & (k > 0)
k = np.clip(k, 1, None)
take = lambda a, kk: np.take_along_axis(a, kk[None], axis=0)[0]
w = (2.0 - take(pv_top, k)) / (take(pv_top, k - 1) - take(pv_top, k))
dt = {name: np.where(found, take(a, k) + w * (take(a, k - 1) - take(a, k)), np.nan)
      for name, a in (('theta', th_top), ('u', u_top), ('v', v_top))}

low = slice(gfs3d.LEVELS.index(925), gfs3d.LEVELS.index(850) + 1)
zeta_low = gfs3d.smooth(zeta[low].mean(axis=0), 6) * 1e5                    # 10^-5 s^-1

data = ccrs.PlateCarree()
fig = plt.figure(figsize=(11, 7.6), dpi=150)
ax = fig.add_subplot(projection=ccrs.LambertConformal(central_longitude=-115,
                                                      central_latitude=45,
                                                      standard_parallels=(35, 60)))
ax.set_extent([-170, -62, 22, 68], crs=data)

filled = ax.contourf(lon, lat, dt['theta'], levels=np.arange(282, 375, 4), cmap='turbo',
                     extend='both', transform=data)
ax.contour(lon, lat, zeta_low, levels=np.arange(8, 41, 4), colors='black', linewidths=0.8,
           transform=data)
step = 14
kt = 1.94384
ax.barbs(lon[::step], lat[::step], dt['u'][::step, ::step] * kt, dt['v'][::step, ::step] * kt,
         length=5, linewidth=0.45, color='#222222', transform=data)
ax.coastlines(linewidth=0.8, color='#111111')
ax.gridlines(linewidth=0.25, color='#555555', linestyle='--')

fig.colorbar(filled, ax=ax, shrink=0.75, pad=0.02, label='Potential temperature on 2 PVU (K)')
ax.set_title('Dynamic tropopause, 925-850 hPa cyclonic vorticity', loc='left', fontsize=11)
ax.set_title('GFS  06 UTC 27 Oct 2026', loc='right', fontsize=9, color='#555555')
ax.text(0, -0.02, 'Black contours: 925-850 hPa relative vorticity from 8 x 10$^{-5}$ s$^{-1}$, '
        'every 4. Barbs: wind on the tropopause (kt).', fontsize=8, color='#444444',
        va='top', transform=ax.transAxes)

fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images',
                         'dynamic_tropopause.png'), bbox_inches='tight')
