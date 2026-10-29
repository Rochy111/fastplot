"""Quasi-geostrophic forcing for vertical motion against the model's own
vertical motion, at 700 hPa over North America and the western Atlantic.

(a) Q-vectors (arrows) and the forcing term of the QG omega equation,
    -2 div Q, computed from the GFS height and temperature at 700 hPa:

        Q = -(R/p) ( dVg/dx . grad T ,  dVg/dy . grad T ),

    with the geostrophic wind Vg = (g/f) k x grad Z. Q converges where
    balanced dynamics forces ascent (red) and diverges where it forces
    descent (blue). Heights and temperatures are smoothed to synoptic scales
    first, since QG theory describes the large-scale balanced flow.
(b) The GFS's own vertical motion at 700 hPa (omega, Pa s^-1; negative is
    upward). Where the two agree, the model's ascent is the balanced
    response to the large-scale flow; where they differ (convection, terrain,
    the tropics) something QG theory leaves out is at work.
"""
import os

import numpy as np

import fastplot.crs as ccrs
import fastplot.pyplot as plt
import gfs3d

grib = gfs3d.open_gfs()
lat, lon = gfs3d.axes(grib)
z, t, omega = (gfs3d.levels(grib, n, (700,))[0] for n in ('HGT', 'TMP', 'VVEL'))

z, t = gfs3d.smooth(z, 12), gfs3d.smooth(t, 12)
f = gfs3d.coriolis(lat) * np.ones_like(z)
f = np.where(np.abs(lat)[:, None] >= 15, f, np.nan)                   # no balance near the equator
ug, vg = -gfs3d.G / f * gfs3d.ddy(z, lat), gfs3d.G / f * gfs3d.ddx(z, lat, lon)
tx, ty = gfs3d.ddx(t, lat, lon), gfs3d.ddy(t, lat)
rp = 287.04 / 70000.0
q1 = -rp * (gfs3d.ddx(ug, lat, lon) * tx + gfs3d.ddx(vg, lat, lon) * ty)
q2 = -rp * (gfs3d.ddy(ug, lat) * tx + gfs3d.ddy(vg, lat) * ty)
forcing = gfs3d.smooth(-2 * (gfs3d.ddx(q1, lat, lon) + gfs3d.ddy(q2, lat)), 3) * 1e17
omega = gfs3d.smooth(omega, 2)

data = ccrs.PlateCarree()
proj = ccrs.LambertConformal(central_longitude=-85, central_latitude=42)
fig, axes = plt.subplots(1, 2, figsize=(15, 5.0), dpi=150, subplot_kw={'projection': proj})

for ax in axes:
    ax.set_extent([-125, -45, 25, 60], crs=data)
    ax.coastlines(linewidth=0.5, color='#333333')
    heights = ax.contour(lon, lat, z / 10, levels=np.arange(270, 330, 3), colors='black',
                         linewidths=0.6, transform=data)
    ax.clabel(heights, levels=heights.levels[::2], fmt='%d', fontsize=6)

ax = axes[0]
filled = ax.contourf(lon, lat, forcing, levels=[-3, -2, -1.5, -1, -0.5, -0.2, 0.2, 0.5, 1, 1.5, 2, 3],
                     cmap='RdBu_r', extend='both', transform=data)
step = 7
qx, qy = np.meshgrid(lon[::step], lat[::step])
sub = (slice(None, None, step), slice(None, None, step))
ax.quiver(qx, qy, q1[sub], q2[sub], scale=3e-11, width=0.0016, color='#333333',
          transform=data)
fig.colorbar(filled, ax=ax, orientation='horizontal', pad=0.04, shrink=0.85,
             label='-2 div Q (10$^{-17}$ K m$^{-2}$ s$^{-1}$), positive forces ascent')
ax.set_title('(a) QG forcing and Q-vectors at 700 hPa', loc='left', fontsize=10)

ax = axes[1]
filled = ax.contourf(lon, lat, omega, levels=[-1.2, -0.8, -0.5, -0.3, -0.15, -0.05, 0.05, 0.15,
                                              0.3, 0.5, 0.8, 1.2],
                     cmap='RdBu', extend='both', transform=data)
fig.colorbar(filled, ax=ax, orientation='horizontal', pad=0.04, shrink=0.85,
             label='Omega (Pa s$^{-1}$), negative is upward')
ax.set_title('(b) GFS vertical motion at 700 hPa', loc='left', fontsize=10)

fig.suptitle('Balanced forcing and model ascent, GFS 06 UTC 27 Oct 2026 (700 hPa height '
             'contoured every 30 m)', fontsize=12)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images',
                         'qg_forcing.png'), bbox_inches='tight')
