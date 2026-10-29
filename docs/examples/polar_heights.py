"""500 hPa heights seen from above the North Pole: the troughs and ridges of
the jet stream's wave pattern."""
import os

import fastgrib
import numpy as np

import fastplot.crs as ccrs
import fastplot.pyplot as plt

GFS = 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20261027/00/atmos/gfs.t00z.pgrb2.0p25.f006'
grib = fastgrib.open(os.environ.get('GFS_FILE', GFS))

z500 = grib.read('HGT:500 mb') / 10.0                # decametres
lat, lon = grib.latlon()
lat, lon = lat[:, 0], lon[0]
data = ccrs.PlateCarree()

fig = plt.figure(figsize=(8, 8.4), dpi=150)
ax = fig.add_subplot(projection=ccrs.NorthPolarStereo(central_longitude=-100))
ax.set_extent([-180, 180, 22, 90], crs=data)

shaded = ax.contourf(lon, lat, z500, levels=np.arange(486, 600, 6), cmap='Spectral_r',
                     extend='both', transform=data)
lines = ax.contour(lon, lat, z500, levels=np.arange(486, 600, 6), colors='k',
                   linewidths=0.5, transform=data)
ax.clabel(lines, levels=lines.levels[::2], fmt='%d', fontsize=7)
ax.coastlines(linewidth=0.5, color='#333333')
ax.gridlines(linewidth=0.3, color='#666666', linestyle='--')

cb = fig.colorbar(shaded, ax=ax, orientation='horizontal', shrink=0.75, pad=0.03, aspect=40)
cb.set_label('500 hPa geopotential height (dam)')
ax.set_title('500 hPa heights', loc='left', fontsize=11)
ax.set_title('GFS  valid 06 UTC 27 Oct 2026', loc='right', fontsize=9, color='#555555')

fig.savefig(os.path.join(os.path.dirname(__file__), '..', 'images', 'polar_heights.png'),
            bbox_inches='tight')
