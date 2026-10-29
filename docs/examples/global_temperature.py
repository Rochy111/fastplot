"""Global 2 m temperature from GFS, with the freezing line picked out."""
import os

import fastgrib
import numpy as np

import fastplot.crs as ccrs
import fastplot.feature as cfeature
import fastplot.pyplot as plt

GFS = 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20261027/00/atmos/gfs.t00z.pgrb2.0p25.f006'
grib = fastgrib.open(os.environ.get('GFS_FILE', GFS))

t2m = grib.read('TMP:2 m above ground') - 273.15
lat, lon = grib.latlon()
lat, lon = lat[:, 0], lon[0]

fig = plt.figure(figsize=(11, 6.2), dpi=150)
ax = fig.add_subplot(projection=ccrs.Robinson(central_longitude=-150))
data = ccrs.PlateCarree()

shaded = ax.contourf(lon, lat, t2m, levels=np.arange(-44, 45, 2), cmap='RdYlBu_r',
                     extend='both', transform=data)
ax.contour(lon, lat, t2m, levels=[0], colors='#1f3a93', linewidths=1.1, transform=data)
ax.coastlines(linewidth=0.5, color='#333333')
ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor='#555555')
ax.gridlines(linewidth=0.3, color='white', alpha=0.6)
ax.set_global()

cb = fig.colorbar(shaded, ax=ax, orientation='horizontal', shrink=0.6, pad=0.04, aspect=40)
cb.set_label('Temperature at 2 m (°C)')
ax.set_title('GFS 0.25°  2 m temperature, 0 °C line in blue', loc='left', fontsize=11)
ax.set_title('Valid 06 UTC 27 Oct 2026', loc='right', fontsize=9, color='#555555')

fig.savefig(os.path.join(os.path.dirname(__file__), '..', 'images', 'global_temperature.png'),
            bbox_inches='tight')
