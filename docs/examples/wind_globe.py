"""10 m wind speed on a globe, with streamlines tracing the flow."""
import os

import fastgrib
import numpy as np

import fastplot.crs as ccrs
import fastplot.pyplot as plt

GFS = 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20261027/00/atmos/gfs.t00z.pgrb2.0p25.f006'
grib = fastgrib.open(os.environ.get('GFS_FILE', GFS))

u, v = grib.winds(level='10 m above ground')
speed = np.hypot(u, v)
lat, lon = grib.latlon()
lat, lon = lat[:, 0], lon[0]
data = ccrs.PlateCarree()

fig = plt.figure(figsize=(8, 8), dpi=150, facecolor='#0e1117')
ax = fig.add_subplot(projection=ccrs.Orthographic(central_longitude=-40, central_latitude=25))
ax.set_global()

ax.pcolormesh(lon, lat, speed, cmap='magma', vmin=0, vmax=22, transform=data)
step = 8
polar = np.abs(lat) > 75                            # lon/lat lines crowd at the poles
u[polar], v[polar] = np.nan, np.nan
ax.streamplot(lon[::step], lat[::step], u[::step, ::step], v[::step, ::step], density=3,
              color='white', linewidth=0.35, arrowsize=0.6, transform=data)
ax.coastlines(linewidth=0.5, color='#9fb4c7')
ax.spines['geo'].set_edgecolor('#3a4250')
ax.set_title('10 m wind, GFS  06 UTC 27 Oct 2026', color='#d8dde6', fontsize=11)

fig.savefig(os.path.join(os.path.dirname(__file__), '..', 'images', 'wind_globe.png'),
            bbox_inches='tight', facecolor=fig.get_facecolor())
