# fastplot

fastplot draws weather maps and scientific plots using the matplotlib and
cartopy API you already know. Instead of building a tree of Python objects
and reprojecting polygons, it renders straight into pixels with numba on
every CPU core, or with CUDA on an NVIDIA GPU.

A global GFS map with filled contours, labelled contour lines, coastlines,
borders, gridline labels and a colorbar takes about **15 ms** to draw and save
as PNG. The same figure takes matplotlib and cartopy around 14 seconds.

<figure markdown>
![Surface analysis over North America](images/surface_analysis.png)
<figcaption>Sea-level pressure and 10 m wind from a GFS run, drawn and saved in about 25 ms.
The code is in the <a href="gallery/#surface-analysis">gallery</a>.</figcaption>
</figure>

## Why use it

- **Keep your code.** Swap `matplotlib.pyplot` for `fastplot.pyplot` and
  `cartopy.crs` for `fastplot.crs`. Most scripts need nothing else changed.
- **Same look.** Layout, colours, ticks, fonts and colorbars follow
  matplotlib's rules, so a figure looks like the one you would have made.
- **Fast enough for every frame.** Hourly maps for a whole forecast, or a
  map per request on a web server, stop being a batch job.
- **Uses the machine it is on.** Thread counts and the GPU switch-over are
  worked out from the hardware at import, not tuned for one computer.

## A first map

```python
import fastplot.pyplot as plt
import fastplot.crs as ccrs

fig = plt.figure(figsize=(10, 5.5), dpi=150)
ax = fig.add_subplot(projection=ccrs.Robinson())
filled = ax.contourf(lon, lat, t2m, levels=range(-40, 46, 2), cmap='RdYlBu_r',
                     extend='both', transform=ccrs.PlateCarree())
ax.coastlines(linewidth=0.5)
fig.colorbar(filled, ax=ax, orientation='horizontal', label='2 m temperature (°C)')
fig.savefig('t2m.png')
```

Head to [Getting started](getting-started.md) to install it, or browse the
[gallery](gallery.md) for complete examples.

<div class="grid-2" markdown>

[![Global 2 m temperature](images/global_temperature.png)](gallery.md#global-temperature)

[![A sounding with a hodograph](images/sounding.png)](gallery.md#a-sounding)

[![10 m wind on a globe](images/wind_globe.png)](gallery.md#wind-on-a-globe)

[![A five-day meteogram](images/meteogram.png)](gallery.md#a-meteogram)

</div>

## Where it fits

fastplot covers the parts of matplotlib, cartopy and MetPy's plotting that
weather and climate work leans on: maps of gridded fields, line and bar
charts, Skew-T diagrams and hodographs, polar plots and triangulations. It
does not try to be all of matplotlib. There are no 3-D axes yet, and vector
formats (PDF, SVG) hold the drawn picture rather than paths. The
[differences page](reference/differences.md) lists everything that behaves
differently.

It pairs well with [fastgrib](https://rochy111.github.io/fastgrib/), which
reads GRIB2 files into numpy arrays just as quickly.
