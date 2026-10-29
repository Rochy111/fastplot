# Getting started

## Install

fastplot needs Python 3.9 or newer. It installs with numpy, numba and Pillow.

```bash
pip install git+https://github.com/Rochy111/fastplot
```

To draw on an NVIDIA GPU as well, add the `gpu` extra (cupy for CUDA 12):

```bash
pip install "fastplot[gpu] @ git+https://github.com/Rochy111/fastplot"
```

Working from a clone, use `pip install -e .` instead.

## Compile once after installing

numba compiles fastplot's kernels for your CPU the first time they run and
keeps the result on disk. Left to happen on its own, the first map you draw
waits about 16 seconds for that. Run this once after installing or upgrading
and it is done up front, in parallel, in about 12 seconds:

```bash
python -m fastplot warm
```

After that, `import fastplot` starts loading the compiled kernels in the
background while your script reads its data, so the first figure rarely
waits. A complete script in a fresh process, from import to saved PNG, takes
about a second.

!!! tip "Containers and CI"
    Put `python -m fastplot warm` in your image build or job setup, after
    the install step. The cache lives next to the installed package.

## Your first map

```python
import numpy as np
import fastplot.pyplot as plt
import fastplot.crs as ccrs
import fastplot.feature as cfeature

lon = np.arange(0, 360, 1.0)
lat = np.arange(-90, 90.1, 1.0)
field = np.cos(np.radians(lat))[:, None] * np.sin(np.radians(3 * lon))[None, :]

fig = plt.figure(figsize=(10, 5), dpi=120)
ax = fig.add_subplot(projection=ccrs.PlateCarree())
cf = ax.contourf(lon, lat, field, levels=12, cmap='viridis', transform=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linewidth=0.3)
ax.gridlines(draw_labels=True, linewidth=0.3)
fig.colorbar(cf, ax=ax, shrink=0.7)
fig.savefig('first_map.png')
```

If you have a GRIB file, [fastgrib](https://rochy111.github.io/fastgrib/)
gives you the arrays:

```python
import fastgrib

grib = fastgrib.open('gfs.t00z.pgrb2.0p25.f006')
t2m = grib.read('TMP:2 m above ground') - 273.15
lat, lon = grib.latlon()            # 2-D; fastplot takes these or the 1-D axes
```

## Your first plot

Plain plots work the way they do in matplotlib:

```python
import fastplot.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(hours, temperature, color='tab:red', label='Temperature')
ax.fill_between(hours, dewpoint, temperature, alpha=0.2)
ax.set_xlabel('Hour')
ax.set_ylabel('°C')
ax.legend()
fig.tight_layout()
fig.savefig('temperature.png')
```

## Moving an existing script over

Most of the work is in the imports:

| matplotlib / cartopy | fastplot |
|---|---|
| `import matplotlib.pyplot as plt` | `import fastplot.pyplot as plt` |
| `import matplotlib.ticker as mticker` | `import fastplot.ticker as mticker` |
| `import matplotlib.dates as mdates` | `import fastplot.dates as mdates` |
| `import matplotlib.patches as mpatches` | `import fastplot.patches as mpatches` |
| `import cartopy.crs as ccrs` | `import fastplot.crs as ccrs` |
| `import cartopy.feature as cfeature` | `import fastplot.feature as cfeature` |
| `from metpy.plots import SkewT, Hodograph` | `from fastplot.skewt import SkewT, Hodograph` |

matplotlib colormap and norm objects can be passed to fastplot as they are,
and so can matplotlib tickers and date locators. If a figure comes out
differently from what you expect, check
[Differences from matplotlib](reference/differences.md) first.

## Next steps

- The [gallery](gallery.md) has complete scripts for maps, soundings and charts.
- [Maps](guide/maps.md) covers projections, features and the field methods.
- [Many frames](guide/frames.md) shows how to draw a forecast's worth of maps quickly.
- [CPU, GPU and threads](guide/hardware.md) explains how fastplot picks its settings.
