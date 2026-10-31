# Getting started

## Installation

fastplot needs Python 3.9 or later. Its dependencies are numpy, numba,
Pillow and freetype-py, all of which install from wheels.

```bash
pip install git+https://github.com/Rochy111/fastplot
```

For drawing on an NVIDIA GPU, install the `gpu` extra, which adds cupy for
CUDA 12:

```bash
pip install "fastplot[gpu] @ git+https://github.com/Rochy111/fastplot"
```

From a clone of the repository, use `pip install -e .` instead.

To upgrade, add `-U`: `pip install -U git+https://github.com/Rochy111/fastplot`.
Each commit's version is taken from git (`0.2.1.dev4+g1a2b3c4` is the fourth
commit after the v0.2.0 tag), so pip sees a newer commit as a newer version.
Installs made before v0.2.0 all said 0.1.0; their upgrade works the same way.

### Compiling the kernels

fastplot's drawing code is compiled by numba for your exact CPU. `pip
install` does this for you, which adds about 30 seconds to the install, so
there is nothing to run afterwards and the first figure is already at full
speed.

Importing fastplot starts loading the compiled kernels on a background
thread while your script reads its data. A script that imports fastplot,
draws one map and saves it takes 0.8 to 1.1 seconds from start to finish,
and about 2 seconds the first time after installing.

If the compiled kernels stop matching, because numba was upgraded or the
install was copied to a different CPU, fastplot notices when it is imported
and compiles them again in a low-priority background process. Figures still
work while that runs, and the first ones are slower. An editable install
(`pip install -e`) gets its kernels this way too.

In a container image built on one machine and run on others, the same thing
happens on the first run on each new CPU. To avoid that, run
`python -m fastplot warm` in the container at start-up, or set
`NUMBA_CPU_NAME=generic` both when building and when running, at the cost
of some speed.

## A first map

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

For model output in GRIB files, [fastgrib](https://rochy111.github.io/fastgrib/)
returns the arrays:

```python
import fastgrib

grib = fastgrib.open('gfs.t00z.pgrb2.0p25.f006')
t2m = grib.read('TMP:2 m above ground') - 273.15
lat, lon = grib.latlon()      # 2-D arrays; pass them as they are, or lat[:, 0] and lon[0]
```

## A first plot

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

## Porting a script

Most of the change is in the imports.

| matplotlib, cartopy, MetPy | fastplot |
|---|---|
| `import matplotlib.pyplot as plt` | `import fastplot.pyplot as plt` |
| `import matplotlib.ticker as mticker` | `import fastplot.ticker as mticker` |
| `import matplotlib.dates as mdates` | `import fastplot.dates as mdates` |
| `import matplotlib.patches as mpatches` | `import fastplot.patches as mpatches` |
| `import cartopy.crs as ccrs` | `import fastplot.crs as ccrs` |
| `import cartopy.feature as cfeature` | `import fastplot.feature as cfeature` |
| `from metpy.plots import SkewT, Hodograph` | `from fastplot.skewt import SkewT, Hodograph` |

matplotlib's colormap, norm, ticker and date-locator objects can be passed
to fastplot unchanged. If a figure comes out differently from what you
expect, [Differences from matplotlib](reference/differences.md) is the
first place to look.
