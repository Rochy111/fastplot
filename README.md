# fastplot

[![tests](https://github.com/Rochy111/fastplot/actions/workflows/tests.yml/badge.svg)](https://github.com/Rochy111/fastplot/actions/workflows/tests.yml)
[![docs](https://img.shields.io/badge/docs-rochy111.github.io%2Ffastplot-blue)](https://rochy111.github.io/fastplot/)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![licence](https://img.shields.io/badge/licence-MIT-green)

Weather maps and scientific plots with the matplotlib and cartopy API,
drawn straight into pixels by numba on every CPU core, or by CUDA on an
NVIDIA GPU. A global map with filled contours, labelled contours, coastlines,
borders and a colorbar takes about 15 ms to draw and save. matplotlib and
cartopy take around 14 seconds.

![Sea-level pressure and 10 m wind over North America](docs/images/surface_analysis.png)

**[Documentation](https://rochy111.github.io/fastplot/)** ·
[Gallery](https://rochy111.github.io/fastplot/gallery/) ·
[Getting started](https://rochy111.github.io/fastplot/getting-started/)

## Install

```bash
pip install git+https://github.com/Rochy111/fastplot
python -m fastplot warm      # compile the kernels once, about 12 s
```

Add the `gpu` extra (`"fastplot[gpu] @ git+https://..."`) to draw on an NVIDIA
GPU with cupy. Python 3.9+, numpy 1 or 2, Linux, Windows and macOS.

## Example

```python
import fastplot.pyplot as plt          # instead of matplotlib.pyplot
import fastplot.crs as ccrs            # instead of cartopy.crs

fig = plt.figure(figsize=(10, 5.5), dpi=150)
ax = fig.add_subplot(projection=ccrs.Robinson())
filled = ax.contourf(lon, lat, t2m, levels=range(-40, 46, 2), cmap='RdYlBu_r',
                     extend='both', transform=ccrs.PlateCarree())
ax.coastlines(linewidth=0.5)
fig.colorbar(filled, ax=ax, orientation='horizontal', label='2 m temperature (°C)')
fig.savefig('t2m.png')
```

Most matplotlib and cartopy scripts only need their imports changed.

## Speed

| One frame, PNG written | matplotlib (+ cartopy) | fastplot |
|---|---:|---:|
| Global map: contourf, contours, labels, coastlines, colorbar | 14,016 ms | 14.5 ms |
| Lambert CONUS: contourf, contours, states | 23,492 ms | 12.0 ms |
| pcolormesh with 1,800 wind barbs | 1,952 ms | 11.8 ms |
| Skew-T with adiabats, CAPE and CIN | 70.8 ms | 6.8 ms |
| Scatter, 20,000 points | 158.8 ms | 4.9 ms |

i7-13700K, warm frames. The [speed page](https://rochy111.github.io/fastplot/speed/)
has the full tables and explains where the time goes.

## What's included

- **Maps:** twelve projections, `pcolormesh`, `contourf`, `contour`,
  `clabel`, barbs, quiver, streamlines, Natural Earth features and labelled
  gridlines.
- **Plots:** lines, scatter, bars, histograms, fills, error bars, log and
  date axes, twin and shared axes, legends, annotations, patches,
  triangulations and polar axes.
- **Meteorology:** MetPy's `SkewT` and `Hodograph`.
- **Style:** `rcParams`, matplotlib's style sheets, every colormap and norm.
- **Many frames:** `fastplot.frames()` draws a forecast's worth of maps
  several at a time.
- **Any machine:** thread counts and the GPU switch-over come from the
  hardware it runs on.

| | | |
|---|---|---|
| ![Global temperature](docs/images/global_temperature.png) | ![Sounding](docs/images/sounding.png) | ![Wind on a globe](docs/images/wind_globe.png) |
| ![500 hPa heights](docs/images/polar_heights.png) | ![Meteogram](docs/images/meteogram.png) | ![Wind rose](docs/images/wind_rose.png) |

See the [differences from matplotlib](https://rochy111.github.io/fastplot/reference/differences/)
before relying on pixel-exact output.

## Related

[fastgrib](https://github.com/Rochy111/fastgrib) reads GRIB2 and GRIB1 files
into numpy arrays, as quickly as fastplot draws them.

## Licence

MIT. Natural Earth data is public domain, and the DejaVu fonts are under
their own licence. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
