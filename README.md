# fastplot

Weather maps -- what matplotlib + cartopy draw for model output -- drawn
straight into pixels by numba on every CPU core, or by CUDA on an NVIDIA
GPU, with matplotlib's and cartopy's API. A GFS 0.25 deg global map with
filled contours, contour lines, coastlines, borders, gridline labels and a
colorbar takes **~20 ms** to build, draw and write as PNG (cartopy: ~14 s), and
looks the same: same layout, same colours, same fonts.

```python
import fastplot as fp
import fastplot.crs as ccrs
import fastplot.feature as cfeature

fig = fp.figure(figsize=(12, 6.5), dpi=150)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
cf = ax.contourf(lon, lat, t2m, levels=range(-40, 46, 2), cmap='turbo', extend='both',
                 transform=ccrs.PlateCarree())
cs = ax.contour(lon, lat, t2m, levels=range(-40, 46, 10), colors='k', linewidths=0.4,
                transform=ccrs.PlateCarree())
ax.clabel(cs, fontsize=7)
ax.coastlines(linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3)
ax.gridlines(draw_labels=True, linewidth=0.2)
fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.06, shrink=0.8, label='2 m temperature (C)')
ax.set_title('GFS 0.25 deg  TMP 2 m')
fig.savefig('t2m.png')
```

Plain (non-map) plots go through the same machinery, with matplotlib's
pyplot, Axes, rcParams and style sheets -- lines, scatter, bars,
histograms, fills, error bars, log/symlog/logit and date axes, twin and
shared axes, legends, annotations with arrows, patches, colorbars,
`tight_layout` -- and come out as matplotlib's Agg would draw them (pixel
snapping, line caps, the same ticks, labels and layout), 3-15x faster:

```python
import fastplot.pyplot as plt

fig, ax = plt.subplots()
ax.semilogy(T, p, 'r', label='T')
ax.fill_betweenx(p, Td, T, color='0.8', alpha=0.5)
ax.set_ylim(1050, 100)
ax.legend()
fig.tight_layout()
fig.savefig('sounding.png')
```

Porting a matplotlib + cartopy script is mostly its imports:

```python
import matplotlib.pyplot as plt   ->  import fastplot.pyplot as plt
import matplotlib.ticker as mticker, matplotlib.dates as mdates, matplotlib.patches as mpatches
                                  ->  fastplot.ticker, fastplot.dates, fastplot.patches
import cartopy.crs as ccrs        ->  import fastplot.crs as ccrs
import cartopy.feature as cfeature ->  import fastplot.feature as cfeature
```

## Install

```
pip install git+https://github.com/Rochy111/fastplot              # numpy, numba, Pillow
pip install "fastplot[gpu] @ git+https://github.com/Rochy111/fastplot"   # + cupy (CUDA 12)
```

From a clone, `pip install -e .`. Python 3.9+, numpy 1.x or 2.x. The first
figure of each kind compiles fastplot's kernels once (numba keeps them);
`fastplot.warm()` does all of it up front, for a job's set-up.

## Speed

One frame, GFS 0.25 deg 2 m temperature (1440x721) at 1800x975 px, PNG
written (`tools/bench.py`, Windows 11, 24 threads, RTX 5070 Ti):

| case | matplotlib + cartopy | fastplot, CPU | fastplot, GPU | CPU speed-up |
|---|---:|---:|---:|---:|
| `pcolormesh` | 2177 ms | 9.9 ms | 9.1 ms | ~220x |
| `contourf` (43 levels) | 18831 ms | 10.7 ms | 9.1 ms | ~1760x |
| `contour` + `clabel` | 2205 ms | 30.9 ms | 30.0 ms | 71x |
| pcolormesh + coastlines, borders, gridline labels, colorbar | 2756 ms | 12.0 ms | 12.1 ms | ~230x |
| contourf + the same | 19495 ms | 11.8 ms | 11.4 ms | ~1650x |
| contourf + contour + the same | 14016 ms | 21.8 ms | 21.4 ms | ~640x |
| Lambert CONUS: contourf, contour, states | 23492 ms | 23.0 ms | 19.0 ms | ~1000x |
| pcolormesh + 1,800 wind barbs | 1952 ms | 16.0 ms | 16.3 ms | 122x |

Warm frames (each a new figure, PNG written), median of 9 for fastplot and 5
for matplotlib + cartopy, whose times vary by ~30% run to run. Plain
matplotlib without cartopy is faster than these -- ~185 ms imshow, ~370 ms
pcolormesh, ~650 ms contourf for the same field -- so on a plain
(non-map) figure fastplot's gain is 10-30x; on cartopy maps most of
cartopy's time is reprojecting polygons, which fastplot never does. A
field read by fastgrib with `gpu=True` draws (pcolormesh, coastlines,
colorbar) in ~7 ms without leaving the card.

Plain axes, 800x500 px, PNG written, warm frames, median of 7
(`tools/bench_plain.py`; CPU):

| case | matplotlib | fastplot | speed-up |
|---|---:|---:|---:|
| `plot`, 100k points | 71.5 ms | 24.7 ms | 2.9x |
| `scatter`, 20k points coloured by value | 158.8 ms | 22.8 ms | 7.0x |
| `hist`, 1M values, 100 bins | 63.6 ms | 20.6 ms | 3.1x |
| `bar` (50) with error bars | 78.2 ms | 6.5 ms | 12x |
| `pcolormesh` 500x500 | 163.6 ms | 10.9 ms | 15x |
| `contourf` + `contour` 300x200 | 83.3 ms | 9.7 ms | 8.6x |
| `imshow` RGB 600x400 | 106.1 ms | 20.4 ms | 5.2x |
| log-pressure `fill_betweenx` | 68.5 ms | 5.9 ms | 12x |

A whole script in a fresh process -- import, one full map, save -- is
~1 s (cartopy: >10 s), most of it numba loading its compiled kernels. Frames of an animation reuse the figure:
`m.set_array(next_field)` then `fig.savefig(...)` redraws only what
changed; and a new figure of the same map reuses the coastlines, borders
and graticule already projected, segmented and binned for it.

## Why it is fast

matplotlib builds a tree of Python artists and Agg paints their paths;
cartopy first reprojects every polygon through shapely -- a contourf of a
global field is ~21,000 rings, one Python call each. fastplot instead:

- **samples fields per pixel**: each pixel's map x/y goes back to
  lon/lat and into the grid (a division on a regular axis), and takes the
  cell's value (pcolormesh) or the bilinear one (contourf); 2-D grids
  (WRF, HRRR) are rasterized cell by cell. Nothing depends on how many
  contour polygons there would have been;
- **colours by table lookup**, bands exactly matplotlib's;
- **draws lines by tiles**: contour lines (marching squares on the grid,
  the same lines as matplotlib's contourpy), coastlines, borders, barbs,
  gridlines become segments binned into 32x32 px tiles, each tile on its
  own core, anti-aliased;
- **caches text** as glyph masks (Pillow's FreeType, DejaVu Sans, kerning
  and matplotlib's mathtext layout);
- **writes PNG** with numba filters and deflate on every core;
- **lays out once** from text metrics -- the same positions matplotlib's
  subplot parameters, gridspecs, aspect and colorbar code give.

## CPU and GPU

```python
fp.set_device('auto')     # default: the GPU for big work, when it is usable
fp.set_device('gpu')      # always the GPU (the CPU if it cannot)
fp.set_device('cpu')
fp.gpu_info()             # {'usable': True, 'device': 'NVIDIA GeForce RTX 5070 Ti', ...}
fig.savefig('a.png', device='gpu')    # or per figure
```

On the GPU, fields are sampled, rasterized, coloured and blended by CUDA
kernels; lines and text stay on the CPU (they are a few ms there). The
two draw **the same image, byte for byte** (tested), so the choice is
only about speed. Fields read by fastgrib with `gpu=True` are drawn
without leaving the card.

Anything that goes wrong on the GPU -- no cupy, no CUDA device, an old
driver, a card missing something a kernel needs, out of memory -- is not
an error: the frame is drawn again on the CPU, a warning says why (once),
and the process keeps to the CPU.

## What there is

| | |
|---|---|
| figures | `figure`, `subplots`, `add_subplot` (incl. `GridSpec`), `add_axes`, `subplots_adjust`, `suptitle`, `fig.text`, `colorbar`, `savefig` (PNG, JPEG, WebP, ..., `bbox_inches='tight'`, `dpi`, `transparent`), `draw()` -> RGBA array |
| fields | `pcolormesh` (nearest, flat, gouraud; 1-D or 2-D coordinates), `imshow`, `contourf`, `contour`, `clabel`, `barbs`, `scatter`, `plot`, `text`, `annotate` |
| colour | every matplotlib colormap and colour name, `ListedColormap`, `LinearSegmentedColormap.from_list`, norms (`Normalize`, `LogNorm`, `BoundaryNorm`, `TwoSlopeNorm`, `CenteredNorm`, `SymLogNorm`, `NoNorm`; matplotlib norm and colormap objects are accepted as they are), `extend`, under/over/bad |
| plain axes | `pyplot` (current figure/axes, every Axes method), `plot` (format strings, every matplotlib marker), `scatter`, `bar`/`barh` (+ error bars, `bar_label`), `hist` (bar, step, stepfilled; stacked, density, cumulative), `fill_between`/`fill_betweenx` (where, interpolate, step), `stackplot`, `step`, `stairs`, `stem`, `errorbar`, `axhline`/`axvline`/`axhspan`/`axvspan`, `hlines`/`vlines`, `semilogx`/`semilogy`/`loglog`, `pcolormesh`/`imshow` (scalar and RGB(A))/`contourf`/`contour`/`clabel`, `text`, `annotate` (every arrow and connection style), `add_patch` (`fastplot.patches`: Rectangle, Circle, Ellipse, Polygon, Wedge, Arc, RegularPolygon, FancyBboxPatch, FancyArrowPatch), `legend` (every loc incl. 'best', columns, titles; `fig.legend`) |
| axes | linear, log, symlog, logit, function scales; dates (datetime, datetime64, pandas) and categories (strings); `fastplot.ticker` and `fastplot.dates` locators and formatters (matplotlib's own objects accepted too); limits, margins, autoscale (sticky edges), inversion, aspect, `twinx`/`twiny`, `sharex`/`sharey`, spines, `tick_params`, minor ticks, grids, `tight_layout`, skew (for Skew-T) |
| style | `rcParams`, `rc_context`, `style.use` with matplotlib's 31 style sheets (ggplot, seaborn-v0_8-*, bmh, dark_background, ...) |
| maps | `PlateCarree`, `Mercator`, `LambertConformal`, `Stereographic`, `NorthPolarStereo`, `SouthPolarStereo`, `Orthographic`, `LambertAzimuthalEqualArea`, `AlbersEqualArea`, `Robinson`, `Miller`, `RotatedPole` (sphere); `set_extent`, `set_global`, `coastlines`, `add_feature` (`BORDERS`, `STATES`, `COASTLINE`, `LAKES`, `LAND`, `OCEAN`, `RIVERS`, `COUNTIES`, `NaturalEarthFeature`), `gridlines(draw_labels=...)` |

Natural Earth at 110m and 50m ships with fastplot; 10m is read from
cartopy's cache if it has one, or downloaded once (`FASTPLOT_DATA_DIR`).

## Differences from matplotlib + cartopy

- Projections are spherical (cartopy's default globe for most is WGS84):
  a map's geometry is self-consistent (data, coastlines and gridlines go
  through the same projection), but native x/y in metres differ slightly.
- `set_extent` on a curved outline (a conic map) is the exact bound;
  cartopy's is a coarse approximation, so the aspect can differ by ~0.3%.
- Text at angles other than 0/90/180/270 (contour labels) is placed as
  matplotlib places it, but its glyphs are rotated as an image.
- Contour-label placement follows matplotlib's algorithm on fastplot's
  lines; labels can land on slightly different points.
- Gridline labels go on the map's four edges, horizontal, on every
  projection; cartopy rotates them along curved (conic, polar) graticules.
- Thin barbs (linewidth well under 1 pt) come out a little lighter than
  matplotlib's.
- Drawing figures from several threads at once needs numba's OpenMP or
  TBB threading layer (the default wheels have one); its fallback
  workqueue layer cannot run parallel kernels from two threads.
- Plain axes are drawn as matplotlib's Agg draws (pixels snapped, butt and
  projecting caps, markers one after another); maps keep fastplot's own
  line drawing (round caps, no snapping), so map images stay as they were.
- Glyphs are rasterized by fastplot (FreeType outlines, its own coverage),
  not by Agg with FreeType's forced autohinting: text sits where
  matplotlib's sits, with the same metrics, but its pixels differ a
  little -- most of what differs on a plain plot (~0.2-1% of pixels).
- Line joins are round (matplotlib's patch edges are mitred): a corner
  pixel can differ. A scatter's markers are placed to 1/8 px.
- Not there (yet): `quiver`, `streamplot`, hatching, `tricontourf`,
  polar and 3-D axes, `subplot_mosaic`, vector output (PDF/SVG).

## Tests

`python -m pytest` (matplotlib, cartopy and contourpy as references; the
GPU tests skip without a GPU). `tests/test_cartesian.py` draws plain plots
with both libraries and compares limits, ticks and labels, legend and
`tight_layout` geometry, and the pixels. See `docs/internals.md` for how it works.
