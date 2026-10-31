# Speed

All times are for warm frames (each a new figure, PNG written), measured on
Windows 11 with an Intel i7-13700K (8 performance and 8 efficiency cores)
and an RTX 5070 Ti. fastplot's numbers are the median of 9 runs and
matplotlib's the median of 5, since its times vary by about 30% between
runs. The scripts are in `tools/`, so you can measure your own machine.

![Speed comparison](images/speed_chart.png)

## Maps

One GFS 0.25° field (1440 x 721) at 1800 x 975 pixels, `tools/bench.py`:

| Figure | matplotlib + cartopy | fastplot CPU | fastplot GPU | Speed-up |
|---|---:|---:|---:|---:|
| `pcolormesh` | 2,177 ms | 6.1 ms | 6.7 ms | 360x |
| `contourf`, 43 levels | 18,831 ms | 5.1 ms | 5.3 ms | 3,700x |
| `contour` + `clabel` | 2,205 ms | 14.9 ms | 14.0 ms | 150x |
| pcolormesh, coastlines, borders, gridline labels, colorbar | 2,756 ms | 7.7 ms | 7.6 ms | 360x |
| contourf and the same | 19,495 ms | 7.0 ms | 6.5 ms | 2,800x |
| contourf, contour and the same | 14,016 ms | 14.5 ms | 14.5 ms | 970x |
| Lambert CONUS: contourf, contour, states | 23,492 ms | 12.0 ms | 11.3 ms | 1,950x |
| pcolormesh and 1,800 wind barbs | 1,952 ms | 11.8 ms | 11.5 ms | 165x |

Most of cartopy's time goes into reprojecting polygons one by one, which
fastplot never does. Plain matplotlib without cartopy is quicker, at about
185 ms for `imshow`, 370 ms for `pcolormesh` and 650 ms for `contourf` of the
same field, so on non-map figures the gain is more like 10 to 30 times.

## Plots

800 x 500 pixels, `tools/bench_plain.py`, CPU:

| Figure | matplotlib | fastplot | Speed-up |
|---|---:|---:|---:|
| `plot`, 100k points | 71.5 ms | 8.0 ms | 9x |
| `scatter`, 20k points coloured by value | 158.8 ms | 4.9 ms | 32x |
| `hist`, 1M values, 100 bins | 63.6 ms | 3.3 ms | 19x |
| `bar` (50) with error bars | 78.2 ms | 3.1 ms | 25x |
| `pcolormesh` 500 x 500 | 163.6 ms | 4.5 ms | 36x |
| `contourf` + `contour` 300 x 200 | 83.3 ms | 4.3 ms | 19x |
| `imshow` RGB 600 x 400 | 106.1 ms | 3.9 ms | 27x |
| Log-pressure `fill_betweenx` | 68.5 ms | 2.7 ms | 25x |
| Skew-T with barbs, CAPE/CIN, adiabats and mixing lines, 700 x 700 | 70.8 ms | 6.8 ms | 10x |
| `streamplot` x3, including integration | 418 ms | 20 ms | 21x |
| Two hodographs with rings, coloured lines and vectors | 52.3 ms | 5.8 ms | 9x |
| `tricontourf`, `tricontour` + `clabel`, `tripcolor` x2, `triplot` (400 points) | 109.3 ms | 8.5 ms | 13x |

## How it works

matplotlib builds a tree of Python artist objects and Agg paints their paths
one at a time. cartopy first reprojects every polygon through shapely; a
global `contourf` is around 21,000 rings, each a separate Python call.

fastplot does not build polygons for fields at all. For every pixel of the
map it takes the pixel's projected position, converts it back to longitude
and latitude, and looks up the value in the data grid, so the cost depends
on the size of the image rather than on the shape of the contours. Colours
come from a lookup table with matplotlib's band edges.

Lines (contours, which are the same lines contourpy finds, coastlines,
borders, barbs and gridlines) are stroked and rasterized as Agg does it, with
the edges sorted into bands of rows and each band drawn on its own core.
Text is cached as glyph masks rendered with FreeType, glyph by glyph at
matplotlib's positions, with kerning and matplotlib's mathtext layout. Layout is computed once from text
sizes using matplotlib's rules for subplot parameters, gridspecs, aspect
ratios and colorbars. Finally the PNG encoder filters and compresses the
image on every core.

## Start-up

A fresh process that imports fastplot, draws one figure and writes its PNG,
timed from outside the process, median of 7 runs. matplotlib and cartopy
take over ten seconds for a map. In a long-running process, or with
[`frames`](guide/frames.md), only the first figure pays this.

| Script | Before | Now |
|---|---:|---:|
| Line plot | 1,495 ms | 715 ms |
| Global map, 1440 x 721 field: `contourf`, coastlines, colorbar | 1,680 ms | 1,056 ms |
| Lambert CONUS, 1799 x 1059 field: `contourf`, contours, labels, states | 1,662 ms | 1,076 ms |

What changed: a kernel found in numba's cache now loads without numba setting
up its compiler, and fastplot's and numba's compiled modules are read from
one file rather than hundreds. `python -m fastplot warm` writes that file,
and otherwise the first run does, as it exits.

## Long jobs

Times per frame, each frame a new figure with a PNG written, fields as in the
table above. "Long job" is a process that has run for a few seconds.

| Figure | | Before | Now |
|---|---|---:|---:|
| Six panels | first frames | 18.5 ms | 16.6 ms |
| | long job | 22.1 ms | 17.2 ms |
| | with `frames` | 32.0 ms | 22.5 ms |
| Global map | first frames | 10.7 ms | 14.0 ms |
| | long job | 10.0 ms | 13.5 ms |
| | with `frames` | 9.1 ms | 12.4 ms |
| Lambert `contourf` on a 2-D grid | first frames | 69.6 ms | 45.1 ms |
| | long job | 69.8 ms | 43.9 ms |
| | with `frames` | 37.8 ms | 25.5 ms |
| Lambert `pcolormesh` and barbs on a 2-D grid | first frames | 51.3 ms | 18.6 ms |
| | long job | 53.0 ms | 19.0 ms |
| | with `frames` | 30.6 ms | 11.4 ms |

The global map is slower because gridline labels now follow cartopy's rules,
which cost about 4 ms a frame.

A large grid's nodes are projected once: from the third frame on, a grid whose
coordinates are the same to the bit in the same view reuses them, and a mesh of
nearest or flat cells keeps which cell shows at each pixel, so a frame only
looks up its values. A grid that changes every frame costs what it did.

On a hybrid CPU under Windows, a process with no window in front can be moved
to the efficiency cores after about two seconds, doubling its frame times, and
fastplot now asks Windows not to
([details](guide/hardware.md#how-the-settings-are-chosen)). Whether that
happens varies from run to run, so the "long job" times above do not always
show it.

## Lines, markers and meshes

Matching Agg pixel for pixel cost time at first; these are the draws affected,
a figure drawn again, against the rasterizer before the Agg port.

| Figure | Before the port | Now |
|---|---:|---:|
| 20 lines of 500 points | 4.6 ms | 4.7 ms |
| One line of 1M points | 23.5 ms | 20.4 ms |
| 2,000 bars | 20.0 ms | 10.4 ms |
| `contour`, 20 levels | 3.3 ms | 2.4 ms |
| 10m coastlines, borders and states, whole globe | 37.1 ms | 29.1 ms |
| Leaning mesh, 1.9M cells, 1-D coordinates | 6.0 ms | 4.9 ms |
| Leaning mesh, 1.9M cells, 2-D coordinates | 34.8 ms | 5.8 ms |

A scatter of 2,000 markers or more with varying colours or sizes is stamped: each
marker is placed to 1/8 of a pixel, and 64 masks per size serve them all. The
masks come from the same Agg rasterizer as an exact marker, so the difference is
a shade at each marker's edge, which can't be seen at normal zoom. Squares and
other rectilinear markers stay exact, because snapping could move them a whole
pixel. With `rcParams['fastplot.exact'] = True`, every marker is exact, and
scatters of up to 200,000 markers differ from matplotlib's by at most one pixel
in the cases measured.

| Scatter | `fastplot.exact` | Default | Mean difference | Largest |
|---|---:|---:|---:|---:|
| 5,000 circles, translucent | 11.3 ms | 5.4 ms | 0.31 | 18 |
| 20,000 circles | 17.7 ms | 6.4 ms | 0.50 | 26 |
| 20,000 circles of many sizes, black edges | 23.6 ms | 13.1 ms | 1.23 | 38 |
| 50,000 circles | 37.4 ms | 12.6 ms | 0.42 | 28 |
| 50,000 triangles of many sizes, translucent | 28.1 ms | 15.6 ms | 0.49 | 20 |
| 20,000 squares | 12.1 ms | 12.4 ms | 0 | 0 |

Differences are in levels of 255 from matplotlib's image, the mean over the
whole figure.

The tables in this section and the two above were measured side by side on a
busy machine with made-up fields, each number the median of at least 5 runs.

## Against 0.3.0

Matching matplotlib costs some of what approximate drawing saved. These are
eight new figures, each drawn and saved to PNG, against the released 0.3.0, run
in turn for 9 rounds on a machine in use with made-up fields.

| Figure | 0.3.0 | Now | Change |
|---|---:|---:|---:|
| `plot`, 100,000 points | 11.4 ms | 9.4 ms | -18% |
| `pcolormesh` and 1,800 barbs | 17.7 ms | 14.7 ms | -17% |
| `streamplot`, 3 times | 15.0 ms | 14.9 ms | -1% |
| Global map, contours | 24.4 ms | 24.8 ms | +2% |
| `scatter`, 20,000 points | 6.7 ms | 6.9 ms | +3% |
| Lambert map, US states | 20.8 ms | 21.4 ms | +3% |
| `pcolormesh` and coastlines | 10.6 ms | 11.0 ms | +4% |
| Skew-T with adiabats | 6.5 ms | 7.2 ms | +11% |

Changes under about 5% vary from run to run. What is left on maps and Skew-T is
the price of matching matplotlib: stroking as Agg does costs a little more than
the tiles it replaced, 7.0 against 6.5 ms on the global map in a profile, and
maps also follow cartopy's chords and look for holes in the field.

Some settings were tried for more speed and left as they are, because they gave
too little for what they cost:

- **PNG.** No level, filter or zlib strategy saves more than about 0.5 ms, and
  some double the file size.
- **Line simplification.** A threshold looser than matplotlib's saves 10 to 20%
  on dense lines, but moves the tips of the lines themselves.
- **Contour precision.** Tracing in single precision would save about 0.4 ms of a
  global map, and could change which way a saddle turns.

Writing to new memory from every thread at once is slow on Windows: threads
queue for the same page faults, and a parallel copy of a 1M-cell field took
2.5 ms longer than a serial one. Kernels that run on every core work in kept
memory where they can (`_arena`).
