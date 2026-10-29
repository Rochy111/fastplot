# fastplot internals

fastplot draws maps of gridded fields -- the figures cartopy + matplotlib
draw for weather models -- straight into a pixel buffer, with no artist
tree, no path objects and no per-polygon Python. Every per-pixel and
per-vertex loop is a numba kernel (CPU, every core, GIL-free) or a CUDA
kernel run through cupy (GPU); the two give the same image.

## Why it is fast

matplotlib + cartopy, one GFS 0.25 deg field to an 1800x975 PNG (see
`tools/bench.py`): ~185 ms for imshow, ~370 ms pcolormesh, ~650 ms
contourf, 9-11 s cartopy contourf -- the last almost all in projecting
21,000 contour rings one by one through shapely. fastplot instead:

1. **Fields are sampled per pixel, not drawn as polygons.** For each output
   pixel: screen -> projection x/y -> lon/lat (the projection's inverse) ->
   fractional grid index -> value (nearest: pcolormesh; bilinear:
   contourf/imshow). The result is a float32 *value raster* the size of
   the map, NaN off the data. Curvilinear grids (WRF, HRRR) go the other
   way: grid nodes projected forward, each cell's two triangles
   rasterized with their values interpolated (or the cell's own value,
   flat). Nothing ever depends on the number of contour polygons.
2. **Colouring is a table lookup** of the value raster (`_color.shade`):
   filled contours are the value's level bin, exactly matplotlib's colour.
3. **Lines are geometry, drawn by tiles.** Contour lines (marching squares
   on the grid), coastlines, borders, barbs, gridlines become screen-space
   segments, binned into 32x32 px tiles; each tile's pixels take the
   anti-aliased coverage of the segments crossing it, in parallel.
4. **Text is a cached glyph mask** (Pillow's FreeType), blended in place.
5. **PNG is filtered by numba and deflated on every core** (`_png`).
6. **Layout is computed once** from text metrics, never by drawing twice.
7. Static geometry (coastlines, borders, graticule: projected, segmented,
   binned) is cached by what it depends on -- feature, projection, extent,
   pixel box -- across figures, so a new figure of the same map reuses it.
8. Cylindrical maps (PlateCarree, Mercator, Miller) of lon/lat grids are
   separable: a pixel's grid index is its column's along x and its row's
   along y, so W + H projections stand for W x H (bit for bit the same).

## Conventions

- Canvas: `numpy.uint8` `(H, W, 4)` RGBA, straight (not premultiplied)
  alpha, row 0 at the top. Screen coordinates: x right, y down, in pixels;
  pixel (i, j) covers [j, j+1) x [i, i+1), its centre is (j+0.5, i+0.5).
- Blending is "over" in float32, the same on CPU and GPU, for a source
  colour (r, g, b, a) in 0..255 and a coverage m in 0..1:

      f = (a / 255) * m                                   # float32
      dst_c = uint8(src_c * f + dst_c * (1 - f) + 0.5)    # c = r, g, b
      dst_a = uint8(255 * f + dst_a * (1 - f) + 0.5)

  In numba every constant and operand is `np.float32` (a float64 literal
  would promote the expression to double and the GPU would differ).
- Kernels use `from ._jit import njit, prange` (numba imported on the first
  kernel call), `@njit(cache=True, nogil=True)` or `parallel=True` with
  `prange` where the work is big enough. No fastmath: CPU and GPU must
  round alike.
- A module with GPU work exports its CUDA C as module-level string
  constants (`CUDA = r'''...'''`) of `__device__` functions and
  `extern "C" __global__` kernels; `_gpu.py` concatenates them into one
  `cupy.RawModule`, compiled with `('--std=c++14', '--fmad=false')`.
  cupy is imported only there, only when the GPU is used.
- Python 3.9+, numpy 1.22+ and 2.x. Dependencies: numpy, numba, Pillow.
  Optional: cupy (GPU). Nothing else at run time (no matplotlib, cartopy,
  shapely or pyproj); tests and tools may use them as references.
- Docstrings say what a thing is and why it is that way, as prose; code
  comments are short and trail the line they explain.

## Modules

| module | what |
|---|---|
| `_jit` | numba, deferred (as fastgrib's) |
| `_device` | CPU or GPU, picked per call; GPU failures fall back to CPU once, with a warning |
| `_proj` | projections: forward / inverse, numba and CUDA C, sphere |
| `_color` | colormaps, colour names, norms; values -> RGBA (`shade`) |
| `_features` | Natural Earth lines and polygons, bundled 110m/50m, 10m fetched |
| `_text` | fonts, text metrics and masks, glyph blending |
| `_png` | PNG encoder: numba filters, parallel deflate |
| `_raster` | value rasters: inverse sampling and forward cell rasterization |
| `_render` | the canvas operations (CPU), View, Coloring |
| `_ticker` | tick locators and formatters (matplotlib's, cartopy's) |
| `_contour` | marching squares: contour lines as polylines on the grid |
| `_clabel` | contour labels, placed as matplotlib's ContourLabeler places them |
| `_draw` | anti-aliased lines, polygon fill, tiles |
| `_barbs` | wind barb and arrow geometry |
| `_gpu` | the GPU canvas: field kernels in CUDA (cupy), compiled once per process |
| `_style` | matplotlib's rcParams defaults; font sizes, colours to bytes, dashes |
| `_layout` | GridSpec / SubplotSpec positions, shrinking a box to an aspect |
| `_artists` | Artist base; projecting points and lines (seam-safe); features, lines, markers, scatter, barbs, text |
| `_fields` | grids; pcolormesh/imshow, contourf, contour + clabel |
| `_gridliner` | graticule lines and edge labels |
| `_colorbar` | the colorbar, its axes and the room it takes (make_axes[_gridspec]) |
| `_axes` | MapAxes: projection, extent, the plotting methods |
| `_figure` | Figure: layout, `tight_layout` (a draw that only measures), the draw (CPU or GPU, falling back), savefig |
| `_axesbase` | what map and plain axes share: titles, the field / line / text methods, legends, patches |
| `_cartesian` | plain axes: Axis (scale, locators, ticks, limits, categories, dates), spines, autoscale as matplotlib's, twin / shared axes, the draw in zorder |
| `_plots` | the plain plot types (bar, hist, fills, steps, stems, error bars, ...) and their containers |
| `_scale` | linear, log, symlog, logit, function scales (matplotlib's transforms and default locators) |
| `_dates` | dates as day numbers; date locators and formatters (dateutil's rrule ported) |
| `_path` | matplotlib's Path: codes, unit shapes, affines, Bezier flattening (numba) |
| `_markers` | every matplotlib marker as geometry (MarkerStyle's paths and transforms) |
| `_patches` | patches, box styles, fancy arrows and connection styles, as matplotlib computes them |
| `_legend` | Legend: matplotlib's offsetbox packing, 'best' search, handlers |
| `_annotation` | Annotation: text and arrow, matplotlib's coordinate systems |
| `_rc` | rcParams (matplotlib 3.11 defaults), `rc_context`, style sheets |
| `transforms`, `pyplot`, `ticker`, `dates`, `patches`, `colors`, `cm`, `gridspec`, `style` | the public matplotlib-named modules |

## Plain axes: Agg's look

A plain axes (`_cartesian.Axes`) maps data through its scales and limits
(and an optional skew) to its box; everything else is the map machinery.
What makes its pictures matplotlib's, pixel for pixel, is Agg's own
behaviour, reproduced where plain axes draw (`ax._snap`, `ax._agg`; maps
keep their drawing, so their images are unchanged):

- **Snapping** (`_draw.snap`, Agg's PathSnapper): a path of horizontal and
  vertical segments only (spines, grids, bars, axhline, a square marker)
  has its vertices moved to pixel centres for an odd whole stroke width,
  to pixel edges otherwise; tick marks and single-colour markers are
  placed at whole pixels first (draw_markers).
- **Caps** (`_draw.BUTT`, `PROJECTING`, `ROUND`): at a polyline's two ends
  the coverage is the one across the line times the box filter along it
  over the stroke's extent; dashed lines are box-filtered over the dashes
  themselves (butt dash ends, anti-aliased). Interior vertices join round.
  `LEGACY` is the maps' round-capped drawing, kept bit for bit.
- **Markers in order** (`_draw.stamp_markers`): matplotlib composites each
  marker -- face, then edge -- before the next, so overlaps darken and the
  last drawn is on top. A marker's face and edge coverage is drawn once per
  1/8-px sub-position in use (all in one kernel call) and stamped, marker
  by marker, in one numba loop.
- **Layout** is matplotlib's: subplot parameters, gridspecs, colorbars
  (make_axes_gridspec), aspect, `tight_layout` (`_layout.tight_subplotpars`,
  fed by a draw on a canvas that paints nothing), legends (offsetbox
  packing in font-size units, 'best' by the same badness).

## The GPU canvas

Fields (sampling, cell rasterization, colouring, blending) run on the card;
lines, polygons and text run on the CPU's tiles. The canvas moves to host
memory when the first of those is drawn and back only if another field
follows, so a frame is usually one upload of the field and one download of
the image. That keeps the GPU-only code to the per-pixel kernels, and the
two devices' images identical (tests/test_figure.py). cupy always compiles
with `-ftz=true`: kernels that must match numba bit for bit load float32
values through their bits (see _color).

## Colour bytes

matplotlib turns a float colour into bytes two ways: images truncate
(`uint8(255 c)`), Agg's vector artists round (`uint8(255 c + 0.5)`).
pcolormesh and contourf are vector artists in matplotlib, so their
rasters here use 'nearest'; imshow uses 'floor'; lines, text and
markers 'nearest'.

## Speed without changing a bit

Every optimization keeps the image (and the PNG) byte for byte: a kernel
that replaces a numpy expression computes it in the same order --
np.cumsum's running sum, numpy's pairwise row sums, np.interp's branches,
libm pow where numpy called it -- and the tests hold each to its numpy
reference. What depends on the CPU's vector code paths or sort order
(hypot, arctan2, argsort) is left to numpy's own calls.
