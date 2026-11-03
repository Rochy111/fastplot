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
3. **Lines are geometry, drawn as Agg draws them.** Contour lines (marching
   squares on the grid), coastlines, borders, barbs and gridlines are
   stroked into outlines by a port of Agg's stroker, in `_agg`, and filled
   by a port of its rasterizer. Edges are sorted into bands of rows, each
   band on its own core; paths within a band blend in order.
4. **Text is a cached glyph mask**: each glyph rendered by FreeType
   (freetype-py) at its fractional position, as matplotlib's Agg does, then
   blended in place.
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

  over an opaque pixel (dst_a 255: every pixel of a figure with a
  background). Over a partly transparent one the destination's colour
  weighs by its alpha too, and the sum is divided by the new alpha:

      w = (dst_a / 255) * (1 - f);  out_a = f + w         # out_a 0: unchanged
      dst_c = uint8((src_c * f + dst_c * w) / out_a + 0.5)
      dst_a = uint8(255 * out_a + 0.5)

  -- else a transparent page's colour would bleed into anti-aliased edges.
  The blend lives in `_draw.blend_px`, `_text.blend_mask_kernel` and their
  CUDA twins (`fp_blend`, the text kernel). numba's on-disk cache checks
  only a function's own file: after changing `blend_px`, clear
  `fastplot/__pycache__/*.nbi *.nbc`, or callers in other modules
  (`_render._blend_image`, `_hatch._blend`) keep the old blend compiled in.
  In numba every constant and operand is `np.float32` (a float64 literal
  would promote the expression to double and the GPU would differ).
- Kernels use `from ._jit import njit, prange` (numba imported on the first
  kernel call, or earlier by the preload thread `import fastplot` starts),
  `@njit(cache=True, nogil=True)` or `parallel=True` with `prange` where
  the work is big enough. No fastmath: CPU and GPU must round alike. A
  prange over rows or bands of the canvas walks them in a stride
  (`_draw._stride`), so each thread's share is spread down the picture
  rather than one run of it: a plot dense in the middle keeps every
  thread busy.
- Kernels get C-contiguous, writable arrays (`_draw._rw`): numba types a
  read-only or strided array apart and compiles the kernel again for it
  (seconds each on a new machine). Read-only on purpose (cached text
  masks) is read-only always. After changing kernels, run
  `tools/make_warm_manifest.py` so `warm()` knows what to compile.
- Never take a tuple of arrays apart in code that runs per edge or per
  pixel: each unpacking counts references on each array, atomically. Passing
  the arrays as arguments made every line and polygon about a third faster
  (`_agg._line`).
- A cached kernel loads without numba setting up its compiler only on the
  releases in `_jit.CACHE_ONLY_NUMBA`, because compiled code missing a
  symbol stops the process rather than raising. Any other release works
  and starts slower. To add one, run the tests and, in a fresh process,
  `compile` every signature in every kernel's cache index with the release
  listed: a process that gets through all of them has every symbol.
- A module with GPU work exports its CUDA C as module-level string
  constants (`CUDA = r'''...'''`) of `__device__` functions and
  `extern "C" __global__` kernels; `_gpu.py` concatenates them into one
  `cupy.RawModule`, compiled with `('--std=c++14', '--fmad=false')`.
  cupy is imported only there, only when the GPU is started: under
  `'auto'`, not for a figure that the last process found too small for
  the card (`gpu.txt` in the cache directory, `_hw.gpu_seen`), nor before
  the CPU time spent on bigger ones has paid for starting it (`_device`).
- Python 3.9+, numpy 1.22+ and 2.x. Dependencies: numpy, numba, Pillow,
  freetype-py.
  Optional: cupy (GPU). Nothing else at run time (no matplotlib, cartopy,
  shapely or pyproj); tests and tools may use them as references.
- Docstrings say what a thing is and why it is that way, as prose; code
  comments are short and trail the line they explain.

## Modules

| module | what |
|---|---|
| `_jit` | numba, deferred (as fastgrib's): kernels run from the kernel image, or numba imported where it cannot; a cached kernel loaded without numba's compiler set up; a draw's thread cap, raised for many markers' masks, on the image's thread pool or numba's; idle workers parked; `Choice`, which runs a step on one thread or all of them as this machine has timed each faster |
| `_image` | the kernel image: numba's compiled kernels, the parallel ones too, linked into one file for this CPU and run without numba (see [below](#the-kernel-image)); its kept memory |
| `_freetype` | FreeType's few calls fastplot makes, from the library freetype-py ships, without importing freetype-py |
| `_npz` | numpy's `.npz` files read without `zipfile`, each array by numpy's own `.npy` reader |
| `_bundle` | imports from one file: the compiled modules of fastplot, numba, llvmlite, freetype and Pillow, checked against their sources as `.pyc` files are |
| `_hw` | the machine: CPUs allowed (affinity, cgroup quota), cores and performance cores, the GPU; the draw threads and the GPU threshold drawn from them, or from tune()'s file |
| `_tune` | `tune()`: draws timed at thread counts and on CPU / GPU, kept per machine; `hardware()` |
| `_warm` | `warm()`, run by `setup.py` while installing and by `keep_up()` in the background when the file it leaves beside numba's cache is missing or out of date: recipes of every figure kind; the kernel manifest (`tools/make_warm_manifest.py`) shared out among processes |
| `_batch` | `frames()`: figures drawn on threads at once, numba's teams sized to share the cores |
| `_arena` | per-thread scratch buffers kept between calls (contour working arrays, a field's value raster and colour layer, line simplification, a PNG's palette indices): no fresh pages each frame |
| `_device` | CPU or GPU, picked per call; GPU failures fall back to CPU once, with a warning |
| `_proj` | projections on cartopy's Earth: forward / inverse, numba and CUDA C; on an ellipsoid the spherical formulas on an auxiliary latitude |
| `_color` | colormaps, colour names, norms; values -> RGBA (`shade`) |
| `_features` | Natural Earth lines and polygons, bundled 110m/50m, 10m fetched |
| `_text` | fonts, text metrics and masks, glyph blending |
| `_png` | PNG encoder: numba filters, parallel deflate |
| `_raster` | value rasters: inverse sampling and forward cell rasterization |
| `_render` | the canvas operations (CPU), View, Coloring |
| `_ticker` | tick locators and formatters (matplotlib's, cartopy's) |
| `_contour` | marching squares: contour lines as polylines on the grid, in contourpy's order and direction |
| `_clabel` | contour labels, placed as matplotlib's ContourLabeler places them |
| `_agg` | Agg's stroker, dashes, clipper and rasterizer, band by band |
| `_draw` | lines, polygons, paths and markers on the canvas, by way of `_agg` |
| `_barbs` | wind barb and arrow geometry |
| `_gpu` | the GPU canvas: field kernels in CUDA (cupy), compiled once per process |
| `_style` | matplotlib's rcParams defaults; font sizes, colours to bytes, dashes |
| `_layout` | GridSpec / SubplotSpec positions, shrinking a box to an aspect |
| `_constrained` | constrained layout: matplotlib's margins per grid row and column, solved directly |
| `_artists` | Artist base; projecting points and lines (seam-safe); features, lines, markers, scatter, barbs, text |
| `_fields` | grids; pcolormesh/imshow, contourf, contour + clabel |
| `_gridliner` | graticule lines and edge labels |
| `_colorbar` | the colorbar, its axes and the room it takes (make_axes[_gridspec]) |
| `_axes` | MapAxes: projection, extent, the plotting methods |
| `_figure` | Figure: layout, `tight_layout` (a draw that only measures), the draw (CPU or GPU, falling back), savefig |
| `_axesbase` | what map and plain axes share: titles, the field / line / text methods, legends, patches |
| `_cartesian` | plain axes: Axis (scale, locators, ticks, limits, categories, dates), spines, autoscale as matplotlib's, twin / shared axes, the draw in zorder |
| `_plots` | the plain plot types (bar, hist, fills, steps, stems, error bars, ...) and their containers |
| `_stats` | pie, boxplot, violinplot, axline, broken_barh, eventplot, hexbin, hist2d, matshow |
| `_secondary` | inset axes and their zoom indicator; secondary axes |
| `_scale` | linear, log, symlog, logit, function scales (matplotlib's transforms and default locators) |
| `_dates` | dates as day numbers; date locators and formatters (dateutil's rrule ported) |
| `_path` | matplotlib's Path: codes, unit shapes, affines, Bezier flattening (numba) |
| `_markers` | every matplotlib marker as geometry (MarkerStyle's paths and transforms) |
| `_patches` | patches, box styles, fancy arrows and connection styles, as matplotlib computes them |
| `_legend` | Legend: matplotlib's offsetbox packing, 'best' search, handlers |
| `_annotation` | Annotation: text and arrow, matplotlib's coordinate systems |
| `_quiver` | Quiver: matplotlib's arrow polygons, width and scale |
| `_streamplot` | streamplot: matplotlib's mask, seeds and RK12 integrator in one numba kernel (the same trajectories, bit for bit) |
| `_polar` | PolarAxes: matplotlib's polar transform onto the box, theta / radial locators and formatters, grid circles and arcs, labels |
| `_tri` | Triangulation (Qhull through scipy as matplotlib calls it, or Bowyer-Watson in numba), a numba triangle rasterizer (top-left rule), marching triangles joined into polylines; tricontourf / tricontour / tripcolor |
| `_triinterp` | TriFinder (bucketed search, matplotlib's tie rules), linear and cubic (reduced HCT) interpolators, TriAnalyzer |
| `_trirefine` | UniformTriRefiner: a numba kernel per level, matplotlib's points and triangles in its order |
| `_hatch` | hatch patterns (matplotlib.hatch), Agg's tile filled and stroked, repeated from the canvas origin under a shape's coverage |
| `skewt` | MetPy's SkewT on a `projection='skewx'` axes, its Hodograph and colored_line; MetPy's thermodynamics (numba moist adiabats) |
| `collections` | LineCollection made without an axes (`add_collection`) |
| `_rc` | rcParams (matplotlib 3.11 defaults), `rc_context`, style sheets |
| `transforms`, `pyplot`, `ticker`, `dates`, `patches`, `colors`, `cm`, `gridspec`, `style` | the public matplotlib-named modules |

## The kernel image

Starting numba is most of a new process's first figure: importing it (about
220 ms with the import bundle), making the dispatchers (55 ms), and typing
each kernel's arguments and loading it from the cache on its first call
(about 5 ms a kernel, and a first contour plot calls 36 of them). So
`warm()` ends by linking what numba compiled into one file
(`fastplot/_image.py`), and a new process runs the kernels from it with
nothing but numpy and ctypes.

The file holds a unit per kernel and argument types: the kernel's object code
from numba's cache (ELF on every platform) with every kernel it calls, small
entry functions made at build time, and what numba's runtime does for the
kernels (memory, arrays that grow) in a few lines of LLVM. A unit is mapped
into executable memory the first time it is called. Its external calls are
bound as numba's JIT binds them: math to numba's `_helperlib`, the C API to
Python's, and the parallel kernels' scheduling to numba's own threading
layer (OpenMP, TBB or workqueue, whichever numba used), so `prange` runs on
the same thread pool, with the same thread counts, as under numba. Those two
compiled modules of numba's are loaded on their own, without numba's Python
(about 2 ms), and must be the very files the image was built against.

One entry is a function of Python's own (`METH_FASTCALL`): it checks each
argument is exactly what the unit was compiled for (an ndarray's dimensions,
type, layout and flags; a Python or numpy number; None; a tuple of these),
releases the GIL, makes numba's native call and returns numbers itself, so a
call costs about 0.2 µs, against numba's 0.4. Anything else (an ndarray
subclass, keywords, a returned array) goes a slower way that ends in the same
call; returned arrays are numba's: a large one keeps its memory, an argument
handed back is that very object.

The file is named for what the CPU reports through CPUID, the Python and numpy
versions and a hash of fastplot's sources, and carries CRCs. Before one is
used, a process of its own draws `warm()`'s figures through it, without
numba, and must get numba's pixels and PNG files. Whatever a file cannot do
goes to numba as before: no file for this machine (another CPU, an edited
fastplot, not x86-64), a damaged one, another numba, a threading layer asked
for that it was not built with, a kernel or argument types it does not hold.
A process that had to start numba starts building a new file in the
background. `FASTPLOT_IMAGE=0` turns the image off; `FASTPLOT_CACHE_DIR`
says where images are kept (else beside numba's cache in the package, or in
the user's cache).

The image also holds fastplot's kept memory, a few hundred bytes of LLVM: a
freed block of 256 KB or more is kept, up to a 64th of the machine's memory
(32 to 256 MB), and handed out again for a request it fits within a quarter.
The kernels' runtime takes its arrays from it, and a draw or a savefig puts
it in place as numpy's memory handler (NEP 49) for the thread drawing, so a
run of frames stops faulting in fresh pages (see [Speed](speed.md#long-jobs)).
It takes a spinlock only to look through its slots, and may be called from
any thread, with or without the GIL.

## Starting fast

A fresh process imports only what a figure needs:

- Text is measured and drawn with FreeType through `_freetype`, a small
  ctypes binding to the library freetype-py ships. Importing freetype-py
  itself costs 30 to 50 ms (28 modules of constants, `ctypes.util`, and
  `platform.system()`, which starts a process on Windows). Characters'
  hinted advances and ink boxes come from FreeType directly, with Pillow's
  rounding, rather than from Pillow, whose import is about 50 ms: the same
  on every character of the bundled fonts at 28 sizes. Pillow is still
  used for other image formats, and freetype-py for text drawn as a path.
- The bundled `.npz` data is read by `_npz` without `zipfile` (7 ms).
- The PNG encoder's helper threads are its own small pool, not
  `concurrent.futures` (15 ms).
- `json`, `calendar`, `subprocess` and `hashlib` are imported only where they
  are used.
- The import bundle is two files, and numba's half is read only when numba
  is about to be imported.
- The kernel image's name is a hash of the sources, kept beside them with
  every file's size and time, so finding it is one directory listing.
- `numpy.ma` (6.5 ms) is imported only for masked input: a check for a
  masked array asks `sys.modules` first, since a masked array can't exist
  before the module does, and `np.unique` (which imports it) is not used
  on the drawing path.
- While the program reads its data, the preload thread that
  `import fastplot` starts does what the first figure would otherwise wait
  for (`_image.prewarm`): it maps the units that this machine's processes
  have mostly used, starts the kept memory and, where no child process can
  be forked (Windows), numba's thread pool and the parallel units (GNU
  OpenMP, numba's layer on Linux, can't be forked once its threads are
  started, and mapping a parallel unit starts them), opens
  the default font's FreeType face, and starts the PNG encoder's threads.
  Which units count as mostly used is learnt per machine: each process
  that runs from the image adds the units it called to a `.used` file
  beside it, and units used by a quarter of processes or more are mapped,
  most used first, at most 120. A unit a draw wants meanwhile is mapped by
  whichever thread gets there first. Mapping a unit takes about 70 µs: the
  file is read straight into executable memory, and every address in it,
  its own and the external ones, is fixed in one vectorised pass.
- Under the `'auto'` device, cupy isn't imported for a figure that the
  last process found too small for the GPU
  ([CPU, GPU and threads](guide/hardware.md#how-the-settings-are-chosen)).

## Plain axes and Agg's rendering

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
- **Markers in order**: matplotlib composites each marker -- face, then
  edge -- before the next, so overlaps darken and the last drawn is on top.
  A path collection (a scatter of many sizes or colours, `_agg.draw_markers`)
  does every marker exactly, at any number of them: what a path puts down
  depends only on its outline, so each marker's coverage is worked out on
  any core into a mask of its own, by the steps any path takes, and row
  bands blend the masks in order. `_draw.exact_markers` keeps the general
  path, which an outline that leaves its box falls back on, and the two
  are tested to give the same bytes. One shape and colour goes through
  `_draw.stamp_markers`, as draw_markers places each at whole pixels.
- **Meshes** (`_raster`): rows of cells used to run on their own cores and
  raced for a pixel on an edge they share; bands of raster rows now take
  the cells in order, so any number of threads gives what one does. In
  nearest and flat cells, which cell shows at a pixel never depends on a
  value, so from the second frame a table of them stands in for the
  rasterizing (`cells_by_table`).
- **Path simplification** (`_draw.simplify`, Agg's PathSimplifier, the
  same vertices as `Path.cleaned(simplify=True)`): a line of 128 points or
  more drops the points within 1/9 px of the line they extend.
- **Skew** (`projection='skewx'`): MetPy's SkewTTransform, in pixels -- x
  moves by tan(rotation) times the height above the axes' bottom; data and
  the x-axis transform (axvline, x grid) skew, the y-axis one does not; x
  ticks go on each edge where its x range reaches.
- **Layout** is matplotlib's: subplot parameters, gridspecs, colorbars
  (make_axes_gridspec), aspect, `tight_layout` (`_layout.tight_subplotpars`,
  fed by a draw on a canvas that paints nothing), legends (offsetbox
  packing in font-size units, 'best' by the same badness).

## The GPU canvas

Fields (sampling, cell rasterization, colouring, blending) run on the card;
lines, polygons and text run on the CPU. The canvas moves to host
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

## Optimisations that keep the output identical

Every optimization keeps the image (and the PNG) byte for byte: a kernel
that replaces a numpy expression computes it in the same order --
np.cumsum's running sum, numpy's pairwise row sums, np.interp's branches,
libm pow where numpy called it -- and the tests hold each to its numpy
reference. What depends on the CPU's vector code paths or sort order
(hypot, arctan2, argsort) is left to numpy's own calls.

Contour lines of a lon/lat grid on a map that shows part of it are cut
before they are projected (`_contour.crop`). The box is the canvas's
lon/lat extent, a pixel past its edges as matplotlib's clipper cuts, padded
by two grid cells and a degree. Each piece is kept to the first point past
the box at either end, so every segment that can reach the canvas is kept
whole. A ring that loses points is one the clipper would cut open anyway, and
dashes start again where a line comes back onto the canvas, so what is drawn
is the same. Labelled lines are kept whole, since matplotlib ranks every
point of a line for its label. So is a level that a cut would leave short
enough to snap (1,024 points). A pole on the canvas, or every longitude,
means no box.
