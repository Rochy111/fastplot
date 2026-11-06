# Changelog

All notable changes to fastplot. Versions follow
[semantic versioning](https://semver.org/): while fastplot is 0.x, a minor
version (0.3.0) can add features and change behaviour; a patch version
(0.2.1) fixes things.

## Unreleased

## 0.5.6 — 2026-11-05

### Changed

- `pcolormesh` of a grid whose rows bend on the map, a geostationary
  satellite's scan (GOES ABI, Himawari, GK-2A) on PlateCarree, draws its
  cells about 1.5x faster, with the same pixels. Each band of raster rows
  looked at every cell of every row of cells that reached it, and a scan
  row is an arc several bands tall (a GOES CONUS row reached 6.5 of 96
  bands on average); a band now passes over runs of 64 cells that miss
  it. GOES CONUS's 0.5 km grid (6000 x 10000 cells) on a 4000 x 1700
  PlateCarree map, 24 threads: the cells 0.15 s rather than 0.22 s, the
  whole draw 0.36 s rather than 0.42 s. Meshes with straight rows draw as
  before.
- An RGB(A) `imshow` on a map writes its pixels onto the canvas about 4x
  faster, the same bytes: the sampled colours were unpacked with four
  boolean-indexed scatters on one core, and now go through one 32-bit
  integer per pixel. A 4000 x 1700 true colour: 0.06 s rather than 0.23 s.

### Fixed

- On Windows with conda's OpenBLAS (`libblas=*=*openblas`), the kernel
  image left out every kernel that can print, 104 of 303: numba's JIT
  found `printf` in `openblas.dll`, which exports one and is searched
  before the C runtime. So the image was marked failed, and every process
  started numba for its first figure. The C runtime's output functions
  are now bound to msvcrt's, where numba finds them elsewhere; functions
  that return numbers stay where numba found them.
- A program that imported fastplot could wait at exit, with no limit, for
  the thread that gets numba ready in the background: it ran its step
  (numba's import, its threads) to the end first. One such script sat
  for minutes after its last line. That thread now stops between steps
  once the program exits, and is waited for 5 seconds at most.

## 0.5.5 — 2026-11-05

### Added

- `barbs` and `quiver` on a map take cartopy's `regrid_shape` (an int, or
  `(nx, ny)`) and `target_extent`: the vectors are interpolated onto a
  regular grid of the map, as many barbs as cartopy draws (780 for
  `regrid_shape=20` on a global Robinson map, not the grid's 32,768).
  They were accepted and ignored. Which data points feed each new point,
  and with what weights, is worked out once for a map and the data's
  coordinates and kept, so later frames each cost a gather (0.02-0.4 ms).
  Working it out takes under a millisecond for 1-D axes (a 0.25° global
  grid: bilinear, read back on the data's grid) and about 20 ms for a
  400 x 600 grid of 2-D coordinates, linear in the grid's own cells as
  projected (cartopy's Delaunay of the same points took 1 s, every
  frame). 40 Lambert frames of barbs on that grid drew in 0.27 s.
  Scattered points are triangulated as cartopy does. On plain axes
  `regrid_shape` warns that it is ignored.
- `fastplot.save_movie(fn, items, 'out.mp4', fps=...)`: frames drawn as
  `fastplot.frames` draws them, piped to ffmpeg in order as raw RGB, with
  no PNGs in between and only a few frames held at a time.
  `.mp4`/`.mov`/`.mkv` are H.264 (x264's `veryfast` preset unless
  `preset=` says otherwise), `.webm` VP9, `.gif` Pillow's. 40 frames of a
  global Robinson map (200-level contourf, labelled contours, barbs,
  coastlines) came out as an MP4 in 0.32 s, where writing them as PNGs
  took 0.39 s; x264's `medium`, ffmpeg's default, took 0.46 s for a file
  5% smaller.
- `fastplot.animation` for scripts ported from `matplotlib.animation`:
  `FuncAnimation`, `ArtistAnimation`, `FFMpegWriter`, `PillowWriter`,
  `writers`, `save()` and `writer.saving(...)` / `grab_frame()`.
- The install notes say to leave out the `gpu` extra where cupy is
  already installed (conda-forge's): fastplot uses any cupy it can import.

### Fixed

- Longitudes round the whole circle with the first repeated at the end
  (0…360 with a cyclic column, as cartopy's `add_cyclic_point` makes
  them) are a grid that wraps, the repeated column dropped: `contourf`
  left a pixel-wide blank line at the first longitude, and `contour` lines
  were not joined there.
- Contour lines that run west across a map's seam through a point exactly
  on it (a ring round a pole on a 0…360 grid, 180° a grid longitude) were
  cut a point early, and drew a straight line across the whole map.

## 0.5.4 — 2026-11-05

### Added

- Text drawn as matplotlib 3.8 and earlier drew it, for figures that
  should match images made with those versions: set
  `rcParams['text.hinting_factor'] = 8`, `'text.hinting'` to
  `'force_autohint'` and `'text.kerning_factor'` to 0, as they had them.
  Glyphs, widths, line spacing and turned text then match those versions
  to the byte, using matplotlib's own font engine, which must be
  installed. On a set of forecast maps, against the same maps drawn by
  matplotlib 3.8, the mean difference fell from 0.7-4.2/255 to 0.01-3.0
  on one case and from 1.2-3.4 to 0.3-2.5 on the other.

### Faster (every pixel unchanged)

- Dashed grid lines from `ax.grid` (a map's too, at its `set_xticks`) are
  kept across figures as `gridlines` are: a map's ticks and grid cost
  1.1 ms a figure to draw rather than 2.3.
- A forecast hour's barbs and isobars, which its products share, are
  stroked once and their coverage recorded as they are first drawn, on
  every thread; the figures after blend it rather than sweep 17 000
  edges again (barb outlines were stroked afresh in every figure): -0.5
  ms an image on a 39-image set of forecast maps, which drew in 346-379
  ms against 415-430 before (a larger map with counties: 417-440 against
  460-483). Recorded coverage is put in row order by a counting sort
  rather than a stable argsort.
- Text drawn as matplotlib 3.8 drew it no longer brings numba up in a
  process's first figure (its turned labels' kernel is in the kernel
  image as they call it), and a text drawn again where it was reuses its
  mask, as fastplot's own text does.

### Less memory

- A process drawing frame after frame holds much less: a render worker
  drawing a 39-image set of forecast maps 6 times (matplotlib 3.8 text)
  settled at 458 MB against 609 MB on 0.5.4.dev44 and 524 MB on 0.5.3
  (with 10m features and counties: 562 MB against 730 and 653), with its
  first pass 0.5 s rather than 1.5, and drew the set 2-7% faster than
  dev44, every pixel the same:
  - A drawn figure's picture goes as soon as the caller lets go of what
    `buffer_rgba()` gave, rather than when Python's cycle collector next
    runs a full collection (a figure lives in reference cycles; a large
    program carried ~20 dropped pictures, 8 MB each at 1875 x 1125).
  - The kernel image's kept memory gives back, at each frame's end, the
    blocks the frame did not take again: about one frame's let-go is
    kept, not up to 256 MB.
  - Kernel units are laid out at the page size, not 64 KB (a worker's
    ~60 units: 2 MB rather than 8), and the units mapped ahead of the
    first figure are those the program's own earlier runs used, not the
    machine's.
  - contourf's NaN-hole edges work in two kept window buffers instead of
    ~19 MB of new arrays a frame.
  - A map's whole Natural Earth features (~50 MB at 10m with states and
    lakes) go once its projected geometry is kept and no frame has read
    them for 8 frames; a new map decodes them again (30-70 ms).
  - The import bundle lets go of a module's code once it is imported,
    holds only the modules the machine's programs import, and numba's is
    not read when the program imported numba itself.

### Changed

- Everything kept from one figure for the next (projected map geometry,
  drawings kept for their arrays, pixel and cell tables, kept regions,
  NaN-hole triangles, recorded ink) now shares one memory budget per
  process instead of a cap each. The budget is 1% of the machine's
  memory, shared among a `multiprocessing` pool's workers, and can be set
  with `FASTPLOT_CACHE_MB` or `fastplot.set_cache_budget(mb)`;
  `fastplot.cache_info()` shows what is held. When it is full, an earlier
  frame's drawings go before a grid's tables and a map's geometry.
  Drawings that hold the same coordinate arrays now share one copy of
  them, and a kept drawing's memo (its stroked outlines and prepared
  edges, which fill in as it is drawn) is counted. Natural Earth features
  stay kept whole, outside the budget, as before. A forecast-map render
  worker drawing 80 frames with new data each frame settled at about
  433 MB, against 607 MB on 0.5.3 and 346 MB on 0.5.2, at 0.5.3's speed.

## 0.5.3 — 2026-11-04

### Added

- `pcolormesh` takes an `(ny, nx, 3 or 4)` RGB(A) array, float 0-1 or
  uint8, on any grid: each cell painted its own colour as Agg rounds it,
  cells of alpha 0 (or NaN, or masked) left unpainted. `alpha` replaces an
  RGB array's and leaves an RGBA one's, as matplotlib does.
- `fastplot.feature.ShapefileFeature(path)` and `ShapelyFeature(geoms, crs)`:
  geometry of your own (Census counties, TIGER roads) drawn as the Natural
  Earth features are, culled to the map and projected once a map.
  `ShapefileFeature` reads the `.shp` with fastplot's own reader.
- Ticks on maps as cartopy's GeoAxes has them: `set_xticks` /
  `set_yticks` (with `crs=` on rectangular projections), `ax.xaxis` /
  `ax.yaxis` formatters and locators, `tick_params`, and `grid` lines at
  the ticks.
- Lines take `path_effects` (`Stroke`, `withStroke`, `Normal`: a halo
  under a trace), `solid_capstyle`, `solid_joinstyle`, `dash_capstyle`,
  `dash_joinstyle` and `clip_on`, which were accepted and ignored.
- `plt.Circle`, `plt.Rectangle`, `plt.Polygon`; `Axes.set_clip_on`.

### Fixed

- A line's halo (`path_effects`) is drawn, its round caps and joins are
  round, and `clip_on=False` lets it past its axes' box: hodograph insets
  differed from matplotlib's by 0.27/255 on average, now by 0.001.
- `contourf` round NaN holes drew a few pixels differently from run to
  run on more than one core: where two of the holes' edge triangles both
  claimed a pixel on their shared edge, the thread that wrote last won.
  Each core now takes its own rows and every triangle in order, which
  gives the one-core picture on any number of cores (tricontourf too).
- `pcolormesh` cells whose edge lies on a pixel's centre: the pixel goes
  to the cell before the edge (left, or above), as Agg fills a QuadMesh,
  on plain axes and on PlateCarree, Mercator and Miller maps. The edges
  are made as matplotlib and cartopy make them (mid-points in the
  coordinates' own dtype, cartopy's wrap-aware steps; a single row or
  column of centres a cell of no size) and placed by their transforms'
  arithmetic, and an aspect-fitted axes' box has `apply_aspect`'s exact
  corners, so the edges that land on pixel centres are the ones that do
  in matplotlib. A map's box rounds half a pixel up, as Agg's clip box.

### Faster (every pixel unchanged)

- `contourf` of a field with NaN holes (a pressure level below the
  ground) fills round them only in a window about the holes, and keeps
  their triangles while the next frame has the same holes: an 850 hPa
  frame takes 21 ms rather than 25.
- A `pcolormesh` of 2-D coordinates drawn again (the next frame of the
  map) shades its cells and takes their colours to the pixels, rather
  than shading every pixel: a 324 x 359 mesh's frame 8 ms rather than 10.
- Axes are made 40% sooner (the settings read once, the default
  locator's steps shared), and an axis whose ticks are `[]` costs nothing
  to draw: 75 hodograph insets a frame.
- `contourf`'s pass that finds which bands touch each pixel vectorizes:
  4-6x faster, 1.5 ms of a 1875 x 1125 map's frame.
- A field on 1-D coordinates drawn larger than its grid interpolates
  each grid row once for the pixel rows between two, not once per pixel
  row: 1.8x faster sampling.
- Lines of 64 to 127 points (a circle, a hodograph ring) go through the
  one-call stroke kernel whenever one core is the way picked for them, and
  colours already converted are kept: 75 hodograph insets draw ~4 ms
  sooner. Opaque rectangles (an axes' background) fill a pixel a word.
- Anti-aliased edges over opaque pixels (lines, fills, text) blend with
  a multiply where Agg divides, by a table exact for every value: strokes
  5-20% faster.
- Contour lines (with their labels) and barbs drawn by a new figure from
  the same arrays as one before -- a forecast frame's products sharing its
  MSLP contours and wind barbs -- take the kept lines, label places and
  barb shapes (the arrays compared to the bit with kept copies, never
  assumed): each repeat draws 1-2 ms sooner.
- Filled contours and pcolormesh go onto the canvas row by row: values,
  touched bands, colours and the blend in one pass over a three-row
  buffer instead of four passes over the image; NaN pixels are skipped and
  mesh cells coloured once (those in view, where they are no more than the
  pixels: a Landsat band drawn small is coloured a pixel at a time). Fill
  kernels run 2.1-2.3x faster.
- A run of bare inset axes (a hodograph map's 60-75) draws its small
  lines in one stroke call, serial or in parallel groups of insets sharing
  no pixel as this machine times it; axes make their tick defaults only
  when used. Insets draw 2.4x faster.
- The next figure of the same map reuses the colorbar, repeated texts, the
  map's extent, outline and frame: kept pixels are pasted only when every
  input and the pixels under them match to the byte.
- Big images (a 7661 x 7801 Landsat band, i7-13700K, on the CPU): an RGB(A)
  `imshow` is no longer converted whole when made: uint8 RGB and float
  RGB(A) are made bytes pixel by pixel as they are drawn (190 ms off each
  figure: an RGB scene saved at 1500 px in 12 ms rather than 208). The
  array is read when drawn, as a uint8 RGBA one already was. Autoscaling
  finds the finite range in one pass on every core (76 ms to 7 for
  float32, 139 to 9 for float64). `bbox_inches='tight'` measures the
  extents as matplotlib does, with a draw that paints nothing, and paints
  once. An image with more cells in view than pixels (drawn at about its
  own size) is shaded a pixel at a time straight onto the canvas, with no
  value raster or colour layer between. A PNG is written in the pieces it
  is made of. The band saved tight at native size: 121 ms rather than
  289; at 1500 px, tight and transparent: 16 ms rather than 90.

- `set_xlim` / `set_ylim` with both limits no longer work out the old
  ones, which autoscaled over every artist: 75 hodograph insets' limits
  took ~55 ms a frame.
- A map's features and gridlines, drawn again by the next figure of the
  same map, are blended from the coverage they had the time before
  instead of rasterized again: a map with 3,000 county edges draws in 16
  ms rather than 23.
- Ellipses and circles are made without matrix products (4x faster), and
  numeric arrays and lists given to plain axes skip the date and category
  checks.

## 0.5.2 — 2026-11-03

### Fixed

- On Python 3.12 and later, one kernel (the check of whether 2-D
  coordinates are a grid) did not compile: numba found a cycle in how it
  added up its rows. A new install's kernel compiling stopped there, so
  every first figure waited on numba, and `contour`, `contourf` and
  `pcolormesh` of 2-D coordinates raised an error. 0.5.0 and 0.5.1 had
  it; Python 3.11 and earlier were not affected.
- The first frame of a field flipped north to south (as GFS ships it), or
  of an RGB image's channel, on some projections could fall back to numba
  for a second or two even with the kernel image: the field is now passed
  in the one form the image holds.
- On Apple silicon (and other arm64 machines), contour pictures follow
  matplotlib's again. There, contourpy and numpy are compiled with a
  multiply and an add fused into one rounding, so matplotlib closes other
  contour rings and turns some vertical labels the other way up than on
  x86-64. fastplot now asks contourpy and `np.interp` once a process which
  arithmetic they use, and rounds the same way. Nothing changes on x86-64,
  where they are not fused.

## 0.5.1 — 2026-11-03

### Faster (every pixel unchanged)

- On Windows the parallel kernels run on a thread pool of fastplot's own
  rather than numba's OpenMP threads, which spun for ~100 ms after every
  step, so a frame's threads never rested. Its threads wait for as long
  as a wake has lately cost on the machine, then sleep. On an i7-13700K
  a frame takes 39 to 75% less CPU time (a six-panel plot 31 ms rather
  than 120, a GFS map 89 rather than 170) and 0 to 8% less time (the GFS
  map 21.2 ms rather than 22.1). `FASTPLOT_POOL=0` goes back to numba's.
  On Linux `FASTPLOT_POOL=1` runs the pool too: 58 to 81% less CPU than
  GNU OpenMP's spinning threads under WSL, but up to 7% slower on some
  figures, so it is not the default there.
- `fastplot.frames()` finds how to split the batch on your machine: its
  first frames are drawn a few ways (fewer workers on several threads
  each, more on one, or one at a time as a loop would), and the rest the
  fastest way, which later batches of the same function reuse. On an
  i7-13700K, 120 frames of a GFS map took 13.7 ms a frame rather than 15.4,
  a line plot of 100,000 points 3.8 rather than 4.7 (7.4 in a loop), and a
  six-panel plot, mostly Python, is drawn one at a time: after the first
  batch as fast as a loop (14.4 ms against 14.2), where four at a time took
  15.4. Giving `workers` or `threads` uses those.
- `fastplot.frames()` keeps the split it learns for a function, so the
  next process on the same machine starts with it, rather than trying a
  few on its first frames. The splits are kept for this CPU and GPU and
  this fastplot only, are tried again when one draws far below its rate
  (other data, a busier machine), and fill one small file in fastplot's
  cache folder, the 64 functions learnt last. A GFS map's first batch in a
  new process took 11.9 ms a frame rather than 13.1.
- An axes' ticks, grid lines and spines, a legend's handles, and other
  short solid lines are snapped, checked and stroked in one kernel call
  rather than a dozen numpy steps each, and a text's colour is taken as
  it comes: six panels with legends took 12.9 ms rather than 14.4, 16
  shared panels 17.0 rather than 19.8, a figure of texts 3.3 rather than
  3.6, a Skew-T 5.8 rather than 6.2.
- Gouraud shading on a mesh (`pcolormesh(..., shading='gouraud')`, a
  quad's four triangles each) works out its tens of thousands of triangles'
  shapes on every core, when this machine has timed that quicker: a panel
  of 64,000 triangles took 20.7 ms rather than 28.7.
- A map whose projection has an edge of its own (Robinson, Lambert
  azimuthal, Albers, polar, Mercator in a box) cuts that edge to the axes
  in a kernel, not in Python: 80 us rather than 2.7 ms an axes. Nine such
  maps in one figure took 12.5 ms rather than 32.7, a Robinson map 5.8
  rather than 11.4.
- Shared axes find each artist's data bounds, and each share group's
  limits, once a layout pass rather than once an axes: 16 shared panels
  with `tight_layout` took 21.1 ms rather than 29.5.
- Colours given row by row (a streamplot's, a scatter's) are taken as one
  array, and a streamplot's colours and widths are read off its field once
  for every line, its start points found by a kernel: streamlines with
  quivers and barbs on a map took 10.1 ms rather than 12.2, three
  streamplots 16.8 rather than 19.1, a scatter of colours 7.8 rather than
  8.5.
- A new process's first figure on Windows no longer spends 25 ms asking
  WMI for the CPU's architecture: a line plot's first image took 209 ms
  rather than 230.
- Gouraud-shaded `tripcolor` works out each triangle's shape once and draws
  bands of rows on every core, each band taking the triangles in order: a
  figure of triangulations (filled and line contours, two `tripcolor`, a
  `triplot`) takes 11.5 ms rather than 18.3. Line contours on a
  triangulation are joined in a kernel, and `Triangulation.edges` is found
  4 times sooner.
- Contour labels on a global map take 1.4 ms less a frame (8%): the lines
  kept whole are copied once, by a kernel, not gathered by index and
  joined.
- Texts drawn again at the same place, such as tick and gridline labels
  in a later figure, reuse their layout: a six-panel line plot takes 5%
  less.
- A new install no longer starts numba the first time a script uses a
  triangulation's interpolators or refiner, half-filled markers, or lines
  given in another coordinate system on a cylindrical map: `warm()` now
  compiles them too. The first such figure took 0.5 to 0.9 seconds more
  than it does now (3 seconds where numba also had to compile).

### Faster PNGs with the `png` extra (the same pixels, other bytes)

- Where the `isal` package is installed (`pip install fastplot[png]`),
  PNGs are deflated by Intel's ISA-L rather than zlib. On an i7-13700K a
  500 x 500 pcolormesh took 2.9 ms rather than 4.1, with 71% less CPU
  time; a pcolormesh map 6.0 rather than 7.9; six panels 12.1 rather than
  13.1; a 4K map 40.4 rather than 42.4. The decoded pixels are the same,
  but the files are other bytes and up to 7% larger.
  `FASTPLOT_DEFLATE=zlib` keeps zlib's. fastplot never needs it: without
  it nothing changes.

### Fixed

- Drawing figures from several threads at once, as `frames()` does, could
  crash the process (an access violation, about one run in ten in a test
  of 8 threads) the first time each kernel ran. Two threads setting up the
  same kernel's fast entry, or numpy's kept-memory handler, at once could
  each make one, and the record the first relied on was freed. Each is now
  made once. Making the default font's FreeType face ahead of time also
  waits for any glyph being drawn.
- `contour` and `contourf` of 2-D coordinates no longer spend about 0.8 ms
  each checking whether those coordinates are a grid. That check ran on
  all of numba's threads where a draw uses fewer, and changing numba's
  number of threads costs more than the check. It now runs at a draw's
  count, on one thread or all of them as is faster. A small contour plot
  took about 40% longer in 0.5.0 than in 0.4.0, and is now faster than in
  0.4.0, and a Lambert map with 2-D coordinates takes 14.4 ms rather than
  16.5.
- Choosing between one thread and all of them no longer depends on a
  thread start being cheap. A size that keeps running on all threads gets
  one try on a single thread once it has run as many times as there are
  threads, so a machine whose threads are slow to wake finds the faster way
  too.

## 0.5.0 — 2026-11-03

### Faster (every pixel unchanged)

- A new process's first figure no longer waits for numba: `warm()`, and so
  `pip install`, now also links the compiled kernels, the parallel ones
  included, into one file for your CPU, which later processes run them from
  without importing numba. With the import changes below, a script that
  draws one figure gets its first image in 0.21 to 0.29 seconds rather than
  0.6 to 0.95: a line plot 210 ms rather than 703, a Lambert CONUS map with
  contours, labels and states 282 rather than 945.
  Kernel calls also cost half what they did, so small figures redrawn in a
  long-running process are no slower, and often a little faster.
- Wherever that file cannot serve (another CPU, an edited fastplot, an
  upgraded numba, a kernel it does not hold, a damaged file, not x86-64),
  fastplot uses numba as before, and rebuilds the file in the background
  after the first such figure. `FASTPLOT_IMAGE=0` turns it off;
  `FASTPLOT_CACHE_DIR` says where it is kept (else beside numba's cache in
  the package, or in your user cache).
- A colorbar with `extend` on a mesh, and meshes through `LogNorm`,
  `TwoSlopeNorm`, `BoundaryNorm` or `SymLogNorm`, no longer compile a kernel
  the first time they are drawn after installing: `warm()` now covers them.
- `import fastplot` takes about 25 ms rather than 90, and a first figure's
  draw and save another 50 to 60 ms less: text no longer imports Pillow or
  freetype-py (FreeType is called directly, with the same results), the
  bundled data is read without `zipfile`, the PNG encoder's threads no
  longer import `concurrent.futures`, and `numpy.ma` is imported only for
  masked input. While a script reads its data, fastplot gets ready on a
  background thread what this machine's scripts have mostly needed first.
- PNG pieces are compressed at zlib memory level 7 rather than 9: less
  state to set up for each of the encoder's pieces, so it's faster. The
  bytes of a file change; the decoded pixels are identical.
- In a long run of frames, the large arrays a draw lets go of are handed to
  the next draw instead of back to the system, which on Windows makes each
  fresh page cost a fault: 4 to 7% off a frame of the larger maps.
- Contour labels on maps place a third faster (2 ms of a global map's frame).
- On a machine with a CUDA card and cupy, the default `'auto'` device no
  longer starts the GPU (0.4 to 0.8 s) for every figure of 500x500 pixels
  or more only to find it too small for the card. What the card can do is
  kept from the last process, so smaller figures never import cupy: a
  process's first line plot on an RTX 5070 Ti went from 417 ms to 30 ms,
  and a first contour plot from 453 ms to 39. Bigger figures stay on the
  CPU until the time spent on them would have paid for starting the GPU.
  Once started, the GPU is timed against the CPU, and if it isn't faster
  `'auto'` stops sending figures of that size to it, here and in later
  processes. See
  [CPU, GPU and threads](https://rochy111.github.io/fastplot/guide/hardware/).
- `fastplot.warm()` waits for a kernel file another process is already
  building instead of returning before it is there.
- Scatter, quiver and collection colours given as float64 through
  `LogNorm`, `TwoSlopeNorm`, `BoundaryNorm`, `SymLogNorm` or `NoNorm`, and
  lines or points of 16,384 or more transformed from one projection to
  another, no longer compile a kernel the first time after installing.
- Contour lines of a global field on a map of part of the world are cut to
  what can reach the image before they are projected and stroked: on a
  Lambert CONUS map of a GFS field 9 lines in 10 are never projected, and a
  frame took 17.5 ms rather than 19.8 (the first 73 rather than 91).
  Lines with labels are kept whole, as matplotlib ranks every point of them.
- A map's first contour labels no longer import contourpy (8 ms).
- Whether a step runs on one thread or all of them is no longer set by
  fixed sizes. fastplot times the steps as it draws and keeps the faster
  way for each size on your machine. This covers projecting points, cutting
  lines at the canvas and a map's seam, dashing, stroking and filling.
  Results are the same to the bit either way. On a Lambert CONUS map of a
  GFS field, two projections of 15,500 points had been running on one
  thread for 1.3 ms each where all of them take 0.1 ms. See
  [CPU, GPU and threads](https://rochy111.github.io/fastplot/guide/hardware/#how-the-settings-are-chosen).
- Contours and filled contours of 2-D coordinates check whether they're a
  grid of 1-D axes in one pass on every core, 2 ms sooner on a 1799x1059
  grid. Telling which contour rings matplotlib leaves open also runs on
  every core.
- Lambert CONUS frames from a GFS field now take 15.6 ms (19.8 before the
  changes above), and with 2-D coordinates 16.5 ms (18.5).

### Fixed

- Maps of part of the world in a Lambert conformal (or another conic)
  projection no longer draw long straight lines across the map from small
  contours on the far side of the earth, where the projection is cut;
  cartopy draws none. (Labelled contours still can: they are drawn whole.)
- On a map whose edge cuts some contour lines (Robinson, conics), the
  rings contourpy leaves open are drawn open, with their ends
  capped, as matplotlib draws them; they were drawn closed.

### Changed

- `pip install` compiles the kernels for your CPU, so the first figure is
  already at full speed and `python -m fastplot warm` is no longer needed. A
  new install's first map takes 2 to 3 seconds rather than 16. The install
  takes about 30 seconds longer, and building needs numba.
- When the compiled kernels don't match the machine, for example after a
  numba upgrade, in an editable install or on a different CPU, importing
  fastplot compiles them again in a low-priority background process.
  `FASTPLOT_AUTOWARM=0` turns this off, and `FASTPLOT_BUILD_KERNELS=0` skips
  compiling during the install.

## 0.4.0 — 2026-10-30

Maps now line up with cartopy's and figures come very close to matplotlib's,
with a port of Agg's rasterizer, cartopy's Earth and its gridline labels, and
fresh scripts start in about half the time. Projected x/y values change by
about 0.2%, and map shapes from `set_extent` can change by several percent.

### Added

- `crs.Globe`, and `globe=` on every projection, as in cartopy: a model's
  own Earth, such as HRRR's sphere of 6371229 m.
- `\left` and `\right` delimiters sized to their contents.
- `\mathbb`, `\mathcal`, `\mathfrak` and `\mathscr` in every letter and
  digit, and `\mathtt`, from the fonts matplotlib uses.
- DejaVu Serif and DejaVu Sans Mono are bundled, so `font.family='serif'`
  and `'monospace'` work without matplotlib installed.
- `mathtext.fontset`: `dejavusans`, `dejavuserif`, `stix`, `stixsans` and
  `cm`, with matplotlib's fonts, glyph rules and constants. The fonts are
  bundled.
- `Gridliner` labels follow cartopy: rotated along curved gridlines on conic
  and polar maps, inline where cartopy puts them inline, and thinned where
  they would overlap.
- `mathtext.default`, the font math starts in, including `regular`.
- Figures of different `mathtext.fontset` values can be drawn on several
  threads at once.

### Changed

- **Faster.** A fresh process that draws one figure takes 35 to 50% less
  time: a line plot went from about 1.4 to 0.7 s. New frames on a 2-D grid
  such as HRRR's take a third to two thirds less, and six-panel figures about
  10% less. Details are on the
  [speed page](https://rochy111.github.io/fastplot/speed/).
- On a hybrid CPU under Windows, fastplot asks Windows at the first draw not
  to move the process to the efficiency cores, which doubled frame times
  after about two seconds in a batch job. Set `FASTPLOT_ECOQOS=1` to leave
  the process as it was.
- `fastplot.warm()` also writes one file holding the compiled modules of
  fastplot, numba, llvmlite, freetype and Pillow, read at import; a run that
  finds none writes it as it exits, once. Set `FASTPLOT_BUNDLE=0` to import
  the usual way.
- `frames` uses 2 numba threads per frame on 8 performance cores, not 3.
- New figures of lines and barbs take about 17% less time than in 0.3.0, and
  scatters and most maps take about the same with far closer output. Skew-T
  diagrams take about 10% longer. Details are on the
  [speed page](https://rochy111.github.io/fastplot/speed/).
- `rcParams['fastplot.exact']`: scatters of 2,000 markers or more are placed to
  1/8 of a pixel by default, from masks drawn by the Agg rasterizer, which takes
  about a third of the time. Set it to `True` to place every marker exactly.
- Clipping, dashing and the seam split of map lines run on every core, grid
  labels are placed once a view, and Skew-T adiabats are worked out once.
- **Lines and patches bend on maps as cartopy bends them.** Long segments
  follow `project_linear`'s chords, straight lines of the source or, from
  `Geodetic`, geodesics of its ellipsoid. A line from a `PlateCarree` runs
  on through its degrees: a step from 170 to -170 now crosses the whole map
  as it does in cartopy, where it used to take the short way over the seam.
  Use `Geodetic` for the short way.
- **Text is blended as Agg blends it**, on 8 bits, with overlapping glyphs in
  turn, and now matches matplotlib byte for byte.
- `contourf` follows contourpy and Agg round masked points, and translucent
  bands show the darker seams matplotlib's do. `tricontourf` fills every
  pixel its hull touches.
- Contour lines keep the field's precision. A node exactly on a level counts
  as under it, as in contourpy, and a ring whose ends differ in the last bit
  stays open.
- **Lines and polygons are drawn by a port of Agg's rasterizer.** Stroking,
  dashes, curve flattening, clipping and coverage follow Agg vertex by
  vertex, so strokes, patch corners and dense lines match matplotlib to the
  pixel, in about the time lines took before, and less for bars, contours
  and dense lines.
- Markers are drawn as matplotlib's `draw_markers` draws them: each marker's
  snap threshold, cap and join.
- Patches mitre their corners, spines have projecting caps, and each patch
  and bar is filled and edged in turn.
- Map features, coastlines and borders are stroked, snapped and clipped as
  cartopy's are, and polygons are drawn one geometry at a time.
- Meshes on maps that lean off the pixel grid follow matplotlib's QuadMesh,
  in array order, from 1-D as well as 2-D coordinates. From its second frame
  a mesh keeps which cell shows at each pixel, so one of 1.9 million cells
  redraws in about 5 ms.
- Text used as a marker (`'$x$'`) is drawn from outlines, as matplotlib's
  TextPath is.
- **Projections use cartopy's Earth.** Lambert, Albers, Mercator,
  Stereographic and LAEA are on the WGS84 ellipsoid, and Orthographic,
  Robinson and Miller on a sphere of 6378137 m, where before all used a
  sphere of 6370997 m. Maps now line up with cartopy's, but projected x/y
  values change by about 0.2%. Pass `radius=` to keep a sphere.
- A rectangular map frame is snapped to pixels, as Agg snaps it.
- `set_extent` follows cartopy's tracing of the box. A Lambert box whose
  curved side stays within the projection's threshold now uses its corners,
  which can change a map's aspect ratio by several percent: a 29-degree
  wide CONUS box's axes went from 0.844 to 0.886 of the figure (cartopy
  0.889).
- `pcolormesh` cells that lean off the pixel grid are filled as Agg fills
  them: corners rounded to whole pixels, then every pixel touched, later
  cells on top. Grids aligned with the map keep the faster path unless the
  grid's edge shows.
- Gouraud meshes and `tripcolor` grow by half a pixel and antialias their
  outline, as Agg draws them.

### Fixed

- `pcolormesh` with 2-D coordinates on their own projection, such as an HRRR
  grid on its Lambert map, now reaches half a cell past its outermost nodes
  where the grid ends inside the map. It stopped short of cartopy's edge, by a
  few pixels on a coarse grid.
- The canvas is the same size as matplotlib's when the figure's size falls just
  under a whole pixel: 9.37 inches at 100 dpi gives 937 pixels, not 936.
- A mesh with pixel centres exactly on the edge between rows of cells, or one
  folded over itself, could come out differently from one run to the next and
  from one drawn on a single thread. Later cells now always win, as in Agg.
- `shading='nearest'` on 2-D coordinates reaches half a cell past the
  outermost nodes in the axes limits, as matplotlib's does.
- A `\mathcal`-style letter followed by a script no longer raises an error.
- Contour labels on maps land where cartopy's do, keep the angles of the
  box before its aspect is applied, and are clipped to the map.
- `axhspan`, `axvspan` and `fill_between` take `color` for their edge too,
  and bar error bars default to black.
- Primes, integrals' limits, overlines and rotated fraction and radical bars
  in mathtext match matplotlib.
- `axis('equal')`, `autoscale(False)` and `axline` no longer lose or skip a
  pending autoscale, and a colorbar's `remove()` keeps its axes in place.
- Symlog colorbars use matplotlib's tick format, and inset indicators
  connect the corners matplotlib connects.
- Contour levels of rectilinear lines are snapped to pixels, as Agg snaps
  them, and contour labels that tie on symmetric data land where
  matplotlib's do.
- A scale change that autoscales settles the other axis too, as matplotlib's
  does, so `set_xscale('log')` then `set_yscale('symlog')` keeps its limits.
- A top axis label sits on its baseline.
- Legend rows take their own text's descent, and the frame and shadow are
  the patches matplotlib draws.
- Polar artists stop at the circle.

## 0.3.0 — 2026-10-30

Figures come much closer to matplotlib 3.11's, pixel for pixel, and many
more of matplotlib's plot types and layout tools are there.

### Added

- `layout='constrained'` (and `fig.set_layout_engine`): matplotlib's
  constrained layout, colorbars included, with `supxlabel` and `supylabel`.
- Plot types: `pie`, `boxplot` / `bxp`, `violinplot`, `axline`,
  `broken_barh`, `eventplot`, `hexbin`, `hist2d`, `matshow`.
- Inset axes (`inset_axes`, `indicate_inset_zoom`) and secondary axes
  (`secondary_xaxis`, `secondary_yaxis`).
- `fastplot.lines.Line2D` and `ax.add_line`; `PowerNorm`, `FuncNorm`; the
  asinh scale.
- Contour sets have `get_paths()`, `allsegs` and `allkinds`, as
  matplotlib's do.
- pyplot: `subplot2grid`, `figaspect`, `setp`, `getp`, `imread`, `imsave`,
  `sci`, `clim`, `fig.canvas`, `fig.autofmt_xdate`.
- Mathtext: `\sqrt[n]`, `\overline`, `\underline`, accents, `\mathbb` and
  the other letter styles, `\lim`-style operators with limits. Large
  operators come from DejaVu Sans Display, and radicals from the STIX size
  fonts, as in matplotlib. Both are bundled.

### Changed

- Text is rendered by FreeType glyph by glyph at matplotlib's positions,
  as Agg draws it, so plain and rotated text match matplotlib to the
  pixel. **freetype-py is now a dependency.** Drawing text-heavy figures
  is also faster on a cold start.
- Contour lines are traced in contourpy's order and direction: on plain
  axes `clabel` puts labels where matplotlib does, and dashed contours
  start their dashes at the same point.
- `contourf` boundary pixels go to the higher band, as Agg fills the
  polygons.
- Bilinear `imshow` fills the image out to its edges.
- Gouraud shading (`pcolormesh`, `tripcolor`) interpolates colours, not
  values.
- Rotated tick labels stay centred under their ticks
  (`rotation_mode='default'`), as matplotlib's do.

### Fixed

- Many small differences found by drawing hundreds of figures with both
  libraries, among them:
  - autoscaling of empty, tight and aspect-locked axes;
  - the legend's `'best'` location;
  - marker centring;
  - masked data in `hist`, `errorbar`, `fill_between` and `step`;
  - colorbars on log, symlog and two-slope norms;
  - titles over deep mathtext, which are now raised;
  - `fmt='%1.1f'` contour labels, which used a Unicode minus.

## 0.2.2 — 2026-10-29

Found while porting the HRRR and RTMA map pipelines from cartopy and
matplotlib.

### Added

- `fig.add_artist`: patches drawn on the figure, in `fig.transFigure`
  unless their transform says otherwise.
- `Artist.set(**kw)`, `get_zorder`, `get_visible`; `Patch.set_transform`,
  `get_transform`, `set_hatch`; `set_zorder` / `get_zorder` on every kind
  of axes.
- On map axes, patches and lines in `ax.transAxes`, `fig.transFigure` or
  another transform are drawn there instead of projected.
- The maps guide covers a colorbar in axes of your own, and boxes in axes
  or figure coordinates.

### Changed

- `fig.colorbar(cax=...)` turns `cax` itself into the colorbar's axes
  (`cb.ax is cax`), so a title set on it is drawn.
- The figure draws its children by zorder, as matplotlib sorts them.

## 0.2.1 — 2026-10-29

### Added

- RGB and RGBA `imshow` with a transform on map axes (it raised
  `NotImplementedError`).

### Fixed

- A global field ending at 180° E autoscaled a Robinson map to stop near
  90° E.

## 0.2.0 — 2026-10-29

The first tagged release.

- Maps with cartopy's interface: twelve projections; `pcolormesh`,
  `contourf`, `contour` and `clabel`; barbs, quiver and streamlines;
  Natural Earth features; labelled gridlines.
- Plain axes with matplotlib's interface: lines, scatter, bars,
  histograms, fills, error bars, log and date axes, twin and shared axes,
  legends, annotations, patches, polar axes, triangulations.
- MetPy's `SkewT` and `Hodograph`.
- `rcParams`, matplotlib's style sheets, colormaps and norms.
- Drawing on every CPU core with numba, or with CUDA on an NVIDIA GPU;
  `fastplot.frames()` for many figures at once; `python -m fastplot warm`
  to compile the kernels ahead of time.
- The version comes from git (setuptools-scm), so
  `pip install -U git+https://github.com/Rochy111/fastplot` upgrades.
