# Changelog

All notable changes to fastplot. Versions follow
[semantic versioning](https://semver.org/): while fastplot is 0.x, a minor
version (0.3.0) can add features and change behaviour; a patch version
(0.2.1) fixes things.

## Unreleased

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
