# Changelog

All notable changes to fastplot. Versions follow
[semantic versioning](https://semver.org/): while fastplot is 0.x, a minor
version (0.3.0) can add features and change behaviour; a patch version
(0.2.1) fixes things.

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
