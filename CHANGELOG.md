# Changelog

All notable changes to fastplot. Versions follow
[semantic versioning](https://semver.org/): while fastplot is 0.x, a minor
version (0.3.0) can add features and change behaviour; a patch version
(0.2.1) fixes things.

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
