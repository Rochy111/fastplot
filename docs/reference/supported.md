# What is supported

A quick list of what fastplot implements. The guide pages explain each area
in more detail.

## Figures and output

| | |
|---|---|
| Creating | `figure`, `subplots`, `subplot_mosaic`, `add_subplot` (with `GridSpec`), `add_axes` |
| Layout | `subplots_adjust`, `tight_layout`, `suptitle`, `fig.text`, `fig.legend` |
| Saving | `savefig` to PNG, JPEG, WebP and other Pillow formats; PDF, SVG and EPS holding the drawn picture; `bbox_inches='tight'`, `dpi`, `transparent` |
| Arrays | `fig.draw()` returns an RGBA numpy array |
| Colorbars | `fig.colorbar` with matplotlib's placement, ticks, format and extensions |

## Maps

| | |
|---|---|
| Projections | `PlateCarree`, `Mercator`, `Miller`, `LambertConformal`, `AlbersEqualArea`, `Stereographic`, `NorthPolarStereo`, `SouthPolarStereo`, `Orthographic`, `LambertAzimuthalEqualArea`, `Robinson`, `RotatedPole` |
| Area | `set_extent`, `set_global`, or fitted to the data |
| Fields | `pcolormesh` (nearest, flat, gouraud), `imshow`, `contourf`, `contour`, `clabel`, on 1-D axes or 2-D curvilinear grids |
| Vectors | `barbs`, `quiver`, `quiverkey`, `streamplot` |
| Features | `coastlines`, `add_feature` with `COASTLINE`, `BORDERS`, `STATES`, `LAKES`, `LAND`, `OCEAN`, `RIVERS`, `COUNTIES`, `NaturalEarthFeature` |
| Gridlines | `gridlines(draw_labels=True)`, `xlocs`, `ylocs` |

## Plain axes

| | |
|---|---|
| Lines and points | `plot`, `scatter`, `step`, `stairs`, `stem`, `errorbar`, `hlines`, `vlines`, `axhline`, `axvline` |
| Areas and bars | `fill_between`, `fill_betweenx`, `stackplot`, `axhspan`, `axvspan`, `bar`, `barh`, `bar_label`, `hist` |
| Fields and vectors | `pcolormesh`, `imshow`, `contourf`, `contour`, `clabel`, `quiver`, `barbs`, `streamplot` |
| Triangles | `tricontourf`, `tricontour`, `tripcolor`, `triplot`; `fastplot.tri` with `Triangulation`, `TriFinder`, `LinearTriInterpolator`, `CubicTriInterpolator`, `UniformTriRefiner`, `TriAnalyzer` |
| Shapes and text | `add_patch` (`fastplot.patches`), `text`, `annotate`, `legend`, hatching, text outlines with `fastplot.patheffects` (`withStroke`, `Stroke`, `Normal`) |
| Scales | linear, log, symlog, logit, function; dates; categories |
| Axes | `twinx`, `twiny`, shared axes, inversion, aspect, margins, spines, `tick_params`, minor ticks, grids |
| Polar | `projection='polar'`, full circle, log radius |

## Meteorology

| | |
|---|---|
| Skew-T | `fastplot.skewt.SkewT` with MetPy's methods |
| Hodograph | `fastplot.skewt.Hodograph` with MetPy's methods |
| Thermodynamics | `dry_lapse`, `moist_lapse`, `saturation_vapor_pressure`, `saturation_mixing_ratio`, `dewpoint_from_vapor_pressure` |

## Style and colour

| | |
|---|---|
| Settings | `rcParams`, `rc_context`, `rcdefaults`, `style.use` with matplotlib's 31 style sheets |
| Colormaps | every matplotlib colormap, `ListedColormap`, `LinearSegmentedColormap.from_list` |
| Norms | `Normalize`, `LogNorm`, `BoundaryNorm`, `TwoSlopeNorm`, `CenteredNorm`, `SymLogNorm`, `NoNorm` |
| Interop | matplotlib colormaps, norms, tickers and date locators are accepted as they are |

## Not there yet

- 3-D axes (`mplot3d`).
- Partial polar wedges (`set_thetamin`, `set_thetamax`).
- Vector output: PDF, SVG and EPS contain the picture, not paths.
