# Plots and charts

Non-map figures use `fastplot.pyplot`, which follows matplotlib's pyplot:
the current figure and axes, `plt.subplots`, every `Axes` method listed
below, `rcParams` and the style sheets. They are drawn the way matplotlib's
Agg backend draws: lines snapped to pixels, the same caps and joins, the
same path simplification, ticks, labels and layout.

```python
import fastplot.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
ax.plot(x, y, 'o-', label='observed')
ax.set_xlabel('Distance (km)')
ax.legend()
fig.tight_layout()
fig.savefig('profile.png')
```

## What you can draw

| Kind | Methods |
|---|---|
| Lines and points | `plot` (format strings, every marker), `scatter`, `step`, `stairs`, `stem`, `errorbar`, `hlines`, `vlines`, `axhline`, `axvline` |
| Areas | `fill_between`, `fill_betweenx` (with `where`, `interpolate`, `step`), `stackplot`, `axhspan`, `axvspan` |
| Bars and counts | `bar`, `barh` (error bars, `bar_label`), `hist` (bar, step, stepfilled; stacked, density, cumulative) |
| Fields | `pcolormesh`, `imshow`, `contourf`, `contour`, `clabel` |
| Vectors | `quiver`, `quiverkey`, `barbs`, `streamplot` |
| Triangulations | `tricontourf`, `tricontour`, `tripcolor`, `triplot`, with `fastplot.tri` |
| Shapes | `add_patch` with `Rectangle`, `Circle`, `Ellipse`, `Polygon`, `Wedge`, `Arc`, `RegularPolygon`, `FancyBboxPatch`, `FancyArrowPatch` |
| Text | `text`, `annotate` (every arrow and connection style), `legend`, `fig.legend`, `suptitle` |

Hatching works on bars, fills, patches and `contourf(hatches=...)`.

## Axes and scales

- Linear, `log`, `symlog`, `logit` and function scales, with `semilogx`,
  `semilogy` and `loglog`.
- Dates from `datetime`, `numpy.datetime64` or pandas, with
  `fastplot.dates` locators and formatters.
- Categories from strings.
- `twinx`, `twiny`, `sharex`, `sharey`, inverted axes, aspect ratios,
  margins and sticky edges.
- `fastplot.ticker` locators and formatters. matplotlib's own ticker and
  date objects also work.
- Spines, `tick_params`, minor ticks and grids.

## Layout

`subplots`, `subplot_mosaic`, `GridSpec`, `add_axes`, `subplots_adjust`
and `tight_layout` place axes with matplotlib's arithmetic, so a figure
keeps its proportions when you switch libraries.

## Polar axes

`fig.add_subplot(projection='polar')` gives matplotlib's polar axes for the
full circle. `set_theta_zero_location`, `set_theta_direction`,
`set_rlim`, `set_rorigin`, `set_rlabel_position`, `set_thetagrids` and
`set_rgrids` are all there, and a log radius works. Bars on polar axes make
wind roses; `pcolormesh` and `contourf` draw radar sweeps.

## Triangulations

`fastplot.tri` matches `matplotlib.tri`: `Triangulation`, `TriFinder`,
`LinearTriInterpolator`, `CubicTriInterpolator`, `UniformTriRefiner` and
`TriAnalyzer`. With scipy installed, triangulations are the same triangles
in the same order as matplotlib's. Without it, fastplot uses its own
Delaunay triangulation.

## Style

```python
plt.style.use('ggplot')
with plt.rc_context({'font.size': 9, 'axes.grid': True}):
    fig, ax = plt.subplots()
```

All 31 of matplotlib's style sheets ship with fastplot, and `rcParams`
takes the same keys.

## Colours

Every matplotlib colormap and named colour, `ListedColormap`,
`LinearSegmentedColormap.from_list`, and the norms `Normalize`, `LogNorm`,
`BoundaryNorm`, `TwoSlopeNorm`, `CenteredNorm`, `SymLogNorm` and `NoNorm`.
`extend` and the under, over and bad colours work as in matplotlib.
