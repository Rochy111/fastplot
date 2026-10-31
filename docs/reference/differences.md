# Differences from matplotlib

fastplot aims to produce the figure that matplotlib and cartopy would. The
test suite draws the same figures with both libraries and compares axis
limits, ticks, labels, legend and layout geometry, and pixels. The
differences below remain, either by design or because matching them would
cost more speed than they are worth.

## Maps

| Topic | Behaviour in fastplot |
|---|---|
| Datums | `Globe` sets the Earth's shape, but `datum`, `towgs84` and `nadgrids` are accepted and not used: fastplot does no datum shifts. Named ellipses are `WGS84`, `GRS80` and `sphere`; give any other by its axes. |
| Map edges | Lines and patches bend as cartopy's `project_linear` bends them, inside the map. A line leaving a projection's domain, behind the globe on Orthographic say, stops at its last projected vertex, where cartopy bisects to the edge. |
| Large meshes | A curved or leaning `pcolormesh` of 16 million cells or more is sampled per pixel rather than drawn cell by cell. |

## Approximated for speed

fastplot draws to match matplotlib's pixels. One thing is drawn approximately by
default, because matching it exactly takes more than twice as long and the
difference can't be seen at normal zoom. To turn it off:

```python
fastplot.rcParams['fastplot.exact'] = True
```

| Topic | Behaviour in fastplot |
|---|---|
| Large scatters | A scatter of 2,000 markers or more, with colours or sizes that vary, has each marker's position rounded to 1/8 of a pixel. If sizes vary, its size is rounded to 1/8 of a pixel too. Across the figure, pixels are on average 0.2 to 1.2 levels of 255 from matplotlib's, 38 at most in the cases measured, all of it at marker edges. No marker moves further than 1/16 of a pixel, and order, colours and transparency are unchanged. Squares, crosses and other rectilinear markers stay exact, as does any scatter with markers under 1 px across, or with too many sizes for its number of markers. |

## Plain axes

| Topic | Behaviour in fastplot |
|---|---|
| Text | Each glyph is rendered by FreeType (freetype-py) at its exact position and angle, as matplotlib 3.11's Agg renders it, and blended as Agg blends it, so plain and rotated text match matplotlib to the byte. Without freetype-py installed, glyphs are rendered on whole pixels and shifted, which makes edges slightly crisper than matplotlib's and rotates text at other angles as an image. |
| Scatter hatches | `scatter(hatch=...)` is not supported. |
| Contour lines | Lines, labels and dashes follow contourpy to the last bit, down to which rings it leaves open. Labels that tie on symmetric data need contourpy installed, as matplotlib's does, to settle the tie as matplotlib does. `contour` leaves out a cell with a masked corner, where contourpy keeps its triangle, and on integer data a ring through nodes exactly on the level can start at another vertex. |
| Filled contours | A band's edge is found from each pixel's value, so on a coarse grid it can curve across a cell where matplotlib's is straight, by under a pixel. Pixels where three bands meet can take a neighbouring band. |
| Mathtext | `mathtext.fontset` `custom` is not supported and gives `dejavusans`. Sizes of scaled delimiters and radicals are rounded differently, by under 0.1 px. |

## Polar axes

Only full circles are supported; `set_thetamin`, `set_thetamax` and
`set_thetalim` raise `NotImplementedError`.

## Output formats

PDF, SVG and EPS files contain the rendered image, losslessly at the
`savefig` dpi, rather than vector paths.

## Threads

Drawing figures from several threads at once needs numba's OpenMP or TBB
threading layer, which the standard wheels for Linux and Windows include.
numba installed with pip on Apple silicon has only the `workqueue` layer,
which cannot run parallel code from two threads at the same time. fastplot
detects this and takes a lock, so concurrent drawing is safe there but runs
one figure at a time.

## Floating point

fastplot's arithmetic gives identical results on every platform. numpy and
matplotlib are compiled with fused multiply-add instructions on Apple
silicon, so their own results differ in the last bit between platforms.
The tests allow for this where they compare against them.
