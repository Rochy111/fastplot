# Differences from matplotlib

fastplot aims to produce the figure that matplotlib and cartopy would. The
test suite draws the same figures with both libraries and compares axis
limits, ticks, labels, legend and layout geometry, and pixels. The
differences below remain, either by design or because matching them would
cost more speed than they are worth.

## Maps

| Topic | Behaviour in fastplot |
|---|---|
| Earth model | Projections use a sphere. cartopy uses the WGS84 ellipsoid for most of them, so projected x/y values in metres differ slightly. Data, coastlines and gridlines all go through the same projection, so a map is consistent with itself. |
| Extents on conic maps | `set_extent` uses the exact bounds of the curved outline. cartopy approximates them, so the aspect ratio of a Lambert map can differ by about 0.3%. |
| Gridline labels | Placed on the map's four edges and always horizontal. cartopy rotates them along curved gridlines on conic and polar maps. |
| Contour labels | Placed with matplotlib's rules on lines traced in contourpy's order and direction. On plain axes they land where matplotlib puts them. On maps cartopy splits and resamples the projected lines, so a label can sit slightly further along a line, and labels are only placed on the part of a line inside the map. |
| Line ends | Map lines use round caps and are not snapped to pixels. |
| Thin barbs | Barbs drawn well under 1 pt wide come out slightly lighter. |

## Plain axes

| Topic | Behaviour in fastplot |
|---|---|
| Text | Each glyph is rendered by FreeType (freetype-py) at its exact position and angle, as matplotlib 3.11's Agg renders it, so plain and rotated text match matplotlib to the pixel. Where glyphs overlap, their coverage is combined before blending rather than blended one by one, which can move a pixel by a level or two. Without freetype-py installed, glyphs are rendered on whole pixels and shifted, which makes edges slightly crisper than matplotlib's and rotates text at other angles as an image. |
| Line joins | Always round. matplotlib mitres patch edges, so a corner pixel can differ. |
| Markers | Scatter markers are positioned to 1/8 of a pixel. |
| Dense lines | Many segments of one line crossing within a single pixel come out slightly lighter than Agg draws them. |
| Curvilinear meshes | `pcolormesh` cells whose sides are not along the pixel grid (skewed or projected cells) take the pixels whose centres they cover. Agg gives a pixel that two cells share to the cell drawn last, so a boundary can sit up to one pixel away. |
| Gouraud edges | Gouraud-shaded meshes and `tripcolor` stop at their outline. Agg grows each triangle by half a pixel and antialiases it, so the outline pixels differ. |

## Polar axes

Only full circles are supported; `set_thetamin`, `set_thetamax` and
`set_thetalim` raise `NotImplementedError`. Artists are clipped to the
axes' bounding box rather than to the circle.

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
