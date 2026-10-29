# Differences from matplotlib

fastplot aims to produce the figure matplotlib and cartopy would, and the
test suite compares the two on limits, ticks, labels, legend and layout
geometry, and pixels. Some differences are deliberate, though, and a few
are not worth the speed they would cost. They are all listed here.

## Maps

- **Spherical projections.** cartopy uses the WGS84 ellipsoid for most
  projections, and fastplot a sphere. Everything on a map goes through the
  same projection, so the map is consistent, but projected x/y values in
  metres differ slightly from cartopy's.
- **Extents on conic maps.** `set_extent` on a curved outline (a Lambert map,
  say) uses the exact bounds. cartopy approximates them, so the aspect ratio
  can differ by about 0.3%.
- **Gridline labels** go on the map's four edges and stay horizontal.
  cartopy rotates them along curved gridlines on conic and polar maps.
- **Contour labels** follow matplotlib's placement rules on fastplot's own
  lines, so a label can land a little further along a line than
  matplotlib's would.
- **Line drawing on maps** uses round caps and no pixel snapping, which keeps
  map images the same as earlier versions.
- **Thin wind barbs** (well under 1 pt) come out slightly lighter.

## Plain axes

- **Text.** Glyphs are rasterised by fastplot from the font outlines, not by
  Agg with FreeType's autohinter. Text sits where matplotlib puts it, with
  the same metrics, but individual pixels differ a little. This accounts
  for most of the remaining difference on a typical plot, about 0.2 to 1% of
  pixels.
- **Rotated text** at angles other than multiples of 90° is placed exactly,
  but its glyphs are rotated as an image.
- **Line joins** are round. matplotlib mitres patch edges, so a corner pixel
  can differ.
- **Scatter markers** are placed to 1/8 of a pixel.
- **Dense noisy lines.** Many segments of one line crossing inside a single
  pixel come out a little lighter than Agg's.

## Polar axes

- Only the full circle is supported, not partial wedges.
- Artists are clipped to the axes' box rather than the circle.

## Output

- PDF, SVG and EPS files hold the drawn picture, losslessly at the savefig
  dpi, not vector paths.

## Threads

Drawing figures from several threads at once needs numba's OpenMP or TBB
threading layer. The default wheels on Linux and Windows have one. pip's
numba on Apple silicon only has the `workqueue` layer, which can't run
parallel work from two threads at once; fastplot then takes turns between
threads automatically, so it stays safe but doesn't gain speed.

## Numbers

fastplot's arithmetic gives the same bits on every platform. numpy and
matplotlib, compiled with fused multiply-adds on Apple silicon, can differ
from themselves on x86 in the last bit. The tests allow for that where they
compare the two.
