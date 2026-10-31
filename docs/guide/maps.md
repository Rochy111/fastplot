# Maps

Map axes behave like cartopy's `GeoAxes`. You give a figure's axes a
projection, set what part of the world to show, then draw fields and
features on it, telling each one which coordinate system its data is in.

```python
import fastplot.pyplot as plt
import fastplot.crs as ccrs
import fastplot.feature as cfeature

fig = plt.figure(figsize=(10, 7), dpi=150)
ax = fig.add_subplot(projection=ccrs.LambertConformal(central_longitude=-97))
ax.set_extent([-125, -66, 22, 52], crs=ccrs.PlateCarree())
ax.contourf(lon, lat, field, levels=20, transform=ccrs.PlateCarree())
ax.add_feature(cfeature.STATES, linewidth=0.3)
ax.coastlines()
```

`plt.subplots(2, 2, subplot_kw={'projection': ccrs.Robinson()})` gives a grid
of map axes, as in matplotlib.

## Projections

| Projection | Typical use |
|---|---|
| `PlateCarree` | Global and regional lat-lon maps |
| `LambertConformal` | Mid-latitude regions (CONUS, Europe) |
| `Mercator`, `Miller` | Tropics, world maps |
| `Stereographic`, `NorthPolarStereo`, `SouthPolarStereo` | Polar regions |
| `Orthographic` | A globe seen from space |
| `LambertAzimuthalEqualArea` | Continents, ocean basins |
| `AlbersEqualArea` | Equal-area regional maps |
| `Robinson` | World maps |
| `RotatedPole` | Rotated-pole model grids (COSMO, ICON-LAM) |

Each takes the same arguments as cartopy's (`central_longitude`,
`central_latitude`, `standard_parallels` and so on), and the same Earth:
WGS84 by default, or a sphere where cartopy's default is one (`Orthographic`,
`Robinson`, `Miller`). So projected x/y metres agree with cartopy's.

To describe a model's own Earth, pass a `Globe`, as you would in cartopy. HRRR
files give a sphere:

```python
globe = ccrs.Globe(ellipse='sphere', semimajor_axis=6371229, semiminor_axis=6371229)
hrrr = ccrs.LambertConformal(central_longitude=-97.5, central_latitude=38.5,
                             standard_parallels=(38.5, 38.5), globe=globe)
```

## Choosing the area

- `ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())` shows a
  box of longitudes and latitudes.
- `ax.set_global()` shows the whole projection.
- Without either, the map fits the data you draw.

Where the projection's own edge shows inside the axes, as for a whole
Robinson map or an orthographic globe, the frame and the background follow
that edge, as in cartopy.

## Putting data on the map

Pass `transform=` with the coordinate system your data is in. For model
output on a latitude-longitude grid that is `ccrs.PlateCarree()`.

With 1-D axes, `ax.contourf(lon, lat, z)` takes `lon` of length `nx` and
`lat` of length `ny`. Longitudes can run from 0 to 360 or from -180 to 180,
and can wrap through 360 as ECMWF's grids do; latitudes can run in either
direction. For curvilinear grids such as WRF, HRRR or NAM, pass 2-D `lon`
and `lat` arrays of the same shape as the field. NaN values and masked
points are left blank.

### Field methods

| Method | What it draws |
|---|---|
| `pcolormesh` | Grid cells, `shading='nearest'`, `'flat'` or `'gouraud'` |
| `contourf` | Filled bands between levels, with `extend` and `hatches` |
| `contour` | Contour lines, same lines as matplotlib's contourpy |
| `clabel` | Labels along contour lines, placed as matplotlib places them |
| `imshow` | An image, scalar or RGB(A), with `extent` and `transform`; reprojected pixel by pixel |
| `barbs` | Wind barbs, in any units you pass |
| `quiver`, `quiverkey` | Arrows and an arrow key |
| `streamplot` | Streamlines of a vector field |
| `scatter`, `plot` | Points and lines, in any coordinate system |
| `text`, `annotate` | Labels at map coordinates |

Fields are not drawn as polygons. Each output pixel looks up its own value
in the grid, which is why a 1440 x 721 contourf takes a few milliseconds no
matter how many contour shapes it would have had.

## Coastlines, borders and other features

```python
ax.coastlines(linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.4)
ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor='grey')
ax.add_feature(cfeature.LAND, facecolor='#eee8d5')
ax.add_feature(cfeature.OCEAN)
```

The available features are `COASTLINE`, `BORDERS`, `STATES`, `LAKES`,
`LAND`, `OCEAN`, `RIVERS` and `COUNTIES`, plus any Natural Earth layer
through `cfeature.NaturalEarthFeature`.

Natural Earth's 110 m and 50 m layers ship with fastplot, and the right one
is picked for the size of the map. The 10 m layers (and US counties) come
from cartopy's cache if you have one, or are downloaded once into
`FASTPLOT_DATA_DIR`.

## Gridlines

```python
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='grey', linestyle='--')
```

Labels go on the map's four edges and are always horizontal. cartopy
rotates them along curved graticules on conic and polar maps. `xlocs` and
`ylocs` set where the lines go.

## Colorbars

```python
cf = ax.contourf(lon, lat, t2m, levels=range(-40, 41, 4), extend='both', transform=pc)
cb = fig.colorbar(cf, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
cb.set_label('2 m temperature (°C)')
```

Colorbars take matplotlib's arguments (`orientation`, `shrink`, `pad`,
`aspect`, `ticks`, `format`, `extend`) and are placed with the same layout
rules.

To put the bar exactly where you want it, give it axes of its own. As in
matplotlib, those axes become the colorbar's: `cb.ax is cax`, and a title
set on them is kept.

```python
cax = fig.add_axes((0.91, 0.25, 0.015, 0.5))
cb = fig.colorbar(cf, cax=cax)
cax.set_title('°C', pad=6, fontsize=9)
```

## Boxes and labels in axes or figure coordinates

A patch or line on a map normally goes through the projection. With
`transform=ax.transAxes` or `fig.transFigure` it is placed in those
coordinates instead, and it doesn't count when the map sizes itself to its
data. `fig.add_artist` draws a patch on the figure itself. Axes, colorbars
and figure artists are drawn in `zorder` order, as in matplotlib.

```python
from fastplot.patches import Rectangle

box = Rectangle((0.01, 0.01), 0.3, 0.08, transform=ax.transAxes)
box.set(fc='white', ec='black', lw=0.5, alpha=0.85, zorder=5)
ax.add_patch(box)
ax.text(0.02, 0.05, 'Valid 12Z Thu', transform=ax.transAxes, va='center', zorder=6)

fig.add_artist(Rectangle((0, 0.95), 1, 0.05, fc='#1f3a5f', transform=fig.transFigure))
```

## Saving

`fig.savefig('map.png')` writes the PNG with a multi-threaded encoder.
JPEG and WebP work too, and `bbox_inches='tight'`, `dpi=` and
`transparent=True` behave as in matplotlib. `fig.draw()` returns the
picture as an RGBA numpy array if you want to put it somewhere else.
