# Soundings and hodographs

`fastplot.skewt` has MetPy's `SkewT` and `Hodograph` classes, with the same
methods and arguments, plus the few thermodynamic functions needed to draw
the background lines and a parcel. pint quantities are accepted wherever
MetPy accepts them.

```python
import fastplot.pyplot as plt
from fastplot.skewt import SkewT

fig = plt.figure(figsize=(7, 7))
skew = SkewT(fig, rotation=45)
skew.plot(p, T, 'r')
skew.plot(p, Td, 'g')
skew.plot_barbs(p, u, v)
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()
skew.ax.set_xlim(-30, 40)
fig.savefig('skewt.png')
```

A complete example, with a parcel, shaded CAPE and CIN and a hodograph
inset, is in the [gallery](../gallery/sounding.md).

## SkewT

| Method | What it does |
|---|---|
| `plot(p, t, ...)` | A temperature profile, any matplotlib line style |
| `plot_barbs(p, u, v)` | Wind barbs down the right-hand side |
| `plot_dry_adiabats()` | Dry adiabats |
| `plot_moist_adiabats()` | Moist (pseudo-) adiabats |
| `plot_mixing_lines()` | Lines of constant mixing ratio |
| `shade_cape(p, t, parcel)` | Red where the parcel is warmer than the air around it |
| `shade_cin(p, t, parcel, dewpoint)` | Blue where it is colder, below the level of free convection |
| `shade_area(y, x1, x2, which)` | Any shading between two profiles |

`SkewT(fig, rotation=45, rect=(left, bottom, width, height))` places the
diagram anywhere in a figure, leaving room for a hodograph or text.
`skew.ax` is an ordinary axes for titles, limits and labels.

## Hodograph

```python
from fastplot.skewt import Hodograph

ax = fig.add_axes((0.7, 0.55, 0.25, 0.3))
hodo = Hodograph(ax, component_range=80)
hodo.add_grid(increment=20)
hodo.plot_colormapped(u, v, p, cmap='plasma_r')
```

`plot`, `plot_colormapped` (through a colormap, or a colour per interval of
height) and `wind_vectors` match MetPy's.

## Thermodynamics

| Function | Returns |
|---|---|
| `dry_lapse(p, t0, p0)` | Temperature (K) of air lifted dry-adiabatically |
| `moist_lapse(p, t0, p0)` | Temperature (K) along moist adiabats |
| `saturation_vapor_pressure(t)` | hPa, over liquid water |
| `saturation_mixing_ratio(p, t)` | kg/kg |
| `dewpoint_from_vapor_pressure(e)` | °C |

The moist adiabats integrate the same equation MetPy 1.7 does and agree
with it to within 0.001 K.
