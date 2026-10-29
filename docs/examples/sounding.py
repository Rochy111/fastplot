"""A springtime sounding on a Skew-T, with the parcel's CAPE and CIN shaded and
the wind profile on a hodograph. The profile is made up for the example: a
moist boundary layer under a capping inversion, steep lapse rates aloft and
winds veering and strengthening with height."""
import os

import numpy as np

import fastplot.pyplot as plt
from fastplot.skewt import Hodograph, SkewT, dry_lapse, moist_lapse

p = np.array([1000, 975, 950, 925, 900, 875, 850, 825, 800, 750, 700, 650, 600, 550, 500,
              450, 400, 350, 300, 250, 200, 150, 100], float)                   # hPa
T = np.array([29.0, 27.0, 24.9, 22.7, 20.6, 19.4, 20.2, 20.8, 19.2, 14.6, 9.0, 3.6, -1.6,
              -6.9, -12.6, -18.9, -26.0, -34.0, -43.0, -52.5, -58.5, -60.5, -64.0])  # °C
Td = np.array([20.5, 19.8, 19.0, 18.0, 16.8, 15.2, 10.2, 2.0, -1.5, -4.0, -7.5, -12.0,
               -17.0, -23.0, -28.5, -34.0, -40.0, -46.0, -53.0, -61.0, -70.0, -76.0, -82.0])
speed = np.array([10, 18, 24, 28, 30, 31, 32, 33, 34, 37, 40, 44, 48, 52, 56, 60, 65, 70, 74,
                  76, 70, 58, 46], float)                                         # knots
direction = np.array([150, 165, 180, 190, 200, 205, 210, 215, 220, 225, 230, 235, 238, 240,
                      243, 245, 248, 250, 252, 254, 256, 258, 260], float)       # degrees
u = -speed * np.sin(np.radians(direction))
v = -speed * np.cos(np.radians(direction))

# The surface parcel: dry to its lifting condensation level (Bolton 1980), moist above.
t0, td0 = T[0] + 273.15, Td[0] + 273.15
t_lcl = 1.0 / (1.0 / (td0 - 56.0) + np.log(t0 / td0) / 800.0) + 56.0
p_lcl = p[0] * (t_lcl / t0) ** (1 / 0.2857)
parcel = np.where(p >= p_lcl, dry_lapse(p, t0, p[0]),
                  moist_lapse(p, t_lcl, p_lcl)) - 273.15

fig = plt.figure(figsize=(9, 8), dpi=150)
skew = SkewT(fig, rotation=45, rect=(0.08, 0.08, 0.6, 0.84))
skew.plot_dry_adiabats(linewidth=0.6, alpha=0.45)
skew.plot_moist_adiabats(linewidth=0.6, alpha=0.45)
skew.plot_mixing_lines(linewidth=0.6, alpha=0.6)
skew.shade_cape(p, T, parcel, alpha=0.25)
skew.shade_cin(p, T, parcel, Td, alpha=0.25)
skew.plot(p, T, color='#c0392b', linewidth=2)
skew.plot(p, Td, color='#1e8449', linewidth=2)
skew.plot(p, parcel, color='black', linewidth=1.2, linestyle='--')
skew.plot_barbs(p[::2], u[::2], v[::2], length=6)
skew.ax.set_xlim(-30, 45)
skew.ax.set_xlabel('Temperature (°C)')
skew.ax.set_ylabel('Pressure (hPa)')
skew.ax.set_title('Afternoon sounding (illustrative)', loc='left', fontsize=11)

hodo_ax = fig.add_axes((0.71, 0.55, 0.26, 0.3))
hodo = Hodograph(hodo_ax, component_range=80)
hodo.add_grid(increment=20, linewidth=0.5)
line = hodo.plot_colormapped(u, v, p, cmap='plasma_r', linewidth=2.5)
hodo_ax.set_title('Hodograph (kt), coloured by pressure', fontsize=8)
hodo_ax.tick_params(labelsize=7)

fig.text(0.71, 0.45, 'Surface parcel', fontsize=9, fontweight='bold')
fig.text(0.71, 0.42, 'LCL  %.0f hPa' % p_lcl, fontsize=9)
fig.text(0.71, 0.395, 'Shear, surface to 500 hPa  %.0f kt' % np.hypot(u[14] - u[0], v[14] - v[0]),
         fontsize=9)

fig.savefig(os.path.join(os.path.dirname(__file__), '..', 'images', 'sounding.png'),
            bbox_inches='tight')
