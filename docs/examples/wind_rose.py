"""A wind rose: how often the wind blows from each direction, and how hard.
The year of hourly winds is made up for the example: prevailing
south-westerlies, with a weaker easterly regime."""
import os

import numpy as np

import fastplot.pyplot as plt

rng = np.random.default_rng(11)
n = 8760                                             # a year of hourly observations
westerly = rng.random(n) < 0.7
direction = np.where(westerly, rng.normal(235, 35, n), rng.normal(80, 30, n)) % 360
speed = np.where(westerly, rng.gamma(3.2, 3.4, n), rng.gamma(2.4, 2.6, n))       # knots

sectors = 16
width = 2 * np.pi / sectors
centre = np.arange(sectors) * 360 / sectors
which = np.round(direction / (360 / sectors)).astype(int) % sectors
bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, np.inf)]
colours = ['#d6eaf8', '#85c1e9', '#3498db', '#2471a3', '#1a5276', '#0b2545']

fig = plt.figure(figsize=(8.4, 7.2), dpi=150)
ax = fig.add_subplot(projection='polar')
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_axisbelow(True)

bottom = np.zeros(sectors)
for (lo, hi), colour in zip(bins, colours):
    share = np.bincount(which[(speed >= lo) & (speed < hi)], minlength=sectors) / n * 100
    label = '%d+ kt' % lo if np.isinf(hi) else '%d-%d kt' % (lo, hi)
    ax.bar(np.radians(centre), share, width=width * 0.92, bottom=bottom, color=colour,
           label=label)
    bottom += share

ax.set_thetagrids(range(0, 360, 45), ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
ax.set_rlabel_position(112)
ax.set_rgrids([3, 6, 9, 12], ['3%', '6%', '9%', '12%'], fontsize=8, color='#555555')
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False,
          title='Wind speed', title_fontsize=8)
ax.set_title('Wind rose, one year of hourly winds (illustrative)', fontsize=11, pad=16)

fig.savefig(os.path.join(os.path.dirname(__file__), '..', 'images', 'wind_rose.png'),
            bbox_inches='tight')
