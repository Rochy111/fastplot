"""A five-day meteogram for one place: a cold front comes through on the third
day. The series are made up for the example, but behave like a real station's:
a daily cycle, a wind shift and pressure jump at the front, then showers."""
import os
from datetime import datetime, timedelta

import numpy as np

import fastplot.dates as mdates
import fastplot.pyplot as plt

hours = np.arange(0, 5 * 24 + 1)
time = [datetime(2026, 10, 27) + timedelta(hours=int(h)) for h in hours]
front = 58                                           # hour of the frontal passage
after = 1 / (1 + np.exp(-(hours - front) / 2.5))     # 0 before the front, 1 after it
daily = np.sin(2 * np.pi * (hours - 9) / 24)

temp = 17 + 6 * daily - 9 * after + 0.6 * np.sin(hours / 3.1)
dewp = np.minimum(12 + 1.5 * daily - 10 * after, temp - 0.8)
pres = 1014 - 0.12 * np.clip(hours, 0, front) + 9 * after + 1.2 * np.sin(2 * np.pi * hours / 12)
speed = 9 + 3 * daily + 14 * np.exp(-((hours - front) / 6) ** 2) + 5 * after
gust = speed * (1.35 + 0.15 * np.sin(hours / 2.3))
direction = 190 + 110 * after                        # southerly, veering north-westerly
rng = np.random.default_rng(4)
rain = np.where((hours > front - 3) & (hours < front + 20),
                rng.gamma(1.2, 1.2, hours.size) * np.exp(-((hours - front - 2) / 8) ** 2), 0)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7.5), dpi=150, sharex=True,
                                    gridspec_kw={'height_ratios': [1.4, 1, 1]})

ax1.plot(time, temp, color='#c0392b', linewidth=1.8, label='Temperature')
ax1.plot(time, dewp, color='#1e8449', linewidth=1.8, label='Dewpoint')
ax1.fill_between(time, dewp, temp, color='#f5b7b1', alpha=0.35)
ax1.axvline(time[front], color='#555555', linestyle='--', linewidth=0.8)
ax1.text(time[front], 22.5, '  cold front', color='#555555',
         fontsize=8, va='top')
ax1.set_ylabel('°C')
ax1.legend(loc='upper right', fontsize=8, frameon=False, ncol=2)
ax1.grid(True, linewidth=0.3)

ax2.fill_between(time, 0, gust, color='#aed6f1', label='Gust')
ax2.plot(time, speed, color='#1f618d', linewidth=1.6, label='Mean wind')
every = slice(0, None, 6)                           # a wind barb every six hours
u = -speed * np.sin(np.radians(direction))
v = -speed * np.cos(np.radians(direction))
ax2.barbs(mdates.date2num(np.array(time)[every]), np.full(hours[every].size, 35.0),
          u[every], v[every], length=5.5, linewidth=0.6, color='#1b2631')
ax2.set_ylabel('kt')
ax2.set_ylim(0, 42)
ax2.legend(loc='center left', fontsize=8, frameon=False, ncol=2)
ax2.grid(True, linewidth=0.3)

ax3.plot(time, pres, color='#5b2c6f', linewidth=1.6)
ax3.set_ylabel('hPa')
ax3.grid(True, linewidth=0.3)
rain_ax = ax3.twinx()
rain_ax.bar(time, rain, width=1 / 24, color='#2e86c1', alpha=0.7)
rain_ax.set_ylabel('mm/h', color='#2e86c1')
rain_ax.tick_params(axis='y', colors='#2e86c1')
rain_ax.set_ylim(0, 8)

ax3.xaxis.set_major_locator(mdates.DayLocator())
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%a %d %b'))
ax3.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
ax1.set_title('Five-day meteogram (illustrative)', loc='left', fontsize=11)
ax1.set_title('Temperature, wind, pressure and rain', loc='right', fontsize=9, color='#555555')
fig.tight_layout()

fig.savefig(os.path.join(os.path.dirname(__file__), '..', 'images', 'meteogram.png'))
