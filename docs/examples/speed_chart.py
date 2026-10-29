"""Milliseconds to draw a figure and save it as PNG: matplotlib (with cartopy
for the maps) and fastplot on the CPU. Numbers from tools/bench.py and
tools/bench_plain.py on an i7-13700K. The chart is drawn with fastplot."""
import os

import numpy as np

import fastplot.pyplot as plt

cases = ['Global map\ncontours', 'Lambert map\nUS states',
         'pcolormesh\n1,800 barbs', 'pcolormesh\ncoastlines', 'streamplot\nx3',
         'scatter\n20k points', 'Skew-T\nadiabats', 'plot\n100k points']
times = {                                            # ms per figure
    'matplotlib (+ cartopy)': [14016, 23492, 1952, 2756, 418, 158.8, 70.8, 71.5],
    'fastplot': [14.5, 12.0, 11.8, 7.7, 20, 4.9, 6.8, 8.0],
}
colours = ['#b3b6b7', '#2471a3']

fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
x = np.arange(len(cases))
w = 0.36
for k, ((name, ms), colour) in enumerate(zip(times.items(), colours)):
    pos = x + (k - 0.5) * w
    ax.bar(pos, ms, width=w * 0.92, color=colour, label=name)
    for px, v in zip(pos, ms):
        ax.text(px, v * 1.08, format(v, ',.0f') if v >= 100 else format(v, '.1f'),
                ha='center', fontsize=7, color='#333333')

ax.set_yscale('log')
ax.set_ylim(1, 1e5)
ax.set_xticks(x, cases, fontsize=8.5)
ax.set_ylabel('ms per figure (log scale)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', linewidth=0.3)
ax.set_axisbelow(True)
ax.legend(loc='upper right', ncol=2, frameon=False, fontsize=8.5)
ax.set_title('Drawing a figure and saving it', loc='left', fontsize=11)
fig.tight_layout()

fig.savefig(os.path.join(os.path.dirname(__file__), '..', 'images', 'speed_chart.png'))
