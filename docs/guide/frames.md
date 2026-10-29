# Many frames

Forecast graphics often draw the same map many times, once for each forecast
hour, ensemble member or variable. fastplot has three ways to speed this up,
and they can be combined.

## Updating an existing figure

If only the data changes, update the artist and save again. Only what
changed is redrawn.

```python
fig = plt.figure(figsize=(12, 6.5), dpi=150)
ax = fig.add_subplot(projection=ccrs.PlateCarree())
mesh = ax.pcolormesh(lon, lat, fields[0], cmap='turbo', transform=ccrs.PlateCarree())
ax.coastlines()
fig.colorbar(mesh, ax=ax)

for hour, field in zip(hours, fields):
    mesh.set_array(field)
    ax.set_title('Forecast hour %d' % hour)
    fig.savefig('frame_%03d.png' % hour)
```

## New figures of the same map

You don't have to restructure a script to reuse figures. A new figure of a
map that was drawn before reuses the coastlines, borders and gridlines that
were already projected for it. On projected maps (Lambert, polar, Robinson
and so on) it also reuses where each pixel lands on the data grid, which is
most of the work for a field. A Lambert CONUS frame from GFS goes from
15.5 ms to 12 ms from the second frame on, with an identical image.

## Drawing several figures at once

`fastplot.frames` runs your function over a list of items, several at a
time on threads, and returns the results in order:

```python
import fastplot

def hour(h):
    fig = fastplot.figure(figsize=(12, 6.5), dpi=150)
    ax = fig.add_subplot(projection=ccrs.PlateCarree())
    ...
    fig.savefig('t2m_%03d.png' % h)

fastplot.frames(hour, range(0, 385, 3))
```

The heavy parts of a frame (numba kernels and PNG compression) release
Python's GIL, so while one frame is in Python another can be drawing. On
an 8 performance-core machine this gives about 1.35 times as many frames per
second as drawing them one by one. The images are byte-for-byte the same.

A few things to know:

- Use `fastplot.figure` or `fastplot.subplots` inside the function, not
  `pyplot`. pyplot's "current figure" is shared between threads.
- `frames(fn, items, workers=None, threads=None)` picks how many frames to
  draw at once, and how many numba threads each gets, from your CPU. Pass
  them to override.
- The first exception stops the remaining frames and is raised from
  `frames`.
- On numba's `workqueue` threading layer (pip's numba on Apple silicon has
  no other), parallel work can't run from two threads at once, so frames
  are drawn one at a time there. Installing numba from conda-forge, which
  brings an OpenMP runtime, lifts that.

## Processes

Threads are usually the better choice, because fastplot's data (map
features, fonts, colormaps, the per-pixel tables) is loaded once and
shared. If you already have a process pool, keep it for the whole job and
warm each worker as it starts, so the first frame doesn't pay for loading
numba. `processes=1` keeps the warm-up inside the worker:

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(4, initializer=fastplot.warm, initargs=(1,)) as pool:
    list(pool.map(hour, range(0, 385, 3)))
```
