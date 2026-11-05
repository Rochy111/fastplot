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
Python's GIL, so while one frame is in Python another can be drawing. The
images are byte-for-byte the same.

How best to split the work depends on the machine and the figure: a few
frames at a time on several threads each, or many at a time on one thread
each, or one at a time as a loop would. `frames` finds out as it goes. Its
first frames are drawn a few ways, and the rest the way that drew the most
frames a second. Later batches of the same function start with that way,
in this process and in later ones on the same machine: the way is kept in
a small file in fastplot's cache folder, for this CPU and GPU only, and
learnt again if the function changes or draws far slower than it did. On an i7-13700K, a GFS map drew at 13.7 ms a frame this way
(19.7 in a plain loop), a Lambert CONUS map at 10.3 ms (15.3), and a line
plot of 100,000 points at 3.8 ms (7.4). Give `workers` or `threads` to set
the split yourself.

A figure that is mostly Python between small kernels, such as six panels
of lines and text, has little to gain: `frames` ends up drawing it one at
a time, as a loop does. The first batch pays for trying the other ways
(about 10% on six panels); later batches run as fast as the loop.

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

## Movies

`fastplot.save_movie` draws frames as `frames` does and pipes them
straight to ffmpeg, with no PNGs in between. The function returns the
figure (or an `(H, W, 3 or 4)` image) instead of saving it:

```python
def hour(h):
    fig = fastplot.figure(figsize=(12, 6.5), dpi=100)
    ...
    return fig

fastplot.save_movie(hour, range(0, 385, 3), 't2m.mp4', fps=8)
```

Frames are written in order as they are drawn, and only a few are held at
a time, so a long movie needs no more memory than a short one. The
extension picks the format: `.mp4`, `.mov` and `.mkv` are H.264 (`crf=18`,
x264's `veryfast` preset, yuv420p, an odd size padded to even), `.webm` is
VP9, and `.gif` is written by Pillow without ffmpeg. `codec`, `crf`,
`preset`, `bitrate` (kbit/s), `extra_args` (ffmpeg output options) and
`metadata` pass through to ffmpeg, and `workers` and `threads` work as
they do for `frames`. ffmpeg is taken from `ffmpeg=`, `FASTPLOT_FFMPEG`,
your `PATH`, or the `imageio-ffmpeg` package. On a global Robinson map
(200-level contourf, labelled contours, 780 barbs, coastlines), 40 frames
came out as an MP4 in 0.32 s, less than the 0.39 s it took to write them
as PNGs. `preset='medium'` (ffmpeg's own default) took 0.46 s for a file
5% smaller.

Scripts written for `matplotlib.animation` can import
`fastplot.animation` instead. It has `FuncAnimation`, `ArtistAnimation`,
`FFMpegWriter` and `PillowWriter`, with `anim.save(...)` and the
`with writer.saving(fig, path, dpi): ... writer.grab_frame()` loop. Those
frames are drawn one after another, because each one changes the same
figure.

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
