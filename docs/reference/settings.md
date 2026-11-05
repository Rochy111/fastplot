# Settings

## Functions

| Function | What it does |
|---|---|
| `fastplot.warm(processes=None)` | Compile, or load from the cache, every kernel, and link them into the kernel file a new process runs them from without numba. Installing already does this. |
| `fastplot.set_device(name)` | `'auto'`, `'cpu'` or `'gpu'` for every figure from now on |
| `fastplot.get_device()` | The current device setting |
| `fastplot.gpu_info()` | Whether a GPU is usable, its name, and why not if it isn't |
| `fastplot.tune(save=True)` | Measure the best thread count and GPU threshold on this machine |
| `fastplot.hardware()` | What fastplot found about the machine, and where each setting came from |
| `fastplot.frames(fn, items)` | Draw many figures at once on threads ([Many frames](../guide/frames.md)) |
| `fastplot.set_cache_budget(mb=None)` | Megabytes fastplot may keep from one figure for the next; `None` works it out again ([memory](#memory)) |
| `fastplot.cache_info()` | The budget, what is held against it, and each cache's size |

## Command line

```bash
python -m fastplot warm        # compile every kernel now (installing already does)
python -m fastplot tune        # measure and save this machine's settings
python -m fastplot hardware    # show what was found
```

## Environment variables

| Variable | Effect |
|---|---|
| `FASTPLOT_THREADS` | Drawing threads, overriding `tune()` and the core count |
| `FASTPLOT_DEVICE` | `auto`, `cpu` or `gpu` |
| `FASTPLOT_GPU_WORK` | The `auto` device's GPU threshold, in pixels plus grid points |
| `FASTPLOT_DEFLATE` | `zlib` writes PNGs with zlib even where the `isal` package (the `png` extra) is installed: zlib's bytes, a little slower |
| `FASTPLOT_POOL` | `0` runs parallel steps on numba's OpenMP threads rather than fastplot's own pool (the default on Windows); `1` uses the pool on Linux too ([details](../guide/hardware.md#idle-threads)) |
| `FASTPLOT_PARK` | `0` leaves numba's idle threads spinning between draws |
| `FASTPLOT_ECOQOS` | `1` lets Windows move a background process to the efficiency cores of a hybrid CPU ([details](../guide/hardware.md#how-the-settings-are-chosen)) |
| `FASTPLOT_IMAGE` | `0` runs the kernels through numba rather than the kernel file |
| `FASTPLOT_PRELOAD` | `0` stops the background thread that opens the kernel file (or starts numba) at import, and the check below with it |
| `FASTPLOT_AUTOWARM` | `0` stops fastplot compiling its kernels in the background when they don't match this machine |
| `FASTPLOT_BUILD_KERNELS` | `0` while installing skips compiling the kernels, for a faster install and a slower first run |
| `FASTPLOT_CACHE_DIR` | Where the kernel file, `tune()`'s results and what the `auto` device learnt about the GPU are kept |
| `FASTPLOT_DATA_DIR` | Where Natural Earth's 10 m layers are downloaded to |
| `FASTPLOT_CACHE_MB` | The memory budget for what is kept across figures, in megabytes; `0` keeps nothing ([memory](#memory)) |

numba's own variables apply too, for example `NUMBA_NUM_THREADS` to cap
its thread pool, or `NUMBA_CACHE_DIR` when the install directory isn't
writable.

## Memory

A figure drawn after another of the same map reuses work that depends only
on the map or the data, rather than redoing it:

- the map's projected coastlines, borders and gridlines;
- per-pixel tables of where each pixel lands on the grid;
- contours and barbs drawn from the same arrays, which a forecast hour's products share;
- the parts of the figure that stay the same, such as a colorbar.

All of it is kept within one memory budget for the process. When the
budget is full, the entry that has gone unused longest goes first, with a
frame's drawings going before a grid's tables and a map's geometry. The
Natural Earth features themselves are not counted: they are kept while
frames read them, and go once a map's projected geometry is kept and no
frame has read them for 8 frames (a new map reads them again, in tens of
milliseconds).

Some memory is kept apart from the budget, and sized by what the frames
use rather than by a setting: the scratch buffers a draw works in (each
thread's, reused frame to frame), and freed blocks of large arrays, of
which a frame's end gives back those the frame did not take again. A
drawn figure's picture is let go as soon as the caller lets go of what
`buffer_rgba()` gave, without waiting for Python's cycle collector.

The budget is 1% of the machine's memory, at least 32 MB and at most
1 GB. In a `multiprocessing` worker, that 1% is shared among the workers
likely to run beside it, counted as the CPUs divided by the worker's
drawing threads. On a 64 GB, 24-thread machine, a worker running 3
threads gets about 80 MB, and a single process about 650 MB. Set
`FASTPLOT_CACHE_MB` or call `fastplot.set_cache_budget(mb)` to choose it
yourself. `fastplot.cache_info()` shows what is held.

## Text as matplotlib 3.8

fastplot draws text as matplotlib 3.11 does. matplotlib 3.8 and earlier
drew it differently: glyphs hinted by FreeType's autohinter at 8 times the
horizontal resolution, each line laid out and drawn as one bitmap, and
lines 1.2 font heights apart. That text is about 2% narrower, its letters
have slightly different shapes, and multi-line text sits a pixel or so
lower. To draw a figure's text that way, so that it matches images made
with those versions, set matplotlib 3.8's text settings:

```python
fastplot.rcParams.update({'text.hinting_factor': 8, 'text.hinting': 'force_autohint',
                          'text.kerning_factor': 0})
```

This needs matplotlib installed: the glyphs come from its own font engine
and FreeType, so they are the pixels that matplotlib draws. Plain text,
upright or at any angle, then matches it to the byte. Mathtext is still
drawn as matplotlib 3.11 draws it. Without matplotlib, fastplot warns once
and keeps its own text.
