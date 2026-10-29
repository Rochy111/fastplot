# Settings

## Functions

| Function | What it does |
|---|---|
| `fastplot.warm(processes=None)` | Compile, or load from the cache, every kernel. Run once after installing. |
| `fastplot.set_device(name)` | `'auto'`, `'cpu'` or `'gpu'` for every figure from now on |
| `fastplot.get_device()` | The current device setting |
| `fastplot.gpu_info()` | Whether a GPU is usable, its name, and why not if it isn't |
| `fastplot.tune(save=True)` | Measure the best thread count and GPU threshold on this machine |
| `fastplot.hardware()` | What fastplot found about the machine, and where each setting came from |
| `fastplot.frames(fn, items)` | Draw many figures at once on threads ([Many frames](../guide/frames.md)) |

## Command line

```bash
python -m fastplot warm        # compile every kernel, in parallel
python -m fastplot tune        # measure and save this machine's settings
python -m fastplot hardware    # show what was found
```

## Environment variables

| Variable | Effect |
|---|---|
| `FASTPLOT_THREADS` | Drawing threads, overriding `tune()` and the core count |
| `FASTPLOT_DEVICE` | `auto`, `cpu` or `gpu` |
| `FASTPLOT_GPU_WORK` | The `auto` device's GPU threshold, in pixels plus grid points |
| `FASTPLOT_PARK` | `0` leaves numba's idle threads spinning between draws |
| `FASTPLOT_PRELOAD` | `0` stops the background loading of numba at import |
| `FASTPLOT_CACHE_DIR` | Where `tune()` keeps its results |
| `FASTPLOT_DATA_DIR` | Where Natural Earth's 10 m layers are downloaded to |

numba's own variables apply too, for example `NUMBA_NUM_THREADS` to cap
its thread pool, or `NUMBA_CACHE_DIR` when the install directory isn't
writable.
