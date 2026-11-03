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
| `FASTPLOT_PARK` | `0` leaves numba's idle threads spinning between draws |
| `FASTPLOT_ECOQOS` | `1` lets Windows move a background process to the efficiency cores of a hybrid CPU ([details](../guide/hardware.md#how-the-settings-are-chosen)) |
| `FASTPLOT_IMAGE` | `0` runs the kernels through numba rather than the kernel file |
| `FASTPLOT_PRELOAD` | `0` stops the background thread that opens the kernel file (or starts numba) at import, and the check below with it |
| `FASTPLOT_AUTOWARM` | `0` stops fastplot compiling its kernels in the background when they don't match this machine |
| `FASTPLOT_BUILD_KERNELS` | `0` while installing skips compiling the kernels, for a faster install and a slower first run |
| `FASTPLOT_CACHE_DIR` | Where the kernel file, `tune()`'s results and what the `auto` device learnt about the GPU are kept |
| `FASTPLOT_DATA_DIR` | Where Natural Earth's 10 m layers are downloaded to |

numba's own variables apply too, for example `NUMBA_NUM_THREADS` to cap
its thread pool, or `NUMBA_CACHE_DIR` when the install directory isn't
writable.
