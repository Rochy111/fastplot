# Working on fastplot

## Setting up

```bash
git clone https://github.com/Rochy111/fastplot
cd fastplot
pip install -e ".[test]"
python -m fastplot warm
```

## Tests

```bash
python -m pytest
```

The tests use matplotlib, cartopy and contourpy as references. They draw
the same figures with both libraries and compare limits, ticks, labels,
legend and layout geometry, and pixels. GPU tests are skipped when there is
no GPU, and tests that compare against matplotlib's own output are skipped
on matplotlib older than 3.11, whose text shaping, legends and colormap
names differ.

CI runs the suite on Linux (Python 3.10, 3.12 and 3.13, numpy 1 and 2),
Windows and macOS on every push.

## Guidelines

A change to the drawing code should not change any pixel unless that is its
purpose. Compare images before and after the change, and check that the CPU
and GPU still produce identical images.

Kernels are numba functions compiled with `cache=True` and `nogil=True`.
They take writable, contiguous arrays so that each compiles for a single
signature. If you add a kernel or change its argument types, regenerate
the warm-up manifest with `python tools/make_warm_manifest.py`.

Measure performance before and after a change. `tools/bench.py` covers
maps, `tools/bench_plain.py` plain plots and `tools/bench_png.py` the PNG
encoder; each prints medians you can compare.

## Building these docs

```bash
pip install mkdocs-material
mkdocs serve
```

The gallery images are made by the scripts in `docs/examples/`. Run one to
redraw its picture into `docs/images/`.

## How it is organised

[How it works inside](internals.md) walks through the rendering pipeline,
the conventions the code follows and what each module does.
