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

## Making a release

Versions follow [semantic versioning](https://semver.org/). While fastplot
is 0.x, bump the minor version (0.3.0 to 0.4.0) for new features or changed
behaviour, and the patch version (0.3.0 to 0.3.1) for fixes only. The
package version comes from the git tag through setuptools-scm, so there is
no version number to edit in the code.

1. Under a new heading in [`CHANGELOG.md`](changelog.md), list what was
   added, changed and fixed since the last release, with the date.
2. Commit it to `main` and push.
3. Tag the commit and push the tag:

    ```bash
    git tag -a v0.4.0 -m "fastplot 0.4.0"
    git push origin v0.4.0
    ```

4. Publish the release on GitHub, with that version's changelog section as
   its notes:

    ```bash
    gh release create v0.4.0 --title "fastplot 0.4.0" --notes-file notes.md
    ```

## How it is organised

[How it works inside](internals.md) walks through the rendering pipeline,
the conventions the code follows and what each module does.
