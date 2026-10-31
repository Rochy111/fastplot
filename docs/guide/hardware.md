# CPU, GPU and threads

fastplot is meant to run well on whatever machine it lands on, from a
laptop to a container on a big server, without hand tuning. This page
explains what it decides and how to change it.

## CPU or GPU

```python
import fastplot as fp

fp.set_device('auto')     # the default: the GPU for big figures, when one is usable
fp.set_device('gpu')      # always the GPU, falling back to the CPU if it can't
fp.set_device('cpu')
fp.gpu_info()             # {'usable': True, 'device': 'NVIDIA GeForce RTX 5070 Ti', ...}

fig.savefig('map.png', device='gpu')   # or for one figure
```

On the GPU, fields are sampled, coloured and blended by CUDA kernels. Lines
and text stay on the CPU, where they take a few milliseconds. Both draw
exactly the same image, byte for byte, which the tests check. Choosing
a device only changes the speed.

If anything goes wrong on the GPU (no cupy, no CUDA device, an old driver,
out of memory), that frame is drawn again on the CPU, a warning says why
once, and the process stays on the CPU. Fields that are already on the
card, such as arrays read by fastgrib with `gpu=True`, are drawn there
without being copied back.

## How the settings are chosen

When it is imported, fastplot inspects the machine, which takes under a
millisecond.

It counts the CPUs that the process is allowed to use, taking an affinity
mask or a container's CPU quota into account, so a 4-CPU container on a
64-core host counts as 4.

Drawing uses one thread per performance core, with a minimum of 4 and a
maximum of 16: the P-cores of a hybrid Intel CPU, the performance cluster
on Apple silicon, or the large cores of an ARM big.LITTLE design. A figure
is made of many short parallel steps, and adding hyper-threads or
efficiency cores makes each step slower to start while using more power.
On a 24-thread i7-13700K this selects the 8 P-cores, which is faster than
using all 24 threads and uses about a quarter of the CPU time.

On a hybrid CPU under Windows, a busy process with no window in front (a
batch job, a service, a script behind another program) gets moved to the
efficiency cores after about two seconds, and every frame after that takes
twice as long. At its first draw fastplot asks Windows not to do this, for
that process and for as long as it runs. The cost is that the process uses
the performance cores while it works. To leave the process as it was, set
`FASTPLOT_ECOQOS=1`.

In `'auto'` mode, a figure is drawn on the GPU once its work (output pixels
plus grid points) passes a threshold. The threshold is scaled by the number
of CPU drawing threads relative to the GPU's multiprocessors, and is four
times higher for integrated graphics.

## Measuring the machine

```python
fastplot.tune()       # about 5 seconds
```

`tune()` times a plot and a map at several thread counts, and the map on
the CPU and GPU at several sizes. It keeps the fewest threads within 3% of
the fastest and saves the result in fastplot's cache directory. The saved
result is keyed by CPU, allowed CPUs, GPU and fastplot version, so a
different machine, or the same container resized, ignores it.

`fastplot.hardware()` reports what was found and where each setting came
from. From the command line:

```bash
python -m fastplot hardware
python -m fastplot tune
```

## Idle threads

numba's OpenMP threads on Windows keep spinning for about 100 ms after each
parallel step, waiting for more work. For a program that draws a frame and
then pauses, that meant several cores busy for longer than the frame took.
fastplot sends them to sleep when a draw finishes and before PNG
compression, so a frame costs 15 to 120 ms of CPU time instead of 380 to
500, at the same speed. Set `FASTPLOT_PARK=0` to turn this off.

## Overrides

Every automatic choice can be fixed with an environment variable. See
[Settings](../reference/settings.md).
