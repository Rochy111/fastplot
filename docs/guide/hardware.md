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
**exactly the same image**, byte for byte, which the tests check. Choosing
a device only changes the speed.

If anything goes wrong on the GPU (no cupy, no CUDA device, an old driver,
out of memory), that frame is drawn again on the CPU, a warning says why
once, and the process stays on the CPU. Fields that are already on the
card, such as arrays read by fastgrib with `gpu=True`, are drawn there
without being copied back.

## How the settings are chosen

At import, fastplot looks at the machine. This takes under a millisecond.

**CPUs.** It counts the CPUs this process may actually use, including an
affinity mask or a container's CPU quota. A 4-CPU container on a 64-core
host gets 4.

**Drawing threads.** One per performance core, between 4 and 16. That
means the P-cores of a hybrid Intel CPU, the performance cluster on Apple
silicon, or the big cores of an ARM big.LITTLE chip. A figure is many short
parallel steps, and threads on hyper-threaded siblings or efficiency cores
make each step slower to start while using more power. On a 24-thread
i7-13700K this picks the 8 P-cores, which is faster than using all 24 and
uses a quarter of the CPU time.

**GPU threshold.** In `'auto'` mode a figure goes to the GPU once its
work (pixels plus grid points) passes a threshold. That threshold scales
with the CPU's drawing threads against the GPU's multiprocessors, and is
four times higher for integrated graphics.

## Measuring instead of guessing

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
