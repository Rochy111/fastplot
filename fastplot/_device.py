"""CPU or GPU: which one draws a figure.

    fastplot.set_device('auto' | 'cpu' | 'gpu')      # or FASTPLOT_DEVICE=...
    fastplot.gpu_info()                               # what was found, or why not

A figure draws on the GPU when the device is 'gpu', or under 'auto' when
the GPU is usable and the work is big enough to gain from it (fields
handed in as cupy arrays -- fastgrib's read(gpu=True) -- always are). The
GPU is usable when cupy imports, finds a CUDA device and compiles
fastplot's kernels; that is found out once per process, the first time
it matters, and never costs a CPU-only user anything (cupy is not
imported until then).

Any failure on the GPU -- no cupy, an old driver, a card without what a
kernel needs, out of memory -- is not an error: the frame is drawn again
on the CPU, a warning says why (once), and that process keeps to the CPU
from then on. The two give the same image.

Starting the GPU costs a process 0.4-0.8 s (cupy, the CUDA context, the
kernels, cupy's own first steps) -- more than a figure the card is meant
for gains from it, on the RTX 5070 Ti it was measured on. So 'auto':
  - knows from the last process (_hw.gpu_seen) the work from which the
    card draws, and whether it can: a smaller figure does not import cupy;
  - and draws on the CPU, all the same, until the CPU time spent on
    figures the card would have drawn adds up to what starting it took
    here: a process that draws one big map keeps to the CPU, one that
    draws hundreds moves to the card having lost at most that much time
    (ski rental: rent until the rent paid would have bought it);
  - and, the card started, times GPU_TRIES figures on it against those
    the CPU drew: unless the card took under GPU_WIN of the CPU's time
    for the work, the work it starts from is raised past them (here and
    for the next process) and the CPU draws them again. Figures bigger
    than that are tried, and judged, by a process that draws them.
    FASTPLOT_GPU_WORK or tune()'s threshold are not second-guessed.
A machine where the GPU was never looked at rents from GPU_LOOK_WORK up
to GPU_START_GUESS before it looks.
"""
import os
import threading
import time
import warnings

DEVICES = ('auto', 'cpu', 'gpu')
GPU_LOOK_WORK = 250_000             # 'auto': work below which the GPU is not even looked at
                                     # (the threshold itself is the card's: _hw.gpu_work)
GPU_START_GUESS = 0.5               # seconds to start the GPU, where it was never timed
GPU_TRIES = 3                       # figures on the card timed before it is judged
GPU_WIN = 0.9                       # its time for the work against the CPU's, to keep it

_lock = threading.Lock()
_rent = [0.0]                        # CPU seconds on figures the card would have drawn, before it started
_look = [0.0]                        # seconds gpu_usable() took, or 0: not timed in this process
_first = [True]                      # no figure drawn on the card yet (its time is part of starting)
_rates = {'cpu': [], 'gpu': []}      # seconds per work of the figures the card would draw, each way
_device = os.environ.get('FASTPLOT_DEVICE', 'auto').lower()
if _device not in DEVICES:
    _device = 'auto'
_usable = None                       # None: not looked at yet
_why = ''                            # why the GPU is not used
_name = ''
_warned = False                      # set_device('gpu') without a GPU: said once


def set_device(device='auto'):
    """Draw on 'cpu', 'gpu' (falling back to the CPU if it cannot) or
    'auto' (the GPU when it is usable and the figure is big enough)."""
    global _device
    device = str(device).lower()
    if device not in DEVICES:
        raise ValueError('device must be one of %s, not %r' % (DEVICES, device))
    _device = device


def get_device():
    return _device


def gpu_usable():
    """True when cupy, a CUDA device and fastplot's kernels all work."""
    global _usable, _why, _name
    if _usable is not None:
        return _usable
    with _lock:
        if _usable is not None:
            return _usable
        t = time.perf_counter()
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('no CUDA device')
            from . import _gpu
            _gpu.module()                               # compile (or load) the kernels
            props = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.Device().id)
            _name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
            _usable = True
        except Exception as e:                          # noqa: BLE001 -- any failure means CPU
            _why = '%s: %s' % (type(e).__name__, e)
            _usable = False
        _look[0] = time.perf_counter() - t
    _looked()
    return _usable


def _looked():
    """Keep what looking at the GPU found for the next process -- unless it
    is the card known already, and what was learnt of it. Twice the look
    for starting it, until its first figure is timed."""
    from . import _hw
    seen = _hw.gpu_seen()
    if not (_usable and seen is not None and seen[0] and seen[3] == _name):
        _hw.see_gpu(_usable, _hw.gpu_work(env=False) if _usable else 0, 2 * _look[0], _name)


def gpu_info():
    """{'usable': bool, 'device': name or '', 'reason': why not, 'mode': ...}."""
    ok = gpu_usable()
    return {'usable': ok, 'device': _name, 'reason': '' if ok else _why, 'mode': _device}


def gpu_failed(exc):
    """A GPU draw raised: warn once, and keep to the CPU from now on."""
    global _usable, _why
    with _lock:
        first = _usable is not False
        _usable = False
        _why = '%s: %s' % (type(exc).__name__, exc)
    if first:
        warnings.warn('fastplot: drawing on the GPU failed (%s); drawing on the CPU instead'
                      % _why, RuntimeWarning, stacklevel=3)


def pick(work, on_gpu=False):
    """'gpu' or 'cpu' for a figure of `work` (grid points + pixels);
    on_gpu: some input already lives on the card."""
    if _device == 'cpu':
        return 'cpu'
    if _device == 'gpu':
        global _warned
        if gpu_usable():
            return 'gpu'
        if not _warned:
            _warned = True
            warnings.warn('fastplot: set_device("gpu") but the GPU is not usable (%s); '
                          'drawing on the CPU' % _why, RuntimeWarning, stacklevel=3)
        return 'cpu'
    if on_gpu:
        return 'gpu' if gpu_usable() else 'cpu'
    if work < GPU_LOOK_WORK:
        return 'cpu'
    if _usable is None and _renting(work):
        return 'cpu'
    if gpu_usable() and work >= _seen()[1]:
        return 'gpu'
    return 'cpu'


def _seen():
    """(usable, work from which the card draws, seconds to start it) as the
    last process found them, FASTPLOT_GPU_WORK before that work; where the
    GPU was never looked at, a guess."""
    from . import _hw
    seen = _hw.gpu_seen()
    usable, w, start = seen[:3] if seen is not None else (True, GPU_LOOK_WORK, GPU_START_GUESS)
    n = _hw._env_int('FASTPLOT_GPU_WORK')
    return usable, w if n is None else n, start


def _keep(work=None, start=None):
    """Keep what this process learnt of the card it started."""
    from . import _hw
    seen = _hw.gpu_seen()
    if seen is not None and seen[0]:
        _hw.see_gpu(True, seen[1] if work is None else work, seen[2] if start is None else start, seen[3])


def _renting(work):
    """'auto', the GPU not started in this process: whether a figure of
    `work` draws on the CPU without looking -- the card is known not to
    draw it, or the rent paid is still short of starting it."""
    usable, w, start = _seen()
    return not usable or work < w or _rent[0] < start


def drew(dev, work, seconds):
    """A figure 'auto' sent to dev took `seconds`: on the CPU, rent paid
    while the card would have drawn it and is not started; the card's
    first, the rest of what starting it costs; its next GPU_TRIES, timed
    against the CPU's."""
    if _device != 'auto' or work < GPU_LOOK_WORK:
        return
    if dev == 'cpu':
        usable, w, _ = _seen()
        if usable and work >= w:
            with _lock:
                if _usable is None:
                    _rent[0] += seconds
                if len(_rates['cpu']) < 64:
                    _rates['cpu'].append((work, seconds))
        return
    with _lock:
        first, _first[0] = _first[0], False
        if not first:
            _rates['gpu'].append((work, seconds))
        judge = len(_rates['gpu']) == GPU_TRIES and not first
    if first and _look[0]:
        _keep(start=_look[0] + seconds)
    if judge:
        _judge()


def _judge():
    """The card's GPU_TRIES figures against the CPU's (seconds per work,
    the median of each): kept if it won, else the work it draws from
    raised past them, and the CPU draws them again."""
    from . import _hw
    if (len(_rates['cpu']) < GPU_TRIES or _hw._env_int('FASTPLOT_GPU_WORK') is not None
            or _hw.tuned(_name).get('work')):
        return
    cpu, gpu = (sorted(s / w for w, s in _rates[d]) for d in ('cpu', 'gpu'))
    if gpu[len(gpu) // 2] >= GPU_WIN * cpu[len(cpu) // 2]:
        _keep(work=2 * max(w for w, _ in _rates['gpu']))
