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
"""
import os
import threading
import warnings

DEVICES = ('auto', 'cpu', 'gpu')
AUTO_GPU_WORK = 1_000_000           # 'auto': grid points + pixels from which the GPU wins
                                     # (an 1800x975 map of GFS 0.25 deg: 2.8M; 1.2-3x faster there)

_lock = threading.Lock()
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
    return _usable


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
    if (on_gpu or work >= AUTO_GPU_WORK) and gpu_usable():
        return 'gpu'
    return 'cpu'
