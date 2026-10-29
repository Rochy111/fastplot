"""Scratch memory kept from one call to the next, per thread.

    a = _arena.take('contour.pts', (n, 4), np.float64)   # n rows, contents undefined

A kernel's working arrays (marching squares' segments, the links between
them) are big and live for one call. Fresh ones cost the OS a page fault
every 4 KB -- ~1 ms for 10 MB, often more than the kernel that fills
them -- so each thread keeps its own, by name, grown when a call wants
more (by half again, so a growing series reallocates rarely) and reused
after. A buffer is never handed out of the call that took it: results
are fresh arrays. Buffers over KEEP_MAX bytes are not kept (one huge
frame does not pin its memory for good); release() lets every one go.
"""
import threading

import numpy as np

KEEP_MAX = 256 << 20            # bytes: a buffer bigger than this is not kept

_LOCAL = threading.local()


def take(name, shape, dtype):
    """This thread's buffer `name` as an array of shape and dtype (its
    contents whatever the last call left)."""
    shape = (shape,) if isinstance(shape, (int, np.integer)) else tuple(int(s) for s in shape)
    dtype = np.dtype(dtype)
    n = dtype.itemsize
    for s in shape:
        n *= s
    bufs = getattr(_LOCAL, 'bufs', None)
    if bufs is None:
        bufs = _LOCAL.bufs = {}
    b = bufs.get(name)
    if b is None or b.size < n:
        size = max(n, int(1.5 * b.size) if b is not None else 0, 4096)
        b = np.empty(size, np.uint8)
        if size <= KEEP_MAX:
            bufs[name] = b
        else:
            bufs.pop(name, None)
    return b[:n].view(dtype).reshape(shape)


def release():
    """Every buffer this thread keeps let go."""
    _LOCAL.bufs = {}
