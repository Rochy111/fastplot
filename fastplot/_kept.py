"""Drawings kept across figures by what they follow from.

freeze(v): a value as a key that tells it to the bit (arrays by their
bytes, floats by their bits), or TypeError where it cannot.

region(r, key, draw): draw(r) -- something that paints r's canvas only
through r and adds to r.extents, its pixels a function of key and of the
pixels under them -- or, when it was drawn before under key over the same
pixels, those pixels as it left them, pasted, and the extents it added.

Where a drawing can paint is found once, exactly, by drawing it a second
time on a transparent probe canvas: Agg's blends (_agg._blend) leave
alpha 0 only where they change nothing, whatever is under them, and the
float blend (_draw.blend_px) only below 1/510 coverage, which changes no
opaque pixel -- so the pixels a drawing can change over opaque ones are the
probe's non-zero ones. The pixels under them are kept as they were (opaque,
or nothing is kept) and must be the same, byte for byte, for the kept
pixels to be used: the drawing would have made the same of them."""
import threading
from collections import OrderedDict

import numpy as np

from ._render import CpuCanvas

_FLOATS = (float, np.floating)


def freeze(v):
    """v as a hashable key telling it apart to the bit; TypeError for what
    cannot be (an object of no known kind)."""
    t = type(v)                                         # (the common kinds first, exactly)
    if t is str or v is None or t is bool:
        return v
    if t is float:
        return (float, v.hex())
    if t is tuple or t is list:
        return (t, tuple(map(freeze, v)))
    if t is int:
        return (int, v)
    if t is dict:
        return (dict, tuple((freeze(k), freeze(u)) for k, u in v.items()))
    if v is None or isinstance(v, (bool, str, bytes, type)):
        return v                                        # (a class: itself)
    if isinstance(v, int) and not isinstance(v, bool):
        return (int, v)
    if isinstance(v, _FLOATS):
        return (type(v), float(v).hex())
    if isinstance(v, np.integer):
        return (type(v), int(v))
    if isinstance(v, np.ndarray):
        if v.dtype.kind == 'O':
            raise TypeError('an object array')
        return ('a', v.dtype.str, v.shape, np.ascontiguousarray(v).tobytes())
    if isinstance(v, (list, tuple)):
        return (type(v), tuple(freeze(u) for u in v))
    if isinstance(v, dict):
        return (dict, tuple((freeze(k), freeze(u)) for k, u in v.items()))
    raise TypeError('cannot key %r' % type(v).__name__)


def frozen(v):
    """freeze(v), or None where it cannot be."""
    try:
        return freeze(v)
    except TypeError:
        return None


_STYLES = {}


def styles(color, width, dashes=None):
    """_draw.Styles(color, width, dashes) of one style (color uint8[4],
    width px, dashes None or [(on, off, ...)] px), made once: shared, so
    only read by its users (as the kernels do)."""
    from . import _draw
    c = np.asarray(color, np.uint8)
    key = None
    if c.shape == (4,) and type(width) is float and (dashes is None or all(
            type(v) is float for d in dashes for v in d)):
        # (Python floats: their reprs tell them apart, -0.0 too)
        key = (c.tobytes(), repr(width), None if dashes is None else repr(dashes))
        got = _STYLES.get(key)
        if got is not None:
            return got
    st = _draw.Styles(c, width, dashes)
    if key is not None:
        if len(_STYLES) >= 256:
            _STYLES.clear()
        _STYLES[key] = st
    return st


_RC_SEEN = []                   # [(deep copy of an rc dict, its token)], most recent first
_RC_NEXT = [0]


def rc_token(d):
    """A token for the settings d holds (a figure's rc dict): the same one
    for settings equal to the last few seen, compared value by value with
    copies of them; None where they cannot be compared."""
    import copy
    try:
        for snap, tok in _RC_SEEN:
            if type(snap) is type(d) and len(snap) == len(d) and snap == d:
                return tok
        snap = copy.deepcopy(d)
        if not snap == d:                               # (a value not equal to itself: NaN ...)
            return None
    except Exception:                                   # noqa: BLE001 -- arrays, odd objects
        return None
    with _LOCK:
        _RC_NEXT[0] += 1
        tok = _RC_NEXT[0]
        _RC_SEEN.insert(0, (snap, tok))
        del _RC_SEEN[8:]
    return tok


# key -> (box, before, after, extents): box (x0, y0, x1, y1) on the canvas,
# before the pixels under it (one uint32 when all alike, else a copy),
# after the same box once drawn
_REGIONS = OrderedDict()
_BYTES = [0]
_MAX_BYTES = 64 << 20
_MAX_N = 64
_MISSES = {}                    # key -> times its pixels under it were not the kept ones
_LOCK = threading.Lock()


def _touched(a):
    """The box (x0, y0, x1, y1) of a probe canvas' non-zero pixels; None if none."""
    nz = a.view(np.uint32).reshape(a.shape[0], a.shape[1]) != 0
    rows = np.flatnonzero(nz.any(axis=1))
    if not rows.size:
        return None
    cols = np.flatnonzero(nz[rows[0]:rows[-1] + 1].any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _under(view):
    """The pixels under a box as kept: one uint32 when they are all alike."""
    w = view.view(np.uint32)
    v = w.flat[0]
    return v if (w == v).all() else view.copy()


def _same(view, before):
    if isinstance(before, np.ndarray):
        return np.array_equal(view, before)
    return bool((view.view(np.uint32) == before).all())


def _nbytes(e):
    return e[2].nbytes + (e[1].nbytes if isinstance(e[1], np.ndarray) else 4)


def region(r, key, draw):
    """See the module's docstring. r: a _Renderer on a host canvas (any
    other: drawn as it is); key None: drawn as it is."""
    c = r.c
    if key is None or type(c) is not CpuCanvas or r.measuring or _MISSES.get(key, 0) >= 3:
        draw(r)
        return
    with _LOCK:
        got = _REGIONS.get(key)
        if got is not None:
            _REGIONS.move_to_end(key)
    if got is not None:
        (x0, y0, x1, y1), before, after, exts = got
        view = c.a[y0:y1, x0:x1]
        if _same(view, before):
            view[...] = after
            r.extents.extend(exts)
            return
        _MISSES[key] = _MISSES.get(key, 0) + 1
        draw(r)
        return
    probe = r.probe()
    draw(probe)
    box = _touched(probe.c.a)
    del probe
    n = len(r.extents)
    if box is None:
        before = after = np.zeros((0, 0, 4), np.uint8)
        box = (0, 0, 0, 0)
        draw(r)
    else:
        x0, y0, x1, y1 = box
        view = c.a[y0:y1, x0:x1]
        if not (view[..., 3] == 255).all():             # (only over opaque pixels: see above)
            _MISSES[key] = 3
            draw(r)
            return
        before = _under(view)
        draw(r)
        after = view.copy()
    e = (box, before, after, tuple(r.extents[n:]))
    size = _nbytes(e)
    if size > _MAX_BYTES // 4:
        return
    with _LOCK:
        old = _REGIONS.pop(key, None)
        if old is not None:
            _BYTES[0] -= _nbytes(old)
        _REGIONS[key] = e
        _BYTES[0] += size
        while len(_REGIONS) > 1 and (len(_REGIONS) > _MAX_N or _BYTES[0] > _MAX_BYTES):
            _, gone = _REGIONS.popitem(last=False)
            _BYTES[0] -= _nbytes(gone)


def clear():
    with _LOCK:
        _REGIONS.clear()
        _BYTES[0] = 0
        _MISSES.clear()
