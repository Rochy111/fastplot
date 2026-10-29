"""Colours, colormaps and norms, and the kernels that turn a value raster
into RGBA -- matplotlib's colours to the byte, without matplotlib.

    rgba = to_rgba('xkcd:sky blue')             # floats, as matplotlib's
    cmap = get_cmap('RdBu_r')                   # any registered name, or a
                                                # matplotlib Colormap object
    norm = make_norm(LogNorm(), data=z)         # or Norm('log', 1, 1e3)
    img = shade(z, cmap, norm)                  # (H, W, 4) uint8

    bands, under, over = contour_colors(levels, cmap='viridis', extend='both')
    img = shade_bands(z, levels, bands, under, over)

**Colours.** to_rgba / to_rgba_array follow matplotlib.colors to the
letter: the same names (base, CSS4, 'tab:', 'xkcd:', read from
data/colors.npz, which tools/make_cmaps.py writes from matplotlib), hex,
grey strings, 'C0'.. (matplotlib's default cycle) and 'none'.

**Colormaps.** Every colormap matplotlib registers is bundled
(data/cmaps.npz): a ListedColormap as its colours, a
LinearSegmentedColormap as its segment tables or functions -- what
matplotlib builds its lookup table from -- so the table, reversed() (how
the '_r' maps are made, here as there) and resampled(n) come out bit for
bit as matplotlib's, and the LUT matplotlib gives at the map's own size is
bundled with them. A colormap's table holds N colours then under, over and
bad, matplotlib's `_lut` layout.

**Bytes.** matplotlib makes bytes of a float colour two ways. An image
(imshow, and to_rgba(..., bytes=True)) truncates, uint8(255 c). Agg, which
paints every vector artist -- pcolormesh's quads, contourf's polygons,
lines -- from a float colour, rounds, uint8(255 c + 0.5). Both are here as
`rounding='floor'` and `rounding='nearest'`; a value raster standing in
for pcolormesh wants 'nearest', one standing in for imshow 'floor'.

**Norms.** Norm is matplotlib's Normalize family as plain data;
make_norm resolves any of them (a Norm, a matplotlib norm object duck-typed
by class name, a scale name, or None plus vmin/vmax) and fills missing
limits from the data the way a ScalarMappable autoscales, NaN and inf
excluded. A matplotlib norm fastplot cannot read (PowerNorm, FuncNorm, a
subclass of one's own) is called as it is, on the CPU.

**Kernels.** shade() reproduces ScalarMappable.to_rgba exactly, including
its mixed precision: on float32 data Normalize subtracts and divides in
double, rounding each result to float32, and Colormap multiplies by N in
float32 before flooring; TwoSlopeNorm and clipped norms work in double
throughout; BoundaryNorm bins in double but tests its ends in float32.
The kernels do the same, in numba on every core and in CUDA C (`CUDA`),
bit-identical. Non-finite values are 'bad', as in a plot (matplotlib masks
them when the data is set). Filled contours are a level lookup
(shade_bands) with contourf's own colours (contour_colors) and interval
convention.

**CUDA kernels** (`CUDA`, compiled with ('--std=c++14', '--fmad=false')).
shade_plan(norm, cmap, alpha, rounding) / bands_plan(levels, band_lut,
under, over, log) give the kernel's name and the arguments that go
between (v, n) and out, numpy scalars and arrays (upload the arrays):

    kern(grid, block, (v, np.int32(n), *plan.args, out))

v is const float* (n float32 values), out unsigned char* (n RGBA pixels,
contiguous, 4-byte aligned), lut the (N + 3, 4) uint8 table (N colours,
under, over, bad; alpha already in it), N its colour count. Every kernel
grid-strides, so any launch shape covers n. In order:

    shade_linear  (v, n, double vmin, double vmax, int clip, lut, int N, out)
    shade_log     (v, n, double vmin, double vmax, double log10(vmin),
                   double log10(vmax), int clip, lut, int N, out)
    shade_symlog  (v, n, double vmin, double vmax, double linthresh,
                   double linscale_adj, double logc, double log(base),
                   double t(vmin), double t(vmax), int clip, lut, int N, out)
    shade_twoslope(v, n, double vmin, double vcenter, double vmax, lut,
                   int N, out)
    shade_nonorm  (v, n, lut, int N, out)
    shade_boundary(v, n, const double* boundaries, int m, const int* rows
                   (m + 1: table row per count of boundaries <= v),
                   int clip, double vmin, double vmax, int lo_row,
                   int hi_row, lut, int N, out)
    shade_bands   (v, n, const double* levels, int m, const unsigned char*
                   tab ((m + 2) RGBA rows: below, the m - 1 bands, above,
                   invalid), int eq0 (row of v == levels[0]), int log,
                   double levels[0], double 1 / step, int uniform, out)

A 'func' norm (a matplotlib norm fastplot does not read) has no kernel
(plan.kernel None): it runs on the CPU.
"""
import functools
import itertools
import math
import os
import re
import threading
from collections.abc import Sequence
from numbers import Real
from typing import NamedTuple

import numpy as np

from . import _ticker
from ._jit import capped, njit, prange

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
_LOCK = threading.RLock()

__all__ = ['to_rgba', 'to_rgba_array', 'to_rgba_bytes', 'is_color_like',
           'Colormap', 'ListedColormap', 'LinearSegmentedColormap', 'get_cmap',
           'cmap_names', 'Norm', 'make_norm', 'auto_levels', 'contour_colors',
           'shade', 'shade_bands', 'shade_plan', 'bands_plan', 'ShadePlan', 'CUDA',
           'CUDA_KERNELS']


# --------------------------------------------------------------------------
# colours

_NAMES = None                   # name -> RGBA tuple
_CYCLE = None                   # the default property cycle, as names
_RGBA_CACHE = {}
_NTH_RE = re.compile(r'\AC[0-9]+\Z')
_HEX_RE = re.compile('#[a-fA-F0-9]+')


def _names():
    global _NAMES, _CYCLE
    if _NAMES is None:
        with _LOCK:
            if _NAMES is None:
                with np.load(os.path.join(_DATA, 'colors.npz')) as z:
                    names, rgba, cycle = z['names'], z['rgba'], z['cycle']
                _CYCLE = [tuple(float(v) for v in c) for c in cycle]
                _NAMES = {str(n): tuple(float(v) for v in c) for n, c in zip(names, rgba)}
    return _NAMES


def _is_nth_color(c):
    return isinstance(c, str) and _NTH_RE.match(c)


def is_color_like(c):
    """Whether `c` is a colour matplotlib would take."""
    if _is_nth_color(c):
        return True
    try:
        to_rgba(c)
    except (TypeError, ValueError):
        return False
    return True


def to_rgba(c, alpha=None):
    """`c` as a tuple of four floats in 0..1, as matplotlib.colors.to_rgba
    makes it: a name, '#rgb[a]' / '#rrggbb[aa]', a grey level string ('0.5'),
    'Cn' (matplotlib's default cycle), 'none' (transparent, whatever the
    alpha), a sequence of 3 or 4 numbers, or a (colour, alpha) pair. `alpha`
    replaces the colour's own alpha."""
    if isinstance(c, tuple) and len(c) == 2:
        if alpha is None:
            c, alpha = c
        else:
            c = c[0]
    if _is_nth_color(c):
        _names()
        c = _CYCLE[int(c[1:]) % len(_CYCLE)]
    try:
        rgba = _RGBA_CACHE[c, alpha]
    except (KeyError, TypeError):
        rgba = None
    if rgba is None:
        rgba = _to_rgba_no_colorcycle(c, alpha)
        try:
            _RGBA_CACHE[c, alpha] = rgba
        except TypeError:
            pass
    return rgba


def _to_rgba_no_colorcycle(c, alpha=None):
    if alpha is not None and not 0 <= alpha <= 1:
        raise ValueError("'alpha' must be between 0 and 1, inclusive")
    orig_c = c
    if c is np.ma.masked:
        return (0., 0., 0., 0.)
    if isinstance(c, str):
        if c.lower() == 'none':
            return (0., 0., 0., 0.)
        names = _names()
        try:
            c = names[c]
        except KeyError:
            if len(c) != 1:
                try:
                    c = names[c.lower()]
                except KeyError:
                    pass
        else:                                   # a stored name: already RGBA
            return c if alpha is None else c[:3] + (alpha,)
        if not isinstance(c, str):
            return c if alpha is None else c[:3] + (alpha,)
    if isinstance(c, str):
        if _HEX_RE.fullmatch(c):
            if len(c) == 7:
                return (*[n / 0xff for n in bytes.fromhex(c[1:])],
                        alpha if alpha is not None else 1.)
            elif len(c) == 4:
                return (*[int(n, 16) / 0xf for n in c[1:]],
                        alpha if alpha is not None else 1.)
            elif len(c) == 9:
                color = [n / 0xff for n in bytes.fromhex(c[1:])]
                if alpha is not None:
                    color[-1] = alpha
                return tuple(color)
            elif len(c) == 5:
                color = [int(n, 16) / 0xf for n in c[1:]]
                if alpha is not None:
                    color[-1] = alpha
                return tuple(color)
            raise ValueError(f'Invalid hex color specifier: {orig_c!r}')
        try:
            c = float(c)
        except ValueError:
            pass
        else:
            if not (0 <= c <= 1):
                raise ValueError(f'Invalid string grayscale value {orig_c!r}. '
                                 f'Value must be within 0-1 range')
            return c, c, c, alpha if alpha is not None else 1.
        raise ValueError(f'Invalid RGBA argument: {orig_c!r}')
    if isinstance(c, np.ndarray):
        if c.ndim == 2 and c.shape[0] == 1:
            c = c.reshape(-1)
    if not np.iterable(c):
        raise ValueError(f'Invalid RGBA argument: {orig_c!r}')
    if len(c) not in [3, 4]:
        raise ValueError('RGBA sequence should have length 3 or 4')
    if not all(isinstance(x, Real) for x in c):
        raise ValueError(f'Invalid RGBA argument: {orig_c!r}')
    c = tuple(map(float, c))
    if len(c) == 3 and alpha is None:
        alpha = 1
    if alpha is not None:
        c = c[:3] + (alpha,)
    if any(elem < 0 or elem > 1 for elem in c):
        raise ValueError('RGBA values should be within 0-1 range')
    return c


def _str_lower_equal(obj, s):
    return isinstance(obj, str) and obj.lower() == s


def to_rgba_array(c, alpha=None):
    """`c` -- one colour or a sequence of them, or an (n, 3|4) array -- as an
    (n, 4) float array, as matplotlib.colors.to_rgba_array makes it ('none'
    alone is an empty (0, 4) array; masked rows of a masked array are
    zero). `alpha` may be a sequence, one value per colour."""
    if isinstance(c, tuple) and len(c) == 2 and isinstance(c[1], Real):
        if alpha is None:
            c, alpha = c
        else:
            c = c[0]
    if np.iterable(alpha):
        alpha = np.asarray(alpha).ravel()
    if (isinstance(c, np.ndarray) and c.dtype.kind in 'if'
            and c.ndim == 2 and c.shape[1] in [3, 4]):
        mask = c.mask.any(axis=1) if np.ma.is_masked(c) else None
        c = np.ma.getdata(c)
        if np.iterable(alpha):
            if c.shape[0] == 1 and alpha.shape[0] > 1:
                c = np.tile(c, (alpha.shape[0], 1))
            elif c.shape[0] != alpha.shape[0]:
                raise ValueError('The number of colors must match the number'
                                 ' of alpha values if there are more than one'
                                 ' of each.')
        if c.shape[1] == 3:
            result = np.column_stack([c, np.zeros(len(c))])
            result[:, -1] = alpha if alpha is not None else 1.
        else:
            result = c.copy()
            if alpha is not None:
                result[:, -1] = alpha
        if mask is not None:
            result[mask] = 0
        if np.any((result < 0) | (result > 1)):
            raise ValueError('RGBA values should be within 0-1 range')
        return result
    if _str_lower_equal(c, 'none'):
        return np.zeros((0, 4), float)
    try:
        if np.iterable(alpha):
            return np.array([to_rgba(c, a) for a in alpha], float)
        return np.array([to_rgba(c, alpha)], float)
    except TypeError:
        pass
    except ValueError as e:
        if e.args == ("'alpha' must be between 0 and 1, inclusive", ):
            raise e
    if isinstance(c, str):
        raise ValueError(f'{c!r} is not a valid color value.')
    if len(c) == 0:
        return np.zeros((0, 4), float)
    if isinstance(c, Sequence):
        lens = {len(cc) if isinstance(cc, (list, tuple)) else -1 for cc in c}
        if lens == {3}:
            rgba = np.column_stack([c, np.ones(len(c))])
        elif lens == {4}:
            rgba = np.array(c)
        else:
            rgba = np.array([to_rgba(cc) for cc in c])
    else:
        rgba = np.array([to_rgba(cc) for cc in c])
    if alpha is not None:
        rgba[:, 3] = alpha
        if isinstance(c, Sequence):
            none_mask = [isinstance(cc, str) and cc == 'none' for cc in c]
            rgba[:, 3][none_mask] = 0
    return rgba


def _to_bytes(f, rounding):
    """Float RGBA (any shape) as uint8: 'floor' as matplotlib's images make
    it, uint8(255 f); 'nearest' as Agg paints a float colour,
    uint8(255 f + 0.5)."""
    f = np.asarray(f, dtype=float)
    if rounding == 'floor':
        return (f * 255).astype(np.uint8)
    if rounding == 'nearest':
        return np.floor(f * 255 + 0.5).astype(np.uint8)
    raise ValueError("rounding must be 'floor' or 'nearest', not %r" % (rounding,))


_BYTES_CACHE = {}                   # (colour, alpha, rounding) -> uint8[4], as to_rgba's cache


def to_rgba_bytes(c, alpha=None, rounding='floor'):
    """`c` as uint8[4]: 'floor' is matplotlib's bytes for an image or a
    colormap (uint8(255 v)), 'nearest' Agg's for a line or a face colour
    (uint8(255 v + 0.5)). A new array each call."""
    try:
        return _BYTES_CACHE[c, alpha, rounding].copy()
    except (KeyError, TypeError):
        pass
    out = _to_bytes(to_rgba(c, alpha), rounding)
    try:
        _BYTES_CACHE[c, alpha, rounding] = out.copy()
    except TypeError:                                   # (unhashable: a list, an array)
        pass
    return out


# --------------------------------------------------------------------------
# colormaps

def _create_lookup_table(N, data, gamma=1.0):
    """matplotlib.colors._create_lookup_table: N values of the mapping
    `data` (a callable, or (x, y0, y1) rows) sampled at linspace(0, 1, N)
    ** gamma -- its arithmetic, step for step."""
    if callable(data):
        xind = np.linspace(0, 1, N) ** gamma
        lut = np.clip(np.array(data(xind), dtype=float), 0, 1)
        return lut
    adata = np.array(data)
    if adata.ndim != 2 or adata.shape[1] != 3:
        raise ValueError('data must be an (M, 3) array')
    x = adata[:, 0]
    y0 = adata[:, 1]
    y1 = adata[:, 2]
    if x[0] != 0. or x[-1] != 1.0:
        raise ValueError('data mapping points must start with x=0 and end with x=1')
    if (np.diff(x) < 0).any():
        raise ValueError('data mapping points must have x in increasing order')
    if N == 1:
        lut = np.array(y0[-1])
    else:
        x = x * (N - 1)
        xind = (N - 1) * np.linspace(0, 1, N) ** gamma
        ind = np.searchsorted(x, xind)[1:-1]
        distance = (xind[1:-1] - x[ind - 1]) / (x[ind] - x[ind - 1])
        lut = np.concatenate([
            [y1[0]],
            distance * (y0[ind] - y1[ind - 1]) + y1[ind - 1],
            [y0[-1]],
        ])
    return np.clip(lut, 0.0, 1.0)


# matplotlib's colormap functions (matplotlib/_cm.py), the same expressions
_G = {
    0: lambda x: 0,
    1: lambda x: 0.5,
    2: lambda x: 1,
    3: lambda x: x,
    4: lambda x: x ** 2,
    5: lambda x: x ** 3,
    6: lambda x: x ** 4,
    7: lambda x: np.sqrt(x),
    8: lambda x: np.sqrt(np.sqrt(x)),
    9: lambda x: np.sin(x * np.pi / 2),
    10: lambda x: np.cos(x * np.pi / 2),
    11: lambda x: np.abs(x - 0.5),
    12: lambda x: (2 * x - 1) ** 2,
    13: lambda x: np.sin(x * np.pi),
    14: lambda x: np.abs(np.cos(x * np.pi)),
    15: lambda x: np.sin(x * 2 * np.pi),
    16: lambda x: np.cos(x * 2 * np.pi),
    17: lambda x: np.abs(np.sin(x * 2 * np.pi)),
    18: lambda x: np.abs(np.cos(x * 2 * np.pi)),
    19: lambda x: np.abs(np.sin(x * 4 * np.pi)),
    20: lambda x: np.abs(np.cos(x * 4 * np.pi)),
    21: lambda x: 3 * x,
    22: lambda x: 3 * x - 1,
    23: lambda x: 3 * x - 2,
    24: lambda x: np.abs(3 * x - 1),
    25: lambda x: np.abs(3 * x - 2),
    26: lambda x: (3 * x - 1) / 2,
    27: lambda x: (3 * x - 2) / 2,
    28: lambda x: np.abs((3 * x - 1) / 2),
    29: lambda x: np.abs((3 * x - 2) / 2),
    30: lambda x: x / 0.32 - 0.78125,
    31: lambda x: 2 * x - 0.84,
    33: lambda x: np.abs(2 * x - 0.5),
    34: lambda x: 2 * x,
    35: lambda x: 2 * x - 0.5,
    36: lambda x: 2 * x - 1,
}


def _g32(x):
    ret = np.zeros(len(x))
    m = (x < 0.25)
    ret[m] = 4 * x[m]
    m = (x >= 0.25) & (x < 0.92)
    ret[m] = -2 * x[m] + 1.84
    m = (x >= 0.92)
    ret[m] = x[m] / 0.08 - 11.5
    return ret


_G[32] = _g32

_FUNCS = {
    'flag_red': lambda x: 0.75 * np.sin((x * 31.5 + 0.25) * np.pi) + 0.5,
    'flag_green': lambda x: np.sin(x * 31.5 * np.pi),
    'flag_blue': lambda x: 0.75 * np.sin((x * 31.5 - 0.25) * np.pi) + 0.5,
    'prism_red': lambda x: 0.75 * np.sin((x * 20.9 + 0.25) * np.pi) + 0.67,
    'prism_green': lambda x: 0.75 * np.sin((x * 20.9 - 0.25) * np.pi) + 0.33,
    'prism_blue': lambda x: -1.1 * np.sin((x * 20.9) * np.pi),
    'gist_heat_red': lambda x: 1.5 * x,
    'gist_heat_green': lambda x: 2 * x - 1,
    'gist_heat_blue': lambda x: 4 * x - 3,
    'gist_yarg': lambda x: 1 - x,
}


def _ch_helper(gamma, s, r, h, p0, p1, x):
    """matplotlib's cubehelix channel."""
    xg = x ** gamma
    a = h * xg * (1 - xg) / 2
    phi = 2 * np.pi * (s / 3 + r * x)
    return xg + a * (p0 * np.cos(phi) + p1 * np.sin(phi))


def _segment_function(spec):
    """The function a name in cmaps.npz stands for."""
    if spec.startswith('ch:'):
        return functools.partial(_ch_helper, *map(float, spec[3:].split(',')))
    if spec[0] == 'g' and spec[1:].isdigit():
        return _G[int(spec[1:])]
    return _FUNCS[spec]


def _reverser(func, x):
    return func(1 - x)


class Colormap:
    """A colormap: N colours plus the colours of values under the range, over
    it, and bad (NaN), as matplotlib.colors.Colormap holds them and with its
    defaults (under: the first colour, over: the last, bad: transparent).

    `lut` (N, 4) uint8 is matplotlib's `cmap(np.arange(N), bytes=True)` and
    `under` / `over` / `bad` its bytes for those; table(rounding) is the
    (N + 3, 4) uint8 table the kernels index (colours, under, over, bad),
    and rgba the float colours. Colormaps are not changed in place:
    reversed(), resampled(), with_extremes() and with_alpha() make new ones,
    as matplotlib's do."""

    def __init__(self, name, N=256, *, bad=None, under=None, over=None):
        self.name = name
        self.N = int(N)
        self._rgba_bad = (0.0, 0.0, 0.0, 0.0) if bad is None else to_rgba(bad)
        self._rgba_under = None if under is None else to_rgba(under)
        self._rgba_over = None if over is None else to_rgba(over)
        self._alpha = None              # with_alpha(): every colour's alpha
        self._alpha_ext = False         # ... and the extremes' too
        self._native = None             # bundled bytes of the N colours
        self._cache = {}
        self.colorbar_extend = False

    # the N colours, float64 (N, 4); a subclass makes them
    def _colors(self):
        raise NotImplementedError

    def _float_table(self):
        """float64 (N + 3, 4): matplotlib's `_lut` (colours, under, over, bad)."""
        t = self._cache.get('f')
        if t is None:
            N = self.N
            t = np.ones((N + 3, 4), float)
            t[:N] = self._colors()
            if self._alpha is not None:
                t[:N, 3] = self._alpha
            t[N] = self._rgba_under if self._rgba_under else t[0]
            t[N + 1] = self._rgba_over if self._rgba_over else t[N - 1]
            t[N + 2] = self._rgba_bad
            if self._alpha is not None and self._alpha_ext:
                t[N:, 3] = self._alpha
            t.setflags(write=False)
            self._cache['f'] = t
        return t

    def table(self, rounding='floor'):
        """uint8 (N + 3, 4): the N colours, then under, over and bad, as
        bytes ('floor': matplotlib's image bytes; 'nearest': Agg's)."""
        t = self._cache.get(rounding)
        if t is None:
            if rounding == 'floor' and self._native is not None and self._alpha is None:
                N = self.N
                t = np.empty((N + 3, 4), np.uint8)
                t[:N] = self._native
                t[N] = _to_bytes(self._rgba_under, 'floor') if self._rgba_under else t[0]
                t[N + 1] = _to_bytes(self._rgba_over, 'floor') if self._rgba_over else t[N - 1]
                t[N + 2] = _to_bytes(self._rgba_bad, 'floor')
            else:
                t = _to_bytes(self._float_table(), rounding)
            t.setflags(write=False)
            self._cache[rounding] = t
        return t

    @property
    def lut(self):
        """(N, 4) uint8: matplotlib's `cmap(np.arange(N), bytes=True)`."""
        return self.table()[:self.N]

    @property
    def under(self):
        return self.table()[self.N]

    @property
    def over(self):
        return self.table()[self.N + 1]

    @property
    def bad(self):
        return self.table()[self.N + 2]

    @property
    def rgba(self):
        """(N, 4) float64: the colours."""
        return self._float_table()[:self.N]

    # matplotlib's accessors, so a fastplot colormap passes for one
    def get_under(self):
        return np.array(self._float_table()[self.N])

    def get_over(self):
        return np.array(self._float_table()[self.N + 1])

    def get_bad(self):
        return np.array(self._float_table()[self.N + 2])

    def _rows(self, X):
        """Table rows of X as Colormap.__call__ picks them: floats are
        fractions of the range (x N, N itself the last colour), integers
        indices; < 0 under, >= N over, NaN or masked bad."""
        xa = np.array(X, copy=True)
        if not xa.dtype.isnative:
            xa = xa.byteswap().view(xa.dtype.newbyteorder())
        if xa.dtype.kind == 'f':
            xa *= self.N
            xa[xa == self.N] = self.N - 1
        mask_under = xa < 0
        mask_over = xa >= self.N
        mask_bad = X.mask if np.ma.is_masked(X) else np.isnan(xa)
        with np.errstate(invalid='ignore'):
            xa = xa.astype(int)
        xa[mask_under] = self.N
        xa[mask_over] = self.N + 1
        xa[mask_bad] = self.N + 2
        return xa, mask_bad

    def __call__(self, X, alpha=None, bytes=False, rounding='floor'):
        """RGBA of X, as matplotlib's Colormap.__call__ (floats in 0..1, or
        bytes; `rounding` picks the bytes)."""
        xa, mask_bad = self._rows(X)
        lut = self.table(rounding) if bytes else self._float_table()
        rgba = lut.take(xa, axis=0, mode='clip')
        if alpha is not None:
            alpha = np.clip(alpha, 0, 1)
            if bytes:
                alpha = (np.floor(alpha * 255 + 0.5) if rounding == 'nearest'
                         else alpha * 255)
            if alpha.shape not in [(), xa.shape]:
                raise ValueError(f'alpha is array-like but its shape {alpha.shape} '
                                 f'does not match that of X {xa.shape}')
            rgba[..., -1] = alpha
            if (lut[-1] == 0).all():
                rgba[mask_bad] = (0, 0, 0, 0)
        if not np.iterable(X):
            rgba = tuple(rgba)
        return rgba

    def copy(self):
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        new._cache = {}
        return new

    __copy__ = copy

    def with_extremes(self, *, bad=None, under=None, over=None):
        """A copy with these colours for bad, under and over values (None
        keeps the current one)."""
        new = self.copy()
        if bad is not None:
            new._rgba_bad = to_rgba(bad)
        if under is not None:
            new._rgba_under = to_rgba(under)
        if over is not None:
            new._rgba_over = to_rgba(over)
        new._alpha_ext = False                  # matplotlib rebuilds those rows
        return new

    def set_bad(self, color='k', alpha=None):
        """matplotlib's (in place): the colour of bad (NaN, masked) values."""
        self._set_extreme('_rgba_bad', color, alpha)

    def set_under(self, color='k', alpha=None):
        """matplotlib's (in place): the colour of values under the norm's range."""
        self._set_extreme('_rgba_under', color, alpha)

    def set_over(self, color='k', alpha=None):
        """matplotlib's (in place): the colour of values over the norm's range."""
        self._set_extreme('_rgba_over', color, alpha)

    def set_extremes(self, *, bad=None, under=None, over=None):
        for name, c in (('_rgba_bad', bad), ('_rgba_under', under), ('_rgba_over', over)):
            if c is not None:
                self._set_extreme(name, c, None)

    def _set_extreme(self, name, color, alpha):
        setattr(self, name, to_rgba(color, alpha))
        self._alpha_ext = False
        self._cache = {}

    def with_alpha(self, alpha):
        """A copy with every colour's alpha (extremes' too) set to `alpha`."""
        if not isinstance(alpha, Real):
            raise TypeError(f"'alpha' must be numeric or None, not {type(alpha)}")
        if not 0 <= alpha <= 1:
            raise ValueError("'alpha' must be between 0 and 1, inclusive")
        new = self.copy()
        new._alpha, new._alpha_ext = float(alpha), True
        return new

    def _swapped_extremes(self, new):
        new._rgba_over = self._rgba_under
        new._rgba_under = self._rgba_over
        new._rgba_bad = self._rgba_bad
        return new

    def reversed(self, name=None):
        raise NotImplementedError

    def resampled(self, lutsize):
        raise NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, Colormap):
            return NotImplemented
        return np.array_equal(self._float_table(), other._float_table())

    __hash__ = object.__hash__

    def __repr__(self):
        return '<%s %r N=%d>' % (type(self).__name__, self.name, self.N)


class ListedColormap(Colormap):
    """A colormap of a list of colours (matplotlib.colors.ListedColormap):
    one per entry, or N of them, the list cut or repeated."""

    def __init__(self, colors, name='unnamed', N=None, *, bad=None, under=None,
                 over=None):
        if N is None:
            self.colors = colors
            N = len(colors)
        elif isinstance(colors, str):
            self.colors = [colors] * N
        elif np.iterable(colors):
            self.colors = list(itertools.islice(itertools.cycle(colors), N))
        else:
            self.colors = [float(colors)] * N
        super().__init__(name, N, bad=bad, under=under, over=over)

    def _colors(self):
        return to_rgba_array(self.colors)

    def resampled(self, lutsize):
        """lutsize colours sampled from this map, as matplotlib resamples a
        ListedColormap; extremes kept."""
        new = ListedColormap(self(np.linspace(0, 1, lutsize)), name=self.name)
        new._rgba_over, new._rgba_under = self._rgba_over, self._rgba_under
        new._rgba_bad = self._rgba_bad
        return new

    def reversed(self, name=None):
        """The colours backwards (name + '_r'), under and over swapped."""
        if name is None:
            name = self.name + '_r'
        if isinstance(self.colors, np.ndarray):
            colors_r = self.colors[::-1]
        else:
            colors_r = list(reversed(self.colors))
        return self._swapped_extremes(ListedColormap(colors_r, name=name))


class LinearSegmentedColormap(Colormap):
    """A colormap of piecewise-linear channels (or channel functions),
    matplotlib.colors.LinearSegmentedColormap: `segmentdata` maps 'red',
    'green', 'blue' (and optionally 'alpha') to (x, y0, y1) rows or to
    functions of x in 0..1."""

    def __init__(self, name, segmentdata, N=256, gamma=1.0, *, bad=None,
                 under=None, over=None):
        super().__init__(name, N, bad=bad, under=under, over=over)
        self._segmentdata = segmentdata
        self._gamma = gamma

    def _colors(self):
        sd, N = self._segmentdata, self.N
        c = np.ones((N, 4), float)
        c[:, 0] = _create_lookup_table(N, sd['red'], self._gamma)
        c[:, 1] = _create_lookup_table(N, sd['green'], self._gamma)
        c[:, 2] = _create_lookup_table(N, sd['blue'], self._gamma)
        if 'alpha' in sd:
            c[:, 3] = _create_lookup_table(N, sd['alpha'], 1)
        return c

    @staticmethod
    def from_list(name, colors, N=256, gamma=1.0, *, bad=None, under=None, over=None):
        """A map running through `colors` evenly (or through (value, colour)
        pairs, values 0 to 1 increasing), as matplotlib's from_list."""
        if not np.iterable(colors):
            raise ValueError('colors must be iterable')
        try:
            r, g, b, a = to_rgba_array(colors).T
            vals = np.linspace(0, 1, len(colors))
        except Exception as e:
            try:
                _vals, _colors = itertools.zip_longest(*colors)
            except Exception as e2:
                raise e2 from e
            vals = np.asarray(_vals)
            if np.min(vals) < 0 or np.max(vals) > 1 or np.any(np.diff(vals) < 0):
                raise ValueError('the values passed in the (value, color) pairs '
                                 'must increase monotonically from 0 to 1.')
            r, g, b, a = to_rgba_array(_colors).T
        cdict = {
            'red': np.column_stack([vals, r, r]),
            'green': np.column_stack([vals, g, g]),
            'blue': np.column_stack([vals, b, b]),
            'alpha': np.column_stack([vals, a, a]),
        }
        return LinearSegmentedColormap(name, cdict, N, gamma, bad=bad, under=under,
                                       over=over)

    def resampled(self, lutsize):
        """The same segments at lutsize entries (gamma back to 1, as in
        matplotlib); extremes kept."""
        new = LinearSegmentedColormap(self.name, self._segmentdata, lutsize)
        new._rgba_over, new._rgba_under = self._rgba_over, self._rgba_under
        new._rgba_bad = self._rgba_bad
        return new

    def reversed(self, name=None):
        """The segments mirrored (name + '_r'), under and over swapped --
        matplotlib's construction, so the table is matplotlib's '_r' table
        (not this one backwards: the two differ in the last bit)."""
        if name is None:
            name = self.name + '_r'
        data_r = {}
        for key, data in self._segmentdata.items():
            if callable(data):
                data_r[key] = functools.partial(_reverser, data)
            else:
                a = np.asarray(data, dtype=float)[::-1]
                data_r[key] = np.column_stack([1.0 - a[:, 0], a[:, 2], a[:, 1]])
        new = LinearSegmentedColormap(name, data_r, self.N, self._gamma)
        return self._swapped_extremes(new)


class _ForeignColormap(Colormap):
    """A matplotlib Colormap object (or anything like one: .N, callable,
    get_under/get_over/get_bad), read through its own interface."""

    def __init__(self, obj):
        super().__init__(getattr(obj, 'name', 'unnamed'), obj.N)
        self._obj = obj
        self._rgba_under = tuple(float(v) for v in np.asarray(obj.get_under(), float))
        self._rgba_over = tuple(float(v) for v in np.asarray(obj.get_over(), float))
        self._rgba_bad = tuple(float(v) for v in np.asarray(obj.get_bad(), float))
        self.colorbar_extend = getattr(obj, 'colorbar_extend', False)

    def _colors(self):
        c = np.asarray(self._obj(np.arange(self.N)), dtype=float)
        if c.shape != (self.N, 4):
            raise ValueError('colormap %r gave %s for %d indices' % (self.name, c.shape, self.N))
        return c

    def reversed(self, name=None):
        if name is None:
            return _ForeignColormap(self._obj.reversed())
        return _ForeignColormap(self._obj.reversed(name))

    def resampled(self, lutsize):
        return _ForeignColormap(self._obj.resampled(lutsize))


_CMDATA = None
_CMCACHE = {}


def _cmap_data():
    global _CMDATA
    if _CMDATA is None:
        with _LOCK:
            if _CMDATA is None:
                with np.load(os.path.join(_DATA, 'cmaps.npz')) as z:
                    d = {k: z[k] for k in z.files}
                d['index'] = {str(n): i for i, n in enumerate(d['names'])}
                _CMDATA = d
    return _CMDATA


def _bundled(name):
    """The bundled base map `name`, built as matplotlib builds it."""
    cm = _CMCACHE.get(name)
    if cm is not None:
        return cm
    d = _cmap_data()
    i = d['index'][name]
    N = int(d['N'][i])
    kind = int(d['kind'][i])
    under, over, bad = (None if np.isnan(e).any() else tuple(float(v) for v in e)
                        for e in d['extremes'][i])
    if kind == 0:
        cols = d['listed'][d['lst0'][i]:d['lst0'][i + 1]]
        cm = ListedColormap(cols, name, bad=bad, under=under, over=over)
    else:
        sd = {}
        if kind == 1:
            for ci, ch in enumerate(('red', 'green', 'blue', 'alpha')):
                a, b = d['seg_idx'][i, ci]
                if a >= 0:
                    sd[ch] = d['seg'][a:b]
        else:
            for ci, ch in enumerate(('red', 'green', 'blue')):
                sd[ch] = _segment_function(str(d['func'][i, ci]))
        cm = LinearSegmentedColormap(name, sd, N, float(d['gamma'][i]), bad=bad,
                                     under=under, over=over)
    cm._native = d['lut'][d['lut0'][i]:d['lut0'][i + 1]]
    with _LOCK:
        return _CMCACHE.setdefault(name, cm)


def cmap_names():
    """Every colormap name get_cmap knows: matplotlib's registry (base maps
    and their '_r'), sorted."""
    base = [str(n) for n in _cmap_data()['names']]
    return sorted(base + [n + '_r' for n in base])


def get_cmap(cmap=None, lut=None):
    """A Colormap from `cmap`: None (matplotlib's default, 'viridis'), a name
    matplotlib registers ('_r' for reversed), a fastplot Colormap, a
    matplotlib Colormap object (read through its interface, never
    imported), or a list of colours (a ListedColormap). `lut` resamples it
    to that many entries, as matplotlib's get_cmap(name, lut)."""
    if cmap is None:
        cmap = 'viridis'
    if isinstance(cmap, Colormap):
        cm = cmap
    elif isinstance(cmap, str):
        idx = _cmap_data()['index']
        if cmap in idx:
            cm = _bundled(cmap)
        elif cmap.endswith('_r') and cmap[:-2] in idx:
            key = cmap
            cm = _CMCACHE.get(key)
            if cm is None:
                cm = _bundled(cmap[:-2]).reversed()
                with _LOCK:
                    cm = _CMCACHE.setdefault(key, cm)
        else:
            raise ValueError('%r is not a known colormap name; see cmap_names()' % cmap)
    elif hasattr(cmap, 'N') and callable(cmap) and hasattr(cmap, 'get_bad'):
        cm = _ForeignColormap(cmap)
    elif np.iterable(cmap):
        cm = ListedColormap(cmap)
    else:
        raise TypeError('cannot make a colormap of %r' % (cmap,))
    if lut is not None:
        cm = cm.resampled(int(lut))
    return cm


# --------------------------------------------------------------------------
# norms

_KINDS = ('linear', 'log', 'symlog', 'twoslope', 'centered', 'boundary', 'nonorm',
          'func')


class Norm:
    """How values map onto a colormap: matplotlib's norms as plain data.

    kind:
      'linear'   Normalize(vmin, vmax, clip)
      'log'      LogNorm(vmin, vmax, clip): non-positive values are bad
      'symlog'   SymLogNorm(linthresh, linscale, vmin, vmax, clip, base)
      'twoslope' TwoSlopeNorm(vcenter, vmin, vmax)
      'centered' CenteredNorm(vcenter, halfrange, clip): linear about vcenter
      'boundary' BoundaryNorm(boundaries, ncolors, clip, extend)
      'nonorm'   NoNorm(): values are the colormap's own input
      'func'     a matplotlib norm fastplot does not read, called as it is
                 (`func`; CPU only)

    Missing vmin / vmax are filled from data by make_norm, as a
    ScalarMappable autoscales its norm."""

    def __init__(self, kind='linear', vmin=None, vmax=None, *, clip=False,
                 vcenter=None, halfrange=None, boundaries=None, ncolors=None,
                 extend='neither', linthresh=None, linscale=1.0, base=10.0, func=None):
        if kind not in _KINDS:
            raise ValueError('Norm kind must be one of %s, not %r' % (_KINDS, kind))
        self.kind = kind
        self.vmin = None if vmin is None else float(vmin)
        self.vmax = None if vmax is None else float(vmax)
        self.clip = bool(clip)
        self.vcenter = vcenter
        self.halfrange = halfrange
        self.boundaries = None
        self.ncolors = ncolors
        self.extend = extend
        self.linthresh = linthresh
        self.linscale = linscale
        self.base = base
        self.func = func
        if kind == 'twoslope':
            if vcenter is None:
                raise ValueError("Norm('twoslope') needs vcenter")
            if self.vmax is not None and vcenter >= self.vmax or \
                    self.vmin is not None and vcenter <= self.vmin:
                raise ValueError('vmin, vcenter, and vmax must be in ascending order')
        elif kind == 'centered':
            if vcenter is None:
                self.vcenter = 0
            if halfrange is not None:
                self.vmin, self.vmax = _centered_limits(self.vcenter, halfrange)
        elif kind == 'boundary':
            if boundaries is None:
                raise ValueError("Norm('boundary') needs boundaries")
            if clip and extend != 'neither':
                raise ValueError("'clip=True' is not compatible with 'extend'")
            b = np.asarray(boundaries)
            if len(b) < 2:
                raise ValueError('You must provide at least 2 boundaries (1 region) '
                                 f'but you passed in {boundaries!r}')
            self.boundaries = b
            self.vmin, self.vmax = _sanitize(b[0]), _sanitize(b[-1])
        elif kind == 'symlog':
            if linthresh is None:
                raise ValueError("Norm('symlog') needs linthresh")
        elif kind == 'func' and func is None:
            raise ValueError("Norm('func') needs func")

    def scaled(self):
        return self.kind in ('boundary', 'nonorm') or (
            self.vmin is not None and self.vmax is not None)

    def copy(self):
        new = Norm.__new__(Norm)
        new.__dict__.update(self.__dict__)
        return new

    def __repr__(self):
        extra = ''
        if self.kind == 'boundary':
            extra = ', %d boundaries, ncolors=%s, extend=%r' % (
                len(self.boundaries), self.ncolors, self.extend)
        elif self.kind in ('twoslope', 'centered'):
            extra = ', vcenter=%r' % (self.vcenter,)
        elif self.kind == 'symlog':
            extra = ', linthresh=%r' % (self.linthresh,)
        return 'Norm(%r, %r, %r%s%s)' % (self.kind, self.vmin, self.vmax,
                                         ', clip=True' if self.clip else '', extra)


def _sanitize(v):
    """matplotlib's _sanitize_extrema: a Python number."""
    if v is None:
        return None
    try:
        return v.item()
    except AttributeError:
        return float(v)


def _centered_limits(vcenter, halfrange):
    """CenteredNorm's vmin, vmax after `halfrange = h` -- its setters in
    order (each limit set moves the other to the mirror image), in the
    arithmetic of h's type."""
    vmin = vmax = None
    v = _sanitize(vcenter - abs(halfrange))
    if v != vmin:
        vmin, vmax = v, 2 * vcenter - v
    v = _sanitize(vcenter + abs(halfrange))
    if v != vmax:
        vmax, vmin = v, 2 * vcenter - v
    return vmin, vmax


def _class_names(obj):
    return [c.__name__ for c in type(obj).__mro__]


def _from_matplotlib(obj, N):
    """A Norm from a matplotlib norm object, read by its class name and
    attributes (matplotlib is never imported)."""
    cls = type(obj).__name__
    names = _class_names(obj)
    if 'Normalize' not in names and 'Norm' not in names:
        raise TypeError('%r is not a norm' % (obj,))
    vmin, vmax = _sanitize(getattr(obj, 'vmin', None)), _sanitize(getattr(obj, 'vmax', None))
    clip = bool(getattr(obj, 'clip', False))
    if cls == 'NoNorm':
        return Norm('nonorm')
    if cls == 'BoundaryNorm':
        return Norm('boundary', boundaries=np.asarray(obj.boundaries),
                    ncolors=int(obj.Ncmap), extend=obj.extend, clip=clip)
    if cls == 'TwoSlopeNorm':
        return Norm('twoslope', vmin, vmax, vcenter=obj.vcenter)
    if cls == 'CenteredNorm':
        n = Norm('centered', vcenter=obj.vcenter, clip=clip)
        n.vmin, n.vmax = vmin, vmax
        n.halfrange = obj.halfrange
        return n
    if cls == 'LogNorm' and getattr(getattr(obj, '_scale', None), 'base', 10) == 10:
        return Norm('log', vmin, vmax, clip=clip)
    if cls == 'SymLogNorm':
        sc = obj._scale
        return Norm('symlog', vmin, vmax, clip=clip, linthresh=float(sc.linthresh),
                    linscale=float(sc.linscale), base=float(sc.base))
    if cls == 'Normalize':
        return Norm('linear', vmin, vmax, clip=clip)
    return Norm('func', vmin, vmax, clip=clip, func=obj)


def _finite(data):
    a = np.asarray(data)
    if a.dtype.kind not in 'fc':
        return a.ravel()
    return a[np.isfinite(a)]


def _autoscale(n, data):
    """Fill n's missing limits from data, as matplotlib's autoscale_None
    does for that norm on the data with NaN / inf masked."""
    if n.kind in ('boundary', 'nonorm'):
        return
    if n.kind == 'func':
        import copy
        f = copy.copy(n.func)
        if n.vmin is not None:
            f.vmin = n.vmin
        if n.vmax is not None:
            f.vmax = n.vmax
        a = np.ma.masked_invalid(np.asarray(data))
        f.autoscale_None(a)
        n.func = f
        n.vmin, n.vmax = _sanitize(f.vmin), _sanitize(f.vmax)
        return
    if n.kind == 'centered':
        if n.vmin is None or n.vmax is None:
            a = _finite(data)
            if a.size:
                h = max(n.vcenter - a.min(), a.max() - n.vcenter)
                n.halfrange = h
                n.vmin, n.vmax = _centered_limits(n.vcenter, h)
        return
    if n.kind == 'log':
        a = _finite(data)
        a = a[a > 0]
    else:
        a = _finite(data)
    if a.size:
        if n.vmin is None:
            n.vmin = _sanitize(a.min())
        if n.vmax is None:
            n.vmax = _sanitize(a.max())
    if n.kind == 'twoslope' and n.vmin is not None and n.vmax is not None:
        if n.vmin >= n.vcenter:
            n.vmin = _sanitize(n.vcenter - (n.vmax - n.vcenter))
        if n.vmax <= n.vcenter:
            n.vmax = _sanitize(n.vcenter + (n.vcenter - n.vmin))


def make_norm(norm=None, vmin=None, vmax=None, data=None, N=256):
    """A Norm with its limits, from what a matplotlib plotting call takes:

    norm  None (Normalize), 'linear' / 'log' / 'symlog' (matplotlib's scale
          names), a Norm, or a matplotlib norm object (Normalize, LogNorm,
          SymLogNorm, TwoSlopeNorm, CenteredNorm, BoundaryNorm, NoNorm read
          by class name; any other is called as it is)
    vmin, vmax  limits, as the vmin= / vmax= of a plotting call (an error
          with a norm object, as in matplotlib; with a Norm they fill its
          missing limits)
    data  the values: limits still missing come from them, NaN and inf
          left out, as a ScalarMappable autoscales
    N     ncolors of a boundary Norm given none (the colormap's N)

    The result is a new Norm; the norm passed is never changed (matplotlib
    autoscales its object in place: to keep one frame's limits for the
    next, keep the Norm this returns)."""
    explicit = vmin is not None or vmax is not None
    if norm is None:
        n = Norm('linear')
    elif isinstance(norm, str):
        if norm == 'linear':
            n = Norm('linear')
        elif norm == 'log':
            n = Norm('log')
        elif norm == 'symlog':
            n = Norm('symlog', linthresh=2.0)       # SymmetricalLogScale's defaults
        else:
            raise ValueError('unsupported norm name %r' % norm)
    elif isinstance(norm, Norm):
        n = norm.copy()
        if explicit:
            for v, attr in ((vmin, 'vmin'), (vmax, 'vmax')):
                if v is not None and getattr(n, attr) is not None and \
                        getattr(n, attr) != _sanitize(v):
                    raise ValueError('vmin/vmax given with a Norm that has them')
    else:
        if explicit:
            raise ValueError('Passing a Normalize instance simultaneously with '
                             'vmin/vmax is not supported.  Please pass vmin/vmax '
                             'as arguments to the norm object when creating it')
        n = _from_matplotlib(norm, N)
    if explicit:
        if n.kind == 'centered':
            raise ValueError('vmin/vmax with a centered norm: give halfrange')
        if vmin is not None:
            n.vmin = _sanitize(vmin)
        if vmax is not None:
            n.vmax = _sanitize(vmax)
    if n.kind == 'boundary' and n.ncolors is None:
        n.ncolors = int(N)
    if n.kind == 'boundary':
        _check_boundary(n)
    if not n.scaled() and data is not None:
        _autoscale(n, data)
    elif n.kind == 'twoslope' and n.scaled():
        _autoscale(n, np.zeros(0))
    return n


def _check_boundary(n):
    nreg = len(n.boundaries) - 1 + (n.extend in ('min', 'both')) + (n.extend in ('max', 'both'))
    if nreg > n.ncolors:
        raise ValueError(f'There are {nreg} color bins including extensions, but '
                         f'ncolors = {n.ncolors}; ncolors must equal or exceed the '
                         f'number of bins')


# --------------------------------------------------------------------------
# contour levels and colours

def auto_levels(n, zmin, zmax, extend='neither'):
    """The levels contourf(z, n, extend=extend) picks for data spanning
    zmin..zmax: MaxNLocator(n + 1, min_n_ticks=1) ticks, trimmed to the data
    as ContourSet._autolev trims them (n None: matplotlib's 7). Linear
    scales only (a LogNorm contour's LogLocator is not reproduced)."""
    N = 7 if n is None else int(n)
    zmin, zmax = float(zmin), float(zmax)
    lev = _ticker.MaxNLocator(N + 1, min_n_ticks=1).tick_values(zmin, zmax)
    under = np.nonzero(lev < zmin)[0]
    i0 = under[-1] if len(under) else 0
    over = np.nonzero(lev > zmax)[0]
    i1 = over[0] + 1 if len(over) else len(lev)
    if extend in ('min', 'both'):
        i0 += 1
    if extend in ('max', 'both'):
        i1 -= 1
    if i1 - i0 < 3:
        i0, i1 = 0, len(lev)
    return lev[i0:i1]


def _resize_sequence(seq, N):
    n = len(seq)
    if N == n:
        return seq
    if N < n:
        return seq[:N]
    return list(itertools.islice(itertools.cycle(seq), N))


def _is_log(norm):
    if norm is None or isinstance(norm, str):
        return norm == 'log'
    if isinstance(norm, Norm):
        return norm.kind == 'log'
    return 'LogNorm' in _class_names(norm)


def contour_colors(levels, cmap=None, colors=None, norm=None, vmin=None, vmax=None,
                   extend='neither', alpha=None, *, rounding='nearest'):
    """The colours matplotlib's contourf(..., levels=levels, cmap=cmap,
    colors=colors, norm=norm, vmin=vmin, vmax=vmax, extend=extend,
    alpha=alpha) gives its filled bands: (bands (len(levels) - 1, 4),
    under, over), under / over the colours of the extended bands below and
    above the levels (None when not extended).

    As ContourSet does it: each band takes the colour of its middle value
    (the geometric middle for a log norm), the norm's limits defaulting to
    the first and last level, the extended bands at +-1e250 (under / over of
    the colormap); with `colors`, band i takes colour i (cycled or cut to
    the bands; a colour each for the extended bands too, given enough).
    matplotlib's facecolours are floats: `rounding` 'nearest' gives them as
    Agg paints them (the PNG's bytes), 'floor' truncated, None the floats.
    """
    levels = np.asarray(levels, np.float64)
    if levels.ndim != 1 or len(levels) < 2:
        raise ValueError('Filled contours require at least 2 levels.')
    if np.min(np.diff(levels)) <= 0.0:
        raise ValueError('Contour levels must be increasing')
    if extend not in ('neither', 'min', 'max', 'both'):
        raise ValueError('extend must be neither, min, max or both, not %r' % (extend,))
    emin, emax = extend in ('min', 'both'), extend in ('max', 'both')
    logscale = _is_log(norm)
    lower, upper = (1e-250, 1e250) if logscale else (-1e250, 1e250)
    _levels = list(levels)
    if emin:
        _levels.insert(0, lower)
    if emax:
        _levels.append(upper)
    _levels = np.asarray(_levels)
    if logscale:
        layers = np.sqrt(_levels[:-1]) * np.sqrt(_levels[1:])
    else:
        layers = 0.5 * (_levels[:-1] + _levels[1:])
    if colors is not None:
        if cmap is not None:
            raise ValueError('Either colors or cmap must be None')
        seq = [colors] if is_color_like(colors) else colors
        ncolors = len(levels) - 1
        total = ncolors + int(emin) + int(emax)
        use = len(seq) == total and (emin or emax)
        i0 = 1 if use and emin else 0
        cm = ListedColormap(_resize_sequence(seq[i0:], ncolors),
                            under=seq[0] if use and emin else None,
                            over=seq[-1] if use and emax else None)
        cvalues = np.arange(-1 if emin else 0, ncolors + (1 if emax else 0))
        rgba = cm(cvalues, alpha=alpha)
    else:
        cm = get_cmap(cmap)
        n = make_norm(norm, N=cm.N) if not isinstance(norm, Norm) else norm.copy()
        if vmin is not None:
            n.vmin = _sanitize(vmin)
        if vmax is not None:
            n.vmax = _sanitize(vmax)
        if not n.scaled():
            _autoscale(n, levels)
        elif n.kind == 'twoslope':
            _autoscale(n, levels)
        if extend != 'neither':
            n.clip = False
        if n.kind == 'func':                        # matplotlib's object, on a copy
            import copy
            f = copy.copy(norm)
            if vmin is not None:
                f.vmin = vmin
            if vmax is not None:
                f.vmax = vmax
            f.autoscale_None(levels)
            if extend != 'neither':
                f.clip = False
            rgba = cm(f(layers), alpha=alpha)
        else:
            plan = shade_plan(n, cm, alpha=alpha, rounding=None)
            rows = np.empty((1, len(layers)), np.uint32)
            _run_cpu(plan, layers.reshape(1, -1), rows, row_index=True)
            rgba = plan.args[-2][rows[0]]           # the float table, alpha applied
    if rounding is not None:
        rgba = _to_bytes(rgba, rounding)
    under = rgba[0] if emin else None
    over = rgba[-1] if emax else None
    bands = rgba[int(emin):len(rgba) - int(emax)]
    return bands, under, over


# --------------------------------------------------------------------------
# kernels: the per-value row functions (CPU) -- the CUDA C below is the
# same, line for line. x arrives as float64 (exactly the float32 value when
# `single`); every float32 rounding matplotlib does on float32 data is an
# explicit np.float32() here.

@njit(cache=True, nogil=True)
def _row_scaled(tN, N):
    """The table row of a value already multiplied by N (Colormap.__call__).
    A NaN made by the arithmetic (inf / inf, from limits near the float
    range's ends) is not masked: matplotlib's int cast and clipped take
    give it the first colour."""
    if tN != tN:
        return 0
    if tN == N:
        tN = N - 1.0
    if tN < 0.0:
        return N
    if tN >= N:
        return N + 1
    return int(tN)


@njit(cache=True, nogil=True)
def _row_linear(x, single, vmin, vmax, clip, N):
    if not math.isfinite(x):
        return N + 2
    if vmin == vmax:
        return 0
    if clip:                                    # np.clip(.., float64) -> float64
        d = min(max(x, vmin), vmax)
        return _row_scaled((d - vmin) / (vmax - vmin) * N, N)
    if single:
        t = np.float64(np.float32(x - vmin))
        t = np.float64(np.float32(t / (vmax - vmin)))
        return _row_scaled(np.float64(np.float32(t * N)), N)
    return _row_scaled((x - vmin) / (vmax - vmin) * N, N)


@njit(cache=True, nogil=True)
def _row_log(x, single, vmin, vmax, tvmin, tvmax, clip, N):
    if not math.isfinite(x):
        return N + 2
    if vmin == vmax:
        return 0
    if clip:                                    # np.clip(.., Python float): float32
        if single:
            x = min(max(x, np.float64(np.float32(vmin))), np.float64(np.float32(vmax)))
        else:
            x = min(max(x, vmin), vmax)
    if x <= 0.0:
        return N + 2
    if single:
        t = np.float64(np.float32(math.log10(x)))
        t = np.float64(np.float32(t - tvmin))
        t = np.float64(np.float32(t / (tvmax - tvmin)))
        return _row_scaled(np.float64(np.float32(t * N)), N)
    t = (math.log10(x) - tvmin) / (tvmax - tvmin)
    return _row_scaled(t * N, N)


@njit(cache=True, nogil=True)
def _row_symlog(x, single, vmin, vmax, lth, lsa, logc, logb, tvmin, tvmax, clip, N):
    """SymLogNorm on masked data: numpy.ma makes its Python scalars float64
    arrays, so all is double but numpy's float32 log of float32 data."""
    if not math.isfinite(x):
        return N + 2
    if vmin == vmax:
        return 0
    if clip:                                    # MaskedArray.clip: float32 bounds
        if single:
            x = min(max(x, np.float64(np.float32(vmin))), np.float64(np.float32(vmax)))
        else:
            x = min(max(x, vmin), vmax)
    if abs(x) <= lth:
        o = x * lsa
    else:
        s = lth if x > 0 else -lth
        if single:
            o = s * (logc + np.float64(np.float32(math.log(abs(x)))) / logb)
        else:
            o = s * (logc + math.log(abs(x)) / logb)
    t = (o - tvmin) / (tvmax - tvmin)
    if not math.isfinite(t):
        return N + 2
    return _row_scaled(t * N, N)


@njit(cache=True, nogil=True)
def _row_twoslope(x, vmin, vc, vmax, N):
    if not math.isfinite(x):
        return N + 2
    if x < vmin:                                # np.interp, left=-inf, right=inf
        return N
    if x > vmax:
        return N + 1
    if x < vc:
        if x == vmin:
            t = 0.0
        else:
            t = (0.5 - 0.0) / (vc - vmin) * (x - vmin) + 0.0
    elif x < vmax:
        if x == vc:
            t = 0.5
        else:
            t = (1.0 - 0.5) / (vmax - vc) * (x - vc) + 0.5
    else:
        t = 1.0
    return _row_scaled(t * N, N)


@njit(cache=True, nogil=True)
def _row_nonorm(x, single, N):
    if not math.isfinite(x):
        return N + 2
    if single:
        return _row_scaled(np.float64(np.float32(x * N)), N)
    return _row_scaled(x * N, N)


@njit(cache=True, nogil=True)
def _count_le(levels, x):
    """Number of levels <= x (searchsorted right, np.digitize)."""
    lo, hi = 0, levels.shape[0]
    while lo < hi:
        mid = (lo + hi) >> 1
        if levels[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True, nogil=True)
def _row_boundary(x, single, bounds, rows, clip, vmin, vmax, lo_row, hi_row, N):
    if not math.isfinite(x):
        return N + 2
    if single:                                  # its ends compared in float32
        vmin = np.float64(np.float32(vmin))
        vmax = np.float64(np.float32(vmax))
    if clip:
        x = min(max(x, vmin), vmax)
    r = rows[_count_le(bounds, x)]
    if x < vmin:
        r = lo_row
    if x >= vmax:
        r = hi_row
    return r


@njit(cache=True, nogil=True, parallel=True)
def _k_linear(v, single, vmin, vmax, clip, lut, N, out):
    for i in prange(v.shape[0]):
        for j in range(v.shape[1]):
            out[i, j] = lut[_row_linear(np.float64(v[i, j]), single, vmin, vmax, clip, N)]


@njit(cache=True, nogil=True, parallel=True)
def _k_log(v, single, vmin, vmax, tvmin, tvmax, clip, lut, N, out):
    for i in prange(v.shape[0]):
        for j in range(v.shape[1]):
            out[i, j] = lut[_row_log(np.float64(v[i, j]), single, vmin, vmax, tvmin,
                                     tvmax, clip, N)]


@njit(cache=True, nogil=True, parallel=True)
def _k_symlog(v, single, vmin, vmax, lth, lsa, logc, logb, tvmin, tvmax, clip, lut, N,
              out):
    for i in prange(v.shape[0]):
        for j in range(v.shape[1]):
            out[i, j] = lut[_row_symlog(np.float64(v[i, j]), single, vmin, vmax, lth,
                                        lsa, logc, logb, tvmin, tvmax, clip, N)]


@njit(cache=True, nogil=True, parallel=True)
def _k_twoslope(v, single, vmin, vc, vmax, lut, N, out):
    for i in prange(v.shape[0]):
        for j in range(v.shape[1]):
            out[i, j] = lut[_row_twoslope(np.float64(v[i, j]), vmin, vc, vmax, N)]


@njit(cache=True, nogil=True, parallel=True)
def _k_nonorm(v, single, lut, N, out):
    for i in prange(v.shape[0]):
        for j in range(v.shape[1]):
            out[i, j] = lut[_row_nonorm(np.float64(v[i, j]), single, N)]


@njit(cache=True, nogil=True, parallel=True)
def _k_boundary(v, single, bounds, m, rows, clip, vmin, vmax, lo_row, hi_row, lut, N,
                out):
    for i in prange(v.shape[0]):
        for j in range(v.shape[1]):
            out[i, j] = lut[_row_boundary(np.float64(v[i, j]), single, bounds, rows,
                                          clip, vmin, vmax, lo_row, hi_row, N)]


@njit(cache=True, nogil=True)
def _band_row(x, levels, m, eq0, log, l0, inv_d, uniform):
    """The row of x in a band table: k = number of levels < x (band k - 1
    holds x: z1 < x <= z2), x == levels[0] its own row (eq0), m + 1 for
    NaN / inf (and x <= 0 on a log scale)."""
    if not math.isfinite(x) or (log and x <= 0.0):
        return m + 1
    if x == l0:
        return eq0
    if uniform:
        g = (x - l0) * inv_d + 1.0              # a guess, then exact
        if g < 0.0:
            k = 0
        elif g > m:
            k = m
        else:
            k = int(g)
        while k > 0 and levels[k - 1] >= x:
            k -= 1
        while k < m and levels[k] < x:
            k += 1
        return k
    lo, hi = 0, m
    while lo < hi:
        mid = (lo + hi) >> 1
        if levels[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True, nogil=True, parallel=True)
def _k_bands(v, single, levels, m, tab, eq0, log, l0, inv_d, uniform, out):
    for i in prange(v.shape[0]):
        for j in range(v.shape[1]):
            out[i, j] = tab[_band_row(np.float64(v[i, j]), levels, m, eq0, log, l0,
                                      inv_d, uniform)]


# --------------------------------------------------------------------------
# plans: a norm and a colormap as one kernel's arguments, the same for the
# CPU and the GPU

class ShadePlan(NamedTuple):
    """One shading as a kernel call: `kernel` the CUDA kernel's name (None:
    CPU only), `args` its arguments between (values, n) and out, in order --
    numpy scalars (np.float64 = double, np.int32 = int) and numpy arrays
    (pointers: upload them). The CPU runs the numba kernel of the same name
    with the same arguments."""
    kernel: str
    args: tuple


def _norm_table(cmap, alpha, rounding):
    """The (N + 3, 4) table with `alpha` applied as Colormap.__call__
    applies it: every row's alpha, but bad stays transparent when bad is
    (0, 0, 0, 0) in these bytes."""
    if rounding is None:
        t = np.array(cmap._float_table())
    else:
        t = np.array(cmap.table(rounding))
    if alpha is not None:
        a = np.clip(alpha, 0, 1)
        if np.ndim(a):
            raise ValueError('alpha must be a scalar')
        if rounding == 'floor':
            a = a * 255
        elif rounding == 'nearest':
            a = np.floor(a * 255 + 0.5)
        bad_zero = (t[-1] == 0).all()
        t[:, 3] = a
        if bad_zero:
            t[-1] = 0
    return t


def _boundary_rows(n, Ncm):
    """BoundaryNorm.__call__'s colour index for each count of boundaries
    <= x (0..m), and those of x < vmin and x >= vmax, as table rows."""
    b = n.boundaries
    m = len(b)
    n_regions = m - 1
    offset = 0
    if n.extend in ('min', 'both'):
        n_regions += 1
        offset = 1
    if n.extend in ('max', 'both'):
        n_regions += 1
    iret = np.arange(m + 1) - 1 + offset
    Ncmap = int(n.ncolors)
    if Ncmap > n_regions:
        if n_regions == 1:
            iret[iret == 0] = (Ncmap - 1) // 2
        else:
            iret = (Ncmap - 1) / (n_regions - 1) * iret
    iret = iret.astype(np.int16).astype(np.int64)
    max_col = Ncmap - 1 if n.clip else Ncmap

    def row(i):
        return np.where(i < 0, Ncm, np.where(i >= Ncm, Ncm + 1, i))
    return (row(iret).astype(np.int32), int(row(np.int64(-1))), int(row(np.int64(max_col))))


def _symlog_t(n, x):
    """SymmetricalLogTransform on a float64 array, numpy's own arithmetic."""
    linscale_adj = n.linscale / (1.0 - 1.0 / n.base)
    log_base = np.log(n.base)
    abs_a = np.abs(x)
    inside = abs_a <= n.linthresh
    if np.all(inside):
        return x * linscale_adj
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.sign(x) * n.linthresh * (
            linscale_adj - np.log(n.linthresh) / log_base + np.log(abs_a) / log_base)
    out[inside] = x[inside] * linscale_adj
    return out


def shade_plan(norm, cmap, alpha=None, rounding='floor'):
    """The kernel and arguments shading with `norm` (a resolved Norm, see
    make_norm), `cmap` (a Colormap), `alpha` and `rounding` takes -- the
    same for the numba and the CUDA kernel. rounding None gives the float
    table (for contour_colors)."""
    cm = cmap
    N = cm.N
    tab = _norm_table(cm, alpha, rounding)
    lut = tab if rounding is None else np.ascontiguousarray(tab)
    n = norm
    f64, i32 = np.float64, np.int32
    if n.kind == 'nonorm':
        return ShadePlan('shade_nonorm', (lut, i32(N)))
    if n.kind == 'boundary':
        rows, lo_row, hi_row = _boundary_rows(n, N)
        bounds = np.ascontiguousarray(n.boundaries, dtype=np.float64)
        return ShadePlan('shade_boundary', (bounds, i32(len(bounds)), rows, i32(n.clip),
                                            f64(n.vmin), f64(n.vmax), i32(lo_row),
                                            i32(hi_row), lut, i32(N)))
    if not n.scaled():
        raise ValueError('the norm has no limits: give vmin/vmax or data (make_norm)')
    vmin, vmax = float(n.vmin), float(n.vmax)
    if n.kind == 'func':
        return ShadePlan(None, (n, lut, i32(N)))
    if n.kind == 'twoslope':
        vc = float(n.vcenter)
        if not vmin <= vc <= vmax:
            raise ValueError('vmin, vcenter, vmax must increase monotonically')
        return ShadePlan('shade_twoslope', (f64(vmin), f64(vc), f64(vmax), lut, i32(N)))
    if vmin > vmax:
        raise ValueError('minvalue must be less than or equal to maxvalue')
    if n.kind in ('linear', 'centered'):
        return ShadePlan('shade_linear', (f64(vmin), f64(vmax), i32(n.clip), lut, i32(N)))
    if n.kind == 'log':
        with np.errstate(divide='ignore', invalid='ignore'):
            tv = np.log10(np.array([vmin, vmax]))
        if vmin != vmax and not np.isfinite(tv).all():
            raise ValueError('Invalid vmin or vmax')
        return ShadePlan('shade_log', (f64(vmin), f64(vmax), f64(tv[0]), f64(tv[1]),
                                       i32(n.clip), lut, i32(N)))
    if n.kind == 'symlog':
        lth = float(n.linthresh)
        lsa = n.linscale / (1.0 - 1.0 / n.base)
        log_base = np.log(n.base)
        logc = lsa - np.log(lth) / log_base
        tv = _symlog_t(n, np.array([vmin, vmax]))
        return ShadePlan('shade_symlog', (f64(vmin), f64(vmax), f64(lth), f64(lsa),
                                          f64(logc), f64(log_base), f64(tv[0]),
                                          f64(tv[1]), i32(n.clip), lut, i32(N)))
    raise ValueError('cannot shade with %r' % (n,))


_CPU_KERNELS = {
    'shade_linear': lambda: _k_linear,
    'shade_log': lambda: _k_log,
    'shade_symlog': lambda: _k_symlog,
    'shade_twoslope': lambda: _k_twoslope,
    'shade_nonorm': lambda: _k_nonorm,
    'shade_boundary': lambda: _k_boundary,
    'shade_bands': lambda: _k_bands,
}


def _cpu_args(args):
    """Plan arguments as the numba kernels take them: tables as uint32
    rows, numpy scalars as Python numbers (one compiled signature)."""
    out = []
    for a in args:
        if isinstance(a, np.ndarray):
            if a.dtype == np.uint8 and a.ndim == 2 and a.shape[1] == 4:
                a = np.ascontiguousarray(a).view(np.uint32).reshape(-1)
            out.append(a)
        elif isinstance(a, np.integer):
            out.append(int(a))
        elif isinstance(a, np.floating):
            out.append(float(a))
        else:
            out.append(a)
    return out


@capped
def _run_cpu(plan, v2, out32, row_index=False):
    """Run `plan` on the 2-D values v2 into out32 ((H, W) uint32: RGBA
    bytes, or table rows when row_index)."""
    single = v2.dtype == np.float32
    if plan.kernel is None:                         # a matplotlib norm, called
        n, lut, N = plan.args
        t = n.func(np.ma.masked_invalid(v2))
        rows = np.clip(_RowsOnly(int(N))._rows(t)[0], 0, int(N) + 2)  # take(mode='clip')
        out32[...] = rows if row_index else np.ascontiguousarray(lut).view(np.uint32).reshape(-1)[rows]
        return out32
    args = _cpu_args(plan.args)
    if row_index:                                   # rows, not colours
        k = 2 if plan.kernel == 'shade_bands' else -2
        nrows = len(plan.args[k])
        args[k] = np.arange(nrows, dtype=np.uint32)
    _CPU_KERNELS[plan.kernel]()(v2, single, *args, out32)
    return out32


class _RowsOnly(Colormap):
    """Colormap.__call__'s row choice for N colours (no colours needed)."""

    def __init__(self, N):
        self.N = N


def _prep_values(values):
    """Values as a float array the way process_value types them (float
    kept; small ints / bool float32, others float64), viewed 2-D."""
    v = np.asarray(values)
    if v.dtype.kind != 'f' or v.dtype.itemsize not in (4, 8):
        if v.dtype.kind in 'biu' and v.dtype.itemsize <= 2:
            v = v.astype(np.float32)
        else:
            v = v.astype(np.float64)
    if not v.dtype.isnative:
        v = v.astype(v.dtype.newbyteorder('='))
    if v.ndim == 0:
        v2 = v.reshape(1, 1)
    elif v.ndim == 1:
        v2 = v.reshape(1, -1)
    elif v.ndim == 2:
        v2 = v
    else:
        v2 = v.reshape(-1, v.shape[-1])
    return v, v2


def _prep_out(out, shape, shape2):
    """out (shape + (4,)) uint8, and its pixels as a (shape2) uint32 view
    (or a scratch array to copy back when it cannot be viewed so)."""
    if out is None:
        out = np.empty(shape + (4,), np.uint8)
    elif out.shape != shape + (4,) or out.dtype != np.uint8:
        raise ValueError('out must be uint8 %s, not %s %s' % (shape + (4,), out.dtype, out.shape))
    o2 = out if out.ndim == 3 else out.reshape(shape2 + (4,))
    try:
        if o2.strides[-1] != 1 or o2.strides[-2] != 4 or o2.ctypes.data % 4:
            raise ValueError
        o32 = o2.view(np.uint32)[..., 0]
        return out, o32, None
    except (ValueError, AttributeError):
        tmp = np.empty(o2.shape[:2], np.uint32)
        return out, tmp, o2


def shade(values, cmap=None, norm=None, out=None, alpha=None, *, vmin=None, vmax=None,
          rounding='floor'):
    """RGBA uint8 (values.shape + (4,)) of a value raster, as matplotlib's
    ScalarMappable(norm, cmap).to_rgba(values, alpha, bytes=True) with
    NaN / inf masked (as a plot masks them): bad for those, under / over
    beyond the norm's range, alpha (a scalar) over every colour but a
    transparent bad. `cmap`, `norm`, `vmin`, `vmax` as get_cmap and
    make_norm take them (limits missing come from the values). rounding
    'nearest' gives Agg's bytes for the same float colours (pcolormesh's
    pixels) instead of the image's. Writes into `out` when given."""
    v, v2 = _prep_values(values)
    cm = get_cmap(cmap)
    n = norm if isinstance(norm, Norm) and norm.scaled() and vmin is None and vmax is None \
        else make_norm(norm, vmin, vmax, data=v, N=cm.N)
    plan = shade_plan(n, cm, alpha=alpha, rounding=rounding)
    out, o32, back = _prep_out(out, v.shape, v2.shape)
    _run_cpu(plan, v2, o32)
    if back is not None:
        back[...] = o32.view(np.uint8).reshape(back.shape)
    return out


def bands_plan(levels, band_lut, under=None, over=None, *, log=False):
    """The 'shade_bands' plan of shade_bands (see there), for the CPU and
    the GPU alike."""
    lev = np.ascontiguousarray(levels, dtype=np.float64)
    m = len(lev)
    band_lut = np.asarray(band_lut, np.uint8)
    if m < 2 or band_lut.shape != (m - 1, 4):
        raise ValueError('band_lut must be (len(levels) - 1, 4) uint8, not %s for %d levels'
                         % (band_lut.shape, m))
    if np.any(np.diff(lev) <= 0):
        raise ValueError('levels must be increasing')
    tab = np.zeros((m + 2, 4), np.uint8)
    tab[1:m] = band_lut
    if under is not None:
        tab[0] = np.asarray(under, np.uint8)
    if over is not None:
        tab[m] = np.asarray(over, np.uint8)
    eq0 = 0 if under is not None else 1
    d = (lev[-1] - lev[0]) / (m - 1)
    dev = np.abs(lev - (lev[0] + d * np.arange(m))).max()
    uniform = int(dev < 0.25 * d)
    return ShadePlan('shade_bands', (lev, np.int32(m), tab, np.int32(eq0), np.int32(bool(log)),
                                     np.float64(lev[0]), np.float64(1.0 / d),
                                     np.int32(uniform)))


def shade_bands(values, levels, band_lut, under=None, over=None, out=None, *, log=False):
    """RGBA uint8 of a value raster as contourf fills it: band i (colour
    band_lut[i]) holds levels[i] < z <= levels[i + 1], the lowest band
    levels[0] too; below levels[0] `under` (z <= levels[0] when given),
    above levels[-1] `over`, each transparent (0, 0, 0, 0) when None, as
    are NaN and inf (and z <= 0 with log=True, as a log contourf masks
    them). contour_colors gives band_lut, under and over. Writes into
    `out` when given."""
    v, v2 = _prep_values(values)
    plan = bands_plan(levels, band_lut, under, over, log=log)
    out, o32, back = _prep_out(out, v.shape, v2.shape)
    _run_cpu(plan, v2, o32)
    if back is not None:
        back[...] = o32.view(np.uint8).reshape(back.shape)
    return out


# --------------------------------------------------------------------------
# CUDA

CUDA = r'''
// fastplot._color: values -> RGBA, bytes identical to the numba kernels.
// Every kernel: v = float32 values (n of them), out = n RGBA pixels
// (4 n bytes, 4-byte aligned); lut = (N + 3) RGBA rows, 4-byte aligned:
// N colours, then under (row N), over (N + 1), bad (N + 2). Grid-stride
// loops: any launch shape covers n.
//
// All arithmetic is double, float32 roundings explicit (fpc_f32): cupy
// compiles with -ftz=true, which flushes float32 subnormals in float
// operations and conversions, and numpy / numba keep them. So a value is
// loaded through its bits (fpc_load) and rounded to float32 by hand in the
// subnormal range; double operations are not flushed.

__device__ __forceinline__ double fpc_load(const float* v, int i)
{
    unsigned int b = ((const unsigned int*)v)[i];
    if ((b & 0x7f800000u) == 0u) {                  // zero or subnormal
        double m = ldexp((double)(b & 0x007fffffu), -149);
        return (b >> 31) ? -m : m;
    }
    return (double)__uint_as_float(b);
}

__device__ __forceinline__ double fpc_f32(double x)
{
    if (fabs(x) < 1.1754943508222875e-38)          // float32's subnormal range
        return ldexp(rint(ldexp(x, 149)), -149);   // (round half even)
    return (double)(float)x;
}

__device__ __forceinline__ int fpc_row_scaled(double tN, int N)
{
    if (tN != tN) return 0;
    if (tN == (double)N) tN = N - 1.0;
    if (tN < 0.0) return N;
    if (tN >= (double)N) return N + 1;
    return (int)tN;
}

__device__ __forceinline__ void fpc_put(unsigned char* out, int i,
                                        const unsigned char* lut, int row)
{
    ((unsigned int*)out)[i] = ((const unsigned int*)lut)[row];
}

#define FPC_LOOP for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
                      i += blockDim.x * gridDim.x)

extern "C" __global__ void shade_linear(const float* v, int n, double vmin,
    double vmax, int clip, const unsigned char* lut, int N, unsigned char* out)
{
    FPC_LOOP {
        double x = fpc_load(v, i);
        int r;
        if (!isfinite(x)) r = N + 2;
        else if (vmin == vmax) r = 0;
        else if (clip) {
            double d = fmin(fmax(x, vmin), vmax);
            r = fpc_row_scaled((d - vmin) / (vmax - vmin) * N, N);
        } else {
            double t = fpc_f32(x - vmin);
            t = fpc_f32(t / (vmax - vmin));
            r = fpc_row_scaled(fpc_f32(t * N), N);
        }
        fpc_put(out, i, lut, r);
    }
}

extern "C" __global__ void shade_log(const float* v, int n, double vmin,
    double vmax, double tvmin, double tvmax, int clip, const unsigned char* lut,
    int N, unsigned char* out)
{
    FPC_LOOP {
        double x = fpc_load(v, i);
        int r;
        if (!isfinite(x)) r = N + 2;
        else if (vmin == vmax) r = 0;
        else {
            if (clip) x = fmin(fmax(x, fpc_f32(vmin)), fpc_f32(vmax));
            if (x <= 0.0) r = N + 2;
            else {
                double t = fpc_f32(log10(x));
                t = fpc_f32(t - tvmin);
                t = fpc_f32(t / (tvmax - tvmin));
                r = fpc_row_scaled(fpc_f32(t * N), N);
            }
        }
        fpc_put(out, i, lut, r);
    }
}

extern "C" __global__ void shade_symlog(const float* v, int n, double vmin,
    double vmax, double lth, double lsa, double logc, double logb, double tvmin,
    double tvmax, int clip, const unsigned char* lut, int N, unsigned char* out)
{
    FPC_LOOP {
        double x = fpc_load(v, i);
        int r;
        if (!isfinite(x)) r = N + 2;
        else if (vmin == vmax) r = 0;
        else {
            if (clip) x = fmin(fmax(x, fpc_f32(vmin)), fpc_f32(vmax));
            double o;
            if (fabs(x) <= lth) o = x * lsa;
            else {
                double s = x > 0 ? lth : -lth;
                o = s * (logc + fpc_f32(log(fabs(x))) / logb);
            }
            double t = (o - tvmin) / (tvmax - tvmin);
            r = isfinite(t) ? fpc_row_scaled(t * N, N) : N + 2;
        }
        fpc_put(out, i, lut, r);
    }
}

extern "C" __global__ void shade_twoslope(const float* v, int n, double vmin,
    double vc, double vmax, const unsigned char* lut, int N, unsigned char* out)
{
    FPC_LOOP {
        double x = fpc_load(v, i);
        int r;
        if (!isfinite(x)) r = N + 2;
        else if (x < vmin) r = N;
        else if (x > vmax) r = N + 1;
        else {
            double t;
            if (x < vc) t = (x == vmin) ? 0.0 : (0.5 - 0.0) / (vc - vmin) * (x - vmin) + 0.0;
            else if (x < vmax) t = (x == vc) ? 0.5 : (1.0 - 0.5) / (vmax - vc) * (x - vc) + 0.5;
            else t = 1.0;
            r = fpc_row_scaled(t * N, N);
        }
        fpc_put(out, i, lut, r);
    }
}

extern "C" __global__ void shade_nonorm(const float* v, int n,
    const unsigned char* lut, int N, unsigned char* out)
{
    FPC_LOOP {
        double x = fpc_load(v, i);
        int r = isfinite(x) ? fpc_row_scaled(fpc_f32(x * N), N) : N + 2;
        fpc_put(out, i, lut, r);
    }
}

extern "C" __global__ void shade_boundary(const float* v, int n,
    const double* bounds, int m, const int* rows, int clip, double vmin,
    double vmax, int lo_row, int hi_row, const unsigned char* lut, int N,
    unsigned char* out)
{
    double lo = fpc_f32(vmin), hi = fpc_f32(vmax);
    FPC_LOOP {
        double x = fpc_load(v, i);
        int r;
        if (!isfinite(x)) r = N + 2;
        else {
            if (clip) x = fmin(fmax(x, lo), hi);
            int a = 0, b = m;
            while (a < b) {
                int mid = (a + b) >> 1;
                if (bounds[mid] <= x) a = mid + 1; else b = mid;
            }
            r = rows[a];
            if (x < lo) r = lo_row;
            if (x >= hi) r = hi_row;
        }
        fpc_put(out, i, lut, r);
    }
}

extern "C" __global__ void shade_bands(const float* v, int n,
    const double* levels, int m, const unsigned char* tab, int eq0, int logscale,
    double l0, double inv_d, int uniform, unsigned char* out)
{
    FPC_LOOP {
        double x = fpc_load(v, i);
        int k;
        if (!isfinite(x) || (logscale && x <= 0.0)) k = m + 1;
        else if (x == l0) k = eq0;
        else if (uniform) {
            double g = (x - l0) * inv_d + 1.0;
            k = g < 0.0 ? 0 : (g > (double)m ? m : (int)g);
            while (k > 0 && levels[k - 1] >= x) k--;
            while (k < m && levels[k] < x) k++;
        } else {
            int a = 0, b = m;
            while (a < b) {
                int mid = (a + b) >> 1;
                if (levels[mid] < x) a = mid + 1; else b = mid;
            }
            k = a;
        }
        fpc_put(out, i, tab, k);
    }
}
'''
CUDA_KERNELS = ('shade_linear', 'shade_log', 'shade_symlog', 'shade_twoslope',
                'shade_nonorm', 'shade_boundary', 'shade_bands')
