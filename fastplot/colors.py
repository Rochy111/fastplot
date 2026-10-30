"""Colours, colormaps and norms, named as matplotlib.colors':

    import fastplot.colors as mcolors
    norm = mcolors.BoundaryNorm(levels, ncolors=256, extend='both')
    cmap = mcolors.ListedColormap(['#ffffff', '#04e9e7', '#019ff4'])
    ax.pcolormesh(x, y, z, cmap=cmap, norm=norm)

The norms are fastplot's (_color.Norm) under matplotlib's class names,
so the renderer reads them directly; calling one maps values to 0..1 as
matplotlib's does (for code that uses norm(v) itself). matplotlib's own
norm and colormap objects are accepted everywhere too.
"""
import numpy as np

from ._color import (Colormap, LinearSegmentedColormap, ListedColormap, Norm,  # noqa: F401
                     cmap_names, is_color_like, to_rgba, to_rgba_array)


class _NormBase(Norm):
    def __call__(self, value, clip=None):
        v = np.ma.asarray(value, dtype=float)
        if self.vmin is None or self.vmax is None:
            self.autoscale_None(v)
        vmin, vmax = float(self.vmin), float(self.vmax)
        if self.kind == 'log':
            with np.errstate(invalid='ignore', divide='ignore'):
                out = (np.log10(v) - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))
        elif self.kind == 'twoslope':
            vc = float(self.vcenter)
            out = np.ma.where(v < vc, 0.5 * (v - vmin) / (vc - vmin),
                              0.5 + 0.5 * (v - vc) / (vmax - vc))
        else:
            out = (v - vmin) / (vmax - vmin) if vmax != vmin else v * 0.0
        if clip if clip is not None else self.clip:
            out = np.ma.clip(out, 0.0, 1.0)
        return out

    def inverse(self, value):
        """matplotlib's Normalize.inverse: 0..1 back to data values."""
        v = np.asarray(value, float)
        vmin, vmax = float(self.vmin), float(self.vmax)
        if self.kind == 'log':
            return vmin * (vmax / vmin) ** v
        if self.kind == 'twoslope':
            vc = float(self.vcenter)
            return np.where(v < 0.5, vmin + 2 * v * (vc - vmin), vc + (2 * v - 1) * (vmax - vc))
        return vmin + v * (vmax - vmin)

    def autoscale(self, A):
        a = np.ma.masked_invalid(np.asarray(A, float))
        self.vmin, self.vmax = float(a.min()), float(a.max())

    def autoscale_None(self, A):
        a = np.ma.masked_invalid(np.asarray(A, float))
        if self.vmin is None:
            self.vmin = float(a.min())
        if self.vmax is None:
            self.vmax = float(a.max())

    def scaled(self):
        return self.vmin is not None and self.vmax is not None


class Normalize(_NormBase):
    def __init__(self, vmin=None, vmax=None, clip=False):
        super().__init__('linear', vmin, vmax, clip=clip)


class LogNorm(_NormBase):
    def __init__(self, vmin=None, vmax=None, clip=False):
        super().__init__('log', vmin, vmax, clip=clip)


class SymLogNorm(_NormBase):
    def __init__(self, linthresh, linscale=1.0, vmin=None, vmax=None, clip=False, *, base=10):
        super().__init__('symlog', vmin, vmax, clip=clip, linthresh=linthresh,
                         linscale=linscale, base=base)


class TwoSlopeNorm(_NormBase):
    def __init__(self, vcenter, vmin=None, vmax=None):
        super().__init__('twoslope', vmin, vmax, vcenter=vcenter)


class CenteredNorm(_NormBase):
    def __init__(self, vcenter=0, halfrange=None, clip=False):
        super().__init__('centered', vcenter=vcenter, halfrange=halfrange, clip=clip)


class BoundaryNorm(_NormBase):
    def __init__(self, boundaries, ncolors, clip=False, *, extend='neither'):
        super().__init__('boundary', boundaries=boundaries, ncolors=ncolors, clip=clip,
                         extend=extend)

    def __call__(self, value, clip=None):
        """matplotlib's: each value's colour index, an integer -- its bin among
        the boundaries (after an extension's), spread over ncolors when there
        are more colours than bins; -1 under vmin, ncolors at or over vmax
        (ncolors - 1 with clip). A colormap takes the integers as indices."""
        clip = self.clip if clip is None else clip
        scalar = np.ndim(value) == 0
        v = np.ma.asarray(value, dtype=float)
        mask = np.ma.getmaskarray(v)
        x = np.atleast_1d(v.filled(self.vmax + 1))
        if clip:
            x = np.clip(x, self.vmin, self.vmax)
            top = self.ncolors - 1
        else:
            top = self.ncolors
        offset = 1 if self.extend in ('min', 'both') else 0
        regions = len(self.boundaries) - 1 + offset + (self.extend in ('max', 'both'))
        idx = np.digitize(x, self.boundaries) - 1 + offset
        if self.ncolors > regions:
            if regions == 1:
                idx[idx == 0] = (self.ncolors - 1) // 2
            else:
                idx = (self.ncolors - 1) / (regions - 1) * idx
        idx = idx.astype(np.int16)
        idx[x < self.vmin] = -1
        idx[x >= self.vmax] = top
        out = np.ma.array(idx, mask=mask)
        return int(out[0]) if scalar else out


class NoNorm(_NormBase):
    def __init__(self, vmin=None, vmax=None, clip=False):
        super().__init__('nonorm', vmin, vmax, clip=clip)


class _Mapping:
    """What a 'func' Norm calls: matplotlib's norm protocol (vmin, vmax,
    clip, autoscale_None, __call__) for a mapping fastplot has no kernel for."""

    def __init__(self, fn, vmin, vmax, clip):
        self.fn, self.vmin, self.vmax, self.clip = fn, vmin, vmax, clip

    def autoscale_None(self, A):
        a = np.ma.masked_invalid(np.asarray(A, float))
        if self.vmin is None and a.count():
            self.vmin = float(a.min())
        if self.vmax is None and a.count():
            self.vmax = float(a.max())

    def __call__(self, value, clip=None):
        v = np.ma.masked_invalid(np.ma.asarray(value, dtype=float))
        self.autoscale_None(v)
        vmin, vmax = float(self.vmin), float(self.vmax)
        if clip if clip is not None else self.clip:
            v = np.ma.clip(v, vmin, vmax)
        return self.fn(v, vmin, vmax)


def _power(gamma):
    def fn(v, vmin, vmax):
        if vmin == vmax:
            return v * 0.0
        r = (v - vmin) / (vmax - vmin)
        return np.ma.where(r > 0, np.ma.power(np.ma.maximum(r, 0), gamma), r)
    return fn


class PowerNorm(_NormBase):
    """matplotlib's PowerNorm: (v - vmin) / (vmax - vmin), to gamma where > 0."""

    def __init__(self, gamma, vmin=None, vmax=None, clip=False):
        self.gamma = gamma
        super().__init__('func', vmin, vmax, clip=clip,
                         func=_Mapping(_power(gamma), vmin, vmax, clip))

    def __call__(self, value, clip=None):
        self.func.vmin, self.func.vmax = self.vmin, self.vmax
        out = self.func(value, clip)
        self.vmin, self.vmax = self.func.vmin, self.func.vmax
        return out

    def inverse(self, value):
        v = np.asarray(value, float)
        return np.power(np.maximum(v, 0), 1.0 / self.gamma) * (self.vmax - self.vmin) + self.vmin


class FuncNorm(_NormBase):
    """matplotlib's FuncNorm: functions (forward, inverse) between vmin and vmax."""

    def __init__(self, functions, vmin=None, vmax=None, clip=False):
        fwd, inv = functions
        self._fwd, self._inv = fwd, inv

        def fn(v, lo, hi):
            a, b = fwd(np.float64(lo)), fwd(np.float64(hi))
            return (fwd(v) - a) / (b - a) if b != a else v * 0.0
        super().__init__('func', vmin, vmax, clip=clip, func=_Mapping(fn, vmin, vmax, clip))

    __call__ = PowerNorm.__call__

    def inverse(self, value):
        a, b = self._fwd(np.float64(self.vmin)), self._fwd(np.float64(self.vmax))
        return self._inv(np.asarray(value, float) * (b - a) + a)


def to_rgb(c):
    return to_rgba(c)[:3]


def to_hex(c, keep_alpha=False):
    r = to_rgba(c)
    vals = r if keep_alpha else r[:3]
    return '#' + ''.join('%02x' % int(round(v * 255)) for v in vals)


def from_levels_and_colors(levels, colors, extend='neither'):
    """matplotlib's: a ListedColormap and BoundaryNorm for discrete levels."""
    n_extra = {'neither': 0, 'min': 1, 'max': 1, 'both': 2}[extend]
    ncolors = len(levels) - 1
    if len(colors) != ncolors + n_extra:
        raise ValueError('with extend=%r, %d levels need %d colours' % (extend, len(levels),
                                                                        ncolors + n_extra))
    lo = 1 if extend in ('min', 'both') else 0
    cmap = ListedColormap(colors[lo:lo + ncolors])
    if extend in ('min', 'both'):
        cmap = cmap.with_extremes(under=colors[0])
    if extend in ('max', 'both'):
        cmap = cmap.with_extremes(over=colors[-1])
    return cmap, BoundaryNorm(levels, ncolors=ncolors, extend=extend)


__all__ = ['Colormap', 'ListedColormap', 'LinearSegmentedColormap', 'Normalize', 'LogNorm',
           'SymLogNorm', 'TwoSlopeNorm', 'CenteredNorm', 'BoundaryNorm', 'NoNorm', 'PowerNorm',
           'FuncNorm', 'to_rgba',
           'to_rgba_array', 'to_rgb', 'to_hex', 'is_color_like', 'cmap_names',
           'from_levels_and_colors']
