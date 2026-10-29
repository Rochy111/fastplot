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


class NoNorm(_NormBase):
    def __init__(self, vmin=None, vmax=None, clip=False):
        super().__init__('nonorm', vmin, vmax, clip=clip)


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
           'SymLogNorm', 'TwoSlopeNorm', 'CenteredNorm', 'BoundaryNorm', 'NoNorm', 'to_rgba',
           'to_rgba_array', 'to_rgb', 'to_hex', 'is_color_like', 'cmap_names',
           'from_levels_and_colors']
