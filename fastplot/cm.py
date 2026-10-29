"""Colormaps by name (fastplot.cm.viridis), get_cmap, the colormap
registry, and ScalarMappable -- for a colorbar of a norm and colormap with
no artist behind it, as matplotlib's:

    fig.colorbar(fastplot.cm.ScalarMappable(norm=norm, cmap='turbo'), ax=ax)
"""
from ._color import Colormap, cmap_names, get_cmap, make_norm


class ScalarMappable:
    """A norm and a colormap (and optionally data to scale the norm by)."""

    axes = None
    extend = None

    def __init__(self, norm=None, cmap=None):
        self.cmap = get_cmap(cmap)
        self._norm_in = norm
        self._A = None
        self._vmin = self._vmax = None

    def set_array(self, A):
        self._A = A

    def get_array(self):
        return self._A

    def set_clim(self, vmin=None, vmax=None):
        if isinstance(vmin, (tuple, list)):
            vmin, vmax = vmin
        self._vmin, self._vmax = vmin, vmax

    def set_cmap(self, cmap):
        self.cmap = get_cmap(cmap)

    def set_norm(self, norm):
        self._norm_in = norm

    def norm_spec(self):
        return make_norm(self._norm_in, vmin=self._vmin, vmax=self._vmax, data=self._A,
                         N=self.cmap.N)

    @property
    def norm(self):
        return self.norm_spec()


class ColormapRegistry:
    """matplotlib.colormaps: colormaps by name."""

    def __getitem__(self, name):
        return get_cmap(name)

    def __iter__(self):
        return iter(cmap_names())

    def __len__(self):
        return len(cmap_names())

    def __contains__(self, name):
        return name in cmap_names()

    def get_cmap(self, cmap):
        return get_cmap(cmap)

    def register(self, cmap, *, name=None, force=False):
        from . import _color
        _color._CMCACHE[name or cmap.name] = cmap


colormaps = ColormapRegistry()


def __getattr__(name):                                  # cm.viridis, cm.turbo_r, ...
    if name.startswith('__'):
        raise AttributeError(name)
    try:
        return get_cmap(name)
    except (KeyError, ValueError) as e:
        raise AttributeError(name) from e


__all__ = ['Colormap', 'ScalarMappable', 'get_cmap', 'colormaps']
