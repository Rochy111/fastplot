"""fastplot: maps of gridded fields -- what matplotlib + cartopy draw for
weather models -- drawn straight into pixels by numba (every CPU core) or
CUDA (cupy), 10-100x faster, with their API.

    import fastplot as fp
    import fastplot.crs as ccrs

    fig = fp.figure(figsize=(12, 6.5), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    cf = ax.contourf(lon, lat, t2m, levels=range(-40, 46, 2), cmap='turbo',
                     extend='both', transform=ccrs.PlateCarree())
    ax.coastlines(); ax.add_feature(fp.feature.BORDERS, linewidth=0.3)
    ax.gridlines(draw_labels=True)
    fig.colorbar(cf, ax=ax, orientation='horizontal', label='2 m temperature (C)')
    fig.savefig('t2m.png')

CPU or GPU is picked per figure (fastplot.set_device); fields read by
fastgrib with gpu=True stay on the card. See README.md.
"""
from . import cm, colors, crs, dates, feature, gridspec, patches, style, ticker  # noqa: F401
from . import patheffects, tri  # noqa: F401
from . import transforms  # noqa: F401
from ._rc import rc, rc_context, rcdefaults, rcParams                # noqa: F401
from .cm import colormaps                                            # noqa: F401
from ._device import get_device, gpu_info, set_device               # noqa: F401
from ._figure import Figure, GridSpec, close, figure, subplots      # noqa: F401
from ._color import get_cmap                                        # noqa: F401
from ._tune import hardware, tune                                   # noqa: F401
from ._warm import warm                                             # noqa: F401
from ._batch import frames                                          # noqa: F401

try:
    from ._version import __version__                               # written when installed
except ImportError:                                                 # a checkout never installed
    __version__ = '0+unknown'
__all__ = ['figure', 'subplots', 'close', 'Figure', 'GridSpec', 'crs', 'feature', 'cm',
           'colors', 'colormaps', 'dates', 'gridspec', 'patches', 'style', 'ticker', 'transforms',
           'patheffects', 'tri', 'rcParams', 'rc', 'rc_context', 'rcdefaults', 'set_device', 'get_device',
           'gpu_info', 'get_cmap', 'warm', 'tune', 'hardware', 'frames']

from . import _jit                                                  # noqa: E402
_jit.preload()                  # numba made ready while the program reads its data

