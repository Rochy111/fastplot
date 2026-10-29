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
from . import transforms  # noqa: F401
from ._rc import rc, rc_context, rcdefaults, rcParams                # noqa: F401
from .cm import colormaps                                            # noqa: F401
from ._device import get_device, gpu_info, set_device               # noqa: F401
from ._figure import Figure, GridSpec, close, figure, subplots      # noqa: F401
from ._color import get_cmap                                        # noqa: F401

__version__ = '0.1.0'
__all__ = ['figure', 'subplots', 'close', 'Figure', 'GridSpec', 'crs', 'feature', 'cm',
           'colors', 'colormaps', 'dates', 'gridspec', 'patches', 'style', 'ticker', 'transforms', 'rcParams', 'rc',
           'rc_context', 'rcdefaults', 'set_device', 'get_device', 'gpu_info', 'get_cmap',
           'warm']


def warm():
    """Compile (or load from numba's cache) every CPU kernel, by drawing a
    small figure of each kind once -- for a job's set-up, so the first real
    frame is not the one that pays."""
    import numpy as np
    lon = np.linspace(0, 359, 36)
    lat = np.linspace(-90, 90, 19)
    z = np.cos(np.radians(lat))[:, None] * np.sin(np.radians(lon))[None, :]
    fig = figure(figsize=(3, 2), dpi=50)
    ax = fig.add_subplot(projection=crs.PlateCarree())
    cf = ax.contourf(lon, lat, z, levels=8, transform=crs.PlateCarree())
    ax.pcolormesh(lon, lat, z, transform=crs.PlateCarree(), alpha=0.5)
    cs = ax.contour(lon, lat, z, levels=5, colors='k', transform=crs.PlateCarree())
    ax.clabel(cs)
    ax.barbs(lon[::6], lat[::6], 20 * z[::6, ::6], 20 * z[::6, ::6], transform=crs.PlateCarree())
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.set_title('warm')
    fig.colorbar(cf, ax=ax)
    fig.draw(device='cpu')
    import io
    fig.savefig(io.BytesIO(), device='cpu')
