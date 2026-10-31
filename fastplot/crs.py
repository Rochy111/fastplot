"""Projections, named as cartopy.crs's, on cartopy's default Earth:

    import fastplot.crs as ccrs
    ax = fig.add_subplot(projection=ccrs.LambertConformal(central_longitude=-97))
    ax.pcolormesh(lon, lat, z, transform=ccrs.PlateCarree())
"""
from ._proj import (AlbersEqualArea, Geodetic, Globe, LambertAzimuthalEqualArea,
                    LambertConformal, Mercator, Miller, NorthPolarStereo, Orthographic,
                    PlateCarree, Robinson, RotatedPole, SouthPolarStereo, Stereographic)

__all__ = ['AlbersEqualArea', 'Geodetic', 'Globe', 'LambertAzimuthalEqualArea',
           'LambertConformal', 'Mercator', 'Miller', 'NorthPolarStereo', 'Orthographic',
           'PlateCarree', 'Robinson', 'RotatedPole', 'SouthPolarStereo', 'Stereographic']
