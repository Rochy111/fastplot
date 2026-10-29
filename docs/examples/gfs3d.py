"""Shared by the diagnostic examples: GFS fields on pressure levels, read with
fastgrib, and the finite differences on a sphere the diagnostics are built
from.

The fields come from the GFS 0.5° run of 00 UTC 27 October 2026, six-hour
forecast, in NOAA's archive on AWS. fastgrib reads the file's index and
fetches only the records asked for. Set GFS_FILE_0P50 to read a local copy.
"""
import os

import numpy as np

import fastgrib

URL = 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20261027/00/atmos/gfs.t00z.pgrb2.0p50.f006'
LEVELS = (1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350,
          300, 250, 200, 150, 100, 70, 50)                     # hPa, bottom to top

R_EARTH = 6.371e6          # m
G = 9.80665                # m s-2
OMEGA = 7.2921e-5          # s-1
KAPPA = 0.2857             # R_d / c_p


def open_gfs():
    return fastgrib.open(os.environ.get('GFS_FILE_0P50', URL))


def levels(grib, name, levels=LEVELS):
    """One variable on the given pressure levels, as (level, lat, lon), bottom
    to top. All records are fetched in one call."""
    recs = grib.select(name=name, level_kind='isobaricInhPa',
                       level_value=lambda v: v in set(levels))
    recs = sorted(recs, key=lambda r: -r.level_value)
    if [r.level_value for r in recs] != list(levels):
        raise ValueError('%s: levels %s' % (name, [r.level_value for r in recs]))
    data = grib.read(recs).astype(np.float64)
    return data[None] if data.ndim == 2 else data          # one level: still (1, lat, lon)


def axes(grib):
    """The grid's 1-D latitudes (south to north) and longitudes (east from 0)."""
    lat, lon = grib.latlon()
    return lat[:, 0].astype(np.float64), lon[0].astype(np.float64)


def ddx(f, lat, lon):
    """d/dx on the sphere along the last axis, periodic in longitude."""
    dlam = np.radians(lon[1] - lon[0])
    dfdl = (np.roll(f, -1, axis=-1) - np.roll(f, 1, axis=-1)) / (2 * dlam)
    coslat = np.maximum(np.cos(np.radians(lat)), 1e-6)[:, None]
    return dfdl / (R_EARTH * coslat)


def ddy(f, lat):
    """d/dy on the sphere along the second-last axis."""
    return np.gradient(f, np.radians(lat), axis=-2) / R_EARTH


def ddp(f, p_hpa):
    """d/dp along the first axis, p in Pa (levels need not be evenly spaced)."""
    return np.gradient(f, np.asarray(p_hpa, np.float64) * 100.0, axis=0)


def coriolis(lat):
    return 2 * OMEGA * np.sin(np.radians(lat))[:, None]


def vorticity(u, v, lat, lon):
    """Relative vorticity, with the curvature term of a spherical grid."""
    tan = np.tan(np.radians(lat))[:, None]
    return ddx(v, lat, lon) - ddy(u, lat) + u * tan / R_EARTH


def theta(t, p_hpa):
    """Potential temperature of temperature t (K) at pressure p (hPa, along axis 0
    or a scalar)."""
    p = np.asarray(p_hpa, np.float64)
    if p.ndim:
        p = p[:, None, None]
    return t * (1000.0 / p) ** KAPPA


def smooth(f, passes=1):
    """A 9-point smoother on the last two axes, periodic in longitude."""
    for _ in range(passes):
        g = np.concatenate([f[..., :1, :], f, f[..., -1:, :]], axis=-2)
        g = np.concatenate([g[..., -1:], g, g[..., :1]], axis=-1)
        sides = g[..., 2:, 1:-1] + g[..., :-2, 1:-1] + g[..., 1:-1, 2:] + g[..., 1:-1, :-2]
        corners = g[..., 2:, 2:] + g[..., 2:, :-2] + g[..., :-2, 2:] + g[..., :-2, :-2]
        f = (4 * g[..., 1:-1, 1:-1] + 2 * sides + corners) / 16
    return f


def bilinear(f, lat, lon, plat, plon):
    """f (..., lat, lon) at points (plat, plon) by bilinear interpolation."""
    j = np.interp(plat, lat, np.arange(lat.size))
    i = ((np.asarray(plon) - lon[0]) % 360.0) / (lon[1] - lon[0])
    j0 = np.clip(np.floor(j).astype(int), 0, lat.size - 2)
    i0 = np.floor(i).astype(int) % lon.size
    i1 = (i0 + 1) % lon.size
    wj, wi = j - j0, i - np.floor(i)
    return ((1 - wj) * ((1 - wi) * f[..., j0, i0] + wi * f[..., j0, i1])
            + wj * ((1 - wi) * f[..., j0 + 1, i0] + wi * f[..., j0 + 1, i1]))
