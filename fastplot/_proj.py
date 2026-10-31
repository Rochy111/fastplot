"""Map projections, on cartopy's Earth, for the per-pixel and per-vertex
loops of the renderer: numba scalar functions, the same math as CUDA C, and
the classes users build maps with (cartopy's names and keywords).

    p = LambertConformal(central_longitude=-97.5, standard_parallels=(38.5,))
    x, y = p.forward(lon, lat)                # any shape, NaN off the map
    lon, lat = p.inverse(x, y)                # NaN outside p.boundary
    x0, x1, y0, y1 = extent_xy(p, (-120, -73, 23, 50))   # ax.set_extent's
    x2, y2 = transform_points(RotatedPole(...), p, rlon, rlat)

    @njit                                      # in another module's kernel
    def k(kind, prm, ...):
        x, y = fwd(kind, prm, lon, lat)        # one point; no allocation
        lon, lat = inv(kind, prm, x, y)

A projection is two numbers to a kernel: `kind`, a small int, and
`params`, 24 float64 holding everything its formulas need, computed once
here (LCC's cone constant n, R c and R rho0, a stereographic's 2 k0 R ...)
so a point costs only its own trigonometry. `fwd` and `inv` branch on
kind; within a loop the branch is always taken the same way, which the
CPU predicts and which costs a GPU warp nothing. They are plain
@njit(cache=True, nogil=True, error_model='numpy') functions (no
exceptions, no division checks), not IR-inlined: a caller's LLVM module
links them in, and IR inlining a function of eleven projections into every
kernel would cost each its compile time for nothing measurable (~31 ns a
point on one core, LCC, the call included). `CUDA` is the same code in C,
`fp_fwd` and `fp_inv`, operation for operation, so the two differ only by
the last bits of a libm function.

The Earth is cartopy's: WGS84 where cartopy's default is an ellipsoid
(Mercator, the conics, Stereographic, LAEA), a sphere of 6378137 m where
it is a sphere (Orthographic, Robinson, Miller); `globe=` or `radius=`
choose another. On a sphere the formulas are PROJ's, mostly in PROJ's
order of operations. On an ellipsoid each is the same spherical formula
with phi replaced by an auxiliary latitude -- the conformal one for
conformal projections, the authalic one for equal-area ones -- which is
what PROJ's ellipsoidal formulas are too, and only constants differ. The
latitude is converted by a short series in sin 2 phi (_aux), both ways, so
the inverse needs no iteration: a pixel costs ~15 ns more. `pyproj` with
`.proj4` agrees to ~1e-6 m forward and ~1e-10 deg inverse (less only
within a degree of an azimuthal antipode, where 1 + cos c cancels).

Coordinates: lon/lat in degrees; x/y in metres, except PlateCarree,
Geodetic and RotatedPole whose x/y are (rotated) degrees, as in cartopy.
Forward x of a cylindrical projection is lon - central_longitude wrapped
into [-180, 180) (times the scale); `cylindrical` and `period` let line
drawing split segments that jump across that seam. The others cut along
a meridian too, central_longitude + 180, but not at a fixed x (Robinson's
oval edge, a cone's two straight edges): `seam` marks them, and lines are
cut there in lon/lat before projecting. Forward gives NaN where
a point has no image: the far side of an Orthographic, the antipode of a
LAEA or Stereographic, LCC beyond its cutoff latitude, |lat| = 90 on
Mercator, |lat| > 90 anywhere. Points outside the drawn domain but with an
image (Mercator beyond its latitude limits, a Stereographic far out) keep
it, as in cartopy and PROJ: lines then run off the map and are clipped.
Inverse gives NaN outside the valid domain (the region `boundary` rings,
Mercator's latitude limits included) and longitudes in
[central_longitude - 180, central_longitude + 180). The lon/lat-like
coordinate systems (PlateCarree, Geodetic, RotatedPole) wrap any x instead,
so data on 0..360 longitudes passes through them as it is.

Arrays of points run in a numba kernel, on every core past PAR_MIN points.
Nothing here needs numba to build a projection: limits and boundaries are
closed-form numpy.
"""
import math
import warnings
from functools import lru_cache

import numpy as np
from ._jit import njit, prange

R_SPHERE = 6370997.0            # PROJ's +ellps=sphere radius, metres
WGS84_A = 6378137.0             # metres
WGS84_F = 1.0 / 298.257223563
NPARAM = 24                     # params of every projection; the last 8 are _aux's
AUX_FWD, AUX_INV = 16, 20       # params: lat -> auxiliary latitude, and back
PAR_MIN = 1 << 14               # points from which arrays run in parallel

# kind codes
GEODETIC = 0
PLATE_CARREE = 1
MERCATOR = 2
LAMBERT_CONFORMAL = 3
STEREOGRAPHIC = 4
ORTHOGRAPHIC = 5
LAMBERT_AZIMUTHAL_EQUAL_AREA = 6
ALBERS_EQUAL_AREA = 7
ROBINSON = 8
MILLER = 9
ROTATED_POLE = 10

D2R = math.pi / 180.0
R2D = 180.0 / math.pi
HALFPI = math.pi / 2.0
FORTPI = math.pi / 4.0
NAN = float('nan')
EPS10 = 1e-10
LIMTOL = 1e-12                  # relative slack of the inverse domain checks
CUTTOL = 1e-9                   # degrees of slack at an LCC cutoff, inverse

# Robinson's table (PROJ robin.cpp, 5-degree nodes; float there, so float32
# here: the same doubles once promoted). Rows c0..c3 of X (parallel length)
# and Y (distance from the equator), flat.
_ROB_X = np.array([
    1.0, 2.2199e-17, -7.15515e-05, 3.1103e-06,
    0.9986, -0.000482243, -2.4897e-05, -1.3309e-06,
    0.9954, -0.00083103, -4.48605e-05, -9.86701e-07,
    0.99, -0.00135364, -5.9661e-05, 3.6777e-06,
    0.9822, -0.00167442, -4.49547e-06, -5.72411e-06,
    0.973, -0.00214868, -9.03571e-05, 1.8736e-08,
    0.96, -0.00305085, -9.00761e-05, 1.64917e-06,
    0.9427, -0.00382792, -6.53386e-05, -2.6154e-06,
    0.9216, -0.00467746, -0.00010457, 4.81243e-06,
    0.8962, -0.00536223, -3.23831e-05, -5.43432e-06,
    0.8679, -0.00609363, -0.000113898, 3.32484e-06,
    0.835, -0.00698325, -6.40253e-05, 9.34959e-07,
    0.7986, -0.00755338, -5.00009e-05, 9.35324e-07,
    0.7597, -0.00798324, -3.5971e-05, -2.27626e-06,
    0.7186, -0.00851367, -7.01149e-05, -8.6303e-06,
    0.6732, -0.00986209, -0.000199569, 1.91974e-05,
    0.6213, -0.010418, 8.83923e-05, 6.24051e-06,
    0.5722, -0.00906601, 0.000182, 6.24051e-06,
    0.5322, -0.00677797, 0.000275608, 6.24051e-06,
], np.float32).astype(np.float64)
_ROB_Y = np.array([
    -5.20417e-18, 0.0124, 1.21431e-18, -8.45284e-11,
    0.062, 0.0124, -1.26793e-09, 4.22642e-10,
    0.124, 0.0124, 5.07171e-09, -1.60604e-09,
    0.186, 0.0123999, -1.90189e-08, 6.00152e-09,
    0.248, 0.0124002, 7.10039e-08, -2.24e-08,
    0.31, 0.0123992, -2.64997e-07, 8.35986e-08,
    0.372, 0.0124029, 9.88983e-07, -3.11994e-07,
    0.434, 0.0123893, -3.69093e-06, -4.35621e-07,
    0.4958, 0.0123198, -1.02252e-05, -3.45523e-07,
    0.5571, 0.0121916, -1.54081e-05, -5.82288e-07,
    0.6176, 0.0119938, -2.41424e-05, -5.25327e-07,
    0.6769, 0.011713, -3.20223e-05, -5.16405e-07,
    0.7346, 0.0113541, -3.97684e-05, -6.09052e-07,
    0.7903, 0.0109107, -4.89042e-05, -1.04739e-06,
    0.8435, 0.0103431, -6.4615e-05, -1.40374e-09,
    0.8936, 0.00969686, -6.4636e-05, -8.547e-06,
    0.9394, 0.00840947, -0.000192841, -4.2106e-06,
    0.9761, 0.00616527, -0.000256, -4.2106e-06,
    1.0, 0.00328947, -0.000319159, -4.2106e-06,
], np.float32).astype(np.float64)
_ROB_FXC = 0.8487
_ROB_FYC = 1.3523
_ROB_C1 = 11.45915590261646417544      # 1 / 5 degrees, in radians
_ROB_RC1 = 0.08726646259971647884      # 5 degrees, in radians
_ROB_NODES = 18
_ROB_ONEEPS = 1.000001
_ROB_EPS = 1e-10
_ROB_MAXIT = 100

_MILL_YMAX = 1.25 * math.log(math.tan(FORTPI + 0.4 * HALFPI))   # 2.3034125433763912


# ---------------------------------------------------------------- kernels

@njit(cache=True, nogil=True, error_model='numpy')
def _wrap180(d):
    """d degrees into [-180, 180); NaN stays NaN."""
    if d < -180.0 or d >= 180.0:
        d -= 360.0 * np.floor((d + 180.0) / 360.0)
        if d >= 180.0:                          # (rounding)
            d -= 360.0
    return d


@njit(cache=True, nogil=True, error_model='numpy')
def _lon_outd(lon0, d):
    """lon0 + d degrees, in [lon0 - 180, lon0 + 180)."""
    lon = lon0 + _wrap180(d)
    if lon >= lon0 + 180.0:
        lon -= 360.0
    return lon


@njit(cache=True, nogil=True, error_model='numpy')
def _lon_out(lon0, lam):
    """lon0 + lam radians, in [lon0 - 180, lon0 + 180)."""
    return _lon_outd(lon0, lam * R2D)


@njit(cache=True, nogil=True, error_model='numpy')
def _clamp1(t):
    if t > 1.0:
        return 1.0
    if t < -1.0:
        return -1.0
    return t


@njit(cache=True, nogil=True, error_model='numpy')
def _azinv(p, x, y, rho, c):
    """(lam, phi) radians of the point at azimuthal (x, y), rho = |(x, y)|,
    angular distance c from the centre: the inverse every azimuthal
    projection shares once it has c. p[2..5]: sin phi0, cos phi0, mode
    (1 north pole, -1 south, 0 oblique), phi0."""
    mode = p[4]
    if rho == 0.0:
        return 0.0, p[5]
    if mode > 0.0:
        return math.atan2(x, -y), HALFPI - c
    if mode < 0.0:
        return math.atan2(x, y), c - HALFPI
    s0, c0 = p[2], p[3]
    sc = math.sin(c)
    cc = math.cos(c)
    phi = math.asin(_clamp1(cc * s0 + y * sc * c0 / rho))
    return math.atan2(x * sc, rho * c0 * cc - y * s0 * sc), phi


@njit(cache=True, nogil=True, error_model='numpy')
def _aux(phi, p, i):
    """phi as an auxiliary latitude (or back): phi + the series in sin 2k
    phi, k = 1..4, written as sin 2 phi times a cubic in cos 2 phi, its
    coefficients p[i..i+3] (_aux_params). Exact at the poles and equator."""
    s = math.sin(phi + phi)
    c = math.cos(phi + phi)
    return phi + s * (p[i] + c * (p[i + 1] + c * (p[i + 2] + c * p[i + 3])))


@njit(cache=True, nogil=True, error_model='numpy')
def _lat_out(p, phi):
    """phi radians (auxiliary, on an ellipsoid) as latitude in degrees."""
    if p[AUX_INV] != 0.0:
        phi = _aux(phi, p, AUX_INV)
    return phi * R2D


@njit(cache=True, nogil=True, error_model='numpy')
def _rob_v(t, i, z):
    j = 4 * i
    return t[j] + z * (t[j + 1] + z * (t[j + 2] + z * t[j + 3]))


@njit(cache=True, nogil=True, error_model='numpy')
def _rob_dv(t, i, z):
    j = 4 * i
    return t[j + 1] + 2.0 * z * t[j + 2] + z * z * 3.0 * t[j + 3]


@njit(cache=True, nogil=True, error_model='numpy')
def fwd(kind, p, lon, lat):
    """(x, y) of the point (lon, lat) degrees on projection (kind, p);
    NaN where it has no image. For other kernels, point by point."""
    if not (lat >= -90.0 and lat <= 90.0) or lon != lon:     # (NaN too)
        return NAN, NAN
    if kind <= PLATE_CARREE:
        return _wrap180(lon - p[1]), lat
    lam = _wrap180(lon - p[1]) * D2R
    phi = lat * D2R
    if p[AUX_FWD] != 0.0:                               # an ellipsoid: the same formulas
        phi = _aux(phi, p, AUX_FWD)
    R = p[0]
    if kind == MERCATOR:
        if lat == 90.0 or lat == -90.0:
            return NAN, NAN
        return p[2] * lam, p[2] * math.asinh(math.tan(phi))
    if kind == LAMBERT_CONFORMAL:
        n = p[2]
        if (n > 0.0 and lat < p[5]) or (n < 0.0 and lat > p[5]):
            return NAN, NAN
        if abs(abs(phi) - HALFPI) < EPS10:
            if phi * n <= 0.0:
                return NAN, NAN
            rho = 0.0
        else:
            rho = p[3] * math.pow(math.tan(FORTPI + 0.5 * phi), -n)
        t = lam * n
        return rho * math.sin(t), p[4] - rho * math.cos(t)
    if kind == STEREOGRAPHIC or kind == ORTHOGRAPHIC or kind == LAMBERT_AZIMUTHAL_EQUAL_AREA:
        mode = p[4]
        sl = math.sin(lam)
        cl = math.cos(lam)
        if mode == 0.0:
            s0, c0 = p[2], p[3]
            sp = math.sin(phi)
            cp = math.cos(phi)
            if kind == ORTHOGRAPHIC:
                if s0 * sp + c0 * cp * cl < -EPS10:     # the far side
                    return NAN, NAN
                k = R
            else:
                d = 1.0 + s0 * sp + c0 * cp * cl        # (PROJ's order: matters at the antipode)
                if d <= EPS10:
                    return NAN, NAN
                if kind == LAMBERT_AZIMUTHAL_EQUAL_AREA:
                    k = math.sqrt(2.0 / d)
                    return p[6] * k * cp * sl, p[7] * k * (c0 * sp - s0 * cp * cl)
                k = p[6] / d
            return k * cp * sl, k * (c0 * sp - s0 * cp * cl)
        if mode < 0.0:                          # south pole: as north, mirrored
            phi = -phi
            cl = -cl
        if kind == STEREOGRAPHIC:
            if abs(phi + HALFPI) < 1e-8:
                return NAN, NAN
            rho = p[6] * math.tan(FORTPI - 0.5 * phi)
        elif kind == ORTHOGRAPHIC:
            if phi < -EPS10:
                return NAN, NAN
            rho = R * math.cos(phi)
        else:
            if abs(phi + HALFPI) < EPS10:
                return NAN, NAN
            rho = 2.0 * p[6] * math.sin(FORTPI - 0.5 * phi)
        return rho * sl, -rho * cl
    if kind == ALBERS_EQUAL_AREA:
        t = p[4] - p[3] * math.sin(phi)
        if t < 0.0:
            return NAN, NAN
        rho = p[5] * math.sqrt(t)
        t = lam * p[2]
        return R * (rho * math.sin(t)), R * (p[6] - rho * math.cos(t))
    if kind == ROBINSON:
        dphi = abs(phi)
        i = int(np.floor(dphi * _ROB_C1 + 1e-15))
        if i >= _ROB_NODES:
            i = _ROB_NODES
        dphi = R2D * (dphi - _ROB_RC1 * i)
        x = _rob_v(_ROB_X, i, dphi) * _ROB_FXC * lam
        y = _rob_v(_ROB_Y, i, dphi) * _ROB_FYC
        if phi < 0.0:
            y = -y
        return R * x, R * y
    if kind == MILLER:
        return R * lam, R * (math.log(math.tan(FORTPI + phi * 0.4)) * 1.25)
    if kind == ROTATED_POLE:
        sp = math.sin(phi)
        cp = math.cos(phi)
        cl = math.cos(lam)
        lr = math.atan2(cp * math.sin(lam), p[2] * cp * cl + p[3] * sp) + p[4]
        pr = math.asin(_clamp1(p[2] * sp - p[3] * cp * cl))
        return _wrap180(lr * R2D), pr * R2D
    return NAN, NAN


@njit(cache=True, nogil=True, error_model='numpy')
def inv(kind, p, x, y):
    """(lon, lat) degrees of the point (x, y) of projection (kind, p); NaN
    outside its domain. For other kernels, point by point."""
    if kind <= PLATE_CARREE:
        if not (y >= -90.0 and y <= 90.0) or not abs(x) < np.inf:
            return NAN, NAN
        return _lon_outd(p[1], x), y
    R = p[0]
    if kind == MERCATOR:
        a = p[2]
        if not (abs(x) <= p[5] and y >= p[3] and y <= p[4]):
            return NAN, NAN
        return _lon_out(p[1], x / a), _lat_out(p, math.atan(math.sinh(y / a)))
    if kind == LAMBERT_CONFORMAL:
        n = p[2]
        dx = x
        dy = p[4] - y
        rho = math.hypot(dx, dy)
        if rho == 0.0:
            lam = 0.0
            phi = HALFPI if n > 0.0 else -HALFPI
        else:
            if n < 0.0:
                rho = -rho
                dx = -dx
                dy = -dy
            phi = 2.0 * math.atan(math.pow(p[3] / rho, 1.0 / n)) - HALFPI
            lam = math.atan2(dx, dy) / n
            if not abs(lam) <= math.pi * (1.0 + LIMTOL):
                return NAN, NAN
        lat = _lat_out(p, phi)
        if (n > 0.0 and lat < p[5] - CUTTOL) or (n < 0.0 and lat > p[5] + CUTTOL) or not abs(lat) <= 90.0:
            return NAN, NAN
        return _lon_out(p[1], lam), lat
    if kind == STEREOGRAPHIC:
        u = x / p[7]
        v = y / p[8]
        if not (u * u + v * v <= 1.0 + LIMTOL):
            return NAN, NAN
        rho = math.hypot(x, y)
        lam, phi = _azinv(p, x, y, rho, 2.0 * math.atan(rho / p[6]))
        return _lon_out(p[1], lam), _lat_out(p, phi)
    if kind == ORTHOGRAPHIC:
        rho = math.hypot(x, y)
        s = rho / R
        if not s <= 1.0 + EPS10:
            return NAN, NAN
        lam, phi = _azinv(p, x, y, rho, math.asin(_clamp1(s)))
        return _lon_out(p[1], lam), phi * R2D
    if kind == LAMBERT_AZIMUTHAL_EQUAL_AREA:
        rq = p[8]
        x *= rq / p[6]
        y *= rq / p[7]
        rho = math.hypot(x, y)
        s = 0.5 * rho / rq
        if not s <= 1.0:
            return NAN, NAN
        lam, phi = _azinv(p, x, y, rho, 2.0 * math.asin(s))
        return _lon_out(p[1], lam), _lat_out(p, phi)
    if kind == ALBERS_EQUAL_AREA:
        n = p[2]
        dx = x / R
        dy = p[6] - y / R
        rho = math.hypot(dx, dy)                # (0 only at the apex: off the map)
        if n < 0.0:
            rho = -rho
            dx = -dx
            dy = -dy
        q = rho / p[5]
        s = (p[4] - q * q) / p[3]
        if not abs(s) <= 1.0 + LIMTOL:
            return NAN, NAN
        phi = math.asin(_clamp1(s))
        lam = math.atan2(dx, dy) / n
        if not abs(lam) <= math.pi * (1.0 + LIMTOL):
            return NAN, NAN
        return _lon_out(p[1], lam), _lat_out(p, phi)
    if kind == ROBINSON:
        lam = x / R / _ROB_FXC
        phi = abs(y / R / _ROB_FYC)
        if not phi < 1.0:
            if not phi <= _ROB_ONEEPS:
                return NAN, NAN
            phi = HALFPI if y > 0.0 else -HALFPI
            lam /= _ROB_X[4 * _ROB_NODES]
        else:
            i = int(np.floor(phi * _ROB_NODES))
            while True:                         # into the node interval
                if _ROB_Y[4 * i] > phi:
                    i -= 1
                elif _ROB_Y[4 * i + 4] <= phi:
                    i += 1
                else:
                    break
            t = 5.0 * (phi - _ROB_Y[4 * i]) / (_ROB_Y[4 * i + 4] - _ROB_Y[4 * i])
            for _ in range(_ROB_MAXIT):         # Newton-Raphson
                t1 = (_rob_v(_ROB_Y, i, t) - phi) / _rob_dv(_ROB_Y, i, t)
                t -= t1
                if abs(t1) < _ROB_EPS:
                    break
            phi = (5.0 * i + t) * D2R
            if y < 0.0:
                phi = -phi
            lam /= _rob_v(_ROB_X, i, t)
        if not abs(lam) <= math.pi * (1.0 + LIMTOL):
            return NAN, NAN
        return _lon_out(p[1], lam), phi * R2D
    if kind == MILLER:
        if not (abs(x) <= p[2] and abs(y) <= p[3]):
            return NAN, NAN
        return _lon_out(p[1], x / R), 2.5 * (math.atan(math.exp(0.8 * (y / R))) - FORTPI) * R2D
    if kind == ROTATED_POLE:
        if not (y >= -90.0 and y <= 90.0) or not abs(x) < np.inf:
            return NAN, NAN
        lr = x * D2R - p[4]
        pr = y * D2R
        sp = math.sin(pr)
        cp = math.cos(pr)
        cl = math.cos(lr)
        phi = math.asin(_clamp1(p[2] * sp + p[3] * cp * cl))
        lam = math.atan2(cp * math.sin(lr), p[2] * cp * cl - p[3] * sp)
        return _wrap180(lam * R2D + p[1]), phi * R2D
    return NAN, NAN


@njit(cache=True, nogil=True, error_model='numpy')
def _fwd_ser(kind, p, a, b, x, y):
    for i in range(a.size):
        x[i], y[i] = fwd(kind, p, a[i], b[i])


@njit(cache=True, nogil=True, parallel=True, error_model='numpy')
def _fwd_par(kind, p, a, b, x, y):
    for i in prange(a.size):
        x[i], y[i] = fwd(kind, p, a[i], b[i])


@njit(cache=True, nogil=True, error_model='numpy')
def _inv_ser(kind, p, a, b, x, y):
    for i in range(a.size):
        x[i], y[i] = inv(kind, p, a[i], b[i])


@njit(cache=True, nogil=True, parallel=True, error_model='numpy')
def _inv_par(kind, p, a, b, x, y):
    for i in prange(a.size):
        x[i], y[i] = inv(kind, p, a[i], b[i])


@njit(cache=True, nogil=True, error_model='numpy')
def _xf_ser(k1, p1, k2, p2, a, b, x, y):
    for i in range(a.size):
        lon, lat = inv(k1, p1, a[i], b[i])
        x[i], y[i] = fwd(k2, p2, lon, lat)


@njit(cache=True, nogil=True, parallel=True, error_model='numpy')
def _xf_par(k1, p1, k2, p2, a, b, x, y):
    for i in prange(a.size):
        lon, lat = inv(k1, p1, a[i], b[i])
        x[i], y[i] = fwd(k2, p2, lon, lat)


def _points(a, b):
    """Two coordinate arrays as flat contiguous float64, and their shape."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        a, b = np.broadcast_arrays(a, b)
    shape = a.shape
    return np.ascontiguousarray(a).reshape(-1), np.ascontiguousarray(b).reshape(-1), shape


def _shaped(x, y, shape):
    if shape == ():
        return x[0], y[0]
    return x.reshape(shape), y.reshape(shape)


def _run(ser, par, args, a, b):
    a, b, shape = _points(a, b)
    x = np.empty(a.size)
    y = np.empty(a.size)
    (par if a.size >= PAR_MIN else ser)(*args, a, b, x, y)
    return _shaped(x, y, shape)


# ---------------------------------------------------------------- classes

def _line(x0, y0, x1, y1, n):
    """n points from (x0, y0) to (x1, y1), the last left out."""
    t = np.arange(n) / n
    return np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t], axis=1)


def _rect(x0, x1, y0, y1, n=180):
    """The closed rectangle ring, counter-clockwise, n points a side."""
    ring = np.concatenate([_line(x0, y0, x1, y0, n), _line(x1, y0, x1, y1, n),
                           _line(x1, y1, x0, y1, n), _line(x0, y1, x0, y0, n)])
    return np.concatenate([ring, ring[:1]])


def _ellipse(a, b, n=360):
    t = np.linspace(0.0, 2.0 * np.pi, n + 1)
    ring = np.stack([a * np.cos(t), b * np.sin(t)], axis=1)
    ring[-1] = ring[0]
    return ring


def _ccw(ring):
    """The ring, counter-clockwise (the shoelace area positive)."""
    x, y = ring[:, 0], ring[:, 1]
    if np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) < 0:
        return np.ascontiguousarray(ring[::-1])
    return ring


def _fmt(v):
    return repr(float(v))


class Projection:
    """A map projection (or lon/lat coordinate system) on a sphere: `kind`
    and `params` for the kernels, its domain (`x_limits`, `y_limits`,
    `boundary`), and forward / inverse for arrays."""

    kind = -1
    cylindrical = False
    period = 0.0
    seam = False                # cut along central_longitude + 180, not cylindrical
    lat_limits = (-90.0, 90.0)  # the latitudes it has an image of
    threshold = 0.5             # cartopy's: how far a straight line may miss a curve

    def _setup(self, params, args, proj4, xl, yl, boundary, aux=None):
        p = np.zeros(NPARAM)
        p[:len(params)] = params
        if aux is not None:
            p[AUX_FWD:] = aux
        self.params = p
        self._args = args
        self.proj4 = proj4
        self.x_limits = (float(xl[0]), float(xl[1]))
        self.y_limits = (float(yl[0]), float(yl[1]))
        self.boundary = np.ascontiguousarray(boundary, dtype=np.float64)

    @property
    def radius(self):
        return float(self.params[0])

    def forward(self, lon, lat):
        """(x, y) of lon/lat degrees (any shapes that broadcast); NaN where
        a point has no image."""
        return _run(_fwd_ser, _fwd_par, (self.kind, self.params), lon, lat)

    def inverse(self, x, y):
        """(lon, lat) degrees of x/y; NaN outside the domain, longitudes in
        [central_longitude - 180, central_longitude + 180)."""
        return _run(_inv_ser, _inv_par, (self.kind, self.params), x, y)

    def as_geodetic(self):
        return Geodetic(radius=self.radius)

    def __eq__(self, other):
        if not isinstance(other, Projection):
            return NotImplemented
        return self.kind == other.kind and np.array_equal(self.params, other.params)

    def __hash__(self):
        return hash((self.kind, self.params.tobytes()))

    def __repr__(self):
        a = ', '.join('%s=%r' % kv for kv in self._args)
        return '%s(%s)' % (type(self).__name__, a)


def _radius(radius):
    r = float(radius)
    if not r > 0:
        raise ValueError('radius must be positive, got %r' % (radius,))
    return r


_ELLIPSES = {'WGS84': (WGS84_A, WGS84_F), 'GRS80': (WGS84_A, 1.0 / 298.257222101),
             'sphere': (R_SPHERE, 0.0), None: (WGS84_A, 0.0)}


class Globe:
    """The shape of the Earth, as cartopy.crs.Globe takes it: a named
    `ellipse`, then semimajor_axis, and semiminor_axis, flattening or
    inverse_flattening to change it. `datum`, `towgs84` and `nadgrids` are
    accepted and not used: fastplot does no datum shifts."""

    def __init__(self, datum=None, ellipse='WGS84', semimajor_axis=None, semiminor_axis=None,
                 flattening=None, inverse_flattening=None, towgs84=None, nadgrids=None):
        if ellipse not in _ELLIPSES:
            raise ValueError('ellipse %r: fastplot knows %s'
                             % (ellipse, sorted(k for k in _ELLIPSES if k)))
        self.datum, self.ellipse = datum, ellipse
        self.towgs84, self.nadgrids = towgs84, nadgrids
        a, f = _ELLIPSES[ellipse]
        if semimajor_axis is not None:
            a = _radius(semimajor_axis)
        if semiminor_axis is not None:
            f = 1.0 - float(semiminor_axis) / a
        elif flattening is not None:
            f = float(flattening)
        elif inverse_flattening is not None:
            f = 1.0 / float(inverse_flattening)
        if not 0.0 <= f < 1.0:
            raise ValueError('a globe\'s flattening must be in [0, 1), got %r' % (f,))
        self.semimajor_axis, self.flattening = a, f
        self.semiminor_axis = a * (1.0 - f)

    def __repr__(self):
        return 'Globe(semimajor_axis=%r, flattening=%r)' % (self.semimajor_axis, self.flattening)


def _earth(radius, globe, ellipsoid=True):
    """(a, f) of a projection's Earth: `radius`, a sphere; `globe`; else
    cartopy's default, WGS84 or (ellipsoid False) a sphere of its a."""
    if radius is not None:
        return _radius(radius), 0.0
    if globe is not None:
        return globe.semimajor_axis, globe.flattening if ellipsoid else 0.0
    return WGS84_A, WGS84_F if ellipsoid else 0.0


def _earth_args(radius, globe):
    """A projection's repr of its Earth, when it was asked for."""
    return ([('radius', float(radius))] if radius is not None else []) + \
        ([('globe', globe)] if globe is not None else [])


def _p4_earth(a, f):
    return '+R=%s' % _fmt(a) if f == 0.0 else '+a=%s +rf=%s' % (_fmt(a), _fmt(1.0 / f))


def _msfn(s, c, e2):
    """PROJ's msfn: a parallel's radius over a, from sin and cos latitude."""
    return c / np.sqrt(1.0 - e2 * s * s)


def _qsfn(s, e):
    """PROJ's qsfn: the area term of the equal-area formulas, 2 sin phi on
    a sphere."""
    if e == 0.0:
        return 2.0 * s
    es = e * s
    return (1.0 - e * e) * (s / (1.0 - es * es) - np.log((1.0 - es) / (1.0 + es)) / (2.0 * e))


def _conformal(phi, e):
    """Conformal latitude of phi radians (scalars or arrays)."""
    if e == 0.0:
        return phi
    s = e * np.sin(phi)
    return 2.0 * np.arctan(np.tan(FORTPI + 0.5 * phi) * ((1.0 - s) / (1.0 + s)) ** (0.5 * e)) - HALFPI


def _authalic(phi, e):
    """Authalic latitude of phi radians (scalars or arrays)."""
    if e == 0.0:
        return phi
    return np.arcsin(np.clip(_qsfn(np.sin(phi), e) / _qsfn(1.0, e), -1.0, 1.0))


def _series(f):
    """Coefficients a_k of f(x) = sum a_k sin 2 k x on (0, pi/2), from
    values at x = j pi / 2N, j = 1..N-1 (a discrete sine transform)."""
    n = f.size + 1
    x = np.arange(1, n) * HALFPI / n
    return np.array([2.0 / n * np.sum(f * np.sin(2.0 * k * x)) for k in range(1, 9)])


def _cubic(a):
    """sum a_k sin 2 k x, k = 1..4, as sin 2x (d0 + d1 c + d2 c^2 + d3 c^3),
    c = cos 2x: sin 2 k x / sin 2x are the Chebyshev polynomials U."""
    return a[0] - a[2], 2.0 * a[1] - 4.0 * a[3], 4.0 * a[2], 8.0 * a[3]


@lru_cache(maxsize=16)
def _aux_params(f, kind):
    """params[16:24] of an ellipsoid of flattening f: `_aux`'s coefficients
    from latitude to its auxiliary latitude (kind 'conformal' or
    'authalic') and back. They fall off like f^k, so four are good to
    1e-13 rad, a micrometre; found numerically for any f, eight terms
    behind the four kept so that the way back is as good."""
    e = math.sqrt(f * (2.0 - f))
    to_aux = _authalic if kind == 'authalic' else _conformal
    n = 64
    x = np.arange(1, n) * HALFPI / n
    a = _series(to_aux(x, e) - x)
    phi = x.copy()                                      # latitudes of those auxiliary ones
    for _ in range(8):
        phi = x - sum(a[k] * np.sin(2.0 * (k + 1) * phi) for k in range(8))
    b = _series(phi - x)
    return _cubic(a) + _cubic(b)


class Geodetic(Projection):
    """Longitude / latitude degrees: x = lon wrapped into [-180, 180),
    y = lat. The crs of lon/lat data (transform=Geodetic()); any x is
    accepted on inverse (it wraps)."""

    kind = GEODETIC
    cylindrical = True
    period = 360.0

    def __init__(self, radius=R_SPHERE, globe=None):
        if isinstance(radius, Globe):                   # cartopy's Geodetic(globe)
            radius = radius.semimajor_axis
        r = _radius(radius)
        self.central_longitude = 0.0
        self._setup([r, 0.0], [('radius', r)] if r != R_SPHERE else [],
                    '+proj=longlat +R=%s +no_defs' % _fmt(r),
                    (-180.0, 180.0), (-90.0, 90.0), _rect(-180.0, 180.0, -90.0, 90.0))


class PlateCarree(Projection):
    """Equirectangular, x and y in degrees as cartopy's: x = lon -
    central_longitude wrapped into [-180, 180), y = lat. On inverse any x is
    accepted (it wraps), so 0..360 data longitudes are fine."""

    kind = PLATE_CARREE
    cylindrical = True
    period = 360.0

    def __init__(self, central_longitude=0.0, radius=R_SPHERE, globe=None):
        r = _radius(radius)
        c = float(central_longitude)
        self.central_longitude = c
        args = [('central_longitude', c)] + ([('radius', r)] if r != R_SPHERE else [])
        self._setup([r, c], args,
                    '+proj=eqc +lon_0=%s +R=%s +to_meter=%s +no_defs'
                    % (_fmt(c), _fmt(r), _fmt(r * D2R)),
                    (-180.0, 180.0), (-90.0, 90.0), _rect(-180.0, 180.0, -90.0, 90.0))


class Mercator(Projection):
    """Mercator: x = a k0 (lon - lon0), y = a k0 asinh(tan chi), chi the
    conformal latitude; k0 the parallel latitude_true_scale's scale, or
    scale_factor. Drawn between min_ and max_latitude (its limits; inverse
    is NaN beyond them), defined forward for any |lat| < 90."""

    kind = MERCATOR
    cylindrical = True
    threshold = 55659.74539663679

    def __init__(self, central_longitude=0.0, min_latitude=-80.0, max_latitude=84.0,
                 latitude_true_scale=None, scale_factor=None, radius=None, globe=None):
        r, fl = _earth(radius, globe)
        e2, e = fl * (2.0 - fl), math.sqrt(fl * (2.0 - fl))
        c = float(central_longitude)
        lo, hi = float(min_latitude), float(max_latitude)
        if not -90 < lo < hi < 90:
            raise ValueError('need -90 < min_latitude < max_latitude < 90')
        p4 = '+proj=merc +lon_0=%s' % _fmt(c)
        if scale_factor is not None:
            if latitude_true_scale is not None:
                raise ValueError('give latitude_true_scale or scale_factor, not both')
            k0 = float(scale_factor)
            p4 += ' +k_0=%s' % _fmt(k0)
        elif latitude_true_scale is not None:
            ts = float(latitude_true_scale) * D2R
            k0 = float(_msfn(math.sin(ts), math.cos(ts), e2))
            p4 += ' +lat_ts=%s' % _fmt(latitude_true_scale)
        else:
            k0 = 1.0
        a = r * k0
        y0 = a * math.asinh(math.tan(_conformal(lo * D2R, e)))
        y1 = a * math.asinh(math.tan(_conformal(hi * D2R, e)))
        xm = math.pi * a
        self.central_longitude = c
        self.period = 2.0 * xm
        args = [('central_longitude', c), ('min_latitude', lo), ('max_latitude', hi)]
        if latitude_true_scale is not None:
            args.append(('latitude_true_scale', float(latitude_true_scale)))
        if scale_factor is not None:
            args.append(('scale_factor', float(scale_factor)))
        args += _earth_args(radius, globe)
        self._setup([r, c, a, y0, y1, xm], args, p4 + ' %s +units=m +no_defs' % _p4_earth(r, fl),
                    (-xm, xm), (y0, y1), _rect(-xm, xm, y0, y1),
                    _aux_params(fl, 'conformal') if fl else None)


class LambertConformal(Projection):
    """Lambert conformal conic, one or two standard parallels;
    the cone's apex at the pole the parallels lean to (a southern cone for
    southern parallels). The map stops at `cutoff` latitude (forward and
    inverse NaN beyond it); the wedge outside the cone's unrolled 360 deg is
    NaN on inverse."""

    kind = LAMBERT_CONFORMAL
    seam = True
    threshold = 1e5

    def __init__(self, central_longitude=-96.0, central_latitude=39.0,
                 standard_parallels=(33.0, 45.0), cutoff=-30.0, radius=None, globe=None):
        r, fl = _earth(radius, globe)
        e2, e = fl * (2.0 - fl), math.sqrt(fl * (2.0 - fl))
        c, lat0, cut = float(central_longitude), float(central_latitude), float(cutoff)
        sp = np.atleast_1d(np.asarray(standard_parallels, dtype=np.float64))
        if not 1 <= sp.size <= 2:
            raise ValueError('1 or 2 standard parallels, got %r' % (standard_parallels,))
        lat1 = float(sp[0])
        lat2 = float(sp[-1])
        phi1, phi2, phi0 = lat1 * D2R, lat2 * D2R, lat0 * D2R
        if abs(phi1 + phi2) < EPS10:
            raise ValueError('standard parallels symmetric about the equator: no cone')
        def tan_chi(phi):                       # 1 / PROJ's tsfn
            return float(math.tan(FORTPI + 0.5 * _conformal(phi, e)))
        n = math.sin(phi1)
        cosphi = float(_msfn(n, math.cos(phi1), e2))
        if abs(phi1 - phi2) >= EPS10:           # secant (PROJ's lcc)
            n = math.log(cosphi / float(_msfn(math.sin(phi2), math.cos(phi2), e2))) / math.log(
                tan_chi(phi2) / tan_chi(phi1))
        cc = cosphi * math.pow(tan_chi(phi1), n) / n
        rho0 = 0.0 if abs(abs(phi0) - HALFPI) < EPS10 else cc * math.pow(tan_chi(phi0), -n)
        RC, Rrho0 = r * cc, r * rho0
        if (n > 0 and cut >= 90) or (n < 0 and cut <= -90):
            raise ValueError('cutoff %r leaves nothing of the cone' % cutoff)
        self.central_longitude = c
        self.cutoff = cut
        self.lat_limits = (cut, 90.0) if n > 0 else (-90.0, cut)
        # the cutoff arc, apex to apex
        lam = np.linspace(-math.pi, math.pi, 721)
        rc = RC * math.pow(tan_chi(cut * D2R), -n)
        arc = np.stack([rc * np.sin(n * lam), Rrho0 - rc * np.cos(n * lam)], axis=1)
        apex = np.array([[0.0, Rrho0]])
        ring = _ccw(np.concatenate([apex, arc, apex]))
        args = [('central_longitude', c), ('central_latitude', lat0),
                ('standard_parallels', tuple(float(v) for v in sp)), ('cutoff', cut)]
        args += _earth_args(radius, globe)
        p4 = ('+proj=lcc +lon_0=%s +lat_0=%s +lat_1=%s +lat_2=%s %s +units=m +no_defs'
              % (_fmt(c), _fmt(lat0), _fmt(lat1), _fmt(lat2), _p4_earth(r, fl)))
        self._setup([r, c, n, RC, Rrho0, cut], args, p4,
                    (ring[:, 0].min(), ring[:, 0].max()),
                    (ring[:, 1].min(), ring[:, 1].max()), ring,
                    _aux_params(fl, 'conformal') if fl else None)


class Stereographic(Projection):
    """Stereographic, any aspect. true_scale_latitude (polar aspects only,
    as in PROJ) or scale_factor sets the scale. Drawn within cartopy's
    ellipse of +-5e7 m (scaled to the Earth's axes); inverse is NaN beyond
    it, forward only at the antipode."""

    kind = STEREOGRAPHIC
    threshold = 1e5

    def __init__(self, central_latitude=0.0, central_longitude=0.0,
                 true_scale_latitude=None, scale_factor=None, radius=None, globe=None):
        r, fl = _earth(radius, globe)
        e2, e = fl * (2.0 - fl), math.sqrt(fl * (2.0 - fl))
        c, lat0 = float(central_longitude), float(central_latitude)
        phi0 = lat0 * D2R
        if not abs(lat0) <= 90:
            raise ValueError('central_latitude %r' % central_latitude)
        mode = 0.0
        if abs(abs(phi0) - HALFPI) < EPS10:
            mode = 1.0 if phi0 > 0 else -1.0
        p4 = '+proj=stere +lat_0=%s +lon_0=%s' % (_fmt(lat0), _fmt(c))
        k0 = 1.0
        if true_scale_latitude is not None:
            if scale_factor is not None:
                raise ValueError('give true_scale_latitude or scale_factor, not both')
            if mode == 0.0:
                warnings.warn('true_scale_latitude is only used by polar stereographic '
                              'projections; consider scale_factor', stacklevel=2)
            p4 += ' +lat_ts=%s' % _fmt(true_scale_latitude)
        if scale_factor is not None:
            k0 = float(scale_factor)
            p4 += ' +k_0=%s' % _fmt(k0)
        chi0 = float(_conformal(phi0, e))
        if mode == 0.0:                         # p[6] is PROJ's akm1 / cos(chi0)
            s0, c0 = math.sin(chi0), math.cos(chi0)
            s = math.sin(phi0)
            akm1 = 2.0 * k0 * math.cos(phi0) / math.sqrt(1.0 - e2 * s * s) / c0
        else:
            s0, c0 = mode, 0.0
            akm1 = 2.0 * k0 / math.sqrt(pow(1.0 + e, 1.0 + e) * pow(1.0 - e, 1.0 - e))
            if true_scale_latitude is not None:
                phits = abs(float(true_scale_latitude) * D2R)
                if abs(phits - HALFPI) >= EPS10:
                    s = math.sin(phits)
                    akm1 = float(_msfn(s, math.cos(phits), e2)) / math.tan(
                        FORTPI - 0.5 * float(_conformal(phits, e)))
        xl = r * (5e7 / WGS84_A)                # cartopy's limits, WGS84's 5e7 m scaled
        yl = r * (1.0 - fl) * (5e7 / 6356752.3142)
        self.central_longitude = c
        self.central_latitude = lat0
        args = self._names(c, lat0, true_scale_latitude, scale_factor) + \
            _earth_args(radius, globe)
        self._setup([r, c, s0, c0, mode, chi0, akm1 * r, xl, yl], args,
                    p4 + ' %s +units=m +no_defs' % _p4_earth(r, fl),
                    (-xl, xl), (-yl, yl), _ellipse(xl, yl),
                    _aux_params(fl, 'conformal') if fl else None)

    @staticmethod
    def _names(c, lat0, ts, sf):
        args = [('central_latitude', lat0), ('central_longitude', c)]
        if ts is not None:
            args.append(('true_scale_latitude', float(ts)))
        if sf is not None:
            args.append(('scale_factor', float(sf)))
        return args


class NorthPolarStereo(Stereographic):
    """Stereographic centred on the north pole."""

    def __init__(self, central_longitude=0.0, true_scale_latitude=None, radius=None, globe=None):
        super().__init__(90.0, central_longitude, true_scale_latitude, None, radius, globe)
        self._args = [kv for kv in self._args if kv[0] != 'central_latitude']


class SouthPolarStereo(Stereographic):
    """Stereographic centred on the south pole."""

    def __init__(self, central_longitude=0.0, true_scale_latitude=None, radius=None, globe=None):
        super().__init__(-90.0, central_longitude, true_scale_latitude, None, radius, globe)
        self._args = [kv for kv in self._args if kv[0] != 'central_latitude']


def _azimuthal(lat0):
    phi0 = float(lat0) * D2R
    if not abs(float(lat0)) <= 90:
        raise ValueError('central_latitude %r' % lat0)
    if abs(abs(phi0) - HALFPI) < EPS10:
        m = 1.0 if phi0 > 0 else -1.0
        return m, 0.0, m, phi0
    return math.sin(phi0), math.cos(phi0), 0.0, phi0


class Orthographic(Projection):
    """Orthographic (the globe seen from infinitely far), always a sphere,
    as cartopy's. The far hemisphere is NaN forward; the disc, as cartopy's,
    is 0.99999 R."""

    kind = ORTHOGRAPHIC
    threshold = 255122.9287452

    def __init__(self, central_longitude=0.0, central_latitude=0.0, radius=None, globe=None):
        r, _ = _earth(radius, globe, ellipsoid=False)
        c, lat0 = float(central_longitude), float(central_latitude)
        s0, c0, mode, phi0 = _azimuthal(lat0)
        a = r * 0.99999
        self.central_longitude = c
        self.central_latitude = lat0
        args = [('central_longitude', c), ('central_latitude', lat0)] + \
            _earth_args(radius, globe)
        self._setup([r, c, s0, c0, mode, phi0], args,
                    '+proj=ortho +lon_0=%s +lat_0=%s +R=%s +units=m +no_defs'
                    % (_fmt(c), _fmt(lat0), _fmt(r)),
                    (-a, a), (-a, a), _ellipse(a, a))


class LambertAzimuthalEqualArea(Projection):
    """Lambert azimuthal equal-area: the whole Earth in a disc of radius 2
    Rq (Rq the sphere of its area), its antipode (the rim) NaN forward; on
    an ellipsoid the authalic sphere, x and y scaled to keep the scale true
    at the centre. Limits as cartopy's (1.9999 a wide, the antipode less
    0.01 deg high)."""

    kind = LAMBERT_AZIMUTHAL_EQUAL_AREA
    threshold = 25511.2723726

    def __init__(self, central_longitude=0.0, central_latitude=0.0, radius=None, globe=None):
        r, fl = _earth(radius, globe)
        e2, e = fl * (2.0 - fl), math.sqrt(fl * (2.0 - fl))
        c, lat0 = float(central_longitude), float(central_latitude)
        s0, c0, mode, phi0 = _azimuthal(lat0)
        beta0 = float(_authalic(phi0, e))
        rq = r * math.sqrt(0.5 * float(_qsfn(1.0, e)))
        xmf = ymf = rq
        if mode == 0.0:                         # PROJ's dd
            s0, c0 = math.sin(beta0), math.cos(beta0)
            s = math.sin(phi0)
            dd = math.cos(phi0) / (math.sqrt(1.0 - e2 * s * s) * (rq / r) * c0)
            xmf, ymf = rq * dd, rq / dd
        # the top of cartopy's ellipse: the image of the antipode less 0.01 deg
        beta = float(_authalic((-lat0 + 0.01) * D2R, e))
        if mode == 0.0:
            b = abs(ymf * math.sqrt(2.0 / (1.0 + s0 * math.sin(beta) - c0 * math.cos(beta)))
                    * (c0 * math.sin(beta) + s0 * math.cos(beta)))
        else:
            b = 2.0 * rq * math.sin(FORTPI - 0.5 * beta * mode)
        a = r * 1.9999
        self.central_longitude = c
        self.central_latitude = lat0
        args = [('central_longitude', c), ('central_latitude', lat0)] + \
            _earth_args(radius, globe)
        self._setup([r, c, s0, c0, mode, beta0, xmf, ymf, rq], args,
                    '+proj=laea +lon_0=%s +lat_0=%s %s +units=m +no_defs'
                    % (_fmt(c), _fmt(lat0), _p4_earth(r, fl)),
                    (-a, a), (-b, b), _ellipse(a, b),
                    _aux_params(fl, 'authalic') if fl else None)


class AlbersEqualArea(Projection):
    """Albers equal-area conic, one or two standard parallels: the whole
    Earth, an annulus sector between the images of the poles."""

    kind = ALBERS_EQUAL_AREA
    seam = True
    threshold = 1e5

    def __init__(self, central_longitude=0.0, central_latitude=0.0,
                 standard_parallels=(20.0, 50.0), radius=None, globe=None):
        r, fl = _earth(radius, globe)
        e2, e = fl * (2.0 - fl), math.sqrt(fl * (2.0 - fl))
        c, lat0 = float(central_longitude), float(central_latitude)
        sp = np.atleast_1d(np.asarray(standard_parallels, dtype=np.float64))
        if not 1 <= sp.size <= 2:
            raise ValueError('1 or 2 standard parallels, got %r' % (standard_parallels,))
        lat1, lat2 = float(sp[0]), float(sp[-1])
        phi1, phi2, phi0 = lat1 * D2R, lat2 * D2R, lat0 * D2R
        if abs(phi1 + phi2) < EPS10:
            raise ValueError('standard parallels symmetric about the equator: no cone')
        n = sinphi = math.sin(phi1)             # PROJ's aea
        m1, q1 = float(_msfn(sinphi, math.cos(phi1), e2)), float(_qsfn(sinphi, e))
        if abs(phi1 - phi2) >= EPS10:
            s2 = math.sin(phi2)
            m2, q2 = float(_msfn(s2, math.cos(phi2), e2)), float(_qsfn(s2, e))
            n = (m1 * m1 - m2 * m2) / (q2 - q1)
        n2 = n * float(_qsfn(1.0, e))           # 2 n on a sphere: then sin of the authalic lat
        cc = m1 * m1 + n * q1
        dd = 1.0 / n
        rho0 = dd * math.sqrt(cc - n * float(_qsfn(math.sin(phi0), e)))
        lam = np.linspace(-math.pi, math.pi, 361)
        rn = r * dd * math.sqrt(max(cc - n2, 0.0))
        rs = r * dd * math.sqrt(cc + n2)
        north = np.stack([rn * np.sin(n * lam), r * rho0 - rn * np.cos(n * lam)], axis=1)
        south = np.stack([rs * np.sin(n * lam), r * rho0 - rs * np.cos(n * lam)], axis=1)
        ring = _ccw(np.concatenate([north, south[::-1], north[:1]]))
        self.central_longitude = c
        args = [('central_longitude', c), ('central_latitude', lat0),
                ('standard_parallels', tuple(float(v) for v in sp))] + _earth_args(radius, globe)
        p4 = ('+proj=aea +lon_0=%s +lat_0=%s +lat_1=%s +lat_2=%s %s +units=m +no_defs'
              % (_fmt(c), _fmt(lat0), _fmt(lat1), _fmt(lat2), _p4_earth(r, fl)))
        self._setup([r, c, n, n2, cc, dd, rho0], args, p4,
                    (ring[:, 0].min(), ring[:, 0].max()),
                    (ring[:, 1].min(), ring[:, 1].max()), ring,
                    _aux_params(fl, 'authalic') if fl else None)


def _rob_np(lat):
    """Robinson's (X, Y) factors of latitudes (degrees), numpy: x = R FXC X
    lam, y = R FYC Y sign(lat)."""
    dphi = np.abs(np.asarray(lat, dtype=np.float64) * D2R)
    i = np.minimum(np.floor(dphi * _ROB_C1 + 1e-15).astype(np.int64), _ROB_NODES)
    z = R2D * (dphi - _ROB_RC1 * i)
    tx, ty = _ROB_X.reshape(-1, 4)[i], _ROB_Y.reshape(-1, 4)[i]
    v = [t[:, 0] + z * (t[:, 1] + z * (t[:, 2] + z * t[:, 3])) for t in (tx, ty)]
    return v[0], v[1]


class Robinson(Projection):
    """Robinson, PROJ's table (5-degree nodes, cubic between); the inverse
    solves the table's latitude by Newton's method."""

    kind = ROBINSON
    seam = True
    threshold = 1e4

    def __init__(self, central_longitude=0.0, radius=None, globe=None):
        r, _ = _earth(radius, globe, ellipsoid=False)
        c = float(central_longitude)
        lat = np.linspace(-90.0, 90.0, 361)
        X, Y = _rob_np(lat)
        x = r * X * _ROB_FXC * math.pi
        y = r * Y * _ROB_FYC * np.sign(lat)
        right = np.stack([x, y], axis=1)
        left = np.stack([-x, y], axis=1)[::-1]
        ring = _ccw(np.concatenate([right, left, right[:1]]))
        self.central_longitude = c
        args = [('central_longitude', c)] + _earth_args(radius, globe)
        self._setup([r, c], args, '+proj=robin +lon_0=%s +R=%s +units=m +no_defs'
                    % (_fmt(c), _fmt(r)),
                    (ring[:, 0].min(), ring[:, 0].max()),
                    (ring[:, 1].min(), ring[:, 1].max()), ring)


class Miller(Projection):
    """Miller cylindrical, in metres, always a sphere, as cartopy's (of
    radius 6378137 m unless told otherwise)."""

    kind = MILLER
    cylindrical = True

    def __init__(self, central_longitude=0.0, radius=None, globe=None):
        r, _ = _earth(radius, globe, ellipsoid=False)
        c = float(central_longitude)
        xm, ym = math.pi * r, _MILL_YMAX * r
        self.central_longitude = c
        self.period = 2.0 * xm
        args = [('central_longitude', c)] + _earth_args(radius, globe)
        self._setup([r, c, xm, ym], args, '+proj=mill +lon_0=%s +R=%s +units=m +no_defs'
                    % (_fmt(c), _fmt(r)), (-xm, xm), (-ym, ym), _rect(-xm, xm, -ym, ym))


class RotatedPole(Projection):
    """Rotated longitude / latitude (degrees) of the sphere whose north pole
    is at (pole_longitude, pole_latitude), turned by
    central_rotated_longitude about it: PROJ's ob_tran as cartopy builds it,
    the crs of rotated-pole model grids (COSMO, ICON-LAM, HIRLAM, ...).
    Inverse takes any x (it wraps) and gives longitudes in [-180, 180)."""

    kind = ROTATED_POLE
    cylindrical = True
    period = 360.0

    def __init__(self, pole_longitude=0.0, pole_latitude=90.0,
                 central_rotated_longitude=0.0, radius=R_SPHERE, globe=None):
        r = _radius(radius)
        plon, plat, crl = float(pole_longitude), float(pole_latitude), float(
            central_rotated_longitude)
        php = plat * D2R
        lon0 = 180.0 + plon
        self.central_longitude = 0.0
        args = [('pole_longitude', plon), ('pole_latitude', plat),
                ('central_rotated_longitude', crl)]
        if r != R_SPHERE:
            args.append(('radius', r))
        p4 = ('+proj=ob_tran +o_proj=longlat +o_lon_p=%s +o_lat_p=%s +lon_0=%s +R=%s +no_defs'
              % (_fmt(crl), _fmt(plat), _fmt(lon0), _fmt(r)))
        self._setup([r, lon0, math.sin(php), math.cos(php), crl * D2R], args, p4,
                    (-180.0, 180.0), (-90.0, 90.0), _rect(-180.0, 180.0, -90.0, 90.0))


# ---------------------------------------------------------------- functions

def transform_points(src, dst, x, y):
    """(x2, y2) in dst of the points (x, y) of src (any shapes that
    broadcast), through lon/lat in one pass; NaN where either step is."""
    if src == dst:
        a, b, shape = _points(x, y)
        return _shaped(a.copy(), b.copy(), shape)
    return _run(_xf_ser, _xf_par, (src.kind, src.params, dst.kind, dst.params), x, y)


EXTENT_SAMPLES = 2048           # points an edge of an extent rectangle


def _straight(t0, p0, t1, p1, at, threshold):
    """cartopy's straightAndDomain for a line inside the map: does the
    chord p0 -> p1 stand for the curve between t0 and t1? Its middle must
    stay within threshold of the chord, less towards the chord's ends."""
    mx, my = at(0.5 * (t0 + t1))
    sx, sy = p1[0] - p0[0], p1[1] - p0[1]
    dx, dy = mx - p0[0], my - p0[1]
    h2 = sx * sx + sy * sy
    if h2 == 0.0:                                       # a loop (-180 to 180 round a pole)
        return dx == 0.0 and dy == 0.0
    along = (sx * dx + sy * dy) / h2
    if not 0.0 < along < 1.0:
        return False
    return abs(dx * sy - dy * sx) / math.sqrt(h2) <= threshold * 2.0 * (0.5 - abs(0.5 - along))


def _vincenty_terms(u2):
    """Vincenty's A and B of a geodesic, from u^2."""
    return (1.0 + u2 / 16384.0 * (4096.0 + u2 * (-768.0 + u2 * (320.0 - 175.0 * u2))),
            u2 / 1024.0 * (256.0 + u2 * (-128.0 + u2 * (74.0 - 47.0 * u2))))


def _vincenty_series(B, s, c2m, ss, cs):
    """Vincenty's delta sigma."""
    return B * ss * (c2m + B / 4.0 * (cs * (-1.0 + 2.0 * c2m * c2m) - B / 6.0 * c2m * (
        -3.0 + 4.0 * ss * ss) * (-3.0 + 4.0 * c2m * c2m)))


def geodesic_inverse(lon0, lat0, lon1, lat1, a=6378137.0, f=1 / 298.257223563):
    """The geodesic from one point to another on an ellipsoid (degrees;
    Vincenty, as far as the ends aren't antipodal): (start azimuth rad,
    length m)."""
    b = a * (1.0 - f)
    L = np.radians(lon1 - lon0)
    U0, U1 = np.arctan((1.0 - f) * np.tan(np.radians(lat0))), \
        np.arctan((1.0 - f) * np.tan(np.radians(lat1)))
    su0, cu0, su1, cu1 = np.sin(U0), np.cos(U0), np.sin(U1), np.cos(U1)
    lam = L
    with np.errstate(invalid='ignore', divide='ignore'):
        for _ in range(30):
            sl, cl = np.sin(lam), np.cos(lam)
            ss = np.hypot(cu1 * sl, cu0 * su1 - su0 * cu1 * cl)
            cs = su0 * su1 + cu0 * cu1 * cl
            s = np.arctan2(ss, cs)
            sa = cu0 * cu1 * sl / ss
            c2a = 1.0 - sa * sa
            c2m = np.where(c2a != 0.0, cs - 2.0 * su0 * su1 / c2a, 0.0)
            C = f / 16.0 * c2a * (4.0 + f * (4.0 - 3.0 * c2a))
            new = L + (1.0 - C) * f * sa * (s + C * ss * (c2m + C * cs * (-1.0 + 2.0 * c2m * c2m)))
            step, lam = np.abs(new - lam), new
            if not np.nanmax(step, initial=0.0) > 1e-13:
                break
        A, B = _vincenty_terms(c2a * (a * a - b * b) / (b * b))
        sl, cl = np.sin(lam), np.cos(lam)
        azim = np.arctan2(cu1 * sl, cu0 * su1 - su0 * cu1 * cl)
        return azim, np.where(ss == 0.0, 0.0, b * A * (s - _vincenty_series(B, s, c2m, ss, cs)))


def geodesic_direct(lon0, lat0, azim, s12, a=6378137.0, f=1 / 298.257223563):
    """Where a geodesic leaving (lon0, lat0) at azim (rad) is after s12 m:
    (lon, lat), degrees (Vincenty)."""
    b = a * (1.0 - f)
    U0 = np.arctan((1.0 - f) * np.tan(np.radians(lat0)))
    su, cu = np.sin(U0), np.cos(U0)
    sz, cz = np.sin(azim), np.cos(azim)
    s1 = np.arctan2(su, cu * cz)
    sa = cu * sz
    c2a = 1.0 - sa * sa
    A, B = _vincenty_terms(c2a * (a * a - b * b) / (b * b))
    s = s12 / (b * A)
    for _ in range(30):
        c2m = np.cos(2.0 * s1 + s)
        ss, cs = np.sin(s), np.cos(s)
        new = s12 / (b * A) + _vincenty_series(B, s, c2m, ss, cs)
        step, s = np.abs(new - s), new
        if not np.nanmax(step, initial=0.0) > 1e-13:
            break
    ss, cs = np.sin(s), np.cos(s)
    c2m = np.cos(2.0 * s1 + s)
    t = su * ss - cu * cs * cz
    lat = np.arctan2(su * cs + cu * ss * cz, (1.0 - f) * np.hypot(sa, t))
    lam = np.arctan2(ss * sz, cu * cs - su * ss * cz)
    C = f / 16.0 * c2a * (4.0 + f * (4.0 - 3.0 * c2a))
    L = lam - (1.0 - C) * f * sa * (s + C * ss * (c2m + C * cs * (-1.0 + 2.0 * c2m * c2m)))
    return lon0 + np.degrees(L), np.degrees(lat)


def _chords_straight(t0, x0, y0, tc, xc, yc, mx, my, threshold):
    """_straight, of arrays; a chord with no length that its middle stays on
    passes, and so does anything at all, as along is then NaN."""
    sx, sy, dx, dy = xc - x0, yc - y0, mx - x0, my - y0
    h2 = sx * sx + sy * sy
    with np.errstate(invalid='ignore', divide='ignore'):
        along = (sx * dx + sy * dy) / h2
        ok = (along > 0.0) & (along < 1.0)
        ok &= np.abs(dx * sy - dy * sx) / np.sqrt(h2) <= threshold * 2.0 * (0.5 - np.abs(0.5 - along))
    return (ok | np.isnan(along)) & np.isfinite(x0 + y0 + xc + yc)


def geodesic_of(src):
    """The ellipsoid (a, f) a line of src runs on as a geodesic, cartopy's
    SphericalInterpolator; None for any but a Geodetic, whose lines are
    straight in its coordinates."""
    if not isinstance(src, Geodetic):
        return None
    r = float(src.params[0])
    return (6378137.0, 1 / 298.257223563) if r == R_SPHERE else (r, 0.0)


def trace_lines(src, proj, x, y, X, Y, offsets, ellipsoid=False):
    """cartopy's project_linear inside the map (trace.pyx, without the
    domain's edge): polylines (x, y) of src, projected to (X, Y) at their
    vertices, with what a chord needs added wherever a segment bends by
    more than proj.threshold, the points between them on a straight line
    of src or a geodesic of an ellipsoid (by default geodesic_of(src)).
    Only a segment longer than the threshold can bend so (its middle stays
    within half its length of the chord, and it would take a bad
    projection to go past), so most lines cost one test. The segments are
    bisected all at once, a round a transform_points."""
    ellipsoid = geodesic_of(src) if ellipsoid is False else ellipsoid
    found = _bisected(src, proj, x, y, X, Y, offsets, ellipsoid)
    if found is None:
        return X, Y, offsets
    seg, t, px, py = found
    at = np.ascontiguousarray(seg + 1, np.int64)        # new points go before these
    outx, outy = np.empty(X.size + at.size), np.empty(Y.size + at.size)
    _insert_points(np.ascontiguousarray(X, np.float64), np.ascontiguousarray(Y, np.float64), at,
                   np.ascontiguousarray(px, np.float64), np.ascontiguousarray(py, np.float64),
                   outx, outy)
    return outx, outy, offsets + np.searchsorted(at, offsets, side='left')


@njit(cache=True, nogil=True, error_model='numpy')
def _insert_points(X, Y, at, px, py, outx, outy):
    """np.insert of (px, py) into X and Y before the indices at, which are in
    order: the lists merged, one pass for both."""
    i = 0
    for k in range(at.size):
        for j in range(i, at[k]):
            outx[j + k], outy[j + k] = X[j], Y[j]
        i = at[k]
        outx[i + k], outy[i + k] = px[k], py[k]
    k = at.size
    for j in range(i, X.size):
        outx[j + k], outy[j + k] = X[j], Y[j]


def trace_source(src, proj, x, y, offsets):
    """trace_lines's points for shapes that are projected later: the
    polylines (x, y) with them added in src."""
    X, Y = transform_points(src, proj, x, y)
    ellipsoid = geodesic_of(src)
    found = _bisected(src, proj, x, y, X, Y, offsets, ellipsoid)
    if found is None:
        return x, y, offsets
    seg, t, px, py = found
    keep = np.isfinite(px)
    seg, t = seg[keep], t[keep]
    if ellipsoid is not None:
        azim, s12 = geodesic_inverse(x[seg], y[seg], x[seg + 1], y[seg + 1], *ellipsoid)
        u, v = geodesic_direct(x[seg], y[seg], azim, s12 * t, *ellipsoid)
    else:
        u, v = x[seg] + t * (x[seg + 1] - x[seg]), y[seg] + t * (y[seg + 1] - y[seg])
    at = seg + 1
    return (np.insert(x, at, u), np.insert(y, at, v),
            offsets + np.searchsorted(at, offsets, side='left'))


@njit(cache=True, nogil=True, parallel=True, error_model='numpy')
def _long_segments(X, Y, th2, out):
    for i in prange(out.size):
        dx, dy = X[i + 1] - X[i], Y[i + 1] - Y[i]
        out[i] = dx * dx + dy * dy > th2


def _bisected(src, proj, x, y, X, Y, offsets, ellipsoid):
    """The points trace_lines adds, in order: (the segment each is on, its
    fraction of that, its X, Y; NaN where a line breaks), or None."""
    if (src == proj and ellipsoid is None) or not X.size:
        return None
    th = proj.threshold
    long = np.empty(X.size - 1, np.bool_)
    _long_segments(np.ascontiguousarray(X, np.float64), np.ascontiguousarray(Y, np.float64),
                   float(th * th), long)
    long[offsets[1:-1] - 1] = False                     # no segment from a line to the next
    seg = np.flatnonzero(long)
    if not seg.size:
        return None
    ax, ay, bx, by = x[seg], y[seg], x[seg + 1], y[seg + 1]
    key = (src, proj, ellipsoid, seg.tobytes(), ax.tobytes(), ay.tobytes(), bx.tobytes(),
           by.tobytes())
    found = _TRACED.get(key, False)                     # (a figure drawn again: the same lines)
    if found is False:
        found = _bisect_chords(src, proj, X, Y, seg, ax, ay, bx, by, ellipsoid, th)
        if len(_TRACED) >= 4096:
            _TRACED.clear()
        _TRACED[key] = found
    return found


_TRACED = {}


def _bisect_chords(src, proj, X, Y, seg, ax, ay, bx, by, ellipsoid, th):
    if ellipsoid is not None:
        azim, s12 = geodesic_inverse(ax, ay, bx, by, *ellipsoid)

        def at(i, t):
            return transform_points(src, proj, *geodesic_direct(ax[i], ay[i], azim[i], s12[i] * t,
                                                                *ellipsoid))
    else:
        def at(i, t):
            return transform_points(src, proj, ax[i] + t * (bx[i] - ax[i]),
                                    ay[i] + t * (by[i] - ay[i]))
    ex, ey = X[seg + 1], Y[seg + 1]
    n = seg.size
    t0, x0, y0 = np.zeros(n), X[seg].copy(), Y[seg].copy()
    tmin, xmin, ymin = t0.copy(), x0.copy(), y0.copy()
    tmax, xmax, ymax = np.ones(n), ex.copy(), ey.copy()
    tc, xc, yc = np.ones(n), ex.copy(), ey.copy()
    live = np.arange(n)
    out = []
    while live.size:
        i = live
        mx, my = at(i, 0.5 * (t0[i] + tc[i]))
        ok = _chords_straight(t0[i], x0[i], y0[i], tc[i], xc[i], yc[i], mx, my, th)
        k, q = i[ok], i[~ok]
        tmin[k], xmin[k], ymin[k] = tc[k], xc[k], yc[k]
        tmax[q], xmax[q], ymax[q] = tc[q], xc[q], yc[q]
        done = np.abs(tmax[i] - tmin[i]) <= 1e-6
        d, g = i[done], i[~done]
        if g.size:
            tc[g] = 0.5 * (tmin[g] + tmax[g])
            xc[g], yc[g] = at(g, tc[g])
        moved = tmin[d] != t0[d]
        t = np.where(moved, tmin[d], tmax[d])
        px, py = np.where(moved, xmin[d], xmax[d]), np.where(moved, ymin[d], ymax[d])
        if not moved.all():                             # no chord gets past here: a new line
            cut = np.flatnonzero(~moved)
            out.append((d[cut], t[cut] - 1e-9, np.full(cut.size, np.nan),
                        np.full(cut.size, np.nan)))
        on = t < 1.0
        if on.any():
            out.append((d[on], t[on], px[on], py[on]))
        d, t, px, py = d[on], t[on], px[on], py[on]
        t0[d], x0[d], y0[d] = t, px, py
        tmin[d], xmin[d], ymin[d] = t, px, py
        tmax[d], xmax[d], ymax[d] = 1.0, ex[d], ey[d]
        tc[d], xc[d], yc[d] = 1.0, ex[d], ey[d]
        live = np.concatenate([g, d])
    if not out:
        return None
    who = np.concatenate([o[0] for o in out])
    t = np.concatenate([o[1] for o in out])
    px, py = np.concatenate([o[2] for o in out]), np.concatenate([o[3] for o in out])
    order = np.lexsort((t, who))
    return seg[who[order]], t[order], px[order], py[order]


@lru_cache(maxsize=64)
def _traced_bounds(proj, crs, x0, x1, y0, y1):
    """The bounds of the rectangle's outline as cartopy projects it
    (trace.pyx, project_linear): each side followed by chords as long as
    _straight allows, the bounds those chords' ends make -- on a wide conic
    map the corners, not the curved side's true extreme, which can be a
    hundred km further. None if the outline leaves the map."""
    (lx0, lx1), (ly0, ly1) = proj.x_limits, proj.y_limits
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))
    xs, ys = [], []
    for (ax, ay), (bx, by) in zip(corners, corners[1:]):
        def at(t, ax=ax, ay=ay, bx=bx, by=by):
            px, py = transform_points(crs, proj, np.array([ax + t * (bx - ax)]),
                                      np.array([ay + t * (by - ay)]))
            return float(px[0]), float(py[0])
        t, p, end = 0.0, at(0.0), at(1.0)
        for _ in range(4096):
            if not (math.isfinite(p[0]) and math.isfinite(p[1]) and
                    lx0 <= p[0] <= lx1 and ly0 <= p[1] <= ly1):
                return None
            xs.append(p[0])
            ys.append(p[1])
            if t >= 1.0:
                break
            t_min, p_min, t_max, p_max = t, p, 1.0, end
            tc, pc = t_max, p_max
            while abs(t_max - t_min) > 1e-6:
                if math.isfinite(pc[0]) and math.isfinite(pc[1]) and \
                        _straight(t, p, tc, pc, at, proj.threshold):
                    t_min, p_min = tc, pc
                else:
                    t_max, p_max = tc, pc
                tc = 0.5 * (t_min + t_max)
                pc = at(tc)
            if t_min == t:
                return None
            t, p = t_min, p_min
    return min(xs), max(xs), min(ys), max(ys)


def extent_xy(proj, extent, crs=None):
    """(x0, x1, y0, y1): the box in proj's x/y that cartopy's
    GeoAxes.set_extent(extent, crs) sets -- extent = (x0, x1, y0, y1) in crs
    (PlateCarree() by default), the bounds of that rectangle's outline,
    straight in crs, as cartopy projects it (_traced_bounds), or, where the
    outline leaves the map, as projected (densely sampled; parts without an
    image left out). On a cylindrical proj an outline crossing the seam
    spans its whole x range, as cartopy's split lines do. Clipped to proj's
    limits."""
    crs = PlateCarree() if crs is None else crs
    x0, x1, y0, y1 = (float(v) for v in extent)
    if crs == proj and (x0, x1) == proj.x_limits and (y0, y1) == proj.y_limits:
        return x0, x1, y0, y1
    if not proj.cylindrical and not isinstance(crs, Geodetic):
        b = _traced_bounds(proj, crs, x0, x1, y0, y1)
        if b is not None:
            return b
    n = EXTENT_SAMPLES
    ring = np.concatenate([_line(x0, y0, x1, y0, n), _line(x1, y0, x1, y1, n),
                           _line(x1, y1, x0, y1, n), _line(x0, y1, x0, y0, n),
                           [[x0, y0]]])
    px, py = transform_points(crs, proj, ring[:, 0], ring[:, 1])
    ok = np.isfinite(px) & np.isfinite(py)
    if not ok.any():
        raise ValueError('extent %r has no image in %r (x_limits=%r, y_limits=%r)'
                         % (tuple(extent), proj, proj.x_limits, proj.y_limits))
    px, py = px[ok], py[ok]
    (lx0, lx1), (ly0, ly1) = proj.x_limits, proj.y_limits
    if proj.cylindrical:
        per = proj.period
        u = np.unwrap(px, period=per)           # the outline, continuous across the seam
        a, b = u.min(), u.max()
        tol = 1e-9 * per
        k = np.ceil((lx0 - tol - a) / per)
        a, b = a + k * per, b + k * per
        if b - a >= per - tol or b > lx1 + tol:
            bx0, bx1 = lx0, lx1                 # round the world, or across the seam
        else:
            bx0, bx1 = a, b
    else:
        bx0, bx1 = px.min(), px.max()
    by0, by1 = py.min(), py.max()
    return (float(min(max(bx0, lx0), lx1)), float(min(max(bx1, lx0), lx1)),
            float(min(max(by0, ly0), ly1)), float(min(max(by1, ly0), ly1)))


# ---------------------------------------------------------------- CUDA

def _cuda_table(t):
    return ', '.join('%.9ef' % v for v in t.astype(np.float32))


CUDA = r'''
// fastplot._proj: map projections (see the Python module).
// kind codes and params as there; NaN where numba's fwd / inv give NaN.
#define FP_D2R 0.017453292519943295
#define FP_R2D 57.29577951308232
#define FP_PI 3.141592653589793
#define FP_HALFPI 1.5707963267948966
#define FP_FORTPI 0.7853981633974483
#define FP_EPS10 1e-10
#define FP_LIMTOL 1e-12
#define FP_NAN __longlong_as_double(0x7ff8000000000000ll)

__device__ const float fp_rob_x[76] = {@ROBX@};
__device__ const float fp_rob_y[76] = {@ROBY@};

__device__ __forceinline__ double fp_wrap180(double d) {
    if (d < -180.0 || d >= 180.0) {
        d -= 360.0 * floor((d + 180.0) / 360.0);
        if (d >= 180.0) d -= 360.0;
    }
    return d;
}

__device__ __forceinline__ double fp_lon_outd(double lon0, double d) {
    double lon = lon0 + fp_wrap180(d);
    if (lon >= lon0 + 180.0) lon -= 360.0;
    return lon;
}

__device__ __forceinline__ double fp_lon_out(double lon0, double lam) {
    return fp_lon_outd(lon0, lam * FP_R2D);
}

__device__ __forceinline__ double fp_clamp1(double t) {
    return t > 1.0 ? 1.0 : (t < -1.0 ? -1.0 : t);
}

__device__ __forceinline__ void fp_azinv(const double* p, double x, double y, double rho,
                                         double c, double* lam, double* phi) {
    double mode = p[4];
    if (rho == 0.0) { *lam = 0.0; *phi = p[5]; return; }
    if (mode > 0.0) { *lam = atan2(x, -y); *phi = FP_HALFPI - c; return; }
    if (mode < 0.0) { *lam = atan2(x, y); *phi = c - FP_HALFPI; return; }
    double s0 = p[2], c0 = p[3];
    double sc = sin(c), cc = cos(c);
    *phi = asin(fp_clamp1(cc * s0 + y * sc * c0 / rho));
    *lam = atan2(x * sc, rho * c0 * cc - y * s0 * sc);
}

__device__ __forceinline__ double fp_aux(double phi, const double* p, int i) {
    double s = sin(phi + phi), c = cos(phi + phi);
    return phi + s * (p[i] + c * (p[i + 1] + c * (p[i + 2] + c * p[i + 3])));
}

__device__ __forceinline__ double fp_lat_out(const double* p, double phi) {
    if (p[20] != 0.0) phi = fp_aux(phi, p, 20);
    return phi * FP_R2D;
}

__device__ __forceinline__ double fp_rob_v(const float* t, int i, double z) {
    int j = 4 * i;
    return (double)t[j] + z * ((double)t[j + 1] + z * ((double)t[j + 2] + z * (double)t[j + 3]));
}

__device__ __forceinline__ double fp_rob_dv(const float* t, int i, double z) {
    int j = 4 * i;
    return (double)t[j + 1] + 2.0 * z * (double)t[j + 2] + z * z * 3.0 * (double)t[j + 3];
}

__device__ void fp_fwd(int kind, const double* p, double lon, double lat, double* x, double* y) {
    *x = FP_NAN; *y = FP_NAN;
    if (!(lat >= -90.0 && lat <= 90.0) || lon != lon) return;
    if (kind <= 1) { *x = fp_wrap180(lon - p[1]); *y = lat; return; }
    double lam = fp_wrap180(lon - p[1]) * FP_D2R;
    double phi = lat * FP_D2R;
    if (p[16] != 0.0) phi = fp_aux(phi, p, 16);             // an ellipsoid
    double R = p[0];
    if (kind == 2) {                                        // Mercator
        if (lat == 90.0 || lat == -90.0) return;
        *x = p[2] * lam; *y = p[2] * asinh(tan(phi));
        return;
    }
    if (kind == 3) {                                        // Lambert conformal
        double n = p[2], rho;
        if ((n > 0.0 && lat < p[5]) || (n < 0.0 && lat > p[5])) return;
        if (fabs(fabs(phi) - FP_HALFPI) < FP_EPS10) {
            if (phi * n <= 0.0) return;
            rho = 0.0;
        } else {
            rho = p[3] * pow(tan(FP_FORTPI + 0.5 * phi), -n);
        }
        double t = lam * n;
        *x = rho * sin(t); *y = p[4] - rho * cos(t);
        return;
    }
    if (kind == 4 || kind == 5 || kind == 6) {              // azimuthal
        double mode = p[4];
        double sl = sin(lam), cl = cos(lam), rho;
        if (mode == 0.0) {
            double s0 = p[2], c0 = p[3];
            double sp = sin(phi), cp = cos(phi);
            double k;
            if (kind == 5) {
                if (s0 * sp + c0 * cp * cl < -FP_EPS10) return;
                k = R;
            } else {
                double d = 1.0 + s0 * sp + c0 * cp * cl;
                if (d <= FP_EPS10) return;
                if (kind == 6) {
                    k = sqrt(2.0 / d);
                    *x = p[6] * k * cp * sl; *y = p[7] * k * (c0 * sp - s0 * cp * cl);
                    return;
                }
                k = p[6] / d;
            }
            *x = k * cp * sl; *y = k * (c0 * sp - s0 * cp * cl);
            return;
        }
        if (mode < 0.0) { phi = -phi; cl = -cl; }
        if (kind == 4) {
            if (fabs(phi + FP_HALFPI) < 1e-8) return;
            rho = p[6] * tan(FP_FORTPI - 0.5 * phi);
        } else if (kind == 5) {
            if (phi < -FP_EPS10) return;
            rho = R * cos(phi);
        } else {
            if (fabs(phi + FP_HALFPI) < FP_EPS10) return;
            rho = 2.0 * p[6] * sin(FP_FORTPI - 0.5 * phi);
        }
        *x = rho * sl; *y = -rho * cl;
        return;
    }
    if (kind == 7) {                                        // Albers
        double t = p[4] - p[3] * sin(phi);
        if (t < 0.0) return;
        double rho = p[5] * sqrt(t);
        t = lam * p[2];
        *x = R * (rho * sin(t)); *y = R * (p[6] - rho * cos(t));
        return;
    }
    if (kind == 8) {                                        // Robinson
        double dphi = fabs(phi);
        int i = (int)floor(dphi * 11.45915590261646417544 + 1e-15);
        if (i >= 18) i = 18;
        dphi = FP_R2D * (dphi - 0.08726646259971647884 * i);
        double xx = fp_rob_v(fp_rob_x, i, dphi) * 0.8487 * lam;
        double yy = fp_rob_v(fp_rob_y, i, dphi) * 1.3523;
        if (phi < 0.0) yy = -yy;
        *x = R * xx; *y = R * yy;
        return;
    }
    if (kind == 9) {                                        // Miller
        *x = R * lam; *y = R * (log(tan(FP_FORTPI + phi * 0.4)) * 1.25);
        return;
    }
    if (kind == 10) {                                       // rotated pole
        double sp = sin(phi), cp = cos(phi), cl = cos(lam);
        double lr = atan2(cp * sin(lam), p[2] * cp * cl + p[3] * sp) + p[4];
        double pr = asin(fp_clamp1(p[2] * sp - p[3] * cp * cl));
        *x = fp_wrap180(lr * FP_R2D); *y = pr * FP_R2D;
        return;
    }
}

__device__ void fp_inv(int kind, const double* p, double x, double y, double* lon, double* lat) {
    *lon = FP_NAN; *lat = FP_NAN;
    if (kind <= 1) {
        if (!(y >= -90.0 && y <= 90.0) || !isfinite(x)) return;
        *lon = fp_lon_outd(p[1], x); *lat = y;
        return;
    }
    double R = p[0];
    if (kind == 2) {
        double a = p[2];
        if (!(fabs(x) <= p[5] && y >= p[3] && y <= p[4])) return;
        *lon = fp_lon_out(p[1], x / a); *lat = fp_lat_out(p, atan(sinh(y / a)));
        return;
    }
    if (kind == 3) {
        double n = p[2], dx = x, dy = p[4] - y, lam, phi;
        double rho = hypot(dx, dy);
        if (rho == 0.0) {
            lam = 0.0; phi = n > 0.0 ? FP_HALFPI : -FP_HALFPI;
        } else {
            if (n < 0.0) { rho = -rho; dx = -dx; dy = -dy; }
            phi = 2.0 * atan(pow(p[3] / rho, 1.0 / n)) - FP_HALFPI;
            lam = atan2(dx, dy) / n;
            if (!(fabs(lam) <= FP_PI * (1.0 + FP_LIMTOL))) return;
        }
        double la = fp_lat_out(p, phi);
        if ((n > 0.0 && la < p[5] - 1e-9) || (n < 0.0 && la > p[5] + 1e-9) || !(fabs(la) <= 90.0)) return;
        *lon = fp_lon_out(p[1], lam); *lat = la;
        return;
    }
    if (kind == 4 || kind == 5 || kind == 6) {
        double rq = p[8];
        if (kind == 6) { x *= rq / p[6]; y *= rq / p[7]; }
        double rho = hypot(x, y), c, lam, phi;
        if (kind == 4) {
            double u = x / p[7], v = y / p[8];
            if (!(u * u + v * v <= 1.0 + FP_LIMTOL)) return;
            c = 2.0 * atan(rho / p[6]);
        } else if (kind == 5) {
            double s = rho / R;
            if (!(s <= 1.0 + FP_EPS10)) return;
            c = asin(fp_clamp1(s));
        } else {
            double s = 0.5 * rho / rq;
            if (!(s <= 1.0)) return;
            c = 2.0 * asin(s);
        }
        fp_azinv(p, x, y, rho, c, &lam, &phi);
        *lon = fp_lon_out(p[1], lam); *lat = fp_lat_out(p, phi);
        return;
    }
    if (kind == 7) {
        double n = p[2], dx = x / R, dy = p[6] - y / R, lam, phi;
        double rho = hypot(dx, dy);
        if (n < 0.0) { rho = -rho; dx = -dx; dy = -dy; }
        double q = rho / p[5];
        double s = (p[4] - q * q) / p[3];
        if (!(fabs(s) <= 1.0 + FP_LIMTOL)) return;
        phi = asin(fp_clamp1(s));
        lam = atan2(dx, dy) / n;
        if (!(fabs(lam) <= FP_PI * (1.0 + FP_LIMTOL))) return;
        *lon = fp_lon_out(p[1], lam); *lat = fp_lat_out(p, phi);
        return;
    }
    if (kind == 8) {
        double lam = x / R / 0.8487;
        double phi = fabs(y / R / 1.3523);
        if (!(phi < 1.0)) {
            if (!(phi <= 1.000001)) return;
            phi = y > 0.0 ? FP_HALFPI : -FP_HALFPI;
            lam /= (double)fp_rob_x[4 * 18];
        } else {
            int i = (int)floor(phi * 18);
            for (;;) {
                if ((double)fp_rob_y[4 * i] > phi) --i;
                else if ((double)fp_rob_y[4 * i + 4] <= phi) ++i;
                else break;
            }
            double t = 5.0 * (phi - (double)fp_rob_y[4 * i])
                       / ((double)fp_rob_y[4 * i + 4] - (double)fp_rob_y[4 * i]);
            for (int it = 0; it < 100; ++it) {
                double t1 = (fp_rob_v(fp_rob_y, i, t) - phi) / fp_rob_dv(fp_rob_y, i, t);
                t -= t1;
                if (fabs(t1) < 1e-10) break;
            }
            phi = (5.0 * i + t) * FP_D2R;
            if (y < 0.0) phi = -phi;
            lam /= fp_rob_v(fp_rob_x, i, t);
        }
        if (!(fabs(lam) <= FP_PI * (1.0 + FP_LIMTOL))) return;
        *lon = fp_lon_out(p[1], lam); *lat = phi * FP_R2D;
        return;
    }
    if (kind == 9) {
        if (!(fabs(x) <= p[2] && fabs(y) <= p[3])) return;
        *lon = fp_lon_out(p[1], x / R);
        *lat = 2.5 * (atan(exp(0.8 * (y / R))) - FP_FORTPI) * FP_R2D;
        return;
    }
    if (kind == 10) {
        if (!(y >= -90.0 && y <= 90.0) || !isfinite(x)) return;
        double lr = x * FP_D2R - p[4], pr = y * FP_D2R;
        double sp = sin(pr), cp = cos(pr), cl = cos(lr);
        double phi = asin(fp_clamp1(p[2] * sp + p[3] * cp * cl));
        double lam = atan2(cp * sin(lr), p[2] * cp * cl - p[3] * sp);
        *lon = fp_wrap180(lam * FP_R2D + p[1]); *lat = phi * FP_R2D;
        return;
    }
}
'''.replace('@ROBX@', _cuda_table(_ROB_X)).replace('@ROBY@', _cuda_table(_ROB_Y))
