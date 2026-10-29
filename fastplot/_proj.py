"""Map projections on a sphere, for the per-pixel and per-vertex loops of
the renderer: numba scalar functions, the same math as CUDA C, and the
classes users build maps with (cartopy's names and keywords).

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
`params`, 16 float64 holding everything its formulas need, computed once
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

The Earth is a sphere of radius `radius` (6370997 m, PROJ's 'sphere',
cartopy's `globe` dropped): at map scale the ellipsoid moves things by
under 0.2 % of the map, and spherical formulas are short and invertible
in closed form (Robinson's inverse is the one Newton iteration, PROJ's).
The formulas are PROJ's spherical ones, mostly in PROJ's order of
operations, so `pyproj` with `.proj4` agrees to ~1e-8 m forward and
~1e-13 deg inverse (less only within a degree of an azimuthal antipode,
where 1 + cos c cancels).

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

import numpy as np
from ._jit import njit, prange

R_SPHERE = 6370997.0            # PROJ's +ellps=sphere radius, metres
NPARAM = 16                     # params of every projection
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
                k = p[6] / d if kind == STEREOGRAPHIC else R * math.sqrt(2.0 / d)
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
            rho = 2.0 * R * math.sin(FORTPI - 0.5 * phi)
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
        return _lon_out(p[1], x / a), math.atan(math.sinh(y / a)) * R2D
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
        lat = phi * R2D
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
        return _lon_out(p[1], lam), phi * R2D
    if kind == ORTHOGRAPHIC:
        rho = math.hypot(x, y)
        s = rho / R
        if not s <= 1.0 + EPS10:
            return NAN, NAN
        lam, phi = _azinv(p, x, y, rho, math.asin(_clamp1(s)))
        return _lon_out(p[1], lam), phi * R2D
    if kind == LAMBERT_AZIMUTHAL_EQUAL_AREA:
        rho = math.hypot(x, y)
        s = 0.5 * rho / R
        if not s <= 1.0:
            return NAN, NAN
        lam, phi = _azinv(p, x, y, rho, 2.0 * math.asin(s))
        return _lon_out(p[1], lam), phi * R2D
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
        return _lon_out(p[1], lam), phi * R2D
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

    def _setup(self, params, args, proj4, xl, yl, boundary):
        p = np.zeros(NPARAM)
        p[:len(params)] = params
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


class Geodetic(Projection):
    """Longitude / latitude degrees: x = lon wrapped into [-180, 180),
    y = lat. The crs of lon/lat data (transform=Geodetic()); any x is
    accepted on inverse (it wraps)."""

    kind = GEODETIC
    cylindrical = True
    period = 360.0

    def __init__(self, radius=R_SPHERE):
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

    def __init__(self, central_longitude=0.0, radius=R_SPHERE):
        r = _radius(radius)
        c = float(central_longitude)
        self.central_longitude = c
        args = [('central_longitude', c)] + ([('radius', r)] if r != R_SPHERE else [])
        self._setup([r, c], args,
                    '+proj=eqc +lon_0=%s +R=%s +to_meter=%s +no_defs'
                    % (_fmt(c), _fmt(r), _fmt(r * D2R)),
                    (-180.0, 180.0), (-90.0, 90.0), _rect(-180.0, 180.0, -90.0, 90.0))


class Mercator(Projection):
    """Mercator, spherical: x = R k0 (lon - lon0), y = R k0 asinh(tan lat);
    k0 = cos(latitude_true_scale) or scale_factor. Drawn between min_ and
    max_latitude (its limits; inverse is NaN beyond them), defined forward
    for any |lat| < 90."""

    kind = MERCATOR
    cylindrical = True

    def __init__(self, central_longitude=0.0, min_latitude=-80.0, max_latitude=84.0,
                 latitude_true_scale=None, scale_factor=None, radius=R_SPHERE):
        r = _radius(radius)
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
            k0 = math.cos(float(latitude_true_scale) * D2R)
            p4 += ' +lat_ts=%s' % _fmt(latitude_true_scale)
        else:
            k0 = 1.0
        a = r * k0
        y0 = a * math.asinh(math.tan(lo * D2R))
        y1 = a * math.asinh(math.tan(hi * D2R))
        xm = math.pi * a
        self.central_longitude = c
        self.period = 2.0 * xm
        args = [('central_longitude', c), ('min_latitude', lo), ('max_latitude', hi)]
        if latitude_true_scale is not None:
            args.append(('latitude_true_scale', float(latitude_true_scale)))
        if scale_factor is not None:
            args.append(('scale_factor', float(scale_factor)))
        if r != R_SPHERE:
            args.append(('radius', r))
        self._setup([r, c, a, y0, y1, xm], args, p4 + ' +R=%s +units=m +no_defs' % _fmt(r),
                    (-xm, xm), (y0, y1), _rect(-xm, xm, y0, y1))


class LambertConformal(Projection):
    """Lambert conformal conic, spherical, one or two standard parallels;
    the cone's apex at the pole the parallels lean to (a southern cone for
    southern parallels). The map stops at `cutoff` latitude (forward and
    inverse NaN beyond it); the wedge outside the cone's unrolled 360 deg is
    NaN on inverse."""

    kind = LAMBERT_CONFORMAL
    seam = True

    def __init__(self, central_longitude=-96.0, central_latitude=39.0,
                 standard_parallels=(33.0, 45.0), cutoff=-30.0, radius=R_SPHERE):
        r = _radius(radius)
        c, lat0, cut = float(central_longitude), float(central_latitude), float(cutoff)
        sp = np.atleast_1d(np.asarray(standard_parallels, dtype=np.float64))
        if not 1 <= sp.size <= 2:
            raise ValueError('1 or 2 standard parallels, got %r' % (standard_parallels,))
        lat1 = float(sp[0])
        lat2 = float(sp[-1])
        phi1, phi2, phi0 = lat1 * D2R, lat2 * D2R, lat0 * D2R
        if abs(phi1 + phi2) < EPS10:
            raise ValueError('standard parallels symmetric about the equator: no cone')
        n = math.sin(phi1)
        cosphi = math.cos(phi1)
        if abs(phi1 - phi2) >= EPS10:           # secant (PROJ's lcc, e = 0)
            n = math.log(cosphi / math.cos(phi2)) / math.log(
                math.tan(FORTPI + 0.5 * phi2) / math.tan(FORTPI + 0.5 * phi1))
        cc = cosphi * math.pow(math.tan(FORTPI + 0.5 * phi1), n) / n
        rho0 = 0.0 if abs(abs(phi0) - HALFPI) < EPS10 else cc * math.pow(
            math.tan(FORTPI + 0.5 * phi0), -n)
        RC, Rrho0 = r * cc, r * rho0
        if (n > 0 and cut >= 90) or (n < 0 and cut <= -90):
            raise ValueError('cutoff %r leaves nothing of the cone' % cutoff)
        self.central_longitude = c
        self.cutoff = cut
        self.lat_limits = (cut, 90.0) if n > 0 else (-90.0, cut)
        # the cutoff arc, apex to apex
        lam = np.linspace(-math.pi, math.pi, 721)
        phic = cut * D2R
        rc = RC * math.pow(math.tan(FORTPI + 0.5 * phic), -n)
        arc = np.stack([rc * np.sin(n * lam), Rrho0 - rc * np.cos(n * lam)], axis=1)
        apex = np.array([[0.0, Rrho0]])
        ring = _ccw(np.concatenate([apex, arc, apex]))
        args = [('central_longitude', c), ('central_latitude', lat0),
                ('standard_parallels', tuple(float(v) for v in sp)), ('cutoff', cut)]
        if r != R_SPHERE:
            args.append(('radius', r))
        p4 = ('+proj=lcc +lon_0=%s +lat_0=%s +lat_1=%s +lat_2=%s +R=%s +units=m +no_defs'
              % (_fmt(c), _fmt(lat0), _fmt(lat1), _fmt(lat2), _fmt(r)))
        self._setup([r, c, n, RC, Rrho0, cut], args, p4,
                    (ring[:, 0].min(), ring[:, 0].max()),
                    (ring[:, 1].min(), ring[:, 1].max()), ring)


class Stereographic(Projection):
    """Stereographic, spherical, any aspect. true_scale_latitude (polar
    aspects only, as in PROJ) or scale_factor sets the scale. Drawn within
    cartopy's ellipse of +-5e7 m (scaled to the sphere); inverse is NaN
    beyond it, forward only at the antipode."""

    kind = STEREOGRAPHIC

    def __init__(self, central_latitude=0.0, central_longitude=0.0,
                 true_scale_latitude=None, scale_factor=None, radius=R_SPHERE):
        r = _radius(radius)
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
        akm1 = 2.0 * k0
        if mode != 0.0 and true_scale_latitude is not None:
            phits = abs(float(true_scale_latitude) * D2R)
            if abs(phits - HALFPI) >= EPS10:
                akm1 = math.cos(phits) / math.tan(FORTPI - 0.5 * phits)
        if mode == 0.0:
            s0, c0 = math.sin(phi0), math.cos(phi0)
        else:
            s0, c0 = mode, 0.0
        xl = r * (5e7 / 6378137.0)              # cartopy's limits, WGS84's 5e7 m scaled
        yl = r * (5e7 / 6356752.3142)
        self.central_longitude = c
        self.central_latitude = lat0
        args = self._names(c, lat0, true_scale_latitude, scale_factor)
        if r != R_SPHERE:
            args.append(('radius', r))
        self._setup([r, c, s0, c0, mode, phi0, akm1 * r, xl, yl], args,
                    p4 + ' +R=%s +units=m +no_defs' % _fmt(r),
                    (-xl, xl), (-yl, yl), _ellipse(xl, yl))

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

    def __init__(self, central_longitude=0.0, true_scale_latitude=None, radius=R_SPHERE):
        super().__init__(90.0, central_longitude, true_scale_latitude, None, radius)
        self._args = [kv for kv in self._args if kv[0] != 'central_latitude']


class SouthPolarStereo(Stereographic):
    """Stereographic centred on the south pole."""

    def __init__(self, central_longitude=0.0, true_scale_latitude=None, radius=R_SPHERE):
        super().__init__(-90.0, central_longitude, true_scale_latitude, None, radius)
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
    """Orthographic (the globe seen from infinitely far), spherical. The far
    hemisphere is NaN forward; the disc, as cartopy's, is 0.99999 R."""

    kind = ORTHOGRAPHIC

    def __init__(self, central_longitude=0.0, central_latitude=0.0, radius=R_SPHERE):
        r = _radius(radius)
        c, lat0 = float(central_longitude), float(central_latitude)
        s0, c0, mode, phi0 = _azimuthal(lat0)
        a = r * 0.99999
        self.central_longitude = c
        self.central_latitude = lat0
        args = [('central_longitude', c), ('central_latitude', lat0)]
        if r != R_SPHERE:
            args.append(('radius', r))
        self._setup([r, c, s0, c0, mode, phi0], args,
                    '+proj=ortho +lon_0=%s +lat_0=%s +R=%s +units=m +no_defs'
                    % (_fmt(c), _fmt(lat0), _fmt(r)),
                    (-a, a), (-a, a), _ellipse(a, a))


class LambertAzimuthalEqualArea(Projection):
    """Lambert azimuthal equal-area, spherical: the whole sphere in a disc
    of radius 2R, its antipode (the rim) NaN forward. Limits as cartopy's
    (1.9999 R wide, the antipode less 0.01 deg high)."""

    kind = LAMBERT_AZIMUTHAL_EQUAL_AREA

    def __init__(self, central_longitude=0.0, central_latitude=0.0, radius=R_SPHERE):
        r = _radius(radius)
        c, lat0 = float(central_longitude), float(central_latitude)
        s0, c0, mode, phi0 = _azimuthal(lat0)
        a = r * 1.9999
        b = 2.0 * r * math.sin(0.5 * (180.0 - 0.01) * D2R)
        self.central_longitude = c
        self.central_latitude = lat0
        args = [('central_longitude', c), ('central_latitude', lat0)]
        if r != R_SPHERE:
            args.append(('radius', r))
        self._setup([r, c, s0, c0, mode, phi0], args,
                    '+proj=laea +lon_0=%s +lat_0=%s +R=%s +units=m +no_defs'
                    % (_fmt(c), _fmt(lat0), _fmt(r)),
                    (-a, a), (-b, b), _ellipse(a, b))


class AlbersEqualArea(Projection):
    """Albers equal-area conic, spherical, one or two standard parallels:
    the whole sphere, an annulus sector between the images of the poles."""

    kind = ALBERS_EQUAL_AREA
    seam = True

    def __init__(self, central_longitude=0.0, central_latitude=0.0,
                 standard_parallels=(20.0, 50.0), radius=R_SPHERE):
        r = _radius(radius)
        c, lat0 = float(central_longitude), float(central_latitude)
        sp = np.atleast_1d(np.asarray(standard_parallels, dtype=np.float64))
        if not 1 <= sp.size <= 2:
            raise ValueError('1 or 2 standard parallels, got %r' % (standard_parallels,))
        lat1, lat2 = float(sp[0]), float(sp[-1])
        phi1, phi2, phi0 = lat1 * D2R, lat2 * D2R, lat0 * D2R
        if abs(phi1 + phi2) < EPS10:
            raise ValueError('standard parallels symmetric about the equator: no cone')
        n = sinphi = math.sin(phi1)
        cosphi = math.cos(phi1)
        if abs(phi1 - phi2) >= EPS10:
            n = 0.5 * (n + math.sin(phi2))
        n2 = n + n
        cc = cosphi * cosphi + n2 * sinphi
        dd = 1.0 / n
        rho0 = dd * math.sqrt(cc - n2 * math.sin(phi0))
        lam = np.linspace(-math.pi, math.pi, 361)
        rn = r * dd * math.sqrt(max(cc - n2, 0.0))
        rs = r * dd * math.sqrt(cc + n2)
        north = np.stack([rn * np.sin(n * lam), r * rho0 - rn * np.cos(n * lam)], axis=1)
        south = np.stack([rs * np.sin(n * lam), r * rho0 - rs * np.cos(n * lam)], axis=1)
        ring = _ccw(np.concatenate([north, south[::-1], north[:1]]))
        self.central_longitude = c
        args = [('central_longitude', c), ('central_latitude', lat0),
                ('standard_parallels', tuple(float(v) for v in sp))]
        if r != R_SPHERE:
            args.append(('radius', r))
        p4 = ('+proj=aea +lon_0=%s +lat_0=%s +lat_1=%s +lat_2=%s +R=%s +units=m +no_defs'
              % (_fmt(c), _fmt(lat0), _fmt(lat1), _fmt(lat2), _fmt(r)))
        self._setup([r, c, n, n2, cc, dd, rho0], args, p4,
                    (ring[:, 0].min(), ring[:, 0].max()),
                    (ring[:, 1].min(), ring[:, 1].max()), ring)


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

    def __init__(self, central_longitude=0.0, radius=R_SPHERE):
        r = _radius(radius)
        c = float(central_longitude)
        lat = np.linspace(-90.0, 90.0, 361)
        X, Y = _rob_np(lat)
        x = r * X * _ROB_FXC * math.pi
        y = r * Y * _ROB_FYC * np.sign(lat)
        right = np.stack([x, y], axis=1)
        left = np.stack([-x, y], axis=1)[::-1]
        ring = _ccw(np.concatenate([right, left, right[:1]]))
        self.central_longitude = c
        args = [('central_longitude', c)] + ([('radius', r)] if r != R_SPHERE else [])
        self._setup([r, c], args, '+proj=robin +lon_0=%s +R=%s +units=m +no_defs'
                    % (_fmt(c), _fmt(r)),
                    (ring[:, 0].min(), ring[:, 0].max()),
                    (ring[:, 1].min(), ring[:, 1].max()), ring)


class Miller(Projection):
    """Miller cylindrical, spherical, in metres (cartopy's Miller defaults
    to a sphere of radius 6378137 m; fastplot's 6370997 m)."""

    kind = MILLER
    cylindrical = True

    def __init__(self, central_longitude=0.0, radius=R_SPHERE):
        r = _radius(radius)
        c = float(central_longitude)
        xm, ym = math.pi * r, _MILL_YMAX * r
        self.central_longitude = c
        self.period = 2.0 * xm
        args = [('central_longitude', c)] + ([('radius', r)] if r != R_SPHERE else [])
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
                 central_rotated_longitude=0.0, radius=R_SPHERE):
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


def extent_xy(proj, extent, crs=None):
    """(x0, x1, y0, y1): the box in proj's x/y that cartopy's
    GeoAxes.set_extent(extent, crs) sets -- extent = (x0, x1, y0, y1) in crs
    (PlateCarree() by default), the bounds of that rectangle's outline,
    straight in crs, as projected (densely sampled; parts without an image
    left out). On a cylindrical proj an outline crossing the seam spans its
    whole x range, as cartopy's split lines do. Clipped to proj's limits."""
    crs = PlateCarree() if crs is None else crs
    x0, x1, y0, y1 = (float(v) for v in extent)
    if crs == proj and (x0, x1) == proj.x_limits and (y0, y1) == proj.y_limits:
        return x0, x1, y0, y1
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
// fastplot._proj: map projections on a sphere (see the Python module).
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
                k = kind == 4 ? p[6] / d : R * sqrt(2.0 / d);
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
            rho = 2.0 * R * sin(FP_FORTPI - 0.5 * phi);
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
        *lon = fp_lon_out(p[1], x / a); *lat = atan(sinh(y / a)) * FP_R2D;
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
        double la = phi * FP_R2D;
        if ((n > 0.0 && la < p[5] - 1e-9) || (n < 0.0 && la > p[5] + 1e-9) || !(fabs(la) <= 90.0)) return;
        *lon = fp_lon_out(p[1], lam); *lat = la;
        return;
    }
    if (kind == 4 || kind == 5 || kind == 6) {
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
            double s = 0.5 * rho / R;
            if (!(s <= 1.0)) return;
            c = 2.0 * asin(s);
        }
        fp_azinv(p, x, y, rho, c, &lam, &phi);
        *lon = fp_lon_out(p[1], lam); *lat = phi * FP_R2D;
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
        *lon = fp_lon_out(p[1], lam); *lat = phi * FP_R2D;
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
