"""fastplot._proj against PROJ (pyproj, from each projection's .proj4) and
cartopy: forward, inverse, round trips, domains, boundaries, extents, the
scalar fwd / inv inside other numba kernels, and the CUDA C on the GPU."""
import math
import warnings

import numpy as np
import pytest

from fastplot import _jit, _proj as P

pyproj = pytest.importorskip('pyproj')

R = P.R_SPHERE
GEO = '+proj=longlat +R=%r +no_defs' % R


def _projections():
    return [
        P.Mercator(),
        P.Mercator(central_longitude=150, min_latitude=-60, max_latitude=70,
                   latitude_true_scale=30),
        P.Mercator(central_longitude=-75, scale_factor=0.9),
        P.LambertConformal(),
        P.LambertConformal(central_longitude=-97.5, central_latitude=38.5,
                           standard_parallels=(38.5,)),              # HRRR
        P.LambertConformal(central_longitude=140, central_latitude=-35,
                           standard_parallels=(-30, -50), cutoff=10),  # southern cone
        P.LambertConformal(central_longitude=10, central_latitude=60,
                           standard_parallels=(55, 65), cutoff=0),
        P.Stereographic(central_latitude=45, central_longitude=10),
        P.Stereographic(),
        P.Stereographic(central_latitude=-30, central_longitude=-60, scale_factor=0.95),
        P.NorthPolarStereo(central_longitude=-45),
        P.NorthPolarStereo(true_scale_latitude=60),
        P.SouthPolarStereo(true_scale_latitude=-71),
        P.SouthPolarStereo(central_longitude=180),
        P.Orthographic(-100, 40),
        P.Orthographic(),
        P.Orthographic(0, 90),
        P.Orthographic(30, -90),
        P.LambertAzimuthalEqualArea(10, 52),
        P.LambertAzimuthalEqualArea(),
        P.LambertAzimuthalEqualArea(0, 90),
        P.LambertAzimuthalEqualArea(-150, -90),
        P.AlbersEqualArea(),
        P.AlbersEqualArea(-96, 23, (29.5, 45.5)),
        P.AlbersEqualArea(130, 0, (-18, -36)),
        P.AlbersEqualArea(0, 0, (40,)),
        P.Robinson(),
        P.Robinson(central_longitude=150),
        P.Miller(),
        P.Miller(central_longitude=-60),
        P.PlateCarree(),
        P.PlateCarree(central_longitude=180),
        P.PlateCarree(central_longitude=-30.5),
        P.RotatedPole(),
        P.RotatedPole(pole_longitude=-170, pole_latitude=40),          # COSMO-EU-like
        P.RotatedPole(pole_longitude=10, pole_latitude=30, central_rotated_longitude=-15),
        P.Geodetic(),
    ]


PROJS = _projections()
IDS = [repr(p) for p in PROJS]
LENIENT = (P.PLATE_CARREE, P.GEODETIC, P.ROTATED_POLE)   # x wraps on inverse


def _lonlat(n, seed=0):
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-180, 180, n)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    return lon, lat


def _finite(a):
    a = np.asarray(a, dtype=np.float64)
    return np.where(np.isfinite(a) & (np.abs(a) < 1e30), a, np.nan)


def _pyproj(src, dst, a, b):
    t = pyproj.Transformer.from_crs(src, dst, always_xy=True)
    with np.errstate(all='ignore'):
        x, y = t.transform(a, b, errcheck=False)
    return _finite(x), _finite(y)


def _angle(lon1, lat1, lon2, lat2):
    """Great-circle distance, degrees (stable at small angles)."""
    l1, p1, l2, p2 = (np.radians(v) for v in (lon1, lat1, lon2, lat2))
    h = np.sin(0.5 * (p2 - p1)) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(0.5 * (l2 - l1)) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(h, 0, 1))))


def _dx(proj, a, b):
    d = np.abs(a - b)
    if proj.cylindrical:
        d = np.minimum(d, np.abs(d - proj.period))      # the seam: +-180 alike
    return d


def _centre_distance(proj, lon, lat):
    c = getattr(proj, 'central_latitude', 0.0)
    return _angle(lon, lat, proj.central_longitude, c)


def _xy_tol(proj, x, y):
    """1e-3 m (1e-9 deg for degree crs), and 1e-13 of far-out coordinates
    (a stereographic near its antipode)."""
    base = 1e-9 if proj.kind in LENIENT else 1e-3
    return base + 1e-13 * (np.abs(x) + np.abs(y))


# ---------------------------------------------------------------- forward / inverse

@pytest.mark.parametrize('proj', PROJS, ids=IDS)
def test_forward_vs_pyproj(proj):
    lon, lat = _lonlat(100000)
    x, y = proj.forward(lon, lat)
    xr, yr = _pyproj(GEO, proj.proj4, lon, lat)
    if proj.kind == P.LAMBERT_CONFORMAL:                # PROJ has no cutoff
        n = proj.params[2]
        beyond = lat < proj.cutoff if n > 0 else lat > proj.cutoff
        assert np.isnan(x[beyond]).all() and np.isnan(y[beyond]).all()
        assert np.isfinite(x[~beyond]).all()
        xr[beyond] = yr[beyond] = np.nan
    np.testing.assert_array_equal(np.isnan(x), np.isnan(xr))
    np.testing.assert_array_equal(np.isnan(y), np.isnan(x))
    ok = np.isfinite(x)
    assert ok.sum() > 30000
    if proj.kind in (P.STEREOGRAPHIC, P.LAMBERT_AZIMUTHAL_EQUAL_AREA):
        ok &= _centre_distance(proj, lon, lat) < 179     # (1 + cos c ~ 0: libm's last bits)
    tol = _xy_tol(proj, xr[ok], yr[ok])
    assert (_dx(proj, x[ok], xr[ok]) <= tol).all()
    assert (np.abs(y[ok] - yr[ok]) <= tol).all()


def _inside_far(proj, x, y, margin):
    """(inside the boundary ring, farther than margin from it) of points."""
    from matplotlib.path import Path
    ring = proj.boundary
    inside = Path(ring).contains_points(np.stack([x, y], axis=1))
    a, b = ring[:-1], ring[1:]
    far = np.ones(x.size, bool)
    for s in range(0, x.size, 2000):
        px, py = x[s:s + 2000, None], y[s:s + 2000, None]
        ex, ey = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
        t = np.clip(((px - a[:, 0]) * ex + (py - a[:, 1]) * ey) / (ex * ex + ey * ey + 1e-300), 0, 1)
        d = np.hypot(px - a[:, 0] - t * ex, py - a[:, 1] - t * ey).min(axis=1)
        far[s:s + 2000] = d > margin
    return inside, far


@pytest.mark.parametrize('proj', PROJS, ids=IDS)
def test_inverse_vs_pyproj(proj):
    rng = np.random.default_rng(1)
    (x0, x1), (y0, y1) = proj.x_limits, proj.y_limits
    w, h = x1 - x0, y1 - y0
    x = rng.uniform(x0 - 0.1 * w, x1 + 0.1 * w, 20000)
    y = rng.uniform(y0 - 0.1 * h, y1 + 0.1 * h, 20000)
    lon, lat = proj.inverse(x, y)
    np.testing.assert_array_equal(np.isnan(lon), np.isnan(lat))
    ok = np.isfinite(lon)
    assert ok.sum() > 5000
    c = proj.central_longitude
    assert (lon[ok] >= c - 180).all() and (lon[ok] < c + 180).all()
    assert (np.abs(lat[ok]) <= 90).all()
    lonr, latr = _pyproj(proj.proj4, GEO, x[ok], y[ok])
    assert np.isfinite(lonr).all()
    assert _angle(lon[ok], lat[ok], lonr, latr).max() < 1e-6
    # NaN exactly outside the domain the boundary rings (away from the ring itself)
    if proj.kind in LENIENT:
        assert np.isfinite(lon[np.abs(y) <= 90]).all()
        assert np.isnan(lon[np.abs(y) > 90]).all()
    else:
        inside, far = _inside_far(proj, x, y, 1e-4 * max(w, h))
        np.testing.assert_array_equal(ok[far], inside[far])


@pytest.mark.parametrize('proj', PROJS, ids=IDS)
def test_round_trip(proj):
    lon, lat = _lonlat(100000, seed=2)
    x, y = proj.forward(lon, lat)
    lon2, lat2 = proj.inverse(x, y)
    ok = np.isfinite(lon2)
    # finite forward images inside the domain come back
    if proj.kind in (P.STEREOGRAPHIC, P.LAMBERT_CONFORMAL, P.MERCATOR):
        inside = np.isfinite(x)                          # (Mercator / stereo: within limits)
        if proj.kind == P.MERCATOR:
            inside &= (y >= proj.y_limits[0]) & (y <= proj.y_limits[1])
        if proj.kind == P.STEREOGRAPHIC:
            inside &= (x / proj.x_limits[1]) ** 2 + (y / proj.y_limits[1]) ** 2 < 1
    else:
        inside = np.isfinite(x)
    np.testing.assert_array_equal(ok, inside)
    err = _angle(lon[ok], lat[ok], lon2[ok], lat2[ok])
    # ill-conditioned next to an antipode / the orthographic limb
    far = _centre_distance(proj, lon[ok], lat[ok])
    if proj.kind in (P.LAMBERT_AZIMUTHAL_EQUAL_AREA, P.STEREOGRAPHIC):
        err = err[far < 179]
    if proj.kind == P.ORTHOGRAPHIC:
        err = err[far < 89.9]
    assert err.max() < 1e-8                              # 1 mm


def test_plate_carree_rotated_pole_vs_cartopy():
    ccrs = pytest.importorskip('cartopy.crs')
    lon, lat = _lonlat(20000, seed=3)
    for c in (0, 180, -30.5, 97):
        r = ccrs.PlateCarree(central_longitude=c).transform_points(ccrs.Geodetic(), lon, lat)
        x, y = P.PlateCarree(c).forward(lon, lat)
        assert (np.minimum(np.abs(x - r[:, 0]), 360 - np.abs(x - r[:, 0])) < 1e-9).all()
        assert (np.abs(y - r[:, 1]) < 1e-12).all()
        lon2, lat2 = P.PlateCarree(c).inverse(x, y)
        assert (np.abs((lon2 - lon + 180) % 360 - 180) < 1e-9).all()
    for args in [(0, 90, 0), (-170, 40, 0), (-123.34, 79.95, 0), (10, 30, -15), (193, 41, 7)]:
        cp, fp = ccrs.RotatedPole(*args), P.RotatedPole(*args)
        r = cp.transform_points(ccrs.Geodetic(), lon, lat)
        x, y = fp.forward(lon, lat)
        assert ((x >= -180) & (x < 180)).all()
        assert (np.abs((x - r[:, 0] + 180) % 360 - 180) < 1e-9).all()
        assert (np.abs(y - r[:, 1]) < 1e-9).all()
        g = ccrs.Geodetic().transform_points(cp, r[:, 0], r[:, 1])
        lon2, lat2 = fp.inverse(r[:, 0], r[:, 1])
        assert ((lon2 >= -180) & (lon2 < 180)).all()
        assert _angle(lon2, lat2, g[:, 0], g[:, 1]).max() < 1e-9


def test_plate_carree_wraps_any_longitude():
    p = P.PlateCarree()
    lon, lat = p.inverse([0.0, 200.0, 359.75, -190.0, 540.0], [0.0, 10.0, -10.0, 5.0, 90.0])
    np.testing.assert_allclose(lon, [0, -160, -0.25, 170, -180], atol=1e-12)
    x, y = P.PlateCarree(180).forward([0.0, 179.0, 181.0, 360.0], 0.0)
    np.testing.assert_allclose(x, [-180, -1, 1, -180], atol=1e-12)
    x, y = P.Geodetic().forward([190.0, -180.0, 180.0], [0, 0, 0])
    np.testing.assert_allclose(x, [-170, -180, -180])


# ---------------------------------------------------------------- NaN domains

def test_nan_where_no_image():
    nan = np.isnan
    # every projection: |lat| > 90 and NaN input
    for p in PROJS:
        x, y = p.forward([0.0, 0.0, np.nan, 10.0], [91.0, -90.5, 0.0, np.nan])
        assert nan(x).all() and nan(y).all(), p
        lon, lat = p.inverse([np.nan, 0.0], [0.0, np.nan])
        assert nan(lon).all() and nan(lat).all(), p
    o = P.Orthographic(-100, 40)
    x, _ = o.forward([-100.0, 80.0, -100.0, -100.0], [40.0, -40.0, -50.0 + 1e-6, -50.0 - 1e-6])
    assert np.isfinite(x[0]) and nan(x[1]) and np.isfinite(x[2]) and nan(x[3])
    lon, _ = o.inverse([0.0, R * 1.0001, R * 0.7, 0.0], [0.0, 0.0, R * 0.72, -R * 1.01])
    assert np.isfinite(lon[0]) and nan(lon[1]) and nan(lon[2]) and nan(lon[3])
    m = P.Mercator()
    x, y = m.forward([0.0, 0.0, 0.0, 0.0], [90.0, -90.0, 85.0, 89.9])
    assert nan(x[:2]).all() and np.isfinite(x[2:]).all()            # beyond the limits: kept
    lon, _ = m.inverse([0.0, 0.0, 0.0, m.x_limits[1] * 1.001], [m.y_limits[1] * 1.001, m.y_limits[0] * 0.999, 0.0, 0.0])
    assert nan(lon[0]) and np.isfinite(lon[1]) and np.isfinite(lon[2]) and nan(lon[3])
    lcc = P.LambertConformal()                                       # cutoff -30
    x, _ = lcc.forward([0.0, 0.0, -96.0, 84.0], [-29.9, -30.1, -90.0, 90.0])
    assert np.isfinite(x[0]) and nan(x[1]) and nan(x[2]) and np.isfinite(x[3])
    # the wedge behind the apex is not on the cone
    lon, _ = lcc.inverse([0.0, 1e5, -1e5], [lcc.params[4] + 1e6, lcc.params[4] + 1e6, lcc.params[4] + 1e6])
    assert nan(lon).all()
    s = P.LambertConformal(140, -35, (-30, -50), cutoff=10)
    x, _ = s.forward([140.0, 140.0, 140.0], [9.0, 11.0, -90.0])
    assert np.isfinite(x[0]) and nan(x[1]) and np.isfinite(x[2])
    la = P.LambertAzimuthalEqualArea(10, 52)
    x, _ = la.forward([-170.0, -170.0], [-52.0, -51.0])
    assert nan(x[0]) and np.isfinite(x[1])
    lon, _ = la.inverse([0.0, 2.0001 * R], [1.9999 * R, 0.0])
    assert np.isfinite(lon[0]) and nan(lon[1])
    st = P.NorthPolarStereo()
    x, _ = st.forward([0.0, 0.0], [-90.0, -89.0])
    assert nan(x[0]) and np.isfinite(x[1])
    lon, _ = st.inverse([0.0, 0.0], [st.y_limits[1] * 0.99, st.y_limits[1] * 1.01])
    assert np.isfinite(lon[0]) and nan(lon[1])
    rb = P.Robinson()
    lon, _ = rb.inverse([rb.x_limits[1] * 0.999, rb.x_limits[1] * 0.999, 0.0],
                        [0.0, rb.y_limits[1] * 0.9, rb.y_limits[1] * 1.001])
    assert np.isfinite(lon[0]) and nan(lon[1]) and nan(lon[2])
    al = P.AlbersEqualArea()
    lon, _ = al.inverse([0.0], [al.params[6] * R])                   # the apex, inside the hole
    assert nan(lon).all()


def test_shapes_scalars_and_parallel_path():
    p = P.LambertConformal()
    x, y = p.forward(-96.0, 39.0)
    assert np.ndim(x) == 0 and abs(x) < 1e-6 and abs(y) < 1e-6
    lon = np.linspace(-130, -60, 12).reshape(3, 4)
    x, y = p.forward(lon, 40.0)                                      # broadcast
    assert x.shape == (3, 4) and y.shape == (3, 4)
    lon, lat = _lonlat(P.PAR_MIN * 4 + 7, seed=4)
    xs, ys = np.empty(lon.size), np.empty(lon.size)
    P._fwd_ser(p.kind, p.params, lon, lat, xs, ys)
    xp, yp = p.forward(lon, lat)                                     # parallel
    np.testing.assert_array_equal(xs, xp)
    np.testing.assert_array_equal(ys, yp)
    f = np.asfortranarray(np.stack([lon, lat]).T)                    # non-contiguous views
    x2, y2 = p.forward(f[:, 0], f[:, 1])
    np.testing.assert_array_equal(x2, xp)


# ---------------------------------------------------------------- from other kernels

def test_fwd_inv_from_other_numba_kernels():
    _jit.load()
    import numba

    @numba.njit(cache=False, parallel=True)
    def project_all(kind, prm, lon, lat, x, y):
        for i in numba.prange(lon.size):
            x[i], y[i] = P.fwd(kind, prm, lon[i], lat[i])

    @numba.njit(cache=False)
    def round_trip(kind, prm, lon, lat):
        worst = 0.0
        for i in range(lon.size):
            x, y = P.fwd(kind, prm, lon[i], lat[i])
            lo, la = P.inv(kind, prm, x, y)
            if lo == lo:
                d = abs(la - lat[i])
                if d > worst:
                    worst = d
        return worst

    fwd, inv = P.fwd, P.inv

    @numba.njit(cache=False)
    def via_globals(kind, prm, x, y):                    # bound as globals, as `from ._proj import fwd`
        lo, la = inv(kind, prm, x, y)
        return fwd(kind, prm, lo, la)

    lon, lat = _lonlat(50000, seed=5)
    for p in PROJS:
        x, y = np.empty(lon.size), np.empty(lon.size)
        project_all(p.kind, p.params, lon, lat, x, y)
        xr, yr = p.forward(lon, lat)
        np.testing.assert_array_equal(x, xr)
        np.testing.assert_array_equal(y, yr)
        assert round_trip(p.kind, p.params, lon[:5000], lat[:5000]) < 1e-6
    p = P.LambertConformal()
    x, y = via_globals(p.kind, p.params, 1e5, -2e5)
    assert abs(x - 1e5) < 1e-6 and abs(y + 2e5) < 1e-6


def test_fwd_from_fastplot_deferred_kernel():
    """A kernel declared through fastplot's deferred njit before numba is
    up, as other modules do, calling fwd and inv (fresh interpreter)."""
    import os
    import subprocess
    import sys
    code = (
        "import sys\n"
        "from fastplot._jit import njit, prange\n"
        "from fastplot._proj import fwd, inv, Robinson\n"
        "assert 'numba' not in sys.modules\n"
        "@njit(nogil=True, parallel=True)\n"
        "def k(kind, p, lon, lat, out):\n"
        "    for i in prange(lon.size):\n"
        "        x, y = fwd(kind, p, lon[i], lat[i])\n"
        "        out[i] = inv(kind, p, x, y)[1] - lat[i]\n"
        "import numpy as np\n"
        "p = Robinson(central_longitude=10)\n"
        "lat = np.linspace(-89, 89, 1001); out = np.empty(lat.size)\n"
        "k(p.kind, p.params, lat * 2, lat, out)\n"
        "print(np.abs(out).max() < 1e-9)\n")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, FASTPLOT_PRELOAD='0')       # numba left unloaded till the call
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=root,
                         env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == 'True'


def test_kernel_made_after_preload_with_a_stand_in():
    """A module that took the prange stand-in and a kernel's stand-in while
    numba was loading, its kernel defined once numba is up (fresh
    interpreter): the stand-ins are swapped before it is made."""
    import os
    import subprocess
    import sys
    code = (
        "from fastplot import _jit\n"
        "from fastplot._jit import njit, prange\n"
        "from fastplot._proj import fwd, Robinson\n"
        "assert type(prange) is _jit._Prange or prange is __import__('numba').prange\n"
        "_jit._PRELOAD['thread'] and _jit._PRELOAD['thread'].join()\n"
        "_jit.load()\n"
        "assert _jit._numba is not None\n"
        "prange = _jit._STAND_IN\n"
        "@njit(nogil=True, parallel=True)\n"
        "def k(kind, p, lon, lat, out):\n"
        "    for i in prange(lon.size):\n"
        "        out[i] = fwd(kind, p, lon[i], lat[i])[1]\n"
        "import numpy as np\n"
        "p = Robinson()\n"
        "lat = np.linspace(-80, 80, 101); out = np.empty(lat.size)\n"
        "k(p.kind, p.params, lat, lat, out)\n"
        "print(np.isfinite(out).all())\n")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=root)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == 'True'


# ---------------------------------------------------------------- transform_points

def test_transform_points():
    lcc = P.LambertConformal(central_longitude=-97.5, central_latitude=38.5, standard_parallels=(38.5,))
    merc = P.Mercator(central_longitude=-100)
    lon, lat = _lonlat(30000, seed=6)
    m = (lat > 10) & (lat < 70)
    x, y = lcc.forward(lon[m], lat[m])
    x2, y2 = P.transform_points(lcc, merc, x, y)
    xr, yr = _pyproj(lcc.proj4, merc.proj4, x, y)
    assert (np.abs(x2 - xr) < 1e-3).all() and (np.abs(y2 - yr) < 1e-3).all()
    xs, ys = P.transform_points(lcc, lcc, x, y)
    np.testing.assert_array_equal(xs, x)
    assert xs is not x
    # a rotated-pole grid onto a map, against cartopy
    ccrs = pytest.importorskip('cartopy.crs')
    rlon, rlat = np.meshgrid(np.linspace(-20, 20, 81), np.linspace(-15, 15, 61))
    rp = P.RotatedPole(-170, 40)
    x, y = P.transform_points(rp, P.Orthographic(10, 50), rlon, rlat)
    sph = ccrs.Globe(ellipse=None, semimajor_axis=R, semiminor_axis=R)
    r = ccrs.Orthographic(10, 50, globe=sph).transform_points(ccrs.RotatedPole(-170, 40), rlon, rlat)
    assert x.shape == rlon.shape
    assert np.abs(x - r[..., 0]).max() < 1e-3 and np.abs(y - r[..., 1]).max() < 1e-3


# ---------------------------------------------------------------- extents

def _cartopy_bounds(cproj, extent, crs):
    """What cartopy's set_extent computes: the bounds of the extent's
    outline, projected (cartopy's own adaptive interpolation)."""
    import shapely
    x1, x2, y1, y2 = extent
    line = shapely.LineString([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]])
    b = cproj.project_geometry(line, crs).bounds
    return np.array([b[0], b[2], b[1], b[3]])


def test_extent_xy_vs_cartopy():
    ccrs = pytest.importorskip('cartopy.crs')
    pytest.importorskip('shapely')
    sph = ccrs.Globe(ellipse=None, semimajor_axis=R, semiminor_axis=R)
    pc = ccrs.PlateCarree()
    cases = [
        # fastplot, cartopy, extent, tolerance (fraction of the span: cartopy's
        # interpolation cuts curved edges short by up to ~2 % on Orthographic)
        (P.LambertConformal(), ccrs.LambertConformal(globe=sph), (-120, -73, 23, 50), 1e-3),
        (P.LambertConformal(-97.5, 38.5, (38.5,)),
         ccrs.LambertConformal(-97.5, 38.5, standard_parallels=(38.5,), globe=sph),
         (-125, -66, 22, 52), 1e-3),
        (P.PlateCarree(180), ccrs.PlateCarree(180), (120, 240, -10, 50), 1e-12),
        (P.PlateCarree(180), ccrs.PlateCarree(180), (100, -100, -30, 60), 1e-6),
        (P.PlateCarree(), ccrs.PlateCarree(), (170, 190, -10, 10), 1e-6),
        (P.PlateCarree(), ccrs.PlateCarree(), (-180, 180, -90, 90), 1e-12),
        (P.NorthPolarStereo(), ccrs.NorthPolarStereo(globe=sph), (-180, 180, 50, 90), 1e-2),
        (P.SouthPolarStereo(), ccrs.SouthPolarStereo(globe=sph), (-180, 180, -90, -60), 2e-2),
        (P.Mercator(), ccrs.Mercator(globe=sph), (-100, -60, 10, 40), 1e-9),
        (P.Mercator(central_longitude=180), ccrs.Mercator(central_longitude=180, globe=sph),
         (120, 250, -20, 50), 1e-9),
        (P.Orthographic(-100, 40), ccrs.Orthographic(-100, 40, globe=sph), (-130, -70, 20, 55), 2e-2),
        (P.LambertAzimuthalEqualArea(10, 52), ccrs.LambertAzimuthalEqualArea(10, 52, globe=sph),
         (-20, 40, 30, 70), 5e-3),
        (P.AlbersEqualArea(-96, 23, (29.5, 45.5)),
         ccrs.AlbersEqualArea(-96, 23, standard_parallels=(29.5, 45.5), globe=sph),
         (-120, -73, 23, 50), 2e-3),
        (P.Robinson(), ccrs.Robinson(globe=sph), (-30, 60, -40, 40), 2e-3),
    ]
    for fp, cp, e, tol in cases:
        a = _cartopy_bounds(cp, e, pc)
        b = np.array(P.extent_xy(fp, e))
        span = max(a[1] - a[0], a[3] - a[2])
        assert np.abs(a - b).max() <= tol * span, (fp, e, a, b)
    # in a rotated-pole crs
    rp, rc = P.RotatedPole(-170, 40), ccrs.RotatedPole(-170, 40)
    a = _cartopy_bounds(ccrs.LambertConformal(globe=sph), (-20, 20, -15, 15), rc)
    b = np.array(P.extent_xy(P.LambertConformal(), (-20, 20, -15, 15), rp))
    assert np.abs(a - b).max() < 1e-3 * (a[1] - a[0])


def test_extent_xy_exact():
    """The true bounds of the projected outline, where cartopy's are approximate."""
    lcc = P.LambertConformal()
    x0, x1, y0, y1 = P.extent_xy(lcc, (-120, -73, 23, 50))
    lon = np.linspace(-120, -73, 200001)
    xb, yb = lcc.forward(lon, 23.0)
    xt, yt = lcc.forward(lon, 50.0)
    assert abs(y0 - yb.min()) < 1.0 and abs(y1 - yt.max()) < 1.0
    assert abs(x0 - lcc.forward(-120.0, 23.0)[0]) < 1e-6 and abs(x1 - lcc.forward(-73.0, 23.0)[0]) < 1e-6
    o = P.Orthographic(-100, 40)
    assert abs(P.extent_xy(o, (-130, -70, 20, 55))[2] + R * math.sin(math.radians(20))) < 1.0
    st = P.NorthPolarStereo()
    rho = -st.forward(0.0, 50.0)[1]
    np.testing.assert_allclose(P.extent_xy(st, (-180, 180, 50, 90)), (-rho, rho, -rho, rho), rtol=1e-7)
    np.testing.assert_allclose(P.extent_xy(P.PlateCarree(180), (120, 240, -10, 50)), (-60, 60, -10, 50))
    # crs given; the projection's own domain
    m = P.Mercator()
    assert P.extent_xy(m, m.x_limits + m.y_limits, crs=m) == m.x_limits + m.y_limits
    with pytest.raises(ValueError):
        P.extent_xy(P.Orthographic(0, 0), (170, 175, -5, 5))


# ---------------------------------------------------------------- the objects

@pytest.mark.parametrize('proj', PROJS, ids=IDS)
def test_boundary_and_limits(proj):
    b = proj.boundary
    assert b.ndim == 2 and b.shape[1] == 2 and b.shape[0] >= 200 and b.dtype == np.float64
    assert np.isfinite(b).all()
    np.testing.assert_array_equal(b[0], b[-1])
    x, y = b[:, 0], b[:, 1]
    assert np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) > 0                # counter-clockwise
    assert np.allclose(proj.x_limits, (x.min(), x.max()))
    assert np.allclose(proj.y_limits, (y.min(), y.max()))
    # the ring, pulled a hair towards its centre, is all in the domain
    cx, cy = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
    if proj.kind in (P.LAMBERT_CONFORMAL, P.ALBERS_EQUAL_AREA):
        lon, lat = proj.inverse(x, y)                                  # (not star-shaped)
        assert np.isfinite(lon).mean() > 0.95
    else:
        lon, lat = proj.inverse(cx + (x - cx) * (1 - 1e-7), cy + (y - cy) * (1 - 1e-7))
        assert np.isfinite(lon).all()
    assert proj.cylindrical == (proj.kind in (P.PLATE_CARREE, P.MERCATOR, P.MILLER,
                                              P.GEODETIC, P.ROTATED_POLE))
    if proj.cylindrical:
        assert proj.period == pytest.approx(proj.x_limits[1] - proj.x_limits[0], rel=1e-12)
    else:
        assert proj.period == 0
    assert proj.params.shape == (P.NPARAM,) and proj.params.dtype == np.float64


def test_limits_match_cartopy():
    ccrs = pytest.importorskip('cartopy.crs')
    sph = ccrs.Globe(ellipse=None, semimajor_axis=R, semiminor_axis=R)
    pairs = [
        (P.PlateCarree(), ccrs.PlateCarree(), 1e-12),
        (P.Mercator(central_longitude=40), ccrs.Mercator(central_longitude=40, globe=sph), 1e-9),
        (P.LambertConformal(), ccrs.LambertConformal(globe=sph), 1e-3),
        (P.NorthPolarStereo(), ccrs.NorthPolarStereo(globe=sph), 1e-9),
        (P.Orthographic(10, 20), ccrs.Orthographic(10, 20, globe=sph), 1e-9),
        (P.LambertAzimuthalEqualArea(10, 52), ccrs.LambertAzimuthalEqualArea(10, 52, globe=sph), 1e-6),
        (P.AlbersEqualArea(), ccrs.AlbersEqualArea(globe=sph), 1e-3),
        (P.Robinson(), ccrs.Robinson(globe=sph), 1e-6),
        (P.Miller(), ccrs.Miller(globe=sph), 1e-9),
        (P.RotatedPole(-170, 40), ccrs.RotatedPole(-170, 40), 1e-12),
    ]
    for fp, cp, tol in pairs:
        span = cp.x_limits[1] - cp.x_limits[0]
        assert np.abs(np.subtract(fp.x_limits, cp.x_limits)).max() <= tol * span, fp
        assert np.abs(np.subtract(fp.y_limits, cp.y_limits)).max() <= tol * span, fp


def test_eq_hash_repr_proj4():
    a, b = P.LambertConformal(), P.LambertConformal(central_longitude=-96.0)
    assert a == b and hash(a) == hash(b) and len({a, b}) == 1
    assert a != P.LambertConformal(cutoff=-20)
    assert P.NorthPolarStereo() == P.Stereographic(central_latitude=90)
    assert P.PlateCarree() != P.Geodetic()
    assert P.Mercator() != P.Mercator(radius=6371000.0)
    assert repr(a) == ('LambertConformal(central_longitude=-96.0, central_latitude=39.0, '
                       'standard_parallels=(33.0, 45.0), cutoff=-30.0)')
    assert repr(P.NorthPolarStereo(true_scale_latitude=60)) == \
        'NorthPolarStereo(central_longitude=0.0, true_scale_latitude=60.0)'
    for p in PROJS:
        pyproj.CRS(p.proj4)                                        # usable
        assert isinstance(p.kind, int) and 0 <= p.kind <= 10
    codes = {P.Geodetic: 0, P.PlateCarree: 1, P.Mercator: 2, P.LambertConformal: 3,
             P.Stereographic: 4, P.NorthPolarStereo: 4, P.SouthPolarStereo: 4,
             P.Orthographic: 5, P.LambertAzimuthalEqualArea: 6, P.AlbersEqualArea: 7,
             P.Robinson: 8, P.Miller: 9, P.RotatedPole: 10}
    for cls, k in codes.items():
        assert cls.kind == k
    with pytest.raises(ValueError):
        P.LambertConformal(standard_parallels=(30, -30))
    with pytest.raises(ValueError):
        P.Mercator(latitude_true_scale=10, scale_factor=0.9)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(UserWarning):
            P.Stereographic(central_latitude=45, true_scale_latitude=60)


def test_no_numba_to_build():
    """Building a projection (limits, boundary) runs no kernel."""
    import subprocess
    import sys
    code = ('import sys; from fastplot import _proj as P\n'
            'ps = [P.LambertConformal(), P.Robinson(), P.AlbersEqualArea(), P.LambertAzimuthalEqualArea(0, 52),'
            ' P.Mercator(), P.Orthographic(), P.NorthPolarStereo(), P.Miller(), P.RotatedPole(10, 40)]\n'
            '[p.boundary for p in ps]\n'
            'print("numba" in sys.modules)\n')
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, FASTPLOT_PRELOAD='0')       # (preload imports numba on its own)
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=root,
                         env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == 'False'


# ---------------------------------------------------------------- GPU

TEST_KERNELS = r'''
extern "C" __global__ void t_fwd(int kind, const double* p, const double* a, const double* b,
                                 double* x, double* y, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) fp_fwd(kind, p, a[i], b[i], x + i, y + i);
}
extern "C" __global__ void t_inv(int kind, const double* p, const double* a, const double* b,
                                 double* x, double* y, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) fp_inv(kind, p, a[i], b[i], x + i, y + i);
}
'''


@pytest.fixture(scope='module')
def gpu():
    cp = pytest.importorskip('cupy')
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('no GPU')
    except Exception:
        pytest.skip('no GPU')
    mod = cp.RawModule(code=P.CUDA + TEST_KERNELS, options=('--std=c++14', '--fmad=false'))
    return cp, mod.get_function('t_fwd'), mod.get_function('t_inv')


def _close(proj, c, g, scale, cyc):
    """g vs c: the same NaNs, else within 1e-9 relative (scale: the size of
    a coordinate near zero; cyc: its period, if it wraps)."""
    np.testing.assert_array_equal(np.isnan(c), np.isnan(g))
    ok = np.isfinite(c)
    d = np.abs(g[ok] - c[ok])
    if cyc:
        d = np.minimum(d, np.abs(d - cyc))
    assert (d <= 1e-9 * (np.abs(c[ok]) + scale)).all(), (proj, d.max())


@pytest.mark.parametrize('proj', PROJS, ids=IDS)
def test_gpu_matches_numba(gpu, proj):
    cp, kf, ki = gpu
    lon, lat = _lonlat(200000, seed=7)
    n = lon.size
    dp = cp.asarray(proj.params)
    da, db = cp.asarray(lon), cp.asarray(lat)
    dx, dy = cp.empty(n), cp.empty(n)
    blocks = ((n + 255) // 256,)
    kf(blocks, (256,), (np.int32(proj.kind), dp, da, db, dx, dy, np.int64(n)))
    x, y = proj.forward(lon, lat)
    deg = proj.kind in LENIENT
    s = 1.0 if deg else proj.radius * 1e-6
    _close(proj, x, cp.asnumpy(dx), s, proj.period)
    _close(proj, y, cp.asnumpy(dy), s, 0)
    # inverse, of points over the whole limits box
    rng = np.random.default_rng(8)
    (x0, x1), (y0, y1) = proj.x_limits, proj.y_limits
    xs = np.concatenate([rng.uniform(x0, x1, n // 2), x[: n // 2]])
    ys = np.concatenate([rng.uniform(y0, y1, n // 2), y[: n // 2]])
    kd = cp.asarray(xs), cp.asarray(ys)
    ki(blocks, (256,), (np.int32(proj.kind), dp, kd[0], kd[1], dx, dy, np.int64(n)))
    lo, la = proj.inverse(xs, ys)
    glo, gla = cp.asnumpy(dx), cp.asnumpy(dy)
    if proj.kind == P.ORTHOGRAPHIC:     # the limb: lat from asin(~1), the last bit of rho x 1e6
        keep = np.hypot(xs, ys) < 0.9999 * proj.radius
        lo, la, glo, gla = lo[keep], la[keep], glo[keep], gla[keep]
    _close(proj, lo, glo, 1.0, 360.0)
    _close(proj, la, gla, 1.0, 0)
