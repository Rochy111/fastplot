"""fastplot._color against matplotlib: colour parsing, every registered
colormap's table, norms and their autoscaling, shade() byte for byte with
ScalarMappable.to_rgba, contourf's band colours and levels, the band
interval convention, and the CUDA kernels against the numba ones."""
import io

import numpy as np
import pytest

import matplotlib
matplotlib.use('agg')
import matplotlib.colors as mcolors                     # noqa: E402
import matplotlib.pyplot as plt                         # noqa: E402
from matplotlib.cm import ScalarMappable                # noqa: E402

from fastplot import _color as C                        # noqa: E402

RNG = np.random.default_rng(20261028)


# ---------------------------------------------------------------- colours

COLOR_SPECS = [
    'k', 'r', 'g', 'b', 'c', 'm', 'y', 'w', 'red', 'Red', 'RED', 'darkslategray',
    'DarkSlateGrey', 'tab:blue', 'tab:olive', 'TAB:Orange', 'xkcd:sky blue',
    'xkcd:Sky Blue', 'xkcd:grey', 'xkcd:gray', 'xkcd:cloudy blue', '#0f0', '#0f08',
    '#00ff00', '#00FF0080', '#a1b2c3', '0.5', '0', '1', '.25', 'C0', 'C1', 'C9', 'C10',
    'C123', 'none', 'None', 'NONE', (0.1, 0.2, 0.3), (0.1, 0.2, 0.3, 0.4), [1, 0, 0],
    np.array([0.2, 0.4, 0.6]), np.array([[0.2, 0.4, 0.6, 0.8]]), ('red', 0.3),
    ('#0f08', 0.5), ((0.1, 0.2, 0.3, 0.4), None),
    (np.float32(0.1), np.float64(0.2), 1), 'aliceblue', 'rebeccapurple', 'grey',
]
BAD_SPECS = ['notacolor', '#12', '#1234567', '1.5', '-0.1', (0.1, 0.2), (1.2, 0, 0),
             ('red', 2.0), 'C', 'Cx', [0.1, 0.2, 0.3, 0.4, 0.5], ('0.1', 0.2, 0.3), 0.5]


@pytest.mark.parametrize('spec', COLOR_SPECS, ids=repr)
def test_to_rgba_matches(spec):
    assert C.to_rgba(spec) == mcolors.to_rgba(spec)
    for a in (0.0, 0.37, 1.0):
        if isinstance(spec, tuple) and len(spec) == 2:
            continue
        assert C.to_rgba(spec, a) == mcolors.to_rgba(spec, a)


@pytest.mark.parametrize('spec', BAD_SPECS, ids=repr)
def test_to_rgba_rejects(spec):
    with pytest.raises((ValueError, TypeError)):
        mcolors.to_rgba(spec)
    with pytest.raises((ValueError, TypeError)):
        C.to_rgba(spec)
    assert C.is_color_like(spec) == mcolors.is_color_like(spec)


def test_every_named_colour():
    names = list(mcolors._colors_full_map)
    assert len(names) > 1100
    for n in names:
        assert C.to_rgba(n) == mcolors.to_rgba(n), n
        assert C.to_rgba(n.upper()) == mcolors.to_rgba(n.upper()) if len(n) > 1 else True


ARRAY_SPECS = [
    ['r', 'g', 'b'], ['none'], 'none', ['none', 'red'], [(1, 0, 0), (0, 1, 0)],
    [(1, 0, 0, 0.5), (0, 1, 0, 1)], np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
    np.array([[0.1, 0.2, 0.3, 0.4]]), ['C0', '#ff000080', '0.3', (0.2, 0.3, 0.4)], [],
    'red', (0.1, 0.2, 0.3), ('red', 0.5), np.ma.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                                                      mask=[[0, 1, 0], [0, 0, 0]]),
]


@pytest.mark.parametrize('spec', ARRAY_SPECS, ids=repr)
def test_to_rgba_array_matches(spec):
    a, b = C.to_rgba_array(spec), mcolors.to_rgba_array(spec)
    assert a.shape == b.shape and a.dtype == b.dtype
    np.testing.assert_array_equal(a, b)
    if not (isinstance(spec, tuple) and len(spec) == 2):
        np.testing.assert_array_equal(C.to_rgba_array(spec, 0.25),
                                      mcolors.to_rgba_array(spec, 0.25))


def test_to_rgba_bytes():
    for spec in COLOR_SPECS:
        f = np.array(mcolors.to_rgba(spec))
        np.testing.assert_array_equal(C.to_rgba_bytes(spec), (f * 255).astype(np.uint8))
        np.testing.assert_array_equal(C.to_rgba_bytes(spec, rounding='nearest'),
                                      np.floor(f * 255 + 0.5).astype(np.uint8))
    # a colormap's bytes for a colour are the same truncation
    cm = mcolors.ListedColormap(['#123456', 'xkcd:sky blue'])
    np.testing.assert_array_equal(cm([0, 1], bytes=True),
                                  [C.to_rgba_bytes('#123456'), C.to_rgba_bytes('xkcd:sky blue')])


# ---------------------------------------------------------------- colormaps

ALL_CMAPS = sorted(matplotlib.colormaps)


def _mpl_bytes(cm):
    return (cm(np.arange(cm.N), bytes=True), (np.array(cm.get_under()) * 255).astype(np.uint8),
            (np.array(cm.get_over()) * 255).astype(np.uint8),
            (np.array(cm.get_bad()) * 255).astype(np.uint8))


def _check_same(ours, theirs):
    lut, under, over, bad = _mpl_bytes(theirs)
    assert ours.N == theirs.N
    np.testing.assert_array_equal(ours.lut, lut)
    np.testing.assert_array_equal(ours.under, under)
    np.testing.assert_array_equal(ours.over, over)
    np.testing.assert_array_equal(ours.bad, bad)
    np.testing.assert_array_equal(ours._float_table(), theirs._lut)


def test_registry_complete():
    assert C.cmap_names() == ALL_CMAPS
    assert len(ALL_CMAPS) == 182


@pytest.mark.parametrize('name', ALL_CMAPS)
def test_cmap_lut(name):
    ours, theirs = C.get_cmap(name), matplotlib.colormaps[name]
    assert ours.name == theirs.name
    _check_same(ours, theirs)
    # the bundled bytes and the bytes of the rebuilt floats agree
    if ours._native is not None:
        np.testing.assert_array_equal(ours._native, C._to_bytes(ours.rgba, 'floor'))


@pytest.mark.parametrize('name', ALL_CMAPS[::3])
@pytest.mark.parametrize('n', [2, 7, 11, 256, 300])
def test_cmap_resampled_reversed(name, n):
    ours, theirs = C.get_cmap(name), matplotlib.colormaps[name]
    _check_same(ours.resampled(n), theirs.resampled(n))
    _check_same(ours.resampled(n).reversed(), theirs.resampled(n).reversed())
    _check_same(ours.reversed().reversed(), theirs.reversed().reversed())
    _check_same(C.get_cmap(name, n), plt.get_cmap(name, n))


@pytest.mark.parametrize('name', ['viridis', 'jet', 'RdBu', 'tab10', 'gnuplot', 'cubehelix'])
def test_cmap_extremes(name):
    ours, theirs = C.get_cmap(name), matplotlib.colormaps[name]
    kw = dict(under='xkcd:sky blue', over=(0.1, 0.2, 0.3, 0.4), bad='#ff000080')
    _check_same(ours.with_extremes(**kw), theirs.with_extremes(**kw))
    _check_same(ours.with_extremes(**kw).reversed(), theirs.with_extremes(**kw).reversed())
    _check_same(ours.with_extremes(over='k').resampled(9),
                theirs.with_extremes(over='k').resampled(9))
    _check_same(ours.with_alpha(0.3), theirs.with_alpha(0.3))


def test_listed_and_from_list():
    cols = ['r', '#00ff0080', (0.1, 0.2, 0.3), 'xkcd:sky blue', 'C3']
    _check_same(C.ListedColormap(cols, 'x'), mcolors.ListedColormap(cols, 'x'))
    _check_same(C.get_cmap(cols), mcolors.ListedColormap(cols))
    _check_same(C.ListedColormap(cols, under='k', over='w', bad='0.5'),
                mcolors.ListedColormap(cols, under='k', over='w', bad='0.5'))
    with pytest.warns(Warning):
        theirs = mcolors.ListedColormap(cols, N=12)
    _check_same(C.ListedColormap(cols, N=12), theirs)
    for N in (2, 10, 256, 1000):
        _check_same(C.LinearSegmentedColormap.from_list('f', cols, N),
                    mcolors.LinearSegmentedColormap.from_list('f', cols, N))
    pairs = [(0, 'navy'), (0.2, 'w'), (0.21, 'r'), (1, '#ffff0080')]
    _check_same(C.LinearSegmentedColormap.from_list('p', pairs),
                mcolors.LinearSegmentedColormap.from_list('p', pairs))
    _check_same(C.LinearSegmentedColormap.from_list('p', pairs, 64, gamma=0.7),
                mcolors.LinearSegmentedColormap.from_list('p', pairs, 64, gamma=0.7))
    _check_same(C.LinearSegmentedColormap.from_list('p', pairs).reversed(),
                mcolors.LinearSegmentedColormap.from_list('p', pairs).reversed())


def test_foreign_cmap_objects():
    for theirs in (matplotlib.colormaps['plasma'].with_extremes(under='r', over='b', bad='g'),
                   mcolors.ListedColormap(['r', 'g', 'b']),
                   plt.get_cmap('RdBu_r', 12).with_alpha(0.5)):
        ours = C.get_cmap(theirs)
        _check_same(ours, theirs)
        _check_same(ours.reversed(), theirs.reversed())
        _check_same(ours.resampled(5), theirs.resampled(5))
    assert C.get_cmap().name == 'viridis'
    assert C.get_cmap(C.get_cmap('jet')) is C.get_cmap('jet')
    with pytest.raises(ValueError):
        C.get_cmap('nope')


def test_colormap_call():
    ours, theirs = C.get_cmap('RdBu'), matplotlib.colormaps['RdBu']
    x = np.array([-0.5, 0, 0.3, 0.999, 1, 1.5, np.nan])
    np.testing.assert_array_equal(ours(x), theirs(x))
    np.testing.assert_array_equal(ours(x, bytes=True), theirs(x, bytes=True))
    np.testing.assert_array_equal(ours(x, alpha=0.4, bytes=True), theirs(x, alpha=0.4, bytes=True))
    np.testing.assert_array_equal(ours(np.arange(-2, 260)), theirs(np.arange(-2, 260)))
    assert ours(0.5) == theirs(0.5)


# ---------------------------------------------------------------- data

def _field(shape=(240, 330), lo=-3.0, hi=3.0, edges=(), N=256, dtype=np.float32):
    """Random values plus the troublesome ones: the limits, the bin edges
    of an N-colour map over them, their float32 neighbours, NaN, inf."""
    z = RNG.uniform(lo - 0.3 * (hi - lo), hi + 0.3 * (hi - lo), shape).astype(dtype)
    special = [lo, hi, 0.0, -0.0, np.nan, np.inf, -np.inf, *edges]
    special += list(lo + (hi - lo) * np.arange(N + 1) / N)
    sp = np.array(special, dtype)
    sp = np.concatenate([sp, np.nextafter(sp, np.inf, dtype=dtype),
                         np.nextafter(sp, -np.inf, dtype=dtype)])
    f = z.ravel()
    idx = RNG.choice(f.size, sp.size * 4, replace=False)
    f[idx] = np.tile(sp, 4)
    f[RNG.choice(f.size, f.size // 50, replace=False)] = np.nan
    return z


def _reference(values, norm, cmap, alpha=None):
    sm = ScalarMappable(norm=norm, cmap=cmap)
    m = np.ma.masked_invalid(values)
    sm.set_array(m)
    return sm.to_rgba(m, alpha=alpha, bytes=True), sm


NORMS = {
    'linear': lambda: mcolors.Normalize(-2, 2.5),
    'linear-clip': lambda: mcolors.Normalize(-2, 2.5, clip=True),
    'linear-auto': lambda: mcolors.Normalize(),
    'linear-vmin': lambda: mcolors.Normalize(vmin=-1.3),
    'linear-same': lambda: mcolors.Normalize(0.5, 0.5),
    'linear-odd': lambda: mcolors.Normalize(-0.7, 0.1),
    'twoslope': lambda: mcolors.TwoSlopeNorm(0.3, -1.0, 3.0),
    'twoslope-auto': lambda: mcolors.TwoSlopeNorm(0.5),
    'twoslope-auto2': lambda: mcolors.TwoSlopeNorm(5.0),
    'centered': lambda: mcolors.CenteredNorm(),
    'centered-hr': lambda: mcolors.CenteredNorm(0.3, 1.7),
    'centered-vc': lambda: mcolors.CenteredNorm(vcenter=1.1),
    'centered-clip': lambda: mcolors.CenteredNorm(0.2, 1.0, clip=True),
    'boundary': lambda: mcolors.BoundaryNorm([-2, -1, 0, 0.7, 1.5, 3], 256),
    'boundary-both': lambda: mcolors.BoundaryNorm([-2, -1, 0, 0.7, 1.5, 3], 256, extend='both'),
    'boundary-min': lambda: mcolors.BoundaryNorm([-2, -1, 0, 0.7, 1.5, 3], 256, extend='min'),
    'boundary-max': lambda: mcolors.BoundaryNorm([-2, -1, 0, 0.7, 1.5, 3], 256, extend='max'),
    'boundary-exact': lambda: mcolors.BoundaryNorm([-2, -1, 0, 0.7, 1.5, 3], 5),
    'boundary-exact-both': lambda: mcolors.BoundaryNorm([-2, -1, 0, 0.7, 3], 6, extend='both'),
    'boundary-clip': lambda: mcolors.BoundaryNorm([-2, -1, 0, 0.7, 1.5, 3], 256, clip=True),
    'boundary-one': lambda: mcolors.BoundaryNorm([-1, 1.1], 256),
    'boundary-over': lambda: mcolors.BoundaryNorm(np.linspace(-2, 2, 11), 300),
    'nonorm': lambda: mcolors.NoNorm(),
    'power': lambda: mcolors.PowerNorm(0.5, -1, 3),
    'power-auto': lambda: mcolors.PowerNorm(2.0),
}
LOG_NORMS = {
    'log': lambda: mcolors.LogNorm(0.01, 10),
    'log-auto': lambda: mcolors.LogNorm(),
    'log-clip': lambda: mcolors.LogNorm(0.1, 5, clip=True),
    'symlog': lambda: mcolors.SymLogNorm(0.5, vmin=-3, vmax=3),
    'symlog-e': lambda: mcolors.SymLogNorm(0.2, linscale=0.5, vmin=-2, vmax=3, base=np.e),
    'symlog-inside': lambda: mcolors.SymLogNorm(10, vmin=-3, vmax=3),
    'symlog-clip': lambda: mcolors.SymLogNorm(0.5, vmin=-1, vmax=2, clip=True),
}
SHADE_CMAPS = ['viridis', 'jet', 'RdBu_r', 'tab10', 'twilight', 'cubehelix_r', 'gist_ncar']


def _cmap_pair(name):
    if name == 'extremes':
        t = matplotlib.colormaps['coolwarm'].with_extremes(under='k', over='#00ff0080', bad='w')
    elif name == 'discrete':
        t = plt.get_cmap('jet', 10)
    else:
        t = matplotlib.colormaps[name]
    return C.get_cmap(t), t


def _ours(values, mnorm, cm, **kw):
    """shade with a fresh copy of the norm (fastplot autoscales its own)."""
    return C.shade(values, cm, mnorm, **kw)


@pytest.mark.parametrize('cname', SHADE_CMAPS + ['extremes', 'discrete'])
@pytest.mark.parametrize('nname', sorted(NORMS))
def test_shade_matches_matplotlib(cname, nname):
    ours_cm, theirs_cm = _cmap_pair(cname)
    edges = [-2, -1, 0, 0.7, 1.5, 3, 0.3, 0.5, 1.1, -1.3, 2.5]
    z = _field(edges=edges, N=ours_cm.N)
    ref, _ = _reference(z, NORMS[nname](), theirs_cm)
    got = _ours(z, NORMS[nname](), ours_cm)
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize('nname', ['linear', 'linear-clip', 'linear-auto', 'twoslope',
                                   'centered-vc', 'boundary-both', 'nonorm', 'log',
                                   'log-clip', 'symlog', 'symlog-inside'])
def test_shade_float64_data(nname):
    norms = {**NORMS, **LOG_NORMS}
    z = _field(edges=[0.01, 10, 0.1, 5, 0.3], dtype=np.float64)
    ref, _ = _reference(z, norms[nname](), matplotlib.colormaps['viridis'])
    np.testing.assert_array_equal(_ours(z, norms[nname](), 'viridis'), ref)


def _f32_log_differs(z, fn):
    """Where numpy's float32 log (a SIMD / libm approximation) is not the
    correctly rounded value fastplot uses."""
    with np.errstate(all='ignore'):
        a = np.abs(z)
        return getattr(np, fn)(a) != getattr(np, fn)(a.astype(np.float64)).astype(np.float32)


@pytest.mark.parametrize('cname', ['viridis', 'jet', 'extremes', 'discrete'])
@pytest.mark.parametrize('nname', sorted(LOG_NORMS))
def test_shade_log_norms(cname, nname):
    """Log and symlog norms take numpy's float32 log, whose last bit varies
    with the machine's SIMD; fastplot's is correctly rounded. Pixels may
    then differ by one colour step, but only where those logs differ."""
    ours_cm, theirs_cm = _cmap_pair(cname)
    z = _field(lo=-1.0, hi=10.0, edges=[0.01, 0.1, 5, 0.5, 0.2, 2, -2, 3, -3])
    pos = RNG.uniform(-4, 1.2, z.shape).astype(np.float32)
    z = np.where(RNG.random(z.shape) < 0.5, (10.0 ** pos).astype(np.float32), z)
    ref, _ = _reference(z, LOG_NORMS[nname](), theirs_cm)
    got = _ours(z, LOG_NORMS[nname](), ours_cm)
    bad = (got != ref).any(-1)
    explained = _f32_log_differs(z, 'log10' if nname.startswith('log') else 'log')
    assert not (bad & ~explained).any()
    assert bad.mean() < 1e-3


@pytest.mark.parametrize('alpha', [0.0, 0.3, 0.5, 1.0])
@pytest.mark.parametrize('cname', ['viridis', 'extremes'])
def test_shade_alpha(alpha, cname):
    ours_cm, theirs_cm = _cmap_pair(cname)
    z = _field()
    ref, _ = _reference(z, mcolors.Normalize(-2, 2), theirs_cm, alpha=alpha)
    np.testing.assert_array_equal(C.shade(z, ours_cm, mcolors.Normalize(-2, 2), alpha=alpha), ref)


def test_shade_nearest_rounding():
    """rounding='nearest' is the float colour as Agg paints it."""
    z = _field()
    sm = ScalarMappable(mcolors.Normalize(-2, 2), 'RdBu_r')
    f = sm.to_rgba(np.ma.masked_invalid(z), alpha=0.6)
    want = np.floor(f * 255 + 0.5).astype(np.uint8)
    np.testing.assert_array_equal(
        C.shade(z, 'RdBu_r', mcolors.Normalize(-2, 2), alpha=0.6, rounding='nearest'), want)


def test_pcolormesh_pixels_are_nearest():
    """pcolormesh's PNG holds the 'nearest' bytes (and imshow's the floor
    ones would differ)."""
    z = RNG.normal(size=(10, 10)).astype(np.float32)
    fig = plt.figure(figsize=(1, 1), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.pcolormesh(z, cmap='jet', vmin=-2, vmax=2, antialiased=False)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    from PIL import Image
    png = np.array(Image.open(buf))
    centres = png[5::10, 5::10][::-1]
    np.testing.assert_array_equal(centres, C.shade(z, 'jet', vmin=-2, vmax=2, rounding='nearest'))


def test_shade_out_and_views():
    z = _field((50, 70))
    canvas = np.zeros((80, 100, 4), np.uint8)
    view = canvas[10:60, 20:90]
    r = C.shade(z, 'magma', out=view, vmin=-1, vmax=1)
    assert r is view
    np.testing.assert_array_equal(view, C.shade(z, 'magma', vmin=-1, vmax=1))
    assert (canvas[:10] == 0).all() and (canvas[:, :20] == 0).all()
    # a strided value raster, and an out that cannot be viewed as pixels
    zz = np.asfortranarray(z)[::2, ::3]
    odd = np.zeros((zz.shape[1], zz.shape[0], 4), np.uint8).transpose(1, 0, 2)
    C.shade(zz, 'magma', out=odd, vmin=-1, vmax=1)
    np.testing.assert_array_equal(odd, C.shade(np.ascontiguousarray(zz), 'magma', vmin=-1, vmax=1))
    # 1-D values
    np.testing.assert_array_equal(C.shade(z[3], 'magma', vmin=-1, vmax=1),
                                  C.shade(z[3:4], 'magma', vmin=-1, vmax=1)[0])
    # int data: float64 as in matplotlib's process_value
    zi = RNG.integers(-5, 5, (20, 20)).astype(np.int32)
    ref, _ = _reference(zi, mcolors.Normalize(-3, 3), matplotlib.colormaps['viridis'])
    np.testing.assert_array_equal(C.shade(zi, 'viridis', vmin=-3, vmax=3), ref)


def test_shade_vmin_vmax_and_strings():
    z = _field(lo=0.01, hi=10)
    z[np.abs(z) > 1e30] = 7.0       # a scale norm masks its overflow, Normalize does not
    for norm, kw in ((None, dict(vmin=-1, vmax=2)), (None, dict(vmax=2)), ('log', dict(vmin=0.1)),
                     ('linear', {}), ('log', {})):
        sm = ScalarMappable(norm=norm, cmap='viridis')
        m = np.ma.masked_invalid(z)
        sm.set_array(m)
        if kw:
            sm.set_clim(kw.get('vmin'), kw.get('vmax'))
            sm.autoscale_None()
        ref = sm.to_rgba(m, bytes=True)
        got = C.shade(z, 'viridis', norm, **kw)
        bad = (got != ref).any(-1)
        assert not (bad & ~_f32_log_differs(z, 'log10')).any(), (norm, kw)


# ---------------------------------------------------------------- norms

@pytest.mark.parametrize('nname', sorted({**NORMS, **LOG_NORMS}))
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_make_norm_autoscale(nname, dtype):
    mk = {**NORMS, **LOG_NORMS}[nname]
    z = _field(lo=-1.5, hi=4.0, dtype=dtype)
    _, sm = _reference(z, mk(), 'viridis')
    n = C.make_norm(mk(), data=z)
    t = sm.norm
    if n.kind not in ('boundary', 'nonorm'):
        assert (n.vmin, n.vmax) == (t.vmin, t.vmax)
        assert type(n.vmin) is type(t.vmin) is float
    assert n.kind == {'Normalize': 'linear', 'LogNorm': 'log', 'SymLogNorm': 'symlog',
                      'TwoSlopeNorm': 'twoslope', 'CenteredNorm': 'centered',
                      'BoundaryNorm': 'boundary', 'NoNorm': 'nonorm'}.get(type(t).__name__, 'func')


def test_make_norm_rules():
    with pytest.raises(ValueError):
        C.make_norm(mcolors.Normalize(0, 1), vmin=0)
    n = C.make_norm(None, vmin=2, data=np.array([1, 5, np.nan], np.float32))
    assert (n.vmin, n.vmax) == (2.0, 5.0)
    n = C.make_norm(C.Norm('log'), vmax=100, data=np.array([-1, 0, 0.5, 7], np.float32))
    assert (n.kind, n.vmin, n.vmax) == ('log', 0.5, 100.0)
    spec = C.Norm('linear')
    C.make_norm(spec, data=np.arange(3.0))
    assert spec.vmin is None                     # never changed in place
    n = C.make_norm(C.Norm('boundary', boundaries=[0, 1, 2]), N=10)
    assert n.ncolors == 10
    with pytest.raises(ValueError):
        C.make_norm(C.Norm('boundary', boundaries=[0, 1, 2], extend='both'), N=3)
    with pytest.raises(ValueError):
        C.shade(np.ones((2, 2), np.float32), 'viridis', vmin=2, vmax=1)
    with pytest.raises(ValueError):
        C.shade(np.ones((2, 2), np.float32), 'viridis', 'log', vmin=0, vmax=1)
    n = C.Norm('centered', vcenter=1, halfrange=2.5)
    m = mcolors.CenteredNorm(1, 2.5)
    assert (n.vmin, n.vmax) == (m.vmin, m.vmax)


# ---------------------------------------------------------------- contours

def _contourf(levels, z=None, **kw):
    if z is None:
        z = RNG.normal(size=(30, 40)) * 2
    fig, ax = plt.subplots()
    cs = ax.contourf(z, levels=levels, **kw)
    fc, lev = cs.get_facecolor(), cs.levels
    plt.close(fig)
    return fc, lev


LEVELS = [-3, -1.5, -0.5, 0, 0.3, 1, 2.2, 4]
CONTOUR_CASES = [
    dict(),
    dict(cmap='jet'),
    dict(cmap='RdBu_r', vmin=-2),
    dict(cmap='RdBu_r', vmin=-5, vmax=1),
    dict(cmap='tab10'),
    dict(cmap='gist_ncar_r', alpha=0.5),
    dict(cmap=lambda: matplotlib.colormaps['viridis'].with_extremes(under='k', over='0.8')),
    dict(cmap=lambda: plt.get_cmap('coolwarm', 9)),
    dict(colors=['r', 'g', 'b', 'c', 'm', 'y', 'k']),
    dict(colors=['r', 'g', 'b']),
    dict(colors=['r', 'g', 'b', 'c', 'm', 'y', 'k', '0.5']),
    dict(colors=['r', 'g', 'b', 'c', 'm', 'y', 'k', '0.5', 'xkcd:sky blue']),
    dict(colors='red'),
    dict(colors=(0.1, 0.2, 0.3)),
    dict(colors=['r', 'g', 'b', 'c', 'm', 'y', 'k', '0.5'], alpha=0.3),
    dict(norm=lambda: mcolors.BoundaryNorm(LEVELS, 256)),
    dict(norm=lambda: mcolors.BoundaryNorm(LEVELS, 256, extend='both'), cmap='RdBu_r'),
    dict(norm=lambda: mcolors.BoundaryNorm(LEVELS, 9, extend='both'), cmap='turbo'),
    dict(norm=lambda: mcolors.TwoSlopeNorm(0.0)),
    dict(norm=lambda: mcolors.TwoSlopeNorm(0.3, -4, 4)),
    dict(norm=lambda: mcolors.CenteredNorm()),
    dict(norm=lambda: mcolors.Normalize(-1, 1)),
    dict(norm=lambda: mcolors.Normalize(-1, 1, clip=True)),
    dict(norm=lambda: mcolors.PowerNorm(0.5)),
    dict(norm=lambda: mcolors.NoNorm()),
]


def _resolve(kw):
    return {k: (v() if callable(v) and k in ('norm', 'cmap') else v) for k, v in kw.items()}


@pytest.mark.parametrize('extend', ['neither', 'min', 'max', 'both'])
@pytest.mark.parametrize('case', range(len(CONTOUR_CASES)))
def test_contour_colors(case, extend):
    kw = CONTOUR_CASES[case]
    if 'norm' in kw and 'BoundaryNorm' in repr(kw['norm']()) and extend != 'neither' \
            and kw['norm']().extend == 'neither' and kw['norm']().Ncmap < 10:
        pytest.skip('too few colours')
    fc, _ = _contourf(LEVELS, extend=extend, **_resolve(kw))
    bands, under, over = C.contour_colors(LEVELS, extend=extend, rounding=None, **_resolve(kw))
    got = np.vstack([x for x in ([under] if under is not None else []) + [bands]
                     + ([over] if over is not None else [])])
    np.testing.assert_array_equal(got, fc)
    assert (under is not None) == (extend in ('min', 'both'))
    assert (over is not None) == (extend in ('max', 'both'))
    b8, u8, o8 = C.contour_colors(LEVELS, extend=extend, **_resolve(kw))
    np.testing.assert_array_equal(b8, np.floor(bands * 255 + 0.5).astype(np.uint8))


@pytest.mark.parametrize('extend', ['neither', 'min', 'max', 'both'])
def test_contour_colors_log(extend):
    lev = [0.01, 0.1, 0.5, 1, 5, 10, 100]
    z = 10 ** RNG.uniform(-3, 3, (30, 40))
    fc, _ = _contourf(lev, z=z, norm=mcolors.LogNorm(), cmap='plasma', extend=extend)
    bands, under, over = C.contour_colors(lev, 'plasma', norm=mcolors.LogNorm(), extend=extend,
                                          rounding=None)
    got = np.vstack([x for x in ([under] if under is not None else []) + [bands]
                     + ([over] if over is not None else [])])
    np.testing.assert_array_equal(got, fc)
    b2, _, _ = C.contour_colors(lev, 'plasma', norm='log', extend=extend, rounding=None)
    np.testing.assert_array_equal(b2, bands)


def test_contourf_pixels_are_band_colours():
    """A rendered contourf of a ramp: each band's middle pixels hold the
    band's 'nearest' bytes."""
    lev = [0, 1, 2, 3, 4, 5, 6]
    z = np.tile(np.linspace(-0.5, 6.5, 200), (20, 1))
    fig = plt.figure(figsize=(2, 0.2), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.contourf(np.linspace(-0.5, 6.5, 200), np.arange(20), z, levels=lev, cmap='RdYlBu',
                extend='both')
    ax.set_xlim(-0.5, 6.5)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    from PIL import Image
    png = np.array(Image.open(buf))
    bands, under, over = C.contour_colors(lev, 'RdYlBu', extend='both')
    row = png[10]
    x_of = lambda v: int((v + 0.5) / 7 * 200)              # noqa: E731
    np.testing.assert_array_equal(row[x_of(-0.25)], under)
    np.testing.assert_array_equal(row[x_of(6.25)], over)
    for i in range(6):
        np.testing.assert_array_equal(row[x_of(i + 0.5)], bands[i])


@pytest.mark.parametrize('extend', ['neither', 'min', 'max', 'both'])
def test_auto_levels(extend):
    for trial in range(60):
        a = RNG.choice([1e-6, 1e-2, 1, 37, 1e4, 1e9])
        lo = RNG.normal() * a + RNG.choice([0, 0, 1000 * a])
        hi = lo + abs(RNG.normal()) * a * RNG.choice([1e-9, 1e-3, 1, 50]) + (trial == 0) * 0
        if trial == 1:
            hi = lo                                         # flat field
        n = RNG.choice([None, 1, 3, 5, 10, 17, 40])
        z = np.full((4, 4), lo)
        z[0, 0] = hi
        z[1, 1] = (lo + hi) / 2
        fig, ax = plt.subplots()
        cs = ax.contourf(z, n, extend=extend) if n is not None else ax.contourf(z, extend=extend)
        want = cs.levels
        plt.close(fig)
        got = C.auto_levels(n, float(np.min(z)), float(np.max(z)), extend)
        np.testing.assert_array_equal(got, want, err_msg=repr((n, lo, hi)))


# ---------------------------------------------------------------- bands

def _bands_reference(z, levels, band_lut, under, over, log=False):
    """contourf's intervals, plainly: z1 < z <= z2, the first closed below
    unless extended."""
    m = len(levels)
    out = np.zeros(z.shape + (4,), np.uint8)
    f = np.asarray(z, np.float64)
    ok = np.isfinite(f) & ((f > 0) if log else True)
    k = np.searchsorted(levels, f, side='left')
    for i in range(m - 1):
        sel = ok & (k == i + 1)
        if i == 0 and under is None:
            sel |= ok & (f == levels[0])
        out[sel] = band_lut[i]
    if under is not None:
        out[ok & (f <= levels[0])] = under
    if over is not None:
        out[ok & (f > levels[-1])] = over
    return out


@pytest.mark.parametrize('levels', [np.linspace(-2, 2, 9), np.array([-3, -1, -0.5, 0, 0.1, 0.7, 2.5]),
                                    np.arange(-2, 2.01, 0.1), np.array([0.0, 1.0]),
                                    np.geomspace(0.01, 100, 12)])
@pytest.mark.parametrize('extend', ['neither', 'min', 'max', 'both'])
def test_shade_bands(levels, extend):
    lv = np.asarray(levels, np.float64)
    band_lut, under, over = C.contour_colors(lv, 'turbo', extend=extend)
    z = _field(lo=lv[0], hi=lv[-1], edges=list(lv) + list(lv.astype(np.float32)))
    log = bool(lv[0] > 0)
    got = C.shade_bands(z, lv, band_lut, under, over, log=log)
    np.testing.assert_array_equal(got, _bands_reference(z, lv, band_lut, under, over, log))
    # values exactly at the levels: the band below, the first level in band 0
    at = lv.reshape(1, -1)
    g = C.shade_bands(at, lv, band_lut, under, over)[0]
    np.testing.assert_array_equal(g[0], under if under is not None else band_lut[0])
    np.testing.assert_array_equal(g[1:], band_lut)
    nan = C.shade_bands(np.array([[np.nan, np.inf, -np.inf]], np.float32), lv, band_lut, under, over)
    assert (nan == 0).all()


def test_shade_bands_uniform_path_exact():
    """The guessed (uniform-level) path agrees with a plain binary search
    for levels that are uniform only up to rounding."""
    lv = np.arange(-1.0, 1.0001, 0.05)
    lut = RNG.integers(0, 255, (len(lv) - 1, 4), dtype=np.uint8)
    z = _field(lo=-1, hi=1, edges=list(lv))
    plan = C.bands_plan(lv, lut, lut[0], lut[-1])
    assert plan.args[-1] == 1
    got = C.shade_bands(z, lv, lut, lut[0], lut[-1])
    np.testing.assert_array_equal(got, _bands_reference(z, lv, lut, lut[0], lut[-1]))


# ---------------------------------------------------------------- GPU

def _cupy():
    try:
        import cupy
        cupy.cuda.runtime.getDeviceCount()
        return cupy
    except Exception:
        return None


cp = _cupy()
needs_gpu = pytest.mark.skipif(cp is None, reason='no cupy / GPU')


@pytest.fixture(scope='module')
def module():
    return cp.RawModule(code=C.CUDA, options=('--std=c++14', '--fmad=false'))


def _gpu_run(module, plan, values):
    v = cp.asarray(np.ascontiguousarray(values, np.float32))
    out = cp.zeros(v.shape + (4,), cp.uint8)
    args = [cp.asarray(a) if isinstance(a, np.ndarray) else a for a in plan.args]
    k = module.get_function(plan.kernel)
    n = v.size
    k(((n + 255) // 256,), (256,), (v, np.int32(n), *args, out))
    return cp.asnumpy(out)


@needs_gpu
@pytest.mark.parametrize('nname', sorted(set(NORMS) | set(LOG_NORMS)))
@pytest.mark.parametrize('cname', ['viridis', 'extremes'])
def test_gpu_equals_cpu(module, nname, cname):
    mk = {**NORMS, **LOG_NORMS}[nname]
    ours_cm, _ = _cmap_pair(cname)
    z = _field((300, 413), lo=-1, hi=3, edges=[0.01, 0.1, 5, -2, 0.7, 1.5])
    z = np.where(RNG.random(z.shape) < 0.3, (10.0 ** RNG.uniform(-3, 1, z.shape)), z).astype(np.float32)
    n = C.make_norm(mk(), data=z, N=ours_cm.N)
    for alpha, rounding in ((None, 'floor'), (0.4, 'nearest')):
        plan = C.shade_plan(n, ours_cm, alpha=alpha, rounding=rounding)
        cpu = C.shade(z, ours_cm, n, alpha=alpha, rounding=rounding)
        if plan.kernel is None:
            continue                                        # a matplotlib norm, CPU only
        np.testing.assert_array_equal(_gpu_run(module, plan, z), cpu)


@needs_gpu
@pytest.mark.parametrize('extend', ['neither', 'min', 'both'])
def test_gpu_bands_equals_cpu(module, extend):
    for lv in (np.linspace(-2, 2, 21), np.array([-3, -1, -0.5, 0, 0.1, 0.7, 2.5]),
               np.geomspace(0.01, 100, 12)):
        lut, under, over = C.contour_colors(lv, 'jet', extend=extend)
        z = _field((256, 300), lo=lv[0], hi=lv[-1], edges=list(lv))
        log = bool(lv[0] > 0)
        plan = C.bands_plan(lv, lut, under, over, log=log)
        np.testing.assert_array_equal(_gpu_run(module, plan, z),
                                      C.shade_bands(z, lv, lut, under, over, log=log))


def test_no_matplotlib_at_import():
    import subprocess
    import sys
    code = ('import sys, numpy as np; import fastplot._color as c; '
            'c.shade(np.ones((3, 3), np.float32), "jet", vmin=0, vmax=2); '
            'c.contour_colors([0, 1, 2], "RdBu_r", extend="both"); '
            'assert "matplotlib" not in sys.modules, "matplotlib imported"')
    subprocess.run([sys.executable, '-c', code], check=True,
                   cwd=__import__('os').path.dirname(__import__('os').path.dirname(__file__)))
