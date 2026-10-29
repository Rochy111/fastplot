"""fastplot's locators, formatters and scales against matplotlib's: real
matplotlib Axes are drawn and their ticks, labels and offset texts compared
with fastplot's for the same view and tick space."""
import math
import warnings

import numpy as np
import pytest

from fastplot import _scale, _ticker
from fastplot._ticker import strip_mathdefault

mpl = pytest.importorskip('matplotlib')
mpl.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.scale as mscale        # noqa: E402
import matplotlib.ticker as mt           # noqa: E402

FIGSIZE = (6.4, 4.8)


@pytest.fixture(scope='module')
def ax():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    yield ax
    plt.close(fig)


def our_tick_space(ax, axis):
    """XAxis / YAxis.get_tick_space from the axes box and the label size."""
    pos = ax.get_position()
    length = (pos.width * FIGSIZE[0] if axis == 'x' else pos.height * FIGSIZE[1]) * 72
    return _ticker.tick_space(length, 10.0, axis)


def draw(ax, lim, scale='linear', axis='x', minor_on=False, major=None, minor=None,
         major_fmt=None, minor_fmt=None, **scale_kw):
    """Ticks, labels and offset text of a drawn matplotlib axis."""
    ax.cla()
    getattr(ax, 'set_%sscale' % axis)(scale, **scale_kw)
    a = ax.xaxis if axis == 'x' else ax.yaxis
    if minor_on:
        a.set_minor_locator(mt.AutoMinorLocator())
    for setter, v in ((a.set_major_locator, major), (a.set_minor_locator, minor),
                      (a.set_major_formatter, major_fmt), (a.set_minor_formatter, minor_fmt)):
        if v is not None:
            setter(v)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        getattr(ax, 'set_%slim' % axis)(*lim)
    ax.figure.canvas.draw()
    assert a.get_tick_space() == our_tick_space(ax, axis)
    return dict(view=a.get_view_interval(), ts=a.get_tick_space(),
                major=np.asarray(a.get_majorticklocs(), float),
                minor=np.asarray(a.get_minorticklocs(), float),
                labels=[strip_mathdefault(t.get_text()) for t in a.get_majorticklabels()],
                mlabels=[strip_mathdefault(t.get_text()) for t in a.get_minorticklabels()],
                offset=strip_mathdefault(a.get_offset_text().get_text()))


def ours(ref, scale, locators=None):
    vmin, vmax = ref['view']
    M, m, F, f = locators or scale.default_locators()
    t = M.tick_values(vmin, vmax, ref['ts'])
    labels, offset = F.format_ticks(t, vmin, vmax)
    mt_ = m.tick_values(vmin, vmax, ref['ts'], major=t)
    mt_ = _ticker.remove_overlapping_locs(mt_, t, vmin, vmax, scale.transform)
    mlabels, _ = f.format_ticks(mt_, vmin, vmax)
    return dict(major=t, minor=mt_, labels=labels, mlabels=mlabels, offset=offset)


def check(ref, got, what):
    for k in ('major', 'minor'):
        assert ref[k].shape == got[k].shape, (what, k, ref[k], got[k])
        assert np.allclose(ref[k], got[k], rtol=1e-12, atol=0), (what, k, ref[k], got[k])
    for k in ('labels', 'mlabels', 'offset'):
        assert list(ref[k]) == list(got[k]) if k != 'offset' else ref[k] == got[k], \
            (what, k, ref[k], got[k])


def compare(ax, lim, scale='linear', axis='x', minor_on=False, **kw):
    ref = draw(ax, lim, scale, axis, minor_on, **kw)
    s = _scale.get_scale(scale, **kw)
    check(ref, ours(ref, s, s.default_locators(minor_visible=minor_on)),
          (lim, scale, axis, minor_on, kw))


# ------------------------------------------------------------ default scales

def test_linear_random_ranges(ax):
    rng = np.random.default_rng(0)
    for i in range(120):
        a, b = sorted(rng.uniform(-1, 1, 2) * 10.0 ** rng.integers(-9, 10))
        if i % 4 == 1:                                   # big offsets
            off = 10.0 ** rng.integers(2, 10) * rng.choice([-1, 1])
            a, b = a * 10.0 ** -rng.integers(1, 5) + off, b * 10.0 ** -rng.integers(1, 5) + off
        if i % 7 == 0:
            a, b = b, a                                  # inverted
        compare(ax, (a, b), axis='xy'[i % 2], minor_on=i % 3 == 0)


def test_linear_special_ranges(ax):
    for lim in [(0, 1), (-1, 1), (0, 1e-12), (1e15, 1e15 + 3), (-5e-7, 3e-7), (99999, 100001),
                (0.001, 0.0012), (-1e20, 1e20), (1, 1), (0, 0), (1e300, 2e300), (-273.15, 40),
                (1000, 1000.0001), (123456789, 123456799)]:
        for axis in 'xy':
            compare(ax, lim, axis=axis, minor_on=True)


def test_log_scales(ax):
    rng = np.random.default_rng(1)
    for i in range(60):
        base = [10, 10, 2, np.e][i % 4]
        e0 = rng.uniform(-12, 12)
        e1 = e0 + (rng.uniform(0.05, 1.5) if i % 3 == 0 else rng.uniform(0.5, 30))
        compare(ax, (base ** e0, base ** e1), 'log', axis='xy'[i % 2], base=base)
    for lim in [(1, 10), (1, 1000), (0.5, 5), (2, 3), (1e-3, 1e3), (1e-30, 1e30), (10, 10.5),
                (1e5, 1e5 * 1.01), (3, 3e9), (1, 100)]:
        for axis in 'xy':
            compare(ax, lim, 'log', axis=axis)
            compare(ax, lim, 'log', axis=axis, base=2)


def test_log_subs_and_nonpositive(ax):
    for lim in [(1, 1e4), (0.3, 30), (1e-2, 1e8)]:
        compare(ax, lim, 'log', subs=[2, 5])
        compare(ax, lim, 'log', subs=[1, 2, 3], nonpositive='mask')


def test_symlog(ax):
    rng = np.random.default_rng(2)
    for i in range(60):
        lt = 10.0 ** rng.integers(-2, 3)
        a, b = sorted(rng.uniform(-1, 1, 2) * 10.0 ** rng.uniform(-2, 9))
        kw = dict(linthresh=lt)
        if i % 3 == 0:
            kw.update(linscale=rng.choice([0.5, 2.0]), base=rng.choice([2, 10]))
        if i % 4 == 0:
            kw['subs'] = [2, 5]
        compare(ax, (a, b), 'symlog', axis='xy'[i % 2], **kw)
    compare(ax, (-1e6, 1e6), 'symlog')
    compare(ax, (0.5, 1.5), 'symlog')


def test_logit(ax):
    rng = np.random.default_rng(3)
    for i in range(40):
        a, b = 10 ** -rng.uniform(0.1, 8), 1 - 10 ** -rng.uniform(0.1, 8)
        if i % 3 == 0:
            a, b = sorted(rng.uniform(0.01, 0.99, 2))
        compare(ax, (a, b), 'logit', axis='xy'[i % 2])


def test_func_scale(ax):
    fns = (lambda x: np.sqrt(np.abs(x)), lambda x: np.asarray(x) ** 2)
    for lim in [(0, 100), (3, 7.5), (1e3, 1e5)]:
        ref = draw(ax, lim, 'function', functions=fns)
        s = _scale.get_scale('function', functions=fns)
        check(ref, ours(ref, s), lim)


# -------------------------------------------------------- explicit locators

def test_multiple_and_auto_minor(ax):
    rng = np.random.default_rng(4)
    s = _scale.LinearScale()
    for i in range(50):
        a, b = sorted(rng.uniform(-1, 1, 2) * 10.0 ** rng.uniform(-3, 4))
        base = 10.0 ** rng.integers(-3, 3) * rng.choice([1, 2, 2.5, 3, 5])
        off = rng.choice([0, 0.5 * base, 0.1])
        n = [None, 2, 3, 4, 5, 'auto'][i % 6]
        while (b - a) / base > 40:
            base *= 10
        ref = draw(ax, (a, b), major=mt.MultipleLocator(base, off), minor=mt.AutoMinorLocator(n),
                   axis='xy'[i % 2])
        got = ours(ref, s, (_ticker.MultipleLocator(base, off), _ticker.AutoMinorLocator(n),
                            _ticker.ScalarFormatter(), _ticker.NullFormatter()))
        check(ref, got, (a, b, base, off, n))


def test_maxn_options(ax):
    rng = np.random.default_rng(5)
    s = _scale.LinearScale()
    opts = [dict(integer=True), dict(symmetric=True), dict(prune='lower'), dict(prune='upper'),
            dict(prune='both'), dict(min_n_ticks=4), dict(steps=[1, 3, 10]), dict(nbins=4),
            dict(nbins='auto', steps=[1, 2, 5, 10]), dict(nbins=20, integer=True, min_n_ticks=1)]
    for i in range(80):
        kw = dict(opts[i % len(opts)])
        nbins = kw.pop('nbins', 'auto')
        a, b = sorted(rng.uniform(-1, 1, 2) * 10.0 ** rng.uniform(-2, 5))
        ref = draw(ax, (a, b), major=mt.MaxNLocator(nbins, **kw), axis='xy'[i % 2])
        got = ours(ref, s, (_ticker.MaxNLocator(nbins, **kw), _ticker.NullLocator(),
                            _ticker.ScalarFormatter(), _ticker.NullFormatter()))
        check(ref, got, (a, b, nbins, kw))
        assert np.allclose(mt.MaxNLocator(nbins, **kw).view_limits(a, b),
                           _ticker.MaxNLocator(nbins, **kw).view_limits(a, b))


def test_simple_locators():
    rng = np.random.default_rng(6)
    for _ in range(50):
        a, b = sorted(rng.uniform(-1, 1, 2) * 10.0 ** rng.uniform(-3, 3))
        n = int(rng.integers(0, 15))
        assert np.array_equal(mt.LinearLocator(n).tick_values(a, b),
                              _ticker.LinearLocator(n).tick_values(a, b))
        assert np.allclose(mt.LinearLocator(n).view_limits(a, b),
                           _ticker.LinearLocator(n).view_limits(a, b))
        base, off = rng.uniform(0.1, 3), rng.uniform(0, 1)
        assert np.array_equal(mt.IndexLocator(base, off).tick_values(a, b),
                              _ticker.IndexLocator(base, off).tick_values(a, b))
        assert np.array_equal(mt.MultipleLocator(base).tick_values(a, b),
                              _ticker.MultipleLocator(base).tick_values(a, b))
        assert np.allclose(mt.MultipleLocator(base).view_limits(a, b),
                           _ticker.MultipleLocator(base).view_limits(a, b))
        assert np.allclose(mt.Locator().nonsingular(a, b), _ticker.Locator().nonsingular(a, b))
        assert np.allclose(mt.Locator().view_limits(a, a), _ticker.Locator().view_limits(a, a))
    presets = {(0.0, 1.0): [0, 0.3, 1]}
    assert list(_ticker.LinearLocator(presets=presets).tick_values(0, 1)) == [0, 0.3, 1]
    assert _ticker.NullLocator().tick_values(0, 1).size == 0
    loc = mt.LogLocator()
    loc.create_dummy_axis()
    for lim in [(0, 10), (5, 5), (-1, -2), (np.nan, 1), (1e-3, 1e3)]:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            loc.axis._get_shared_axis = lambda: [loc.axis]
            loc.axis.get_minpos = lambda: 0.25
            assert np.allclose(loc.nonsingular(*lim), _ticker.LogLocator().nonsingular(*lim, 0.25))
            assert np.allclose(loc.view_limits(*lim), _ticker.LogLocator().view_limits(*lim, 0.25))
    for lim in [(0, 1e-3), (0.3, 0.3), (-1, 2), (0.2, 0.9)]:
        assert np.allclose(mt.LogitLocator().nonsingular(*lim), _ticker.LogitLocator().nonsingular(*lim))


def test_symlog_locator_subs():
    for lim in [(-1e5, 1e5), (-3, 7e3), (5, 5e4), (-0.5, 0.5)]:
        for subs in (None, [1, 2, 5], np.arange(1, 10)):
            ref = mt.SymmetricalLogLocator(subs=subs, linthresh=1.0, base=10).tick_values(*lim)
            got = _ticker.SymmetricalLogLocator(subs=subs, linthresh=1.0, base=10).tick_values(*lim)
            assert np.allclose(ref, got)


# --------------------------------------------------------------- formatters

def test_formatters_on_axes(ax):
    s = _scale.LinearScale()
    cases = [
        (mt.ScalarFormatter(useMathText=True), _ticker.ScalarFormatter(useMathText=True)),
        (mt.ScalarFormatter(useOffset=False), _ticker.ScalarFormatter(useOffset=False)),
        (mt.ScalarFormatter(useOffset=1000.0), _ticker.ScalarFormatter(useOffset=1000.0)),
        (mt.EngFormatter('Hz'), _ticker.EngFormatter('Hz')),
        (mt.EngFormatter('m', places=2, sep=''), _ticker.EngFormatter('m', places=2, sep='')),
        (mt.EngFormatter(useOffset=True), _ticker.EngFormatter(useOffset=True)),
        (mt.PercentFormatter(), _ticker.PercentFormatter()),
        (mt.PercentFormatter(1.0, decimals=1), _ticker.PercentFormatter(1.0, decimals=1)),
        (mt.StrMethodFormatter('{x:.2f} km'), _ticker.StrMethodFormatter('{x:.2f} km')),
        (mt.FormatStrFormatter('%.1e'), _ticker.FormatStrFormatter('%.1e')),
        (mt.FuncFormatter(lambda x, p: '%d:%s' % (p, x)),
         _ticker.FuncFormatter(lambda x, p: '%d:%s' % (p, x))),
        (mt.NullFormatter(), _ticker.NullFormatter()),
    ]
    rng = np.random.default_rng(7)
    for i in range(120):
        mf, ff = cases[i % len(cases)]
        a, b = sorted(rng.uniform(-1, 1, 2) * 10.0 ** rng.uniform(-8, 10))
        if i % 3 == 0:
            off = 10.0 ** rng.integers(3, 8)
            a, b = a * 1e-4 + off, b * 1e-4 + off
        ref = draw(ax, (a, b), major_fmt=mf, axis='xy'[i % 2])
        got = ours(ref, s, (_ticker.AutoLocator(), _ticker.NullLocator(), ff,
                            _ticker.NullFormatter()))
        check(ref, got, (a, b, type(mf).__name__))
    ref = draw(ax, (0, 5), major=mt.FixedLocator([1, 2, 3]),
               major_fmt=mt.FixedFormatter(['a', 'b', 'c']))
    got = ours(ref, s, (_ticker.FixedLocator([1, 2, 3]), _ticker.NullLocator(),
                        _ticker.FixedFormatter(['a', 'b', 'c']), _ticker.NullFormatter()))
    check(ref, got, 'fixed')


def test_log_formatters_on_axes(ax):
    rng = np.random.default_rng(8)
    s10 = _scale.LogScale()
    for i in range(30):
        e0 = rng.uniform(-6, 6)
        e1 = e0 + (rng.uniform(0.1, 1.2) if i % 2 else rng.uniform(1, 12))
        lim = (10 ** e0, 10 ** e1)
        for mk, fk in [(lambda **k: mt.LogFormatter(**k), lambda **k: _ticker.LogFormatter(**k)),
                       (lambda **k: mt.LogFormatterExponent(**k),
                        lambda **k: _ticker.LogFormatterExponent(**k)),
                       (lambda **k: mt.LogFormatterMathtext(**k),
                        lambda **k: _ticker.LogFormatterMathtext(**k))]:
            ref = draw(ax, lim, 'log', major_fmt=mk(labelOnlyBase=True),
                       minor_fmt=mk(labelOnlyBase=False), axis='xy'[i % 2])
            M, m, _, _ = s10.default_locators()
            got = ours(ref, s10, (M, m, fk(labelOnlyBase=True), fk(labelOnlyBase=False)))
            check(ref, got, (lim, i))
    for lim in [(-1e3, 1e4), (-5, 0.5), (3, 3e5)]:
        ref = draw(ax, lim, 'symlog', linthresh=1, minor=mt.SymmetricalLogLocator(
            linthresh=1, base=10, subs=np.arange(2, 10)), minor_fmt=mt.LogFormatterSciNotation())
        s = _scale.SymmetricalLogScale(linthresh=1)
        M, _, F, _ = s.default_locators()
        got = ours(ref, s, (M, _ticker.SymmetricalLogLocator(s, np.arange(2, 10)), F,
                            _ticker.LogFormatterSciNotation(linthresh=1)))
        check(ref, got, lim)


def test_log_formatter_labels():
    f = _ticker.LogFormatterSciNotation()
    labels, off = f.format_ticks([1e-2, 1e-1, 1, 10, 100], 1e-2, 1e2)
    assert labels == ['$10^{-2}$', '$10^{-1}$', '$10^{0}$', '$10^{1}$', '$10^{2}$'] and off == ''
    f = _ticker.LogFormatterSciNotation(labelOnlyBase=False)
    assert f.format_ticks([20, 30, 50], 15, 60)[0] == [r'$2\times10^{1}$', r'$3\times10^{1}$', '']
    assert f.format_ticks([20, 30, 40], 18, 42)[0][2] == r'$4\times10^{1}$'   # < 0.4 decades
    assert '\\mathdefault' not in ''.join(_ticker.LogitFormatter().format_ticks([0.01, 0.5], 0, 1)[0])


def test_scalar_formatter_offsets_match_matplotlib():
    rng = np.random.default_rng(9)
    for _ in range(300):
        off = rng.choice([-1, 1]) * 10.0 ** rng.integers(2, 12) * rng.uniform(1, 9)
        a, b = sorted(off + rng.uniform(-1, 1, 2) * 10.0 ** rng.uniform(-3, 3))
        locs = mt.MaxNLocator(7).tick_values(a, b)
        for math_text in (False, True):
            ref = mt.ScalarFormatter(useMathText=math_text)
            ref.create_dummy_axis()
            ref.axis.set_view_interval(a, b)
            rl = [strip_mathdefault(x) for x in ref.format_ticks(locs)]
            ro = strip_mathdefault(ref.get_offset())
            gl, go = _ticker.ScalarFormatter(useMathText=math_text).format_ticks(locs, a, b)
            assert (gl, go) == (rl, ro), (a, b)


def test_existing_api():
    f = _ticker.ScalarFormatter()
    assert f.format([0.1, 0.2, 0.3], 0.05, 0.35) == (['0.1', '0.2', '0.3'], '')
    assert f.format([], 0, 1) == ([], '')
    assert isinstance(_ticker.auto_locator(), _ticker.MaxNLocator)
    assert _ticker.format_values(None, [-1.5, 0, 1.5]) == ['−1.5', '0.0', '1.5']
    assert _ticker.as_locator([1, 2]).tick_values().tolist() == [1, 2]
    lab, _ = _ticker.LongitudeFormatter().format_ticks([-120, 0, 180], -180, 180)
    assert lab == ['120°W', '0°', '180°']


def test_strip_mathdefault():
    assert strip_mathdefault(r'$\mathdefault{10^{2}}$') == '$10^{2}$'
    assert strip_mathdefault(r'$\times\mathdefault{10^{3}}\mathdefault{}$') == r'$\times10^{3}$'
    assert strip_mathdefault('1.5') == '1.5'


# -------------------------------------------- matplotlib objects, duck typed

def test_foreign_locators_and_formatters(ax):
    s = _scale.LinearScale()
    rng = np.random.default_rng(10)
    for i in range(30):
        a, b = sorted(rng.uniform(-1, 1, 2) * 10.0 ** rng.uniform(-3, 7))
        mloc, mmin = mt.MaxNLocator('auto', steps=[1, 2, 5, 10]), mt.AutoMinorLocator()
        mfmt = mt.ScalarFormatter(useMathText=True)
        ref = draw(ax, (a, b), major=mloc, minor=mmin, major_fmt=mfmt, axis='xy'[i % 2])
        got = ours(ref, s, (_ticker.get_locator(mt.MaxNLocator('auto', steps=[1, 2, 5, 10])),
                            _ticker.get_locator(mt.AutoMinorLocator()),
                            _ticker.get_formatter(mt.ScalarFormatter(useMathText=True)),
                            _ticker.get_formatter(mt.NullFormatter())))
        check(ref, got, (a, b))
    for lim in [(1e-3, 1e5), (2, 40)]:
        ref = draw(ax, lim, 'log')
        sl = _scale.LogScale()
        got = ours(ref, sl, (_ticker.get_locator(mt.LogLocator(), sl),
                             _ticker.get_locator(mt.LogLocator(subs='auto'), sl),
                             _ticker.get_formatter(mt.LogFormatterSciNotation(), sl),
                             _ticker.get_formatter(mt.LogFormatterSciNotation(
                                 labelOnlyBase=False), sl)))
        check(ref, got, lim)
    assert isinstance(_ticker.get_formatter('{x:.1f}'), _ticker.StrMethodFormatter)
    assert isinstance(_ticker.get_formatter('%.1f'), _ticker.FormatStrFormatter)
    assert _ticker.get_formatter(lambda x, p: 'v%g' % x).format_ticks([1, 2])[0] == ['v1', 'v2']
    assert _ticker.get_formatter(['a', 'b']).format_ticks([1, 2])[0] == ['a', 'b']
    assert _ticker.get_locator([1, 2, 3]).tick_values(0, 1).tolist() == [1, 2, 3]


# ------------------------------------------------------------------- scales

def _same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


def test_scale_transforms_match_matplotlib():
    x = np.concatenate([[-1e10, -5, -1, -0.3, 0, 1e-300, 1e-8, 0.2, 0.5, 0.999, 1, 2, 7.5, 1e5,
                         1e300, np.inf, -np.inf, np.nan], np.linspace(-20, 20, 101)])
    xs = np.linspace(-30, 30, 121)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for name, kw in [('linear', {}), ('log', {}), ('log', dict(base=2)),
                         ('log', dict(base=np.e, nonpositive='mask')), ('log', dict(base=3.0)),
                         ('symlog', {}), ('symlog', dict(linthresh=0.1, linscale=0.3, base=2)),
                         ('symlog', dict(linthresh=5, base=np.e)), ('logit', {}),
                         ('logit', dict(nonpositive='clip'))]:
            ref = mscale.scale_factory(name, None, **kw)
            got = _scale.get_scale(name, **kw)
            tr = ref.get_transform()
            assert got.name == name
            with np.errstate(all='ignore'):
                assert _same(tr.transform(x), got.transform(x)), (name, kw)
                assert _same(tr.inverted().transform(xs), got.inverse(xs)), (name, kw)
            for lim in [(-1, 5), (0, 0.5), (0.2, 1), (1e-3, 2)]:
                for minpos in (1e-5, np.inf):
                    assert ref.limit_range_for_scale(*lim, minpos) == \
                        got.limit_range_for_scale(*lim, minpos)
            assert np.array_equal(ref.val_in_range(x), got.val_in_range(x))
    assert np.isscalar(_scale.LogScale().transform(100.0))
    assert _scale.LogScale().transform(100.0) == 2.0


def test_default_locator_types():
    def kinds(s, **kw):
        return [type(v).__name__ for v in s.default_locators(**kw)]
    assert kinds(_scale.LinearScale()) == ['AutoLocator', 'NullLocator', 'ScalarFormatter',
                                           'NullFormatter']
    assert kinds(_scale.LinearScale(), minor_visible=True)[1] == 'AutoMinorLocator'
    assert kinds(_scale.LogScale()) == ['LogLocator', 'LogLocator', 'LogFormatterSciNotation',
                                        'LogFormatterSciNotation']
    M, m, F, f = _scale.LogScale(base=2).default_locators()
    assert M.base == 2 and m._subs == 'auto' and not F.labelOnlyBase and not f.labelOnlyBase
    assert _scale.LogScale(subs=[2, 5]).default_locators()[3].labelOnlyBase
    assert kinds(_scale.SymmetricalLogScale()) == [
        'SymmetricalLogLocator', 'SymmetricalLogLocator', 'LogFormatterSciNotation',
        'NullFormatter']
    assert kinds(_scale.LogitScale()) == ['LogitLocator', 'LogitLocator', 'LogitFormatter',
                                          'LogitFormatter']
    with pytest.raises(ValueError):
        _scale.get_scale('nope')


def test_foreign_scale(ax):
    for name, kw, lim in [('log', dict(base=2), (1, 1e4)), ('symlog', dict(linthresh=3), (-1e3, 50)),
                          ('linear', {}, (-3, 8)), ('logit', {}, (1e-4, 0.9))]:
        ms = mscale.scale_factory(name, None, **kw)
        s = _scale.get_scale(ms)
        assert isinstance(s, _scale.ForeignScale) and s.name == name
        x = np.array([0.01, 0.5, 2.0, 30.0])
        with np.errstate(all='ignore'):
            assert _same(s.transform(x), ms.get_transform().transform(x))
        ref = draw(ax, lim, name, **kw)
        check(ref, ours(ref, s), (name, kw))


def test_tick_space_formula(ax):
    for size in [(3, 2), (6.4, 4.8), (12, 3), (2, 9)]:
        fig, a = plt.subplots(figsize=size)
        pos = a.get_position()
        assert a.xaxis.get_tick_space() == _ticker.tick_space(pos.width * size[0] * 72, 10, 'x')
        assert a.yaxis.get_tick_space() == _ticker.tick_space(pos.height * size[1] * 72, 10, 'y')
        plt.close(fig)
    assert math.isclose(_ticker.nonsingular(1, 1)[0], 0.999)
