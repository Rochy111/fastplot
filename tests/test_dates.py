"""fastplot's date conversion, locators and formatters against
matplotlib.dates: the same floats from date2num, the same datetimes from
num2date, and on drawn matplotlib date axes the same ticks, labels and
offset strings."""
import datetime
import warnings

import numpy as np
import pytest

from fastplot import _dates, _ticker

mpl = pytest.importorskip('matplotlib')
mpl.use('Agg')
import matplotlib.dates as mdates        # noqa: E402
import matplotlib.pyplot as plt          # noqa: E402

UTC = datetime.timezone.utc


@pytest.fixture(scope='module')
def ax():
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    yield ax
    plt.close(fig)


def draw(ax, lim, axis='x', loc=None, fmt=None, concise=False):
    ax.cla()
    d0 = datetime.datetime(2000, 1, 1)
    ax.plot([d0, d0], [0, 1]) if axis == 'x' else ax.plot([0, 1], [d0, d0])
    a = ax.xaxis if axis == 'x' else ax.yaxis
    if loc is not None:
        a.set_major_locator(loc)
    if concise:
        a.set_major_formatter(mdates.ConciseDateFormatter(a.get_major_locator()))
    elif fmt is not None:
        a.set_major_formatter(fmt)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        getattr(ax, 'set_%slim' % axis)(*lim)
        ax.figure.canvas.draw()
    return dict(view=a.get_view_interval(), major=np.asarray(a.get_majorticklocs(), float),
                labels=[t.get_text() for t in a.get_majorticklabels()],
                offset=a.get_offset_text().get_text())


def check(ref, loc, fmt, what):
    vmin, vmax = ref['view']
    t = loc.tick_values(vmin, vmax)
    labels, offset = fmt.format_ticks(t, vmin, vmax)
    assert t.shape == ref['major'].shape and np.array_equal(t, ref['major']), \
        (what, t, ref['major'])
    assert labels == ref['labels'], (what, labels, ref['labels'])
    assert offset == ref['offset'], (what, offset, ref['offset'])


def _span(rng, i):
    span_s = 10 ** rng.uniform(-3.5, 10.3)               # 0.3 ms .. 600 years
    start = rng.uniform(-60, 80) * 365.25 * 86400
    if i % 3 == 0:
        start = round(start / 86400) * 86400             # midnight starts
    a, b = start / 86400, (start + span_s) / 86400
    return (b, a) if i % 5 == 0 else (a, b)              # some inverted


# ---------------------------------------------------------------- conversion

def test_date2num_matches_matplotlib():
    pd = pytest.importorskip('pandas')
    rng = np.random.default_rng(0)
    ns = rng.integers(-2 ** 62, 2 ** 62, 200)
    dt64 = ns.astype('datetime64[ns]')
    cases = [dt64, dt64.astype('datetime64[s]'), dt64.astype('datetime64[us]'),
             dt64.astype('datetime64[D]'), dt64.astype('datetime64[m]'),
             np.array(['2020-01-01T00:00:00.123456789', 'NaT'], 'datetime64[ns]'),
             np.datetime64('2021-06-01T12:30'), dt64[:5].astype('datetime64[us]').tolist(),
             datetime.datetime(2000, 2, 29, 23, 59, 59, 999999), datetime.date(1999, 12, 31),
             datetime.datetime(1, 1, 1), datetime.datetime(9999, 12, 31, 23),
             datetime.datetime(2020, 1, 1, 6, tzinfo=datetime.timezone(datetime.timedelta(hours=5))),
             [datetime.datetime(2020, 3, 1, tzinfo=UTC), datetime.datetime(2020, 3, 2, tzinfo=UTC)],
             (datetime.date(2001, 1, 1), datetime.date(2002, 1, 1)),
             pd.Timestamp('2020-01-01 00:00:00.000000123'),
             pd.Timestamp('2020-01-01 03:00', tz='UTC'),
             pd.date_range('2020-01-01', periods=50, freq='37min'),
             pd.date_range('2020-01-01', periods=10, freq='h', tz='UTC'),
             pd.Series(pd.date_range('1900-01-01', periods=10, freq='7D')),
             pd.DatetimeIndex(dt64[:20]), np.array([], 'datetime64[ns]')]
    for c in cases:
        ref = mdates.date2num(c)
        got = _dates.date2num(c)
        assert np.shape(ref) == np.shape(got), c
        assert np.array_equal(np.asarray(ref, float), np.asarray(got, float), equal_nan=True), c


def test_num2date_matches_matplotlib():
    rng = np.random.default_rng(1)
    xs = np.concatenate([rng.uniform(-700000, 2900000, 300), rng.uniform(-30, 30, 300),
                         [0, 0.5, 25567.999999999, -719162.0, 2932896.9]])
    for x in xs:
        assert mdates.num2date(x) == _dates.num2date(x)
    assert mdates.num2date(xs[:50]) == _dates.num2date(xs[:50])
    tz = datetime.timezone(datetime.timedelta(hours=-3, minutes=-30))
    assert mdates.num2date(xs[:20], tz) == _dates.num2date(xs[:20], tz)
    with pytest.raises(ValueError):
        _dates.num2date(1e8)


def test_round_trips():
    pd = pytest.importorskip('pandas')
    rng = np.random.default_rng(2)
    us = rng.integers(-2 ** 52, 2 ** 52, 100).astype('datetime64[us]')
    for d in us[:50].tolist():
        assert _dates.num2date(_dates.date2num(d)).replace(tzinfo=None) - d < \
            datetime.timedelta(microseconds=21)
    near = np.datetime64('1975-01-01') + rng.integers(0, 10 ** 15, 50).astype('timedelta64[us]')
    for d in near.tolist():                                   # 1 us exact near the epoch
        assert _dates.num2date(_dates.date2num(d)).replace(tzinfo=None) == d
    idx = pd.date_range('2023-03-01', periods=30, freq='17h')
    assert np.array_equal(_dates.to_num(idx), mdates.date2num(idx))


def test_is_date_like():
    pd = pytest.importorskip('pandas')
    yes = [datetime.datetime(2020, 1, 1), datetime.date(2020, 1, 1), np.datetime64('2020'),
           np.array(['2020-01-01'], 'datetime64[D]'), [datetime.datetime(2020, 1, 1)],
           (None, datetime.date(2020, 1, 1)), pd.Timestamp('2020'),
           pd.date_range('2020', periods=3), pd.date_range('2020', periods=3, tz='UTC'),
           pd.Series(pd.date_range('2020', periods=3)),
           np.array([datetime.datetime(2020, 1, 1)], object)]
    no = [1.0, [1, 2], np.arange(3.0), 'x', None, [], pd.Series([1.0, 2.0]),
          np.array(['a'], object)]
    assert all(_dates.is_date_like(v) for v in yes)
    assert not any(_dates.is_date_like(v) for v in no)
    assert _dates.to_num([1, 2]).tolist() == [1.0, 2.0]


# ---------------------------------------------------------- automatic dates

def test_auto_date_locator_and_formatters(ax):
    rng = np.random.default_rng(3)
    for i in range(160):
        lim = _span(rng, i)
        concise = i % 2 == 0
        ref = draw(ax, lim, axis='xy'[(i // 2) % 2], concise=concise)
        loc = _dates.AutoDateLocator()
        fmt = _dates.ConciseDateFormatter(loc) if concise else _dates.AutoDateFormatter(loc)
        check(ref, loc, fmt, (i, lim, concise))


def test_auto_date_spans(ax):
    t0 = mdates.date2num(datetime.datetime(2024, 2, 27, 13, 7, 3, 250000))
    spans = [1e-8, 3e-7, 2e-6, 1e-5, 1 / 86400, 7 / 86400, 50 / 86400, 1 / 1440, 7 / 1440, 1 / 24,
             0.2, 0.5, 1, 2.5, 6, 10, 20, 45, 70, 100, 200, 400, 800, 1500, 3000, 8000, 20000,
             60000, 200000]
    for s in spans:
        for concise in (False, True):
            ref = draw(ax, (t0, t0 + s), concise=concise)
            loc = _dates.AutoDateLocator()
            fmt = _dates.ConciseDateFormatter(loc) if concise else _dates.AutoDateFormatter(loc)
            check(ref, loc, fmt, (s, concise))


def test_auto_date_locator_options(ax):
    rng = np.random.default_rng(4)
    for i, kw in enumerate([dict(interval_multiples=False), dict(minticks=3, maxticks=7),
                            dict(minticks=8), dict(maxticks={_dates.DAILY: 5}),
                            dict(interval_multiples=False, minticks=2)] * 12):
        a = rng.uniform(-30, 60) * 365.25
        lim = (a, a + 10 ** rng.uniform(-5, 5))
        mkw = dict(kw)
        if 'maxticks' in kw and isinstance(kw['maxticks'], dict):
            mkw['maxticks'] = {mdates.DAILY: 5}
        ml = mdates.AutoDateLocator(**mkw)
        ref = draw(ax, lim, loc=ml, fmt=mdates.AutoDateFormatter(ml))
        fl = _dates.AutoDateLocator(**kw)
        check(ref, fl, _dates.AutoDateFormatter(fl), (lim, kw))


def test_concise_formatter_options(ax):
    rng = np.random.default_rng(5)
    opts = [dict(show_offset=False),
            dict(formats=['%y', '%b', '%d', '%H', '%M', '%S.%f']),
            dict(zero_formats=['', '%Y', '%B', '%d %b', '%Hh', '%M']),
            dict(offset_formats=['', '%Y', '%b %Y', '%d %b %Y', '%d %b %Y', '%d %b %Y %H:%M'])]
    for i in range(40):
        kw = opts[i % len(opts)]
        lim = _span(rng, i)
        ml = mdates.AutoDateLocator()
        ref = draw(ax, lim, loc=ml, fmt=mdates.ConciseDateFormatter(ml, **kw))
        fl = _dates.AutoDateLocator()
        check(ref, fl, _dates.ConciseDateFormatter(fl, **kw), (lim, kw))


# --------------------------------------------------------- explicit locators

LOCATORS = [
    ('YearLocator', {}, 365 * 30), ('YearLocator', dict(base=5, month=7, day=4), 365 * 60),
    ('YearLocator', dict(base=100), 365 * 900),
    ('MonthLocator', {}, 400), ('MonthLocator', dict(bymonth=[1, 7], bymonthday=15), 2000),
    ('MonthLocator', dict(interval=3), 1500), ('MonthLocator', dict(bymonthday=-1), 700),
    ('MonthLocator', dict(bymonthday=31), 700),
    ('WeekdayLocator', {}, 90), ('WeekdayLocator', dict(byweekday=(0, 4), interval=2), 60),
    ('DayLocator', {}, 20), ('DayLocator', dict(interval=3), 60),
    ('DayLocator', dict(bymonthday=[1, 15]), 200),
    ('HourLocator', {}, 1), ('HourLocator', dict(interval=5), 3),
    ('HourLocator', dict(byhour=[0, 12]), 9), ('MinuteLocator', dict(interval=7), 0.1),
    ('MinuteLocator', dict(byminute=[0, 30]), 0.4), ('SecondLocator', dict(interval=10), 0.002),
    ('SecondLocator', dict(bysecond=[0, 15, 30, 45]), 0.004),
    ('MicrosecondLocator', dict(interval=500), 5e-8),
]


def test_explicit_locators(ax):
    rng = np.random.default_rng(6)
    for name, kw, span in LOCATORS:
        for j in range(6):
            a = rng.uniform(-30, 60) * 365.25
            if j % 3 == 0:
                a = round(a)
            lim = (a, a + span * rng.uniform(0.5, 2))
            mkw = dict(kw)
            if name == 'WeekdayLocator' and 'byweekday' in kw:
                mkw['byweekday'] = tuple(mdates.WEEKDAYS[d] for d in kw['byweekday'])
            ml = getattr(mdates, name)(**mkw)
            ref = draw(ax, lim, loc=ml, fmt=mdates.AutoDateFormatter(ml))
            fl = getattr(_dates, name)(**kw)
            check(ref, fl, _dates.AutoDateFormatter(fl), (name, kw, lim))


def test_date_formatter(ax):
    rng = np.random.default_rng(7)
    for fmt in ['%Y-%m-%d %H:%M:%S.%f', '%b %d', '%j %a %p', '%d/%m/%y']:
        for _ in range(5):
            a = rng.uniform(-30, 60) * 365.25
            lim = (a, a + 10 ** rng.uniform(-3, 4))
            ref = draw(ax, lim, fmt=mdates.DateFormatter(fmt))
            check(ref, _dates.AutoDateLocator(), _dates.DateFormatter(fmt), (fmt, lim))


def test_time_zones(ax):
    zones = [datetime.timezone(datetime.timedelta(hours=5, minutes=30)),
             datetime.timezone(datetime.timedelta(hours=-8))]
    rng = np.random.default_rng(8)
    for tz in zones:
        for i in range(20):
            lim = _span(rng, i + 1)
            ml = mdates.AutoDateLocator(tz=tz)
            mf = (mdates.ConciseDateFormatter(ml, tz=tz) if i % 2 else
                  mdates.AutoDateFormatter(ml, tz=tz))
            ref = draw(ax, lim, loc=ml, fmt=mf)
            fl = _dates.AutoDateLocator(tz=tz)
            ff = (_dates.ConciseDateFormatter(fl, tz=tz) if i % 2 else
                  _dates.AutoDateFormatter(fl, tz=tz))
            check(ref, fl, ff, (tz, lim))


def test_nonsingular_and_defaults():
    for loc, mloc in [(_dates.AutoDateLocator(), mdates.AutoDateLocator()),
                      (_dates.DayLocator(interval=2), mdates.DayLocator(interval=2)),
                      (_dates.HourLocator(), mdates.HourLocator())]:
        for lim in [(10.0, 10.0), (5.0, 3.0), (np.nan, 1.0), (1.0, 1.0 + 1e-8), (0.0, 30.0)]:
            assert np.allclose(loc.nonsingular(*lim), mloc.nonsingular(*lim))
    M, m, F, f = _dates.default_locators(concise=True)
    assert isinstance(M, _dates.AutoDateLocator) and isinstance(F, _dates.ConciseDateFormatter)
    assert isinstance(m, _ticker.NullLocator)
    assert _dates.ConciseDateFormatter(M).format_ticks([], 0, 1) == ([], '')


def test_foreign_date_locator_and_formatter(ax):
    rng = np.random.default_rng(9)
    for i in range(12):
        lim = _span(rng, i)
        ref = draw(ax, lim, concise=True)
        ml = mdates.AutoDateLocator()
        loc = _ticker.get_locator(ml)
        fmt = _ticker.get_formatter(mdates.ConciseDateFormatter(ml))
        check(ref, loc, fmt, lim)
