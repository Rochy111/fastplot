"""Dates on axes as matplotlib.dates handles them, without dateutil.

A date is a float: days since matplotlib's default epoch 1970-01-01T00:00
UTC (`date2num` / `num2date`, computed with matplotlib's numpy expressions so
the floats are the same bits). Time zone aware inputs are converted to UTC;
naive ones are taken as UTC.

matplotlib's date locators are dateutil rrules (``rrule.between`` over a
window of the view widened by its own length), and AutoDateLocator measures
the view with ``relativedelta``. Both are reimplemented here for what the
locators use -- yearly to secondly frequencies with an interval counted from
the rule's start, bymonth / bymonthday (also negative) / byweekday / byhour /
byminute / bysecond and dateutil's defaults for the missing ones -- so the
ticks are the same datetimes, quirks included (YearLocator keeps the view
start's microseconds, which can drop its first tick, as matplotlib's does).
Rules run in the locator's time zone wall clock, as matplotlib's rrulewrapper
does; with the default UTC there is no daylight saving to care about.

Formatters follow fastplot's protocol (format_ticks(locs, vmin, vmax) ->
(labels, offset_text), see _ticker): DateFormatter (strftime),
AutoDateFormatter (format from the locator's unit, matplotlib's
date.autoformatter rcParams) and ConciseDateFormatter (its offset string is
the offset text; vmin > vmax means an inverted axis).
"""
import calendar
import datetime

import numpy as np

from . import _ticker

UTC = datetime.timezone.utc
_EPOCH = '1970-01-01T00:00:00'          # rcParams['date.epoch']

YEARLY, MONTHLY, WEEKLY, DAILY, HOURLY, MINUTELY, SECONDLY = range(7)   # dateutil's
MICROSECONDLY = SECONDLY + 1
MO, TU, WE, TH, FR, SA, SU = range(7)

HOURS_PER_DAY = 24.
MIN_PER_HOUR = 60.
SEC_PER_MIN = 60.
MONTHS_PER_YEAR = 12.
DAYS_PER_WEEK = 7.
DAYS_PER_MONTH = 30.
DAYS_PER_YEAR = 365.0
MINUTES_PER_DAY = MIN_PER_HOUR * HOURS_PER_DAY
SEC_PER_HOUR = SEC_PER_MIN * MIN_PER_HOUR
SEC_PER_DAY = SEC_PER_HOUR * HOURS_PER_DAY
SEC_PER_WEEK = SEC_PER_DAY * DAYS_PER_WEEK
MUSECONDS_PER_DAY = 1e6 * SEC_PER_DAY


def get_epoch():
    return _EPOCH


def set_epoch(epoch):
    """matplotlib.dates.set_epoch (call it before converting any date)."""
    global _EPOCH
    _EPOCH = epoch


def _get_tzinfo(tz=None):
    if tz is None or tz == 'UTC':
        return UTC
    if isinstance(tz, str):
        import zoneinfo
        return zoneinfo.ZoneInfo(tz)
    if isinstance(tz, datetime.tzinfo):
        return tz
    raise TypeError('tz must be string or tzinfo subclass, not %r.' % (tz,))


# --------------------------------------------------------------- conversion

def _unpack(x):
    """cbook._unpack_to_numpy: pandas / xarray objects to numpy."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, 'to_numpy'):
        return x.to_numpy()
    if hasattr(x, 'values'):
        v = x.values
        if isinstance(v, np.ndarray):
            return v
    return x


def _dt64_to_ordinalf(d):
    dseconds = d.astype('datetime64[s]')
    extra = (d - dseconds).astype('timedelta64[ns]')
    t0 = np.datetime64(get_epoch(), 's')
    dt = (dseconds - t0).astype(np.float64)
    dt += extra.astype(np.float64) / 1.0e9
    dt = dt / SEC_PER_DAY
    nat = np.isnat(d)
    if nat.any():
        dt[nat] = np.nan
    return dt


def date2num(d):
    """matplotlib.dates.date2num: datetime / date / numpy datetime64 (any
    unit) / pandas Timestamp, DatetimeIndex, Series, or sequences of them
    -> days since the epoch (float, or a float array). Aware datetimes are
    converted to UTC."""
    d = _unpack(d)
    iterable = np.iterable(d)
    if not iterable:
        d = [d]
    masked = np.ma.is_masked(d)
    mask = np.ma.getmask(d)
    d = np.asarray(d)
    if not np.issubdtype(d.dtype, np.datetime64):
        if not d.size:
            return d
        if getattr(d.flat[0], 'tzinfo', None) is not None:
            d = np.asarray([v.astimezone(UTC).replace(tzinfo=None)
                            if getattr(v, 'tzinfo', None) is not None else v
                            for v in d.ravel()]).reshape(d.shape)
        d = d.astype('datetime64[us]')
    d = np.ma.masked_array(d, mask=mask) if masked else d
    d = _dt64_to_ordinalf(d)
    return d if iterable else d[0]


def _from_ordinalf(x, tz=None):
    tz = _get_tzinfo(tz)
    dt = np.datetime64(get_epoch()) + np.timedelta64(int(np.round(x * MUSECONDS_PER_DAY)), 'us')
    if dt < np.datetime64('0001-01-01') or dt >= np.datetime64('10000-01-01'):
        raise ValueError('Date ordinal %s converts to %s (using epoch %s), but dates must '
                         'be between year 0001 and 9999.' % (x, dt, get_epoch()))
    dt = dt.tolist().replace(tzinfo=UTC).astimezone(tz)
    if np.abs(x) > 70 * 365:                    # float64 resolution: round to 20 us
        ms = round(dt.microsecond / 20) * 20
        if ms == 1000000:
            dt = dt.replace(microsecond=0) + datetime.timedelta(seconds=1)
        else:
            dt = dt.replace(microsecond=ms)
    return dt


_from_ordinalf_v = np.vectorize(_from_ordinalf, otypes='O')


def num2date(x, tz=None):
    """matplotlib.dates.num2date: days since the epoch -> aware datetime(s)
    in *tz* (default UTC); a list for sequences."""
    if np.ndim(x) == 0:
        return _from_ordinalf(float(x), tz)
    return _from_ordinalf_v(x, _get_tzinfo(tz)).tolist()


def is_date_like(x):
    """Whether *x* (a value, an array, a sequence, a pandas object) holds
    dates: datetime / date, numpy datetime64, pandas Timestamps."""
    if isinstance(x, (datetime.date, np.datetime64)):
        return True
    if isinstance(x, (str, bytes)) or x is None:
        return False
    if not isinstance(x, np.ndarray):
        if hasattr(x, 'to_numpy') or hasattr(x, 'values'):
            v = _unpack(x)
            if v is not x:
                return is_date_like(v)
        if isinstance(x, (list, tuple)):
            for v in x:
                if v is not None:
                    return is_date_like(v)
            return False
        return False
    if x.dtype.kind == 'M':
        return True
    if x.dtype == object and x.size:
        for v in x.flat:
            if v is not None:
                return isinstance(v, (datetime.date, np.datetime64))
    return False


def to_num(values):
    """Axis coordinates: dates as date2num floats, anything else as floats."""
    if is_date_like(values):
        return np.asarray(date2num(values), float)
    return np.asarray(values, float)


# ----------------------------------------- relativedelta / rrule (dateutil's)

def _sign(x):
    return 1 if x >= 0 else -1


def _split_months(months):
    if abs(months) > 11:
        s = _sign(months)
        div, mod = divmod(months * s, 12)
        return div * s, mod * s
    return 0, months


class _RelDelta:
    """dateutil.relativedelta.relativedelta(dt1, dt2) and adding it back."""

    def __init__(self, dt1, dt2):
        months = (dt1.year - dt2.year) * 12 + (dt1.month - dt2.month)
        self.years, self.months = _split_months(months)
        self.days = self.hours = self.minutes = self.seconds = self.microseconds = 0
        dtm = self.add_to(dt2)
        if dt1 < dt2:
            compare, inc = (lambda a, b: a > b), 1
        else:
            compare, inc = (lambda a, b: a < b), -1
        while compare(dt1, dtm):
            months += inc
            self.years, self.months = _split_months(months)
            dtm = self.add_to(dt2)
        delta = dt1 - dtm
        self.seconds = delta.seconds + delta.days * 86400
        self.microseconds = delta.microseconds
        self._fix()

    def _fix(self):
        for small, big, n in (('microseconds', 'seconds', 1000000), ('seconds', 'minutes', 60),
                              ('minutes', 'hours', 60), ('hours', 'days', 24)):
            v = getattr(self, small)
            if abs(v) > n - 1:
                s = _sign(v)
                div, mod = divmod(v * s, n)
                setattr(self, small, mod * s)
                setattr(self, big, getattr(self, big) + div * s)
        if abs(self.months) > 11:
            s = _sign(self.months)
            div, mod = divmod(self.months * s, 12)
            self.months = mod * s
            self.years += div * s

    def negated(self):
        out = object.__new__(_RelDelta)
        for k in ('years', 'months', 'days', 'hours', 'minutes', 'seconds', 'microseconds'):
            setattr(out, k, -getattr(self, k))
        return out

    def add_to(self, other, sign=1):
        year = other.year + sign * self.years
        month = other.month
        months = sign * self.months
        if months:
            month += months
            if month > 12:
                year += 1
                month -= 12
            elif month < 1:
                year -= 1
                month += 12
        day = min(calendar.monthrange(year, month)[1], other.day)
        return other.replace(year=year, month=month, day=day) + datetime.timedelta(
            days=sign * self.days, hours=sign * self.hours, minutes=sign * self.minutes,
            seconds=sign * self.seconds, microseconds=sign * self.microseconds)


def _tuple(v):
    """A by-rule as dateutil keeps it: sorted unique ints (weekday objects
    with a .weekday, such as dateutil's MO, count as their weekday)."""
    if v is None:
        return None
    if not isinstance(v, (list, tuple, range, set, frozenset, np.ndarray)):
        v = (v,)
    return tuple(sorted({int(getattr(x, 'weekday', x)) for x in v}))


class _RRule:
    """dateutil's rrule for the locators: freq (YEARLY..SECONDLY), interval,
    bymonth, bymonthday, byweekday, byhour, byminute, bysecond; dtstart
    (microseconds dropped, as dateutil does) and until set by the locators;
    between(after, before, inc=True). Naive datetimes (the zone's wall
    clock)."""

    def __init__(self, freq, **kw):
        kw['freq'] = freq
        self._construct = kw

    def set(self, **kw):
        self._construct.update(kw)

    @property
    def freq(self):
        return self._construct['freq']

    @property
    def interval(self):
        return self._construct.get('interval', 1)

    def _compile(self):
        c = self._construct
        freq = c['freq']
        dtstart = c['dtstart'].replace(microsecond=0)
        bymonth, bymonthday, byweekday = c.get('bymonth'), c.get('bymonthday'), c.get('byweekday')
        if bymonthday is None and byweekday is None:    # dateutil's defaults
            if freq == YEARLY:
                if bymonth is None:
                    bymonth = dtstart.month
                bymonthday = dtstart.day
            elif freq == MONTHLY:
                bymonthday = dtstart.day
            elif freq == WEEKLY:
                byweekday = dtstart.weekday()
        self.bymonth = _tuple(bymonth)
        md = _tuple(bymonthday) or ()
        self.bymd = tuple(x for x in md if x > 0)
        self.bynmd = tuple(x for x in md if x < 0)
        self.byweekday = _tuple(byweekday)
        bh, bm, bs = c.get('byhour'), c.get('byminute'), c.get('bysecond')
        self.byhour = _tuple(bh) if bh is not None else ((dtstart.hour,) if freq < HOURLY else None)
        self.byminute = _tuple(bm) if bm is not None else ((dtstart.minute,) if freq < MINUTELY else None)
        self.bysecond = _tuple(bs) if bs is not None else ((dtstart.second,) if freq < SECONDLY else None)
        return dtstart

    def _day_ok(self, d):
        if self.bymonth and d.month not in self.bymonth:
            return False
        if self.byweekday and d.weekday() not in self.byweekday:
            return False
        if self.bymd or self.bynmd:
            n = calendar.monthrange(d.year, d.month)[1]
            if d.day not in self.bymd and (d.day - n - 1) not in self.bynmd:
                return False
        return True

    def _month_days(self, year, month):
        n = calendar.monthrange(year, month)[1]
        if self.bymd or self.bynmd:
            days = sorted({x for x in self.bymd if x <= n} | {n + 1 + x for x in self.bynmd if x >= -n})
        else:
            days = range(1, n + 1)
        out = []
        for day in days:
            d = datetime.date(year, month, day)
            if not self.byweekday or d.weekday() in self.byweekday:
                out.append(d)
        return out

    def _times(self):
        return sorted(datetime.time(h, m, s) for h in self.byhour for m in self.byminute
                      for s in self.bysecond)

    def _iter(self, stop):
        """Occurrences >= dtstart, in order, until one passes *stop*."""
        dtstart = self._compile()
        freq, interval = self.freq, self.interval
        if freq <= DAILY:
            times = self._times()
            for d in self._days(dtstart, interval, stop):
                for t in times:
                    res = datetime.datetime.combine(d, t)
                    if res > stop:
                        return
                    if res >= dtstart:
                        yield res
            return
        unit = {HOURLY: datetime.timedelta(hours=1), MINUTELY: datetime.timedelta(minutes=1),
                SECONDLY: datetime.timedelta(seconds=1)}[freq]
        step = unit * interval
        t = dtstart.replace(**{HOURLY: dict(minute=0, second=0), MINUTELY: dict(second=0),
                               SECONDLY: {}}[freq])
        subs = ([(m, s) for m in self.byminute for s in self.bysecond] if freq == HOURLY else
                [s for s in self.bysecond] if freq == MINUTELY else None)
        while t <= stop:
            skip_to = None
            if not self._day_ok(t.date()):
                skip_to = datetime.datetime.combine(t.date() + datetime.timedelta(days=1),
                                                    datetime.time())
            elif self.byhour and t.hour not in self.byhour:
                skip_to = t.replace(minute=0, second=0) + datetime.timedelta(hours=1) \
                    if freq > HOURLY else None
            elif freq >= MINUTELY and self.byminute and t.minute not in self.byminute:
                skip_to = t.replace(second=0) + datetime.timedelta(minutes=1) \
                    if freq > MINUTELY else None
            elif freq == SECONDLY and self.bysecond and t.second not in self.bysecond:
                skip_to = None
            else:
                if freq == HOURLY:
                    cands = [t.replace(minute=m, second=s) for m, s in subs]
                elif freq == MINUTELY:
                    cands = [t.replace(second=s) for s in subs]
                else:
                    cands = [t]
                for res in cands:
                    if res > stop:
                        return
                    if res >= dtstart:
                        yield res
                t = t + step
                continue
            if skip_to is None or skip_to <= t:
                t = t + step
            else:                                    # whole periods to the boundary
                k = -((t - skip_to) // step)
                t = t + step * max(k, 1)

    def _days(self, dtstart, interval, stop):
        freq = self.freq
        if freq == YEARLY:
            year = dtstart.year
            while year <= min(stop.year, 9999):
                for month in (self.bymonth or range(1, 13)):
                    yield from self._month_days(year, month)
                year += interval
        elif freq == MONTHLY:
            k = dtstart.year * 12 + dtstart.month - 1
            while k // 12 <= min(stop.year, 9999):
                year, month = divmod(k, 12)
                if not self.bymonth or month + 1 in self.bymonth:
                    yield from self._month_days(year, month + 1)
                k += interval
        else:
            d = dtstart.date()
            if freq == WEEKLY:
                d = d - datetime.timedelta(days=d.weekday())      # wkst Monday
                n = 7
            else:
                n = 1
            stop_d = stop.date()
            while d <= stop_d:
                for i in range(n):
                    di = d + datetime.timedelta(days=i)
                    if self._day_ok(di):
                        yield di
                try:
                    d = d + datetime.timedelta(days=n * interval)
                except OverflowError:
                    return

    def between(self, after, before, inc=True):
        stop = min(before, self._construct.get('until') or before)
        out = []
        for res in self._iter(stop):
            if (res >= after) if inc else (res > after):
                if inc or res < before:
                    out.append(res)
        return out


# ----------------------------------------------------------------- locators

def _naive(dt, tz):
    """An aware datetime as the wall clock of *tz* (naive ones are UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(tz).replace(tzinfo=None)


class DateLocator(_ticker.Locator):
    """matplotlib.dates.DateLocator: tick_values(vmin, vmax) takes date
    numbers (or datetimes) and returns date numbers."""
    hms0d = {'byhour': 0, 'byminute': 0, 'bysecond': 0}

    def __init__(self, tz=None):
        self.tz = _get_tzinfo(tz)

    def set_tzinfo(self, tz):
        self.tz = _get_tzinfo(tz)

    def _dt_range(self, vmin, vmax):
        if isinstance(vmin, datetime.date) and isinstance(vmax, datetime.date):
            if not isinstance(vmin, datetime.datetime):
                vmin = datetime.datetime.fromordinal(vmin.toordinal())
            if not isinstance(vmax, datetime.datetime):
                vmax = datetime.datetime.fromordinal(vmax.toordinal())
            return _naive(vmin, self.tz), _naive(vmax, self.tz)
        vmin, vmax = sorted((float(vmin), float(vmax)))
        return _naive(num2date(vmin, self.tz), self.tz), _naive(num2date(vmax, self.tz), self.tz)

    def tick_values(self, vmin, vmax, tick_space=None, major=None):
        try:
            dmin, dmax = self._dt_range(vmin, vmax)
        except ValueError:
            return np.array([], float)
        return self._ticks(dmin, dmax)

    def _nums(self, dts):
        return np.asarray(date2num([d.replace(tzinfo=self.tz) for d in dts]), float)

    def _get_unit(self):
        return 1

    def _get_interval(self):
        return 1

    def nonsingular(self, vmin, vmax):
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return (date2num(datetime.date(1970, 1, 1)), date2num(datetime.date(1970, 1, 2)))
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        unit = self._get_unit()
        interval = self._get_interval()
        if abs(vmax - vmin) < 1e-6:
            vmin -= 2 * unit * interval
            vmax += 2 * unit * interval
        return vmin, vmax


def _unit_generic(freq):
    return {YEARLY: DAYS_PER_YEAR, MONTHLY: DAYS_PER_MONTH, WEEKLY: DAYS_PER_WEEK, DAILY: 1.0,
            HOURLY: 1.0 / HOURS_PER_DAY, MINUTELY: 1.0 / MINUTES_PER_DAY,
            SECONDLY: 1.0 / SEC_PER_DAY}.get(freq, -1)


class RRuleLocator(DateLocator):
    """matplotlib.dates.RRuleLocator: the rule's occurrences in the view,
    the rule restarted at the view start minus the view's length."""

    def __init__(self, rule, tz=None):
        super().__init__(tz)
        self.rule = rule

    def _ticks(self, dmin, dmax):
        start, stop = self._create_rrule(dmin, dmax)
        dates = self.rule.between(start, stop, True)
        if not dates:
            return self._nums([dmin, dmax])
        return self._nums(dates)

    def _create_rrule(self, vmin, vmax):
        delta = _RelDelta(vmax, vmin)
        try:
            start = delta.add_to(vmin, -1)
        except (ValueError, OverflowError):
            start = datetime.datetime(1, 1, 1)
        try:
            stop = delta.add_to(vmax, 1)
        except (ValueError, OverflowError):
            stop = datetime.datetime(9999, 12, 31, 23, 59, 59)
        self.rule.set(dtstart=start, until=stop)
        return vmin, vmax

    def _get_unit(self):
        return _unit_generic(self.rule.freq)

    def _get_interval(self):
        return self.rule.interval


class YearLocator(RRuleLocator):
    """Every *base* years on (month, day), from the multiple of base at or
    below the view start to the one at or above its end."""

    def __init__(self, base=1, month=1, day=1, tz=None):
        super().__init__(_RRule(YEARLY, interval=base, bymonth=month, bymonthday=day,
                                **self.hms0d), tz)
        self.base = _ticker._Edge(base, 0)

    def _create_rrule(self, vmin, vmax):
        ymin = max(self.base.le(vmin.year) * self.base.step, 1)
        ymax = min(self.base.ge(vmax.year) * self.base.step, 9999)
        c = self.rule._construct
        start = vmin.replace(year=int(ymin), month=c.get('bymonth', 1), day=c.get('bymonthday', 1),
                             hour=0, minute=0, second=0)          # microsecond kept, as matplotlib
        stop = start.replace(year=int(ymax))
        self.rule.set(dtstart=start, until=stop)
        return start, stop


class MonthLocator(RRuleLocator):
    """Months in *bymonth* (default all) on *bymonthday*, every *interval*."""

    def __init__(self, bymonth=None, bymonthday=1, interval=1, tz=None):
        if bymonth is None:
            bymonth = range(1, 13)
        super().__init__(_RRule(MONTHLY, bymonth=bymonth, bymonthday=bymonthday,
                                interval=interval, **self.hms0d), tz)


class WeekdayLocator(RRuleLocator):
    """Days that are *byweekday* (0 = Monday ... 6; MO..SU), every *interval*."""

    def __init__(self, byweekday=1, interval=1, tz=None):
        super().__init__(_RRule(DAILY, byweekday=byweekday, interval=interval, **self.hms0d), tz)


class DayLocator(RRuleLocator):
    """Days of the month in *bymonthday* (default all), every *interval*."""

    def __init__(self, bymonthday=None, interval=1, tz=None):
        if interval != int(interval) or interval < 1:
            raise ValueError('interval must be an integer greater than 0')
        if bymonthday is None:
            bymonthday = range(1, 32)
        super().__init__(_RRule(DAILY, bymonthday=bymonthday, interval=interval,
                                **self.hms0d), tz)


class HourLocator(RRuleLocator):
    """Hours in *byhour* (default all), every *interval*."""

    def __init__(self, byhour=None, interval=1, tz=None):
        if byhour is None:
            byhour = range(24)
        super().__init__(_RRule(HOURLY, byhour=byhour, interval=interval, byminute=0,
                                bysecond=0), tz)


class MinuteLocator(RRuleLocator):
    """Minutes in *byminute* (default all), every *interval*."""

    def __init__(self, byminute=None, interval=1, tz=None):
        if byminute is None:
            byminute = range(60)
        super().__init__(_RRule(MINUTELY, byminute=byminute, interval=interval, bysecond=0), tz)


class SecondLocator(RRuleLocator):
    """Seconds in *bysecond* (default all), every *interval*."""

    def __init__(self, bysecond=None, interval=1, tz=None):
        if bysecond is None:
            bysecond = range(60)
        super().__init__(_RRule(SECONDLY, bysecond=bysecond, interval=interval), tz)


class MicrosecondLocator(DateLocator):
    """Every *interval* microseconds: a MultipleLocator in microseconds since
    the view start's day (precise only near the epoch, as matplotlib's)."""

    def __init__(self, interval=1, tz=None):
        super().__init__(tz=tz)
        self._interval = interval
        self._wrapped = _ticker.MultipleLocator(interval)

    def _ticks(self, dmin, dmax):
        nmin, nmax = self._nums([dmin, dmax])
        t0 = np.floor(nmin)
        nmin = (nmin - t0) * MUSECONDS_PER_DAY
        nmax = (nmax - t0) * MUSECONDS_PER_DAY
        return self._wrapped.tick_values(nmin, nmax) / MUSECONDS_PER_DAY + t0

    def _get_unit(self):
        return 1. / MUSECONDS_PER_DAY

    def _get_interval(self):
        return self._interval


class AutoDateLocator(DateLocator):
    """matplotlib.dates.AutoDateLocator: the coarsest frequency (yearly ...
    microsecondly) with at least *minticks* periods in the view, the
    smallest of its intervald intervals giving at most maxticks[freq] ticks;
    with interval_multiples, ticks on multiples of the interval (Jan/Apr/...,
    days 1, 8, 15, 22, hours 0, 6, 12, 18). get_locator(dmin, dmax) returns
    the locator it picks; its frequency then sets AutoDateFormatter's format."""

    def __init__(self, tz=None, minticks=5, maxticks=None, interval_multiples=True):
        super().__init__(tz=tz)
        self._freq = YEARLY
        self._freqs = [YEARLY, MONTHLY, DAILY, HOURLY, MINUTELY, SECONDLY, MICROSECONDLY]
        self.minticks = minticks
        self.maxticks = {YEARLY: 11, MONTHLY: 12, DAILY: 11, HOURLY: 12, MINUTELY: 11,
                         SECONDLY: 11, MICROSECONDLY: 8}
        if maxticks is not None:
            try:
                self.maxticks.update(maxticks)
            except TypeError:
                self.maxticks = dict.fromkeys(self._freqs, maxticks)
        self.interval_multiples = interval_multiples
        self.intervald = {
            YEARLY: [1, 2, 4, 5, 10, 20, 40, 50, 100, 200, 400, 500, 1000, 2000, 4000, 5000,
                     10000],
            MONTHLY: [1, 2, 3, 4, 6],
            DAILY: [1, 2, 3, 7, 14, 21],
            HOURLY: [1, 2, 3, 4, 6, 12],
            MINUTELY: [1, 5, 10, 15, 30],
            SECONDLY: [1, 5, 10, 15, 30],
            MICROSECONDLY: [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000,
                            50000, 100000, 200000, 500000, 1000000]}
        if interval_multiples:
            self.intervald[DAILY] = [1, 2, 4, 7, 14]
        self._byranges = [None, range(1, 13), range(1, 32), range(0, 24), range(0, 60),
                          range(0, 60), None]

    def _ticks(self, dmin, dmax):
        return self.get_locator(dmin, dmax)._ticks(dmin, dmax)

    def nonsingular(self, vmin, vmax):
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return (date2num(datetime.date(1970, 1, 1)), date2num(datetime.date(1970, 1, 2)))
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        if vmin == vmax:
            vmin = vmin - DAYS_PER_YEAR * 2
            vmax = vmax + DAYS_PER_YEAR * 2
        return vmin, vmax

    def _get_unit(self):
        if self._freq in [MICROSECONDLY]:
            return 1. / MUSECONDS_PER_DAY
        return _unit_generic(self._freq)

    def get_locator(self, dmin, dmax):
        if not isinstance(dmin, datetime.datetime):
            dmin, dmax = self._dt_range(dmin, dmax)
        elif dmin.tzinfo is not None or dmax.tzinfo is not None:
            dmin, dmax = _naive(dmin, self.tz), _naive(dmax, self.tz)
        delta = _RelDelta(dmax, dmin)
        tdelta = dmax - dmin
        if dmin > dmax:
            delta = delta.negated()
            tdelta = -tdelta
        num_years = float(delta.years)
        num_months = num_years * MONTHS_PER_YEAR + delta.months
        num_days = tdelta.days
        num_hours = num_days * HOURS_PER_DAY + delta.hours
        num_minutes = num_hours * MIN_PER_HOUR + delta.minutes
        num_seconds = np.floor(tdelta.total_seconds())
        num_us = np.floor(tdelta.total_seconds() * 1e6)
        nums = [num_years, num_months, num_days, num_hours, num_minutes, num_seconds, num_us]
        use_rrule = [True] * 6 + [False]
        byranges = [None, 1, 1, 0, 0, 0, None]
        for i, (freq, num) in enumerate(zip(self._freqs, nums)):
            if num < self.minticks:                     # too few: a finer frequency
                byranges[i] = None
                continue
            for interval in self.intervald[freq]:
                if num <= interval * (self.maxticks[freq] - 1):
                    break
            self._freq = freq
            if self._byranges[i] and self.interval_multiples:
                byranges[i] = self._byranges[i][::interval]
                if i in (DAILY, WEEKLY):                # matplotlib's test (index 2 = days)
                    if interval == 14:
                        byranges[i] = [1, 15]
                    elif interval == 7:
                        byranges[i] = [1, 8, 15, 22]
                interval = 1
            else:
                byranges[i] = self._byranges[i]
            break
        else:
            interval = 1
        if freq == YEARLY and self.interval_multiples:
            return YearLocator(interval, tz=self.tz)
        if use_rrule[i]:
            _, bymonth, bymonthday, byhour, byminute, bysecond, _ = byranges
            rule = _RRule(self._freq, interval=interval, dtstart=dmin, until=dmax,
                          bymonth=bymonth, bymonthday=bymonthday, byhour=byhour,
                          byminute=byminute, bysecond=bysecond)
            return RRuleLocator(rule, tz=self.tz)
        return MicrosecondLocator(interval, tz=self.tz)


# --------------------------------------------------------------- formatters

class DateFormatter(_ticker.Formatter):
    """strftime of the date in *tz* (default UTC)."""

    def __init__(self, fmt, tz=None, *, usetex=None):
        self.tz = _get_tzinfo(tz)
        self.fmt = fmt

    def __call__(self, x, pos=0):
        return num2date(x, self.tz).strftime(self.fmt)

    def set_tzinfo(self, tz):
        self.tz = _get_tzinfo(tz)


AUTOFORMATS = {DAYS_PER_YEAR: '%Y', DAYS_PER_MONTH: '%Y-%m', 1: '%Y-%m-%d',
               1 / HOURS_PER_DAY: '%m-%d %H', 1 / MINUTES_PER_DAY: '%d %H:%M',
               1 / SEC_PER_DAY: '%H:%M:%S', 1 / MUSECONDS_PER_DAY: '%M:%S.%f'}
# rcParams date.autoformatter.year ... .microsecond


class AutoDateFormatter(_ticker.Formatter):
    """matplotlib.dates.AutoDateFormatter: the first format of *scaled*
    (unit in days -> strftime format or f(x, pos)) whose unit is at least
    the locator's (an AutoDateLocator's: that of the frequency it picked
    for the view)."""

    def __init__(self, locator, tz=None, defaultfmt='%Y-%m-%d', *, usetex=None):
        self._locator = locator
        self._tz = tz
        self.defaultfmt = defaultfmt
        self.scaled = dict(AUTOFORMATS)

    def format_ticks(self, locs, vmin=None, vmax=None):
        if (isinstance(self._locator, AutoDateLocator) and vmin is not None and vmax is not None
                and np.isfinite(vmin) and np.isfinite(vmax)):
            try:
                self._locator.get_locator(vmin, vmax)    # its frequency for this view
            except ValueError:
                pass
        return super().format_ticks(locs, vmin, vmax)

    def __call__(self, x, pos=None):
        try:
            unit = float(self._locator._get_unit())
        except AttributeError:
            unit = 1
        fmt = next((f for scale, f in sorted(self.scaled.items()) if scale >= unit),
                   self.defaultfmt)
        if isinstance(fmt, str):
            return DateFormatter(fmt, self._tz)(x, pos)
        if callable(fmt):
            return fmt(x, pos)
        raise TypeError('Unexpected type passed to %r.' % self)


class ConciseDateFormatter(_ticker.Formatter):
    """matplotlib.dates.ConciseDateFormatter: labels as short as the ticks
    allow ('Jan', '15', '12:00'), the zero of each level written one level up
    ('2024', 'Feb', 'Mar-01'), and the rest of the date as the offset text
    (last tick's, first one's on an inverted axis)."""

    def __init__(self, locator=None, tz=None, formats=None, offset_formats=None,
                 zero_formats=None, show_offset=True, *, usetex=None):
        self._locator = locator
        self._tz = tz
        self.defaultfmt = '%Y'
        if formats:
            if len(formats) != 6:
                raise ValueError('formats argument must be a list of 6 format strings (or None)')
            self.formats = formats
        else:
            self.formats = ['%Y', '%b', '%d', '%H:%M', '%H:%M', '%S.%f']
        if zero_formats:
            if len(zero_formats) != 6:
                raise ValueError('zero_formats argument must be a list of 6 format strings '
                                 '(or None)')
            self.zero_formats = zero_formats
        elif formats:
            self.zero_formats = [''] + self.formats[:-1]
        else:
            self.zero_formats = [''] + self.formats[:-1]
            self.zero_formats[3] = '%b-%d'
        if offset_formats:
            if len(offset_formats) != 6:
                raise ValueError('offset_formats argument must be a list of 6 format strings '
                                 '(or None)')
            self.offset_formats = offset_formats
        else:
            self.offset_formats = ['', '%Y', '%Y-%b', '%Y-%b-%d', '%Y-%b-%d', '%Y-%b-%d %H:%M']
        self._offset_string = ''
        self.show_offset = show_offset

    def __call__(self, x, pos=None):
        return DateFormatter(self.defaultfmt, self._tz)(x, pos=pos)

    def format_ticks(self, locs, vmin=None, vmax=None):
        locs = np.asarray(locs, float)
        inverted = vmin is not None and vmax is not None and vmax < vmin
        if vmin is not None:
            self._view = (vmin, vmax)
        if not locs.size:
            self._offset_string = ''
            return [], ''
        tickdatetime = [num2date(v, tz=self._tz) for v in locs]
        tickdate = np.array([t.timetuple()[:6] for t in tickdatetime])
        fmts, zerofmts, offsetfmts = self.formats, self.zero_formats, self.offset_formats
        show_offset = self.show_offset
        for level in range(5, -1, -1):              # the finest field that varies
            unique = np.unique(tickdate[:, level])
            if len(unique) > 1:
                if level < 2 and np.any(unique == 1):
                    show_offset = False
                break
            elif level == 0:
                level = 5
        zerovals = [0, 1, 1, 0, 0, 0, 0]
        labels = [''] * len(tickdate)
        for nn in range(len(tickdate)):
            if level < 5:
                fmt = zerofmts[level] if tickdate[nn][level] == zerovals[level] else fmts[level]
            else:
                fmt = (zerofmts[level] if tickdatetime[nn].second == tickdatetime[nn].microsecond
                       == 0 else fmts[level])
            labels[nn] = tickdatetime[nn].strftime(fmt)
        if level >= 5:                              # trim the zeros all labels share
            trailing = min((len(s) - len(s.rstrip('0')) for s in labels if '.' in s),
                           default=None)
            if trailing:
                for nn in range(len(labels)):
                    if '.' in labels[nn]:
                        labels[nn] = labels[nn][:-trailing].rstrip('.')
        if show_offset:
            t = tickdatetime[0] if inverted else tickdatetime[-1]
            self._offset_string = t.strftime(offsetfmts[level])
        else:
            self._offset_string = ''
        return labels, self._offset_string

    def get_offset(self):
        return self._offset_string


def default_locators(tz=None, concise=False, interval_multiples=True):
    """What matplotlib's date converter puts on a date axis (rcParams
    date.converter 'auto' or 'concise'): (major locator, minor locator,
    major formatter, minor formatter)."""
    loc = AutoDateLocator(tz=tz, interval_multiples=interval_multiples)
    fmt = ConciseDateFormatter(loc, tz=tz) if concise else AutoDateFormatter(loc, tz=tz)
    return loc, _ticker.NullLocator(), fmt, _ticker.NullFormatter()
