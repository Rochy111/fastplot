"""matplotlib.dates: dates as day numbers, and their ticks.

    import fastplot.dates as mdates
    ax.plot(times, temp)                          # datetimes, datetime64, pandas
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %HZ'))

Day numbers since the epoch (1970-01-01, set_epoch), as matplotlib's;
the locators and formatters (Auto, Concise, Year ... Microsecond) pick
the ticks and labels matplotlib does (tests/test_dates.py).
"""
from ._dates import (MO, SU, TU, WE, TH, FR, SA, UTC, AutoDateFormatter,  # noqa: F401
                     AutoDateLocator, ConciseDateFormatter, DateFormatter, DateLocator,
                     DayLocator, HourLocator, MicrosecondLocator, MinuteLocator, MonthLocator,
                     RRuleLocator, SecondLocator, WeekdayLocator, YearLocator, date2num,
                     get_epoch, num2date, set_epoch)

__all__ = ['date2num', 'num2date', 'get_epoch', 'set_epoch', 'DateLocator', 'RRuleLocator',
           'YearLocator', 'MonthLocator', 'WeekdayLocator', 'DayLocator', 'HourLocator',
           'MinuteLocator', 'SecondLocator', 'MicrosecondLocator', 'AutoDateLocator',
           'DateFormatter', 'AutoDateFormatter', 'ConciseDateFormatter', 'UTC', 'MO', 'TU',
           'WE', 'TH', 'FR', 'SA', 'SU']
