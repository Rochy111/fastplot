"""matplotlib.ticker: where ticks go and how they read.

    import fastplot.ticker as mticker
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

The same tick values and labels as matplotlib 3.11's (tests/test_ticker.py
draws matplotlib's to compare); matplotlib's own locator and formatter
objects are accepted on fastplot axes too.
"""
from ._ticker import (AutoLocator, AutoMinorLocator, EngFormatter, FixedFormatter,  # noqa: F401
                      FixedLocator, FormatStrFormatter, Formatter, FuncFormatter, IndexLocator,
                      LatitudeFormatter, LatitudeLocator, LinearLocator, Locator,
                      LogFormatter, LogFormatterExponent, LogFormatterMathtext,
                      LogFormatterSciNotation, LogitFormatter, LogitLocator, LogLocator,
                      LongitudeFormatter, LongitudeLocator, MaxNLocator, MultipleLocator,
                      NullFormatter, NullLocator, PercentFormatter, ScalarFormatter,
                      StrMethodFormatter, SymmetricalLogLocator)

__all__ = ['Locator', 'MaxNLocator', 'AutoLocator', 'FixedLocator', 'NullLocator',
           'IndexLocator', 'LinearLocator', 'MultipleLocator', 'LogLocator',
           'SymmetricalLogLocator', 'AutoMinorLocator', 'LogitLocator', 'LongitudeLocator',
           'LatitudeLocator', 'Formatter', 'NullFormatter', 'FixedFormatter', 'FuncFormatter',
           'FormatStrFormatter', 'StrMethodFormatter', 'ScalarFormatter', 'EngFormatter',
           'PercentFormatter', 'LogFormatter', 'LogFormatterExponent', 'LogFormatterMathtext',
           'LogFormatterSciNotation', 'LogitFormatter', 'LongitudeFormatter',
           'LatitudeFormatter']
