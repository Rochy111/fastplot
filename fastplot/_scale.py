"""Axis scales as matplotlib.scale defines them: linear, log, symlog, logit
and function.

A scale is the axis' coordinate transform (data -> the space where ticks are
spaced evenly and where the axis is linear on screen) plus the locators and
formatters matplotlib's ``set_default_locators_and_formatters`` installs for
it. ``transform`` / ``inverse`` are vectorized numpy and compute exactly
matplotlib's expressions (np.log10 for base 10, exp(x log b) back, -1000 for
clipped non-positive values), so that a point lands on the same pixel.

`get_scale` takes a name ('linear', 'log', 'symlog', 'logit', 'function')
with matplotlib's keyword arguments, a fastplot scale, or a matplotlib
ScaleBase object (anything with get_transform()), which is wrapped: its
transform, its range limits and the locators/formatters it would set.
"""
import numpy as np

from . import _ticker


class ScaleBase:
    """The scale protocol: name, transform(a), inverse(a),
    limit_range_for_scale(vmin, vmax, minpos) and default_locators()."""
    name = None

    def transform(self, a):
        raise NotImplementedError

    def inverse(self, a):
        raise NotImplementedError

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return vmin, vmax

    def default_locators(self, minor_visible=False):
        """(major locator, minor locator, major formatter, minor formatter),
        matplotlib's defaults; minor_visible is rcParams[x/ytick.minor.visible]."""
        raise NotImplementedError

    def val_in_range(self, val):
        arr = np.asarray(val)
        res = np.isfinite(arr)
        return bool(res) if arr.ndim == 0 else res


def _arr(a):
    return np.array(a, dtype=float, ndmin=1), np.ndim(a) == 0


def _out(out, scalar):
    return out[0] if scalar else out


def _linear_defaults(minor_visible):
    minor = _ticker.AutoMinorLocator() if minor_visible else _ticker.NullLocator()
    return _ticker.AutoLocator(), minor, _ticker.ScalarFormatter(), _ticker.NullFormatter()


class LinearScale(ScaleBase):
    """The identity: AutoLocator + ScalarFormatter, no minor ticks."""
    name = 'linear'

    def transform(self, a):
        return np.asarray(a, float)

    inverse = transform

    def default_locators(self, minor_visible=False):
        return _linear_defaults(minor_visible)


class FuncScale(ScaleBase):
    """matplotlib's FuncScale: functions=(forward, inverse), linear ticks."""
    name = 'function'

    def __init__(self, functions):
        forward, inverse = functions
        if not (callable(forward) and callable(inverse)):
            raise ValueError('arguments to FuncTransform must be functions')
        self._forward, self._inverse = forward, inverse

    def transform(self, a):
        return self._forward(np.asarray(a, float))

    def inverse(self, a):
        return self._inverse(np.asarray(a, float))

    def default_locators(self, minor_visible=False):
        return _linear_defaults(minor_visible)


_LOG = {np.e: np.log, 2: np.log2, 10: np.log10}
_EXP = {np.e: np.exp, 2: np.exp2}


class LogScale(ScaleBase):
    """matplotlib's LogScale: log_base, non-positive values clipped to -1000
    (nonpositive='clip') or left invalid ('mask': -inf at 0, NaN below, as
    matplotlib's transform gives before masking)."""
    name = 'log'

    def __init__(self, base=10, subs=None, nonpositive='clip'):
        if base <= 0 or base == 1:
            raise ValueError('The log base cannot be <= 0 or == 1')
        if nonpositive not in ('clip', 'mask'):
            raise ValueError("nonpositive must be 'clip' or 'mask'")
        self.base = base
        self.subs = subs
        self.nonpositive = nonpositive

    def transform(self, a):
        a, scalar = _arr(a)
        with np.errstate(divide='ignore', invalid='ignore'):
            f = _LOG.get(self.base)
            out = f(a) if f else np.log(a) / np.log(self.base)
            if self.nonpositive == 'clip':
                out[a <= 0] = -1000
        return _out(out, scalar)

    def inverse(self, a):
        a, scalar = _arr(a)
        f = _EXP.get(self.base)
        return _out(f(a) if f else np.exp(a * np.log(self.base)), scalar)

    def limit_range_for_scale(self, vmin, vmax, minpos):
        if not np.isfinite(minpos):
            minpos = 1e-300
        return (minpos if vmin <= 0 else vmin, minpos if vmax <= 0 else vmax)

    def default_locators(self, minor_visible=False):
        return (_ticker.LogLocator(self.base), _ticker.LogLocator(self.base, self.subs),
                _ticker.LogFormatterSciNotation(self.base),
                _ticker.LogFormatterSciNotation(self.base, labelOnlyBase=self.subs is not None))

    def val_in_range(self, val):
        arr = np.asarray(val)
        with np.errstate(invalid='ignore'):
            res = np.isfinite(arr) & (arr > 0)
        return bool(res) if arr.ndim == 0 else res


class SymmetricalLogScale(ScaleBase):
    """matplotlib's SymmetricalLogScale: linear within +-linthresh (stretched
    by linscale decades), logarithmic beyond. matplotlib's default linthresh
    is 2."""
    name = 'symlog'

    def __init__(self, base=10, linthresh=2, subs=None, linscale=1):
        if base <= 1.0:
            raise ValueError("'base' must be larger than 1")
        if linthresh <= 0.0:
            raise ValueError("'linthresh' must be positive")
        if linscale <= 0.0:
            raise ValueError("'linscale' must be positive")
        self.base, self.linthresh, self.linscale, self.subs = base, linthresh, linscale, subs

    def transform(self, a):
        a, scalar = _arr(a)
        linscale_adj = self.linscale / (1.0 - 1.0 / self.base)
        log_base = np.log(self.base)
        abs_a = np.abs(a)
        inside = abs_a <= self.linthresh
        if np.all(inside):
            return _out(a * linscale_adj, scalar)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.sign(a) * self.linthresh * (
                linscale_adj - np.log(self.linthresh) / log_base + np.log(abs_a) / log_base)
        out[inside] = a[inside] * linscale_adj
        return _out(out, scalar)

    def inverse(self, a):
        a, scalar = _arr(a)
        linscale_adj = self.linscale / (1.0 - 1.0 / self.base)
        invlinthresh = self.transform(np.array([self.linthresh], float))[0]
        abs_a = np.abs(a)
        inside = abs_a <= invlinthresh
        if np.all(inside):
            return _out(a / linscale_adj, scalar)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.sign(a) * self.linthresh * np.exp(
                (abs_a / self.linthresh - linscale_adj) * np.log(self.base))
        out[inside] = a[inside] / linscale_adj
        return _out(out, scalar)

    def default_locators(self, minor_visible=False):
        return (_ticker.SymmetricalLogLocator(self),
                _ticker.SymmetricalLogLocator(self, self.subs),
                _ticker.LogFormatterSciNotation(self.base, linthresh=self.linthresh),
                _ticker.NullFormatter())


class LogitScale(ScaleBase):
    """matplotlib's LogitScale: log10(p / (1 - p)) for probabilities, 0 and
    1 clipped to -+1000 (nonpositive='clip') or invalid ('mask')."""
    name = 'logit'

    def __init__(self, nonpositive='mask', *, one_half=r'\frac{1}{2}', use_overline=False):
        if nonpositive not in ('clip', 'mask'):
            raise ValueError("nonpositive must be 'clip' or 'mask'")
        self.nonpositive = nonpositive
        self._one_half = one_half
        self._use_overline = use_overline

    def transform(self, a):
        a, scalar = _arr(a)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.log10(a / (1 - a))
        if self.nonpositive == 'clip':
            out[a <= 0] = -1000
            out[1 <= a] = 1000
        return _out(out, scalar)

    def inverse(self, a):
        a, scalar = _arr(a)
        return _out(1.0 / (1 + 10 ** (-a)), scalar)

    def limit_range_for_scale(self, vmin, vmax, minpos):
        if not np.isfinite(minpos):
            minpos = 1e-7
        return (minpos if vmin <= 0 else vmin, 1 - minpos if vmax >= 1 else vmax)

    def default_locators(self, minor_visible=False):
        kw = dict(one_half=self._one_half, use_overline=self._use_overline)
        return (_ticker.LogitLocator(), _ticker.LogitLocator(minor=True),
                _ticker.LogitFormatter(**kw), _ticker.LogitFormatter(minor=True, **kw))

    def val_in_range(self, val):
        arr = np.asarray(val)
        with np.errstate(invalid='ignore'):
            res = (0 < arr) & (arr < 1)
        return bool(res) if arr.ndim == 0 else res


class _Recorder:
    """The axis matplotlib's set_default_locators_and_formatters is given."""
    axis_name = 'x'

    def __init__(self):
        self.got = {}

    def set_major_locator(self, v):
        self.got['major_locator'] = v

    def set_minor_locator(self, v):
        self.got['minor_locator'] = v

    def set_major_formatter(self, v):
        self.got['major_formatter'] = v

    def set_minor_formatter(self, v):
        self.got['minor_formatter'] = v

    def set(self, **kw):
        for k, v in kw.items():
            getattr(self, 'set_' + k)(v)


class ForeignScale(ScaleBase):
    """A matplotlib ScaleBase object behind the protocol."""

    def __init__(self, obj):
        self.obj = obj
        self.name = getattr(obj, 'name', None)
        tr = obj.get_transform()
        for attr in ('base', 'linthresh', 'linscale'):     # LogFormatter reads linthresh
            if hasattr(tr, attr):
                setattr(self, attr, getattr(tr, attr))

    def get_transform(self):
        return self.obj.get_transform()

    def transform(self, a):
        a, scalar = _arr(a)
        return _out(np.asarray(self.obj.get_transform().transform(a), float), scalar)

    def inverse(self, a):
        a, scalar = _arr(a)
        return _out(np.asarray(self.obj.get_transform().inverted().transform(a), float), scalar)

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return self.obj.limit_range_for_scale(vmin, vmax, minpos)

    def default_locators(self, minor_visible=False):
        rec = _Recorder()
        self.obj.set_default_locators_and_formatters(rec)
        g = rec.got
        minor = g.get('minor_locator')
        if self.name in ('linear', 'function') and minor_visible:
            minor = _ticker.AutoMinorLocator()
        return (_ticker.get_locator(g.get('major_locator'), self),
                _ticker.get_locator(minor, self),
                _ticker.get_formatter(g.get('major_formatter'), self),
                _ticker.get_formatter(g.get('minor_formatter'), self))

    def val_in_range(self, val):
        return self.obj.val_in_range(val) if hasattr(self.obj, 'val_in_range') else \
            ScaleBase.val_in_range(self, val)


class AsinhLocator(_ticker.Locator):
    """matplotlib's AsinhLocator: ticks even on the asinh screen, rounded
    to base-n numbers (and multiples subs of them)."""

    def __init__(self, linear_width, numticks=11, symthresh=0.2, base=10, subs=None):
        self.linear_width, self.numticks, self.symthresh = linear_width, numticks, symthresh
        self.base, self.subs = base, subs

    def tick_values(self, vmin, vmax, tick_space=None, major=None):
        import math
        a = self.linear_width
        ymin, ymax = a * np.arcsinh(np.array([vmin, vmax]) / a)
        ys = np.linspace(ymin, ymax, self.numticks)
        zero_dev = abs(ys / (ymax - ymin))
        if ymin * ymax < 0:
            ys = np.hstack([ys[zero_dev > 0.5 / self.numticks], 0.0])
        xs = a * np.sinh(ys / a)
        zero_xs = ys == 0
        with np.errstate(divide='ignore'):
            if self.base > 1:
                lb = math.log(self.base)
                powers = (np.where(zero_xs, 0, np.sign(xs)) * np.power(
                    self.base, np.where(zero_xs, 0.0,
                                        np.floor(np.log(np.abs(xs) + zero_xs * 1e-6) / lb))))
                qs = np.outer(powers, self.subs).flatten() if self.subs else powers
            else:
                powers = (np.where(xs >= 0, 1, -1) * np.power(10, np.where(
                    zero_xs, 0.0, np.floor(np.log10(np.abs(xs) + zero_xs * 1e-6)))))
                qs = powers * np.round(xs / powers)
        ticks = np.array(sorted(set(qs.tolist())))
        return ticks if len(ticks) >= 2 else np.linspace(vmin, vmax, self.numticks)


_ASINH_SUBS = {3: (2,), 4: (2,), 5: (2,), 8: (2, 4), 10: (2, 5), 16: (2, 4, 8), 64: (4, 16),
               1024: (256, 512)}


class AsinhScale(ScaleBase):
    """matplotlib's AsinhScale: linear_width * asinh(x / linear_width) --
    linear near zero, logarithmic far from it, for either sign."""
    name = 'asinh'

    def __init__(self, linear_width=1.0, *, base=10, subs='auto'):
        if linear_width <= 0:
            raise ValueError('Scale parameter linear_width must be strictly positive')
        self.linear_width, self.base = float(linear_width), int(base)
        self.subs = _ASINH_SUBS.get(self.base) if subs == 'auto' else subs

    def transform(self, a):
        return self.linear_width * np.arcsinh(np.asarray(a, float) / self.linear_width)

    def inverse(self, a):
        return self.linear_width * np.sinh(np.asarray(a, float) / self.linear_width)

    def default_locators(self, minor_visible=False):
        major = AsinhLocator(self.linear_width, base=self.base)
        minor = AsinhLocator(self.linear_width, base=self.base, subs=self.subs)
        fmt = _ticker.LogFormatterSciNotation(self.base) if self.base > 1 else \
            _ticker.StrMethodFormatter('{x:.3g}')
        return major, minor, fmt, _ticker.NullFormatter()


_SCALES = {'linear': LinearScale, 'log': LogScale, 'symlog': SymmetricalLogScale,
           'logit': LogitScale, 'function': FuncScale, 'asinh': AsinhScale}


def get_scale(scale, **kw):
    """A scale from a name and matplotlib's keyword arguments (log: base,
    subs, nonpositive; symlog: base, linthresh, linscale, subs; logit:
    nonpositive, one_half, use_overline; function: functions=(fwd, inv)),
    a fastplot scale, or a matplotlib scale object."""
    if isinstance(scale, ScaleBase):
        return scale
    if isinstance(scale, str):
        try:
            cls = _SCALES[scale]
        except KeyError:
            raise ValueError('%r is not a valid scale name; supported: %s'
                             % (scale, ', '.join(sorted(_SCALES)))) from None
        return cls(**kw)
    if hasattr(scale, 'get_transform'):
        return ForeignScale(scale)
    raise TypeError('cannot make a scale from %r' % (scale,))
