"""Tick locations and labels as matplotlib and cartopy choose them.

The locators and formatters of matplotlib.ticker (and cartopy's longitude /
latitude ones), ported line by line so that a colorbar, a gridline label or
a Cartesian axis ticks and labels exactly where matplotlib 3.11 would: users
port matplotlib scripts and expect the same numbers on their axes.

matplotlib's locators and formatters read the axis they are attached to
(its view interval, the room for labels, the major ticks, the scale's
transform). Here there is no Axis object, so that state is passed in:

- A **locator** answers ``tick_values(vmin, vmax, tick_space=None,
  major=None)`` with an array of tick values (outside ticks included, as
  matplotlib returns them). *tick_space* is what matplotlib's
  ``Axis.get_tick_space()`` returns (see `tick_space`), used by
  ``nbins='auto'`` / ``numticks='auto'``; *major* are the major ticks, which
  minor locators (AutoMinorLocator) derive theirs from. ``view_limits`` and
  ``nonsingular`` are matplotlib's, for autoscaling.
- A **formatter** answers ``format_ticks(locs, vmin, vmax)`` with
  ``(labels, offset_text)``: the labels of all *locs* and the text drawn at
  the end of the axis (ScalarFormatter's ``×1e3+1e4``, ConciseDateFormatter's
  date). vmin > vmax means an inverted axis. ``__call__(x, pos)`` formats one
  value with the state of the last ``format_ticks``, as in matplotlib.

Labels are matplotlib's strings, with axes.unicode_minus (U+2212) and
mathtext where matplotlib uses it, but never ``\\mathdefault{...}``
(fastplot's mathtext has no such command): ``$10^{2}$``,
``$2\\times10^{1}$``. matplotlib Locator / Formatter objects a user passes
are accepted through `get_locator` / `get_formatter`, which hand them a
stand-in axis carrying that state.
"""
import itertools
import math
import string

import numpy as np

MINUS = '\N{MINUS SIGN}'       # matplotlib's axes.unicode_minus
MIN_EXPONENT = 0               # rcParams['axes.formatter.min_exponent']


def tick_space(length_pt, labelsize_pt=10.0, axis='x'):
    """matplotlib's XAxis / YAxis.get_tick_space: how many ticks fit on an
    axis *length_pt* points long, assuming labels at most 3:1 wide (x) or
    two label heights apart (y)."""
    size = labelsize_pt * (3 if axis == 'x' else 2)
    return int(np.floor(length_pt / size)) if size > 0 else 2 ** 31 - 1


def nonsingular(vmin, vmax, expander=0.001, tiny=1e-15, increasing=True):
    """matplotlib.transforms._nonsingular: widen a range that is empty (or
    too small for double precision) and turn inf/NaN into (-expander,
    expander)."""
    if not (np.isfinite(vmin) and np.isfinite(vmax)):
        return -expander, expander
    swapped = vmax < vmin
    if swapped:
        vmin, vmax = vmax, vmin
    vmin, vmax = float(vmin), float(vmax)
    maxabs = max(abs(vmin), abs(vmax))
    if maxabs < (1e6 / tiny) * np.finfo(float).tiny:
        vmin, vmax = -expander, expander
    elif vmax - vmin <= maxabs * tiny:
        if vmax == 0 and vmin == 0:
            vmin, vmax = -expander, expander
        else:
            vmin -= expander * abs(vmin)
            vmax += expander * abs(vmax)
    if swapped and not increasing:
        vmin, vmax = vmax, vmin
    return vmin, vmax


def _nonsingular(vmin, vmax, expander=1e-13, tiny=1e-14):
    return nonsingular(vmin, vmax, expander, tiny, increasing=False)


def remove_overlapping_locs(minor, major, vmin, vmax, transform=None):
    """Axis.get_minorticklocs: drop the minor ticks that land on a major one
    (within 1e-5 of the view, in the scale's transformed space)."""
    minor = np.asarray(minor, float)
    major = np.asarray(major, float)
    if not minor.size or not major.size:
        return minor
    tr = (lambda a: a) if transform is None else transform
    tmin, tmaj = np.asarray(tr(minor), float), np.asarray(tr(major), float)
    lo, hi = sorted(np.asarray(tr(np.array([vmin, vmax], float)), float))
    tol = (hi - lo) * 1e-5
    mask = np.isclose(tmin[:, None], tmaj[None, :], atol=tol, rtol=0).any(axis=1)
    return minor[~mask]


# ----------------------------------------------------------------- locators

class Locator:
    """matplotlib.ticker.Locator: the protocol (module docstring) and the
    default autoscaling helpers."""
    MAXTICKS = 1000

    def tick_values(self, vmin, vmax, tick_space=None, major=None):
        raise NotImplementedError

    def nonsingular(self, vmin, vmax):
        return nonsingular(vmin, vmax, expander=0.05)

    def view_limits(self, vmin, vmax):
        return nonsingular(vmin, vmax)


def _scale_range(vmin, vmax, n=1, threshold=100):
    dv = abs(vmax - vmin)
    meanv = (vmax + vmin) / 2
    if abs(meanv) / dv < threshold:
        offset = 0
    else:
        offset = math.copysign(10 ** (math.log10(abs(meanv)) // 1), meanv)
    scale = 10 ** (math.log10(dv / n) // 1)
    return scale, offset


class _Edge:
    """matplotlib's _Edge_integer: n * step <= x, allowing for the precision
    lost to a large offset."""

    def __init__(self, step, offset):
        if step <= 0:
            raise ValueError("'step' must be positive")
        self.step = step
        self._offset = abs(offset)

    def closeto(self, ms, edge):
        if self._offset > 0:
            digits = np.log10(self._offset / self.step)
            tol = max(1e-10, 10 ** (digits - 12))
            tol = min(0.4999, tol)
        else:
            tol = 1e-10
        return abs(ms - edge) < tol

    def le(self, x):
        d, m = divmod(x, self.step)
        return d + 1 if self.closeto(m / self.step, 1) else d

    def ge(self, x):
        d, m = divmod(x, self.step)
        return d if self.closeto(m / self.step, 0) else d + 1


class MaxNLocator(Locator):
    """matplotlib.ticker.MaxNLocator: at most nbins + 1 'nice' ticks in the
    view (nbins an int, or 'auto' from tick_space), multiples of *steps*."""

    def __init__(self, nbins=10, steps=None, integer=False, min_n_ticks=2, prune=None,
                 symmetric=False):
        self.nbins = nbins if nbins == 'auto' else int(nbins)
        self.integer = integer
        self.min_n_ticks = max(1, min_n_ticks)
        if prune not in ('upper', 'lower', 'both', None):
            raise ValueError('prune must be upper, lower, both or None')
        self.prune = prune
        self.symmetric = symmetric
        self.set_steps(steps)

    def set_params(self, **kw):
        for k in ('nbins', 'integer', 'min_n_ticks', 'prune', 'symmetric'):
            if k in kw:
                v = kw.pop(k)
                setattr(self, k, v if k != 'nbins' or v == 'auto' else int(v))
        if 'steps' in kw:
            self.set_steps(kw.pop('steps'))
        if kw:
            raise TypeError('unknown parameters %s' % sorted(kw))

    def set_steps(self, steps):
        if steps is None:                               # MaxNLocator's own default
            steps = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]
        steps = np.asarray(steps, float)
        if np.any(np.diff(steps) <= 0) or steps[-1] > 10 or steps[0] < 1:
            raise ValueError('steps must be an increasing sequence of numbers '
                             'between 1 and 10 inclusive')
        if steps[0] != 1:
            steps = np.concatenate([[1], steps])
        if steps[-1] != 10:
            steps = np.concatenate([steps, [10]])
        self.steps = steps
        self._extended = np.concatenate([0.1 * steps[:-1], steps, [10 * steps[1]]])

    def _nbins(self, tick_space):
        if self.nbins != 'auto':
            return self.nbins
        if tick_space is None:                          # no axis: matplotlib's 9
            return 9
        return int(np.clip(tick_space, max(1, self.min_n_ticks - 1), 9))

    def raw_ticks(self, vmin, vmax, nbins):
        scale, offset = _scale_range(vmin, vmax, nbins)
        _vmin, _vmax = vmin - offset, vmax - offset
        steps = self._extended * scale
        if self.integer:
            steps = steps[(steps < 1) | (np.abs(steps - np.round(steps)) < 0.001)]
        raw_step = (_vmax - _vmin) / nbins
        large = steps >= raw_step
        istep = np.nonzero(large)[0][0] if large.any() else len(steps) - 1
        ticks = None
        for step in steps[:istep + 1][::-1]:
            if self.integer and np.floor(_vmax) - np.ceil(_vmin) >= self.min_n_ticks - 1:
                step = max(1, step)
            best_vmin = (_vmin // step) * step
            edge = _Edge(step, offset)
            low = edge.le(_vmin - best_vmin)
            high = edge.ge(_vmax - best_vmin)
            ticks = np.arange(low, high + 1) * step + best_vmin
            nticks = ((ticks <= _vmax) & (ticks >= _vmin)).sum()
            if nticks >= self.min_n_ticks:
                break
        return ticks + offset

    def tick_values(self, vmin, vmax, tick_space=None, major=None):
        if self.symmetric:
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax
        vmin, vmax = nonsingular(vmin, vmax, expander=1e-13, tiny=1e-14)
        locs = self.raw_ticks(vmin, vmax, self._nbins(tick_space))
        if self.prune == 'lower':
            locs = locs[1:]
        elif self.prune == 'upper':
            locs = locs[:-1]
        elif self.prune == 'both':
            locs = locs[1:-1]
        return locs

    def view_limits(self, vmin, vmax):
        if self.symmetric:
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax
        return nonsingular(vmin, vmax, expander=1e-12, tiny=1e-13)


class AutoLocator(MaxNLocator):
    """matplotlib's AutoLocator: MaxNLocator('auto', steps=[1, 2, 2.5, 5, 10])."""

    def __init__(self):
        super().__init__('auto', steps=[1, 2, 2.5, 5, 10])


def auto_locator():
    """matplotlib's AutoLocator."""
    return AutoLocator()


class FixedLocator(Locator):
    """matplotlib.ticker.FixedLocator, nbins thinning included (keeping the
    subset with the smallest absolute value, zero if it is there)."""

    def __init__(self, locs, nbins=None):
        self.locs = np.asarray(locs, float)
        self.nbins = None if nbins is None else max(nbins, 2)

    def tick_values(self, vmin=None, vmax=None, tick_space=None, major=None):
        if self.nbins is None or not self.locs.size:
            return self.locs
        step = max(int(np.ceil(len(self.locs) / self.nbins)), 1)
        ticks = self.locs[::step]
        for i in range(1, step):
            t1 = self.locs[i::step]
            if np.abs(t1).min() < np.abs(ticks).min():
                ticks = t1
        return ticks


class NullLocator(Locator):
    """No ticks."""

    def tick_values(self, vmin=None, vmax=None, tick_space=None, major=None):
        return np.array([], float)


class IndexLocator(Locator):
    """matplotlib.ticker.IndexLocator: every *base*-th value from vmin +
    offset (matplotlib calls it with the data limits, not the view)."""

    def __init__(self, base, offset):
        self.base = base
        self.offset = offset

    def tick_values(self, vmin, vmax, tick_space=None, major=None):
        return np.arange(vmin + self.offset, vmax + 1e-12, self.base)


class LinearLocator(Locator):
    """matplotlib.ticker.LinearLocator: *numticks* (default 11) evenly spaced
    ticks over the view, or presets[(vmin, vmax)]."""

    def __init__(self, numticks=None, presets=None):
        self._numticks = numticks
        self.presets = {} if presets is None else presets

    @property
    def numticks(self):
        return self._numticks if self._numticks is not None else 11

    @numticks.setter
    def numticks(self, n):
        self._numticks = n

    def tick_values(self, vmin, vmax, tick_space=None, major=None):
        vmin, vmax = nonsingular(vmin, vmax, expander=0.05)
        if (vmin, vmax) in self.presets:
            return np.asarray(self.presets[(vmin, vmax)], float)
        if self.numticks == 0:
            return np.array([], float)
        return np.linspace(vmin, vmax, self.numticks)

    def view_limits(self, vmin, vmax):
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        if vmin == vmax:
            vmin -= 1
            vmax += 1
        return nonsingular(vmin, vmax)


class MultipleLocator(Locator):
    """matplotlib.ticker.MultipleLocator: every multiple of *base* (plus
    *offset*), one beyond each end of the view."""

    def __init__(self, base=1.0, offset=0.0):
        self._edge = _Edge(base, 0)
        self._offset = offset

    def tick_values(self, vmin, vmax, tick_space=None, major=None):
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        step = self._edge.step
        vmin -= self._offset
        vmax -= self._offset
        vmin = self._edge.ge(vmin) * step
        n = (vmax - vmin + 0.001 * step) // step
        return vmin - step + np.arange(n + 3) * step + self._offset

    def view_limits(self, dmin, dmax):
        return nonsingular(dmin, dmax)


# log helpers (matplotlib.ticker's _decade_* and _is_close_to_int)

def _is_decade(x, *, base=10, rtol=None):
    if not np.isfinite(x):
        return False
    if x == 0.0:
        return True
    lx = np.log(abs(x)) / np.log(base)
    if rtol is None:
        return np.isclose(lx, np.round(lx))
    return np.isclose(lx, np.round(lx), rtol=rtol)


def _decade_less_equal(x, base):
    return (x if x == 0 else -_decade_greater_equal(-x, base) if x < 0 else
            base ** np.floor(np.log(x) / np.log(base)))


def _decade_greater_equal(x, base):
    return (x if x == 0 else -_decade_less_equal(-x, base) if x < 0 else
            base ** np.ceil(np.log(x) / np.log(base)))


def _decade_less(x, base):
    if x < 0:
        return -_decade_greater(-x, base)
    less = _decade_less_equal(x, base)
    if less == x:
        less /= base
    return less


def _decade_greater(x, base):
    if x < 0:
        return -_decade_less(-x, base)
    greater = _decade_greater_equal(x, base)
    if greater == x:
        greater *= base
    return greater


def _is_close_to_int(x):
    return math.isclose(x, round(x))


class LogLocator(Locator):
    """matplotlib.ticker.LogLocator: ticks at subs[j] * base**i, decades
    strided to at most numticks ('auto': tick_space clipped to 2..9). subs
    'auto' (the minor locator) puts ticks between the decades -- none when
    there are 10 decades or more; with fewer than two ticks it falls back to
    an AutoLocator, as matplotlib does."""

    def __init__(self, base=10.0, subs=(1.0,), *, numticks=None):
        self._base = float(base)
        self._set_subs(subs)
        self.numticks = 'auto' if numticks is None else numticks

    @property
    def base(self):
        return self._base

    def _set_subs(self, subs):
        if subs is None:
            self._subs = 'auto'
        elif isinstance(subs, str):
            if subs not in ('all', 'auto'):
                raise ValueError("subs must be None, 'all', 'auto' or floats")
            self._subs = subs
        else:
            self._subs = np.asarray(subs, dtype=float)

    def set_params(self, base=None, subs=None, *, numticks=None):
        if base is not None:
            self._base = float(base)
        if subs is not None:
            self._set_subs(subs)
        if numticks is not None:
            self.numticks = numticks

    def _log_b(self, x):
        return (np.log10(x) if self._base == 10 else np.log2(x) if self._base == 2 else
                np.log(x) / np.log(self._base))

    def tick_values(self, vmin, vmax, tick_space=None, major=None):
        if self.numticks != 'auto':
            n_request = self.numticks
        else:
            n_request = 9 if tick_space is None else int(np.clip(tick_space, 2, 9))
        b = self._base
        if vmin <= 0.0 or not np.isfinite(vmin):
            raise ValueError('Data cannot be log-scaled because all values are <= 0.')
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        efmin, efmax = self._log_b([vmin, vmax])
        emin = math.ceil(efmin)
        emax = math.floor(efmax)
        n_avail = emax - emin + 1                       # decade ticks available
        if isinstance(self._subs, str):
            if n_avail >= 10 or b < 3:
                if self._subs == 'auto':
                    return np.array([])
                subs = np.array([1.0])
            else:
                subs = np.arange(2.0 if self._subs == 'auto' else 1.0, b)
        else:
            subs = self._subs
        # the largest tick count <= n_request that a stride can give, then the
        # largest stride giving it, then an offset putting ticks on multiples
        stride = n_avail // (n_request + 1) + 1
        nr = math.ceil(n_avail / stride)
        if nr <= n_request:
            n_request = nr
        if n_request == 0:                              # two ticks just outside
            decades = [emin - 1, emax + 1]
            stride = decades[1] - decades[0]
        elif n_request == 1:                            # one near the centre
            mid = round((efmin + efmax) / 2)
            stride = max(mid - (emin - 1), (emax + 1) - mid)
            decades = [mid - stride, mid, mid + stride]
        else:
            stride = (n_avail - 1) // (n_request - 1)
            if stride < n_avail / n_request:
                stride = n_avail // n_request
            olo = max(n_avail - stride * n_request, 0)
            ohi = min(n_avail - stride * (n_request - 1), stride)
            offset = (-emin) % stride
            if not olo <= offset < ohi:
                offset = olo
            decades = range(emin + offset - stride, emax + stride + 1, stride)
        is_minor = len(subs) > 1 or (len(subs) == 1 and subs[0] != 1.0)
        if is_minor:
            if stride == 1 or n_avail <= 1:
                ticklocs = np.concatenate([subs * b ** decade
                                           for decade in range(emin - 1, emax + 1)])
            else:
                ticklocs = np.array([])
        else:
            ticklocs = b ** np.array(decades)
        if (len(subs) > 1 and stride == 1
                and len(decades) - 2 + ((vmin <= ticklocs) & (ticklocs <= vmax)).sum() <= 1):
            return AutoLocator().tick_values(vmin, vmax)
        return ticklocs

    def view_limits(self, vmin, vmax, minpos=np.inf):
        return self.nonsingular(vmin, vmax, minpos)

    def nonsingular(self, vmin, vmax, minpos=np.inf):
        """LogLocator.nonsingular; *minpos* is the smallest positive data
        value (matplotlib's Axis.get_minpos), inf when there is none."""
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 1, 10
        elif vmax <= 0:
            vmin, vmax = 1, 10
        else:
            if not np.isfinite(minpos):
                minpos = 1e-300
            if vmin <= 0:
                vmin = minpos
            if vmin == vmax:
                vmin = _decade_less(vmin, self._base)
                vmax = _decade_greater(vmax, self._base)
        return vmin, vmax


class SymmetricalLogLocator(Locator):
    """matplotlib.ticker.SymmetricalLogLocator: decades on both log sides of
    a symlog axis and 0 in its linear part, strided to numticks (15)."""

    def __init__(self, transform=None, subs=None, linthresh=None, base=None):
        if transform is not None:
            self._base, self._linthresh = transform.base, transform.linthresh
        elif linthresh is not None and base is not None:
            self._base, self._linthresh = base, linthresh
        else:
            raise ValueError('Either transform, or both linthresh and base, must be provided.')
        self._subs = [1.0] if subs is None else subs
        self.numticks = 15

    def tick_values(self, vmin, vmax, tick_space=None, major=None):
        linthresh = self._linthresh
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        if -linthresh <= vmin < vmax <= linthresh:     # only the linear part
            return np.array(sorted({vmin, 0, vmax}), float)
        has_a = vmin < -linthresh
        has_c = vmax > linthresh
        has_b = (has_a and vmax > -linthresh) or (has_c and vmin < linthresh)
        base = self._base

        def get_log_range(lo, hi):
            return (np.floor(np.log(lo) / np.log(base)), np.ceil(np.log(hi) / np.log(base)))

        a_lo, a_hi = (0, 0)
        if has_a:
            a_lo, a_hi = get_log_range(abs(min(-linthresh, vmax)), abs(vmin) + 1)
        c_lo, c_hi = (0, 0)
        if has_c:
            c_lo, c_hi = get_log_range(max(linthresh, vmin), vmax + 1)
        total_ticks = (a_hi - a_lo) + (c_hi - c_lo)
        if has_b:
            total_ticks += 1
        stride = max(total_ticks // (self.numticks - 1), 1)
        decades = []
        if has_a:
            decades.extend(-1 * (base ** (np.arange(a_lo, a_hi, stride)[::-1])))
        if has_b:
            decades.append(0.0)
        if has_c:
            decades.extend(base ** (np.arange(c_lo, c_hi, stride)))
        subs = np.asarray(self._subs)
        if len(subs) > 1 or subs[0] != 1.0:
            ticklocs = []
            for decade in decades:
                if decade == 0:
                    ticklocs.append(decade)
                else:
                    ticklocs.extend(subs * decade)
        else:
            ticklocs = decades
        return np.array(ticklocs, float)

    def view_limits(self, vmin, vmax):
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        return nonsingular(vmin, vmax)


class AutoMinorLocator(Locator):
    """matplotlib.ticker.AutoMinorLocator: n divisions of the major step
    (n 'auto' = rcParams x/ytick.minor.ndivs: 5 when the step's mantissa is
    1, 2.5, 5 or 10, else 4), over the view. Needs the major ticks."""

    def __init__(self, n=None):
        self.ndivs = 'auto' if n is None else n

    def tick_values(self, vmin, vmax, tick_space=None, major=None):
        majorlocs = np.unique(np.asarray([] if major is None else major, float))
        if len(majorlocs) < 2:
            return np.array([], float)
        majorstep = majorlocs[1] - majorlocs[0]
        if self.ndivs == 'auto':
            mantissa = 10 ** (np.log10(majorstep) % 1)
            ndivs = 5 if np.isclose(mantissa, [1, 2.5, 5, 10]).any() else 4
        else:
            ndivs = self.ndivs
        minorstep = majorstep / ndivs
        vmin, vmax = sorted((vmin, vmax))
        t0 = majorlocs[0]
        tmin = round((vmin - t0) / minorstep)
        tmax = round((vmax - t0) / minorstep) + 1
        return np.arange(tmin, tmax) * minorstep + t0


class LogitLocator(MaxNLocator):
    """matplotlib.ticker.LogitLocator: ticks at 10^-k, 1/2, 1 - 10^-k on a
    logit axis (subsampled to nbins; MaxNLocator ticks once zoomed in)."""

    def __init__(self, minor=False, *, nbins='auto'):
        self.minor = minor
        super().__init__(nbins=nbins, steps=[1, 2, 5, 10])

    def tick_values(self, vmin, vmax, tick_space=None, major=None):
        if self.nbins == 'auto':
            nbins = 9 if tick_space is None else max(tick_space, 2)
        else:
            nbins = self.nbins

        def ideal_ticks(x):
            return 10 ** x if x < 0 else 1 - (10 ** (-x)) if x > 0 else 0.5

        vmin, vmax = self.nonsingular(vmin, vmax)
        binf = int(np.floor(np.log10(vmin)) if vmin < 0.5 else 0 if vmin < 0.9
                   else -np.ceil(np.log10(1 - vmin)))
        bsup = int(np.ceil(np.log10(vmax)) if vmax <= 0.5 else 1 if vmax <= 0.9
                   else -np.floor(np.log10(1 - vmax)))
        numideal = bsup - binf - 1
        if numideal >= 2:
            if numideal > nbins:
                sub = math.ceil(numideal / nbins)
                if self.minor:
                    t = [ideal_ticks(b) for b in range(binf, bsup + 1) if b % sub != 0]
                else:
                    t = [ideal_ticks(b) for b in range(binf, bsup + 1) if b % sub == 0]
                return np.array(t, float)
            if self.minor:
                t = []
                for b in range(binf, bsup):
                    if b < -1:
                        t.extend(np.arange(2, 10) * 10 ** b)
                    elif b == -1:
                        t.extend(np.arange(2, 5) / 10)
                    elif b == 0:
                        t.extend(np.arange(6, 9) / 10)
                    else:
                        t.extend(1 - np.arange(2, 10)[::-1] * 10 ** (-b - 1))
                return np.array(t, float)
            return np.array([ideal_ticks(b) for b in range(binf, bsup + 1)], float)
        if self.minor:
            return np.array([], float)
        return MaxNLocator.tick_values(self, vmin, vmax, tick_space)

    def nonsingular(self, vmin, vmax, minpos=None):
        standard_minpos = 1e-7
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= 0 or vmin >= 1:
            vmin, vmax = standard_minpos, 1 - standard_minpos
        else:
            if minpos is None or not np.isfinite(minpos):
                minpos = standard_minpos
            if vmin <= 0:
                vmin = minpos
            if vmax >= 1:
                vmax = 1 - minpos
            if vmin == vmax:
                vmin, vmax = 0.1 * vmin, 1 - 0.1 * vmin
        return vmin, vmax

    def view_limits(self, vmin, vmax):
        return nonsingular(vmin, vmax)


class LongitudeLocator(MaxNLocator):
    """cartopy.mpl.ticker.LongitudeLocator (dms=False)."""

    def __init__(self, nbins=8):
        super().__init__(nbins=nbins)

    def _guess_steps(self, vmin, vmax):
        dv = abs(vmax - vmin)
        if dv > 180:
            dv -= 180
        self.set_steps([1, 2, 3, 6, 10] if dv > 50.0 else [1, 1.5, 2, 2.5, 3, 5, 10])

    def raw_ticks(self, vmin, vmax, nbins):
        self._guess_steps(vmin, vmax)
        return MaxNLocator.raw_ticks(self, vmin, vmax, nbins)


class LatitudeLocator(LongitudeLocator):
    """cartopy.mpl.ticker.LatitudeLocator (dms=False)."""

    def tick_values(self, vmin, vmax, tick_space=None, major=None):
        return LongitudeLocator.tick_values(self, max(vmin, -90.0), min(vmax, 90.0))

    def raw_ticks(self, vmin, vmax, nbins):
        self._guess_steps(max(vmin, -90.0), min(vmax, 90.0))
        t = MaxNLocator.raw_ticks(self, vmin, vmax, nbins)
        return t[(t >= -90) & (t <= 90)]


def as_locator(loc):
    """A locator, or a sequence of locations as a FixedLocator (anything
    with tick_values -- a cartopy or matplotlib locator -- is kept as is)."""
    if loc is None or hasattr(loc, 'tick_values'):
        return loc
    return FixedLocator(np.asarray(loc, float))


# --------------------------------------------------------------- formatters

def _minus(s):
    return s.replace('-', MINUS)


def strip_mathdefault(s):
    """'$\\mathdefault{10^{2}}$' -> '$10^{2}$': matplotlib's mathtext labels
    for fastplot's mathtext, which has no \\mathdefault."""
    key = '\\mathdefault{'
    while key in s:
        i = s.index(key)
        j, depth = i + len(key), 1
        while j < len(s) and depth:
            depth += {'{': 1, '}': -1}.get(s[j], 0)
            j += 1
        s = s[:i] + s[i + len(key):j - 1] + s[j:]
    return s


class Formatter:
    """matplotlib.ticker.Formatter, the view interval held here instead of
    read from an axis: format_ticks(locs, vmin, vmax) -> (labels, offset)."""
    _locs = ()
    _view = (0.0, 1.0)

    def format_ticks(self, locs, vmin=None, vmax=None):
        locs = np.asarray(locs, float) if not isinstance(locs, np.ndarray) else locs
        if vmin is None or vmax is None:
            vmin, vmax = (float(np.min(locs)), float(np.max(locs))) if len(locs) else (0.0, 1.0)
        self._view = (vmin, vmax)
        self.set_locs(locs)
        return [self(v, i) for i, v in enumerate(locs)], self.get_offset()

    def __call__(self, x, pos=None):
        raise NotImplementedError

    def set_locs(self, locs):
        self._locs = locs

    def get_offset(self):
        return ''

    def format_data(self, value):
        return self(value)

    fix_minus = staticmethod(_minus)


class NullFormatter(Formatter):
    """No labels."""

    def __call__(self, x, pos=None):
        return ''


class FixedFormatter(Formatter):
    """matplotlib.ticker.FixedFormatter: the i-th string for the i-th tick."""

    def __init__(self, seq):
        self.seq = list(seq)
        self.offset_string = ''

    def __call__(self, x, pos=None):
        return '' if pos is None or pos >= len(self.seq) else self.seq[pos]

    def get_offset(self):
        return self.offset_string


class FuncFormatter(Formatter):
    """matplotlib.ticker.FuncFormatter: func(x, pos) -> label."""

    def __init__(self, func):
        self.func = func
        self.offset_string = ''

    def __call__(self, x, pos=None):
        return self.func(x, pos)

    def get_offset(self):
        return self.offset_string


class FormatStrFormatter(Formatter):
    """matplotlib.ticker.FormatStrFormatter: fmt % x (hyphen minus, as
    matplotlib)."""

    def __init__(self, fmt):
        self.fmt = fmt

    def __call__(self, x, pos=None):
        return self.fmt % x


class _UnicodeMinusFormat(string.Formatter):
    def format_field(self, value, format_spec):
        return _minus(super().format_field(value, format_spec))


class StrMethodFormatter(Formatter):
    """matplotlib.ticker.StrMethodFormatter: fmt.format(x=x, pos=pos), the
    fields' minus signs unicode."""

    def __init__(self, fmt):
        self.fmt = fmt

    def __call__(self, x, pos=None):
        return _UnicodeMinusFormat().format(self.fmt, x=x, pos=pos)


class ScalarFormatter(Formatter):
    """matplotlib.ticker.ScalarFormatter with its rcParams defaults: an
    offset when it saves offset_threshold (4) digits, scientific notation
    outside powerlimits (-5, 6), the fewest decimals telling the ticks apart,
    unicode minus. useMathText writes '$\\times10^{3}$'-style offsets and
    '$1.5$' labels. useLocale is ignored. format(locs, vmin, vmax) is
    format_ticks."""

    def __init__(self, useOffset=None, useMathText=None, useLocale=None, *, usetex=None,
                 use_offset=None, offset_threshold=None, powerlimits=None):
        from ._rc import active                         # rcParams axes.formatter.*, as made
        rc = active()
        if useOffset is None:
            useOffset = rc.get('axes.formatter.useoffset', True) if use_offset is None \
                else use_offset
        if offset_threshold is None:
            offset_threshold = rc.get('axes.formatter.offset_threshold', 4)
        if powerlimits is None:
            powerlimits = rc.get('axes.formatter.limits', (-5, 6))
        if useMathText is None:
            useMathText = rc.get('axes.formatter.use_mathtext', False)
        self.offset_threshold = offset_threshold
        self.set_useOffset(useOffset)
        self._useMathText = bool(useMathText) or bool(usetex)
        self.orderOfMagnitude = 0
        self._fmt = ''
        self._scientific = True
        self.powerlimits = tuple(powerlimits)

    def set_useOffset(self, val):
        if isinstance(val, bool):
            self.offset = 0
            self.use_offset = val
        else:
            self.use_offset = False
            self.offset = val

    def set_scientific(self, b):
        self._scientific = bool(b)

    def set_powerlimits(self, lims):
        self.powerlimits = tuple(lims)

    def format(self, locs, vmin, vmax):
        return self.format_ticks(np.asarray(locs, float), vmin, vmax)

    def __call__(self, x, pos=None):
        if len(self._locs) == 0:
            return ''
        xp = (x - self.offset) / (10. ** self.orderOfMagnitude)
        if abs(xp) < 1e-8:
            xp = 0
        return _minus(self._fmt % xp)

    def format_data(self, value):
        e = math.floor(math.log10(abs(value)))
        s = round(value / 10 ** e, 10)
        significand = _minus(('%d' if s % 1 == 0 else '%1.10g') % s)
        if e == 0:
            return significand
        exponent = _minus('%d' % e)
        if self._useMathText:
            exponent = '10^{%s}' % exponent
            return exponent if s == 1 else r'%s \times %s' % (significand, exponent)
        return '%se%s' % (significand, exponent)

    def get_offset(self):
        if len(self._locs) == 0:
            return ''
        if self.orderOfMagnitude or self.offset:
            offset_str = sci_str = ''
            if self.offset:
                offset_str = self.format_data(self.offset)
                if self.offset > 0:
                    offset_str = '+' + offset_str
            if self.orderOfMagnitude:
                if self._useMathText:
                    sci_str = self.format_data(10 ** self.orderOfMagnitude)
                else:
                    sci_str = '1e%d' % self.orderOfMagnitude
            if self._useMathText:
                if sci_str:
                    sci_str = r'\times' + sci_str
                s = '$%s%s$' % (sci_str, offset_str)
            else:
                s = sci_str + offset_str
            return _minus(s)
        return ''

    def set_locs(self, locs):
        self._locs = locs
        if len(locs):
            if self.use_offset:
                self._compute_offset()
            self._set_order_of_magnitude()
            self._set_format()

    def _visible(self):
        vmin, vmax = sorted(self._view)
        locs = np.asarray(self._locs)
        return locs[(vmin <= locs) & (locs <= vmax)], vmin, vmax

    def _compute_offset(self):
        locs, vmin, vmax = self._visible()
        if not len(locs):
            self.offset = 0
            return
        lmin, lmax = locs.min(), locs.max()
        if lmin == lmax or lmin <= 0 <= lmax:          # one tick, or both signs
            self.offset = 0
            return
        abs_min, abs_max = sorted([abs(float(lmin)), abs(float(lmax))])
        sign = math.copysign(1, lmin)
        oom_max = np.ceil(math.log10(abs_max))
        oom = 1 + next(o for o in itertools.count(oom_max, -1)
                       if abs_min // 10 ** o != abs_max // 10 ** o)
        if (abs_max - abs_min) / 10 ** oom <= 1e-2:     # straddling a big round number
            oom = 1 + next(o for o in itertools.count(oom_max, -1)
                           if abs_max // 10 ** o - abs_min // 10 ** o > 1)
        n = self.offset_threshold - 1
        self.offset = (sign * (abs_max // 10 ** oom) * 10 ** oom
                       if abs_max // 10 ** oom >= 10 ** n else 0)

    def _set_order_of_magnitude(self):
        if not self._scientific:
            self.orderOfMagnitude = 0
            return
        lo, hi = self.powerlimits
        if lo == hi != 0:                               # fixed scaling
            self.orderOfMagnitude = lo
            return
        locs, vmin, vmax = self._visible()
        locs = np.abs(locs)
        if not len(locs):
            self.orderOfMagnitude = 0
            return
        if self.offset:
            oom = math.floor(math.log10(vmax - vmin))
        else:
            val = locs.max()
            oom = 0 if val == 0 else math.floor(math.log10(val))
        self.orderOfMagnitude = oom if (oom <= lo or oom >= hi) else 0

    def _set_format(self):
        _locs = (list(self._locs) + list(self._view)) if len(self._locs) < 2 else self._locs
        locs = (np.asarray(_locs) - self.offset) / 10. ** self.orderOfMagnitude
        rng = np.ptp(locs)
        if rng == 0:
            rng = np.max(np.abs(locs))
        if rng == 0:
            rng = 1
        if len(self._locs) < 2:
            locs = locs[:-2]
        rng_oom = int(math.floor(math.log10(rng)))
        sig = max(0, 3 - rng_oom)
        thresh = 1e-3 * 10 ** rng_oom
        while sig >= 0:
            if np.abs(locs - np.round(locs, decimals=sig)).max() < thresh:
                sig -= 1
            else:
                break
        sig += 1
        self._fmt = '%1.' + str(sig) + 'f'
        if self._useMathText:
            self._fmt = '$%s$' % self._fmt


class EngFormatter(ScalarFormatter):
    """matplotlib.ticker.EngFormatter: SI prefixes, '10 kHz'."""
    ENG_PREFIXES = {-30: 'q', -27: 'r', -24: 'y', -21: 'z', -18: 'a', -15: 'f', -12: 'p',
                    -9: 'n', -6: '\N{MICRO SIGN}', -3: 'm', 0: '', 3: 'k', 6: 'M', 9: 'G',
                    12: 'T', 15: 'P', 18: 'E', 21: 'Z', 24: 'Y', 27: 'R', 30: 'Q'}

    def __init__(self, unit='', places=None, sep=' ', *, usetex=None, useMathText=None,
                 useOffset=False):
        self.unit = unit
        self.places = places
        self.sep = sep
        super().__init__(useOffset=useOffset, useMathText=useMathText, usetex=usetex)

    def __call__(self, x, pos=None):
        if len(self._locs) == 0 or self.offset == 0:
            return _minus(self.format_data(x))
        xp = (x - self.offset) / (10. ** self.orderOfMagnitude)
        if abs(xp) < 1e-8:
            xp = 0
        return _minus(self._fmt % xp)

    def set_locs(self, locs):
        self._locs = locs
        if len(locs):
            vmin, vmax = sorted(self._view)
            if self.use_offset:
                self._compute_offset()
                if self.offset != 0:
                    self.offset = round((vmin + vmax) / 2, 3)
            self.orderOfMagnitude = math.floor(math.log(vmax - vmin, 1000)) * 3
            self._set_format()

    def get_offset(self):
        if len(self._locs) == 0 or not self.offset:
            return ''
        offset_str = self.format_data(self.offset)
        if self.offset > 0:
            offset_str = '+' + offset_str
        sci_str = self.format_data(10 ** self.orderOfMagnitude)
        if self._useMathText:
            s = '$%s%s$' % (r'\times' + sci_str if sci_str else '', offset_str)
        else:
            s = sci_str + offset_str
        return _minus(s)

    def format_data(self, value):
        sign = 1
        fmt = 'g' if self.places is None else '.%df' % self.places
        if value < 0:
            sign = -1
            value = -value
        if value != 0:
            pow10 = int(math.floor(math.log10(value) / 3) * 3)
        else:
            pow10 = 0
            value = 0.0
        pow10 = int(np.clip(pow10, min(self.ENG_PREFIXES), max(self.ENG_PREFIXES)))
        mant = sign * value / (10.0 ** pow10)
        if abs(float(format(mant, fmt))) >= 1000 and pow10 < max(self.ENG_PREFIXES):
            mant /= 1000
            pow10 += 3
        prefix = self.ENG_PREFIXES[pow10]
        suffix = '%s%s%s' % (self.sep, prefix, self.unit) if (self.unit or prefix) else ''
        if self._useMathText:
            return '$%s$%s' % (format(mant, fmt), suffix)
        return '%s%s' % (format(mant, fmt), suffix)


class PercentFormatter(Formatter):
    """matplotlib.ticker.PercentFormatter: x / xmax * 100 with decimals from
    the view's range."""

    def __init__(self, xmax=100, decimals=None, symbol='%', is_latex=False):
        self.xmax = xmax + 0.0
        self.decimals = decimals
        self.symbol = symbol or ''

    def __call__(self, x, pos=None):
        display_range = abs(self._view[1] - self._view[0])
        return _minus(self.format_pct(x, display_range))

    def format_pct(self, x, display_range):
        x = 100.0 * (x / self.xmax)
        if self.decimals is None:
            scaled = 100.0 * (display_range / self.xmax)
            if scaled <= 0:
                decimals = 0
            else:
                decimals = min(max(math.ceil(2.0 - math.log10(2.0 * scaled)), 0), 5)
        else:
            decimals = self.decimals
        return '%s%s' % (format(x, '0.%df' % int(decimals)), self.symbol)


class LogFormatter(Formatter):
    """matplotlib.ticker.LogFormatter: labels at the decades (labelOnlyBase)
    and, when few decades are in view, between them (minor_thresholds:
    at most 1 major tick -> label 1,2,3,4,6 x base^n; under 0.4 decades ->
    every integer coefficient). *linthresh* for a symlog axis."""

    def __init__(self, base=10.0, labelOnlyBase=False, minor_thresholds=None, linthresh=None):
        self._base = float(base)
        self.labelOnlyBase = labelOnlyBase
        self.minor_thresholds = (1, 0.4) if minor_thresholds is None else minor_thresholds
        self._sublabels = None
        self._linthresh = linthresh

    def set_base(self, base):
        self._base = float(base)

    def set_label_minor(self, labelOnlyBase):
        self.labelOnlyBase = labelOnlyBase

    def set_locs(self, locs=None):
        self._locs = locs
        if np.isinf(self.minor_thresholds[0]):
            self._sublabels = None
            return
        linthresh = self._linthresh
        vmin, vmax = sorted(self._view)
        if linthresh is None and vmin <= 0:
            self._sublabels = {1}
            return
        b = self._base
        if linthresh is not None:                       # symlog: the log parts only
            numdec = numticks = 0
            if vmin < -linthresh:
                rhs = min(vmax, -linthresh)
                numticks += (math.floor(math.log(abs(rhs), b))
                             - math.floor(math.nextafter(math.log(abs(vmin), b), -math.inf)))
                numdec += math.log(vmin / rhs, b)
            if vmax > linthresh:
                lhs = max(vmin, linthresh)
                numticks += (math.floor(math.log(vmax, b))
                             - math.floor(math.nextafter(math.log(lhs, b), -math.inf)))
                numdec += math.log(vmax / lhs, b)
        else:
            lmin = math.log(vmin, b)
            lmax = math.log(vmax, b)
            numticks = math.floor(lmax) - math.floor(math.nextafter(lmin, -math.inf))
            numdec = abs(lmax - lmin)
        if numticks > self.minor_thresholds[0]:
            self._sublabels = {1}
        elif numdec > self.minor_thresholds[1]:
            c = np.geomspace(1, b, int(b) // 2 + 1)
            self._sublabels = set(np.round(c))
        else:
            self._sublabels = set(np.arange(1, b + 1))

    def _num_to_string(self, x, vmin, vmax):
        return self._pprint_val(x, vmax - vmin) if 1 <= x <= 10000 else '%1.0e' % x

    def _skip(self, x):
        b = self._base
        fx = math.log(x) / math.log(b)
        is_x_decade = _is_close_to_int(fx)
        exponent = round(fx) if is_x_decade else np.floor(fx)
        coeff = round(b ** (fx - exponent))
        if self.labelOnlyBase and not is_x_decade:
            return True, fx, is_x_decade
        if self._sublabels is not None and coeff not in self._sublabels:
            return True, fx, is_x_decade
        return False, fx, is_x_decade

    def __call__(self, x, pos=None):
        if x == 0.0:
            return '0'
        x = abs(x)
        if self._skip(x)[0]:
            return ''
        vmin, vmax = nonsingular(*self._view, expander=0.05)
        return _minus(self._num_to_string(x, vmin, vmax))

    def _pprint_val(self, x, d):
        if abs(x) < 1e4 and x == int(x):
            return '%d' % x
        fmt = ('%1.3e' if d < 1e-2 else '%1.3f' if d <= 1 else '%1.2f' if d <= 10 else
               '%1.1f' if d <= 1e5 else '%1.1e')
        s = fmt % x
        tup = s.split('e')
        if len(tup) == 2:
            mantissa = tup[0].rstrip('0').rstrip('.')
            exponent = int(tup[1])
            s = '%se%d' % (mantissa, exponent) if exponent else mantissa
        else:
            s = s.rstrip('0').rstrip('.')
        return s


class LogFormatterExponent(LogFormatter):
    """matplotlib.ticker.LogFormatterExponent: the exponent, '3'."""

    def _num_to_string(self, x, vmin, vmax):
        fx = math.log(x) / math.log(self._base)
        if 1 <= abs(fx) <= 10000:
            fd = math.log(vmax - vmin) / math.log(self._base)
            return self._pprint_val(fx, fd)
        return '%1.0g' % fx


class LogFormatterMathtext(LogFormatter):
    """matplotlib.ticker.LogFormatterMathtext: '$10^{3}$' (without
    \\mathdefault), '$10^{0.30}$' between decades."""

    def _non_decade_format(self, sign, base, fx):
        return '$%s%s^{%.2f}$' % (sign, base, fx)

    def __call__(self, x, pos=None):
        if x == 0:
            return '$0$'
        sign = '-' if x < 0 else ''
        x = abs(x)
        b = self._base
        skip, fx, is_x_decade = self._skip(x)
        if skip:
            return ''
        if is_x_decade:
            fx = round(fx)
        base = '%d' % b if b % 1 == 0.0 else '%s' % b
        if abs(fx) < MIN_EXPONENT:
            return '$%s%g$' % (sign, x)
        if not is_x_decade:
            return self._non_decade_format(sign, base, fx)
        return '$%s%s^{%d}$' % (sign, base, fx)


class LogFormatterSciNotation(LogFormatterMathtext):
    """matplotlib.ticker.LogFormatterSciNotation, the log axes' default:
    '$10^{3}$', '$2\\times10^{3}$' (without \\mathdefault)."""

    def _non_decade_format(self, sign, base, fx):
        b = float(base)
        exponent = math.floor(fx)
        coeff = b ** (fx - exponent)
        if _is_close_to_int(coeff):
            coeff = round(coeff)
        return r'$%s%g\times%s^{%d}$' % (sign, coeff, base, exponent)


class LogitFormatter(Formatter):
    """matplotlib.ticker.LogitFormatter: '$10^{-2}$', '$\\frac{1}{2}$',
    '$1-10^{-2}$' (without \\mathdefault)."""

    def __init__(self, *, use_overline=False, one_half=r'\frac{1}{2}', minor=False,
                 minor_threshold=25, minor_number=6):
        self._use_overline = use_overline
        self._one_half = one_half
        self._minor = minor
        self._labelled = set()
        self._minor_threshold = minor_threshold
        self._minor_number = minor_number

    def set_locs(self, locs):
        self._locs = np.array(locs)
        self._labelled.clear()
        if not self._minor:
            return
        if all(_is_decade(x, rtol=1e-7) or _is_decade(1 - x, rtol=1e-7)
               or (_is_close_to_int(2 * x) and int(np.round(2 * x)) == 1) for x in locs):
            return
        if len(locs) < self._minor_threshold:
            if len(locs) < self._minor_number:
                self._labelled.update(locs)
            else:
                diff = np.diff(-np.log(1 / self._locs - 1))
                space_pessimistic = np.minimum(np.concatenate(((np.inf,), diff)),
                                               np.concatenate((diff, (np.inf,))))
                space_sum = np.concatenate(((0,), diff)) + np.concatenate((diff, (0,)))
                good = sorted(range(len(self._locs)),
                              key=lambda i: (space_pessimistic[i], space_sum[i]))
                self._labelled.update(locs[i] for i in good[-self._minor_number:])

    def _format_value(self, x, locs, sci_notation=True):
        if sci_notation:
            exponent = math.floor(np.log10(x))
            min_precision = 0
        else:
            exponent = 0
            min_precision = 1
        value = x * 10 ** (-exponent)
        if len(locs) < 2:
            precision = min_precision
        else:
            diff = np.sort(np.abs(locs - x))[1]
            precision = -np.log10(diff) + exponent
            precision = (int(np.round(precision)) if _is_close_to_int(precision)
                         else math.ceil(precision))
            precision = max(precision, min_precision)
        mantissa = '%.*f' % (precision, value)
        if not sci_notation:
            return mantissa
        return r'%s\cdot10^{%d}' % (mantissa, exponent)

    def _one_minus(self, s):
        return r'\overline{%s}' % s if self._use_overline else '1-%s' % s

    def __call__(self, x, pos=None):
        if self._minor and x not in self._labelled:
            return ''
        if x <= 0 or x >= 1:
            return ''
        if _is_close_to_int(2 * x) and round(2 * x) == 1:
            s = self._one_half
        elif x < 0.5 and _is_decade(x, rtol=1e-7):
            s = '10^{%d}' % round(math.log10(x))
        elif x > 0.5 and _is_decade(1 - x, rtol=1e-7):
            s = self._one_minus('10^{%d}' % round(math.log10(1 - x)))
        elif x < 0.1:
            s = self._format_value(x, self._locs)
        elif x > 0.9:
            s = self._one_minus(self._format_value(1 - x, 1 - self._locs))
        else:
            s = self._format_value(x, self._locs, sci_notation=False)
        return '$%s$' % s


def lon_label(lon):
    """cartopy's LongitudeFormatter defaults: 120°W, 0°, 180°."""
    lon = (lon + 180.0) % 360.0 - 180.0
    if abs(lon) < 1e-9:
        return '0°'
    if abs(abs(lon) - 180.0) < 1e-9:
        return '180°'
    return '%g' % abs(lon) + '°' + ('W' if lon < 0 else 'E')


def lat_label(lat):
    """cartopy's LatitudeFormatter defaults: 30°N, 0°, 30°S."""
    if abs(lat) < 1e-9:
        return '0°'
    return '%g' % abs(lat) + '°' + ('S' if lat < 0 else 'N')


class LongitudeFormatter(Formatter):
    """cartopy's LongitudeFormatter defaults as an axis formatter (lon_label)."""

    def __call__(self, x, pos=None):
        return lon_label(x)


class LatitudeFormatter(Formatter):
    """cartopy's LatitudeFormatter defaults as an axis formatter (lat_label)."""

    def __call__(self, x, pos=None):
        return lat_label(x)


def format_values(fmt, values):
    """Contour-label style formatting: a % format, a {} format, a callable,
    a dict level -> text, or None (matplotlib's default ScalarFormatter
    over all the values)."""
    values = np.asarray(values, float)
    if fmt is None:
        if not values.size:
            return []
        labels, _ = ScalarFormatter(use_offset=False).format(values, values.min(), values.max())
        return labels
    if isinstance(fmt, dict):
        return [str(fmt.get(v, fmt.get(float(v), '%g' % v))) for v in values]
    if callable(fmt):
        return [str(fmt(v)) for v in values]
    # a format string as given: an ASCII hyphen, as matplotlib's `fmt % level`
    if '{' in fmt:
        return [fmt.format(v) for v in values]
    return [fmt % v for v in values]


# ------------------------------------------- matplotlib objects (duck typed)

class _AxisStub:
    """What a matplotlib Locator / Formatter reads from its axis, filled in
    from the arguments of a tick_values / format_ticks call."""
    __name__ = 'xaxis'                  # ConciseDateFormatter asks it; inversion follows view
    axis_name = 'x'

    def __init__(self, view, tick_space=None, major=None, minpos=np.inf, scale=None):
        self.view = tuple(view)
        self.data = tuple(view)
        self.tick_space = tick_space
        self.major = major
        self.minpos = minpos
        self.scale = scale

    def get_view_interval(self):
        return self.view

    def set_view_interval(self, vmin, vmax, ignore=False):
        self.view = (vmin, vmax)

    def get_data_interval(self):
        return self.data

    def set_data_interval(self, vmin, vmax, ignore=False):
        self.data = (vmin, vmax)

    def get_minpos(self):
        return self.minpos

    def get_tick_space(self):
        return 9 if self.tick_space is None else self.tick_space

    def get_majorticklocs(self):
        return np.asarray([] if self.major is None else self.major, float)

    def get_inverted(self):
        return self.view[1] < self.view[0]

    def get_scale(self):
        return getattr(self.scale, 'name', 'linear')

    def get_transform(self):
        s = self.scale
        if s is None:
            raise AttributeError('no transform')
        return s.get_transform() if hasattr(s, 'get_transform') else s

    def _get_shared_axis(self):
        return [self]


class _Borrowed:
    """Temporarily give a matplotlib TickHelper the stub axis."""

    def __init__(self, obj, stub):
        self.obj, self.stub = obj, stub

    def __enter__(self):
        self.old = getattr(self.obj, 'axis', None)
        if hasattr(self.obj, 'set_axis'):
            self.obj.set_axis(self.stub)
        return self.obj

    def __exit__(self, *exc):
        if hasattr(self.obj, 'set_axis'):
            self.obj.set_axis(self.old)


class ForeignLocator(Locator):
    """A matplotlib (or cartopy) Locator object behind fastplot's protocol:
    tick_values(vmin, vmax) with the stub axis set, or its __call__ (which
    reads the stub's view) when tick_values can't take numbers (date
    locators) or is not implemented (AutoMinorLocator)."""

    def __init__(self, obj, scale=None):
        self.obj = obj
        self.scale = scale
        self.minpos = np.inf

    def _stub(self, vmin, vmax, tick_space=None, major=None):
        return _AxisStub((vmin, vmax), tick_space, major, self.minpos, self.scale)

    def tick_values(self, vmin, vmax, tick_space=None, major=None):
        with _Borrowed(self.obj, self._stub(vmin, vmax, tick_space, major)) as o:
            try:
                t = o.tick_values(vmin, vmax)
            except (NotImplementedError, TypeError, AttributeError):
                t = o()
        return np.asarray(t, float)

    def nonsingular(self, vmin, vmax):
        with _Borrowed(self.obj, self._stub(vmin, vmax)) as o:
            return o.nonsingular(vmin, vmax) if hasattr(o, 'nonsingular') else \
                Locator.nonsingular(self, vmin, vmax)

    def view_limits(self, vmin, vmax):
        with _Borrowed(self.obj, self._stub(vmin, vmax)) as o:
            return o.view_limits(vmin, vmax) if hasattr(o, 'view_limits') else \
                Locator.view_limits(self, vmin, vmax)


class ForeignFormatter(Formatter):
    """A matplotlib Formatter object behind fastplot's protocol: its
    format_ticks (or set_locs + __call__) and get_offset with the stub axis
    set, \\mathdefault stripped from the labels."""

    def __init__(self, obj, scale=None):
        self.obj = obj
        self.scale = scale

    def format_ticks(self, locs, vmin=None, vmax=None):
        locs = np.asarray(locs, float)
        if vmin is None or vmax is None:
            vmin, vmax = (locs.min(), locs.max()) if locs.size else (0.0, 1.0)
        self._view = (vmin, vmax)
        with _Borrowed(self.obj, _AxisStub((vmin, vmax), scale=self.scale)) as o:
            if hasattr(o, 'format_ticks'):
                labels = o.format_ticks(locs)
            else:
                if hasattr(o, 'set_locs'):
                    o.set_locs(locs)
                labels = [o(v, i) for i, v in enumerate(locs)]
            offset = o.get_offset() if hasattr(o, 'get_offset') else ''
        return [strip_mathdefault(str(s)) for s in labels], strip_mathdefault(str(offset))

    def __call__(self, x, pos=None):
        with _Borrowed(self.obj, _AxisStub(self._view, scale=self.scale)) as o:
            return strip_mathdefault(str(o(x, pos)))


def get_locator(loc, scale=None):
    """An axis locator from what a user passes: None, a fastplot Locator, a
    matplotlib / cartopy Locator object (wrapped), or tick locations."""
    if loc is None or isinstance(loc, Locator):
        return loc
    if hasattr(loc, 'tick_values'):
        return ForeignLocator(loc, scale)
    return FixedLocator(np.asarray(loc, float))


def get_formatter(fmt, scale=None):
    """An axis formatter from what a user passes: None, a fastplot
    Formatter, a matplotlib Formatter object (wrapped), a '{x}' or '%'
    format string, a function f(x, pos) or a sequence of labels."""
    if fmt is None or isinstance(fmt, Formatter):
        return fmt
    if hasattr(fmt, 'format_ticks') or hasattr(fmt, 'set_locs'):
        return ForeignFormatter(fmt, scale)
    if isinstance(fmt, str):
        return StrMethodFormatter(fmt) if '{' in fmt else FormatStrFormatter(fmt)
    if callable(fmt):
        return FuncFormatter(fmt)
    return FixedFormatter([str(s) for s in fmt])
