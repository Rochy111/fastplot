"""Skew-T log-p diagrams with MetPy's SkewT API.

    from fastplot.skewt import SkewT

    skew = SkewT(rotation=45)
    skew.plot(p, T, 'r')
    skew.plot(p, Td, 'g')
    skew.plot_barbs(p, u, v)
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()
    skew.shade_cape(p, T, parcel)
    skew.ax.set_xlim(-40, 50)
    skew.ax.figure.savefig('skewt.png')

The axes is projection='skewx': temperature skewed by `rotation` degrees
in axes units (x += tan(rotation) * y), pressure on a log axis, x ticks
where each edge's range reaches -- MetPy's SkewXAxes. Pressure is in
hPa, temperature in degC, mixing ratio in kg/kg; pint quantities (MetPy's
units) are converted to those. The background lines use MetPy's
formulas and constants (MetPy 1.7): dry adiabats by Poisson's equation,
moist adiabats integrated from 1000 hPa (MetPy's moist lapse rate, with
Ambaum's (2020) saturation vapour pressure over liquid water; RK4 in numba
here, scipy's solve_ivp there), mixing lines by Bolton's dewpoint.
"""
import numpy as np

from . import ticker
from ._jit import capped, njit, prange
from . import transforms as _transforms
from .collections import LineCollection
from ._layout import SubplotSpec

__all__ = ['SkewT', 'Hodograph', 'colored_line']

# MetPy's constants (metpy.constants)
RD = 8.314462618 / 28.96546e-3                         # J / (kg K)
CP_D = 3.5 * RD
KAPPA = RD / CP_D
EPSILON = 18.015268 / 28.96546
LV = 2.50084e6                                          # J / kg
RV = 8.314462618 / 18.015268e-3
CP_L = 4219.4
CP_V = 1860.078011865639
T0 = 273.16                                             # the triple point
ZERO_C = 273.15


def _mag(q, unit):
    """A pint quantity in unit, or numbers as they are."""
    if hasattr(q, 'm_as'):
        return np.asarray(q.m_as(unit), float)
    return np.asarray(q, float)


def _ok(*arrs):
    """MetPy's _delete_masked_points: the points finite in every array."""
    arrs = [np.ma.filled(np.ma.asarray(a, float), np.nan) for a in np.broadcast_arrays(*arrs)]
    keep = np.logical_and.reduce([np.isfinite(a) for a in arrs])
    return [a[keep] for a in arrs]


def dry_lapse(pressure, t0, p0=1000.0):
    """Temperature (K) of air lifted dry-adiabatically from t0 (K) at p0."""
    return t0 * (np.asarray(pressure, float) / p0) ** KAPPA


@njit(cache=True, nogil=True)
def saturation_vapor_pressure(t):
    """Over liquid water, hPa, of temperature t (K): Ambaum (2020) eq. 13,
    the latent heat by his eq. 15 (MetPy's)."""
    latent = LV - (CP_L - CP_V) * (t - T0)
    return 6.112 * (T0 / t) ** ((CP_L - CP_V) / RV) * np.exp((LV / T0 - latent / t) / RV)


@njit(cache=True, nogil=True)
def saturation_mixing_ratio(p, t):
    e = saturation_vapor_pressure(t)
    return EPSILON * e / (p - e)


@njit(cache=True, nogil=True)
def _moist_rate(p, t):
    """MetPy's moist lapse rate dT/dp (K / hPa)."""
    rs = saturation_mixing_ratio(p, t)
    return (RD * t + LV * rs) / (CP_D + LV * LV * rs * EPSILON / (RD * t * t)) / p


@njit(cache=True, nogil=True, parallel=True)
def _moist_lapse(pressure, order, t0, p0, step, out):
    """out[i, k]: the adiabat through t0[i] at p0, at pressure[k]; each
    adiabat by RK4 from p0 out to the pressures in `order` (nearest first),
    either side of p0 on its own."""
    for i in prange(t0.size):
        for direction in (-1.0, 1.0):
            p, t = p0, t0[i]
            for k in order:
                target = pressure[k]
                if (target - p0) * direction < 0.0:
                    continue
                n = max(1, int(np.ceil(abs(target - p) / step)))
                h = (target - p) / n
                for _ in range(n):
                    k1 = _moist_rate(p, t)
                    k2 = _moist_rate(p + h / 2, t + h / 2 * k1)
                    k3 = _moist_rate(p + h / 2, t + h / 2 * k2)
                    k4 = _moist_rate(p + h, t + h * k3)
                    t = t + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                    p += h
                p = target
                out[i, k] = t


@capped
def moist_lapse(pressure, t0, p0=1000.0, step=1.0):
    """Temperature (K) along moist adiabats through t0 (K, any shape) at p0
    (hPa), at each pressure: MetPy's dT/dp integrated by RK4 in steps of at
    most `step` hPa (numba, an adiabat a core). Shape t0.shape +
    pressure.shape."""
    pressure = np.ascontiguousarray(np.atleast_1d(np.asarray(pressure, float)))
    t0 = np.asarray(t0, float)
    out = np.empty((t0.size, pressure.size))
    order = np.argsort(np.abs(pressure - p0), kind='stable').astype(np.int64)
    _moist_lapse(pressure, order, np.ascontiguousarray(t0.ravel()), float(p0), float(step), out)
    return out.reshape(t0.shape + pressure.shape)


def dewpoint_from_vapor_pressure(e):
    """Bolton's inversion, degC, of vapour pressure e (hPa)."""
    val = np.log(e / 6.112)
    return 243.5 * val / (17.67 - val)


class SkewT:
    """MetPy's SkewT: a skew-T log-p diagram in a figure."""

    def __init__(self, fig=None, rotation=30, subplot=None, rect=None, aspect=80.5):
        if fig is None:
            from . import pyplot
            fig = pyplot.figure(figsize=(7, 7))
        self.fig = fig
        kw = dict(projection='skewx', rotation=rotation)
        if rect is not None:
            self.ax = fig.add_axes(rect, **kw)
        elif isinstance(subplot, (SubplotSpec, int)):
            self.ax = fig.add_subplot(subplot, **kw)
        elif subplot is not None:
            self.ax = fig.add_subplot(*subplot, **kw)
        else:
            self.ax = fig.add_subplot(1, 1, 1, **kw)
        ax = self.ax
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_ylim(1050, 100)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.set_xlim(-40, 50)
        ax.grid(True)
        self.mixing_lines = self.dry_adiabats = self.moist_adiabats = None
        ax.set_aspect(aspect, adjustable='box')

    # -- data
    def plot(self, pressure, t, *args, **kwargs):
        """Temperature against pressure (ax.plot): its lines."""
        t, pressure = _ok(_mag(t, 'degC'), _mag(pressure, 'hPa'))
        return self.ax.plot(t, pressure, *args, **kwargs)

    def plot_barbs(self, pressure, u, v, c=None, xloc=1.0, x_clip_radius=0.1,
                   y_clip_radius=0.08, plot_units=None, **kwargs):
        """Wind barbs at xloc (axes fraction) beside the diagram, drawn
        past the axes' edge by the clip radii (axes fractions)."""
        if plot_units is not None:
            u, v = _mag(u, plot_units), _mag(v, plot_units)
        else:
            u = np.asarray(getattr(u, 'magnitude', u), float)
            v = np.asarray(getattr(v, 'magnitude', v), float)
        pressure = _mag(pressure, 'hPa')
        x = np.full_like(pressure, xloc, dtype=float)
        tr = _transforms.blended_transform_factory(self.ax.transAxes, self.ax.transData)
        kwargs.setdefault('zorder', 2)
        b = self.ax.barbs(x, pressure, u, v, transform=tr, **kwargs)
        b.clip_axes = (xloc - x_clip_radius, -y_clip_radius, xloc + x_clip_radius,
                       1.0 + y_clip_radius)
        return b

    # -- the background
    def _lines(self, attr, t, pressure, kwargs, defaults):
        old = getattr(self, attr)
        if old is not None:
            old.remove()
        for k, v in defaults.items():
            kwargs.setdefault(k, v)
        lc = LineCollection([np.column_stack([ti, pressure]) for ti in t], **kwargs)
        setattr(self, attr, lc)
        return self.ax.add_collection(lc)

    def _pressures(self, pressure):
        if pressure is None:
            return np.linspace(*self.ax.get_ylim())
        return _mag(pressure, 'hPa')

    def plot_dry_adiabats(self, t0=None, pressure=None, **kwargs):
        """Dry adiabats through t0 (degC) at 1000 hPa: every 10 degC across
        the diagram by default."""
        if t0 is None:
            xmin, xmax = self.ax.get_xlim()
            t0 = np.arange(xmin, xmax + 1, 10)
        t0 = _mag(t0, 'degC')
        pressure = self._pressures(pressure)
        t = dry_lapse(pressure, (t0 + ZERO_C)[:, None]) - ZERO_C
        return self._lines('dry_adiabats', t, pressure, kwargs,
                           dict(colors='r', linestyles='dashed', alpha=0.5))

    def plot_moist_adiabats(self, t0=None, pressure=None, **kwargs):
        """Moist adiabats through t0 (degC) at 1000 hPa: every 10 degC
        below 0 and 5 degC above by default."""
        if t0 is None:
            xmin, xmax = self.ax.get_xlim()
            t0 = np.concatenate((np.arange(xmin, 0, 10), np.arange(0, xmax + 1, 5)))
        t0 = _mag(t0, 'degC')
        pressure = self._pressures(pressure)
        t = moist_lapse(pressure, t0 + ZERO_C) - ZERO_C
        return self._lines('moist_adiabats', t, pressure, kwargs,
                           dict(colors='b', linestyles='dashed', alpha=0.5))

    def plot_mixing_lines(self, mixing_ratio=None, pressure=None, **kwargs):
        """Lines of constant saturation mixing ratio (kg/kg), 600 hPa down."""
        if mixing_ratio is None:
            mixing_ratio = np.array([0.0004, 0.001, 0.002, 0.004, 0.007, 0.01, 0.016, 0.024,
                                     0.032])
        mixing_ratio = np.asarray(getattr(mixing_ratio, 'magnitude', mixing_ratio),
                                  float).reshape(-1, 1)
        if pressure is None:
            pressure = np.linspace(600, max(self.ax.get_ylim()))
        else:
            pressure = _mag(pressure, 'hPa')
        e = pressure * mixing_ratio / (EPSILON + mixing_ratio)
        td = dewpoint_from_vapor_pressure(e)
        return self._lines('mixing_lines', td, pressure, kwargs,
                           dict(colors='g', linestyles='dashed', alpha=0.8))

    # -- shading
    def shade_area(self, y, x1, x2=0, which='both', **kwargs):
        """fill_betweenx of x1 and x2 where x1 > x2 ('positive'), x1 < x2
        ('negative') or everywhere ('both')."""
        x1, x2 = np.broadcast_arrays(np.asarray(x1, float), np.asarray(x2, float))
        props = {'positive': dict(facecolor='tab:red', alpha=0.4, where=x1 > x2),
                 'negative': dict(facecolor='tab:blue', alpha=0.4, where=x1 < x2),
                 'both': dict(facecolor='tab:green', alpha=0.4, where=None)}
        if which not in props:
            raise ValueError("which must be 'positive', 'negative' or 'both'")
        args = props[which]
        args.update(kwargs)
        where = args.pop('where')
        arrs = (y, x1, x2) if where is None else (y, x1, x2, where)
        arrs = _ok(*arrs)
        if where is not None:
            arrs[3] = arrs[3].astype(bool)
            args['where'] = arrs.pop()
        return self.ax.fill_betweenx(*arrs, interpolate=True, **args)

    def shade_cape(self, pressure, t, t_parcel, **kwargs):
        """Red where the parcel is warmer than the environment."""
        kwargs.setdefault('facecolor', 'tab:red')
        kwargs.setdefault('alpha', 0.3)
        return self.shade_area(_mag(pressure, 'hPa'), _mag(t_parcel, 'degC'), _mag(t, 'degC'),
                               which='positive', **kwargs)

    def shade_cin(self, pressure, t, t_parcel, dewpoint=None, **kwargs):
        """Blue where the parcel is colder than the environment (with a
        dewpoint: only below the level of free convection, the first level
        from the surface up where the parcel turns warmer)."""
        kwargs.setdefault('facecolor', 'tab:blue')
        kwargs.setdefault('alpha', 0.3)
        p, te, tp = _mag(pressure, 'hPa'), _mag(t, 'degC'), _mag(t_parcel, 'degC')
        if dewpoint is not None:
            warm = np.nonzero(tp[1:] > te[1:])[0]
            if warm.size:
                keep = p >= p[warm[0] + 1]
                p, te, tp = p[keep], te[keep], tp[keep]
        return self.shade_area(p, tp, te, which='negative', **kwargs)


def colored_line(x, y, c, cmap=None, norm=None, **kwargs):
    """MetPy's colored_line: a LineCollection of the segments between
    consecutive points, segment i coloured by c[i] through cmap and norm
    (scaled to c when not given); NaN points left out."""
    from . import _color
    x, y, c = (np.asarray(a, float) for a in (x, y, c))
    keep = ~(np.isnan(x) | np.isnan(y) | np.isnan(c))
    x, y, c = x[keep], y[keep], c[keep]
    pts = np.column_stack([x, y])
    segs = np.stack([pts[:-1], pts[1:]], 1)
    cm = _color.get_cmap(cmap)
    cnorm = _color.make_norm(norm, data=c, N=cm.N)
    vals = c[:len(segs)].astype(np.float32)
    out = np.empty((1, vals.size, 4), np.uint8)
    if vals.size:
        _color.shade(vals[None, :], cm, cnorm, out=out, rounding='nearest')
    lc = LineCollection(list(segs), colors=out[0] if vals.size else 'k', **kwargs)
    lc.cmap, lc.norm, lc.values = cm, cnorm, c        # a colorbar's mappable
    lc.norm_spec = lambda: cnorm
    return lc


class Hodograph:
    """MetPy's Hodograph: wind components on equal axes, range rings."""

    def __init__(self, ax=None, component_range=80):
        if ax is None:
            from . import pyplot
            ax = pyplot.figure().add_subplot(1, 1, 1)
        self.ax = ax
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-component_range, component_range)
        ax.set_ylim(-component_range, component_range)
        self.max_range = 1.4142135 * component_range

    def add_grid(self, increment=10., **kwargs):
        """Dashed grey rings every `increment`, and the axes through 0."""
        from .patches import Circle
        grid_args = {'color': 'grey', 'linestyle': 'dashed', 'zorder': 2.0 - 0.001}
        grid_args.update(kwargs)
        circle_args = dict(grid_args)
        circle_args['edgecolor'] = circle_args.pop('color', None)
        circle_args['fill'] = False
        self.rings = []
        for r in np.arange(increment, self.max_range, increment):
            c = Circle((0, 0), radius=r, **circle_args)
            self.ax.add_patch(c)
            self.rings.append(c)
        self.yaxis = self.ax.axvline(0, **grid_args)
        self.xaxis = self.ax.axhline(0, **grid_args)

    @staticmethod
    def _form_line_args(kwargs):
        args = {'linewidth': 3}
        args.update(kwargs)
        return args

    def plot(self, u, v, **kwargs):
        """The hodograph line (3 wide)."""
        u, v = _ok(_mag(u, 'knots') if hasattr(u, 'm_as') else u,
                   _mag(v, 'knots') if hasattr(v, 'm_as') else v)
        return self.ax.plot(u, v, **self._form_line_args(kwargs))

    def wind_vectors(self, u, v, **kwargs):
        """Arrows from the origin to each (u, v), in data units."""
        args = {'units': 'xy', 'scale': 1}
        args.update(kwargs)
        u = np.asarray(getattr(u, 'magnitude', u), float)
        v = np.asarray(getattr(v, 'magnitude', v), float)
        zero = np.zeros_like(u)
        return self.ax.quiver(zero, zero, u, v, **args)

    def plot_colormapped(self, u, v, c, intervals=None, colors=None, **kwargs):
        """The hodograph coloured by c: through a colormap, or (colors and
        intervals) a colour per interval, the line cut at each boundary
        (with heights as pint quantities, as MetPy does)."""
        from . import colors as mcolors
        u = np.asarray(getattr(u, 'magnitude', u), float)
        v = np.asarray(getattr(v, 'magnitude', v), float)
        if colors:
            cmap = mcolors.ListedColormap(colors).with_extremes(over='none', under='none')
            if hasattr(intervals, 'units') and hasattr(c, 'm_as'):
                c = c.m_as(intervals.units)
                bounds = np.asarray(intervals.magnitude, float)
                lo, hi = np.nanmin(c), np.nanmax(c)
                extra = np.sort([b for b in bounds if b not in c and lo <= b <= hi])
                order = np.argsort(c)
                u = np.concatenate([u, np.interp(extra, c[order], u[order])])
                v = np.concatenate([v, np.interp(extra, c[order], v[order])])
                c = np.concatenate([c, extra])
                s = np.argsort(c, kind='stable')
                u, v, c, intervals = u[s], v[s], c[s], bounds
            intervals = np.asarray(getattr(intervals, 'magnitude', intervals), float)
            kwargs['cmap'] = cmap
            kwargs['norm'] = mcolors.BoundaryNorm(intervals, cmap.N)
        c = np.asarray(getattr(c, 'magnitude', c), float)
        lc = colored_line(u, v, c, **self._form_line_args(kwargs))
        self.ax.add_collection(lc)
        return lc
