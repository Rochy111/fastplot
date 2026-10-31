"""matplotlib's Axes: plain x/y axes -- scales, limits, ticks, spines,
grid, axis labels -- and AxesBase's hooks answered through them.

    ax = fig.add_subplot()            # (no projection)
    ax.plot(t, T); ax.set_yscale('log'); ax.set_xlabel('time')

Data -> pixels: each axis' scale (linear, log, symlog, ...) then its
limits map the data to 0..1 across the axes; an optional skew (x += k y,
in those axes units -- a SkewT's) and the axes' box in the canvas finish
it. Autoscaling is matplotlib's: the union of the artists' data limits,
margins (5%) added in the scale's own space, never past a sticky edge
(images, pcolormesh, bars' bases), nonsingular. Ticks are the scale's
locators and formatters (_ticker, _scale, _dates), drawn where matplotlib
draws them: marks outward from the spines, labels a pad beyond, the axis
label beyond the labels, the offset text at the axis' end.

The axis (grid, ticks) draws at zorder 1.5 (axes.axisbelow 'line': above
images and patches, below lines), the spines at 2.5, among the artists.
Fields on plain axes are sampled per column and per row (a pixel's grid
index along x depends only on its column) with _raster's separable
kernel, on the CPU or the GPU; a skewed axes draws them by cells.
"""
import inspect

import numpy as np

from . import _draw, _raster, _scale, _ticker
from ._jit import njit
from . import transforms as _transforms
from ._style import _FONT_SCALE, _dashes, fontsize_pt, rc, rgba
from ._layout import SubplotSpec, _fit, spec_box
from ._axesbase import AxesBase
from ._plots import PlotMixin
from ._stats import StatsMixin

SPINE_ZORDER = 2.5
OFFSET_PAD = 3.0                    # Axis.OFFSETTEXTPAD, points
_AXISBELOW = {True: 0.5, 'line': 1.5, False: 2.5}   # axes.axisbelow -> the axis' zorder
_DEFAULT_LIMS = {'log': (1, 10), 'logit': (0.1, 0.9)}   # a locator's nonsingular(-inf, inf),
                                                        # where it isn't (-0.05, 0.05)


_ACCEPTS = {}                       # (function, name) -> takes it? (inspect.signature is slow)


def _accepts(fn, name):
    key = (getattr(fn, '__func__', fn), name)          # a bound method: its function's
    try:
        return _ACCEPTS[key]
    except (KeyError, TypeError):
        pass
    try:
        ok = name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        ok = False
    try:
        _ACCEPTS[key] = ok
    except TypeError:                                   # (unhashable)
        pass
    return ok


def tick_values(loc, vmin, vmax, space=None, major=None):
    """A locator's ticks, whatever of (tick_space, major) it takes."""
    kw = {}
    if space is not None and _accepts(loc.tick_values, 'tick_space'):
        kw['tick_space'] = space
    if major is not None and _accepts(loc.tick_values, 'major'):
        kw['major'] = major
    return np.asarray(loc.tick_values(vmin, vmax, **kw), float)


def format_ticks(fmt, locs, vmin, vmax):
    """(labels, offset text) of a formatter (format_ticks, or the older format)."""
    if hasattr(fmt, 'format_ticks') and _accepts(fmt.format_ticks, 'vmin'):
        return fmt.format_ticks(locs, vmin, vmax)
    if hasattr(fmt, 'format') and not isinstance(fmt, str):
        return fmt.format(locs, vmin, vmax)
    raise TypeError('not a formatter: %r' % (fmt,))


def _mid_edges(c):
    """Cell edges round centres c: the mid-points, half a step out at the
    ends (matplotlib's pcolormesh shading='nearest')."""
    c = np.asarray(c, np.float64)
    if c.size < 2:
        return np.r_[c - 0.5, c + 0.5]
    mid = 0.5 * (c[1:] + c[:-1])
    return np.r_[c[0] - (mid[0] - c[0]), mid, c[-1] + (c[-1] - mid[-1])]


def _across(pos, a, b, axis):
    """Segments across an axis (ticks, grid lines), (n, 2, 2): at each pos
    along coordinate `axis`, from a to b along the other."""
    s = np.empty((pos.size, 2, 2))
    s[:, :, axis] = pos[:, None]
    s[:, 0, 1 - axis] = a
    s[:, 1, 1 - axis] = b
    return s


class _CategoryLocator(_ticker.Locator):
    """matplotlib's StrCategoryLocator: a tick at every category."""

    def __init__(self, units):
        self.units = units

    def tick_values(self, vmin=None, vmax=None, tick_space=None, major=None):
        return np.asarray(list(self.units.values()), float)


class _CategoryFormatter(_ticker.Formatter):
    """matplotlib's StrCategoryFormatter: a category's tick its name."""

    def __init__(self, units):
        self.units = units

    def __call__(self, x, pos=None):
        names = {v: k for k, v in self.units.items()}
        return names.get(int(np.round(x)), '') if np.isfinite(x) else ''

    def get_offset(self):
        return ''


def _is_categorical(v):
    a = v if isinstance(v, np.ndarray) else np.asarray(v, dtype=object)
    if a.dtype.kind in 'US':
        return True
    return a.dtype.kind == 'O' and a.size > 0 and all(isinstance(e, (str, bytes))
                                                      for e in a.ravel())


def _as_formatter(f):
    """matplotlib's set_major_formatter: a Formatter, a str ('{x:.1f}' or
    '%.1f'), or a function f(x, pos)."""
    if isinstance(f, str):
        return _ticker.StrMethodFormatter(f) if '{' in f else _ticker.FormatStrFormatter(f)
    if callable(f) and not hasattr(f, 'format_ticks') and not hasattr(f, 'format'):
        return _ticker.FuncFormatter(f)
    return _ticker.as_formatter(f) if hasattr(_ticker, 'as_formatter') else f


class Spine:
    """One side of the frame (ax.spines['left'] ...)."""

    zorder = SPINE_ZORDER

    def __init__(self, axes, side):
        self.axes, self.side = axes, side
        self.linewidth, self.edgecolor = rc('axes.linewidth'), rc('axes.edgecolor')
        self.visible = rc('axes.spines.' + side)
        self.position = ('outward', 0.0)
        self.linestyle = '-'
        self.alpha = None
        self.bounds = None

    def set_linewidth(self, w):
        self.linewidth = w

    def set_edgecolor(self, c):
        self.edgecolor = c

    set_color = set_edgecolor

    def set_linestyle(self, ls):
        self.linestyle = ls

    def set_alpha(self, a):
        self.alpha = a

    def set_visible(self, v):
        self.visible = bool(v)

    def get_visible(self):
        return self.visible

    def set_bounds(self, low=None, high=None):
        if high is None and np.iterable(low):
            low, high = low
        self.bounds = None if low is None and high is None else (low, high)

    def get_bounds(self):
        return self.bounds

    def set_color(self, c):
        self.edgecolor = c

    def get_edgecolor(self):
        return self.edgecolor

    def get_linewidth(self):
        return self.linewidth

    def set_zorder(self, z):
        self.zorder = z

    def set_capstyle(self, c):
        pass

    def set_smart_bounds(self, b):
        pass

    def set(self, **kw):
        for k, v in kw.items():
            getattr(self, 'set_' + {'lw': 'linewidth', 'ls': 'linestyle', 'ec': 'edgecolor',
                                    'c': 'color'}.get(k, k))(v)

    def set_position(self, position):
        """('outward', points), ('axes', fraction), ('data', value), 'zero', 'center'."""
        if position == 'center':
            position = ('axes', 0.5)
        elif position == 'zero':
            position = ('data', 0.0)
        self.position = (position[0], float(position[1]))

    def get_position(self):
        return self.position

    def offset_px(self, r):
        """Where the spine sits: its coordinate across the axes, in canvas px."""
        ax = self.axes
        x0, y0, x1, y1 = ax.box_f
        kind, amount = self.position
        vertical = self.side in ('left', 'right')
        if kind == 'outward':
            d = r.px(amount)
            return {'left': x0 - d, 'right': x1 + d, 'bottom': y1 + d, 'top': y0 - d}[self.side]
        if kind == 'axes':
            return x0 + amount * (x1 - x0) if vertical else y1 - amount * (y1 - y0)
        if vertical:                                    # ('data', value)
            return float(ax._data_px(np.array([amount]), np.array([ax.get_ylim()[0]]))[0][0])
        return float(ax._data_px(np.array([ax.get_xlim()[0]]), np.array([amount]))[1][0])


class Spines(dict):
    """ax.spines: matplotlib's Spines -- by name, as attributes
    (spines.left), or several at once (spines[['top', 'right']],
    spines[:]) whose setters act on each."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key != slice(None):
                raise ValueError('Spines does not support slicing except for the fully open '
                                 'slice [:] to access all spines.')
            return _SpinesProxy(list(self.values()))
        if isinstance(key, (list, tuple)):
            return _SpinesProxy([dict.__getitem__(self, k) for k in key])
        return dict.__getitem__(self, key)


class _SpinesProxy:
    """Several spines: each set_* call made on all of them."""

    def __init__(self, spines):
        self._spines = spines

    def __getattr__(self, name):
        if not name.startswith('set'):
            raise AttributeError(name)

        def call(*args, **kw):
            for s in self._spines:
                getattr(s, name)(*args, **kw)
        return call


class Axis:
    """An axis of plain axes (ax.xaxis, ax.yaxis): its scale, limits,
    locators and formatters, tick parameters, label and grid."""

    def __init__(self, axes, name):
        self.axes = axes
        self.axis_name = name
        self.set_scale('linear')
        self.lim = None                                 # set by set_xlim; None: autoscale
        self.auto = True
        self.settled = None                             # (artists then, limits): autoscaled early
        self.margin = rc('axes.%smargin' % name)
        self.shared = [self]                            # the axes' axes sharing these limits
        self.dates = False
        self.categories = None                          # str -> its integer (category axes)
        self.inverted = False                           # kept through autoscaling
        t = name + 'tick.'
        s1, s2 = ('bottom', 'top') if name == 'x' else ('left', 'right')
        grid = rc('axes.grid') and rc('axes.grid.axis') in (name, 'both')
        which = rc('axes.grid.which')
        self.ticks = {}
        for w in ('major', 'minor'):
            lc = rc(t + 'labelcolor')
            self.ticks[w] = dict(
                size=rc(t + w + '.size'), width=rc(t + w + '.width'), pad=rc(t + w + '.pad'),
                color=rc(t + 'color'), labelsize=rc(t + 'labelsize'),
                labelcolor=rc(t + 'color') if lc == 'inherit' else lc,
                direction=rc(t + 'direction'), rotation=0.0, on1=rc(t + s1), on2=rc(t + s2),
                label1=rc(t + 'label' + s1), label2=rc(t + 'label' + s2),
                grid=grid and which in (w, 'both'), grid_kw={})
        self.label_text = ''
        self.label_kw = {}
        self.labelpad = rc('axes.labelpad')
        self.label_position = 'bottom' if name == 'x' else 'left'
        self.offset_position = None
        self.visible = True
        self.fixed_labels = None                        # set_xticklabels
        self.tick_label_kw = {}

    # -- scale, locators, formatters
    def set_scale(self, name, **kw):
        from . import _scale
        self.scale = _scale.get_scale(name, **kw)
        (self.major_locator, self.minor_locator, self.major_formatter,
         self.minor_formatter) = self.scale.default_locators()
        self._minor_visible = getattr(self.scale, 'name', 'linear') not in ('linear',) or \
            rc(self.axis_name + 'tick.minor.visible')

    def get_scale(self):
        return getattr(self.scale, 'name', 'linear')

    def set_major_locator(self, loc):
        self.major_locator = _ticker.get_locator(loc, self.scale)    # (matplotlib's wrapped)

    def set_minor_locator(self, loc):
        self.minor_locator = _ticker.get_locator(loc, self.scale)
        self._minor_visible = True

    def set_major_formatter(self, f):
        self.major_formatter = _ticker.get_formatter(f, self.scale)

    def set_minor_formatter(self, f):
        self.minor_formatter = _ticker.get_formatter(f, self.scale)

    def get_major_locator(self):
        return self.major_locator

    def get_major_formatter(self):
        return self.major_formatter

    def get_minor_locator(self):
        return self.minor_locator

    def get_minor_formatter(self):
        return self.minor_formatter

    def set_ticks(self, ticks, labels=None, minor=False, **kw):
        loc = _ticker.FixedLocator(np.asarray(ticks, float))
        if minor:
            self.minor_locator = loc
            self._minor_visible = True
        else:
            self.major_locator = loc
        t = np.asarray(ticks, float)
        t = t[np.isfinite(t)]
        if t.size:                                      # matplotlib: the view grows to hold them
            if self.lim is not None and not self.auto:
                lo, hi = self.lim
                inv = lo > hi
                lo, hi = min(lo, hi, t.min()), max(lo, hi, t.max())
                self.axes._set_lim(self.axis_name, (hi, lo) if inv else (lo, hi), auto=None)
            else:
                self._tick_span = (float(t.min()), float(t.max()))
        if labels is not None:
            self.set_ticklabels(labels, minor=minor, **kw)
        return list(ticks)

    def set_ticklabels(self, labels, minor=False, **kw):
        fixed = [str(t) for t in labels]
        f = _ticker.FixedFormatter(fixed)
        if minor:
            self.minor_formatter = f
        else:
            self.major_formatter = f
        self.tick_label_kw.update(kw)

    def get_majorticklocs(self):
        vmin, vmax = sorted(self.get_view_interval())
        return tick_values(self.major_locator, vmin, vmax)

    def category_values(self, v):
        """matplotlib's StrCategoryConverter: each new string the next
        integer, the axis ticking every category with its name."""
        if self.categories is None:                     # one map for the shared axes
            cats = next((a.categories for a in self.shared if a.categories is not None), {})
            for a in self.shared:
                a.categories = cats
                a.major_locator = _CategoryLocator(cats)
                a.major_formatter = _CategoryFormatter(cats)
        a = np.asarray(v, dtype=object)
        for e in a.ravel():
            e = e.decode() if isinstance(e, bytes) else e
            self.categories.setdefault(e, float(len(self.categories)))
        return np.array([self.categories[e.decode() if isinstance(e, bytes) else e]
                         for e in a.ravel()], float).reshape(a.shape)

    def axis_date(self, tz=None):
        from . import _dates
        if not self.dates:                              # the shared axes too, as matplotlib's
            loc = _dates.AutoDateLocator()                # shared converter
            fmt = _dates.AutoDateFormatter(loc)
            for a in self.shared:
                if not a.dates:
                    a.dates = True
                    a.major_locator, a.major_formatter = loc, fmt

    # -- ticks
    def set_tick_params(self, which='major', reset=False, **kw):
        whiches = ('major', 'minor') if which == 'both' else (which,)
        alias = {'length': 'size', 'labelrotation': 'rotation', 'tickdir': 'direction',
                 'tick1On': 'side1', 'tick2On': 'side2', 'label1On': 'labelside1',
                 'label2On': 'labelside2'}
        side1, side2 = ('bottom', 'top') if self.axis_name == 'x' else ('left', 'right')
        for w in whiches:
            t = self.ticks[w]
            for k, v in kw.items():
                k = alias.get(k, k)
                if k == 'colors':
                    t['color'] = t['labelcolor'] = v
                elif k in (side1, 'side1'):
                    t['on1'] = bool(v)
                elif k in (side2, 'side2'):
                    t['on2'] = bool(v)
                elif k in ('label' + side1, 'labelside1'):
                    t['label1'] = bool(v)
                elif k in ('label' + side2, 'labelside2'):
                    t['label2'] = bool(v)
                elif k.startswith('grid_'):
                    t['grid_kw'][k[5:]] = v
                elif k in ('size', 'width', 'pad', 'color', 'labelsize', 'labelcolor',
                           'direction', 'rotation'):
                    t[k] = v
                    if k == 'labelsize':
                        t['labelsize_set'] = True
                elif k == 'gridOn':
                    t['grid'] = bool(v)
                elif k == 'labelfontfamily':
                    self.tick_label_kw['family'] = v
                elif k in ('zorder', 'labelrotation_mode'):
                    if k == 'labelrotation_mode':
                        self.tick_label_kw['rotation_mode'] = v
                    continue
                else:
                    raise TypeError('tick_params: unsupported keyword %r' % k)

    def set_ticks_position(self, position):
        a, b = ('bottom', 'top') if self.axis_name == 'x' else ('left', 'right')
        for w in ('major', 'minor'):
            t = self.ticks[w]
            if position == a:
                t.update(on1=True, on2=False, label1=True, label2=False)
            elif position == b:
                t.update(on1=False, on2=True, label1=False, label2=True)
            elif position == 'both':
                t.update(on1=True, on2=True)
            elif position == 'none':
                t.update(on1=False, on2=False)
            elif position == 'default':
                t.update(on1=True, on2=False, label1=True, label2=False)
        if position in (a, b):
            self.label_position = position

    def tick_bottom(self):
        self.set_ticks_position('bottom')

    def tick_top(self):
        self.set_ticks_position('top')

    def tick_left(self):
        self.set_ticks_position('left')

    def tick_right(self):
        self.set_ticks_position('right')

    def set_label_position(self, position):
        self.label_position = position

    def set_label_text(self, label, **kw):
        self.label_text = label
        self.label_kw.update(kw)

    def get_label_text(self):
        return self.label_text

    def set_visible(self, v):
        self.visible = bool(v)

    def get_visible(self):
        return self.visible

    def grid(self, visible=None, which='major', **kw):
        whiches = ('major', 'minor') if which == 'both' else (which,)
        for w in whiches:
            t = self.ticks[w]
            if visible is None and not kw:
                t['grid'] = not t['grid']
            else:
                t['grid'] = True if visible is None else bool(visible)
            t['grid_kw'].update(kw)

    # -- limits
    def get_view_interval(self):
        return self.axes._get_lim(self.axis_name)

    def get_minorticklocs(self):
        if not self._minor_visible:
            return np.array([])
        vmin, vmax = sorted(self.get_view_interval())
        major = self.get_majorticklocs()
        minor = tick_values(self.minor_locator, vmin, vmax, major=major)
        tol = 1e-5 * max(abs(vmax - vmin), 1e-300)
        return np.array([v for v in minor if np.abs(major - v).min(initial=np.inf) > tol])

    def get_ticklocs(self, *, minor=False):
        return self.get_minorticklocs() if minor else self.get_majorticklocs()

    def get_ticklabels(self, minor=False, which=None):
        return self.axes._ticklabel_texts(self, 'minor' if minor or which == 'minor' else 'major')

    def get_major_ticks(self, numticks=None):
        """matplotlib's Tick objects, for what scripts do with them:
        tick.label1 / label2 (Text-like), tick1line / tick2line / gridline
        visibility, set_pad. Changes apply to the axis' ticks."""
        labs = self.get_ticklabels()
        n = len(labs) if numticks is None else numticks
        return [_Tick(self, 'major', labs[k] if k < len(labs) else _TickLabel(self, '', 0.0))
                for k in range(n)]

    def get_minor_ticks(self, numticks=None):
        labs = self.get_ticklabels(minor=True)
        n = len(labs) if numticks is None else numticks
        return [_Tick(self, 'minor', labs[k] if k < len(labs) else _TickLabel(self, '', 0.0))
                for k in range(n)]

    get_majorticklabels = get_ticklabels

    def get_minorticklabels(self):
        return self.get_ticklabels(minor=True)

    def set_label_coords(self, x, y, transform=None):
        self.label_coords = (float(x), float(y))

    @property
    def label(self):
        return _LabelText(self)

    def get_label(self):
        return _LabelText(self)

    def set_tick_params_all(self, **kw):
        self.set_tick_params('both', **kw)

    def get_tick_space(self):
        return 9

    def set_data_interval(self, vmin, vmax, ignore=False):
        pass

    def set_view_interval(self, vmin, vmax, ignore=False):
        self.axes._set_lim(self.axis_name, (vmin, vmax), auto=None)

    def get_data_interval(self):
        lo, hi, _, _ = self.axes._data_limits(self.axis_name)
        return np.array([lo, hi])

    def set_units(self, u):
        pass

    def update_units(self, data):
        return False

    def convert_units(self, x):
        return self.axes._to_num(self, x)

    def set_major_ticks_visible(self, v):
        self.set_tick_params('major', **{s: bool(v) for s in self._sides()})

    def _sides(self):
        return ('bottom', 'top') if self.axis_name == 'x' else ('left', 'right')

    def get_ticks_position(self):
        t = self.ticks['major']
        a, b = self._sides()
        if t['on1'] and t['on2']:
            return 'default' if not t['label2'] else 'unknown'
        return a if t['on1'] else (b if t['on2'] else 'unknown')

    def get_label_position(self):
        return self.label_position

    def set_minor_visible(self, v):
        self._minor_visible = bool(v)

    def axis_name_(self):
        return self.axis_name

    def set_inverted(self, inverted):
        for a in self.shared:
            a.inverted = bool(inverted)
        if self.lim is not None and not self.auto:
            lo, hi = sorted(self.lim)
            self.axes._set_lim(self.axis_name, (hi, lo) if inverted else (lo, hi), auto=None)

    def get_inverted(self):
        a, b = self.axes._get_lim(self.axis_name)
        return a > b

    def transform(self, v):
        return self.scale.transform(np.asarray(v, float))


class _TickLabel:
    """One tick label as matplotlib's Text: get_text(); its setters act on
    all the axis' tick labels (what code that loops over them does)."""

    def __init__(self, axis, text, value):
        self.axis, self._text, self._value = axis, text, value

    def get_text(self):
        return self._text

    def get_position(self):
        return (self._value, 0) if self.axis.axis_name == 'x' else (0, self._value)

    def __repr__(self):
        return 'Text(%r)' % self._text

    def __str__(self):
        return self._text

    def _set(self, k, v):
        self.axis.tick_label_kw[k] = v

    def set_rotation(self, r):
        self._set('rotation', r)

    def set_fontsize(self, s):
        self._set('fontsize', s)

    set_size = set_fontsize

    def set_color(self, c):
        self._set('color', c)

    def set_ha(self, ha):
        self._set('ha', ha)

    set_horizontalalignment = set_ha

    def set_va(self, va):
        self._set('va', va)

    set_verticalalignment = set_va

    def set_fontweight(self, w):
        self._set('fontweight', w)

    set_weight = set_fontweight

    def set_rotation_mode(self, m):
        self._set('rotation_mode', m)

    def set_visible(self, v):
        """This tick's label only (the axis keeps the others)."""
        hidden = self.axis.__dict__.setdefault('_hidden_labels', set())
        if v:
            hidden.discard(float(self._value))
        else:
            hidden.add(float(self._value))

    def set_fontfamily(self, f):
        self._set('family', f)

    set_family = set_fontfamily

    def set_fontstyle(self, s):
        self._set('fontstyle', s)

    def set_text(self, s):
        pass

    def set(self, **kw):
        for k, v in kw.items():
            getattr(self, 'set_' + {'fontsize': 'fontsize', 'size': 'fontsize',
                                    'horizontalalignment': 'ha', 'c': 'color'}.get(k, k))(v)


class _TickLine:
    """tick.tick1line / tick2line / gridline: their visibility, colour, size."""

    def __init__(self, axis, which, part):
        self.axis, self.which, self.part = axis, which, part

    def set_visible(self, v):
        a, b = self.axis._sides()
        key = {'tick1': a, 'tick2': b}.get(self.part)
        if key is not None:
            self.axis.set_tick_params(self.which, **{key: bool(v)})
        else:
            self.axis.ticks[self.which]['grid'] = bool(v)

    def set_color(self, c):
        if self.part == 'grid':
            self.axis.ticks[self.which]['grid_kw']['color'] = c
        else:
            self.axis.set_tick_params(self.which, color=c)

    def set_markersize(self, s):
        self.axis.set_tick_params(self.which, size=s)

    def set_markeredgewidth(self, w):
        self.axis.set_tick_params(self.which, width=w)

    def set_linewidth(self, w):
        if self.part == 'grid':
            self.axis.ticks[self.which]['grid_kw']['linewidth'] = w

    def set_linestyle(self, ls):
        if self.part == 'grid':
            self.axis.ticks[self.which]['grid_kw']['linestyle'] = ls


class _Tick:
    """One of get_major_ticks()'s: its label and lines; setters act on the axis."""

    def __init__(self, axis, which, label):
        self.axis, self.which = axis, which
        self.label1 = self.label = label
        self.label2 = label
        self.tick1line = _TickLine(axis, which, 'tick1')
        self.tick2line = _TickLine(axis, which, 'tick2')
        self.gridline = _TickLine(axis, which, 'grid')

    def set_pad(self, p):
        self.axis.set_tick_params(self.which, pad=p)

    def get_loc(self):
        return self.label1.get_position()[0 if self.axis.axis_name == 'x' else 1]

    def set_visible(self, v):
        self.tick1line.set_visible(v)
        self.label1.set_visible(v)


class _LabelText:
    """An axis label as matplotlib's Text (ax.xaxis.label)."""

    def __init__(self, axis):
        self.axis = axis

    def get_text(self):
        return self.axis.label_text

    def set_text(self, s):
        self.axis.label_text = s

    def _set(self, k, v):
        self.axis.label_kw[k] = v

    def set_color(self, c):
        self._set('color', c)

    def get_color(self):
        return self.axis.label_kw.get('color', rc('axes.labelcolor'))

    def set_fontsize(self, s):
        self._set('fontsize', s)

    set_size = set_fontsize

    def set_fontweight(self, w):
        self._set('fontweight', w)

    set_weight = set_fontweight

    def set_rotation(self, r):
        self._set('rotation', r)

    def set_ha(self, ha):
        self._set('ha', ha)

    set_horizontalalignment = set_ha

    def set_va(self, va):
        self._set('va', va)

    set_verticalalignment = set_va

    def set_visible(self, v):
        if not v:
            self.axis.label_text = ''

    def set_position(self, xy):
        pass

    def set_fontstyle(self, s):
        self._set('fontstyle', s)

    def set_family(self, f):
        self._set('family', f)

    set_fontfamily = set_family

    def set_labelpad(self, p):
        self.axis.labelpad = p

    def set(self, **kw):
        for k, v in kw.items():
            getattr(self, 'set_' + {'size': 'fontsize', 'c': 'color',
                                    'horizontalalignment': 'ha'}.get(k, k))(v)


def _takes_minpos(fn):
    """Whether a locator's nonsingular has a minpos parameter."""
    try:
        return 'minpos' in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


@njit(cache=True, nogil=True)
def _linear_px(x, y, x0, ux, sx, w, y1, uy, sy, h, out):
    """_data_px for linear scales into out (n, 2): numpy's arithmetic,
    one operation after another, in one pass."""
    for i in range(x.size):
        out[i, 0] = x0 + ((x[i] - ux) * sx) * w
        out[i, 1] = y1 - ((y[i] - uy) * sy) * h


class Axes(StatsMixin, PlotMixin, AxesBase):
    """matplotlib's Axes: plain x/y axes."""

    name = 'rectilinear'

    def __init__(self, fig, spec, label='', sharex=None, sharey=None):
        super().__init__(fig, spec, label)
        self.xaxis = Axis(self, 'x')
        self.yaxis = Axis(self, 'y')
        self.spines = Spines((s, Spine(self, s)) for s in ('left', 'right', 'bottom', 'top'))
        self.transData = _transforms.TransData(self)
        self._aspect = 'auto'
        self._adjustable = 'box'
        self._skew = 0.0                                # x += skew * height, in px (SkewT)
        self._lims = {}
        self._twin_of = None
        self._tight = False
        self.zorder = 0
        if sharex is not None:
            self.sharex(sharex)
        if sharey is not None:
            self.sharey(sharey)

    # -- sharing and twins
    def _share(self, other, name):
        mine = getattr(self, name + 'axis')
        theirs = getattr(other, name + 'axis')
        group = theirs.shared
        for a in mine.shared:
            if a not in group:
                group.append(a)
        for a in group:
            a.shared = group
        mine.scale = theirs.scale
        mine.major_locator, mine.minor_locator = theirs.major_locator, theirs.minor_locator
        mine.major_formatter, mine.minor_formatter = theirs.major_formatter, \
            theirs.minor_formatter
        mine.lim, mine.auto, mine.dates = theirs.lim, theirs.auto, theirs.dates
        mine.categories = theirs.categories

    def sharex(self, other):
        self._share(other, 'x')

    def sharey(self, other):
        self._share(other, 'y')

    def twinx(self):
        """A second y axis on the right, sharing x (matplotlib's twinx)."""
        tw = self.figure._add_axes_object(Axes(self.figure, self.spec, sharex=self))
        tw._twin_of = self
        tw.anchor = self.anchor
        tw.patch_visible = False
        tw.yaxis.tick_right()
        tw.yaxis.set_label_position('right')
        tw.xaxis.set_tick_params(which='both', bottom=False, labelbottom=False)
        return tw

    def twiny(self):
        """A second x axis on top, sharing y (matplotlib's twiny)."""
        tw = self.figure._add_axes_object(Axes(self.figure, self.spec, sharey=self))
        tw._twin_of = self
        tw.anchor = self.anchor
        tw.patch_visible = False
        tw.xaxis.tick_top()
        tw.xaxis.set_label_position('top')
        tw.yaxis.set_tick_params(which='both', left=False, labelleft=False)
        return tw

    # -- axes in and beside this one (fastplot._secondary)
    def inset_axes(self, bounds, *, transform=None, zorder=5, **kw):
        """matplotlib's inset_axes: an axes at bounds (x, y, w, h) in this
        one's axes coordinates (or data, transform=ax.transData)."""
        from ._secondary import inset_axes
        return inset_axes(self, bounds, transform=transform, zorder=zorder, **kw)

    def indicate_inset(self, bounds=None, inset_ax=None, *, transform=None, facecolor='none',
                       edgecolor='0.5', alpha=0.5, zorder=None, **kw):
        """matplotlib's indicate_inset: the rectangle bounds (x, y, w, h,
        data) and connectors to inset_ax."""
        from ._secondary import InsetIndicator
        lw = kw.pop('linewidth', kw.pop('lw', None))
        ls = kw.pop('linestyle', kw.pop('ls', '-'))
        edgecolor = kw.pop('ec', kw.pop('color', edgecolor))
        kw.pop('label', None)
        if kw:
            raise TypeError('indicate_inset: unsupported keyword(s): %s' % ', '.join(sorted(kw)))
        ind = InsetIndicator(self, None if bounds is None else tuple(bounds), inset_ax,
                             facecolor, edgecolor, alpha, lw, ls)
        if zorder is not None:
            ind.zorder = zorder
        return self._add(ind)

    def indicate_inset_zoom(self, inset_ax, **kw):
        """matplotlib's indicate_inset_zoom: the inset's view limits as the box."""
        return self.indicate_inset(None, inset_ax, **kw)

    def secondary_xaxis(self, location, functions=None, *, transform=None, **kw):
        """matplotlib's secondary_xaxis: a second x axis ('top', 'bottom' or
        an axes fraction), its values functions[0] of this one's."""
        from ._secondary import SecondaryAxis
        sec = SecondaryAxis(self, 'x', location, functions, transform)
        for k, v in kw.items():
            getattr(sec, 'set_' + k)(v)
        return self.figure._add_axes_object(sec) and self._keep_current(sec)

    def secondary_yaxis(self, location, functions=None, *, transform=None, **kw):
        """matplotlib's secondary_yaxis: a second y axis ('left', 'right' or
        an axes fraction)."""
        from ._secondary import SecondaryAxis
        sec = SecondaryAxis(self, 'y', location, functions, transform)
        for k, v in kw.items():
            getattr(sec, 'set_' + k)(v)
        return self.figure._add_axes_object(sec) and self._keep_current(sec)

    def _keep_current(self, child):
        """A child axes added: this one stays the figure's current axes."""
        self.figure._current_axes = self
        return child

    def label_outer(self):
        """Only the outer subplots keep their tick labels and axis labels."""
        ss = self.spec if isinstance(self.spec, SubplotSpec) else None
        if ss is None:
            return
        if ss.r1 != ss.gs.nrows - 1:
            self.xaxis.set_tick_params(which='both', labelbottom=False)
            self.xaxis.label_text = ''
        if ss.c0 != 0:
            self.yaxis.set_tick_params(which='both', labelleft=False)
            self.yaxis.label_text = ''

    # -- limits and scales
    def _set_lim(self, name, lim, auto=False, emit=True):
        axis = getattr(self, name + 'axis')
        for a in axis.shared:
            a.lim = None if lim is None else (float(lim[0]), float(lim[1]))
            if lim is not None and lim[0] != lim[1]:
                a.inverted = lim[0] > lim[1]
            if auto is not None:
                a.auto = auto

    def _get_lim(self, name):
        if self._placed is not None:
            return self._lims[name]
        av = getattr(self, '_aspect_view', None)       # adjustable='datalim': the limits the
        if av is not None:                              # last draw widened, while the data
            base = {n: self._base_lim(n) for n in ('x', 'y')}   # and limits are unchanged
            if base == av[0]:
                return av[1][name]
        return self._base_lim(name)

    def _base_lim(self, name):
        axis = getattr(self, name + 'axis')
        if axis.lim is not None and not axis.auto:
            return axis.lim
        if axis.settled is not None and axis.settled[0] == len(self.artists):
            return axis.settled[1]
        lo, hi = self._autoscaled(name)
        return (hi, lo) if axis.inverted else (lo, hi)

    def set_xlim(self, left=None, right=None, emit=True, auto=False, *, xmin=None, xmax=None):
        return self._set_limits('x', left, right, xmin, xmax, auto)

    def set_ylim(self, bottom=None, top=None, emit=True, auto=False, *, ymin=None, ymax=None):
        return self._set_limits('y', bottom, top, ymin, ymax, auto)

    def _set_limits(self, name, a, b, amin, amax, auto):
        if b is None and np.iterable(a) and not isinstance(a, str):
            a, b = a
        a = amin if a is None else a
        b = amax if b is None else b
        axis = getattr(self, name + 'axis')
        a, b = (self._to_num(axis, a), self._to_num(axis, b))
        for v in (a, b):
            if v is not None and not np.isfinite(float(v)):
                raise ValueError('Axis limits cannot be NaN or Inf')
        old = self._get_lim(name)
        lim = (old[0] if a is None else a, old[1] if b is None else b)
        if axis.get_scale() == 'log':                   # matplotlib: a bound <= 0 is ignored
            lim = (old[0] if lim[0] <= 0 else lim[0], old[1] if lim[1] <= 0 else lim[1])
        if lim[0] == lim[1]:                            # matplotlib widens identical limits
            lo, hi = self._nonsingular(axis, float(lim[0]), float(lim[1]))
            lim = (lo, hi)
        self._set_lim(name, lim, auto=auto)
        return lim

    def get_xlim(self):
        self._looked = True                             # (working them out settles them)
        return self._get_lim('x')

    def get_ylim(self):
        self._looked = True
        return self._get_lim('y')

    def set_xbound(self, lower=None, upper=None):
        lo, hi = sorted(self.get_xlim())
        inv = self.xaxis.get_inverted()
        lim = (lo if lower is None else lower, hi if upper is None else upper)
        self.set_xlim(*(lim[::-1] if inv else lim))

    def set_ybound(self, lower=None, upper=None):
        lo, hi = sorted(self.get_ylim())
        inv = self.yaxis.get_inverted()
        lim = (lo if lower is None else lower, hi if upper is None else upper)
        self.set_ylim(*(lim[::-1] if inv else lim))

    def invert_xaxis(self):
        self.xaxis.set_inverted(not self.xaxis.get_inverted())

    def invert_yaxis(self):
        self.yaxis.set_inverted(not self.yaxis.get_inverted())

    def xaxis_inverted(self):
        return self.xaxis.get_inverted()

    def yaxis_inverted(self):
        return self.yaxis.get_inverted()

    def autoscale(self, enable=True, axis='both', tight=None):
        if tight is not None:
            self._tight = bool(tight)
        for name in ('x', 'y'):
            if axis in (name, 'both'):
                if tight:                               # matplotlib: tight zeroes the margins
                    getattr(self, name + 'axis').margin = 0.0
                if enable is None:
                    continue
                if not enable:                          # off: the limits stay as they are now,
                    a = getattr(self, name + 'axis')    # the default if none were worked out yet
                    fresh = a.auto and a.lim is None and a.get_scale() == 'linear' and \
                        not getattr(self, '_looked', False)
                    self._set_lim(name, (0.0, 1.0) if fresh else self._get_lim(name), auto=False)
                else:
                    for a in getattr(self, name + 'axis').shared:
                        a.auto = True
                        a.settled = None

    def autoscale_view(self, tight=None, scalex=True, scaley=True):
        if tight is not None:
            self._tight = bool(tight)
        for name, on in (('x', scalex), ('y', scaley)):  # (data limits are found each time)
            axis = getattr(self, name + 'axis')
            if on and axis.auto:
                axis.lim = axis.settled = None
        self._autoscale_requested = True                # patches count from now on

    def set_autoscale_on(self, b):
        self.autoscale(b)

    def set_autoscalex_on(self, b):
        self.autoscale(b, 'x')

    def set_autoscaley_on(self, b):
        self.autoscale(b, 'y')

    def relim(self, visible_only=False):
        self._stale_bounds = []                         # data limits are found at each draw

    def margins(self, *margins, x=None, y=None, tight=True):
        if len(margins) == 1:
            x = y = margins[0]
        elif len(margins) == 2:
            x, y = margins
        if x is None and y is None:
            return self.xaxis.margin, self.yaxis.margin
        if x is not None:
            self.xaxis.margin = float(x)
        if y is not None:
            self.yaxis.margin = float(y)

    def set_xmargin(self, m):
        self.xaxis.margin = float(m)

    def set_ymargin(self, m):
        self.yaxis.margin = float(m)

    def set_xscale(self, value, **kw):
        self._set_scale('x', self.yaxis, value, kw)

    def set_yscale(self, value, **kw):
        self._set_scale('y', self.xaxis, value, kw)

    def _set_scale(self, name, other, value, kw):
        """Axis._set_axes_scale: a scale whose locator has other default
        limits autoscales at once, and that settles what is pending on the
        other axis at the scale it has now. A later symlog, say, which has
        a linear axis's default limits, does not autoscale again."""
        axis = getattr(self, name + 'axis')
        before = _DEFAULT_LIMS.get(axis.get_scale())
        for a in axis.shared:
            a.set_scale(value, **kw)
        if before != _DEFAULT_LIMS.get(axis.get_scale()):
            axis.settled = None
            if other.auto and (other.settled is None or other.settled[0] != len(self.artists)):
                other.settled = (len(self.artists), self._base_lim(other.axis_name))

    def get_xscale(self):
        return self.xaxis.get_scale()

    def get_yscale(self):
        return self.yaxis.get_scale()

    def set_aspect(self, aspect, adjustable=None, anchor=None, share=False):
        self._aspect = 'auto' if aspect in ('auto', None) else \
            (1.0 if aspect == 'equal' else float(aspect))
        if adjustable is not None:
            self._adjustable = adjustable
        if anchor is not None:
            self.anchor = anchor

    def get_aspect(self):
        return self._aspect

    def axis(self, arg=None, **kw):
        if arg in ('off', False):
            self.frame_on = False
            self.patch_visible = False
            self.xaxis.visible = self.yaxis.visible = False
        elif arg in ('on', True):
            self.frame_on = True
            self.xaxis.visible = self.yaxis.visible = True
        elif arg in ('equal', 'scaled', 'image'):
            self.autoscale(True)                        # matplotlib: autoscale on, then aspect
            self._autoscale_requested = True
            self._tight = False
            self.set_aspect('equal', adjustable='datalim' if arg == 'equal' else 'box')
            if arg == 'image':
                self._tight = True
        elif arg == 'tight':
            self._tight = True
        elif arg == 'square':                           # equal spans, a square box
            self.set_aspect('equal', adjustable='box', anchor='C')
            xlim, ylim = self.get_xlim(), self.get_ylim()
            edge = max(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]))
            self.set_xlim(xlim[0], xlim[0] + edge * np.sign(xlim[1] - xlim[0] or 1))
            self.set_ylim(ylim[0], ylim[0] + edge * np.sign(ylim[1] - ylim[0] or 1))
        elif arg in ('auto', 'normal'):
            self.set_aspect('auto')
            self.autoscale(True)
            self._tight = False
        elif arg is not None and not isinstance(arg, str) and np.iterable(arg):
            x0, x1, y0, y1 = arg
            self.set_xlim(x0, x1)
            self.set_ylim(y0, y1)
        elif isinstance(arg, str):
            raise ValueError('Unrecognized string %r to axis; try \'on\' or \'off\'' % arg)
        if kw:
            xmin, xmax = kw.pop('xmin', None), kw.pop('xmax', None)
            ymin, ymax = kw.pop('ymin', None), kw.pop('ymax', None)
            kw.pop('emit', None)
            if kw:
                raise TypeError('axis: unsupported keyword(s): %s' % ', '.join(sorted(kw)))
            if xmin is not None or xmax is not None:
                self.set_xlim(xmin, xmax)
            if ymin is not None or ymax is not None:
                self.set_ylim(ymin, ymax)
        return (*self.get_xlim(), *self.get_ylim())

    def set_axis_on(self):
        self.axis('on')

    # -- ticks, labels, grid (matplotlib's Axes methods)
    def set_xticks(self, ticks, labels=None, *, minor=False, **kw):
        return self.xaxis.set_ticks(self._to_num(self.xaxis, ticks), labels, minor, **kw)

    def set_yticks(self, ticks, labels=None, *, minor=False, **kw):
        return self.yaxis.set_ticks(self._to_num(self.yaxis, ticks), labels, minor, **kw)

    def set_xticklabels(self, labels, *, minor=False, **kw):
        self.xaxis.set_ticklabels(labels, minor, **kw)

    def set_yticklabels(self, labels, *, minor=False, **kw):
        self.yaxis.set_ticklabels(labels, minor, **kw)

    def get_xticks(self, minor=False):
        return self.xaxis.get_ticklocs(minor=minor)

    def get_yticks(self, minor=False):
        return self.yaxis.get_ticklocs(minor=minor)

    def get_xticklabels(self, minor=False, which=None):
        return self.xaxis.get_ticklabels(minor, which)

    def get_yticklabels(self, minor=False, which=None):
        return self.yaxis.get_ticklabels(minor, which)

    def get_xmajorticklabels(self):
        return self.xaxis.get_ticklabels()

    def get_ymajorticklabels(self):
        return self.yaxis.get_ticklabels()

    def _ticklabel_texts(self, axis, which):
        """matplotlib's tick label Texts: each its text; a change to one
        (rotation, size, colour, alignment) applies to the axis' labels."""
        locs = axis.get_ticklocs(minor=which == 'minor')
        vmin, vmax = sorted(axis.get_view_interval())
        fmt = axis.minor_formatter if which == 'minor' else axis.major_formatter
        if len(locs):
            labels = format_ticks(fmt, np.asarray(locs, float), vmin, vmax)[0]
        else:
            labels = []
        return [_TickLabel(axis, s, v) for s, v in zip(labels, locs)]

    def get_xbound(self):
        return tuple(sorted(self.get_xlim()))

    def get_ybound(self):
        return tuple(sorted(self.get_ylim()))

    def get_xaxis(self):
        return self.xaxis

    def get_yaxis(self):
        return self.yaxis

    def get_lines(self):
        from ._artists import Line2D
        return [a for a in self.artists if isinstance(a, Line2D)]

    def get_xmargin(self):
        return self.xaxis.margin

    def get_ymargin(self):
        return self.yaxis.margin

    def get_autoscale_on(self):
        return self.xaxis.auto and self.yaxis.auto

    def get_autoscalex_on(self):
        return self.xaxis.auto

    def get_autoscaley_on(self):
        return self.yaxis.auto

    def tick_params(self, axis='both', **kw):
        which = kw.pop('which', 'major')
        for name in ('x', 'y'):
            if axis in (name, 'both'):
                kk = {k: v for k, v in kw.items()
                      if not (name == 'x' and k in ('left', 'right', 'labelleft', 'labelright'))
                      and not (name == 'y' and k in ('bottom', 'top', 'labelbottom', 'labeltop'))}
                getattr(self, name + 'axis').set_tick_params(which, **kk)

    def minorticks_on(self):
        for a in (self.xaxis, self.yaxis):
            if a.get_scale() == 'linear':
                a.minor_locator = _ticker.AutoMinorLocator()
            a._minor_visible = True

    def minorticks_off(self):
        for a in (self.xaxis, self.yaxis):
            a.minor_locator = _ticker.NullLocator()
            a._minor_visible = False

    def locator_params(self, axis='both', tight=None, **kw):
        for name in ('x', 'y'):
            if axis in (name, 'both'):
                loc = getattr(self, name + 'axis').major_locator
                if hasattr(loc, 'set_params'):
                    loc.set_params(**kw)
                elif 'nbins' in kw and hasattr(loc, 'nbins'):
                    loc.nbins = kw['nbins']

    def ticklabel_format(self, *, axis='both', style=None, scilimits=None, useOffset=None,
                         useMathText=None, **kw):
        for name in ('x', 'y'):
            if axis in (name, 'both'):
                f = getattr(self, name + 'axis').major_formatter
                if useOffset is not None and hasattr(f, 'use_offset'):
                    f.use_offset = bool(useOffset)
                if scilimits is not None and hasattr(f, 'powerlimits'):
                    f.powerlimits = tuple(scilimits)
                if style == 'plain' and hasattr(f, 'powerlimits'):
                    f.powerlimits = (-np.inf, np.inf)
                if useMathText is not None and hasattr(f, 'use_math_text'):
                    f.use_math_text = bool(useMathText)

    def set_xlabel(self, xlabel, fontdict=None, labelpad=None, *, loc=None, **kw):
        self._set_label(self.xaxis, xlabel, fontdict, labelpad, loc, kw)
        return self.xaxis.label

    def set_ylabel(self, ylabel, fontdict=None, labelpad=None, *, loc=None, **kw):
        self._set_label(self.yaxis, ylabel, fontdict, labelpad, loc, kw)
        return self.yaxis.label

    def _set_label(self, axis, text, fontdict, labelpad, loc, kw):
        kw = dict(fontdict or {}, **kw)
        axis.label_text = text
        axis.label_kw = kw
        if labelpad is not None:
            axis.labelpad = labelpad
        if loc is not None:
            axis.label_kw['loc'] = loc

    def get_xlabel(self):
        return self.xaxis.label_text

    def get_ylabel(self):
        return self.yaxis.label_text

    def grid(self, visible=None, which='major', axis='both', **kw):
        for name in ('x', 'y'):
            if axis in (name, 'both'):
                getattr(self, name + 'axis').grid(visible, which, **kw)

    def set_axisbelow(self, b):
        self._axis_zorder = 0.5 if b is True else (1.5 if b == 'line' else 2.5)

    def xaxis_date(self, tz=None):
        """The x axis as dates: matplotlib's day numbers, date ticks and labels."""
        self.xaxis.axis_date(tz)

    def yaxis_date(self, tz=None):
        """The y axis as dates: matplotlib's day numbers, date ticks and labels."""
        self.yaxis.axis_date(tz)

    # -- coordinates: AxesBase's hooks
    def _to_num(self, axis, v):
        """Values on an axis: dates (datetime, datetime64, pandas) as
        matplotlib's day numbers, switching the axis to dates."""
        if v is None or isinstance(v, (int, float)):
            return v
        if isinstance(v, str) or _is_categorical(v):
            return axis.category_values([v] if isinstance(v, str) else v)
        from . import _dates
        if _dates.is_date_like(v):
            axis.axis_date()
            return _dates.to_num(v)
        return v

    def _xy(self, x, y):
        return self._to_num(self.xaxis, x), self._to_num(self.yaxis, y)

    def _src(self, transform):
        if transform is None or transform is self.transData or transform == 'data':
            return 'data'
        if transform is self.transAxes or transform == 'axes':
            return 'axes'
        if isinstance(transform, _transforms.Transform):
            return transform
        raise TypeError('transform must be ax.transData, ax.transAxes, fig.transFigure or a '
                        'fastplot.transforms transform on plain axes')

    def _data_px(self, x, y):
        """Data -> canvas pixels: scale, limits, skew, box."""
        p = self._xf
        u = (self.xaxis.scale.transform(np.asarray(x, np.float64)) - p[0]) * p[1]
        v = (self.yaxis.scale.transform(np.asarray(y, np.float64)) - p[2]) * p[3]
        x0, y0, x1, y1 = self.box_f
        if self._skew:                                  # MetPy's SkewTTransform, in pixels
            return x0 + u * (x1 - x0) + self._skew * v * (y1 - y0), y1 - v * (y1 - y0)
        return x0 + u * (x1 - x0), y1 - v * (y1 - y0)

    def _shape_px(self, x, y, src):
        """A shape's vertices -> canvas pixels, one copy (see MapAxes)."""
        return [self._points_px(x, y, src)]

    def _points_px(self, x, y, src):
        if isinstance(src, str):
            if src == 'data':
                return self._data_px(x, y)
            return self.transAxes.to_px(x, y)
        return src.to_px(x, y)

    def _lines_px(self, x, y, offsets, src, parts=False):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        if (src == 'data' and not self._skew and x.ndim == 1 and x.shape == y.shape
                and type(self)._data_px is Axes._data_px
                and type(self.xaxis.scale) is _scale.LinearScale
                and type(self.yaxis.scale) is _scale.LinearScale):
            p = self._xf                                # (the common case, fused)
            x0, y0, x1, y1 = self.box_f
            xy = np.empty((x.size, 2))
            _linear_px(x, y, float(x0), float(p[0]), float(p[1]), float(x1 - x0), float(y1),
                       float(p[2]), float(p[3]), float(y1 - y0), xy)
        else:
            px, py = self._points_px(x, y, src)
            xy = np.column_stack([px, py])
        out = (xy, np.asarray(offsets, np.int64))
        return out + (np.arange(len(offsets) - 1),) if parts else out

    def _vectors_px(self, x, y, u, v, src):
        """matplotlib's barbs on plain axes: (u, v) is the direction on the
        screen as it is (the data's scales do not turn it)."""
        px, py = self._points_px(np.asarray(x, np.float64), np.asarray(y, np.float64), src)
        return px, py, np.asarray(u, float), -np.asarray(v, float), np.hypot(u, v)

    def _default_extent(self, src, shape):
        ny, nx = shape[:2]
        return (-0.5, nx - 0.5, ny - 0.5, -0.5)

    def _bounds(self, artist):
        return artist.data_bounds()

    def _draw_field(self, r, grid, z, mode, coloring, alpha):
        """A field on plain axes: per column and row (separable) when the
        grid has 1-D axes and the axes no skew; by cells otherwise."""
        bx0, by0, bx1, by1 = self.box
        W, H = bx1 - bx0, by1 - by0
        if W <= 0 or H <= 0:
            return
        if grid.rect and self._separable():
            cols, rows = self._sep_tables(grid, mode, bx0, by0, W, H)
            r.c.field_sep(self.box, cols, rows, z, grid, mode, coloring, alpha)
            return
        if grid.rect:
            xc, yc = grid.xaxis.coords, grid.yaxis.coords
            if mode == _raster.IMAGE:                   # by cells: between the nodes
                mode = _raster.BILINEAR
            if mode == _raster.NEAREST:                 # cells by their corners
                if not grid.edges:                      # matplotlib's 'nearest': mid-points
                    xc, yc = _mid_edges(xc), _mid_edges(yc)
                mode = _raster.FLAT
            xs, ys = np.meshgrid(xc, yc)
            if grid.yaxis.flip:
                xs, ys = xs[::-1], ys[::-1]
            if grid.xaxis.flip:
                xs, ys = xs[:, ::-1], ys[:, ::-1]
        else:
            xs, ys = grid.x2d, grid.y2d
        px, py = self._data_px(xs, ys)
        r.c.field_cells_px(self.box, np.ascontiguousarray(px - bx0 - 0.5),
                           np.ascontiguousarray(py - by0 - 0.5), z, mode, coloring, alpha)

    def _sep_tables(self, grid, mode, bx0, by0, W, H):
        """Per column and row of the raster: the grid nodes and weights
        (_raster._axis), from each pixel centre's data coordinate."""
        p = self._xf
        x0, y0, x1, y1 = self.box_f
        u = (np.arange(W) + bx0 + 0.5 - x0) / (x1 - x0)
        v = (y1 - (np.arange(H) + by0 + 0.5)) / (y1 - y0)
        xd = self.xaxis.scale.inverse(u / p[1] + p[0])
        yd = self.yaxis.scale.inverse(v / p[3] + p[2])
        return (_raster.axis_table(xd, grid.xaxis, mode), _raster.axis_table(yd, grid.yaxis, mode))

    # -- autoscaling (matplotlib's autoscale_view)
    def _data_limits(self, name):
        """Union of the shared axes' artists' data limits along name:
        (lo, hi, minpos, stickies)."""
        lo, hi, minpos = np.inf, -np.inf, np.inf
        stickies = []
        axis = getattr(self, name + 'axis')
        arts = [art for ax in {id(a.axes): a.axes for a in axis.shared}.values()
                for art in ax.artists if art.visible]
        # matplotlib autoscales once a plotting call asks, and for its axis:
        # add_patch alone does not, nor an axhspan for x
        if not any(getattr(art, 'requests_autoscale', True) and self._asks(art, name)
                   for art in arts) and not getattr(self, '_autoscale_requested', False):
            return lo, hi, minpos, stickies
        stale = [b for ax in {id(a.axes): a.axes for a in axis.shared}.values()
                 for b in getattr(ax, '_stale_bounds', [])]
        for b in [self._art_bounds(art) for art in arts] + stale:
            if b is None:
                continue
            blo, bhi, bmp, bst = b[0 if name == 'x' else 1]
            if np.isfinite(blo) and np.isfinite(bhi):
                lo, hi = min(lo, blo), max(hi, bhi)
            if np.isfinite(bmp):
                minpos = min(minpos, bmp)
            stickies.extend(bst)
        return lo, hi, minpos, stickies

    _seen_bounds = None

    def _art_bounds(self, art):
        """art.data_bounds(), once a placing (a line of 1M points: ~3 ms each)."""
        seen = self._seen_bounds
        if seen is None:
            return art.data_bounds()
        b = seen.get(id(art), seen)
        if b is seen:
            b = seen[id(art)] = art.data_bounds()
        return b

    def _asks(self, a, name):
        """Does an artist autoscale this axis?"""
        b = getattr(a, '_bounds', None)
        if b is not None and hasattr(b, 'x'):
            return (b.x if name == 'x' else b.y) is not None
        return self._art_bounds(a) is not None

    def _autoscaled(self, name):
        axis = getattr(self, name + 'axis')
        lo, hi, minpos, stickies = self._data_limits(name)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            plotted = any(getattr(a, 'requests_autoscale', True) and a.visible and
                          self._asks(a, name)
                          for ax in {id(s.axes): s.axes for s in axis.shared}.values()
                          for a in ax.artists)
            if not plotted or axis.get_scale() != 'linear':
                lo, hi = (1.0, 10.0) if axis.get_scale() == 'log' else (0.0, 1.0)
                return self._nonsingular(axis, lo, hi)
            lo, hi = -0.05, 0.05                        # matplotlib: an empty plot's limits
        x0, x1 = self._nonsingular(axis, lo, hi, minpos)
        if axis.get_scale() == 'log':
            stickies = [s for s in stickies if s > 0]
        st = np.sort(np.asarray(stickies, float))
        tol = 1e-5 * max(abs(x0), abs(x1), abs(x1 - x0))
        i0 = np.searchsorted(st, x0 + tol) - 1
        x0bound = st[i0] if i0 != -1 else None
        i1 = np.searchsorted(st, x1 - tol)
        x1bound = st[i1] if i1 != len(st) else None
        if hasattr(axis.scale, 'limit_range_for_scale'):
            x0, x1 = axis.scale.limit_range_for_scale(x0, x1, minpos)
        t0, t1 = axis.scale.transform(np.array([x0, x1], float))
        delta = (t1 - t0) * axis.margin
        if not np.isfinite(delta):
            delta = 0.0
        x0, x1 = axis.scale.inverse(np.array([t0 - delta, t1 + delta]))
        if x0bound is not None:
            x0 = max(x0, x0bound)
        if x1bound is not None:
            x1 = min(x1, x1bound)
        loc = axis.major_locator
        if hasattr(loc, 'view_limits') and not self._tight:     # tight: margins only
            x0, x1 = loc.view_limits(x0, x1)
        span = getattr(axis, '_tick_span', None)        # set_ticks beyond the data
        if span is not None:
            x0, x1 = min(x0, span[0]), max(x1, span[1])
        return float(x0), float(x1)

    def _nonsingular(self, axis, lo, hi, minpos=np.inf):
        """The locator's nonsingular(lo, hi), told the data's smallest
        positive value where it takes one (log, logit: matplotlib's read it
        from the axis -- a bar from 0 on a log axis starts at it, not at
        1e-300)."""
        loc = axis.major_locator
        if hasattr(loc, 'nonsingular'):
            if _takes_minpos(loc.nonsingular):
                return loc.nonsingular(lo, hi, minpos=minpos)
            return loc.nonsingular(lo, hi)
        return _ticker._nonsingular(lo, hi, expander=0.05)

    # -- layout
    def _place(self, fig, W, H):
        self._placed = None
        lims = {}
        self._seen_bounds = {}                          # each artist's bounds found once
        for name in ('x', 'y'):
            lims[name] = self._base_lim(name)
        self._seen_bounds = None
        base = dict(lims)
        if self._twin_of is not None and self._twin_of._placed is not None:
            box = self._twin_of._pos
        else:
            box = spec_box(self, fig)
            self._orig_pos = box                        # before the aspect (constrained layout)
            if self._aspect != 'auto':
                tx = self.xaxis.scale.transform(np.array(lims['x'], float))
                ty = self.yaxis.scale.transform(np.array(lims['y'], float))
                dr = self._data_ratio(tx, ty)
                if dr is not None:
                    ratio = self._aspect * dr
                    if self._adjustable == 'box':
                        box = _fit(box, ratio, self.anchor, W, H)
                    else:
                        lims = self._datalim_aspect(lims, box, W, H)
                        self._aspect_view = (base, dict(lims))
            if self.box_aspect is not None:
                box = _fit(box, self.box_aspect, self.anchor, W, H)
        self._pos = box
        ox, oy = getattr(fig, '_org', (0.0, 0.0))     # the figure's place on the canvas
        x0, x1 = box.l * W + ox, (box.l + box.w) * W + ox
        y0, y1 = (1 - box.b - box.h) * H + oy, (1 - box.b) * H + oy
        self.box_f = (x0, y0, x1, y1)
        self.box = tuple(int(np.floor(v + 0.5)) for v in (x0, y0, x1, y1))
        self.box_clip = self.box
        self._lims = lims
        tx = self.xaxis.scale.transform(np.array(lims['x'], float))
        ty = self.yaxis.scale.transform(np.array(lims['y'], float))
        self._xf = (tx[0], 1.0 / (tx[1] - tx[0]) if tx[1] != tx[0] else 0.0,
                    ty[0], 1.0 / (ty[1] - ty[0]) if ty[1] != ty[0] else 0.0)
        self.view_key = ('cartesian', lims['x'], lims['y'], self.xaxis.get_scale(),
                         self.yaxis.get_scale(), self._skew, self.box_f)
        self._placed = lims
        self._looked = True

    def _datalim_aspect(self, lims, box, W, H):
        """adjustable='datalim': matplotlib's apply_aspect -- one axis' view
        widened (or narrowed) about its centre to the aspect, which one
        decided by how the data's limits (+5%) fit each."""
        bw, bh = box.w * W, box.h * H
        if bw <= 0 or bh <= 0:
            return lims
        data_ratio = (bh / bw) / self._aspect
        tx = self.xaxis.scale.transform(np.array(lims['x'], float))
        ty = self.yaxis.scale.transform(np.array(lims['y'], float))
        xsize, ysize = abs(tx[1] - tx[0]), abs(ty[1] - ty[0])
        if not xsize or not ysize:
            return lims
        y_expander = data_ratio * xsize / ysize - 1
        if abs(y_expander) < 0.005:
            return lims
        dl = {}
        for n, t in (('x', tx), ('y', ty)):
            lo, hi = self._data_limits(n)[:2]
            if not (np.isfinite(lo) and np.isfinite(hi)):
                lo, hi = sorted(lims[n])
            dl[n] = getattr(self, n + 'axis').scale.transform(np.array([lo, hi], float))
        xr = 1.05 * (dl['x'][1] - dl['x'][0])
        yr = 1.05 * (dl['y'][1] - dl['y'][0])
        xmarg, ymarg = xsize - xr, ysize - yr
        Ysize, Xsize = data_ratio * xsize, ysize / data_ratio
        Xmarg, Ymarg = Xsize - xr, Ysize - yr
        shared_x, shared_y = len(self.xaxis.shared) > 1, len(self.yaxis.shared) > 1
        if shared_y:
            adjust_y = False
        else:
            if xmarg > 0 and ymarg > 0:
                adjy = (Ymarg > 0 and y_expander < 0) or (Xmarg < 0 and y_expander > 0)
            else:
                adjy = y_expander > 0
            adjust_y = shared_x or adjy
        if adjust_y:
            c, half, t, n = 0.5 * (ty[0] + ty[1]), Ysize / 2.0, ty, 'y'
        else:
            c, half, t, n = 0.5 * (tx[0] + tx[1]), Xsize / 2.0, tx, 'x'
        new = np.array([c - half, c + half]) if t[1] >= t[0] else np.array([c + half, c - half])
        lims[n] = tuple(getattr(self, n + 'axis').scale.inverse(new))
        return lims

    @property
    def extent(self):
        return (*self._get_lim('x'), *self._get_lim('y'))

    # -- drawing
    def _draw_background(self, r):
        if self.patch_visible:
            b = self.box
            r.c.fill_rect(b[0], b[1], b[2], b[3], rgba(self.facecolor))

    def _separable(self):
        """Can a field be drawn per column and row (no skew, no polar)?"""
        return not self._skew

    def _data_ratio(self, tx, ty):
        """matplotlib's get_data_ratio: the view's height over width (scaled)."""
        dx, dy = abs(tx[1] - tx[0]), abs(ty[1] - ty[0])
        return dy / dx if dx > 0 and dy > 0 else None

    def _draw(self, r):
        self._draw_background(r)
        self._top_extent = None
        self._ticks_cache = {}
        azo = getattr(self, '_axis_zorder', None)
        if azo is None:
            azo = _AXISBELOW.get(rc('axes.axisbelow'), 1.5)
        items = [(a.zorder, k, a) for k, a in enumerate(self.artists) if a.visible]
        items.append((azo, len(items), '_axis'))
        items.append((SPINE_ZORDER, len(items), '_spines'))
        if self.legend_ is not None:
            items.append((self.legend_.zorder, len(items), self.legend_))
        for _, _, a in sorted(items, key=lambda t: (t[0], t[1])):
            if a == '_axis':
                self._draw_axes_lines(r)
            elif a == '_spines':
                self._draw_spines(r)
            else:
                self._render_artist(r, a)
        exts = self._draw_tick_labels(r)
        r.extents.append(self.box_f)
        self._draw_axis_labels(r, exts)
        self._draw_titles(r)

    def _render_artist(self, r, a):
        a.render(r, self)

    def _ticks(self, r, axis, which):
        """The ticks of an axis to draw: (data values, labels, offset text)."""
        key = (axis.axis_name, which)
        if key in self._ticks_cache:
            return self._ticks_cache[key]
        lim = self._lims[axis.axis_name]
        vmin, vmax = min(lim), max(lim)
        if self._skew and axis.axis_name == 'x':        # the bottom's and the top's x ranges
            (a, b), (c, d) = self._edge_xlim(0), self._edge_xlim(1)
            vmin, vmax = min(a, b, c, d), max(a, b, c, d)
        t = axis.ticks[which]
        # shared axes share matplotlib's locator, whose tick space is its first
        # axis' (the one shared with)
        lead = axis.shared[0] if axis.shared and getattr(axis.shared[0].axes, 'box_f',
                                                         None) is not None else axis
        x0, y0, x1, y1 = lead.axes.box_f
        length = (x1 - x0) if axis.axis_name == 'x' else (y1 - y0)
        lt = lead.ticks[which]
        if lt.get('labelsize_set'):
            size = fontsize_pt(lt['labelsize'])
        else:                                           # matplotlib reads rcParams as they are now
            from ._rc import rcParams as live
            ls = live[axis.axis_name + 'tick.labelsize']
            size = float(live['font.size']) * _FONT_SCALE[ls] if isinstance(ls, str) \
                else float(ls)
        space = int(np.floor(length * 72.0 / r.dpi / (size * (3 if axis.axis_name == 'x' else 2))))
        if which == 'major':
            locs = tick_values(axis.major_locator, vmin, vmax, space)
            fmt = axis.major_formatter
        else:
            if not axis._minor_visible:
                self._ticks_cache[key] = (np.empty(0), [], '')
                return self._ticks_cache[key]
            major = self._ticks(r, axis, 'major')[0]
            locs = tick_values(axis.minor_locator, vmin, vmax, space, major=major)
            fmt = axis.minor_formatter
            if major.size and locs.size:                # matplotlib's remove_overlapping_locs
                tm = axis.transform(major)
                tl = axis.transform(locs)
                keep = ~np.isclose(tl[:, None], tm[None, :], rtol=0, atol=1e-5 *
                                   max(1.0, np.nanmax(np.abs(tm)))).any(axis=1)
                locs = locs[keep]
        # the ticks inside the view (matplotlib's _interval_contains_close)
        tv = axis.transform(locs)
        ta, tb = sorted(axis.transform(np.array([vmin, vmax])))
        eps = 1e-10 * max(abs(tb - ta), 1e-300)
        inside = (tv >= ta - eps) & (tv <= tb + eps)
        labels, offset = format_ticks(fmt, locs, vmin, vmax) if locs.size else ([], '')
        labels = [lab for lab, k in zip(labels, inside) if k]
        out = (locs[inside], labels, offset)
        self._ticks_cache[key] = out
        return out

    def _edge_xlim(self, v):
        """The data x range along the axes' bottom (v 0) or top (v 1) edge
        -- MetPy's lower_xlim / upper_xlim when skewed."""
        p = self._xf
        x0, y0, x1, y1 = self.box_f
        u = np.array([0.0, 1.0]) - self._skew * v * (y1 - y0) / (x1 - x0)
        x = self.xaxis.scale.inverse(p[0] + u / p[1]) if p[1] else np.array(self._lims['x'])
        return float(x[0]), float(x[1])

    def _tick_side(self, axis, locs, pos, side):
        """The ticks of an axis' side 1 (bottom, left) or 2: (positions px,
        which locs). A skewed x axis ticks each edge where its x range
        reaches (MetPy's SkewXTick lower/upper_in_bounds)."""
        if not (self._skew and axis.axis_name == 'x'):
            return pos, np.ones(locs.shape, bool)
        a, b = sorted(self._edge_xlim(0 if side == 1 else 1))
        eps = 1e-10 * max(abs(b - a), 1e-300)
        keep = (locs >= a - eps) & (locs <= b + eps)
        if side == 2:
            pos = self._data_px(locs, np.full(locs.shape, self._lims['y'][1]))[0]
        return pos[keep], keep

    def _tick_px(self, axis, locs):
        """Tick positions along an axis, canvas px (x for x ticks, y for y)."""
        if axis.axis_name == 'x':
            lo = self._lims['y'][0]
            px, _ = self._data_px(locs, np.full(locs.shape, lo))
            return px
        lo = self._lims['x'][0]
        return self._data_px(np.full(locs.shape, lo), locs)[1]

    def _draw_axes_lines(self, r):
        """Grid lines and tick marks (the axis, at its zorder)."""
        x0, y0, x1, y1 = self.box_f
        for axis in (self.xaxis, self.yaxis):
            if not axis.visible:
                continue
            for which in ('major', 'minor'):
                t = axis.ticks[which]
                locs = self._ticks(r, axis, which)[0]
                if not locs.size:
                    continue
                pos = self._tick_px(axis, locs)
                if t['grid']:
                    self._draw_grid(r, axis, locs, t['grid_kw'])
                lw = r.px(t['width'])
                L = r.px(t['size'])
                if lw <= 0 or L <= 0:
                    continue
                out_, in_ = {'out': (L, 0.0), 'in': (0.0, L), 'inout': (L / 2, L / 2)}[
                    t['direction']]
                # Agg's draw_markers: a tick placed at whole pixels, its path snapped
                sv = float(_draw.snap_value(lw))
                out_, in_, nout, nin = (np.floor(v + 0.5) + sv for v in (out_, in_, -out_, -in_))
                segs = []
                pos0 = pos
                for side, on in ((1, t['on1']), (2, t['on2'])):
                    if not on:
                        continue
                    pos = np.floor(self._tick_side(axis, locs, pos0, side)[0] + 0.5) + sv
                    if axis.axis_name == 'x':
                        e = np.floor(self.spines['bottom' if side == 1 else 'top'].offset_px(r)
                                     + 0.5)
                        a, b = (e + out_, e + nin) if side == 1 else (e + nout, e + in_)
                        segs.append(_across(pos, a, b, 0))
                    else:
                        e = np.floor(self.spines['left' if side == 1 else 'right'].offset_px(r)
                                     + 0.5)
                        a, b = (e + nout, e + in_) if side == 1 else (e + out_, e + nin)
                        segs.append(_across(pos, a, b, 1))
                if segs:
                    s = np.concatenate(segs).reshape(-1, 2)
                    r.lines(s, np.arange(0, s.shape[0] + 1, 2), rgba(t['color']), lw, None,
                            cap=_draw.BUTT)

    def _draw_grid(self, r, axis, locs, kw):
        x0, y0, x1, y1 = self.box_f
        color = kw.get('color', kw.get('c', rc('grid.color')))
        lw = kw.get('linewidth', kw.get('lw', rc('grid.linewidth')))
        ls = kw.get('linestyle', kw.get('ls', rc('grid.linestyle')))
        alpha = kw.get('alpha', rc('grid.alpha'))
        if self._skew and axis.axis_name == 'x':        # skewed: lines along the data's y
            ylo, yhi = self._lims['y']
            n = locs.size
            xs = np.repeat(locs, 2)
            ys = np.tile([ylo, yhi], n)
            px, py = self._data_px(xs, ys)
            xy = np.c_[px, py]
        else:
            pos = self._tick_px(axis, locs)
            if axis.axis_name == 'x':
                xy = _across(pos, y0, y1, 0).reshape(-1, 2)
            else:
                xy = _across(pos, x0, x1, 1).reshape(-1, 2)
        d = _dashes(ls, lw)
        r.lines(xy, np.arange(0, xy.shape[0] + 1, 2), rgba(color, alpha), r.px(lw),
                self.box_clip, None if d is None else tuple(r.px(v) for v in d), self._snap,
                _draw.PROJECTING if d is None else _draw.BUTT)

    def _draw_spines(self, r):
        if not self.frame_on:
            return
        x0, y0, x1, y1 = self.box_f
        for side, sp in self.spines.items():
            if not sp.visible or sp.linewidth <= 0:
                continue
            e = sp.offset_px(r)
            if side in ('left', 'right'):
                a, b = y1, y0
                if sp.bounds is not None:
                    a = self._data_px(np.array([self._lims['x'][0]]), np.array([sp.bounds[0]]))[1][0]
                    b = self._data_px(np.array([self._lims['x'][0]]), np.array([sp.bounds[1]]))[1][0]
                xy = np.array([[e, a], [e, b]])
            else:
                a, b = x0, x1
                if sp.bounds is not None:
                    a = self._data_px(np.array([sp.bounds[0]]), np.array([self._lims['y'][0]]))[0][0]
                    b = self._data_px(np.array([sp.bounds[1]]), np.array([self._lims['y'][0]]))[0][0]
                xy = np.array([[a, e], [b, e]])
            d = _dashes(sp.linestyle, sp.linewidth)
            r.lines(xy, np.array([0, 2]), rgba(sp.edgecolor, sp.alpha), r.px(sp.linewidth), None,
                    None if d is None else tuple(r.px(v) for v in d), self._snap,
                    _draw.PROJECTING, join=_draw.JOIN_MITER)    # a Spine's cap and join
            r.extents.append(self._spine_extent(r, side, xy))

    def _spine_extent(self, r, side, xy):
        """matplotlib's Spine.get_window_extent: the spine, and its axis'
        major ticks reaching out of it (tight bboxes and layouts count them)."""
        x0, y0 = xy.min(axis=0)
        x1, y1 = xy.max(axis=0)
        axis = self.yaxis if side in ('left', 'right') else self.xaxis
        t = axis.ticks['major']
        on = t['on1'] if side in ('left', 'bottom') else t['on2']
        if axis.visible and on and not isinstance(axis.major_locator, _ticker.NullLocator):
            size = r.px(t['size'])
            out = {'out': size, 'inout': size / 2, 'in': 0.0}.get(t['direction'], size)
            inn = {'out': 0.0, 'inout': size / 2, 'in': size}.get(t['direction'], 0.0)
            if side == 'left':
                x0, x1 = x0 - out, x1 + inn
            elif side == 'right':
                x0, x1 = x0 - inn, x1 + out
            elif side == 'bottom':
                y0, y1 = y0 - inn, y1 + out
            else:
                y0, y1 = y0 - out, y1 + inn
        return (x0, y0, x1, y1)

    def _draw_tick_labels(self, r):
        """Tick labels and offset texts; their extents per side."""
        exts = {'bottom': [], 'top': [], 'left': [], 'right': []}
        for axis in (self.xaxis, self.yaxis):
            if not axis.visible:
                continue
            for which in ('major', 'minor'):
                t = axis.ticks[which]
                locs, labels, offset = self._ticks(r, axis, which)
                size = fontsize_pt(axis.tick_label_kw.get('fontsize', axis.tick_label_kw.get(
                    'size', t['labelsize'])))
                color = axis.tick_label_kw.get('color', t['labelcolor'])
                rot = axis.tick_label_kw.get('rotation', t['rotation'])
                rot = 90.0 if rot == 'vertical' else (0.0 if rot in ('horizontal', None) else
                                                      float(rot))
                tkw = axis.tick_label_kw
                tweight = tkw.get('fontweight', tkw.get('weight', 'normal'))
                tstyle = tkw.get('fontstyle', tkw.get('style', 'normal'))
                tfamily = tkw.get('family', tkw.get('fontfamily'))
                hidden = getattr(axis, '_hidden_labels', None)
                if hidden and locs.size:                # a tick's label.set_visible(False)
                    hv = np.array(sorted(hidden))
                    labels = ['' if np.isclose(v, hv, rtol=0, atol=1e-9 * max(1.0, abs(v))).any()
                              else lab for v, lab in zip(locs, labels)]
                L = r.px(t['size'])
                gap = {'out': L, 'in': 0.0, 'inout': L / 2}[t['direction']] + r.px(t['pad'])
                pos = self._tick_px(axis, locs) if locs.size else locs
                pos0 = pos
                for side, on in ((1, t['label1']), (2, t['label2'])):
                    if not on or not locs.size:
                        continue
                    pos, keep = self._tick_side(axis, locs, pos0, side)
                    side_labels = [lab for lab, k in zip(labels, keep) if k]
                    if axis.axis_name == 'x':
                        name = 'bottom' if side == 1 else 'top'
                        e = self.spines[name].offset_px(r)
                        y = e + gap if side == 1 else e - gap
                        # matplotlib's tick labels: centred, rotation_mode 'default',
                        # whatever the rotation, unless set
                        ha = axis.tick_label_kw.get('ha', axis.tick_label_kw.get(
                            'horizontalalignment', 'center'))
                        va = 'top' if side == 1 else 'bottom'
                        va = axis.tick_label_kw.get('va', axis.tick_label_kw.get(
                            'verticalalignment', va))
                        mode = axis.tick_label_kw.get('rotation_mode', 'default')
                        for p, lab in zip(pos, side_labels):
                            ext = r.text(lab, p, y, size, color, ha=ha, va=va, rotation=rot,
                                         rotation_mode=mode, weight=tweight, style=tstyle,
                                         family=tfamily)
                            if ext is not None:
                                exts[name].append(ext)
                    else:
                        name = 'left' if side == 1 else 'right'
                        e = self.spines[name].offset_px(r)
                        x = e - gap if side == 1 else e + gap
                        ha = axis.tick_label_kw.get('ha', axis.tick_label_kw.get(
                            'horizontalalignment', 'right' if side == 1 else 'left'))
                        va = axis.tick_label_kw.get('va', axis.tick_label_kw.get(
                            'verticalalignment', 'center_baseline'))
                        for p, lab in zip(pos, side_labels):
                            ext = r.text(lab, x, p, size, color, ha=ha, va=va, rotation=rot,
                                         weight=tweight, style=tstyle, family=tfamily)
                            if ext is not None:
                                exts[name].append(ext)
                if which == 'major' and offset:
                    self._draw_offset(r, axis, offset, size, color, exts)
        return exts

    def _draw_offset(self, r, axis, text, size, color, exts):
        """The offset / order-of-magnitude text at the axis' end."""
        x0, y0, x1, y1 = self.box_f
        pad = r.px(OFFSET_PAD)
        if axis.axis_name == 'x':
            top = axis.ticks['major']['label2'] and not axis.ticks['major']['label1']
            if top:
                y = min([e[1] for e in exts['top']], default=y0) - pad
                ext = r.text(text, x1, y, size, color, ha='right', va='bottom')
            else:
                y = max([e[3] for e in exts['bottom']], default=y1) + pad
                ext = r.text(text, x1, y, size, color, ha='right', va='top')
            exts['top' if top else 'bottom'].append(ext)
        else:
            right = axis.ticks['major']['label2'] and not axis.ticks['major']['label1']
            ext = r.text(text, x1 if right else x0, y0 - pad, size, color,   # (matplotlib's:
                         ha='right' if right else 'left', va='baseline')    # on its baseline)
            exts['top'].append(ext)

    def _draw_axis_labels(self, r, exts):
        x0, y0, x1, y1 = self.box_f
        for axis in (self.xaxis, self.yaxis):
            if not axis.visible or not axis.label_text:
                continue
            kw = axis.label_kw
            size = fontsize_pt(kw.get('fontsize', kw.get('size', rc('axes.labelsize'))))
            color = kw.get('color', kw.get('c', rc('axes.labelcolor')))
            weight = kw.get('fontweight', kw.get('weight', rc('axes.labelweight')))
            pad = r.px(axis.labelpad)
            loc = kw.get('loc', 'center')
            style = kw.get('fontstyle', kw.get('style', 'normal'))
            family = kw.get('family', kw.get('fontfamily'))
            coords = getattr(axis, 'label_coords', None)
            if axis.axis_name == 'x':
                x, ha = {'left': (x0, 'left'), 'center': (0.5 * (x0 + x1), 'center'),
                         'right': (x1, 'right')}[loc]
                ha = kw.get('ha', kw.get('horizontalalignment', ha))
                rot = kw.get('rotation', 0.0)
                rot = 90.0 if rot == 'vertical' else (0.0 if rot in ('horizontal', None)
                                                      else float(rot))
                top = axis.label_position == 'top'
                va = kw.get('va', kw.get('verticalalignment', 'baseline' if top else 'top'))
                if coords is not None:                  # set_label_coords: axes fractions
                    x, y = x0 + coords[0] * (x1 - x0), y1 - coords[1] * (y1 - y0)
                elif top:
                    y = min([e[1] for e in exts['top']],
                            default=self.spines['top'].offset_px(r)) - pad
                else:
                    y = max([e[3] for e in exts['bottom']],
                            default=self.spines['bottom'].offset_px(r)) + pad
                ext = r.text(axis.label_text, x, y, size, color, ha=ha, va=va, rotation=rot,
                             weight=weight, style=style, family=family)
                exts['top' if top else 'bottom'].append(ext)
            else:
                y, ha = {'bottom': (y1, 'left'), 'center': (0.5 * (y0 + y1), 'center'),
                         'top': (y0, 'right')}[loc]
                ha = kw.get('ha', kw.get('horizontalalignment', ha))
                rot = kw.get('rotation', 90.0)
                rot = 90.0 if rot == 'vertical' else (0.0 if rot in ('horizontal', None)
                                                      else float(rot))
                right = axis.label_position == 'right'
                va = kw.get('va', kw.get('verticalalignment', 'top' if right else 'bottom'))
                if coords is not None:
                    x, y = x0 + coords[0] * (x1 - x0), y1 - coords[1] * (y1 - y0)
                elif right:
                    x = max([e[2] for e in exts['right']],
                            default=self.spines['right'].offset_px(r)) + pad
                else:
                    x = min([e[0] for e in exts['left']],
                            default=self.spines['left'].offset_px(r)) - pad
                r.text(axis.label_text, x, y, size, color, ha=ha, va=va, rotation=rot,
                       rotation_mode='anchor', weight=weight, style=style, family=family)
        tops = exts['top']
        self._top_extent = min(e[1] for e in tops if e is not None) if any(
            e is not None for e in tops) else None

    def _draw_titles(self, r):
        super()._draw_titles(r)

    def get_tightbbox(self, r=None):
        return self.box_f
