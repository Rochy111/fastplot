"""Axes that live inside or alongside another: matplotlib's inset_axes
(and indicate_inset_zoom's box and connectors) and secondary_xaxis /
secondary_yaxis (a second scale of the same data, by a pair of functions).

Both are plain Axes placed from their parent at each draw: an inset at
bounds in the parent's axes (or data) coordinates, a secondary axis on the
parent's box with the parent's limits mapped through its functions.
"""
import numpy as np

from ._cartesian import Axes
from ._layout import _Box, spec_box


class InsetSpec:
    """Where an inset sits: bounds (x, y, width, height) in its parent's
    axes coordinates, or in data coordinates (transform=ax.transData)."""

    def __init__(self, parent, bounds, data=False):
        self.parent, self.bounds, self.data = parent, tuple(float(v) for v in bounds), data

    def box(self, fig):
        p = self.parent
        x, y, w, h = self.bounds
        pos = getattr(p, '_pos', None) or spec_box(p, fig)
        if self.data and getattr(p, 'box_f', None) is not None:
            W, H = fig._px_size
            ox, oy = getattr(fig, '_org', (0.0, 0.0))
            xs, ys = p._data_px(np.array([x, x + w]), np.array([y, y + h]))
            return _Box((min(xs) - ox) / W, 1 - (max(ys) - oy) / H, abs(xs[1] - xs[0]) / W,
                        abs(ys[1] - ys[0]) / H)
        return _Box(pos.l + x * pos.w, pos.b + y * pos.h, w * pos.w, h * pos.h)


def inset_axes(parent, bounds, *, transform=None, zorder=5, **kw):
    """matplotlib's Axes.inset_axes."""
    fig = parent.figure
    data = transform is not None and transform is getattr(parent, 'transData', None)
    ax = Axes(fig, InsetSpec(parent, bounds, data))
    ax.zorder = zorder
    for k in ('xlim', 'ylim'):
        if k in kw:
            getattr(ax, 'set_' + k)(*kw.pop(k))
    for k in ('title', 'xlabel', 'ylabel', 'xscale', 'yscale', 'aspect', 'facecolor'):
        if k in kw:
            getattr(ax, 'set_' + k)(kw.pop(k))
    kw.pop('label', None)
    if kw.pop('frameon', True) is False:
        ax.set_frame_on(False)
    if kw:
        raise TypeError('inset_axes: unsupported keyword(s): %s' % ', '.join(sorted(kw)))
    fig._add_axes_object(ax)
    fig._current_axes = parent
    return ax


class InsetIndicator:
    """indicate_inset / indicate_inset_zoom: a rectangle on the parent and
    the connectors from its corners to the inset's that cross neither box."""

    zorder = 4.99

    def __init__(self, parent, bounds, inset, facecolor='none', edgecolor='0.5', alpha=0.5,
                 linewidth=None, linestyle='-'):
        from ._style import rc
        self.axes, self.parent, self.bounds, self.inset = parent, parent, bounds, inset
        self.facecolor, self.edgecolor, self.alpha = facecolor, edgecolor, alpha
        self.linewidth = rc('patch.linewidth') if linewidth is None else linewidth
        self.linestyle = linestyle
        self.visible = True
        self.label = None

    def data_bounds(self):
        return None

    def work(self):
        return 0

    def remove(self):
        if self in self.parent.artists:
            self.parent.artists.remove(self)

    def _rect(self):
        if self.bounds is not None:
            return self.bounds
        (x0, x1), (y0, y1) = self.inset.get_xlim(), self.inset.get_ylim()
        return (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))

    def render(self, r, ax):
        from . import _draw
        from ._style import _dashes, rgba
        x, y, w, h = self._rect()
        px, py = ax._data_px(np.array([x, x + w, x + w, x]), np.array([y, y, y + h, y + h]))
        ring = np.c_[px, py]
        fc, ec = rgba(self.facecolor, self.alpha), rgba(self.edgecolor, self.alpha)
        lw = r.px(self.linewidth)
        closed = _draw.snap(np.r_[ring, ring[:1]], np.array([0, 5]), lw, ax._snap)
        if fc[3]:
            r.c.polygons(closed[:4], np.array([0, 4]), fc, ax.box_clip, 'nonzero')
        d = _dashes(self.linestyle, self.linewidth)
        dash = None if d is None else tuple(r.px(v) for v in d)
        r.lines(closed, np.array([0, 5]), ec, lw, ax.box_clip, dash, ax._snap, _draw.BUTT)
        ins = getattr(self.inset, 'box_f', None)
        if self.inset is None or ins is None:
            return
        ix0, iy0, ix1, iy1 = ins                        # the inset's corners, as the rect's
        corners = [(ix0, iy1), (ix1, iy1), (ix1, iy0), (ix0, iy0)]
        rbox = (px.min(), py.min(), px.max(), py.max())
        segs = []
        for (ax_, ay_), (bx, by) in zip(ring, corners):
            if not (_crosses(ax_, ay_, bx, by, rbox) or _crosses(ax_, ay_, bx, by, ins)):
                segs += [[ax_, ay_], [bx, by]]
        if segs:
            s = np.asarray(segs, float)
            r.lines(s, np.arange(0, s.shape[0] + 1, 2), ec, lw, None, dash, ax._snap,
                    _draw.BUTT)


def _crosses(x0, y0, x1, y1, box):
    """Does the segment pass through the box's interior (not just touch)?"""
    bx0, by0, bx1, by1 = box
    t = np.linspace(0.02, 0.98, 49)
    xs, ys = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
    eps = 0.5
    return bool(((xs > bx0 + eps) & (xs < bx1 - eps) & (ys > by0 + eps) &
                 (ys < by1 - eps)).any())


class SecondaryAxis(Axes):
    """matplotlib's SecondaryAxis: the parent's box and limits, one axis
    shown -- its values the parent's through functions (forward, inverse),
    ticked in its own units."""

    def __init__(self, parent, orientation, location, functions, transform=None):
        super().__init__(parent.figure, parent.spec)
        self._parent = parent
        self._orientation = orientation
        if functions is None:
            functions = (lambda v: v, lambda v: v)
        elif not isinstance(functions, tuple) and hasattr(functions, 'to_px'):
            functions = (lambda v: v, lambda v: v)
        elif not (isinstance(functions, tuple) and len(functions) == 2):
            raise ValueError('functions argument of secondary Axes must be a two-tuple of '
                             'callable functions with the first function being the transform '
                             'and the second being the inverse')
        self.patch_visible = False
        self._twin_of = None
        self.zorder = parent.zorder
        self.set_functions(functions)
        self.set_location(location)
        other = self.yaxis if orientation == 'x' else self.xaxis
        other.set_visible(False)
        for s in self.spines.values():
            s.set_visible(False)

    def set_location(self, location):
        """'top' / 'bottom' (x), 'left' / 'right' (y), or a number: the
        parent's axes fraction its spine sits at."""
        x = self._orientation == 'x'
        sides = ('bottom', 'top') if x else ('left', 'right')
        if isinstance(location, str):
            if location not in sides:
                raise ValueError('location must be %r, %r, or a float, not %r'
                                 % (sides[1], sides[0], location))
            side, pos = location, None
        else:
            side, pos = sides[1] if float(location) > 0.5 else sides[0], float(location)
        self._side = side
        axis = self.xaxis if x else self.yaxis
        axis.set_ticks_position(side)
        axis.set_label_position(side)
        sp = self.spines[side]
        sp.set_visible(True)
        if pos is not None:
            sp.set_position(('axes', pos))

    def set_functions(self, functions):
        """The scale of this axis: the parent's (log: through its log, as
        matplotlib's 'functionlog') after the inverse function."""
        self._functions = functions
        f, inv = functions
        name = self._orientation
        pscale = getattr(self._parent, name + 'axis').scale
        axis = getattr(self, name + 'axis')
        axis.set_scale('function', functions=(
            lambda v: pscale.transform(inv(np.asarray(v, float))),
            lambda v: f(pscale.inverse(np.asarray(v, float)))))
        if getattr(pscale, 'name', '') == 'log':        # 'functionlog': log ticks and labels
            (axis.major_locator, axis.minor_locator, axis.major_formatter,
             axis.minor_formatter) = pscale.default_locators()
            axis._minor_visible = True
        self._pscale = pscale

    def _place(self, fig, W, H):
        p = self._parent
        name = self._orientation
        if getattr(p, name + 'axis').scale is not self._pscale:    # the parent's changed
            loc = getattr(self, name + 'axis')
            kept = loc.major_locator, loc.major_formatter, loc.minor_locator
            self.set_functions(self._functions)
            loc.major_locator, loc.major_formatter, loc.minor_locator = kept
        lim = np.array(p._lims[name] if p._placed is not None else p._get_lim(name), float)
        new = self._functions[0](lim)
        self._set_lim(name, (float(new[0]), float(new[1])), auto=False)
        other = 'y' if name == 'x' else 'x'
        self._set_lim(other, p._lims[other] if p._placed is not None else p._get_lim(other),
                      auto=False)
        self.spec = getattr(p, '_pos', None) or p.spec  # exactly the parent's box
        super()._place(fig, W, H)
