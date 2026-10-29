"""matplotlib's PolarAxes (projection='polar'), the full circle.

    ax = fig.add_subplot(projection='polar')
    ax.plot(theta, r); ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
    ax.pcolormesh(theta_edges, r_edges, reflectivity)        # a radar sweep

Data are (theta in radians, r). As matplotlib's transforms do it: theta
turns by the direction and offset, r is measured from the origin (the
lower r limit unless set_rorigin) and scaled so the upper limit lies on
the axes' circle, the axes square ('equal', anchored at the centre).
Lines join their points straight on the screen (matplotlib's Line2D does
not interpolate); grid circles and bars follow arcs (its gridlines and
bars do). Theta ticks every 45 degrees, labelled outside the circle (the
title above them, as matplotlib's automatic title position puts it);
radial ticks matplotlib's AutoLocator's above the origin, the axis taking
0 in and labelled along rlabel_position (22.5 degrees). Fields
(pcolormesh, contourf, imshow of values) are drawn by cells, their corners
through the polar mapping, as matplotlib's QuadMesh and contour polygons
are. Not here: partial wedges (set_thetamin / set_thetamax), clipping to
the circle (artists inside the limits need none).
"""
import numpy as np

from . import _draw, _path, _ticker
from ._cartesian import Axes, tick_values
from ._style import _dashes, fontsize_pt, rc, rgba

_ZERO = {'N': 0.5, 'NW': 0.75, 'W': 1.0, 'SW': 1.25, 'S': 1.5, 'SE': 1.75, 'E': 0.0, 'NE': 0.25}


def _full_circle(t0, t1):
    return abs(abs(t1 - t0) - 2 * np.pi) < 1.74e-14


class ThetaLocator(_ticker.Locator):
    """Every 45 degrees round the full circle (matplotlib's)."""

    def tick_values(self, vmin, vmax, tick_space=None, major=None):
        if _full_circle(vmin, vmax):
            return min(vmin, vmax) + np.arange(8) * 2 * np.pi / 8
        raise NotImplementedError('partial polar wedges are not in fastplot yet')


class ThetaFormatter(_ticker.Formatter):
    """Degrees with a degree sign, as many digits as the view needs."""

    def __call__(self, x, pos=None):
        vmin, vmax = self._view
        d = np.rad2deg(abs(vmax - vmin))
        digits = max(-int(np.log10(d) - 1.5), 0)
        return f'{np.rad2deg(x):0.{digits}f}\N{DEGREE SIGN}'

    def get_offset(self):
        return ''


class RadialLocator(_ticker.Locator):
    """matplotlib's: the base locator's ticks above the origin; 0 taken
    into the view (when the scale allows it)."""

    def __init__(self, base, ax):
        self.base, self.ax = base, ax

    def tick_values(self, vmin, vmax, tick_space=None, major=None):
        ticks = np.asarray(tick_values(self.base, vmin, vmax, tick_space), float)
        ax = self.ax
        rorigin = ax.get_rorigin() * ax.get_rsign()
        if ax.get_rmin() <= rorigin:
            ticks = ticks[ticks > rorigin]
        return ticks

    def _zero_in_bounds(self):
        lo, _ = self.ax.yaxis.scale.limit_range_for_scale(0, 1, 1e-5)
        return lo == 0

    def nonsingular(self, vmin, vmax):
        if self._zero_in_bounds() and (vmin, vmax) == (-np.inf, np.inf):
            return 0, 1
        return self.base.nonsingular(vmin, vmax) if hasattr(self.base, 'nonsingular') else \
            _ticker.nonsingular(vmin, vmax, expander=0.05)

    def view_limits(self, vmin, vmax):
        if hasattr(self.base, 'view_limits'):
            vmin, vmax = self.base.view_limits(vmin, vmax)
        if self._zero_in_bounds() and vmax > vmin:
            vmin = min(0, vmin)
        return _ticker.nonsingular(vmin, vmax)


class PolarAxes(Axes):
    name = 'polar'

    def __init__(self, fig, spec, sharex=None, sharey=None, theta_offset=0.0,
                 theta_direction=1, rlabel_position=22.5):
        super().__init__(fig, spec, sharex=sharex, sharey=sharey)
        self._theta_offset = float(theta_offset)
        self._theta_direction = theta_direction
        self._rlabel_position = float(rlabel_position)
        self._rorigin = None
        self.set_aspect('equal', adjustable='box', anchor='C')
        self.set_xlim(0.0, 2 * np.pi)
        self.xaxis.major_locator = ThetaLocator()
        self.xaxis.major_formatter = ThetaFormatter()
        self.yaxis.major_locator = RadialLocator(self.yaxis.major_locator, self)
        self.grid(rc('polaraxes.grid'))
        for t in (*self.xaxis.ticks.values(), *self.yaxis.ticks.values()):
            t['on1'] = t['on2'] = False                 # ticks position 'none'
        self.spines['polar'] = self.spines['top']       # the circle (matplotlib's name)
        self.spines['start'], self.spines['end'] = self.spines['left'], self.spines['right']
        for name in ('start', 'end'):
            self.spines[name].visible = False

    def _data_ratio(self, tx, ty):
        return 1.0                                      # matplotlib's get_data_ratio

    # -- matplotlib's PolarAxes API
    def set_yscale(self, value, **kw):
        super().set_yscale(value, **kw)
        self.yaxis.major_locator = RadialLocator(self.yaxis.major_locator, self)

    set_rscale = set_yscale

    def set_theta_offset(self, offset):
        self._theta_offset = float(offset)

    def get_theta_offset(self):
        return self._theta_offset

    def set_theta_zero_location(self, loc, offset=0.0):
        self.set_theta_offset(_ZERO[loc] * np.pi + np.deg2rad(offset))

    def set_theta_direction(self, direction):
        if direction in ('clockwise', -1):
            self._theta_direction = -1
        elif direction in ('counterclockwise', 'anticlockwise', 1):
            self._theta_direction = 1
        else:
            raise ValueError('direction must be 1, -1, clockwise or counterclockwise')

    def get_theta_direction(self):
        return self._theta_direction

    def set_thetamin(self, thetamin):
        raise NotImplementedError('partial polar wedges are not in fastplot yet')

    set_thetamax = set_thetamin

    def set_thetalim(self, *args, **kw):
        raise NotImplementedError('partial polar wedges are not in fastplot yet')

    def set_rmax(self, rmax):
        self.set_ylim(self.get_ylim()[0], rmax)

    def get_rmax(self):
        return self.get_ylim()[1]

    def set_rmin(self, rmin):
        self.set_ylim(rmin, self.get_ylim()[1])

    def get_rmin(self):
        return self.get_ylim()[0]

    def set_rorigin(self, rorigin):
        self._rorigin = rorigin

    def get_rorigin(self):
        return self.get_ylim()[0] if self._rorigin is None else self._rorigin

    def get_rsign(self):
        lo, hi = self.get_ylim()
        return np.sign(hi - lo)

    def set_rlim(self, bottom=None, top=None, *, emit=True, auto=False, **kw):
        bottom = kw.pop('rmin', bottom)
        top = kw.pop('rmax', top)
        return self.set_ylim(bottom, top, emit=emit, auto=auto)

    def set_rlabel_position(self, value):
        self._rlabel_position = float(value)

    def get_rlabel_position(self):
        return self._rlabel_position

    def set_rticks(self, *args, **kw):
        return self.set_yticks(*args, **kw)

    def set_thetagrids(self, angles, labels=None, fmt=None, **kw):
        angles = np.asarray(angles, float)
        self.set_xticks(np.deg2rad(angles), labels)
        if fmt is not None and labels is None:
            self.xaxis.set_major_formatter(_ticker.FuncFormatter(
                lambda x, pos: fmt % np.rad2deg(x)))
        self._label_style('x', kw)
        return self.xaxis

    def set_rgrids(self, radii, labels=None, angle=None, fmt=None, **kw):
        self.set_yticks(radii, labels)
        if angle is not None:
            self.set_rlabel_position(angle)
        if fmt is not None and labels is None:
            self.yaxis.set_major_formatter(_ticker.FormatStrFormatter(fmt))
        self._label_style('y', kw)
        return self.yaxis

    def _label_style(self, axis, kw):
        """matplotlib's Text keywords of set_thetagrids / set_rgrids (size,
        colour) on that axis' tick labels."""
        style = {}
        size = kw.get('fontsize', kw.get('size'))
        if size is not None:
            style['labelsize'] = size
        if kw.get('color') is not None:
            style['labelcolor'] = kw['color']
        if style:
            self.tick_params(axis=axis, **style)

    # -- the mapping
    def _r_scaled(self):
        """(rmin, rmax, origin) through the radial scale."""
        lo, hi = self._lims['y']
        s = self.yaxis.scale.transform(np.array([lo, hi, self.get_rorigin()], float))
        return s[0], s[1], s[2]

    def _data_px(self, x, y):
        """(theta, r) -> canvas pixels: matplotlib's PolarTransform and
        PolarAffine onto the axes' box."""
        lo, hi, ro = self._r_scaled()
        t = np.asarray(x, np.float64) * self._theta_direction + self._theta_offset
        rr = (self.yaxis.scale.transform(np.asarray(y, np.float64)) - ro) * np.sign(hi - lo)
        with np.errstate(invalid='ignore'):
            rr = np.where(rr >= 0, rr, np.nan)
        k = 0.5 / abs(hi - lo) if hi != lo else 0.0
        x0, y0, x1, y1 = self.box_f
        return (x0 + (0.5 + rr * np.cos(t) * k) * (x1 - x0),
                y1 - (0.5 + rr * np.sin(t) * k) * (y1 - y0))

    def _separable(self):
        return False

    def _data_limits(self, name):
        lo, hi, minpos, stickies = super()._data_limits(name)
        if name == 'y':                                 # the radial axis' sticky 0
            stickies = list(stickies) + [0.0]
        return lo, hi, minpos, stickies

    def _arc(self, r, t0=0.0, t1=2 * np.pi, steps=None):
        """A circle (arc) of radius r in data, as px: Agg's Bezier arc, flat."""
        lo, hi, ro = self._r_scaled()
        rr = (self.yaxis.scale.transform(np.array([r], float))[0] - ro) * np.sign(hi - lo)
        x0, y0, x1, y1 = self.box_f
        rad = rr * 0.5 / abs(hi - lo) * (x1 - x0)
        a0 = t0 * self._theta_direction + self._theta_offset
        a1 = t1 * self._theta_direction + self._theta_offset
        p = _path.arc(np.rad2deg(min(a0, a1)), np.rad2deg(max(a0, a1)))
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        v = p.vertices * rad
        xy, offs, _ = _path.flatten(np.column_stack([cx + v[:, 0], cy - v[:, 1]]), p.codes)
        return xy, offs

    def _interp_polys(self, polys, steps=100):
        """matplotlib's path interpolation (bars: 100 steps): an edge at one
        radius an arc, one at one angle straight, others interpolated."""
        out = []
        for p in polys:
            p = np.asarray(p, float)
            ring = np.r_[p, p[:1]]
            pts = [ring[:1]]
            for a, b in zip(ring[:-1], ring[1:]):
                if a[0] == b[0]:
                    pts.append(b[None])
                else:
                    f = np.linspace(0, 1, steps + 1)[1:, None]
                    pts.append(a + (b - a) * f)
            out.append(np.concatenate(pts)[:-1])
        return out

    # -- drawing
    def _draw_background(self, r):
        if not self.patch_visible:
            return
        x0, y0, x1, y1 = self.box_f
        p = _path.unit_circle()
        rad = 0.5 * (x1 - x0)
        v = p.vertices * rad
        xy, offs, _ = _path.flatten(np.column_stack([0.5 * (x0 + x1) + v[:, 0],
                                                     0.5 * (y0 + y1) - v[:, 1]]), p.codes)
        r.c.polygons(xy, offs, rgba(self.facecolor), None, 'nonzero')

    def _draw_spines(self, r):
        if not self.frame_on:
            return
        sp = self.spines.get('polar', self.spines.get('top'))
        if sp is None or not sp.visible or sp.linewidth <= 0:
            return
        xy, offs = self._arc(self._lims['y'][1])
        d = _dashes(sp.linestyle, sp.linewidth)
        r.lines(xy, offs, rgba(sp.edgecolor, sp.alpha), r.px(sp.linewidth), None,
                None if d is None else tuple(r.px(v) for v in d), None, _draw.BUTT)

    def _draw_axes_lines(self, r):
        """The grid: radial lines at the theta ticks, circles at the r ticks."""
        lo, hi = self._lims['y']
        for axis in (self.xaxis, self.yaxis):
            if not axis.visible:
                continue
            for which in ('major', 'minor'):
                t = axis.ticks[which]
                if not t['grid']:
                    continue
                locs = self._ticks(r, axis, which)[0]
                if not locs.size:
                    continue
                kw = t['grid_kw']
                color = kw.get('color', kw.get('c', rc('grid.color')))
                lw = kw.get('linewidth', kw.get('lw', rc('grid.linewidth')))
                ls = kw.get('linestyle', kw.get('ls', rc('grid.linestyle')))
                alpha = kw.get('alpha', rc('grid.alpha'))
                d = _dashes(ls, lw)
                dash = None if d is None else tuple(r.px(v) for v in d)
                cap = _draw.PROJECTING if d is None else _draw.BUTT
                if axis is self.xaxis:
                    n = locs.size
                    px, py = self._data_px(np.repeat(locs, 2), np.tile([lo, hi], n))
                    xy = np.column_stack([px, py])
                    offs = np.arange(0, 2 * n + 1, 2)
                else:
                    parts = [self._arc(v) for v in locs]
                    xy = np.concatenate([p[0] for p in parts])
                    sizes = np.cumsum([0] + [len(p[0]) for p in parts])
                    offs = np.concatenate([p[1][:-1] + sizes[k] for k, p in enumerate(parts)] +
                                          [sizes[-1:]])
                r.lines(xy, offs, rgba(color, alpha), r.px(lw), self.box_clip, dash, None, cap)

    def _draw_tick_labels(self, r):
        """Theta labels just outside the circle (matplotlib's ThetaTick:
        14 pt out along the angle, centred); radial labels at
        rlabel_position, bottom-left of their point."""
        exts = {'bottom': [], 'top': [], 'left': [], 'right': []}
        lo, hi = self._lims['y']
        t = self.xaxis.ticks['major']
        if self.xaxis.visible and t['label1']:
            locs, labels, _ = self._ticks(r, self.xaxis, 'major')
            size = fontsize_pt(self.xaxis.tick_label_kw.get('fontsize', t['labelsize']))
            color = self.xaxis.tick_label_kw.get('color', t['labelcolor'])
            # matplotlib's ThetaTick: the tick's pad (its own and its length out) + 7 pt
            pad = r.px(t['pad'] + t['size'] * {'in': 0.0, 'inout': 0.5, 'out': 1.0}[
                t['direction']] + 7.0)
            px, py = self._data_px(locs, np.full(locs.shape, hi))
            ang = locs * self._theta_direction + self._theta_offset
            for x, y, a, lab in zip(px, py, ang, labels):
                e = r.text(lab, x + pad * np.cos(a), y - pad * np.sin(a), size, color,
                           ha='center', va='center')
                if e is not None:
                    exts['bottom'].append(e)
            exts['top'] = list(exts['bottom'])          # the title goes above them
        t = self.yaxis.ticks['major']
        if self.yaxis.visible and t['label1']:
            locs, labels, _ = self._ticks(r, self.yaxis, 'major')
            size = fontsize_pt(self.yaxis.tick_label_kw.get('fontsize', t['labelsize']))
            color = self.yaxis.tick_label_kw.get('color', t['labelcolor'])
            th = np.full(locs.shape, np.deg2rad(self._rlabel_position))
            px, py = self._data_px(th, locs)
            for x, y, lab in zip(px, py, labels):
                if np.isfinite(x) and np.isfinite(y):
                    e = r.text(lab, x, y, size, color, ha='left', va='bottom')
                    if e is not None:
                        exts['left'].append(e)
        return exts

