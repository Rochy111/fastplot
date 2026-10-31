"""cartopy's Gridliner: graticule lines and their labels on the map's edges."""
import numpy as np

from . import _proj, _ticker
from ._style import _dashes, fontsize_pt, rc, rgba
from . import _draw
from ._artists import Artist, _project_lines, cached_geometry
from ._text import box_corners, family_key

_X_INLINE = ('LambertConformal', 'RotatedPole')
_POLAR = ('NorthPolarStereo', 'SouthPolarStereo', 'Stereographic')
_ROTATE = _POLAR + ('AlbersEqualArea', 'LambertConformal')


def _label_fn(fmt, default):
    """A tick label maker: the default, a callable (matplotlib Formatters
    are), or a %- or {}-format string."""
    if fmt is None:
        return default
    if callable(fmt):
        return fmt
    return (lambda v: fmt.format(v)) if '{' in fmt else (lambda v: fmt % v)


def _inside_parts(pts, ring):
    """shapely's line.intersection(polygon): the pieces of the polyline pts
    (n, 2) inside the closed ring, each a list of points, its vertices and
    where it crosses the ring."""
    a, d = pts[:-1], np.diff(pts, axis=0)
    p, e = ring[:-1], np.diff(ring, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        den = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
        wx, wy = p[None, :, 0] - a[:, None, 0], p[None, :, 1] - a[:, None, 1]
        t = (wx * e[None, :, 1] - wy * e[None, :, 0]) / den
        u = (wx * d[:, None, 1] - wy * d[:, None, 0]) / den
    hit = (den != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    ins = _inside(ring, pts[:, 0], pts[:, 1])
    out, cur = [], None
    for k in range(a.shape[0]):
        if not hit[k].any():
            if ins[k]:
                if cur is None:
                    cur = [pts[k]]
                cur.append(pts[k + 1])
            elif cur is not None:
                out.append(cur)
                cur = None
            continue
        knots = np.unique(np.r_[0.0, t[k][hit[k]], 1.0])
        mid = 0.5 * (knots[:-1] + knots[1:])
        at = _inside(ring, a[k, 0] + mid * d[k, 0], a[k, 1] + mid * d[k, 1])
        for j in range(mid.size):
            if at[j]:
                if cur is None:
                    cur = [a[k] + knots[j] * d[k]]
                cur.append(a[k] + knots[j + 1] * d[k])
            elif cur is not None:
                out.append(cur)
                cur = None
    if cur is not None:
        out.append(cur)
    return out


def _inside(ring, x, y):
    """Whether points are inside a closed ring (even-odd)."""
    x0, y0 = ring[:-1, 0][:, None], ring[:-1, 1][:, None]
    x1, y1 = ring[1:, 0][:, None], ring[1:, 1][:, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        hit = ((y0 > y) != (y1 > y)) & (x < x0 + (y - y0) * (x1 - x0) / (y1 - y0))
    return (hit.sum(axis=0) % 2) == 1


def _close(a, b):
    return np.abs(a - b) <= 1e-10 + 1e-5 * np.abs(b)


def _crossing(p, q):
    """matplotlib's path_intersects_path of two polylines: any two segments
    meeting, its tolerances (collinear ones only when they overlap)."""
    for (x1, y1), (x2, y2) in zip(p[:-1], p[1:]):
        if _close((x1 - x2) ** 2 + (y1 - y2) ** 2, 0.0):
            continue
        for (x3, y3), (x4, y4) in zip(q[:-1], q[1:]):
            if _close((x3 - x4) ** 2 + (y3 - y4) ** 2, 0.0):
                continue
            den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            if _close(den, 0.0):
                area = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
                if _close(area, 0.0):
                    a0, a1, b0, b1 = (y1, y2, y3, y4) if x1 == x2 == x3 else (x1, x2, x3, x4)
                    if (min(a0, a1) <= min(b0, b1) <= max(a0, a1)) or \
                            (min(b0, b1) <= min(a0, a1) <= max(b0, b1)):
                        return True
                continue
            u1 = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
            u2 = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den
            if (u1 > 0.0 or _close(u1, 0.0)) and (u1 < 1.0 or _close(u1, 1.0)) and \
                    (u2 > 0.0 or _close(u2, 0.0)) and (u2 < 1.0 or _close(u2, 1.0)):
                return True
    return False


def _intersects(p, q, filled=True):
    """Path.intersects_path: filled, one inside the other counts too."""
    if _crossing(p, q):
        return True
    return filled and (bool(_inside(p, q[:, 0], q[:, 1]).all()) or
                       bool(_inside(q, p[:, 0], p[:, 1]).all()))


def _round_to(x, base):
    """Gridliner._round."""
    if np.isnan(base):
        base = 5
    return int(base * round(x / base))


class _Label:
    """cartopy's Label: a text, its turned box, where it sits."""

    def __init__(self, text, x, y, kw, style, path, xy, loc, visible):
        self.text, self.x, self.y, self.kw, self.style = text, x, y, kw, style
        self.path, self.xy, self.loc, self.visible = path, xy, loc, visible

    def overlaps(self, other):
        a, b = self.path, other.path
        if a[:, 0].min() > b[:, 0].max() or b[:, 0].min() > a[:, 0].max() or \
                a[:, 1].min() > b[:, 1].max() or b[:, 1].min() > a[:, 1].max():
            return False
        return _intersects(a, b)


def _side_value(v):
    if isinstance(v, str):
        v = v.lower()
    if not isinstance(v, (list, bool)) and v not in ('x', 'y'):
        raise ValueError('Invalid draw_labels argument: %s' % (v,))
    return v


class Gridliner(Artist):
    """cartopy's Gridliner: graticule lines and their labels."""

    zorder = 2.0

    def __init__(self, ax, crs, draw_labels=False, xlocs=None, ylocs=None, color=None,
                 linewidth=None, linestyle='-', alpha=None, zorder=None, xpadding=5,
                 ypadding=5, x_inline=None, y_inline=None, auto_inline=True, rotate_labels=None,
                 offset_angle=25):
        super().__init__(ax, zorder, alpha)
        self.crs = crs
        sides = ('top', 'bottom', 'left', 'right', 'geo')
        if isinstance(draw_labels, (list, tuple)):
            xy = [v for v in ('x', 'y') if v in draw_labels]
            value = True if not xy else xy[0] if len(xy) == 1 else ['x', 'y']
            values = {s: value if s in draw_labels else False for s in sides}
        elif isinstance(draw_labels, dict):
            values = {s: draw_labels.get(s, False) for s in sides}
        else:
            values = dict.fromkeys(sides, draw_labels)
        for s in sides:
            setattr(self, s + '_labels', _side_value(values[s]))
        name = type(ax.projection).__name__
        polar = name in _POLAR
        self.x_inline = bool(auto_inline) and name in _X_INLINE and not polar
        self.y_inline = bool(auto_inline) and polar
        if x_inline is not None:
            self.x_inline = x_inline
        if y_inline is not None:
            self.y_inline = y_inline
        if not draw_labels:
            self.inline_labels = False
        elif self.x_inline and self.y_inline:
            self.inline_labels = True
        elif self.x_inline:
            self.inline_labels = 'x'
        elif self.y_inline:
            self.inline_labels = 'y'
        else:
            self.inline_labels = False
        self.rotate_labels = name in _ROTATE if rotate_labels is None else rotate_labels
        self.offset_angle = offset_angle
        self.xlocator = _ticker.as_locator(xlocs) or _ticker.LongitudeLocator()
        self.ylocator = _ticker.as_locator(ylocs) or _ticker.LatitudeLocator()
        self.xformatter = self.yformatter = None
        self.color = rc('grid.color') if color is None else color
        self.linewidth = rc('grid.linewidth') if linewidth is None else linewidth
        self.linestyle = linestyle
        self.xlabel_style = {}
        self.ylabel_style = {}
        self.xpadding, self.ypadding = xpadding, ypadding
        self.xlines = self.ylines = True
        self.n_steps = 100

    # cartopy's property names
    @property
    def xlocs(self):
        return self.xlocator

    @xlocs.setter
    def xlocs(self, v):
        self.xlocator = _ticker.as_locator(v)

    @property
    def ylocs(self):
        return self.ylocator

    @ylocs.setter
    def ylocs(self, v):
        self.ylocator = _ticker.as_locator(v)

    def _domain(self, ax):
        """cartopy's _axes_domain: the range of this crs's x and y over a
        100 by 100 grid of the map, what the grid's points show of it."""
        P, crs = ax.projection, self.crs
        e = ax.extent
        t = np.linspace(1e-9, 1 - 1e-9, 100)
        fx, fy = np.meshgrid(t, t)
        X, Y = (e[0] + fx * (e[1] - e[0])).ravel(), (e[2] + fy * (e[3] - e[2])).ravel()
        lon, lat = _proj.transform_points(P, crs, X, Y)
        if ax._outline() is not None:
            ok = ax._on_map(X, Y)
            lon, lat = lon[ok], lat[ok]
        xl, yl = tuple(crs.x_limits), tuple(crs.y_limits)
        fin = lat[np.isfinite(lat)]
        if not fin.size:
            return xl, yl
        lon_lim = np.clip((np.nanmin(lon), np.nanmax(lon)), *xl)
        lat_lim = np.clip((np.nanmin(lat), fin.max()), *yl)
        if abs(lon_lim[1] - lon_lim[0]) / abs(xl[1] - xl[0]) > 0.9:    # nearly all: all
            lon_lim = xl
        return tuple(float(v) for v in lon_lim), tuple(float(v) for v in lat_lim)

    def _ticks(self, ax):
        (lon0, lon1), (lat0, lat1) = cached_geometry(('grid-domain', self.crs, ax.view_key),
                                                     lambda: self._domain(ax))
        lons = np.asarray(self.xlocator.tick_values(lon0, lon1), float)
        lats = np.asarray(self.ylocator.tick_values(lat0, lat1), float)
        xl, yl = self.crs.x_limits, self.crs.y_limits
        lons = lons[(lons >= max(lon0, xl[0])) & (lons <= min(lon1, xl[1]))]
        lats = lats[(lats >= max(lat0, yl[0])) & (lats <= min(lat1, yl[1]))]
        return lons, lats, (lon0, lon1, lat0, lat1)

    def _meridians(self, lons, lats, lims):
        """The meridians' points in the crs: x and y (lines, n_steps)."""
        lat0, lat1 = lims[2:]
        if lats.size:
            lat0, lat1 = min(lat0, lats.min()), max(lat1, lats.max())
        n = self.n_steps
        return (np.repeat(lons[:, None], n, axis=1),
                np.broadcast_to(np.linspace(lat0, lat1, n), (lons.size, n)))

    def _parallels(self, lons, lats, lims):
        lon0, lon1 = lims[:2]
        if lons.size:
            lon0, lon1 = min(lon0, lons.min()), max(lon1, lons.max())
        n = self.n_steps
        return (np.broadcast_to(np.linspace(lon0, lon1, n), (lats.size, n)),
                np.repeat(lats[:, None], n, axis=1))

    def _geometry(self, ax, lons, lats, lims, width):
        """The graticule as polylines (and a memo of their outlines), and a
        dict for its edge crossings -- one per map view and set of ticks.
        Meridians reach the outermost parallel and parallels the outermost
        meridian, wherever the domain stops: cartopy's, which sets where
        each line's dashes start."""
        xs, ys = [], []
        if self.xlines:
            xs, ys = self._meridians(lons, lats, lims)
            xs, ys = list(xs), list(ys)
        if self.ylines:
            x2, y2 = self._parallels(lons, lats, lims)
            xs, ys = xs + list(x2), ys + list(y2)
        segs = None
        if xs:
            offs = np.arange(len(xs) + 1) * self.n_steps
            xy, off = _project_lines(ax, np.concatenate(xs), np.concatenate(ys), offs, self.crs)
            segs = _draw.segments(_draw.snap(xy, off, width), off)
        return segs, {}, {}

    def render(self, r, ax):
        lons, lats, lims = self._ticks(ax)
        w = r.px(self.linewidth)
        key = ('graticule', self.crs, ax.view_key, lons.tobytes(), lats.tobytes(), lims,
               self.n_steps, self.xlines, self.ylines, w)
        segs, memo, crossings = cached_geometry(key, lambda: self._geometry(ax, lons, lats, lims,
                                                                            w))
        if segs is not None:
            d = _dashes(self.linestyle, self.linewidth)
            styles = _draw.Styles(rgba(self.color, self.alpha), r.px(self.linewidth),
                                  None if d is None else [tuple(r.px(v) for v in d)])
            r.c.segments(segs, styles, ax.box_clip, memo)
        self._labels(r, ax, lons, lats, crossings, lims)

    # cartopy's rules for which labels show and how they turn
    def _draw_this_label(self, xylabel, loc):
        show = getattr(self, loc + '_labels')
        if show is True and loc != 'geo':
            show = 'x' if loc in ('top', 'bottom') else 'y'
        if not show:
            return False
        if isinstance(show, str):
            show = [show]
        return not (isinstance(show, list) and xylabel not in show)

    @staticmethod
    def _loc_from_angle(angle):
        angle %= 360
        if angle > 180:
            angle -= 360
        if abs(angle) <= 45:
            return 'right'
        if abs(angle) >= 135:
            return 'left'
        return 'top' if angle > 45 else 'bottom'

    @staticmethod
    def _loc_from_spine(spines, xylabel, x, y):
        order = ('bottom', 'top', 'left', 'right') if xylabel == 'x' else \
            ('left', 'right', 'bottom', 'top')
        for side in order:
            v = x if side in ('left', 'right') else y
            if round(v, 2) == round(spines[side], 2):
                return side
        return 'geo'

    @staticmethod
    def _loc_from_overlap(spines, xylabel, path):
        """Which side a label's box reaches past the furthest, if any."""
        side_max, dist_max = None, None
        for side, index, xy, ge in (('left', 0, 'x', False), ('bottom', 1, 'y', False),
                                    ('right', 0, 'x', True), ('top', 1, 'y', True)):
            if xy == xylabel:
                continue
            c, s = path[:-1, index], spines[side]
            if not ((c >= s) if ge else (c <= s)).any():
                continue
            dist = c.min() - s if ge else s - c.max()
            if side_max is None or dist > dist_max:
                side_max, dist_max = side, dist
        return 'geo' if side_max is None else side_max

    def _text_specs(self, angle, loc, xylabel):
        """Rotation and alignment of a label."""
        if angle > 180:
            angle -= 360
        if loc == 'geo':
            loc = self._loc_from_angle(angle)
        if not self.rotate_labels:
            kw = dict(rotation=0.0, ha='center', va='center')
            if loc == 'right':
                kw['ha'] = 'left'
            elif loc == 'left':
                kw['ha'] = 'right'
            elif loc == 'top':
                kw['va'] = 'bottom'
            elif loc == 'bottom':
                kw['va'] = 'top'
        else:
            if isinstance(self.rotate_labels, (float, int)) and \
                    not isinstance(self.rotate_labels, bool):
                angle = self.rotate_labels
            kw = dict(rotation=angle, rotation_mode='anchor', va='center')
            if -90 + self.offset_angle < angle < 90 + self.offset_angle:
                kw.update(ha='left', rotation=angle)
            else:
                kw.update(ha='right', rotation=angle + 180)
        if getattr(self, xylabel + 'padding') < 0:      # inside the map
            flip = dict(left='right', right='left', top='bottom', bottom='top')
            kw['ha'] = flip.get(kw['ha'], kw['ha'])
            kw['va'] = flip.get(kw['va'], kw['va'])
        return kw

    def _offset(self, r, angle, loc, xylabel, factor=1.0):
        """The padding, px, y up, along angle (or square to a side)."""
        if self.rotate_labels is False and loc != 'geo':
            angle = {'top': 90.0, 'right': 0.0, 'bottom': -90.0, 'left': 180.0}[loc]
        p = factor * r.px(getattr(self, xylabel + 'padding'))
        return p * np.cos(np.deg2rad(angle)), p * np.sin(np.deg2rad(angle))

    def _midpoints(self, lim, ticks):
        cent = np.diff(ticks).mean() / 2 if len(ticks) > 1 else np.nan
        lq, uq = (90, 90) if type(self.axes.projection).__name__ in _POLAR else (25, 75)
        return _round_to(np.percentile(lim, lq), cent), _round_to(np.percentile(lim, uq), cent)

    def _ends(self, ax, r, ring, x, y):
        """Each piece of a line (points in the crs) inside the map, by its
        two ends: pairs of (end, a point a little way in), in y-up pixels."""
        X, Y = _proj.transform_points(self.crs, ax.projection, x, y)
        px, py = ax.to_px(X, Y)
        pts = np.column_stack([px, r.H + r.oy - py])
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] < 2:
            return []
        parts = _inside_parts(pts, ring)
        if len(parts) > 4:                              # lines along the edge: as one
            parts = [[p for q in parts for p in q]]
        out = []
        for c in parts:
            n2 = min(len(c), 8)
            if n2 >= 2:
                out.append(((c[0], c[n2 - 1]), (c[-1], c[-n2])))
        return out

    def _labels(self, r, ax, lons, lats, crossings, lims):
        """cartopy's labels: where each line leaves the map, turned along it
        on conic and polar maps, or on the map itself where inline; moved
        out until clear of the frame, then none overlapping another. The top
        ones' extent goes to the axes (the title sits above it). crossings:
        the lines' ends found so far for this view."""
        if not any((self.left_labels, self.right_labels, self.bottom_labels, self.top_labels,
                    self.inline_labels, self.geo_labels)):
            return
        top = r.H + r.oy
        tops = []
        for q, _ in cached_geometry(self._label_key(r, ax, lons, lats, lims),
                                    lambda: self._placed(r, ax, lons, lats, crossings, lims)):
            size = fontsize_pt(q.style.get('size', q.style.get('fontsize')))
            ext = r.text(q.text, q.x, top - q.y, size, q.style.get('color', 'black'),
                         ha=q.kw.get('ha', 'left'), va=q.kw.get('va', 'baseline'),
                         rotation=q.kw.get('rotation', 0.0),
                         rotation_mode=q.kw.get('rotation_mode', 'default'),
                         weight=q.style.get('weight', q.style.get('fontweight', 'normal')))
            if q.loc == 'top' and ext is not None:
                tops.append(ext)
        if tops:
            ax._top_extent = min(e[1] for e in tops)

    def _label_key(self, r, ax, lons, lats, lims):
        """All that where the labels go depends on: the frames of a forecast
        place them once (6-7 ms a frame, the same each time). The texts
        themselves stand for the formatters, whatever state those keep."""
        xfmt = _label_fn(self.xformatter, _ticker.lon_label)
        yfmt = _label_fn(self.yformatter, _ticker.lat_label)
        sizes = [fontsize_pt(s.get('size', s.get('fontsize')))
                 for s in (self.xlabel_style, self.ylabel_style)]
        return ('labels', self.crs, ax.view_key, r.dpi, r.H, r.oy, lons.tobytes(), lats.tobytes(),
                lims, self.n_steps, tuple(str(xfmt(t)) for t in lons.tolist()),
                tuple(str(yfmt(t)) for t in lats.tolist()), tuple(sizes),
                repr(sorted(self.xlabel_style.items())), repr(sorted(self.ylabel_style.items())),
                repr((self.left_labels, self.right_labels, self.bottom_labels, self.top_labels,
                      self.inline_labels, self.geo_labels, self.x_inline, self.y_inline,
                      self.rotate_labels, self.offset_angle, self.xpadding, self.ypadding)),
                family_key(None), rc('mathtext.fontset'), rc('mathtext.default'))

    def _placed(self, r, ax, lons, lats, crossings, lims):
        """The labels that show, with their boxes (what cached_geometry
        counts them by)."""
        b = ax.box_f
        top = r.H + r.oy
        ring = ax._frame().copy()
        ring[:, 1] = top - ring[:, 1]
        spines = dict(left=b[0], right=b[2], bottom=top - b[3], top=top - b[1])
        x_mid = self._midpoints(lims[:2], lons) if self.y_inline else None
        y_mid = self._midpoints(lims[2:], lats) if self.x_inline else None
        labels = []
        for xylabel, (lx, ly), ticks, fmt, style in (
                ('x', self._meridians(lons, lats, lims), lons,
                 _label_fn(self.xformatter, _ticker.lon_label), self.xlabel_style),
                ('y', self._parallels(lons, lats, lims), lats,
                 _label_fn(self.yformatter, _ticker.lat_label), self.ylabel_style)):
            x_inline = self.x_inline and xylabel == 'x'
            y_inline = self.y_inline and xylabel == 'y'
            padding = getattr(self, xylabel + 'padding')
            size = fontsize_pt(style.get('size', style.get('fontsize')))
            weight = style.get('weight', style.get('fontweight', 'normal'))
            for k, tick in enumerate(ticks.tolist()):
                key = (xylabel, tick)
                if key not in crossings:
                    crossings[key] = self._ends(ax, r, ring, lx[k], ly[k])
                text = str(fmt(tick))
                for pair in crossings[key]:
                    for i, (pt0, pt1) in enumerate(pair):
                        if (x_inline or y_inline) and abs(tick) == 180:
                            continue
                        x0, y0 = pt0
                        angle = 0.0
                        if x_inline or y_inline:
                            loc = 'inline'
                            kw = dict(rotation=0.0, ha='center', va='center')
                            lo, la = (tick, y_mid[i]) if x_inline else (x_mid[i], tick)
                            X, Y = _proj.transform_points(self.crs, ax.projection,
                                                          np.array([float(lo)]),
                                                          np.array([float(la)]))
                            px, py = ax.to_px(X, Y)
                            x0, y0 = float(px[0]), top - float(py[0])
                            dx = dy = 0.0
                        else:
                            angle = np.degrees(np.arctan2(y0 - pt1[1], x0 - pt1[0]))
                            loc = self._loc_from_spine(spines, xylabel, x0, y0)
                            kw = self._text_specs(angle, loc, xylabel)
                            dx, dy = self._offset(r, angle, loc, xylabel)
                        for name in ('rotation', 'ha', 'va'):
                            if name in style:
                                kw[name] = style[name]

                        def box(ox, oy):
                            c = box_corners(text, r.px(size), kw.get('ha', 'left'),
                                            kw.get('va', 'baseline'), kw.get('rotation', 0.0),
                                            kw.get('rotation_mode', 'default'), weight)
                            return np.vstack([c, c[:1]]) + (x0 + ox, y0 + oy)
                        path = box(dx, dy)
                        if not (x_inline or y_inline) and loc == 'geo':
                            new = self._loc_from_overlap(spines, xylabel, path)
                            if new != loc:
                                loc = new
                                dx, dy = self._offset(r, angle, loc, xylabel)
                                kw = self._text_specs(angle, loc, xylabel)
                                for name in ('rotation', 'ha', 'va'):
                                    if name in style:
                                        kw[name] = style[name]
                                path = box(dx, dy)
                        visible = False
                        if not self._draw_this_label(xylabel, loc):
                            pass
                        elif loc == 'geo' and not getattr(self, self._loc_from_angle(angle) +
                                                          '_labels'):
                            pass
                        elif x_inline or y_inline:
                            visible = bool(_inside(ring, np.array([x0]), np.array([y0]))[0])
                        else:
                            factor = 1.0
                            while factor < 5:
                                if _intersects(ring, path, padding > 0):
                                    dx, dy = self._offset(r, angle, loc, xylabel, factor)
                                    path = box(dx, dy)
                                    factor += 0.2
                                else:
                                    visible = True
                                    break
                        labels.append(_Label(text, x0 + dx, y0 + dy, kw, style, path, xylabel,
                                             loc, visible))
        labels.sort(key=lambda q: q.loc in ('left', 'right', 'top', 'bottom'), reverse=True)
        shown = []
        for q in labels:
            if q.visible:
                if any(q.overlaps(o) for o in shown):
                    continue
                shown.append(q)
        return [(q, q.path) for q in shown]
