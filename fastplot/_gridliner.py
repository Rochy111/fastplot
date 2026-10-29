"""cartopy's Gridliner: graticule lines and their labels on the map's edges."""
import numpy as np

from . import _proj, _ticker
from ._style import _dashes, fontsize_pt, rc, rgba
from . import _draw
from ._artists import Artist, _project_lines, cached_geometry

_SAMPLES = 400                  # points along a graticule line when finding where it leaves


def _label_fn(fmt, default):
    """A tick label maker: the default, a callable (matplotlib Formatters
    are), or a %- or {}-format string."""
    if fmt is None:
        return default
    if callable(fmt):
        return fmt
    return (lambda v: fmt.format(v)) if '{' in fmt else (lambda v: fmt % v)


def _edge_crossings(ax, crs, edge, lon=None, lat=None):
    """Where the meridian `lon` (or the parallel `lat`) meets an edge of the
    map: x positions on the bottom/top edge, y on the left/right one."""
    b = ax.box_f
    P = ax.projection
    if P.cylindrical and P.kind != _proj.ROTATED_POLE:   # meridians upright, parallels level
        if lon is not None:
            if edge not in ('bottom', 'top'):
                return []
            X, _ = _proj.transform_points(crs, P, np.array([lon]), np.array([0.0]))
            px, _ = ax.to_px(X, np.array([0.0]))
            xs = [px[0] + k * ax.period_px for k in (-1, 0, 1)]
            return [x for x in xs if np.isfinite(x) and b[0] - 0.5 <= x <= b[2] + 0.5]
        if edge not in ('left', 'right'):
            return []
        c = 0.5 * (ax.extent[0] + ax.extent[1])
        lo_c = _proj.transform_points(P, crs, np.array([c]), np.array([0.0]))[0]
        _, Y = _proj.transform_points(crs, P, lo_c, np.array([lat]))
        y = ax.to_px(np.array([c]), Y)[1][0]
        return [y] if np.isfinite(y) and b[1] + 1.0 < y < b[3] - 1.0 else []
    n = _SAMPLES
    if lon is not None:
        X, Y = _proj.transform_points(crs, P, np.full(n, lon), np.linspace(-90.0, 90.0, n))
    else:
        lo = np.linspace(-180.0, 180.0, 4 * n)
        X, Y = _proj.transform_points(crs, P, lo, np.full(lo.size, lat))
    px, py = ax.to_px(X, Y)
    if edge in ('bottom', 'top'):
        along, across, lo_, hi_ = px, py, b[0] - 0.5, b[2] + 0.5
        e = b[3] if edge == 'bottom' else b[1]
    else:
        along, across, lo_, hi_ = py, px, b[1] + 1.0, b[3] - 1.0
        e = b[0] if edge == 'left' else b[2]
    s = across - e
    ok = np.isfinite(s[:-1]) & np.isfinite(s[1:]) & np.isfinite(along[:-1]) & \
        np.isfinite(along[1:])
    out = []
    for k in np.nonzero(ok & (np.sign(s[:-1]) != np.sign(s[1:])))[0]:
        t = s[k] / (s[k] - s[k + 1])
        v = along[k] + t * (along[k + 1] - along[k])
        if lo_ <= v <= hi_:
            out.append(v)
    return out


class Gridliner(Artist):
    """cartopy's Gridliner: graticule lines and their labels."""

    zorder = 2.0

    def __init__(self, ax, crs, draw_labels=False, xlocs=None, ylocs=None, color=None,
                 linewidth=None, linestyle='-', alpha=None, zorder=None, xpadding=5,
                 ypadding=5):
        super().__init__(ax, zorder, alpha)
        self.crs = crs
        on = bool(draw_labels)
        self.top_labels = self.bottom_labels = self.left_labels = self.right_labels = on
        if isinstance(draw_labels, (list, tuple, dict, str)):
            sides = draw_labels if not isinstance(draw_labels, str) else [draw_labels]
            for side in ('top', 'bottom', 'left', 'right'):
                setattr(self, side + '_labels', side in sides)
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

    def _ticks(self, ax):
        lon0, lon1, lat0, lat1 = ax.lonlat_extent()
        lons = np.asarray(self.xlocator.tick_values(lon0, lon1), float)
        lats = np.asarray(self.ylocator.tick_values(lat0, lat1), float)
        xl, yl = self.crs.x_limits, self.crs.y_limits
        lons = lons[(lons >= max(lon0, xl[0]) - 1e-9) & (lons <= lon1 + 1e-9)]
        lats = lats[(lats >= max(lat0, yl[0]) - 1e-9) & (lats <= min(lat1, yl[1]) + 1e-9)]
        return lons, lats, (lon0, lon1, lat0, lat1)

    def _geometry(self, ax, lons, lats, lims):
        """The graticule as segments (and a memo of their binning), and a
        dict for its edge crossings -- one per map view and set of ticks."""
        lon0, lon1, lat0, lat1 = lims
        n = self.n_steps
        xs, ys = [], []
        if self.xlines:
            for lo in lons:
                xs.append(np.full(n, lo))
                ys.append(np.linspace(max(lat0, -90), min(lat1, 90), n))
        if self.ylines:
            for la in lats:
                xs.append(np.linspace(lon0, lon1, n))
                ys.append(np.full(n, la))
        segs = None
        if xs:
            offs = np.arange(len(xs) + 1) * n
            segs = _draw.segments(*_project_lines(ax, np.concatenate(xs), np.concatenate(ys),
                                                  offs, self.crs))
        return segs, {}, {}

    def render(self, r, ax):
        lons, lats, lims = self._ticks(ax)
        key = ('graticule', self.crs, ax.view_key, lons.tobytes(), lats.tobytes(), lims,
               self.n_steps, self.xlines, self.ylines)
        segs, memo, crossings = cached_geometry(key, lambda: self._geometry(ax, lons, lats,
                                                                            lims))
        if segs is not None:
            d = _dashes(self.linestyle, self.linewidth)
            styles = _draw.Styles(rgba(self.color, self.alpha), r.px(self.linewidth),
                                  None if d is None else [tuple(r.px(v) for v in d)])
            r.c.segments(segs, styles, ax.box_clip, memo)
        self._labels(r, ax, lons, lats, crossings)

    def _labels(self, r, ax, lons, lats, crossings):
        """Labels where the lines leave the map, none overlapping another;
        the top ones' extent goes to the axes (the title sits above it).
        crossings: the edge crossings found so far for this view."""
        if not (self.top_labels or self.bottom_labels or self.left_labels or self.right_labels):
            return
        b = ax.box_f                                    # float box: x0, y0, x1, y1
        xs, ys = self.xlabel_style, self.ylabel_style
        xsize = fontsize_pt(xs.get('size', xs.get('fontsize')))
        ysize = fontsize_pt(ys.get('size', ys.get('fontsize')))
        xcol, ycol = xs.get('color', 'black'), ys.get('color', 'black')
        padx, pady = r.px(self.xpadding), r.px(self.ypadding)
        fmtx = _label_fn(self.xformatter, _ticker.lon_label)
        fmty = _label_fn(self.yformatter, _ticker.lat_label)
        seen, tops = [], []

        def cross(edge, lon=None, lat=None):
            k = (edge, lon, lat)
            if k not in crossings:
                crossings[k] = _edge_crossings(ax, self.crs, edge, lon=lon, lat=lat)
            return crossings[k]

        def put(text, x, y, ha, va, size, color, top=False):
            ext = r.measure(text, x, y, size, ha=ha, va=va)
            if ext is None or any(ext[0] < e[2] and ext[2] > e[0] and ext[1] < e[3] and
                                  ext[3] > e[1] for e in seen):
                return
            seen.append(ext)
            r.text(text, x, y, size, color, ha=ha, va=va)
            if top:
                tops.append(ext)

        for lon in lons:
            lbl = fmtx(lon)
            if self.bottom_labels:
                for x in cross('bottom', lon=lon):
                    put(lbl, x, b[3] + padx, 'center', 'top', xsize, xcol)
            if self.top_labels:
                for x in cross('top', lon=lon):
                    put(lbl, x, b[1] - padx, 'center', 'bottom', xsize, xcol, top=True)
        for lat in lats:
            lbl = fmty(lat)
            if self.left_labels:
                for y in cross('left', lat=lat):
                    put(lbl, b[0] - pady, y, 'right', 'center', ysize, ycol)
            if self.right_labels:
                for y in cross('right', lat=lat):
                    put(lbl, b[2] + pady, y, 'left', 'center', ysize, ycol)
        if tops:
            ax._top_extent = min(e[1] for e in tops)
