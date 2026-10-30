"""cartopy's GeoAxes: a map's projection and extent, the map-only
methods (coastlines, features, gridlines, set_extent), and the answers to
AxesBase's hooks through the projection: points, lines (seam-safe on
cylindrical maps), vectors and fields."""
import numpy as np

from . import _proj
from . import feature as _feature
from . import transforms as _transforms
from ._render import View
from ._style import _ignore, rc, rgba
from ._layout import SubplotSpec, _fit, spec_box
from ._artists import FeatureArtist, _project_lines, _wrap_x
from ._axesbase import AxesBase
from ._gridliner import Gridliner


def _inside(ring, x, y):
    """Whether points (x, y) are inside a closed ring (even-odd)."""
    x0, y0 = ring[:-1, 0][:, None], ring[:-1, 1][:, None]
    x1, y1 = ring[1:, 0][:, None], ring[1:, 1][:, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        hit = ((y0 > y) != (y1 > y)) & (x < x0 + (y - y0) * (x1 - x0) / (y1 - y0))
    return (hit.sum(axis=0) % 2) == 1


def _clip_ring(ring, x0, x1, y0, y1):
    """A ring clipped to a rectangle (Sutherland-Hodgman), closed."""
    pts = ring[:-1]
    for axis, lim, keep_above in ((0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)):
        if not len(pts):
            break
        v = pts[:, axis]
        ins = v >= lim if keep_above else v <= lim
        nxt = np.roll(np.arange(len(pts)), -1)
        out = []
        for k in range(len(pts)):
            a, b = pts[k], pts[nxt[k]]
            ia, ib = ins[k], ins[nxt[k]]
            if ia:
                out.append(a)
            if ia != ib:
                t = (lim - a[axis]) / (b[axis] - a[axis])
                out.append(a + t * (b - a))
        pts = np.array(out).reshape(-1, 2)
    return np.vstack([pts, pts[:1]]) if len(pts) else pts


class _Spine:
    def __init__(self):
        self.linewidth, self.edgecolor = rc('axes.linewidth'), rc('axes.edgecolor')
        self.visible = True

    def set_linewidth(self, w):
        self.linewidth = w

    def set_edgecolor(self, c):
        self.edgecolor = c

    set_color = set_edgecolor

    def set_visible(self, v):
        self.visible = bool(v)


class MapAxes(AxesBase):
    """cartopy's GeoAxes: a map in a projection, and what is drawn on it."""

    name = 'map'

    def __init__(self, fig, spec, projection, label=''):
        super().__init__(fig, spec, label)
        self.projection = projection if projection is not None else _proj.PlateCarree()
        self.spines = {'geo': _Spine()}
        self.transData = self.projection
        self._extent = None                             # set by set_extent / set_global
        self._aspect = 'equal'                          # set_aspect

    # -- AxesBase's hooks, through the projection
    def _src(self, transform):
        """The crs the data's coordinates are in (cartopy's transform=), or
        'axes', or another coordinate system (transforms)."""
        if transform is None:
            return self.projection
        if transform is self.transAxes or transform == 'axes':
            return 'axes'
        if isinstance(transform, _transforms.Transform):
            return transform
        if not hasattr(transform, 'kind'):
            raise TypeError('transform must be a fastplot.crs projection (or ax.transAxes)')
        return transform

    def _page_px(self, x, y, src):
        """Points in 'axes' or a transforms coordinate system (fig.transFigure,
        ...) -> canvas pixels, straight, as matplotlib draws them on a GeoAxes;
        None for a crs, which goes through the projection."""
        if isinstance(src, str):
            return self.transAxes.to_px(x, y)
        if isinstance(src, _transforms.Transform):
            return src.to_px(x, y)
        return None

    def _points_px(self, x, y, src):
        """Points in src -> canvas pixels (cylindrical maps: wrapped onto the map)."""
        page = self._page_px(x, y, src)
        if page is not None:
            return page
        X, Y = _proj.transform_points(src, self.projection, x, y)
        return self.to_px(_wrap_x(X, self.projection, self.extent[0]), Y)

    def _lines_px(self, x, y, offsets, src, parts=False):
        page = self._page_px(np.asarray(x, np.float64), np.asarray(y, np.float64), src)
        if page is not None:
            offsets = np.asarray(offsets, np.int64)
            out = (np.column_stack(page).astype(np.float64), offsets)
            return out + (np.arange(offsets.size - 1),) if parts else out
        return _project_lines(self, x, y, offsets, src, parts)

    def _shape_px(self, x, y, src):
        """A shape's vertices (a patch's path) -> canvas pixels, one copy per
        place it shows. On a cylindrical map the vertices are kept continuous
        in x and the shape is moved a whole number of periods to sit over the
        map, not wrapped vertex by vertex (which tears a shape crossing the
        map's edge across all of it); where it still runs off one edge of a
        map a period wide it is drawn again a period over, as lines are."""
        page = self._page_px(np.asarray(x, np.float64), np.asarray(y, np.float64), src)
        if page is not None:
            return [page]
        P = self.projection
        X, Y = _proj.transform_points(src, P, np.asarray(x, np.float64),
                                      np.asarray(y, np.float64))
        if not (P.cylindrical and P.period) or not X.size:
            return [self.to_px(X, Y)]
        per, (lo, hi) = P.period, self.extent[:2]
        ok = np.isfinite(X)
        if not ok.any():
            return [self.to_px(X, Y)]
        X = X.copy()
        X[ok] = np.unwrap(X[ok], period=per)
        mid = 0.5 * (np.nanmin(X) + np.nanmax(X))
        X += -per * np.round((mid - 0.5 * (lo + hi)) / per)
        out = [self.to_px(X, Y)]
        for shift in (per, -per):
            Xs = X + shift
            if np.nanmax(Xs) > lo and np.nanmin(Xs) < hi:
                out.append(self.to_px(Xs, Y))
        return out

    def _vectors_px(self, x, y, u, v, src):
        """Points and the screen direction of (u, v) at each, as cartopy's
        transform_vectors: a small step along (u, v) in the source crs."""
        P = self.projection
        X, Y = _proj.transform_points(src, P, x, y)
        mag = np.hypot(u, v)
        with np.errstate(invalid='ignore', divide='ignore'):
            eps = 1e-3 / np.where(mag > 0, mag, 1.0)
        X2, Y2 = _proj.transform_points(src, P, x + u * eps, y + v * eps)
        Xw = _wrap_x(X, P, self.extent[0])
        px, py = self.to_px(Xw, Y)
        px2, py2 = self.to_px(X2 + (Xw - X), Y2)
        dx, dy = px2 - px, py2 - py
        if P.cylindrical and P.period:                  # a step across the seam
            dx = np.where(np.abs(dx) > 0.5 * self.period_px, dx - np.sign(dx) * self.period_px,
                          dx)
        return px, py, dx, dy, mag

    def _draw_field(self, r, grid, z, mode, coloring, alpha):
        if grid.rect:
            r.c.field_rect(self.box, self.view, grid.src, grid.xaxis, grid.yaxis, z, mode,
                           coloring, alpha)
        else:
            r.c.field_cells(self.box, self.view, grid.src, grid.x2d, grid.y2d, z, mode,
                            coloring, alpha, self.period_px)

    def _default_extent(self, src, shape):
        return (src.x_limits[0], src.x_limits[1], src.y_limits[0], src.y_limits[1])

    def _bounds(self, artist):
        return artist.datalim(self.projection)

    # -- extent
    def set_extent(self, extents, crs=None):
        x0, x1, y0, y1 = _proj.extent_xy(self.projection, tuple(extents), crs)
        self._extent = (x0, x1, y0, y1)

    def set_aspect(self, aspect, adjustable=None, anchor=None, share=False):
        """cartopy's (matplotlib's): 'equal' (the default: a map's x and y to
        the same scale), 'auto' (the map stretched to fill the axes' box) or
        a number (y scale over x)."""
        if aspect != 'auto' and aspect != 'equal':
            aspect = float(aspect)
        self._aspect = aspect
        if anchor is not None:
            self.anchor = anchor

    def get_aspect(self):
        return 1.0 if self._aspect == 'equal' else self._aspect

    def set_global(self):
        self._extent = (self.projection.x_limits[0], self.projection.x_limits[1],
                        self.projection.y_limits[0], self.projection.y_limits[1])

    def set_xlim(self, a, b=None):
        a, b = (a if b is None else (a, b))
        e = self._extent or self._auto_extent()
        self._extent = (a, b, e[2], e[3])

    def set_ylim(self, a, b=None):
        a, b = (a if b is None else (a, b))
        e = self._extent or self._auto_extent()
        self._extent = (e[0], e[1], a, b)

    def get_extent(self, crs=None):
        e = self._extent or self._auto_extent()
        if crs is None or crs == self.projection:
            return e
        lon0, lon1, lat0, lat1 = self._ll_extent(e, crs)
        return (lon0, lon1, lat0, lat1)

    def _auto_extent(self):
        lims = [self._bounds(a) for a in self.artists if a.visible]
        lims = [l for l in lims if l is not None and all(np.isfinite(l))]
        xl, yl = self.projection.x_limits, self.projection.y_limits
        if not lims:
            return (xl[0], xl[1], yl[0], yl[1])
        x0 = max(min(l[0] for l in lims), xl[0])
        x1 = min(max(l[1] for l in lims), xl[1])
        y0 = max(min(l[2] for l in lims), yl[0])
        y1 = min(max(l[3] for l in lims), yl[1])
        if x1 <= x0 or y1 <= y0:
            return (xl[0], xl[1], yl[0], yl[1])
        return (x0, x1, y0, y1)

    @property
    def extent(self):
        if self._placed is not None:                    # fixed for the length of a draw
            return self._placed
        return self._extent or self._auto_extent()

    def _ll_extent(self, e, crs=None):
        """lon/lat bounding box of the map (cartopy's _axes_domain)."""
        crs = crs or _proj.PlateCarree()
        n = 40
        xs = np.linspace(e[0], e[1], n)
        ys = np.linspace(e[2], e[3], n)
        gx, gy = np.meshgrid(xs, ys)
        lon, lat = _proj.transform_points(self.projection, crs, gx.ravel(), gy.ravel())
        ok = np.isfinite(lon) & np.isfinite(lat)
        if not ok.any():
            return (-180.0, 180.0, -90.0, 90.0)
        lon, lat = lon[ok], lat[ok]
        if self.projection.cylindrical:
            c = 0.5 * (e[0] + e[1])
            clon = _proj.transform_points(self.projection, crs, np.array([c]),
                                          np.array([0.5 * (e[2] + e[3])]))[0][0]
            lon = clon + ((lon - clon + 180.0) % 360.0) - 180.0
            span = (e[1] - e[0]) / self.projection.period * 360.0
            lo = clon - span / 2 if np.isfinite(clon) else lon.min()
            return (lo, lo + span, max(lat.min(), -90.0), min(lat.max(), 90.0))
        # does the map hold a pole? then every longitude
        lat0, lat1 = lat.min(), lat.max()
        for pole in (90.0, -90.0):
            X, Y = _proj.transform_points(crs, self.projection, np.array([0.0]), np.array([pole]))
            if np.isfinite(X[0]) and e[0] <= X[0] <= e[1] and e[2] <= Y[0] <= e[3]:
                if pole > 0:
                    lat1 = 90.0
                else:
                    lat0 = -90.0
                return (-180.0, 180.0, lat0, lat1)
        lo = lon.min()
        hi = lon.max()
        if hi - lo > 180:                               # across the antimeridian
            l2 = np.where(lon < 0, lon + 360, lon)
            if l2.max() - l2.min() < hi - lo:
                lo, hi = l2.min(), l2.max()
        return (lo, hi, lat0, lat1)

    def lonlat_extent(self):
        return self._ll

    def lonlat_bbox(self):
        lo, hi, la0, la1 = self._ll
        if hi - lo >= 359.0:
            return None
        pad = 1.0
        return (lo - pad, hi + pad, la0 - pad, la1 + pad)

    # -- appearance

    # -- data

    def coastlines(self, resolution='auto', color=None, **kw):
        f = _feature.COASTLINE.with_scale(resolution)
        kw['edgecolor'] = color if color is not None else kw.pop('edgecolor', 'black')
        kw['facecolor'] = 'none'
        return self._add(FeatureArtist(self, f, **kw))

    def add_feature(self, feature, **kw):
        return self._add(FeatureArtist(self, feature, **kw))

    def gridlines(self, crs=None, draw_labels=False, xlocs=None, ylocs=None, **kw):
        crs = crs or _proj.PlateCarree()
        g = Gridliner(self, crs, draw_labels, xlocs, ylocs,
                      color=kw.pop('color', kw.pop('c', None)),
                      linewidth=kw.pop('linewidth', kw.pop('lw', None)),
                      linestyle=kw.pop('linestyle', kw.pop('ls', '-')),
                      alpha=kw.pop('alpha', None), zorder=kw.pop('zorder', None),
                      xpadding=kw.pop('xpadding', 5), ypadding=kw.pop('ypadding', 5))
        _ignore(kw, 'gridlines', ('dms', 'x_inline', 'y_inline', 'auto_inline',
                                  'rotate_labels', 'formatter_kwargs', 'labels_bbox_style'))
        return self._add(g)

    def _outline(self):
        """Where the projection's own edge shows in the axes' box (a whole
        Robinson's oval, an Orthographic's disc): that edge, cut by the box,
        in pixels; None where the box is all map."""
        ring = self.projection.boundary
        if ring.ndim != 2 or len(ring) < 4:
            return None
        e = self.extent
        ex, ey = 1e-9 * (e[1] - e[0]), 1e-9 * (e[3] - e[2])
        cx = np.array([e[0] + ex, e[1] - ex, e[1] - ex, e[0] + ex])
        cy = np.array([e[2] + ey, e[2] + ey, e[3] - ey, e[3] - ey])
        if _inside(ring, cx, cy).all():
            return None
        cut = _clip_ring(ring, e[0], e[1], e[2], e[3])
        if len(cut) < 4:
            return None
        px, py = self.to_px(cut[:, 0], cut[:, 1])
        return np.stack([px, py], axis=1)

    def _frame(self):
        """The map's frame in pixels, as cartopy's: its outline, else the box."""
        ring = self._outline()
        if ring is not None:
            return ring
        x0, y0, x1, y1 = self.box_f
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])

    # -- layout and drawing
    def _place(self, fig, W, H):
        box = spec_box(self, fig)
        self._orig_pos = box                            # before the aspect (constrained layout)
        self._placed = None
        e = self._placed = self.extent
        ratio = (e[3] - e[2]) / (e[1] - e[0]) if e[1] != e[0] else 1.0
        if self._aspect != 'auto':                      # 'auto': the map fills the box
            a = 1.0 if self._aspect == 'equal' else float(self._aspect)
            box = _fit(box, ratio * a, self.anchor, W, H)
        self._pos = box
        ox, oy = getattr(fig, '_org', (0.0, 0.0))     # the figure's place on the canvas
        x0, x1 = box.l * W + ox, (box.l + box.w) * W + ox
        y0, y1 = (1 - box.b - box.h) * H + oy, (1 - box.b) * H + oy
        self.box_f = (x0, y0, x1, y1)
        bx0, by0 = int(round(x0)), int(round(y0))
        bx1, by1 = int(round(x1)), int(round(y1))
        self.box = (bx0, by0, bx1, by1)
        self.box_clip = self.box
        self._sx = (x1 - x0) / (e[1] - e[0])
        self._sy = (y1 - y0) / (e[3] - e[2])
        self.view = View(self.projection, e[0] + (bx0 + 0.0 - x0) / self._sx, 1.0 / self._sx,
                         e[3] - (by0 - y0) / self._sy, -1.0 / self._sy)
        self.period_px = self.projection.period * self._sx if self.projection.cylindrical else 0.0
        self.view_key = (self.projection, e, self.box_f)
        self._ll = self._ll_extent(e)

    def to_px(self, X, Y):
        e = self.extent
        x0, y0 = self.box_f[0], self.box_f[1]
        return x0 + (np.asarray(X) - e[0]) * self._sx, y0 + (e[3] - np.asarray(Y)) * self._sy

    def _draw(self, r):
        b = self.box
        if self.patch_visible:
            ring = self._outline()
            if ring is None:
                r.c.fill_rect(b[0], b[1], b[2], b[3], rgba(self.facecolor))
            else:                                       # cartopy's background: the map alone
                r.c.polygons(np.ascontiguousarray(ring), np.array([0, len(ring)], np.int64),
                             rgba(self.facecolor), self.box_clip)
        self._top_extent = None
        drawn = [a for a in self.artists if a.visible]
        if self.legend_ is not None:
            drawn.append(self.legend_)
        for a in sorted(drawn, key=lambda a: a.zorder):
            a.render(r, self)
        sp = self.spines['geo']
        if self.frame_on and sp.visible and sp.linewidth > 0:
            ring = self._frame()
            r.lines(ring, np.array([0, len(ring)]), rgba(sp.edgecolor), r.px(sp.linewidth),
                    None)
        r.extents.append(self.box_f)
        self._draw_titles(r)


