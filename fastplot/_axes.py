"""cartopy's GeoAxes: a map's projection and extent, the map-only
methods (coastlines, features, gridlines, set_extent), and the answers to
AxesBase's hooks through the projection: points, lines (seam-safe on
cylindrical maps), vectors and fields."""
import math

import numpy as np

from . import _draw, _kept, _proj, _raster
from . import feature as _feature
from ._jit import njit
from . import transforms as _transforms
from ._render import View
from ._style import _ignore, rc, rgba
from ._layout import SubplotSpec, _fit, fig_aspect, spec_box
from ._artists import FeatureArtist, _project_lines, _wrap_x, cached_geometry
from ._axesbase import AxesBase
from ._gridliner import Gridliner


_FRAME_KEPT = True              # the map's frame drawn as kept geometry (False: r.lines, as before)
_LL = {}                        # (map, crs, extent) -> lon/lat extent: the next figure's
_OUTLINES = {}                  # (map, extent, box) -> its outline in px, read only, or None


def _inside(ring, x, y):
    """Whether points (x, y) are inside a closed ring (even-odd)."""
    x0, y0 = ring[:-1, 0][:, None], ring[:-1, 1][:, None]
    x1, y1 = ring[1:, 0][:, None], ring[1:, 1][:, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        hit = ((y0 > y) != (y1 > y)) & (x < x0 + (y - y0) * (x1 - x0) / (y1 - y0))
    return (hit.sum(axis=0) % 2) == 1


def _clip_ring(ring, x0, x1, y0, y1):
    """A ring clipped to a rectangle (Sutherland-Hodgman), closed."""
    pts = _clip_ring_open(np.ascontiguousarray(ring[:-1], np.float64), float(x0), float(x1), float(y0),
                          float(y1))
    return np.vstack([pts, pts[:1]]) if len(pts) else pts


@njit(cache=True, nogil=True)
def _clip_ring_open(pts, x0, x1, y0, y1):
    """_clip_ring's edges, the ring open: against x >= x0, x <= x1, y >= y0,
    y <= y1 in turn, each vertex kept where inside and a crossing put in
    where an edge goes in or out (a + t (b - a), as numpy worked it out
    here before: a projection's boundary is thousands of points)."""
    cur = pts
    m = pts.shape[0]
    for side in range(4):
        if m == 0:
            break
        axis = 0 if side < 2 else 1
        lim = x0 if side == 0 else (x1 if side == 1 else (y0 if side == 2 else y1))
        above = side % 2 == 0
        out = np.empty((2 * m, 2))
        q = 0
        for k in range(m):
            k2 = k + 1 if k + 1 < m else 0
            va, vb = cur[k, axis], cur[k2, axis]
            ia = va >= lim if above else va <= lim
            ib = vb >= lim if above else vb <= lim
            if ia:
                out[q, 0], out[q, 1] = cur[k, 0], cur[k, 1]
                q += 1
            if ia != ib:
                t = (lim - va) / (vb - va)
                out[q, 0] = cur[k, 0] + t * (cur[k2, 0] - cur[k, 0])
                out[q, 1] = cur[k, 1] + t * (cur[k2, 1] - cur[k, 1])
                q += 1
        cur = out[:q]
        m = q
    return cur


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

    def _traced_shape(self, v, codes, src):
        """A patch's path as cartopy projects a shape: its straight sides
        bent as project_linear bends them (_proj.trace_lines)."""
        P = self.projection
        if not isinstance(src, _proj.Projection) or src == P or \
                (codes is not None and ((codes == 3).any() or (codes == 4).any())):
            return v, codes                                 # (not in a crs, or curved)
        n = v.shape[0]
        c = np.full(n, 2, np.uint8) if codes is None else codes
        if codes is None:
            c[0] = 1
        starts = np.flatnonzero(c == 1)
        x, y = v[:, 0].astype(np.float64), v[:, 1].astype(np.float64)
        shut = np.flatnonzero(c == 79)
        if shut.size:                                   # a CLOSEPOLY's vertex means nothing
            first = starts[np.searchsorted(starts, shut, side='right') - 1]
            x, y = x.copy(), y.copy()
            x[shut], y[shut] = x[first], y[first]
        offs = np.r_[starts, n].astype(np.int64)
        tx, ty, to = _proj.trace_source(src, P, x, y, offs)
        if to is offs:
            return v, codes
        out = np.full(tx.size, 2, np.uint8)
        out[to[:-1]] = 1
        closed = np.isin(offs[1:] - 1, shut)
        out[to[1:][closed] - 1] = 79
        return np.column_stack([tx, ty]), out

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

    def _leaning_mesh(self, grid, mode):
        """A pcolormesh on rectilinear axes whose cells lean on this map: its
        cell corners, drawn by cells (Agg's rounded corners, every touched
        pixel) where sampling pixel centres would give another picture. Not
        a whole circle of longitude, which sampling joins up over the seam."""
        if not grid.mesh or mode != _raster.NEAREST or grid.xaxis.periodic or \
                self.period_px > 0.0 or min(grid.shape) < 2 or \
                grid.shape[0] * grid.shape[1] >= 1 << 24:
            return None
        cx, cy = grid.cell_corners()
        if not _raster.axes_lean(self.view, grid.src, cx, cy):
            return None
        # in the array's own order: later cells are painted over earlier ones
        return cx[::-1] if grid.xaxis.flip else cx, cy[::-1] if grid.yaxis.flip else cy

    def _mesh_nodes(self, grid, cx, cy):
        """The corners in the raster's pixels, kept for the next frame."""
        def nodes():
            return _raster.forward_nodes(self.view, grid.src, *np.meshgrid(cx, cy))
        return cached_geometry(('mesh-nodes', grid.src, self.view_key, cx.tobytes(), cy.tobytes()),
                               nodes)

    def _mesh_tables(self, grid, mode):
        """A pcolormesh's column and row tables on a map whose x follows
        longitude alone and y latitude (PlateCarree, Mercator, Miller): its
        cell edges projected and taken to the canvas as cartopy's QuadMesh
        takes them -- transData's matrix made as matplotlib's BboxTransformFrom
        and BboxTransformTo make theirs and np.dot joins them, applied as
        matplotlib's affine_transform does, then Agg's y flip -- so a pixel
        whose centre is on an edge goes to the cell before it, as Agg fills
        it. None where cartopy draws the mesh otherwise (across the seam,
        beyond the projection's x range, a whole circle) or an edge has no
        place."""
        if not grid.mesh or mode != _raster.NEAREST or grid.xaxis.periodic or \
                self.view.kind not in _raster._SEP_MAP or grid.src.kind not in _raster._SEP_SRC:
            return None
        src = grid.src
        # cartopy's _wrap_args: a wrappable projection's centres made edges
        # with its x range as the wrap; a Geodetic's left to matplotlib
        wrap = abs(src.x_limits[1] - src.x_limits[0]) if src.kind == _proj.PLATE_CARREE else 0.0
        got = grid.mesh_corners(wrap)
        if got is None:
            return None
        cx, cy = got
        P = self.projection
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            X, _ = _proj.transform_points(grid.src, P, cx, np.zeros_like(cx))
            _, Y = _proj.transform_points(grid.src, P, np.full(cy.shape, 0.5 * (cx[0] + cx[-1])),
                                          cy)
            X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
            xl = P.x_limits
            if not (np.isfinite(X).all() and np.isfinite(Y).all()) or \
                    X.min() < min(xl) or X.max() > max(xl):
                return None
            W, H = self.figure._px_size
            ox, oy = getattr(self.figure, '_org', (0.0, 0.0))
            box, e = self._pos, self.extent
            x0d, x1d = box.x0 * W, box.x1 * W              # the axes' display box
            y0d, y1d = box.y0 * H, box.y1 * H
            to = np.array([[x1d - x0d, 0.0, x0d], [0.0, y1d - y0d, y0d], [0.0, 0.0, 1.0]], float)
            sx, sy = 1.0 / (e[1] - e[0]), 1.0 / (e[3] - e[2])
            frm = np.array([[sx, 0.0, -e[0] * sx], [0.0, sy, -e[2] * sy], [0.0, 0.0, 1.0]],
                           float)
            m = np.dot(to, frm)
            px = X * m[0, 0] + m[0, 2]
            py = H - (Y * m[1, 1] + m[1, 2])
        if ox:
            px = px + ox
        if oy:
            py = py + oy
        bx0, by0, bx1, by1 = self.box
        cols = _raster.mesh_table(px, bx1 - bx0, bx0 + 0.5)
        rows = _raster.mesh_table(py, by1 - by0, by0 + 0.5)
        return None if cols is None or rows is None else (cols, rows)

    def _draw_field(self, r, grid, z, mode, coloring, alpha):
        if grid.rect:
            corners = self._leaning_mesh(grid, mode)
            if corners is not None:
                sx, sy = self._mesh_nodes(grid, *corners)
                r.c.field_cells_px(self.box, sx, sy, z, _raster.FLAT, coloring, alpha)
                return
            tables = self._mesh_tables(grid, mode)
            if tables is not None:
                r.c.field_sep(self.box, tables[0], tables[1], z, grid, mode, coloring, alpha)
                return
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
        """_ll_extent_of, kept across figures of the same map."""
        try:
            key = (_proj.value_key(self.projection), None if crs is None else
                   _proj.value_key(crs), np.array(e, np.float64).tobytes(),
                   tuple(map(type, e)))
        except (AttributeError, TypeError, ValueError):
            return self._ll_extent_of(e, crs)
        got = _LL.get(key)
        if got is None:
            got = self._ll_extent_of(e, crs)
            if len(_LL) >= 256:
                _LL.clear()
            _LL[key] = got
        return got

    def _ll_extent_of(self, e, crs=None):
        """lon/lat bounding box of the map (cartopy's _axes_domain)."""
        crs = crs or _proj.plate_carree()
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
                      xpadding=kw.pop('xpadding', 5), ypadding=kw.pop('ypadding', 5),
                      x_inline=kw.pop('x_inline', None), y_inline=kw.pop('y_inline', None),
                      auto_inline=kw.pop('auto_inline', True),
                      rotate_labels=kw.pop('rotate_labels', None),
                      offset_angle=kw.pop('offset_angle', 25))
        _ignore(kw, 'gridlines', ('dms', 'formatter_kwargs', 'labels_bbox_style'))
        return self._add(g)

    # -- ticks: cartopy's GeoAxes keeps matplotlib's x and y axes, hidden
    # until set_xticks / set_yticks, in the projection's coordinates. Here a
    # plain Axes on the map's box (its twin: placed where the map is, its
    # limits the extent) carries them; the map draws its axis lines at the
    # axis' zorder among its artists, and its tick labels after them.
    _tick_ax = None

    def _ticker(self):
        t = self._tick_ax
        if t is None:
            from ._cartesian import Axes
            from . import _ticker
            t = self._tick_ax = Axes(self.figure, self.spec)
            t._twin_of = self
            t.patch_visible = False
            for s in t.spines.values():
                s.set_visible(False)
            for a in (t.xaxis, t.yaxis):
                a.visible = False                       # (cartopy: until ticks are set)
                a.set_major_locator(_ticker.NullLocator())
        return t

    @property
    def xaxis(self):
        return self._ticker().xaxis

    @property
    def yaxis(self):
        return self._ticker().yaxis

    def _tick_values(self, ticks, crs, which):
        """Ticks in crs (None: the projection's own x or y) as projection x or
        y. As cartopy: from one rectangular coordinate system to another
        only, y = 0 for the x ticks and x = 0 for the y ticks."""
        t = np.asarray(ticks, np.float64).ravel()
        if crs is None or crs == self.projection or not t.size:
            return t
        if not (getattr(crs, 'cylindrical', False) and self.projection.cylindrical):
            raise RuntimeError('Cannot handle non-rectangular coordinate systems.')
        zero = np.zeros(t.size)
        X, Y = _proj.transform_points(crs, self.projection, *((t, zero) if which == 'x'
                                                               else (zero, t)))
        return np.asarray(X if which == 'x' else Y, np.float64)

    def set_xticks(self, ticks, labels=None, *, minor=False, crs=None, **kw):
        t = self._ticker()
        t.xaxis.visible = True
        return t.set_xticks(self._tick_values(ticks, crs, 'x'), labels, minor=minor, **kw)

    def set_yticks(self, ticks, labels=None, *, minor=False, crs=None, **kw):
        t = self._ticker()
        t.yaxis.visible = True
        return t.set_yticks(self._tick_values(ticks, crs, 'y'), labels, minor=minor, **kw)

    def get_xticks(self, minor=False):
        return self._ticker().get_xticks(minor)

    def get_yticks(self, minor=False):
        return self._ticker().get_yticks(minor)

    def set_xticklabels(self, labels, *, minor=False, **kw):
        self._ticker().set_xticklabels(labels, minor=minor, **kw)

    def set_yticklabels(self, labels, *, minor=False, **kw):
        self._ticker().set_yticklabels(labels, minor=minor, **kw)

    def tick_params(self, axis='both', **kw):
        self._ticker().tick_params(axis, **kw)

    def grid(self, visible=None, which='major', axis='both', **kw):
        """matplotlib's grid: lines at the ticks (once set), in the
        projection's x and y."""
        self._ticker().grid(visible, which, axis, **kw)

    def _place_ticks(self, fig, W, H):
        t = self._tick_ax
        if t is None:
            return
        e = self._placed
        t._set_lim('x', (e[0], e[1]))
        t._set_lim('y', (e[2], e[3]))
        t._place(fig, W, H)
        t._ticks_cache = {}

    def _outline(self):
        """_outline_of, kept across figures of the same map in the same box
        (a fresh copy of it each time)."""
        e, b = self.extent, self.box_f
        try:
            key = (_proj.value_key(self.projection), np.array(e, np.float64).tobytes(),
                   tuple(map(type, e)), np.array(b, np.float64).tobytes(), tuple(map(type, b)),
                   self._sx, self._sy)
        except (AttributeError, TypeError, ValueError):
            return self._outline_of()
        if key in _OUTLINES:
            got = _OUTLINES[key]
        else:
            got = self._outline_of()
            if len(_OUTLINES) >= 64:
                _OUTLINES.clear()
            _OUTLINES[key] = got
        return None if got is None else got.copy()

    def _outline_of(self):
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

    def _on_map(self, X, Y):
        """Which points (projected) are inside the projection's own edge."""
        return _inside(self.projection.boundary, X, Y)

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
            box = _fit(box, ratio * a, self.anchor, W, H, fig_aspect(fig))
        self._pos = box
        ox, oy = getattr(fig, '_org', (0.0, 0.0))     # the figure's place on the canvas
        x0, x1 = box.l * W + ox, (box.l + box.w) * W + ox
        y0, y1 = (1 - box.b - box.h) * H + oy, (1 - box.b) * H + oy
        self.box_f = (x0, y0, x1, y1)
        # Agg's clip box (and plain axes' box): half a pixel rounds up
        bx0, by0 = math.floor(x0 + 0.5), math.floor(y0 + 0.5)
        bx1, by1 = math.floor(x1 + 0.5), math.floor(y1 + 0.5)
        self.box = (bx0, by0, bx1, by1)
        self.box_clip = self.box
        self._sx = (x1 - x0) / (e[1] - e[0])
        self._sy = (y1 - y0) / (e[3] - e[2])
        self.view = View(self.projection, e[0] + (bx0 + 0.0 - x0) / self._sx, 1.0 / self._sx,
                         e[3] - (by0 - y0) / self._sy, -1.0 / self._sy)
        self.period_px = self.projection.period * self._sx if self.projection.cylindrical else 0.0
        self.view_key = (self.projection, e, self.box_f)
        self._ll = self._ll_extent(e)
        self._place_ticks(fig, W, H)

    def to_px(self, X, Y):
        e = self.extent
        x0, y0 = self.box_f[0], self.box_f[1]
        return x0 + (np.asarray(X) - e[0]) * self._sx, y0 + (e[3] - np.asarray(Y)) * self._sy

    def _draw(self, r):
        b = self.box
        fresh, r.fresh = r.fresh, False
        if self.patch_visible:
            ring = self._outline()
            if ring is None:
                face = rgba(self.facecolor)
                if not (fresh and r.same_as_bg(face)):  # (else it would only paint the canvas'
                    r.c.fill_rect(b[0], b[1], b[2], b[3], face)     # own colour over itself)
            else:                                       # cartopy's background: the map alone
                r.c.polygons(np.ascontiguousarray(ring), np.array([0, len(ring)], np.int64),
                             rgba(self.facecolor), self.box_clip)
        self._top_extent = None
        drawn = [a for a in self.artists if a.visible]
        if self.legend_ is not None:
            drawn.append(self.legend_)
        t = self._tick_ax
        if t is not None and (t.xaxis.visible or t.yaxis.visible):
            from ._cartesian import _AXISBELOW
            azo = getattr(t, '_axis_zorder', None)
            drawn.append(_TickLines(t, _AXISBELOW.get(rc('axes.axisbelow'), 1.5)
                                    if azo is None else azo))
        else:
            t = None
        for a in sorted(drawn, key=lambda a: a.zorder):
            a.render(r, self)
        sp = self.spines['geo']
        if self.frame_on and sp.visible and sp.linewidth > 0 and not _FRAME_KEPT:
            ring = self._frame()
            lw = r.px(sp.linewidth)
            offs = np.array([0, len(ring)])
            r.lines(_draw.snap(ring, offs, lw), offs, rgba(sp.edgecolor), lw, None)  # a box, snapped
        elif self.frame_on and sp.visible and sp.linewidth > 0 and not r.measuring:
            lw = r.px(sp.linewidth)

            def frame():                                # a box (or the map's outline), snapped
                ring = self._frame()
                offs = np.array([0, len(ring)], np.int64)
                return _draw.segments(np.ascontiguousarray(_draw.snap(ring, offs, lw),
                                                           np.float64), offs), {}
            # as r.lines draws it, its outline kept with it (the next figure of the map blends
            # its ink: a box's sides are rows the whole map wide to sweep); the frame follows
            # from the view (projection, extent, box) and the width
            segs, memo = cached_geometry(('map-frame', self.view_key, _kept.freeze(lw)), frame)
            if segs.shape[0]:
                r.c.segments(segs, _kept.styles(rgba(sp.edgecolor), lw), None, memo)
        if t is not None:
            t._draw_tick_labels(r)
        r.extents.append(self.box_f)
        self._draw_titles(r)


class _TickLines:
    """A map's tick marks and grid lines (its ticker's), drawn at the axis'
    zorder among the map's artists."""

    visible = True

    def __init__(self, ticker, zorder):
        self.ticker, self.zorder = ticker, zorder

    def render(self, r, ax):
        self.ticker._draw_axes_lines(r)


