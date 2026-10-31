"""Gridded fields on a map: their grid (1-D axes sampled per pixel, or
2-D coordinates drawn by cells), pcolormesh/imshow, contourf, and contour
lines (their labels: _clabel)."""
import numpy as np

from . import _clabel, _color, _contour, _draw, _jit, _proj, _raster, _render
from ._jit import njit, prange
from ._render import Coloring
from ._style import _dashes, rc, rgba
from ._artists import Artist, _Mappable, _as_host, _axis_bounds, _gather, _lims, _on_gpu


def _rectilinear(x, y):
    """1-D axes from x, y if they are (or are a meshgrid of) 1-D axes."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim == 1 and y.ndim == 1:
        return x, y
    if x.ndim == 2 and y.ndim == 2 and x.shape == y.shape and \
            np.array_equal(x[-1], x[0]) and np.array_equal(y[:, -1], y[:, 0]) and \
            np.array_equal(x, np.broadcast_to(x[:1], x.shape)) and \
            np.array_equal(y, np.broadcast_to(y[:, :1], y.shape)):
        return x[0], y[:, 0]
    return None                                         # 2-D (a transposed meshgrid too)


def _is_lonlat_like(crs):
    return getattr(crs, 'kind', None) in (_proj.PLATE_CARREE, _proj.GEODETIC, _proj.ROTATED_POLE)


def _field(z):
    """A field as drawn: cupy as it is, numpy float32 with NaN where masked."""
    return z if _on_gpu(z) else np.ma.filled(np.ma.asarray(z, dtype=np.float32), np.nan)


def _auto_levels(levels, z):
    """Levels as given, or chosen as contour/contourf choose them (None or a count)."""
    if levels is None or np.ndim(levels) == 0:
        h = _as_host(z)
        levels = _color.auto_levels(7 if levels is None else int(levels), np.nanmin(h),
                                    np.nanmax(h))
    return np.asarray(levels, np.float64)


def _per_level(values, n):
    """One value per level, cycling a sequence as matplotlib does."""
    values = list(values)
    return [values[k % len(values)] for k in range(n)]


class _Grid:
    """A field's grid: 1-D axes (sampled per pixel) or 2-D coordinates
    (drawn by cells), in a source crs."""

    def __init__(self, x, y, shape, src, edges=False):
        self.src = src
        self.shape = shape
        self.edges = edges
        self.mesh = False                               # a pcolormesh's: a QuadMesh in Agg
        r = _rectilinear(_as_host(x), _as_host(y))
        if r is not None:
            self.xaxis = _raster.Axis(r[0], angular=_is_lonlat_like(src), edges=edges)
            self.yaxis = _raster.Axis(r[1], edges=edges)
            self.x2d = self.y2d = None
        else:
            self.xaxis = self.yaxis = None
            self.x2d = np.ascontiguousarray(_as_host(x), np.float64)
            self.y2d = np.ascontiguousarray(_as_host(y), np.float64)

    @property
    def rect(self):
        return self.xaxis is not None

    def cell_corners(self):
        """A rectilinear grid's cell corners along each axis, increasing, as
        pcolormesh makes them (shading 'nearest': half way between nodes)."""
        out = []
        for a in (self.xaxis, self.yaxis):
            c = a.coords
            if not a.edges and c.size > 1:
                d = 0.5 * np.diff(c)
                c = np.r_[c[0] - d[0], c[:-1] + d, c[-1] + d[-1]]
            out.append(c)
        return out

    def bounds_xy(self, proj):
        """The grid's extent in proj x/y (autoscaling)."""
        if self.rect:
            xs = self.xaxis.coords
            ys = self.yaxis.coords
            if not self.edges:
                xs = np.r_[xs[0] - 0.5 * self.xaxis.dx, xs, xs[-1] + 0.5 * self.xaxis.dx] \
                    if xs.size > 1 else xs
                dy = ys[1] - ys[0] if ys.size > 1 else 0.0
                ys = np.r_[ys[0] - 0.5 * dy, ys, ys[-1] + 0.5 * dy] if ys.size > 1 else ys
            n = 64
            e = 1e-7 * np.sign(xs[-1] - xs[0]) if _is_lonlat_like(self.src) else 0.0
            xe = np.linspace(xs[0] + e, xs[-1] - e, n)  # 180 E exactly is on a map's seam
            ye = np.linspace(ys[0], ys[-1], n)
            bx = np.concatenate((xe, np.full(n, xe[-1]), xe[::-1], np.full(n, xe[0])))
            by = np.concatenate((np.full(n, ys[0]), ye, np.full(n, ys[-1]), ye[::-1]))
            if self.xaxis.periodic and _is_lonlat_like(self.src):
                by = np.clip(by, -90, 90)
        else:
            x, y = self.x2d, self.y2d
            bx = np.concatenate((x[0], x[:, -1], x[-1, ::-1], x[::-1, 0]))
            by = np.concatenate((y[0], y[:, -1], y[-1, ::-1], y[::-1, 0]))
        X, Y = _proj.transform_points(self.src, proj, bx, by)
        if self.rect and self.xaxis.periodic and proj.cylindrical:      # global: every longitude
            ok = np.isfinite(Y)
            return (proj.x_limits[0], proj.x_limits[1], Y[ok].min(), Y[ok].max()) \
                if ok.any() else None
        return _lims(X, Y, proj)


class _GridArtist(Artist):
    """An artist of a field on a grid."""

    cells = False                                       # nearest cells: half a cell beyond

    def datalim(self, proj):
        return self.grid.bounds_xy(proj)

    def data_bounds(self):
        """The grid's extent, sticky (as matplotlib's QuadMesh, AxesImage
        and ContourSet make theirs)."""
        g = self.grid
        if g.src != 'data':
            return None
        ext = getattr(self, 'extent', None)             # imshow: its extent, whatever the sampling
        if ext is not None:
            x0, x1, y0, y1 = (float(v) for v in ext)
            bx, by = _axis_bounds(np.array([x0, x1])), _axis_bounds(np.array([y0, y1]))
            return bx[:3] + ((x0, x1),), by[:3] + ((y0, y1),)
        if g.rect:
            out = []
            for a in (g.xaxis, g.yaxis):
                c = a.coords
                if self.cells and not g.edges and c.size > 1:
                    d = np.diff(c)
                    c = np.r_[c[0] - 0.5 * d[0], c[:-1] + 0.5 * d, c[-1] + 0.5 * d[-1]]
                b = _axis_bounds(c)
                out.append(b[:3] + ((b[0], b[1]),))
            return tuple(out)
        x2, y2 = g.x2d, g.y2d
        if self.cells and not g.edges and min(x2.shape) > 1:
            x2, y2 = _raster._cell_corners(x2), _raster._cell_corners(y2)
        bx, by = _axis_bounds(x2), _axis_bounds(y2)
        return bx[:3] + ((bx[0], bx[1]),), by[:3] + ((by[0], by[1]),)

    def work(self):
        return int(np.prod(self.grid.shape))


class _FieldArtist(_Mappable, _GridArtist):
    """A field drawn as a value raster: sampled (or rasterized by cells) at
    `mode`, coloured by coloring()."""

    def _draw_field(self, r, ax, mode, alpha):
        z = self.z
        if not self.on_gpu or r.c.device == 'cpu':
            z = np.ascontiguousarray(_as_host(z), np.float32)
        ax._draw_field(r, self.grid, z, mode, self.coloring(), alpha)

    def get_array(self):
        return self.z


class FieldArtist(_FieldArtist):
    """pcolormesh / imshow: a field coloured by cmap + norm, nearest cells."""

    zorder = 1.0

    def __init__(self, ax, grid, z, mode, cmap=None, norm=None, vmin=None, vmax=None,
                 alpha=None, zorder=None, rounding='nearest'):
        super().__init__(ax, zorder, alpha)
        self.grid, self.mode = grid, mode
        self.cells = mode not in (_raster.BILINEAR, _raster.IMAGE)
        self.rounding = rounding                        # imshow: 'floor', as matplotlib's images
        self.cmap = _color.get_cmap(cmap)
        self._norm_in, self._vmin, self._vmax = norm, vmin, vmax
        self._norm = None
        self.set_array(z)
        if not self.on_gpu:                             # matplotlib scales the norm now: later
            self.norm_spec()                            # set_data keeps its limits

    def set_array(self, z):
        self.z = _field(z)
        self.on_gpu = _on_gpu(self.z)
        self._coloring = None
        if self.on_gpu:
            self._norm = None

    set_data = set_array

    def norm_spec(self):
        if self._norm is None:
            data = _as_host(self.z) if (self._vmin is None or self._vmax is None) else None
            self._norm = _color.make_norm(self._norm_in, vmin=self._vmin, vmax=self._vmax,
                                          data=data, N=self.cmap.N)
        return self._norm

    @property
    def norm(self):
        return self.norm_spec()

    def coloring(self):
        if self._coloring is None:
            self._coloring = Coloring(cmap=self.cmap, norm=self.norm_spec(),
                                      rounding=self.rounding)
        return self._coloring

    def render(self, r, ax):
        self._draw_field(r, ax, self.mode, self.alpha)
        edges = getattr(self, 'edges', None)
        if edges is not None:
            self._draw_edges(r, ax, *edges)

    def _draw_edges(self, r, ax, color, width, flat):
        """pcolormesh's edgecolors: every cell's outline (the grid's lines)."""
        g = self.grid
        if g.rect:
            xe, ye = g.xaxis.coords, g.yaxis.coords
            if not g.edges:
                from ._cartesian import _mid_edges
                xe, ye = _mid_edges(xe), _mid_edges(ye)
            X, Y = np.meshgrid(xe, ye)
        else:
            X, Y = g.x2d, g.y2d
        # each cell's own outline, as matplotlib's QuadMesh strokes them (a
        # shared edge drawn twice)
        rings = [(X[:-1, :-1], Y[:-1, :-1]), (X[:-1, 1:], Y[:-1, 1:]), (X[1:, 1:], Y[1:, 1:]),
                 (X[1:, :-1], Y[1:, :-1]), (X[:-1, :-1], Y[:-1, :-1])]
        xs = np.stack([a.ravel() for a, _ in rings], axis=1).ravel()
        ys = np.stack([b.ravel() for _, b in rings], axis=1).ravel()
        off = np.arange(0, xs.size + 1, 5, dtype=np.int64)
        xy, off = ax._lines_px(xs, ys, off, g.src)
        # QuadMesh is not antialiased: its edges whole pixels wide, at least one
        r.lines(xy, off, rgba(color, self.alpha), max(1.0, float(np.rint(r.px(width)))),
                ax.box_clip, snap=ax._snap)


class ContourfArtist(_FieldArtist):
    """contourf: a field's level bands, bilinear between nodes."""

    zorder = 1.0

    def __init__(self, ax, grid, z, levels, cmap=None, colors=None, norm=None, vmin=None,
                 vmax=None, extend='neither', alpha=None, zorder=None):
        super().__init__(ax, zorder, alpha)
        self.grid = grid
        self.extend = extend
        self.colors = colors
        self.cmap = None if colors is not None else _color.get_cmap(cmap)
        self._norm_in, self._vmin, self._vmax = norm, vmin, vmax
        self._levels_in = levels
        self.set_array(z)

    def set_array(self, z):
        self.z = _field(z)
        self.on_gpu = _on_gpu(self.z)
        self.levels = _auto_levels(self._levels_in, self.z)
        lut, under, over = _color.contour_colors(self.levels, cmap=self.cmap, colors=self.colors,
                                                 norm=self._norm_in, vmin=self._vmin,
                                                 vmax=self._vmax, extend=self.extend,
                                                 alpha=self.alpha)
        self.band_lut, self.under, self.over = lut, under, over
        self._coloring = Coloring(levels=self.levels, band_lut=lut, under=under, over=over)

    set_data = set_array

    def coloring(self):
        return self._coloring

    hatches = None

    def render(self, r, ax):
        holes = self._holes()
        if not holes and (not self.hatches or not any(self.hatches)):
            self._draw_field(r, ax, _raster.BILINEAR, None)  # alpha is in the band colours
            return
        c0 = r.c                                        # hatched: on the CPU (the same image),
        r.c = c0._to_cpu() if hasattr(c0, '_to_cpu') else c0   # its value raster kept
        try:
            self._draw_field(r, ax, _raster.BILINEAR, None)
            if self.hatches and any(self.hatches):
                self._hatch_bands(r)
            if holes:
                self._fill_hole_edges(r, ax)
        finally:
            r.c = c0

    def _holes(self):
        """Has the field NaNs a grid's cells stop at? (Worked out once a
        field: all the rest is for them alone.)"""
        if self.on_gpu or self.z.ndim != 2:
            return False
        if getattr(self, '_holes_of', None) is not self.z:
            self._holes_of, self._has_holes = self.z, bool(np.isnan(self.z).any())
        return self._has_holes

    def _fill_hole_edges(self, r, ax):
        """What sampling cell by cell leaves out round masked points: a cell
        with one corner masked keeps the triangle of the others (contourpy's
        corner_mask), and Agg fills, not antialiased, every pixel the
        polygons touch, so the pixels past their outline that they reach
        into. Drawn from the cells along the holes alone, over what is
        there. (2-D grids and any map: their nodes projected.)"""
        from ._tri import _touch_hull, raster_triangles
        tri, x, y, zv = self._hole_cells()
        if not tri.size:
            return
        px, py = ax._points_px(x, y, self.grid.src)
        px, py = np.asarray(px, np.float64), np.asarray(py, np.float64)
        box, main = r.c.last_values
        zt = zv[tri]
        keep = np.isfinite(px[tri] + py[tri]).all(axis=1)
        add = np.full(main.shape, np.nan, np.float32)
        raster_triangles(add, np.ascontiguousarray(px - box[0]), np.ascontiguousarray(py - box[1]),
                         np.ascontiguousarray(tri[keep]), zt[keep], False)
        gap = np.isnan(main)
        all_of = np.where(gap, add, main)
        _touch_hull(all_of, px, py, box, tri[keep])
        r.c.field_values(box, np.where(gap, all_of, np.float32(np.nan)), self.coloring(), None)

    def _hole_cells(self):
        """The triangles along the holes (node indices, rows 3), and their
        nodes' x, y and z: made once a field."""
        if getattr(self, '_hole_cells_of', (None,))[0] is self.z:
            return self._hole_cells_of[1]
        g, z = self.grid, self.z
        if g.rect:
            z = z[::-1] if g.yaxis.flip else z
            z = z[:, ::-1] if g.xaxis.flip else z
        valid = np.isfinite(z)
        count = (valid[:-1, :-1].astype(np.int8) + valid[:-1, 1:] + valid[1:, 1:] + valid[1:, :-1])
        short = count < 4
        near = short.copy()                             # whole cells next to a masked one
        near[1:] |= short[:-1]
        near[:-1] |= short[1:]
        near[:, 1:] |= short[:, :-1]
        near[:, :-1] |= short[:, 1:]
        i, j = np.nonzero(near & (count == 4))
        sw, se, ne, nw = (i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)
        ny, nx = z.shape

        def node(p):
            return p[0] * nx + p[1]
        tri = [np.c_[node(sw), node(se), node(ne)], np.c_[node(sw), node(ne), node(nw)]]
        i, j = np.nonzero(count == 3)
        corners = np.stack([np.c_[i, j], np.c_[i, j + 1], np.c_[i + 1, j + 1], np.c_[i + 1, j]], 1)
        ok = valid[corners[..., 0], corners[..., 1]]
        ids = (corners[..., 0] * nx + corners[..., 1])[ok].reshape(-1, 3)
        tri.append(ids)
        tri = np.concatenate(tri).astype(np.int64)
        used, tri = np.unique(tri, return_inverse=True)
        tri = tri.reshape(-1, 3)
        ri, rj = used // nx, used % nx
        if g.rect:
            x, y = g.xaxis.coords[rj], g.yaxis.coords[ri]
        else:
            x, y = g.x2d[ri, rj], g.y2d[ri, rj]
        cells = (tri, x, y, np.ascontiguousarray(z[ri, rj], np.float64))
        self._hole_cells_of = (self.z, cells)
        return cells

    def _hatch_bands(self, r):
        """matplotlib's contourf hatches: cycled over the filled regions,
        the extended bands first and last (under, bands, over)."""
        from . import _hatch
        box, values = r.c.last_values
        lv = self.levels
        with np.errstate(invalid='ignore'):
            k = np.searchsorted(lv, values, side='left') - 1   # levels[k] < z <= levels[k+1]
            k[values == lv[0]] = 0
            k[~np.isfinite(values)] = -9
        regions = []
        if self.extend in ('min', 'both'):
            regions.append(-1)
        regions += list(range(len(lv) - 1))
        if self.extend in ('max', 'both'):
            regions.append(len(lv) - 1)
        color = _hatch.color_for(None, None, self.alpha, collection=True)
        for i, band in enumerate(regions):
            pattern = self.hatches[i % len(self.hatches)]
            if not pattern:
                continue
            if band == -1:
                mask = values < lv[0]
            elif band == len(lv) - 1:
                mask = values > lv[-1]
            else:
                mask = k == band
            if mask.any():
                _hatch.fill_mask(r.c.a, mask.astype(np.float32), box[0], box[1], pattern,
                                 color, r.dpi)


class ContourSet(_GridArtist):
    """contour: lines at levels, with optional labels (clabel)."""

    zorder = 2.0

    def __init__(self, ax, grid, z, levels, colors=None, cmap=None, norm=None, vmin=None,
                 vmax=None, linewidths=None, linestyles=None, negative_linestyles='dashed',
                 alpha=None, zorder=None):
        super().__init__(ax, zorder, alpha)
        self.grid = grid
        self.z = _as_host(z)
        self.levels = _auto_levels(levels, self.z)
        L = self.levels.size
        if colors is not None:
            one = isinstance(colors, str) or (np.ndim(colors) == 1 and len(colors) in (3, 4)
                                              and not isinstance(colors[0], str))
            cl = [colors] if one else list(colors)
            self.monochrome = len(cl) == 1
            self.line_colors = np.array([rgba(c, alpha) for c in _per_level(cl, L)], np.uint8)
        else:
            self.monochrome = False
            cm = _color.get_cmap(cmap)
            nrm = _color.make_norm(norm, vmin=vmin, vmax=vmax, data=self.levels, N=cm.N)
            out = np.empty((1, L, 4), np.uint8)
            _color.shade(self.levels[None, :].astype(np.float32), cm, nrm, out=out, alpha=alpha,
                         rounding='nearest')
            self.line_colors = out[0]
        lw = (rc('contour.linewidth') or rc('lines.linewidth')) if linewidths is None \
            else linewidths
        self.linewidths = np.array(_per_level(np.ravel(lw), L), float)
        if linestyles is None:
            ls = [negative_linestyles if (self.monochrome and v < 0) else 'solid'
                  for v in self.levels]
        elif isinstance(linestyles, (str, tuple)):
            ls = [linestyles] * L
        else:
            ls = _per_level(linestyles, L)
        self.linestyles = ls
        self.labels = None
        self._lines = None
        self._cache = None

    def _grid_lines(self):
        """Polylines in the grid's coordinates (made once)."""
        if self._lines is None:
            z = self.z
            g = self.grid
            if g.rect:
                xc, yc = g.xaxis.coords, g.yaxis.coords
                if g.xaxis.periodic:                    # join the seam
                    xc = np.r_[xc, xc[0] + 360.0]
                z = _contour.field64(z, g.yaxis.flip, g.xaxis.flip)
                x, y, off, lof = _contour.lines(z, self.levels, periodic=g.xaxis.periodic,
                                                cols=xc, rows=yc)
                self._open = _contour.open_rings(z, self.levels, x, y, off, lof, xc, yc)
            else:
                fi, fj, off, lof = _contour.lines(z, self.levels)
                x = _bilinear(g.x2d, fi, fj)
                y = _bilinear(g.y2d, fi, fj)
                self._open = np.zeros(off.size - 1, bool)
            self._lines = (x, y, off, lof)
        return self._lines

    def _segs(self):
        """Per level, its lines as (vertices, codes) in the grid's
        coordinates, in the order and direction matplotlib's contour
        generator gives them; a closed line ends on its first point, coded
        CLOSEPOLY."""
        from ._path import CLOSEPOLY, LINETO, MOVETO
        x, y, off, lof = self._grid_lines()
        out = [[] for _ in range(self.levels.size)]
        for k in range(off.size - 1):
            a, b = int(off[k]), int(off[k + 1])
            v = np.c_[x[a:b], y[a:b]]
            c = np.full(b - a, LINETO, np.uint8)
            c[0] = MOVETO
            if b - a > 2 and (v[0] == v[-1]).all() and not self._open[k]:
                c[-1] = CLOSEPOLY
            out[int(lof[k])].append((v, c))
        return out

    def get_paths(self):
        """matplotlib's: one Path a level, its lines joined (empty: none)."""
        from ._path import Path
        return [Path(np.concatenate([v for v, _ in lv]), np.concatenate([c for _, c in lv]))
                if lv else Path(np.empty((0, 2))) for lv in self._segs()]

    # a level without lines: one empty segment (matplotlib's empty Path)
    @property
    def allsegs(self):
        return [[v for v, _ in lv] or [np.empty((0, 2))] for lv in self._segs()]

    @property
    def allkinds(self):
        return [[c for _, c in lv] or [None] for lv in self._segs()]

    def render(self, r, ax):
        key = (ax.view_key, r.dpi, None if self.labels is None else self.labels['key'])
        if self._cache is None or self._cache[0] != key:
            x, y, off, lof = self._grid_lines()
            xy, poff, which = ax._lines_px(x, y, off, self.grid.src, parts=True)
            lof = lof[which]                            # copies across the seam keep their level
            if self._open.any() and poff.size == off.size:
                xy = xy.copy()                          # a ring left open is not closed in px
                xy[poff[1:][self._open] - 1, 0] += 1e-9
            texts = []
            if self.labels is not None:
                xy, poff, lof, texts = _clabel.place(r, ax, self, xy, poff, lof)
            if ax._snap is not False:
                xy = _snap_levels(xy, poff, lof, r.px(self.linewidths))
            segs = _draw.segments(xy, poff, lof, lof)   # a level is one path, as in matplotlib
            self._cache = (key, segs, texts, {})
        _, segs, texts, memo = self._cache
        dashes = []
        for k, ls in enumerate(self.linestyles):
            d = _dashes(ls, self.linewidths[k])
            dashes.append(() if d is None else tuple(r.px(v) for v in d))
        styles = _draw.Styles(self.line_colors, r.px(self.linewidths), dashes)
        r.c.segments(segs, styles, ax.box_clip, memo)
        for args, kw in texts:
            r.text(*args, **kw)


def _snap_levels(xy, poff, lof, widths):
    """Agg's 'auto' snapping of contour lines: a level is one path, snapped
    if it has at most 1024 vertices and every segment is rectilinear."""
    per_level = np.bincount(lof, np.diff(poff), minlength=widths.size)
    small = np.flatnonzero(per_level[lof] <= 1024)
    if not small.size:
        return xy
    at = _gather(poff, small)
    off = np.concatenate(([0], np.cumsum(np.diff(poff)[small])))
    straight = np.ones(widths.size, bool)
    straight[lof[small]] = True
    np.logical_and.at(straight, lof[small], _draw.rectilinear(xy[at], off))
    keep = np.flatnonzero(straight[lof[small]])
    if not keep.size:
        return xy
    at = _gather(poff, small[keep])
    off = np.r_[0, np.cumsum(np.diff(poff)[small[keep]])]
    xy = xy.copy()
    xy[at] = _draw.snap(xy[at], off, widths[lof[small[keep]]], True)
    return xy


@njit(cache=True, nogil=True, parallel=True)
def _bilinear_into(a, fi, fj, out):
    ny, nx = a.shape
    for k in prange(fi.size):
        i0 = min(max(np.int64(np.floor(fi[k])), 0), ny - 2)
        j0 = min(max(np.int64(np.floor(fj[k])), 0), nx - 2)
        ti, tj = fi[k] - i0, fj[k] - j0
        out[k] = ((a[i0, j0] * (1 - tj) + a[i0, j0 + 1] * tj) * (1 - ti)
                  + (a[i0 + 1, j0] * (1 - tj) + a[i0 + 1, j0 + 1] * tj) * ti)


def _bilinear(a, fi, fj):
    out = np.empty(fi.shape, np.float64)
    _bilinear_into(a, np.ascontiguousarray(fi), np.ascontiguousarray(fj), out)
    return out


class RGBImage(Artist):
    """imshow of an RGB(A) array (float 0..1 or uint8): each canvas pixel
    takes the image pixel under its centre (matplotlib's nearest sampling
    when an image is drawn larger than it is). On plain axes through the
    axes' scales; on a map through the projection, sampled as a scalar
    imshow's field is (its grid). Row 0 is at the top for origin 'upper'."""

    zorder = 0.0

    def __init__(self, ax, Z, extent, origin, alpha=None, zorder=None, grid=None):
        super().__init__(ax, zorder, alpha)
        self.grid = grid
        Z = np.asarray(Z)
        if Z.shape[2] == 3 and Z.dtype == np.uint8:
            Z = np.concatenate([Z, np.full(Z.shape[:2] + (1,), 255, Z.dtype)], axis=2)
        if Z.dtype != np.uint8:                         # matplotlib's: clipped, * 255, floored
            out = np.empty(Z.shape[:2] + (4,), np.uint8)    # (RGB: alpha 255)
            with _jit.threads():
                _render.rgba_bytes(np.ascontiguousarray(Z, np.float64), out)
            Z = out
        self.rgba = np.ascontiguousarray(Z)
        self.extent = tuple(float(v) for v in extent)
        self.origin = origin

    def data_bounds(self):
        x0, x1, y0, y1 = self.extent
        return (_axis_bounds(np.array([x0, x1]), (x0, x1)),
                _axis_bounds(np.array([y0, y1]), (y0, y1)))

    def datalim(self, proj):
        return None if self.grid is None else self.grid.bounds_xy(proj)

    def work(self):
        return int(np.prod(self.rgba.shape[:2]))

    def _render_map(self, r, ax):
        """The image's red, green and blue bytes packed in one float32
        (24 bits: exact), sampled nearest like a field; its alpha a second
        field when it has any below 255. Off the image: NaN, nothing drawn."""
        bx0, by0, bx1, by1 = ax.box
        h, w = by1 - by0, bx1 - bx0
        if h <= 0 or w <= 0:
            return
        g = self.grid
        if not hasattr(self, '_packed'):
            c = self.rgba.astype(np.int32)
            self._packed = ((c[..., 0] << 16) | (c[..., 1] << 8) | c[..., 2]).astype(np.float32)
            a = self.rgba[..., 3]
            self._alpha = None if (a == 255).all() else a.astype(np.float32)
        v = np.empty((h, w), np.float32)
        _raster.sample_rect(v, ax.view, g.src, g.xaxis, g.yaxis, self._packed, _raster.NEAREST)
        out = np.zeros((h, w, 4), np.uint8)
        ok = np.isfinite(v)
        k = v[ok].astype(np.int32)
        out[ok, 0], out[ok, 1], out[ok, 2] = k >> 16, (k >> 8) & 255, k & 255
        if self._alpha is None:
            out[ok, 3] = 255
        else:
            _raster.sample_rect(v, ax.view, g.src, g.xaxis, g.yaxis, self._alpha, _raster.NEAREST)
            out[ok, 3] = v[ok].astype(np.uint8)
        r.c.image(out, bx0, by0, 1.0 if self.alpha is None else self.alpha)

    def render(self, r, ax):
        if self.grid is not None and hasattr(ax, 'view'):
            return self._render_map(r, ax)
        if not hasattr(ax, '_xf') or getattr(ax, '_skew', 0):
            raise NotImplementedError('RGB(A) imshow is drawn on plain, unskewed axes and maps')
        bx0, by0, bx1, by1 = ax.box_clip
        if bx1 <= bx0 or by1 <= by0:
            return
        x0f, y0f, x1f, y1f = ax.box_f
        p = ax._xf
        u = (np.arange(bx0, bx1) + 0.5 - x0f) / (x1f - x0f)
        v = (y1f - (np.arange(by0, by1) + 0.5)) / (y1f - y0f)
        xs = ax.xaxis.scale.inverse(p[0] + u / p[1]) if p[1] else np.full(u.shape, np.nan)
        ys = ax.yaxis.scale.inverse(p[2] + v / p[3]) if p[3] else np.full(v.shape, np.nan)
        ny, nx = self.rgba.shape[:2]
        x0, x1, y0, y1 = self.extent
        ytop, ybot = (y1, y0) if self.origin == 'upper' else (y0, y1)
        with np.errstate(invalid='ignore'):
            col = np.floor((xs - x0) / (x1 - x0) * nx)
            row = np.floor((ys - ytop) / (ybot - ytop) * ny)
        col = np.where((col >= 0) & (col < nx), col, -1).astype(np.int64)
        row = np.where((row >= 0) & (row < ny), row, -1).astype(np.int64)
        r.c.image_nearest(self.rgba, row, col, bx0, by0, 1.0 if self.alpha is None else
                          self.alpha)
