"""matplotlib.patches: shapes added to axes.

    import fastplot.patches as mpatches
    ax.add_patch(mpatches.Rectangle((0, 0), 2, 1, facecolor='none', edgecolor='r'))
    ax.add_patch(mpatches.Circle((5, 5), 1, alpha=0.4))
    h = mpatches.Patch(color='C1', label='region')     # a legend handle

The geometry is matplotlib's (_patches: the same vertices and Bezier
codes), in the patch's coordinates (data unless transform says
otherwise), mapped to the canvas and flattened there; filled nonzero,
then stroked -- snapped and butt-capped on plain axes, as Agg draws.
"""
import numpy as np

from . import _draw, _patches, _path
from ._artists import Artist, _axis_bounds
from ._style import _dashes, rc, rgba

__all__ = ['Patch', 'Rectangle', 'Circle', 'Ellipse', 'Polygon', 'Wedge', 'Arc',
           'RegularPolygon', 'FancyBboxPatch', 'FancyArrowPatch']


class Patch(Artist):
    """matplotlib's Patch: face and edge, no geometry of its own (a
    legend handle) unless a subclass gives one."""

    zorder = 1.0
    clip = True                                         # to the axes box
    requests_autoscale = False                          # add_patch does not autoscale
    _edge_default = False                               # filled, without an edge unless given one

    def __init__(self, *, facecolor=None, edgecolor=None, color=None, linewidth=None,
                 linestyle='solid', alpha=None, fill=True, zorder=None, label=None,
                 transform=None, hatch=None, visible=True, **kw):
        kw.pop('antialiased', None)
        kw.pop('aa', None)
        if kw.pop('clip_on', True) is False:
            self.clip = False
        self.capstyle = kw.pop('capstyle', 'butt')
        self.joinstyle = kw.pop('joinstyle', 'miter')
        for k in ('snap', 'rasterized', 'gid', 'url', 'picker', 'path_effects', 'sketch_params',
                  'animated', 'in_layout'):
            kw.pop(k, None)
        fc = kw.pop('fc', facecolor)
        ec = kw.pop('ec', edgecolor)
        lw = kw.pop('lw', linewidth)
        linestyle = kw.pop('ls', linestyle)
        if kw:
            raise TypeError('%s: unsupported keyword(s): %s' % (type(self).__name__,
                                                                ', '.join(sorted(kw))))
        super().__init__(None, zorder, alpha, visible)
        if color is not None:
            fc = color if fc is None else fc
            ec = color if ec is None else ec
        self.facecolor = rc('patch.facecolor') if fc is None else fc
        self.edgecolor = (rc('patch.edgecolor') if rc('patch.force_edgecolor') or not fill or
                          self._edge_default else 'none') if ec is None else ec
        self.fill = fill
        self.linewidth = rc('patch.linewidth') if lw is None else lw
        self.linestyle = linestyle
        self.label = label
        self.transform = transform
        self.hatch = hatch

    # matplotlib's setters
    def set_transform(self, t):
        self.transform = t
        if self.axes is not None:                       # already added: its coordinates anew
            self.src = self.axes._src(t)

    def get_transform(self):
        return self.transform

    def set_hatch(self, h):
        self.hatch = h

    def set_facecolor(self, c):
        self.facecolor = c

    def set_edgecolor(self, c):
        self.edgecolor = c

    def set_color(self, c):
        self.facecolor = self.edgecolor = c

    def set_linewidth(self, w):
        self.linewidth = w

    def set_linestyle(self, ls):
        self.linestyle = ls

    def set_alpha(self, a):
        self.alpha = a

    def set_clip_on(self, b):
        self.clip = bool(b)

    def get_facecolor(self):
        return tuple(v / 255.0 for v in self._colors()[0])

    def get_edgecolor(self):
        return tuple(v / 255.0 for v in self._colors()[1])

    def get_linewidth(self):
        return self.linewidth

    def set_fill(self, b):
        self.fill = bool(b)

    def set_label(self, s):
        self.label = s

    def get_label(self):
        return self.label

    def get_path(self):
        """matplotlib's get_path() under get_patch_transform(): a _path.Path
        in the patch's coordinates (None: no geometry)."""
        return None

    def _colors(self):
        fc = rgba(self.facecolor, self.alpha) if self.fill else np.zeros(4, np.uint8)
        ec = rgba(self.edgecolor, self.alpha)
        return fc, ec

    def legend_style(self):
        fc, ec = self._colors()
        return dict(kind='patch', facecolor=fc, edgecolor=ec, linewidth=self.linewidth,
                    linestyle=self.linestyle)

    def data_bounds(self):
        p = self.get_path()
        if p is None or self.src != 'data' or not len(p.vertices):
            return None
        v = _path.curve_extrema(p.vertices, p.codes)
        return _axis_bounds(v[:, 0]), _axis_bounds(v[:, 1])

    def render(self, r, ax):
        s = getattr(r, 'strokes', None)                 # (an axes' lines put off: _strokes)
        if s is not None and self._put_off(r, ax, s):
            return
        p = self.get_path()
        if p is None or not len(p.vertices):
            return
        v, codes = p.vertices, p.codes
        if hasattr(ax, '_traced_shape'):
            v, codes = ax._traced_shape(v, codes, self.src)
        for px, py in ax._shape_px(v[:, 0], v[:, 1], self.src):
            self._draw_path(r, ax, np.column_stack([px, py]), codes)

    def _put_off(self, r, ax, s):
        """render() into a Strokes (s): an edge alone (no face, no hatch),
        solid, in data coordinates of linear axes -- the vertices mapped as
        _data_px maps them, flattened, and the edge a group of the parts
        unsnapped, small_lines snapping them by ax._snap as _draw.snap did
        (the same width, the same snap value). True if done (False:
        render() as ever, nothing put in)."""
        lin = s.linear
        if lin is None or self.src != 'data' or self.hatch or hasattr(ax, '_traced_shape'):
            return False
        fc, ec = self._colors()
        if fc[3]:
            return False
        lw = r.px(self.linewidth) if ec[3] and self.linestyle not in ('None', 'none') else 0.0
        if type(lw) is not float and np.ndim(lw) != 0:
            return False
        lw = float(lw)
        if lw > 0 and _dashes(self.linestyle, self.linewidth) is not None:
            return False
        cap, join = _draw.CAPS.get(self.capstyle), _draw.JOINS.get(self.joinstyle)
        if cap is None or join is None:
            return False
        if type(self).get_path is Ellipse.get_path and self.angle == 0:
            if not lw > 0:                              # (an ellipse: nothing shows)
                return True
            n0, n1, p0, p1 = s.ellipse(self.width * 0.5, self.height * 0.5,  # (ellipse()'s
                                       float(self.center[0]), float(self.center[1]))  # numbers)
            if n1 == n0:
                return True                             # (flattened to nothing)
            if n1 - n0 < s.low:
                return self._edge(r, ax, s, n1, p1, p0, p1, lw, cap, join, ec)
            xy, offs = s.xy[n0:n1].copy(), s.off[p0:p1 + 1] - n0
        else:
            p = self.get_path()
            if p is None or not len(p.vertices):
                return True
            v = p.vertices
            if v.ndim != 2 or v.shape[1] != 2:
                return False
            pts = np.empty((v.shape[0], 2))
            s.to_px(np.ascontiguousarray(v[:, 0], np.float64),
                    np.ascontiguousarray(v[:, 1], np.float64), *lin, pts)
            xy, offs, closed = _path.flatten(pts, p.codes)
        n = xy.shape[0]
        if not n or not lw > 0:
            return True
        if n >= s.low:                                  # (_draw_path's stroke, from here: its
            r.lines_now(xy, offs, ec, lw, ax.box_clip if self.clip else None, None,  # snap,
                        ax._snap, cap=cap, join=join, filled=False)         # lines')
            return True
        if offs.size < 2 or offs[0] != 0 or offs[-1] != n:
            return False
        p0, p1 = s.parts(xy, offs)
        return self._edge(r, ax, s, s.n, s.p, p0, p1, lw, cap, join, ec)

    def _edge(self, r, ax, s, n, p, p0, p1, lw, cap, join, ec):
        """The edge: parts p0..p1 of s (its points kept up to n, p), stroked
        as _draw_path's r.lines would."""
        box = s.box_clip if self.clip else s.box_all
        if box is None:
            return True
        s.take(n, p)
        s.stroke(p0, p1, ax._snap, float(_draw.snap_value(lw)), False, False, lw, cap, join,
                 box, ec)
        return True

    def _draw_path(self, r, ax, pts, codes):
        xy, offs, closed = _path.flatten(pts, codes)
        if not xy.shape[0]:
            return
        fc, ec = self._colors()
        lw = r.px(self.linewidth) if ec[3] and self.linestyle not in ('None', 'none') else 0.0
        xy = _draw.snap(xy, offs, lw, ax._snap)
        clip = ax.box_clip if self.clip else None
        if fc[3] and closed.any():
            r.c.polygons(xy, offs, fc, clip, 'nonzero')
        if self.hatch and closed.any():
            from ._hatch import color_for
            r.c.hatch(xy, offs, self.hatch, color_for(ec), r.dpi, clip)
        if lw > 0:
            d = _dashes(self.linestyle, self.linewidth)
            r.lines(xy, offs, ec, lw, clip, None if d is None else tuple(r.px(x) for x in d),
                    cap=_draw.CAPS[self.capstyle], join=_draw.JOINS[self.joinstyle],
                    filled=bool(fc[3]) and closed.any())


class Rectangle(Patch):
    def __init__(self, xy, width, height, *, angle=0.0, rotation_point='xy', **kw):
        super().__init__(**kw)
        self.xy, self.width, self.height = tuple(xy), width, height
        self.angle, self.rotation_point = angle, rotation_point

    def get_path(self):
        return _patches.rectangle(self.xy, self.width, self.height, self.angle,
                                  self.rotation_point)

    def get_xy(self):
        return self.xy

    def set_xy(self, xy):
        self.xy = tuple(xy)

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def set_width(self, w):
        self.width = w

    def set_height(self, h):
        self.height = h

    def set_bounds(self, x, y, w, h):
        self.xy, self.width, self.height = (x, y), w, h


class Ellipse(Patch):
    def __init__(self, xy, width, height, *, angle=0.0, **kw):
        super().__init__(**kw)
        self.center, self.width, self.height, self.angle = tuple(xy), width, height, angle

    def get_path(self):
        return _patches.ellipse(self.center, self.width, self.height, self.angle)

    def set_center(self, xy):
        self.center = tuple(xy)


class Circle(Ellipse):
    def __init__(self, xy, radius=5.0, **kw):
        super().__init__(xy, 2 * radius, 2 * radius, **kw)
        self.radius = radius

    def set_radius(self, r):
        self.radius = r
        self.width = self.height = 2 * r


class Polygon(Patch):
    def __init__(self, xy, *, closed=True, **kw):
        super().__init__(**kw)
        self.xy, self.closed = np.asarray(xy, float), closed

    def get_path(self):
        return _patches.polygon(self.xy, self.closed)

    def get_xy(self):
        return self.xy

    def set_xy(self, xy):
        self.xy = np.asarray(xy, float)


class RegularPolygon(Patch):
    def __init__(self, xy, numVertices, *, radius=5.0, orientation=0.0, **kw):
        super().__init__(**kw)
        self.xy, self.numvertices = tuple(xy), numVertices
        self.radius, self.orientation = radius, orientation

    def get_path(self):
        return _patches.regular_polygon(self.xy, self.numvertices, self.radius,
                                        self.orientation)


class Wedge(Patch):
    def __init__(self, center, r, theta1, theta2, *, width=None, **kw):
        super().__init__(**kw)
        self.center, self.r, self.theta1, self.theta2 = tuple(center), r, theta1, theta2
        self.width = width

    def get_path(self):
        return _patches.wedge(self.center, self.r, self.theta1, self.theta2, self.width)

    def set_width(self, w):
        self.width = w

    def set_radius(self, r):
        self.r = r

    def set_center(self, c):
        self.center = tuple(c)

    def set_theta1(self, t):
        self.theta1 = t

    def set_theta2(self, t):
        self.theta2 = t


class Arc(Ellipse):
    """An elliptical arc: stroked, never filled."""

    def __init__(self, xy, width, height, *, angle=0.0, theta1=0.0, theta2=360.0, **kw):
        kw['fill'] = False
        super().__init__(xy, width, height, angle=angle, **kw)
        self.theta1, self.theta2 = theta1, theta2

    def get_path(self):
        return _patches.arc(self.center, self.width, self.height, self.angle, self.theta1,
                            self.theta2)


class FancyBboxPatch(Patch):
    """A box with a box style ('round,pad=0.3', 'square', 'sawtooth' ...),
    in the patch's coordinates (mutation_scale in them, as matplotlib's)."""

    _edge_default = True

    def __init__(self, xy, width, height, boxstyle='round', *, mutation_scale=1.0,
                 mutation_aspect=1.0, **kw):
        super().__init__(**kw)
        self.xy, self.width, self.height = tuple(xy), width, height
        self.boxstyle, self.mutation_scale, self.mutation_aspect = (boxstyle, mutation_scale,
                                                                    mutation_aspect)

    def get_path(self):
        return _patches.fancy_bbox(self.xy, self.width, self.height, self.boxstyle,
                                   self.mutation_scale, self.mutation_aspect)


class FancyArrowPatch(Patch):
    """An arrow from posA to posB (data coordinates unless transform),
    matplotlib's arrow and connection styles; mutation_scale in points."""

    zorder = 1.0
    _edge_default = True

    def __init__(self, posA=None, posB=None, *, arrowstyle='simple', connectionstyle='arc3',
                 mutation_scale=1.0, shrinkA=2.0, shrinkB=2.0, **kw):
        kw.setdefault('linewidth', 1.0)
        kw.setdefault('joinstyle', 'round')
        kw.setdefault('capstyle', 'round')
        super().__init__(**kw)
        self.posA, self.posB = posA, posB
        self.arrowstyle, self.connectionstyle = arrowstyle, connectionstyle
        self.mutation_scale, self.shrinkA, self.shrinkB = mutation_scale, shrinkA, shrinkB

    def set_positions(self, posA, posB):
        self.posA, self.posB = posA, posB

    def data_bounds(self):
        if self.src != 'data':
            return None
        v = np.array([self.posA, self.posB], float)
        return _axis_bounds(v[:, 0]), _axis_bounds(v[:, 1])

    def _geometry(self, r, ax):
        """The arrow's flattened paths, px: [(xy, offsets, fillable)]."""
        v = np.array([self.posA, self.posB], float)
        px, py = ax._points_px(v[:, 0], v[:, 1], self.src)
        if not np.isfinite(np.r_[px, py]).all():
            return []
        k = r.dpi / 72.0
        paths, fills = _patches.arrow_paths((px[0], py[0]), (px[1], py[1]), self.arrowstyle,
                                            self.connectionstyle,
                                            mutation_scale=self.mutation_scale * k,
                                            shrinkA=self.shrinkA * k, shrinkB=self.shrinkB * k,
                                            linewidth=self.linewidth * k)
        out = []
        for p, fillable in zip(paths, fills):
            xy, offs, _ = _path.flatten(p.vertices, p.codes)
            if xy.shape[0]:
                out.append((xy, offs, fillable))
        return out

    def _batch_key(self):
        """What arrows drawn together by _patches.arc3_arrows share (src,
        styles), or None: a style it does not draw."""
        if not (isinstance(self.arrowstyle, str) and isinstance(self.connectionstyle, str)):
            return None
        name, _ = _patches._parse(self.arrowstyle, {}, _patches.ARROWSTYLES, 'arrow style')
        cname, ckw = _patches._parse(self.connectionstyle, {}, _patches.CONNECTIONSTYLES,
                                     'connection style')
        if name not in _patches.HEAD_STYLES or cname != 'arc3':
            return None
        src = self.src if isinstance(self.src, str) else id(self.src)
        return src, self.arrowstyle, float(ckw['rad'])

    def render(self, r, ax):
        _draw_arrows(r, ax, [self])


def _take(xy, offs, sel):
    """The parts sel (bool) of a polyline set: (xy, offsets)."""
    if sel.all():
        return xy, offs
    lo, hi = offs[:-1][sel], offs[1:][sel]
    n = hi - lo
    o = np.zeros(n.size + 1, np.int64)
    np.cumsum(n, out=o[1:])
    return xy[np.repeat(lo - o[:-1], n) + np.arange(o[-1])], o


def _draw_arrows(r, ax, arrows):
    """FancyArrowPatches drawn together: the faces filled colour by colour,
    then every edge in one pass of the line kernel (a style per distinct
    colour and width). Arrows of a head style ('->', '-|>' ...) along arc3
    connections are made all at once (_patches.arc3_arrows_flat: the same
    numbers as one at a time), the rest one by one. A measuring draw
    (tight_layout) skips them: arrows paint, they take no room."""
    if getattr(r, 'measuring', False):
        return
    fkeys, skeys, chunks, groups, seen = {}, {}, [], {}, {}
    k = r.dpi / 72.0

    def keys(a):                    # (fill colour #, edge style #), -1: none
        same = (id(a.facecolor), id(a.edgecolor), a.alpha, a.fill, a.linewidth)
        if same not in seen:                            # (a streamplot's arrows: all alike)
            fc, ec = a._colors()
            lw = r.px(a.linewidth)
            seen[same] = (fkeys.setdefault(fc.tobytes(), len(fkeys)) if fc[3] else -1,
                          skeys.setdefault((ec.tobytes(), float(lw)), len(skeys))
                          if lw > 0 and ec[3] else -1)
        return seen[same]

    for a in arrows:
        g = a._batch_key()
        if g is not None:
            groups.setdefault(g, []).append(a)
            continue
        f, s = keys(a)
        for xy, offs, fillable in a._geometry(r, ax):
            m = offs.size - 1
            chunks.append((xy, offs, np.full(m, f if fillable else -1), np.full(m, s)))
    for (_, style, rad), g in groups.items():
        v = np.array([(a.posA, a.posB) for a in g], np.float64)
        px, py = ax._points_px(v[..., 0].ravel(), v[..., 1].ravel(), g[0].src)
        P = np.stack([px, py], 1).reshape(-1, 2, 2)
        ok = np.isfinite(P).all((1, 2))
        if not ok.all():
            g, P = [a for a, o in zip(g, ok) if o], P[ok]
        if not g:
            continue
        num = np.array([(a.mutation_scale, a.shrinkA, a.shrinkB, a.linewidth) for a in g],
                       np.float64) * k
        fs = np.array([keys(a) for a in g]).reshape(-1, 2)
        xy, offs, arrow, fillable = _patches.arc3_arrows_flat(
            P[:, 0], P[:, 1], style, rad, mutation_scale=num[:, 0], shrinkA=num[:, 1],
            shrinkB=num[:, 2], linewidth=num[:, 3])
        chunks.append((xy, offs, np.where(fillable, fs[arrow, 0], -1), fs[arrow, 1]))
    if not chunks:
        return
    if len(chunks) == 1:
        xy, offs, fk, sk = chunks[0]
    else:
        n = np.cumsum([0] + [c[0].shape[0] for c in chunks])
        xy = np.concatenate([c[0] for c in chunks])
        offs = np.concatenate([c[1][:-1] + n[i] for i, c in enumerate(chunks)] + [n[-1:]])
        fk = np.concatenate([c[2] for c in chunks])
        sk = np.concatenate([c[3] for c in chunks])
    offs = offs.astype(np.int64)
    faces = np.zeros((offs.size - 1, 4), np.uint8)      # each path filled, then stroked, in turn
    for key, i in fkeys.items():
        faces[fk == i] = np.frombuffer(key, np.uint8)
    keys = sorted(skeys, key=skeys.get)
    st = _draw.Styles(np.array([np.frombuffer(c, np.uint8) for c, _ in keys]).reshape(-1, 4),
                      np.array([w for _, w in keys]), caps=_draw.ROUND)
    r.c.paths(xy, offs, faces, sk, st, ax.box_clip)


class ArrowBatch(Artist):
    """Many FancyArrowPatches as one artist (streamplot's arrows)."""

    zorder = 2.0

    def __init__(self, ax, arrows, zorder=None):
        super().__init__(ax, zorder)
        self.arrows = list(arrows)

    def render(self, r, ax):
        _draw_arrows(r, ax, [a for a in self.arrows if a.visible])
