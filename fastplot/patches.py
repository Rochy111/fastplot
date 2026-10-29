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

    def __init__(self, *, facecolor=None, edgecolor=None, color=None, linewidth=None,
                 linestyle='solid', alpha=None, fill=True, zorder=None, label=None,
                 transform=None, hatch=None, visible=True, **kw):
        kw.pop('antialiased', None)
        kw.pop('aa', None)
        kw.pop('clip_on', None)
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
        self.edgecolor = (rc('patch.edgecolor') if rc('patch.force_edgecolor') or not fill
                          else 'none') if ec is None else ec
        self.fill = fill
        self.linewidth = rc('patch.linewidth') if lw is None else lw
        self.linestyle = linestyle
        self.label = label
        self.transform = transform
        self.hatch = hatch

    # matplotlib's setters
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
        v = p.vertices
        return _axis_bounds(v[:, 0]), _axis_bounds(v[:, 1])

    def render(self, r, ax):
        p = self.get_path()
        if p is None or not len(p.vertices):
            return
        v = p.vertices
        px, py = ax._points_px(v[:, 0], v[:, 1], self.src)
        xy, offs, closed = _path.flatten(np.column_stack([px, py]), p.codes)
        if not xy.shape[0]:
            return
        fc, ec = self._colors()
        lw = r.px(self.linewidth) if ec[3] and self.linestyle not in ('None', 'none') else 0.0
        xy = _draw.snap(xy, offs, lw, ax._snap)
        clip = ax.box_clip if self.clip else None
        if fc[3] and closed.any():
            r.c.polygons(xy, offs, fc, clip, 'nonzero')
        if lw > 0:
            d = _dashes(self.linestyle, self.linewidth)
            r.lines(xy, offs, ec, lw, clip, None if d is None else tuple(r.px(x) for x in d),
                    cap=_draw.BUTT if ax._agg else _draw.LEGACY)


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

    def __init__(self, posA=None, posB=None, *, arrowstyle='simple', connectionstyle='arc3',
                 mutation_scale=1.0, shrinkA=2.0, shrinkB=2.0, **kw):
        kw.setdefault('linewidth', 1.0)
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

    def render(self, r, ax):
        v = np.array([self.posA, self.posB], float)
        px, py = ax._points_px(v[:, 0], v[:, 1], self.src)
        k = r.dpi / 72.0
        paths, fills = _patches.arrow_paths((px[0], py[0]), (px[1], py[1]), self.arrowstyle,
                                            self.connectionstyle,
                                            mutation_scale=self.mutation_scale * k,
                                            shrinkA=self.shrinkA * k, shrinkB=self.shrinkB * k,
                                            linewidth=self.linewidth * k)
        fc, ec = self._colors()
        lw = r.px(self.linewidth)
        for p, fillable in zip(paths, fills):
            xy, offs, _ = _path.flatten(p.vertices, p.codes)
            if fillable and fc[3]:
                r.c.polygons(xy, offs, fc, ax.box_clip, 'nonzero')
            if lw > 0 and ec[3]:
                r.lines(xy, offs, ec, lw, ax.box_clip, cap=_draw.ROUND)
