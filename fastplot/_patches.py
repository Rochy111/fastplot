"""matplotlib's patches, box styles and fancy arrows, as Paths.

    p = rectangle((x, y), w, h, angle=30)              # data coordinates
    xy, offsets, closed = _path.flatten(apply(data_to_px, p.vertices), p.codes)

    box = fancy_bbox((x0, y0), w, h, 'round,pad=0.3', mutation_scale=fontsize_px,
                     ydown=True)                       # a text's frame, px
    fill, stroke = fancy_arrow(posA, posB, '->', 'arc3,rad=0.2', mutation_scale=..., ...)

Patches. Each function returns what matplotlib draws: the patch's
get_path() under its get_patch_transform(), in the patch's own
coordinates (data, usually) -- the same vertices and codes, Bezier
control points and all. The caller maps them to pixels (its data
transform, the y flip included) and flattens them (_path.flatten); a
Rectangle's or an Ellipse's angle is in those coordinates, as in
matplotlib. Arc is a stroke only. matplotlib fills a patch nonzero.

Box styles (FancyBboxPatch, for text frames and legend boxes) are
matplotlib's BoxStyle classes: the box (x0, y0, width, height) grown by
pad * mutation_size, its outline square, round (quadratic corners),
round4 (cubic), sawtooth, roundtooth, an arrow (larrow, rarrow, darrow),
circle or ellipse. They are in matplotlib's display frame, y up; with
ydown=True the box is given and returned on the screen (x0, y0 its top
left corner), the outline being matplotlib's mirrored -- the picture
matplotlib would draw.

Fancy arrows (annotate) are FancyArrowPatch in display space: the
connection style makes the path from posA to posB (arc3, angle3, angle,
arc, bar), clipped where it leaves patchA and enters patchB (here a box
(x0, y0, x1, y1) or a closed Path, in px) and then shrunk by circles of
shrinkA / shrinkB px around its ends -- each by matplotlib's bisection of
the Bezier segment that crosses the boundary (to 0.01 px) -- and the
arrow style turns it into paths: the line and its heads or brackets
('-', '->', '-|>', ']-[' ...: stroked, the '|>' heads also filled) or one
closed outline (simple, fancy, wedge: filled and stroked). Everything is
in pixels: mutation_scale (matplotlib's, in points, * dpi / 72), shrinks,
linewidth (it pulls the heads back by half a line width, as matplotlib's
_get_arrow_wedge). ydown=True (the default) takes and returns screen
coordinates, the arrow being matplotlib's mirrored, so angles (angleA,
angleB, the sign of rad) turn the way they do in matplotlib's picture.
"""
import math

import numpy as np

from . import _path
from ._path import (CLOSEPOLY, CURVE3, CURVE4, LINETO, MOVETO, Path, chain, rotation,
                    scaling, translation)

# ------------------------------------------------------------------ patches


def rectangle(xy, width, height, angle=0.0, rotation_point='xy'):
    """Rectangle(xy, width, height, angle=, rotation_point=): 'xy',
    'center' or a point (x, y)."""
    x0, y0 = float(xy[0]), float(xy[1])
    x1, y1 = x0 + width, y0 + height
    bw, bh = x1 - x0, y1 - y0
    if isinstance(rotation_point, str):
        if rotation_point == 'center':
            rp = x0 + bw / 2., y0 + bh / 2.
        elif rotation_point == 'xy':
            rp = x0, y0
        else:
            raise ValueError("rotation_point must be 'xy', 'center' or (x, y)")
    else:
        rp = float(rotation_point[0]), float(rotation_point[1])
    M = chain(scaling(bw, bh), translation(x0, y0), translation(-rp[0], -rp[1]),
              rotation(float(angle)), translation(rp[0], rp[1]))
    return _path.unit_rectangle().transformed(M)


def ellipse(xy, width, height, angle=0.0):
    """Ellipse(xy, width, height, angle=): the unit circle's Beziers."""
    M = chain(scaling(width * 0.5, height * 0.5), rotation(angle), translation(xy[0], xy[1]))
    return _path.unit_circle().transformed(M)


def circle(xy, radius=5.0):
    """Circle(xy, radius)."""
    return ellipse(xy, radius * 2, radius * 2)


def polygon(xy, closed=True):
    """Polygon(xy, closed=): closed, the first vertex repeated unless it is
    already last; open, a repeated last vertex dropped (as set_xy)."""
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    n = len(xy)
    if closed:
        if n == 1 or n > 1 and (xy[0] != xy[-1]).any():
            xy = np.concatenate([xy, [xy[0]]])
    elif n > 2 and (xy[0] == xy[-1]).all():
        xy = xy[:-1]
    return Path(xy, closed=closed)


def regular_polygon(xy, numVertices, radius=5.0, orientation=0.0):
    """RegularPolygon(xy, numVertices, radius=, orientation= radians)."""
    M = chain(scaling(radius), _rotation_rad(orientation), translation(xy[0], xy[1]))
    return _path.unit_regular_polygon(numVertices).transformed(M)


def _rotation_rad(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wedge(center, r, theta1, theta2, width=None):
    """Wedge(center, r, theta1, theta2, width=): a pie slice, or with width
    a slice of a ring (the inner arc reversed)."""
    if abs((theta2 - theta1) - 360) <= 1e-12:
        theta1, theta2 = 0, 360
        connector = MOVETO
    else:
        connector = LINETO
    a = _path.arc(theta1, theta2)
    if width is not None:
        v1 = a.vertices
        v2 = a.vertices[::-1] * (r - width) / r
        v = np.concatenate([v1, v2, [(0, 0)]])
        c = [*a.codes, connector, *a.codes[1:], CLOSEPOLY]
    else:
        v = np.concatenate([a.vertices, [(0, 0), (0, 0)]])
        c = [*a.codes, connector, CLOSEPOLY]
    return Path(v * r + np.asarray(center, np.float64), c)


def arc(xy, width, height, angle=0.0, theta1=0.0, theta2=360.0):
    """Arc(xy, width, height, angle=, theta1=, theta2=): an elliptic arc
    (stroked, never filled); the angles stretched to the ellipse as
    matplotlib's _theta_stretch. (matplotlib's splitting of huge arcs at
    the axes box is not done.)"""
    if width != height and not (theta1 != theta2 and theta1 % 360 == theta2 % 360):
        theta1 = _theta_stretch(theta1, width / height)
        theta2 = _theta_stretch(theta2, width / height)
    M = chain(scaling(width * 0.5, height * 0.5), rotation(angle), translation(xy[0], xy[1]))
    return _path.arc(theta1, theta2).transformed(M)


def _theta_stretch(theta, scale):
    theta = np.deg2rad(theta)
    stheta = np.rad2deg(np.arctan2(scale * np.sin(theta), np.cos(theta)))
    return (stheta + 360) % 360


def shadow(path, ox, oy):
    """Shadow(patch, ox, oy): the patch's path moved by (ox, oy). matplotlib
    converts ox, oy from points to pixels and adds them in the patch's
    own coordinates (before its data transform) -- pixels for a patch in
    pixels, as a legend frame; on a y-down screen a shadow below is +oy.
    Its colour is (1 - shade) * the face colour (shade 0.7), alpha 0.5."""
    return path.transformed(translation(ox, oy))


# -------------------------------------------------------------- box styles

_PARSED = {}                        # (style string, table) -> (name, params), no kwargs


def _parse(style, kwargs, table, what):
    """matplotlib's _Style(name, **kw): 'name,key=value,...' -> (name, params).
    A string without kwargs is parsed once (a streamplot's arrows: the same
    two styles each)."""
    if not kwargs and isinstance(style, str):
        hit = _PARSED.get((style, id(table)))
        if hit is not None:
            return hit[0], dict(hit[1])
        name, params = _parse_style(style, kwargs, table, what)
        _PARSED[style, id(table)] = (name, dict(params))
        return name, params
    return _parse_style(style, kwargs, table, what)


def _parse_style(style, kwargs, table, what):
    parts = style.replace(' ', '').split(',')
    name = parts[0].lower()
    if name not in table:
        raise ValueError('Unknown %s: %r' % (what, style))
    try:
        args = {k: float(v) for k, v in (p.split('=') for p in parts[1:])}
    except ValueError as err:
        raise ValueError('Incorrect %s argument: %r' % (what, style)) from err
    args.update(kwargs)
    defaults = table[name][1]
    bad = set(args) - set(defaults)
    if bad:
        raise TypeError('%s %r takes no argument %s' % (what, name, ', '.join(sorted(bad))))
    return name, {**defaults, **args}


def _box_square(x0, y0, width, height, mutation_size, pad):
    pad = mutation_size * pad
    width, height = width + 2 * pad, height + 2 * pad
    x0, y0 = x0 - pad, y0 - pad
    x1, y1 = x0 + width, y0 + height
    return Path.create_closed([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _box_circle(x0, y0, width, height, mutation_size, pad):
    pad = mutation_size * pad
    width, height = width + 2 * pad, height + 2 * pad
    x0, y0 = x0 - pad, y0 - pad
    return _path.circle((x0 + width / 2, y0 + height / 2), max(width, height) / 2)


def _box_ellipse(x0, y0, width, height, mutation_size, pad):
    pad = mutation_size * pad
    width, height = width + 2 * pad, height + 2 * pad
    x0, y0 = x0 - pad, y0 - pad
    a = width / math.sqrt(2)
    b = height / math.sqrt(2)
    return _path.unit_circle().transformed(
        chain(scaling(a, b), translation(x0 + width / 2, y0 + height / 2)))


def _check_arrow(head_width, head_angle):
    if head_width < 0:
        raise ValueError("'head_width' must be nonnegative")
    if head_angle % 360 == 0:
        raise ValueError("'head_angle' must be nonzero")


def _box_rarrow(x0, y0, width, height, mutation_size, pad, head_width, head_angle):
    _check_arrow(head_width, head_angle)
    pad = mutation_size * pad
    dx, dy = width + 2 * pad, height + 2 * pad
    x0, y0 = x0 - pad, y0 - pad
    x1, y1 = x0 + dx, y0 + dy
    head_dy = head_width * dy
    mid_y = (y0 + y1) / 2
    shaft_y0 = mid_y - head_dy / 2
    shaft_y1 = mid_y + head_dy / 2
    cot = 1 / math.tan(math.radians(head_angle / 2))
    if cot > 0:
        tip_x = x1 + cot * min(dy, head_dy) / 2
        shaft_x = tip_x - cot * head_dy / 2
        return Path.create_closed([
            (x0, y0), (shaft_x, y0), (shaft_x, shaft_y0), (tip_x, mid_y),
            (shaft_x, shaft_y1), (shaft_x, y1), (x0, y1)])
    dx = min(-cot * max(head_dy - dy, 0) / 2, dx)       # (cot < 0: a reversed head)
    mid_y0 = min(shaft_y0, y0) - dx / cot
    mid_y1 = max(shaft_y1, y1) + dx / cot
    return Path.create_closed([
        (x0, y0), (x1 - dx, mid_y0), (x1, shaft_y0), (x1, shaft_y1), (x1 - dx, mid_y1),
        (x0, y1)])


def _box_larrow(x0, y0, width, height, mutation_size, pad, head_width, head_angle):
    p = _box_rarrow(x0, y0, width, height, mutation_size, pad, head_width, head_angle)
    p.vertices[:, 0] = 2 * x0 + width - p.vertices[:, 0]
    return p


def _box_darrow(x0, y0, width, height, mutation_size, pad, head_width, head_angle):
    _check_arrow(head_width, head_angle)
    pad = mutation_size * pad
    dx, dy = width + 2 * pad, height + 2 * pad
    x0, y0 = x0 - pad, y0 - pad
    x1, y1 = x0 + dx, y0 + dy
    head_dy = head_width * dy
    mid_y = (y0 + y1) / 2
    shaft_y0 = mid_y - head_dy / 2
    shaft_y1 = mid_y + head_dy / 2
    cot = 1 / math.tan(math.radians(head_angle / 2))
    if cot > 0:
        tip_x0 = x0 - cot * min(dy, head_dy) / 2
        shaft_x0 = tip_x0 + cot * head_dy / 2
        tip_x1 = x1 + cot * min(dy, head_dy) / 2
        shaft_x1 = tip_x1 - cot * head_dy / 2
        return Path.create_closed([
            (shaft_x0, y1), (shaft_x0, shaft_y1), (tip_x0, mid_y), (shaft_x0, shaft_y0),
            (shaft_x0, y0), (shaft_x1, y0), (shaft_x1, shaft_y0), (tip_x1, mid_y),
            (shaft_x1, shaft_y1), (shaft_x1, y1)])
    dx = min(-cot * max(head_dy - dy, 0) / 2, dx / 2)
    mid_y0 = min(shaft_y0, y0) - dx / cot
    mid_y1 = max(shaft_y1, y1) + dx / cot
    return Path.create_closed([
        (x0, shaft_y0), (x0 + dx, mid_y0), (x1 - dx, mid_y0), (x1, shaft_y0),
        (x1, shaft_y1), (x1 - dx, mid_y1), (x0 + dx, mid_y1), (x0, shaft_y1)])


def _box_round(x0, y0, width, height, mutation_size, pad, rounding_size):
    pad = mutation_size * pad
    dr = mutation_size * rounding_size if rounding_size else pad
    width, height = width + 2 * pad, height + 2 * pad
    x0, y0 = x0 - pad, y0 - pad
    x1, y1 = x0 + width, y0 + height
    cp = [(x0 + dr, y0), (x1 - dr, y0), (x1, y0), (x1, y0 + dr), (x1, y1 - dr), (x1, y1),
          (x1 - dr, y1), (x0 + dr, y1), (x0, y1), (x0, y1 - dr), (x0, y0 + dr), (x0, y0),
          (x0 + dr, y0), (x0 + dr, y0)]
    com = [MOVETO, LINETO, CURVE3, CURVE3, LINETO, CURVE3, CURVE3, LINETO, CURVE3, CURVE3,
           LINETO, CURVE3, CURVE3, CLOSEPOLY]
    return Path(cp, com)


def _box_round4(x0, y0, width, height, mutation_size, pad, rounding_size):
    pad = mutation_size * pad
    dr = mutation_size * rounding_size if rounding_size else pad / 2.
    width = width + 2 * pad - 2 * dr
    height = height + 2 * pad - 2 * dr
    x0, y0 = x0 - pad + dr, y0 - pad + dr
    x1, y1 = x0 + width, y0 + height
    cp = [(x0, y0), (x0 + dr, y0 - dr), (x1 - dr, y0 - dr), (x1, y0), (x1 + dr, y0 + dr),
          (x1 + dr, y1 - dr), (x1, y1), (x1 - dr, y1 + dr), (x0 + dr, y1 + dr), (x0, y1),
          (x0 - dr, y1 - dr), (x0 - dr, y0 + dr), (x0, y0), (x0, y0)]
    com = [MOVETO] + [CURVE4] * 12 + [CLOSEPOLY]
    return Path(cp, com)


def _sawtooth_vertices(x0, y0, width, height, mutation_size, pad, tooth_size):
    pad_ = mutation_size * pad
    tooth = pad * .5 * mutation_size if tooth_size is None else tooth_size * mutation_size
    hsz = tooth / 2
    width = width + 2 * pad_ - tooth
    height = height + 2 * pad_ - tooth
    dsx_n = round((width - tooth) / (tooth * 2)) * 2
    dsy_n = round((height - tooth) / (tooth * 2)) * 2
    x0, y0 = x0 - pad_ + hsz, y0 - pad_ + hsz
    x1, y1 = x0 + width, y0 + height
    xs = [x0, *np.linspace(x0 + hsz, x1 - hsz, 2 * dsx_n + 1),
          *([x1, x1 + hsz, x1, x1 - hsz] * dsy_n)[:2 * dsy_n + 2],
          x1, *np.linspace(x1 - hsz, x0 + hsz, 2 * dsx_n + 1),
          *([x0, x0 - hsz, x0, x0 + hsz] * dsy_n)[:2 * dsy_n + 2]]
    ys = [*([y0, y0 - hsz, y0, y0 + hsz] * dsx_n)[:2 * dsx_n + 2],
          y0, *np.linspace(y0 + hsz, y1 - hsz, 2 * dsy_n + 1),
          *([y1, y1 + hsz, y1, y1 - hsz] * dsx_n)[:2 * dsx_n + 2],
          y1, *np.linspace(y1 - hsz, y0 + hsz, 2 * dsy_n + 1)]
    return [*zip(xs, ys), (xs[0], ys[0])]


def _box_sawtooth(x0, y0, width, height, mutation_size, pad, tooth_size):
    return Path(_sawtooth_vertices(x0, y0, width, height, mutation_size, pad, tooth_size),
                closed=True)


def _box_roundtooth(x0, y0, width, height, mutation_size, pad, tooth_size):
    v = _sawtooth_vertices(x0, y0, width, height, mutation_size, pad, tooth_size)
    v = np.concatenate([v, [v[0]]])
    codes = [MOVETO] + [CURVE3, CURVE3] * ((len(v) - 1) // 2) + [CLOSEPOLY]
    return Path(v, codes)


_ARROW_KW = {'pad': 0.3, 'head_width': 1.5, 'head_angle': 90}
BOXSTYLES = {
    'square': (_box_square, {'pad': 0.3}),
    'circle': (_box_circle, {'pad': 0.3}),
    'ellipse': (_box_ellipse, {'pad': 0.3}),
    'rarrow': (_box_rarrow, _ARROW_KW),
    'larrow': (_box_larrow, _ARROW_KW),
    'darrow': (_box_darrow, _ARROW_KW),
    'round': (_box_round, {'pad': 0.3, 'rounding_size': None}),
    'round4': (_box_round4, {'pad': 0.3, 'rounding_size': None}),
    'sawtooth': (_box_sawtooth, {'pad': 0.3, 'tooth_size': None}),
    'roundtooth': (_box_roundtooth, {'pad': 0.3, 'tooth_size': None}),
}


def boxstyle_path(boxstyle, x0, y0, width, height, mutation_size, **kwargs):
    """BoxStyle(boxstyle, **kwargs)(x0, y0, width, height, mutation_size):
    the outline around a box, matplotlib's frame (y up)."""
    name, kw = _parse(boxstyle, kwargs, BOXSTYLES, 'box style')
    return BOXSTYLES[name][0](x0, y0, width, height, mutation_size, **kw)


def fancy_bbox(xy, width, height, boxstyle='round', mutation_scale=1.0, mutation_aspect=1.0,
               ydown=False, **kwargs):
    """FancyBboxPatch(xy, width, height, boxstyle, mutation_scale=,
    mutation_aspect=).get_path(). ydown: xy is the box's top left on a
    y-down screen and the path comes back there (matplotlib's, mirrored)."""
    x, y = float(xy[0]), float(xy[1])
    if ydown:
        y = -(y + height)
    m = 1.0 if mutation_aspect is None else mutation_aspect
    p = boxstyle_path(boxstyle, x, y / m, width, height / m, mutation_scale, **kwargs)
    v = p.vertices * [1, m]
    if ydown:
        v[:, 1] = -v[:, 1]
    return Path._raw(v, p.codes)


# ------------------------------------------------------------ Bezier tools
# (matplotlib.bezier, as it computes them)

class NonIntersectingPathException(ValueError):
    pass


def get_intersection(cx1, cy1, cos_t1, sin_t1, cx2, cy2, cos_t2, sin_t2):
    """Where the line through (cx1, cy1) at angle t1 meets that through
    (cx2, cy2) at t2."""
    line1_rhs = sin_t1 * cx1 - cos_t1 * cy1
    line2_rhs = sin_t2 * cx2 - cos_t2 * cy2
    a, b = sin_t1, -cos_t1
    c, d = sin_t2, -cos_t2
    ad_bc = a * d - b * c
    if abs(ad_bc) < 1e-12:
        raise ValueError('Given lines do not intersect. Please verify that the angles are '
                         'not equal or differ by 180 degrees.')
    a_, b_ = d, -b
    c_, d_ = -c, a
    a_, b_, c_, d_ = (k / ad_bc for k in [a_, b_, c_, d_])
    return a_ * line1_rhs + b_ * line2_rhs, c_ * line1_rhs + d_ * line2_rhs


def get_normal_points(cx, cy, cos_t, sin_t, length):
    if length == 0.:
        return cx, cy, cx, cy
    cos_t1, sin_t1 = sin_t, -cos_t
    cos_t2, sin_t2 = -sin_t, cos_t
    return (length * cos_t1 + cx, length * sin_t1 + cy,
            length * cos_t2 + cx, length * sin_t2 + cy)


def split_de_casteljau(beta, t):
    beta = np.asarray(beta)
    beta_list = [beta]
    while True:
        beta = beta[:-1] * (1 - t) + beta[1:] * t
        beta_list.append(beta)
        if len(beta) == 1:
            break
    return [b[0] for b in beta_list], [b[-1] for b in reversed(beta_list)]


_BEZIER_ORDERS = {}                     # N -> (orders, reversed, binomial coefficients)


def _bezier(control_points):
    """matplotlib's BezierSegment: t -> the point, as a tuple. Its
    point_at_t's numpy calls, value for value (the weights np.power of
    (1 - t) and t, one product, the same matmul), with less around them."""
    cp = np.asarray(control_points)
    N = len(cp)
    if N not in _BEZIER_ORDERS:
        orders = np.arange(N)
        _BEZIER_ORDERS[N] = (orders, orders[::-1].copy(),
                             [math.factorial(N - 1) // (math.factorial(i)
                                                        * math.factorial(N - 1 - i))
                              for i in range(N)])
    orders, rev, coeff = _BEZIER_ORDERS[N]
    px = (cp.T * coeff).T
    power = np.power

    def at(t):
        w = power(1 - t, rev)
        w *= power(t, orders)
        p = w @ px
        return tuple(p) if p.shape[0] != 2 else (p[0], p[1])
    return at


def find_bezier_t_intersecting_with_closedpath(bezier_point_at_t, inside_closedpath, t0=0.,
                                               t1=1., tolerance=0.01):
    start = bezier_point_at_t(t0)
    end = bezier_point_at_t(t1)
    start_inside = inside_closedpath(start)
    end_inside = inside_closedpath(end)
    if start_inside == end_inside and start != end:
        raise NonIntersectingPathException(
            'Both points are on the same side of the closed path')
    while True:
        if np.hypot(start[0] - end[0], start[1] - end[1]) < tolerance:
            return t0, t1
        middle_t = 0.5 * (t0 + t1)
        middle = bezier_point_at_t(middle_t)
        middle_inside = inside_closedpath(middle)
        if start_inside ^ middle_inside:
            t1 = middle_t
            if end == middle:
                return t0, t1
            end = middle
        else:
            t0 = middle_t
            if start == middle:
                return t0, t1
            start = middle
            start_inside = middle_inside


def split_bezier_intersecting_with_closedpath(bezier, inside_closedpath, tolerance=0.01):
    t0, t1 = find_bezier_t_intersecting_with_closedpath(_bezier(bezier), inside_closedpath,
                                                        tolerance=tolerance)
    return split_de_casteljau(bezier, (t0 + t1) / 2.)


_NV = {MOVETO: 1, LINETO: 1, CURVE3: 2, CURVE4: 3, CLOSEPOLY: 1}


def _segments(path):
    """Path.iter_segments(): (flat control points, code) per segment."""
    v, c = path.vertices, path.full_codes()
    i = 0
    while i < len(v):
        code = int(c[i])
        if code == 0:
            break
        m = _NV.get(code, 1)
        yield v[i:i + m].ravel(), code
        i += m


def split_path_inout(path, inside, tolerance=0.01):
    """matplotlib.bezier.split_path_inout: the path cut where it first
    crosses the boundary of `inside` -> (path_in, path_out)."""
    it = _segments(path)
    ctl_points, _ = next(it)
    begin_inside = inside(ctl_points[-2:])
    ctl_points_old = ctl_points
    iold = 0
    i = 1
    for ctl_points, _ in it:
        iold = i
        i += len(ctl_points) // 2
        if inside(ctl_points[-2:]) != begin_inside:
            bezier_path = np.concatenate([ctl_points_old[-2:], ctl_points])
            break
        ctl_points_old = ctl_points
    else:
        raise ValueError('The path does not intersect with the patch')
    bp = bezier_path.reshape((-1, 2))
    left, right = split_bezier_intersecting_with_closedpath(bp, inside, tolerance)
    if len(left) == 2:
        codes_left, codes_right = [LINETO], [MOVETO, LINETO]
    elif len(left) == 3:
        codes_left, codes_right = [CURVE3, CURVE3], [MOVETO, CURVE3, CURVE3]
    else:
        codes_left, codes_right = [CURVE4] * 3, [MOVETO] + [CURVE4] * 3
    verts_left, verts_right = left[1:], right[:]
    if path.codes is None:
        path_in = Path(np.concatenate([path.vertices[:i], verts_left]))
        path_out = Path(np.concatenate([verts_right, path.vertices[i:]]))
    else:
        path_in = Path(np.concatenate([path.vertices[:iold], verts_left]),
                       np.concatenate([path.codes[:iold], codes_left]))
        path_out = Path(np.concatenate([verts_right, path.vertices[i:]]),
                        np.concatenate([codes_right, path.codes[i:]]))
    return path_in, path_out


def inside_circle(cx, cy, r):
    r2 = r ** 2

    def f(xy):
        x, y = xy
        return (x - cx) ** 2 + (y - cy) ** 2 < r2
    return f


def inside_box(x0, y0, x1, y1):
    """Inside the box, as matplotlib's Rectangle.contains (no edge: radius
    0) decides it in its y-up frame -- edges included but the bottom one,
    which matters when a straight connector's bisection lands on it."""
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)

    def f(xy):
        x, y = xy
        return bool(x0 <= x <= x1 and y0 < y <= y1)
    return f


def inside_path(path, tol=0.01):
    """Inside a closed path (flattened; even-odd)."""
    xy, offs, _ = _path.flatten(path.vertices, path.codes, tol)
    rings = [xy[offs[k]:offs[k + 1]] for k in range(len(offs) - 1)]

    def f(p):
        x, y = p
        inside = False
        for r in rings:
            xa, ya = r[:, 0], r[:, 1]
            xb, yb = np.roll(xa, -1), np.roll(ya, -1)
            cross = (ya > y) != (yb > y)
            with np.errstate(divide='ignore', invalid='ignore'):
                xc = xa + (y - ya) * (xb - xa) / (yb - ya)
            inside ^= bool(np.count_nonzero(cross & (x < xc)) % 2)
        return inside
    return f


def get_cos_sin(x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    d = (dx * dx + dy * dy) ** .5
    if d == 0:
        return 0.0, 0.0
    return dx / d, dy / d


def check_if_parallel(dx1, dy1, dx2, dy2, tolerance=1.e-5):
    theta1 = np.arctan2(dx1, dy1)
    theta2 = np.arctan2(dx2, dy2)
    dtheta = abs(theta1 - theta2)
    if dtheta < tolerance:
        return 1
    elif abs(dtheta - np.pi) < tolerance:
        return -1
    return False


def get_parallels(bezier2, width):
    """Two quadratic Beziers parallel to bezier2 at width either side."""
    c1x, c1y = bezier2[0]
    cmx, cmy = bezier2[1]
    c2x, c2y = bezier2[2]
    if check_if_parallel(c1x - cmx, c1y - cmy, cmx - c2x, cmy - c2y) == -1:
        cos_t1, sin_t1 = get_cos_sin(c1x, c1y, c2x, c2y)
        cos_t2, sin_t2 = cos_t1, sin_t1
    else:
        cos_t1, sin_t1 = get_cos_sin(c1x, c1y, cmx, cmy)
        cos_t2, sin_t2 = get_cos_sin(cmx, cmy, c2x, c2y)
    c1x_l, c1y_l, c1x_r, c1y_r = get_normal_points(c1x, c1y, cos_t1, sin_t1, width)
    c2x_l, c2y_l, c2x_r, c2y_r = get_normal_points(c2x, c2y, cos_t2, sin_t2, width)
    try:
        cmx_l, cmy_l = get_intersection(c1x_l, c1y_l, cos_t1, sin_t1, c2x_l, c2y_l, cos_t2,
                                        sin_t2)
        cmx_r, cmy_r = get_intersection(c1x_r, c1y_r, cos_t1, sin_t1, c2x_r, c2y_r, cos_t2,
                                        sin_t2)
    except ValueError:
        cmx_l, cmy_l = 0.5 * (c1x_l + c2x_l), 0.5 * (c1y_l + c2y_l)
        cmx_r, cmy_r = 0.5 * (c1x_r + c2x_r), 0.5 * (c1y_r + c2y_r)
    return ([(c1x_l, c1y_l), (cmx_l, cmy_l), (c2x_l, c2y_l)],
            [(c1x_r, c1y_r), (cmx_r, cmy_r), (c2x_r, c2y_r)])


def _find_control_points(c1x, c1y, mmx, mmy, c2x, c2y):
    return [(c1x, c1y), (.5 * (4 * mmx - (c1x + c2x)), .5 * (4 * mmy - (c1y + c2y))),
            (c2x, c2y)]


def make_wedged_bezier2(bezier2, width, w1=1., wm=0.5, w2=0.):
    """Two quadratic Beziers either side of bezier2, width * w1 apart at its
    start, * wm in its middle, * w2 at its end."""
    c1x, c1y = bezier2[0]
    cmx, cmy = bezier2[1]
    c3x, c3y = bezier2[2]
    cos_t1, sin_t1 = get_cos_sin(c1x, c1y, cmx, cmy)
    cos_t2, sin_t2 = get_cos_sin(cmx, cmy, c3x, c3y)
    c1x_l, c1y_l, c1x_r, c1y_r = get_normal_points(c1x, c1y, cos_t1, sin_t1, width * w1)
    c3x_l, c3y_l, c3x_r, c3y_r = get_normal_points(c3x, c3y, cos_t2, sin_t2, width * w2)
    c12x, c12y = (c1x + cmx) * .5, (c1y + cmy) * .5
    c23x, c23y = (cmx + c3x) * .5, (cmy + c3y) * .5
    c123x, c123y = (c12x + c23x) * .5, (c12y + c23y) * .5
    cos_t123, sin_t123 = get_cos_sin(c12x, c12y, c23x, c23y)
    c123x_l, c123y_l, c123x_r, c123y_r = get_normal_points(c123x, c123y, cos_t123, sin_t123,
                                                           width * wm)
    return (_find_control_points(c1x_l, c1y_l, c123x_l, c123y_l, c3x_l, c3y_l),
            _find_control_points(c1x_r, c1y_r, c123x_r, c123y_r, c3x_r, c3y_r))


# --------------------------------------------------------- connection styles

def _arc3(posA, posB, rad):
    x1, y1 = posA
    x2, y2 = posB
    x12, y12 = (x1 + x2) / 2., (y1 + y2) / 2.
    dx, dy = x2 - x1, y2 - y1
    cx, cy = x12 + rad * dy, y12 - rad * dx
    return Path([(x1, y1), (cx, cy), (x2, y2)], [MOVETO, CURVE3, CURVE3])


def _angle3(posA, posB, angleA, angleB):
    x1, y1 = posA
    x2, y2 = posB
    cosA, sinA = math.cos(math.radians(angleA)), math.sin(math.radians(angleA))
    cosB, sinB = math.cos(math.radians(angleB)), math.sin(math.radians(angleB))
    cx, cy = get_intersection(x1, y1, cosA, sinA, x2, y2, cosB, sinB)
    return Path([(x1, y1), (cx, cy), (x2, y2)], [MOVETO, CURVE3, CURVE3])


def _angle(posA, posB, angleA, angleB, rad):
    x1, y1 = posA
    x2, y2 = posB
    cosA, sinA = math.cos(math.radians(angleA)), math.sin(math.radians(angleA))
    cosB, sinB = math.cos(math.radians(angleB)), math.sin(math.radians(angleB))
    cx, cy = get_intersection(x1, y1, cosA, sinA, x2, y2, cosB, sinB)
    vertices = [(x1, y1)]
    codes = [MOVETO]
    if rad == 0.:
        vertices.append((cx, cy))
        codes.append(LINETO)
    else:
        dx1, dy1 = x1 - cx, y1 - cy
        f1 = rad / np.hypot(dx1, dy1)
        dx2, dy2 = x2 - cx, y2 - cy
        f2 = rad / np.hypot(dx2, dy2)
        vertices.extend([(cx + dx1 * f1, cy + dy1 * f1), (cx, cy),
                         (cx + dx2 * f2, cy + dy2 * f2)])
        codes.extend([LINETO, CURVE3, CURVE3])
    vertices.append((x2, y2))
    codes.append(LINETO)
    return Path(vertices, codes)


def _arcstyle(posA, posB, angleA, angleB, armA, armB, rad):
    x1, y1 = posA
    x2, y2 = posB
    vertices = [(x1, y1)]
    rounded = []
    codes = [MOVETO]
    if armA:
        cosA, sinA = math.cos(math.radians(angleA)), math.sin(math.radians(angleA))
        d = armA - rad
        rounded.append((x1 + d * cosA, y1 + d * sinA))
        d = armA
        rounded.append((x1 + d * cosA, y1 + d * sinA))
    if armB:
        cosB, sinB = math.cos(math.radians(angleB)), math.sin(math.radians(angleB))
        x_armB, y_armB = x2 + armB * cosB, y2 + armB * sinB
        if rounded:
            xp, yp = rounded[-1]
            dx, dy = x_armB - xp, y_armB - yp
            dd = (dx * dx + dy * dy) ** .5
            rounded.append((xp + rad * dx / dd, yp + rad * dy / dd))
            vertices.extend(rounded)
            codes.extend([LINETO, CURVE3, CURVE3])
        else:
            xp, yp = vertices[-1]
            dx, dy = x_armB - xp, y_armB - yp
            dd = (dx * dx + dy * dy) ** .5
        d = dd - rad
        rounded = [(xp + d * dx / dd, yp + d * dy / dd), (x_armB, y_armB)]
    if rounded:
        xp, yp = rounded[-1]
        dx, dy = x2 - xp, y2 - yp
        dd = (dx * dx + dy * dy) ** .5
        rounded.append((xp + rad * dx / dd, yp + rad * dy / dd))
        vertices.extend(rounded)
        codes.extend([LINETO, CURVE3, CURVE3])
    vertices.append((x2, y2))
    codes.append(LINETO)
    return Path(vertices, codes)


def _bar(posA, posB, armA, armB, fraction, angle):
    x1, y1 = posA
    x20, y20 = x2, y2 = posB
    theta1 = math.atan2(y2 - y1, x2 - x1)
    dx, dy = x2 - x1, y2 - y1
    dd = (dx * dx + dy * dy) ** .5
    ddx, ddy = dx / dd, dy / dd
    if angle is not None:
        theta0 = np.deg2rad(angle)
        dtheta = theta1 - theta0
        dl = dd * math.sin(dtheta)
        dL = dd * math.cos(dtheta)
        x2, y2 = x1 + dL * math.cos(theta0), y1 + dL * math.sin(theta0)
        armB = armB - dl
        dx, dy = x2 - x1, y2 - y1
        dd2 = (dx * dx + dy * dy) ** .5
        ddx, ddy = dx / dd2, dy / dd2
    arm = max(armA, armB)
    f = fraction * dd + arm
    cx1, cy1 = x1 + f * ddy, y1 - f * ddx
    cx2, cy2 = x2 + f * ddy, y2 - f * ddx
    return Path([(x1, y1), (cx1, cy1), (cx2, cy2), (x20, y20)],
                [MOVETO, LINETO, LINETO, LINETO])


CONNECTIONSTYLES = {
    'arc3': (_arc3, {'rad': 0.}),
    'angle3': (_angle3, {'angleA': 90, 'angleB': 0}),
    'angle': (_angle, {'angleA': 90, 'angleB': 0, 'rad': 0.}),
    'arc': (_arcstyle, {'angleA': 0, 'angleB': 0, 'armA': None, 'armB': None, 'rad': 0.}),
    'bar': (_bar, {'armA': 0., 'armB': 0., 'fraction': 0.3, 'angle': None}),
}


def _clip(path, in_start, in_stop):
    if in_start:
        try:
            _, path = split_path_inout(path, in_start)
        except ValueError:
            pass
    if in_stop:
        try:
            path, _ = split_path_inout(path, in_stop)
        except ValueError:
            pass
    return path


def _inside_patch(patch):
    if patch is None:
        return None
    if callable(patch):
        return patch
    if isinstance(patch, Path):
        return inside_path(patch)
    return inside_box(*patch)


def connect(posA, posB, connectionstyle='arc3', shrinkA=2., shrinkB=2., patchA=None,
            patchB=None, **kwargs):
    """ConnectionStyle(connectionstyle, **kwargs)(posA, posB, shrinkA,
    shrinkB, patchA, patchB), matplotlib's display frame (y up): the path
    made, clipped at the patches (a box (x0, y0, x1, y1), a closed Path or
    a function (x, y) -> inside), then shrunk."""
    name, kw = _parse(connectionstyle, kwargs, CONNECTIONSTYLES, 'connection style')
    path = CONNECTIONSTYLES[name][0](tuple(posA), tuple(posB), **kw)
    path = _clip(path, _inside_patch(patchA), _inside_patch(patchB))
    return _clip(path,
                 inside_circle(*path.vertices[0], shrinkA) if shrinkA else None,
                 inside_circle(*path.vertices[-1], shrinkB) if shrinkB else None)


# -------------------------------------------------------------- arrow styles

_CURVE_KW = {'head_length': .4, 'head_width': .2, 'widthA': 1., 'widthB': 1., 'lengthA': 0.2,
             'lengthB': 0.2, 'angleA': 0, 'angleB': 0, 'scaleA': None, 'scaleB': None}
_CURVE_STYLES = {           # name: (the arrow, its parameters with their defaults)
    '-': ('-', {}),
    '<-': ('<-', _CURVE_KW), '->': ('->', _CURVE_KW), '<->': ('<->', _CURVE_KW),
    '<|-': ('<|-', _CURVE_KW), '-|>': ('-|>', _CURVE_KW), '<|-|>': ('<|-|>', _CURVE_KW),
    ']-': (']-', {'widthA': 1., 'lengthA': 0.2, 'angleA': 0}),
    '-[': ('-[', {'widthB': 1., 'lengthB': 0.2, 'angleB': 0}),
    ']-[': (']-[', {'widthA': 1., 'lengthA': 0.2, 'angleA': 0, 'widthB': 1., 'lengthB': 0.2,
                    'angleB': 0}),
    '|-|': ('|-|', {'widthA': 1., 'angleA': 0, 'widthB': 1., 'angleB': 0}),
    ']->': (']->', {'widthA': 1., 'lengthA': 0.2, 'angleA': None}),
    '<-[': ('<-[', {'widthB': 1., 'lengthB': 0.2, 'angleB': None}),
}


def _get_arrow_wedge(x0, y0, x1, y1, head_dist, cos_t, sin_t, linewidth):
    dx, dy = x0 - x1, y0 - y1
    cp_distance = np.hypot(dx, dy)
    pad_projected = (.5 * linewidth / sin_t)
    if cp_distance == 0:
        cp_distance = 1
    ddx = pad_projected * dx / cp_distance
    ddy = pad_projected * dy / cp_distance
    dx = dx / cp_distance * head_dist
    dy = dy / cp_distance * head_dist
    dx1, dy1 = cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy
    dx2, dy2 = cos_t * dx - sin_t * dy, sin_t * dx + cos_t * dy
    v = [(x1 + ddx + dx1, y1 + ddy + dy1), (x1 + ddx, y1 + ddy), (x1 + ddx + dx2, y1 + ddy + dy2)]
    return v, [MOVETO, LINETO, LINETO], ddx, ddy


def _get_bracket(x0, y0, x1, y1, width, length, angle):
    cos_t, sin_t = get_cos_sin(x1, y1, x0, y0)
    x1, y1, x2, y2 = get_normal_points(x0, y0, cos_t, sin_t, width)
    dx, dy = length * cos_t, length * sin_t
    v = [(x1 + dx, y1 + dy), (x1, y1), (x2, y2), (x2 + dx, y2 + dy)]
    if angle:
        x0, y0 = float(x0), float(y0)
        v = _path.apply(chain(translation(-x0, -y0), rotation(angle), translation(x0, y0)), v)
    return v, [MOVETO, LINETO, LINETO, LINETO]


def _curve(path, mutation_size, linewidth, arrow, head_length=.4, head_width=.2, widthA=1.,
           widthB=1., lengthA=0.2, lengthB=0.2, angleA=0, angleB=0, scaleA=None,
           scaleB=None):
    """ArrowStyle._Curve.transmute: ([paths], [fillable])."""
    begin, end = arrow.split('-', 1)
    begin_head, begin_bracket = begin in ('<', '<|'), begin in (']', '|')
    end_head, end_bracket = end in ('>', '|>'), end in ('[', '|')
    fillbegin, fillend = begin == '<|', end == '|>'
    if begin_head or end_head:
        hl = head_length * mutation_size
        hw = head_width * mutation_size
        head_dist = np.hypot(hl, hw)
        cos_t, sin_t = hl / head_dist, hw / head_dist
    scaleA = mutation_size if scaleA is None else scaleA
    scaleB = mutation_size if scaleB is None else scaleB
    x0, y0 = path.vertices[0]
    x1, y1 = path.vertices[1]
    has_begin = begin_head and (x0, y0) != (x1, y1)
    vA, cA, ddxA, ddyA = (_get_arrow_wedge(x1, y1, x0, y0, head_dist, cos_t, sin_t, linewidth)
                          if has_begin else ([], [], 0, 0))
    x2, y2 = path.vertices[-2]
    x3, y3 = path.vertices[-1]
    has_end = end_head and (x2, y2) != (x3, y3)
    vB, cB, ddxB, ddyB = (_get_arrow_wedge(x2, y2, x3, y3, head_dist, cos_t, sin_t, linewidth)
                          if has_end else ([], [], 0, 0))
    paths = [Path(np.concatenate([[(x0 + ddxA, y0 + ddyA)], path.vertices[1:-1],
                                  [(x3 + ddxB, y3 + ddyB)]]), path.codes)]
    fills = [False]
    if has_begin:
        if fillbegin:
            paths.append(Path([*vA, (0, 0)], [*cA, CLOSEPOLY]))
            fills.append(True)
        else:
            paths.append(Path(vA, cA))
            fills.append(False)
    elif begin_bracket:
        x0, y0 = path.vertices[0]
        x1, y1 = path.vertices[1]
        v, c = _get_bracket(x0, y0, x1, y1, widthA * scaleA, lengthA * scaleA, angleA)
        paths.append(Path(v, c))
        fills.append(False)
    if has_end:
        if fillend:
            paths.append(Path([*vB, (0, 0)], [*cB, CLOSEPOLY]))
            fills.append(True)
        else:
            paths.append(Path(vB, cB))
            fills.append(False)
    elif end_bracket:
        x0, y0 = path.vertices[-1]
        x1, y1 = path.vertices[-2]
        v, c = _get_bracket(x0, y0, x1, y1, widthB * scaleB, lengthB * scaleB, angleB)
        paths.append(Path(v, c))
        fills.append(False)
    return paths, fills


def _quadratic(path):
    segs = list(_segments(path))
    if len(segs) != 2 or segs[0][1] != MOVETO or segs[1][1] != CURVE3:
        raise ValueError("'path' is not a valid quadratic Bezier curve (simple, fancy and "
                         "wedge arrows need connectionstyle arc3 or angle3)")
    return [*segs[0][0], *segs[1][0]]


def _point_along_a_line(x0, y0, x1, y1, d):
    dx, dy = x0 - x1, y0 - y1
    ff = d / (dx * dx + dy * dy) ** .5
    return x0 - ff * dx, y0 - ff * dy


def _from_pairs(pp):
    return Path([p for c, p in pp], [c for c, p in pp])


def _simple(path, mutation_size, linewidth, head_length, head_width, tail_width):
    x0, y0, x1, y1, x2, y2 = _quadratic(path)
    head_length = head_length * mutation_size
    in_f = inside_circle(x2, y2, head_length)
    arrow_path = [(x0, y0), (x1, y1), (x2, y2)]
    try:
        arrow_out, arrow_in = split_bezier_intersecting_with_closedpath(arrow_path, in_f)
    except NonIntersectingPathException:
        x0, y0 = _point_along_a_line(x2, y2, x1, y1, head_length)
        x1n, y1n = 0.5 * (x0 + x2), 0.5 * (y0 + y2)
        arrow_in = [(x0, y0), (x1n, y1n), (x2, y2)]
        arrow_out = None
    head_width = head_width * mutation_size
    head_left, head_right = make_wedged_bezier2(arrow_in, head_width / 2., wm=.5)
    if arrow_out is not None:
        tail_left, tail_right = get_parallels(arrow_out, tail_width * mutation_size / 2.)
        pp = [(MOVETO, tail_right[0]), (CURVE3, tail_right[1]), (CURVE3, tail_right[2]),
              (LINETO, head_right[0]), (CURVE3, head_right[1]), (CURVE3, head_right[2]),
              (CURVE3, head_left[1]), (CURVE3, head_left[0]), (LINETO, tail_left[2]),
              (CURVE3, tail_left[1]), (CURVE3, tail_left[0]), (LINETO, tail_right[0]),
              (CLOSEPOLY, tail_right[0])]
    else:
        pp = [(MOVETO, head_right[0]), (CURVE3, head_right[1]), (CURVE3, head_right[2]),
              (CURVE3, head_left[1]), (CURVE3, head_left[0]), (CLOSEPOLY, head_left[0])]
    return _from_pairs(pp), True


def _fancy(path, mutation_size, linewidth, head_length, head_width, tail_width):
    x0, y0, x1, y1, x2, y2 = _quadratic(path)
    head_length = head_length * mutation_size
    arrow_path = [(x0, y0), (x1, y1), (x2, y2)]
    in_f = inside_circle(x2, y2, head_length)
    try:
        path_out, path_in = split_bezier_intersecting_with_closedpath(arrow_path, in_f)
    except NonIntersectingPathException:
        x0, y0 = _point_along_a_line(x2, y2, x1, y1, head_length)
        x1n, y1n = 0.5 * (x0 + x2), 0.5 * (y0 + y2)
        arrow_path = [(x0, y0), (x1n, y1n), (x2, y2)]
        path_head = arrow_path
    else:
        path_head = path_in
    in_f = inside_circle(x2, y2, head_length * .8)
    path_out, path_in = split_bezier_intersecting_with_closedpath(arrow_path, in_f)
    path_tail = path_out
    head_l, head_r = make_wedged_bezier2(path_head, head_width * mutation_size / 2., wm=.6)
    tail_width = tail_width * mutation_size
    tail_left, tail_right = make_wedged_bezier2(path_tail, tail_width * .5, w1=1., wm=0.6,
                                                w2=0.3)
    in_f = inside_circle(x0, y0, tail_width * .3)
    path_in, path_out = split_bezier_intersecting_with_closedpath(arrow_path, in_f)
    tail_start = path_in[-1]
    pp = [(MOVETO, tail_start), (LINETO, tail_right[0]), (CURVE3, tail_right[1]),
          (CURVE3, tail_right[2]), (LINETO, head_r[0]), (CURVE3, head_r[1]),
          (CURVE3, head_r[2]), (CURVE3, head_l[1]), (CURVE3, head_l[0]),
          (LINETO, tail_left[2]), (CURVE3, tail_left[1]), (CURVE3, tail_left[0]),
          (LINETO, tail_start), (CLOSEPOLY, tail_start)]
    return _from_pairs(pp), True


def _wedge(path, mutation_size, linewidth, tail_width, shrink_factor):
    x0, y0, x1, y1, x2, y2 = _quadratic(path)
    b_plus, b_minus = make_wedged_bezier2([(x0, y0), (x1, y1), (x2, y2)],
                                          tail_width * mutation_size / 2., wm=shrink_factor)
    pp = [(MOVETO, b_plus[0]), (CURVE3, b_plus[1]), (CURVE3, b_plus[2]),
          (LINETO, b_minus[2]), (CURVE3, b_minus[1]), (CURVE3, b_minus[0]),
          (CLOSEPOLY, b_minus[0])]
    return _from_pairs(pp), True


ARROWSTYLES = {name: (None, kw) for name, (_, kw) in _CURVE_STYLES.items()}
ARROWSTYLES.update({
    'simple': (_simple, {'head_length': .5, 'head_width': .5, 'tail_width': .2}),
    'fancy': (_fancy, {'head_length': .4, 'head_width': .4, 'tail_width': .4}),
    'wedge': (_wedge, {'tail_width': .3, 'shrink_factor': 0.5}),
})


def arrowstyle_paths(arrowstyle, path, mutation_size, linewidth, aspect_ratio=1.0,
                     **kwargs):
    """ArrowStyle(arrowstyle, **kwargs)(path, mutation_size, linewidth,
    aspect_ratio): ([paths], [fillable]) -- always lists."""
    name, kw = _parse(arrowstyle, kwargs, ARROWSTYLES, 'arrow style')
    if aspect_ratio is not None:
        path = Path(path.vertices / [1, aspect_ratio], path.codes)
    if name in _CURVE_STYLES:
        arrow = _CURVE_STYLES[name][0]
        if name == '-':
            kw = {'head_length': .2, 'head_width': .1}
        elif name == '|-|':
            kw = dict(kw, lengthA=0, lengthB=0)
        paths, fills = _curve(path, mutation_size, linewidth, arrow, **kw)
        if aspect_ratio is not None:
            paths = [Path(p.vertices * [1, aspect_ratio], p.codes) for p in paths]
    else:
        p, f = ARROWSTYLES[name][0](path, mutation_size, linewidth, **kw)
        paths, fills = [p], [f]
    return paths, fills


def arrow_paths(posA=None, posB=None, arrowstyle='simple', connectionstyle='arc3',
                mutation_scale=1.0, mutation_aspect=1.0, shrinkA=2.0, shrinkB=2.0,
                linewidth=1.0, patchA=None, patchB=None, path=None, ydown=True,
                arrowprops=None, connectionprops=None):
    """FancyArrowPatch(posA, posB, arrowstyle=, connectionstyle=, ...) in
    display space: ([paths], [fillable]) -- each path stroked, the
    fillable ones also filled. Pixels throughout; ydown: screen
    coordinates in (posA, posB, patches, path) and out. A string style may
    carry its parameters ('->,head_width=0.3'); arrowprops /
    connectionprops: more of them, as dicts."""
    def flip(xy):
        xy = np.array(xy, np.float64)
        if ydown:
            xy[..., 1] = -xy[..., 1]
        return xy

    def flip_patch(p):
        if p is None or not ydown:
            return p
        if callable(p):
            return lambda q: p((q[0], -q[1]))
        if isinstance(p, Path):
            return Path._raw(flip(p.vertices), p.codes)
        x0, y0, x1, y1 = p
        return (x0, -y1, x1, -y0)

    if path is None:
        if posA is None or posB is None:
            raise ValueError('Either posA and posB, or path need to be provided')
        cpath = connect(flip(posA), flip(posB), connectionstyle or 'arc3', shrinkA, shrinkB,
                        flip_patch(patchA), flip_patch(patchB), **(connectionprops or {}))
    else:
        cpath = Path._raw(flip(path.vertices), path.codes)
    m = 1.0 if mutation_aspect is None else mutation_aspect
    paths, fills = arrowstyle_paths(arrowstyle, cpath, mutation_scale, linewidth, m,
                                    **(arrowprops or {}))
    if ydown:
        paths = [Path._raw(flip(p.vertices), p.codes) for p in paths]
    return paths, fills


def fancy_arrow(posA=None, posB=None, arrowstyle='simple', connectionstyle='arc3', **kwargs):
    """The arrow as matplotlib draws it: (paths to fill, paths to stroke),
    in px (see arrow_paths for the arguments)."""
    paths, fills = arrow_paths(posA, posB, arrowstyle, connectionstyle, **kwargs)
    return [p for p, f in zip(paths, fills) if f], paths


# ------------------------------------------------- many arc3 arrows at once

HEAD_STYLES = {'-': ('', ''), '->': ('', '>'), '-|>': ('', '|>'), '<-': ('<', ''),
               '<->': ('<', '>'), '<|-': ('<|', ''), '<|-|>': ('<|', '|>')}


def _at3(P, t):
    """_bezier's point of each quadratic P[i] (n, 3, 2) at t[i]."""
    s = 1 - t
    return ((s * s)[:, None] * P[:, 0] + (s * t)[:, None] * (2 * P[:, 1])
            + (t * t)[:, None] * P[:, 2])


def _shrink3(P, r, at_start, tolerance=0.01):
    """_clip(path, inside_circle(start, r), None) (at_start) or (None,
    inside_circle(end, r)) for many quadratic paths: each that crosses its
    circle split there, by find_bezier_t_intersecting_with_closedpath's
    bisection run on all of them together, step for step."""
    c = P[:, 0] if at_start else P[:, 2]
    r2 = r ** 2

    def inside(p, cx, cy, r2):
        return (p[:, 0] - cx) ** 2 + (p[:, 1] - cy) ** 2 < r2

    cross = (r > 0) & (inside(P[:, 0], c[:, 0], c[:, 1], r2)
                       != inside(P[:, 2], c[:, 0], c[:, 1], r2))
    if not cross.any():
        return P
    Q, cx, cy, r2 = P[cross], c[cross, 0], c[cross, 1], r2[cross]
    n = len(Q)
    t0, t1 = np.zeros(n), np.ones(n)
    start, end = _at3(Q, t0), _at3(Q, t1)
    si = inside(start, cx, cy, r2)
    live = np.ones(n, bool)
    while True:
        live &= np.hypot(start[:, 0] - end[:, 0], start[:, 1] - end[:, 1]) >= tolerance
        if not live.any():
            break
        tm = 0.5 * (t0 + t1)
        mid = _at3(Q, tm)
        mi = inside(mid, cx, cy, r2)
        left = live & (si ^ mi)
        right = live & ~(si ^ mi)
        stall = (left & (end == mid).all(1)) | (right & (start == mid).all(1))
        t1 = np.where(left, tm, t1)
        end = np.where(left[:, None], mid, end)
        t0 = np.where(right, tm, t0)
        start = np.where(right[:, None], mid, start)
        si = np.where(right, mi, si)
        live &= ~stall
    t = ((t0 + t1) / 2.)[:, None, None]                 # split_de_casteljau
    a = Q[:, :-1] * (1 - t) + Q[:, 1:] * t
    m = a[:, :1] * (1 - t) + a[:, 1:] * t
    P = P.copy()
    P[cross] = (np.concatenate([m, a[:, 1:], Q[:, 2:]], 1) if at_start
                else np.concatenate([Q[:, :1], a[:, :1], m], 1))
    return P


def _wedges(x0, y0, x1, y1, head_dist, cos_t, sin_t, linewidth):
    """_get_arrow_wedge for many: (vertices (n, 3, 2), ddx, ddy)."""
    dx, dy = x0 - x1, y0 - y1
    cp_distance = np.hypot(dx, dy)
    pad_projected = (.5 * linewidth / sin_t)
    cp_distance = np.where(cp_distance == 0, 1, cp_distance)
    ddx = pad_projected * dx / cp_distance
    ddy = pad_projected * dy / cp_distance
    dx = dx / cp_distance * head_dist
    dy = dy / cp_distance * head_dist
    dx1, dy1 = cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy
    dx2, dy2 = cos_t * dx - sin_t * dy, sin_t * dx + cos_t * dy
    v = np.stack([np.stack([x1 + ddx + dx1, y1 + ddy + dy1], 1),
                  np.stack([x1 + ddx, y1 + ddy], 1),
                  np.stack([x1 + ddx + dx2, y1 + ddy + dy2], 1)], 1)
    return v, ddx, ddy


def arc3_arrows(posA, posB, arrowstyle='-|>', rad=0., mutation_scale=1.0, shrinkA=2.0,
                shrinkB=2.0, linewidth=1.0, ydown=True):
    """arrow_paths(posA[i], posB[i], arrowstyle, 'arc3,rad=...', ...) for
    many arrows of one head style (HEAD_STYLES; its parameters in the
    string as there) at once, value for value: posA, posB (n, 2) px, the
    rest scalars or (n,). Returns (line, heads): the lines (n, 3, 2), each
    MOVETO, CURVE3, CURVE3, and per head [(vertices (n, 3, 2), has (n,)
    bool, filled)] -- a head path being its vertices, closed if filled."""
    name, kw = _parse(arrowstyle, {}, ARROWSTYLES, 'arrow style')
    begin, end = HEAD_STYLES[name]
    A = np.array(posA, np.float64).reshape(-1, 2)
    B = np.array(posB, np.float64).reshape(-1, 2)
    n = len(A)
    if ydown:
        A[:, 1] = -A[:, 1]
        B[:, 1] = -B[:, 1]
    d = B - A
    C = np.stack([(A[:, 0] + B[:, 0]) / 2. + rad * d[:, 1],
                  (A[:, 1] + B[:, 1]) / 2. - rad * d[:, 0]], 1)
    P = np.stack([A, C, B], 1)
    P = _shrink3(P, np.broadcast_to(np.asarray(shrinkA, np.float64), n), True)
    P = _shrink3(P, np.broadcast_to(np.asarray(shrinkB, np.float64), n), False)
    line, heads = P.copy(), []
    if begin or end:
        ms = np.asarray(mutation_scale, np.float64)
        hl, hw = kw['head_length'] * ms, kw['head_width'] * ms
        head_dist = np.hypot(hl, hw)
        cos_t, sin_t = hl / head_dist, hw / head_dist
        lw = np.asarray(linewidth, np.float64)
    if begin:
        (x0, y0), (x1, y1) = P[:, 0].T, P[:, 1].T
        has = (x0 != x1) | (y0 != y1)
        v, ddx, ddy = _wedges(x1, y1, x0, y0, head_dist, cos_t, sin_t, lw)
        line[:, 0, 0] = x0 + np.where(has, ddx, 0)
        line[:, 0, 1] = y0 + np.where(has, ddy, 0)
        heads.append((v, has, begin == '<|'))
    if end:
        (x2, y2), (x3, y3) = P[:, 1].T, P[:, 2].T
        has = (x2 != x3) | (y2 != y3)
        v, ddx, ddy = _wedges(x2, y2, x3, y3, head_dist, cos_t, sin_t, lw)
        line[:, 2, 0] = x3 + np.where(has, ddx, 0)
        line[:, 2, 1] = y3 + np.where(has, ddy, 0)
        heads.append((v, has, end == '|>'))
    if ydown:
        line[..., 1] = -line[..., 1]
        for v, _, _ in heads:
            v[..., 1] = -v[..., 1]
    return line, heads


def arc3_arrows_flat(posA, posB, arrowstyle='-|>', rad=0., **kwargs):
    """arc3_arrows flattened as _path.flatten would each path: (xy,
    offsets, arrow (k,) int -- the arrow each part is of --, fillable (k,)
    bool), the parts in matplotlib's order (each arrow's line, then its
    heads)."""
    line, heads = arc3_arrows(posA, posB, arrowstyle, rad, **kwargs)
    n = len(line)
    verts, codes, sizes, fills = [line], [np.array([MOVETO, CURVE3, CURVE3])], [np.full(n, 3)], [False]
    for v, has, filled in heads:
        if filled:
            v = np.concatenate([v, np.zeros((n, 1, 2))], 1)
            codes.append(np.array([MOVETO, LINETO, LINETO, CLOSEPOLY]))
        else:
            codes.append(np.array([MOVETO, LINETO, LINETO]))
        verts.append(v)
        sizes.append(np.where(has, v.shape[1], 0))
        fills.append(filled)
    keep = np.concatenate([np.repeat((s > 0)[:, None], v.shape[1], 1)
                           for s, v in zip(sizes, verts)], 1)
    v = np.concatenate(verts, 1)[keep]
    c = np.broadcast_to(np.concatenate(codes), keep.shape)[keep].astype(np.uint8)
    sizes = np.stack(sizes, 1)                          # (n, paths per arrow)
    paths = sizes.ravel() > 0
    po = np.concatenate([[0], np.cumsum(sizes.ravel()[paths])])
    xy, offs, _, pp = _path.flatten_many(v, c, po)
    path_arrow = np.repeat(np.arange(n), sizes.shape[1])[paths]
    path_fill = np.tile(np.array(fills), n)[paths]
    per = np.diff(pp)
    return xy, offs, np.repeat(path_arrow, per), np.repeat(path_fill, per)
