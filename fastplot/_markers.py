"""matplotlib's markers, as geometry in pixels.

    g = marker_geom('o', size_px, fillstyle='full')    # a MarkerGeom
    placed = place(g, centers_px)                      # polygons and polylines for _draw

A marker is built exactly as matplotlib's MarkerStyle builds it (its
_set_* methods): the same unit path, the same transform -- the 0.5
scaling that makes most markers `markersize` across, the rotations, the
half-filled variants and their alternate halves, the per-marker join and
cap styles -- then scaled to the size in pixels (matplotlib's markersize
in points * dpi / 72, as Line2D scales it) and flipped to the screen
(y down). Vertices are relative to the marker's centre.

What matplotlib does with them (Line2D.draw): the path is filled with
markerfacecolor -- for every marker, the "unfilled" ones too (a '+' or a
'1' encloses no area; a caret's open path is filled closed) -- unless the
fillstyle is 'none'; the alternate path (half fillstyles) is filled with
markerfacecoloralt; then both are stroked, centred on the path, with
markeredgecolor at markeredgewidth. Agg fills nonzero. scatter() fills
unfilled markers' faces with nothing and strokes them in the face colour.
The pixel marker ',' is a 1 px square, never scaled, never stroked.

`place` flattens a marker once and copies it to every centre in numba,
ready for _draw.fill_polygons (rings) and _draw.draw_lines (polylines).
"""
import numpy as np

from . import _path
from ._path import Path, chain, rotation, scaling, translation

(TICKLEFT, TICKRIGHT, TICKUP, TICKDOWN, CARETLEFT, CARETRIGHT, CARETUP, CARETDOWN,
 CARETLEFTBASE, CARETRIGHTBASE, CARETUPBASE, CARETDOWNBASE) = range(12)

MARKERS = {
    '.': 'point', ',': 'pixel', 'o': 'circle', 'v': 'triangle_down', '^': 'triangle_up',
    '<': 'triangle_left', '>': 'triangle_right', '1': 'tri_down', '2': 'tri_up',
    '3': 'tri_left', '4': 'tri_right', '8': 'octagon', 's': 'square', 'p': 'pentagon',
    '*': 'star', 'h': 'hexagon1', 'H': 'hexagon2', '+': 'plus', 'x': 'x', 'D': 'diamond',
    'd': 'thin_diamond', '|': 'vline', '_': 'hline', 'P': 'plus_filled', 'X': 'x_filled',
    TICKLEFT: 'tickleft', TICKRIGHT: 'tickright', TICKUP: 'tickup', TICKDOWN: 'tickdown',
    CARETLEFT: 'caretleft', CARETRIGHT: 'caretright', CARETUP: 'caretup',
    CARETDOWN: 'caretdown', CARETLEFTBASE: 'caretleftbase',
    CARETRIGHTBASE: 'caretrightbase', CARETUPBASE: 'caretupbase',
    CARETDOWNBASE: 'caretdownbase', 'None': 'nothing', 'none': 'nothing', ' ': 'nothing',
    '': 'nothing',
}
FILLSTYLES = ('full', 'left', 'right', 'bottom', 'top', 'none')
_HALF = ('left', 'right', 'bottom', 'top')
_EMPTY = Path(np.empty((0, 2)))

# matplotlib's marker paths (markers.MarkerStyle class attributes)
_TRI = Path.create_closed([[0, 1], [-1, -1], [1, -1]])
_TRI_U = Path.create_closed([[0, 1], [-3 / 5, -1 / 5], [3 / 5, -1 / 5]])
_TRI_D = Path.create_closed([[-3 / 5, -1 / 5], [3 / 5, -1 / 5], [1, -1], [-1, -1]])
_TRI_L = Path.create_closed([[0, 1], [0, -1], [-1, -1]])
_TRI_R = Path.create_closed([[0, 1], [0, -1], [1, -1]])
_LINE = Path([[0.0, -1.0], [0.0, 1.0]])
_TICKH = Path([[0.0, 0.0], [1.0, 0.0]])
_TICKV = Path([[-0.0, 0.0], [-0.0, 1.0]])
_ML = [1, 2, 1, 2]
_TRIPATH = Path([[0.0, 0.0], [0.0, -1.0], [0.0, 0.0], [0.8, 0.5], [0.0, 0.0], [-0.8, 0.5]],
                [1, 2, 1, 2, 1, 2])
_CARET = Path([[-1.0, 1.5], [0.0, 0.0], [1.0, 1.5]])
_CARET_BASE = Path([[-1.0, 0.0], [0.0, -1.5], [1.0, 0]])
_PLUS = Path([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]], _ML)
_X = Path([[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]], _ML)
_PLUS_FILLED = Path.create_closed(np.array([
    (-1, -3), (+1, -3), (+1, -1), (+3, -1), (+3, +1), (+1, +1),
    (+1, +3), (-1, +3), (-1, +1), (-3, +1), (-3, -1), (-1, -1)]) / 6)
_PLUS_FILLED_T = Path.create_closed(np.array([
    (+3, 0), (+3, +1), (+1, +1), (+1, +3), (-1, +3), (-1, +1), (-3, +1), (-3, 0)]) / 6)
_X_FILLED = Path.create_closed(np.array([
    (-1, -2), (0, -1), (+1, -2), (+2, -1), (+1, 0), (+2, +1),
    (+1, +2), (0, +1), (-1, +2), (-2, +1), (-1, 0), (-2, -1)]) / 4)
_X_FILLED_T = Path.create_closed(np.array([
    (+1, 0), (+2, +1), (+1, +2), (0, +1), (-1, +2), (-2, +1), (-1, 0)]) / 4)


class _Style:
    """A MarkerStyle's state while its _set_* method runs (marker units,
    y up): path, transform, alt path and transform, filled, join, cap."""

    def __init__(self, fillstyle, joinstyle, capstyle):
        self.fillstyle = fillstyle
        self.user_join = joinstyle
        self.path = _EMPTY
        self.M = np.eye(3)
        self.alt_path = None
        self.alt_M = None
        self.join = 'round'                     # (matplotlib's default, user's or not)
        self.cap = capstyle or 'butt'
        self.filled = fillstyle != 'none'

    def half(self):
        return self.fillstyle in _HALF

    def joinstyle(self, default):
        self.join = self.user_join or default


def _circle(s, size=1.0):
    s.M = scaling(0.5 * size)
    if not s.half():
        s.path = _path.unit_circle()
    else:
        s.path = s.alt_path = _path.unit_circle_righthalf()
        s.M = chain(s.M, rotation({'right': 0, 'top': 90, 'left': 180, 'bottom': 270}
                                  [s.fillstyle]))
        s.alt_M = chain(s.M, rotation(180.0))


def _pixel(s):
    s.path = _path.unit_rectangle()
    s.M = translation(-0.49999, -0.49999)


def _triangle(s, rot, skip):
    s.M = chain(scaling(0.5), rotation(rot))
    if not s.half():
        s.path = _TRI
    else:
        mp = [_TRI_U, _TRI_L, _TRI_D, _TRI_R]
        a, b = {'top': (0, 2), 'bottom': (2, 0), 'left': (1, 3), 'right': (3, 1)}[s.fillstyle]
        s.path, s.alt_path = mp[(a + skip) % 4], mp[(b + skip) % 4]
        s.alt_M = s.M
    s.joinstyle('miter')


def _square(s):
    s.M = translation(-0.5, -0.5)
    if not s.half():
        s.path = _path.unit_rectangle()
    else:
        s.path = Path([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5], [0.0, 0.0]])
        s.alt_path = Path([[0.0, 0.5], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0], [0.0, 0.5]])
        s.M = chain(s.M, rotation({'bottom': 0, 'right': 90, 'top': 180, 'left': 270}
                                  [s.fillstyle]))
        s.alt_M = s.M
    s.joinstyle('miter')


def _diamond(s):
    s.M = chain(translation(-0.5, -0.5), rotation(45))
    if not s.half():
        s.path = _path.unit_rectangle()
    else:
        s.path = Path([[0, 0], [1, 0], [1, 1], [0, 0]])
        s.alt_path = Path([[0, 0], [0, 1], [1, 1], [0, 0]])
        s.M = chain(s.M, rotation({'right': 0, 'top': 90, 'left': 180, 'bottom': 270}
                                  [s.fillstyle]))
        s.alt_M = s.M
    s.joinstyle('miter')


def _thin_diamond(s):
    _diamond(s)
    s.M = chain(s.M, scaling(0.6, 1.0))         # (matplotlib scales the alt transform too:
    if s.alt_M is not None:                     # the same object)
        s.alt_M = s.M


def _halves(s, polypath, parts):
    if not s.half():
        s.path = polypath
    else:
        top, bottom, left, right = parts(polypath.vertices)
        s.path, s.alt_path = {'top': (top, bottom), 'bottom': (bottom, top),
                              'left': (left, right), 'right': (right, left)}[s.fillstyle]
        s.alt_M = s.M


def _pentagon(s):
    s.M = scaling(0.5)

    def parts(v):
        y = (1 + np.sqrt(5)) / 4.
        return (Path(v[[0, 1, 4, 0]]), Path(v[[1, 2, 3, 4, 1]]),
                Path([v[0], v[1], v[2], [0, -y], v[0]]),
                Path([v[0], v[4], v[3], [0, -y], v[0]]))
    _halves(s, _path.unit_regular_polygon(5), parts)
    s.joinstyle('miter')


def _star(s):
    s.M = scaling(0.5)

    def parts(v):
        return (Path(np.concatenate([v[0:4], v[7:10], v[0:1]])),
                Path(np.concatenate([v[3:8], v[3:4]])),
                Path(np.concatenate([v[0:6], v[0:1]])),
                Path(np.concatenate([v[0:1], v[5:10], v[0:1]])))
    _halves(s, _path.unit_regular_star(5, innerCircle=0.381966), parts)
    s.joinstyle('bevel')


def _hexagon1(s):
    s.M = scaling(0.5)

    def parts(v):
        x = np.abs(np.cos(5 * np.pi / 6.))
        return (Path(np.concatenate([[(-x, 0)], v[[1, 0, 5]], [(x, 0)]])),
                Path(np.concatenate([[(-x, 0)], v[2:5], [(x, 0)]])),
                Path(v[0:4]), Path(v[[0, 5, 4, 3]]))
    _halves(s, _path.unit_regular_polygon(6), parts)
    s.joinstyle('miter')


def _hexagon2(s):
    s.M = chain(scaling(0.5), rotation(30))

    def parts(v):
        x, y = np.sqrt(3) / 4, 3 / 4.
        return (Path(v[[1, 0, 5, 4, 1]]), Path(v[1:5]),
                Path(np.concatenate([[(x, y)], v[:3], [(-x, -y), (x, y)]])),
                Path(np.concatenate([[(x, y)], v[5:2:-1], [(-x, -y)]])))
    _halves(s, _path.unit_regular_polygon(6), parts)
    s.joinstyle('miter')


def _octagon(s):
    s.M = scaling(0.5)
    if not s.half():
        s.M = chain(s.M, rotation(22.5))
        s.path = _path.unit_regular_polygon(8)
    else:
        x = np.sqrt(2.) / 4.
        s.path = s.alt_path = Path([[0, -1], [0, 1], [-x, 1], [-1, x], [-1, -x], [-x, -1],
                                    [0, -1]])
        s.M = chain(s.M, rotation({'left': 0, 'bottom': 90, 'right': 180, 'top': 270}
                                  [s.fillstyle]))
        s.alt_M = chain(s.M, rotation(180.0))
    s.joinstyle('miter')


def _lines(s, path, M):
    s.M = M
    s.filled = False
    s.path = path


def _caret(s, path, rot):
    _lines(s, path, chain(scaling(0.5), rotation(rot)))
    s.joinstyle('miter')


def _filled_cross(s, full, half):
    s.M = np.eye(3)
    s.joinstyle('miter')
    if not s.half():
        s.path = full
    else:
        s.path = s.alt_path = half
        s.M = rotation({'top': 0, 'left': 90, 'bottom': 180, 'right': 270}[s.fillstyle])
        s.alt_M = chain(s.M, rotation(180))


def _nothing(s):
    s.filled = False


_SET = {
    'nothing': _nothing,
    'circle': _circle,
    'point': lambda s: _circle(s, 0.5),
    'pixel': _pixel,
    'triangle_up': lambda s: _triangle(s, 0.0, 0),
    'triangle_down': lambda s: _triangle(s, 180.0, 2),
    'triangle_left': lambda s: _triangle(s, 90.0, 3),
    'triangle_right': lambda s: _triangle(s, 270.0, 1),
    'square': _square,
    'diamond': _diamond,
    'thin_diamond': _thin_diamond,
    'pentagon': _pentagon,
    'star': _star,
    'hexagon1': _hexagon1,
    'hexagon2': _hexagon2,
    'octagon': _octagon,
    'vline': lambda s: _lines(s, _LINE, scaling(0.5)),
    'hline': lambda s: _lines(s, _LINE, chain(scaling(0.5), rotation(90))),
    'tickleft': lambda s: _lines(s, _TICKH, scaling(-1.0, 1.0)),
    'tickright': lambda s: _lines(s, _TICKH, scaling(1.0, 1.0)),
    'tickup': lambda s: _lines(s, _TICKV, scaling(1.0, 1.0)),
    'tickdown': lambda s: _lines(s, _TICKV, scaling(1.0, -1.0)),
    'tri_down': lambda s: _lines(s, _TRIPATH, scaling(0.5)),
    'tri_up': lambda s: _lines(s, _TRIPATH, chain(scaling(0.5), rotation(180))),
    'tri_left': lambda s: _lines(s, _TRIPATH, chain(scaling(0.5), rotation(270))),
    'tri_right': lambda s: _lines(s, _TRIPATH, chain(scaling(0.5), rotation(90))),
    'caretdown': lambda s: _caret(s, _CARET, 0),
    'caretup': lambda s: _caret(s, _CARET, 180),
    'caretleft': lambda s: _caret(s, _CARET, 270),
    'caretright': lambda s: _caret(s, _CARET, 90),
    'caretdownbase': lambda s: _caret(s, _CARET_BASE, 0),
    'caretupbase': lambda s: _caret(s, _CARET_BASE, 180),
    'caretleftbase': lambda s: _caret(s, _CARET_BASE, 270),
    'caretrightbase': lambda s: _caret(s, _CARET_BASE, 90),
    'plus': lambda s: _lines(s, _PLUS, scaling(0.5)),
    'x': lambda s: _lines(s, _X, scaling(0.5)),
    'plus_filled': lambda s: _filled_cross(s, _PLUS_FILLED, _PLUS_FILLED_T),
    'x_filled': lambda s: _filled_cross(s, _X_FILLED, _X_FILLED_T),
}


def _custom(s, path):
    rescale = np.max(np.abs(path.vertices))
    s.M = scaling(0.5 / rescale)
    s.path = path


def _tuple(s, marker):
    if len(marker) == 2:
        numsides, rotation_ = marker[0], 0.0
    else:
        numsides, rotation_ = marker[0], marker[2]
    sym = marker[1]
    if sym == 0:
        s.path = _path.unit_regular_polygon(numsides)
        s.joinstyle('miter')
    elif sym == 1:
        s.path = _path.unit_regular_star(numsides)
        s.joinstyle('bevel')
    elif sym == 2:
        s.path = _path.unit_regular_asterisk(numsides)
        s.filled = False
        s.joinstyle('bevel')
    else:
        raise ValueError('Unexpected tuple marker: %r' % (marker,))
    s.M = chain(scaling(0.5), rotation(rotation_))


def _extents(v, c):
    """Path.get_extents of a path of lines and quadratic curves: the box of
    its end points and of the curves' turning points."""
    ends = [], [], []
    k = 0
    for k in range(c.size):
        if c[k] == 3 and (k == 0 or c[k - 1] != 3 or len(ends[1]) and ends[1][-1] != k - 1):
            ends[0].append(k - 1)                       # (the end point before this control)
            ends[1].append(k)
            ends[2].append(k + 1)
    on = np.ones(c.size, np.bool_)
    on[c == 79] = False
    on[ends[1]] = False                                 # controls count only as they pull
    p = v[on]
    lo, hi = p.min(axis=0), p.max(axis=0)
    if ends[1]:
        a, b, d = v[ends[0]], v[ends[1]], v[ends[2]]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (a - b) / (a - 2.0 * b + d)
            q = (1 - t) ** 2 * a + 2 * t * (1 - t) * b + t ** 2 * d
        q = np.where((t > 0.0) & (t < 1.0), q, a)
        lo, hi = np.minimum(lo, q.min(axis=0)), np.maximum(hi, q.max(axis=0))
    return lo, hi


def _mathtext(s, marker):
    """A '$...$' marker (MarkerStyle._set_mathtext_path): the text's outline,
    its box centred and its larger side 1."""
    from ._text import text_path
    v, c = text_path(marker)
    if not len(v):
        return
    lo, hi = _extents(v, c)
    w, h = hi - lo
    s.path = Path(v, c)
    s.M = chain(_path.translation(-lo[0] - 0.5 * w, -lo[1] - 0.5 * h), scaling(1.0 / max(w, h)))


def _style(marker, fillstyle, joinstyle, capstyle):
    """The marker's _Style (matplotlib's MarkerStyle._set_marker), y up."""
    fillstyle = 'full' if fillstyle is None else fillstyle
    if fillstyle not in FILLSTYLES:
        raise ValueError('fillstyle must be one of %s, not %r' % (FILLSTYLES, fillstyle))
    s = _Style(fillstyle, joinstyle, capstyle)
    if marker is None:
        name = 'nothing'
    elif isinstance(marker, str) and len(marker) > 2 and marker[0] == marker[-1] == '$':
        _mathtext(s, marker)
        return s, None
    else:
        try:
            name = MARKERS.get(marker)
        except TypeError:                       # (unhashable: vertices)
            name = None
    if name is not None:
        _SET[name](s)
    elif isinstance(marker, Path):
        _custom(s, marker)
    elif isinstance(marker, np.ndarray) and marker.ndim == 2 and marker.shape[1] == 2:
        _custom(s, Path(marker))
    elif (hasattr(marker, '__len__') and len(marker) in (2, 3) and
          np.ndim(marker[0]) == 0 and marker[1] in (0, 1, 2)):
        _tuple(s, marker)
    else:
        try:
            p = Path(marker)
        except (ValueError, TypeError) as err:
            raise ValueError('Unrecognized marker style %r' % (marker,)) from err
        _custom(s, p)
    return s, name


class MarkerGeom:
    """A marker in pixels, relative to its centre, y down.

    fill      Path: matplotlib's get_path() under get_transform(), scaled:
              filled with markerfacecolor (and stroked)
    fill_alt  Path or None: the other half of a half fillstyle (get_alt_path()
              under get_alt_transform()), filled with markerfacecoloralt
    stroke    Path or None: what is stroked with markeredgecolor -- fill, and
              fill_alt after it (None for an empty marker)
    filled    matplotlib's is_filled()
    joinstyle, capstyle   matplotlib's defaults for the marker ('miter',
              'round', 'bevel'; 'butt', 'round', 'projecting')
    pixel     the ',' marker (a 1 px square, unscaled, its edge width 0)
    snap_threshold   matplotlib's: a line's marker of this size in px or more
              is snapped, a smaller one is not (None: only if rectilinear)
    """

    __slots__ = ('marker', 'fillstyle', 'size', 'fill', 'fill_alt', 'stroke', 'filled',
                 'joinstyle', 'capstyle', 'pixel', 'snap_threshold', '_flat')

    def __bool__(self):
        return bool(len(self.fill))

    def edge_width(self, markeredgewidth_px):
        """The stroke's width in px: matplotlib centres the edge on the path
        (half of it outside), at markeredgewidth, and none for ','."""
        return 0.0 if self.pixel else float(markeredgewidth_px)

    def flat(self, tol=0.5):
        """The flattened (fill, fill_alt, stroke) as (xy, offsets, closed)
        each (or None), remembered per tol."""
        key = float(tol)
        if self._flat is None or self._flat[0] != key:
            f = tuple(None if p is None or not len(p) else _path.flatten(p.vertices, p.codes, key)
                      for p in (self.fill, self.fill_alt, self.stroke))
            self._flat = (key, f)
        return self._flat[1]


_FLIP = scaling(1.0, -1.0)

# matplotlib's MarkerStyle.get_snap_threshold, by marker; the rest have none
_SNAP_THRESHOLDS = {m: t for t, names in (
    (float('inf'), ('point', 'circle')),
    (1.0, ('plus', 'vline', 'hline', 'tickleft', 'tickright', 'tickup', 'tickdown')),
    (2.0, ('square',)),
    (3.0, ('x', 'caretleft', 'caretright', 'caretup', 'caretdown', 'caretleftbase',
           'caretrightbase', 'caretupbase', 'caretdownbase')),
    (5.0, ('triangle_down', 'triangle_up', 'triangle_left', 'triangle_right', 'tri_down',
           'tri_up', 'tri_left', 'tri_right', 'octagon', 'pentagon', 'star', 'diamond',
           'thin_diamond', 'plus_filled', 'x_filled'))) for m in names}


def marker_geom(marker, size, fillstyle='full', transform=None, joinstyle=None,
                capstyle=None):
    """The MarkerGeom of a marker (anything MarkerStyle takes but mathtext:
    a code, a Path, (n, 2) vertices, (numsides, style[, angle])) of size
    px (markersize points * dpi / 72; for scatter, sqrt(s) * dpi / 72).
    transform: a 3x3 affine in marker units, y up (MarkerStyle's
    transform=, applied after the marker's own)."""
    s, name = _style(marker, fillstyle, joinstyle, capstyle)
    pixel = name == 'pixel'
    U = np.eye(3) if transform is None else np.asarray(transform, np.float64)
    S = np.eye(3) if pixel else scaling(float(size))  # (Line2D: ',' is never scaled)
    g = MarkerGeom()
    g.marker, g.fillstyle, g.size, g.pixel, g._flat = marker, s.fillstyle, size, pixel, None
    g.fill = s.path.transformed(chain(s.M, U, S, _FLIP))
    g.fill_alt = None if s.alt_path is None else \
        s.alt_path.transformed(chain(s.alt_M, U, S, _FLIP))
    g.stroke = None if not len(g.fill) else \
        g.fill if g.fill_alt is None else _path.make_compound_path(g.fill, g.fill_alt)
    g.filled = s.filled
    g.joinstyle, g.capstyle = s.join, s.cap
    g.snap_threshold = _SNAP_THRESHOLDS.get(name)
    return g


class Placed:
    """Markers placed at centres, for _draw:

    fill, fill_alt   (xy, ring_offsets) for fill_polygons(..., rule='nonzero'), or None
    stroke           (xy, offsets) for draw_lines, or None
    index            the centres kept (finite), in order
    rings, alt_rings, lines   parts per marker: marker j's fill rings are
                     ring_offsets[j rings : (j + 1) rings + 1], and so on
    """

    __slots__ = ('fill', 'fill_alt', 'stroke', 'index', 'rings', 'alt_rings', 'lines')


def place(geom, centers_px, tol=0.5):
    """A MarkerGeom flattened once and copied to every finite centre
    (n, 2) px (numba, in parallel)."""
    c = np.ascontiguousarray(centers_px, dtype=np.float64).reshape(-1, 2)
    ok = np.isfinite(c).all(axis=1)
    index = np.nonzero(ok)[0]
    if index.size != len(c):
        c = np.ascontiguousarray(c[index])
    out = Placed()
    out.index = index
    flats = geom.flat(tol)
    res = []
    for f in flats:
        if f is None or not f[0].shape[0]:
            res.append((None, 0))
        else:
            res.append((_path.translate_many(f[0], f[1], c), f[1].size - 1))
    (out.fill, out.rings), (out.fill_alt, out.alt_rings), (out.stroke, out.lines) = res
    return out
