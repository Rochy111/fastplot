"""matplotlib's Path model, its unit shapes, affines, and flattening.

    p = Path(vertices, codes)                          # codes: matplotlib's, or None
    q = p.transformed(affine(scale, rotate_deg, translate))
    xy, offsets, closed = flatten(q.vertices, q.codes, tol=0.1)

A Path is matplotlib's: (n, 2) float64 vertices and n uint8 codes --
MOVETO 1, LINETO 2, CURVE3 3 (a control point, then the end), CURVE4 4
(two control points, then the end), CLOSEPOLY 79 (its vertex ignored),
STOP 0 -- or codes None, a polyline. The unit shapes (circle, arcs,
regular polygons and stars, the rectangle) are built with matplotlib's
own formulas, so their vertices and codes are matplotlib's: markers,
patches and arrows made of them are matplotlib's shapes, to the last
bit of the Bezier control points.

fastplot's rasterizer draws polygons and polylines, not curves, so a path
in pixels is *flattened*: every curve is cut into a number of chords
picked from its control polygon -- the second differences of the control
points bound the curve's second derivative, and a curve cut into n equal
parameter steps lies within max|B''| / (8 n^2) of its chords -- so the
polyline is within `tol` px of the curve both ways, the same count every
time (no adaptive recursion, no data-dependent branching between runs).
Each MOVETO starts a new part, CLOSEPOLY closes the part (its first point
repeated at the end); a non-finite vertex breaks the part, as matplotlib's
NaN removal does. `flatten_many` does the same for many paths in parallel.

Affines are 3x3 matrices that act on column vectors, as matplotlib's
Affine2D (its `get_matrix()`): `affine(scale, rotate_deg, translate)`
scales, then rotates (degrees, counterclockwise in a y-up frame), then
translates; `apply(M, xy)` maps points.
"""
import numpy as np

from ._jit import njit, prange

STOP = 0
MOVETO = 1
LINETO = 2
CURVE3 = 3
CURVE4 = 4
CLOSEPOLY = 79

MAX_SUB = 1 << 16                   # chords a curve at most (a curve ~10^9 px long)


class Path:
    """matplotlib's Path: vertices (n, 2) float64 and codes (n,) uint8 or
    None (a polyline). closed=True with codes None makes the last vertex a
    CLOSEPOLY (ignored, as matplotlib's `Path(..., closed=True)`)."""

    __slots__ = ('vertices', 'codes')

    def __init__(self, vertices, codes=None, closed=False):
        v = np.array(vertices, dtype=np.float64)
        if v.size == 0:
            v = v.reshape(0, 2)
        if v.ndim != 2 or v.shape[1] != 2:
            raise ValueError('vertices must be (N, 2), not %s' % (v.shape,))
        if codes is not None and len(v):
            codes = np.array(codes, dtype=np.uint8)
            if codes.shape != (len(v),):
                raise ValueError('codes must be (N,) for (N, 2) vertices')
            if codes[0] != MOVETO:
                raise ValueError('the first code must be MOVETO (1)')
        elif closed and len(v):
            codes = np.full(len(v), LINETO, np.uint8)
            codes[0] = MOVETO
            codes[-1] = CLOSEPOLY
        else:
            codes = None
        self.vertices = v
        self.codes = codes

    @classmethod
    def create_closed(cls, vertices):
        """A closed polygon through vertices (the CLOSEPOLY added, as
        matplotlib's Path._create_closed)."""
        v = np.asarray(vertices, dtype=np.float64)
        return cls(np.concatenate([v, v[:1]]), closed=True)

    def __len__(self):
        return len(self.vertices)

    def __bool__(self):
        return bool(len(self.vertices))

    def __repr__(self):
        return 'Path(%r, %r)' % (self.vertices, self.codes)

    def full_codes(self):
        """The codes, those of a polyline (MOVETO, LINETO ...) if None."""
        if self.codes is not None:
            return self.codes
        c = np.full(len(self.vertices), LINETO, np.uint8)
        if len(c):
            c[0] = MOVETO
        return c

    def transformed(self, M):
        """This path under the 3x3 affine M (codes kept)."""
        return Path._raw(apply(M, self.vertices), self.codes)

    def flatten(self, tol=0.1):
        """flatten() of this path (its vertices taken as pixels)."""
        return flatten(self.vertices, self.codes, tol)

    @classmethod
    def _raw(cls, vertices, codes):
        p = cls.__new__(cls)
        p.vertices = vertices
        p.codes = codes
        return p


def make_compound_path(*paths):
    """One path of several (matplotlib's Path.make_compound_path: codes
    made explicit, STOPs dropped)."""
    if not paths:
        return Path(np.empty((0, 2)))
    v = np.concatenate([p.vertices for p in paths])
    c = np.concatenate([p.full_codes() for p in paths]) if len(v) else np.empty(0, np.uint8)
    keep = c != STOP
    return Path._raw(v[keep], c[keep])


# ------------------------------------------------------------- affines

def affine(scale=1.0, rotate_deg=0.0, translate=(0.0, 0.0)):
    """3x3: scale (a number or (sx, sy)), then rotate by rotate_deg
    (counterclockwise with y up), then translate."""
    sx, sy = (scale, scale) if np.ndim(scale) == 0 else scale
    M = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])
    if rotate_deg:
        M = rotation(rotate_deg) @ M
    M[0, 2] += translate[0]
    M[1, 2] += translate[1]
    return M


def rotation(deg):
    """3x3 rotation by deg degrees (matplotlib's Affine2D().rotate_deg)."""
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def scaling(sx, sy=None):
    """3x3 scaling (Affine2D().scale)."""
    return np.array([[sx, 0.0, 0.0], [0.0, sx if sy is None else sy, 0.0], [0.0, 0.0, 1.0]])


def translation(tx, ty):
    """3x3 translation (Affine2D().translate)."""
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def chain(*Ms):
    """The affine applying Ms[0] first, then Ms[1] ... (matplotlib's
    `A + B`, or Affine2D().a().b() method chains)."""
    out = np.eye(3)
    for M in Ms:
        out = np.asarray(M, np.float64) @ out
    return out


def apply(M, xy):
    """Points (n, 2) under the 3x3 affine M."""
    xy = np.asarray(xy, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    out = np.empty(xy.shape, np.float64)
    out[..., 0] = M[0, 0] * xy[..., 0] + M[0, 1] * xy[..., 1] + M[0, 2]
    out[..., 1] = M[1, 0] * xy[..., 0] + M[1, 1] * xy[..., 1] + M[1, 2]
    return out


# ------------------------------------------------------------ unit shapes

def unit_rectangle():
    """The unit square (0, 0) - (1, 1), closed."""
    return Path([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], closed=True)


def unit_regular_polygon(n):
    """A regular n-gon in the unit circle, pointing up, closed."""
    theta = (2 * np.pi / n) * np.arange(n + 1) + np.pi / 2
    return Path(np.column_stack((np.cos(theta), np.sin(theta))), closed=True)


def unit_regular_star(n, innerCircle=0.5):
    """A regular n-pointed star, radius 1 (inner radius innerCircle), closed."""
    ns2 = n * 2
    theta = 2 * np.pi / ns2 * np.arange(ns2 + 1)
    theta += np.pi / 2.0
    r = np.ones(ns2 + 1)
    r[1::2] = innerCircle
    return Path((r * np.vstack((np.cos(theta), np.sin(theta)))).T, closed=True)


def unit_regular_asterisk(n):
    """An n-armed asterisk (a star with inner radius 0)."""
    return unit_regular_star(n, 0.0)


_MAGIC = 0.2652031
_SQRTHALF = np.sqrt(0.5)
_MAGIC45 = _SQRTHALF * _MAGIC
_CIRCLE = np.array([
    [0.0, -1.0], [_MAGIC, -1.0], [_SQRTHALF - _MAGIC45, -_SQRTHALF - _MAGIC45],
    [_SQRTHALF, -_SQRTHALF], [_SQRTHALF + _MAGIC45, -_SQRTHALF + _MAGIC45], [1.0, -_MAGIC],
    [1.0, 0.0], [1.0, _MAGIC], [_SQRTHALF + _MAGIC45, _SQRTHALF - _MAGIC45],
    [_SQRTHALF, _SQRTHALF], [_SQRTHALF - _MAGIC45, _SQRTHALF + _MAGIC45], [_MAGIC, 1.0],
    [0.0, 1.0], [-_MAGIC, 1.0], [-_SQRTHALF + _MAGIC45, _SQRTHALF + _MAGIC45],
    [-_SQRTHALF, _SQRTHALF], [-_SQRTHALF - _MAGIC45, _SQRTHALF - _MAGIC45], [-1.0, _MAGIC],
    [-1.0, 0.0], [-1.0, -_MAGIC], [-_SQRTHALF - _MAGIC45, -_SQRTHALF + _MAGIC45],
    [-_SQRTHALF, -_SQRTHALF], [-_SQRTHALF + _MAGIC45, -_SQRTHALF - _MAGIC45], [-_MAGIC, -1.0],
    [0.0, -1.0], [0.0, -1.0]])


def circle(center=(0.0, 0.0), radius=1.0):
    """A circle of 8 cubic Beziers (matplotlib's Path.circle), closed."""
    codes = np.full(26, CURVE4, np.uint8)
    codes[0] = MOVETO
    codes[-1] = CLOSEPOLY
    return Path(_CIRCLE * radius + np.asarray(center, np.float64), codes)


def unit_circle():
    """The unit circle (matplotlib's Path.unit_circle)."""
    return circle((0.0, 0.0), 1.0)


def unit_circle_righthalf():
    """The right half of the unit circle, closed by its diameter."""
    v = np.concatenate([_CIRCLE[:13], [[0.0, -1.0]]])
    codes = np.full(14, CURVE4, np.uint8)
    codes[0] = MOVETO
    codes[-1] = CLOSEPOLY
    return Path(v, codes)


def arc(theta1, theta2, n=None, is_wedge=False):
    """The unit circle's arc from theta1 to theta2 degrees, cubic Beziers
    (matplotlib's Path.arc: theta2 unwrapped to within 360 of theta1, a
    whole number of turns a full circle; n segments, by default 2^k for
    a span over k quarter turns); is_wedge: from and back to the centre."""
    halfpi = np.pi * 0.5
    eta1 = theta1
    n_turns = (theta2 - theta1) / 360
    nearest_turn = np.rint(n_turns)
    if nearest_turn != 0 and abs(n_turns - nearest_turn) <= 1e-12:
        eta2 = theta1 + 360
    else:
        eta2 = theta2 - 360 * np.floor(n_turns)
    eta1, eta2 = np.deg2rad([eta1, eta2])
    if n is None:
        n = int(2 ** np.ceil((eta2 - eta1) / halfpi))
    if n < 1:
        raise ValueError('n must be >= 1 or None')
    deta = (eta2 - eta1) / n
    t = np.tan(0.5 * deta)
    alpha = np.sin(deta) * (np.sqrt(4.0 + 3.0 * t * t) - 1) / 3.0
    steps = np.linspace(eta1, eta2, n + 1, True)
    cos_eta = np.cos(steps)
    sin_eta = np.sin(steps)
    xA, yA = cos_eta[:-1], sin_eta[:-1]
    xA_dot, yA_dot = -yA, xA
    xB, yB = cos_eta[1:], sin_eta[1:]
    xB_dot, yB_dot = -yB, xB
    if is_wedge:
        length = n * 3 + 4
        vertices = np.zeros((length, 2), float)
        codes = np.full(length, CURVE4, np.uint8)
        vertices[1] = [xA[0], yA[0]]
        codes[0:2] = [MOVETO, LINETO]
        codes[-2:] = [LINETO, CLOSEPOLY]
        off = 2
        end = length - 2
    else:
        length = n * 3 + 1
        vertices = np.empty((length, 2), float)
        codes = np.full(length, CURVE4, np.uint8)
        vertices[0] = [xA[0], yA[0]]
        codes[0] = MOVETO
        off = 1
        end = length
    vertices[off:end:3, 0] = xA + alpha * xA_dot
    vertices[off:end:3, 1] = yA + alpha * yA_dot
    vertices[off + 1:end:3, 0] = xB - alpha * xB_dot
    vertices[off + 1:end:3, 1] = yB - alpha * yB_dot
    vertices[off + 2:end:3, 0] = xB
    vertices[off + 2:end:3, 1] = yB
    return Path(vertices, codes)


def wedge(theta1, theta2, n=None):
    """The unit circle's wedge from theta1 to theta2 degrees (Path.wedge)."""
    return arc(theta1, theta2, n, True)


# ------------------------------------------------------------ flattening

@njit(cache=True, nogil=True)
def _n_quad(x0, y0, x1, y1, x2, y2, tol):
    """Chords for a quadratic Bezier within tol: |B''| = 2|P0 - 2P1 + P2|."""
    dx, dy = x0 - 2.0 * x1 + x2, y0 - 2.0 * y1 + y2
    d = np.sqrt(dx * dx + dy * dy)
    n = np.ceil(np.sqrt(d / (4.0 * tol)))
    return np.int64(min(max(n, 1.0), MAX_SUB))


@njit(cache=True, nogil=True)
def _n_cubic(x0, y0, x1, y1, x2, y2, x3, y3, tol):
    """Chords for a cubic Bezier within tol: |B''| <= 6 max|second difference|."""
    ax, ay = x0 - 2.0 * x1 + x2, y0 - 2.0 * y1 + y2
    bx, by = x1 - 2.0 * x2 + x3, y1 - 2.0 * y2 + y3
    d = max(np.sqrt(ax * ax + ay * ay), np.sqrt(bx * bx + by * by))
    n = np.ceil(np.sqrt(0.75 * d / tol))
    return np.int64(min(max(n, 1.0), MAX_SUB))


@njit(cache=True, nogil=True)
def _fin(v, i):
    return np.isfinite(v[i, 0]) and np.isfinite(v[i, 1])


@njit(cache=True, nogil=True)
def _walk(v, c, a, b, tol, xy, offsets, closed, p, k, write, cap):
    """Flatten the path v[a:b], codes c[a:b], writing points from xy[p]
    and parts from offsets[k] / closed[k] (write) or only counting them.
    Parts of fewer than 2 points are dropped (a dropped part's point is
    written only below cap, the end of this path's points). Returns (p, k)
    after it."""
    ps = p                                  # the open part's first point (output)
    open_ = False                           # a part is being written
    cur = False                             # there is a current point
    cx = cy = sx = sy = 0.0                 # current point, part's start point
    i = a
    while i < b:
        code = c[i]
        if code == STOP:
            break
        if code == MOVETO or code == LINETO:
            nxt = i + 1
            if not _fin(v, i):                      # NaN: the part ends here
                if open_:
                    if p - ps >= 2:
                        if write:
                            offsets[k] = ps
                            closed[k] = False
                        k += 1
                    else:
                        p = ps
                open_ = False
                cur = False
            elif code == MOVETO or not cur:
                if open_:
                    if p - ps >= 2:
                        if write:
                            offsets[k] = ps
                            closed[k] = False
                        k += 1
                    else:
                        p = ps
                ps = p
                if write and p < cap:               # (a lone last point: dropped)
                    xy[p, 0], xy[p, 1] = v[i, 0], v[i, 1]
                p += 1
                cx, cy = v[i, 0], v[i, 1]
                sx, sy = cx, cy
                open_ = True
                cur = True
            else:
                if not open_:                       # after a CLOSEPOLY: from its start
                    ps = p
                    if write:
                        xy[p, 0], xy[p, 1] = cx, cy
                    p += 1
                    sx, sy = cx, cy
                    open_ = True
                if write:
                    xy[p, 0], xy[p, 1] = v[i, 0], v[i, 1]
                p += 1
                cx, cy = v[i, 0], v[i, 1]
            i = nxt
        elif code == CURVE3 or code == CURVE4:
            m = 2 if code == CURVE3 else 3
            if i + m > b:
                break
            ok = cur
            for q in range(m):
                ok = ok and _fin(v, i + q)
            if not ok:                              # a curve with a NaN: skipped
                if open_:
                    if p - ps >= 2:
                        if write:
                            offsets[k] = ps
                            closed[k] = False
                        k += 1
                    else:
                        p = ps
                open_ = False
                cur = _fin(v, i + m - 1)
                if cur:
                    cx, cy = v[i + m - 1, 0], v[i + m - 1, 1]
                i += m
                continue
            if not open_:
                ps = p
                if write:
                    xy[p, 0], xy[p, 1] = cx, cy
                p += 1
                sx, sy = cx, cy
                open_ = True
            x0, y0 = cx, cy
            if code == CURVE3:
                x1, y1 = v[i, 0], v[i, 1]
                x2, y2 = v[i + 1, 0], v[i + 1, 1]
                n = _n_quad(x0, y0, x1, y1, x2, y2, tol)
                if write:
                    for j in range(1, n):
                        t = j / n
                        u = 1.0 - t
                        xy[p, 0] = u * u * x0 + 2.0 * u * t * x1 + t * t * x2
                        xy[p, 1] = u * u * y0 + 2.0 * u * t * y1 + t * t * y2
                        p += 1
                    xy[p, 0], xy[p, 1] = x2, y2
                    p += 1
                else:
                    p += n
                cx, cy = x2, y2
            else:
                x1, y1 = v[i, 0], v[i, 1]
                x2, y2 = v[i + 1, 0], v[i + 1, 1]
                x3, y3 = v[i + 2, 0], v[i + 2, 1]
                n = _n_cubic(x0, y0, x1, y1, x2, y2, x3, y3, tol)
                if write:
                    for j in range(1, n):
                        t = j / n
                        u = 1.0 - t
                        b0, b1, b2, b3 = u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t
                        xy[p, 0] = b0 * x0 + b1 * x1 + b2 * x2 + b3 * x3
                        xy[p, 1] = b0 * y0 + b1 * y1 + b2 * y2 + b3 * y3
                        p += 1
                    xy[p, 0], xy[p, 1] = x3, y3
                    p += 1
                else:
                    p += n
                cx, cy = x3, y3
            i += m
        elif code == CLOSEPOLY:
            if open_:
                if cx != sx or cy != sy:            # back to the start, once
                    if write:
                        xy[p, 0], xy[p, 1] = sx, sy
                    p += 1
                if p - ps >= 2:
                    if write:
                        offsets[k] = ps
                        closed[k] = True
                    k += 1
                else:
                    p = ps
                open_ = False
                cx, cy = sx, sy
            i += 1
        else:
            i += 1                                  # (not a code: skipped)
    if open_:
        if p - ps >= 2:
            if write:
                offsets[k] = ps
                closed[k] = False
            k += 1
        else:
            p = ps
    return p, k


def _codes_for(vertices, codes):
    if codes is None:
        c = np.full(len(vertices), LINETO, np.uint8)
        if len(c):
            c[0] = MOVETO
        return c
    return np.ascontiguousarray(codes, dtype=np.uint8)


def flatten(vertices_px, codes=None, tol=0.1):
    """A path (pixels; codes matplotlib's or None) as polylines within tol
    px of it: (xy (m, 2) float64, offsets (k+1) int64, closed (k,) bool).
    Part j is xy[offsets[j]:offsets[j+1]]; a closed part ends on its
    first point."""
    v = np.ascontiguousarray(vertices_px, dtype=np.float64).reshape(-1, 2)
    c = _codes_for(v, codes)
    tol = float(tol)
    if not tol > 0:
        raise ValueError('tol must be > 0')
    dummy2 = np.empty((0, 2), np.float64)
    dummy1 = np.empty(0, np.int64)
    dummyb = np.empty(0, np.bool_)
    n, k = _walk(v, c, 0, len(v), tol, dummy2, dummy1, dummyb, 0, 0, False, 0)
    xy = np.empty((n, 2), np.float64)
    offsets = np.empty(k + 1, np.int64)
    closed = np.empty(k, np.bool_)
    _walk(v, c, 0, len(v), tol, xy, offsets, closed, 0, 0, True, n)
    offsets[k] = n
    return xy, offsets, closed


@njit(cache=True, nogil=True, parallel=True)
def _count_many(v, c, po, tol, npts, nparts):
    d2 = np.empty((0, 2), np.float64)
    d1 = np.empty(0, np.int64)
    db = np.empty(0, np.bool_)
    for j in prange(po.size - 1):
        n, k = _walk(v, c, po[j], po[j + 1], tol, d2, d1, db, 0, 0, False, 0)
        npts[j + 1] = n
        nparts[j + 1] = k


@njit(cache=True, nogil=True, parallel=True)
def _fill_many(v, c, po, tol, p0, k0, xy, offsets, closed):
    for j in prange(po.size - 1):
        _walk(v, c, po[j], po[j + 1], tol, xy, offsets, closed, p0[j], k0[j], True,
              p0[j + 1])


def flatten_many(vertices_px, codes, path_offsets, tol=0.1):
    """Many paths flattened at once (in parallel): path j is
    vertices_px[path_offsets[j]:path_offsets[j+1]] with its codes (codes
    one array for all, or None: every path a polyline). Returns (xy,
    offsets, closed, path_parts): parts as flatten's, those of path j
    being parts path_parts[j] .. path_parts[j+1]-1."""
    v = np.ascontiguousarray(vertices_px, dtype=np.float64).reshape(-1, 2)
    po = np.ascontiguousarray(path_offsets, dtype=np.int64)
    if codes is None:
        c = np.full(len(v), LINETO, np.uint8)
        c[po[:-1][po[:-1] < len(v)]] = MOVETO
    else:
        c = np.ascontiguousarray(codes, dtype=np.uint8)
    npaths = max(po.size - 1, 0)
    npts = np.zeros(npaths + 1, np.int64)
    nparts = np.zeros(npaths + 1, np.int64)
    if npaths:
        _count_many(v, c, po, float(tol), npts, nparts)
    np.cumsum(npts, out=npts)
    np.cumsum(nparts, out=nparts)
    xy = np.empty((npts[-1], 2), np.float64)
    offsets = np.empty(nparts[-1] + 1, np.int64)
    closed = np.empty(nparts[-1], np.bool_)
    if npaths:
        _fill_many(v, c, po, float(tol), npts, nparts, xy, offsets, closed)
    offsets[-1] = npts[-1]
    return xy, offsets, closed, nparts


@njit(cache=True, nogil=True, parallel=True)
def _translate_many(xy, centers, out):
    m = xy.shape[0]
    for i in prange(centers.shape[0]):
        cx, cy = centers[i, 0], centers[i, 1]
        b = i * m
        for q in range(m):
            out[b + q, 0] = xy[q, 0] + cx
            out[b + q, 1] = xy[q, 1] + cy


def translate_many(xy, offsets, centers):
    """Polylines (xy, offsets) copied to every centre (n, 2): (xy (n m, 2),
    offsets (n k + 1)) -- copy i's parts are i k .. i k + k - 1."""
    xy = np.ascontiguousarray(xy, dtype=np.float64)
    centers = np.ascontiguousarray(centers, dtype=np.float64).reshape(-1, 2)
    offsets = np.asarray(offsets, np.int64)
    n, m, k = centers.shape[0], xy.shape[0], offsets.size - 1
    out = np.empty((n * m, 2), np.float64)
    if n and m:
        _translate_many(xy, centers, out)
    offs = np.empty(n * k + 1, np.int64)
    if k > 0:
        offs[:-1] = (offsets[:-1][None, :] + m * np.arange(n)[:, None]).ravel()
    offs[-1] = n * m
    return out, offs
