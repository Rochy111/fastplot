"""Natural Earth geometry -- coastlines, borders, states, lakes, rivers, land,
ocean, US counties -- as flat numpy arrays, the shape the line and fill
kernels take: one (n, 2) float64 array of lon/lat vertices for a whole
feature, and offsets saying where each part (or ring, or polygon) starts.

    c = lines('coastline', '50m', bbox=(-130, -60, 20, 55))
    c.xy[c.offsets[k]:c.offsets[k + 1]]      # part k, (lon, lat) rows
    p = polygons('land', scale_for(extent))
    p.edges()                                # every ring as Lines

110m and 50m of every feature ship with fastplot (fastplot/data, made by
tools/make_features.py). 10m, and the counties that exist only at 10m,
are converted on first use from the shapefile cartopy already keeps
($CARTOPY_DATA_DIR, or ~/.local/share/cartopy, shapefiles/natural_earth/
<category>/ne_10m_<name>.shp), or else from Natural Earth's own download,
and the result is cached as .npz in $FASTPLOT_DATA_DIR or the platform's
cache directory -- so cartopy, shapely and pyshp are never needed. The
shapefile reader here is plain numpy over the ESRI layout.

Coordinates are stored as integer micro-degrees (1e-6 deg, ~0.1 m: far
finer than the float32 the drawing kernels use), delta-coded and
zigzagged, each byte of the 32-bit deltas in a plane of its own, then
deflated: neighbouring vertices are close, so the high planes are almost
all zero and the whole bundle stays ~2 MB. Every loaded feature is kept
in memory per (name, scale); its arrays are read-only, being shared.

Natural Earth is in the public domain (https://www.naturalearthdata.com/
about/terms-of-use/).
"""
import os
import struct
import sys
import threading
import warnings
from collections import namedtuple

import numpy as np

SCALES = ('110m', '50m', '10m')
URL = 'https://naturalearth.s3.amazonaws.com/{scale}_{category}/ne_{scale}_{name}.zip'
FORMAT = 1                      # .npz layout version: a cache of another is rebuilt
UNIT = 1e6                      # stored coordinates: integer multiples of 1e-6 degree

# fastplot's name -> (Natural Earth category, Natural Earth name, scales it exists at)
FEATURES = {
    'coastline': ('physical', 'coastline', SCALES),
    'borders': ('cultural', 'admin_0_boundary_lines_land', SCALES),
    'states': ('cultural', 'admin_1_states_provinces_lines', SCALES),
    'states_lakes': ('cultural', 'admin_1_states_provinces_lakes', SCALES),
    'lakes': ('physical', 'lakes', SCALES),
    'rivers': ('physical', 'rivers_lake_centerlines', SCALES),
    'land': ('physical', 'land', SCALES),
    'ocean': ('physical', 'ocean', SCALES),
    'counties': ('cultural', 'admin_2_counties', ('10m',)),
}

_POLYGONS = ('states_lakes', 'lakes', 'land', 'ocean', 'counties')
_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
_CACHE = {}                     # (name, scale) -> Lines | Polygons, and (name, scale, 'edges')
_LOCK = threading.RLock()


class Lines(namedtuple('Lines', 'xy offsets bbox')):
    """Polylines: `xy` (n, 2) float64 lon/lat, part k being
    xy[offsets[k]:offsets[k + 1]] (`offsets` int64, m + 1), and `bbox`
    (m, 4) float64, each part's (lon0, lat0, lon1, lat1)."""
    __slots__ = ()

    @property
    def parts(self):
        return len(self.offsets) - 1


class Polygons(namedtuple('Polygons', 'xy ring_offsets poly_offsets bbox')):
    """Polygons with holes: ring j is xy[ring_offsets[j]:ring_offsets[j + 1]],
    polygon k is rings poly_offsets[k] .. poly_offsets[k + 1] - 1, the first
    its exterior and the rest its holes, and `bbox` (p, 4) each polygon's
    (lon0, lat0, lon1, lat1). Rings keep the shapefile's winding (exterior
    clockwise, holes counter-clockwise) and are closed (last vertex = first)."""
    __slots__ = ()

    @property
    def parts(self):
        return len(self.poly_offsets) - 1

    def edges(self):
        """Every ring as a part of Lines (outlines), sharing `xy`."""
        return Lines(self.xy, self.ring_offsets, _readonly(_part_bbox(self.xy, self.ring_offsets)))


def names():
    """The feature names lines() and polygons() take."""
    return list(FEATURES)


def kind(name):
    """'lines' or 'polygons': what a feature is at its source."""
    _feature(name)
    return 'polygons' if name in _POLYGONS else 'lines'


def lines(name, scale='50m', bbox=None):
    """A feature as Lines: its polylines, or for a polygon feature (land,
    lakes, states_lakes ...) the edges of every ring. `bbox` (lon0, lon1,
    lat0, lat1) keeps only the parts whose bounding box meets it (see
    select)."""
    scale = _scale(name, scale)
    if kind(name) == 'lines':
        out = _load(name, scale)
    else:
        key = (name, scale, 'edges')
        with _LOCK:
            out = _CACHE.get(key)
            if out is None:
                out = _CACHE[key] = _load(name, scale).edges()
    return out if bbox is None else select(out, bbox)


def polygons(name, scale='50m', bbox=None):
    """A polygon feature (land, ocean, lakes, states_lakes, counties) as
    Polygons; `bbox` keeps the polygons whose bounding box meets it."""
    scale = _scale(name, scale)
    if kind(name) != 'polygons':
        raise ValueError('%r is a line feature (lines() draws it); polygon features: %s'
                         % (name, ', '.join(_POLYGONS)))
    out = _load(name, scale)
    return out if bbox is None else select(out, bbox)


def scale_for(extent):
    """The Natural Earth scale cartopy's coastlines(resolution='auto') -- its
    AdaptiveScaler('110m', (('50m', 50), ('10m', 15))) -- picks for a map
    showing `extent` (lon0, lon1, lat0, lat1, degrees): the smaller of the
    longitude and latitude spans decides, '10m' up to 15 degrees, '50m' up
    to 50, '110m' beyond (and for a span of 0 or NaN, or extent None). As
    in select(), lon0 > lon1 is a box across the antimeridian, spanning
    lon1 - lon0 + 360 (cartopy, which never gets such an extent, would
    take |lon1 - lon0|)."""
    if extent is None:
        return '110m'
    lon0, lon1, lat0, lat1 = (float(v) for v in extent)
    width = lon1 - lon0 if lon1 >= lon0 else lon1 - lon0 + 360.0
    span = min(width, abs(lat1 - lat0))
    if span != 0 and span <= 15:                # NaN fails both: '110m'
        return '10m'
    if span != 0 and span <= 50:
        return '50m'
    return '110m'


def select(feature, bbox):
    """The parts of Lines, or polygons of Polygons, whose bounding box meets
    `bbox` (lon0, lon1, lat0, lat1), as a new feature of the same kind:
    a superset of the geometries that truly intersect it, decided over the
    part boxes at once. lon0 > lon1 means the box crosses the antimeridian;
    longitudes may be -180..180 or 0..360 (a part at -100 meets 250..300),
    and a box 360 or more wide keeps every longitude."""
    keep = _hits(feature.bbox, bbox)
    if keep.all():
        return feature
    k = np.flatnonzero(keep)
    if isinstance(feature, Lines):
        start, stop = feature.offsets[k], feature.offsets[k + 1]
        return Lines(_readonly(feature.xy[_ranges(start, stop)]), _readonly(_offsets(stop - start)),
                     _readonly(feature.bbox[k]))
    r = _ranges(feature.poly_offsets[k], feature.poly_offsets[k + 1])
    start, stop = feature.ring_offsets[r], feature.ring_offsets[r + 1]
    return Polygons(_readonly(feature.xy[_ranges(start, stop)]), _readonly(_offsets(stop - start)),
                    _readonly(_offsets(feature.poly_offsets[k + 1] - feature.poly_offsets[k])),
                    _readonly(feature.bbox[k]))


def _hits(bb, bbox):
    """Which of the boxes bb (m, 4: lon0, lat0, lon1, lat1, longitudes in
    -180..180) meet the query box (lon0, lon1, lat0, lat1)."""
    lon0, lon1, lat0, lat1 = (float(v) for v in bbox)
    if not np.isfinite([lon0, lon1, lat0, lat1]).all():
        raise ValueError('bbox %r: not finite' % (bbox,))
    lat0, lat1 = min(lat0, lat1), max(lat0, lat1)
    keep = (bb[:, 1] <= lat1) & (bb[:, 3] >= lat0)
    width = lon1 - lon0 if lon1 >= lon0 else lon1 - lon0 + 360.0
    if width < 360:
        q0 = (lon0 + 180.0) % 360.0 - 180.0     # the box as [q0, q0 + width], q0 in [-180, 180)
        q1 = q0 + width
        a, b = bb[:, 0], bb[:, 2]
        on = (a <= q1) & (b >= q0)
        on |= a <= q1 - 360.0                   # the box's part past +180, from -180
        on |= b >= q0 + 360.0                   # (only when q0 = -180 and b = 180)
        keep &= on
    return keep


def _ranges(start, stop):
    """np.concatenate([np.arange(a, b) for a, b in zip(start, stop)]), at once."""
    n = (stop - start).astype(np.int64)
    before = np.cumsum(n) - n
    return np.arange(int(n.sum()), dtype=np.int64) + np.repeat(start - before, n)


def _offsets(counts):
    out = np.zeros(len(counts) + 1, np.int64)
    np.cumsum(counts, out=out[1:])
    return out


def _readonly(a):
    a.setflags(write=False)
    return a


def _part_bbox(xy, offsets):
    """(lon0, lat0, lon1, lat1) of each part (none empty), (m, 4)."""
    m = len(offsets) - 1
    out = np.empty((m, 4))
    if m:
        s = offsets[:-1]
        out[:, :2] = np.minimum.reduceat(xy, s, axis=0)
        out[:, 2:] = np.maximum.reduceat(xy, s, axis=0)
    return out


# -- loading ------------------------------------------------------------------

def _feature(name):
    try:
        return FEATURES[name]
    except (KeyError, TypeError):
        raise ValueError('no feature %r; there are %s' % (name, ', '.join(FEATURES))) from None


def _scale(name, scale):
    """`scale` checked; for a feature missing at it (counties below 10m),
    the nearest finer scale it has."""
    have = _feature(name)[2]
    if scale not in SCALES:
        raise ValueError('scale %r: one of %s' % (scale, ', '.join(SCALES)))
    for s in SCALES[SCALES.index(scale):]:
        if s in have:
            return s
    raise ValueError('%r exists at %s only' % (name, ', '.join(have)))   # (not reached)


def cache_dir():
    """Where converted 10m features are kept: $FASTPLOT_DATA_DIR, else the
    platform's cache directory (%LOCALAPPDATA%\\fastplot, ~/Library/Caches/
    fastplot, $XDG_CACHE_HOME/fastplot or ~/.cache/fastplot)."""
    d = os.environ.get('FASTPLOT_DATA_DIR')
    if d:
        return d
    home = os.path.expanduser('~')
    if sys.platform == 'win32':
        base = os.environ.get('LOCALAPPDATA') or os.path.join(home, 'AppData', 'Local')
    elif sys.platform == 'darwin':
        base = os.path.join(home, 'Library', 'Caches')
    else:
        base = os.environ.get('XDG_CACHE_HOME') or os.path.join(home, '.cache')
    return os.path.join(base, 'fastplot')


def _cartopy_dirs():
    """Cartopy's Natural Earth directories: $CARTOPY_DATA_DIR's (pre-existing
    data), then the one it downloads to ($XDG_DATA_HOME or ~/.local/share)."""
    out = []
    if os.environ.get('CARTOPY_DATA_DIR'):
        out.append(os.environ['CARTOPY_DATA_DIR'])
    base = os.environ.get('XDG_DATA_HOME') or os.path.join(os.path.expanduser('~'), '.local', 'share')
    out.append(os.path.join(base, 'cartopy'))
    return [os.path.join(d, 'shapefiles', 'natural_earth') for d in out]


def _load(name, scale):
    """The whole feature, from memory, the bundled data, the cache directory
    or its source, in that order."""
    key = (name, scale)
    with _LOCK:
        out = _CACHE.get(key)
        if out is None:
            out = _CACHE[key] = _read(name, scale)
        return out


def _read(name, scale):
    fn = 'ne_%s_%s.npz' % (scale, name)
    for path in (os.path.join(_DATA, fn), os.path.join(cache_dir(), fn)):
        if os.path.exists(path):
            try:
                with np.load(path, allow_pickle=False) as z:
                    if int(z['format']) == FORMAT:
                        return decode(z)
            except Exception as e:              # damaged cache: made again
                warnings.warn('fastplot: %s unreadable (%s), rebuilding it' % (path, e))
    data = encode(*_from_shp(read_shp(_source(name, scale)), kind(name)), source=fn[:-4])
    _save(os.path.join(cache_dir(), fn), data)
    return decode(data)


def _save(path, data):
    """np.savez_compressed to `path` through a temporary file and os.replace,
    so a reader never sees half a file; a cache that cannot be written is
    only a warning."""
    tmp = None
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = '%s.%d.%d.tmp' % (path, os.getpid(), threading.get_ident())
        with open(tmp, 'wb') as f:
            np.savez_compressed(f, **data)
        os.replace(tmp, path)
    except OSError as e:
        warnings.warn('fastplot: could not cache %s (%s)' % (path, e))
        if tmp is not None and os.path.exists(tmp):
            os.remove(tmp)


def _source(name, scale):
    """The bytes of a feature's .shp: a shapefile or zip put in the cache
    directory by hand, cartopy's copy, or Natural Earth's download."""
    category, ne, _ = _feature(name)
    base = 'ne_%s_%s' % (scale, ne)
    here = cache_dir()
    for path in [os.path.join(here, base + '.shp')] + [
            os.path.join(d, category, base + '.shp') for d in _cartopy_dirs()]:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return f.read()
    path = os.path.join(here, base + '.zip')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return _shp_from_zip(f.read(), base, path)
    url = URL.format(scale=scale, category=category, name=ne)
    try:
        data = _download(url)
    except Exception as e:
        raise OSError('fastplot: Natural Earth %s %s could not be downloaded from %s (%s). '
                      'Download it by hand into %s, or set FASTPLOT_DATA_DIR to a directory '
                      'holding %s.zip (or CARTOPY_DATA_DIR to a cartopy data directory with '
                      'shapefiles/natural_earth/%s/%s.shp).'
                      % (scale, name, url, e, here, base, category, base)) from e
    return _shp_from_zip(data, base, url)


def _shp_from_zip(data, base, where):
    import io
    import zipfile
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        shps = [n for n in z.namelist() if n.lower().endswith('.shp')]
        pick = [n for n in shps if os.path.basename(n) == base + '.shp'] or shps
        if not pick:
            raise OSError('fastplot: %s holds no .shp' % where)
        return z.read(pick[0])


def _download(url, retries=3, timeout=120):
    """The bytes at url (urllib, through any proxy set)."""
    import urllib.error
    import urllib.request
    err = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'fastplot'})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                raise
            err = e
        except OSError as e:
            err = e
    raise err


# -- storage ------------------------------------------------------------------

def encode(what, xy, offsets, poly_offsets=None, source=''):
    """A feature as the arrays of its .npz: micro-degree coordinates,
    delta-coded along the vertices (one stream per axis), zigzagged
    (sign to the low bit) and split into four byte planes, so deflate
    finds the high planes nearly empty; part and polygon sizes as counts."""
    q = np.round(np.asarray(xy, np.float64) * UNIT).astype(np.int64)
    if len(q) and np.abs(q).max() >= 2 ** 30:      # (deltas then fit 32 bits)
        raise ValueError('coordinates beyond +-1073 degrees')
    d = np.diff(q.T, axis=1, prepend=np.zeros((2, 1), np.int64))     # (2, n)
    z = np.ascontiguousarray(((d << 1) ^ (d >> 63)).astype('<u4'))
    out = {'format': np.int32(FORMAT), 'kind': np.int8(what == 'polygons'),
           'coords': np.ascontiguousarray(z.view(np.uint8).reshape(2, -1, 4).transpose(0, 2, 1)),
           'counts': np.diff(offsets).astype(np.int32), 'source': np.str_(source)}
    if what == 'polygons':
        out['rings'] = np.diff(poly_offsets).astype(np.int32)
    return out


def decode(z):
    """Lines or Polygons from encode()'s arrays (or the .npz holding them)."""
    planes = z['coords']                                    # (2, 4, n) uint8
    u = np.ascontiguousarray(planes.transpose(0, 2, 1)).view('<u4')[..., 0]   # (2, n)
    d = ((u >> 1) ^ (np.uint32(0) - (u & 1))).view(np.int32)
    xy = np.empty((u.shape[1], 2))
    np.divide(np.cumsum(d, axis=1, dtype=np.int64).T, UNIT, out=xy)   # (the double nearest q 1e-6)
    _readonly(xy)
    offsets = _readonly(_offsets(z['counts']))
    if int(z['kind']):
        polys = _readonly(_offsets(z['rings']))
        return Polygons(xy, offsets, polys, _readonly(_poly_bbox(xy, offsets, polys)))
    return Lines(xy, offsets, _readonly(_part_bbox(xy, offsets)))


def _poly_bbox(xy, rings, polys):
    """Each polygon's box, over all its rings."""
    bb = _part_bbox(xy, rings)
    if len(polys) < 2:
        return np.empty((0, 4))
    s = polys[:-1]
    return np.hstack([np.minimum.reduceat(bb[:, :2], s, axis=0), np.maximum.reduceat(bb[:, 2:], s, axis=0)])


# -- shapefiles ---------------------------------------------------------------

Shp = namedtuple('Shp', 'shape_type xy part_offsets record_offsets')
_SHP_TYPES = {3: 'polyline', 5: 'polygon', 13: 'polyline', 15: 'polygon',
              23: 'polyline', 25: 'polygon'}


def read_shp(src):
    """The vertices of an ESRI shapefile (.shp) of polylines or polygons --
    shape types 3, 5 and their Z (13, 15) and M (23, 25) forms, whose Z
    and M values are skipped -- as Shp: `shape_type` (the header's), `xy`
    (n, 2) float64, part k = xy[part_offsets[k]:part_offsets[k + 1]] (empty
    parts dropped), and record i's parts part_offsets indices
    record_offsets[i] .. record_offsets[i + 1] - 1 (none for a null shape).
    `src` is a path or the file's bytes. Per the ESRI whitepaper (1998):
    the file code, length and record headers big-endian, all else little."""
    if isinstance(src, (str, os.PathLike)):
        with open(src, 'rb') as f:
            src = f.read()
    buf = bytes(src)
    if len(buf) < 100 or struct.unpack_from('>i', buf, 0)[0] != 9994:
        raise ValueError('not a shapefile (.shp): no file code 9994')
    end = min(2 * struct.unpack_from('>i', buf, 24)[0], len(buf))
    stype = struct.unpack_from('<i', buf, 32)[0]
    if stype not in _SHP_TYPES and stype != 0:
        raise ValueError('shapefile of shape type %d: only polylines and polygons '
                         '(3, 5, 13, 15, 23, 25) are read' % stype)
    chunks, starts, stops, nrec = [], [], [], []
    n = 0
    pos = 100
    while pos + 12 <= end:
        clen = struct.unpack_from('>i', buf, pos + 4)[0]
        c, pos = pos + 8, pos + 8 + 2 * clen
        t = struct.unpack_from('<i', buf, c)[0]
        if t == 0:                              # null shape
            nrec.append(0)
            continue
        if t != stype:
            raise ValueError('shapefile of type %d has a record of type %d at byte %d' % (stype, t, c - 8))
        limit = min(pos, len(buf))
        nparts, npoints = struct.unpack_from('<2i', buf, c + 36) if c + 44 <= limit else (-1, -1)
        p = c + 44 + 4 * nparts
        if nparts < 0 or npoints < 0 or p + 16 * npoints > limit:
            raise ValueError('shapefile record at byte %d is cut short' % (c - 8))
        part = np.frombuffer(buf, '<i4', nparts, c + 44).astype(np.int64)
        stop = np.append(part[1:], npoints)
        full = stop > part
        starts.append(part[full] + n)
        stops.append(stop[full] + n)
        nrec.append(int(full.sum()))
        chunks.append(np.frombuffer(buf, '<f8', 2 * npoints, p))
        n += npoints
    xy = np.concatenate(chunks).reshape(-1, 2) if chunks else np.empty((0, 2))
    start = np.concatenate(starts) if starts else np.empty(0, np.int64)
    stop = np.concatenate(stops) if stops else np.empty(0, np.int64)
    if len(start) and not (start[1:] == stop[:-1]).all():   # (parts not back to back)
        xy = xy[_ranges(start, stop)]
        stop = np.cumsum(stop - start)
    return Shp(stype, xy, np.append(np.int64(0), stop), _offsets(np.array(nrec, np.int64)))


def _from_shp(shp, what):
    """(what, xy, offsets[, poly_offsets]) of a feature from read_shp():
    polylines as they are, every part of every record; polygon records'
    rings grouped into polygons as pyshp (and so cartopy) groups them."""
    if what == 'lines':
        return 'lines', shp.xy, shp.part_offsets
    if _SHP_TYPES.get(shp.shape_type, 'polygon') != 'polygon':
        raise ValueError('a polygon feature from a shapefile of polylines')
    order, polys = organize(shp.xy, shp.part_offsets, shp.record_offsets)
    start, stop = shp.part_offsets[order], shp.part_offsets[order + 1]
    xy = shp.xy if (order == np.arange(len(order))).all() else shp.xy[_ranges(start, stop)]
    return 'polygons', xy, _offsets(stop - start), polys


def signed_area2(xy, offsets):
    """Twice each ring's signed area (rings of 2 vertices or more; none
    empty), positive counter-clockwise, as pyshp computes it: the sum of
    x_i (y_i+1 - y_i-1) for i = 1 .. n-1, with y_n = y_1."""
    s, e = offsets[:-1], offsets[1:]
    if not len(s):
        return np.zeros(0)
    x, y = xy[:, 0], xy[:, 1]
    g = np.arange(len(x))
    nxt = np.minimum(g + 1, len(x) - 1)
    nxt[e - 1] = np.minimum(s + 1, e - 1)       # the last vertex's next: the second
    t = x * (y[nxt] - y[np.maximum(g - 1, 0)])
    t[s] = 0                                    # (i from 1)
    return np.add.reduceat(t, s)


def organize(xy, ring_offsets, record_offsets):
    """Polygons of shapefile rings, pyshp's way (its organize_polygon_rings):
    in each record the clockwise rings are exteriors, the others holes; with
    one exterior every hole is its; with none, or no holes, every ring is a
    polygon of its own; else a hole goes to the exterior whose box strictly
    contains its box -- among several, the ones containing a sample point
    inside the hole, the smallest of those -- and a hole no exterior holds
    becomes a polygon of its own, after the others. Returns the ring order
    (exterior first, holes in file order) and poly_offsets into it."""
    cw = signed_area2(xy, ring_offsets) < 0
    nrings = np.diff(record_offsets)
    rec = np.repeat(np.arange(len(nrings)), nrings)
    n_ext = np.diff(_offsets(cw)[record_offsets])          # exteriors per record
    each = (n_ext == nrings) | (n_ext == 0)                 # every ring a polygon
    npoly = np.where(each, nrings, 1)
    ring = np.arange(len(cw))
    local = np.where(each[rec], ring - record_offsets[:-1][rec], 0)
    rank = np.where(each[rec], 0, (~cw).astype(np.int64))
    hard = np.flatnonzero(~each & (n_ext > 1))
    if len(hard):
        bb = _part_bbox(xy, ring_offsets)
        area = np.abs(signed_area2(xy, ring_offsets))
        for k in hard:
            r = np.arange(record_offsets[k], record_offsets[k + 1])
            npoly[k], local[r], rank[r] = _group(xy, ring_offsets, r, cw[r], bb[r], area[r])
    pid = (_offsets(npoly)[:-1])[rec] + local
    order = np.lexsort((ring, rank, pid))
    return order, _offsets(np.bincount(pid, minlength=int(npoly.sum())))


def _group(xy, offsets, r, cw, bb, area):
    """organize() for one record of several exteriors and some holes: the
    number of polygons, and each ring's polygon and rank in it."""
    ext, holes = np.flatnonzero(cw), np.flatnonzero(~cw)
    local = np.zeros(len(r), np.int64)
    rank = np.zeros(len(r), np.int64)
    local[ext] = np.arange(len(ext))
    orphans = []
    for h in holes:
        e, hb = bb[ext], bb[h]
        cand = ext[(e[:, 0] < hb[0]) & (hb[2] < e[:, 2]) & (e[:, 1] < hb[1]) & (hb[3] < e[:, 3])]
        if len(cand) > 1:
            ring = xy[offsets[r[h]]:offsets[r[h] + 1]]
            pt = _ring_sample(ring, ccw=True)
            cand = np.array([c for c in cand if _contains(xy[offsets[r[c]]:offsets[r[c] + 1]], pt)],
                            np.int64)
            if len(cand) > 1:
                cand = cand[np.argsort(area[cand], kind='stable')[:1]]
        if len(cand):
            local[h], rank[h] = local[cand[0]], 1
        else:
            orphans.append(h)
    local[orphans] = len(ext) + np.arange(len(orphans))
    return len(ext) + len(orphans), local, rank


def _contains(ring, pt):
    """Whether a ring contains a point: crossings along +x (pyshp's test)."""
    tx, ty = pt
    x0, y0, x1, y1 = ring[:-1, 0], ring[:-1, 1], ring[1:, 0], ring[1:, 1]
    cross = (y0 >= ty) != (y1 >= ty)
    xf0, xf1 = x0 >= tx, x1 >= tx
    with np.errstate(divide='ignore', invalid='ignore'):
        xi = x1 - (y1 - ty) * (x0 - x1) / (y0 - y1)
    hit = cross & np.where(xf0 == xf1, xf0, xi >= tx)
    return bool(np.count_nonzero(hit) & 1)


def _ring_sample(ring, ccw):
    """A point inside a ring: the centroid of the first vertex triplet that
    winds as the ring does and lies inside it (pyshp's ring_sample)."""
    if len(ring) < 3:
        return tuple(ring[0])
    pts = [tuple(p) for p in ring] + [tuple(ring[1])]
    tri = []
    for p in pts:
        if p not in tri:
            tri.append(p)
        if len(tri) == 3:
            (ax, ay), (bx, by), (cx, cy) = tri
            if (ay - by) * (ax - cx) != (ay - cy) * (ax - bx):
                a2 = ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)
                if (a2 >= 0) == ccw:
                    c = ((ax + bx + cx) / 3.0, (ay + by + cy) / 3.0)
                    if _contains(ring, c):
                        return c
            tri.pop(0)
    return tuple(ring[0])                       # (degenerate: pyshp raises)
