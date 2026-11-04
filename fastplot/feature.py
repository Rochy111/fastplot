"""Natural Earth features, named as cartopy.feature's:

    ax.add_feature(fastplot.feature.BORDERS, linewidth=0.5)
    ax.add_feature(fastplot.feature.STATES.with_scale('10m'), edgecolor='gray')
    ax.add_feature(fastplot.feature.NaturalEarthFeature('cultural', 'admin_2_counties', '10m'))

110m and 50m ship with fastplot; 10m is read from cartopy's cache or
downloaded once (see _features). Geometry of your own -- a shapefile, or
shapely geometries -- is a feature too, drawn the same way:

    ax.add_feature(fastplot.feature.ShapefileFeature('tl_2021_us_primaryroads.shp'),
                   edgecolor='black', linewidth=0.4)
    ax.add_feature(fastplot.feature.ShapelyFeature(geoms, fastplot.crs.PlateCarree()))
"""
import contextlib
import os
import threading

import numpy as np

_SCOPE = threading.local()      # .stats: path -> (path, size, mtime), for the draw in progress


@contextlib.contextmanager
def one_look():
    """For the block (a figure's draw): each shapefile's file looked at
    once, however many features and calls read it."""
    old = getattr(_SCOPE, 'stats', None)
    _SCOPE.stats = {}
    try:
        yield
    finally:
        _SCOPE.stats = old

COLORS = {'land': np.array((240, 240, 220)) / 256.0,
          'land_alt1': np.array((220, 220, 220)) / 256.0,
          'water': np.array((152, 183, 226)) / 256.0}

# Natural Earth names -> fastplot's
_NE_NAMES = {'coastline': 'coastline', 'admin_0_boundary_lines_land': 'borders',
             'admin_1_states_provinces_lines': 'states',
             'admin_1_states_provinces_lakes': 'states_lakes', 'lakes': 'lakes',
             'rivers_lake_centerlines': 'rivers', 'land': 'land', 'ocean': 'ocean',
             'admin_2_counties': 'counties'}


class Feature:
    """A Natural Earth feature and how to draw it: edgecolor, facecolor
    ('none'/'never': not filled), linewidth (points), linestyle, alpha,
    zorder; scale 'auto' (by the map's extent, as cartopy) or a fixed one."""

    def __init__(self, name, scale='auto', **kwargs):
        self.name = name
        self.scale = scale
        self.kwargs = kwargs

    def with_scale(self, scale):
        return Feature(self.name, scale, **self.kwargs)

    def __repr__(self):
        return '<Feature %s %s>' % (self.name, self.scale)

    # what FeatureArtist draws: the geometry in `crs` (None: PlateCarree),
    # the parts whose bounding box meets bbox
    crs = None

    def _kind(self):
        from . import _features
        return _features.kind(self.name)

    def _lines(self, scale, bbox):
        from . import _features
        return _features.lines(self.name, scale, bbox=bbox)

    def _polygons(self, scale, bbox):
        from . import _features
        return _features.polygons(self.name, scale, bbox=bbox)

    def _key(self, scale):
        return (self.name, scale)


def NaturalEarthFeature(category, name, scale='auto', **kwargs):     # noqa: N802 -- cartopy's name
    """cartopy.feature.NaturalEarthFeature for the features fastplot knows."""
    if name not in _NE_NAMES:
        raise ValueError('fastplot has no Natural Earth feature %r (it has: %s)'
                         % (name, ', '.join(sorted(_NE_NAMES))))
    return Feature(_NE_NAMES[name], scale, **kwargs)


BORDERS = Feature('borders', edgecolor='black', facecolor='never')
STATES = Feature('states_lakes', edgecolor='black', facecolor='none')
COASTLINE = Feature('coastline', edgecolor='black', facecolor='never')
LAKES = Feature('lakes', edgecolor='none', facecolor=COLORS['water'])
LAND = Feature('land', edgecolor='none', facecolor=COLORS['land'], zorder=-1)
OCEAN = Feature('ocean', edgecolor='none', facecolor=COLORS['water'], zorder=-1)
RIVERS = Feature('rivers', edgecolor=COLORS['water'], facecolor='never')
COUNTIES = Feature('counties', scale='10m', edgecolor='black', facecolor='none')


class _GeometryFeature(Feature):
    """A feature of geometry of its own (a shapefile's, shapely's), drawn as
    the Natural Earth ones are: the parts that can show picked by their
    bounding boxes, projected once a map and kept for the next figure of
    it. `crs`: the coordinate system it is in (None: PlateCarree)."""

    def __init__(self, crs=None, **kwargs):
        super().__init__(None, None, **kwargs)
        self.crs = crs
        self._geom = None                               # (Lines or None, Polygons or None)

    def with_scale(self, scale):
        return self                                     # (one scale: its own)

    def _parts(self):
        raise NotImplementedError

    def _both(self):
        if self._geom is None:
            self._geom = self._parts()
        return self._geom

    def _kind(self):
        return 'polygons' if self._both()[1] is not None else 'lines'

    def _lines(self, scale, bbox):
        """Its lines, and its polygons' rings as lines."""
        from . import _features
        ln = self._all_lines
        if ln is None:
            lines, polys = self._both()
            out = [] if lines is None else [lines]
            if polys is not None:
                out.append(polys.edges())
            if not out:
                ln = _features.Lines(np.empty((0, 2)), np.zeros(1, np.int64), np.empty((0, 4)))
            else:
                ln = out[0] if len(out) == 1 else _join_lines(out)
            self._all_lines = ln
        return ln if bbox is None else _features.select(ln, bbox)

    _all_lines = None

    def _polygons(self, scale, bbox):
        from . import _features
        polys = self._both()[1]
        return polys if bbox is None or polys is None else _features.select(polys, bbox)

    _made = [0]

    def _key(self, scale):
        k = getattr(self, '_token', None)
        if k is None:                                   # (not id(): it is reused once freed)
            _GeometryFeature._made[0] += 1
            k = self._token = ('geometry', _GeometryFeature._made[0])
        return k

    def geometries(self):
        """Every part's coordinates, (n, 2) arrays: the lines, then the rings."""
        ln = self._lines(None, None)
        return [ln.xy[ln.offsets[k]:ln.offsets[k + 1]] for k in range(ln.parts)]


def _join_lines(parts):
    from . import _features as f
    xy = np.concatenate([p.xy for p in parts])
    counts = np.concatenate([np.diff(p.offsets) for p in parts])
    return f.Lines(f._readonly(xy), f._readonly(f._offsets(counts)),
                   f._readonly(np.concatenate([p.bbox for p in parts])))


def _lines_of(f, parts):
    xy = f._readonly(np.ascontiguousarray(np.concatenate(parts), np.float64))
    off = f._readonly(f._offsets(np.array([len(p) for p in parts], np.int64)))
    return f.Lines(xy, off, f._readonly(f._part_bbox(xy, off)))


class ShapefileFeature(_GeometryFeature):
    """A feature from an ESRI shapefile (.shp) of polylines or polygons --
    Census counties, TIGER roads -- read by fastplot's own reader (no
    cartopy, shapely or pyshp needed), once a process per file:

        counties = ShapefileFeature('cb_2018_us_county_20m.shp', facecolor='none',
                                    edgecolor='gray', linewidth=0.3)
        ax.add_feature(counties)

    A polygon file's rings are grouped into polygons as cartopy's reader
    groups them, so a facecolor fills them as cartopy does; its edges are
    every ring. Coordinates are taken as `crs` (None: PlateCarree, lon/lat
    degrees -- the Census files' NAD83 is within a metre of it)."""

    _read = {}                                          # (path, size, mtime) -> geometry

    def __init__(self, path, crs=None, **kwargs):
        super().__init__(crs, **kwargs)
        self.path = os.fspath(path)

    def __repr__(self):
        return '<ShapefileFeature %s>' % self.path

    def _file(self):
        """The file as it is (path, size, mtime): looked at once a draw."""
        memo = getattr(_SCOPE, 'stats', None)
        got = memo.get(self.path) if memo is not None else None
        if got is None:
            st = os.stat(self.path)
            got = (os.path.abspath(self.path), st.st_size, st.st_mtime_ns)
            if memo is not None:
                memo[self.path] = got
        return got

    def _both(self):
        return self._parts()                            # (_read keeps it, by the file as it is)

    def _lines(self, scale, bbox):
        key = self._file()
        if self._lines_of != key:                       # (a changed file: its lines again)
            self._all_lines, self._lines_of = None, key
        return super()._lines(scale, bbox)

    _lines_of = None

    def _parts(self):
        from . import _features as f
        key = self._file()
        got = ShapefileFeature._read.get(key)
        if got is None:
            shp = f.read_shp(self.path)
            if f._SHP_TYPES.get(shp.shape_type) == 'polygon':
                _, xy, rings, polys = f._from_shp(shp, 'polygons')
                xy, rings, polys = (f._readonly(np.ascontiguousarray(a))
                                    for a in (np.asarray(xy, np.float64), rings, polys))
                got = (None, f.Polygons(xy, rings, polys,
                                        f._readonly(f._poly_bbox(xy, rings, polys))))
            else:
                xy = f._readonly(np.ascontiguousarray(shp.xy, np.float64))
                off = f._readonly(np.asarray(shp.part_offsets, np.int64))
                got = (f.Lines(xy, off, f._readonly(f._part_bbox(xy, off))), None)
            ShapefileFeature._read[key] = got
        return got

    def _key(self, scale):
        return ('shapefile',) + self._file()


class ShapelyFeature(_GeometryFeature):
    """cartopy.feature.ShapelyFeature(geometries, crs, **kwargs): shapely
    geometries (Polygon, LineString, LinearRing, their Multi forms and
    collections of them) in `crs`, drawn as a feature."""

    def __init__(self, geometries, crs=None, **kwargs):
        super().__init__(crs, **kwargs)
        self._source = list(geometries)

    def _parts(self):
        from . import _features as f
        lines, rings, polys = [], [], []

        def walk(g):
            if g is None or g.is_empty:
                return
            t = g.geom_type
            if t == 'Polygon':
                rs = [np.asarray(g.exterior.coords, np.float64)[:, :2]] + \
                    [np.asarray(r.coords, np.float64)[:, :2] for r in g.interiors]
                rs = [r for r in rs if len(r) >= 2]
                if rs:
                    polys.append(len(rs))
                    rings.extend(rs)
            elif t in ('LineString', 'LinearRing'):
                c = np.asarray(g.coords, np.float64)[:, :2]
                if len(c) >= 2:
                    lines.append(c)
            elif hasattr(g, 'geoms'):
                for s in g.geoms:
                    walk(s)

        for g in self._source:
            walk(g)
        ln = _lines_of(f, lines) if lines else None
        pg = None
        if rings:
            r = _lines_of(f, rings)
            po = f._readonly(f._offsets(np.array(polys, np.int64)))
            pg = f.Polygons(r.xy, r.offsets, po, f._readonly(f._poly_bbox(r.xy, r.offsets, po)))
        return ln, pg
