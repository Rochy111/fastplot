"""fastplot._features: Natural Earth geometry against cartopy + shapely, bbox
selection, scale_for against cartopy's AdaptiveScaler, the shapefile
reader on hand-made files, and where 10m data is found, fetched and kept
(no network: downloads are monkeypatched)."""
import io
import os
import struct
import zipfile

import numpy as np
import pytest

from fastplot import _features as F

NE_DIRS = F._cartopy_dirs()                 # cartopy's caches, as the environment has them


def cartopy_shp(category, ne, scale):
    for d in NE_DIRS:
        p = os.path.join(d, category, 'ne_%s_%s.shp' % (scale, ne))
        if os.path.exists(p):
            return p
    pytest.skip('ne_%s_%s.shp not in cartopy\'s cache' % (scale, ne))


@pytest.fixture
def fresh(monkeypatch, tmp_path):
    """An empty memory cache and cache dir, no cartopy data, no network."""
    monkeypatch.setattr(F, '_CACHE', {})
    monkeypatch.setenv('FASTPLOT_DATA_DIR', str(tmp_path / 'cache'))
    monkeypatch.delenv('CARTOPY_DATA_DIR', raising=False)
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'xdg'))

    def offline(url, *a, **k):
        raise OSError('no network in tests')
    monkeypatch.setattr(F, '_download', offline)
    return tmp_path


# -- against cartopy ----------------------------------------------------------

def reference_lines(path):
    cio = pytest.importorskip('cartopy.io.shapereader')
    return [np.asarray(g.coords) for geom in cio.Reader(path).geometries()
            for g in getattr(geom, 'geoms', [geom])]


def reference_polygons(path):
    cio = pytest.importorskip('cartopy.io.shapereader')
    return [p for geom in cio.Reader(path).geometries() for p in getattr(geom, 'geoms', [geom])]


def same(a, b):
    """Equal to float32-level precision (here: within the 1e-6 deg grid)."""
    return a.shape == b.shape and np.abs(a - b).max() <= 1e-6


@pytest.mark.parametrize('scale', ['110m', '50m'])
@pytest.mark.parametrize('name', ['coastline', 'borders'])
def test_lines_match_cartopy(name, scale):
    category, ne, _ = F.FEATURES[name]
    ref = reference_lines(cartopy_shp(category, ne, scale))
    got = F.lines(name, scale)
    assert got.parts == len(ref)
    assert all(same(got.xy[got.offsets[k]:got.offsets[k + 1]], r) for k, r in enumerate(ref))
    assert got.xy.dtype == np.float64 and got.offsets.dtype == np.int64
    assert got.bbox.shape == (len(ref), 4)


@pytest.mark.parametrize('scale', ['110m', '50m'])
def test_land_matches_cartopy(scale):
    ref = reference_polygons(cartopy_shp('physical', 'land', scale))
    got = F.polygons('land', scale)
    assert got.parts == len(ref)
    assert list(np.diff(got.poly_offsets)) == [1 + len(p.interiors) for p in ref]
    rings = [np.asarray(r.coords) for p in ref for r in [p.exterior, *p.interiors]]
    assert len(got.ring_offsets) - 1 == len(rings)
    assert all(same(got.xy[got.ring_offsets[k]:got.ring_offsets[k + 1]], r) for k, r in enumerate(rings))
    bounds = np.array([p.bounds for p in ref])
    assert np.abs(got.bbox - bounds).max() <= 1e-6
    e = F.lines('land', scale)                  # ring edges
    assert e.parts == len(rings) and e.xy is got.xy


@pytest.mark.parametrize('name', ['states_lakes', 'ocean', 'lakes'])
def test_holes_grouped_as_cartopy(name):
    category, ne, _ = F.FEATURES[name]
    ref = reference_polygons(cartopy_shp(category, ne, '50m'))
    got = F.polygons(name, '50m')
    assert list(np.diff(got.poly_offsets)) == [1 + len(p.interiors) for p in ref]


def test_10m_from_cartopy_cache(fresh, monkeypatch):
    ne_dirs = NE_DIRS
    monkeypatch.setattr(F, '_cartopy_dirs', lambda: ne_dirs)
    path = cartopy_shp('physical', 'coastline', '10m')
    got = F.lines('coastline', '10m')
    ref = reference_lines(path)
    assert got.parts == len(ref)
    assert all(same(got.xy[got.offsets[k]:got.offsets[k + 1]], r) for k, r in enumerate(ref))
    cached = fresh / 'cache' / 'ne_10m_coastline.npz'
    assert cached.exists() and not [p for p in os.listdir(cached.parent) if p.endswith('.tmp')]
    # a new process: from the .npz, with cartopy's copy gone
    monkeypatch.setattr(F, '_CACHE', {})
    monkeypatch.setattr(F, '_cartopy_dirs', lambda: [])
    again = F.lines('coastline', '10m')
    assert np.array_equal(again.xy, got.xy) and np.array_equal(again.offsets, got.offsets)


# -- API ----------------------------------------------------------------------

def test_names_and_kinds():
    assert F.names() == ['coastline', 'borders', 'states', 'states_lakes', 'lakes', 'rivers',
                         'land', 'ocean', 'counties']
    for n in F.names():
        assert F.kind(n) in ('lines', 'polygons')
    with pytest.raises(ValueError, match='line feature'):
        F.polygons('coastline')
    with pytest.raises(ValueError, match='no feature'):
        F.lines('roads')
    with pytest.raises(ValueError, match='scale'):
        F.lines('coastline', '1m')


@pytest.mark.parametrize('scale', ['110m', '50m'])
def test_every_bundled_feature_loads(fresh, scale):
    for n in F.names():
        if scale not in F.FEATURES[n][2]:
            continue
        f = F.lines(n, scale)
        assert f.parts > 0 and len(f.xy) == f.offsets[-1]
        assert (np.diff(f.offsets) > 0).all()
        assert np.abs(f.xy[:, 0]).max() <= 180 and np.abs(f.xy[:, 1]).max() <= 90
        assert not f.xy.flags.writeable
        if F.kind(n) == 'polygons':
            p = F.polygons(n, scale)
            assert p.poly_offsets[-1] == len(p.ring_offsets) - 1
            r = p.xy[p.ring_offsets[:-1]], p.xy[p.ring_offsets[1:] - 1]
            assert np.array_equal(*r)           # closed rings


def test_bbox_is_part_extent():
    f = F.lines('coastline', '110m')
    for k in range(f.parts):
        part = f.xy[f.offsets[k]:f.offsets[k + 1]]
        assert np.array_equal(f.bbox[k], [*part.min(0), *part.max(0)])


# -- bbox selection -----------------------------------------------------------

def brute(bb, box):
    """Reference: a part meets the box if some 360-degree shift of it does."""
    lon0, lon1, lat0, lat1 = box
    lat0, lat1 = sorted((lat0, lat1))
    if lon1 < lon0:
        lon1 += 360
    out = np.zeros(len(bb), bool)
    for k, (a, s, b, n) in enumerate(bb):
        if s > lat1 or n < lat0:
            continue
        out[k] = lon1 - lon0 >= 360 or any(a + sh <= lon1 and b + sh >= lon0
                                           for sh in (-720, -360, 0, 360, 720))
    return out


BOXES = [(-130, -60, 20, 55), (250, 300, 20, 55), (170, -170, -60, 0), (350, 10, 40, 60),
         (-190, -170, -60, 60), (0, 360, -90, 90), (-180, 180, 60, 90), (100, 100.5, 0, 1),
         (180, 190, -20, 20), (-10, 10, 55, 45), (359, 361, 50, 52), (-500, -480, 0, 30)]


@pytest.mark.parametrize('box', BOXES)
def test_bbox_selection(box):
    f = F.lines('coastline', '50m')
    want = brute(f.bbox, box)
    g = F.lines('coastline', '50m', bbox=box)
    assert g.parts == want.sum()
    k = np.flatnonzero(want)
    assert np.array_equal(g.bbox, f.bbox[k])
    for j, i in enumerate(k[:50]):
        assert np.array_equal(g.xy[g.offsets[j]:g.offsets[j + 1]], f.xy[f.offsets[i]:f.offsets[i + 1]])


def test_bbox_wrap_cases():
    f = F.lines('coastline', '110m')
    us = F.lines('coastline', '110m', bbox=(-125, -65, 25, 50))
    assert us.parts == F.lines('coastline', '110m', bbox=(235, 295, 25, 50)).parts > 0
    # a part at lon -100 meets the box 250..300
    lone = F.Lines(np.array([[-100.0, 40], [-99, 41]]), np.array([0, 2]), np.array([[-100.0, 40, -99, 41]]))
    assert F.select(lone, (250, 300, 30, 50)).parts == 1
    assert F.select(lone, (250, 255, 30, 50)).parts == 0
    # across the antimeridian: Fiji (~178 E .. 180 / -180 .. -179) from both sides
    fiji = F.lines('coastline', '50m', bbox=(175, -175, -20, -15))
    assert fiji.parts > 0 and (fiji.bbox[:, 0] < 0).any() and (fiji.bbox[:, 2] > 0).any()
    assert F.lines('coastline', '50m', bbox=(175, 185, -20, -15)).parts == fiji.parts
    whole = F.lines('coastline', '110m', bbox=(-180, 180, -90, 90))
    assert whole is f                           # nothing dropped: the feature itself


def test_polygon_selection_keeps_holes():
    p = F.polygons('lakes', '50m')
    s = F.polygons('lakes', '50m', bbox=(-95, -80, 40, 50))        # the Great Lakes, islands as holes
    k = np.flatnonzero(brute(p.bbox, (-95, -80, 40, 50)))
    assert s.parts == len(k) and (np.diff(s.poly_offsets) > 1).any()
    for j, i in enumerate(k):
        rs, re = p.poly_offsets[i], p.poly_offsets[i + 1]
        js = s.poly_offsets[j]
        assert s.poly_offsets[j + 1] - js == re - rs
        for r in range(re - rs):
            a = p.xy[p.ring_offsets[rs + r]:p.ring_offsets[rs + r + 1]]
            b = s.xy[s.ring_offsets[js + r]:s.ring_offsets[js + r + 1]]
            assert np.array_equal(a, b)
    e = s.edges()
    assert e.parts == len(s.ring_offsets) - 1


def test_bbox_rejects_nan():
    with pytest.raises(ValueError):
        F.lines('coastline', '110m', bbox=(np.nan, 10, 0, 10))


# -- scale_for ----------------------------------------------------------------

def test_scale_for_matches_cartopy():
    cf = pytest.importorskip('cartopy.feature')
    ref = cf.AdaptiveScaler('110m', (('50m', 50), ('10m', 15)))
    rng = np.random.default_rng(0)
    exts = [(-180, 180, -90, 90), (-5, 6, 45, 56), (-5, 5, 45, 56), (0, 15, 0, 15), (0, 15.0001, 0, 30),
            (0, 50, 0, 60), (0, 50.0001, 0, 60), (0, 0, 0, 10), (10, 10, 5, 5), (0, 360, -90, 90),
            (100, 120, 30, 30), (230, 300, 20, 55)]
    for _ in range(2000):
        lon0 = rng.uniform(-180, 180)
        exts.append((lon0, lon0 + rng.uniform(0, 80) ** rng.uniform(0.5, 1.4),
                     *sorted(rng.uniform(-90, 90, 2))))
    for e in exts:
        assert F.scale_for(e) == ref.scale_from_extent(e), e
    assert F.scale_for(None) == ref.scale_from_extent(None) == '110m'
    assert F.scale_for((np.nan, 0, 0, 1)) == ref.scale_from_extent((np.nan, 0, 0, 1))
    assert cf.COASTLINE.scaler._limits == (('50m', 50), ('10m', 15))      # what scale_for mirrors
    assert F.scale_for((170, -170, 0, 30)) == '50m'                         # across 180: 20 wide


# -- the shapefile reader -----------------------------------------------------

def make_shp(stype, records, with_m=True):
    """A .shp per the ESRI whitepaper: records are lists of parts (lists of
    (x, y)) or None (a null shape); Z and M values are made up."""
    body = b''
    allpts = [p for r in records if r for part in r for p in part]
    for i, rec in enumerate(records):
        if rec is None:
            content = struct.pack('<i', 0)
        else:
            pts = np.array([p for part in rec for p in part], float)
            parts = np.cumsum([0] + [len(p) for p in rec[:-1]])
            n = len(pts)
            content = struct.pack('<i4d2i', stype, *pts.min(0), *pts.max(0), len(rec), n)
            content += struct.pack('<%di' % len(rec), *parts) + pts.astype('<f8').tobytes()
            z, m = 100 + np.arange(n, dtype='<f8'), -np.arange(n, dtype='<f8')
            if stype in (13, 15):
                content += struct.pack('<2d', z.min(), z.max()) + z.tobytes()
            if stype in (23, 25) or (stype in (13, 15) and with_m):
                content += struct.pack('<2d', m.min(), m.max()) + m.tobytes()
        body += struct.pack('>2i', i + 1, len(content) // 2) + content
    xy = np.array(allpts, float) if allpts else np.zeros((1, 2))
    head = struct.pack('>7i', 9994, 0, 0, 0, 0, 0, (100 + len(body)) // 2)
    head += struct.pack('<2i4d4d', 1000, stype, *xy.min(0), *xy.max(0), 0, 0, 0, 0)
    return head + body


SQUARE_CW = [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)]
HOLE_CCW = [(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)]
FAR_CW = [(20, 0), (20, 5), (25, 5), (25, 0), (20, 0)]


@pytest.mark.parametrize('stype, with_m', [(3, False), (5, False), (13, True), (13, False),
                                           (15, True), (15, False), (23, True), (25, True)])
def test_read_shp_variants(stype, with_m):
    recs = [[SQUARE_CW, HOLE_CCW], None, [FAR_CW], [[(1, 1), (2, 3), (5, 8)]]]
    data = make_shp(stype, recs, with_m)
    s = F.read_shp(data)
    assert s.shape_type == stype
    want = [p for r in recs if r for p in r]
    assert len(s.part_offsets) - 1 == len(want)
    for k, w in enumerate(want):
        assert np.array_equal(s.xy[s.part_offsets[k]:s.part_offsets[k + 1]], np.array(w, float))
    assert list(np.diff(s.record_offsets)) == [2, 0, 1, 1]
    shapefile = pytest.importorskip('shapefile')                  # pyshp agrees
    ref = [sh for sh in shapefile.Reader(shp=io.BytesIO(data)).iterShapes()]
    assert len(ref) == 4
    pts = np.concatenate([np.array(sh.points, float).reshape(-1, 2) for sh in ref if sh.points])
    assert np.array_equal(pts, s.xy)


def test_read_shp_from_path(tmp_path):
    p = tmp_path / 'x.shp'
    p.write_bytes(make_shp(3, [[[(0, 0), (1, 1)]]]))
    assert F.read_shp(str(p)).xy.tolist() == [[0, 0], [1, 1]]
    assert F.read_shp(p).xy.tolist() == [[0, 0], [1, 1]]


def test_read_shp_rejects():
    with pytest.raises(ValueError, match='9994'):
        F.read_shp(b'\0' * 100)
    pts = make_shp(3, [[[(0, 0), (1, 1)]]])
    bad = pts[:32] + struct.pack('<i', 1) + pts[36:]                 # points
    with pytest.raises(ValueError, match='shape type 1'):
        F.read_shp(bad)
    with pytest.raises(ValueError, match='cut short'):
        F.read_shp(pts[:-8])


def test_polygon_grouping_like_pyshp():
    shapefile = pytest.importorskip('shapefile')
    inner_cw = [(2.5, 2.5), (2.5, 3.5), (3.5, 3.5), (3.5, 2.5), (2.5, 2.5)]   # island in the hole
    big_ccw_far = [(50, 50), (60, 50), (60, 60), (50, 60), (50, 50)]          # orphan hole
    recs = [[SQUARE_CW, HOLE_CCW],                      # one exterior: simple
            [HOLE_CCW, SQUARE_CW],                      # hole first in the file
            [FAR_CW, SQUARE_CW, HOLE_CCW, inner_cw],    # several exteriors, one hole
            [SQUARE_CW, big_ccw_far, FAR_CW],           # an orphan
            [HOLE_CCW],                                 # only a hole
            [FAR_CW, SQUARE_CW]]                        # no holes
    data = make_shp(5, recs)
    got = F.decode(F.encode(*F._from_shp(F.read_shp(data), 'polygons')))
    want = []
    for sh in shapefile.Reader(shp=io.BytesIO(data)).iterShapes():
        g = sh.__geo_interface__
        want += [g['coordinates']] if g['type'] == 'Polygon' else g['coordinates']
    assert got.parts == len(want)
    for k, poly in enumerate(want):
        rs, re = got.poly_offsets[k], got.poly_offsets[k + 1]
        assert re - rs == len(poly)
        for j, ring in enumerate(poly):
            a = got.xy[got.ring_offsets[rs + j]:got.ring_offsets[rs + j + 1]]
            assert np.array_equal(a, np.array(ring, float))


def test_encode_roundtrip():
    rng = np.random.default_rng(1)
    xy = np.column_stack([rng.uniform(-180, 180, 5000), rng.uniform(-90, 90, 5000)])
    xy[:3] = [[-180, -90], [180, 90], [-180, 90]]
    off = np.array([0, 1, 2, 4000, 5000])
    d = F.decode(F.encode('lines', xy, off))
    assert np.abs(d.xy - xy).max() <= 5e-7 and np.array_equal(d.offsets, off)
    assert np.array_equal(d.xy[:3], xy[:3])
    with pytest.raises(ValueError):
        F.encode('lines', np.array([[2000.0, 0]]), np.array([0, 1]))


# -- where the 10m data comes from --------------------------------------------

def counties_zip():
    shp = make_shp(5, [[SQUARE_CW, HOLE_CCW], [FAR_CW]])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('ne_10m_admin_2_counties.README.html', '<html/>')
        z.writestr('ne_10m_admin_2_counties.shp', shp)
    return buf.getvalue()


def test_download_and_cache(fresh, monkeypatch):
    asked = []

    def download(url, *a, **k):
        asked.append(url)
        return counties_zip()
    monkeypatch.setattr(F, '_download', download)
    p = F.polygons('counties', '50m')           # counties: 10m only, so 10m
    assert asked == ['https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_2_counties.zip']
    assert p.parts == 2 and list(np.diff(p.poly_offsets)) == [2, 1]
    assert F.polygons('counties', '10m') is p
    assert F.lines('counties', '10m').parts == 3
    cache = fresh / 'cache'
    assert os.listdir(cache) == ['ne_10m_counties.npz']           # no temp file left
    monkeypatch.setattr(F, '_CACHE', {})        # a new process: from the cache
    q = F.polygons('counties')
    assert len(asked) == 1 and np.array_equal(q.xy, p.xy)


def test_failed_download_names_url_and_env(fresh):
    with pytest.raises(OSError) as e:
        F.lines('coastline', '10m')
    msg = str(e.value)
    assert 'https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_coastline.zip' in msg
    assert 'FASTPLOT_DATA_DIR' in msg and 'no network in tests' in msg
    assert not (fresh / 'cache').exists() or not os.listdir(fresh / 'cache')


def test_hand_placed_zip_and_cartopy_dir(fresh, monkeypatch):
    cache = fresh / 'cache'
    cache.mkdir()
    (cache / 'ne_10m_admin_2_counties.zip').write_bytes(counties_zip())
    assert F.polygons('counties').parts == 2
    # CARTOPY_DATA_DIR's shapefiles/natural_earth/<category>/ne_10m_<name>.shp
    monkeypatch.setattr(F, '_CACHE', {})
    d = fresh / 'cdata' / 'shapefiles' / 'natural_earth' / 'physical'
    d.mkdir(parents=True)
    (d / 'ne_10m_lakes.shp').write_bytes(make_shp(5, [[FAR_CW]]))
    monkeypatch.setenv('CARTOPY_DATA_DIR', str(fresh / 'cdata'))
    assert F.polygons('lakes', '10m').parts == 1
    assert (cache / 'ne_10m_lakes.npz').exists()


def test_stale_cache_rebuilt(fresh, monkeypatch):
    cache = fresh / 'cache'
    cache.mkdir()
    (cache / 'ne_10m_counties.npz').write_bytes(b'not a zip')
    monkeypatch.setattr(F, '_download', lambda url, *a, **k: counties_zip())
    with pytest.warns(UserWarning, match='rebuilding'):
        assert F.polygons('counties').parts == 2
    with np.load(cache / 'ne_10m_counties.npz') as z:
        assert int(z['format']) == F.FORMAT


def test_unwritable_cache_only_warns(fresh, monkeypatch):
    blocker = fresh / 'file'
    blocker.write_bytes(b'')
    monkeypatch.setenv('FASTPLOT_DATA_DIR', str(blocker / 'sub'))    # under a file: cannot be made
    monkeypatch.setattr(F, '_download', lambda url, *a, **k: counties_zip())
    with pytest.warns(UserWarning, match='could not cache'):
        assert F.polygons('counties').parts == 2


def test_cache_dir_defaults(monkeypatch):
    monkeypatch.setenv('FASTPLOT_DATA_DIR', '/some/where')
    assert F.cache_dir() == '/some/where'
    monkeypatch.delenv('FASTPLOT_DATA_DIR')
    home = os.path.expanduser('~')
    monkeypatch.setattr(F.sys, 'platform', 'win32')
    monkeypatch.setenv('LOCALAPPDATA', os.path.join('C:', 'LAD'))
    assert F.cache_dir() == os.path.join('C:', 'LAD', 'fastplot')
    monkeypatch.setattr(F.sys, 'platform', 'darwin')
    assert F.cache_dir() == os.path.join(home, 'Library', 'Caches', 'fastplot')
    monkeypatch.setattr(F.sys, 'platform', 'linux')
    monkeypatch.setenv('XDG_CACHE_HOME', '/xdg')
    assert F.cache_dir() == os.path.join('/xdg', 'fastplot')
    monkeypatch.delenv('XDG_CACHE_HOME')
    assert F.cache_dir() == os.path.join(home, '.cache', 'fastplot')


def test_cartopy_dirs(monkeypatch):
    monkeypatch.setenv('CARTOPY_DATA_DIR', '/pre')
    monkeypatch.setenv('XDG_DATA_HOME', '/data')
    assert F._cartopy_dirs() == [os.path.join('/pre', 'shapefiles', 'natural_earth'),
                                 os.path.join('/data', 'cartopy', 'shapefiles', 'natural_earth')]
