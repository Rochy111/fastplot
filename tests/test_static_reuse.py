"""What a new figure of the same map, colorbar and texts takes from the
last one -- kept text blends, a colorbar's pixels, a map's extents, outline
and frame, its background fill -- is what drawing them afresh gives: every
pixel the same, and drawn anew whatever they follow from changes."""
import os
import struct
import tempfile

import numpy as np
import pytest

import fastplot
import fastplot.crs as fcrs
import fastplot.feature as ffeature
import fastplot.patheffects as fpe
from fastplot import _axes, _colorbar, _figure, _kept, _proj, _rc
from fastplot.colors import Normalize


def _rgba(fig):
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def _no_text_memo(monkeypatch):
    monkeypatch.setattr(_figure._Renderer, '_text_key', lambda self, *a: None)


def _no_regions(monkeypatch):
    monkeypatch.setattr(_kept, 'region', lambda r, key, draw: draw(r))


def _map_figure(cmap='turbo', vmin=-10.0, vmax=30.0, ticks=None, extend='both', title='T 2 m',
                valid='Valid: 12Z', facecolor=None, field=True):
    """A forecast-map frame in small: a map with its extent, graticule, features,
    a field, a colorbar in a cax, headers and tick labels."""
    fig = fastplot.figure(figsize=(6, 3.6), dpi=90, facecolor=facecolor)
    pc = fcrs.PlateCarree()
    rect = [0.08, 0.1, 0.78, 0.78]
    ax = fig.add_axes(rect, projection=pc)
    e = (-104.0, -94.0, 30.0, 36.0)
    ax.set_extent(e, crs=pc)
    xt, yt = np.linspace(e[0], e[1], 6), np.linspace(e[2], e[3], 4)
    ax.gridlines(crs=pc, xlocs=xt, ylocs=yt, linestyle=':', alpha=0.4, color='gray',
                 linewidth=0.5, zorder=1.5)
    tax = fig.add_axes(rect)
    tax.patch.set_visible(False)
    for s in tax.spines.values():
        s.set_visible(False)
    tax.set_xlim(e[0], e[1])
    tax.set_ylim(e[2], e[3])
    tax.set_xticks(xt)
    tax.set_yticks(yt)
    tax.tick_params(axis='both', labelsize=8, pad=3, width=0.5)
    ax.add_feature(ffeature.STATES, edgecolor='#1f1f1f', facecolor='none', linewidth=0.65)
    ax.coastlines(color='black', linewidth=0.7)
    if field:
        lon, lat = np.meshgrid(np.linspace(-105, -93, 60), np.linspace(29, 37, 40))
        z = 10 + 15 * np.sin(lon / 2) * np.cos(lat / 3)
        ax.contourf(lon, lat, z, levels=np.linspace(vmin, vmax, 40), cmap=cmap,
                    norm=Normalize(vmin, vmax), extend='both', transform=pc)
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
    sm = fastplot.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin, vmax))
    cb = fig.colorbar(sm, cax=cax, ticks=ticks if ticks is not None else
                      np.linspace(vmin, vmax, 9), extend=extend)
    cb.ax.tick_params(labelsize=9, width=0.5)
    fig.text(pos.x0, pos.y1 + 0.06, title, ha='left', fontsize=13, fontweight='bold')
    fig.text(pos.x1, pos.y1 + 0.06, 'FASTPLOT', ha='right', fontsize=13, fontweight='bold')
    fig.text(pos.x1, pos.y1 + 0.01, valid, ha='right', fontsize=9, color='#444444')
    fig.text(0.5, 0.02, 'turned', rotation=30, fontsize=8, alpha=0.6, ha='center')
    fig.text(0.2, 0.03, '0-2 km', fontsize=8, color='#D77A61',
             path_effects=[fpe.withStroke(linewidth=2, foreground='black')])
    fig.text(0.3, 0.03, 'x', fontsize=9, path_effects=[fpe.Stroke(linewidth=3, foreground='w',
                                                                  alpha=0.5), fpe.Normal()])
    return fig


def _frames(n=3, **kw):
    return [_rgba(_map_figure(**kw)) for _ in range(n)]


def _fresh():
    _kept.clear()
    _figure._TEXTS.clear()
    _colorbar._SCALES.clear()


def test_frames_of_the_same_map_draw_as_the_first(monkeypatch):
    """Figure after figure of the same map, bar and texts, taking what the
    one before kept: the pixels of a draw that keeps nothing."""
    _fresh()
    kept = _frames(4)
    with monkeypatch.context() as m:
        _no_text_memo(m)
        _no_regions(m)
        m.setattr(_axes, '_FRAME_KEPT', False)
        m.setattr(_figure._Renderer, 'same_as_bg', lambda self, c: False)
        plain = _frames(1)[0]
    for f in kept:
        assert np.array_equal(f, plain)
    assert len(_kept._REGIONS) >= 1 and not _kept._MISSES      # (the bar was kept, and taken)


@pytest.mark.parametrize('change', [
    dict(cmap='viridis'), dict(vmin=-5.0), dict(vmax=40.0), dict(ticks=[0.0, 10.0, 20.0]),
    dict(extend='neither'), dict(extend='max'), dict(title='Dew point'),
    dict(valid='Valid: 13Z'), dict(facecolor='#eeeeee')])
def test_what_a_kept_drawing_follows_from_draws_it_anew(monkeypatch, change):
    """Another colormap, norm, ticks, extensions, text or background: the
    figure is what it is drawn afresh, not the kept one."""
    _frames(2)                                          # (the plain one kept)
    got = _frames(2, **change)
    with monkeypatch.context() as m:
        _no_text_memo(m)
        _no_regions(m)
        m.setattr(_axes, '_FRAME_KEPT', False)
        want = _frames(1, **change)[0]
    for f in got:
        assert np.array_equal(f, want)


def test_texts_follow_the_font_settings(monkeypatch):
    """A text drawn again under other font settings is drawn in them."""
    _fresh()
    a = _frames(1)[0]
    with _rc.rc_context({'font.family': ['serif']}):
        b = _frames(2)
    with monkeypatch.context() as m, _rc.rc_context({'font.family': ['serif']}):
        _no_text_memo(m)
        want = _frames(1)[0]
    assert not np.array_equal(a, want)
    for f in b:
        assert np.array_equal(f, want)


def test_colormap_changed_in_place_is_drawn_anew():
    """A colormap's under colour set in place (matplotlib's set_under):
    the bar's extension in the new colour."""
    from fastplot import _color
    cm = _color.get_cmap('viridis').copy()
    a = [_rgba(_map_figure(cmap=cm, field=False)) for _ in range(2)]
    cm.set_under('magenta')
    b = _rgba(_map_figure(cmap=cm, field=False))
    _fresh()
    want = _rgba(_map_figure(cmap=cm, field=False))
    assert np.array_equal(a[0], a[1]) and not np.array_equal(a[0], b)
    assert np.array_equal(b, want)


def test_region_drawn_anew_over_other_pixels():
    """A kept region over pixels other than those it was kept over (here:
    a translucent box over another background) is drawn, not pasted."""
    from fastplot._render import CpuCanvas

    def draw(r):
        r.c.polygons(np.array([[10.5, 10.5], [60.2, 12.0], [40.0, 50.7]]), np.array([0, 3]),
                     np.array([200, 30, 30, 120], np.uint8), None, 'nonzero')
        r.extents.append((10.0, 10.0, 60.0, 51.0))

    fig = fastplot.figure(figsize=(1, 1), dpi=80)
    out = []
    for bg in ((255, 255, 255, 255), (255, 255, 255, 255), (0, 90, 200, 255)):
        r = _figure._Renderer(fig, CpuCanvas(80, 80, bg), 80.0, 80.0, 80.0)
        _kept.region(r, ('test-region',), draw)
        ref = _figure._Renderer(fig, CpuCanvas(80, 80, bg), 80.0, 80.0, 80.0)
        draw(ref)
        assert np.array_equal(r.c.a, ref.c.a) and r.extents == ref.extents
        out.append(r.c.a.copy())
    assert np.array_equal(out[0], out[1]) and not np.array_equal(out[0], out[2])
    assert _kept._MISSES.get(('test-region',)) == 1
    _kept.clear()


def test_region_not_kept_over_transparent_pixels():
    from fastplot._render import CpuCanvas
    fig = fastplot.figure(figsize=(1, 1), dpi=80)

    def draw(r):
        r.c.fill_rect(5, 5, 30, 30, np.array([10, 200, 10, 100], np.uint8))

    for _ in range(2):
        r = _figure._Renderer(fig, CpuCanvas(40, 40, (0, 0, 0, 0)), 80.0, 40.0, 40.0)
        _kept.region(r, ('test-clear',), draw)
        ref = _figure._Renderer(fig, CpuCanvas(40, 40, (0, 0, 0, 0)), 80.0, 40.0, 40.0)
        draw(ref)
        assert np.array_equal(r.c.a, ref.c.a)
    assert ('test-clear',) not in _kept._REGIONS
    _kept.clear()


def test_map_background_left_off_only_over_its_own_colour(monkeypatch):
    def img(fc, ax_fc):
        fig = fastplot.figure(figsize=(3, 2), dpi=60, facecolor=fc)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=fcrs.PlateCarree())
        ax.set_extent((-100, -90, 30, 36))
        ax.set_facecolor(ax_fc)
        ax.coastlines()
        return _rgba(fig)

    cases = [('white', 'white'), ('white', '#ddeeff'), ('#ddeeff', 'white'),
             ('white', (1, 1, 1, 0.5)), ('#123456', '#123456')]
    got = [img(*c) for c in cases]
    monkeypatch.setattr(_figure._Renderer, 'same_as_bg', lambda self, c: False)
    for g, c in zip(got, cases):
        assert np.array_equal(g, img(*c))


def test_map_frame_drawn_from_its_kept_outline(monkeypatch):
    """The map's frame, its outline kept and from the third figure its ink:
    the pixels r.lines draws, on a box and on a Robinson oval."""
    for proj, e in ((fcrs.PlateCarree(), (-100, -90, 30, 36)), (fcrs.Robinson(), None),
                    (fcrs.LambertConformal(), (-120, -70, 20, 50))):
        def img():
            fig = fastplot.figure(figsize=(3, 2), dpi=70)
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=proj)
            if e is None:
                ax.set_global()
            else:
                ax.set_extent(e, crs=fcrs.PlateCarree())
            ax.spines['geo'].set_linewidth(1.3)
            return _rgba(fig)
        kept = [img() for _ in range(4)]
        with monkeypatch.context() as m:
            m.setattr(_axes, '_FRAME_KEPT', False)
            want = img()
        for k in kept:
            assert np.array_equal(k, want)


def test_extents_kept_as_made():
    """extent_xy, a map's lon/lat extent and its outline, kept: what they
    are made, to the bit (-0.0, types)."""
    cases = [(fcrs.PlateCarree(), (-104.5, -94.25, 30.0, 36.0), None),
             (fcrs.PlateCarree(central_longitude=180), (100, 250, -10, 40), None),
             (fcrs.LambertConformal(), (-120, -70, 20, 50), None),
             (fcrs.Mercator(), (-0.0, 30.0, -0.0, 20.0), fcrs.PlateCarree()),
             (fcrs.Robinson(), (-180, 180, -90, 90), None)]
    for proj, ext, crs in cases:
        for _ in range(2):
            assert _proj.extent_xy(proj, ext, crs) == _proj._extent_xy(proj, ext, crs)
            fig = fastplot.figure(figsize=(3, 2), dpi=50)
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=proj)
            ax.set_extent(ext, crs=crs)
            fig.canvas.buffer_rgba()
            e = ax.extent
            assert ax._ll_extent(e) == ax._ll_extent_of(e)
            ax._place(fig, *fig._size_f(fig.dpi))
            a, b = ax._outline(), ax._outline_of()
            assert (a is None and b is None) or np.array_equal(a, b)
            ax._placed = None
    a = _proj.extent_xy(fcrs.PlateCarree(), (-0.0, 10.0, 0.0, 5.0))
    b = _proj.extent_xy(fcrs.PlateCarree(), (0.0, 10.0, 0.0, 5.0))
    assert str(a) == str(_proj._extent_xy(fcrs.PlateCarree(), (-0.0, 10.0, 0.0, 5.0)))
    assert str(b) == str(_proj._extent_xy(fcrs.PlateCarree(), (0.0, 10.0, 0.0, 5.0)))


def test_projections_equal_by_value():
    a, b = fcrs.PlateCarree(), fcrs.PlateCarree()
    assert a == b and hash(a) == hash(b) and a == a
    assert a != fcrs.PlateCarree(central_longitude=10) and a != fcrs.Geodetic()
    lc = fcrs.LambertConformal(central_longitude=-100)
    assert lc == fcrs.LambertConformal(central_longitude=-100) and lc != fcrs.LambertConformal()
    assert np.array_equal(a.boundary, b.boundary) and a.boundary is not b.boundary


def test_freeze_tells_values_apart_to_the_bit():
    f = _kept.freeze
    assert f(-0.0) != f(0.0) and f(1) != f(1.0) and f(True) != f(1)
    assert f(np.float32(1.5)) != f(1.5)
    assert f(np.zeros(3)) != f(np.zeros(3, np.float32)) and f(np.zeros(3)) != f(np.zeros((1, 3)))
    assert f([1, 2]) != f((1, 2)) and f({'a': 1}) == f({'a': 1})
    assert f(float('nan')) == f(float('nan'))
    with pytest.raises(TypeError):
        f(object())
    assert _kept.frozen(lambda: 0) is None


def test_styles_kept_as_made():
    from fastplot import _draw
    for args in ((np.array([1, 2, 3, 255], np.uint8), 1.25, None),
                 (np.array([1, 2, 3, 255], np.uint8), 1.25, [(2.0, 1.0)]),
                 (np.array([9, 2, 3, 128], np.uint8), 0.5, None)):
        s, t = _kept.styles(*args), _draw.Styles(*args)
        assert _kept.styles(*args) is s
        for k in vars(t):
            assert np.array_equal(np.asarray(getattr(s, k)), np.asarray(getattr(t, k)))
    assert _kept.styles(np.array([1, 2, 3, 255], np.uint8), -0.0) is not \
        _kept.styles(np.array([1, 2, 3, 255], np.uint8), 0.0)


def _write_shp(path, parts):
    recs = []
    for k, ring in enumerate(parts):
        ring = np.asarray(ring, float)
        body = struct.pack('<i4d2i', 3, ring[:, 0].min(), ring[:, 1].min(),
                           ring[:, 0].max(), ring[:, 1].max(), 1, len(ring))
        body += struct.pack('<i', 0) + ring.astype('<f8').tobytes()
        recs.append(struct.pack('>2i', k + 1, len(body) // 2) + body)
    data = b''.join(recs)
    head = struct.pack('>7i', 9994, 0, 0, 0, 0, 0, (100 + len(data)) // 2)
    head += struct.pack('<2i4d4d', 1000, 3, -180, -90, 180, 90, 0, 0, 0, 0)
    with open(path, 'wb') as f:
        f.write(head + data)


def test_shapefile_looked_at_once_a_draw_and_read_again_when_changed(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'roads.shp')
        _write_shp(path, [[(-104, 29), (-95, 34)]])
        roads = ffeature.ShapefileFeature(path)

        def img():
            fig = fastplot.figure(figsize=(3, 2), dpi=60)
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection=fcrs.PlateCarree())
            ax.set_extent((-105, -93, 28, 36))
            ax.add_feature(roads, edgecolor='k', linewidth=1)
            ax.add_feature(roads, edgecolor='r', linewidth=0.5)
            return _rgba(fig)

        a = img()
        looks = []
        stat = os.stat
        monkeypatch.setattr(ffeature.os, 'stat', lambda p, *k, **kw: looks.append(p) or
                            stat(p, *k, **kw))
        assert np.array_equal(img(), a)
        assert looks == [path]                          # (two features, one look)
        monkeypatch.setattr(ffeature.os, 'stat', stat)
        _write_shp(path, [[(-104, 34), (-95, 29)]])
        os.utime(path, ns=(1, stat(path).st_mtime_ns + 10 ** 9))
        b = img()
        assert not np.array_equal(a, b)
