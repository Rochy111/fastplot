"""Saving big images (a Landsat scene): each faster way against the way it
replaced, byte for byte --
  - RGB(A) imshow made bytes pixel by pixel as drawn, not converted whole
    when made (RGBImage);
  - autoscale's least and greatest finite value in one pass on the threads
    (_color._finite_range), numpy's isfinite, mask, min and max the other;
  - more cells in view than pixels: each pixel shaded from its cell's value
    as it goes onto the canvas (_render._pixels_linear), not sampled into a
    raster, shaded into a layer and blended;
  - bbox_inches='tight' measured by a draw that paints nothing, not by a
    painting one (Figure._measure);
  - a PNG written in the pieces it is made of (_png.save)."""
import io

import numpy as np
import pytest

import fastplot.pyplot as plt
from fastplot import _color, _figure, _fields, _jit, _png, _render, colors, crs


def _rng():
    return np.random.default_rng(11)


def _scene(shape=(700, 900)):
    """A band: values 0..1 clipped, NaN off the scene, a few above and below."""
    rng = _rng()
    z = np.clip(rng.normal(0.5, 0.3, shape), 0, 1).astype(np.float32)
    z[:, :60] = np.nan
    z[rng.random(shape) < 0.01] = np.nan
    return z


# --- RGB(A) imshow, made bytes as drawn

def _rgb_cases():
    rng = _rng()
    f = rng.uniform(-0.3, 1.3, (60, 80, 3)).astype(np.float32)
    f[::7, ::5, 1] = np.nan
    f[::11, ::3, 2] = np.inf
    f[3::13, ::4, 0] = -np.inf
    u8 = (rng.random((60, 80, 3)) * 255).astype(np.uint8)
    return {
        'u8 rgb': u8,
        'u8 rgb strided': np.dstack([u8, u8])[::-1, ::2, 1:4],
        'f32 rgb': f,
        'f32 rgba': np.dstack([f, rng.uniform(-0.2, 1.2, f.shape[:2]).astype(np.float32)]),
        'f64 rgb F-order': np.asfortranarray(f.astype(np.float64)),
        'f64 rgba': np.dstack([f, np.full(f.shape[:2], 0.6, np.float32)]).astype(np.float64),
        'f16 rgb': f.astype(np.float16),
        'u16 rgb': (u8 * 3).astype(np.uint16),
    }


def _draw_rgb(data, kw, lim, dpi):
    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.imshow(data, **kw)
    if lim:
        ax.set_xlim(*lim[:2])
        ax.set_ylim(*lim[2:])
    out = fig.draw(dpi=dpi)
    plt.close('all')
    return out


@pytest.mark.parametrize('name', list(_rgb_cases()))
@pytest.mark.parametrize('kw,lim,dpi', [
    ({}, None, 40),                                 # more pixels of the image than of the canvas
    ({'alpha': 0.55}, None, 100),
    ({'origin': 'lower'}, (10, 30, 5, 25), 60),     # zoomed in: an image pixel many canvas ones
    ({'extent': (5.0, -3.0, 0.0, 4.0)}, None, 50),
])
def test_rgb_made_bytes_as_drawn(name, kw, lim, dpi):
    """An RGB(A) image draws as its uint8 RGBA conversion (matplotlib's
    bytes: NaN 0, clipped, * 255, floored; RGB alpha 255) drawn as given."""
    data = _rgb_cases()[name]
    want = _draw_rgb(_fields._rgba_of(np.asarray(data)), kw, lim, dpi)
    assert np.array_equal(_draw_rgb(data, kw, lim, dpi), want)


def test_rgb_not_converted_when_made():
    rng = _rng()
    for data in (rng.random((50, 40, 3)).astype(np.float32), rng.random((50, 40, 4)),
                 (rng.random((50, 40, 3)) * 255).astype(np.uint8)):
        fig = plt.figure()
        im = fig.add_subplot().imshow(data)
        assert im._rgba is None and im.src.dtype == data.dtype
        assert np.array_equal(im.rgba, _fields._rgba_of(data))     # on demand: as before
        plt.close('all')


def test_rgb_on_a_map_as_before():
    """On a map the image is sampled through the projection from its RGBA."""
    rng = _rng()
    data = rng.uniform(-0.1, 1.1, (40, 50, 4)).astype(np.float32)

    def draw(d):
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection=crs.LambertConformal(-100, 40))
        ax.set_extent((-110, -90, 32, 48), crs.PlateCarree())
        ax.imshow(d, extent=(-108, -92, 34, 46), transform=crs.PlateCarree(), alpha=0.8)
        out = fig.draw(dpi=60)
        plt.close('all')
        return out
    assert np.array_equal(draw(data), draw(_fields._rgba_of(data)))


# --- autoscale

def _range_inputs():
    rng = _rng()
    out = []
    for dt in ('float32', 'float64'):
        a = (rng.normal(size=200_001) * 50).astype(dt)
        b = a.copy()
        b[rng.integers(0, b.size, 3000)] = np.nan
        b[rng.integers(0, b.size, 40)] = np.inf
        b[rng.integers(0, b.size, 40)] = -np.inf
        z = np.zeros(100_000, dt)
        z[rng.integers(0, z.size, 500)] = -0.0
        out += [a, b, b[::3], np.abs(b), -np.abs(b), z, np.abs(z), -np.abs(z),
                np.r_[z, dt == 'float32' and np.float32(2) or 2.0], np.full(70_000, np.nan, dt),
                np.full(70_000, np.inf, dt), np.asfortranarray(b[:200_000].reshape(400, 500))]
    for dt in ('uint8', 'uint16', 'int16', 'int32', 'int64'):
        info = np.iinfo(dt)
        a = rng.integers(info.min, info.max, 150_000, dtype=dt, endpoint=True)
        out += [a, a[::2], np.zeros(80_000, dt), np.full(80_000, info.min, dt),
                np.full(80_000, info.max, dt)]
    out += [np.ma.masked_less(rng.normal(size=(300, 300)).astype(np.float32), 0),
            np.array([np.nan, 1.0, -0.0, 0.0]), np.zeros(0, np.float32),
            (rng.random(100_000) > 0.5), rng.normal(size=90_000).astype(np.float16)]
    return out


def _same(x, y):
    if x is None or y is None:
        return x is y
    return all(type(p) is type(q) and (bool(p == q) and np.signbit(p) == np.signbit(q))
               for p, q in zip(x, y))


@pytest.mark.parametrize('way', [0, 1])
@pytest.mark.parametrize('positive', [False, True])
def test_finite_range_as_numpy(monkeypatch, way, positive):
    """The least and greatest finite value (> 0 too if positive), numpy's
    scalars of the data's type -- a zero of either sign as numpy's min and
    max give it -- on the threads and numpy's way alike."""
    c = _jit.Choice()
    c.force = way
    monkeypatch.setattr(_color, '_RANGE', c)
    for a in _range_inputs():
        want = _color._numpy_range(np.asarray(a), positive)
        assert _same(_color._finite_range(a, positive), want), (a.dtype, a.shape)


def test_finite_range_threads_tell_zeros():
    """Zeros of both signs at an end: left to numpy, which picks one."""
    z = np.zeros(10_000, np.float32)
    z[5000] = -0.0
    assert _color._range_threads(z, False) is _color._UNSURE
    assert _same(_color._range_threads(np.r_[-np.abs(z), np.float32(1)], False),
                 (np.float32(-0.0), np.float32(1)))
    assert _same(_color._range_threads(np.r_[z, np.float32(-1)] * 0 + 3, False),
                 (np.float32(3), np.float32(3)))


@pytest.mark.parametrize('norm', [None, 'log', colors.CenteredNorm(), colors.TwoSlopeNorm(0.2)])
def test_make_norm_limits_as_before(monkeypatch, norm):
    rng = _rng()
    data = [_scene(), _scene().astype(np.float64), (rng.random((300, 400)) * 4e4).astype(np.uint16),
            np.ma.masked_greater(_scene(), 0.8)]
    for d in data:
        got = _color.make_norm(norm, data=d)
        monkeypatch.setattr(_color, '_finite_range',
                            lambda a, positive=False: _color._numpy_range(np.asarray(a), positive))
        want = _color.make_norm(norm, data=d)
        monkeypatch.undo()
        for attr in ('vmin', 'vmax', 'halfrange'):
            g, w = getattr(got, attr, None), getattr(want, attr, None)
            assert type(g) is type(w) and g == w


# --- more cells than pixels: shaded as they go

def _draw_field(z, kw, size=(3, 2), dpi=50, proj=None):
    fig = plt.figure(figsize=size)
    if proj is None:
        ax = fig.add_axes([0.05, 0.1, 0.9, 0.85])
        ax.imshow(z, interpolation='nearest', **kw)
    else:
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection=proj)
        ax.set_extent((-120, -100, 30, 45), crs.PlateCarree())
        ax.imshow(z, extent=(-125, -95, 25, 50), transform=crs.PlateCarree(), **kw)
    out = fig.draw(dpi=dpi)
    plt.close('all')
    return out


def _cmap_with_ends():
    cm = plt.get_cmap('viridis').copy()
    cm.set_under('red')
    cm.set_over((0, 0, 1, 0.5))
    cm.set_bad((1, 0, 1, 0.3))
    return cm


@pytest.mark.parametrize('kw', [
    {}, {'origin': 'lower'}, {'extent': (90.0, -10.0, 0.0, 70.0)},
    {'extent': (90.0, -10.0, 0.0, 70.0), 'origin': 'lower'},
    {'alpha': 0.45}, {'vmin': 0.2, 'vmax': 0.7, 'cmap': 'cmap_ends'},
    {'vmin': 0.2, 'vmax': 0.7, 'cmap': 'cmap_ends', 'alpha': 0.6},
    {'norm': 'clip'}, {'vmin': 0.5, 'vmax': 0.5}, {'cmap': 'few'},
])
def test_pixels_shaded_as_they_go(monkeypatch, kw):
    z = _scene()
    z[::37, ::41] = np.inf
    z[5::53, ::29] = -np.inf
    kw = dict(kw)
    if kw.get('cmap') == 'cmap_ends':
        kw['cmap'] = _cmap_with_ends()
    elif kw.get('cmap') == 'few':
        kw['cmap'] = plt.get_cmap('magma', 7)
    if kw.get('norm') == 'clip':
        kw['norm'] = colors.Normalize(0.1, 0.9, clip=True)
    taken = []
    real = _render.CpuCanvas._pixels_direct

    def spy(self, *a):
        taken.append(real(self, *a))
        return taken[-1]
    monkeypatch.setattr(_render.CpuCanvas, '_pixels_direct', spy)
    got = _draw_field(z, kw)
    assert taken == [True]
    monkeypatch.setattr(_render.CpuCanvas, '_pixels_direct', lambda self, *a: False)
    assert np.array_equal(got, _draw_field(z, kw))


def test_pixels_shaded_as_they_go_on_a_map(monkeypatch):
    z = _scene((900, 1000))
    got = _draw_field(z, {'cmap': 'magma'}, proj=crs.PlateCarree())
    monkeypatch.setattr(_render.CpuCanvas, '_pixels_direct', lambda self, *a: False)
    assert np.array_equal(got, _draw_field(z, {'cmap': 'magma'}, proj=crs.PlateCarree()))


def test_other_norms_keep_their_way(monkeypatch):
    """A log norm (not a linear one) is sampled and shaded as before."""
    z = _scene() + 0.01
    called = []
    monkeypatch.setattr(_render, '_pixels_linear', lambda *a: called.append(1))
    _draw_field(z, {'norm': colors.LogNorm(0.05, 1)})
    assert not called


def test_linear_rows_as_row_linear():
    """The rows a value takes, by selects, as _k_linear's _row_linear takes
    them: NaN, inf, under, over, the limits, their neighbours, every bit
    pattern's sample."""
    rng = _rng()
    for vmin, vmax in ((0.0, 1.0), (0.2, 0.7), (-3e38, 3e38), (0.5, 0.5), (1e-30, 1e-29),
                       (6000.0, 42523.0)):
        for N in (1, 7, 256, 65536):
            for clip in (0, 1):
                span = vmax - vmin or 1.0
                edges = (vmin + span * np.arange(N + 1) / N).astype(np.float32)
                v = np.concatenate([
                    np.array([0.0, -0.0, np.nan, np.inf, -np.inf, 3.4e38, -3.4e38, 1e-45],
                             np.float32),
                    edges, np.nextafter(edges, np.float32(np.inf)),
                    np.nextafter(edges, np.float32(-np.inf)),
                    rng.integers(0, 2 ** 32, 5000, dtype=np.uint32).view(np.float32)])
                lut = np.arange(N + 3, dtype=np.uint32)
                want = np.empty((1, v.size), np.uint32)
                with np.errstate(all='ignore'):
                    _color._k_linear(v.reshape(1, -1), True, vmin, vmax, clip, lut, N, want)
                got = np.empty(v.size, np.int64)
                with np.errstate(all='ignore'):
                    _render._linear_rows(v, vmin, vmax, clip, N, got)
                assert np.array_equal(got, want[0]), (vmin, vmax, N, clip)


# --- bbox_inches='tight': measured by a draw that paints nothing

def _figures():
    z = _scene((300, 280))

    def satima():
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(z, cmap='gray')
        ax.axis('off')
        return fig

    def ticks():
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot()
        im = ax.imshow(z)
        ax.set_title('B4')
        ax.set_xlabel('x px')
        fig.colorbar(im, label='reflectance')
        fig.suptitle('scene')
        return fig

    def legend():
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot()
        ax.plot([0, 1], [0, 1], label='a line')
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        fig.text(0.98, 0.02, 'outside', ha='left')
        return fig

    def layout():
        fig, axs = plt.subplots(1, 2, figsize=(4, 2), layout='constrained')
        axs[0].imshow(z)
        axs[1].plot([1, 2, 3])
        axs[1].set_ylabel('y')
        return fig

    def tight():
        fig, axs = plt.subplots(2, 1, figsize=(3, 4))
        axs[0].imshow(z)
        axs[1].bar([1, 2], [3, 4])
        fig.tight_layout()
        return fig

    def a_map():
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(projection=crs.PlateCarree())
        ax.set_extent((-10, 30, 35, 60), crs.PlateCarree())
        ax.imshow(z, extent=(-10, 30, 35, 60), transform=crs.PlateCarree())
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        ax.set_title('map')
        return fig
    return {f.__name__: f for f in (satima, ticks, legend, layout, tight, a_map)}


@pytest.mark.parametrize('name', list(_figures()))
def test_measured_extents_are_drawn_extents(name):
    fig = _figures()[name]()
    fig.draw(dpi=80)
    drawn = list(fig._extents)
    fig._measure(80)
    assert fig._extents == drawn
    plt.close('all')


@pytest.mark.parametrize('name', list(_figures()))
@pytest.mark.parametrize('kw', [
    dict(pad_inches=0), dict(pad_inches=0, transparent=True, facecolor='none'),
    dict(pad_inches=0.13), dict(),
])
def test_tight_save_as_with_a_painting_draw(monkeypatch, name, kw):
    def save(fig):
        b = io.BytesIO()
        fig.savefig(b, format='png', dpi=77, bbox_inches='tight', **kw)
        return b.getvalue()
    got = save(_figures()[name]())
    plt.close('all')

    def painting(self, dpi, facecolor=None):
        raise RuntimeError('measure the old way')
    monkeypatch.setattr(_figure.Figure, '_measure', painting)
    assert got == save(_figures()[name]())
    plt.close('all')


def test_tight_save_paints_once(monkeypatch):
    fig = _figures()['satima']()
    drew = []
    real = _figure.Figure._draw_on
    monkeypatch.setattr(_figure.Figure, '_draw_on',
                        lambda self, *a, **k: drew.append(1) or real(self, *a, **k))
    fig.savefig(io.BytesIO(), format='png', bbox_inches='tight', pad_inches=0)
    assert len(drew) == 1
    plt.close('all')


# --- PNG written in pieces

def test_png_written_in_pieces(tmp_path):
    rng = _rng()
    for img in ((rng.random((700, 900, 4)) * 255).astype(np.uint8),
                np.repeat((rng.random((600, 800, 1)) * 4).astype(np.uint8) * 60, 4, axis=2)):
        want = _png.encode(img, dpi=100, metadata={'Software': 'fastplot'})
        b = io.BytesIO()
        _png.save(b, img, dpi=100, metadata={'Software': 'fastplot'})
        assert b.getvalue() == want
        p = tmp_path / 'a.png'
        _png.save(str(p), img, dpi=100, metadata={'Software': 'fastplot'})
        assert p.read_bytes() == want

        class Sink:
            def __init__(self):
                self.got = []

            def write(self, data):
                self.got.append(bytes(data))
        s = Sink()
        _png.save(s, img, dpi=100, metadata={'Software': 'fastplot'})
        assert s.got == [want]
