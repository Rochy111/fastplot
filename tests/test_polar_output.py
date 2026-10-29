"""Polar axes against matplotlib's PolarAxes, and the PDF / SVG / EPS
output."""
import io
import re
import zlib

import numpy as np
import pytest

import fastplot as fp
import fastplot.pyplot as fplt


@pytest.fixture(autouse=True)
def _fresh():
    fp.rcdefaults()
    yield
    fplt.close('all')


def _mpl():
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _pixels_off(mfig, ffig):
    from PIL import Image
    buf = io.BytesIO()
    mfig.savefig(buf, format='png')
    a = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert('RGB')).astype(int)
    b = ffig.draw(device='cpu')[..., :3].astype(int)
    assert a.shape == b.shape
    return (np.abs(a - b).max(axis=2) > 32).mean()


@pytest.mark.parametrize('setup', [
    dict(),
    dict(zero='N', direction=-1),
    dict(zero='W', offset=10, rlim=(0.5, 3)),
    dict(rscale='log', rlim=(0.1, 10)),
])
def test_polar_mapping(setup):
    """(theta, r) -> pixels, matplotlib's transData exactly."""
    plt = _mpl()
    th = np.linspace(0, 2 * np.pi, 37)
    r = np.linspace(0.2, 3, 37)
    axs = []
    for lib in (plt, fplt):
        fig = lib.figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(projection='polar')
        ax.plot(th, r)
        if 'zero' in setup:
            ax.set_theta_zero_location(setup['zero'], setup.get('offset', 0))
        if 'direction' in setup:
            ax.set_theta_direction(setup['direction'])
        if 'rscale' in setup:
            ax.set_rscale(setup['rscale'])
        if 'rlim' in setup:
            ax.set_rlim(*setup['rlim'])
        axs.append((fig, ax))
    (mfig, max_), (ffig, fax) = axs
    mfig.canvas.draw()
    ffig.draw(device='cpu')
    np.testing.assert_allclose(fax.get_ylim(), max_.get_ylim(), rtol=1e-12)
    H = ffig._px_size[1]
    ref = max_.transData.transform(np.column_stack([th, r]))
    px, py = fax._data_px(th, r)
    ok = np.isfinite(ref).all(axis=1)
    np.testing.assert_allclose(px[ok], ref[ok, 0], atol=1e-6)
    np.testing.assert_allclose(py[ok], H - ref[ok, 1], atol=1e-6)


def test_polar_ticks_like_matplotlib():
    plt = _mpl()
    th = np.linspace(0, 2 * np.pi, 100)
    out = []
    for lib in (plt, fplt):
        fig = lib.figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(projection='polar')
        ax.plot(th, 1 + 0.5 * np.sin(3 * th))
        out.append((fig, ax))
    (mfig, max_), (ffig, fax) = out
    mfig.canvas.draw()
    ffig.draw(device='cpu')
    for name in ('x', 'y'):
        maxis = getattr(max_, name + 'axis')
        mt = [(t.get_loc(), t.label1.get_text()) for t in maxis.get_major_ticks()]
        vmin, vmax = sorted(maxis.get_view_interval())
        mt = [m for m in mt if vmin - 1e-9 <= m[0] <= vmax + 1e-9]
        locs, labels, _ = fax._ticks_cache[(name, 'major')]
        np.testing.assert_allclose(locs, [m[0] for m in mt], atol=1e-12)
        assert [s.replace('−', '-') for s in labels] == \
            [m[1].replace('−', '-') for m in mt]


def test_polar_pictures():
    """Lines, a wind rose of bars, a radar-like pcolormesh: the pixels."""
    plt = _mpl()
    th = np.linspace(0, 2 * np.pi, 200)
    figs = []
    for lib in (plt, fplt):
        fig = lib.figure(figsize=(12, 4), dpi=100)
        ax = fig.add_subplot(1, 3, 1, projection='polar')
        ax.plot(th, 1 + 0.5 * np.sin(3 * th))
        ax.set_title('lines')
        ax = fig.add_subplot(1, 3, 2, projection='polar')
        ax.bar(np.linspace(0, 2 * np.pi, 12, endpoint=False), np.arange(1, 13), width=0.45,
               alpha=0.6, edgecolor='k')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax = fig.add_subplot(1, 3, 3, projection='polar')
        A, R = np.meshgrid(np.linspace(0, 2 * np.pi, 73), np.linspace(0, 100, 41))
        ax.pcolormesh(A, R, np.sin(3 * A[:-1, :-1]) * np.exp(-R[:-1, :-1] / 60), cmap='turbo')
        figs.append(fig)
    assert _pixels_off(*figs) < 0.016                   # today ~1.1%, text included


# ---------------------------------------------------------------- output

def _fig():
    fig, ax = fplt.subplots(figsize=(4, 3), dpi=80)
    ax.plot([0, 1], [0, 1])
    return fig


@pytest.mark.parametrize('transparent', [False, True])
def test_pdf(transparent):
    """A valid one-page PDF, its pixels lossless, alpha a soft mask."""
    fig = _fig()
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', transparent=transparent)
    b = buf.getvalue()
    assert b.startswith(b'%PDF-1.4') and b.rstrip().endswith(b'%%EOF')
    start = int(re.search(rb'startxref\n(\d+)', b).group(1))
    assert b[start:start + 4] == b'xref'
    for k, o in enumerate(int(x) for x in re.findall(rb'(\d{10}) 00000 n', b)):
        assert b[o:o + 10].startswith(b'%d 0 obj' % (k + 1))
    assert (b'/SMask' in b) == transparent
    m = re.search(rb'/DeviceRGB [^>]*/Length (\d+)[^>]*>>\nstream\n', b)
    px = np.frombuffer(zlib.decompress(b[m.end():m.end() + int(m.group(1))]), np.uint8)
    img = fig.draw(device='cpu', facecolor=(0, 0, 0, 0) if transparent else None)
    np.testing.assert_array_equal(px.reshape(img.shape[0], img.shape[1], 3), img[..., :3])
    assert b'/MediaBox [0 0 288.0000 216.0000]' in b    # 4 x 3 inches, in points


def test_svg_and_eps():
    import base64
    from PIL import Image
    fig = _fig()
    buf = io.BytesIO()
    fig.savefig(buf, format='svg')
    s = buf.getvalue().decode()
    assert s.startswith('<?xml') and 'width="288pt" height="216pt"' in s
    png = base64.b64decode(re.search(r'base64,([^"]+)"', s).group(1))
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(png)).convert('RGBA')),
                                  fig.draw(device='cpu'))
    buf = io.BytesIO()
    fig.savefig(buf, format='eps')
    assert buf.getvalue().startswith(b'%!PS-Adobe')
