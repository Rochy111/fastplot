"""fastplot._png: every PNG it writes decodes (Pillow, and zlib for the
stream's adler32) to exactly the pixels given, whatever the colour type,
bit depth, filter, size, strides or thread count."""
import io
import struct
import threading
import zlib

import numpy as np
import pytest
from PIL import Image

from fastplot import _png
from fastplot._png import encode, save

FILTERS = list(range(_png.AUTO + 1))
RNG = np.random.default_rng(1234)


# --- helpers

def chunks(data):
    """[(type, body)] of a PNG, every CRC checked."""
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    out, p = [], 8
    while p < len(data):
        n, = struct.unpack('>I', data[p:p + 4])
        kind, body = data[p + 4:p + 8], data[p + 8:p + 8 + n]
        crc, = struct.unpack('>I', data[p + 8 + n:p + 12 + n])
        assert crc == zlib.crc32(kind + body), kind
        out.append((kind, body))
        p += 12 + n
    assert out[0][0] == b'IHDR' and out[-1] == (b'IEND', b'')
    return out


def ihdr(data):
    """(width, height, bit depth, colour type)."""
    return struct.unpack('>IIBB', chunks(data)[0][1][:10])


def inflate(data):
    """The zlib stream of the IDATs, inflated (zlib checks its adler32)."""
    return zlib.decompress(b''.join(b for k, b in chunks(data) if k == b'IDAT'))


def decoded(data, img):
    """The PNG decoded by Pillow, in img's channels."""
    im = Image.open(io.BytesIO(data))
    im.load()
    if img.ndim == 2:
        return np.asarray(im.convert('L'))
    mode = {1: 'L', 2: 'LA', 3: 'RGB', 4: 'RGBA'}[img.shape[2]]
    back = np.asarray(im.convert(mode))
    return back[..., None] if img.shape[2] == 1 else back


def roundtrip(img, **kw):
    """encode(img) checked against Pillow and zlib; the bytes."""
    data = _png._encode(img, **kw)
    np.testing.assert_array_equal(decoded(data, img), img)
    w, h, depth, ctype = ihdr(data)
    assert (w, h) == (img.shape[1], img.shape[0])
    rowbytes = (w * depth * {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype] + 7) // 8
    raw = inflate(data)
    assert len(raw) == h * (1 + rowbytes)
    assert all(f <= 4 for f in raw[::1 + rowbytes])
    return data


def smooth(h, w, c=4, seed=0):
    """A smooth colourful image of many colours."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    ch = [127.5 + 127.5 * np.sin(x / (7 + 3 * k) + y / (11 + 2 * k) + seed + k) for k in range(c)]
    return np.ascontiguousarray(np.stack(ch, -1).round().astype(np.uint8))


def few(h, w, ncol, alpha=False, seed=0):
    """An RGBA image of exactly ncol colours (blocks of them, as filled
    contours are), translucent ones too if `alpha`."""
    rng = np.random.default_rng(seed)
    cols = np.unique(rng.integers(0, 256, (ncol * 2, 4), dtype=np.uint8), axis=0)
    rng.shuffle(cols)
    cols = cols[:ncol]
    cols[:, 3] = 255 if not alpha else np.minimum(cols[:, 3], 254)
    if alpha:
        cols[: ncol // 2, 3] = 255                      # half opaque, half not
    assert len(np.unique(cols, axis=0)) == ncol
    y, x = np.mgrid[0:h, 0:w]
    lab = ((x // 5) * 7 + (y // 3) * 3 + (x * y) // 97) % ncol
    lab.flat[:ncol] = np.arange(ncol)                   # every colour present
    return np.ascontiguousarray(cols[lab])


# --- colour types

def test_rgba_partial_alpha():
    img = smooth(61, 83)
    img[..., 3] = np.where(img[..., 3] > 40, img[..., 3], 255)
    data = roundtrip(img)
    assert ihdr(data)[2:] == (8, 6)


def test_opaque_rgba_written_as_rgb():
    img = smooth(50, 70)
    img[..., 3] = 255
    data = roundtrip(img)
    assert ihdr(data)[2:] == (8, 2)
    assert Image.open(io.BytesIO(data)).mode == 'RGB'


def test_rgb():
    data = roundtrip(smooth(45, 77, 3))
    assert ihdr(data)[2:] == (8, 2)


def test_gray():
    data = roundtrip(smooth(40, 90, 1)[..., 0])
    assert ihdr(data)[2:] == (8, 0)


def test_gray_few_levels_packs():
    img = (smooth(40, 90, 1)[..., 0] // 32 * 32).astype(np.uint8)     # 8 levels
    data = roundtrip(img)
    assert ihdr(data)[2:] == (4, 3)
    assert ihdr(_png._encode(img, palette=False))[2:] == (8, 0)


def test_gray_alpha_input():
    img = smooth(30, 41, 2)
    assert ihdr(roundtrip(img))[3] == 4
    img[..., 1] = 255
    assert ihdr(roundtrip(img))[2:] == (8, 0)


@pytest.mark.parametrize('ncol, depth', [(1, 1), (2, 1), (3, 2), (4, 2), (5, 4), (16, 4),
                                         (17, 8), (200, 8), (256, 8)])
@pytest.mark.parametrize('alpha', [False, True])
def test_palette(ncol, depth, alpha):
    img = few(37, 53, ncol, alpha and ncol > 1)
    data = roundtrip(img)
    assert ihdr(data)[2:] == (depth, 3)
    kinds = [k for k, _ in chunks(data)]
    plte = dict(chunks(data))[b'PLTE']
    assert len(plte) == 3 * ncol
    assert (b'tRNS' in kinds) == (alpha and ncol > 1)
    if b'tRNS' in kinds:
        assert len(dict(chunks(data))[b'tRNS']) == ncol - ncol // 2   # translucent ones first
    unpal = _png._encode(img, palette=False)
    assert ihdr(unpal)[3] == (6 if alpha and ncol > 1 else 2)
    np.testing.assert_array_equal(decoded(unpal, img), img)


def test_257_colours_not_indexed():
    img = few(37, 53, 257)
    data = roundtrip(img)
    assert ihdr(data)[2:] == (8, 2)


def test_palette_bands_meet_colours_in_other_orders():
    """Pieces each meet their colours in their own order (and some only in
    one piece); the palette maps every one back right."""
    h, w = 1200, 300                                    # several pieces
    img = few(h, w, 250, alpha=True)
    img[h // 2:] = img[h // 2:, ::-1]
    img[-50:] = img[-50:, :, [2, 1, 0, 3]][:, ::-1]     # (new colours at the end)
    n = len(np.unique(img.reshape(-1, 4), axis=0))
    assert len(_png._cuts(h, w)) > 3
    data = roundtrip(img)
    assert ihdr(data)[3] == (3 if n <= 256 else 6)


def test_more_than_256_over_pieces():
    """No piece has more than 256 colours, the image has: not indexed."""
    h, w = 1200, 300
    a = few(h // 2, w, 200, seed=1)
    b = few(h - h // 2, w, 200, seed=2)
    img = np.concatenate([a, b])
    assert len(np.unique(img.reshape(-1, 4), axis=0)) > 256
    assert ihdr(roundtrip(img))[2:] == (8, 2)


def test_tiny_image_many_rows_of_one_colour():
    img = np.zeros((300, 1, 4), np.uint8)
    img[..., 3] = 255
    assert ihdr(roundtrip(img))[2:] == (1, 3)


# --- sizes, strides, filters, threads

SHAPES = [(1, 1), (1, 9), (9, 1), (3, 7), (2, 3), (5, 1000), (1000, 5)]


def images(h, w):
    rgba = smooth(h, w, 4, seed=h + w)
    opaque = rgba.copy()
    opaque[..., 3] = 255
    return {'rgba': rgba, 'opaque': opaque, 'rgb': smooth(h, w, 3),
            'gray': smooth(h, w, 1)[..., 0], 'pal': few(h, w, min(h * w, 12)),
            'pal_alpha': few(h, w, min(h * w, 40), alpha=True)}


@pytest.mark.parametrize('shape', SHAPES)
def test_sizes(shape):
    for name, img in images(*shape).items():
        roundtrip(img)


@pytest.mark.parametrize('ft', FILTERS)
def test_filters(ft):
    for shape in [(1, 1), (3, 7), (40, 33)]:
        for name, img in images(*shape).items():
            data = roundtrip(img, filter=ft)
            if ft <= 4:
                raw = inflate(data)
                assert set(raw[::len(raw) // ihdr(data)[1]]) == {ft}


def test_filter_names():
    img = smooth(20, 30)
    for name, code in _png._FILTERS.items():
        assert _png._encode(img, filter=name) == _png._encode(img, filter=code)
    with pytest.raises(ValueError):
        _png._encode(img, filter='best')
    with pytest.raises(ValueError):
        _png._encode(img, filter=9)


def test_auto_chooses_none_or_adaptive():
    """AUTO is None or ADAPTIVE, whichever deflates smaller: adaptive for a
    continuous image, None for one coloured through a colour table."""
    cont = smooth(300, 400)
    y, x = np.mgrid[0:300, 0:400]
    table = smooth(1, 256)[0]
    table[:, 3] = 255
    lut = table[((np.sin(x / 37.0) * np.cos(y / 23.0) + 1) * 127.9).astype(int)]
    for img, want in ((cont, 'adaptive'), (lut, 'none')):
        auto = roundtrip(img, filter='auto', palette=False)
        assert auto == _png._encode(img, filter=want, palette=False)
        assert len(auto) < len(_png._encode(img, filter={'none': 'adaptive'}.get(want, 'none'),
                                            palette=False))


def test_non_contiguous():
    big = smooth(80, 120)
    roundtrip(big[::2, ::3])
    roundtrip(big[..., :3])                             # RGB view of RGBA
    roundtrip(big[..., 2])                              # gray, strided
    roundtrip(big.transpose(1, 0, 2))
    roundtrip(few(40, 60, 9)[::-1, ::-1])
    roundtrip(np.asfortranarray(smooth(30, 50, 3)))


@pytest.mark.parametrize('ft', [None] + FILTERS)
def test_threads_give_the_same_bytes(ft):
    for name, img in images(700, 333).items():
        assert len(_png._cuts(*img.shape[:2])) > 2
        one = _png._encode(img, threads=1, filter=ft)
        assert one == _png._encode(img, threads=3, filter=ft)
        assert one == _png._encode(img, filter=ft)
    assert len([k for k, _ in chunks(one) if k == b'IDAT']) > 1


def test_concurrent_calls():
    imgs = [smooth(400, 500, seed=s) for s in range(3)] + [few(400, 500, 30)]
    want = [encode(im) for im in imgs]
    got = [None] * 12

    def go(i):
        got[i] = encode(imgs[i % 4])

    ts = [threading.Thread(target=go, args=(i,)) for i in range(12)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert got == want * 3


@pytest.mark.parametrize('level', range(10))
def test_levels(level):
    img = smooth(200, 300)
    data = roundtrip(img, level=level)
    cmf, flg = b''.join(b for k, b in chunks(data) if k == b'IDAT')[:2]
    assert cmf == 0x78 and (cmf * 256 + flg) % 31 == 0
    roundtrip(few(200, 300, 30), level=level)


def test_default_levels():
    img = smooth(100, 300)
    assert encode(img) == encode(img, level=_png.LEVEL)
    pal = few(100, 300, 20)
    assert encode(pal) == encode(pal, level=_png.LEVEL_PALETTE)


def test_strategy():
    img = smooth(200, 300)
    for st in (zlib.Z_FILTERED, zlib.Z_HUFFMAN_ONLY, zlib.Z_RLE, zlib.Z_FIXED):
        roundtrip(img, strategy=st)


def test_adler32_combine():
    rng = np.random.default_rng(7)
    for n in (0, 1, 5, 5552, 65521, 100000, 300007):
        data = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        cut = sorted(rng.integers(0, n + 1, 3))
        a = 1
        for p, q in zip([0] + cut, cut + [n]):
            a = _png._adler32_combine(a, zlib.adler32(data[p:q]), q - p)
        assert a == zlib.adler32(data)
    ff = b'\xff' * 70000
    assert _png._adler32_combine(1, zlib.adler32(ff), 70000) == zlib.adler32(ff)


# --- chunks, files, errors

def test_dpi_and_metadata():
    img = smooth(20, 30)
    meta = {'Software': 'fastplot 0.1', 'Title': 'T2m', 'Skip': None,
            'Comment': 'température 25 °C', 'Author': 'Zoë 日本'}
    data = encode(img, dpi=100, metadata=meta)
    im = Image.open(io.BytesIO(data))
    assert im.info['dpi'] == pytest.approx((100, 100), abs=0.01)
    assert im.text['Software'] == 'fastplot 0.1'
    assert im.text['Title'] == 'T2m'
    assert im.text['Comment'] == 'température 25 °C'        # Latin-1: tEXt
    assert im.text['Author'] == 'Zoë 日本'                    # not: iTXt
    assert 'Skip' not in im.text
    kinds = [k for k, _ in chunks(data)]
    assert kinds.index(b'pHYs') < kinds.index(b'IDAT') and b'iTXt' in kinds
    phys = dict(chunks(data))[b'pHYs']
    assert struct.unpack('>IIB', phys) == (3937, 3937, 1)       # 100 dpi, as libpng writes it
    im2 = Image.open(io.BytesIO(encode(img, dpi=(72, 300))))
    assert im2.info['dpi'] == pytest.approx((72, 300), abs=0.01)
    assert b'pHYs' not in [k for k, _ in chunks(encode(img))]
    with pytest.raises(ValueError):
        encode(img, dpi=0)
    with pytest.raises(ValueError):
        encode(img, metadata={'': 'x'})


def test_save(tmp_path):
    img = few(30, 40, 7)
    p = tmp_path / 'a.png'
    save(p, img, dpi=96)
    save(str(tmp_path / 'b.png'), img)
    b = io.BytesIO()
    save(b, img, dpi=96)
    assert p.read_bytes() == b.getvalue() == encode(img, dpi=96)
    assert (tmp_path / 'b.png').read_bytes() == encode(img)
    np.testing.assert_array_equal(np.asarray(Image.open(p).convert('RGBA')), img)


def test_errors():
    with pytest.raises(TypeError):
        encode(np.zeros((4, 4, 4), np.float32))
    with pytest.raises(ValueError):
        encode(np.zeros((4, 4, 5), np.uint8))
    with pytest.raises(ValueError):
        encode(np.zeros((4,), np.uint8))
    with pytest.raises(ValueError):
        encode(np.zeros((0, 4, 4), np.uint8))
    with pytest.raises(ValueError):
        encode(np.zeros((4, 4, 4), np.uint8), level=10)
    with pytest.raises(ValueError):
        encode(np.zeros((4, 4, 4), np.uint8), palette='yes')


def test_pillow_agrees_on_a_map_like_image():
    """A larger image with AA-like edges between flat areas, as a map:
    the same pixels as Pillow's own PNG, smaller than level 1's."""
    base = np.repeat(np.repeat(few(98, 180, 18), 10, 0), 10, 1)
    edge = smooth(980, 1800)
    img = np.where((np.arange(1800) % 97 < 2)[None, :, None], edge, base)
    img[..., 3] = 255
    data = roundtrip(img)
    ref = io.BytesIO()
    Image.fromarray(img).save(ref, 'PNG', compress_level=1)
    np.testing.assert_array_equal(decoded(ref.getvalue(), img), decoded(data, img))
