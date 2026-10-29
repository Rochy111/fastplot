"""PNG files of rendered maps in a few milliseconds: the image looked at,
filtered and deflated a band of rows per core, and written as the palette
it is when it has 256 colours or fewer.

    data = encode(img)          # uint8 (H, W, 4) RGBA, (H, W, 3) RGB, (H, W) gray
    save('t2m.png', img, dpi=100, metadata={'Software': 'fastplot'})

matplotlib's savefig (Pillow's PNG writer: a filter chosen per row and
zlib at level 6, on one thread) takes 40-120 ms over an 1800x975 map,
longer than fastplot takes to draw one. Here (tools/bench_png.py,
i7-13700K, 24 threads) the same maps take 3-8 ms, to files 10-20% bigger
than level 6's, or 10% smaller when they can be indexed:

1. **The image is looked at first** (`_scan`, a band of rows per thread):
   are all its alphas 255, and how many colours has it -- each band hashing
   its pixels' colours, RGBA read as one 32-bit word (a run of one colour
   costs a compare), and writing down each pixel's place in its band's list
   as it goes, so an indexed image needs no second look; every band stops
   counting once any has seen 257. RGBA with every alpha 255 is written as
   RGB. 256 colours or fewer, as an indexed image (PLTE, and tRNS for the
   translucent entries, sorted first so that tRNS stops at the last of
   them), 1, 2, 4 or 8 bits a pixel as the count allows: filled contours
   drawn without anti-aliased text are that, a quarter of the bytes (or an
   eighth, a sixteenth) to filter and deflate, in half the time of the same
   pixels as RGB, to 60% of the size. A gray image is indexed only if 16
   levels or fewer let it pack below 8 bits.
2. **Scanlines are filtered by numba** (`_filter`, branch-free and
   vectorized where the filter allows). A filter sees raw rows only, its
   own and the one above, so any band of rows filters on its own. No one
   filter suits every map: a map's colours come out of a colour table, a
   few hundred triples with edges and text blended between them, and
   deflate finds those triples repeated as they are -- unfiltered (None) is
   the smallest for imshow-like and contour maps, by 14-21%, where libpng's
   per-row choice (the least sum of |residual| as signed bytes over the
   five filters, `ADAPTIVE`; choosing by the entropy of the residuals, as
   zopflipng can, chose the same) is larger. On a continuous image (shaded
   relief under a translucent field) or one of blocky cells it is the other
   way round, None up to 2.2x larger. So true colour is filtered `AUTO`:
   four 32 KB samples, spread over the image, tried both ways, and the
   image filtered as the smaller came out -- by `_lz_bits`, a model of
   zlib's fast levels in numba, which chose as zlib did on every map tried
   and keeps the trial off the GIL (0.2-0.5 ms). Within 2% of the better of
   the two filters on every map tried. Indexed images are not filtered,
   as libpng does: an index is not a quantity, and its rows repeat as they
   are.
3. **Deflate runs on every core.** The image is cut at row boundaries into
   pieces of PIECE pixels (96 Ki: 19 for 1800x975), each filtered into its
   thread's own buffer, still in cache, and deflated raw there by zlib,
   which lets go of the GIL while it works; flushed with Z_SYNC_FLUSH, to
   end on a byte, the last with Z_FINISH. End to end the pieces are one
   deflate stream (a 2-byte zlib header before them, the adler32 of the
   whole after), each written as an IDAT chunk of its own, so that its CRC
   too is taken on its own thread; the pieces' adler32s are combined at the
   end (adler32_combine's arithmetic, in Python). A piece starts with an
   empty window: priming it with the 32 KB before it (as pigz does) made
   files 0.1-2% smaller, for 32 KB more to deflate a piece (or, through
   zlib's zdict, more time holding the GIL). Smaller pieces lose more to
   zlib's set-up (which holds the GIL) and to threads starting one after
   another; bigger ones leave cores idle, the slow efficiency cores last.
   The cuts depend on the image's size alone, never on the thread count:
   the file is the same bytes on 1 thread or 24. An image of one piece
   (under 96 Ki pixels, 384 KB of RGBA) is encoded on the calling thread,
   with no pool to wake; the pool is the module's, made once, a thread for
   each CPU the process may use (_hw.cpus). The scan and the trial run on
   numba's threads, a draw's count of them; then numba's workers are let
   go (_jit.park), so that they do not spin on the cores deflate needs.
4. **zlib level 2 for true colour, 3 for indexed** (LEVEL, LEVEL_PALETTE).
   zlib's levels 1-3 are one fast deflate, looking harder for matches as
   they go. On the maps tried, level 2 was 1-18% smaller than level 1 in
   the same time (13% more on shaded relief); level 3 a few percent smaller
   still, but 15% slower on imshow and 50% on relief; level 6, matplotlib's,
   2-5x slower. Indexed rows are a byte a pixel and repeat, and level 3
   deflates them to 30% less than level 1 in the same time (level 6: 27%
   less again, 20% slower).

Of the ~5 ms an RGBA imshow map takes, the scan, the trial, the filtering,
CRCs and adler32s are about one; deflate is the rest. Every kernel is
compiled once and cached (cache=True): a new process's first encode()
takes ~0.4 s, most of it importing numba.
"""
import os
import struct
import sys
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from . import _hw, _jit
from ._jit import njit, prange

LEVEL = 2                       # zlib level when none is given: true colour
LEVEL_PALETTE = 3               # and indexed
PIECE = 3 << 15                 # pixels in a piece (a band of rows, a thread's at a time)
PIECE_MIN = 1 << 15             # and at least, a smaller image's (a twelfth of it)
TRIAL = 1 << 15                 # AUTO: bytes of a sample tried both ways
TRIALS = 4                      # AUTO: samples
TABLE = 1024                    # colour hash table slots (a power of 2 > 2 * 257)

NONE, SUB, UP, AVERAGE, PAETH, ADAPTIVE, AUTO = range(7)
FILTER_RGB = AUTO               # true colour and gray
FILTER_PALETTE = NONE           # indexed
_FILTERS = {'none': NONE, 'sub': SUB, 'up': UP, 'average': AVERAGE, 'paeth': PAETH,
            'adaptive': ADAPTIVE, 'auto': AUTO}
_SIG = b'\x89PNG\r\n\x1a\n'
_ADLER_BASE = 65521


# --- kernels (nogil: a band of rows a call, on the pool's threads, but for _survey and _trial)

@njit(cache=True, nogil=True, inline='always')
def _key(px, p32, i, j, cin):
    """Pixel (i, j)'s colour as one int, r | g << 8 | b << 16 | a << 24
    (gray: r = g = b; no alpha: 255). RGBA is read as the uint32 it is
    (p32, px seen as little-endian words)."""
    if cin == 4:
        return np.int64(p32[i, j])
    p = j * cin
    if cin == 3:
        return (np.int64(px[i, p]) | (np.int64(px[i, p + 1]) << 8)
                | (np.int64(px[i, p + 2]) << 16) | (np.int64(255) << 24))
    v = np.int64(px[i, p])
    a = np.int64(px[i, p + 1]) if cin == 2 else np.int64(255)
    return v | (v << 8) | (v << 16) | (a << 24)


@njit(cache=True, nogil=True, inline='always')
def _slot(key):
    """key's first slot in a colour table of TABLE slots."""
    x = key ^ (key >> 16)
    x = (x * 0x45D9F3B) & 0xFFFFFFFF
    x ^= x >> 16
    return x & (TABLE - 1)


@njit(cache=True, nogil=True)
def _count(px, p32, i, w, cin, table, local, cols, n, idx):
    """Row i's colours added to a band's (n of them so far, in cols, met in
    that order; table/local: colour -> place in cols), each pixel's place
    written to idx[i]: the new n, or 257 on meeting a 257th colour. (Alpha
    is _alpha_ok's: a flag set in this loop made it 25x slower.)"""
    last = np.int64(-1)
    v = np.uint8(0)
    for j in range(w):
        k = _key(px, p32, i, j, cin)
        if k != last:
            last = k
            s = _slot(k)
            while table[s] != -1 and table[s] != k:
                s = (s + 1) & (TABLE - 1)
            if table[s] == -1:
                if n == 256:
                    return 257
                table[s] = k
                local[s] = n
                cols[n] = k
                n += 1
            v = local[s]
        idx[i, j] = v
    return n


@njit(cache=True, nogil=True)
def _alpha_ok(px, i, w, cin):
    """Row i's alphas all 255 (cin 2 or 4)."""
    for j in range(w):
        if px[i, j * cin + cin - 1] != 255:
            return False
    return True


@njit(cache=True, nogil=True)
def _scan(px, p32, i0, i1, w, cin, count, cols, stop, res, b, idx):
    """Rows i0:i1 looked at: whether their alphas are all 255 (res[b, 1]),
    and, if `count`, their distinct colours (cols[:res[b, 0]], in the order
    met; res[b, 0] = 257 for more than 256), each pixel's place in that
    list written to idx. Counting stops once any band has passed 256
    (stop[0], set then); alpha is looked at to the end."""
    table = np.full(TABLE, -1, np.int64)
    local = np.zeros(TABLE, np.uint8)
    n = 0
    op = 1
    hasa = cin == 2 or cin == 4
    for i in range(i0, i1):
        if count and stop[0] != 0:
            count = False
        if count:
            n = _count(px, p32, i, w, cin, table, local, cols, n, idx)
            if n > 256:
                stop[0] = 1
                count = False
        elif not hasa or op == 0:
            break                                       # nothing left to learn
        if hasa and op == 1 and not _alpha_ok(px, i, w, cin):
            op = 0
    res[b, 0] = n
    res[b, 1] = op


@njit(cache=True, nogil=True)
def _merge(cols, res, out):
    """Every band's distinct colours into out[:n]; n, or 257 for more than
    256."""
    table = np.full(TABLE, -1, np.int64)
    n = 0
    for b in range(res.shape[0]):
        if res[b, 0] > 256:
            return 257
        for q in range(res[b, 0]):
            k = cols[b, q]
            s = _slot(k)
            while table[s] != -1 and table[s] != k:
                s = (s + 1) & (TABLE - 1)
            if table[s] == -1:
                if n == 256:
                    return 257
                table[s] = k
                out[n] = k
                n += 1
    return n


@njit(cache=True, nogil=True)
def _luts(keys, cols, res, lut):
    """lut[b, q]: the palette index (keys' order) of band b's colour q."""
    table = np.full(TABLE, -1, np.int64)
    vals = np.zeros(TABLE, np.uint8)
    for v in range(keys.shape[0]):
        s = _slot(keys[v])
        while table[s] != -1:
            s = (s + 1) & (TABLE - 1)
        table[s] = keys[v]
        vals[s] = v
    for b in range(res.shape[0]):
        for q in range(res[b, 0]):
            s = _slot(cols[b, q])
            while table[s] != cols[b, q]:
                s = (s + 1) & (TABLE - 1)
            lut[b, q] = vals[s]


@njit(cache=True, nogil=True)
def _pack(idx, i0, i1, w, lut, depth, out, r0, c0):
    """Rows i0:i1 of band indices as palette indices (through lut), `depth`
    bits each, high bits first, into out[r0:, c0:] (whole bytes, the last
    one's spare bits 0); out[r, 0] = 0 (filter None) if c0 is 1."""
    per = 8 // depth
    for i in range(i0, i1):
        r = r0 + i - i0
        if c0:
            out[r, 0] = 0
        src = idx[i]
        dst = out[r, c0:]
        if depth == 8:
            for j in range(w):
                dst[j] = lut[src[j]]
            continue
        o = 0
        for j0 in range(0, w - per + 1, per):
            acc = 0
            for q in range(per):
                acc = (acc << depth) | lut[src[j0 + q]]
            dst[o] = acc
            o += 1
        m = w - o * per
        if m:
            acc = 0
            for q in range(m):
                acc = (acc << depth) | lut[src[o * per + q]]
            dst[o] = acc << (depth * (per - m))


@njit(cache=True, nogil=True, inline='always')
def _abs(x):
    m = x >> 31
    return (x ^ m) - m


@njit(cache=True, nogil=True, inline='always')
def _paeth(a, b, c):
    """PNG's Paeth predictor (int32s), branch-free."""
    pa = _abs(b - c)
    pb = _abs(a - c)
    pc = _abs(a + b - c - c)
    t = b if pb <= pc else c
    return a if (pa <= pb) & (pa <= pc) else t


@njit(cache=True, nogil=True, inline='always')
def _cost(v):
    """libpng's measure of a residual byte v (0-255): |v| as a signed byte."""
    w = np.int32(256) - v
    return v if v < w else w


@njit(cache=True, nogil=True)
def _choose(cur, prev, n, bpp):
    """libpng's choice of filter for a row: the least sum of |residual| as
    signed bytes, ties to the lower filter."""
    s = np.zeros(5, np.int64)
    for k in range(bpp):                                # a = c = 0
        x = np.int32(cur[k])
        b = np.int32(prev[k])
        s[0] += _cost(x)
        s[1] += _cost(x)
        s[2] += _cost((x - b) & 255)
        s[3] += _cost((x - (b >> 1)) & 255)
        s[4] += _cost((x - b) & 255)
    cx, ca, pb, pc = cur[bpp:n], cur[:n - bpp], prev[bpp:n], prev[:n - bpp]   # (no wraparound)
    s0 = s1 = s2 = s3 = s4 = np.int64(0)
    for k in range(n - bpp):
        x = np.int32(cx[k])
        a = np.int32(ca[k])
        b = np.int32(pb[k])
        c = np.int32(pc[k])
        s0 += _cost(x)
        s1 += _cost((x - a) & 255)
        s2 += _cost((x - b) & 255)
        s3 += _cost((x - ((a + b) >> 1)) & 255)
        s4 += _cost((x - _paeth(a, b, c)) & 255)
    s[0] += s0
    s[1] += s1
    s[2] += s2
    s[3] += s3
    s[4] += s4
    best = 0
    for f in range(1, 5):
        if s[f] < s[best]:
            best = f
    return best


@njit(cache=True, nogil=True)
def _apply(cur, prev, n, bpp, ft, dst):
    """Row `cur` (the row above: `prev`) filtered with filter ft: dst[0] the
    filter, dst[1:n + 1] the bytes."""
    dst[0] = ft
    d = dst[1 + bpp:1 + n]
    cx, ca, pb, pc = cur[bpp:n], cur[:n - bpp], prev[bpp:n], prev[:n - bpp]
    m = n - bpp
    if ft == 0:
        for k in range(n):
            dst[1 + k] = cur[k]
        return
    if ft == 2:
        for k in range(n):
            dst[1 + k] = (np.int32(cur[k]) - np.int32(prev[k])) & 255
        return
    for k in range(bpp):                                # a = c = 0
        x = np.int32(cur[k])
        b = np.int32(prev[k])
        dst[1 + k] = x if ft == 1 else (x - (b >> 1)) & 255 if ft == 3 else (x - b) & 255
    if ft == 1:
        for k in range(m):
            d[k] = (np.int32(cx[k]) - np.int32(ca[k])) & 255
    elif ft == 3:
        for k in range(m):
            d[k] = (np.int32(cx[k]) - ((np.int32(ca[k]) + np.int32(pb[k])) >> 1)) & 255
    else:
        for k in range(m):
            d[k] = (np.int32(cx[k]) - _paeth(np.int32(ca[k]), np.int32(pb[k]),
                                             np.int32(pc[k]))) & 255


@njit(cache=True, nogil=True)
def _filter(px, i0, i1, w, cin, cout, ft, out):
    """Rows i0:i1 of PNG's filtered stream into out[:i1 - i0]: each row's
    filter, then the first `cout` channels of its w pixels (px: rows of
    w * cin bytes) filtered with filter ft (0-4) or the row's own choice
    (ADAPTIVE). A filter sees raw rows only, so any band of rows filters on
    its own."""
    n = w * cout
    zero = np.zeros(n, np.uint8)
    bufa = np.empty(n, np.uint8)
    bufb = np.empty(n, np.uint8)
    prev = zero
    for i in range(max(0, i0 - 1), i1):
        if cin == cout:
            cur = px[i]
        else:                                           # RGBA -> RGB, GA -> G
            cur = bufa if (i & 1) else bufb
            row = px[i]
            if cin == 4:
                for j in range(w):
                    cur[3 * j] = row[4 * j]
                    cur[3 * j + 1] = row[4 * j + 1]
                    cur[3 * j + 2] = row[4 * j + 2]
            else:
                for j in range(w):
                    cur[j] = row[2 * j]
        if i >= i0:
            f = _choose(cur, prev, n, cout) if ft == 5 else ft
            _apply(cur, prev, n, cout, f, out[i - i0])
        prev = cur


@njit(cache=True, nogil=True)
def _lz_bits(buf):
    """About how many bits zlib's fast levels deflate buf (bytes) to: a
    greedy parse, each match the last place the next 4 bytes were seen
    (within 32 KB), literals costed at the entropy of their histogram,
    matches at their length's and distance's extra bits plus 12. AUTO's
    measure, in place of deflating the samples: it chose as zlib did on
    every map tried, and runs on numba's threads, where zlib would need the
    GIL and a pool's thread for each sample."""
    n = buf.shape[0]
    head = np.full(1 << 15, -1, np.int64)
    hist = np.zeros(256, np.int64)
    nlit = 0
    nmatch = 0
    mbits = 0.0
    i = 0
    while i + 4 <= n:
        x = (np.int64(buf[i]) | (np.int64(buf[i + 1]) << 8) | (np.int64(buf[i + 2]) << 16)
             | (np.int64(buf[i + 3]) << 24))
        h = ((x * 2654435761) & 0xFFFFFFFF) >> 17
        c = head[h]
        head[h] = i
        if (c >= 0 and i - c <= 32768 and buf[c] == buf[i] and buf[c + 1] == buf[i + 1]
                and buf[c + 2] == buf[i + 2] and buf[c + 3] == buf[i + 3]):
            m = 4
            while i + m < n and m < 258 and buf[c + m] == buf[i + m]:
                m += 1
            mbits += 12.0 + max(0.0, np.log2(m) - 2.0) + max(0.0, np.log2(i - c) - 1.0)
            nmatch += 1
            i += m
        else:
            hist[buf[i]] += 1
            nlit += 1
            i += 1
    for k in range(i, n):
        hist[buf[k]] += 1
        nlit += 1
    tot = nlit + nmatch
    bits = mbits + (nmatch * np.log2(tot / nmatch) if nmatch else 0.0)
    for v in range(256):
        if hist[v]:
            bits += hist[v] * np.log2(tot / hist[v])
    return bits


@njit(cache=True, nogil=True, parallel=True)
def _survey(px, p32, cuts, w, cin, count, cols, stop, res, idx):
    """_scan of every band, on numba's threads."""
    for b in prange(cuts.shape[0] - 1):
        _scan(px, p32, cuts[b], cuts[b + 1], w, cin, count, cols[b], stop, res, b, idx)


@njit(cache=True, nogil=True, parallel=True)
def _trial(src, w, cin, cout, starts, m, bits):
    """AUTO's trial: rows starts[s]:starts[s] + m filtered None (bits[s, 0])
    and ADAPTIVE (bits[s, 1]), and _lz_bits of each."""
    rowlen = 1 + w * cout
    for t in prange(2 * starts.shape[0]):
        s, q = t // 2, t % 2
        rows = np.empty((m, rowlen), np.uint8)
        _filter(src, starts[s], starts[s] + m, w, cin, cout, NONE if q == 0 else ADAPTIVE, rows)
        bits[s, q] = _lz_bits(rows.ravel())


# --- threads

_POOL = None
_POOL_LOCK = threading.Lock()
_LOCAL = threading.local()          # each thread's buffer, kept between calls


def _forked():
    """In a child of fork() the parent's pool threads are gone: a new pool."""
    global _POOL, _POOL_LOCK
    _POOL = None
    _POOL_LOCK = threading.Lock()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_forked)


def _pool():
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                _POOL = ThreadPoolExecutor(max_workers=_hw.cpus(),
                                           thread_name_prefix='fastplot-png')
    return _POOL


def _run(fn, n, threads):
    """fn(k) for k in range(n), on the calling thread and up to threads - 1
    of the pool's, each taking the next k as it finishes one (the cores
    are not all as fast); helpers not started by the time the work runs
    out are called off, never waited for."""
    threads = max(1, min(threads, n))
    if threads == 1:
        for k in range(n):
            fn(k)
        return
    lock = threading.Lock()
    nxt = [0]

    def work():
        while True:
            with lock:
                k = nxt[0]
                nxt[0] = k + 1
            if k >= n:
                return
            fn(k)

    futs = [_pool().submit(work) for _ in range(threads - 1)]
    try:
        work()
    finally:
        for f in futs:
            if not f.cancel():
                f.result()


def _parallel(kernel, threads, *args):
    """A parallel kernel run on at most a draw's numba threads (and
    `threads`): few enough that the scan's regions start fast."""
    _jit.load()
    nb = _jit._numba
    old = nb.get_num_threads()
    nb.set_num_threads(max(1, min(old, threads, _hw.draw_threads())))
    try:
        with _jit.serial():
            kernel(*args)
    finally:
        nb.set_num_threads(old)


def _buffer(size):
    """This thread's scratch buffer, at least `size` bytes, kept for the
    next call (a new one costs a page fault every 4 KB)."""
    buf = getattr(_LOCAL, 'buf', None)
    if buf is None or buf.size < size:
        buf = _LOCAL.buf = np.empty(max(size, 4 * PIECE + (1 << 16)), np.uint8)
    return buf


# --- zlib

def _adler32_combine(a1, a2, len2):
    """adler32 of A + B from adler32(A), adler32(B) and len(B), as zlib's
    adler32_combine."""
    rem = len2 % _ADLER_BASE
    s1 = ((a1 & 0xFFFF) + (a2 & 0xFFFF) - 1) % _ADLER_BASE
    s2 = ((a1 >> 16) + (a2 >> 16) + rem * (a1 & 0xFFFF) - rem) % _ADLER_BASE
    return (s2 << 16) | s1


def _zlib_header(level):
    """zlib's two header bytes for a 32 KB window at `level`."""
    flg = (0 if level < 2 else 1 if level < 6 else 2 if level == 6 else 3) << 6
    return bytes((0x78, flg + 31 - (0x78 * 256 + flg) % 31))


def _cuts(h, w):
    """Row boundaries of the pieces an image is scanned, filtered and
    deflated in: about PIECE pixels each, down to PIECE_MIN for a smaller
    image (a twelfth of it), so that a plot of half a million pixels is a
    dozen pieces, not five -- 30-40% faster there, the same file within
    a few bytes; evened out, and a function of the image's size alone, so
    the file never depends on the thread count."""
    piece = min(PIECE, max(PIECE_MIN, h * w // 12))
    rows = max(1, min(h, piece // w))
    pieces = -(-h // rows)
    rows = -(-h // pieces)
    return [min(k * rows, h) for k in range(pieces)] + [h]


# --- chunks

def _chunk(kind, body):
    return (struct.pack('>I', len(body)) + kind + body
            + struct.pack('>I', zlib.crc32(body, zlib.crc32(kind))))


def _phys(dpi):
    """pHYs: pixels per metre, rounded as libpng and Pillow round them."""
    dx, dy = (dpi, dpi) if np.ndim(dpi) == 0 else dpi
    dx, dy = float(dx), float(dy)
    if not (dx > 0 and dy > 0):
        raise ValueError('dpi must be positive, not %r' % (dpi,))
    return _chunk(b'pHYs', struct.pack('>IIB', int(dx / 0.0254 + 0.5), int(dy / 0.0254 + 0.5), 1))


def _text(key, value):
    """tEXt, or iTXt where the text is not Latin-1."""
    k = str(key).encode('latin-1')
    if not 1 <= len(k) <= 79 or b'\0' in k:
        raise ValueError('a PNG text key is 1-79 Latin-1 characters, not %r' % (key,))
    try:
        return _chunk(b'tEXt', k + b'\0' + str(value).encode('latin-1'))
    except UnicodeEncodeError:
        return _chunk(b'iTXt', k + b'\0\0\0\0\0' + str(value).encode('utf-8'))


_IEND = _chunk(b'IEND', b'')
_NO32 = np.zeros((1, 1), np.uint32)     # p32 when the pixels are not RGBA words
_CRC_IDAT = zlib.crc32(b'IDAT')


# --- encoding

def _prepare(img):
    """img as C-contiguous rows of bytes, (h, w * cin), and h, w, cin."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        raise TypeError('PNG: a uint8 image, not %s' % a.dtype)
    if a.ndim == 3 and a.shape[2] in (1, 2, 3, 4):
        h, w, cin = a.shape
    elif a.ndim == 2:
        (h, w), cin = a.shape, 1
    else:
        raise ValueError('PNG: an image of shape (H, W), (H, W, 3) or (H, W, 4), not %s'
                         % (a.shape,))
    if not (1 <= h < 1 << 31 and 1 <= w < 1 << 31):
        raise ValueError('PNG: no %d x %d image' % (w, h))
    return np.ascontiguousarray(a).reshape(h, w * cin), h, w, cin


def _filter_code(filter):
    if filter is None or isinstance(filter, (int, np.integer)) and 0 <= filter <= AUTO:
        return filter
    try:
        return _FILTERS[str(filter).lower()]
    except KeyError:
        raise ValueError('filter: 0-%d or one of %s, not %r'
                         % (AUTO, sorted(_FILTERS), filter)) from None


def _encode(img, level=None, dpi=None, metadata=None, threads=None, palette='auto',
            filter=None, strategy=zlib.Z_DEFAULT_STRATEGY):
    """encode(), with the choices it makes itself left open (tests and
    tools/bench_png.py): `filter` (0-4, ADAPTIVE, AUTO or their names;
    None: FILTER_RGB or FILTER_PALETTE) and zlib's `strategy`."""
    if level is not None and not 0 <= int(level) <= 9:
        raise ValueError('level: 0-9, not %r' % (level,))
    filter = _filter_code(filter)
    if palette not in ('auto', True, False):
        raise ValueError("palette: 'auto' or False, not %r" % (palette,))
    ncpu = _hw.cpus()
    threads = ncpu if threads is None else min(max(1, int(threads)), ncpu)

    px, h, w, cin = _prepare(img)
    p32 = _NO32
    if cin == 4:                                        # RGBA as words r | g << 8 ...
        p32 = px.view('<u4') if sys.byteorder == 'little' else px.view('<u4').astype(np.uint32)
    hasa = cin in (2, 4)
    cuts = _cuts(h, w)
    n = len(cuts) - 1
    keys, opaque = None, True
    if palette or hasa:
        count = bool(palette)
        cols = np.empty((n, 256), np.int64)
        res = np.empty((n, 2), np.int64)
        stop = np.zeros(1, np.uint8)
        idx = np.empty((h, w) if count else (1, 1), np.uint8)
        if n > 1 and threads > 1:
            _parallel(_survey, threads, px, p32, np.array(cuts), w, cin, count, cols, stop, res,
                      idx)
        else:
            for b in range(n):
                _scan(px, p32, cuts[b], cuts[b + 1], w, cin, count, cols[b], stop, res, b, idx)
        opaque = bool(res[:, 1].all())
        if count:
            out = np.empty(256, np.int64)
            ncol = _merge(cols, res, out)
            if ncol <= (16 if cin == 1 or cin == 2 and opaque else 256):   # (gray: < 8 bits)
                keys = out[:ncol]
                keys = keys[np.lexsort((keys, (keys >> 24) == 255))]   # translucent first

    head = []
    fused = False
    if keys is not None:
        ncol = len(keys)
        depth = 1 if ncol <= 2 else 2 if ncol <= 4 else 4 if ncol <= 16 else 8
        lut = np.zeros((n, 256), np.uint8)
        _luts(keys, cols, res, lut)
        rowbytes = (w * depth + 7) // 8
        ft = FILTER_PALETTE if filter is None else filter
        fused = ft == NONE                              # indices packed straight into the rows
        if not fused:
            src = np.empty((h, rowbytes), np.uint8)
            _run(lambda b: _pack(idx, cuts[b], cuts[b + 1], w, lut[b], depth, src, cuts[b], 0),
                 n, threads)
        scin, cout, wpix, ctype = 1, 1, rowbytes, 3
        u = keys.astype('<u4').view(np.uint8).reshape(ncol, 4)
        head.append(_chunk(b'PLTE', u[:, :3].tobytes()))
        nt = int((u[:, 3] != 255).sum())
        if nt:
            head.append(_chunk(b'tRNS', u[:nt, 3].tobytes()))
        level = LEVEL_PALETTE if level is None else int(level)
    else:
        depth = 8
        cout = cin - 1 if hasa and opaque else cin
        ctype = (0, 4, 2, 6)[cout - 1]
        src, scin, wpix = px, cin, w
        ft = FILTER_RGB if filter is None else filter
        level = LEVEL if level is None else int(level)
    rowlen = 1 + wpix * cout

    if ft == AUTO:                                      # None or ADAPTIVE, as deflates smaller
        ns = min(TRIALS, n)
        m = min(cuts[1], max(2, TRIAL // rowlen))
        mid = [min(h - m, max(0, cuts[k] + (cuts[k + 1] - cuts[k] - m) // 2))
               for k in ((2 * s + 1) * n // (2 * ns) for s in range(ns))]   # pieces spread out
        bits = np.zeros((ns, 2))
        _parallel(_trial, threads if n > 1 else 1, src, wpix, scin, cout, np.array(mid), m, bits)
        ft = NONE if bits[:, 0].sum() <= bits[:, 1].sum() else ADAPTIVE

    hdr = _zlib_header(level)
    crc0 = zlib.crc32(hdr, _CRC_IDAT)
    pieces = [None] * n

    def piece(k):
        i0, i1 = cuts[k], cuts[k + 1]
        rows = _buffer((i1 - i0) * rowlen)[:(i1 - i0) * rowlen].reshape(i1 - i0, rowlen)
        if fused:
            _pack(idx, i0, i1, w, lut[k], depth, rows, 0, 1)
        else:
            _filter(src, i0, i1, wpix, scin, cout, ft, rows)
        data = memoryview(rows).cast('B')
        z = zlib.compressobj(level, zlib.DEFLATED, -15, 9, strategy)
        c1 = z.compress(data)
        c2 = z.flush(zlib.Z_FINISH if k == n - 1 else zlib.Z_SYNC_FLUSH)
        crc = zlib.crc32(c2, zlib.crc32(c1, crc0 if k == 0 else _CRC_IDAT))
        pieces[k] = (c1, c2, crc, zlib.adler32(data), len(data))

    _jit.park()                                         # numba's workers off the cores deflate uses
    _run(piece, n, threads)

    adler = 1
    for p in pieces:
        adler = _adler32_combine(adler, p[3], p[4])
    tail = struct.pack('>I', adler)
    idat = []
    for k, (c1, c2, crc, _, _) in enumerate(pieces):
        first, last = k == 0, k == n - 1
        idat += [struct.pack('>I', len(c1) + len(c2) + 2 * first + 4 * last), b'IDAT']
        idat += [hdr, c1, c2] if first else [c1, c2]
        if last:
            idat.append(tail)
            crc = zlib.crc32(tail, crc)
        idat.append(struct.pack('>I', crc))

    out = [_SIG, _chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, depth, ctype, 0, 0, 0))] + head
    if dpi is not None:
        out.append(_phys(dpi))
    for k, v in (metadata or {}).items():
        if v is not None:
            out.append(_text(k, v))
    return b''.join(out + idat + [_IEND])


def encode(img, *, level=None, dpi=None, metadata=None, threads=None, palette='auto'):
    """The PNG file of an image, as bytes.

    img: uint8 (H, W, 4) RGBA (straight alpha), (H, W, 3) RGB or (H, W)
    gray, any strides. RGBA whose alpha is all 255 is written as RGB; an
    image of 256 colours or fewer (palette='auto') as an indexed PNG of 1,
    2, 4 or 8 bits a pixel, with tRNS for translucent colours -- the same
    pixels, smaller and sooner (palette=False: never). level: zlib's, 0-9
    (None: LEVEL). dpi: a pHYs chunk (a number, or (x, y)), as matplotlib
    writes one. metadata: {key: text}, tEXt chunks (iTXt where the text is
    not Latin-1; None skipped). threads: how many at most (None: one per
    logical CPU; an image under PIECE pixels uses the calling thread alone);
    the bytes are the same for any number."""
    return _encode(img, level, dpi, metadata, threads, palette)


def save(file, img, **kw):
    """encode(img, **kw) written to `file`: a path, or a binary file object."""
    data = encode(img, **kw)
    if hasattr(file, 'write'):
        file.write(data)
    else:
        with open(os.fspath(file), 'wb') as f:
            f.write(data)
