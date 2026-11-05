"""The kernel image (_image.py): kernels run with no numba at all -- the
parallel ones on numba's own threading layer -- and draw as numba does, and
whatever an image cannot do -- no image for this machine, a damaged one, a
kernel or argument types it does not hold -- goes to numba. In fresh
interpreters, each with images of its own (FASTPLOT_CACHE_DIR): an image is
looked for once a process."""
import json
import marshal
import os
import shutil
import struct
import subprocess
import sys
import time

import numpy as np
import pytest

from fastplot import _image

pytestmark = pytest.mark.skipif(_image.cpu_id() is None, reason='no kernel image on this machine')

# figures of the common kinds, drawn and saved: their pixels and PNG files
DRAW = '''
import hashlib, io, json, sys, warnings
warnings.simplefilter('ignore')
import numpy as np
import fastplot
from fastplot import _jit, crs, feature
from fastplot import pyplot as plt


def figures():
    x = np.linspace(0, 10, 3000)
    fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
    ax.plot(x, np.sin(x), 'o-', markevery=300, label='a')
    ax.plot(x, np.cos(x), '--', lw=2, label='b')
    ax.legend()
    ax.set_title('lines')
    yield fig
    xx, yy = np.meshgrid(np.linspace(-3, 3, 60), np.linspace(-2, 2, 40))
    zz = np.exp(-xx ** 2 - yy ** 2)
    fig, axs = plt.subplots(1, 2, figsize=(5, 2.5), dpi=80)
    cf = axs[0].contourf(xx, yy, zz, levels=8)
    axs[0].clabel(axs[0].contour(xx, yy, zz, levels=4, colors='k'))
    fig.colorbar(cf, ax=axs[0])
    m = axs[1].pcolormesh(zz, cmap='RdBu_r')
    fig.colorbar(m, ax=axs[1], extend='both')
    yield fig
    pc = crs.PlateCarree()
    lon, lat = np.linspace(0, 359, 72), np.linspace(-90, 90, 37)
    z = (np.cos(np.radians(lat))[:, None] * np.sin(np.radians(lon))[None, :]).astype(np.float32)
    fig = fastplot.figure(figsize=(4, 3), dpi=80)
    ax = fig.add_subplot(projection=crs.LambertConformal(central_longitude=-97, central_latitude=38))
    ax.set_extent([-120, -70, 22, 52], crs=pc)
    ax.contourf(lon, lat, z, levels=8, transform=pc)
    ax.coastlines()
    ax.add_feature(feature.BORDERS)
    ax.gridlines(draw_labels=True)
    yield fig
    fig, ax = plt.subplots(figsize=(3, 2), dpi=80)
    ax.text(0.1, 0.5, 'Text $\\\\alpha^2$', fontsize=14)
    ax.scatter(np.arange(40), np.arange(40) % 7, c=np.arange(40), s=np.arange(40) + 4.0)
    yield fig
    fig = fastplot.figure(figsize=(16, 9), dpi=100)              # (every thread a draw has)
    ax = fig.add_subplot(projection=pc)
    ax.pcolormesh(lon, lat, z, transform=pc)
    ax.coastlines()
    yield fig


values = []
for fig in figures():
    values.append(hashlib.sha256(fig.draw().tobytes()).hexdigest())
    b = io.BytesIO()
    fig.savefig(b, format='png')
    values.append(hashlib.sha256(b.getvalue()).hexdigest())
    plt.close('all')
print(json.dumps(dict(values=values, image=_jit.image() is not None, numba='numba' in sys.modules)))
'''


def run(code, *args, cache=None, **env):
    """What a fresh interpreter printed last, as JSON: with an image unless
    `env` says otherwise, whatever the tests themselves run with."""
    e = {k: v for k, v in os.environ.items() if k not in ('FASTPLOT_IMAGE', 'FASTPLOT_CACHE_DIR')}
    e.update(FASTPLOT_AUTOWARM='0', FASTPLOT_IMAGE_BUILD='0', FASTPLOT_DEVICE='cpu')
    e.update(env)
    if cache is not None:
        e['FASTPLOT_CACHE_DIR'] = str(cache)
    out = subprocess.run([sys.executable, '-c', code] + [str(a) for a in args], capture_output=True, text=True,
                         timeout=900, env=e)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture(scope='module')
def images(tmp_path_factory):
    """A directory with this machine's image, built as warm() builds one."""
    d = tmp_path_factory.mktemp('images')
    env = dict(os.environ, FASTPLOT_CACHE_DIR=str(d), FASTPLOT_AUTOWARM='0', FASTPLOT_IMAGE_BUILD='1',
               FASTPLOT_DEVICE='cpu', FASTPLOT_PRELOAD='0')
    env.pop('FASTPLOT_IMAGE', None)
    out = subprocess.run([sys.executable, '-c', 'import fastplot; fastplot.warm(processes=1)'], capture_output=True,
                         text=True, timeout=3600, env=env)
    assert out.returncode == 0, out.stderr[-3000:]
    assert os.path.exists(d / _image._name()) and not os.path.exists(d / (_image._name() + '.failed'))
    return d


@pytest.fixture(scope='module')
def by_numba():
    return run(DRAW, FASTPLOT_IMAGE='0')['values']


def copy(images, tmp_path):
    d = tmp_path / 'images'
    shutil.copytree(images, d)
    return d


def test_draws_without_numba_as_numba_does(images, by_numba):
    got = run(DRAW, cache=images)
    assert got['image'] and not got['numba']
    assert got['values'] == by_numba


def test_without_the_preload_thread_too(images, by_numba):
    got = run(DRAW, cache=images, FASTPLOT_PRELOAD='0')
    assert got['image'] and not got['numba']
    assert got['values'] == by_numba


def test_figures_from_many_threads_at_once(images):
    code = '''
import hashlib, json, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import fastplot
from fastplot import _jit


def draw(i):
    fig = fastplot.figure(figsize=(3, 2), dpi=60)
    ax = fig.add_subplot()
    ax.plot(np.arange(50), np.sin(np.arange(50) / (1 + i % 4)))
    ax.contourf(np.random.default_rng(i % 4).random((20, 30)), levels=5)
    ax.set_title('t%d' % (i % 4))
    return hashlib.sha256(fig.draw().tobytes()).hexdigest()


with ThreadPoolExecutor(8) as ex:
    got = list(ex.map(draw, range(32)))
print(json.dumps(dict(values=got[:4], same=got == got[:4] * 8, image=_jit.image() is not None,
                      numba='numba' in sys.modules)))
'''
    got = run(code, cache=images)
    assert got['image'] and not got['numba'] and got['same']
    assert got['values'] == run(code, FASTPLOT_IMAGE='0')['values']


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='no fork here')
def test_workers_forked_after_the_image_was_used(images, by_numba):
    """Units the parent mapped come along; those a child maps it reads for
    itself, at its own place in the file the parent opened; the parallel
    kernels start a pool of their own in each. (The parent runs no parallel
    kernel first: GNU OpenMP, numba's layer on Linux, cannot be forked once it
    has run one, and numba stops such a child -- under numba too, which runs
    one as it starts.)"""
    code = DRAW.replace('values = []', '''
import multiprocessing


def hashes(_):
    out = []
    for fig in figures():
        out.append(hashlib.sha256(fig.draw().tobytes()).hexdigest())
        b = io.BytesIO()
        fig.savefig(b, format='png')
        out.append(hashlib.sha256(b.getvalue()).hexdigest())
        plt.close('all')
    return out, 'numba' in sys.modules


from fastplot import _draw
_draw._stride(7)                                    # (the image open, a serial unit mapped)
with multiprocessing.get_context('fork').Pool(2) as pool:
    got = pool.map(hashes, range(2))
print(json.dumps(dict(values=got[0][0], same=got[0] == got[1], numba=any(n for _, n in got),
                      image=_jit.image() is not None)))
raise SystemExit
''')
    got = run(code, cache=images)
    assert got['image'] and not got['numba'] and got['same']
    assert got['values'] == by_numba


def test_prewarm_maps_what_this_machine_mostly_uses(images):
    """The units a quarter of processes or more used, mapped ahead -- but
    where a child can be forked, neither numba's pool nor a parallel unit
    (mapping one starts the pool): GNU OpenMP cannot be forked once started."""
    code = '''
import json, os
from fastplot import _image
im = _image.open_image()
keys = [(name, key) for name, units in im.kernels.items() for key in units]
par = [k for k in keys if _image._needs_layer(im, k)][:5]
ser = [k for k in keys if not _image._needs_layer(im, k)][:5]
_image._usage = lambda im: dict(processes=4, units=dict.fromkeys(ser + par, 1))


def mapped(k):
    u = im.units.get(id(im.kernels[k[0]][k[1]]))
    return u is not None and u.fn is not None


_image.prewarm(im)
print(json.dumps(dict(fork=hasattr(os, 'fork'), par=len(par), serial=all(map(mapped, ser)),
                      parallel=any(map(mapped, par)), pool=_image._parts.get('pool') is not None)))
'''
    got = run(code, cache=images, FASTPLOT_PRELOAD='0')
    assert got['par'] and got['serial']
    assert got['parallel'] == got['pool'] == (not got['fork'])


def test_thread_counts_on_the_layer_without_numba(images):
    code = '''
import json, sys
import numpy as np
import fastplot
from fastplot import _jit, _raster
z = np.zeros(100000, np.float32)
z[::7] = np.nan
z[3] = np.inf
with _jit.threads(2):
    inside = _jit.team(), _raster._not_finite(z)
size, layer = _jit.pool_size(), _jit.layer()
outside = _jit.team()
_jit.park()
print(json.dumps(dict(inside=inside, outside=outside, after_park=_jit.team(), size=size, layer=layer,
                      got=_raster._not_finite(z), image=_jit.image() is not None, numba='numba' in sys.modules)))
'''
    got = run(code, cache=images)
    assert got['image'] and not got['numba']
    numba = run(code, FASTPLOT_IMAGE='0')
    assert got['inside'] == [min(2, got['size']), numba['got']] and got['got'] == numba['got']
    assert got['outside'] == got['after_park'] == got['size'] == numba['size']
    assert got['layer'] == numba['layer']


def test_numba_coming_up_later_keeps_the_count(images):
    code = '''
import json, sys
import numpy as np
from fastplot import _jit, _raster
z = np.zeros(1000, np.float32)
with _jit.threads(3):
    _raster._not_finite(z)
    _jit.load()                                     # numba up: the same DLL, the same count
    import numba
    got = _jit.team(), numba.get_num_threads(), numba.threading_layer()
print(json.dumps(dict(got=got, layer=_jit.layer(), image=_jit.image() is not None)))
'''
    got = run(code, cache=images)
    assert got['image'] and got['got'][:2] == [3, 3]
    assert got['got'][2] == ('omp' if got['layer'] == 'fastplot' else got['layer'])     # (what it stands in for)


def test_the_way_in_from_python(images, tmp_path):
    """A unit's function of Python's own takes what it can (it is the
    stand-in's first try), and leaves anything else -- an ndarray subclass,
    a numpy scalar of another width, keywords -- to the slower ways, with the
    same results."""
    code = '''
import json, sys
import numpy as np
from fastplot import _draw, _jit, _raster


class Sub(np.ndarray):
    pass


z = np.zeros(1000, np.float32)
z[::9] = np.nan
m = np.memmap(sys.argv[1], np.float32, 'w+', shape=(1000,))
m[:] = z
calls = [(z,), (z,), (z.view(Sub),), (m,), (m,)]
got = [list(_raster._not_finite(*a)) for a in calls]
entries = len(getattr(_raster._not_finite, 'entries', ()))
steps = [_draw._stride(n) for n in (7, 7, np.int64(40), np.int32(40), np.uint8(9))]
numba = 'numba' in sys.modules
im = _jit.image()
missed = im.missed if im is not None else 0
steps.append(_draw._stride(n=9))                    # (keywords: numba's)
print(json.dumps(dict(got=got, steps=steps, entries=entries, image=im is not None, numba=numba,
                      missed=missed)))
'''
    got = run(code, tmp_path / 'mm.bin', cache=images)
    assert got['image'] and got['entries'] == 1 and not got['numba'] and not got['missed']
    numba = run(code, tmp_path / 'mm2.bin', FASTPLOT_IMAGE='0')
    assert got['got'] == numba['got'] and got['steps'] == numba['steps']


def test_an_error_raised_as_numba_raises_it(images):
    code = '''
import json, sys
import numpy as np
from fastplot import _jit, _png
idx, lut, out = np.zeros((2, 4), np.uint8), np.zeros(4, np.uint8), np.zeros((2, 5), np.uint8)
got = []
for _ in range(2):                                  # (the second: the unit's own way in)
    try:
        _png._pack(idx, 0, 2, 4, lut, 0, out, 0, 1)     # (a depth of 0 bits)
        got.append(None)
    except Exception as e:
        got.append([type(e).__name__, str(e)])
print(json.dumps(dict(got=got, image=_jit.image() is not None, numba='numba' in sys.modules)))
'''
    got = run(code, cache=images)
    assert got['image'] and not got['numba']
    want = run(code, FASTPLOT_IMAGE='0')['got']
    assert got['got'] == want and want[0][0] == 'ZeroDivisionError'


def test_arrays_returned_as_numba_returns_them(images):
    """A kernel's own array, small (copied out) or large (numpy's view of
    its memory, let go with it), and an argument handed back as it came: the
    argument itself."""
    code = '''
import gc, hashlib, json, sys
import numpy as np
from fastplot import _draw, _jit
got = []
for n in (40, 40000):                               # (40000 x 2 doubles: not copied)
    xy = np.random.default_rng(n).uniform(0, 500, (n, 2))
    offsets = np.arange(0, n + 1, 20, dtype=np.int64)
    sv = np.full(offsets.size - 1, 0.5)
    for _ in range(2):                              # (the second: the unit's own way in)
        out, snapped = _draw._snap_parts(xy, offsets, sv, False)
        out2, snapped2 = _draw._snap_parts(xy, offsets, sv, True)     # (slanted: none snapped)
        gc.collect()
        got.append([hashlib.sha256(out.tobytes()).hexdigest(), snapped, out is xy, out.flags.writeable,
                    out2 is xy, snapped2])
        del out, out2
        gc.collect()
print(json.dumps(dict(got=got, image=_jit.image() is not None, numba='numba' in sys.modules)))
'''
    got = run(code, cache=images)
    assert got['image'] and not got['numba']
    want = run(code, FASTPLOT_IMAGE='0')['got']
    assert got['got'] == want and want[0][4] is True


def test_kept_memory_between_frames(images):
    """A run of frames reuses the memory the one before let go of; numpy's
    handler is its own again outside a draw; arrays made in a draw (an image
    kept from an early frame) are whole after later frames have reused
    memory."""
    code = '''
import ctypes, hashlib, io, json, sys
import numpy as np
import fastplot
from fastplot import _jit
try:
    from numpy._core.multiarray import get_handler_name
except ImportError:
    from numpy.core.multiarray import get_handler_name
kept, sums = [], []
for k in range(6):
    fig = fastplot.figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot()
    z = np.random.default_rng(k % 2).random((300, 400))
    ax.pcolormesh(z)
    ax.contour(z, levels=4, colors='k')
    a = fig.draw()
    kept.append(a)
    sums.append(hashlib.sha256(a.tobytes()).hexdigest())
    fig.savefig(io.BytesIO(), format='png')
im = _jit.image()
hits = ctypes.c_int64.from_address(im.kept()['fastplot_kept_hits']).value if im else 0
print(json.dumps(dict(hits=hits, after=get_handler_name(), early=[hashlib.sha256(a.tobytes()).hexdigest()
                      for a in kept], sums=sums, image=im is not None, numba='numba' in sys.modules)))
'''
    got = run(code, cache=images)
    assert got['image'] and not got['numba'] and got['hits'] > 0
    assert got['after'] == 'default_allocator' and got['early'] == got['sums']
    assert got['sums'] == run(code, FASTPLOT_IMAGE='0')['sums']


def test_kept_memory_gives_back_what_a_frame_did_not_take(images):
    """A frame's end gives back the kept blocks let go before it that it did
    not take again: frames of one size keep one frame's let-go; after one
    much larger frame, the next smaller ones give its blocks back."""
    code = '''
import ctypes, json, sys
import numpy as np
import fastplot
from fastplot import _jit


def frame(w, h):
    fig = fastplot.figure(figsize=(w, h), dpi=100)
    ax = fig.add_subplot()
    ax.pcolormesh(np.random.default_rng(0).random((200, 300)))
    np.asarray(fig.canvas.buffer_rgba()).sum()


for _ in range(3):
    frame(6, 4)
im = _jit.image()                       # (looked for by the first kernel)
held = lambda: ctypes.c_int64.from_address(im.kept()['fastplot_kept_held']).value
small = held()
frame(30, 20)                           # (a 3000 x 2000 canvas: 24 MB)
big = held()
for _ in range(3):
    frame(6, 4)
print(json.dumps(dict(small=small, big=big, after=held(), image=im is not None,
                      numba='numba' in sys.modules)))
'''
    got = run(code, cache=images)
    assert got['image'] and not got['numba']
    assert got['big'] >= 24 << 20 > got['small']
    assert got['after'] <= got['small'] + (1 << 20)


def test_a_dropped_figure_lets_its_picture_go_at_once(images):
    """A figure lives in reference cycles; its picture does not wait for a
    collection: it goes when the caller lets go of what buffer_rgba gave,
    and the next frame's canvas is that memory again (no new pages)."""
    code = '''
import gc, json, weakref
import numpy as np
import fastplot
import fastplot.patheffects as pe
gc.disable()
refs = []
for k in range(3):
    fig = fastplot.figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot()
    ax.contourf(np.random.default_rng(k).random((30, 40)), levels=5)
    ax.plot([0, 10, 20], [0, 5, 3], lw=1.8, path_effects=[pe.withStroke(linewidth=2.5,
                                                                         foreground='w')])
    a = np.asarray(fig.canvas.buffer_rgba())
    again = np.asarray(fig.canvas.buffer_rgba())        # (still held: the same picture)
    same = again.__array_interface__['data'][0] == a.__array_interface__['data'][0]
    refs.append(weakref.ref(fig))
    picture = fig._handed                               # (the figure's weak hold on it)
    del a, again, fig, ax
print(json.dumps(dict(same=same, figures=sum(r() is not None for r in refs),
                      picture=picture() is not None)))
'''
    got = run(code, cache=images)
    assert got['same']
    assert got['figures'] > 0 and not got['picture']   # (figures wait for gc; pictures do not)


def test_prewarm_goes_by_this_programs_own_use(images):
    """Once a program has run here, prewarm maps what its own processes
    used, not what the machine's others did."""
    code = '''
import json
from fastplot import _image
im = _image.open_image()
keys = [(name, key) for name, units in im.kernels.items() for key in units
        if not _image._needs_layer(im, (name, key))]
mine, theirs = keys[:3], keys[3:9]
_image._usage = lambda im: dict(processes=10, units=dict.fromkeys(mine + theirs, 10),
                                programs={_image._program(): dict(processes=2,
                                                                  units=dict.fromkeys(mine, 2))})


def mapped(k):
    u = im.units.get(id(im.kernels[k[0]][k[1]]))
    return u is not None and u.fn is not None


_image.prewarm(im)
print(json.dumps(dict(mine=all(map(mapped, mine)), theirs=any(map(mapped, theirs)))))
'''
    got = run(code, cache=images, FASTPLOT_PRELOAD='0')
    assert got['mine'] and not got['theirs']


def test_what_an_image_does_not_hold_goes_to_numba(images, by_numba):
    code = DRAW.replace('values = []', '''
from fastplot import _image
_jit._look()
_image.open_image().kernels.pop('fastplot._render._fill32')
values = []''')
    got = run(code, cache=images)
    assert got['image'] and got['numba']
    assert got['values'] == by_numba


def test_another_threading_layer_asked_for_goes_to_numba(images, by_numba):
    got = run(DRAW, cache=images, NUMBA_THREADING_LAYER='workqueue')
    assert got['image'] and got['numba']
    assert got['values'] == by_numba


def _damage(d, how):
    pointer = d / _image._name()
    image = d / pointer.read_text().strip()
    b = bytearray(image.read_bytes())
    _, _, _, size, at, n = _image.HEAD.unpack_from(b)
    if how == 'unit':                           # every unit of a kernel every figure needs
        for r in marshal.loads(bytes(b[at:at + n]))['kernels']['fastplot._render._fill32'].values():
            b[r['at'] + r['stored'] // 2] ^= 0xFF
    elif how == 'index':
        b[at + n // 2] ^= 0xFF
    elif how == 'cut':
        b = b[:len(b) // 2]
    elif how == 'head':
        b[:8] = b'NOTANIMG'
    if how == 'gone':
        image.unlink()
    elif how == 'pointer':
        pointer.write_bytes(os.urandom(40))
    else:
        image.write_bytes(bytes(b))


@pytest.mark.parametrize('how', ['unit', 'index', 'cut', 'head', 'gone', 'pointer'])
def test_a_damaged_image_goes_to_numba(images, by_numba, tmp_path, how):
    d = copy(images, tmp_path)
    _damage(d, how)
    got = run(DRAW, cache=d)
    assert got['numba'] and got['image'] == (how == 'unit')     # (only a unit's own CRC is seen late)
    assert got['values'] == by_numba


def test_another_numba_goes_to_numba(images, by_numba, tmp_path):
    """An image is built against numba's compiled helpers as they are: other
    files (an upgrade of numba) and the units that need them are not used."""
    d = copy(images, tmp_path)
    pointer = d / _image._name()
    image = d / pointer.read_text().strip()
    b = bytearray(image.read_bytes())
    _, _, _, size, at, n = _image.HEAD.unpack_from(b)
    index = marshal.loads(bytes(b[at:at + n]))
    index['numba']['helpers'] = (1, 2)
    index['numba']['pool']['file'] = (1, 2)
    tail = marshal.dumps(index)
    b = b[:at] + tail
    struct.pack_into('<8sIIQQQ', b, 0, _image.MAGIC, _image.FORMAT, __import__('zlib').crc32(tail), len(b), at,
                     len(tail))
    image.write_bytes(bytes(b))
    got = run(DRAW, cache=d)
    assert got['image'] and got['numba']
    assert got['values'] == by_numba


def test_no_image_goes_to_numba(by_numba, tmp_path):
    got = run(DRAW, cache=tmp_path)
    assert not got['image'] and got['numba']
    assert got['values'] == by_numba


def test_turned_off_never_looks(images):
    code = '''
import io, json, sys
import fastplot
fig = fastplot.figure(figsize=(2, 2), dpi=50)
fig.add_subplot().plot([0, 1], [1, 0])
fig.savefig(io.BytesIO(), format='png')
print(json.dumps(dict(looked='fastplot._image' in sys.modules, numba='numba' in sys.modules)))
'''
    assert run(code, cache=images, FASTPLOT_IMAGE='0') == dict(looked=False, numba=True)


def test_an_image_answers_for_its_machine_and_sources_only():
    name = _image._name()
    assert name != _image._name('00000000')
    assert _image._warm._sources()[:16] in name
    assert ('py%d.%d' % sys.version_info[:2]) in name and ('np' + np.__version__.split('.')[0]) in name


def test_a_build_that_fails_its_trial_is_never_seen(tmp_path):
    code = '''
import json, os
from fastplot import _image
_image._tried_in_process = lambda *a: False
_image.build_once()
p = _image.path()
print(json.dumps(dict(current=_image.current(), failed=os.path.exists(p + '.failed'), files=os.listdir(os.path.dirname(p)))))
'''
    got = run(code, cache=tmp_path, FASTPLOT_PRELOAD='0')
    assert got['current'] is None and got['failed']
    assert all(n.endswith('.failed') for n in got['files'])


def test_publish_tidies_only_its_own(tmp_path):
    name = _image._name()
    head = name.rpartition('-')[0]
    for n in ('notes.py', head + '-0123456789abcdef.img', head + '-0123456789abcdef.img.1.2', 'kernels-other.img',
              name + '.lock'):
        (tmp_path / n).write_text('x')
    new = tmp_path / (name + '.3.4')
    new.write_text('x')
    _image._publish(str(new), str(tmp_path / name))
    assert sorted(os.listdir(tmp_path)) == sorted(['notes.py', 'kernels-other.img', name, name + '.3.4',
                                                   name + '.lock'])
    assert (tmp_path / name).read_text() == name + '.3.4'


def test_only_what_numba_would_take_goes_in():
    slots, key = [], []
    a = np.zeros(3)
    assert _image._flatten((a, 7, 2.5, np.float32(1), None, True, (1.5, 2.5), (np.zeros(2), 3)), slots, key)
    assert key[1:] == ['i8', 'f8', 'f4', 'n', '?', ('d', 2), ('H', (('f8', 1, 'C', False), 'i8'))]
    assert struct.unpack('<d', struct.pack('<Q', slots[8]))[0] == 2.5
    assert slots[1] == id(a)                            # (an array's parent word: the array itself)
    odd = np.zeros(17, np.uint8)[1:].view(np.float64)
    assert not odd.flags.aligned
    for a in (np.zeros(3, '>f4'), np.ma.masked_array(np.zeros(3)), odd):
        assert not _image._flatten((a,), [], [])


OWN_POOL = pytest.mark.skipif(not __import__('fastplot._pool', fromlist=['x']).possible('omp'),
                              reason="fastplot's own pool stands in for OpenMP on x86-64 Windows and Linux")

REDUCE = '''
import json, sys, threading
import numpy as np
import fastplot
from fastplot import _jit
_jit.load()                                     # (numba as fastplot brings it up: on its pool)
import numba
from numba import njit, prange


@njit(parallel=True)
def total(a):
    s = 0.0
    for i in prange(a.shape[0]):
        s += a[i] * 1.0000001
    return s


x = np.random.default_rng(0).random(1_000_003)
got = {}
most = numba.config.NUMBA_NUM_THREADS               # (a CI runner may have 4)
counts = sorted({min(n, most) for n in (8, 5, 3, 2, 1)}, reverse=True)
for n in counts:
    numba.set_num_threads(n)
    got[n] = [total(x).hex() for _ in range(20)]
bad = []


def hammer(k):
    n = counts[k % len(counts)]
    numba.set_num_threads(n)
    want = got[n][0]
    for _ in range(400):
        if total(x).hex() != want:
            bad.append(k)


ts = [threading.Thread(target=hammer, args=(k,)) for k in range(6)]
for t in ts:
    t.start()
for t in ts:
    t.join()
print(json.dumps(dict(got=got, bad=bad, layer=_jit.layer())))
'''


@OWN_POOL
def test_own_pool_reduces_to_openmps_bits():
    """A float reduction over prange, on any thread count, gives OpenMP's
    bits on fastplot's pool, run after run and from six threads at once."""
    pool = run(REDUCE, FASTPLOT_IMAGE='0', FASTPLOT_POOL='1')
    omp = run(REDUCE, FASTPLOT_IMAGE='0', FASTPLOT_POOL='0')
    assert pool['layer'] == 'fastplot' and omp['layer'] == 'omp'
    assert pool['got'] == omp['got'] and not pool['bad']
    assert all(len(set(v)) == 1 for v in pool['got'].values())


@OWN_POOL
def test_own_pool_in_the_image_draws_as_openmp(images, by_numba):
    code = DRAW.replace('print(json.dumps(dict(', 'print(json.dumps(dict(layer=_jit.layer(), ')
    on = run(code, cache=images, FASTPLOT_POOL='1')
    off = run(code, cache=images, FASTPLOT_POOL='0')
    assert on['image'] and off['image'] and not on['numba']
    assert on['layer'] == 'fastplot' and off['layer'] == 'omp'
    assert on['values'] == off['values'] == by_numba


@OWN_POOL
def test_own_pool_frames_of_every_split_alike(images):
    code = '''
import hashlib, json, sys
import numpy as np
import fastplot
from fastplot import _jit


def frame(i):
    fig = fastplot.figure(figsize=(3, 2), dpi=60)
    ax = fig.add_subplot()
    ax.contourf(np.random.default_rng(i % 3).random((30, 40)), levels=6)
    ax.plot(np.arange(40), np.sin(np.arange(40) / (1 + i % 3)))
    return hashlib.sha256(fig.draw().tobytes()).hexdigest()


one = [frame(i) for i in range(3)]
runs = [fastplot.frames(frame, range(24), workers=w, threads=t) for w, t in ((4, 2), (8, 1), (2, 4), (3, 3))]
print(json.dumps(dict(same=all(r == one * 8 for r in runs), layer=_jit.layer(), image=_jit.image() is not None,
                      numba='numba' in sys.modules)))
'''
    got = run(code, cache=images, FASTPLOT_POOL='1')
    assert got['image'] and not got['numba'] and got['layer'] == 'fastplot' and got['same']


def test_objects_are_laid_out_as_the_units_read_them():
    assert _image._py_works(_image._tuple_items())
    assert not _image._py_works(None)


def test_build_later_once_a_process_and_never_with_autowarm_off(monkeypatch):
    started = []
    monkeypatch.delenv('FASTPLOT_IMAGE', raising=False)
    monkeypatch.setattr(_image, '_spawned', [False])
    monkeypatch.setattr(_image, '_spawn', lambda: started.append(1))
    monkeypatch.setattr(_image, '_can_spawn', lambda: True)
    monkeypatch.setenv('FASTPLOT_AUTOWARM', '0')
    _image.build_later()
    assert not started and not _image._spawned[0]
    monkeypatch.setenv('FASTPLOT_AUTOWARM', '1')
    _image.build_later()
    _image.build_later()
    for _ in range(300):                        # (on a thread of its own)
        if started:
            break
        time.sleep(0.01)
    time.sleep(0.05)
    assert started == [1]
