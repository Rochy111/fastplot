"""fastplot.save_movie and fastplot.animation: frames piped to ffmpeg (or a
GIF by Pillow), in order, every frame there."""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

import fastplot as fp
import fastplot.animation as animation
from fastplot import _movie

try:
    FFMPEG = _movie.find_ffmpeg()
except RuntimeError:
    FFMPEG = None
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason='no ffmpeg')


def _probe(path):
    """(frames, width, height) of a movie, by ffprobe beside ffmpeg."""
    probe = shutil.which('ffprobe') or os.path.join(os.path.dirname(FFMPEG), 'ffprobe')
    if not (os.path.exists(probe) or shutil.which(probe)):
        pytest.skip('no ffprobe')
    out = subprocess.run([probe, '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                          '-show_entries', 'stream=nb_read_frames,width,height', '-of', 'json',
                          path], capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)['streams'][0]
    return int(s['nb_read_frames']), int(s['width']), int(s['height'])


def _frame(k):
    fig = fp.figure(figsize=(3.01, 2.01), dpi=50)         # 150 x 100 px... (odd: 151 x 101)
    ax = fig.add_subplot()
    x = np.linspace(0, 6, 50)
    ax.plot(x, np.sin(x + 0.3 * k))
    ax.set_title('frame %d' % k)
    return fig


@needs_ffmpeg
@pytest.mark.parametrize('ext', ['.mp4', '.webm'])
def test_save_movie_writes_every_frame(tmp_path, ext):
    path = str(tmp_path / ('m' + ext))
    got = fp.save_movie(_frame, range(12), path, fps=6)
    assert got == path
    n, w, h = _probe(path)
    img = _frame(0).draw()
    assert n == 12
    assert (w, h) == (img.shape[1] + img.shape[1] % 2, img.shape[0] + img.shape[0] % 2)


@needs_ffmpeg
def test_save_movie_frames_in_order(tmp_path):
    """Frames of one flat colour each, drawn on several threads: decoded,
    each is its own colour, in the items' order."""
    path = str(tmp_path / 'order.mp4')
    shades = np.linspace(20, 235, 24).astype(np.uint8)

    def flat(k):
        img = np.empty((64, 96, 3), np.uint8)
        img[:] = shades[k]
        return img

    fp.save_movie(flat, range(24), path, fps=10, workers=4, threads=1, crf=0)
    out = subprocess.run([FFMPEG, '-v', 'error', '-i', path, '-f', 'rawvideo', '-pix_fmt',
                          'gray', '-'], capture_output=True, check=True).stdout
    got = np.frombuffer(out, np.uint8).reshape(-1, 64, 96)[:, 32, 48]
    assert got.size == 24
    assert np.all(np.abs(got.astype(int) - shades.astype(int)) <= 3)


def test_save_movie_gif(tmp_path):
    from PIL import Image
    path = str(tmp_path / 'm.gif')
    fp.save_movie(_frame, range(5), path, fps=5)
    im = Image.open(path)
    assert im.n_frames == 5
    assert im.info.get('duration') == 200


@needs_ffmpeg
def test_a_failing_frame_leaves_no_file(tmp_path):
    path = str(tmp_path / 'bad.mp4')

    def bad(k):
        if k == 7:
            raise ValueError('frame 7')
        return _frame(k)

    with pytest.raises(ValueError, match='frame 7'):
        fp.save_movie(bad, range(20), path, workers=2, threads=1)
    assert not os.path.exists(path)


def test_a_slow_failing_frame_with_others_waiting_raises(tmp_path):
    """Frame 0 fails after the rest have run AHEAD of it and wait their
    turn: they give up, and save_movie raises frame 0's error (it hung)."""
    import threading
    import time
    path = str(tmp_path / 'slow.gif')

    def fn(k):
        if k == 0:
            time.sleep(0.5)
            raise ValueError('frame 0')
        return np.zeros((8, 8, 3), np.uint8)

    got = []

    def run():
        try:
            fp.save_movie(fn, range(60), path, workers=4, threads=1)
        except BaseException as e:                      # noqa: BLE001
            got.append(e)
    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(30)
    assert not th.is_alive()
    assert len(got) == 1 and isinstance(got[0], ValueError) and 'frame 0' in str(got[0])


def test_a_failed_gif_leaves_an_earlier_one(tmp_path):
    from PIL import Image
    path = str(tmp_path / 'keep.gif')
    fp.save_movie(lambda k: np.full((8, 8, 3), 40 * k, np.uint8), range(3), path, workers=1)

    def bad(k):
        if k == 2:
            raise ValueError('no')
        return np.zeros((8, 8, 3), np.uint8)
    with pytest.raises(ValueError):
        fp.save_movie(bad, range(4), path, workers=1)
    assert Image.open(path).n_frames == 3


def test_bitrate_minus_one_is_the_encoders():
    enc = _movie.Encoder.__new__(_movie.Encoder)
    enc.__dict__.update(ffmpeg='ffmpeg', fps=10.0, ext='.mp4', codec=None, crf=None, bitrate=-1,
                        extra_args=[], metadata={}, path='x.mp4', preset=None)
    args = enc._args(100, 80)
    assert '-b:v' not in args and '-crf' in args
    assert args[args.index('-preset') + 1] == _movie.PRESET
    enc.preset = 'medium'
    assert enc._args(100, 80)[enc._args(100, 80).index('-preset') + 1] == 'medium'
    enc.preset, enc.extra_args = None, ['-preset', 'slow']
    assert enc._args(100, 80).count('-preset') == 1
    enc.bitrate = 2000
    args = enc._args(100, 80)
    assert args[args.index('-b:v') + 1] == '2000k'


def test_func_animation_frames_as_matplotlib_reads_them():
    import itertools
    fig = fp.figure(figsize=(1, 1), dpi=10)

    def frames_of(**kw):
        seen = []
        a = animation.FuncAnimation(fig, seen.append, **kw)
        _, steps = a._frames()
        for s in steps:
            s()
        return seen
    assert frames_of(frames=itertools.count(), save_count=5) == [0, 1, 2, 3, 4]   # lazily
    assert frames_of(frames=np.int64(3)) == [0, 1, 2]
    assert frames_of(frames=[7, 8, 9], save_count=1) == [7, 8, 9]       # a sequence: all of it
    assert len(frames_of()) == 100
    assert frames_of(frames=lambda: iter('ab')) == ['a', 'b']


@needs_ffmpeg
def test_frames_of_another_size_raise(tmp_path):
    path = str(tmp_path / 'sizes.mp4')

    def grow(k):
        return np.zeros((40, 40 + 2 * k, 3), np.uint8)

    with pytest.raises(ValueError, match='one size'):
        fp.save_movie(grow, range(4), path, workers=1)
    assert not os.path.exists(path)


@needs_ffmpeg
def test_func_animation_saves(tmp_path):
    fig = _frame(0)
    ax = fig.axes[0]
    (line,) = ax.plot([0, 1], [0, 1])
    seen = []

    def update(k):
        seen.append(k)
        line.set_data([0, 1], [0, k / 10.0])

    anim = animation.FuncAnimation(fig, update, frames=9, interval=100)
    path = str(tmp_path / 'anim.mp4')
    calls = []
    anim.save(path, writer='ffmpeg', dpi=40,
              progress_callback=lambda i, n: calls.append((i, n)))
    assert seen == list(range(9))
    assert calls[-1] == (8, 9)
    n, w, h = _probe(path)
    assert n == 9 and w == 120 + 120 % 2


@needs_ffmpeg
def test_writer_saving_grab_frame(tmp_path):
    fig = _frame(0)
    w = animation.FFMpegWriter(fps=4)
    path = str(tmp_path / 'grab.mp4')
    with w.saving(fig, path, dpi=50):
        for k in range(6):
            fig.axes[0].set_title('k %d' % k)
            w.grab_frame()
    assert _probe(path)[0] == 6
    assert animation.writers.is_available('ffmpeg')


def test_artist_animation_shows_one_frame_at_a_time(tmp_path):
    from PIL import Image
    fig = fp.figure(figsize=(2, 2), dpi=40)
    ax = fig.add_subplot()
    frames = [ax.plot([0, 1], [k, k], color='k', lw=4) for k in range(3)]
    anim = animation.ArtistAnimation(fig, frames, interval=50)
    path = str(tmp_path / 'artists.gif')
    anim.save(path, writer='pillow')
    assert Image.open(path).n_frames == 3
    assert [a.get_visible() for fr in frames for a in fr] == [False, False, True]
