"""Movies: frames drawn and piped straight to ffmpeg, no PNGs between.

    def hour(h):
        fig = fastplot.figure(figsize=(12, 6.5), dpi=100)
        ...
        return fig                                  # (or an (H, W, 3 or 4) uint8 image)

    fastplot.save_movie(hour, range(0, 385, 3), 't2m.mp4', fps=8)

save_movie draws as fastplot.frames does (several frames at once on
threads) and writes them in order as they come: each frame is drawn,
handed to ffmpeg's stdin as raw RGB and dropped, a few frames at most held
waiting for an earlier one. ffmpeg encodes in its own process meanwhile.
The extension picks the container; .mp4, .mov, .mkv and .m4v are H.264
(libx264, preset veryfast, yuv420p, crf 18, the size padded to even pixels as yuv420p
needs it), .webm VP9, .gif Pillow's (no ffmpeg). ffmpeg is the one given,
FASTPLOT_FFMPEG, the one on PATH, or imageio-ffmpeg's.

fastplot.animation.FuncAnimation writes through the same Encoder, one
frame after another (its function changes one figure)."""
import os
import threading

import numpy as np

AHEAD = 16                      # frames drawn ahead of the one ffmpeg waits for, at most
PRESET = 'veryfast'             # x264's: 40 frames of a global map encoded in 0.32 s, not 0.46
                                # ('medium', ffmpeg's own), the file 5% larger at the same crf
_H264 = ('.mp4', '.mov', '.mkv', '.m4v')


def find_ffmpeg(path=None):
    """The ffmpeg to run: path, FASTPLOT_FFMPEG, PATH's, imageio-ffmpeg's."""
    import shutil
    for p in (path, os.environ.get('FASTPLOT_FFMPEG')):
        if p:
            return p
    p = shutil.which('ffmpeg')
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                                   # noqa: BLE001 -- not installed / no binary
        pass
    raise RuntimeError('no ffmpeg found: install it (conda install ffmpeg, or pip install '
                       'imageio-ffmpeg), put it on PATH or set FASTPLOT_FFMPEG; a .gif needs none')


def rgb_of(frame, dpi=None, facecolor=None):
    """A frame -- a figure (drawn here) or an image -- as (H, W, 3) uint8."""
    if hasattr(frame, 'draw') and hasattr(frame, 'dpi'):
        img = frame.draw(dpi=dpi, facecolor=facecolor, _transient=True)
    else:
        img = np.asarray(frame)
        if img.dtype != np.uint8:                       # (floats 0..1, as imshow takes them)
            img = np.clip(np.rint(np.asarray(img, np.float64) * 255.0), 0, 255).astype(np.uint8)
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError('a movie frame is a figure or an (H, W, 3 or 4) image, not shape %s'
                         % (img.shape,))
    return np.ascontiguousarray(img[:, :, :3])


class Encoder:
    """Frames (H, W, 3) uint8, in order, into a movie file."""

    def __init__(self, path, fps=10, codec=None, crf=None, bitrate=None, extra_args=(),
                 ffmpeg=None, metadata=None, preset=None):
        self.preset = preset
        self.path = os.fspath(path)
        self.fps = float(fps)
        self.ext = os.path.splitext(self.path)[1].lower()
        self.codec, self.crf, self.bitrate = codec, crf, bitrate
        self.extra_args = list(extra_args or ())
        self.metadata = dict(metadata or {})
        self.ffmpeg = None if self.ext == '.gif' else find_ffmpeg(ffmpeg)
        self.size = None
        self.count = 0
        self._proc = self._gif = None

    def _args(self, W, H):
        a = [self.ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
             '-s', '%dx%d' % (W, H), '-framerate', repr(self.fps), '-i', '-']
        codec = self.codec or ('libvpx-vp9' if self.ext == '.webm' else
                               'libx264' if self.ext in _H264 or not self.ext else None)
        if codec in ('h264', 'libx264', 'libx265', 'hevc') or (codec is None and W % 2 + H % 2):
            a += ['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2']   # (yuv420p: even sizes)
        if codec is not None:
            a += ['-c:v', codec]
        if codec in ('h264', 'libx264', 'libx265', 'hevc') and '-preset' not in self.extra_args:
            a += ['-preset', PRESET if self.preset is None else str(self.preset)]
        if codec in ('h264', 'libx264', 'libx265', 'hevc', 'libvpx-vp9') and \
                '-pix_fmt' not in self.extra_args:
            a += ['-pix_fmt', 'yuv420p']
        if self.bitrate is not None and self.bitrate > 0:      # (matplotlib's -1: the encoder's)
            a += ['-b:v', '%dk' % int(self.bitrate)]
        elif codec in ('h264', 'libx264', 'libx265', 'hevc'):
            a += ['-crf', str(18 if self.crf is None else int(self.crf))]
        elif codec == 'libvpx-vp9':
            a += ['-crf', str(30 if self.crf is None else int(self.crf)), '-b:v', '0']
        if self.ext in ('.mp4', '.mov', '.m4v'):
            a += ['-movflags', '+faststart']
        for k, v in self.metadata.items():
            a += ['-metadata', '%s=%s' % (k, v)]
        return a + self.extra_args + [self.path]

    def write(self, rgb):
        H, W = rgb.shape[:2]
        if self.size is None:
            self.size = (W, H)
            if self.ext == '.gif':
                self._gif = _GifWriter(self.path, self.fps, self.metadata)
            else:
                import subprocess
                self._proc = subprocess.Popen(self._args(W, H), stdin=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
                self._err = []                          # (read as it comes: a full pipe would
                self._err_thread = threading.Thread(    # stop ffmpeg, and our writes with it)
                    target=lambda p=self._proc, out=self._err: out.append(p.stderr.read()),
                    name='fastplot-ffmpeg-stderr', daemon=True)
                self._err_thread.start()
        elif self.size != (W, H):
            raise ValueError('movie frames are all one size: frame %d is %dx%d, the first %dx%d'
                             % ((self.count, W, H) + self.size))
        if self._gif is not None:
            self._gif.add(rgb)
        else:
            try:
                self._proc.stdin.write(memoryview(rgb).cast('B'))
            except (BrokenPipeError, OSError):
                self._fail()
        self.count += 1

    def _stderr(self):
        self._err_thread.join(5.0)
        return b''.join(self._err).decode(errors='replace').strip()[-2000:]

    def _fail(self):
        p = self._proc
        if p is None:                                   # (stopped by abort meanwhile)
            raise RuntimeError('ffmpeg stopped writing %s' % self.path)
        import subprocess
        try:
            p.wait(10.0)                                # (its error said, it ends)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
        raise RuntimeError('ffmpeg failed writing %s:\n%s' % (self.path, self._stderr()))

    def close(self):
        """Finish the file (nothing was written: no file)."""
        if self._gif is not None:
            self._gif.close()
            self._gif = None
        p = self._proc
        if p is None:
            return
        try:
            p.stdin.close()
        except OSError:
            pass
        code = p.wait()
        self._proc = None
        if code != 0:
            raise RuntimeError('ffmpeg failed writing %s:\n%s' % (self.path, self._stderr()))
        self._stderr()

    def abort(self):
        """Stop: ffmpeg's unfinished file removed (a GIF is only written
        by close, so an earlier file of its name is left as it was)."""
        self._gif = None
        p, self._proc = self._proc, None
        if p is not None:
            p.kill()
            p.wait()
            try:
                os.remove(self.path)
            except OSError:
                pass


class _GifWriter:
    """Pillow's animated GIF, its frames given one by one."""

    def __init__(self, path, fps, metadata):
        self.path, self.ms = path, int(round(1000.0 / fps))
        self.frames = []
        self.loop = int(metadata.get('loop', 0))

    def add(self, rgb):
        from PIL import Image
        self.frames.append(Image.fromarray(rgb).quantize(256, method=Image.Quantize.MEDIANCUT))

    def close(self):
        if self.frames:
            first, rest = self.frames[0], self.frames[1:]
            first.save(self.path, save_all=True, append_images=rest, duration=self.ms,
                       loop=self.loop, optimize=False)
        self.frames = []


class _Drawn:
    """fn as save_movie draws it: the frame as RGB, waiting while it is too
    far ahead of the one being written. frames knows it as fn (its code
    and name), so the split frames learnt for fn is a movie's of it too,
    and the other way round."""

    def __init__(self, fn, dpi, gate):
        self.fn, self.dpi, self.gate = fn, dpi, gate
        code = getattr(fn, '__code__', None)
        if code is not None:
            self.__code__ = code
            self.__qualname__ = getattr(fn, '__qualname__', code.co_name)

    def __call__(self, ki):
        k, item = ki
        self.gate.wait_turn(k)
        try:
            rgb = rgb_of(self.fn(item), self.dpi)
        except BaseException:
            self.gate.stop()                            # (frames waiting their turn give up: the
            raise                                       # batch raises this frame's error)
        self.gate.put(k, rgb)


class _Stopped(Exception):
    """A frame not drawn: an earlier one failed (frames raises that one)."""


class _Gate:
    """Frames in order to an Encoder on a thread of its own."""

    def __init__(self, enc, n):
        self.enc, self.n = enc, n
        self.cv = threading.Condition()
        self.done = {}
        self.next = 0
        self.error = None
        self.stopped = False
        self.thread = threading.Thread(target=self._run, name='fastplot-movie', daemon=True)
        self.thread.start()

    def wait_turn(self, k):
        with self.cv:
            while k >= self.next + AHEAD and self.error is None and not self.stopped:
                self.cv.wait()
            if self.error is not None:
                raise self.error
            if self.stopped:
                raise _Stopped(k)

    def stop(self):
        with self.cv:
            self.stopped = True
            self.cv.notify_all()

    def put(self, k, rgb):
        with self.cv:
            self.done[k] = rgb
            self.cv.notify_all()

    def _run(self):
        try:
            while True:
                with self.cv:
                    while self.next not in self.done and not self.stopped and self.next < self.n:
                        self.cv.wait()
                    if self.next >= self.n or self.next not in self.done:
                        return
                    rgb = self.done.pop(self.next)
                self.enc.write(rgb)
                with self.cv:
                    self.next += 1
                    self.cv.notify_all()
        except BaseException as e:                      # noqa: BLE001 -- raised by save_movie
            with self.cv:
                self.error = e
                self.cv.notify_all()

    def finish(self, failed):
        with self.cv:
            self.stopped = failed
            self.cv.notify_all()
        self.thread.join()
        if self.error is not None:
            raise self.error


def save_movie(fn, items, path, fps=10, dpi=None, codec=None, crf=None, bitrate=None,
               extra_args=(), ffmpeg=None, metadata=None, workers=None, threads=None,
               preset=None):
    """A movie of fn(item) for every item, in order: fn returns a figure
    (fastplot.figure / subplots, not pyplot's, as for frames) or an
    (H, W, 3 or 4) image, every frame the same size. Drawn as
    fastplot.frames draws (workers, threads), piped to ffmpeg as they come
    (see the module). path's extension picks the format; codec, crf
    (H.264: 18) or bitrate (kbit/s), preset (x264's, PRESET unless given:
    'medium' is ffmpeg's own, slower, files a little smaller), extra_args (ffmpeg's
    output options) and metadata ({'title': ...}) as ffmpeg takes them.
    Returns path. If a frame or ffmpeg fails, the unfinished file is
    removed and the error raised here."""
    from ._batch import frames
    items = list(items)
    enc = Encoder(path, fps, codec, crf, bitrate, extra_args, ffmpeg, metadata, preset)
    gate = _Gate(enc, len(items))
    try:
        frames(_Drawn(fn, dpi, gate), list(enumerate(items)), workers=workers, threads=threads)
        gate.finish(False)
        enc.close()
    except BaseException:
        try:
            gate.finish(True)
        except BaseException:                           # noqa: BLE001 -- the first error is raised
            pass
        enc.abort()
        raise
    return enc.path
