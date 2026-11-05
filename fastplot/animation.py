"""matplotlib.animation for ported scripts: FuncAnimation, ArtistAnimation
and the movie writers, saved through fastplot's encoder (_movie: frames
piped to ffmpeg as raw RGB, or a GIF by Pillow).

    import fastplot.animation as animation      # for matplotlib.animation

    def update(h):
        mesh.set_array(fields[h])
        title.set_text('hour %d' % h)

    anim = animation.FuncAnimation(fig, update, frames=len(fields), interval=125)
    anim.save('t2m.mp4', fps=8, dpi=100)

    writer = animation.FFMpegWriter(fps=8)      # or by hand, as matplotlib's
    with writer.saving(fig, 't2m.mp4', dpi=100):
        for h in hours:
            update(h)
            writer.grab_frame()

The frames are drawn one after another: each changes the same figure. For
frames that can be drawn apart (a figure each), fastplot.save_movie draws
several at once. There is no window to play in: save is what is drawn.
"""
import itertools

from ._movie import Encoder, rgb_of, find_ffmpeg

__all__ = ['FuncAnimation', 'ArtistAnimation', 'Animation', 'TimedAnimation', 'MovieWriter',
           'FFMpegWriter', 'PillowWriter', 'ImageMagickWriter', 'writers']


class MovieWriter:
    """matplotlib's MovieWriter: fps, codec, bitrate (kbit/s), extra_args
    (ffmpeg's output options) and metadata; saving(fig, outfile, dpi) then
    grab_frame() a frame, or given to Animation.save."""

    def __init__(self, fps=5, codec=None, bitrate=None, extra_args=None, metadata=None):
        self.fps, self.codec, self.bitrate = fps, codec, bitrate
        self.extra_args, self.metadata = extra_args, metadata
        self._enc = self.fig = None
        self.dpi, self._savefig = None, {}

    @classmethod
    def isAvailable(cls):
        try:
            find_ffmpeg()
            return True
        except RuntimeError:
            return False

    def setup(self, fig, outfile, dpi=None):
        self.fig, self.outfile = fig, outfile
        self.dpi = dpi if dpi not in (None, 'figure') else None
        self._enc = Encoder(outfile, self.fps, self.codec, None, self.bitrate,
                            self.extra_args or (), None, self.metadata)

    def grab_frame(self, **savefig_kwargs):
        kw = dict(self._savefig, **savefig_kwargs)
        self._enc.write(rgb_of(self.fig, kw.get('dpi', self.dpi), kw.get('facecolor')))

    def finish(self):
        if self._enc is not None:
            enc, self._enc = self._enc, None
            enc.close()

    def saving(self, fig, outfile, dpi=None, *args, **kwargs):
        """The writer set up for fig and outfile; finished at the end, or the
        unfinished file removed if the block raised."""
        return _Saving(self, fig, outfile, dpi)


class _Saving:
    def __init__(self, writer, fig, outfile, dpi):
        self.w = writer
        writer.setup(fig, outfile, dpi)

    def __enter__(self):
        return self.w

    def __exit__(self, kind, value, tb):
        if kind is None:
            self.w.finish()
        else:
            enc, self.w._enc = self.w._enc, None
            if enc is not None:
                enc.abort()
        return False


class FFMpegWriter(MovieWriter):
    """ffmpeg (the extension picks the format: H.264 for .mp4)."""


FFMpegFileWriter = FFMpegWriter


class PillowWriter(MovieWriter):
    """An animated GIF by Pillow."""


ImageMagickWriter = ImageMagickFileWriter = PillowWriter    # (a GIF all the same)


class _Writers(dict):
    """matplotlib's animation.writers: by name, and is_available."""

    def is_available(self, name):
        cls = self.get(name)
        return cls is not None and (cls is PillowWriter or cls.isAvailable())

    def list(self):
        return [k for k in self if self.is_available(k)]

    def __getitem__(self, name):
        try:
            return dict.__getitem__(self, name)
        except KeyError:
            raise RuntimeError('Requested MovieWriter (%s) not available' % name) from None


writers = _Writers(ffmpeg=FFMpegWriter, ffmpeg_file=FFMpegWriter, pillow=PillowWriter,
                   imagemagick=PillowWriter, imagemagick_file=PillowWriter)


class Animation:
    """What FuncAnimation and ArtistAnimation share: save."""

    def __init__(self, fig, interval=200, repeat_delay=0, repeat=True, blit=False,
                 event_source=None):
        self._fig = fig
        self._interval = interval
        self.repeat, self.repeat_delay, self._blit = repeat, repeat_delay, blit

    def _frames(self):
        """(how many, an iterator of steps), each step drawing a frame."""
        raise NotImplementedError

    def save(self, filename, writer=None, fps=None, dpi=None, codec=None, bitrate=None,
             extra_args=None, metadata=None, extra_anim=None, savefig_kwargs=None, *,
             progress_callback=None):
        """Every frame into filename, as matplotlib's save: writer a
        MovieWriter or its name (the extension picks the format either way),
        fps 1000 / interval unless given."""
        if extra_anim:
            raise ValueError('fastplot.animation: extra_anim is not supported')
        if writer is None or isinstance(writer, str):
            cls = PillowWriter if (writer or '').startswith(('pillow', 'imagemagick')) \
                else FFMpegWriter
            if fps is None:
                fps = 1000.0 / self._interval
            writer = cls(fps=fps, codec=codec, bitrate=bitrate, extra_args=extra_args,
                         metadata=metadata)
        elif fps is not None:
            writer.fps = fps
        writer._savefig = dict(savefig_kwargs or {})
        total, steps = self._frames()
        with writer.saving(self._fig, filename, dpi):
            for k, step in enumerate(steps):
                step()
                writer.grab_frame()
                if progress_callback is not None:
                    progress_callback(k, total)

    def new_frame_seq(self):
        return iter(self._frames()[1])

    def pause(self):
        pass

    def resume(self):
        pass


TimedAnimation = Animation


class FuncAnimation(Animation):
    """matplotlib's FuncAnimation: func(frame, *fargs) before each frame is
    drawn; frames a sequence (all of it saved), a number (range), or an
    iterable, generator function or None (read as it goes, save_count
    frames: 100 unless given); init_func first."""

    def __init__(self, fig, func, frames=None, init_func=None, fargs=None, save_count=None, *,
                 cache_frame_data=True, **kwargs):
        super().__init__(fig, **kwargs)
        self._func, self._init_func = func, init_func
        self._args = tuple(fargs or ())
        unsized = 100 if save_count is None else int(save_count)
        if frames is None:
            self._seq, self._count = itertools.count, unsized
        elif callable(frames):
            self._seq, self._count = frames, unsized
        elif hasattr(frames, '__len__'):
            self._seq, self._count = (lambda: iter(frames)), len(frames)
        elif hasattr(frames, '__iter__'):               # (read lazily, once)
            self._seq, self._count = (lambda: iter(frames)), unsized
        else:
            n = int(frames)
            self._seq, self._count = (lambda: iter(range(n))), n

    def _frames(self):
        seq = itertools.islice(self._seq(), int(self._count))
        func, args = self._func, self._args

        def steps():
            if self._init_func is not None:
                self._init_func()
            for f in seq:
                yield lambda f=f: func(f, *args)
        return self._count, steps()


class ArtistAnimation(Animation):
    """matplotlib's ArtistAnimation: a list of frames, each a list of
    artists, those of one frame shown while it is drawn."""

    def __init__(self, fig, artists, *args, **kwargs):
        super().__init__(fig, *args, **kwargs)
        self._artists = [list(a) if isinstance(a, (list, tuple)) else [a] for a in artists]

    def _frames(self):
        every = [a for fr in self._artists for a in fr]

        def show(fr):
            for a in every:
                a.set_visible(False)
            for a in fr:
                a.set_visible(True)

        return len(self._artists), (lambda fr=fr: show(fr) for fr in self._artists)
