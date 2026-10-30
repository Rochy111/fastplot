"""matplotlib.pyplot for fastplot: the current figure and axes, and the
functions that act on them.

    import fastplot.pyplot as plt
    plt.plot(t, temp, label='T'); plt.xlabel('time'); plt.legend()
    plt.savefig('t.png'); plt.close()

Every Axes method has its function here, acting on gca(); the figure
functions act on gcf(). There is no window: show() does nothing (in a
notebook it displays the figures), savefig() writes them.
"""
from . import _figure
from . import cm, colors, style                                     # noqa: F401
from ._rc import rc, rc_context, rcdefaults, rcParams               # noqa: F401
from .cm import colormaps, get_cmap                                 # noqa: F401
from ._figure import Figure                                         # noqa: F401
from .gridspec import GridSpec                                      # noqa: F401
from .ticker import (AutoLocator, FixedFormatter, FixedLocator,     # noqa: F401
                     FormatStrFormatter, Formatter, FuncFormatter, IndexLocator,
                     LinearLocator, Locator, LogFormatter, LogFormatterExponent,
                     LogFormatterMathtext, LogLocator, MaxNLocator, MultipleLocator,
                     NullFormatter, NullLocator, ScalarFormatter)
from .colors import Normalize                                       # noqa: F401

_figs = {}                          # num -> Figure
_current = [None]                   # the current figure
_last_mappable = [None]             # gci(): the last image / mesh / contour set


def _new_num():
    return max([n for n in _figs if isinstance(n, int)], default=0) + 1


def figure(num=None, figsize=None, dpi=None, facecolor=None, clear=False, **kw):
    """A new figure, or (num given) the figure of that number, made current."""
    if isinstance(num, Figure):
        _current[0] = num
        return num
    if num is not None and num in _figs:
        fig = _figs[num]
        if clear:
            fig.clear()
    else:
        fig = _figure.figure(figsize=figsize, dpi=dpi, facecolor=facecolor, **kw)
        fig.number = num if num is not None else _new_num()
        _figs[fig.number] = fig
    _current[0] = fig
    return fig


def gcf():
    return _current[0] if _current[0] is not None else figure()


def gca(**kw):
    fig = gcf()
    if fig._current_axes is None or fig._current_axes not in fig.axes:
        fig._current_axes = fig.axes[-1] if fig.axes else fig.add_subplot(**kw)
    return fig._current_axes


def sca(ax):
    _current[0] = ax.figure
    ax.figure._current_axes = ax


def gci():
    """The current figure's current image (mappable): the last made with
    plt.*, else the last one in the current axes, then the others."""
    fig = gcf()
    m = _last_mappable[0]
    if m is not None and getattr(getattr(m, 'axes', None), 'figure', None) is fig:
        return m
    axs = list(fig.axes)
    cur = fig._current_axes
    if cur in axs:
        axs.remove(cur)
        axs.insert(0, cur)
    for ax in axs:
        for a in reversed(getattr(ax, 'artists', [])):
            if hasattr(a, 'norm_spec') and hasattr(a, 'cmap') and \
                    (type(a).__name__ != 'Scatter' or getattr(a, 'values', None) is not None):
                return a
    return None


_SUBPLOTS_KW = ('sharex', 'sharey', 'squeeze', 'subplot_kw', 'gridspec_kw', 'width_ratios',
                'height_ratios')


def subplots(nrows=1, ncols=1, **kw):
    fig_kw = {k: kw.pop(k) for k in list(kw) if k not in _SUBPLOTS_KW}
    fig = figure(**fig_kw)
    axs = fig.subplots(nrows, ncols, **kw)
    fig._current_axes = axs if not hasattr(axs, 'ravel') else axs.ravel()[-1]
    return fig, axs


def subplot(*args, **kw):
    """matplotlib's plt.subplot: the axes at that grid position -- the
    existing one if there is (with the same projection), else a new one."""
    fig = gcf()
    key = _subplot_key(args)
    if key is not None and not kw.get('sharex') and not kw.get('sharey'):
        proj = kw.get('projection', 'polar' if kw.get('polar') else None)
        for ax in fig.axes:
            if _spec_key(getattr(ax, 'spec', None)) == key and \
                    getattr(ax, '_twin_of', None) is None and \
                    (proj is None or getattr(ax, 'name', None) == proj or
                     getattr(ax, 'projection', None) is proj):
                fig._current_axes = ax
                return ax
    ax = fig.add_subplot(*args, **kw)
    fig._current_axes = ax
    return ax


def _spec_key(spec):
    """A subplot's place: (grid rows, columns, its rows and columns)."""
    if spec is None or not hasattr(spec, 'gs') or spec.gs.parent is not None:
        return None
    return (spec.gs.nrows, spec.gs.ncols, spec.r0, spec.r1, spec.c0, spec.c1)


def _subplot_key(args):
    """subplot()'s arguments as a _spec_key."""
    if not args:
        args = (1, 1, 1)
    if len(args) == 1 and isinstance(args[0], int):
        n = args[0]
        args = (n // 100, (n // 10) % 10, n % 10)
    if len(args) == 3 and all(isinstance(a, (int, tuple)) for a in args):
        nr, nc, i = args
        i0, i1 = (i, i) if isinstance(i, int) else i
        return (nr, nc, (i0 - 1) // nc, (i1 - 1) // nc, (i0 - 1) % nc, (i1 - 1) % nc)
    if len(args) == 1 and hasattr(args[0], 'gs'):
        return _spec_key(args[0])
    return None


def axes(arg=None, **kw):
    fig = gcf()
    ax = fig.add_subplot(**kw) if arg is None else fig.add_axes(arg, **kw)
    fig._current_axes = ax
    return ax


def subplot_mosaic(mosaic, **kw):
    """A new figure and its axes laid out by the mosaic: (fig, {label: ax})."""
    fig_kw = {k: kw.pop(k) for k in list(kw) if k not in _SUBPLOTS_KW + (
        'empty_sentinel', 'per_subplot_kw')}
    fig = figure(**fig_kw)
    axd = fig.subplot_mosaic(mosaic, **kw)
    fig._current_axes = next(iter(axd.values()), None)
    return fig, axd


def close(fig=None):
    """Forget a figure (the current one, a number, a Figure, or 'all')."""
    if fig == 'all':
        _figs.clear()
        _current[0] = None
        return
    target = _current[0] if fig is None else (_figs.get(fig) if not isinstance(fig, Figure)
                                               else fig)
    for k, v in list(_figs.items()):
        if v is target:
            del _figs[k]
    if _current[0] is target:
        _current[0] = next(reversed(_figs.values()), None) if _figs else None


def get_fignums():
    return sorted(n for n in _figs if isinstance(n, int))


def clf():
    gcf().clear()


def cla():
    gca().cla()


def fignum_exists(num):
    return num in _figs


def delaxes(ax=None):
    ax = gca() if ax is None else ax
    ax.figure.delaxes(ax)


def setp(obj, *args, **kw):
    """matplotlib's setp: set properties on an artist or a list of them."""
    objs = list(obj) if isinstance(obj, (list, tuple)) else [obj]
    if args:
        kw = dict(zip(args[0::2], args[1::2]), **kw)
    out = []
    for o in objs:
        if isinstance(o, (list, tuple)):
            out += setp(o, **kw)
            continue
        for k, v in kw.items():
            f = getattr(o, 'set_' + k, None)
            if f is None:
                from ._artists import _ALIASES
                f = getattr(o, 'set_' + _ALIASES.get(k, k), None)
            if f is None:
                raise AttributeError('%s has no property %r' % (type(o).__name__, k))
            out.append(f(v))
    return out


def getp(obj, prop):
    return getattr(obj, 'get_' + prop)()


def imread(fname, format=None):
    """matplotlib's imread: an image file as an array (PNG: float32 0..1)."""
    import numpy as np
    from PIL import Image
    im = Image.open(fname)
    a = np.asarray(im)
    if (format or str(getattr(fname, 'name', fname))).lower().endswith('png'):
        return (a / (65535.0 if a.dtype == np.uint16 else 255.0)).astype(np.float32)
    return a


def imsave(fname, arr, vmin=None, vmax=None, cmap=None, format=None, origin=None, dpi=100,
           **kw):
    """matplotlib's imsave: an array written as an image, pixel for pixel."""
    import numpy as np
    from PIL import Image
    a = np.asarray(arr)
    if origin == 'lower':
        a = a[::-1]
    if a.ndim == 2:
        from ._color import get_cmap, make_norm
        cm = get_cmap(cmap)
        norm = make_norm(None, vmin=vmin, vmax=vmax, data=a.astype(np.float32), N=cm.N)
        rgba = np.asarray(cm(norm(a), bytes=True), np.uint8)
    else:
        rgba = a if a.dtype == np.uint8 else (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(rgba).save(fname, format=format)


def savefig(*args, **kw):
    return gcf().savefig(*args, **kw)


def show(*args, **kw):
    """No window to open; in IPython/Jupyter the open figures are displayed."""
    try:
        from IPython.display import Image, display
    except ImportError:
        return
    import io
    for fig in list(_figs.values()):
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        display(Image(buf.getvalue()))


def draw():
    pass


def pause(interval):
    pass


def ion():
    pass


def ioff():
    pass


def isinteractive():
    return False


# -- the current axes' methods
def _ax_function(name, remember=False):
    def f(*args, **kw):
        out = getattr(gca(), name)(*args, **kw)
        if remember:
            _last_mappable[0] = out
        return out
    f.__name__ = name
    f.__doc__ = 'gca().%s(...)' % name
    return f


plot = _ax_function('plot')
scatter = _ax_function('scatter', remember=True)
bar = _ax_function('bar')
barh = _ax_function('barh')
bar_label = _ax_function('bar_label')
hist = _ax_function('hist')
fill_between = _ax_function('fill_between')
fill_betweenx = _ax_function('fill_betweenx')
fill = _ax_function('fill')
errorbar = _ax_function('errorbar')
step = _ax_function('step')
semilogx = _ax_function('semilogx')
semilogy = _ax_function('semilogy')
loglog = _ax_function('loglog')
stairs = _ax_function('stairs')
stem = _ax_function('stem')
stackplot = _ax_function('stackplot')
axhline = _ax_function('axhline')
axvline = _ax_function('axvline')
axhspan = _ax_function('axhspan')
axvspan = _ax_function('axvspan')
hlines = _ax_function('hlines')
vlines = _ax_function('vlines')
clabel = _ax_function('clabel')
barbs = _ax_function('barbs')
quiver = _ax_function('quiver', remember=True)
streamplot = _ax_function('streamplot')
quiverkey = _ax_function('quiverkey')
tricontourf = _ax_function('tricontourf', remember=True)
tricontour = _ax_function('tricontour', remember=True)
tripcolor = _ax_function('tripcolor', remember=True)
triplot = _ax_function('triplot')
text = _ax_function('text')
annotate = _ax_function('annotate')
legend = _ax_function('legend')
grid = _ax_function('grid')
tick_params = _ax_function('tick_params')
minorticks_on = _ax_function('minorticks_on')
minorticks_off = _ax_function('minorticks_off')
twinx = _ax_function('twinx')
twiny = _ax_function('twiny')
axis = _ax_function('axis')
margins = _ax_function('margins')
autoscale = _ax_function('autoscale')
locator_params = _ax_function('locator_params')
ticklabel_format = _ax_function('ticklabel_format')
imshow = _ax_function('imshow', remember=True)
add_patch = _ax_function('add_patch')
pcolormesh = _ax_function('pcolormesh', remember=True)
pcolor = _ax_function('pcolor', remember=True)
contour = _ax_function('contour', remember=True)
contourf = _ax_function('contourf', remember=True)
pie = _ax_function('pie')
boxplot = _ax_function('boxplot')
violinplot = _ax_function('violinplot')
axline = _ax_function('axline')
broken_barh = _ax_function('broken_barh')
eventplot = _ax_function('eventplot')
hexbin = _ax_function('hexbin', remember=True)


def hist2d(*args, **kw):
    out = gca().hist2d(*args, **kw)
    _last_mappable[0] = out[3]
    return out


def matshow(A, fignum=None, **kw):
    """matplotlib's plt.matshow: a new figure shaped to the matrix (or the
    current axes, fignum=0) showing it."""
    import numpy as np
    A = np.asanyarray(A)
    if fignum == 0:
        ax = gca()
    else:
        fig = figure(fignum, figsize=figaspect(A))
        ax = fig.add_axes((0.15, 0.09, 0.775, 0.775))
        fig._current_axes = ax
    im = ax.matshow(A, **kw)
    _last_mappable[0] = im
    return im


def figaspect(arg):
    """matplotlib's figaspect: a figure size (w, h) with this height/width
    (a number, or an array's shape), within rcParams figure.figsize."""
    import numpy as np
    isarray = hasattr(arg, 'shape') and not np.isscalar(arg)
    figsize_min, figsize_max = np.array((4.0, 2.0)), np.array((16.0, 16.0))
    arr_ratio = arg.shape[0] / arg.shape[1] if isarray else arg
    fig_height = rcParams['figure.figsize'][1]
    newsize = np.array((fig_height / arr_ratio, fig_height))
    newsize /= min(1.0, *(newsize / figsize_min))
    newsize /= max(1.0, *(newsize / figsize_max))
    return np.clip(newsize, figsize_min, figsize_max)


def sci(im):
    _last_mappable[0] = im


def clim(vmin=None, vmax=None):
    im = gci()
    if im is None:
        raise RuntimeError('You must first define an image, e.g., with imshow')
    im.set_clim(vmin, vmax)


def subplot2grid(shape, loc, rowspan=1, colspan=1, fig=None, **kw):
    fig = gcf() if fig is None else fig
    gs = GridSpec(shape[0], shape[1], figure=fig)
    ax = fig.add_subplot(gs[loc[0]:loc[0] + rowspan, loc[1]:loc[1] + colspan], **kw)
    fig._current_axes = ax
    return ax


def title(label, *args, **kw):
    return gca().set_title(label, *args, **kw)


def xlabel(label, *args, **kw):
    return gca().set_xlabel(label, *args, **kw)


def ylabel(label, *args, **kw):
    return gca().set_ylabel(label, *args, **kw)


def xscale(value, **kw):
    gca().set_xscale(value, **kw)


def yscale(value, **kw):
    gca().set_yscale(value, **kw)


def _lim(name, args, kw):
    ax = gca()
    if not args and not kw:
        return getattr(ax, 'get_%slim' % name)()
    return getattr(ax, 'set_%slim' % name)(*args, **kw)


def xlim(*args, **kw):
    return _lim('x', args, kw)


def ylim(*args, **kw):
    return _lim('y', args, kw)


def _ticks(name, ticks, labels, kw):
    ax = gca()
    if ticks is None:
        if labels is not None or kw:
            getattr(ax, name + 'axis').tick_label_kw.update(kw)
        return getattr(ax, 'get_%sticks' % name)(), getattr(ax, 'get_%sticklabels' % name)()
    getattr(ax, 'set_%sticks' % name)(ticks, labels, **kw)
    return getattr(ax, 'get_%sticks' % name)(), getattr(ax, 'get_%sticklabels' % name)()


def xticks(ticks=None, labels=None, **kw):
    return _ticks('x', ticks, labels, kw)


def yticks(ticks=None, labels=None, **kw):
    return _ticks('y', ticks, labels, kw)


# -- the current figure's
def suptitle(t, **kw):
    return gcf().suptitle(t, **kw)


def supxlabel(t, **kw):
    return gcf().supxlabel(t, **kw)


def supylabel(t, **kw):
    return gcf().supylabel(t, **kw)


def figtext(x, y, s, **kw):
    return gcf().text(x, y, s, **kw)


def colorbar(mappable=None, cax=None, ax=None, **kw):
    mappable = mappable if mappable is not None else gci()
    if mappable is None:
        raise RuntimeError('no mappable for the colorbar: draw an image, mesh or contour set '
                           'first, or pass one')
    return gcf().colorbar(mappable, cax=cax, ax=ax if ax is not None else (
        getattr(mappable, 'axes', None) or gca()), **kw)


def tight_layout(**kw):
    gcf().tight_layout(**kw)


def subplots_adjust(**kw):
    gcf().subplots_adjust(**kw)


def figlegend(*args, **kw):
    return gcf().legend(*args, **kw)


def get_current_fig_manager():
    return None


def set_cmap(cmap):
    rcParams['image.cmap'] = cmap if isinstance(cmap, str) else cmap.name


__all__ = ['figure', 'gcf', 'gca', 'sca', 'gci', 'subplots', 'subplot', 'subplot_mosaic', 'axes', 'close', 'get_fignums', 'clf', 'cla', 'savefig', 'show', 'draw', 'pause', 'ion', 'ioff', 'isinteractive', 'title', 'xlabel', 'ylabel', 'xscale', 'yscale', 'xlim', 'ylim', 'xticks', 'yticks', 'suptitle', 'supxlabel', 'supylabel', 'figtext', 'pie', 'boxplot', 'violinplot', 'axline', 'broken_barh', 'eventplot', 'matshow', 'hexbin', 'hist2d', 'figaspect', 'sci', 'clim', 'subplot2grid', 'fignum_exists', 'delaxes', 'setp', 'getp', 'imread', 'imsave', 'colorbar', 'tight_layout', 'subplots_adjust', 'figlegend', 'get_current_fig_manager', 'set_cmap', 'Figure', 'GridSpec', 'cm', 'colors', 'style', 'rc', 'rc_context', 'rcdefaults', 'rcParams', 'colormaps', 'get_cmap', 'plot', 'scatter', 'bar', 'barh', 'bar_label', 'hist', 'fill_between', 'fill_betweenx', 'fill', 'errorbar', 'step', 'semilogx', 'semilogy', 'loglog', 'stairs', 'stem', 'stackplot', 'axhline', 'axvline', 'axhspan', 'axvspan', 'hlines', 'vlines', 'clabel', 'barbs', 'quiver', 'quiverkey', 'tricontourf', 'tricontour', 'tripcolor', 'triplot', 'streamplot', 'text', 'annotate', 'legend', 'grid', 'tick_params', 'minorticks_on', 'minorticks_off', 'twinx', 'twiny', 'axis', 'margins', 'autoscale', 'locator_params', 'ticklabel_format', 'imshow', 'add_patch', 'pcolormesh', 'pcolor', 'contour', 'contourf']
