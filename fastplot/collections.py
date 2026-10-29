"""matplotlib.collections: many lines (or polygons) as one artist.

    from fastplot.collections import LineCollection
    lc = LineCollection(segments, colors='r', linestyles='dashed', alpha=0.5)
    ax.add_collection(lc)

Made without an axes, bound to one by add_collection; drawn as
matplotlib's collections are (butt caps, each line its colour and width).
"""
from . import _plots
from ._style import rc

__all__ = ['LineCollection']


def _one(*values, default=None):
    return next((v for v in values if v is not None), default)


class LineCollection(_plots.LineCollection):
    def __init__(self, segments, *, colors=None, color=None, linewidths=None, linewidth=None,
                 lw=None, linestyles=None, linestyle=None, ls=None, alpha=None, zorder=None,
                 label=None, transform=None, clip_on=True, antialiaseds=None, **kw):
        if kw:
            raise TypeError('LineCollection: unsupported keyword(s): %s' % ', '.join(sorted(kw)))
        super().__init__(None, segments, 'data', _one(colors, color, default=rc('lines.color')),
                         _one(linewidths, linewidth, lw, default=rc('lines.linewidth')),
                         _one(linestyles, linestyle, ls, default='solid'), alpha, zorder, label)
        self.transform = transform

    def set_segments(self, segments):
        n = len(self.lines)
        super().__init__(self.axes, segments, self.src, self.colors[:1] if n else 'k',
                         self.linewidths[:1] if n else 1.0, self.linestyle, None, self.zorder,
                         self.label)

    def get_segments(self):
        return list(self.lines)
