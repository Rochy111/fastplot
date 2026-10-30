"""matplotlib.lines: Line2D, for lines made by hand -- legend handles
(proxy artists) and ax.add_line.

    from fastplot.lines import Line2D
    handles = [Line2D([0], [0], color='C0', lw=2, label='model'),
               Line2D([], [], marker='o', ls='', color='k', label='obs')]
    ax.legend(handles=handles)
"""
from ._artists import Line2D as _Line2D
from ._rc import active as _rc

__all__ = ['Line2D']


class Line2D(_Line2D):
    """matplotlib's Line2D(xdata, ydata, **kw), in data coordinates once
    added to an axes (ax.add_line)."""

    def __init__(self, xdata, ydata, *, linewidth=None, linestyle=None, color=None,
                 marker=None, markersize=None, markeredgewidth=None, markeredgecolor=None,
                 markerfacecolor=None, markerfacecoloralt='none', fillstyle=None, alpha=None,
                 zorder=None, label=None, transform=None, drawstyle=None, markevery=None,
                 **kw):
        rc = _rc()
        linewidth = kw.pop('lw', linewidth)
        linestyle = kw.pop('ls', linestyle)
        color = kw.pop('c', color)
        markersize = kw.pop('ms', markersize)
        markeredgewidth = kw.pop('mew', markeredgewidth)
        markeredgecolor = kw.pop('mec', markeredgecolor)
        markerfacecolor = kw.pop('mfc', markerfacecolor)
        dashes = kw.pop('dashes', None)
        for k in ('solid_capstyle', 'dash_capstyle', 'solid_joinstyle', 'dash_joinstyle',
                  'antialiased', 'clip_on', 'path_effects', 'rasterized', 'visible', 'gid'):
            kw.pop(k, None)
        if kw:
            raise TypeError('Line2D: unsupported keyword(s): %s' % ', '.join(sorted(kw)))
        if dashes is not None and linestyle is None:
            linestyle = (0, tuple(dashes))
        super().__init__(None, xdata, ydata, 'data', color,
                         rc['lines.linewidth'] if linewidth is None else linewidth,
                         rc['lines.linestyle'] if linestyle is None else linestyle,
                         rc['lines.marker'] if marker is None else marker,
                         rc['lines.markersize'] if markersize is None else markersize,
                         markerfacecolor, markeredgecolor,
                         rc['lines.markeredgewidth'] if markeredgewidth is None
                         else markeredgewidth, alpha, zorder, label, fillstyle or 'full',
                         markerfacecoloralt)
        if color is None:                               # the axes' cycle on add_line
            self.color = None if transform is None else rc['lines.color']
        self.transform = transform
        self.drawstyle = drawstyle or 'default'
        self.markevery = markevery

    def legend_style(self):
        if self.color is None:
            self.color = _rc()['lines.color']
            self.mfc = self.color if self.mfc is None else self.mfc
            self.mec = self.color if self.mec is None else self.mec
        return super().legend_style()
