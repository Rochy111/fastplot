"""matplotlib.collections: many lines (or polygons) as one artist.

    from fastplot.collections import LineCollection
    lc = LineCollection(segments, colors='r', linestyles='dashed', alpha=0.5)
    ax.add_collection(lc)

    lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(0, 1))
    lc.set_array(values)                     # coloured by values; a colorbar can show it
    ax.add_collection(lc); fig.colorbar(lc, ax=ax)

    pc = PatchCollection([Circle(...), Rectangle(...)], cmap='jet', alpha=0.5)
    pc.set_array(values); ax.add_collection(pc)

Made without an axes, bound to one by add_collection; drawn as
matplotlib's collections are (butt caps, each line its colour and width).
"""
import numpy as np

from . import _color, _path, _plots
from ._style import rc

__all__ = ['LineCollection', 'PolyCollection', 'PatchCollection']


def _one(*values, default=None):
    return next((v for v in values if v is not None), default)


class _ScalarMappable:
    """set_array / cmap / norm: values coloured through a colormap (a
    colorbar can show the collection)."""

    def _init_mappable(self, cmap, norm, array=None, clim=None):
        self.cmap = _color.get_cmap(cmap)
        self._norm_in = norm
        self._vmin = self._vmax = None
        if clim is not None:
            self._vmin, self._vmax = clim
        self._array = None if array is None else np.asarray(array, np.float32).ravel()
        self._norm = None

    def set_array(self, A):
        self._array = None if A is None else np.ma.filled(
            np.ma.asarray(A, np.float32), np.nan).ravel()
        self._norm = None

    def get_array(self):
        return self._array

    def set_cmap(self, cmap):
        self.cmap = _color.get_cmap(cmap)

    def get_cmap(self):
        return self.cmap

    def set_norm(self, norm):
        self._norm_in, self._norm = norm, None

    def set_clim(self, vmin=None, vmax=None):
        if isinstance(vmin, (tuple, list)):
            vmin, vmax = vmin
        self._vmin, self._vmax, self._norm = vmin, vmax, None

    def get_clim(self):
        n = self.norm_spec()
        return n.vmin, n.vmax

    def norm_spec(self):
        if self._norm is None:
            self._norm = _color.make_norm(self._norm_in, vmin=self._vmin, vmax=self._vmax,
                                          data=self._array, N=self.cmap.N)
        return self._norm

    @property
    def norm(self):
        return self.norm_spec()

    @norm.setter
    def norm(self, n):
        self.set_norm(n)

    def _mapped(self, n):
        """The values' colours, (n, 4) uint8 (None: no values)."""
        if self._array is None or not self._array.size:
            return None
        out = np.empty((1, self._array.size, 4), np.uint8)
        _color.shade(self._array[None, :], self.cmap, self.norm_spec(), out=out,
                     alpha=self.alpha, rounding='nearest')
        return np.resize(out[0], (n, 4))


class LineCollection(_ScalarMappable, _plots.LineCollection):
    def __init__(self, segments, *, colors=None, color=None, linewidths=None, linewidth=None,
                 lw=None, linestyles=None, linestyle=None, ls=None, alpha=None, zorder=None,
                 label=None, transform=None, clip_on=True, antialiaseds=None, cmap=None,
                 norm=None, array=None, clim=None, capstyle=None, joinstyle=None, **kw):
        for k in ('antialiased', 'rasterized', 'path_effects', 'offsets', 'offset_transform'):
            kw.pop(k, None)
        if kw:
            raise TypeError('LineCollection: unsupported keyword(s): %s' % ', '.join(sorted(kw)))
        super().__init__(None, segments, 'data', _one(colors, color, default=rc('lines.color')),
                         _one(linewidths, linewidth, lw, default=rc('lines.linewidth')),
                         _one(linestyles, linestyle, ls, default='solid'), alpha, zorder, label)
        self.transform = transform
        self._init_mappable(cmap, norm, array, clim)

    def set_segments(self, segments):
        n = len(self.lines)
        super(_ScalarMappable, self).__init__(
            self.axes, segments, self.src, self.colors[:1] if n else 'k',
            self.linewidths[:1] if n else 1.0, self.linestyle, self.alpha, self.zorder,
            self.label)

    set_verts = set_paths = set_segments

    def get_segments(self):
        return list(self.lines)

    def set_color(self, c):
        self.colors = _plots._colors(c, len(self.lines), self.alpha)

    set_colors = set_edgecolor = set_color

    def set_linewidth(self, w):
        self.linewidths = np.resize(np.atleast_1d(np.asarray(w, float)), len(self.lines))

    set_linewidths = set_linewidth

    def set_linestyle(self, ls):
        self.linestyle = ls

    def legend_style(self):
        st = super().legend_style()
        cols = self._mapped(len(self.lines))
        if cols is not None and len(cols):
            st['color'] = cols[0]
        return st

    def render(self, r, ax):
        cols = self._mapped(len(self.lines))
        if cols is not None:
            self.colors = cols
        super().render(r, ax)


class PolyCollection(_ScalarMappable, _plots.PolyCollection):
    """matplotlib's PolyCollection(verts, facecolors=, edgecolors=, ...)."""

    def __init__(self, verts, *, facecolors=None, facecolor=None, edgecolors=None,
                 edgecolor=None, linewidths=None, linewidth=None, lw=None, linestyles=None,
                 linestyle=None, alpha=None, zorder=None, label=None, transform=None,
                 cmap=None, norm=None, array=None, clim=None, closed=True, hatch=None,
                 color=None, **kw):
        for k in ('antialiased', 'antialiaseds', 'rasterized', 'clip_on', 'fc', 'ec'):
            if k == 'fc' and k in kw:
                facecolor = kw.pop(k)
            elif k == 'ec' and k in kw:
                edgecolor = kw.pop(k)
            else:
                kw.pop(k, None)
        if kw:
            raise TypeError('PolyCollection: unsupported keyword(s): %s' % ', '.join(sorted(kw)))
        fc = _one(facecolors, facecolor, color, default=rc('patch.facecolor'))
        ec = _one(edgecolors, edgecolor, color if facecolors is None and facecolor is None
                  else None, default='none')
        super().__init__(None, verts, 'data', fc, ec,
                         _one(linewidths, linewidth, lw, default=rc('patch.linewidth')),
                         _one(linestyles, linestyle, default='solid'), alpha, zorder, label,
                         hatch=hatch)
        self.transform = transform
        self._ec_spec = ec
        self._init_mappable(cmap, norm, array, clim)

    def set_facecolor(self, c):
        self.facecolors = _plots._colors(c, len(self.polys), self.alpha)

    set_facecolors = set_facecolor

    def set_edgecolor(self, c):
        self._ec_spec = c
        self.edgecolors = _plots._colors(c, len(self.polys), self.alpha)

    set_edgecolors = set_edgecolor

    def set_color(self, c):
        self.set_facecolor(c)
        self.set_edgecolor(c)

    def set_linewidth(self, w):
        self.linewidths = np.resize(np.atleast_1d(np.asarray(w, float)), len(self.polys))

    set_linewidths = set_linewidth

    def set_verts(self, verts):
        self.polys = [np.asarray(p, np.float64).reshape(-1, 2) for p in verts]

    set_paths = set_verts

    def get_paths(self):
        return list(self.polys)

    def render(self, r, ax):
        cols = self._mapped(len(self.polys))
        if cols is not None:
            self.facecolors = cols
            if isinstance(self._ec_spec, str) and self._ec_spec == 'face':
                self.edgecolors = cols
        super().render(r, ax)


class PatchCollection(PolyCollection):
    """matplotlib's PatchCollection(patches, match_original=False, ...):
    the patches' outlines (curves flattened) as one collection."""

    def __init__(self, patches, *, match_original=False, **kw):
        polys = []
        fcs, ecs, lws = [], [], []
        for p in patches:
            path = p.get_path()
            if path is None or not len(path.vertices):
                continue
            xy, offs, _closed = _path.flatten(path.vertices, path.codes, tol=1e-3 * max(
                1e-12, float(np.ptp(path.vertices[:, 0]) + np.ptp(path.vertices[:, 1]))))
            for k in range(offs.size - 1):
                polys.append(xy[offs[k]:offs[k + 1]])
                if match_original:
                    fc, ec = p._colors()
                    fcs.append(tuple(fc / 255.0))
                    ecs.append(tuple(ec / 255.0))
                    lws.append(p.linewidth)
        if match_original:
            kw.setdefault('facecolors', fcs)
            kw.setdefault('edgecolors', ecs)
            kw.setdefault('linewidths', lws)
        super().__init__(polys, **kw)
