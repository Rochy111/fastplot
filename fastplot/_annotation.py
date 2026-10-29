"""matplotlib's Annotation: a text at xytext, an arrow from it to xy.

    ax.annotate('max', xy=(t0, v0), xytext=(10, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle='->'))

Coordinates (xycoords, textcoords): 'data', 'axes fraction', 'figure
fraction', 'axes pixels', 'figure pixels', 'offset points' / 'offset
pixels' (textcoords: from xy) or a transform. The arrow is matplotlib's
FancyArrowPatch (_patches.arrow_paths): with arrowstyle, that style,
shrunk 2 pt at each end; without, the 'simple' arrow standing in for
YAArrow (width / headwidth / headlength / shrink), from the corner of the
text box nearest xy. It leaves the text's box padded by 4 pt (or its
bbox patch), mutation_scale the text size; edge black, face C0, round
caps and joins, as matplotlib's defaults.
"""
import numpy as np

from . import _draw, _path
from ._artists import Text
from ._patches import arrow_paths
from ._style import fontsize_pt, rc, rgba

_YA = ('width', 'headwidth', 'headlength', 'shrink', 'frac')


class Annotation(Text):
    zorder = 3.0

    def __init__(self, ax, s, xy, xytext, xycoords, textcoords, arrowprops, annotation_clip,
                 **kw):
        x, y = xy if xytext is None else xytext
        super().__init__(ax, x, y, s, 'annotation', **kw)
        self.xy = xy
        self.xycoords = xycoords
        self.textcoords = xycoords if textcoords is None else textcoords
        self.arrowprops = None if arrowprops is None else dict(arrowprops)
        self.annotation_clip = annotation_clip

    def _px(self, r, ax, coords, x, y, ref=None):
        """A point in coords as canvas px (ref: xy's px, for offsets)."""
        if coords in ('offset points', 'offset pixels', 'offset fontsize'):
            k = {'offset points': r.dpi / 72.0, 'offset pixels': 1.0,
                 'offset fontsize': r.px(fontsize_pt(self.fontsize))}[coords]
            return ref[0] + x * k, ref[1] - y * k
        if coords == 'figure fraction':
            return x * r.W, (1 - y) * r.H
        if coords == 'figure pixels':
            return x, r.H - y
        if coords == 'axes fraction':
            b = ax.box_f
            return b[0] + x * (b[2] - b[0]), b[3] - y * (b[3] - b[1])
        if coords == 'axes pixels':
            b = ax.box_f
            return b[0] + x, b[3] - y
        if hasattr(coords, 'to_px'):
            px, py = coords.to_px(np.array([x], float), np.array([y], float))
            return px[0], py[0]
        src = ax._src(None) if coords == 'data' else coords
        px, py = ax._points_px(np.array([x], float), np.array([y], float), src)
        return px[0], py[0]

    def render(self, r, ax):
        x, y = self.xy
        if self.xycoords == 'data':
            x, y = (float(np.ravel(v)[0]) for v in ax._xy(x, y))
        ex, ey = self._px(r, ax, self.xycoords, x, y)
        if not (np.isfinite(ex) and np.isfinite(ey)):
            return
        clip = self.annotation_clip
        if (clip or (clip is None and self.xycoords == 'data')):
            x0, y0, x1, y1 = ax.box_f
            if not (x0 <= ex <= x1 and y0 <= ey <= y1):
                return
        x, y = self.x, self.y
        if self.textcoords == 'data':
            x, y = (float(np.ravel(v)[0]) for v in ax._xy(x, y))
        tx, ty = self._px(r, ax, self.textcoords, x, y, (ex, ey))
        size = fontsize_pt(self.fontsize)
        box = r.measure(self.s, tx, ty, size, ha=self.ha, va=self.va, rotation=self.rotation,
                        weight=self.weight, rotation_mode=self.rotation_mode)
        if self.arrowprops is not None and box is not None:
            self._arrow(r, ax, box, (ex, ey), size)
        r.text(self.s, tx, ty, size, self.color, ha=self.ha, va=self.va, rotation=self.rotation,
               weight=self.weight, style=self.style, rotation_mode=self.rotation_mode,
               bbox=self.bbox, alpha=self.alpha, family=self.family)

    def _arrow(self, r, ax, box, end, size):
        ap = dict(self.arrowprops)
        k = r.dpi / 72.0
        ms = ap.pop('mutation_scale', size)
        x0, y0, x1, y1 = box
        style = ap.pop('arrowstyle', None)
        relpos = ap.pop('relpos', (0.5, 0.5))
        shrinkA, shrinkB = ap.pop('shrinkA', 2.0), ap.pop('shrinkB', 2.0)
        if style is None:                               # matplotlib's YAArrow stand-in
            w, hw, hl = ap.pop('width', 4.0), ap.pop('headwidth', 12.0), ap.pop('headlength',
                                                                               12.0)
            shrink = ap.pop('shrink', 0.0)
            ap.pop('frac', None)
            style = 'simple,head_length=%r,head_width=%r,tail_width=%r' % (hl / ms, hw / ms,
                                                                           w / ms)
            bx, rx = min(((x0, 0.0), ((x0 + x1) / 2, 0.5), (x1, 1.0)),
                         key=lambda v: abs(v[0] - end[0]))
            by, ry = min(((y1, 0.0), ((y0 + y1) / 2, 0.5), (y0, 1.0)),
                         key=lambda v: abs(v[0] - end[1]))
            relpos = (rx, ry)
            shrinkA = shrinkB = shrink * np.hypot(by - end[1], bx - end[0]) / k
        else:
            for name in _YA:
                ap.pop(name, None)
        begin = (x0 + (x1 - x0) * relpos[0], y1 - (y1 - y0) * relpos[1])
        patchA = ap.pop('patchA', None)
        if patchA is None and str(self.s):
            if self.bbox is not None:                   # its bbox patch (as a box)
                patchA = r._text_box(box, self.bbox, size, draw=False)
            else:
                pad = 4.0 * k
                patchA = (x0 - pad / 2, y0 - pad / 2, x1 + pad / 2, y1 + pad / 2)
        lw = ap.pop('linewidth', ap.pop('lw', 1.0))
        color = ap.pop('color', None)
        ec = rgba(ap.pop('edgecolor', ap.pop('ec', color if color is not None else 'black')),
                  ap.get('alpha'))
        fc = rgba(ap.pop('facecolor', ap.pop('fc', color if color is not None else
                                              rc('patch.facecolor'))), ap.pop('alpha', None))
        cstyle = ap.pop('connectionstyle', 'arc3')
        ap.pop('patchB', None)
        for name in ('zorder', 'clip_on', 'antialiased', 'fill', 'mutation_aspect'):
            ap.pop(name, None)
        paths, fills = arrow_paths(begin, end, style, cstyle, mutation_scale=ms * k,
                                   shrinkA=shrinkA * k, shrinkB=shrinkB * k,
                                   linewidth=lw * k, patchA=patchA)
        w = lw * k
        for p, fillable in zip(paths, fills):
            xy, offs, _ = _path.flatten(p.vertices, p.codes)
            if not xy.shape[0]:
                continue
            if fillable and fc[3]:
                r.c.polygons(xy, offs, fc, None, 'nonzero')
            if w > 0 and ec[3]:
                r.lines(xy, offs, ec, w, None, cap=_draw.ROUND)
