"""matplotlib's Legend: entries (a handle drawn in a box, its label)
packed as matplotlib's offsetbox packs them, framed, and placed by a loc
code -- or by 'best', matplotlib's search of the ten spots for the one
covering the least data.

    ax.legend(); ax.legend(loc='upper left', ncols=2, title='runs')
    fig.legend(handles, labels, loc='lower center')

Sizes are matplotlib's rcParams legend.* in font-size units. Each entry
is [handle box | handletextpad | label] on one baseline; entries stack in
columns (labelspacing apart), columns side by side (columnspacing), a
title above, borderpad round it all; the frame is a box rounded by 0.2
font sizes, framealpha its alpha.
"""
import numpy as np

from . import _draw
from ._jit import njit
from ._style import _dashes, fontsize_pt, rc, rgba

LOCS = {'best': 0, 'upper right': 1, 'upper left': 2, 'lower left': 3, 'lower right': 4,
        'right': 5, 'center left': 6, 'center right': 7, 'lower center': 8, 'upper center': 9,
        'center': 10}
_ANCHOR = {1: (1, 1), 2: (0, 1), 3: (0, 0), 4: (1, 0), 5: (1, 0.5), 6: (0, 0.5), 7: (1, 0.5),
           8: (0.5, 0), 9: (0.5, 1), 10: (0.5, 0.5)}


def handles_and_labels(axes_list):
    """The artists with a label that does not start with '_', in order."""
    handles, labels = [], []
    for ax in axes_list:
        for a in [*ax.artists, *getattr(ax, 'containers', ())]:
            lab = getattr(a, 'label', None)
            if lab is None or str(lab).startswith('_') or not hasattr(a, 'legend_style'):
                continue
            handles.append(a)
            labels.append(str(lab))
    return handles, labels


def parse_args(args, kw, axes_list):
    """matplotlib's legend(), legend(labels), legend(handles, labels) and
    handles=/labels= forms: (handles, labels)."""
    handles, labels = kw.pop('handles', None), kw.pop('labels', None)
    if len(args) == 2:
        handles, labels = args
    elif len(args) == 1:
        labels = args[0]
    elif args:
        raise TypeError('legend takes at most 2 positional arguments')
    if handles is None:
        found, found_labels = handles_and_labels(axes_list)
        if labels is None:
            return found, found_labels
        # legend(labels): the labelled artists in order, then any others
        pool = found + [a for ax in axes_list for a in ax.artists
                        if hasattr(a, 'legend_style') and a not in found]
        handles = pool[:len(labels)]
    handles = [_unwrap(h) for h in handles]
    if labels is None:
        labels = [str(getattr(h, 'label', '')) for h in handles]
    n = min(len(handles), len(labels))
    return list(handles[:n]), [str(s) for s in labels[:n]]


def _unwrap(h):
    """A container (bar, errorbar, stem) stands for its first artist; a list
    (plot's return) for its first line."""
    if isinstance(h, (list, tuple)) and h and not hasattr(h, 'legend_style'):
        return _unwrap(h[0])
    for attr in ('patches', 'lines', 'markerline'):
        inner = getattr(h, attr, None)
        if inner is not None and not hasattr(h, 'legend_style'):
            return _unwrap(inner)
    return h


def _round_rect(x0, y0, x1, y1, rad, n=8):
    """A box with corners rounded by rad (matplotlib's 'round' boxstyle:
    a quadratic Bezier per corner, controlled at the corner), as a ring."""
    rad = min(rad, (x1 - x0) / 2, (y1 - y0) / 2)
    t = np.linspace(0, 1, n + 1)[:, None]

    def corner(p0, c, p1):
        p0, c, p1 = np.asarray(p0), np.asarray(c), np.asarray(p1)
        return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p1
    return np.concatenate([corner((x1 - rad, y0), (x1, y0), (x1, y0 + rad)),
                           corner((x1, y1 - rad), (x1, y1), (x1 - rad, y1)),
                           corner((x0 + rad, y1), (x0, y1), (x0, y1 - rad)),
                           corner((x0, y0 + rad), (x0, y0), (x0 + rad, y0))])


class _LegendText:
    """A legend entry's label as matplotlib's Text."""

    def __init__(self, lg, k):
        self.lg, self.k = lg, k

    def get_text(self):
        return self.lg.entries[self.k][1]

    def set_text(self, s):
        h, _ = self.lg.entries[self.k]
        self.lg.entries[self.k] = (h, str(s))

    def set_color(self, c):
        self.lg._label_colors[self.k] = c

    def get_color(self):
        return self.lg._label_colors.get(self.k, rc('text.color'))

    def set_fontsize(self, s):
        self.lg.fontsize = fontsize_pt(s)

    set_size = set_fontsize

    def set_fontweight(self, w):
        pass

    def set_visible(self, v):
        pass

    def set(self, **kw):
        for k, v in kw.items():
            getattr(self, 'set_' + {'size': 'fontsize', 'c': 'color'}.get(k, k))(v)

    def __repr__(self):
        return 'Text(%r)' % self.get_text()


class _LegendTitle:
    """A legend's title as matplotlib's Text."""

    def __init__(self, lg):
        self.lg = lg

    def get_text(self):
        return self.lg.title or ''

    def set_text(self, s):
        self.lg.title = s

    def set_fontsize(self, s):
        self.lg.title_fontsize = fontsize_pt(s)

    set_size = set_fontsize

    def set_fontweight(self, w):
        self.lg.title_weight = w

    set_weight = set_fontweight

    def set_color(self, c):
        self.lg.title_color = c

    def set_ha(self, ha):
        pass

    set_horizontalalignment = set_ha


class _LegendFrame:
    """legend.get_frame(): the frame's colours, width, alpha."""

    def __init__(self, lg):
        self.lg = lg

    def set_facecolor(self, c):
        self.lg.facecolor = c

    def set_edgecolor(self, c):
        self.lg.edgecolor = c

    def set_linewidth(self, w):
        self.lg.frame_lw = float(w)

    def set_alpha(self, a):
        self.lg.framealpha = a

    def set_boxstyle(self, s, **kw):
        self.lg.fancybox = 'round' in str(s)

    def set_visible(self, v):
        self.lg.frameon = bool(v)

    def set_color(self, c):
        self.lg.facecolor = self.lg.edgecolor = c

    def set_linestyle(self, ls):
        pass

    def get_facecolor(self):
        return tuple(v / 255.0 for v in rgba(self.lg.facecolor))

    def set(self, **kw):
        for k, v in kw.items():
            getattr(self, 'set_' + {'fc': 'facecolor', 'ec': 'edgecolor', 'lw': 'linewidth',
                                    'c': 'color'}.get(k, k))(v)


class Legend:
    """A legend of an axes (or a figure); drawn at zorder 5."""

    zorder = 5.0

    def __init__(self, parent, handles, labels, loc=None, ncols=1, fontsize=None, title=None,
                 title_fontsize=None, frameon=None, framealpha=None, facecolor=None,
                 edgecolor=None, fancybox=None, shadow=None, borderpad=None, labelspacing=None,
                 handlelength=None, handleheight=None, handletextpad=None, borderaxespad=None,
                 columnspacing=None, markerscale=None, numpoints=None, scatterpoints=None,
                 bbox_to_anchor=None, mode=None, labelcolor=None, alignment='center',
                 reverse=False, **kw):
        prop = kw.pop('prop', None)
        if prop is not None and fontsize is None:       # a dict or FontProperties
            fontsize = prop.get('size') if isinstance(prop, dict) else \
                getattr(prop, 'get_size', lambda: None)()
        tprop = kw.pop('title_fontproperties', None)
        if tprop is not None and title_fontsize is None:
            title_fontsize = tprop.get('size') if isinstance(tprop, dict) else \
                getattr(tprop, 'get_size', lambda: None)()
        ncols = kw.pop('ncol', ncols)
        kw.pop('bbox_transform', None)
        kw.pop('draggable', None)
        kw.pop('handler_map', None)
        if kw:
            raise TypeError('legend: unsupported keyword(s): %s' % ', '.join(sorted(kw)))
        self.parent = parent
        self.entries = list(zip(handles, labels))
        if reverse:
            self.entries.reverse()
        self._label_colors = {}
        r = lambda v, k: rc('legend.' + k) if v is None else v    # noqa: E731
        loc = r(loc, 'loc')
        self._outside = None
        if isinstance(loc, str) and loc.startswith('outside '):    # figure legends
            words = loc.split()[1:]
            if words[0] in ('left', 'right') and len(words) > 1:  # 'right upper'
                self._outside = words[0]
                words = [words[1], words[0]]
            elif words[0] in ('upper', 'lower'):
                self._outside = words[0]
            loc = ' '.join(words)
        if isinstance(loc, str) and loc in ('upper', 'lower'):
            loc += ' center'
        self.loc = LOCS[loc] if isinstance(loc, str) else (loc if isinstance(loc, int) else
                                                           tuple(loc))
        self.ncols = max(1, int(ncols))
        self.fontsize = fontsize_pt(r(fontsize, 'fontsize'))
        self.title = title
        self.title_fontsize = fontsize_pt(title_fontsize if title_fontsize is not None else
                                          (rc('legend.title_fontsize') or self.fontsize))
        self.frameon = r(frameon, 'frameon')
        self.framealpha = r(framealpha, 'framealpha')
        fc = r(facecolor, 'facecolor')
        self.facecolor = rc('axes.facecolor') if fc == 'inherit' else fc
        ec = r(edgecolor, 'edgecolor')
        self.edgecolor = rc('axes.edgecolor') if ec == 'inherit' else ec
        self.fancybox = r(fancybox, 'fancybox')
        self.shadow = r(shadow, 'shadow')
        self.pad = {k: r(v, k) for k, v in dict(
            borderpad=borderpad, labelspacing=labelspacing, handlelength=handlelength,
            handleheight=handleheight, handletextpad=handletextpad,
            borderaxespad=borderaxespad, columnspacing=columnspacing).items()}
        self.markerscale = r(markerscale, 'markerscale')
        self.numpoints = r(numpoints, 'numpoints')
        self.scatterpoints = r(scatterpoints, 'scatterpoints')
        self.bbox_to_anchor = bbox_to_anchor
        self.mode = mode
        self.labelcolor = r(labelcolor, 'labelcolor')
        self.alignment = alignment
        self.visible = True

    def set_visible(self, v):
        self.visible = bool(v)

    def get_visible(self):
        return self.visible

    def get_texts(self):
        return [_LegendText(self, k) for k in range(len(self.entries))]

    def get_lines(self):
        return [h for h, _ in self.entries]

    legend_handles = property(get_lines)
    legendHandles = legend_handles

    def set_title(self, title, prop=None):
        self.title = title
        if prop is not None:
            size = prop.get('size') if isinstance(prop, dict) else \
                getattr(prop, 'get_size', lambda: None)()
            if size is not None:
                self.title_fontsize = fontsize_pt(size)

    def get_title(self):
        return _LegendTitle(self)

    def get_frame(self):
        return _LegendFrame(self)

    def set_draggable(self, state, use_blit=False, update='loc'):
        return None

    def get_draggable(self):
        return False

    def set_zorder(self, z):
        self.zorder = z

    def set_bbox_to_anchor(self, bbox, transform=None):
        self.bbox_to_anchor = bbox

    def set_loc(self, loc):
        self.loc = LOCS[loc] if isinstance(loc, str) else loc

    def set_frame_on(self, b):
        self.frameon = bool(b)

    def get_frame_on(self):
        return self.frameon

    def set_ncols(self, n):
        self.ncols = max(1, int(n))

    def remove(self):
        if getattr(self.parent, 'legend_', None) is self:
            self.parent.legend_ = None

    # -- layout (matplotlib's offsetbox, in px)
    def _layout(self, r):
        fs = r.px(self.fontsize)
        P = {k: v * fs for k, v in self.pad.items()}
        rows = []
        lp = r.measure('lp', 0, 0, self.fontsize)      # TextArea: at least an 'lp' high
        for h, lab in self.entries:
            ext = r.measure(lab, 0, 0, self.fontsize) or (0.0, 0.0, 0.0, 0.0)
            w, asc, desc = ext[2] - ext[0], max(-ext[1], -lp[1]), max(ext[3], lp[3])
            hd = 0.35 * fs * (self.pad['handleheight'] - 0.7)     # the handle box's descent
            hh = P['handleheight']
            # [handle | text] on a baseline: height above it and below
            above = max(hh - hd, asc)
            below = max(hd, desc)
            rows.append(dict(h=h, lab=lab, tw=w, above=above, below=below, hd=hd,
                             width=P['handlelength'] + P['handletextpad'] + w))
        cols = [c for c in np.array_split(np.arange(len(rows)), self.ncols) if len(c)]
        col_w = [max(rows[i]['width'] for i in c) for c in cols]
        col_h = [sum(rows[i]['above'] + rows[i]['below'] for i in c) +
                 P['labelspacing'] * (len(c) - 1) for c in cols]
        body_w = sum(col_w) + P['columnspacing'] * (len(cols) - 1) if cols else 0.0
        body_h = max(col_h) if col_h else 0.0
        title_h = title_w = 0.0
        if self.title:
            te = r.measure(self.title, 0, 0, self.title_fontsize,
                           weight=getattr(self, 'title_weight', 'normal'))
            tl = r.measure('lp', 0, 0, self.title_fontsize)
            self._title_asc = max(-te[1], -tl[1])
            title_w, title_h = te[2] - te[0], self._title_asc + max(te[3], tl[3])
        inner_w = max(body_w, title_w)
        inner_h = body_h + (title_h + P['labelspacing'] if self.title else 0.0)
        bp = P['borderpad']
        return dict(P=P, rows=rows, cols=cols, col_w=col_w, body_w=body_w, inner_w=inner_w,
                    title_h=title_h, title_w=title_w, W=inner_w + 2 * bp,
                    H=inner_h + 2 * bp, fs=fs)

    def _parent_box(self, r):
        p = self.parent
        if hasattr(p, 'box_f'):
            box = p.box_f
        else:                                           # a figure
            box = (r.ox, r.oy, r.ox + float(r.W), r.oy + float(r.H))
        if self.bbox_to_anchor is None:
            return box
        b = tuple(self.bbox_to_anchor)
        x0, y0, x1, y1 = box
        if len(b) == 2:
            px = x0 + b[0] * (x1 - x0)
            py = y1 - b[1] * (y1 - y0)
            return (px, py, px, py)
        bx, by, bw, bh = b
        return (x0 + bx * (x1 - x0), y1 - (by + bh) * (y1 - y0),
                x0 + (bx + bw) * (x1 - x0), y1 - by * (y1 - y0))

    def _anchored(self, code, W, H, box, pad):
        """matplotlib's _get_anchored_bbox: the box placed at a loc in the
        parent box padded inward by pad (a point -- bbox_to_anchor=(x, y) --
        pads to an inverted box, so the legend sits pad off the point)."""
        x0, y0, x1, y1 = box
        ax_, ay_ = _ANCHOR[code]
        left = x0 + pad + (x1 - x0 - 2 * pad - W) * ax_
        top = y1 - pad - (y1 - y0 - 2 * pad - H) * ay_ - H
        return left, top

    def _best(self, r, W, H, box, pad):
        """matplotlib's _find_best_position: of locs 1..10, the first with
        the least badness -- line vertices and scatter offsets inside,
        rectangle patches overlapping, lines (and other patches' and
        polygons' outlines) meeting the box."""
        ax = self.parent
        pts, lines, boxes = [], [], []
        for a in ax.artists:
            if not getattr(a, 'visible', True):
                continue
            kind = type(a).__name__
            try:
                if kind == 'Line2D':
                    lines.append(ax._lines_px(a.x, a.y, np.array([0, a.x.size]), a.src)[0])
                elif kind == 'Scatter':
                    pts.append(np.column_stack(a._screen(ax)))
                elif kind == 'PolyCollection':
                    for p in a.polys:
                        px, py = ax._points_px(p[:, 0], p[:, 1], a.src)
                        if getattr(a, 'rectangles', False):
                            boxes.append((px.min(), py.min(), px.max(), py.max()))
                        else:
                            lines.append(np.column_stack([px, py]))
                elif kind == 'Rectangle':
                    v = a.get_path().vertices
                    px, py = ax._points_px(v[:, 0], v[:, 1], a.src)
                    boxes.append((px.min(), py.min(), px.max(), py.max()))
                elif hasattr(a, 'get_path') and kind != 'FancyArrowPatch':
                    p = a.get_path()
                    if p is not None and len(p.vertices):
                        v = p.vertices
                        px, py = ax._points_px(v[:, 0], v[:, 1], a.src)
                        lines.append(np.column_stack([px, py]))
            except Exception:                           # noqa: BLE001 -- not in the count
                continue
        pts = np.concatenate(pts) if pts else np.empty((0, 2))
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        verts = [xy[np.isfinite(xy).all(axis=1)] for xy in lines]
        segs = [np.column_stack([xy[:-1], xy[1:]]) for xy in lines if len(xy) > 1]
        firsts = [v[0] for v in verts if len(v)]
        best = None
        for code in range(1, 11):
            left, top = self._anchored(code, W, H, box, pad)
            l, t, rgt, b = left, top, left + W, top + H

            def inside(p):
                return int(((p[:, 0] > l) & (p[:, 0] < rgt) & (p[:, 1] > t) &
                            (p[:, 1] < b)).sum())
            bad = inside(pts) + sum(inside(v) for v in verts if len(v))
            bad += int(((boxes[:, 0] < rgt) & (boxes[:, 2] > l) & (boxes[:, 1] < b) &
                        (boxes[:, 3] > t)).sum())
            cx, cy = 0.5 * (l + rgt), 0.5 * (t + b)
            met = [2.0 * abs(f[0] - cx) <= W and 2.0 * abs(f[1] - cy) <= H for f in firsts]
            bad += sum(1 for k, sg in enumerate(segs)
                       if (k < len(met) and met[k]) or _crosses(sg, l, t, rgt, b))
            if bad == 0:
                return left, top
            if best is None or bad < best[0]:
                best = (bad, left, top)
        return best[1], best[2]

    # -- drawing
    def render(self, r, ax=None):
        if not self.visible or not self.entries:
            return
        L = self._layout(r)
        P, W, H, fs = L['P'], L['W'], L['H'], L['fs']
        box = self._parent_box(r)
        pad = P['borderaxespad']
        if self.loc == 0 and hasattr(self.parent, 'artists'):
            left, top = self._best(r, W, H, box, pad)
        elif self.loc == 0:                             # a figure's: upper right
            left, top = self._anchored(1, W, H, box, pad)
        elif isinstance(self.loc, tuple):               # loc=(x, y) in parent fractions
            x0, y0, x1, y1 = box
            left = x0 + self.loc[0] * (x1 - x0)
            top = y1 - self.loc[1] * (y1 - y0) - H
        else:
            left, top = self._anchored(self.loc, W, H, box, pad)
        if self.frameon:
            ring = _round_rect(left, top, left + W, top + H, 0.2 * fs) if self.fancybox else \
                np.array([[left, top], [left + W, top], [left + W, top + H], [left, top + H]])
            fc = rgba(self.facecolor, self.framealpha)
            ec = rgba(self.edgecolor, self.framealpha)
            lw = r.px(getattr(self, 'frame_lw', None) or rc('patch.linewidth'))
            closed = _draw.snap(np.r_[ring, ring[:1]], np.array([0, len(ring) + 1]),
                                lw if ec[3] else 0.0, None)
            ring = closed[:-1]
            if self.shadow:
                r.c.polygons(ring + 2.0, np.array([0, len(ring)]), rgba('k', 0.5), None,
                             'nonzero')
            r.c.polygons(ring, np.array([0, len(ring)]), fc, None, 'nonzero')
            if ec[3]:
                r.lines(closed, np.array([0, len(closed)]), ec, lw, None, cap=_draw.BUTT)
        bp = P['borderpad']
        y = top + bp
        a = {'center': 0.5, 'left': 0.0, 'right': 1.0}[self.alignment]
        if self.title:
            tx = left + bp + (L['inner_w'] - L['title_w']) * a
            r.text(self.title, tx, y + self._title_asc, self.title_fontsize,
                   getattr(self, 'title_color', None) or rc('text.color'),
                   weight=getattr(self, 'title_weight', 'normal'))
            y += L['title_h'] + P['labelspacing']
        x = left + bp + (L['inner_w'] - L['body_w']) * a
        for c, cw in zip(L['cols'], L['col_w']):
            yy = y
            for i in c:
                row = L['rows'][i]
                base = yy + row['above']                # the row's baseline
                self._draw_handle(r, row['h'], x, base + row['hd'] - P['handleheight'],
                                  P['handlelength'], P['handleheight'])
                col = self._label_colors.get(i) or self._label_color(row['h'])
                r.text(row['lab'], x + P['handlelength'] + P['handletextpad'], base,
                       self.fontsize, col)
                yy = base + row['below'] + P['labelspacing']
            x += cw + P['columnspacing']
        r.extents.append((left, top, left + W, top + H))

    def _label_color(self, h):
        lc = self.labelcolor
        if lc in (None, 'None', 'none'):
            return rc('text.color')
        if lc in ('linecolor', 'markerfacecolor', 'mfc', 'markeredgecolor', 'mec'):
            st = h.legend_style()
            return st.get('color', st.get('facecolor', 'black'))
        return lc

    def _draw_handle(self, r, h, x, y, w, hgt):
        """A handle in its box (x, y top-left, w x hgt px), as matplotlib's
        HandlerLine2D / HandlerPatch / HandlerPathCollection draw it (Agg's
        snapping and caps)."""
        from ._artists import _markers
        st = h.legend_style()
        kind = st['kind']
        cy = y + hgt / 2.0
        if kind == 'patch':
            ring = np.array([[x, y], [x + w, y], [x + w, y + hgt], [x, y + hgt], [x, y]])
            ec, lw = st.get('edgecolor'), r.px(st.get('linewidth') or 0.0)
            edge = ec is not None and ec[3] and lw > 0
            ring = _draw.snap(ring, np.array([0, 5]), lw if edge else 0.0, None)
            if st['facecolor'] is not None and st['facecolor'][3]:
                r.c.polygons(ring[:4], np.array([0, 4]), st['facecolor'], None, 'nonzero')
            if edge:
                r.lines(ring, np.array([0, 5]), ec, lw, None, cap=_draw.BUTT)
            return
        if kind == 'errorbar':                          # HandlerErrorbar: bars, caps, line
            e = 0.5 * self.fontsize * r.dpi / 72.0       # xerr_size = yerr_size
            cx, b = x + w / 2.0, st['bar']
            for on, seg, cap in ((st['xerr'], [[cx - e, cy], [cx + e, cy]], '|'),
                                 (st['yerr'], [[cx, cy - e], [cx, cy + e]], '_')):
                if not on:
                    continue
                seg = np.array(seg)
                r.lines(_draw.snap(seg, np.array([0, 2]), r.px(b['linewidth']), None),
                        np.array([0, 2]), b['color'], r.px(b['linewidth']), None, cap=_draw.BUTT)
                if st['cap'] is not None:
                    c = dict(st['cap'], marker=cap)
                    _markers(r, seg, cap, r.px(c['markersize']) * self.markerscale, c['mfc'],
                             c['mec'], r.px(c['mew']), None, snap=None)
            if st['line'] is not None:
                self._draw_handle_style(r, st['line'], x, y, w, hgt)
            return
        if kind == 'stem':                              # HandlerStem: baseline, stem, marker
            base_y, top = y + hgt, y + hgt * (1.0 - (0.5 * 0.375 + 0.5))
            self._draw_handle_style(r, dict(st['base'], marker=None), x, base_y - hgt / 2.0,
                                    w, hgt)
            s = st['stem']
            seg = np.array([[x + w / 2.0, base_y], [x + w / 2.0, top]])
            r.lines(_draw.snap(seg, np.array([0, 2]), r.px(s['linewidth']), None),
                    np.array([0, 2]), s['color'], r.px(s['linewidth']), None, cap=_draw.BUTT)
            m = st['marker']
            _markers(r, seg[1:], m['marker'], r.px(m['markersize']) * self.markerscale, m['mfc'],
                     m['mec'], r.px(m['mew']), None, snap=None)
            return
        self._draw_handle_style(r, st, x, y, w, hgt)

    def _draw_handle_style(self, r, st, x, y, w, hgt):
        from ._artists import _markers
        kind = st['kind']
        cy = y + hgt / 2.0
        if kind == 'line':
            lw = st.get('linewidth') or 0.0
            ls = st.get('linestyle', '-')
            if lw > 0 and ls not in ('None', 'none', '', ' '):
                d = _dashes(ls, lw)
                r.lines(np.array([[x, cy], [x + w, cy]]), np.array([0, 2]), rgba(st['color']),
                        r.px(lw), None, None if d is None else tuple(r.px(v) for v in d), None,
                        _draw.PROJECTING if d is None else _draw.BUTT)
            if st.get('marker') not in (None, 'None', 'none', ''):
                xs = np.array([[x + w / 2.0, cy]]) if self.numpoints == 1 else                     np.c_[np.linspace(x + 0.3 * w, x + 0.7 * w, self.numpoints),
                          np.full(self.numpoints, cy)]
                _markers(r, xs, st['marker'], r.px(st['markersize']) * self.markerscale,
                         st['mfc'], st['mec'], r.px(st['mew']), None, snap=None,
                         fillstyle=st.get('fillstyle', 'full'), fc_alt=st.get('mfcalt'))
            return
        if kind == 'scatter':                           # matplotlib's scatteryoffsets: 3/8 up
            _markers(r, np.array([[x + w / 2.0, y + hgt * 0.625]]), st['marker'],
                     np.sqrt(st['s']) * r.dpi / 72.0 * self.markerscale, st['facecolor'],
                     st['edgecolor'], r.px(st['ew']), None, snap=None, scatter=True)


@njit(cache=True, nogil=True)
def _crosses_any(s, cx, cy, w, h):
    """_crosses' test, segment by segment, the same float operations."""
    for k in range(s.shape[0]):
        x1, y1, x2, y2 = s[k, 0], s[k, 1], s[k, 2], s[k, 3]
        dx, dy = abs(x1 - x2), abs(y1 - y2)
        if (abs(x1 + x2 - 2.0 * cx) < dx + w and abs(y1 + y2 - 2.0 * cy) < dy + h and
                2.0 * abs((x1 - cx) * (y1 - y2) - (y1 - cy) * (x1 - x2)) < w * dy + h * dx):
            return True
    return False


def _crosses(s, l, t, rgt, b):
    """Does any segment (x1, y1, x2, y2 rows) meet the box? matplotlib's
    segment_intersects_rectangle, a separating-axis test (NaN never
    meets)."""
    cx, cy, w, h = (l + rgt) * 0.5, (t + b) * 0.5, rgt - l, b - t
    return bool(_crosses_any(np.ascontiguousarray(s, np.float64), float(cx), float(cy),
                             float(w), float(h)))
