"""Where axes go: matplotlib's subplot grid (GridSpec, SubplotSpec, their
nested sub-grids) in figure fractions, and the shrink of a box to an
aspect ratio about an anchor (apply_aspect / set_box_aspect)."""
import numpy as np


class _Box:
    """A rectangle in figure fractions: left, bottom, width, height (with
    matplotlib's Bbox attributes, x0 .. bounds)."""

    def __init__(self, l, b, w, h):
        self.l, self.b, self.w, self.h = float(l), float(b), float(w), float(h)

    def __iter__(self):
        return iter((self.l, self.b, self.w, self.h))

    x0 = xmin = property(lambda self: self.l)
    y0 = ymin = property(lambda self: self.b)
    x1 = xmax = property(lambda self: self.l + self.w)
    y1 = ymax = property(lambda self: self.b + self.h)
    width = property(lambda self: self.w)
    height = property(lambda self: self.h)
    bounds = property(lambda self: (self.l, self.b, self.w, self.h))
    extents = property(lambda self: (self.l, self.b, self.l + self.w, self.b + self.h))
    p0 = property(lambda self: (self.l, self.b))
    p1 = property(lambda self: (self.l + self.w, self.b + self.h))
    intervalx = property(lambda self: np.array([self.l, self.l + self.w]))
    intervaly = property(lambda self: np.array([self.b, self.b + self.h]))

    def __repr__(self):
        return 'Bbox([[%g, %g], [%g, %g]])' % self.extents


def _Bbox(x0, y0, x1, y1):
    return _Box(x0, y0, x1 - x0, y1 - y0)


def _grid(box, nrows, ncols, wspace, hspace, wr=None, hr=None):
    """matplotlib's GridSpecBase.get_grid_positions over box: per cell
    (bottom, top) and (left, right) arrays."""
    left, bottom, right, top = box.l, box.b, box.l + box.w, box.b + box.h
    tot_w, tot_h = right - left, top - bottom
    hr = [1] * nrows if hr is None else list(hr)
    wr = [1] * ncols if wr is None else list(wr)
    cell_h = tot_h / (nrows + hspace * (nrows - 1))
    sep_h = hspace * cell_h
    norm = cell_h * nrows / sum(hr)
    hs = np.cumsum(np.column_stack([[0] + [sep_h] * (nrows - 1), [r * norm for r in hr]]).flat)
    tops, bottoms = top - hs[0::2], top - hs[1::2]
    cell_w = tot_w / (ncols + wspace * (ncols - 1))
    sep_w = wspace * cell_w
    norm = cell_w * ncols / sum(wr)
    ws = np.cumsum(np.column_stack([[0] + [sep_w] * (ncols - 1), [r * norm for r in wr]]).flat)
    lefts, rights = left + ws[0::2], left + ws[1::2]
    return bottoms, tops, lefts, rights


class GridSpec:
    """matplotlib.gridspec.GridSpec (in a figure) or, with `parent`, a
    GridSpecFromSubplotSpec; indexing gives SubplotSpecs."""

    def __init__(self, nrows, ncols, figure=None, parent=None, wspace=None, hspace=None,
                 width_ratios=None, height_ratios=None, left=None, right=None, bottom=None,
                 top=None):
        self.nrows, self.ncols = nrows, ncols
        self.figure, self.parent = figure, parent
        self.wspace, self.hspace = wspace, hspace
        self.width_ratios, self.height_ratios = width_ratios, height_ratios
        self.pars = dict(left=left, right=right, bottom=bottom, top=top)

    def _box(self, fig):
        if self.parent is not None:
            return self.parent.box(fig)
        p = {k: (v if v is not None else fig.subplotpars[k]) for k, v in self.pars.items()}
        return _Box(p['left'], p['bottom'], p['right'] - p['left'], p['top'] - p['bottom'])

    def _spaces(self, fig):
        ws = self.wspace if self.wspace is not None else fig.subplotpars['wspace']
        hs = self.hspace if self.hspace is not None else fig.subplotpars['hspace']
        return ws, hs

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            k = range(self.nrows * self.ncols)[key]
            ks = [k] if isinstance(k, int) else list(k)
            rows = [i // self.ncols for i in ks]
            cols = [i % self.ncols for i in ks]
            return SubplotSpec(self, min(rows), max(rows), min(cols), max(cols))
        r, c = key
        rr = range(self.nrows)[r]
        cc = range(self.ncols)[c]
        rr = [rr] if isinstance(rr, int) else list(rr)
        cc = [cc] if isinstance(cc, int) else list(cc)
        return SubplotSpec(self, min(rr), max(rr), min(cc), max(cc))

    def subplots(self, *, sharex=False, sharey=False, squeeze=True, subplot_kw=None):
        """matplotlib's GridSpec.subplots: an axes in each cell."""
        fig = self.figure
        if fig is None:
            raise ValueError('GridSpec.subplots() only works for GridSpecs created with a '
                             'parent figure')
        return fig._grid_subplots(self, sharex, sharey, squeeze, subplot_kw)

    def get_geometry(self):
        return self.nrows, self.ncols

    def update(self, **kw):
        for k in ('left', 'right', 'bottom', 'top'):
            if k in kw:
                self.pars[k] = kw.pop(k)
        if 'wspace' in kw:
            self.wspace = kw.pop('wspace')
        if 'hspace' in kw:
            self.hspace = kw.pop('hspace')


class SubplotSpec:
    def __init__(self, gs, r0, r1, c0, c1):
        self.gs, self.r0, self.r1, self.c0, self.c1 = gs, r0, r1, c0, c1

    def box(self, fig):
        cl = getattr(fig, '_cl_boxes', None)
        if cl and self.gs.parent is None:               # placed by the constrained layout
            b = cl.get((id(self.gs), self.r0, self.r1, self.c0, self.c1))
            if b is not None:
                return b
        ws, hs = self.gs._spaces(fig)
        b, t, l, r = _grid(self.gs._box(fig), self.gs.nrows, self.gs.ncols, ws, hs,
                           self.gs.width_ratios, self.gs.height_ratios)
        bottom, top = b[self.r1], t[self.r0]
        left, right = l[self.c0], r[self.c1]
        return _Box(left, bottom, right - left, top - bottom)

    def subgridspec(self, nrows, ncols, **kw):
        return GridSpec(nrows, ncols, parent=self, **kw)


def spec_box(ax, fig):
    """Where an axes' spec puts it (figure fractions) -- or, in a constrained
    layout's first pass, where fig.colorbar's make_axes moved it, as
    matplotlib starts from there."""
    pre = getattr(ax, '_cl_pre', None)
    if pre is not None and getattr(fig, '_layout', None) == 'constrained' and \
            not getattr(fig, '_cl_boxes', None):
        return pre
    return ax.spec.box(fig) if hasattr(ax.spec, 'box') else ax.spec


def _anchor(a):
    names = {'C': (0.5, 0.5), 'SW': (0, 0), 'S': (0.5, 0), 'SE': (1, 0), 'E': (1, 0.5),
             'NE': (1, 1), 'N': (0.5, 1), 'NW': (0, 1), 'W': (0, 0.5)}
    return names[a] if isinstance(a, str) else tuple(a)


def _fit(box, ratio, anchor, figw, figh):
    """box (figure fractions) shrunk to height/width = ratio in pixels,
    placed by anchor -- matplotlib's apply_aspect / set_box_aspect."""
    wpx, hpx = box.w * figw, box.h * figh
    if wpx <= 0 or hpx <= 0 or not np.isfinite(ratio) or ratio <= 0:
        return box
    if hpx / wpx > ratio:
        nh = wpx * ratio / figh
        nw = box.w
    else:
        nw = hpx / ratio / figw
        nh = box.h
    ax, ay = _anchor(anchor)
    return _Box(box.l + (box.w - nw) * ax, box.b + (box.h - nh) * ay, nw, nh)


# ---------------------------------------------------------------- tight_layout

def topmost(spec):
    """The figure-level SubplotSpec a (nested) one lies in."""
    while spec.gs.parent is not None:
        spec = spec.gs.parent
    return spec


def tight_subplotpars(groups, figsize, fontsize_pt, pad=1.08, h_pad=None, w_pad=None,
                      rect=None, suptitle_h=0.0, supxlabel_h=0.0, supylabel_w=0.0):
    """matplotlib's get_tight_layout_figure: the subplot parameters (left,
    right, bottom, top, wspace, hspace) that fit every subplot's
    decorations in the figure, pad font sizes apart. groups: per topmost
    SubplotSpec (figure-level grid), (spec, its axes' union box, the union
    of what they draw (xmin, ymin, xmax, ymax)) in figure fractions;
    suptitle_h: the suptitle's height, figure fraction (0: none)."""
    if not groups:
        return {}
    gss = {id(g[0].gs): g[0].gs for g in groups}.values()
    nrows = max(gs.nrows for gs in gss)
    ncols = max(gs.ncols for gs in gss)
    spans = []
    for spec, _, _ in groups:
        dr, mr = divmod(nrows, spec.gs.nrows)
        dc, mc = divmod(ncols, spec.gs.ncols)
        if mr or mc:
            return {}                                   # grids that do not divide: skipped
        spans.append((range(spec.r0 * dr, (spec.r1 + 1) * dr),
                      range(spec.c0 * dc, (spec.c1 + 1) * dc)))
    sup = (suptitle_h, supxlabel_h, supylabel_w)
    kw = _auto_adjust(groups, spans, (nrows, ncols), figsize, fontsize_pt, pad, h_pad, w_pad,
                      None, sup)
    if rect is not None and kw:
        left, bottom, right, top = rect
        if left is not None:
            left += kw['left']
        if bottom is not None:
            bottom += kw['bottom']
        if right is not None:
            right -= 1 - kw['right']
        if top is not None:
            top -= 1 - kw['top']
        kw = _auto_adjust(groups, spans, (nrows, ncols), figsize, fontsize_pt, pad, h_pad,
                          w_pad, (left, bottom, right, top), sup)
    return kw


def _auto_adjust(groups, spans, shape, figsize, fontsize_pt, pad, h_pad, w_pad, rect, sup):
    """matplotlib's _auto_adjust_subplotpars (sup: the suptitle's and
    supxlabel's heights, the supylabel's width, figure fractions)."""
    suptitle_h, supxlabel_h, supylabel_w = sup
    rows, cols = shape
    fw, fh = figsize
    inch = fontsize_pt / 72.0
    pad_in = pad * inch
    vpad_in = h_pad * inch if h_pad is not None else pad_in
    hpad_in = w_pad * inch if w_pad is not None else pad_in
    if rect is None:
        m_left = m_bottom = m_right = m_top = None
    else:
        m_left, m_bottom, right, top = rect
        m_right = 1 - right if right else None
        m_top = 1 - top if top else None
    vspaces = np.zeros((rows + 1, cols))
    hspaces = np.zeros((rows, cols + 1))
    for (_, ax_box, tight), (rs, cs) in zip(groups, spans):
        x0, y0, x1, y1 = ax_box.l, ax_box.b, ax_box.l + ax_box.w, ax_box.b + ax_box.h
        tx0, ty0, tx1, ty1 = tight
        hspaces[rs.start:rs.stop, cs.start] += x0 - tx0
        hspaces[rs.start:rs.stop, cs.stop] += tx1 - x1
        vspaces[rs.start, cs.start:cs.stop] += ty1 - y1
        vspaces[rs.stop, cs.start:cs.stop] += y0 - ty0
    if not m_left:
        m_left = max(hspaces[:, 0].max(), 0) + pad_in / fw
        if supylabel_w:
            m_left += supylabel_w + pad_in / fw
    if not m_right:
        m_right = max(hspaces[:, -1].max(), 0) + pad_in / fw
    if not m_top:
        m_top = max(vspaces[0, :].max(), 0) + pad_in / fh
        if suptitle_h:
            m_top += suptitle_h + pad_in / fh
    if not m_bottom:
        m_bottom = max(vspaces[-1, :].max(), 0) + pad_in / fh
        if supxlabel_h:
            m_bottom += supxlabel_h + pad_in / fh
    if m_left + m_right >= 1 or m_bottom + m_top >= 1:
        return {}                                       # no room: matplotlib warns, leaves it
    kw = dict(left=m_left, right=1 - m_right, bottom=m_bottom, top=1 - m_top)
    if cols > 1:
        hspace = hspaces[:, 1:-1].max() + hpad_in / fw
        h_axes = (1 - m_right - m_left - hspace * (cols - 1)) / cols
        if h_axes < 0:
            return {}
        kw['wspace'] = hspace / h_axes
    if rows > 1:
        vspace = vspaces[1:-1, :].max() + vpad_in / fh
        v_axes = (1 - m_top - m_bottom - vspace * (rows - 1)) / rows
        if v_axes < 0:
            return {}
        kw['hspace'] = vspace / v_axes
    return kw
