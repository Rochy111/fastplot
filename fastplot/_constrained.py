"""matplotlib's constrained layout (layout='constrained'), done before each
draw: every subplot grid's rows and columns get margins that hold their
axes' decorations (tick labels, axis labels, titles) and colorbars, and
the axes share what is left by their width and height ratios -- the
solution matplotlib's kiwi solver reaches, worked out directly:

    margins     per column left / leftcb / right / rightcb, per row
                top / topcb / bottom / bottomcb, the largest any axes
                asks for (matplotlib's edit_outer_margin_mins); 'cb'
                margins hold pads and colorbars, the others decorations
    inner       each column (row) the rest of the figure, less every
                margin, shared by width (height) ratios
    colorbars   made by fig.colorbar beside their parents, in the parents'
                cb margins, pad and their own decorations off them

twice, as matplotlib does, since tick labels change with the axes' sizes.
Axes laid out by add_axes (no grid) keep their places. Nested grids are
laid out by their top-level cell; inside it, as their GridSpec says.
"""
import numpy as np

from ._layout import SubplotSpec, _Box, _anchor, topmost
from ._style import rc

PADS = ('w_pad', 'h_pad', 'wspace', 'hspace')


def pads(fig):
    """The figure's constrained-layout pads: w_pad / h_pad inches, wspace
    / hspace figure fractions (rcParams figure.constrained_layout.*)."""
    out = {k: rc('figure.constrained_layout.' + k) for k in PADS}
    out.update({k: v for k, v in getattr(fig, '_cl_pads', {}).items() if v is not None})
    return out


def do_layout(fig, measure):
    """Lay fig out; measure(fig) -> {id(obj): (pos, tight)} for each axes
    and colorbar axes, both (x0, y0, x1, y1) figure fractions, y up --
    pos the axes box, tight that grown by its decorations."""
    p = pads(fig)
    fw, fh = fig.figsize
    w_pad, h_pad = p['w_pad'] / fw, p['h_pad'] / fh
    fig._cl_boxes = {}
    for cbax in _cl_colorbars(fig):
        cbax.spec = cbax._cl['box0']
    prev = {}                                           # the last pass's solution, per grid
    for _ in range(2):
        m = measure(fig)
        prev = _solve(fig, m, w_pad, h_pad, p['wspace'], p['hspace'], prev)


def _cl_colorbars(fig):
    return [cb.ax for cb in fig.colorbars if getattr(cb.ax, '_cl', None) is not None]


def _cell_of(ax):
    spec = ax.spec
    if not isinstance(spec, SubplotSpec):
        return None
    return topmost(spec)


def _solve(fig, m, w_pad, h_pad, wspace, hspace, prev):
    # the figure's own margins: suptitle, supxlabel, supylabel
    fig_m = dict(left=0.0, right=0.0, top=0.0, bottom=0.0)
    for name, side in (('_suptitle', 'top'), ('_supxlabel', 'bottom'), ('_supylabel', 'left')):
        t = getattr(fig, name, None)
        if t is None or t not in fig.texts or not t.visible or id(t) not in m:
            continue
        if getattr(t, '_autopos', False):
            if side == 'top':
                t.y = 1 - h_pad
            elif side == 'bottom':
                t.y = h_pad
            else:
                t.x = w_pad
        x0, y0, x1, y1 = m[id(t)][1]
        size = (y1 - y0) + 2 * h_pad if side != 'left' else (x1 - x0) + 2 * w_pad
        fig_m[side] = max(fig_m[side], size)
    for lg in fig.legends:                              # legend(loc='outside ...')
        out = getattr(lg, '_outside', None)
        if out is None or lg.bbox_to_anchor is not None or id(lg) not in m:
            continue
        x0, y0, x1, y1 = m[id(lg)][1]
        side = {'upper': 'top', 'lower': 'bottom'}.get(out, out)
        size = (y1 - y0) + 2 * h_pad if side in ('top', 'bottom') else (x1 - x0) + 2 * w_pad
        fig_m[side] = max(fig_m[side], size)
    # the grids: each top-level GridSpec its own set of rows and columns
    grids = {}
    cells = {}
    for ax in fig.axes:
        if getattr(ax, '_twin_of', None) is not None or not getattr(ax, 'visible', True):
            continue
        ss = _cell_of(ax)
        if ss is None or id(ax) not in m:
            continue
        g = grids.setdefault(id(ss.gs), dict(gs=ss.gs, margin=_zero(ss.gs)))
        key = (id(ss.gs), ss.r0, ss.r1, ss.c0, ss.c1)
        cells.setdefault(key, (ss, []))[1].append(ax)
    cbs_of = {}                                         # parent axes -> its colorbar axes
    for cbax in _cl_colorbars(fig):
        for par in cbax._cl['parents']:
            cbs_of.setdefault(id(par), []).append(cbax)
    for key, (ss, axs) in cells.items():
        gs = ss.gs
        mg = grids[id(gs)]['margin']
        pos, tight = _box_union([m[id(a)] for a in axs] +
                                [m[id(t)] for a in axs for t in _twins(fig, a) if id(t) in m])
        margin = _from_padding(gs, ss, w_pad, h_pad, wspace, hspace)
        # (negative where an aspect leaves the axes short of its cell: the
        # slack takes up what overhangs, as matplotlib's margins do)
        margin['left'] += pos[0] - tight[0]
        margin['right'] += tight[2] - pos[2]
        margin['bottom'] += pos[1] - tight[1]
        margin['top'] += tight[3] - pos[3]
        for a in axs:
            for cbax in cbs_of.get(id(a), []):
                if id(cbax) not in m:
                    continue
                info = cbax._cl
                rs, cs = _parent_spans(info['parents'])
                if rs is None:
                    continue
                cpos, cb = m[id(cbax)]
                # the pad is a fraction of the parents' cells as the last pass
                # left them: nothing on the first (matplotlib reads the grid
                # before it is solved)
                s0 = prev.get(id(gs))
                inner = (0.0, 0.0, 0.0, 0.0) if s0 is None else _inner(s0, *rs, *cs)
                pad = info['pad'] * (inner[2] - inner[0] if info['location'] in ('left', 'right')
                                     else inner[3] - inner[1])
                loc = info['location']
                if loc == 'right' and cs[1] == ss.c1:
                    margin['rightcb'] += cb[2] - cb[0] + pad
                elif loc == 'left' and cs[0] == ss.c0:
                    margin['leftcb'] += cb[2] - cb[0] + pad
                elif loc == 'top' and rs[0] == ss.r0:
                    margin['topcb'] += cb[3] - cb[1] + pad
                elif loc == 'bottom' and rs[1] == ss.r1:
                    margin['bottomcb'] += cb[3] - cb[1] + pad
                if loc in ('top', 'bottom'):
                    if cs[0] == ss.c0 and cb[0] < tight[0]:
                        margin['left'] += tight[0] - cb[0]
                    if cs[1] == ss.c1 and cb[2] > tight[2]:
                        margin['right'] += cb[2] - tight[2]
                else:
                    if rs[1] == ss.r1 and cb[1] < tight[1]:
                        margin['bottom'] += tight[1] - cb[1]
                    if rs[0] == ss.r0 and cb[3] > tight[3]:
                        margin['top'] += cb[3] - tight[3]
        for k in ('left', 'leftcb'):
            mg[k][ss.c0] = max(mg[k][ss.c0], margin[k])
        for k in ('right', 'rightcb'):
            mg[k][ss.c1] = max(mg[k][ss.c1], margin[k])
        for k in ('top', 'topcb'):
            mg[k][ss.r0] = max(mg[k][ss.r0], margin[k])
        for k in ('bottom', 'bottomcb'):
            mg[k][ss.r1] = max(mg[k][ss.r1], margin[k])
    # solve each grid inside the figure's margins
    area = (fig_m['left'], fig_m['bottom'], 1 - fig_m['right'], 1 - fig_m['top'])
    sol = {}
    for gid, g in grids.items():
        gs, mg = g['gs'], g['margin']
        p = gs.pars
        x0 = area[0] if p.get('left') is None else max(area[0], p['left'])
        x1 = area[2] if p.get('right') is None else min(area[2], p['right'])
        y0 = area[1] if p.get('bottom') is None else max(area[1], p['bottom'])
        y1 = area[3] if p.get('top') is None else min(area[3], p['top'])
        wr = np.asarray(gs.width_ratios or [1] * gs.ncols, float)
        hr = np.asarray(gs.height_ratios or [1] * gs.nrows, float)
        cm = mg['left'] + mg['leftcb'] + mg['right'] + mg['rightcb']
        rm = mg['top'] + mg['topcb'] + mg['bottom'] + mg['bottomcb']
        inner_w = (x1 - x0 - cm.sum()) * wr / wr.sum()
        inner_h = (y1 - y0 - rm.sum()) * hr / hr.sum()
        if (inner_w <= 0).any() or (inner_h <= 0).any():
            continue                                    # collapsed: matplotlib warns, leaves it
        widths = cm + inner_w
        heights = rm + inner_h
        lefts = x0 + np.r_[0.0, np.cumsum(widths)[:-1]]
        tops = y1 - np.r_[0.0, np.cumsum(heights)[:-1]]
        sol[gid] = (lefts, lefts + widths, tops, tops - heights, mg)
    boxes = {}
    for key, (ss, axs) in cells.items():
        s = sol.get(id(ss.gs))
        if s is None:
            continue
        boxes[key] = _inner(s, ss.r0, ss.r1, ss.c0, ss.c1)
    fig._cl_boxes = {k: _Box(b[0], b[1], b[2] - b[0], b[3] - b[1]) for k, b in boxes.items()}
    # colorbars: beside their first parent's cell, in its cb margins
    done = set()
    for key, (ss, axs) in cells.items():
        s = sol.get(id(ss.gs))
        if s is None:
            continue
        offset = dict(left=0.0, right=0.0, bottom=0.0, top=0.0)
        for a in axs:
            for cbax in reversed(cbs_of.get(id(a), [])):
                if id(cbax) in done or cbax._cl['parents'][0] is not a or id(cbax) not in m:
                    continue
                done.add(id(cbax))
                _place_colorbar(cbax, s, m[id(cbax)], offset)
    return sol


def _twins(fig, ax):
    return [a for a in fig.axes if getattr(a, '_twin_of', None) is ax]


def _zero(gs):
    out = {}
    for k in ('left', 'leftcb', 'right', 'rightcb'):
        out[k] = np.zeros(gs.ncols)
    for k in ('top', 'topcb', 'bottom', 'bottomcb'):
        out[k] = np.zeros(gs.nrows)
    return out


def _from_padding(gs, ss, w_pad, h_pad, wspace, hspace):
    """matplotlib's get_margin_from_padding."""
    ws = (gs.wspace if gs.wspace is not None else wspace) / 2
    hs = (gs.hspace if gs.hspace is not None else hspace) / 2
    margin = dict(leftcb=w_pad, rightcb=w_pad, bottomcb=h_pad, topcb=h_pad, left=0.0,
                  right=0.0, top=0.0, bottom=0.0)
    if ws / gs.ncols > w_pad:
        if ss.c0 > 0:
            margin['leftcb'] = ws / gs.ncols
        if ss.c1 < gs.ncols - 1:
            margin['rightcb'] = ws / gs.ncols
    if hs / gs.nrows > h_pad:
        if ss.r1 < gs.nrows - 1:
            margin['bottomcb'] = hs / gs.nrows
        if ss.r0 > 0:
            margin['topcb'] = hs / gs.nrows
    return margin


def _box_union(items):
    pos = np.array([p for p, _ in items], float)
    tight = np.array([t for _, t in items], float)
    return ((pos[:, 0].min(), pos[:, 1].min(), pos[:, 2].max(), pos[:, 3].max()),
            (tight[:, 0].min(), tight[:, 1].min(), tight[:, 2].max(), tight[:, 3].max()))


def _parent_spans(parents):
    """The rows and columns (first, last) the colorbar's parents span."""
    specs = [_cell_of(p) for p in parents]
    if any(s is None for s in specs):
        return None, None
    return ((min(s.r0 for s in specs), max(s.r1 for s in specs)),
            (min(s.c0 for s in specs), max(s.c1 for s in specs)))


def _inner(s, r0, r1, c0, c1):
    """get_inner_bbox: the cells' box less all their margins (x0, y0, x1, y1)."""
    lefts, rights, tops, bottoms, mg = s
    return (lefts[c0] + mg['left'][c0] + mg['leftcb'][c0],
            bottoms[r1] + mg['bottom'][r1] + mg['bottomcb'][r1],
            rights[c1] - mg['right'][c1] - mg['rightcb'][c1],
            tops[r0] - mg['top'][r0] - mg['topcb'][r0])


def _for_cb(s, r0, r1, c0, c1):
    """get_bbox_for_cb: the cells' box less only their cb margins."""
    lefts, rights, tops, bottoms, mg = s
    return (lefts[c0] + mg['leftcb'][c0], bottoms[r1] + mg['bottomcb'][r1],
            rights[c1] - mg['rightcb'][c1], tops[r0] - mg['topcb'][r0])


def _place_colorbar(cbax, s, measured, offset):
    """matplotlib's reposition_colorbar."""
    info = cbax._cl
    (r0, r1), (c0, c1) = _parent_spans(info['parents'])
    parent = _for_cb(s, r0, r1, c0, c1)
    pb = _inner(s, r0, r1, c0, c1)
    loc, fraction, shrink = info['location'], info['fraction'], info['shrink']
    ax_, ay_ = _anchor(info['anchor'])
    cpos, cbb = measured
    pw, ph = pb[2] - pb[0], pb[3] - pb[1]
    if loc in ('left', 'right'):
        w, h = pw * fraction, ph * shrink
        x0 = pb[0] + (pw - w) * ax_
        y0 = pb[1] + (ph - h) * ay_
        pad = info['pad'] * pw
        if loc == 'right':
            lmargin = cpos[0] - cbb[0]
            x0 += parent[2] - x0 + offset['right'] + pad + lmargin
            offset['right'] += cbb[2] - cbb[0] + pad
        else:
            lmargin = cpos[0] - cbb[0]
            x0 += parent[0] - x0 - (cbb[2] - cbb[0]) - pad + lmargin - offset['left']
            offset['left'] += cbb[2] - cbb[0] + pad
    else:
        w, h = pw * shrink, ph * fraction
        x0 = pb[0] + (pw - w) * ax_
        y0 = pb[1] + (ph - h) * ay_
        pad = info['pad'] * ph
        bmargin = cpos[1] - cbb[1]
        if loc == 'top':
            y0 += parent[3] - y0 + offset['top'] + pad + bmargin
            offset['top'] += cbb[3] - cbb[1] + pad
        else:
            y0 += parent[1] - y0 - (cbb[3] - cbb[1]) - pad + bmargin - offset['bottom']
            offset['bottom'] += cbb[3] - cbb[1] + pad
    cbax.spec = _Box(x0, y0, w, h)
