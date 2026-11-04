"""Axes that draw only their artists (hodograph insets): their small lines
put off and drawn together (_strokes), the lazy parts of Axis, plot()'s
and _place's short cuts -- each against the long way round, to the bit."""
import numpy as np
import pytest

import fastplot
import fastplot.patches as mpatches
import fastplot.patheffects as pe
from fastplot import _agg, _cartesian, _draw, _rc, _strokes, _ticker


def _bare(ax):
    ax.patch.set_alpha(0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def _inset(fig, x, y, k, rng, size=0.07, extra=None):
    ax = fig.add_axes([x, y, size, size])
    _bare(ax)
    ax.plot([-40, 40], [0, 0], color='#666666', linestyle='-', linewidth=0.5, alpha=0.8, zorder=1)
    ax.plot([0, 0], [-40, 40], color='#666666', linestyle='-', linewidth=0.5, alpha=0.8, zorder=1)
    for r in (20, 40):
        ax.add_artist(mpatches.Circle((0, 0), r, fill=False, color='#666666', linestyle='-',
                                      alpha=0.8, linewidth=0.5, zorder=2))
    ax.set_clip_on(False)
    u = np.cumsum(rng.normal(2, 4, 41)) - 20
    v = np.cumsum(rng.normal(1, 4, 41)) - 10
    for i, c in enumerate(['#D77A61', '#D8B4A0', '#DBD3D8', '#EFF1F3', '#223843']):
        ax.plot(list(u[i * 8:i * 8 + 9]), list(v[i * 8:i * 8 + 9]), color=c, linewidth=1.8,
                solid_joinstyle='round', solid_capstyle='round', zorder=20 + i, clip_on=False,
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white', alpha=0.7)])
    if extra is not None:
        extra(ax, k)
    ax.set_aspect('equal')
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    return ax


def _way(m, way):
    """Strokes.flush's choice made: 0 one thread, 1 the runs on a team of 3."""
    real = _agg.CORES.pick
    m.setattr(_agg.CORES, 'pick', lambda key, n: (way, None) if key == 'strokes' else real(key, n))
    m.setattr(_strokes._jit, 'team', lambda: 3)


def _both(make, monkeypatch, way=None):
    """make()'s figure drawn as it is (the flush's way chosen, if way),
    then with every axes drawn the long way (_draw's own ticks, spines
    ... and each line its own call)."""
    with monkeypatch.context() as m:
        if way is not None:
            _way(m, way)
        got = np.asarray(make().draw()).copy()
    with monkeypatch.context() as m:
        m.setattr(_cartesian.Axes, '_draws_artists_only', lambda self: False)
        want = np.asarray(make().draw()).copy()
    return got, want


def _uses_strokes(make, monkeypatch):
    calls = []
    real = _strokes.Strokes.flush

    def flush(self, r):
        calls.append(len(self.groups))
        return real(self, r)
    with monkeypatch.context() as m:
        m.setattr(_strokes.Strokes, 'flush', flush)
        make().draw()
    return calls


def _extra_cases(ax, k):
    """What else an inset may hold: each goes its own way (the steps, the
    renderer), in its turn."""
    kind = k % 9
    if kind == 0:                                       # a NaN: that line to the steps
        ax.plot([-30, -10, np.nan, 10, 30], [5, 15, 0, 15, 5], color='C1', lw=2, zorder=21.5)
    elif kind == 1:                                     # off the canvas, unclipped
        ax.plot([0, 9000], [0, 30], color='C2', lw=1.5, clip_on=False, zorder=22.5)
    elif kind == 2:                                     # dashed: the renderer's own way
        ax.plot([-35, 35], [-30, 30], 'k--', lw=1, zorder=23.5)
    elif kind == 3:                                     # markers
        ax.plot([-20, 0, 20], [10, -10, 10], 'o-', color='C3', ms=3, zorder=20.5)
    elif kind == 4:                                     # a filled patch, a hatched one
        ax.add_patch(mpatches.Rectangle((-30, -30), 20, 15, facecolor='C4', alpha=0.5,
                                        zorder=24.5))
        ax.add_patch(mpatches.Rectangle((5, 5), 20, 15, fill=False, hatch='//', zorder=24.6))
    elif kind == 5:                                     # an outline rectangle, an ellipse
        ax.add_patch(mpatches.Rectangle((-25, -15), 30, 20, fill=False, ec='C5', lw=1.3))
        ax.add_patch(mpatches.Ellipse((10, -5), 30, 12, angle=30, fill=False, ec='C6', lw=0.8))
        ax.add_patch(mpatches.Ellipse((-5, 10), 24, 9, fill=False, ec='C7', lw=2.2))
    elif kind == 6:                                     # text between the lines
        ax.text(0, 0, 'H', fontsize=9, zorder=21.2)
    elif kind == 7:                                     # many points (CORES' to pick), a step
        t = np.linspace(-38, 38, 300)
        ax.plot(t, 30 * np.sin(t / 5), color='C8', lw=0.9, zorder=22.2)
        ax.step([-30, -10, 10, 30], [-20, 0, -5, 15], color='C9', zorder=22.3)
    else:                                               # strokes alone, offsets, odd widths
        ax.plot([-30, 30], [-25, -25], color='C0', lw=1.0, zorder=25,
                path_effects=[pe.Stroke(linewidth=4, foreground='y'), pe.Normal()])
        ax.plot([-30, 30], [25, 25], color='C1', lw=1.0, zorder=25,
                path_effects=[pe.Stroke(linewidth=3, foreground='g', offset=(1, -1)),
                              pe.Normal()])
        ax.plot([-10, -10], [-30, 30], color='C2', lw=2.0, solid_capstyle='butt')
        ax.plot([12, 12], [-30, 30], color='C3', lw=3.0, solid_capstyle='projecting',
                solid_joinstyle='miter', linestyle='solid')
        ax.plot([5], [5], color='C4')                   # one point
        ax.plot([], [], color='C5')
        ax.plot([-5, 5], [0, 0], color='C6', linestyle='None')
        ax.plot([-5, 5], [3, 3], color='C7', lw=0)


@pytest.mark.parametrize('way', [None, 0, 1])
@pytest.mark.parametrize('extra', [None, _extra_cases])
def test_insets_draw_as_the_long_way(extra, way, monkeypatch):
    def make():
        rng = np.random.default_rng(3)
        fig = fastplot.figure(figsize=(15, 9), dpi=125)
        for k in range(40):                             # (overlapping, some at the edges)
            x, y = -0.02 + 1.0 * ((k * 0.137) % 1), -0.02 + 1.0 * ((k * 0.311) % 1)
            _inset(fig, x, y, k, rng, extra=extra)
        return fig
    got, want = _both(make, monkeypatch, way)
    assert np.array_equal(got, want)
    assert sum(_uses_strokes(make, monkeypatch)) > 40 * 10


@pytest.mark.parametrize('way', [0, 1])
def test_big_insets_and_dpi(way, monkeypatch):
    def make():
        rng = np.random.default_rng(5)
        fig = fastplot.figure(figsize=(4, 3), dpi=211)
        for k in range(4):
            _inset(fig, 0.1 + 0.2 * k, 0.15 + 0.1 * k, k, rng, size=0.45, extra=_extra_cases)
        return fig
    got, want = _both(make, monkeypatch, way)
    assert np.array_equal(got, want)


@pytest.mark.parametrize('way', [0, 1])
def test_runs_of_bare_axes_among_others(way, monkeypatch):
    """Bare axes drawn together where the figure draws them one after
    another -- not across another zorder, other axes, an invisible one,
    a visible background; colorbars, texts and legends after them."""
    def make():
        rng = np.random.default_rng(8)
        fig = fastplot.figure(figsize=(8, 6), dpi=100)
        main = fig.add_axes([0.05, 0.05, 0.9, 0.9])
        main.plot([0, 1], [0, 1], lw=3)
        for k in range(30):
            ax = _inset(fig, 0.05 + 0.8 * ((k * 0.37) % 1), 0.05 + 0.8 * ((k * 0.61) % 1), k, rng,
                        size=0.12)
            if k == 7:
                ax.set_zorder(2)                        # (drawn after the rest)
            if k == 11:
                ax.set_visible(False)
            if k == 13:
                ax.patch.set_alpha(0.5)                 # (a background to paint, in its turn)
            if k == 17:
                ax.set_xticks([0])                      # (ticks: drawn its own way)
            if k == 19:
                ax.set_zorder(-1)                       # (drawn before the main axes)
        fig.text(0.5, 0.5, 'over', fontsize=30, color='r')
        cs = main.contourf(np.add.outer(np.arange(5.0), np.arange(6.0)), alpha=0.3)
        fig.colorbar(cs, ax=main)
        return fig
    got, want = _both(make, monkeypatch, way)
    assert np.array_equal(got, want)
    fig = make()
    fig.draw()
    fig.draw()                                          # (marks of the first draw: none left)
    assert not [ax for ax in fig.axes if getattr(ax, '_drawn_in', None) is not None]


def test_only_bare_axes_go_the_short_way():
    fig = fastplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.5, 0.5])
    assert not ax._draws_artists_only()                 # ticks, spines
    _bare(ax)
    assert ax._draws_artists_only()
    ax.set_title('t')
    assert not ax._draws_artists_only()
    ax.set_title('')
    ax.set_xlabel('x')
    assert not ax._draws_artists_only()
    ax.set_xlabel('')
    ax.spines['left'].set_visible(True)
    assert not ax._draws_artists_only()
    ax.set_frame_on(False)
    assert ax._draws_artists_only()
    ax.minorticks_on()                                  # (AutoMinorLocator: ticks)
    assert not ax._draws_artists_only()
    ax.xaxis.set_minor_locator(_ticker.FixedLocator([]))
    assert not ax._draws_artists_only()
    ax.yaxis.set_visible(False)
    assert ax._draws_artists_only()
    ax.legend()
    assert not ax._draws_artists_only()


@pytest.mark.parametrize('way', [0, 1])
def test_strokes_kernel_as_small_lines_one_by_one(way, monkeypatch):
    """Random small lines, several clip boxes, snapped or not, lines and
    rings, edges of fills: _strokes as each small_lines call, in order."""
    _way(monkeypatch, way)
    rng = np.random.default_rng(11)
    H, W = 300, 400
    a = np.zeros((H, W, 4), np.uint8)
    a[..., 3] = 255
    a[..., :3] = rng.integers(0, 255, (H, W, 3))
    want = a.copy()
    got = a.copy()
    canvas = type('C', (), {'a': got, 'device': 'cpu', 'shape': got.shape})()
    s = _strokes.Strokes(canvas)
    boxes = [None, (20, 30, 380, 260), (100.4, 50.6, 220.2, 199.9), (0, 0, 10, 10)]
    calls = []
    for k in range(400):
        n = int(rng.integers(2, 40)) if k % 13 else int(rng.integers(64, 128))  # (CORES' sizes)
        xy = np.cumsum(rng.normal(0, 8, (n, 2)), 0) + rng.uniform(0, [W, H])
        if k % 7 == 0:
            xy = np.round(xy)                           # rectilinear-ish
            xy[1::2, 0] = xy[0::2, 0][:xy[1::2].shape[0]]
        m = int(rng.integers(1, 4))
        cuts = np.sort(rng.choice(np.arange(1, n), size=min(m - 1, n - 2), replace=False))
        offs = np.r_[0, cuts, n].astype(np.int64)
        if k % 11 == 0:                                 # a closed ring
            xy[-1] = xy[0]
        if k % 37 == 0:
            xy[n // 2] = np.nan                         # the steps
        if k % 41 == 0:
            xy[0] = (-50, 20)                           # off the canvas
        args = (xy, offs, rng.integers(0, 256, 4).astype(np.uint8),
                float(rng.choice([0.5, 0.86, 1.0, 2.0, 3.125, 4.34])), boxes[k % 4],
                [False, None, True][k % 3], [_agg.BUTT, _agg.PROJECTING, _agg.ROUND][k % 3],
                [_agg.JOIN_MITER, _agg.JOIN_ROUND, _agg.JOIN_BEVEL][(k // 3) % 3],
                bool(k % 2), bool(k % 5 == 0))
        calls.append(args)
    from fastplot._figure import _Renderer
    from fastplot._render import CpuCanvas

    def renderer(a):
        c = CpuCanvas.__new__(CpuCanvas)
        c.H, c.W, c.a = H, W, a
        return _Renderer(None, c, 100.0, W, H)
    r = renderer(want)                                  # each line its own call
    for xy, offs, col, w, clip, snap, cap, join, line, filled in calls:
        r.lines(xy, offs, col, w, clip, None, snap, cap, False, join, line, filled)
    for xy, offs, col, w, clip, snap, cap, join, line, filled in calls:
        assert s.lines(xy, offs, col, w, clip, snap, cap, join, line, filled)
    s.flush(renderer(got))
    assert np.array_equal(got, want)
    assert not s.groups and s.n == 0


def test_axis_parts_made_later_as_made_then():
    """Ticks, locators and formatters made when first wanted, from the
    settings when the axis was made."""
    def parts(ax):
        return [ax.xaxis.ticks, ax.yaxis.ticks, vars(ax.xaxis.major_formatter),
                vars(ax.yaxis.major_formatter), type(ax.xaxis.major_locator),
                vars(ax.xaxis.major_locator), type(ax.xaxis.minor_locator),
                type(ax.yaxis.minor_formatter), ax.xaxis._minor_visible, ax.spines['top'].visible]
    settings = {'axes.formatter.useoffset': False, 'axes.formatter.limits': (-3, 3),
                'xtick.major.size': 7.5, 'ytick.direction': 'in', 'xtick.minor.visible': True,
                'axes.grid': True, 'axes.spines.top': False}
    with _rc.rc_context(settings):
        fig = fastplot.figure()
        ax = fig.add_subplot()                          # (its parts not asked for in here)
        ref = fastplot.figure().add_subplot()
        want = parts(ref)                               # (asked for at once)
    got = parts(ax)
    assert got == want
    assert ax.xaxis.ticks['major']['size'] == 7.5 and ax.yaxis.ticks['major']['direction'] == 'in'
    plain = fastplot.figure().add_subplot()
    assert parts(plain) != want
    ax2 = fig.add_subplot(212)
    ax2.set_xticks([1, 2])                              # one set, the rest made after
    assert type(ax2.xaxis.major_locator) is _ticker.FixedLocator
    assert type(ax2.xaxis.major_formatter) is type(ref.xaxis.major_formatter)
    ax2.sharex(ax)
    assert ax2.xaxis.major_formatter is ax.xaxis.major_formatter


def test_plot_short_cut_makes_the_same_line():
    fig = fastplot.figure()
    ax = fig.add_subplot()
    cases = [([1, 2, 3], [3, 1, 2]), (np.arange(4.0), np.arange(4)), ([1, 2.5], (3, 4)),
             (np.float32([1, 2]), [True, False]), ([-40, 40], [0, 0])]
    for x, y in cases:
        for kw in ({}, dict(color='r', lw=2, alpha=0.5, zorder=3),
                   dict(solid_capstyle='round', solid_joinstyle='bevel', clip_on=False,
                        path_effects=[pe.withStroke(linewidth=2)])):
            a = ax.plot(x, y, label='a', **kw)[0]
            b = ax.plot(x, y, label=['a'], **kw)[0]     # (a list of labels: the long way)
            va, vb = dict(vars(a)), dict(vars(b))
            for k in ('x', 'y'):
                assert va.pop(k).tobytes() == vb.pop(k).tobytes()
            if 'color' not in kw:                       # (each takes the next colour)
                va.pop('color'), vb.pop('color'), va.pop('mfc'), vb.pop('mfc'), \
                    va.pop('mec'), vb.pop('mec')
            assert va == vb


def test_place_short_cut_numbers():
    fig = fastplot.figure(figsize=(15, 9), dpi=125)
    ax = fig.add_axes([0.123, 0.456, 0.07, 0.07])
    ax.set_aspect('equal')
    ax.set_xlim(-40, 40)
    ax.set_ylim(-33.3, 41.7)
    W, H = fig._size_f(fig.dpi)
    ax._place(fig, W, H)
    got = (ax._xf, ax.box, ax.box_f, ax._pos)
    tx = ax.xaxis.scale.transform(np.array(ax._lims['x'], float))
    ty = ax.yaxis.scale.transform(np.array(ax._lims['y'], float))
    want_xf = (tx[0], 1.0 / (tx[1] - tx[0]), ty[0], 1.0 / (ty[1] - ty[0]))
    assert [float(v) for v in got[0]] == [float(v) for v in want_xf]
    assert got[1] == tuple(int(np.floor(v + 0.5)) for v in got[2])
