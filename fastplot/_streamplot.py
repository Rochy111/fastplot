"""matplotlib's streamplot, its trajectories computed in numba.

    s = ax.streamplot(x, y, u, v, density=1.5, color=speed, cmap='viridis')
    s.lines, s.arrows

The same algorithm as matplotlib.streamplot, step for step: a mask of
30 x 30 cells per unit density; start points spiralling in from the
edge (or given); each line integrated both ways by the adaptive RK12
in grid coordinates, its step size and error matplotlib's, stopping at
a used mask cell, the grid's edge (an Euler step onto it), zero or
missing speed, or maxlength; lines shorter than minlength given back.
One kernel runs the whole loop (matplotlib runs it in Python), so a
field of streamlines is milliseconds. Lines are a LineCollection (a
segment each when coloured or sized by a field), arrows FancyArrowPatch
'-|>' at the middle of each line.
"""
import numpy as np

from ._jit import njit
from ._plots import LineCollection, _Bounds


class StreamplotSet:
    """streamplot's result: the lines (a LineCollection) and the arrows."""

    def __init__(self, lines, arrows):
        self.lines, self.arrows = lines, arrows


def _axis(v, name, which):
    v = np.asarray(v, float)
    if v.ndim == 2:
        line = v[0] if which == 'x' else v[:, 0]
        if not np.allclose(line[None, :] if which == 'x' else line[:, None], v):
            raise ValueError("The %s of '%s' must be equal" % (
                'rows' if which == 'x' else 'columns', name))
        v = line
    elif v.ndim != 1:
        raise ValueError("'%s' can have at maximum 2 dimensions" % name)
    if not (np.diff(v) > 0).all():
        raise ValueError("'%s' must be strictly increasing" % name)
    if not np.allclose(np.diff(v), (v[-1] - v[0]) / (v.size - 1)):
        raise ValueError("'%s' values must be equally spaced" % name)
    return v


@njit(cache=True, nogil=True)
def _interp(a, xi, yi):
    """matplotlib's interpgrid at a point (NaN: masked)."""
    Ny, Nx = a.shape
    x, y = int(xi), int(yi)
    xn = x if x == Nx - 1 else x + 1
    yn = y if y == Ny - 1 else y + 1
    xt, yt = xi - x, yi - y
    a0 = a[y, x] * (1 - xt) + a[y, xn] * xt
    a1 = a[yn, x] * (1 - xt) + a[yn, xn] * xt
    return a0 * (1 - yt) + a1 * yt


# f's outcomes
_OK, _OUT, _STOP = 0, 1, 2


@njit(cache=True, nogil=True)
def _f(u, v, speed, xi, yi, sign, out):
    """forward_time / backward_time: (dx/ds, dy/ds) into out, or why not."""
    ny, nx = u.shape
    if not (0 <= xi <= nx - 1 and 0 <= yi <= ny - 1):
        return _OUT
    ds_dt = _interp(speed, xi, yi)
    if ds_dt != ds_dt:
        return _STOP
    if ds_dt == 0:
        return _STOP
    dt_ds = 1. / ds_dt
    ui = _interp(u, xi, yi)
    vi = _interp(v, xi, yi)
    if ui != ui or vi != vi:
        return _STOP
    out[0] = sign * (ui * dt_ds)
    out[1] = sign * (vi * dt_ds)
    return _OK


@njit(cache=True, nogil=True)
def _mask_cell(xg, yg, gx2m, gy2m):
    return int(np.rint(xg * gx2m)), int(np.rint(yg * gy2m))


@njit(cache=True, nogil=True)
def _rk12(u, v, speed, x0, y0, sign, maxlength, broken, step_scale, err_scale, mask, state,
          traj, gx2m, gy2m, buf):
    """_integrate_rk12: (stotal, count, buf) -- the points in buf, grown
    as needed. state: [current mask x, current mask y, cells used]."""
    ny, nx = u.shape
    mny, mnx = mask.shape
    maxerror = 0.003 * err_scale
    maxds = min(1. / mnx, 1. / mny, 0.1) * step_scale
    ds = maxds
    stotal = 0.0
    xi, yi = x0, y0
    n = 0
    k1 = np.empty(2)
    k2 = np.empty(2)
    while True:
        if n + 2 > buf.shape[0]:
            grow = np.empty((2 * buf.shape[0], 2))
            grow[:n] = buf[:n]
            buf = grow
        if 0 <= xi <= nx - 1 and 0 <= yi <= ny - 1:
            buf[n, 0] = xi
            buf[n, 1] = yi
            n += 1
            r = _f(u, v, speed, xi, yi, sign, k1)
            if r == _OK:
                r = _f(u, v, speed, xi + ds * k1[0], yi + ds * k1[1], sign, k2)
        else:
            r = _OUT
        if r == _OUT:
            if n:                                       # an Euler step onto the edge
                lx, ly = buf[n - 1, 0], buf[n - 1, 1]
                if _f(u, v, speed, lx, ly, sign, k1) == _OK:
                    cx, cy = k1[0], k1[1]
                    dsx = np.inf if cx == 0 else (lx / -cx if cx < 0 else (nx - 1 - lx) / cx)
                    dsy = np.inf if cy == 0 else (ly / -cy if cy < 0 else (ny - 1 - ly) / cy)
                    d = min(dsx, dsy)
                    buf[n, 0] = lx + cx * d
                    buf[n, 1] = ly + cy * d
                    n += 1
                    stotal += d
            break
        if r == _STOP:
            break
        dx1, dy1 = ds * k1[0], ds * k1[1]
        dx2, dy2 = ds * 0.5 * (k1[0] + k2[0]), ds * 0.5 * (k1[1] + k2[1])
        error = np.hypot((dx2 - dx1) / (nx - 1), (dy2 - dy1) / (ny - 1))
        if error < maxerror:
            xi += dx2
            yi += dy2
            if not (0 <= xi <= nx - 1 and 0 <= yi <= ny - 1):
                break
            xm, ym = _mask_cell(xi, yi, gx2m, gy2m)
            if not (state[0] == xm and state[1] == ym):
                if mask[ym, xm] == 0:
                    traj[state[2], 0] = ym
                    traj[state[2], 1] = xm
                    state[2] += 1
                    mask[ym, xm] = 1
                    state[0], state[1] = xm, ym
                elif broken:
                    break
            if stotal + ds > maxlength:
                break
            stotal += ds
        if error == 0:
            ds = maxds
        else:
            ds = min(maxds, 0.85 * ds * (maxerror / error) ** 0.5)
    return stotal, n, buf


@njit(cache=True, nogil=True)
def _streamlines(u, v, speed, starts, precheck, mask, gx2m, gy2m, minlength, maxlength,
                 backward, forward, broken, step_scale, err_scale):
    """Every trajectory, as matplotlib's loop makes them: (points (n, 2)
    grid coordinates, offsets). precheck: skip a start whose mask cell is
    taken (the spiral's)."""
    pts = np.empty((4096, 2))
    offs = [0]
    npts = 0
    mny, mnx = mask.shape
    traj = np.empty((mnx * mny + 8, 2), np.int64)
    state = np.array([-1, -1, 0], np.int64)              # current_xy None: -1, -1
    side = np.empty((1024, 2))
    for s in range(starts.shape[0]):
        x0, y0 = starts[s, 0], starts[s, 1]
        if precheck and mask[int(np.rint(y0 * gy2m)), int(np.rint(x0 * gx2m))] != 0:
            continue
        # start_trajectory
        state[2] = 0
        xm, ym = _mask_cell(x0, y0, gx2m, gy2m)
        if not (state[0] == xm and state[1] == ym):
            if mask[ym, xm] == 0:
                traj[0, 0], traj[0, 1] = ym, xm
                state[2] = 1
                mask[ym, xm] = 1
                state[0], state[1] = xm, ym
            elif broken:
                continue
        stotal = 0.0
        base = npts
        if backward:
            st, nb, side = _rk12(u, v, speed, x0, y0, -1.0, maxlength, broken, step_scale,
                                 err_scale, mask, state, traj, gx2m, gy2m, side)
            stotal += st
            pts = _room(pts, npts, nb)
            for k in range(nb):                         # reversed
                pts[npts, 0] = side[nb - 1 - k, 0]
                pts[npts, 1] = side[nb - 1 - k, 1]
                npts += 1
        if forward:
            state[0], state[1] = _mask_cell(x0, y0, gx2m, gy2m)   # reset_start_point
            st, nf, side = _rk12(u, v, speed, x0, y0, 1.0, maxlength, broken, step_scale,
                                 err_scale, mask, state, traj, gx2m, gy2m, side)
            stotal += st
            pts = _room(pts, npts, nf)
            for k in range(1, nf):
                pts[npts, 0] = side[k, 0]
                pts[npts, 1] = side[k, 1]
                npts += 1
        if stotal > minlength:
            offs.append(npts)
        else:                                           # too short: undone
            npts = base
            for k in range(state[2]):
                mask[traj[k, 0], traj[k, 1]] = 0
    return pts[:npts].copy(), np.array(offs, np.int64)


@njit(cache=True, nogil=True)
def _room(pts, n, more):
    """pts with room for `more` points after its first n."""
    if n + more <= pts.shape[0]:
        return pts
    grow = np.empty((max(2 * pts.shape[0], n + more), 2))
    grow[:n] = pts[:n]
    return grow


def _spiral(ny, nx):
    """matplotlib's _gen_starting_points: the mask cells from the edge in."""
    out = np.empty((nx * ny, 2), np.int64)
    xfirst, yfirst, xlast, ylast = 0, 1, nx - 1, ny - 1
    x = y = 0
    direction = 0                                       # right, up, left, down
    for i in range(nx * ny):
        out[i] = x, y
        if direction == 0:
            x += 1
            if x >= xlast:
                xlast -= 1
                direction = 1
        elif direction == 1:
            y += 1
            if y >= ylast:
                ylast -= 1
                direction = 2
        elif direction == 2:
            x -= 1
            if x <= xfirst:
                xfirst += 1
                direction = 3
        else:
            y -= 1
            if y <= yfirst:
                yfirst += 1
                direction = 0
    return out


def trajectories(x, y, u, v, density=1, minlength=0.1, maxlength=4.0,
                 integration_direction='both', broken_streamlines=True, start_points=None,
                 integration_max_step_scale=1.0, integration_max_error_scale=1.0):
    """The streamlines in data coordinates: (xy (n, 2), offsets), and the
    grid's (x0, y0, width, height)."""
    if integration_max_step_scale <= 0 or integration_max_error_scale <= 0:
        raise ValueError('integration_max_step_scale and _error_scale must be > 0')
    if integration_direction not in ('both', 'forward', 'backward'):
        raise ValueError("integration_direction must be 'both', 'forward' or 'backward'")
    xs, ys = _axis(x, 'x', 'x'), _axis(y, 'y', 'y')
    gnx, gny = xs.size, ys.size
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    mnx, mny = (30 * np.broadcast_to(density, 2)).astype(int)
    if mnx < 0 or mny < 0:
        raise ValueError("'density' must be positive")
    u = np.ma.filled(np.ma.masked_invalid(np.asarray(u, float)), np.nan)
    v = np.ma.filled(np.ma.masked_invalid(np.asarray(v, float)), np.nan)
    if u.shape != (gny, gnx) or v.shape != (gny, gnx):
        raise ValueError("'u' and 'v' must match the shape of the (x, y) grid")
    xd2g, yd2g = 1. / dx, 1. / dy
    ug, vg = u * xd2g, v * yd2g                         # data -> grid
    speed = np.sqrt((ug / (gnx - 1)) ** 2 + (vg / (gny - 1)) ** 2)
    gx2m, gy2m = (mnx - 1) / (gnx - 1), (mny - 1) / (gny - 1)
    if integration_direction == 'both':
        maxlength /= 2.
    if start_points is None:
        cells = _spiral(mny, mnx)
        starts = np.column_stack([cells[:, 0] * (1. / gx2m), cells[:, 1] * (1. / gy2m)])
    else:
        sp = np.asarray(start_points, float).reshape(-1, 2).copy()
        bad = ~((xs[0] <= sp[:, 0]) & (sp[:, 0] <= xs[-1]) & (ys[0] <= sp[:, 1]) &
                (sp[:, 1] <= ys[-1]))
        if bad.any():
            raise ValueError('Starting point (%r, %r) outside of data boundaries'
                             % tuple(sp[np.argmax(bad)]))
        starts = np.column_stack([np.clip((sp[:, 0] - xs[0]) * xd2g, 0, gnx - 1),
                                  np.clip((sp[:, 1] - ys[0]) * yd2g, 0, gny - 1)])
    mask = np.zeros((mny, mnx))
    pts, offs = _streamlines(ug, vg, speed, np.ascontiguousarray(starts), start_points is None,
                             mask, gx2m, gy2m,
                             minlength, maxlength, integration_direction != 'forward',
                             integration_direction != 'backward', bool(broken_streamlines),
                             float(integration_max_step_scale),
                             float(integration_max_error_scale))
    data = np.column_stack([pts[:, 0] / xd2g + xs[0], pts[:, 1] / yd2g + ys[0]])
    return data, pts, offs, (xs[0], ys[0], xs[-1] - xs[0], ys[-1] - ys[0])


def _interp_field(a, gx, gy):
    """matplotlib's interpgrid at many points (no mask check)."""
    Ny, Nx = a.shape
    x, y = gx.astype(int), gy.astype(int)
    xn, yn = np.clip(x + 1, 0, Nx - 1), np.clip(y + 1, 0, Ny - 1)
    xt, yt = gx - x, gy - y
    a0 = a[y, x] * (1 - xt) + a[y, xn] * xt
    a1 = a[yn, x] * (1 - xt) + a[yn, xn] * xt
    return a0 * (1 - yt) + a1 * yt


def streamplot(ax, x, y, u, v, density=1, linewidth=None, color=None, cmap=None, norm=None,
               arrowsize=1, arrowstyle='-|>', minlength=0.1, transform=None, zorder=None,
               start_points=None, maxlength=4.0, integration_direction='both',
               broken_streamlines=True, integration_max_step_scale=1.0,
               integration_max_error_scale=1.0, num_arrows=1):
    """ax.streamplot, as matplotlib's (see the module docstring)."""
    from . import _color
    from .patches import FancyArrowPatch
    from ._style import rc
    if num_arrows < 0:
        raise ValueError('num_arrows must be >= 0')
    zorder = 2.0 if zorder is None else zorder
    if color is None:
        color = ax._next_color('lines')
    lw = rc('lines.linewidth') if linewidth is None else linewidth
    data, grid_pts, offs, (x0, y0, w, h) = trajectories(
        x, y, u, v, density, minlength, maxlength, integration_direction, broken_streamlines,
        start_points, integration_max_step_scale, integration_max_error_scale)
    multicolor = isinstance(color, np.ndarray)
    field_lw = isinstance(lw, np.ndarray)
    shape = np.shape(u)
    if multicolor and color.shape != shape:
        raise ValueError("If 'color' is given, it must match the shape of the (x, y) grid")
    if field_lw and lw.shape != shape:
        raise ValueError("If 'linewidth' is given, it must match the shape of the (x, y) grid")
    src = ax._src(transform)
    lines, seg_vals, seg_lw, arrows = [], [], [], []
    cvals = np.ma.filled(np.ma.masked_invalid(np.asarray(color, float)), np.nan) \
        if multicolor else None
    if multicolor:
        cnorm = _color.make_norm(norm, vmin=np.nanmin(cvals), vmax=np.nanmax(cvals),
                                 data=cvals, N=_color.get_cmap(cmap).N)
        cm = _color.get_cmap(cmap)
    for k in range(offs.size - 1):
        a, b = offs[k], offs[k + 1]
        tx, ty = data[a:b, 0], data[a:b, 1]
        tg = grid_pts[a:b]
        if multicolor or field_lw:
            pts = np.column_stack([tx, ty])
            lines.extend(np.stack([pts[:-1], pts[1:]], 1))
        else:
            lines.append(np.column_stack([tx, ty]))
        s = np.cumsum(np.hypot(np.diff(tx), np.diff(ty)))
        if field_lw:
            lws = _interp_field(np.asarray(lw, float), tg[:, 0], tg[:, 1])[:-1]
            seg_lw.append(lws)
        if multicolor:
            vals = _interp_field(cvals, tg[:, 0], tg[:, 1])[:-1]
            seg_vals.append(vals)
        for j in range(1, num_arrows + 1):
            if not s.size:
                break
            idx = np.searchsorted(s, s[-1] * (j / (num_arrows + 1)))
            tail = (tx[idx], ty[idx])
            head = (np.mean(tx[idx:idx + 2]), np.mean(ty[idx:idx + 2]))
            arrows.append((tail, head, float(lws[idx]) if field_lw else lw,
                           vals[idx] if multicolor else None))
    if multicolor and arrows:                           # the arrows' colours in one call
        out = np.empty((1, len(arrows), 4), np.uint8)
        _color.shade(np.array([[a[3] for a in arrows]], np.float32), cm, cnorm, out=out,
                     rounding='nearest')
        acols = list(out[0])
    else:
        acols = [color] * len(arrows)
    arrows = [FancyArrowPatch(tail, head, arrowstyle=arrowstyle, mutation_scale=10 * arrowsize,
                              color=acol, linewidth=alw, zorder=zorder, transform=transform)
              for (tail, head, alw, _), acol in zip(arrows, acols)]
    if multicolor:
        vals = np.concatenate(seg_vals) if seg_vals else np.empty(0)
        out = np.empty((1, vals.size, 4), np.uint8)
        _color.shade(vals[None, :].astype(np.float32), cm, cnorm, out=out, rounding='nearest')
        colors = out[0]
    else:
        colors = color
    widths = np.concatenate(seg_lw) if field_lw and seg_lw else lw
    bounds = _Bounds(data[:, 0], data[:, 1], (x0, x0 + w), (y0, y0 + h))
    lc = LineCollection(ax, lines, src, colors if len(lines) or not multicolor else 'k',
                        widths, zorder=zorder, bounds=bounds)
    if multicolor:                                      # a colorbar's mappable, as matplotlib's
        lc.cmap, lc.norm = cm, cnorm
        lc.norm_spec = lambda: cnorm
        lc.values = vals
    ax._add(lc)
    for p in arrows:                                    # drawn together, kept as matplotlib's
        p.axes, p.src = ax, src
    from .patches import ArrowBatch
    ax._add(ArrowBatch(ax, arrows, zorder))
    return StreamplotSet(lc, arrows)
