"""Coordinate systems, named as matplotlib's transforms:

    ax.transData      data coordinates (on a map: the map's projection)
    ax.transAxes      0..1 across the axes, (0, 0) its lower left
    fig.transFigure   0..1 across the figure
    blended_transform_factory(ax.transData, ax.transAxes)   x in one, y in another
    offset_copy(ax.transData, fig, x=5, y=5, units='points')   shifted on the page
    IdentityTransform()   canvas pixels, y up (as matplotlib's display coordinates)

They are names, not matrices: what a point in one of them lands on is
known only once the figure is laid out, so `to_px(x, y)` is asked at draw
time, of arrays, and answers canvas pixels (x right, y down).
"""
import numpy as np


class Transform:
    """A coordinate system; to_px(x, y) -> canvas pixels, once laid out."""

    def __add__(self, other):
        if isinstance(other, _Shift):
            return _Offset(self, other.dx, other.dy, other.units, other.fig)
        return NotImplemented

    def to_px(self, x, y):
        raise NotImplementedError


class TransData(Transform):
    def __init__(self, ax):
        self.ax = ax

    def to_px(self, x, y):
        return self.ax._data_px(x, y)

    def __repr__(self):
        return '<transData>'


class TransAxes(Transform):
    def __init__(self, ax):
        self.ax = ax

    def to_px(self, x, y):
        x0, y0, x1, y1 = self.ax.box_f
        return x0 + np.asarray(x, float) * (x1 - x0), y1 - np.asarray(y, float) * (y1 - y0)

    def __repr__(self):
        return '<transAxes>'


class TransFigure(Transform):
    def __init__(self, fig):
        self.fig = fig

    def to_px(self, x, y):
        W, H = self.fig._px_size
        return np.asarray(x, float) * W, (1.0 - np.asarray(y, float)) * H

    def __repr__(self):
        return '<transFigure>'


class IdentityTransform(Transform):
    """Display pixels, y up from the bottom (matplotlib's display space)."""

    fig = None

    def to_px(self, x, y):
        H = self.fig._px_size[1] if self.fig is not None else 0.0
        return np.asarray(x, float), H - np.asarray(y, float)

    def __repr__(self):
        return '<IdentityTransform>'


class _Blended(Transform):
    def __init__(self, tx, ty):
        self.tx, self.ty = tx, ty

    def to_px(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        py = self.ty.to_px(np.zeros_like(y), y)[1]
        if isinstance(self.tx, TransData):              # data x: at the bottom (any scale),
            ax = self.tx.ax                             # then skewed with the height, as
            px = ax._data_px(x, np.full_like(x, ax._lims['y'][0]))[0]   # MetPy's xaxis transform
            if getattr(ax, '_skew', 0.0):
                px = px + ax._skew * (ax.box_f[3] - py)
            return px, py
        return self.tx.to_px(x, np.zeros_like(x))[0], py

    def __repr__(self):
        return '<blended %r, %r>' % (self.tx, self.ty)


class _Shift:
    """ScaledTranslation(dx, dy, fig.dpi_scale_trans): added to a transform."""

    def __init__(self, dx, dy, units, fig):
        self.dx, self.dy, self.units, self.fig = dx, dy, units, fig


class _Offset(Transform):
    def __init__(self, base, dx, dy, units, fig):
        self.base, self.dx, self.dy, self.units, self.fig = base, dx, dy, units, fig

    def to_px(self, x, y):
        px, py = self.base.to_px(x, y)
        k = {'points': 1.0 / 72.0, 'inches': 1.0, 'dots': None}[self.units]
        dpi = self.fig._draw_dpi
        sx = self.dx if k is None else self.dx * k * dpi
        sy = self.dy if k is None else self.dy * k * dpi
        return px + sx, py - sy                         # y up on the page, down in pixels


def blended_transform_factory(x_transform, y_transform):
    """x in one coordinate system, y in another (an axvline's: data x, axes y)."""
    return _Blended(x_transform, y_transform)


def offset_copy(trans, fig=None, x=0.0, y=0.0, units='inches'):
    """trans shifted by (x, y) on the page, in 'inches', 'points' or 'dots'."""
    if units not in ('inches', 'points', 'dots'):
        raise ValueError("units must be 'inches', 'points' or 'dots'")
    return _Offset(trans, x, y, units, fig if fig is not None else getattr(trans, 'ax', None)
                   and trans.ax.figure)


def ScaledTranslation(xt, yt, scale_trans):              # noqa: N802 -- matplotlib's name
    """matplotlib's ScaledTranslation(dx, dy, fig.dpi_scale_trans): inches."""
    return _Shift(xt, yt, 'inches', getattr(scale_trans, 'fig', None))


class DpiScaleTrans:
    """fig.dpi_scale_trans: inches on the page."""

    def __init__(self, fig):
        self.fig = fig
