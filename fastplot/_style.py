"""The small conversions every artist uses, reading matplotlib's
rcParams (_rc) when it draws: font sizes, colours to bytes, dash patterns,
the subplot parameters."""
import numpy as np

from . import _color
from ._rc import rcParams

_FONT_SCALE = {'xx-small': 0.579, 'x-small': 0.694, 'small': 0.833, 'medium': 1.0,
               'large': 1.2, 'x-large': 1.44, 'xx-large': 1.728, 'larger': 1.2,
               'smaller': 0.833}
_PATTERNS = {'dashed': 'lines.dashed_pattern', '--': 'lines.dashed_pattern',
             'dashdot': 'lines.dashdot_pattern', '-.': 'lines.dashdot_pattern',
             'dotted': 'lines.dotted_pattern', ':': 'lines.dotted_pattern'}
_SOLID = ('solid', '-', 'none', 'None', '', ' ')


def rc(key):
    """The current value of an rcParam."""
    return rcParams[key]


def fontsize_pt(size):
    """Points, from a number or matplotlib's size names (relative to font.size)."""
    if size is None:
        return float(rcParams['font.size'])
    if isinstance(size, str):
        return float(rcParams['font.size']) * _FONT_SCALE[size]
    return float(size)


def subplotpars():
    return {k: rcParams['figure.subplot.' + k]
            for k in ('left', 'right', 'bottom', 'top', 'wspace', 'hspace')}


def cycle_color(k):
    """The k-th colour of axes.prop_cycle."""
    cyc = rcParams['axes.prop_cycle']
    return cyc[k % len(cyc)]


def rgba(c, alpha=None):
    """uint8[4] from any matplotlib colour spec; alpha replaces its alpha."""
    if c is None:
        return None
    if isinstance(c, str) and c.lower() in ('none', 'never'):
        return np.zeros(4, np.uint8)
    if isinstance(c, np.ndarray) and c.dtype == np.uint8 and c.shape == (4,):
        out = c.copy()
    else:
        out = np.asarray(_color.to_rgba_bytes(c, rounding='nearest'), np.uint8).copy()
    if alpha is not None:
        out[3] = np.uint8(min(max(float(alpha), 0.0), 1.0) * 255 + 0.5)
    return out


def _dashes(style, lw_pt):
    """matplotlib's dash pattern for a linestyle, in points (scaled by the
    line width when lines.scale_dashes, as matplotlib does)."""
    if style is None:
        return None
    scale = max(lw_pt, 1.0) if rcParams['lines.scale_dashes'] else 1.0
    if isinstance(style, tuple) and len(style) == 2 and not isinstance(style[0], str):
        offset, seq = style
        return None if seq is None else tuple(v * scale for v in seq)
    if style in _SOLID:
        return None
    if style not in _PATTERNS:
        raise ValueError('unknown linestyle %r' % (style,))
    return tuple(v * scale for v in rcParams[_PATTERNS[style]])


def _ignore(kw, fn, names):
    for n in names:
        kw.pop(n, None)
    if kw:
        raise TypeError('%s: unsupported keyword(s): %s' % (fn, ', '.join(sorted(kw))))
