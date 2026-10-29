"""matplotlib's rcParams: the defaults every artist reads when it is
drawn, settable as matplotlib's are.

    fastplot.rcParams['font.size'] = 12
    with fastplot.rc_context({'axes.grid': True}): ...
    fastplot.rc('lines', linewidth=2)
    fastplot.style.use('ggplot')          # matplotlib's style sheets

The keys are matplotlib's and so are their defaults (matplotlib 3.11's
rcParamsDefault, for the keys fastplot reads -- tests/test_rc.py holds
them to it). Values are read at draw time, so a figure drawn inside an
rc_context takes its settings, as in matplotlib.
"""
import contextlib
import copy
import os

DEFAULTS = {
    'font.size': 10.0,
    'font.family': ['sans-serif'],
    'font.weight': 'normal',
    'text.color': 'black',
    'lines.linewidth': 1.5,
    'lines.linestyle': '-',
    'lines.color': 'C0',
    'lines.marker': 'None',
    'lines.markersize': 6.0,
    'lines.markeredgewidth': 1.0,
    'lines.dashed_pattern': [3.7, 1.6],
    'lines.dashdot_pattern': [6.4, 1.6, 1.0, 1.6],
    'lines.dotted_pattern': [1.0, 1.65],
    'lines.scale_dashes': True,
    'polaraxes.grid': True,
    'hatch.color': 'edge',
    'hatch.linewidth': 1.0,
    'path.simplify': True,
    'path.simplify_threshold': 0.111111111111,
    'patch.linewidth': 1.0,
    'patch.edgecolor': 'black',
    'patch.facecolor': 'C0',
    'patch.force_edgecolor': False,
    'axes.linewidth': 0.8,
    'axes.edgecolor': 'black',
    'axes.facecolor': 'white',
    'axes.labelpad': 4.0,
    'axes.titlepad': 6.0,
    'axes.titlesize': 'large',
    'axes.labelsize': 'medium',
    'axes.titleweight': 'normal',
    'axes.labelweight': 'normal',
    'axes.labelcolor': 'black',
    'axes.titlecolor': 'auto',
    'axes.titlelocation': 'center',
    'axes.xmargin': 0.05,
    'axes.ymargin': 0.05,
    'axes.axisbelow': 'line',
    'axes.grid': False,
    'axes.grid.axis': 'both',
    'axes.grid.which': 'major',
    'axes.prop_cycle': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    'axes.unicode_minus': True,
    'axes.formatter.limits': [-5, 6],
    'axes.formatter.useoffset': True,
    'axes.formatter.offset_threshold': 4,
    'axes.formatter.use_mathtext': False,
    'axes.spines.left': True,
    'axes.spines.right': True,
    'axes.spines.top': True,
    'axes.spines.bottom': True,
    'figure.figsize': [6.4, 4.8],
    'figure.dpi': 100.0,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    'figure.titlesize': 'large',
    'figure.titleweight': 'normal',
    'figure.subplot.left': 0.125,
    'figure.subplot.right': 0.9,
    'figure.subplot.bottom': 0.11,
    'figure.subplot.top': 0.88,
    'figure.subplot.wspace': 0.2,
    'figure.subplot.hspace': 0.2,
    'savefig.dpi': 'figure',
    'savefig.facecolor': 'auto',
    'savefig.bbox': None,
    'savefig.pad_inches': 0.1,
    'savefig.transparent': False,
    'savefig.format': 'png',
    'grid.color': '#b0b0b0',
    'grid.linestyle': '-',
    'grid.linewidth': 0.8,
    'grid.alpha': 1.0,
    'image.cmap': 'viridis',
    'image.origin': 'upper',
    'image.interpolation': 'auto',
    'image.aspect': 'equal',
    'legend.loc': 'best',
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.facecolor': 'inherit',
    'legend.edgecolor': '0.8',
    'legend.fancybox': True,
    'legend.shadow': False,
    'legend.numpoints': 1,
    'legend.scatterpoints': 1,
    'legend.markerscale': 1.0,
    'legend.fontsize': 'medium',
    'legend.title_fontsize': None,
    'legend.borderpad': 0.4,
    'legend.labelspacing': 0.5,
    'legend.handlelength': 2.0,
    'legend.handleheight': 0.7,
    'legend.handletextpad': 0.8,
    'legend.borderaxespad': 0.5,
    'legend.columnspacing': 2.0,
    'legend.labelcolor': 'None',
    'hist.bins': 10,
    'errorbar.capsize': 0.0,
    'contour.negative_linestyle': 'dashed',
    'contour.linewidth': None,
    'scatter.marker': 'o',
    'scatter.edgecolors': 'face',
}
for _ax in ('x', 'y'):
    DEFAULTS.update({
        _ax + 'tick.major.size': 3.5, _ax + 'tick.minor.size': 2.0,
        _ax + 'tick.major.width': 0.8, _ax + 'tick.minor.width': 0.6,
        _ax + 'tick.major.pad': 3.5, _ax + 'tick.minor.pad': 3.4,
        _ax + 'tick.color': 'black', _ax + 'tick.labelcolor': 'inherit',
        _ax + 'tick.labelsize': 'medium', _ax + 'tick.direction': 'out',
        _ax + 'tick.minor.visible': False,
    })
DEFAULTS.update({'xtick.top': False, 'xtick.bottom': True, 'xtick.labeltop': False,
                 'xtick.labelbottom': True, 'ytick.left': True, 'ytick.right': False,
                 'ytick.labelleft': True, 'ytick.labelright': False})

_ALIASES = {'lw': 'linewidth', 'ls': 'linestyle', 'c': 'color', 'fc': 'facecolor',
            'ec': 'edgecolor', 'mew': 'markeredgewidth', 'aa': 'antialiased'}


def _parse(key, value):
    """A value as matplotlib's rc validators would read it from a style file."""
    if not isinstance(value, str):
        return value
    v = value.strip().strip('"').strip("'")
    if key == 'axes.prop_cycle':                        # cycler('color', [...])
        import re
        cols = re.findall(r"'([^']*)'|\"([^\"]*)\"", v)
        return [a or b for a, b in cols] if cols else DEFAULTS[key]
    old = DEFAULTS.get(key)
    if isinstance(old, bool):
        return v.lower() in ('true', 'yes', 'on', '1')
    if isinstance(old, (int, float)) and not isinstance(old, bool):
        try:
            return type(old)(float(v)) if isinstance(old, float) else int(float(v))
        except ValueError:
            return v
    if isinstance(old, list) and ',' in v:
        parts = [p.strip() for p in v.split(',')]
        try:
            return [float(p) for p in parts]
        except ValueError:
            return parts
    if v == 'None':
        return None
    if len(v) in (6, 8) and all(ch in '0123456789abcdefABCDEF' for ch in v):
        return '#' + v                                  # style files write colours bare
    return v


class RcParams(dict):
    """matplotlib's rcParams (a dict; unknown keys are kept, unused)."""

    def __setitem__(self, key, value):
        super().__setitem__(key, _parse(key, value) if isinstance(value, str) else value)

    def update(self, *args, **kw):
        for k, v in dict(*args, **kw).items():
            self[k] = v

    def copy(self):
        return RcParams(dict.copy(self))


rcParams = RcParams(copy.deepcopy(DEFAULTS))
rcParamsDefault = RcParams(copy.deepcopy(DEFAULTS))


def rc(group, **kw):
    """matplotlib's rc('lines', linewidth=2, color='r')."""
    for g in ((group,) if isinstance(group, str) else group):
        for k, v in kw.items():
            rcParams['%s.%s' % (g, _ALIASES.get(k, k))] = v


def rcdefaults():
    dict.clear(rcParams)
    dict.update(rcParams, copy.deepcopy(DEFAULTS))


@contextlib.contextmanager
def rc_context(rc=None, fname=None):
    """Settings for the length of a with block."""
    saved = dict(rcParams)
    try:
        if fname is not None:
            rcParams.update(_read_style(fname))
        if rc:
            rcParams.update(rc)
        yield rcParams
    finally:
        dict.clear(rcParams)
        dict.update(rcParams, saved)


# -- style sheets

STYLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'stylelib')


def _read_style(path):
    out = {}
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.split('#', 1)[0].strip() if not line.lstrip().startswith(
                "axes.prop_cycle") else line.strip()
            if not line or ':' not in line:
                continue
            k, v = line.split(':', 1)
            out[k.strip()] = v.strip()
    return out


def available():
    """The bundled style names (matplotlib's)."""
    if not os.path.isdir(STYLE_DIR):
        return []
    return sorted(f[:-9] for f in os.listdir(STYLE_DIR) if f.endswith('.mplstyle'))


def _style_dict(style):
    if isinstance(style, dict):
        return style
    if style in ('default', 'classic_default'):
        return copy.deepcopy(DEFAULTS)
    path = os.path.join(STYLE_DIR, style + '.mplstyle')
    if os.path.isfile(path):
        return _read_style(path)
    if os.path.isfile(style):
        return _read_style(style)
    raise ValueError('%r is not a style (available: %s)' % (style, ', '.join(available())))


def use(style):
    """matplotlib's style.use: a name, a path, a dict, or a list of them."""
    for s in (style if isinstance(style, (list, tuple)) else [style]):
        d = _style_dict(s)
        if s == 'default':
            rcdefaults()
        else:
            rcParams.update(d)


@contextlib.contextmanager
def context(style, after_reset=False):
    with rc_context():
        if after_reset:
            rcdefaults()
        use(style)
        yield
