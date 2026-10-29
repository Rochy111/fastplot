"""Write fastplot/data/cmaps.npz and fastplot/data/colors.npz from the
installed matplotlib -- the colormaps and colour names fastplot._color uses
without importing matplotlib.

    python tools/make_cmaps.py

cmaps.npz holds every colormap matplotlib registers (its base maps; the
'_r' maps are made at run time by reversing them, as matplotlib itself makes
them), each as what it is built from, so that reversed() and resampled()
come out as matplotlib's would:

- ListedColormap: its colours, float64 RGBA (`listed`, rows `lst0:lst1`);
- LinearSegmentedColormap from data: each channel's (x, y0, y1) table
  (`seg`, rows `seg_idx[i, c]`; alpha absent where the map has none);
- LinearSegmentedColormap from functions: each channel's function, named
  (`func`: 'g7', 'flag_red', 'ch:gamma,s,r,h,p0,p1', ...; fastplot._color
  evaluates them with matplotlib's own expressions).

and, for checking and for the native size, every map's LUT as matplotlib
gives it, `cmap(np.arange(N), bytes=True)` (`lut`, rows `lut0:lut1`), with
under / over / bad (float RGBA, NaN where unset). Aliases ('grey' for
'gray' ...) are separate entries, as in matplotlib's registry.

colors.npz: every name matplotlib.colors.to_rgba knows (base, CSS4,
'tab:', 'xkcd:' and the xkcd 'grey' -> 'gray' spellings) with its float64
RGBA, and the default property cycle ('C0'..., float64 RGBA).
"""
import functools
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib import _cm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'fastplot', 'data')

CHANNELS = ('red', 'green', 'blue', 'alpha')


def func_name(f):
    """The name fastplot._color knows a segment function by."""
    for i, g in _cm.gfunc.items():
        if f is g:
            return 'g%d' % i
    if isinstance(f, functools.partial) and f.func is _cm._ch_helper:
        assert not f.keywords
        return 'ch:' + ','.join(repr(float(a)) for a in f.args)
    name = getattr(f, '__name__', '')
    known = ('_flag_red', '_flag_green', '_flag_blue', '_prism_red', '_prism_green',
             '_prism_blue', '_gist_heat_red', '_gist_heat_green', '_gist_heat_blue',
             '_gist_yarg')
    if name in known:
        return name[1:]
    raise ValueError('unknown colormap function %r' % (f,))


def rgba_or_nan(c):
    return np.full(4, np.nan) if c is None else np.asarray(mcolors.to_rgba(c), float)


def build_cmaps():
    reg = mpl.colormaps
    names = sorted(n for n in reg if not n.endswith('_r'))
    for n in reg:                               # every '_r' is a base map reversed
        if n.endswith('_r'):
            assert n[:-2] in names, n
    kind, Ns, gammas, ext = [], [], [], []
    luts, lut0 = [], [0]
    listed, lst0 = [], [0]
    seg, seg_idx = [], []
    funcs = []
    nseg = 0
    for name in names:
        cm = reg[name]
        assert cm.name == name, (cm.name, name)
        N = cm.N
        Ns.append(N)
        lut = cm(np.arange(N), bytes=True)
        luts.append(lut)
        lut0.append(lut0[-1] + N)
        ext.append([rgba_or_nan(cm._rgba_under), rgba_or_nan(cm._rgba_over),
                    rgba_or_nan(cm._rgba_bad)])
        idx = np.full((4, 2), -1, np.int64)
        fn = ['', '', '']
        if isinstance(cm, mcolors.ListedColormap):
            kind.append(0)
            gammas.append(1.0)
            cols = mcolors.to_rgba_array(cm.colors)
            assert cols.shape == (N, 4)
            listed.append(cols)
            lst0.append(lst0[-1] + N)
        elif isinstance(cm, mcolors.LinearSegmentedColormap):
            sd = cm._segmentdata
            gammas.append(float(cm._gamma))
            if any(callable(sd[c]) for c in CHANNELS[:3]):
                kind.append(2)
                assert 'alpha' not in sd
                fn = [func_name(sd[c]) for c in CHANNELS[:3]]
            else:
                kind.append(1)
                for ci, c in enumerate(CHANNELS):
                    if c in sd:
                        a = np.array(sd[c], dtype=float)
                        assert a.ndim == 2 and a.shape[1] == 3
                        seg.append(a)
                        idx[ci] = (nseg, nseg + len(a))
                        nseg += len(a)
            lst0.append(lst0[-1])
        else:
            raise TypeError('%s: %s' % (name, type(cm)))
        seg_idx.append(idx)
        funcs.append(fn)
    return dict(
        names=np.array(names), kind=np.array(kind, np.int8), N=np.array(Ns, np.int32),
        gamma=np.array(gammas), extremes=np.array(ext),
        lut=np.concatenate(luts).astype(np.uint8), lut0=np.array(lut0, np.int64),
        listed=np.concatenate(listed), lst0=np.array(lst0, np.int64),
        seg=np.concatenate(seg), seg_idx=np.array(seg_idx), func=np.array(funcs))


def build_colors():
    full = mcolors._colors_full_map
    names = sorted(full)
    rgba = np.array([mcolors.to_rgba(n) for n in names], float)
    cycle = mpl.rcParamsDefault['axes.prop_cycle'].by_key()['color']
    cycle = np.array([mcolors.to_rgba(c) for c in cycle], float)
    return dict(names=np.array(names), rgba=rgba, cycle=cycle)


def main():
    os.makedirs(OUT, exist_ok=True)
    c = build_cmaps()
    np.savez_compressed(os.path.join(OUT, 'cmaps.npz'), **c)
    k = build_colors()
    np.savez_compressed(os.path.join(OUT, 'colors.npz'), **k)
    for f in ('cmaps.npz', 'colors.npz'):
        p = os.path.join(OUT, f)
        print('%s: %d bytes' % (os.path.normpath(p), os.path.getsize(p)))
    print('%d colormaps (+ %d reversed), %d colour names, matplotlib %s'
          % (len(c['names']), len(c['names']), len(k['names']), mpl.__version__))


if __name__ == '__main__':
    sys.exit(main())
