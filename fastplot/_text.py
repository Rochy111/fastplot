"""Text: fonts, metrics, glyph masks and their blending -- titles, tick
labels, colorbar labels, contour labels drawn as matplotlib's Agg draws
them, for almost nothing.

    ext = text_extent('500 hPa', x, y, 13.9, ha='center', va='top')
    ext = draw_text(canvas, 'T$_{2m}$ ($^\\circ$C)', x, y, 13.9, (0, 0, 0, 1.0))
    tm, x0, y0, ext = place(s, x, y, 13.9, rotation=90)   # then blend it anywhere
    blend_mask(canvas, tm.alpha, x0, y0, rgba)

The font is matplotlib's, DejaVu Sans (bundled in data/fonts, with its
bold and oblique faces), rasterized by Pillow's FreeType with the same
hinting matplotlib's Agg uses (FreeType's default, native TrueType
hinting), so a glyph's pixels are matplotlib's. What matplotlib adds on
top -- harfbuzz shaping through libraqm -- Pillow's raqm-free BASIC layout
does not have, so it is done here: glyphs are placed one by one at
FreeType's hinted advances plus the font's GPOS pair kerning (the 'kern'
feature of the text's script, as harfbuzz applies it) and its standard
ligatures (fi, fl, ff ...), each glyph rendered at its fractional pen
position. A small reader of the font's own tables (cmap, hmtx, OS/2,
GPOS, GSUB, kern) supplies what Pillow does not expose.

Layout is matplotlib's Text._get_layout (3.11), line for line: the box of
a line is its advance width by the larger of its ink and the font's
typographic ascent/descent; multi-line text stacks lines by those metrics
(linespacing None: matplotlib 3.11's default, 'normal') or by
`linespacing` times the font height when a number is given (matplotlib's
fixed spacing, which also pads a single line); the box is rotated and then
aligned (rotation_mode 'default'), or aligned and then rotated about the
anchor ('anchor'). Bounding boxes agree with matplotlib's
Text.get_window_extent (Agg, size_px = fontsize * dpi / 72) to rounding
error, mathtext included. Fractional font sizes need Pillow >= 10.1.

Text between $...$ is a subset of mathtext laid out with matplotlib's own
box model (its TeX-derived sub/superscript rules and DejaVu Sans
constants): ^ and _ (shrunk 0.7, raised and lowered as matplotlib does),
Greek letters, \\circ \\degree \\times \\pm \\cdot and friends, \\mathrm
\\mathbf \\mathit \\text, the spaces \\, \\; \\quad ..., the escapes \\% \\$.
An unknown command is drawn as its own text rather than failing.

A text's pixels are an alpha mask, cached (LRU) by everything that shapes
it; drawing is blend_mask, the "over" blend of docs/internals.md in
float32, a numba kernel and a CUDA kernel (CUDA, below) that give the same
bytes. Masks sit on whole pixels (the anchor is rounded), so glyphs stay
as crisp as the hinter made them; rotations by 90, 180 and 270 degrees
are exact transpositions of the upright mask, other angles a bicubic
resampling of it.
"""
import math
import os
import struct
import threading
import unicodedata
from functools import lru_cache
from typing import NamedTuple

import numpy as np
from ._jit import njit

__all__ = ['Font', 'TextMask', 'font', 'text_extent', 'draw_text', 'place',
           'mask', 'blend_mask', 'blend_mask_kernel', 'CUDA', 'FONT_DIR']

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'fonts')
_BUNDLED = {(False, False): 'DejaVuSans.ttf', (True, False): 'DejaVuSans-Bold.ttf',
            (False, True): 'DejaVuSans-Oblique.ttf', (True, True): 'DejaVuSans-BoldOblique.ttf'}
_BUNDLED_NAMES = {'dejavu sans', 'dejavusans', 'sans-serif', 'sans', 'sans serif'}
MASK_CACHE = 4096                 # masks kept (LRU)
LAYOUT_CACHE = 8192               # layouts kept (LRU)


# ---------------------------------------------------------------- font files

class _Face:
    """The tables of one font file Pillow does not expose: units per em,
    the typographic ascent/descent/line gap matplotlib lays lines out
    with, the character map, unhinted advances (hmtx), GPOS pair kerning
    and GSUB standard ligatures by script, and the legacy 'kern' table
    (mathtext's kerning, FreeType's FT_Get_Kerning)."""

    def __init__(self, path):
        with open(path, 'rb') as f:
            self.data = data = f.read()
        self.path = path
        off = 0
        if data[:4] == b'ttcf':                 # a collection: its first font
            off = struct.unpack_from('>I', data, 12)[0]
        n = struct.unpack_from('>H', data, off + 4)[0]
        self.tables = {}
        for i in range(n):
            tag, _, toff, tlen = struct.unpack_from('>4sIII', data, off + 12 + 16 * i)
            self.tables[tag.decode('latin-1')] = (toff, tlen)
        self.upem = self._u16('head', 18)
        self.num_glyphs = self._u16('maxp', 4)
        if 'OS/2' in self.tables and self.tables['OS/2'][1] >= 74:
            o = self.tables['OS/2'][0]
            a, d, g = struct.unpack_from('>hhh', data, o + 68)
        else:
            o = self.tables['hhea'][0]
            a, d, g = struct.unpack_from('>hhh', data, o + 4)
        self.typo = (a / self.upem, -d / self.upem, g / self.upem)   # per em
        nhm = self._u16('hhea', 34)
        o = self.tables['hmtx'][0]
        adv = np.frombuffer(data, '>u2', 2 * nhm, o)[0::2].astype(np.int32)
        if nhm < self.num_glyphs:
            adv = np.concatenate([adv, np.full(self.num_glyphs - nhm, adv[-1] if nhm else 0, np.int32)])
        self.hadv = adv
        self._cmap_init()
        self._gpos = {}                         # script -> kern lookups
        self._gsub = {}                         # script -> ligatures
        self._kern_cache = {}
        self._kern0 = None
        self._rev = None

    def glyph_box(self, gid):
        """The glyph's unhinted ink box (xMin, yMin, xMax, yMax), font
        units, from 'glyf' (None: no outline, or not a TrueType font)."""
        if 'glyf' not in self.tables or 'loca' not in self.tables:
            return None
        lo = self.tables['loca'][0]
        if self._u16('head', 50):
            a, b = struct.unpack_from('>II', self.data, lo + 4 * gid)
        else:
            a, b = struct.unpack_from('>HH', self.data, lo + 2 * gid)
            a, b = 2 * a, 2 * b
        if b <= a:
            return None
        return struct.unpack_from('>hhhh', self.data, self.tables['glyf'][0] + a + 2)

    def _u16(self, tag, at):
        return struct.unpack_from('>H', self.data, self.tables[tag][0] + at)[0]

    # cmap
    def _cmap_init(self):
        data = self.data
        base = self.tables['cmap'][0]
        n = struct.unpack_from('>H', data, base + 2)[0]
        subs = {}
        for i in range(n):
            pid, eid, so = struct.unpack_from('>HHI', data, base + 4 + 8 * i)
            fmt = struct.unpack_from('>H', data, base + so)[0]
            subs[(pid, eid, fmt)] = base + so
        self._c12 = self._c4 = None
        for key in ((3, 10, 12), (0, 4, 12), (0, 6, 12)):
            if key in subs:
                o = subs[key]
                ng = struct.unpack_from('>I', data, o + 12)[0]
                g = np.frombuffer(data, '>u4', 3 * ng, o + 16).reshape(ng, 3).astype(np.int64)
                self._c12 = (g[:, 0], g[:, 1], g[:, 2])
                break
        for key in ((3, 1, 4), (0, 3, 4), (0, 4, 4), (0, 1, 4), (0, 0, 4), (3, 0, 4)):
            if key in subs:
                o = subs[key]
                seg = struct.unpack_from('>H', data, o + 6)[0] // 2
                end = np.frombuffer(data, '>u2', seg, o + 14).astype(np.int64)
                start = np.frombuffer(data, '>u2', seg, o + 16 + 2 * seg).astype(np.int64)
                delta = np.frombuffer(data, '>i2', seg, o + 16 + 4 * seg).astype(np.int64)
                ro_at = o + 16 + 6 * seg
                ro = np.frombuffer(data, '>u2', seg, ro_at).astype(np.int64)
                self._c4 = (end, start, delta, ro, ro_at)
                break
        self._gid = {}

    def gid(self, ch):
        """The glyph index of a character (0: the font has none)."""
        g = self._gid.get(ch)
        if g is not None:
            return g
        c = ord(ch)
        g = 0
        if self._c12 is not None:
            s, e, sg = self._c12
            i = int(np.searchsorted(e, c))
            if i < len(e) and s[i] <= c:
                g = int(sg[i] + c - s[i])
        elif self._c4 is not None and c < 0x10000:
            end, start, delta, ro, ro_at = self._c4
            i = int(np.searchsorted(end, c))
            if i < len(end) and start[i] <= c:
                if ro[i] == 0:
                    g = int((c + delta[i]) & 0xFFFF)
                else:
                    at = ro_at + 2 * i + int(ro[i]) + 2 * (c - int(start[i]))
                    g = struct.unpack_from('>H', self.data, at)[0]
                    if g:
                        g = (g + int(delta[i])) & 0xFFFF
        if g >= self.num_glyphs:
            g = 0
        self._gid[ch] = g
        return g

    def char_of(self, gid):
        """A character mapped to glyph `gid` among the presentation forms
        and the BMP (for ligature glyphs, which Pillow can only draw by
        character), or None."""
        if self._rev is None:
            rev = {}
            for c in list(range(0xFB00, 0xFB07)) + list(range(0x2000, 0x2070)):
                g = self.gid(chr(c))
                if g and g not in rev:
                    rev[g] = chr(c)
            self._rev = rev
        return self._rev.get(gid)

    # OpenType layout tables
    def _lookups(self, tag, feature, script):
        """Lookup offsets of `feature` for `script` (its default language
        system; DFLT, then latn, when the font has no such script)."""
        if tag not in self.tables:
            return []
        data = self.data
        base = self.tables[tag][0]
        sl, fl, ll = struct.unpack_from('>HHH', data, base + 4)
        sl += base
        fl += base
        ll += base
        scripts = {}
        for i in range(struct.unpack_from('>H', data, sl)[0]):
            stag, so = struct.unpack_from('>4sH', data, sl + 2 + 6 * i)
            scripts[stag.decode('latin-1')] = sl + so
        for s in (script, 'DFLT', 'latn'):
            if s in scripts:
                so = scripts[s]
                break
        else:
            return []
        dls = struct.unpack_from('>H', data, so)[0]
        if not dls:
            return []
        ls = so + dls
        req, nf = struct.unpack_from('>HH', data, ls + 2)
        fidx = list(struct.unpack_from('>%dH' % nf, data, ls + 6))
        if req != 0xFFFF:
            fidx.append(req)
        lidx = set()
        for fi in fidx:
            ftag, fo = struct.unpack_from('>4sH', data, fl + 2 + 6 * fi)
            if ftag.decode('latin-1') == feature:
                fo += fl
                n = struct.unpack_from('>H', data, fo + 2)[0]
                lidx.update(struct.unpack_from('>%dH' % n, data, fo + 4))
        out = []
        for li in sorted(lidx):
            lo = ll + struct.unpack_from('>H', data, ll + 2 + 2 * li)[0]
            ltype, flag, nsub = struct.unpack_from('>HHH', data, lo)
            subs = []
            for k in range(nsub):
                st = lo + struct.unpack_from('>H', data, lo + 6 + 2 * k)[0]
                t = ltype
                if (tag == 'GPOS' and t == 9) or (tag == 'GSUB' and t == 7):
                    t, eo = struct.unpack_from('>HI', data, st + 2)
                    st += eo
                subs.append((t, st))
            out.append(subs)
        return out

    def _coverage(self, o):
        data = self.data
        fmt, n = struct.unpack_from('>HH', data, o)
        if fmt == 1:
            g = np.frombuffer(data, '>u2', n, o + 4)
            return {int(x): i for i, x in enumerate(g)}
        cov = {}
        r = np.frombuffer(data, '>u2', 3 * n, o + 4).reshape(n, 3)
        for s, e, si in r.tolist():
            for g in range(s, e + 1):
                cov[g] = si + g - s
        return cov

    def _classdef(self, o):
        data = self.data
        cls = np.zeros(self.num_glyphs + 1, np.int32)
        fmt = struct.unpack_from('>H', data, o)[0]
        if fmt == 1:
            s, n = struct.unpack_from('>HH', data, o + 2)
            v = np.frombuffer(data, '>u2', n, o + 6)
            e = min(s + n, len(cls))
            cls[s:e] = v[:e - s]
        elif fmt == 2:
            n = struct.unpack_from('>H', data, o + 2)[0]
            for s, e, c in np.frombuffer(data, '>u2', 3 * n, o + 4).reshape(n, 3).tolist():
                cls[s:min(e + 1, len(cls))] = c
        return cls

    @staticmethod
    def _vsize(vf):
        return 2 * bin(vf & 0xFF).count('1')

    @staticmethod
    def _xadv_at(vf):
        """Byte offset of XAdvance in a value record of format vf (None
        when it has none)."""
        return 2 * bin(vf & 3).count('1') if vf & 4 else None

    def _kern_lookups(self, script):
        k = self._gpos.get(script)
        if k is not None:
            return k
        data = self.data
        lookups = []
        for subs in self._lookups('GPOS', 'kern', script):
            tabs = []
            for t, st in subs:
                if t != 2:
                    continue
                fmt, co, vf1, vf2 = struct.unpack_from('>HHHH', data, st)
                xa = self._xadv_at(vf1)
                rec = self._vsize(vf1) + self._vsize(vf2)
                cov = self._coverage(st + co)
                if fmt == 1:
                    n = struct.unpack_from('>H', data, st + 8)[0]
                    sets = struct.unpack_from('>%dH' % n, data, st + 10)
                    tabs.append((1, cov, (st, sets, xa, rec)))
                elif fmt == 2:
                    c1o, c2o, n1, n2 = struct.unpack_from('>HHHH', data, st + 8)
                    c1 = self._classdef(st + c1o)
                    c2 = self._classdef(st + c2o)
                    if xa is None:
                        mat = np.zeros((n1, n2), np.int32)
                    else:
                        raw = np.frombuffer(data, np.uint8, n1 * n2 * rec, st + 16)
                        raw = raw.reshape(n1 * n2, rec)[:, xa:xa + 2]
                        mat = raw.copy().view('>i2').reshape(n1, n2).astype(np.int32)
                    tabs.append((2, cov, (c1, c2, mat)))
            lookups.append(tabs)
        self._gpos[script] = lookups
        return lookups

    def kern(self, script, g1, g2):
        """GPOS 'kern' advance adjustment between glyphs g1, g2 (font units),
        every lookup of the feature applied, each by its first subtable
        covering the pair (as harfbuzz does)."""
        key = (script, g1, g2)
        v = self._kern_cache.get(key)
        if v is not None:
            return v
        v = 0
        for tabs in self._kern_lookups(script):
            for fmt, cov, t in tabs:
                ci = cov.get(g1)
                if ci is None:
                    continue
                if fmt == 2:
                    c1, c2, mat = t
                    a, b = c1[g1], c2[g2]
                    if a < mat.shape[0] and b < mat.shape[1]:
                        v += int(mat[a, b])
                        break
                else:
                    st, sets, xa, rec = t
                    if ci >= len(sets):
                        continue
                    so = st + sets[ci]
                    n = struct.unpack_from('>H', self.data, so)[0]
                    found = False
                    for k in range(n):
                        at = so + 2 + k * (2 + rec)
                        if struct.unpack_from('>H', self.data, at)[0] == g2:
                            if xa is not None:
                                v += struct.unpack_from('>h', self.data, at + 2 + xa)[0]
                            found = True
                            break
                    if found:
                        break
        self._kern_cache[key] = v
        return v

    def ligatures(self, script):
        """{first glyph: [(components, ligature glyph), ...]} of the GSUB
        'liga' feature for `script`, in the font's order of preference."""
        lig = self._gsub.get(script)
        if lig is not None:
            return lig
        data = self.data
        lig = {}
        for subs in self._lookups('GSUB', 'liga', script):
            for t, st in subs:
                if t != 4:
                    continue
                fmt, co, n = struct.unpack_from('>HHH', data, st)
                if fmt != 1:
                    continue
                cov = self._coverage(st + co)
                inv = {i: g for g, i in cov.items()}
                for i in range(n):
                    so = st + struct.unpack_from('>H', data, st + 6 + 2 * i)[0]
                    m = struct.unpack_from('>H', data, so)[0]
                    for k in range(m):
                        lo = so + struct.unpack_from('>H', data, so + 2 + 2 * k)[0]
                        lg, cc = struct.unpack_from('>HH', data, lo)
                        comps = struct.unpack_from('>%dH' % (cc - 1), data, lo + 4)
                        if i in inv:
                            lig.setdefault(inv[i], []).append((comps, lg))
        self._gsub[script] = lig
        return lig

    def kern0(self, g1, g2):
        """The legacy 'kern' table's value for a pair (font units)."""
        if self._kern0 is None:
            pairs = {}
            if 'kern' in self.tables:
                data = self.data
                o = self.tables['kern'][0]
                ver, nt = struct.unpack_from('>HH', data, o)
                if ver == 0:
                    p = o + 4
                    for _ in range(nt):
                        _, ln, cov = struct.unpack_from('>HHH', data, p)
                        if cov >> 8 == 0 and cov & 1:
                            n = struct.unpack_from('>H', data, p + 6)[0]
                            arr = np.frombuffer(data, '>u2', 3 * n, p + 14).reshape(n, 3)
                            keys = (arr[:, 0].astype(np.int64) << 16) | arr[:, 1]
                            vals = arr[:, 2].astype(np.int16)
                            pairs.update(zip(keys.tolist(), vals.tolist()))
                        p += ln
            self._kern0 = pairs
        return self._kern0.get((g1 << 16) | g2, 0)


@lru_cache(maxsize=None)
def _face(path):
    return _Face(path)


# ---------------------------------------------------------------- fonts

def _is_bold(weight):
    if isinstance(weight, (int, float, np.integer, np.floating)):
        return weight >= 550
    w = str(weight).lower().replace('-', ' ').replace('_', ' ')
    return w in ('bold', 'semibold', 'demibold', 'demi', 'heavy', 'extra bold',
                 'extrabold', 'ultrabold', 'ultra bold', 'black')


def _is_italic(style):
    return str(style).lower() in ('italic', 'oblique')


_FONT_DIRS = None


def _system_dirs():
    global _FONT_DIRS
    if _FONT_DIRS is None:
        d = []
        win = os.environ.get('WINDIR')
        if win:
            d.append(os.path.join(win, 'Fonts'))
        la = os.environ.get('LOCALAPPDATA')
        if la:
            d.append(os.path.join(la, 'Microsoft', 'Windows', 'Fonts'))
        d += ['/usr/share/fonts', '/usr/local/share/fonts', os.path.expanduser('~/.fonts'),
              os.path.expanduser('~/.local/share/fonts'), '/Library/Fonts',
              '/System/Library/Fonts', os.path.expanduser('~/Library/Fonts')]
        _FONT_DIRS = [x for x in d if os.path.isdir(x)]
    return _FONT_DIRS


_GENERIC = ('sans-serif', 'serif', 'monospace', 'cursive', 'fantasy')
_NAME_INDEX = None


def _mpl_font_dir():
    """matplotlib's own font folder (its DejaVu Serif, Mono ...), if it is
    installed -- found without importing it."""
    import importlib.util
    try:
        spec = importlib.util.find_spec('matplotlib')
    except (ImportError, ValueError):
        return None
    for loc in (spec.submodule_search_locations or []) if spec else []:
        d = os.path.join(loc, 'mpl-data', 'fonts', 'ttf')
        if os.path.isdir(d):
            return d
    return None


def _name_index():
    """{family name (lower case): {style name (lower case): path}} of the
    fonts in matplotlib's folder and the system's (read once, when a family
    is asked for by name)."""
    global _NAME_INDEX
    if _NAME_INDEX is None:
        from PIL import ImageFont
        idx = {}
        dirs = [d for d in [_mpl_font_dir()] if d] + _system_dirs()
        for d in dirs:
            for root, _, files in os.walk(d):
                for f in files:
                    if not f.lower().endswith(('.ttf', '.otf', '.ttc')):
                        continue
                    p = os.path.join(root, f)
                    try:
                        fam, sty = ImageFont.truetype(p, 10).getname()
                    except Exception:                   # noqa: BLE001 -- not a usable font
                        continue
                    if fam:
                        idx.setdefault(fam.lower(), {}).setdefault((sty or 'regular').lower(), p)
        _NAME_INDEX = idx
    return _NAME_INDEX


def _styled(styles, bold, italic):
    """The face of a family closest to the weight and style asked for."""
    want = ('bold ' if bold else '') + ('italic' if italic else '')
    order = {(False, False): ('regular', 'book', 'roman', 'normal', 'medium'),
             (True, False): ('bold',),
             (False, True): ('italic', 'oblique'),
             (True, True): ('bold italic', 'bold oblique')}[(bold, italic)]
    for s in order:
        if s in styles:
            return styles[s]
    for s, p in styles.items():                         # e.g. 'Condensed Bold'
        if want.strip() and all(w in s for w in want.split()):
            return p
    for s in ('regular', 'book', 'roman', 'normal'):
        if s in styles:
            return styles[s]
    return next(iter(styles.values()))


def _find_font(name, bold, italic):
    """One family name -> its font file, or None."""
    low = name.lower()
    if low in _BUNDLED_NAMES:
        return os.path.join(FONT_DIR, _BUNDLED[(bold, italic)])
    if os.path.isfile(name):
        return os.path.abspath(name)
    styles = _name_index().get(low) or _name_index().get(low.replace('-', ' '))
    if styles:
        return _styled(styles, bold, italic)
    names = {low, low.replace(' ', '')}                 # a file's name: 'arial', 'Arial.ttf'
    if not os.path.splitext(low)[1]:
        names = {n + e for n in names for e in ('.ttf', '.otf', '.ttc')}
    for d in _system_dirs():
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower() in names:
                    return os.path.join(root, f)
    return None


def family_key(family):
    """A text's family as matplotlib's findfont reads it: None is
    rcParams font.family; a name or a list of them, generic ones
    (sans-serif, serif, monospace ...) standing for their rcParams lists.
    The names in order of preference, a tuple (a path stays a path)."""
    from ._rc import active
    rc = active()
    if family is None:
        family = rc.get('font.family', ['sans-serif'])
    fams = [family] if isinstance(family, (str, os.PathLike)) else list(family)
    out = []
    for f in fams:
        f = os.fspath(f)
        if f.lower() in _GENERIC:
            out += [str(v) for v in rc.get('font.' + f.lower(), [])] or ['DejaVu Sans']
        else:
            out.append(f)
    return tuple(out) if out else None


_WARNED = set()


@lru_cache(maxsize=256)
def _font_path(bold, italic, family):
    """The font file for a family: None, bundled DejaVu Sans; else the
    first of family_key's names found -- the bundled DejaVu Sans, a path,
    a family name in matplotlib's or the system's fonts ('DejaVu Serif',
    'Times New Roman'), or a font file's name ('arial', 'Arial.ttf');
    none of them: DejaVu Sans, with a warning, as matplotlib's findfont."""
    if family is None:
        return os.path.join(FONT_DIR, _BUNDLED[(bold, italic)])
    names = (family,) if isinstance(family, str) else tuple(family)
    for n in names:
        if n.lower() in _GENERIC:
            continue
        p = _find_font(n, bold, italic)
        if p is not None:
            return p
    if names not in _WARNED:
        _WARNED.add(names)
        import warnings
        warnings.warn('findfont: Font family %r not found; falling back to DejaVu Sans.'
                      % (list(names),), stacklevel=3)
    return os.path.join(FONT_DIR, _BUNDLED[(bold, italic)])


class Font:
    """One font file at one pixel size: Pillow's FreeType face (hinted as
    matplotlib's Agg hints) and the per-character metrics and glyph
    images text is built from, cached.

    `size_px` is FreeType's size in pixels per em, matplotlib's
    fontsize * dpi / 72; it is kept in FreeType's own 1/64 px steps, the
    size matplotlib's FT_Set_Char_Size asks for. `ascent`, `descent` and
    `line_gap` (px) are the typographic metrics (OS/2, else hhea) that
    matplotlib stacks lines with."""

    def __init__(self, path, size_px):
        from PIL import ImageFont
        self.path = path
        self.size64 = max(int(math.floor(size_px * 64 + 0.5)), 1)
        self.size_px = self.size64 / 64.0
        self.face = _face(path)
        self.pil = ImageFont.truetype(path, self.size_px, layout_engine=ImageFont.Layout.BASIC)
        a, d, g = self.face.typo
        s = self.size_px                        # matplotlib scales by the nominal size
        self.ascent, self.descent, self.line_gap = a * s, d * s, g * s
        self._adv = {}
        self._bbox = {}
        self._lin = {}
        self._glyph = {}
        self._ubox = {}

    def __repr__(self):
        return 'Font(%r, %g px)' % (os.path.basename(self.path), self.size_px)

    def advance(self, ch):
        """FreeType's hinted advance of a character, px."""
        v = self._adv.get(ch)
        if v is None:
            v = self._adv[ch] = float(self.pil.getlength(ch))
        return v

    def linear_advance(self, ch):
        """The unhinted advance (FreeType's linearHoriAdvance), px."""
        v = self._lin.get(ch)
        if v is None:
            v = self._lin[ch] = int(self.face.hadv[self.face.gid(ch)]) * self.size_px / self.face.upem
        return v

    def bbox(self, ch):
        """The hinted glyph's ink box (left, top, right, bottom) in whole
        pixels about its origin on the baseline, y down; (0, 0, 0, 0) for
        one without ink."""
        v = self._bbox.get(ch)
        if v is None:
            if ch.isspace():
                v = (0, 0, 0, 0)
            else:
                v = tuple(int(t) for t in self.pil.getbbox(ch, anchor='ls'))
                if v[2] <= v[0] or v[3] <= v[1]:
                    v = (0, 0, 0, 0)
            self._bbox[ch] = v
        return v

    def has(self, ch):
        return self.face.gid(ch) != 0

    def unhinted_box(self, ch):
        """The unhinted ink box (xmin, ymin, xmax, ymax), px, y up --
        FreeType's metrics with FT_LOAD_NO_HINTING, what matplotlib's
        mathtext lays out with; the hinted box for non-TrueType fonts."""
        v = self._ubox.get(ch)
        if v is None:
            if ch.isspace():
                v = (0.0, 0.0, 0.0, 0.0)
            else:
                b = self.face.glyph_box(self.face.gid(ch))
                if b is None:
                    l, t, r, bt = self.bbox(ch)
                    v = (float(l), float(-bt), float(r), float(-t))
                else:
                    k = self.size_px / self.face.upem
                    v = tuple(math.floor(u * k * 64 + 0.5) / 64 for u in b)
            self._ubox[ch] = v
        return v

    def glyph(self, ch):
        """The glyph's coverage rendered at an integer origin, float32
        0..1, and its top-left pixel (left, top) about the origin; None
        for one without ink."""
        v = self._glyph.get(ch)
        if v is None:
            l, t, r, b = self.bbox(ch)
            if r <= l:
                v = (None, 0, 0)
            else:
                from PIL import Image, ImageDraw
                im = Image.new('L', (r - l + 4, b - t + 4), 0)
                ImageDraw.Draw(im).text((2 - l, 2 - t), ch, fill=255, font=self.pil, anchor='ls')
                a = np.asarray(im)
                rows = np.flatnonzero(a.any(1))
                cols = np.flatnonzero(a.any(0))
                if not len(rows):
                    v = (None, 0, 0)
                else:
                    a = a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
                    v = (a.astype(np.float32) / np.float32(255),
                         int(cols[0]) - 2 + l, int(rows[0]) - 2 + t)
            self._glyph[ch] = v
        return v


try:                                     # FreeType itself: glyphs at fractional pens
    import freetype as _FT
except ImportError:                      # (else whole-pixel glyphs moved by _subpixel)
    _FT = None


@lru_cache(maxsize=64)
def _ft_face(path):
    return _FT.Face(path)


@lru_cache(maxsize=512)
def _font(path, size64):
    return Font(path, size64 / 64.0)


def font(size_px, weight='normal', style='normal', family=None):
    """The (cached) Font of a size in pixels, weight ('normal', 'bold' or
    a number, bold from 550), style ('normal', 'italic'/'oblique') and
    family (None: the bundled DejaVu Sans, matplotlib's; a .ttf/.otf path;
    or a font file's name in the system's font folders)."""
    path = _font_path(_is_bold(weight), _is_italic(style),
                      None if family is None else family_key(family))
    return _font(path, max(int(math.floor(size_px * 64 + 0.5)), 1))


# matplotlib's dejavusans fontset beyond DejaVu Sans: its display font,
# where every glyph it has is taken from (the large operators), and the
# STIX sizes of the radical's sized alternatives
_MATH_EXTRA = {'ex': 'DejaVuSansDisplay.ttf', 's1': 'STIXSizOneSymReg.ttf',
               's2': 'STIXSizTwoSymReg.ttf', 's3': 'STIXSizThreeSymReg.ttf',
               's4': 'STIXSizFourSymReg.ttf', 's5': 'STIXSizFiveSymReg.ttf'}
_MATH_SLANTED = ('∫', '∮')               # \int, \oint: matplotlib's _slanted_symbols


def _math_font(key, size_px, user):
    """mathtext's fonts: 'default' is the text's own font, the others
    matplotlib's dejavusans math fontset."""
    if key == 'default':
        return _font(user.path, max(int(math.floor(size_px * 64 + 0.5)), 1))
    if key in _MATH_EXTRA:
        return _font(os.path.join(FONT_DIR, _MATH_EXTRA[key]),
                     max(int(math.floor(size_px * 64 + 0.5)), 1))
    b, i = {'rm': (False, False), 'it': (False, True), 'bf': (True, False),
            'bfit': (True, True)}[key]
    return _font(os.path.join(FONT_DIR, _BUNDLED[(b, i)]),
                 max(int(math.floor(size_px * 64 + 0.5)), 1))


# ---------------------------------------------------------------- plain lines

class _Line(NamedTuple):
    glyphs: tuple   # ((Font, char, x, y), ...): pen origins, px, y down, from the line's origin
    width: float    # advance width
    ascent: float   # ink above the baseline
    descent: float  # ink below it


_SCRIPTS = {'LATIN': 'latn', 'GREEK': 'grek', 'CYRILLIC': 'cyrl', 'ARMENIAN': 'armn',
            'HEBREW': 'hebr', 'ARABIC': 'arab', 'GEORGIAN': 'geor'}


def _script(text):
    """The OpenType script tag harfbuzz shapes a run with: that of its
    first letter; DFLT for text without letters (digits, punctuation)."""
    for c in text:
        if c.isalpha():
            if c < '\x80':
                return 'latn'
            return _SCRIPTS.get(unicodedata.name(c, ' ').split(' ')[0], 'DFLT')
    return 'DFLT'


def _plain_line(fnt, text):
    """A line laid out as matplotlib 3.11 lays it out with libraqm: hinted
    advances, GPOS kerning (rounded to FreeType's 1/64 px), standard
    ligatures; characters the font lacks are left out (matplotlib warns
    and skips them)."""
    face = fnt.face
    script = _script(text)
    chars = [c for c in text if face.gid(c)]
    gids = [face.gid(c) for c in chars]
    ligs = face.ligatures(script) if gids else {}
    if ligs:
        i = 0
        while i < len(gids):
            for comps, lg in ligs.get(gids[i], ()):
                n = len(comps)
                if tuple(gids[i + 1:i + 1 + n]) == comps:
                    c = face.char_of(lg)
                    if c is not None:
                        gids[i:i + 1 + n] = [lg]
                        chars[i:i + 1 + n] = [c]
                        break
            i += 1
    scale = fnt.size64 / face.upem
    glyphs = []
    pen = 0.0
    top = 0
    bot = 0
    inked = False
    for k, (c, g) in enumerate(zip(chars, gids)):
        glyphs.append((fnt, c, pen, 0.0))
        l, t, r, b = fnt.bbox(c)
        if r > l:
            if not inked:
                top, bot, inked = t, b, True
            else:
                top, bot = min(top, t), max(bot, b)
        else:                                   # FreeType's empty cbox: the origin
            top, bot = min(top, 0), max(bot, 0)
            inked = True
        adv = fnt.advance(c)
        if k + 1 < len(gids):
            kv = face.kern(script, g, gids[k + 1])
            if kv:
                adv += math.floor(kv * scale + 0.5) / 64.0
        pen += adv
    return _Line(tuple(glyphs), pen, float(-top), float(bot))


# ---------------------------------------------------------------- mathtext

SHRINK = 0.7                     # matplotlib's SHRINK_FACTOR
_LEVELS = 6                      # NUM_SIZE_LEVELS
# matplotlib's DejaVuSansFontConstants (per x-height)
_XH = 1120 / 2048
_SUP1 = 845.824 / 1120
_SUB1 = 307.199 / 1120
_SUB2 = 632.832 / 1120
_SUPDROP = 790.527 / 1120
_SUBDROP = 102.4 / 1120
_DELTA = 0.025
_DELTA_SLANTED = 0.2
_SCRIPT_SPACE = 0.05

_GREEK = dict(zip(
    'alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda '
    'mu nu xi pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi psi omega '
    'Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega '
    'Alpha Beta Epsilon Zeta Eta Iota Kappa Mu Nu Omicron Rho Tau Chi omicron'.split(),
    '\u03b1\u03b2\u03b3\u03b4\u03b5\u03b5\u03b6\u03b7\u03b8\u03d1\u03b9\u03ba\u03bb'
    '\u03bc\u03bd\u03be\u03c0\u03d6\u03c1\u03f1\u03c3\u03c2\u03c4\u03c5\u03d5\u03c6\u03c7\u03c8\u03c9'
    '\u0393\u0394\u0398\u039b\u039e\u03a0\u03a3\u03a5\u03a6\u03a8\u03a9'
    '\u0391\u0392\u0395\u0396\u0397\u0399\u039a\u039c\u039d\u039f\u03a1\u03a4\u03a7\u03bf'))
_SYMBOLS = dict(_GREEK)
_SYMBOLS.update({
    'circ': '\u2218', 'degree': '\u00b0', 'times': '\u00d7', 'pm': '\u00b1', 'mp': '\u2213',
    'cdot': '\u22c5', 'div': '\u00f7', 'ast': '\u2217', 'star': '\u22c6', 'bullet': '\u2219',
    'leq': '\u2264', 'le': '\u2264', 'geq': '\u2265', 'ge': '\u2265', 'neq': '\u2260', 'ne': '\u2260',
    'approx': '\u2248', 'sim': '\u223c', 'simeq': '\u2243', 'equiv': '\u2261', 'propto': '\u221d',
    'll': '\u226a', 'gg': '\u226b', 'infty': '\u221e', 'partial': '\u2202', 'nabla': '\u2207',
    'rightarrow': '\u2192', 'to': '\u2192', 'leftarrow': '\u2190', 'uparrow': '\u2191',
    'downarrow': '\u2193', 'leftrightarrow': '\u2194', 'Rightarrow': '\u21d2',
    'Leftarrow': '\u21d0', 'prime': '\u2032', 'ldots': '\u2026', 'cdots': '\u22ef',
    'sum': '\u2211', 'int': '\u222b', 'prod': '\u220f', 'AA': '\u00c5',
    'perthousand': '\u2030', 'cap': '\u2229', 'cup': '\u222a', 'wedge': '\u2227', 'vee': '\u2228',
    'oplus': '\u2295', 'otimes': '\u2297', 'in': '\u2208', 'hbar': '\u0127', 'ell': '\u2113',
    'dagger': '\u2020', 'langle': '\u27e8', 'rangle': '\u27e9', 'lbrace': '{', 'rbrace': '}',
    'lbrack': '[', 'rbrack': ']', 'vert': '|', 'backslash': '\\', 'sharp': '\u266f',
    'checkmark': '\u2713', 'S': '\u00a7', 'P': '\u00b6', 'copyright': '\u00a9',
    'forall': '\u2200', 'exists': '\u2203', 'nexists': '\u2204', 'neg': '\u00ac',
    'lnot': '\u00ac', 'emptyset': '\u2205', 'varnothing': '\u2205', 'angle': '\u2220',
    'triangle': '\u25b3', 'clubsuit': '\u2663', 'spadesuit': '\u2660', 'heartsuit': '\u2661',
    'diamondsuit': '\u2662', 'subset': '\u2282', 'supset': '\u2283', 'subseteq': '\u2286',
    'supseteq': '\u2287', 'notin': '\u2209', 'ni': '\u220b', 'mid': '\u2223',
    'parallel': '\u2225', 'perp': '\u27c2', 'cong': '\u2245', 'doteq': '\u2250',
    'lesssim': '\u2272', 'gtrsim': '\u2273', 'leqslant': '\u2a7d', 'geqslant': '\u2a7e',
    'longrightarrow': '\u27f6', 'longleftarrow': '\u27f5', 'Longrightarrow': '\u27f9',
    'Longleftarrow': '\u27f8', 'longleftrightarrow': '\u27f7', 'mapsto': '\u21a6',
    'Leftrightarrow': '\u21d4', 'Uparrow': '\u21d1', 'Downarrow': '\u21d3',
    'nearrow': '\u2197', 'searrow': '\u2198', 'swarrow': '\u2199', 'nwarrow': '\u2196',
    'hookrightarrow': '\u21aa', 'hookleftarrow': '\u21a9', 'iint': '\u222c', 'iiint': '\u222d',
    'oint': '\u222e', 'surd': '\u221a', 'aleph': '\u2135', 'Re': '\u211c', 'Im': '\u2111',
    'wp': '\u2118', 'imath': '\u0131', 'jmath': '\u0237', 'top': '\u22a4', 'bot': '\u22a5',
    'vdots': '\u22ee', 'ddots': '\u22f1', 'dots': '\u2026', 'colon': ':', 'lfloor': '\u230a',
    'rfloor': '\u230b', 'lceil': '\u2308', 'rceil': '\u2309', 'Vert': '\u2016',
    'setminus': '\u2216', 'smallsetminus': '\u2216', 'odot': '\u2299', 'ominus': '\u2296',
    'oslash': '\u2298', 'bigcirc': '\u25cb', 'diamond': '\u22c4', 'triangleleft': '\u25c1',
    'triangleright': '\u25b7', 'bigtriangleup': '\u25b3', 'bigtriangledown': '\u25bd',
    'cdotp': '\u00b7', 'centerdot': '\u00b7', 'therefore': '\u2234', 'because': '\u2235',
    'flat': '\u266d', 'natural': '\u266e', 'yen': '\u00a5', 'pounds': '\u00a3',
    'euro': '\u20ac', 'dag': '\u2020', 'ddag': '\u2021', 'ddagger': '\u2021',
    'sqsubset': '\u228f', 'sqsupset': '\u2290', 'models': '\u22a7', 'vdash': '\u22a2',
    'dashv': '\u22a3', 'asymp': '\u224d', 'bowtie': '\u22c8', 'wr': '\u2240',
    'amalg': '\u2a3f', 'uplus': '\u228e', 'sqcap': '\u2293', 'sqcup': '\u2294',
    'nleq': '\u2270', 'ngeq': '\u2271', 'nsim': '\u2241', 'micro': '\u00b5',
    'textdegree': '\u00b0', 'implies': '\u27f9', 'iff': '\u27fa', 'coprod': '\u2210',
    'bigcap': '\u22c2', 'bigcup': '\u22c3', 'bigvee': '\u22c1', 'bigwedge': '\u22c0',
    'bigodot': '\u2a00', 'bigoplus': '\u2a01', 'bigotimes': '\u2a02', 'biguplus': '\u2a04',
    'bigsqcup': '\u2a06', 'leftrightarrows': '\u21c6', 'rightleftharpoons': '\u21cc',
    'Box': '\u25a1', 'square': '\u25a1', 'blacksquare': '\u25a0', 'lozenge': '\u25ca',
    'star': '\u22c6', 'circledast': '\u229b', 'hslash': '\u210f', 'mho': '\u2127',
    'eth': '\u00f0', 'beth': '\u2136', 'gimel': '\u2137', 'daleth': '\u2138',
    'digamma': '\u03dd', 'varkappa': '\u03f0', 'backprime': '\u2035',
})
# matplotlib's _overunder_symbols: limits above and below
_OVERUNDER = set('\u2211 \u220f \u2210 \u22c2 \u22c3 \u22c1 \u22c0 \u2a00 \u2a01 \u2a02 '
                 '\u2a04 \u2a06'.split())
_OVERUNDER_FUNCTIONS = set('lim liminf limsup sup max min'.split())
_FUNCTIONS = set('arccos csc ker min arcsin deg lg Pr arctan det lim sec arg dim liminf sin '
                 'cos exp limsup sinh cosh gcd ln sup cot hom log tan coth inf max tanh'.split())
_DELIMS = set('( ) [ ] { } < > | . / \\'.split()) | {'{', '}'}
# matplotlib's _accent_map (dejavusans: the combining marks, or a symbol)
_ACCENTS = {'hat': '\u0302', 'breve': '\u0306', 'bar': '\u0304', 'grave': '\u0300',
            'acute': '\u0301', 'tilde': '\u0303', 'dot': '\u0307', 'ddot': '\u0308',
            'dddot': '\u20db', 'ddddot': '\u20dc', 'vec': '\u20d7', 'overrightarrow': '\u2192',
            'overleftarrow': '\u2190', 'mathring': '\u2218', 'widehat': '\u0302',
            'widetilde': '\u0303', 'widebar': '\u0304'}
_RULE = object()                 # _ship's marker of a filled rule (not a glyph)
# \mathbb / \mathcal / \mathfrak: the letters' Unicode forms that DejaVu Sans has
_LETTERMAPS = {
    'mathbb': {'C': 'ℂ', 'H': 'ℍ', 'N': 'ℕ', 'P': 'ℙ', 'Q': 'ℚ',
               'R': 'ℝ', 'Z': 'ℤ'},
    'mathcal': {'B': 'ℬ', 'E': 'ℰ', 'F': 'ℱ', 'H': 'ℋ', 'I': 'ℐ',
                'L': 'ℒ', 'M': 'ℳ', 'R': 'ℛ', 'e': 'ℯ', 'g': 'ℊ',
                'o': 'ℴ'},
    'mathfrak': {'C': 'ℭ', 'H': 'ℌ', 'I': 'ℑ', 'R': 'ℜ', 'Z': 'ℨ'},
}
_LETTERMAPS['mathscr'] = _LETTERMAPS['mathcal']
# DejaVuSansFontConstants for fractions, per x-height; axis height per em
_NUM2 = 868.352 / 1120
_NUM3 = 970.752 / 1120
_DENOM2 = 768 / 1120
_AXIS = 512 / 2048
_ESCAPES = set('%$#&_{}')
_SPACES = {',': 0.16667, '/': 0.16667, '>': 0.22222, ':': 0.22222, ';': 0.27778,
           ' ': 0.33333, '!': -0.16667, 'thinspace': 0.16667, 'enspace': 0.5,
           'quad': 1.0, 'qquad': 2.0}
_FONTCMDS = {'mathrm': 'rm', 'mathit': 'it', 'mathbf': 'bf', 'mathsf': 'rm',
             'mathdefault': 'default', 'mathregular': 'default', 'mathbfit': 'bfit',
             'boldsymbol': 'bf', 'text': 'rm', 'mbox': 'rm', 'operatorname': 'rm'}
_BINARY = set('+ * \u2212 \u00b1 \u2213 \u00d7 \u00f7 \u2217 \u22c6 \u2218 \u2219 \u22c5 '
              '\u2229 \u222a \u2227 \u2228 \u2295 \u2297 \u2020'.split())
_RELATION = set('= < > : \u2264 \u2265 \u2260 \u2248 \u223c \u2243 \u2261 \u221d \u226a '
                '\u226b \u2208'.split())
_ARROWS = set('\u2192 \u2190 \u2191 \u2193 \u2194 \u21d2 \u21d0'.split())
_SPACED = _BINARY | _RELATION | _ARROWS
_PUNCT = set(',;.!')
_NO_SPACE_AFTER = set('{([<=>:')     # a binary operator after these is unary (matplotlib's rule)


class _Char:
    """mathtext's Char: TeX width/height/depth from the glyph's unhinted
    ink box (matplotlib's vector parse loads glyphs unhinted), the
    unhinted advance kept for the kern that follows it."""
    __slots__ = ('c', 'fk', 'size', 'width', 'height', 'depth', 'advance', 'slanted', 'level')

    def __init__(self, c, fk, size, user):
        if fk != 'default' and _math_font('ex', size, user).has(c):
            fk = 'ex'                             # DejaVuFonts._get_glyph: the display font first
        f = _math_font(fk, size, user)
        if not f.has(c) and fk != 'rm':          # matplotlib falls back to another font
            f = _math_font('rm', size, user)
            fk = 'rm'
        self.c, self.fk, self.size, self.level = c, fk, size, 0
        self.slanted = fk in ('it', 'bfit') or (fk == 'ex' and c in _MATH_SLANTED)
        x0, y0, x1, y1 = f.unhinted_box(c)
        self.advance = f.linear_advance(c)
        self.width = self.advance if c == ' ' else x1 - x0
        self.height = y1                          # FreeType's horiBearingY
        self.depth = -y0                          # -(iceberg - height)


class _Kern:
    __slots__ = ('width', 'level')

    def __init__(self, w):
        self.width, self.level = w, 0


class _HList:
    __slots__ = ('children', 'width', 'height', 'depth', 'shift', 'level')

    def __init__(self, children, kern=True, user=None):
        self.children = list(children)
        self.shift = 0.0
        self.level = 0
        if kern:
            self._kern(user)
        self.pack()

    def _kern(self, user):
        out = []
        ch = self.children
        for i, e in enumerate(ch):
            out.append(e)
            if isinstance(e, _Char):
                nxt = ch[i + 1] if i + 1 < len(ch) else None
                k = e.advance - e.width
                if isinstance(nxt, _Char) and nxt.fk == e.fk and nxt.size == e.size:
                    k += _kern_table(e, nxt, user)
                if k != 0.0:
                    out.append(_Kern(k))
        self.children = out

    def pack(self):
        w = h = d = 0.0
        for p in self.children:
            if isinstance(p, _Char):
                w += p.width
                h = max(h, p.height)
                d = max(d, p.depth)
            elif isinstance(p, _Kern):
                w += p.width
            else:
                w += p.width
                h = max(h, p.height - p.shift)
                d = max(d, p.depth + p.shift)
        self.width, self.height, self.depth = w, h, d


class _FuncList(_HList):
    """A function name's box (\\sin, \\lim ...), which knows its name."""
    __slots__ = ('function_name',)


class _Rule:
    """mathtext's Hrule: a filled bar (a fraction's, a radical's overline);
    in a vertical list as wide as the list."""
    __slots__ = ('width', 'height', 'depth', 'shift', 'level')

    def __init__(self, width, height, depth=0.0):
        self.width, self.height, self.depth = width, height, depth
        self.shift = 0.0
        self.level = 0


def _centered(node, width):
    """mathtext's HCentered packed to width: the node in the middle."""
    pad = (width - node.width) / 2.0
    h = _HList([_Kern(pad), node, _Kern(pad)], kern=False)
    return h


class _VList:
    """A superscript over a subscript (TeX's node759), its natural size."""
    __slots__ = ('children', 'width', 'height', 'depth', 'shift', 'level')

    def __init__(self, children):
        self.children = children
        self.shift = 0.0
        self.level = 0
        x = d = w = 0.0
        for p in children:
            if isinstance(p, _Kern):
                x += d + p.width
                d = 0.0
            else:
                x += d + p.height
                d = p.depth
                w = max(w, p.width)
        self.width, self.height, self.depth = w, x, d


def _kern_table(a, b, user):
    """FreeType's FT_Get_Kerning(FT_KERNING_DEFAULT) from the 'kern'
    table, what matplotlib's mathtext kerns with: scaled, damped below
    25 ppem, rounded to whole pixels."""
    f = _math_font(a.fk, a.size, user)
    v = f.face.kern0(f.face.gid(a.c), f.face.gid(b.c))
    if not v:
        return 0.0
    k = v * f.size64 / f.face.upem
    ppem = int(math.floor(f.size_px + 0.5))
    if ppem < 25:
        k = k * ppem / 25
    return math.floor((k + 32) / 64) * 64 / 64.0


def _shrink(n):
    n.level += 1
    small = n.level < _LEVELS
    if isinstance(n, _Char):
        if small:
            n.size *= SHRINK
            n.width *= SHRINK
            n.height *= SHRINK
            n.depth *= SHRINK
    elif isinstance(n, _Kern):
        if small:
            n.width *= SHRINK
    elif isinstance(n, _Rule):
        if small:
            n.width *= SHRINK
            n.height *= SHRINK
            n.depth *= SHRINK
    else:
        for c in n.children:
            _shrink(c)
        if small:
            n.width *= SHRINK
            n.height *= SHRINK
            n.depth *= SHRINK
            n.shift *= SHRINK


def _is_char_node(n):
    if isinstance(n, _VList):
        return False
    if isinstance(n, _HList):
        return all(_is_char_node(c) for c in n.children)
    return True


def _tokens(s):
    """mathtext tokens of a math string: (kind, value, position)."""
    out = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == '\\':
            j = i + 1
            while j < n and s[j].isalpha():
                j += 1
            if j == i + 1:                       # \, \; \% \$ \{ ...
                if j < n:
                    out.append(('cmd', s[j], i))
                    i = j + 1
                else:
                    out.append(('sym', '\\', i))
                    i = j
                continue
            name = s[i + 1:j]
            if name in ('text', 'mbox') and j < n and s[j] == '{':   # raw, spaces kept
                depth, k = 1, j + 1
                while k < n and depth:
                    depth += {'{': 1, '}': -1}.get(s[k], 0)
                    k += 1
                out.append(('text', s[j + 1:k - 1], i))
                i = k
                continue
            out.append(('cmd', name, i))
            i = j
        elif c in '{}^_':
            out.append((c, c, i))
            i += 1
        elif c.isspace():
            i += 1
        else:
            out.append(('sym', c, i))
            i += 1
    return out


class _MathParser:
    """Recursive descent over mathtext tokens to matplotlib's box tree."""

    def __init__(self, s, size, user):
        self.s, self.size, self.user = s, size, user
        self.toks = _tokens(s)
        self.i = 0
        self.quad = _math_font('it', size, user).linear_advance('m')   # get_quad: 'm' in 'normal'
        # matplotlib's _needs_space_after_subsuper: a function followed by a
        # script -- operators are not spaced until that script is attached,
        # and a thin space follows it
        self.needs_space = False

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None, len(self.s))

    def char(self, c, font):
        if font == 'normal':
            cat = unicodedata.category(c)[0]
            font = 'rm' if cat != 'L' or unicodedata.name(c, '').startswith('GREEK CAPITAL') else 'it'
        return _Char(c, font, self.size, self.user)

    def space(self, frac):
        return _Kern(self.quad * frac)

    def symbol(self, c, loc, font):
        if c == '-':
            c = '\u2212'
        ch = self.char(c, font)
        s = self.s
        if c in _SPACED:
            prev = next((x for x in s[:loc][::-1] if x != ' '), '')
            if self.needs_space or (c in _BINARY and (not s[:loc].split() or
                                                      prev in _NO_SPACE_AFTER)):
                return ch
            return _HList([self.space(0.2), ch, self.space(0.2)], user=self.user)
        if c in _PUNCT:
            prev = next((x for x in s[:loc][::-1] if x != ' '), '')
            nxt = next((x for x in s[loc + 1:] if x != ' '), '')
            if c == ',' and prev == '{' and nxt == '}':
                return ch
            if c == '.' and prev.isdigit() and nxt.isdigit():
                return ch
            return _HList([ch, self.space(0.2)], user=self.user)
        return ch

    def literal(self, text, font):
        return _HList([self.char(c, 'rm' if font == 'normal' else font) for c in text],
                      user=self.user)

    def expr(self, font):
        """Items up to a closing brace or the end."""
        items = []
        while True:
            kind = self.peek()[0]
            if kind is None or kind == '}':
                return items
            items.append(self.subsuper(font))

    def atom(self, font):
        kind, val, loc = self.peek()
        if kind in (None, '}', '^', '_'):
            return None
        self.i += 1
        if kind == 'sym':
            return self.symbol(val, loc, font)
        if kind == '{':
            items = self.expr(font)
            if self.peek()[0] == '}':
                self.i += 1
            return _HList(items, user=self.user)
        if kind == 'text':
            return _HList([self.char(c, 'rm') for c in val], user=self.user)
        # a command
        if val in _ESCAPES:
            return self.char(val, font)
        if val in _SPACES:
            return self.space(_SPACES[val])
        if val in _SYMBOLS:
            return self.symbol(_SYMBOLS[val], loc, font)
        if val in _LETTERMAPS:                          # \mathbb{R}: its letters' own glyphs
            arg = self.group('rm')
            return _HList([self.char(_LETTERMAPS[val].get(c.c, c.c), 'rm') if isinstance(c, _Char)
                           else c for c in (arg.children if isinstance(arg, _HList) else [arg])],
                          user=self.user)
        if val in _FONTCMDS:
            f = _FONTCMDS[val]
            if self.peek()[0] == '{':
                self.i += 1
                items = self.expr(f)
                if self.peek()[0] == '}':
                    self.i += 1
                return _HList(items, user=self.user)
            nxt = self.atom(f)
            return nxt if nxt is not None else _HList([])
        if val in ('left', 'right', 'big', 'Big', 'bigg', 'Bigg', 'displaystyle'):
            nxt = self.atom(font)
            return nxt if nxt is not None else _HList([])
        if val in ('frac', 'dfrac', 'tfrac', 'binom'):
            num, den = self.group(font), self.group(font)
            return self.genfrac(num, den, rule=val != 'binom', display=val == 'dfrac')
        if val == 'sqrt':
            root = None
            if self.peek()[0] == 'sym' and self.peek()[1] == '[':
                self.i += 1
                start = self.i
                while self.peek()[0] not in (None,) and not (self.peek()[0] == 'sym' and
                                                             self.peek()[1] == ']'):
                    self.i += 1
                inner = self.toks[start:self.i]
                if self.peek()[0] == 'sym':
                    self.i += 1
                sub = _MathParser.__new__(_MathParser)
                sub.__dict__.update(self.__dict__)
                sub.toks, sub.i = inner, 0
                root = _HList(sub.expr(font), user=self.user)
            return self.sqrt(self.group(font), root)
        if val in ('overline', 'underline'):
            return self.overline(self.group(font), under=val == 'underline')
        if val in _ACCENTS:
            return self.accent(val, self.group(font))
        if val in _FUNCTIONS:
            return self.function(val, loc)
        return self.literal('\\' + val, font)       # unknown: as written

    # -- matplotlib's _mathtext constructions
    def group(self, font):
        """A command's argument: {...} or one atom."""
        a = self.atom(font)
        return a if a is not None else _HList([])

    def thickness(self):
        return 0.75 / 12 * self.size

    def genfrac(self, num, den, rule=True, display=False):
        """matplotlib's _genfrac (text style; dfrac: display)."""
        th = self.thickness()
        axis = _AXIS * self.size                        # axis_height, px (per em)
        xh = _XH * self.size
        if not display:                             # TEXTSTYLE: both shrink once
            xh *= SHRINK
            _shrink(num)
            _shrink(den)
        if not isinstance(num, _HList):
            num = _HList([num], user=self.user)
        if not isinstance(den, _HList):
            den = _HList([den], user=self.user)
        width = max(num.width, den.width)
        cnum, cden = _centered(num, width), _centered(den, width)
        if rule:
            r = th
            if display:
                up, down, clr = 1529.86 / 1120 * xh, 1548.29 / 1120 * xh, 3 * r
            else:
                up, down, clr = _NUM2 * xh, _DENOM2 * xh, r
            delta = r / 2
            num_clr = max((up - cnum.depth) - (axis + delta), clr)
            den_clr = max((axis - delta) - (cden.height - down), clr)
            v = _VList([cnum, _Kern(num_clr), _Rule(0.0, r * 0.5, r * 0.5), _Kern(den_clr), cden])
            v.shift = cden.height + den_clr + delta - axis
        else:
            up, down, mn = _NUM3 * xh, _DENOM2 * xh, 3 * th
            def_clr = (up - cnum.depth) - (cden.height - down)
            clr = max(def_clr, mn)
            v = _VList([cnum, _Kern(clr), cden])
            v.shift = down + ((mn - def_clr) / 2 if def_clr < mn else 0.0)
        return _HList([_Kern(th), v, _Kern(th)], kern=False)

    def sqrt(self, body, root=None):
        """matplotlib's sqrt: a radical sign stretched to the body, a rule
        over it, an optional root index up-left."""
        th = self.thickness()
        if not isinstance(body, _HList):
            body = _HList([body], user=self.user)
        height = body.height - body.shift + 5 * th
        depth = body.depth + body.shift
        # AutoHeightChar: DejaVu Sans's sign if tall enough, else the first
        # STIX size that is (the largest's metrics are off), scaled to fit
        target = height + depth
        xh = 1120 / 2048 * self.size
        for fk in ('rm', 's1', 's2', 's3', 's4'):
            check = _Char('√', fk, self.size, self.user)
            if check.height + check.depth >= target - 0.2 * xh:
                break
        shift = 0.0
        if fk != 'rm':
            check = _Char('√', fk, self.size * target / (check.height + check.depth),
                          self.user)
            shift = depth - check.depth                # the sign's bottom at the body's
        check.width = check.advance
        sign = _HList([check], kern=False)
        sign.shift = shift
        height, depth = check.height - shift, check.depth + shift
        top = height + 0.06 * self.size                 # vpack's fontsize * dpi / 1200
        padded = _HList([_Kern(2 * th), body, _Kern(2 * th)], kern=False)
        bar = _Rule(0.0, th * 0.5, th * 0.5)
        right = _VList([bar, _Kern(max(top - th - padded.height, 0.0)), padded])
        right.shift = 0.0
        right.height, right.depth = top, max(depth, padded.depth)
        parts = []
        if root is not None:
            _shrink(root)
            _shrink(root)
            rv = _HList([root], kern=False)
            rv.shift = -height * 0.6
            parts += [rv, _Kern(-0.5 * check.width)]
        else:
            parts += [_Kern(0.5 * check.width), _Kern(-0.5 * check.width)]
        return _HList(parts + [sign, right], kern=False)

    def overline(self, body, under=False):
        th = self.thickness()
        if not isinstance(body, _HList):
            body = _HList([body], user=self.user)
        if under:
            v = _VList([body, _Kern(3 * th), _Rule(0.0, th * 0.5, th * 0.5)])
            d = v.height + v.depth + th
            v.height = body.height
            v.depth = d - v.height
            v.shift = 0.0
            return _HList([v], kern=False)
        height = body.height - body.shift + 3 * th
        gap = max(height - th - body.height, 0.0)
        v = _VList([_Rule(0.0, th * 0.5, th * 0.5), _Kern(gap), body])
        v.height, v.depth = height, body.depth + body.shift
        return _HList([v], kern=False)

    def accent(self, name, sym):
        """matplotlib's accent: the mark centred over the symbol (narrow
        ones a quarter of its width on), two rule thicknesses above it."""
        th = self.thickness()
        if not isinstance(sym, _HList):
            sym = _HList([sym], user=self.user)
        mark = _Char(_ACCENTS[name], 'rm', self.size, self.user)
        f = _math_font('rm', self.size, self.user)
        x0, y0, x1, y1 = f.unhinted_box(mark.c)
        mark.width = x1 - x0                            # Accent: its ink, from its left edge
        mark.height, mark.depth = y1 - y0, 0.0
        if name == 'mathring':
            _shrink(mark)
            _shrink(mark)
        acc = _HList([_Kern(-x0), mark], kern=False) if x0 else _HList([mark], kern=False)
        acc.shift = y0                                  # drawn with its ink's bottom on the line
        lift = _HList([acc], kern=False)
        lift.width, lift.height, lift.depth = mark.width, mark.height, 0.0
        if name in ('widehat', 'widetilde', 'widebar'):
            c = _centered(lift, sym.width)
        else:
            c = _centered(_HList([_Kern(sym.width / 4.0), lift], kern=False), sym.width)
        v = _VList([c, _Kern(2 * th), sym])
        v.shift = 0.0
        return _HList([v], kern=False)

    def function(self, name, loc):
        """\\sin, \\log ...: the name in roman, a thin space after it unless
        a delimiter (or a script) follows (matplotlib's operatorname)."""
        chars = [self.char(c, 'rm') for c in name]
        nxt = self.peek()
        follows = nxt[1] if nxt[0] in ('sym', '{', '}', '^', '_') else ''
        if nxt[0] == 'cmd' and nxt[1] in ('{', '}', '|'):
            follows = nxt[1]
        if follows not in _DELIMS | {'^', '_'} and name not in _OVERUNDER_FUNCTIONS:
            chars.append(self.space(_SPACES[',']))
        self.needs_space = follows in ('^', '_')
        h = _FuncList(chars, user=self.user)
        h.function_name = name
        return h

    def overunder(self, nucleus, sub, sup):
        """matplotlib's limits over and under \\sum, \\prod, \\lim ..."""
        rule = 0.75 / 12 * self.size
        width = nucleus.width
        if sup is not None:
            _shrink(sup)
            width = max(width, sup.width)
        if sub is not None:
            _shrink(sub)
            width = max(width, sub.width)
        vgap = rule * 3.0
        items = []
        if sup is not None:
            items += [_centered(sup, width), _Kern(vgap)]
        n = nucleus if isinstance(nucleus, _HList) else _HList([nucleus], user=self.user)
        items.append(_centered(n, width))
        shift = 0.0
        if sub is not None:
            cs = _centered(sub, width)
            items += [_Kern(vgap), cs]
            shift = cs.height + vgap + n.depth
        v = _VList(items)
        v.shift = shift
        return _HList([v], kern=False)

    def subsuper(self, font):
        nucleus = self.atom(font)
        sub = sup = None
        while self.peek()[0] in ('^', '_'):
            op = self.peek()[0]
            self.i += 1
            arg = self.atom(font)
            if arg is None:
                arg = _HList([])
            if op == '_' and sub is None:
                sub = arg
            elif op == '^' and sup is None:
                sup = arg
        if sub is None and sup is None:
            return nucleus if nucleus is not None else _HList([])
        if nucleus is None:
            nucleus = _HList([], kern=False)
            nucleus.width = 0.0
        if (isinstance(nucleus, _Char) and nucleus.c in _OVERUNDER) or \
                getattr(nucleus, 'function_name', None) in _OVERUNDER_FUNCTIONS:
            out = self.overunder(nucleus, sub, sup)
        else:
            out = self.scripts(nucleus, sub, sup)
        if self.needs_space:                            # \log_{10}, \lim_{x}: a thin space after
            out = _HList([out, self.space(_SPACES[','])], kern=False)
        self.needs_space = False
        return out

    def scripts(self, nucleus, sub, sup):
        """matplotlib's Parser.subsuper for a nucleus that is neither an
        over/under nor a drop-sub symbol."""
        size = self.size
        rule = 0.75 / 12 * size
        xh = _XH * size
        last = nucleus
        if isinstance(nucleus, _HList):
            ch = nucleus.children
            if ch:
                if len(ch) >= 2 and isinstance(ch[-1], _Kern) and isinstance(ch[-2], _Char):
                    ch = ch[:-1]
                last = ch[-1]
                if isinstance(last, _Char):
                    last.width = last.advance
            nucleus = _HList(ch, kern=False)
        else:
            if isinstance(nucleus, _Char):
                last.width = last.advance
            nucleus = _HList([nucleus], user=self.user)
        lc_height = last.height
        superkern = subkern = _DELTA * xh
        if isinstance(last, _Char) and last.slanted:
            superkern += _DELTA * xh
            superkern += _DELTA_SLANTED * (lc_height - xh * 2 / 3)
            subkern = 0.0
        if _is_char_node(nucleus):
            shift_up = shift_down = 0.0
        else:
            sxh = _XH * size * SHRINK
            shift_up = nucleus.height - _SUPDROP * sxh
            shift_down = nucleus.depth + _SUBDROP * sxh
        if sup is None:
            x = _HList([_Kern(subkern), sub], user=self.user)
            _shrink(x)
            shift_down = max(shift_down, _SUB1 * xh, x.height - xh * 4 / 5)
            x.shift = shift_down
        else:
            x = _HList([_Kern(superkern), sup], user=self.user)
            _shrink(x)
            shift_up = max(shift_up, _SUP1 * xh, x.depth + xh / 4)
            if sub is None:
                x.shift = -shift_up
            else:
                y = _HList([_Kern(subkern), sub], user=self.user)
                _shrink(y)
                shift_down = max(shift_down, _SUB2 * xh)
                clr = 4 * rule - ((shift_up - x.depth) - (y.height - shift_down))
                if clr > 0.0:
                    shift_down += clr
                    clr = xh * 4 / 5 - shift_up + x.depth
                    if clr > 0:
                        shift_up += clr
                        shift_down -= clr
                x = _VList([x, _Kern((shift_up - x.depth) - (y.height - shift_down)), y])
                x.shift = shift_down
        x.width += _SCRIPT_SPACE * xh
        return _HList([nucleus, x], user=self.user)


def _ship(box, user):
    """matplotlib's ship(): (Font, char, x, y) of every glyph, y down
    from the box's baseline."""
    out = []

    def hlist(b, h, v):
        for p in b.children:
            if isinstance(p, _Char):
                out.append((_math_font(p.fk, p.size, user), p.c, h, v))
                h += p.width
            elif isinstance(p, _Kern):
                h += p.width
            elif isinstance(p, _Rule):
                out.append((_RULE, (p.width, p.height + p.depth), h, v - p.height))
                h += p.width
            elif isinstance(p, _HList):
                hlist(p, h, v + p.shift)
                h += p.width
            else:
                vlist(p, h, v + p.shift)
                h += p.width

    def vlist(b, h, v):
        v -= b.height
        for p in b.children:
            if isinstance(p, _Kern):
                v += p.width
            elif isinstance(p, _Rule):                  # as wide as the list
                out.append((_RULE, (b.width, p.height + p.depth), h, v))
                v += p.height + p.depth
            else:
                v += p.height
                hlist(p, h + p.shift, v)
                v += p.depth

    hlist(box, 0.0, 0.0)
    return out


def _split_math(s):
    """Text and math parts of a mathtext line: [(is_math, str), ...]."""
    parts = []
    buf = []
    math_mode = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == '\\' and i + 1 < len(s) and s[i + 1] == '$':
            buf.append('\\$' if math_mode else '$')
            i += 2
            continue
        if c == '$':
            parts.append((math_mode, ''.join(buf)))
            buf = []
            math_mode = not math_mode
        else:
            buf.append(c)
        i += 1
    parts.append((math_mode, ''.join(buf)))
    return parts


def _is_math(s):
    n = s.count('$') - s.count('\\$')
    return n > 0 and n % 2 == 0


def _math_line(fnt, s, size=None):
    """A mathtext line as matplotlib's 'path' parser lays it out: its
    metrics (ceiled, as VectorParse gives them) and its glyphs. `size`
    is the nominal size in px (fontsize * dpi / 72), which matplotlib's
    layout constants scale with; glyph metrics come from the font at
    FreeType's size."""
    if size is None:
        size = fnt.size_px
    items = []
    for is_math, part in _split_math(s):
        if is_math:
            p = _MathParser(part, size, fnt)
            nodes = p.expr('normal')
            while p.peek()[0] == '}':                  # unbalanced: skip the brace
                p.i += 1
                nodes += p.expr('normal')
            items.append(_HList(nodes, user=fnt))
        else:
            items.append(_HList([_Char(c, 'default', size, fnt) for c in part], user=fnt))
    box = _HList(items, user=fnt)
    W, H, D = box.width, box.height, box.depth
    lift = math.ceil(H) - H                     # VectorParse's baseline, ceil(h) below the top
    glyphs = tuple((f, c, x, y - lift) for f, c, x, y in _ship(box, fnt))
    return _Line(glyphs, float(math.ceil(W)), float(math.ceil(H)), float(math.ceil(D)))


# ---------------------------------------------------------------- layout

class _Layout(NamedTuple):
    lines: tuple        # _Line per line
    xs: tuple           # line origins x (multialignment offsets), px
    ys: tuple           # baselines, matplotlib's y (up, box top = 0), px
    width: float
    descent: float      # the last line's descent
    baseline: float     # from the box top to the last line's baseline


def _norm_ls(linespacing):
    if linespacing is None or linespacing == 'normal':
        return 'normal'
    return float(linespacing)


@lru_cache(maxsize=LAYOUT_CACHE)
def _layout(text, size, bold, italic, family, linespacing, ma):
    fnt = _font(_font_path(bold, italic, family), _size64(size))
    lines = text.split('\n')
    min_a, min_d, gap = (v * size for v in fnt.face.typo)   # matplotlib: the nominal size
    if len(lines) == 1:
        gap = 0.0
    out = []
    ys = []
    thisy = 0.0
    baseline = 0.0
    d = 0.0
    for s in lines:
        if not s:
            ln = _Line((), 0.0, 0.0, 0.0)
        elif _is_math(s):
            ln = _math_line(fnt, s, size)
        else:
            ln = _plain_line(fnt, s.replace('\\$', '$'))
        a, d = ln.ascent, ln.descent
        if linespacing == 'normal':
            a = max(a, min_a) + gap / 2
            d = max(d, min_d) + gap / 2
        else:
            lead = linespacing * (min_a + min_d) - (a + d)
            a += lead / 2
            d += lead / 2
        baseline = a - thisy
        thisy -= a
        out.append(ln)
        ys.append(thisy)
        thisy -= d
    width = max(ln.width for ln in out)
    if ma == 'center':
        xs = tuple(width / 2 - ln.width / 2 for ln in out)
    elif ma == 'right':
        xs = tuple(width - ln.width for ln in out)
    else:
        xs = (0.0,) * len(out)
    return _Layout(tuple(out), xs, tuple(ys), width, d, baseline)


def _rot(angle, x, y):
    if angle == 0:
        return x, y
    r = math.radians(angle)
    c, s = math.cos(r), math.sin(r)
    return c * x - s * y, s * x + c * y


def _geometry(lay, ha, va, rotation, rotation_mode):
    """matplotlib's alignment: the rotated box's bounds and the anchor
    (first line's baseline, box left), both relative to the text's
    position, in matplotlib's y-up pixels."""
    xmin, xmax, ymax, ymin = 0.0, lay.width, 0.0, lay.ys[-1] - lay.descent
    corners = [_rot(rotation, x, y) for x, y in
               ((xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin))]
    cx = [c[0] for c in corners]
    cy = [c[1] for c in corners]
    rx0, rx1, ry0, ry1 = min(cx), max(cx), min(cy), max(cy)
    if rotation_mode != 'anchor':
        ox = rx0 if ha == 'left' else rx1 if ha == 'right' else (rx0 + rx1) / 2
        oy = (ry0 if va == 'bottom' else ry1 if va == 'top' else
              (ry0 + ry1) / 2 if va == 'center' else
              ry0 + lay.descent if va == 'baseline' else
              ry0 + (ry1 - ry0) - lay.baseline / 2)          # center_baseline
    else:
        ox = xmin if ha == 'left' else xmax if ha == 'right' else (xmin + xmax) / 2
        oy = (ymin if va == 'bottom' else ymax if va == 'top' else
              (ymin + ymax) / 2 if va == 'center' else
              ymax - lay.baseline if va == 'baseline' else
              ymax - lay.baseline / 2)
        ox, oy = _rot(rotation, ox, oy)
    ax, ay = _rot(rotation, 0.0, lay.ys[0])
    return (rx0 - ox, ry0 - oy, rx1 - ox, ry1 - oy), (ax - ox, ay - oy)


_HA = ('left', 'center', 'right')
_VA = ('top', 'center', 'baseline', 'center_baseline', 'bottom')


def _check(ha, va, rotation_mode):
    if ha not in _HA:
        raise ValueError('ha must be one of %s, not %r' % (_HA, ha))
    if va not in _VA:
        raise ValueError('va must be one of %s, not %r' % (_VA, va))
    if rotation_mode not in ('default', 'anchor', None):
        raise ValueError("rotation_mode must be 'default' or 'anchor', not %r" % (rotation_mode,))


def _size64(size_px):
    return max(int(math.floor(size_px * 64 + 0.5)), 1)


def _key(size_px, weight, style, family, linespacing):
    if not size_px > 0:
        raise ValueError('size_px must be positive, not %r' % (size_px,))
    fam = family_key(family)
    if fam is not None and len(fam) >= 1 and fam[0].lower() in _BUNDLED_NAMES:
        fam = None                                      # the bundled DejaVu Sans
    return (float(size_px), _is_bold(weight), _is_italic(style), fam, _norm_ls(linespacing))


def _rotation(rotation):
    r = float(rotation) % 360.0
    return 0.0 if r == 360.0 else r


def _locate(text, x, y, size_px, ha, va, rotation, rotation_mode, weight, style, family,
            linespacing, ma=None):
    _check(ha, va, rotation_mode)
    size, b, i, fam, ls = _key(size_px, weight, style, family, linespacing)
    rot = _rotation(rotation)
    lay = _layout(text, size, b, i, fam, ls, _ma(text, ha, ma))
    (x0, y0, x1, y1), (ax, ay) = _geometry(lay, ha, va, rot, rotation_mode)
    # y up -> canvas y down
    return (x + x0, y - y1, x + x1, y - y0), (x + ax, y - ay), (size, b, i, fam, ls, rot)


def _ma(text, ha, ma):
    """The lines' alignment of a text: its multialignment, else its ha."""
    if '\n' not in text:
        return 'left'
    return ha if ma is None else ma


def text_extent(text, x, y, size_px, ha='left', va='baseline', rotation=0,
                rotation_mode='default', weight='normal', style='normal', family=None,
                linespacing=None, ma=None):
    """The (x0, y0, x1, y1) canvas box (y down, float px) text drawn at
    (x, y) occupies -- matplotlib's Text.get_window_extent -- without
    drawing it. `linespacing` None is matplotlib's 'normal' (its 3.11
    default); a number is its fixed spacing, in font heights."""
    text = str(text)
    if not text:
        return (float(x), float(y), float(x), float(y))
    return _locate(text, x, y, size_px, ha, va, rotation, rotation_mode, weight, style,
                   family, linespacing, ma)[0]


# ---------------------------------------------------------------- masks

class TextMask(NamedTuple):
    """A text's coverage, uint8 (h, w), and where its anchor -- the left
    end of the first line's baseline before rotation -- falls in it
    (float px, from the mask's top-left corner)."""
    alpha: np.ndarray
    ax: float
    ay: float


def _subpixel(g, f):
    """Glyph coverage g (float32, h x w) moved right by f (0 < f < 1)
    px: (h, w + 1). Each pixel's ink is taken to be one run of its
    coverage's length, against the side of its darker neighbour (an
    edge) or centred (a thin stroke), and the runs are moved and
    re-integrated over the pixels -- the area coverage FreeType gives the
    outline drawn at that offset, as matplotlib draws it (its hinter
    leaves x alone): about 1/1000 from it on average for DejaVu Sans,
    where linear weights would blur stems by a further pixel. Moved down
    (mathtext's fractional baselines), the same, of the vertically hinted
    glyph, as FreeType moves it."""
    h, w = g.shape
    left = np.zeros_like(g)
    left[:, 1:] = g[:, :-1]
    right = np.zeros_like(g)
    right[:, :-1] = g[:, 1:]
    one = np.float32(1)
    o = np.where(left > right, np.float32(0),
                 np.where(right > left, one - g, (one - g) * np.float32(0.5)))
    stay = np.clip(one - (o + np.float32(f)), 0, g)
    out = np.zeros((h, w + 1), np.float32)
    out[:, :w] = stay
    out[:, 1:] += g - stay
    return out


def _render(lay):
    """The upright mask of a layout: every glyph at its pen position, to
    FreeType's 1/64 px as matplotlib places it (fractional in x, and in y
    within mathtext; lines on whole pixels), coverage composited "over"
    as Agg composites glyphs.

    Pillow renders a glyph on whole pixels only, so each is rendered once
    at its integer origin and moved by the fraction (_subpixel)."""
    y_first = lay.ys[0]
    place_ = []
    gx0 = gy0 = math.inf
    gx1 = gy1 = -math.inf
    for ln, lx, ly in zip(lay.lines, lay.xs, lay.ys):
        dy = float(math.floor(y_first - ly + 0.5))
        for f, c, x, y in ln.glyphs:
            if f is _RULE:                              # Agg's draw_rect_filled, as matplotlib
                (w, hh), x1, y1 = c, lx + x, dy + y     # rounds a mathtext rule
                height = max(int(hh) - 1, 0)
                yy = int(y1 + hh / 2.0 - (height + 1) / 2.0) if height == 0 else int(y1)
                g = np.ones((height + 1, max(int(math.ceil(x1 + w)) - int(x1), 1)), np.float32)
                l = t = 0
                px, py = float(int(x1)), float(yy)
            else:
                g, l, t = f.glyph(c)
                if g is None:
                    continue
                px = math.floor((lx + x) * 64 + 0.5) / 64
                py = math.floor((dy + y) * 64 + 0.5) / 64
            place_.append((g, l, t, px, py))
            gx0 = min(gx0, px + l)
            gx1 = max(gx1, px + l + g.shape[1])
            gy0 = min(gy0, py + t)
            gy1 = max(gy1, py + t + g.shape[0])
    if not place_:
        return TextMask(np.zeros((0, 0), np.uint8), 0.0, 0.0)
    ox = math.floor(gx0) - 1
    oy = math.floor(gy0) - 1
    w = int(math.ceil(gx1)) - ox + 2
    h = int(math.ceil(gy1)) - oy + 2
    acc = np.zeros((h, w), np.float32)
    for g, l, t, px, py in place_:
        ix, iy = math.floor(px), math.floor(py)
        fx, fy = np.float32(px - ix), np.float32(py - iy)
        gh, gw = g.shape
        if fx:
            s = _subpixel(g, fx)
        else:
            s = np.zeros((gh, gw + 1), np.float32)
            s[:, :gw] = g
        if fy:
            s = _subpixel(s.T, fy).T
        else:
            s = np.concatenate([s, np.zeros((1, gw + 1), np.float32)])
        x0, y0 = ix + l - ox, iy + t - oy
        a = acc[y0:y0 + gh + 1, x0:x0 + gw + 1]
        a += s - a * s
    m = np.floor(acc * 255 + 0.5).astype(np.uint8)
    rows = np.flatnonzero(m.any(1))
    if not len(rows):
        return TextMask(np.zeros((0, 0), np.uint8), 0.0, 0.0)
    cols = np.flatnonzero(m.any(0))
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    m = np.ascontiguousarray(m[r0:r1, c0:c1])
    return TextMask(m, float(-ox - c0), float(-oy - r0))


def _rotate(tm, rot, exact=True):
    """The mask turned `rot` degrees counter-clockwise: exactly (array
    rotation) for multiples of 90, else bicubic, its anchor carried."""
    a, ax, ay = tm
    h, w = a.shape
    if rot == 0 or a.size == 0:
        return tm
    if exact and rot in (90.0, 180.0, 270.0):
        k = int(rot // 90)
        r = np.ascontiguousarray(np.rot90(a, k))
        if k == 1:
            return TextMask(r, ay, w - ax)
        if k == 2:
            return TextMask(r, w - ax, h - ay)
        return TextMask(r, h - ay, ax)
    from PIL import Image
    t = math.radians(rot)
    c, s = math.cos(t), math.sin(t)
    # forward (y down): (dx, dy) -> (c dx + s dy, -s dx + c dy)
    xs = [c * x + s * y for x, y in ((0, 0), (w, 0), (w, h), (0, h))]
    ys = [-s * x + c * y for x, y in ((0, 0), (w, 0), (w, h), (0, h))]
    tx = 2 - math.floor(min(xs))
    ty = 2 - math.floor(min(ys))
    nw = int(math.ceil(max(xs))) + tx + 2
    nh = int(math.ceil(max(ys))) + ty + 2
    # inverse, output -> input: (X - tx, Y - ty) -> (c u - s v, s u + c v)
    m = (c, -s, -(c * tx - s * ty), s, c, -(s * tx + c * ty))
    out = Image.fromarray(a).transform((nw, nh), Image.Transform.AFFINE, m,
                                       resample=Image.Resampling.BICUBIC)
    r = np.asarray(out)
    nax, nay = c * ax + s * ay + tx, -s * ax + c * ay + ty
    rows = np.flatnonzero(r.any(1))
    if not len(rows):
        return TextMask(np.zeros((0, 0), np.uint8), 0.0, 0.0)
    cols = np.flatnonzero(r.any(0))
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    return TextMask(np.ascontiguousarray(r[r0:r1, c0:c1]), nax - c0, nay - r0)


@lru_cache(maxsize=MASK_CACHE)
def _mask(text, size, bold, italic, family, linespacing, ma, rot):
    if rot != 0:
        return _rotate(_mask(text, size, bold, italic, family, linespacing, ma, 0.0), rot)
    tm = _render(_layout(text, size, bold, italic, family, linespacing, ma))
    tm.alpha.setflags(write=False)
    return tm


def mask(text, size_px, rotation=0, weight='normal', style='normal', family=None,
         linespacing=None, ma='left'):
    """The (LRU-cached) TextMask of a text: its coverage, uint8 (h, w),
    and its anchor in it. `ma` aligns the lines of a multi-line text
    (left, center, right; draw_text uses its `ha`). The array is shared
    by the cache: read-only."""
    text = str(text)
    size, b, i, fam, ls = _key(size_px, weight, style, family, linespacing)
    if ma not in _HA:
        raise ValueError('ma must be one of %s, not %r' % (_HA, ma))
    if not text:
        return TextMask(np.zeros((0, 0), np.uint8), 0.0, 0.0)
    return _mask(text, size, b, i, fam, ls, ma if '\n' in text else 'left',
                 _rotation(rotation))


def place(text, x, y, size_px, ha='left', va='baseline', rotation=0, rotation_mode='default',
          weight='normal', style='normal', family=None, linespacing=None, ma=None):
    """Everything draw_text does but the blend: (TextMask, x0, y0,
    extent) -- the cached mask, the integer canvas position of its
    top-left pixel (the anchor rounded to whole pixels), and the text's
    (x0, y0, x1, y1) extent. blend_mask(canvas, tm.alpha, x0, y0, rgba),
    or the CUDA kernel on a GPU canvas, then draws exactly what
    draw_text draws."""
    text = str(text)
    if not text:
        e = (float(x), float(y), float(x), float(y))
        return TextMask(np.zeros((0, 0), np.uint8), 0.0, 0.0), 0, 0, e
    ext, (ax, ay), (size, b, i, fam, ls, rot) = _locate(
        text, x, y, size_px, ha, va, rotation, rotation_mode, weight, style, family,
        linespacing, ma)
    if _FT is not None:                                 # matplotlib 3.11's way: each glyph
        ix, iy = math.floor(ax), math.floor(ay)         # by FreeType at its exact place
        fx, fy = round((ax - ix) * 64), round((ay - iy) * 64)
        tm = _ft_mask(text, size, b, i, fam, ls, _ma(text, ha, ma), rot, fx, fy)
        return tm, ix + int(tm.ax), iy + int(tm.ay), ext
    tm = _mask(text, size, b, i, fam, ls, _ma(text, ha, ma), rot)
    x0 = int(math.floor(ax - tm.ax + 0.5))
    y0 = int(math.floor(ay - tm.ay + 0.5))
    return tm, x0, y0, ext


@lru_cache(maxsize=MASK_CACHE)
def _ft_mask(text, size, bold, italic, family, linespacing, ma, rot, fx64, fy64):
    """A text's coverage as matplotlib 3.11's Agg draws it
    (_draw_text_glyphs_and_boxes): every glyph rendered by FreeType with
    the rotation and its exact pen (1/64 px) as the face's transform, put
    down at its bitmap's left / top; a mathtext rule a box, snapped to whole
    pixels and at least one wide. The anchor -- the first line's baseline,
    box left -- sits fx64 / 64, fy64 / 64 px into the integer point the
    returned TextMask's (ax, ay) -- the mask's top-left, about that point,
    as integers -- are measured from."""
    lay = _layout(text, size, bold, italic, family, linespacing, ma)
    th = math.radians(rot)
    c, s = math.cos(th), math.sin(th)
    mat = (int(round(0x10000 * c)), int(round(-0x10000 * s)), int(round(0x10000 * s)),
           int(round(0x10000 * c)))
    fx, fy = fx64 / 64.0, fy64 / 64.0
    y_first = lay.ys[0]
    pieces = []
    for ln, lx, ly in zip(lay.lines, lay.xs, lay.ys):
        dyl = y_first - ly
        for f, ch, x, y in ln.glyphs:
            u, v = lx + x, dyl + y                      # from the anchor, y down, unrotated
            if f is _RULE:
                (w, hh) = ch
                if w < 1:                               # thin bars grow symmetrically
                    u -= (1 - w) / 2
                    w = 1.0
                if hh < 1:
                    v -= (1 - hh) / 2
                    hh = 1.0
                X0, Y0 = fx + u * c + v * s, fy - u * s + v * c
                x0, y0 = math.floor(X0 + 0.5), math.floor(Y0 + 0.5)
                x1, y1 = math.floor(X0 + w + 0.5), math.floor(Y0 + hh + 0.5)
                if x1 > x0 and y1 > y0:
                    pieces.append((np.ones((y1 - y0, x1 - x0), np.float32), x0, y0))
                continue
            X, Y = fx + u * c + v * s, fy - u * s + v * c
            dx, dy = round(64 * X), round(-64 * Y)     # FreeType's y is up
            g, left, top = _ft_glyph(f, ch, mat, dx & 63, dy & 63)
            if g is not None:
                pieces.append((g, (dx >> 6) + left, -((dy >> 6) + top)))
    if not pieces:
        return TextMask(np.zeros((0, 0), np.uint8), 0.0, 0.0)
    c0 = min(p[1] for p in pieces)
    r0 = min(p[2] for p in pieces)
    c1 = max(p[1] + p[0].shape[1] for p in pieces)
    r1 = max(p[2] + p[0].shape[0] for p in pieces)
    acc = np.zeros((r1 - r0, c1 - c0), np.float32)
    for g, cc, rr in pieces:                            # each over the others, as Agg blends
        a = acc[rr - r0:rr - r0 + g.shape[0], cc - c0:cc - c0 + g.shape[1]]
        a += g - a * g
    m = np.floor(acc * 255 + 0.5).astype(np.uint8)
    m.setflags(write=False)
    return TextMask(m, float(c0), float(r0))


_FT_GLYPHS = {}
_FT_LOCK = threading.Lock()


def _ft_glyph(f, ch, mat, fx64, fy64):
    """One glyph as FreeType renders it under matrix mat with its origin
    moved (fx64, fy64) / 64 px (y up): coverage float32 0..1 and its
    bitmap's left and top (px; top up), None when it has no ink. Cached;
    FreeType's faces are shared, so rendering holds a lock."""
    key = (f.path, f.size64, ch, mat, fx64, fy64)
    v = _FT_GLYPHS.get(key)
    if v is None:
        with _FT_LOCK:
            face = _ft_face(f.path)
            face.set_char_size(f.size64, f.size64, 72, 72)
            face.set_transform(_FT.Matrix(*mat), _FT.Vector(fx64, fy64))
            gid = face.get_char_index(ord(ch)) if len(ch) == 1 else 0
            face.load_glyph(gid, _FT.FT_LOAD_DEFAULT)
            face.glyph.render(_FT.FT_RENDER_MODE_NORMAL)
            bm = face.glyph.bitmap
            if bm.width and bm.rows:
                a = np.array(bm.buffer, np.uint8).reshape(bm.rows, bm.pitch)[:, :bm.width]
                v = (a.astype(np.float32) / np.float32(255), int(face.glyph.bitmap_left),
                     int(face.glyph.bitmap_top))
            else:
                v = (None, 0, 0)
            face.set_transform(_FT.Matrix(0x10000, 0, 0, 0x10000), _FT.Vector(0, 0))
        if len(_FT_GLYPHS) > 100000:                    # (bounded: a long run's variety)
            _FT_GLYPHS.clear()
        _FT_GLYPHS[key] = v
    return v


def grow(alpha, radius):
    """A uint8 coverage mask grown by `radius` px (a stroke round the text),
    anti-aliased: each pixel the largest coverage within the radius, faded
    over the last pixel. Returns the grown mask and how far it was padded
    on every side."""
    r = max(float(radius), 0.0)
    pad = int(math.ceil(r)) + 1
    a = alpha.astype(np.float32) / np.float32(255.0)
    h, w = a.shape
    out = np.zeros((h + 2 * pad, w + 2 * pad), np.float32)
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            weight = min(max(r + 0.5 - math.hypot(dx, dy), 0.0), 1.0)
            if weight > 0:
                view = out[pad + dy:pad + dy + h, pad + dx:pad + dx + w]
                np.maximum(view, a * np.float32(weight), out=view)
    return (out * np.float32(255.0) + np.float32(0.5)).astype(np.uint8), pad


def _rgba(color, alpha=None):
    """A colour as float32 [r, g, b, a] in whole 0..255 steps: uint8 or
    integer values as they are, floats as 0..1; `alpha` (0..1) replaces
    the colour's alpha, as matplotlib's Text alpha does."""
    arr = np.asarray(color)
    if arr.shape not in ((3,), (4,)):
        raise ValueError('colour must be 3 or 4 numbers, not %r' % (color,))
    if arr.dtype.kind == 'f':
        v = np.clip(arr.astype(np.float64), 0, 1) * 255.0
    else:
        v = np.clip(arr.astype(np.float64), 0, 255)
    if len(v) == 3:
        v = np.append(v, 255.0)
    if alpha is not None:
        v[3] = min(max(float(alpha), 0.0), 1.0) * 255.0
    return np.floor(v + 0.5).astype(np.float32)


@njit(cache=True, nogil=True)
def blend_mask_kernel(canvas, mask, x0, y0, r, g, b, a):
    """The "over" blend of internals.md, coverage m = mask / 255, of the
    colour (r, g, b, a) (float32, 0..255) through `mask` placed with its
    top-left pixel at (x0, y0); what falls off the canvas is skipped."""
    H, W = canvas.shape[0], canvas.shape[1]
    mh, mw = mask.shape
    i0 = max(0, -y0)
    i1 = min(mh, H - y0)
    j0 = max(0, -x0)
    j1 = min(mw, W - x0)
    inv = np.float32(255.0)
    one = np.float32(1.0)
    half = np.float32(0.5)
    fa = a / inv
    for i in range(i0, i1):
        y = y0 + i
        for j in range(j0, j1):
            mv = mask[i, j]
            if mv == 0:
                continue
            x = x0 + j
            m = np.float32(mv) / inv
            f = fa * m
            k = one - f
            da = np.float32(canvas[y, x, 3])
            if da == inv:                       # over an opaque pixel
                canvas[y, x, 0] = np.uint8(r * f + np.float32(canvas[y, x, 0]) * k + half)
                canvas[y, x, 1] = np.uint8(g * f + np.float32(canvas[y, x, 1]) * k + half)
                canvas[y, x, 2] = np.uint8(b * f + np.float32(canvas[y, x, 2]) * k + half)
                canvas[y, x, 3] = np.uint8(inv * f + da * k + half)
                continue
            w = (da / inv) * k                  # over a partly transparent one: _draw.blend_px
            oa = f + w
            if oa <= np.float32(0.0):
                continue
            canvas[y, x, 0] = np.uint8((r * f + np.float32(canvas[y, x, 0]) * w) / oa + half)
            canvas[y, x, 1] = np.uint8((g * f + np.float32(canvas[y, x, 1]) * w) / oa + half)
            canvas[y, x, 2] = np.uint8((b * f + np.float32(canvas[y, x, 2]) * w) / oa + half)
            canvas[y, x, 3] = np.uint8(inv * oa + half)


def blend_mask(canvas, alpha_mask, x0, y0, rgba):
    """Blend colour `rgba` (draw_text's colour forms) through a uint8
    coverage mask onto a uint8 (H, W, 4) canvas, its top-left pixel at
    integer (x0, y0), clipped to the canvas."""
    if alpha_mask.size == 0:
        return
    if canvas.dtype != np.uint8 or canvas.ndim != 3 or canvas.shape[2] != 4:
        raise ValueError('canvas must be uint8 (H, W, 4)')
    c = rgba if (isinstance(rgba, np.ndarray) and rgba.dtype == np.float32) else _rgba(rgba)
    x0, y0 = int(x0), int(y0)
    H, W = canvas.shape[:2]
    mh, mw = alpha_mask.shape
    if x0 >= W or y0 >= H or x0 + mw <= 0 or y0 + mh <= 0 or c[3] == 0:
        return
    m = np.ascontiguousarray(alpha_mask, np.uint8)
    if m.flags.writeable:                               # one type for the kernel: read-only, as
        m = m.view()                                    # the cached masks are (a view: no copy)
        m.setflags(write=False)
    blend_mask_kernel(canvas, m, x0, y0,
                      c[0], c[1], c[2], c[3])


def draw_text(canvas, text, x, y, size_px, color, ha='left', va='baseline', rotation=0,
              rotation_mode='default', weight='normal', style='normal', family=None,
              linespacing=None, alpha=None):
    """Draw text on a uint8 (H, W, 4) canvas at (x, y) (px, y down), as
    matplotlib's Text draws it, clipped to the canvas; returns its
    (x0, y0, x1, y1) extent (text_extent's). `color`: 4 (or 3) numbers,
    uint8/ints 0..255 or floats 0..1; `alpha` replaces its alpha. place()
    then blend_mask(): the two can never disagree."""
    tm, x0, y0, ext = place(text, x, y, size_px, ha, va, rotation, rotation_mode, weight,
                            style, family, linespacing)
    if tm.alpha.size:
        blend_mask(canvas, tm.alpha, x0, y0, _rgba(color, alpha))
    return ext


def cache_info():
    """(layout, mask) lru_cache statistics."""
    return _layout.cache_info(), _mask.cache_info()


def cache_clear():
    _layout.cache_clear()
    _mask.cache_clear()
    _ft_mask.cache_clear()
    _FT_GLYPHS.clear()


CUDA = r'''
extern "C" __global__ void blend_mask(unsigned char* canvas, int H, int W,
                                      const unsigned char* mask, int mh, int mw,
                                      int x0, int y0, float r, float g, float b, float a)
{
    /* _text.blend_mask: the "over" blend of a colour through a uint8
       coverage mask, one thread per mask pixel, bytes identical to the
       numba kernel (float32, no fused multiply-adds). */
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= mh || j >= mw) return;
    int y = y0 + i, x = x0 + j;
    if (y < 0 || y >= H || x < 0 || x >= W) return;
    unsigned char mv = mask[(size_t)i * mw + j];
    if (mv == 0) return;
    float m = (float)mv / 255.0f;
    float f = (a / 255.0f) * m;
    float k = 1.0f - f;
    unsigned char* p = canvas + ((size_t)y * W + x) * 4;
    float da = (float)p[3];
    if (da == 255.0f) {
        p[0] = (unsigned char)(r * f + (float)p[0] * k + 0.5f);
        p[1] = (unsigned char)(g * f + (float)p[1] * k + 0.5f);
        p[2] = (unsigned char)(b * f + (float)p[2] * k + 0.5f);
        p[3] = (unsigned char)(255.0f * f + da * k + 0.5f);
        return;
    }
    float w = (da / 255.0f) * k;
    float oa = f + w;
    if (oa <= 0.0f) return;
    p[0] = (unsigned char)((r * f + (float)p[0] * w) / oa + 0.5f);
    p[1] = (unsigned char)((g * f + (float)p[1] * w) / oa + 0.5f);
    p[2] = (unsigned char)((b * f + (float)p[2] * w) / oa + 0.5f);
    p[3] = (unsigned char)(255.0f * oa + 0.5f);
}
'''
