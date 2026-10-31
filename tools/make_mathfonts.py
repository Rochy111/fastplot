"""fastplot/data/mathfonts.json: what mathtext's fontsets need of matplotlib's
tables, which would otherwise need matplotlib at run time.

    python tools/make_mathfonts.py

cm        a character -> (Bakoma font, code), matplotlib's latex_to_bakoma
cm_sized  a delimiter -> its sized alternatives, (Bakoma font, code)
stix_sf   \\mathsf (and everything in 'stixsans') as ranges of code points:
          first, last, font, first code, per font class
stix_tt   \\mathtt likewise
"""
import json
import os

from matplotlib import _mathtext
from matplotlib._mathtext_data import latex_to_bakoma, stix_virtual_fonts
from matplotlib.font_manager import FontProperties
from matplotlib.ft2font import LoadFlags

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'fastplot', 'data', 'mathfonts.json')


def main():
    fonts = _mathtext.BakomaFonts(FontProperties(), LoadFlags.NO_HINTING)
    cm = {}
    for sym, (font, num) in sorted(latex_to_bakoma.items()):
        if sym.startswith('\\__'):
            continue
        try:
            c = chr(_mathtext.get_unicode_index(sym))
        except ValueError:
            continue
        cm.setdefault(c, (font, num))
    sized = {}
    for sym, alts in fonts._size_alternatives.items():
        try:
            c = chr(_mathtext.get_unicode_index(sym))
        except ValueError:
            c = {r'\__sqrt__': '√', r'\widehat': '̂', r'\widetilde': '̃'}.get(sym)
            if c is None:
                continue
        got = []
        for name, s in alts:
            font, num, _ = fonts._get_glyph(name, 'it', s)
            got.append((os.path.splitext(os.path.basename(font.fname))[0], int(num)))
        sized[c] = got
    data = {'cm': cm, 'cm_sized': sized,
            'stix_sf': stix_virtual_fonts['sf'], 'stix_tt': stix_virtual_fonts['tt']}
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=True, separators=(',', ':'), sort_keys=True)
    print(OUT, len(cm), 'symbols', len(sized), 'delimiters')


if __name__ == '__main__':
    main()
