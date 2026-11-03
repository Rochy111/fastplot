"""What a fresh process imports to draw: none of the modules fastplot no
longer needs for a figure (Pillow, freetype-py, zipfile, concurrent.futures
...), and what replaces them gives what they gave."""
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from fastplot import _npz, _text

DATA = os.path.join(os.path.dirname(_npz.__file__), 'data')


def test_a_figure_imports_only_what_it_needs():
    code = '''
import io, json, sys
import numpy as np
import fastplot
from fastplot import crs
fig = fastplot.figure(figsize=(4, 3), dpi=80)
ax = fig.add_subplot(projection=crs.PlateCarree())
ax.contourf(np.arange(30.), np.arange(20.), np.arange(600.).reshape(20, 30), cmap='viridis')
ax.coastlines()
ax.set_title('Text $\\\\alpha$')
fig.savefig(io.BytesIO(), format='png')
print(json.dumps(dict(numba='numba' in sys.modules, loaded=sorted(
    m for m in ('PIL', 'PIL.Image', 'freetype', 'zipfile', 'concurrent.futures', 'calendar')
    if m in sys.modules))))
'''
    env = dict(os.environ, FASTPLOT_AUTOWARM='0', FASTPLOT_DEVICE='cpu', FASTPLOT_PRELOAD='0')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert not {'PIL', 'PIL.Image', 'freetype'} & set(got['loaded']), got     # (fastplot's alone)
    if not got['numba']:                                # (numba itself imports some of the rest)
        assert got['loaded'] == [], got


def test_npz_files_read_as_numpy_reads_them():
    n = 0
    for p in glob.glob(os.path.join(DATA, '*.npz')):
        a, b = np.load(p, allow_pickle=False), _npz.load(p)
        assert isinstance(b, _npz.Npz) and sorted(a.files) == sorted(b.files)
        for k in a.files:
            x, y = a[k], b[k]
            assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
            n += 1
    assert n > 50


def test_a_damaged_npz_goes_to_numpy(tmp_path):
    p = tmp_path / 'x.npz'
    np.savez_compressed(p, a=np.arange(10))
    b = bytearray(p.read_bytes())
    b[60] ^= 0xFF                                       # (inside the member: its CRC fails)
    p.write_bytes(bytes(b))
    with pytest.raises(Exception):
        _npz.load(str(p))['a']


@pytest.mark.skipif(_text._FT is None, reason='no FreeType')
def test_character_metrics_are_pillows():
    ImageFont = pytest.importorskip('PIL.ImageFont')
    fonts = sorted(glob.glob(os.path.join(DATA, 'fonts', '*.ttf')))[:6]
    n = 0
    for path in fonts:
        for size in (7.0, 10.0, 13.890625, 24.0):
            f = _text.Font(path, size)
            pil = ImageFont.truetype(path, f.size_px, layout_engine=ImageFont.Layout.BASIC)
            for ch in 'Ag,|jW%$α−.:_-xÅ':
                assert f.advance(ch) == float(pil.getlength(ch)), (path, size, ch)
                want = tuple(int(t) for t in pil.getbbox(ch, anchor='ls'))
                if want[2] <= want[0] or want[3] <= want[1]:
                    want = (0, 0, 0, 0)
                assert f.bbox(ch) == want, (path, size, ch)
                n += 1
    assert n > 300
