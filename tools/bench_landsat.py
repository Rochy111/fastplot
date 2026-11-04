"""Big images, matplotlib vs fastplot: a Landsat 8 scene (about 7,700 x
7,800 pixels a band) drawn with imshow and saved with savefig to a file,
per figure (imshow included), median of 5 after one warm-up, ms.

python tools/bench_landsat.py mpl|fp <folder>

<folder> holds the scene's B2, B3 and B4 GeoTIFFs (`*_B2.TIF`, ...), read
with Pillow. The docs' scene, S = LC08_L1TP_044034_20200114_20200127_01_T1,
is in Google's public Landsat bucket:
https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/044/034/S/S_B4.TIF
(and _B3, _B2). fastplot runs on the CPU (FASTPLOT_DEVICE=cpu) unless that
is set otherwise.
"""
import glob
import os
import statistics
import sys
import tempfile
import time

import numpy as np
from PIL import Image

lib, folder = sys.argv[1], sys.argv[2]
REPS = 5
Image.MAX_IMAGE_PIXELS = None                   # (a band is ~60 M pixels)


def band(b):
    path, = glob.glob(os.path.join(folder, f'*_{b}.TIF'))
    with Image.open(path) as im:
        return np.array(im).astype(np.uint16)


def stretched(a):
    """0..1 between the 2nd and 98th percentiles, NaN outside the scene."""
    m = a > 0
    lo, hi = np.percentile(a[m][::97], (2, 98))
    out = np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1)
    out[~m] = np.nan
    return out


B = {b: band(b) for b in ('B2', 'B3', 'B4')}
H, W = B['B4'].shape
red = stretched(B['B4'])
rgb = (np.dstack([np.nan_to_num(stretched(B[b]), nan=0.0) for b in ('B4', 'B3', 'B2')])
       * 255 + 0.5).astype(np.uint8)
rgba = np.dstack([rgb, (B['B4'] > 0).astype(np.uint8) * 255])
tight = dict(bbox_inches='tight', pad_inches=0, transparent=True)
CASES = [                                       # (name, image, dpi of a 10 x 10 in figure, savefig's)
    ('one band, 1500 x 1500 px', red, 150, {}),
    ('one band, tight + transparent, 1500 px', red, 150, tight),
    ('RGB uint8, 1500 x 1500 px', rgb, 150, {}),
    ('RGBA uint8, tight + transparent, 3000 px', rgba, 300, tight),
    (f'one band, full resolution ({W} x {H})', red, 766, tight),
]

if lib == 'mpl':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
else:
    os.environ.setdefault('FASTPLOT_DEVICE', 'cpu')
    import fastplot.pyplot as plt

out = os.path.join(tempfile.mkdtemp(), 'landsat.png')
for name, data, dpi, kw in CASES:
    times = []
    for rep in range(REPS + 1):                 # (the first warms caches and kernels)
        t = time.perf_counter()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(data, cmap='gray' if data.ndim == 2 else None, interpolation='nearest')
        ax.axis('off')
        fig.savefig(out, dpi=dpi, **kw)
        plt.close('all')
        if rep:
            times.append(time.perf_counter() - t)
    print(f'{name:44s} {statistics.median(times) * 1e3:8.1f} ms', flush=True)
