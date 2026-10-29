"""Builds fastplot/data/ne_<scale>_<name>.npz, the Natural Earth features
fastplot ships: 110m and 50m of every feature that has them (10m and the
counties are converted on first use instead -- see fastplot._features).

    python tools/make_features.py [--src DIR ...] [--scales 110m 50m]

Each shapefile is looked for as ne_<scale>_<ne name>.shp or .zip under the
--src directories (searched recursively), then in cartopy's data
directories, and else downloaded from Natural Earth
(https://naturalearth.s3.amazonaws.com/<scale>_<category>/ne_<scale>_<ne name>.zip).
It is read by fastplot's own shapefile reader -- what the 10m path uses
at run time -- so the bundled and the fetched data are made alike.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fastplot import _features as F                    # noqa: E402


def find(dirs, base):
    """The first ne_... .shp (else .zip) under dirs, or None."""
    for ext in ('.shp', '.zip'):
        for d in dirs:
            for root, _, files in os.walk(d):
                if base + ext in files:
                    return os.path.join(root, base + ext)
    return None


def source(name, scale, dirs):
    category, ne, _ = F.FEATURES[name]
    base = 'ne_%s_%s' % (scale, ne)
    path = find(dirs + F._cartopy_dirs(), base)
    if path is None:
        url = F.URL.format(scale=scale, category=category, name=ne)
        print('  downloading', url)
        return F._shp_from_zip(F._download(url), base, url), url
    with open(path, 'rb') as f:
        data = f.read()
    return (F._shp_from_zip(data, base, path) if path.endswith('.zip') else data), path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--src', nargs='*', default=[], help='directories holding Natural Earth .shp/.zip')
    ap.add_argument('--scales', nargs='*', default=['110m', '50m'])
    ap.add_argument('--out', default=os.path.join(ROOT, 'fastplot', 'data'))
    a = ap.parse_args()
    total = 0
    for scale in a.scales:
        for name, (_, _, scales) in F.FEATURES.items():
            if scale not in scales:
                continue
            data, where = source(name, scale, a.src)
            shp = F.read_shp(data)
            enc = F.encode(*F._from_shp(shp, F.kind(name)), source=os.path.basename(where))
            path = os.path.join(a.out, 'ne_%s_%s.npz' % (scale, name))
            F._save(path, enc)
            size = os.path.getsize(path)
            total += size
            feat = F.decode(enc)
            print('%-5s %-13s %8d vertices %6d parts %8.1f kB  <- %s'
                  % (scale, name, len(feat.xy), feat.parts, size / 1e3, where))
    print('total %.2f MB' % (total / 1e6))


if __name__ == '__main__':
    main()
