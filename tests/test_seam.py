"""Maps cut along a meridian that is not a fixed x (Robinson, conics): lines
crossing it are cut there, filled rings closed along it (or through the
pole they go round), and the frame is the map's own edge."""
import numpy as np
import pytest

import fastplot as fp
import fastplot.crs as ccrs
import fastplot.feature as cfeature

SEAM_PROJS = [ccrs.Robinson(central_longitude=-150), ccrs.Robinson(),
              ccrs.AlbersEqualArea(central_longitude=-150),
              ccrs.LambertConformal(central_longitude=100)]


def _ax(proj, **kw):
    fig = fp.figure(figsize=(5, 3), dpi=80)
    ax = fig.add_subplot(projection=proj)
    ax.set_global()
    return fig, ax


@pytest.mark.parametrize('proj', SEAM_PROJS, ids=lambda p: type(p).__name__)
def test_lines_cut_at_the_seam(proj):
    fig, ax = _ax(proj)
    fig.draw()
    seam = proj.central_longitude + 180
    lon = np.array([seam - 10, seam - 1, seam + 1, seam + 10, seam + 20])
    lat = np.array([10.0, 12, 14, 16, 18])
    xy, off = ax._lines_px(lon, lat, np.array([0, 5]), ccrs.PlateCarree())
    assert off.size - 1 == 2                            # cut once
    w = ax.box_f[2] - ax.box_f[0]
    for a, b in zip(off[:-1], off[1:]):
        assert np.hypot(*np.diff(xy[a:b], axis=0).T).max() < 0.2 * w


@pytest.mark.parametrize('proj', SEAM_PROJS[:3], ids=lambda p: type(p).__name__)
def test_coastlines_have_no_jumps(proj):
    fig, ax = _ax(proj)
    fig.draw()
    from fastplot import _features
    ln = _features.lines('coastline', '110m')
    xy, off = ax._lines_px(ln.xy[:, 0], ln.xy[:, 1], ln.offsets, ccrs.PlateCarree())
    step = np.hypot(*np.diff(xy, axis=0).T)
    inner = np.ones(step.size, bool)
    inner[off[1:-1] - 1] = False                        # not between two lines
    assert np.nanmax(step[inner]) < 0.25 * (ax.box_f[2] - ax.box_f[0])


@pytest.mark.parametrize('proj', [ccrs.Robinson(central_longitude=-150), ccrs.Robinson()],
                         ids=['recentred', 'centred'])
def test_land_round_the_south_pole_filled(proj):
    fig, ax = _ax(proj)
    ax.add_feature(cfeature.LAND, facecolor='red')
    img = fig.draw()
    x0, y0, x1, y1 = ax.box_f
    row = img[int(y1) - 1, int(x0 + 0.3 * (x1 - x0)):int(x0 + 0.7 * (x1 - x0))]
    red = (row[:, 0] > 200) & (row[:, 1] < 60)
    assert red.mean() > 0.95                            # Antarctica reaches the map's foot


def test_frame_is_the_maps_edge():
    fig, ax = _ax(ccrs.Robinson())
    img = fig.draw()
    x0, y0, x1, y1 = (int(round(v)) for v in ax.box_f)
    corner = img[y0 + 2, x0 + 2, :3]
    assert (corner > 250).all()                         # outside the oval: no frame there
    fig, ax = _ax(ccrs.PlateCarree())
    img = fig.draw()
    x0, y0, x1, y1 = (int(round(v)) for v in ax.box_f)
    assert (img[y0:y1, x0, :3] < 200).mean() > 0.9      # a rectangle's left side, drawn
