"""fastplot draws, lays out and names things as matplotlib 3.11 and cartopy
0.26 do; the tests below check that against their output. Older releases
(all that Python 3.10 gets) shape text, place legends, choose ticks and
name colormaps differently -- or, cartopy 0.25 with matplotlib 3.10,
fail on the reference figure itself -- so there these tests are skipped,
and the rest still run."""
import pytest

REFERENCE = {'matplotlib': (3, 11), 'cartopy': (0, 26)}

AGAINST_REFERENCE = {
    'test_cartesian.py': {'test_legend_frame', 'test_tight_layout_matches'},
    'test_color.py': {'test_cmap_extremes', 'test_foreign_cmap_objects',
                      'test_listed_and_from_list', 'test_registry_complete',
                      'test_shade_float64_data', 'test_shade_log_norms',
                      'test_shade_matches_matplotlib'},
    'test_figure.py': {'test_field_pixels_match_cartopy', 'test_layout_matches_cartopy'},
    'test_geometry.py': {'test_boxstyle_matches_matplotlib'},
    'test_skewt_quiver.py': {'test_hatches_like_matplotlib', 'test_streamlines_are_matplotlibs'},
    'test_text.py': {'test_extent_math', 'test_extent_multiline', 'test_extent_plain',
                     'test_extent_weights_and_empty', 'test_glyph_positions_match_raqm',
                     'test_mathtext_glyph_positions', 'test_pixels_like_matplotlib',
                     'test_rotated_pixels_in_place'},
    'test_ticker.py': {'test_log_formatters_on_axes', 'test_log_scales',
                       'test_log_subs_and_nonpositive', 'test_scale_transforms_match_matplotlib',
                       'test_simple_locators', 'test_symlog'},
}


def _old():
    """'matplotlib 3.10.9' (the first reference package older than fastplot's), or None."""
    import importlib
    for name, want in REFERENCE.items():
        try:
            v = importlib.import_module(name).__version__
        except ImportError:
            continue                    # the tests skip themselves
        have = tuple(int(p) for p in v.split('.')[:2] if p.isdigit())
        if have < want:
            return '%s %s' % (name, v)
    return None


def pytest_collection_modifyitems(config, items):
    old = _old()
    if old is None:
        return
    skip = pytest.mark.skip(reason='compares with matplotlib 3.11 / cartopy 0.26; here %s' % old)
    for item in items:
        if item.originalname in AGAINST_REFERENCE.get(item.path.name, ()):
            item.add_marker(skip)
