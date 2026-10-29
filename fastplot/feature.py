"""Natural Earth features, named as cartopy.feature's:

    ax.add_feature(fastplot.feature.BORDERS, linewidth=0.5)
    ax.add_feature(fastplot.feature.STATES.with_scale('10m'), edgecolor='gray')
    ax.add_feature(fastplot.feature.NaturalEarthFeature('cultural', 'admin_2_counties', '10m'))

110m and 50m ship with fastplot; 10m is read from cartopy's cache or
downloaded once (see _features).
"""
import numpy as np

COLORS = {'land': np.array((240, 240, 220)) / 256.0,
          'land_alt1': np.array((220, 220, 220)) / 256.0,
          'water': np.array((152, 183, 226)) / 256.0}

# Natural Earth names -> fastplot's
_NE_NAMES = {'coastline': 'coastline', 'admin_0_boundary_lines_land': 'borders',
             'admin_1_states_provinces_lines': 'states',
             'admin_1_states_provinces_lakes': 'states_lakes', 'lakes': 'lakes',
             'rivers_lake_centerlines': 'rivers', 'land': 'land', 'ocean': 'ocean',
             'admin_2_counties': 'counties'}


class Feature:
    """A Natural Earth feature and how to draw it: edgecolor, facecolor
    ('none'/'never': not filled), linewidth (points), linestyle, alpha,
    zorder; scale 'auto' (by the map's extent, as cartopy) or a fixed one."""

    def __init__(self, name, scale='auto', **kwargs):
        self.name = name
        self.scale = scale
        self.kwargs = kwargs

    def with_scale(self, scale):
        return Feature(self.name, scale, **self.kwargs)

    def __repr__(self):
        return '<Feature %s %s>' % (self.name, self.scale)


def NaturalEarthFeature(category, name, scale='auto', **kwargs):     # noqa: N802 -- cartopy's name
    """cartopy.feature.NaturalEarthFeature for the features fastplot knows."""
    if name not in _NE_NAMES:
        raise ValueError('fastplot has no Natural Earth feature %r (it has: %s)'
                         % (name, ', '.join(sorted(_NE_NAMES))))
    return Feature(_NE_NAMES[name], scale, **kwargs)


BORDERS = Feature('borders', edgecolor='black', facecolor='never')
STATES = Feature('states_lakes', edgecolor='black', facecolor='none')
COASTLINE = Feature('coastline', edgecolor='black', facecolor='never')
LAKES = Feature('lakes', edgecolor='none', facecolor=COLORS['water'])
LAND = Feature('land', edgecolor='none', facecolor=COLORS['land'], zorder=-1)
OCEAN = Feature('ocean', edgecolor='none', facecolor=COLORS['water'], zorder=-1)
RIVERS = Feature('rivers', edgecolor=COLORS['water'], facecolor='never')
COUNTIES = Feature('counties', scale='10m', edgecolor='black', facecolor='none')
