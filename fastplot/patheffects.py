"""matplotlib.patheffects for text: outlines (halos) round labels.

    import fastplot.patheffects as pe
    ax.text(x, y, 'Denver', color='white',
            path_effects=[pe.withStroke(linewidth=3, foreground='black')])

The effects are drawn in list order. Stroke draws the text's outline,
linewidth points wide and centred on the glyphs' edges, in `foreground`;
Normal draws the text itself; withStroke is a Stroke followed by Normal.
fastplot draws text from coverage masks, so a stroke is the mask grown by
half the line width, with an anti-aliased edge. Effects on other artists
are accepted and ignored.
"""


class AbstractPathEffect:
    def __init__(self, offset=(0.0, 0.0)):
        self.offset = (float(offset[0]), float(offset[1]))         # points, x right, y up


class Normal(AbstractPathEffect):
    """The text as it is."""


class Stroke(AbstractPathEffect):
    """An outline of the text, linewidth points wide, in foreground."""

    def __init__(self, offset=(0.0, 0.0), **gc):
        super().__init__(offset)
        self.linewidth = float(gc.pop('linewidth', gc.pop('lw', 2.0)))
        self.foreground = gc.pop('foreground', gc.pop('color', 'black'))
        self.alpha = gc.pop('alpha', None)
        self.gc = gc                                    # (joinstyle, capstyle: round here)


class withStroke(Stroke):
    """A Stroke, then the text drawn over it."""


__all__ = ['AbstractPathEffect', 'Normal', 'Stroke', 'withStroke']
