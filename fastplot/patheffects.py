"""matplotlib.patheffects for text and lines: outlines (halos) round labels
and traces.

    import fastplot.patheffects as pe
    ax.text(x, y, 'Denver', color='white',
            path_effects=[pe.withStroke(linewidth=3, foreground='black')])
    ax.plot(u, v, lw=1.8, path_effects=[pe.withStroke(linewidth=2.5, foreground='white',
                                                      alpha=0.7)])

The effects are drawn in list order. Stroke draws the outline, linewidth
points wide, in `foreground` (with `alpha`, else the artist's); Normal draws
the artist itself; withStroke is a Stroke followed by Normal. A line's
Stroke is the line stroked again at that width, with its caps, joins and
dashes (or the Stroke's `capstyle` / `joinstyle`); its markers are drawn
once, after. fastplot draws text from coverage masks, so a text's stroke is
the mask grown by half the line width, with an anti-aliased edge. Effects
on other artists are accepted and ignored.
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
