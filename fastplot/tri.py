"""matplotlib.tri: unstructured triangular grids.

    from fastplot.tri import Triangulation
    t = Triangulation(x, y)                     # Delaunay (matplotlib's), or triangles=
    ax.tricontourf(t, z); ax.tricontour(t, z, colors='k'); ax.triplot(t, 'k-', lw=0.3)
    t.get_trifinder()(xi, yi)                   # the triangle of each point, -1 off the mesh
    CubicTriInterpolator(t, z)(xi, yi)          # or LinearTriInterpolator; .gradient(xi, yi)
    UniformTriRefiner(t).refine_field(z, subdiv=3); TriAnalyzer(t).get_flat_tri_mask(0.01)
"""
from ._tri import Triangulation
from ._triinterp import (CubicTriInterpolator, LinearTriInterpolator, TrapezoidMapTriFinder,
                         TriAnalyzer, TriFinder, TriInterpolator)
from ._trirefine import TriRefiner, UniformTriRefiner

__all__ = ['Triangulation', 'TriFinder', 'TrapezoidMapTriFinder', 'TriInterpolator',
           'LinearTriInterpolator', 'CubicTriInterpolator', 'TriRefiner', 'UniformTriRefiner',
           'TriAnalyzer']
