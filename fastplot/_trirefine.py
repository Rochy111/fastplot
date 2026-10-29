"""matplotlib.tri's mesh refinement: each triangle split into 4 on its
edges' midpoints, subdiv times over.

    refiner = UniformTriRefiner(t)
    fine, zf = refiner.refine_field(z, subdiv=3)        # a cubic interpolation by default

matplotlib's algorithm on matplotlib's neighbours, so the refined points and
triangles come out in matplotlib's order: the original points, then each
level's new midpoints edge by edge (an edge's owner being its higher-
numbered triangle, a masked triangle's edges each owned by it), each
triangle's four children in place of it.
"""
import numpy as np

from ._jit import njit
from ._tri import Triangulation
from ._triinterp import CubicTriInterpolator, TriInterpolator, neighbors


class TriRefiner:
    """matplotlib.tri.TriRefiner: the base of the mesh refiners, with
    refine_triangulation and refine_field."""

    def __init__(self, triangulation):
        if not isinstance(triangulation, Triangulation):
            raise TypeError('triangulation must be a Triangulation')
        self._triangulation = triangulation


class UniformTriRefiner(TriRefiner):
    """matplotlib.tri.UniformTriRefiner: uniform refinement by recursive
    subdivision, 4**subdiv triangles for each."""

    def refine_triangulation(self, return_tri_index=False, subdiv=3):
        """The refined Triangulation and, when return_tri_index, the index of
        each refined point's original triangle (unmasked ones winning over
        masked ones where a point is in both)."""
        refi_triangulation = self._triangulation
        ntri = refi_triangulation.triangles.shape[0]
        ancestors = np.arange(ntri, dtype=np.int32)
        for _ in range(subdiv):
            refi_triangulation, ancestors = self._refine_triangulation_once(
                refi_triangulation, ancestors)
        if not return_tri_index:
            return refi_triangulation
        refi_npts = refi_triangulation.x.shape[0]
        refi_triangles = refi_triangulation.triangles
        found_index = np.full(refi_npts, -1, dtype=np.int32)
        tri_mask = self._triangulation.mask
        if tri_mask is None:
            found_index[refi_triangles] = np.repeat(ancestors, 3).reshape(-1, 3)
        else:                                           # masked first, unmasked over them
            ancestor_mask = tri_mask[ancestors]
            found_index[refi_triangles[ancestor_mask, :]] = np.repeat(
                ancestors[ancestor_mask], 3).reshape(-1, 3)
            found_index[refi_triangles[~ancestor_mask, :]] = np.repeat(
                ancestors[~ancestor_mask], 3).reshape(-1, 3)
        return refi_triangulation, found_index

    def refine_field(self, z, triinterpolator=None, subdiv=3):
        """(the refined Triangulation, z interpolated at its points), by
        triinterpolator or else a CubicTriInterpolator (min_E)."""
        if triinterpolator is None:
            interp = CubicTriInterpolator(self._triangulation, z)
        else:
            if not isinstance(triinterpolator, TriInterpolator):
                raise TypeError('triinterpolator must be a TriInterpolator')
            interp = triinterpolator
        refi_tri, found_index = self.refine_triangulation(subdiv=subdiv, return_tri_index=True)
        refi_z = interp._interpolate_multikeys(refi_tri.x, refi_tri.y, tri_index=found_index)[0]
        return refi_tri, refi_z

    @staticmethod
    def _refine_triangulation_once(triangulation, ancestors=None):
        """One level: each triangle into 4 on its edges' midpoints (a masked
        triangle's children masked), and the children's ancestors."""
        nbs = neighbors(triangulation)                  # (orients the triangles first)
        triangles = triangulation.triangles
        ntri = np.shape(triangles)[0]
        if ancestors is not None:
            ancestors = np.asarray(ancestors)
            if np.shape(ancestors) != (ntri,):
                raise ValueError('Incompatible shapes provide for triangulation.masked_triangles'
                                 ' and ancestors: %s and %s'
                                 % (np.shape(triangles), np.shape(ancestors)))
        refi_x, refi_y, child_triangles = _refine_once(
            triangulation.x, triangulation.y, np.ascontiguousarray(triangles, np.int32), nbs)
        child_triangulation = Triangulation(refi_x, refi_y, child_triangles)
        if triangulation.mask is not None:
            child_triangulation.set_mask(np.repeat(triangulation.mask, 4))
        if ancestors is None:
            return child_triangulation
        return child_triangulation, np.repeat(ancestors, 4)


@njit(cache=True, nogil=True)
def _refine_once(x, y, tri, nb):
    """matplotlib's _refine_triangulation_once: the points (the old, then
    each edge's midpoint, made by its master -- the higher-numbered of its
    triangles -- apex 0 edges first, then 1, then 2, each in triangle
    order) and the triangles (each one's 4 children in its place)."""
    npts, ntri = x.size, tri.shape[0]
    n_mid = 0
    for t in range(ntri):
        for e in range(3):
            if t > nb[t, e]:
                n_mid += 1
    rx = np.empty(npts + n_mid)
    ry = np.empty(npts + n_mid)
    rx[:npts] = x
    ry[:npts] = y
    mid = np.empty((ntri, 3), np.int32)
    k = npts
    for e in range(3):
        for t in range(ntri):
            if t > nb[t, e]:
                a, b = tri[t, e], tri[t, (e + 1) % 3]
                rx[k] = (x[a] + x[b]) * 0.5
                ry[k] = (y[a] + y[b]) * 0.5
                mid[t, e] = k
                k += 1
    for e in range(3):                                  # a slave takes its master's midpoint
        for t in range(ntri):
            m = nb[t, e]
            if t <= m:
                best = 0                                # (argmin |nb[m] - t|, as matplotlib)
                for j in range(1, 3):
                    if abs(nb[m, j] - t) < abs(nb[m, best] - t):
                        best = j
                mid[t, e] = mid[m, best]
    child = np.empty((4 * ntri, 3), np.int32)
    for t in range(ntri):
        c, m = 4 * t, mid[t]
        child[c, 0], child[c, 1], child[c, 2] = tri[t, 0], m[0], m[2]
        child[c + 1, 0], child[c + 1, 1], child[c + 1, 2] = tri[t, 1], m[1], m[0]
        child[c + 2, 0], child[c + 2, 1], child[c + 2, 2] = tri[t, 2], m[2], m[1]
        child[c + 3, 0], child[c + 3, 1], child[c + 3, 2] = m[0], m[1], m[2]
    return rx, ry, child
