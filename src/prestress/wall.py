"""Build a solid vessel wall from a Vascular Model Repository lumen surface.

VMR models are fluid-domain meshes: the wall is only a surface. To prestress
the wall it has to be given a thickness, which is done here by extruding the
lumen surface outwards into layers of wedge (``penta6``) elements.

The wall thickness follows the local lumen radius, which is estimated from the
surface itself -- a ray cast inwards along the normal hits the far wall, and
half that chord is the radius -- so no centreline is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["WallMesh", "load_lumen_surface", "local_radius",
           "smooth_normals", "extrude_wall"]


def _wedge_volumes(x: np.ndarray, wedges: np.ndarray) -> np.ndarray:
    p = x[wedges]
    tets = ((0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5))
    vol = np.zeros(len(p))
    for a, b, c, d in tets:
        vol += np.einsum("ij,ij->i", p[:, b] - p[:, a],
                         np.cross(p[:, c] - p[:, a],
                                  p[:, d] - p[:, a])) / 6.0
    return vol


@dataclass
class WallMesh:
    """Solid wall discretised into wedge elements, layered through the thickness."""

    nodes: np.ndarray          # (n_node, 3) in vivo coordinates [mm]
    wedges: np.ndarray         # (n_elem, 6) 0-based, penta6 ordering
    lumen_facets: np.ndarray   # (n_facet, 3) 0-based, normal points into the lumen
    ring_nodes: np.ndarray     # node ids on the inlet/outlet rings (all layers)
    normals: np.ndarray        # (n_surf, 3) outward unit normals of the lumen surface
    radius: np.ndarray         # (n_surf,) local lumen radius [mm]
    thickness: np.ndarray      # (n_surf,) wall thickness [mm]
    n_layers: int
    n_surf: int                # nodes per layer

    @property
    def n_node(self) -> int:
        return len(self.nodes)

    @property
    def n_elem(self) -> int:
        return len(self.wedges)

    def volumes(self, coords: np.ndarray | None = None) -> np.ndarray:
        """Signed volume of every wedge, by splitting it into three tetrahedra."""
        x = self.nodes if coords is None else coords
        return _wedge_volumes(x, self.wedges)


def load_lumen_surface(path: str | Path, scale: float = 10.0,
                       decimate: float = 0.88, taubin: int = 30):
    """Read a VMR wall surface, scale it, decimate and smooth it.

    ``scale`` converts the model units to mm (VMR models are in cm).
    ``decimate`` is the fraction of triangles to remove; boundary vertices are
    kept so the inlet and outlet rings survive intact.  ``taubin`` iterations of
    volume-preserving smoothing then take the sharp creases out of the branch
    junctions -- without it the outward extrusion folds there.  Over-smoothing
    is counter-productive: around 30 iterations is the minimum for this model.
    """
    import pyvista as pv

    surf = pv.read(str(path)).clean()
    surf.points = np.asarray(surf.points) * scale
    if decimate:
        surf = surf.decimate_pro(decimate, preserve_topology=True,
                                 boundary_vertex_deletion=False).clean()
    if taubin:
        surf = surf.smooth_taubin(n_iter=taubin, pass_band=0.1,
                                  boundary_smoothing=False)
    surf = surf.compute_normals(cell_normals=False, point_normals=True,
                                consistent_normals=True,
                                auto_orient_normals=True)
    return surf


def _surface_neighbours(faces: np.ndarray, n_pts: int) -> list[np.ndarray]:
    adj: list[set] = [set() for _ in range(n_pts)]
    for a, b, c in faces:
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    return [np.fromiter(s, dtype=int) for s in adj]


def local_radius(surf, max_chord: float = 120.0, smooth_iters: int = 40,
                 clip: tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    """Local lumen radius at every surface point [mm].

    A ray is cast inwards along the point normal; the first hit on the far wall
    gives the chord, and half of it is the radius.  Rays that graze a branch or
    run along the vessel axis give outliers, so the field is clipped to the
    given percentiles and then smoothed over the surface graph.
    """
    pts = np.asarray(surf.points)
    nrm = np.asarray(surf.point_data["Normals"])
    tri = surf.extract_surface().triangulate()

    rad = np.full(len(pts), np.nan)
    for i in range(len(pts)):
        hit, _ = tri.ray_trace(pts[i] - 1e-3 * nrm[i],
                               pts[i] - max_chord * nrm[i], first_point=True)
        if len(hit):
            rad[i] = 0.5 * float(np.linalg.norm(hit - pts[i]))

    good = np.isfinite(rad)
    lo, hi = np.percentile(rad[good], clip)
    rad[~good | (rad < lo) | (rad > hi)] = np.nan

    faces = surf.faces.reshape(-1, 4)[:, 1:]
    nbr = _surface_neighbours(faces, len(pts))
    fill = np.nanmedian(rad)
    out = np.where(np.isfinite(rad), rad, fill)
    for _ in range(smooth_iters):
        out = np.array([out[n].mean() if len(n) else out[i]
                        for i, n in enumerate(nbr)])
    return out


def smooth_normals(surf, iters: int = 5) -> np.ndarray:
    """Outward unit point normals, built from the triangle winding.

    VTK's ``auto_orient_normals`` is unreliable on an open surface: a handful of
    points come back flipped, and smoothing then rotates them by 180 degrees,
    which folds the extrusion.  Building the field from the (consistent)
    winding instead and only then smoothing avoids that.  The sign is fixed by
    a ray cast: along the inward normal the ray must hit the far wall.
    """
    pts = np.asarray(surf.points, dtype=float)
    faces = surf.faces.reshape(-1, 4)[:, 1:].astype(int)

    # area-weighted triangle normals accumulated onto the points
    tn = np.cross(pts[faces[:, 1]] - pts[faces[:, 0]],
                  pts[faces[:, 2]] - pts[faces[:, 0]])          # |tn| = 2 * area
    nrm = np.zeros_like(pts)
    for k in range(3):
        np.add.at(nrm, faces[:, k], tn)
    mag = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, mag, out=np.zeros_like(nrm), where=mag > 1e-12)

    nbr = _surface_neighbours(faces, len(pts))
    for _ in range(iters):
        sm = np.array([nrm[n].mean(axis=0) if len(n) else nrm[i]
                       for i, n in enumerate(nbr)])
        keep = np.einsum("ij,ij->i", sm, nrm) > 0.0   # never flip a normal
        nrm[keep] = sm[keep]
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)

    # global sign: casting inwards has to hit the opposite wall
    tri = surf.extract_surface().triangulate()
    rng = np.random.default_rng(0)
    sample = rng.choice(len(pts), size=min(200, len(pts)), replace=False)
    hits = 0
    for i in sample:
        h, _ = tri.ray_trace(pts[i] - 1e-3 * nrm[i], pts[i] - 120.0 * nrm[i],
                             first_point=True)
        hits += len(h) > 0
    if hits < 0.5 * len(sample):
        nrm = -nrm
    return nrm


def _build(pts, nrm, faces, thickness, n_layers):
    n_surf = len(pts)
    offsets = np.linspace(0.0, 1.0, n_layers + 1)
    nodes = np.concatenate([pts + nrm * (thickness * t)[:, None] for t in offsets])
    wedges = np.concatenate([
        np.column_stack([faces + k * n_surf, faces + (k + 1) * n_surf])
        for k in range(n_layers)
    ])
    return nodes, wedges


def extrude_wall(surf, thickness: np.ndarray, n_layers: int = 2,
                 smooth_normal_iters: int = 5, max_relax: int = 60,
                 grading: float = 0.5, drop_bad: bool = True) -> WallMesh:
    """Extrude the lumen surface outwards into ``n_layers`` of wedge elements.

    Where the outward extrusion would fold -- concave patches at branch
    junctions, whose radius of curvature is smaller than the wall thickness --
    the thickness is relaxed locally until every wedge has positive volume.
    """
    pts = np.asarray(surf.points, dtype=float)
    nrm = smooth_normals(surf, smooth_normal_iters)
    faces = surf.faces.reshape(-1, 4)[:, 1:].astype(int)
    n_surf = len(pts)
    thickness = np.array(thickness, dtype=float)
    nbr = _surface_neighbours(faces, n_surf)

    # A handful of slivers sit in creases where the triangle normal opposes the
    # extrusion direction of its own nodes; those wedges invert for *any*
    # thickness.  Reversing the winding fixes the element.  The facet then
    # carries the luminal pressure the wrong way round, which for the few
    # triangles involved (<0.1 % of the wall area) is immaterial.
    tn = np.cross(pts[faces[:, 1]] - pts[faces[:, 0]],
                  pts[faces[:, 2]] - pts[faces[:, 0]])
    tn /= np.linalg.norm(tn, axis=1, keepdims=True)
    flip = np.einsum("ij,ij->i", tn, nrm[faces].mean(axis=1)) <= 0.0
    if flip.any():
        faces[flip] = faces[flip][:, [0, 2, 1]]
        import warnings
        warnings.warn(f"reversed {flip.sum()} of {len(faces)} wall triangles "
                      "whose normal opposed the extrusion direction")

    # A handful of triangles -- slivers, or spots where the node normals
    # diverge by tens of degrees -- fold at the nominal thickness.  Thin the
    # wall at their nodes only.  Smoothing that reduction is what made an
    # earlier version oscillate: it pulled neighbouring nodes down and inverted
    # *their* elements, so the bad set wandered instead of emptying.
    thickness0 = thickness.copy()
    edge_len = {i: np.linalg.norm(pts[n] - pts[i], axis=1) for i, n in enumerate(nbr)}
    for _ in range(max_relax):
        nodes, wedges = _build(pts, nrm, faces, thickness, n_layers)
        bad = _wedge_volumes(nodes, wedges) <= 0.0
        if not bad.any():
            break
        hit = np.unique(wedges[bad] % n_surf)
        thickness[hit] = np.maximum(0.6 * thickness[hit],
                                    0.01 * thickness0[hit])
        # Grade the field: |h_i - h_j| <= grading * |x_i - x_j| for neighbours.
        # Without this the thinned nodes sit next to full-thickness ones and
        # the element is so skewed that it inverts anyway -- which is what made
        # pure local thinning stall.  The sweep only ever lowers h, so it
        # terminates.
        for _ in range(4):
            upd = np.array([np.min(thickness[n] + grading * edge_len[i])
                            if len(n) else thickness[i]
                            for i, n in enumerate(nbr)])
            thickness = np.minimum(thickness, upd)
    else:
        # The crotch between a branch and the trunk can have a concave radius
        # of curvature smaller than any usable wall thickness; there the
        # outward offset self-intersects however gently it is tapered.  (This
        # is geometry, not meshing: the *undecimated* surface folds worst of
        # all, because it resolves those creases most faithfully.)  A handful
        # of elements are simply dropped.
        if not drop_bad:
            raise RuntimeError(f"{bad.sum()} wedges still invert after "
                               f"{max_relax} passes")
        keep = ~bad
        import warnings
        warnings.warn(f"dropped {bad.sum()} of {len(wedges)} wedges that fold "
                      "at every thickness (concave branch junctions)")
        dropped_tris = np.unique(np.flatnonzero(bad) % len(faces))
        wedges = wedges[keep]
        faces = np.delete(faces, dropped_tris, axis=0)

    # pressure acts on the lumen; FEBio applies t = -p n, so the facet winding
    # has to give a normal pointing *into* the lumen
    lumen = faces[:, [0, 2, 1]].copy()

    edges = {}
    for a, b, c in faces:
        for e in ((a, b), (b, c), (c, a)):
            k = (min(e), max(e))
            edges[k] = edges.get(k, 0) + 1
    ring = np.unique([i for e, n in edges.items() if n == 1 for i in e])
    ring_nodes = np.concatenate([ring + k * n_surf for k in range(n_layers + 1)])

    return WallMesh(nodes=nodes, wedges=wedges, lumen_facets=lumen,
                    ring_nodes=ring_nodes, normals=nrm,
                    radius=np.zeros(n_surf), thickness=thickness,
                    n_layers=n_layers, n_surf=n_surf)


# ---------------------------------------------------------------------------
# fibre architecture
# ---------------------------------------------------------------------------
def _inlet_ring(surf) -> np.ndarray:
    """Point ids of the widest open boundary ring -- the inlet."""
    faces = surf.faces.reshape(-1, 4)[:, 1:].astype(int)
    edges: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for e in ((a, b), (b, c), (c, a)):
            k = (min(e), max(e))
            edges[k] = edges.get(k, 0) + 1
    border = [e for e, n in edges.items() if n == 1]

    # connected components of the boundary edge graph
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    n = surf.n_points
    i = [e[0] for e in border] + [e[1] for e in border]
    j = [e[1] for e in border] + [e[0] for e in border]
    g = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    ncomp, lab = connected_components(g, directed=False)

    pts = np.asarray(surf.points)
    on_border = np.zeros(n, bool)
    on_border[np.unique(np.concatenate(border))] = True
    best, best_r = None, -1.0
    for c in range(ncomp):
        m = on_border & (lab == c)
        if m.sum() < 3:
            continue
        p = pts[m]
        r = float(np.linalg.norm(p - p.mean(0), axis=1).mean())
        if r > best_r:
            best, best_r = np.flatnonzero(m), r
    return best


def axial_direction(surf, smooth_iters: int = 30) -> np.ndarray:
    """Unit "downstream" direction tangent to the lumen surface.

    There is no centreline in a VMR wall surface, so the axial direction is
    taken from the gradient of the geodesic distance to the inlet: Dijkstra
    over the surface graph, smoothed, then the piecewise-linear gradient
    projected into the tangent plane.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra

    pts = np.asarray(surf.points, dtype=float)
    faces = surf.faces.reshape(-1, 4)[:, 1:].astype(int)
    nrm = smooth_normals(surf, 5)
    n = len(pts)

    rows, cols, w = [], [], []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        i, j = faces[:, a], faces[:, b]
        d = np.linalg.norm(pts[i] - pts[j], axis=1)
        rows += [*i, *j]
        cols += [*j, *i]
        w += [*d, *d]
    g = sp.coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()

    phi = dijkstra(g, directed=False, indices=_inlet_ring(surf), min_only=True)
    phi[~np.isfinite(phi)] = np.nanmax(phi[np.isfinite(phi)])

    nbr = _surface_neighbours(faces, n)
    for _ in range(smooth_iters):
        phi = np.array([phi[k].mean() if len(k) else phi[i]
                        for i, k in enumerate(nbr)])

    # piecewise-linear gradient on each triangle, then area-weighted to points
    p0, p1, p2 = pts[faces[:, 0]], pts[faces[:, 1]], pts[faces[:, 2]]
    tn = np.cross(p1 - p0, p2 - p0)
    a2 = np.linalg.norm(tn, axis=1)            # 2 * area
    tnu = tn / a2[:, None]
    grad = (np.cross(tnu, p2 - p1) * phi[faces[:, 0], None]
            + np.cross(tnu, p0 - p2) * phi[faces[:, 1], None]
            + np.cross(tnu, p1 - p0) * phi[faces[:, 2], None]) / a2[:, None]

    acc = np.zeros_like(pts)
    for k in range(3):
        np.add.at(acc, faces[:, k], grad * a2[:, None])
    acc -= nrm * np.einsum("ij,ij->i", acc, nrm)[:, None]
    mag = np.linalg.norm(acc, axis=1, keepdims=True)
    return np.divide(acc, mag, out=np.tile([1.0, 0.0, 0.0], (n, 1)), where=mag > 1e-12)


def fiber_directions(surf, angle_deg: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Two fibre families at +/- ``angle_deg`` from the local axial direction."""
    nrm = smooth_normals(surf, 5)
    ax = axial_direction(surf)
    circ = np.cross(nrm, ax)
    circ /= np.linalg.norm(circ, axis=1, keepdims=True)
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    return c * ax + s * circ, c * ax - s * circ
