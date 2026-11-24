"""Structured hexahedral mesh of a thick-walled cylindrical tube."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import CylinderGeometry

__all__ = ["CylinderMesh", "build_cylinder_mesh", "hex8_shape_grad", "GAUSS_2x2x2"]

# HEX8 node ordering (FEBio / standard): bottom face 0-3, top face 4-7
_HEX_CORNERS = np.array(
    [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ],
    dtype=float,
)

# 2x2x2 Gauss rule in FEBio's order (FEHex8G8 in FECore/FEElementTraits.cpp):
# the integration points follow the node ordering, *not* a lexicographic one.
# Getting this wrong silently permutes any per-integration-point data handed to
# FEBio through a MAT_POINTS element map.
GAUSS_2x2x2 = _HEX_CORNERS / np.sqrt(3.0)


def hex8_shape_grad(xi: np.ndarray) -> np.ndarray:
    """dN/dxi at natural coordinates ``xi`` -> array (8, 3)."""
    s = _HEX_CORNERS * xi  # (8,3)
    return 0.125 * np.stack(
        [
            _HEX_CORNERS[:, 0] * (1 + s[:, 1]) * (1 + s[:, 2]),
            _HEX_CORNERS[:, 1] * (1 + s[:, 0]) * (1 + s[:, 2]),
            _HEX_CORNERS[:, 2] * (1 + s[:, 0]) * (1 + s[:, 1]),
        ],
        axis=1,
    )


@dataclass
class CylinderMesh:
    """Nodes, hexahedra and boundary sets of the idealised vessel."""

    nodes: np.ndarray           # (n_node, 3) in vivo coordinates [mm]
    hexes: np.ndarray           # (n_elem, 8) 0-based connectivity
    inner_facets: np.ndarray    # (n_facet, 4) 0-based, normal points into the lumen
    fix_z: np.ndarray           # node ids with u_z = 0
    fix_x: np.ndarray           # node ids with u_x = 0
    fix_y: np.ndarray           # node ids with u_y = 0
    mat_axis: np.ndarray        # (n_elem, 3, 3) columns = (radial, circ., axial)
    geo: CylinderGeometry

    @property
    def n_node(self) -> int:
        return len(self.nodes)

    @property
    def n_elem(self) -> int:
        return len(self.hexes)

    def jacobians(self, coords: np.ndarray | None = None) -> np.ndarray:
        """Determinant of dx/dxi at every Gauss point -> (n_elem, 8)."""
        x = self.nodes if coords is None else coords
        xe = x[self.hexes]                                   # (ne, 8, 3)
        dn = np.stack([hex8_shape_grad(g) for g in GAUSS_2x2x2])  # (8gp, 8, 3)
        jac = np.einsum("eni,gnj->egij", xe, dn)             # (ne, gp, 3, 3) = dx/dxi
        return np.linalg.det(jac)


def build_cylinder_mesh(geo: CylinderGeometry = CylinderGeometry()) -> CylinderMesh:
    nt, nr, nz = geo.n_theta, geo.n_r, geo.n_z
    if nt % 4:
        raise ValueError("n_theta must be a multiple of 4 for the symmetry constraints")

    theta = 2.0 * np.pi * np.arange(nt) / nt
    rad = geo.r_inner + geo.thickness * np.arange(nr + 1) / nr
    zz = geo.length * np.arange(nz + 1) / nz

    def nid(it, ir, iz):
        return ((iz * (nr + 1)) + ir) * nt + (it % nt)

    nodes = np.empty((nt * (nr + 1) * (nz + 1), 3))
    for iz in range(nz + 1):
        for ir in range(nr + 1):
            for it in range(nt):
                nodes[nid(it, ir, iz)] = (
                    rad[ir] * np.cos(theta[it]),
                    rad[ir] * np.sin(theta[it]),
                    zz[iz],
                )

    hexes = np.empty((nt * nr * nz, 8), dtype=int)
    mat_axis = np.empty((nt * nr * nz, 3, 3))
    e = 0
    for iz in range(nz):
        for ir in range(nr):
            for it in range(nt):
                # local xi ~ radial, eta ~ circumferential, zeta ~ axial
                hexes[e] = [
                    nid(it, ir, iz), nid(it, ir + 1, iz),
                    nid(it + 1, ir + 1, iz), nid(it + 1, ir, iz),
                    nid(it, ir, iz + 1), nid(it, ir + 1, iz + 1),
                    nid(it + 1, ir + 1, iz + 1), nid(it + 1, ir, iz + 1),
                ]
                th = 0.5 * (theta[it] + theta[(it + 1) % nt]) if it + 1 < nt \
                    else theta[it] + np.pi / nt
                er = np.array([np.cos(th), np.sin(th), 0.0])
                et = np.array([-np.sin(th), np.cos(th), 0.0])
                ez = np.array([0.0, 0.0, 1.0])
                mat_axis[e] = np.column_stack([er, et, ez])
                e += 1

    # inner surface, ordered so that the facet normal points into the lumen
    inner = np.empty((nt * nz, 4), dtype=int)
    f = 0
    for iz in range(nz):
        for it in range(nt):
            inner[f] = [
                nid(it, 0, iz), nid(it, 0, iz + 1),
                nid(it + 1, 0, iz + 1), nid(it + 1, 0, iz),
            ]
            f += 1

    fix_z = np.array([nid(it, ir, 0) for ir in range(nr + 1) for it in range(nt)])
    fix_y = np.array(
        [nid(it, ir, iz) for iz in range(nz + 1) for ir in range(nr + 1)
         for it in (0, nt // 2)]
    )
    fix_x = np.array(
        [nid(it, ir, iz) for iz in range(nz + 1) for ir in range(nr + 1)
         for it in (nt // 4, 3 * nt // 4)]
    )

    mesh = CylinderMesh(nodes, hexes, inner, fix_z, fix_x, fix_y, mat_axis, geo)

    # sanity checks
    det = mesh.jacobians()
    if det.min() <= 0:
        raise RuntimeError(f"negative hex Jacobian (min {det.min():.3e})")
    n = _facet_normal(nodes, inner[0])
    mid = nodes[inner[0]].mean(axis=0)
    if n[:2] @ (mid[:2] / np.linalg.norm(mid[:2])) > 0:
        raise RuntimeError("inner facet normal does not point into the lumen")
    return mesh


def _facet_normal(nodes: np.ndarray, facet: np.ndarray) -> np.ndarray:
    p = nodes[facet]
    n = np.cross(p[1] - p[0], p[3] - p[0])
    return n / np.linalg.norm(n)
