"""Deformation gradients and principal stretches at the Gauss points."""

from __future__ import annotations

import numpy as np

from .mesh import GAUSS_2x2x2, CylinderMesh, hex8_shape_grad

__all__ = [
    "deformation_gradient",
    "principal_stretches",
    "right_stretch",
    "gauss_positions",
    "circumferential",
    "prestretch_maps",
    "fiber_nodal_forces",
]


def deformation_gradient(mesh: CylinderMesh, X: np.ndarray, x: np.ndarray) -> np.ndarray:
    """F = dx/dX at every Gauss point of every element -> (n_elem, n_gp, 3, 3).

    ``X`` is the reference configuration (e.g. the stress-free elastin mesh) and
    ``x`` the current one (e.g. the in vivo mesh), both (n_node, 3).
    """
    dn = np.stack([hex8_shape_grad(g) for g in GAUSS_2x2x2])   # (gp, 8, 3)
    Xe, xe = X[mesh.hexes], x[mesh.hexes]                      # (ne, 8, 3)
    dXdxi = np.einsum("eni,gnj->egij", Xe, dn)                 # (ne, gp, 3, 3)
    dxdxi = np.einsum("eni,gnj->egij", xe, dn)
    return dxdxi @ np.linalg.inv(dXdxi)


def right_stretch(F: np.ndarray) -> np.ndarray:
    """Right stretch tensor U = sqrt(F^T F) of a stack of deformation gradients."""
    C = np.swapaxes(F, -1, -2) @ F
    w, v = np.linalg.eigh(C)
    return (v * np.sqrt(np.clip(w, 0, None))[..., None, :]) @ np.swapaxes(v, -1, -2)


def principal_stretches(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Principal stretches (descending) and their directions.

    Returns ``(lam, dirs)`` with shapes ``(..., 3)`` and ``(..., 3, 3)``; the
    columns of ``dirs`` are the principal directions in the *reference*
    configuration (eigenvectors of C = F^T F).
    """
    C = np.swapaxes(F, -1, -2) @ F
    w, v = np.linalg.eigh(C)
    lam = np.sqrt(np.clip(w, 0, None))
    order = np.argsort(-lam, axis=-1)
    lam = np.take_along_axis(lam, order, axis=-1)
    v = np.take_along_axis(v, order[..., None, :], axis=-1)
    return lam, v


def gauss_positions(mesh, coords: np.ndarray) -> np.ndarray:
    """Position of every Gauss point -> (n_elem, n_gp, 3)."""
    from .mesh import GAUSS_2x2x2, _HEX_CORNERS

    xi = GAUSS_2x2x2
    s = _HEX_CORNERS[None, :, :] * xi[:, None, :]          # (gp, 8, 3)
    N = 0.125 * np.prod(1.0 + s, axis=2)                   # (gp, 8)
    return np.einsum("gn,eni->egi", N, coords[mesh.hexes])


def circumferential(pos: np.ndarray) -> np.ndarray:
    """Unit circumferential direction about the z axis at the given positions."""
    e = np.stack([-pos[..., 1], pos[..., 0], np.zeros_like(pos[..., 0])], axis=-1)
    return e / np.linalg.norm(e, axis=-1, keepdims=True)


def prestretch_maps(mesh, X: np.ndarray, x: np.ndarray):
    """Per-Gauss-point ``(|G_e M|, det G_e)`` for the plugin material.

    ``G_e = dx/dX`` maps the stress-free elastin mesh ``X`` onto the in vivo
    mesh ``x``; ``M`` is the circumferential direction in ``X``, matching the
    plugin's ``<fiber type="cylindrical">`` definition.
    """
    Ge = deformation_gradient(mesh, X, x)
    M = circumferential(gauss_positions(mesh, X))
    lam = np.linalg.norm(np.einsum("egij,egj->egi", Ge, M), axis=-1)
    return lam, np.linalg.det(Ge)


def fiber_nodal_forces(mesh, p, lam_fiber: float | None = None,
                       deviatoric: bool = True) -> np.ndarray:
    """Nodal forces that represent the embedded fibre stress *exactly*.

    The weak form of the mixture (Eq. 6 of the manuscript) is

        int sigma_e : de dv  =  int t.du ds  -  int sigma_ef : de dv,

    and the last term is known a priori, because the fibres are deposited at a
    prescribed stretch.  Integrating it against the shape functions on the
    in vivo mesh gives a fixed nodal force vector

        f_a = - int sigma_ef . grad(N_a) dv,

    which any solver can apply.  Unlike the effective pressure of Sec. 2.4 this
    is not an approximation: at convergence the deformed configuration is the
    in vivo one, so the forces are evaluated exactly where they belong.

    With ``deviatoric=True`` only the deviatoric part of ``sigma_ef`` is
    assembled.  For an incompressible mixture that makes no difference -- the
    hydrostatic part does no virtual work, because ``tr(de) = d(ln J) = 0`` --
    and it matches the split FEBio's uncoupled materials make internally, so
    the two formulations then agree at any finite bulk penalty.

    Returns an (n_node, 3) array in the same units as the pressure times mm^2.
    """
    from .material import fiber_cauchy_stress
    from .mesh import GAUSS_2x2x2, hex8_shape_grad

    sigma = fiber_cauchy_stress(p, lam_fiber)
    dn = np.stack([hex8_shape_grad(g) for g in GAUSS_2x2x2])      # (gp, 8, 3)
    xe = mesh.nodes[mesh.hexes]                                   # (ne, 8, 3)
    jac = np.einsum("eni,gnj->egij", xe, dn)                      # dx/dxi
    detJ = np.linalg.det(jac)
    gradN = np.einsum("gnj,egjk->egnk", dn, np.linalg.inv(jac))   # dN_a/dx

    M = circumferential(gauss_positions(mesh, mesh.nodes))        # (ne, gp, 3)
    S = sigma * np.einsum("egi,egj->egij", M, M)                  # sigma_ef tensor
    if deviatoric:
        S = S - np.eye(3) * (np.trace(S, axis1=-2, axis2=-1)[..., None, None] / 3.0)

    fe = -np.einsum("egij,egnj,eg->eni", S, gradN, detJ)          # 2x2x2 weights = 1
    f = np.zeros_like(mesh.nodes)
    np.add.at(f, mesh.hexes, fe)
    return f
