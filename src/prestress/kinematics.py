"""Deformation gradients and principal stretches at the Gauss points."""

from __future__ import annotations

import numpy as np

from .mesh import GAUSS_2x2x2, CylinderMesh, hex8_shape_grad

__all__ = [
    "deformation_gradients", "convect_directions", "stress_from_gp_directions",
    "deformation_gradient",
    "principal_stretches",
    "right_stretch",
    "gauss_positions",
    "circumferential",
    "prestretch_maps",
    "fiber_nodal_forces",
    "fiber_stress_tensors",
    "nodal_forces_from_stress",
    "surface_fiber_forces",
    "laplace_relief_forces",
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


def fiber_stress_tensors(elements, elem_type: str, families, deviatoric: bool = True):
    """Prescribed fibre Cauchy stress at every integration point.

    ``families`` is a sequence of ``(sigma, directions)``: the tissue-level
    Cauchy stress the family carries along its own direction, and that
    direction as an (n_node, 3) nodal field.  Directions are interpolated to
    the integration points and renormalised, so a curved vessel keeps a smooth
    fibre architecture.
    """
    from .elements import element_for

    el = element_for(elem_type)
    N = el.values()                                   # (gp, n)
    S = None
    for sigma, dirs in families:
        m = np.einsum("gn,eni->egi", N, np.asarray(dirs)[elements])
        m /= np.linalg.norm(m, axis=-1, keepdims=True)
        term = sigma * np.einsum("egi,egj->egij", m, m)
        S = term if S is None else S + term
    if deviatoric:
        S = S - np.eye(3) * (np.trace(S, axis1=-2, axis2=-1)[..., None, None] / 3.0)
    return S


def deformation_gradients(X, x, elements, elem_type: str) -> np.ndarray:
    """``F = dx/dX`` at every integration point of any element type.

    Returns ``(n_elem, n_gp, 3, 3)``.  Unlike :func:`deformation_gradient`
    this needs no cylinder.
    """
    from .elements import element_for

    dn = element_for(elem_type).grads()
    Jr = np.einsum("eni,gnj->egij", np.asarray(X)[elements], dn)
    Jc = np.einsum("eni,gnj->egij", np.asarray(x)[elements], dn)
    return np.einsum("egij,egjk->egik", Jc, np.linalg.inv(Jr))


def convect_directions(dirs, F, elements, elem_type: str) -> np.ndarray:
    """Push a nodal direction field forward: ``m = F M / |F M|``.

    A fibre is a material line, so it rotates and stretches with the body.
    Returns the unit directions at the integration points, ``(n_elem, n_gp, 3)``.
    """
    from .elements import element_for

    N = element_for(elem_type).values()
    M = np.einsum("gn,eni->egi", N, np.asarray(dirs)[elements])
    M /= np.linalg.norm(M, axis=-1, keepdims=True)
    m = np.einsum("egij,egj->egi", F, M)
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def stress_from_gp_directions(families_gp, deviatoric: bool = True) -> np.ndarray:
    """Fibre Cauchy stress from directions already given at the Gauss points.

    ``families_gp`` is a sequence of ``(sigma, m)`` with ``m`` of shape
    ``(n_elem, n_gp, 3)``.  :func:`fiber_stress_tensors` is the same thing for
    directions given on the nodes of the reference configuration.
    """
    S = None
    for sigma, m in families_gp:
        term = sigma * np.einsum("egi,egj->egij", m, m)
        S = term if S is None else S + term
    if deviatoric:
        S = S - np.eye(3) * (np.trace(S, axis1=-2, axis2=-1)[..., None, None] / 3.0)
    return S


def nodal_forces_from_stress(nodes, elements, elem_type: str, S) -> np.ndarray:
    """Assemble ``f_a = -int sigma . grad(N_a) dv`` over the given mesh.

    This is the term the manuscript drops between Eqs. 6 and 7.  Because the
    fibre stress is prescribed a priori it can be integrated once, on the
    imaged geometry, and handed to the solver as nodal loads -- exactly, and
    without the solver needing to know anything about fibres.
    """
    from .elements import element_for

    el = element_for(elem_type)
    dn, w = el.grads(), el.weights
    xe = np.asarray(nodes)[elements]
    jac = np.einsum("eni,gnj->egij", xe, dn)
    detJ = np.linalg.det(jac)
    gradN = np.einsum("gnj,egjk->egnk", dn, np.linalg.inv(jac))

    fe = -np.einsum("egij,egnj,eg,g->eni", S, gradN, detJ, w)
    f = np.zeros_like(np.asarray(nodes), dtype=float)
    np.add.at(f, elements, fe)
    return f


def fiber_nodal_forces(mesh, p, lam_fiber: float | None = None,
                       deviatoric: bool = True) -> np.ndarray:
    """Nodal forces representing the embedded fibre stress of the cylinder.

    Thin wrapper that builds the single circumferential family and calls the
    general assembly above.  See ``nodal_forces_from_stress`` for why this is
    exact rather than an approximation.
    """
    from .material import fiber_cauchy_stress

    sigma = fiber_cauchy_stress(p, lam_fiber)
    circ = circumferential(mesh.nodes)
    S = fiber_stress_tensors(mesh.hexes, "hex8", [(sigma, circ)], deviatoric)
    return nodal_forces_from_stress(mesh.nodes, mesh.hexes, "hex8", S)


def surface_fiber_forces(nodes, facets, S_facet) -> np.ndarray:
    """Nodal forces from a traction ``sigma . n`` on triangular facets.

    This is Eq. 9 of the manuscript taken literally: the fibre contribution
    enters only as a surface traction.  For any fibre architecture lying in the
    wall, ``sigma_ef . n`` vanishes identically on the luminal surface, so this
    assembles to (discretisation) noise -- which is the point.
    """
    P = np.asarray(nodes)[facets]
    tn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    area = np.linalg.norm(tn, axis=1) / 2.0
    n = tn / (2.0 * area[:, None])
    trac = np.einsum("fij,fj->fi", S_facet, n) * area[:, None] / 3.0

    f = np.zeros_like(np.asarray(nodes), dtype=float)
    for k in range(3):
        np.add.at(f, facets[:, k], trac)
    return f


def laplace_relief_forces(nodes, facets, hoop_stress, thickness, radius) -> np.ndarray:
    """Nodal forces for the thin-wall pressure relief of Sec. 2.4, made local.

    The manuscript replaces the fibre contribution by ``P_eff = P - sigma h/r``,
    which needs a single hoop stress, thickness and radius.  On a vessel where
    all three vary this is the natural generalisation: the relief is applied
    pointwise as ``-sigma_hoop h/r`` along the luminal normal.
    """
    P = np.asarray(nodes)[facets]
    tn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    area = np.linalg.norm(tn, axis=1) / 2.0
    n = tn / (2.0 * area[:, None])

    hr = (np.asarray(thickness)[facets] / np.asarray(radius)[facets]).mean(axis=1)
    trac = (-hoop_stress * hr)[:, None] * n * area[:, None] / 3.0

    f = np.zeros_like(np.asarray(nodes), dtype=float)
    for k in range(3):
        np.add.at(f, facets[:, k], trac)
    return f
