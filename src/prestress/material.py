"""Constitutive models of the constrained mixture (elastin + embedded fibres).

Units: length [mm], stress [kPa].

Conventions
-----------
The *tissue-level* Cauchy stress of a constituent already includes its mass
fraction, so that ``sigma = sigma_e + sum_ef sigma_ef``.
"""

from __future__ import annotations

import numpy as np

from .params import MaterialParams

__all__ = [
    "fiber_cauchy_stress",
    "fiber_cauchy_stress_2pk",
    "elastin_cauchy_principal",
    "elastin_cauchy",
    "effective_pressure",
]


# ---------------------------------------------------------------------------
# Embedded fibre family
# ---------------------------------------------------------------------------
def fiber_cauchy_stress_2pk(p: MaterialParams, lam: float | None = None) -> float:
    """Scalar 2nd PK stress of *one* fibre family, ``S = 2 dpsi/dI4e``.

    ``S_ef = phi_ef * c1 * (I4e - 1) * exp(c2 (I4e - 1)^2)`` with ``I4e = lam^2``.
    The fibre contribution to the 2nd PK tensor is ``S_ef * G^2 * (M (x) M)``
    when the natural configuration is reached through the deposition stretch
    tensor ``G_ef``; here it is evaluated directly in terms of the elastic
    fibre stretch ``lam``.
    """
    lam = p.lam_dep if lam is None else lam
    i4 = lam * lam
    return p.phi_per_family * p.c1 * (i4 - 1.0) * np.exp(p.c2 * (i4 - 1.0) ** 2)


def fiber_cauchy_stress(p: MaterialParams, lam: float | None = None,
                        jacobian: float = 1.0) -> float:
    """Tissue-level Cauchy stress of the fibre family along its direction [kPa].

    In the in vivo (homeostatic) configuration the mixture is incompressible
    (``J = 1``) and the elastic fibre stretch equals the deposition stretch, so

        sigma_ef = 2 * I4e * dpsi_ef/dI4e
                 = phi_ef * c1 * I4e * (I4e - 1) * exp(c2 (I4e - 1)^2).

    Note
    ----
    Eq. 13 of the manuscript divides by ``J_alpha = det(G_ef) = lam_dep``.  That
    is inconsistent with constrained mixture theory, in which the push-forward
    of the constituent 2nd PK stress uses the *mixture* deformation gradient and
    its Jacobian (``J = 1`` in the in vivo configuration).  Using ``J_ef`` would
    underestimate the fibre stress by a factor ``lam_dep`` (10 % here).  The
    ``jacobian`` argument is exposed only so that this can be checked explicitly.
    """
    lam = p.lam_dep if lam is None else lam
    return lam * lam * fiber_cauchy_stress_2pk(p, lam) / jacobian


def effective_pressure(p: MaterialParams, sigma_ef: float, radius: float,
                       thickness: float) -> float:
    """Effective pressure acting on elastin (Sec. 2.4 of the manuscript).

    The fibre family carries ``sigma_ef`` circumferentially.  For a thin-walled
    tube the resultant of its divergence over the wall is a radial load
    ``sigma_ef * h / r``, so that elastin only has to equilibrate

        P_eff = P - sigma_ef * h / r.
    """
    return p.pressure - sigma_ef * thickness / radius


# ---------------------------------------------------------------------------
# Elastin: uncoupled neo-Hookean (Eq. 11)
# ---------------------------------------------------------------------------
def elastin_cauchy_principal(p: MaterialParams, lams) -> np.ndarray:
    """Principal Cauchy stresses of elastin for principal stretches ``lams``.

    psi = c (I1bar - 3) + K/2 (ln J)^2,  c = phi_e * c_e
    sigma = 2c/J * dev(Bbar) + K ln(J)/J * I
    """
    lams = np.asarray(lams, dtype=float)
    c = p.phi_e * p.c_e
    j = float(np.prod(lams))
    b_bar = j ** (-2.0 / 3.0) * lams ** 2
    dev = b_bar - b_bar.sum() / 3.0
    return 2.0 * c / j * dev + p.K * np.log(j) / j


def elastin_cauchy(p: MaterialParams, F: np.ndarray) -> np.ndarray:
    """Cauchy stress tensor of elastin for a deformation gradient ``F`` (3x3)."""
    F = np.asarray(F, dtype=float)
    c = p.phi_e * p.c_e
    j = float(np.linalg.det(F))
    b = F @ F.T
    b_bar = j ** (-2.0 / 3.0) * b
    dev = b_bar - np.trace(b_bar) / 3.0 * np.eye(3)
    return 2.0 * c / j * dev + p.K * np.log(j) / j * np.eye(3)
