"""0D analytical solution for the prestressed thin-walled cylinder (Sec. 2.4).

The in vivo geometry (r_ivo, h_ivo, l_ivo) and the fibre deposition stretch are
known.  The fibre family carries a known circumferential Cauchy stress, so the
elastin only has to equilibrate the *effective* pressure

    P_eff = P - sigma_ef * h_ivo / r_ivo.

The unloaded (stress-free) elastin dimensions R_o, H_o, L_o follow from

    sigma_e_tt - P_eff * r_ivo / h_ivo = 0      (Laplace, circumferential)
    sigma_e_zz                         = 0      (open tube, no axial force)
    sigma_e_rr                         = 0      (thin wall / plane stress)

three equations for the three unknown stretches.  Near-incompressibility is not
imposed as a constraint but follows from the volumetric penalty K/2 (ln J)^2 in
Eq. 11, so J is a genuine unknown of the system.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .material import effective_pressure, elastin_cauchy_principal, fiber_cauchy_stress
from .params import CylinderGeometry, MaterialParams

__all__ = ["Cylinder0DResult", "solve_cylinder_0d"]


@dataclass
class Cylinder0DResult:
    """Unloaded elastin geometry and prestretch of the thin-walled cylinder."""

    lam_t: float   # circumferential elastin prestretch
    lam_z: float   # axial
    lam_r: float   # radial
    R_inner: float  # unloaded inner radius     [mm]
    H: float        # unloaded wall thickness   [mm]
    L: float        # unloaded length           [mm]
    J: float
    sigma_ef: float       # fibre Cauchy stress, circumferential [kPa]
    sigma_e: np.ndarray   # elastin Cauchy stress (tt, zz, rr)   [kPa]
    p_eff: float          # effective pressure on elastin        [kPa]
    residual: np.ndarray
    radius_convention: str

    def __str__(self) -> str:
        return (
            f"0D thin-walled cylinder ({self.radius_convention}-radius Laplace)\n"
            f"  sigma_ef (circ.)      = {self.sigma_ef:9.4f} kPa\n"
            f"  P_effective           = {self.p_eff:9.4f} kPa\n"
            f"  sigma_e (tt, zz, rr)  = "
            f"{self.sigma_e[0]:9.4f} {self.sigma_e[1]:9.3e} {self.sigma_e[2]:9.3e} kPa\n"
            f"  prestretch (t, z, r)  = {self.lam_t:7.4f} {self.lam_z:7.4f} {self.lam_r:7.4f}"
            f"   (J = {self.J:.6f})\n"
            f"  unloaded R_i, H, L    = {self.R_inner:7.4f} {self.H:7.4f} {self.L:7.4f} mm\n"
            f"  |residual|            = {np.linalg.norm(self.residual):.3e} kPa"
        )


def solve_cylinder_0d(
    p: MaterialParams,
    geo: CylinderGeometry = CylinderGeometry(),
    radius_convention: str = "mid",
    lam_fiber: float | None = None,
    fiber_jacobian: float = 1.0,
) -> Cylinder0DResult:
    """Solve the 0D prestress problem for the thin-walled cylinder.

    Parameters
    ----------
    radius_convention
        ``"mid"`` uses the mid-wall radius in the Laplace relation (the
        consistent thin-wall choice, and what the 3D model reproduces);
        ``"inner"`` uses the inner radius, as written in the manuscript.
    lam_fiber
        Elastic fibre stretch; defaults to the deposition stretch ``p.lam_dep``.
    fiber_jacobian
        Divisor applied to the fibre Cauchy stress.  ``1.0`` is correct;
        ``p.lam_dep`` reproduces the ``1/J_alpha`` factor of Eq. 13.
    """
    if radius_convention == "mid":
        r_lap = geo.r_mid
    elif radius_convention == "inner":
        r_lap = geo.r_inner
    else:
        raise ValueError(f"unknown radius_convention {radius_convention!r}")

    sigma_ef = fiber_cauchy_stress(p, lam_fiber, jacobian=fiber_jacobian)
    p_eff = effective_pressure(p, sigma_ef, r_lap, geo.thickness)
    sigma_target = p_eff * r_lap / geo.thickness

    def residual(x):
        sig = elastin_cauchy_principal(p, np.exp(x))  # (tt, zz, rr)
        return np.array([sig[0] - sigma_target, sig[1], sig[2]])

    sol = least_squares(residual, np.zeros(3), xtol=1e-15, ftol=1e-15, gtol=1e-15)
    lam_t, lam_z, lam_r = np.exp(sol.x)
    sig = elastin_cauchy_principal(p, [lam_t, lam_z, lam_r])

    H = geo.thickness / lam_r
    L = geo.length / lam_z
    if radius_convention == "mid":
        R_inner = geo.r_mid / lam_t - 0.5 * H
    else:
        R_inner = geo.r_inner / lam_t

    return Cylinder0DResult(
        lam_t=lam_t, lam_z=lam_z, lam_r=lam_r,
        R_inner=R_inner, H=H, L=L, J=lam_t * lam_z * lam_r,
        sigma_ef=sigma_ef, sigma_e=sig, p_eff=p_eff,
        residual=sol.fun, radius_convention=radius_convention,
    )
