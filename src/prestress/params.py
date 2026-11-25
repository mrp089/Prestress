"""Geometry, material and solver parameters.

Units throughout: length [mm], stress/pressure [kPa], force [mN].
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

MMHG_TO_KPA = 0.1333223684  # 1 mmHg in kPa


def mmHg_to_kPa(p: float) -> float:
    return p * MMHG_TO_KPA


@dataclass(frozen=True)
class CylinderGeometry:
    """In vivo (imaged) dimensions of the idealised mouse descending thoracic aorta."""

    r_inner: float = 0.33   # in vivo inner radius  [mm]
    thickness: float = 0.04  # in vivo wall thickness [mm]
    length: float = 2.0     # in vivo length         [mm]

    # mesh resolution: n_theta * n_r * n_z = 48 * 3 * 28 = 4032 hexahedra (as in the paper)
    n_theta: int = 48
    n_r: int = 3
    n_z: int = 28

    @property
    def r_mid(self) -> float:
        return self.r_inner + 0.5 * self.thickness

    @property
    def r_outer(self) -> float:
        return self.r_inner + self.thickness

    @property
    def n_elem(self) -> int:
        return self.n_theta * self.n_r * self.n_z


@dataclass(frozen=True)
class MaterialParams:
    """Constrained mixture: isotropic elastin + one circumferential fibre family.

    Elastin (Eq. 11 of the manuscript), an *uncoupled* neo-Hookean solid

        psi_e = phi_e * c_e * (I1bar - 3) + K/2 * (ln J)^2

    so that the elastin shear modulus at the tissue level is ``2 * phi_e * c_e``.
    This maps one-to-one onto the FEBio ``Mooney-Rivlin`` material with
    ``c1 = phi_e * c_e``, ``c2 = 0`` and ``k = K``.

    Embedded fibres (Eq. 10), deposited at stretch ``lam_dep``

        psi_ef = phi_ef * c1/(4 c2) * (exp(c2 (I4e - 1)^2) - 1),   I4e = lam_dep^2
    """

    name: str = "paper"

    # mixture composition
    phi_ef: float = 0.33            # mass fraction of the turning-over fibre family
    # elastin
    c_e: float = 1000.0             # [kPa]  stiffness-like parameter of elastin
    bulk_ratio: float = 1000.0      # K = bulk_ratio * c_e
    mr_c2: float = 0.0              # [kPa] second Mooney-Rivlin parameter of
                                    # the ground substance.  Zero is Eq. 11;
                                    # the heart of Pfaller et al. (2019) uses
                                    # an isochoric Mooney-Rivlin with c2 != 0.
    # fibres
    c1: float = 235.0               # [kPa]
    c2: float = 4.08                # [-]
    lam_dep: float = 1.1            # deposition (pre-)stretch of the fibre family
    n_families: int = 1             # fibre families sharing phi_ef
    fiber_angle_deg: float = 0.0    # +/- this angle from the local axial direction
    # loading
    pressure_mmHg: float = 120.0

    @property
    def phi_e(self) -> float:
        return 1.0 - self.phi_ef

    @property
    def phi_per_family(self) -> float:
        """Mass fraction of one fibre family."""
        return self.phi_ef / self.n_families

    @property
    def K(self) -> float:
        return self.bulk_ratio * self.c_e

    @property
    def mu_e(self) -> float:
        """Tissue-level elastin shear modulus [kPa]."""
        return 2.0 * self.phi_e * self.c_e

    @property
    def pressure(self) -> float:
        """In vivo transmural pressure [kPa]."""
        return mmHg_to_kPa(self.pressure_mmHg)

    def with_(self, **kw) -> "MaterialParams":
        return replace(self, **kw)


# ---------------------------------------------------------------------------
# Homogenized constrained mixture of the myocardium (Gebauer et al. 2023)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CMMConstituent:
    """One constituent of a homogenized constrained mixture.

    Gebauer et al. (Biomech Model Mechanobiol 22:1983-2002, 2023) write the
    mixture energy per unit *mass*, ``Psi = sum_i rho0^i W^i`` (their Eq. 4),
    so every stiffness in their Table 1 is in J/kg and only becomes a stress
    once multiplied by a reference mass density in kg/m^3.  That conversion is
    the one place this is easy to get silently wrong, so it lives here.

    Fibre constituents use the Fung form of their Eq. (5),

        W = a / (2 b) * (exp[b (I4 - 1)^2] - 1),

    and ``b = 0`` marks the isotropic neo-Hookean constituent (elastin, their
    Eq. 9, ``W = a (I1bar - 3)``).
    """

    name: str
    mass_fraction: float            # xi^i of the total reference density
    a: float                        # [J/kg]
    b: float = 0.0                  # [-]  Fung exponent; 0 = neo-Hookean
    lam_h: float = 1.0              # homeostatic (deposition) stretch
    n_families: int = 1             # families sharing ``mass_fraction``

    def density(self, rho0: float) -> float:
        """Reference density of one family [kg/m^3]."""
        return self.mass_fraction * rho0 / self.n_families

    def stiffness_kPa(self, rho0: float) -> float:
        """``rho0^i * a`` of one family, as a stress [kPa]."""
        return self.density(rho0) * self.a * 1e-3

    def sigma_h_kPa(self, rho0: float) -> float:
        """Cauchy stress of one family at its homeostatic stretch [kPa].

        With ``W`` per unit mass and ``I4 = lam_h^2``,

            sigma = 2 rho0^i I4 dW/dI4
                  = 2 rho0^i I4 a (I4 - 1) exp[b (I4 - 1)^2].
        """
        if self.b == 0.0:
            return 0.0                      # isotropic: no fibre direction
        i4 = self.lam_h * self.lam_h
        return (2.0 * self.density(rho0) * i4 * self.a * (i4 - 1.0)
                * np.exp(self.b * (i4 - 1.0) ** 2) * 1e-3)


# Table 1 of Gebauer et al. (2023), verbatim.  The material parameters and
# homeostatic stretches are themselves adopted there from Braeu et al. (2016).
GEBAUER23_RHO0 = 1050.0             # [kg/m^3] total reference mass density
GEBAUER23_KAPPA = 150.0             # [kPa] volumetric penalty
GEBAUER23 = {
    # four collagen families: myocyte-fibre, cross-fibre and +/-45 degrees,
    # with the mass fraction distributed equally between them
    "collagen": CMMConstituent("collagen", 0.1, a=568.0, b=11.2,
                               lam_h=1.062, n_families=4),
    "myocytes": CMMConstituent("myocytes", 0.6, a=7.6, b=11.4, lam_h=1.1),
    "elastin": CMMConstituent("elastin", 0.3, a=72.0),
}
# Active muscle tone, their Eq. (8) after Wilson et al. (2012).  It is part of
# their *systolic* state; the four-chamber heart here is prestressed at end
# diastole with the active stress off, as in Pfaller et al. (2019).
GEBAUER23_ACTIVE = dict(sigma_act_kPa=54.0, lam_0=0.8, lam_max=1.4)


def gebauer23_active_stress_kPa(lam: float = 1.1) -> float:
    """Active myocyte Cauchy stress at fibre stretch ``lam`` [kPa].

    From ``W_act = (sigma_act / rho0) (lam + (lam_max - lam)^3 /
    (3 (lam_max - lam_0)^2))``, the Cauchy stress is ``rho0^m lam dW/dlam``.
    """
    g = GEBAUER23_ACTIVE
    m = GEBAUER23["myocytes"]
    dw = (g["sigma_act_kPa"] / GEBAUER23_RHO0) * (
        1.0 - ((g["lam_max"] - lam) / (g["lam_max"] - g["lam_0"])) ** 2)
    return m.density(GEBAUER23_RHO0) * lam * dw


# ---------------------------------------------------------------------------
# Parameter sets
# ---------------------------------------------------------------------------
# "paper"  -- Table 1 of the manuscript, verbatim.
# "mouse"  -- the same mixture, but with the elastin modulus and the collagen
#             deposition stretch taken consistently from the mouse descending
#             thoracic aorta of Latorre & Humphrey (CMAME 2020), which is the
#             source of c1 and c2 in Table 1.  There,
#                 psi_e = phi_e * mu/2 * (I1 - 3)  with  mu = 89.71 kPa,
#             i.e. c_e = mu/2 = 44.855 kPa in the convention of Eq. 11, and the
#             collagen deposition stretch is G_c = 1.25 (lam_dep = 1.1 is the
#             *active* smooth-muscle stretch lam_M in that reference).
# "implied" -- the elastin modulus that reproduces the unloaded dimensions
#             reported in the manuscript; the least-squares fit to the reported
#             unloaded length is 262 kPa (see docs/corrections.md).
# "aorta"  -- the human thoracic aorta of Sec. 2.4: elastin and two diagonal
#             fibre families at +/-30 degrees to the axial direction, at
#             130 mmHg, with the elastin modulus of Parikh et al. 2023.
PRESETS: dict[str, MaterialParams] = {
    "paper": MaterialParams(name="paper"),
    "aorta": MaterialParams(name="aorta", phi_ef=0.40, c_e=1000.0,
                            n_families=2, fiber_angle_deg=30.0,
                            pressure_mmHg=130.0),
    "mouse": MaterialParams(name="mouse", c_e=89.71 / 2.0, lam_dep=1.25),
    "implied": MaterialParams(name="implied", c_e=260.0),
    # Passive myocardium of Pfaller et al. (2019): an isochoric Mooney-Rivlin
    # (C1 = 7.0, C2 = 0.04 kPa) plus a Sussman-Bathe volumetric penalty
    # (kappa = 1e3 kPa).  No turning-over fibre family and no active stress --
    # the active stress is switched off while prestressing -- so the heart is
    # a purely elastic AIM problem.  phi_ef = 0 makes phi_e = 1, so the
    # written c1 is c_e itself.  pressure_mmHg is the LV end-diastolic value;
    # the other three chambers are set in prestress.heart.
    "heart": MaterialParams(name="heart", phi_ef=0.0, c_e=7.0, mr_c2=0.04,
                            bulk_ratio=1e3 / 7.0, pressure_mmHg=10.0),
    # The same four-chamber geometry as a constrained mixture, with the
    # myocardium of Gebauer et al. (2023).  Only the elastin reaches FEBio as
    # a material; the myocyte and collagen families enter as nodal forces, so
    # phi_ef = 0 and the written c1 is rho0^e * a^e = 22.68 kPa.
    "heart_cmm": MaterialParams(
        name="heart_cmm", phi_ef=0.0,
        c_e=GEBAUER23["elastin"].stiffness_kPa(GEBAUER23_RHO0),
        mr_c2=0.0,
        bulk_ratio=GEBAUER23_KAPPA
        / GEBAUER23["elastin"].stiffness_kPa(GEBAUER23_RHO0),
        pressure_mmHg=10.0),
}
