"""Geometry, material and solver parameters.

Units throughout: length [mm], stress/pressure [kPa], force [mN].
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

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
}
