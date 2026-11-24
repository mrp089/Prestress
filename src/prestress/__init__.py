"""Homeostatic prestressed reference configurations for constrained mixture models.

Python/FEBio re-implementation of the augmented iterative method (AIM) of
Parikh, Pfaller & Loerakker, adapted to constrained mixture models.

Units throughout: length [mm], stress and pressure [kPa], force [mN].
"""

from .params import CylinderGeometry, MaterialParams, PRESETS, mmHg_to_kPa
from .material import (
    effective_pressure,
    elastin_cauchy,
    elastin_cauchy_principal,
    fiber_cauchy_stress,
    fiber_cauchy_stress_2pk,
)
from .analytic import Cylinder0DResult, solve_cylinder_0d

__all__ = [
    "CylinderGeometry",
    "MaterialParams",
    "PRESETS",
    "mmHg_to_kPa",
    "effective_pressure",
    "elastin_cauchy",
    "elastin_cauchy_principal",
    "fiber_cauchy_stress",
    "fiber_cauchy_stress_2pk",
    "Cylinder0DResult",
    "solve_cylinder_0d",
]
