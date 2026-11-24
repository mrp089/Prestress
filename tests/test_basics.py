"""Fast checks that do not need FEBio.

    python -m pytest tests -q        (or just: python tests/test_basics.py)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prestress import PRESETS, CylinderGeometry, solve_cylinder_0d
from prestress.kinematics import deformation_gradient, prestretch_maps
from prestress.material import elastin_cauchy, elastin_cauchy_principal, fiber_cauchy_stress
from prestress.mesh import GAUSS_2x2x2, _HEX_CORNERS, build_cylinder_mesh


def test_gauss_order_matches_febio():
    """FEBio's HEX8 2x2x2 rule follows the node ordering (FEHex8G8).

    Any other order silently permutes MAT_POINTS element maps, which is very
    hard to spot: the AIM still converges, but the fibre stretch is attached to
    the wrong integration point.
    """
    expected = (
        np.array(
            [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
             [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float
        )
        / np.sqrt(3.0)
    )
    assert np.allclose(GAUSS_2x2x2, expected)
    assert np.allclose(GAUSS_2x2x2, _HEX_CORNERS / np.sqrt(3.0))


def test_mesh_is_valid():
    m = build_cylinder_mesh()
    assert m.n_elem == 4032
    assert m.jacobians().min() > 0


def test_identity_kinematics():
    m = build_cylinder_mesh()
    F = deformation_gradient(m, m.nodes, m.nodes)
    assert np.allclose(F, np.eye(3), atol=1e-12)
    lam, J = prestretch_maps(m, m.nodes, m.nodes)
    assert np.allclose(lam, 1.0, atol=1e-12)
    assert np.allclose(J, 1.0, atol=1e-12)


def test_homogeneous_kinematics():
    m = build_cylinder_mesh()
    X = m.nodes.copy()
    X[:, :2] *= 0.9
    X[:, 2] *= 1.05
    lam, J = prestretch_maps(m, X, m.nodes)
    assert np.allclose(lam, 1 / 0.9)
    assert np.allclose(J, 1 / (0.9**2 * 1.05))


def test_elastin_stress_consistent():
    p = PRESETS["paper"]
    lams = np.array([1.1, 0.97, 0.94])
    s1 = elastin_cauchy_principal(p, lams)
    s2 = np.diag(elastin_cauchy(p, np.diag(lams)))
    assert np.allclose(s1, s2)


def test_fiber_stress_value():
    p = PRESETS["paper"]
    i4 = p.lam_dep**2
    expected = p.phi_ef * p.c1 * i4 * (i4 - 1) * np.exp(p.c2 * (i4 - 1) ** 2)
    assert np.isclose(fiber_cauchy_stress(p), expected)


def test_0d_solution_is_equilibrium():
    for name in PRESETS:
        for conv in ("inner", "mid"):
            r = solve_cylinder_0d(PRESETS[name], CylinderGeometry(),
                                  radius_convention=conv)
            assert np.linalg.norm(r.residual) < 1e-8
            assert np.isclose(r.lam_t * r.lam_z * r.lam_r, r.J)


def test_0d_reproduces_reported_dimensions():
    """c_e = 262 kPa reproduces the unloaded dimensions reported in the draft."""
    r = solve_cylinder_0d(PRESETS["paper"].with_(c_e=262.015),
                          radius_convention="inner")
    assert abs(r.R_inner - 0.29) < 0.01
    assert abs(r.H - 0.042) < 0.0005
    assert abs(r.L - 2.10) < 0.005


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
