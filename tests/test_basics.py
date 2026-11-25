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


def test_tet10_matches_febio():
    """The tet10 rule and shape functions reproduce FEBio's FETet10G8.

    FEBio's default for tet10 is the 8-point rule (``FEModelBuilder.cpp``
    sets ``m_ntet10 = FE_TET10G8``), not the 4-point one.  A wrong rule is
    silent: the AIM still converges, just to a different configuration.
    """
    import numpy as np
    from prestress.elements import element_for

    e = element_for("tet10")
    assert e.n_gp == 8 and e.n_node == 10

    # the weights integrate the reference tetrahedron exactly
    assert abs(e.weights.sum() - 1.0 / 6.0) < 1e-12

    N, G = e.values(), e.grads()
    assert np.allclose(N.sum(axis=1), 1.0)          # partition of unity
    assert np.allclose(G.sum(axis=1), 0.0)          # gradients sum to zero

    # a quadratic element reproduces a quadratic field exactly, so the rule
    # must integrate x^2 over the unit tet exactly: int r^2 dV = 1/60
    xi = e.points
    assert abs((e.weights * xi[:, 0] ** 2).sum() - 1.0 / 60.0) < 1e-9

    # shape functions are nodal: N_a(xi_b) = delta_ab at the ten nodes
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                      [.5, 0, 0], [.5, .5, 0], [0, .5, 0],
                      [0, 0, .5], [.5, 0, .5], [0, .5, .5]], dtype=float)
    H = np.stack([e._val(p) for p in nodes])
    assert np.allclose(H, np.eye(10), atol=1e-12)


def test_heart_surfaces_are_outward():
    """The tri6 faces pulled off the tet10 heart close and point outward."""
    import numpy as np

    vtu = Path(__file__).resolve().parents[1] / "eva.vtu"
    if not vtu.exists():        # the heart mesh is not in the repository
        return

    from prestress.elements import element_for
    from prestress.heart import _boundary_faces, load_heart

    h = load_heart(str(vtu))
    t = h.nodes[_boundary_faces(h.elements)[:, :3]]
    v_surf = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0

    e = element_for("tet10")
    det = np.linalg.det(np.einsum("ena,gnb->egab", h.nodes[h.elements], e.grads()))
    v_quad = float((det * e.weights[None, :]).sum())

    assert det.min() > 0.0                                   # no inverted elements
    assert v_surf > 0.0                                      # outward orientation
    assert abs(v_surf - v_quad) / v_quad < 1e-10             # the boundary closes




def test_gebauer23_parameters():
    """Table 1 of Gebauer et al. (2023), and the J/kg -> kPa conversion.

    Their stiffnesses are energies per unit *mass*, so every one of them is
    meaningless until multiplied by a reference density.  Getting that step
    wrong scales every fibre stress by ~10^3 and still runs, so the expected
    values here are worked out independently of the code:

        sigma = 2 rho0^i I4 a (I4 - 1) exp[b (I4 - 1)^2],  I4 = lam_h^2

    collagen, one of four families:
        rho0 = 0.1/4 * 1050 = 26.25 kg/m^3,  I4 = 1.062^2 = 1.127844
        sigma = 2*26.25*1.127844*568*0.127844*exp(11.2*0.127844^2)
              = 5163.6 Pa
    myocytes:
        rho0 = 0.6 * 1050 = 630 kg/m^3,  I4 = 1.21
        sigma = 2*630*1.21*7.6*0.21*exp(11.4*0.21^2) = 4022.8 Pa
    """
    from prestress.params import (GEBAUER23, GEBAUER23_KAPPA, GEBAUER23_RHO0,
                                  PRESETS, gebauer23_active_stress_kPa)

    rho = GEBAUER23_RHO0
    assert rho == 1050.0
    # mass fractions must partition the mixture
    assert abs(sum(c.mass_fraction for c in GEBAUER23.values()) - 1.0) < 1e-12

    col = GEBAUER23["collagen"].sigma_h_kPa(rho)
    myo = GEBAUER23["myocytes"].sigma_h_kPa(rho)
    assert abs(col - 5.1636) < 1e-3, col
    assert abs(myo - 4.0230) < 1e-3, myo
    # the isotropic constituent carries no fibre stress
    assert GEBAUER23["elastin"].sigma_h_kPa(rho) == 0.0
    # rho0^e * a^e, which is what FEBio receives as Mooney-Rivlin c1
    assert abs(GEBAUER23["elastin"].stiffness_kPa(rho) - 22.68) < 1e-9

    # active tone at the homeostatic stretch, their Eq. (8)
    #   630 * 1.1 * (54/1050) * (1 - (0.3/0.6)^2) = 26.73 kPa
    assert abs(gebauer23_active_stress_kPa(1.1) - 26.73) < 1e-2

    m = PRESETS["heart_cmm"]
    assert abs(m.phi_e * m.c_e - 22.68) < 1e-9
    assert abs(m.K - GEBAUER23_KAPPA) < 1e-9
    assert m.mr_c2 == 0.0


def test_heart_cmm_families_are_orthonormal():
    """The five fibre families, including where fiber1 and fiber2 are parallel.

    About 1 % of nodes in the great vessels carry a sheet direction parallel
    to the fibre direction, which makes Gram-Schmidt singular.  The fallback
    must still return unit vectors at the requested angles.
    """
    from prestress.heart import cmm_fiber_families

    class _Fake:
        pass

    rng = np.random.default_rng(0)
    n = 500
    f = rng.normal(size=(n, 3))
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    s = rng.normal(size=(n, 3))
    s -= (s * f).sum(1)[:, None] * f
    s /= np.linalg.norm(s, axis=1, keepdims=True)
    s[:50] = f[:50]                      # degenerate: sheet parallel to fibre

    h = _Fake()
    h.fiber1, h.fiber2 = f, s
    dirs, labels = cmm_fiber_families(h)

    assert len(dirs) == 5 and labels[0] == "myocytes"
    for d in dirs:
        assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-12)
    # family angles measured against the myocyte direction
    for d, want in zip(dirs[1:], (1.0, 0.0, np.cos(np.pi / 4), np.cos(np.pi / 4))):
        got = np.abs((d * dirs[0]).sum(1))
        assert np.allclose(got, want, atol=1e-10), (labels, got[:3], want)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
