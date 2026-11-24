"""Integration checks that need FEBio and the compiled plugin.

    python tests/test_febio.py [workdir]

Each takes a few seconds on the 4032-element cylinder.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prestress import PRESETS, CylinderGeometry, solve_cylinder_0d
from prestress.aim import make_solver, run_aim
from prestress.febio import read_node_log, run_febio, write_cmm_feb, write_elastin_feb
from prestress.kinematics import prestretch_maps
from prestress.mesh import build_cylinder_mesh


def test_plugin_matches_builtin(work: Path):
    """With no fibres, prestress_cmm must reproduce FEBio's Mooney-Rivlin.

    Both then use the same three-field HEX8, so the AIM and the G&R stage of
    the study are discretised identically.
    """
    mesh = build_cylinder_mesh()
    p = PRESETS["implied"]
    a = solve_cylinder_0d(p, CylinderGeometry(), radius_convention="mid")

    f1 = write_elastin_feb(work / "mr" / "m.feb", mesh, p, a.p_eff)
    run_febio(f1)
    x1 = read_node_log(work / "mr" / "m_pos.txt", mesh.n_node)

    f2 = write_cmm_feb(work / "pl" / "p.feb", mesh, p, a.p_eff, mesh.nodes,
                       phi_c=0.0)
    run_febio(f2, plugin=True)
    x2 = read_node_log(work / "pl" / "p_pos.txt", mesh.n_node)

    err = np.abs(x1 - x2).max()
    assert err < 1e-10, f"plugin differs from Mooney-Rivlin by {err:.3e} mm"
    return err


def test_aim_matches_0d(work: Path):
    """The converged AIM prestretch must match the thin-wall solution."""
    geo = CylinderGeometry()
    mesh = build_cylinder_mesh(geo)
    p = PRESETS["paper"]
    a = solve_cylinder_0d(p, geo, radius_convention="mid")

    solve = make_solver(mesh, p, a.p_eff, work / "aim", mode="effective")
    r = run_aim(mesh, solve, max_iter=40, verbose=False)
    assert r.final_residual < 1e-9

    from prestress.kinematics import deformation_gradient, principal_stretches
    lam, _ = principal_stretches(deformation_gradient(mesh, r.X, mesh.nodes))
    med = np.median(lam.reshape(-1, 3), axis=0)
    ref = np.array([a.lam_t, a.lam_z, a.lam_r])
    err = np.abs(med / ref - 1).max()
    assert err < 0.01, f"AIM and 0D differ by {100 * err:.2f} %"
    return err


def test_prestressed_mixture_is_in_equilibrium(work: Path):
    """``mode="embedded"`` solves the exact mixture problem.

    Loading its result as a full mixture must reproduce the in vivo geometry to
    the AIM residual -- this is what catches a permuted integration-point map.
    """
    geo = CylinderGeometry()
    mesh = build_cylinder_mesh(geo)
    p = PRESETS["implied"]

    solve = make_solver(mesh, p, p.pressure, work / "emb", mode="embedded")
    r = run_aim(mesh, solve, max_iter=40, verbose=False)

    lam_ref, J_ref = prestretch_maps(mesh, r.X, mesh.nodes)
    f = write_cmm_feb(work / "chk" / "c.feb", mesh, p, p.pressure, r.X,
                      lam_ref=lam_ref, J_ref=J_ref, phi_c=p.phi_ef, k_d=0.0)
    run_febio(f, plugin=True)
    x = read_node_log(work / "chk" / "c_pos.txt", mesh.n_node)

    err = np.abs(x - mesh.nodes).max()
    assert err < 100 * max(r.final_residual, 1e-11), (
        f"mixture is {err:.3e} mm from the in vivo geometry, "
        f"AIM residual was {r.final_residual:.3e} mm"
    )
    return err


if __name__ == "__main__":
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    for fn in (test_plugin_matches_builtin, test_aim_matches_0d,
               test_prestressed_mixture_is_in_equilibrium):
        v = fn(base / fn.__name__)
        print(f"ok  {fn.__name__}   ({v:.3e})")
    print("\n3 integration tests passed")
