#!/usr/bin/env python3
"""Prestress a patient-specific aorta from the Vascular Model Repository.

The manuscript's second example is a patient-specific human aorta with elastin
and two diagonal fibre families.  This builds the equivalent test case from a
VMR aortofemoral model (see docs/aorta.md) and compares three ways of taking
the prescribed fibre stress into account:

  surface   Eq. 9 taken literally: the fibres enter only through the surface
            traction ``sigma_ef . n`` on the lumen
  laplace   the thin-wall relief of Sec. 2.4, applied pointwise with the local
            thickness and radius
  nodal     the exact weak form: ``f_a = -int sigma_ef . grad(N_a) dv``

    python scripts/run_aorta.py [--variants surface laplace nodal]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prestress import PRESETS
from prestress.aim import make_force_solver, run_aim
from prestress.febio import wall_model
from prestress.kinematics import (fiber_stress_tensors, laplace_relief_forces,
                                  nodal_forces_from_stress, surface_fiber_forces)
from prestress.material import fiber_cauchy_stress
from prestress.vtk import write_vtk
from prestress.wall import (extrude_wall, fiber_directions, load_lumen_surface,
                            local_radius, smooth_normals)

SRC = "/Users/pfaller/repos/Aorta/mesh_example/smoothed/walls_combined.vtp"
VARIANTS = ("surface", "laplace", "nodal")


def build_wall(cache: Path, src: str, thickness_frac: float):
    """Extrude the wall, caching the expensive ray-cast radius."""
    surf = load_lumen_surface(src)
    if cache.is_file():
        r = np.load(cache)["radius"]
    else:
        r = local_radius(surf)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, radius=r)
    h = np.clip(thickness_frac * r, 0.2, 2.0)
    f1, f2 = fiber_directions(surf, PRESETS["aorta"].fiber_angle_deg)
    nrm = smooth_normals(surf, 5)
    mesh = extrude_wall(surf, h, n_layers=2,
                        node_fields={"fib1": f1, "fib2": f2, "radius": r,
                                     "thickness": h, "normal": nrm})
    return surf, mesh, r, f1, f2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--outdir", default="output/aorta")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--thickness-frac", type=float, default=0.10)
    ap.add_argument("--max-iter", type=int, default=25)
    ap.add_argument("--n-steps", type=int, default=10,
                      help="load steps per AIM solve (auto stepper sub-divides)")
    # The cylinder can be driven to 1e-16: it is small and nearly linear.  A
    # 60k-dof patient-specific mesh cannot, and demanding it only burns through
    # max_refs and reports "Max nr of reformations reached" on a solve that had
    # in fact converged.  The AIM floor is then set by these, not by the
    # precision of the log file.
    ap.add_argument("--rtol", type=float, default=1e-10)
    ap.add_argument("--etol", type=float, default=1e-12)
    ap.add_argument("--min-residual", type=float, default=1e-12)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    workroot = Path(args.workdir) if args.workdir else outdir / "febio"

    mat = PRESETS["aorta"]
    surf, mesh, radius, f1, f2 = build_wall(outdir / "radius.npz", args.src,
                                            args.thickness_frac)
    model = wall_model(mesh)
    print(f"wall: {model.n_elem} wedges, {model.n_node} nodes, "
          f"{len(mesh.lumen_facets)} lumen facets, {len(mesh.ring_nodes)} clamped")

    # per-node fields come back from extrude_wall already aligned with the
    # (possibly renumbered) mesh
    nf = mesh.node_fields
    F1, F2, thick, rad = nf["fib1"], nf["fib2"], nf["thickness"], nf["radius"]

    sigma = fiber_cauchy_stress(mat)
    families = [(sigma, F1), (sigma, F2)]
    S_vol = fiber_stress_tensors(model.elements, model.elem_type, families)
    S_fac = _facet_stress(mesh.lumen_facets, families)

    # the circumferential share of the two families, for the thin-wall relief
    a = np.radians(mat.fiber_angle_deg)
    hoop = 2.0 * sigma * np.sin(a) ** 2

    forces = {
        "surface": surface_fiber_forces(mesh.nodes, mesh.lumen_facets, S_fac),
        "laplace": laplace_relief_forces(mesh.nodes, mesh.lumen_facets, hoop,
                                         thick, rad),
        "nodal": nodal_forces_from_stress(mesh.nodes, model.elements,
                                          model.elem_type, S_vol),
    }
    print(f"sigma_ef per family {sigma:.3f} kPa, hoop share {hoop:.3f} kPa, "
          f"pressure {mat.pressure:.3f} kPa")
    for k, f in forces.items():
        print(f"  {k:8s} |f| max {np.linalg.norm(f, axis=1).max():.4e} mN, "
              f"resultant {np.linalg.norm(f.sum(0)):.4e} mN")

    summary, configs = {}, {}
    for v in args.variants:
        print("=" * 78)
        print(f"AIM, {v}")
        print("=" * 78)
        solve = make_force_solver(model, mat, mat.pressure, workroot / v,
                                  forces=forces[v], rtol=args.rtol,
                                  etol=args.etol, min_residual=args.min_residual,
                                  n_steps=args.n_steps, auto_step=True)
        try:
            aim = run_aim(model, solve, max_iter=args.max_iter)
        except RuntimeError as exc:
            # A variant that cannot be loaded at all is a result, not a crash:
            # keep going so the ones that work are still compared.
            print(f"!! {v}: FEBio did not converge -- {str(exc).splitlines()[0]}")
            summary[v] = dict(failed=True)
            continue
        configs[v] = aim.X
        summary[v] = dict(iterations=aim.iterations,
                          residual=list(map(float, aim.residual)))
        np.savez_compressed(outdir / f"aim_{v}.npz", X=aim.X,
                            residual=np.array(aim.residual))
        write_vtk(outdir / f"aorta_{v}.vtk", aim.X, model.elements,
                  point_data={"in_vivo_displacement": mesh.nodes - aim.X},
                  title=f"AIM stress-free aorta ({v})")

    if "nodal" in configs:
        for v in configs:
            if v == "nodal":
                continue
            d = np.abs(configs[v] - configs["nodal"]).max()
            summary[v]["vs_nodal"] = float(d)
            print(f"\nmax |X_{v} - X_nodal| = {d:.4e} mm")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote results to {outdir}")


def _facet_stress(facets, families):
    """Fibre stress tensor on each triangular facet (directions averaged)."""
    S = None
    for sig, dirs in families:
        m = np.asarray(dirs)[facets].mean(axis=1)
        m /= np.linalg.norm(m, axis=-1, keepdims=True)
        term = sig * np.einsum("fi,fj->fij", m, m)
        S = term if S is None else S + term
    return S


if __name__ == "__main__":
    main()
