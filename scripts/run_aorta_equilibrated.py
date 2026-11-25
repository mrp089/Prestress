#!/usr/bin/env python3
"""How far does an imperfectly prestressed aorta end up from the image?

The question Gebauer et al. (BMMB 2023) asked of the generalized prestressing
algorithm, asked here of the AIM on a patient-specific geometry.  A
prestressing algorithm is always stopped at a finite residual, so the
prestressed state is not exactly in mechanobiological equilibrium, and the
vessel moves even though the loading never changes.

The AIM is stopped at a ladder of residuals; each resulting stress-free
elastin configuration is then loaded as the **equilibrated** constrained
mixture -- every family back at its deposition stretch, carrying the
prescribed homeostatic Cauchy stress along its current direction -- and the
distance from the imaged geometry is the spurious displacement.

Run it below the pressure at which the AIM stops converging (see
docs/aorta.md): the default is 97 mmHg, where the AIM reaches 1e-3 mm.

    python scripts/run_aorta_equilibrated.py [--pressure-mmhg 97]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prestress import PRESETS
from prestress.aim import make_force_solver, run_aim
from prestress.febio import wall_model
from prestress.growth import equilibrated_forces_residual
from prestress.kinematics import fiber_stress_tensors, nodal_forces_from_stress
from prestress.material import fiber_cauchy_stress
from prestress.vtk import write_vtk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_aorta import SRC, build_wall


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--outdir", default="output/aorta_equil")
    ap.add_argument("--thickness-frac", type=float, default=0.10)
    ap.add_argument("--pressure-mmhg", type=float, default=97.0)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3])
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--n-steps", type=int, default=10)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--etol", type=float, default=1e-8)
    ap.add_argument("--min-residual", type=float, default=1e-10)
    ap.add_argument("--max-ups", type=int, default=10)
    ap.add_argument("--opt-iter", type=int, default=30)
    ap.add_argument("--beta-max", type=float, default=8.0)
    ap.add_argument("--anderson", type=int, default=5)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    surf, mesh, radius, _, _ = build_wall(
        Path("output/aorta/radius.npz"), args.src, args.thickness_frac)
    model = wall_model(mesh)
    nf = mesh.node_fields
    fibers = [nf["fib1"], nf["fib2"]]

    mat = dataclasses.replace(PRESETS["aorta"],
                              pressure_mmHg=args.pressure_mmhg)
    sig = fiber_cauchy_stress(mat)
    print(f"wall: {model.n_elem} wedges, {model.n_node} nodes")
    print(f"pressure {mat.pressure:.3f} kPa ({args.pressure_mmhg:g} mmHg), "
          f"sigma_h {sig:.3f} kPa per family")

    feb_kw = dict(rtol=args.rtol, etol=args.etol,
                  min_residual=args.min_residual, max_ups=args.max_ups,
                  opt_iter=args.opt_iter, auto_step=True, laugon=False)

    # the exact fibre contribution, as dead nodal forces on the imaged mesh
    # the wall's own node fields, not the surface ones build_wall returns:
    # extrude_wall repeats and renumbers them to match model.elements
    S = fiber_stress_tensors(model.elements, model.elem_type,
                             [(sig, fibers[0]), (sig, fibers[1])])
    forces = nodal_forces_from_stress(mesh.nodes, model.elements,
                                      model.elem_type, S)

    rows = []
    for thr in sorted(args.thresholds, reverse=True):
        tag = f"thr_{thr:.0e}".replace("-", "m").replace("+", "")
        print("=" * 72)
        print(f"AIM stopped at {thr:.1e} mm")
        print("=" * 72)
        solve = make_force_solver(model, mat, mat.pressure,
                                  outdir / "febio" / tag, forces=forces,
                                  n_steps=args.n_steps, **feb_kw)
        aim = run_aim(model, solve, max_iter=args.max_iter, tol=thr,
                      beta_max=args.beta_max, anderson=args.anderson)
        if not aim.converged:
            print(f"!! AIM did not reach {thr:.1e} mm "
                  f"(stopped at {aim.residual[-1]:.3e})")
        r_aim = float(aim.residual[-1])

        print(f"  equilibrating from the {r_aim:.3e} mm configuration")
        r_eq, x_eq, hist = equilibrated_forces_residual(
            model, mat, aim.X, mesh.nodes, fibers,
            outdir / "febio" / f"{tag}_equil", sigma=sig,
            n_steps=args.n_steps, **feb_kw)

        U = np.linalg.norm(x_eq - mesh.nodes, axis=1)
        rows.append(dict(threshold=thr, aim_residual=r_aim,
                         aim_iterations=aim.iterations,
                         equilibrated_max=r_eq,
                         equilibrated_p50=float(np.percentile(U, 50)),
                         equilibrated_p99=float(np.percentile(U, 99)),
                         equil_iterations=len(hist)))
        print(f"  -> spurious displacement: max {r_eq:.4e}  "
              f"p99 {np.percentile(U, 99):.4e}  median "
              f"{np.percentile(U, 50):.4e} mm   (ratio to AIM "
              f"{r_eq / r_aim:.2f})")
        write_vtk(outdir / f"equilibrated_{tag}.vtk", mesh.nodes,
                  model.elements,
                  point_data={"spurious_displacement": x_eq - mesh.nodes,
                              "magnitude_mm": U},
                  title=f"equilibrated state, AIM stopped at {thr:.0e} mm")

    (outdir / "summary.json").write_text(json.dumps(rows, indent=2))
    print("\n AIM residual [mm]   equilibrated max [mm]   ratio")
    for r in rows:
        print(f"   {r['aim_residual']:.3e}          {r['equilibrated_max']:.3e}"
              f"        {r['equilibrated_max'] / r['aim_residual']:.2f}")
    print(f"\nwrote results to {outdir}")


if __name__ == "__main__":
    main()
