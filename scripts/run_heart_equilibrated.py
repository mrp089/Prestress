#!/usr/bin/env python3
"""The four-chamber heart as a constrained mixture, and what stopping costs.

The heart test case elsewhere in this repository is purely elastic (the
passive myocardium of Pfaller et al. 2019), which makes the AIM residual and
the spurious displacement the same number -- reloading the same reference
configuration reproduces the same deformation, so there is nothing to
equilibrate.  To put the heart on the same footing as the cylinder and the
aorta it needs a *mixture*, and this script gives it the myocardium of

    Gebauer, Pfaller, Braeu, Cyron & Wall, "A homogenized constrained mixture
    model of cardiac growth and remodeling: analyzing mechanobiological
    stability and reversal", Biomech Model Mechanobiol 22:1983-2002 (2023),

whose Table 1 is transcribed in ``prestress.params.GEBAUER23``.  Three
constituents: elastin (isotropic, the only one FEBio sees as a material),
cardiomyocytes along the fibre direction, and four collagen families in the
myocyte plane at 0, 90 and +/-45 degrees.

Two deliberate departures from Gebauer et al., both because their parameters
were calibrated on a *systolic* prolate-spheroid left ventricle and are used
here on an *end-diastolic* four-chamber mesh:

  * the active muscle tone of their Eq. (8) is off, as it is in Pfaller
    et al. (2019) during prestressing.  ``--active`` turns it on;
  * the load is end-diastolic (LV/RV 10, LA 10, RA 5 mmHg), not their
    120 mmHg systolic pressure.

So this is their constitutive model on our geometry and loading, not a
reproduction of their simulation.  Their own ladder, for reference (their
Sec. 3.1): eps_pre = 0.1 / 0.01 / 0.001 mm reached in 136 / 393 / 745
iterations, leaving 0.4 / 0.03 / 0.003 mm after G&R.

    python scripts/run_heart_equilibrated.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prestress.aim import run_aim
from prestress.febio import read_node_log, run_febio, write_elastin_feb
from prestress.growth import equilibrated_forces_residual
from prestress.heart import CHAMBERS, cmm_fiber_families, heart_model, load_heart
from prestress.kinematics import fiber_stress_tensors, nodal_forces_from_stress
from prestress.params import (GEBAUER23, GEBAUER23_RHO0, PRESETS,
                              gebauer23_active_stress_kPa)
from prestress.vtk import write_vtk


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", default="eva.vtu")
    ap.add_argument("--outdir", default="output/heart_equil")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3])
    ap.add_argument("--active", action="store_true",
                    help="add the active muscle tone of Gebauer Eq. (8); off "
                         "by default, since this is an end-diastolic state")
    ap.add_argument("--max-iter", type=int, default=40)
    # The equilibration is a fixed point in its own right and its
    # contraction rate is a property of the mesh and material, not of
    # the threshold it starts from.  On the aorta it is 0.19 per
    # iteration and 10 steps reach 2.4e-7 mm.  The heart mixture
    # contracts at 0.42, so the library default of 12 stops it at
    # ~1e-4 mm -- which would show up in the ladder as a floor that
    # is nothing but a truncated fixed point.  Give it room.
    ap.add_argument("--equil-max-iter", type=int, default=40)
    ap.add_argument("--equil-tol", type=float, default=1e-6)
    ap.add_argument("--n-steps", type=int, default=10)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--etol", type=float, default=1e-8)
    ap.add_argument("--min-residual", type=float, default=1e-10)
    ap.add_argument("--max-ups", type=int, default=10)
    ap.add_argument("--opt-iter", type=int, default=30)
    ap.add_argument("--beta-max", type=float, default=8.0)
    ap.add_argument("--anderson", type=int, default=5)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    h = load_heart(args.mesh, surfaces={**CHAMBERS, "vessels": "dsurf20",
                                        "epi": "dsurf8", "base": "dsurf19"})
    model, pressures = heart_model(h)
    mat = PRESETS["heart_cmm"]

    fibers, labels = cmm_fiber_families(h)
    rho = GEBAUER23_RHO0
    sigmas = [GEBAUER23["myocytes"].sigma_h_kPa(rho)]
    if args.active:
        sigmas[0] += gebauer23_active_stress_kPa(GEBAUER23["myocytes"].lam_h)
    sigmas += [GEBAUER23["collagen"].sigma_h_kPa(rho)] * 4

    print(f"nodes {model.n_node}  elements {model.n_elem}")
    print(f"elastin  c1 = {mat.phi_e * mat.c_e:.3f} kPa   k = {mat.K:.1f} kPa")
    for lab, sg in zip(labels, sigmas):
        print(f"  {lab:18s} sigma_h = {sg:7.4f} kPa")
    print(f"  total fibre stress  {sum(sigmas):7.4f} kPa"
          f"{'  (active tone on)' if args.active else ''}")

    feb_kw = dict(rtol=args.rtol, etol=args.etol,
                  min_residual=args.min_residual, max_ups=args.max_ups,
                  opt_iter=args.opt_iter, auto_step=True, laugon=False)

    # the fibre contribution as dead nodal forces on the imaged mesh
    S = fiber_stress_tensors(model.elements, model.elem_type,
                             list(zip(sigmas, fibers)))
    forces = nodal_forces_from_stress(h.nodes, model.elements,
                                      model.elem_type, S)

    rows = []
    for thr in sorted(args.thresholds, reverse=True):
        tag = f"thr_{thr:.0e}".replace("-", "m").replace("+", "")
        print("=" * 72)
        print(f"AIM stopped at {thr:.1e} mm")
        print("=" * 72)
        work = outdir / "febio" / tag
        work.mkdir(parents=True, exist_ok=True)

        def solve(X: np.ndarray, k: int, _w=work) -> np.ndarray:
            feb = write_elastin_feb(_w / f"aim_{k:03d}.feb", model, mat,
                                    mat.pressure, coords=X,
                                    nodal_forces=forces,
                                    n_steps=args.n_steps, **feb_kw)
            run_febio(feb)
            return read_node_log(_w / f"aim_{k:03d}_pos.txt", model.n_node)

        aim = run_aim(model, solve, max_iter=args.max_iter, tol=thr,
                      beta_max=args.beta_max, anderson=args.anderson)
        if not aim.converged:
            print(f"!! AIM did not reach {thr:.1e} mm "
                  f"(stopped at {aim.residual[-1]:.3e})")
        r_aim = float(aim.residual[-1])

        print(f"  equilibrating from the {r_aim:.3e} mm configuration")
        r_eq, x_eq, hist = equilibrated_forces_residual(
            model, mat, aim.X, h.nodes, fibers,
            outdir / "febio" / f"{tag}_equil", sigma=sigmas,
            n_steps=args.n_steps, max_iter=args.equil_max_iter,
            tol=args.equil_tol, **feb_kw)

        U = np.linalg.norm(x_eq - h.nodes, axis=1)
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
        write_vtk(outdir / f"equilibrated_{tag}.vtk", h.nodes, model.elements,
                  point_data={"spurious_displacement": x_eq - h.nodes,
                              "magnitude_mm": U},
                  title=f"equilibrated heart, AIM stopped at {thr:.0e} mm")
        np.savez_compressed(outdir / f"aim_{tag}.npz", X=aim.X,
                            residual=np.asarray(aim.residual))

    (outdir / "summary.json").write_text(json.dumps(rows, indent=2))
    print("\n AIM residual [mm]   equilibrated max [mm]   ratio")
    for r in rows:
        print(f"   {r['aim_residual']:.3e}          {r['equilibrated_max']:.3e}"
              f"        {r['equilibrated_max'] / r['aim_residual']:.2f}")
    print(f"\nwrote results to {outdir}")


if __name__ == "__main__":
    main()
