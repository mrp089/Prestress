#!/usr/bin/env python3
"""Prestress the four-chamber heart of Pfaller et al. (2019) with the AIM.

The dissertation prestressed this geometry with BACI's MULF algorithm
(``PRESTRESS MULF``), a different family from the fixed-point iteration the
AIM belongs to.  Running the AIM on the same mesh with the same loads and
boundary conditions gives a direct comparison on a problem where an accepted
answer already exists.

The active stress is off while prestressing, so this is a purely elastic
problem: no constrained mixture, no fibre nodal forces, just the AIM.  That
makes it the cleanest test of the algorithm itself in this repository -- the
cylinder is idealised and the aorta mixes the algorithm with the question of
how to account for the embedded fibres.

    python scripts/run_heart.py [--max-iter 40] [--la 10] [--ra 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prestress import PRESETS
from prestress.aim import run_aim
from prestress.febio import read_node_log, run_febio, write_elastin_feb
from prestress.heart import CHAMBERS, ED_PRESSURE_MMHG, heart_model, load_heart
from prestress.vtk import write_vtk

SRC = "eva.vtu"
BC_SURFACES = {"vessels": "dsurf20", "epi": "dsurf8", "base": "dsurf19"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--outdir", default="output/heart")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--n-steps", type=int, default=10)
    # Ventricular end-diastolic pressures are the ones Pfaller et al. used.
    # The atrial ones are a modelling choice: physiological mean atrial
    # pressures are 8-12 mmHg on the left and 2-6 mmHg on the right.
    ap.add_argument("--lv", type=float, default=ED_PRESSURE_MMHG["LV"])
    ap.add_argument("--rv", type=float, default=ED_PRESSURE_MMHG["RV"])
    ap.add_argument("--la", type=float, default=ED_PRESSURE_MMHG["LA"])
    ap.add_argument("--ra", type=float, default=ED_PRESSURE_MMHG["RA"])
    ap.add_argument("--k-vessels", type=float, default=1e3)
    ap.add_argument("--k-epi", type=float, default=2.0)
    # Same reasoning as the aorta: see scripts/run_aorta.py.  This mesh is
    # 846,864 dof, so the settings matter even more here.
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--etol", type=float, default=1e-8)
    ap.add_argument("--min-residual", type=float, default=1e-10)
    ap.add_argument("--max-ups", type=int, default=10)
    ap.add_argument("--opt-iter", type=int, default=30)
    ap.add_argument("--laugon", action="store_true")
    ap.add_argument("--augtol", type=float, default=1e-4)
    ap.add_argument("--beta-max", type=float, default=8.0)
    ap.add_argument("--anderson", type=int, default=5)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir or outdir / "febio")

    heart = load_heart(args.src, surfaces={**CHAMBERS, **BC_SURFACES})
    pmm = {"LV": args.lv, "RV": args.rv, "LA": args.la, "RA": args.ra}
    model, p_lv = heart_model(heart, pmm, args.k_vessels, args.k_epi)
    mat = PRESETS["heart"]

    print(f"heart: {model.n_elem} tet10, {model.n_node} nodes "
          f"({3 * model.n_node} dof)")
    print(f"  LV {args.lv:g}  RV {args.rv:g}  LA {args.la:g}  RA {args.ra:g} mmHg"
          f"   ->  LV {p_lv:.4f} kPa")
    print(f"  material  c1 {mat.phi_e * mat.c_e:g}  c2 {mat.mr_c2:g}  "
          f"k {mat.K:g} kPa")
    for name, facets in model.surfaces.items():
        print(f"  surface {name:9s} {len(facets):6d} tri6")

    workdir.mkdir(parents=True, exist_ok=True)

    def solve(X: np.ndarray, k: int) -> np.ndarray:
        stem = f"aim_{k:03d}"
        feb = write_elastin_feb(
            workdir / f"{stem}.feb", model, mat, p_lv, coords=X,
            n_steps=args.n_steps, auto_step=True, rtol=args.rtol,
            etol=args.etol, min_residual=args.min_residual,
            max_ups=args.max_ups, opt_iter=args.opt_iter,
            laugon=args.laugon, augtol=args.augtol)
        run_febio(feb)
        return read_node_log(workdir / f"{stem}_pos.txt", model.n_node)

    aim = run_aim(model, solve, max_iter=args.max_iter,
                  beta_max=args.beta_max, tol=args.tol,
                  anderson=args.anderson)

    U = np.linalg.norm(aim.displacements[-1], axis=1)
    summary = dict(
        iterations=aim.iterations, converged=bool(aim.converged),
        residual=list(map(float, aim.residual)),
        pressures_mmHg=pmm,
        residual_percentiles={f"p{q}": float(np.percentile(U, q))
                              for q in (50, 90, 99, 99.9, 100)},
    )
    print("\nfinal residual |x(X) - x_target| over the mesh:")
    for q in (50, 90, 99, 99.9, 100):
        print(f"   p{q:<5} {np.percentile(U, q):.4e} mm")

    np.savez_compressed(outdir / "aim_heart.npz", X=aim.X,
                        residual=np.array(aim.residual))
    write_vtk(outdir / "heart_unloaded.vtk", aim.X, model.elements,
              point_data={"in_vivo_displacement": heart.nodes - aim.X,
                          "aim_residual_mm": U},
              title="AIM unloaded heart configuration")
    (outdir / "summary_heart.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote results to {outdir}")


if __name__ == "__main__":
    main()
