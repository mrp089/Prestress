#!/usr/bin/env python3
"""Fig. 2 of the manuscript, extended: what does stopping the AIM early cost?

The manuscript's Fig. 2 plots the AIM residual against the iteration count and
stops there.  Gebauer et al. (BMMB 2023) instead run their prestressing to a
*range* of tolerances and report what each one leaves behind.  This script does
the same for the AIM.

The cost of stopping early is measured without any growth & remodelling time
stepping.  At mechanobiological equilibrium every turning-over constituent is
back at its deposition stretch by definition, so the fibre family is a constant
prestress and the equilibrated state follows from one nonlinear elastic solve
(see ``prestress.growth.equilibrated_residual``).

    python scripts/run_thresholds.py [--modes effective nodal embedded]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt

from prestress import PRESETS, CylinderGeometry, solve_cylinder_0d
from prestress.aim import make_solver, run_aim
from prestress.growth import equilibrated_residual
from prestress.mesh import build_cylinder_mesh
from prestress.plotting import use_paper_style

MODE_LABEL = {
    "effective": "effective pressure (Eq. 9)",
    "nodal": "fibre nodal forces",
    "embedded": "embedded fibres",
}
MODE_COLOR = {"effective": "#1f77b4", "nodal": "#2ca02c", "embedded": "#d62728"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="implied")
    ap.add_argument("--modes", nargs="+",
                    default=["effective", "nodal", "embedded"])
    ap.add_argument("--outdir", default="output/thresholds")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()

    use_paper_style()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        summary = json.loads((outdir / "summary.json").read_text())
        _report(summary)
        _figure(summary, outdir)
        print(f"\nredrew figures in {outdir}")
        return

    workroot = Path(args.workdir) if args.workdir else outdir / "febio"
    geo = CylinderGeometry()
    mesh = build_cylinder_mesh(geo)
    mat = PRESETS[args.preset]
    ana = solve_cylinder_0d(mat, geo, radius_convention="mid")
    thresholds = [10.0 ** -e for e in range(1, 12)]

    summary = {}
    for mode in args.modes:
        pressure = ana.p_eff if mode == "effective" else mat.pressure
        print("=" * 78)
        print(f"AIM, {MODE_LABEL[mode]}")
        print("=" * 78)
        solve = make_solver(mesh, mat, pressure, workroot / f"aim_{mode}",
                            mode=mode)
        aim = run_aim(mesh, solve, max_iter=args.max_iter)

        stops = []
        for thr in thresholds:
            try:
                k, X, r = aim.config_at_residual(thr)
            except ValueError:
                continue
            if stops and stops[-1]["iteration"] == k + 1:
                continue
            d, _ = equilibrated_residual(mesh, mat, X,
                                         workroot / f"eq_{mode}_{k + 1:03d}")
            print(f"  threshold {thr:8.0e} -> iteration {k + 1:3d}, "
                  f"residual {r:.3e}, equilibrated {d:.3e} mm")
            stops.append(dict(threshold=thr, iteration=k + 1, residual=r,
                              equilibrated=d))
        summary[mode] = dict(residual=list(map(float, aim.residual)),
                             iterations=aim.iterations, stops=stops)

    order = [m for m in ("effective", "nodal", "embedded") if m in summary]
    summary = {m: summary[m] for m in order}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    _report(summary)
    _figure(summary, outdir)
    print(f"\nwrote results and figures to {outdir}")


def _report(summary) -> None:
    print("\n" + "=" * 78)
    print("Cost of stopping the AIM at a given threshold")
    print("=" * 78)
    for mode, d in summary.items():
        print(f"\n{MODE_LABEL[mode]}  ({d['iterations']} iterations in total)")
        print(f"  {'threshold':>10s} {'iteration':>10s} {'residual [mm]':>15s}"
              f" {'equilibrated [mm]':>19s}")
        for s in d["stops"]:
            print(f"  {s['threshold']:10.0e} {s['iteration']:10d}"
                  f" {s['residual']:15.3e} {s['equilibrated']:19.3e}")


def _figure(summary, outdir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # (A) the manuscript's Fig. 2, with the thresholds drawn in
    for mode, d in summary.items():
        r = np.array(d["residual"])
        ax1.semilogy(np.arange(1, len(r) + 1), r, "-", lw=1.2,
                     color=MODE_COLOR[mode], label=MODE_LABEL[mode])
        k = np.array([s["iteration"] for s in d["stops"]])
        v = np.array([s["residual"] for s in d["stops"]])
        ax1.semilogy(k, v, "o", ms=3.5, color=MODE_COLOR[mode])
    for s in summary[next(iter(summary))]["stops"]:
        ax1.axhline(s["threshold"], color="#cccccc", lw=0.5, zorder=0)
    ax1.set_xlabel("iteration $k$")
    ax1.set_ylabel(r"AIM residual  max$\,|\mathbf{U}_k|$  [mm]")
    ax1.set_title("A  convergence and stopping thresholds", fontsize=8, loc="left")
    ax1.legend(fontsize=6)

    # (B) what stopping there actually costs
    for mode, d in summary.items():
        r = np.array([s["residual"] for s in d["stops"]])
        e = np.array([s["equilibrated"] for s in d["stops"]])
        ax2.loglog(r, e, "o-", ms=3.5, color=MODE_COLOR[mode],
                   label=MODE_LABEL[mode])
    lim = np.array(ax2.get_xlim())
    ax2.loglog(lim, lim, ":", color="#999999", lw=0.8, label="1:1")
    ax2.set_xlim(*lim)
    ax2.set_xlabel(r"AIM residual at which it was stopped  [mm]")
    ax2.set_ylabel("equilibrated residual\ndisplacement  [mm]")
    ax2.set_title("B  cost of stopping early", fontsize=8, loc="left")
    ax2.legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(outdir / "fig2_thresholds.pdf")
    fig.savefig(outdir / "fig2_thresholds.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
