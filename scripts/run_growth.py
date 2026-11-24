#!/usr/bin/env python3
"""How accurate does the prestress have to be?

The AIM is stopped at a range of residual norms; each of the resulting
prestressed configurations is then loaded to the in vivo pressure and allowed
to grow and remodel with the loading held constant.  Any motion after loading
is spurious, and its size is what this script measures as a function of the
prestress residual (cf. Gebauer et al., BMMB 2023).

Two ways of accounting for the embedded fibres during prestressing are
compared:

  effective   the manuscript's black-box method, in which the prescribed fibre
              stress is replaced by an effective pressure on elastin;
  embedded    the fibre family is carried by the material, so the AIM solves
              the exact mixture equilibrium.

    python scripts/run_growth.py [--preset implied] [--outdir output/growth]
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
from prestress.growth import run_growth
from prestress.mesh import build_cylinder_mesh
from prestress.plotting import use_paper_style

MODE_LABEL = {
    "effective": "effective pressure (black box, Eq. 9)",
    "nodal": "fibre nodal forces (black box, exact)",
    "embedded": "embedded fibres (exact mixture)",
}
MODE_COLOR = {"effective": "#1f77b4", "nodal": "#2ca02c", "embedded": "#d62728"}


def pick_levels(aim, levels):
    """Pick AIM iterates whose residual first drops below each level."""
    out = []
    for lv in levels:
        try:
            k, X, r = aim.config_at_residual(lv)
        except ValueError:
            continue
        if out and out[-1][0] == k:
            continue
        out.append((k, X, r))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="implied")
    ap.add_argument("--bulk-ratio", type=float, default=100.0,
                    help="K / c_e; 1000 (Table 1) is a pure numerical penalty "
                         "and conditions the G&R solve badly, see docs/corrections.md")
    ap.add_argument("--outdir", default="output/growth")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--modes", nargs="+",
                    default=["effective", "nodal", "embedded"])
    ap.add_argument("--k-d", type=float, default=1.0)
    ap.add_argument("--K-sigma", type=float, default=1.0)
    ap.add_argument("--t-end", type=float, default=4.0)
    ap.add_argument("--gr-step", type=float, default=0.05)
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="wall-clock budget per G&R run; a run that loses "
                         "stability is kept up to the point it reached")
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw the figures from a previous run's summary.json")
    ap.add_argument("--append", action="store_true",
                    help="keep the modes already in summary.json and add these")
    args = ap.parse_args()

    use_paper_style()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    workroot = Path(args.workdir) if args.workdir else outdir / "febio"

    if args.figures_only:
        summary = json.loads((outdir / "summary.json").read_text())
        _report(summary)
        _figures(summary, outdir)
        print(f"\nredrew figures in {outdir}")
        return

    geo = CylinderGeometry()
    mesh = build_cylinder_mesh(geo)
    mat = PRESETS[args.preset].with_(bulk_ratio=args.bulk_ratio)
    ana = solve_cylinder_0d(mat, geo, radius_convention="mid")
    levels = [10.0 ** -e for e in range(2, 12)]

    summary = {}
    if args.append and (outdir / "summary.json").is_file():
        summary = json.loads((outdir / "summary.json").read_text())
        print(f"appending to existing modes: {', '.join(summary)}\n")
    for mode in args.modes:
        pressure = ana.p_eff if mode == "effective" else mat.pressure
        print("=" * 78)
        print(f"AIM, {MODE_LABEL[mode]}")
        print("=" * 78)
        # The G&R stage cannot use the augmented Lagrangian (it enforces
        # J -> 1, the growth constraint is J -> J_g), so the prestressing must
        # not either: otherwise the two stages treat incompressibility
        # differently and that mismatch, ~1e-4 mm, would masquerade as a
        # prestress error.  scripts/run_thresholds.py has no such constraint
        # and does use it.
        solve = make_solver(mesh, mat, pressure, workroot / f"aim_{mode}",
                            mode=mode, laugon=False)
        aim = run_aim(mesh, solve, max_iter=args.max_iter)

        runs = []
        for k, X, r in pick_levels(aim, levels):
            print(f"\n  G&R from AIM iterate {k + 1} (residual {r:.3e} mm)")
            g = run_growth(
                mesh, mat, X, workroot / f"gr_{mode}_{k + 1:03d}",
                prestress_residual=r, k_d=args.k_d, K_sigma=args.K_sigma,
                t_end=args.t_end, gr_step=args.gr_step, timeout=args.timeout,
            )
            t_mid = 0.5 * (args.t_end + 1.0)
            print(f"    after loading      : {g.drift_loaded:.4e} mm")
            print(f"    after G&R t={t_mid:<5.2f}: {g.drift_at(t_mid):.4e} mm")
            print(f"    at t={g.times[-1]:<5.2f}       : {g.drift_final:.4e} mm"
                  f"{'' if g.completed else '   (solver stopped early)'}")
            runs.append(
                dict(iteration=k + 1, residual=r, completed=g.completed,
                     drift_loaded=g.drift_loaded, drift_mid=g.drift_at(t_mid),
                     drift_final=g.drift_final, t_final=float(g.times[-1]),
                     times=g.times.tolist(), drift=g.drift.tolist(),
                     r_inner=g.r_inner.tolist(), thickness=g.thickness.tolist(),
                     length=g.length.tolist())
            )
        summary[mode] = dict(
            aim_residual=list(map(float, aim.residual)),
            aim_iterations=aim.iterations,
            runs=runs,
        )
        np.savez_compressed(outdir / f"aim_{mode}.npz",
                            X=aim.X, residual=np.array(aim.residual))

    order = [m for m in ("effective", "nodal", "embedded") if m in summary]
    summary = {m: summary[m] for m in order}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    _figures(summary, outdir)
    _report(summary)
    print(f"\nwrote results and figures to {outdir}")


def _report(summary) -> None:
    print("\n" + "=" * 78)
    print("Spurious displacement after growth & remodelling")
    print("=" * 78)
    for mode, d in summary.items():
        print(f"\n{MODE_LABEL[mode]}")
        print(f"  {'AIM it.':>8s} {'residual [mm]':>15s} {'after load [mm]':>17s}"
              f" {'mid G&R [mm]':>14s} {'end G&R [mm]':>14s} {'t_end':>7s}")
        for r in d["runs"]:
            print(f"  {r['iteration']:8d} {r['residual']:15.3e}"
                  f" {r['drift_loaded']:17.3e} {r['drift_mid']:14.3e}"
                  f" {r['drift_final']:14.3e} {r['t_final']:7.2f}"
                  f"{'' if r['completed'] else '  *'}")


def _figures(summary, outdir: Path) -> None:
    # spurious displacement vs prestress residual
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    for mode, d in summary.items():
        r = np.array([q["residual"] for q in d["runs"]])
        dl = np.array([q["drift_loaded"] for q in d["runs"]])
        dm = np.array([q["drift_mid"] for q in d["runs"]])
        c = MODE_COLOR[mode]
        ax.loglog(r, dl, "o--", ms=3, color=c, alpha=0.55,
                  label=f"{mode}: after loading")
        ax.loglog(r, dm, "o-", ms=3, color=c, label=f"{mode}: after G&R")
    lim = np.array(ax.get_xlim())
    ax.loglog(lim, lim, ":", color="#999999", lw=0.8, label="1:1")
    ax.set_xlabel(r"AIM residual  max$|\mathbf{U}_k|$  [mm]")
    ax.set_ylabel(r"spurious displacement  [mm]")
    ax.set_title("Prestress accuracy and spurious G&R")
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=2)
    fig.savefig(outdir / "fig6_drift_vs_residual.pdf")
    fig.savefig(outdir / "fig6_drift_vs_residual.png")
    plt.close(fig)

    # time traces
    n = len(summary)
    fig, axes = plt.subplots(1, n, figsize=(3.9 * n, 2.8), squeeze=False)
    for ax, (mode, d) in zip(axes[0], summary.items()):
        cmap = plt.get_cmap("viridis")
        runs = d["runs"]
        for i, q in enumerate(runs):
            t = np.array(q["times"])
            ax.semilogy(t, np.maximum(np.array(q["drift"]), 1e-16),
                        color=cmap(i / max(1, len(runs) - 1)), lw=1.0,
                        label=f"{q['residual']:.0e}")
        ax.axvline(1.0, color="#999999", ls=":", lw=0.8)
        ax.set_xlabel("G&R time  [1/$k_d$]")
        ax.set_ylabel(r"$\max|\mathbf{x}(t)-\mathbf{x}_{ivo}|$  [mm]")
        ax.set_title(MODE_LABEL[mode], fontsize=7)
        ax.legend(title="AIM residual [mm]", fontsize=5, ncol=1,
                  loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(outdir / "fig7_drift_history.pdf")
    fig.savefig(outdir / "fig7_drift_history.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
