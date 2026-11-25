#!/usr/bin/env python3
"""Convergence of the AIM on all three geometries, and what stopping costs.

Top row: the AIM residual max|U| against iteration, with the stopping
thresholds drawn across it and a marker where each one fires.

Bottom row: what that stopping threshold actually buys.  The prestressed
configuration is handed to the mechanobiologically equilibrated mixture and
released; whatever it moves is spurious, because an equilibrated mixture
handed its own homeostatic state should not move at all.  Plotted against the
AIM residual it was started from, with the identity line for reference: a
point on the line means the AIM residual is an honest estimate of the error,
a point above it means the AIM residual understates the error.

The heart is purely elastic -- no constrained mixture, no fibre turnover --
so there is no separate equilibration step to run.  Reloading the same
reference configuration reproduces the same deformation, and the AIM residual
*is* the spurious displacement, exactly.

    python scripts/plot_convergence.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CASES = ("cylinder", "aorta", "heart")
TITLES = {"cylinder": "idealised cylinder",
          "aorta": "VMR 0012_H_AO_H aorta  (97 mmHg)",
          "heart": "Pfaller19 four-chamber heart"}
# Gebauer et al. (2023), Sec. 3.1: their own prestress ladder on a prolate
# spheroid LV, for comparison -- (threshold, iterations, displacement after
# G&R) in mm.  Their ratio is a constant 3-4x over three decades.
GEBAUER23_LADDER = [(1e-1, 136, 4e-1), (1e-2, 393, 3e-2), (1e-3, 745, 3e-3)]
COLOR = {"cylinder": "#1f77b4", "aorta": "#d62728", "heart": "#2ca02c"}


def _find(root: Path, *candidates: str) -> Path:
    """First candidate that exists under root.

    --root takes either a local ``output/`` tree or an unpacked release
    archive, which flattens the same files under ``ladders/`` and ``aim/``.
    """
    for c in candidates:
        p = root / c
        if p.is_file():
            return p
    raise FileNotFoundError(str(root / candidates[0]))


def load(case: str, root: Path) -> dict:
    """Residual history and, where it exists, the equilibrated ladder."""
    if case == "cylinder":
        d = json.loads(_find(root, "thresholds/summary.json",
                             "ladders/cylinder_thresholds.json").read_text())
        # the nodal variant, to match the aorta run
        v = d.get("nodal", next(iter(d.values())))
        return dict(residual=np.asarray(v["residual"], float),
                    stops=[(s["iteration"], s["residual"], s["equilibrated"])
                           for s in v["stops"]],
                    elastic=False)
    if case == "aorta":
        res = np.load(_find(root, "p97/aim_nodal.npz",
                            "aim/aorta_aim.npz"))["residual"]
        rows = json.loads(_find(root, "aorta_equil/summary.json",
                                "ladders/aorta_equilibrated.json").read_text())
        return dict(residual=np.asarray(res, float),
                    stops=[(r["aim_iterations"], r["aim_residual"],
                            r["equilibrated_max"]) for r in rows],
                    elastic=False)
    if case == "heart":
        h = json.loads(_find(root, "heart/summary_heart.json",
                             "ladders/heart_summary.json").read_text())
        # The elastic heart has no equilibration step.  The constrained
        # mixture of Gebauer et al. (2023) does, and when that ladder has been
        # run it is plotted like the other two.
        try:
            rows = json.loads(_find(root, "heart_equil/summary.json",
                                    "ladders/heart_equilibrated.json").read_text())
        except FileNotFoundError:
            return dict(residual=np.asarray(h["residual"], float),
                        stops=[], elastic=True)
        return dict(residual=np.asarray(h["residual"], float),
                    stops=[(r["aim_iterations"], r["aim_residual"],
                            r["equilibrated_max"]) for r in rows],
                    elastic=False)
    raise SystemExit(case)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("output"))
    ap.add_argument("--out", type=Path,
                    default=Path("output/fig_convergence.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {}
    for c in CASES:
        try:
            data[c] = load(c, args.root)
        except FileNotFoundError as e:
            print(f"  {c}: skipped, missing {e.filename}")
    if not data:
        raise SystemExit("no results found under " + str(args.root))

    fig, axes = plt.subplots(2, len(data), figsize=(4.6 * len(data), 7.2),
                             squeeze=False)

    for col, (case, d) in enumerate(data.items()):
        res, stops, c = d["residual"], d["stops"], COLOR[case]

        # --- top: residual against iteration ------------------------------
        ax = axes[0][col]
        it = np.arange(1, len(res) + 1)
        ax.semilogy(it, res, "-o", color=c, ms=4, lw=1.6, zorder=3)
        for k, (i_stop, r_stop, _) in enumerate(stops):
            ax.axhline(r_stop, color="0.75", lw=0.7, ls=":", zorder=1)
            ax.plot(i_stop, r_stop, "s", color="k", ms=6, mfc="white",
                    mew=1.3, zorder=4,
                    label="stopping threshold" if k == 0 else None)
        ax.set_xlabel("AIM iteration")
        if col == 0:
            ax.set_ylabel(r"residual  $\max|U|$  [mm]")
        ax.set_title(TITLES[case], fontsize=10)
        ax.grid(alpha=0.25, lw=0.5, which="both")
        if stops:
            ax.legend(fontsize=8, loc="upper right")

        # --- bottom: what the stop costs ----------------------------------
        ax = axes[1][col]
        if d["elastic"]:
            lo, hi = res[-1] * 0.5, res[0] * 2
            xs = np.array([lo, hi])
            ax.loglog(xs, xs, "-", color="0.4", lw=1.2, label="identity")
            ax.text(0.5, 0.42,
                    "purely elastic:\nno constrained mixture,\nso no separate "
                    "equilibration.\nThe AIM residual *is*\nthe spurious "
                    "displacement.",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="0.25")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        else:
            r_aim = np.array([s[1] for s in stops])
            r_eq = np.array([s[2] for s in stops])
            lo = min(r_aim.min(), r_eq.min()) * 0.4
            hi = max(r_aim.max(), r_eq.max()) * 2.5
            xs = np.array([lo, hi])
            ax.loglog(xs, xs, "-", color="0.4", lw=1.2, label="identity")
            ax.loglog(r_aim, r_eq, "-o", color=c, ms=6, lw=1.6,
                      label="equilibrated")
            floor = r_eq.min()
            if r_eq.max() / floor > 3:          # a floor worth naming
                ax.axhline(floor, color=c, lw=0.9, ls="--", alpha=0.7)
                ax.text(hi, floor * 1.15, f"floor {floor:.2e} mm",
                        ha="right", va="bottom", fontsize=8, color=c)
            if case == "heart":
                g = np.array(GEBAUER23_LADDER, float)
                ax.loglog(g[:, 0], g[:, 2], "--^", color="0.45", ms=6,
                          lw=1.3, label="Gebauer23 (their LV)")
                lo = min(lo, g[:, 2].min() * 0.4)
                ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.legend(fontsize=8, loc="upper left")
        ax.set_xlabel(r"AIM residual at stop  [mm]")
        if col == 0:
            ax.set_ylabel("spurious displacement\nafter equilibration  [mm]")
        ax.grid(alpha=0.25, lw=0.5, which="both")
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("AIM convergence and the cost of stopping early", fontsize=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    fig.savefig(args.out.with_suffix(".pdf"))
    print(f"wrote {args.out}  (+ .pdf)")
    for case, d in data.items():
        print(f"  {case:9s} {len(d['residual']):3d} iterations, "
              f"final {d['residual'][-1]:.3e} mm, "
              f"{len(d['stops'])} equilibrated stops")


if __name__ == "__main__":
    main()
