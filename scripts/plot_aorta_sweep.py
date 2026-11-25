#!/usr/bin/env python3
"""Plot how the AIM's convergence on the aorta depends on the load.

Reads the per-pressure logs produced by

    scripts/run_aorta.py --variants nodal --pressure-mmhg P --outdir output/pP

and draws the residual history of each, which is the figure that says what the
method can and cannot do on a patient-specific geometry.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RE_AIM = re.compile(r"AIM\s+(\d+):\s+max\|U\| = ([0-9.eE+-]+) mm")


def history(path: Path) -> np.ndarray:
    return np.array([float(m.group(2))
                     for m in RE_AIM.finditer(path.read_text())])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--out", default="output/aorta/fig5_load_sweep")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--p-ref", type=float, default=130.0)
    args = ap.parse_args()

    runs = []
    for lg in args.logs:
        m = re.search(r"aim_p(\d+)", lg.name)
        if not m:
            continue
        h = history(lg)
        if len(h):
            runs.append((float(m.group(1)), h))
    runs.sort()
    if not runs:
        sys.exit("no AIM histories found in the given logs")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    cmap = plt.get_cmap("viridis")
    for p, h in runs:
        c = cmap(0.08 + 0.84 * (p / args.p_ref))
        ax.semilogy(np.arange(1, len(h) + 1), h, "o-", ms=3.5, color=c,
                    label=f"{p:.0f} mmHg  ({100 * p / args.p_ref:.0f} %)")
    ax.axhline(args.tol, color="0.4", lw=0.9, ls="--")
    ax.text(0.98, args.tol * 1.3, f"tol = {args.tol:g} mm", ha="right",
            va="bottom", transform=ax.get_yaxis_transform(), fontsize=8,
            color="0.4")
    ax.set_xlabel("AIM iteration")
    ax.set_ylabel(r"$\max|\mathbf{x}(\mathbf{X}) - \mathbf{x}_\mathrm{target}|$  [mm]")
    ax.set_title("Residual history")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, which="both", lw=0.5)

    # contraction factor: geometric mean of the last few ratios
    ps = [p for p, _ in runs]
    rate = []
    for _, h in runs:
        tail = h[max(1, len(h) - 6):]
        r = tail[1:] / tail[:-1]
        r = r[np.isfinite(r) & (r > 0)]
        rate.append(float(np.exp(np.mean(np.log(r)))) if len(r) else np.nan)
    bx.plot(ps, rate, "o-", color="#1f77b4")
    bx.axhline(1.0, color="crimson", lw=1.0, ls="--")
    bx.text(min(ps), 1.0, " no longer a contraction", color="crimson",
            fontsize=8, va="bottom")
    bx.set_xlabel("in vivo pressure [mmHg]")
    bx.set_ylabel("contraction factor per iteration")
    bx.set_title("Why it stops working")
    bx.grid(alpha=0.25, lw=0.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = Path(f"{args.out}.{ext}")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=170)
    print(f"wrote {args.out}.png / .pdf")
    for (p, h), r in zip(runs, rate):
        print(f"  {p:6.0f} mmHg  {len(h):3d} iterations  "
              f"final {h[-1]:.3e} mm  contraction {r:.3f}")


if __name__ == "__main__":
    main()
