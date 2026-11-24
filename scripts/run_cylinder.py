#!/usr/bin/env python3
"""Cylinder example: 0D analytical solution and 3D AIM in FEBio.

Reproduces Sec. 2.4 / Sec. 3 of the manuscript for the idealised mouse
descending thoracic aorta and verifies the finite element result against the
thin-walled analytical solution.

    python scripts/run_cylinder.py [--preset paper implied] [--outdir output]
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
from prestress.kinematics import (deformation_gradient, gauss_positions,
                                  principal_stretches)
from prestress.mesh import build_cylinder_mesh
from prestress.plotting import COLORS, use_paper_style
from prestress.vtk import write_vtk

# values reported in the manuscript, for reference
PAPER = {
    "analytic": {"R_i": 0.29, "H": 0.042, "L": 2.10},
    "aim": {"R_i": 0.27, "H": 0.044, "L": 2.14},
    "lam_median": (1.172, 0.937, 0.912),
    "lam_range": ((1.159, 1.185), (0.933, 0.939), (0.911, 0.913)),
    "iterations": 26,
    "residual_first": 0.16,
    "residual_floor": 3.7e-9,
}


def summarise(name, geo, mat, aim, analytic):
    F = deformation_gradient(mesh_global, aim.X, geo_nodes_global)
    lam, _ = principal_stretches(F)
    lam = lam.reshape(-1, 3)
    pos = gauss_positions(mesh_global, geo_nodes_global).reshape(-1, 3)
    gauss_rad = np.linalg.norm(pos[:, :2], axis=1)   # in vivo radius of each Gauss point
    rad = np.linalg.norm(aim.X[:, :2], axis=1)       # nodal radii of the unloaded mesh
    out = {
        "preset": name,
        "c_e": mat.c_e,
        "lam_dep": mat.lam_dep,
        "iterations": aim.iterations,
        "residual_first": aim.residual[0],
        "residual_final": aim.final_residual,
        "R_i": float(rad.min()),
        "R_o": float(rad.max()),
        "H": float(rad.max() - rad.min()),
        "L": float(aim.X[:, 2].max() - aim.X[:, 2].min()),
        "lam_median": [float(v) for v in np.median(lam, axis=0)],
        "lam_min": [float(v) for v in lam.min(axis=0)],
        "lam_max": [float(v) for v in lam.max(axis=0)],
        "analytic": {
            "R_i": analytic.R_inner, "H": analytic.H, "L": analytic.L,
            "lam": [analytic.lam_t, analytic.lam_z, analytic.lam_r],
            "sigma_ef": analytic.sigma_ef, "p_eff": analytic.p_eff,
        },
    }
    out["lam_error_pct"] = [
        100.0 * (out["lam_median"][i] / out["analytic"]["lam"][i] - 1.0) for i in range(3)
    ]
    return out, lam, gauss_rad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", nargs="+", default=["paper", "implied"])
    ap.add_argument("--outdir", default="output/cylinder")
    ap.add_argument("--workdir", default=None,
                    help="scratch directory for the FEBio runs")
    ap.add_argument("--radius", default="mid", choices=["mid", "inner"])
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw the figures from a previous run's output")
    args = ap.parse_args()

    use_paper_style()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    workroot = Path(args.workdir) if args.workdir else outdir / "febio"

    if args.figures_only:
        results = json.loads((outdir / "summary.json").read_text())
        stretches, gauss_r = {}, {}
        for name in results:
            d = np.load(outdir / f"aim_{name}.npz")
            stretches[name], gauss_r[name] = d["lam"], d["gauss_radius"]
        _report(results)
        _figures(results, stretches, gauss_r, outdir)
        print(f"\nredrew figures in {outdir}")
        return

    geo = CylinderGeometry()
    mesh = build_cylinder_mesh(geo)
    global mesh_global, geo_nodes_global
    mesh_global, geo_nodes_global = mesh, mesh.nodes
    print(f"mesh: {mesh.n_node} nodes, {mesh.n_elem} hexahedra\n")

    # ---------------------------------------------------------------- 0D
    print("=" * 78)
    print("0D thin-walled analytical solution")
    print("=" * 78)
    for name in args.preset:
        for conv in ("inner", "mid"):
            r = solve_cylinder_0d(PRESETS[name], geo, radius_convention=conv)
            print(f"\n[{name} / {conv}]")
            print(r)

    # ---------------------------------------------------------------- 3D
    results, stretches, gauss_r = {}, {}, {}
    for name in args.preset:
        mat = PRESETS[name]
        ana = solve_cylinder_0d(mat, geo, radius_convention=args.radius)
        print("\n" + "=" * 78)
        print(f"AIM for preset '{name}'  (c_e = {mat.c_e:g} kPa, "
              f"P_eff = {ana.p_eff:.4f} kPa)")
        print("=" * 78)
        solve = make_solver(mesh, mat, ana.p_eff, workroot / name,
                            mode="effective")
        aim = run_aim(mesh, solve, max_iter=args.max_iter)
        res, lam, rad = summarise(name, geo, mat, aim, ana)
        results[name], stretches[name], gauss_r[name] = res, lam, rad
        np.savez_compressed(
            outdir / f"aim_{name}.npz",
            X=aim.X, x=aim.x_target, hexes=mesh.hexes,
            residual=np.array(aim.residual), beta=np.array(aim.beta),
            change=np.array(aim.change), U_final=aim.displacements[-1],
            configs=np.array(aim.configs), lam=lam, gauss_radius=gauss_r[name],
        )
        # the stress-free elastin mesh, carrying the residual displacement and
        # the element-averaged principal prestretches (open in ParaView)
        write_vtk(
            outdir / f"cylinder_{name}.vtk", aim.X, mesh.hexes,
            point_data={
                "residual_displacement": aim.displacements[-1],
                "in_vivo_displacement": mesh.nodes - aim.X,
            },
            cell_data={
                "Ge_principal_1": lam.reshape(mesh.n_elem, -1, 3)[:, :, 0].mean(1),
                "Ge_principal_2": lam.reshape(mesh.n_elem, -1, 3)[:, :, 1].mean(1),
                "Ge_principal_3": lam.reshape(mesh.n_elem, -1, 3)[:, :, 2].mean(1),
            },
            title=f"AIM stress-free elastin configuration ({name})",
        )

    (outdir / "summary.json").write_text(json.dumps(results, indent=2))
    _report(results)
    _figures(results, stretches, gauss_r, outdir)
    print(f"\nwrote results and figures to {outdir}")


def _report(results) -> None:
    print("\n" + "=" * 78)
    print("Comparison with the values reported in the manuscript")
    print("=" * 78)
    hdr = f"{'quantity':24s}" + "".join(f"{k:>14s}" for k in results) + f"{'manuscript':>14s}"
    print(hdr)
    print("-" * len(hdr))

    def row(label, fn, paper):
        vals = "".join(f"{fn(results[k]):>14}" for k in results)
        print(f"{label:24s}{vals}{paper:>14}")

    row("iterations", lambda r: r["iterations"], PAPER["iterations"])
    row("max|U_1| [mm]", lambda r: f"{r['residual_first']:.3e}", f"{PAPER['residual_first']:.2e}")
    row("residual floor [mm]", lambda r: f"{r['residual_final']:.2e}", f"{PAPER['residual_floor']:.1e}")
    row("R_i,o  AIM [mm]", lambda r: f"{r['R_i']:.4f}", f"{PAPER['aim']['R_i']}")
    row("H_o    AIM [mm]", lambda r: f"{r['H']:.4f}", f"{PAPER['aim']['H']}")
    row("L_o    AIM [mm]", lambda r: f"{r['L']:.4f}", f"{PAPER['aim']['L']}")
    row("R_i,o  0D  [mm]", lambda r: f"{r['analytic']['R_i']:.4f}", f"{PAPER['analytic']['R_i']}")
    row("H_o    0D  [mm]", lambda r: f"{r['analytic']['H']:.4f}", f"{PAPER['analytic']['H']}")
    row("L_o    0D  [mm]", lambda r: f"{r['analytic']['L']:.4f}", f"{PAPER['analytic']['L']}")
    for i, lbl in enumerate(("lam_1 (circ.)", "lam_2 (axial)", "lam_3 (radial)")):
        row(f"{lbl} median", lambda r, i=i: f"{r['lam_median'][i]:.4f}",
            f"{PAPER['lam_median'][i]}")
        row(f"{lbl} vs 0D [%]", lambda r, i=i: f"{r['lam_error_pct'][i]:+.2f}",
            "+5.4/-4.4" if i in (0, 2) else "-")


def _figures(results, stretches, gauss_r, outdir: Path) -> None:
    # Fig. 2 -- convergence
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for name in results:
        d = np.load(outdir / f"aim_{name}.npz")
        r = d["residual"]
        ax.semilogy(np.arange(1, len(r) + 1), r, "o-", ms=3,
                    color=COLORS.get(name, None),
                    label=f"{name} ($c_e$={results[name]['c_e']:g} kPa)")
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel(r"max$\,|\mathbf{U}_k|$  [mm]")
    ax.set_title("AIM convergence, cylinder")
    ax.legend()
    fig.savefig(outdir / "fig2_convergence.pdf")
    fig.savefig(outdir / "fig2_convergence.png")
    plt.close(fig)

    # Fig. 4 -- principal prestretch through the wall
    names = list(results)
    fig, axes = plt.subplots(len(names), 3, figsize=(7.2, 2.2 * len(names)),
                             squeeze=False, sharex=True)
    labels = [r"$G_{e,1}$ (circ.)", r"$G_{e,2}$ (axial)", r"$G_{e,3}$ (radial)"]
    for row, name in enumerate(names):
        lam = stretches[name]
        rad = gauss_r[name]
        xi = (rad - rad.min()) / (rad.max() - rad.min())
        for i in range(3):
            ax = axes[row][i]
            ax.plot(xi, lam[:, i], ".", ms=1.5, color=COLORS.get(name, None),
                    label="AIM")
            ax.axhline(results[name]["analytic"]["lam"][i], ls="--", lw=0.9,
                       color="#333333", label="0D")
            if row == len(names) - 1 and i == 1:
                ax.set_xlabel("wall position  (0 = lumen, 1 = adventitia)")
            if i == 0:
                ax.set_ylabel(f"{name}\nprestretch")
            axes[0][i].set_title(labels[i])
    axes[0][0].legend()
    fig.suptitle("Elastin prestretch at the integration points", y=1.0)
    fig.tight_layout()
    fig.savefig(outdir / "fig4_prestretch.pdf")
    fig.savefig(outdir / "fig4_prestretch.png")
    plt.close(fig)

    # Fig. 3 -- residual displacement distribution at convergence
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for name in results:
        d = np.load(outdir / f"aim_{name}.npz")
        u = np.linalg.norm(d["U_final"], axis=1)
        ax.hist(u, bins=40, histtype="step", color=COLORS.get(name, None),
                label=f"{name}: max {u.max():.1e} mm")
    ax.set_xlabel(r"$|\mathbf{U}_k|$ at convergence  [mm]")
    ax.set_title("residual at the numerical floor\n(spikes = log-file precision)")
    ax.set_ylabel("nodes")
    ax.set_yscale("log")
    ax.legend()
    fig.savefig(outdir / "fig3_residual.pdf")
    fig.savefig(outdir / "fig3_residual.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
