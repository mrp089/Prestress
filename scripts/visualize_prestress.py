#!/usr/bin/env python3
"""Render the prestressed state: displacement, principal stretches, histograms.

The manuscript's figures plot the AIM *residual* on the geometry, which is a
convergence diagnostic rather than a result -- once the algorithm has
converged it is, by construction, noise at the solver tolerance.  What the
reader wants is the prestress itself:

  (a) the displacement from the stress-free elastin configuration to the
      imaged one, which is what prestressing actually produces;
  (b) the principal stretches *and their directions*, drawn as glyphs, since
      the eigenvalues alone do not say whether the wall is stretched
      circumferentially, axially or through the thickness;
  (c) histograms of the three principal stretches and of J, over the whole
      mesh rather than a single reported median.

    python scripts/visualize_prestress.py --case aorta
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prestress.elements import element_for
from prestress.kinematics import deformation_gradients, principal_stretches

STRETCH_COLORS = ("#d62728", "#2ca02c", "#1f77b4")      # lam1, lam2, lam3
STRETCH_LABELS = (r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$")


def load_case(case: str, npz: Path | None):
    """Return (nodes_in_vivo, elements, elem_type, X_unloaded, title)."""
    if case == "cylinder":
        import numpy as _np
        from prestress.mesh import build_cylinder_mesh
        d = _np.load(npz or "output/cylinder/aim_paper.npz")
        mesh = build_cylinder_mesh()
        return mesh.nodes, mesh.hexes, "hex8", d["X"], "idealised cylinder"
    if case == "aorta":
        from prestress.febio import wall_model
        from run_aorta import SRC, build_wall
        _, mesh, *_ = build_wall(Path("output/aorta/radius.npz"), SRC, 0.10)
        m = wall_model(mesh)
        d = np.load(npz or "output/aorta_nodal/aim_nodal.npz")
        return mesh.nodes, m.elements, m.elem_type, d["X"], "VMR 0012_H_AO_H aorta"
    if case == "heart":
        from prestress.heart import CHAMBERS, heart_model, load_heart
        h = load_heart("eva.vtu", surfaces={**CHAMBERS, "vessels": "dsurf20",
                                            "epi": "dsurf8", "base": "dsurf19"})
        m, _ = heart_model(h)
        d = np.load(npz or "output/heart/aim_heart.npz")
        return h.nodes, m.elements, m.elem_type, d["X"], "Pfaller19 four-chamber heart"
    raise SystemExit(f"unknown case {case!r}")


def to_cells(elements, values_gp):
    """Average a Gauss-point field over each element."""
    return values_gp.mean(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True,
                    choices=["cylinder", "aorta", "heart"])
    ap.add_argument("--npz", type=Path, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--glyphs", type=int, default=700,
                    help="how many points to draw principal directions at")
    ap.add_argument("--glyph-scale", type=float, default=1.0)
    ap.add_argument("--clip", action="store_true",
                    help="cut the model in half so interior glyphs are visible")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyvista as pv
    pv.OFF_SCREEN = True

    x, elements, etype, X, title = load_case(args.case, args.npz)
    out = Path(args.out or f"output/{args.case}/fig_prestress_{args.case}")
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- kinematics ---------------------------------------------------------
    F = deformation_gradients(X, x, elements, etype)          # X -> in vivo
    lam_gp, dirs_gp = principal_stretches(F)
    lam = to_cells(elements, lam_gp)                          # (n_elem, 3)
    J = to_cells(elements, np.linalg.det(F))
    dirs = dirs_gp[:, 0]                                      # one point/elem
    u = x - X                                                 # displacement
    umag = np.linalg.norm(u, axis=1)

    # --- grid ---------------------------------------------------------------
    vtk_type = {"hex8": 12, "penta6": 13, "tet10": 24}[etype]
    npe = elements.shape[1]
    cells = np.c_[np.full(len(elements), npe), elements].ravel()
    grid = pv.UnstructuredGrid(cells, np.full(len(elements), vtk_type, np.uint8), X)
    grid.point_data["displacement"] = u
    grid.point_data["|u| [mm]"] = umag
    for i in range(3):
        grid.cell_data[f"lambda{i + 1}"] = lam[:, i]
    grid.cell_data["J"] = J

    surf = grid.extract_surface()
    rng = np.random.default_rng(args.seed)
    centres = grid.cell_centers().points
    pick = rng.choice(len(centres), size=min(args.glyphs, len(centres)),
                      replace=False)

    # a sensible glyph length: a fraction of the median element size
    h = float(np.median(grid.compute_cell_sizes(length=False, area=False,
                                                volume=True)["Volume"]) ** (1 / 3))
    glyph_len = 1.3 * h * args.glyph_scale

    # --- panel 1: displacement ---------------------------------------------
    pl = pv.Plotter(off_screen=True, window_size=(900, 1100))
    pl.add_mesh(surf, scalars="|u| [mm]", cmap="viridis", smooth_shading=True,
                scalar_bar_args=dict(title="|u|  [mm]", vertical=True,
                                     position_x=0.86, position_y=0.25,
                                     height=0.5, width=0.05,
                                     title_font_size=18, label_font_size=15))
    pl.add_mesh(surf.copy(), color="k", style="wireframe", opacity=0.06)
    pl.background_color = "white"
    pl.view_isometric()
    pl.camera.zoom(1.25)
    png_u = out.with_name(out.name + "_displacement.png")
    pl.screenshot(str(png_u))
    pl.close()

    # --- panel 2: principal directions --------------------------------------
    pl = pv.Plotter(off_screen=True, window_size=(900, 1100))
    keep = np.ones(len(centres), bool)
    if args.clip:
        mid = 0.5 * (grid.bounds[2] + grid.bounds[3])
        keep = centres[:, 1] <= mid
        shown = grid.clip("y", origin=(0, mid, 0), invert=True).extract_surface()
    else:
        shown = surf
    pl.add_mesh(shown, color="#dddddd", opacity=0.30, smooth_shading=True)
    pick = pick[keep[pick]]
    line = pv.Line((-0.5, 0, 0), (0.5, 0, 0))
    for i, (c, lab) in enumerate(zip(STRETCH_COLORS, STRETCH_LABELS)):
        pts = pv.PolyData(centres[pick])
        pts["dir"] = dirs[pick, :, i]
        pts["mag"] = np.abs(lam[pick, i] - 1.0)
        g = pts.glyph(orient="dir", scale=False, factor=glyph_len, geom=line)
        pl.add_mesh(g, color=c, line_width=3,
                    label=f"{['lambda_1','lambda_2','lambda_3'][i]}  "
                          f"median {np.median(lam[:, i]):.4f}")
    pl.add_legend(bcolor="white", size=(0.32, 0.14), loc="lower left")
    pl.background_color = "white"
    pl.view_isometric()
    pl.camera.zoom(1.25)
    png_d = out.with_name(out.name + "_directions.png")
    pl.screenshot(str(png_d))
    pl.close()

    # --- panel 3: histograms -------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    def _frame(axn, v, c, lab):
        """Fit the axis to the data; a homogeneous field is a spike, not a gap."""
        lo, hi = np.percentile(v, [0.1, 99.9])
        if hi - lo < 1e-12:
            lo, hi = v.min() - 1e-6, v.max() + 1e-6
        pad = 0.08 * (hi - lo)
        axn.hist(v, bins=np.linspace(lo - pad, hi + pad, 81), color=c, alpha=0.85)
        axn.set_xlim(lo - pad, hi + pad)
        inside = lo - pad <= 1.0 <= hi + pad
        if inside:
            axn.axvline(1.0, color="0.3", lw=1.0, ls="--")
        axn.axvline(np.median(v), color="k", lw=1.2)
        axn.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        axn.tick_params(axis="x", labelsize=8)
        return lab if inside else f"{lab}   (1.0 off scale)"

    for i, (axn, c, lab) in enumerate(zip(axes[:3], STRETCH_COLORS, STRETCH_LABELS)):
        v = lam[:, i]
        axn.set_xlabel(_frame(axn, v, c, lab))
        axn.set_title(f"{lab}   median {np.median(v):.4f}\n"
                      f"[{np.percentile(v, 1):.3f}, {np.percentile(v, 99):.3f}] (1-99 %)",
                      fontsize=9)
        axn.set_yticks([])
        axn.grid(alpha=0.2, lw=0.5)
    axes[0].set_ylabel("elements")
    axes[3].set_xlabel(_frame(axes[3], J, "#7f7f7f", "$J$"))
    axes[3].set_title(f"$J$   median {np.median(J):.6f}\n"
                      f"max |J-1| = {np.abs(J - 1).max():.2e}", fontsize=9)
    axes[3].set_yticks([])
    axes[3].grid(alpha=0.2, lw=0.5)
    fig.suptitle(f"prestretch from the stress-free elastin configuration "
                 f"to the imaged one  -  {title}", fontsize=11)
    fig.tight_layout()
    png_h = out.with_name(out.name + "_histograms.png")
    fig.savefig(png_h, dpi=170)
    fig.savefig(png_h.with_suffix(".pdf"))

    grid.save(str(out.with_suffix(".vtu")), binary=True)
    print(f"wrote {png_u}")
    print(f"      {png_d}")
    print(f"      {png_h}  (+ .pdf)")
    print(f"      {out.with_suffix('.vtu')}  (for ParaView)")
    print(f"  |u|      median {np.median(umag):.4f}  max {umag.max():.4f} mm")
    for i in range(3):
        print(f"  lambda{i + 1}  median {np.median(lam[:, i]):.5f}  "
              f"[{lam[:, i].min():.5f}, {lam[:, i].max():.5f}]")
    print(f"  J        median {np.median(J):.6f}  max |J-1| {np.abs(J - 1).max():.3e}")


if __name__ == "__main__":
    main()
