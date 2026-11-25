#!/usr/bin/env python3
"""Render the prestressed state: displacement, principal directions, histograms.

The manuscript's figures plot the AIM *residual* on the geometry, which is a
convergence diagnostic rather than a result -- once the algorithm has
converged it is, by construction, noise at the solver tolerance.  What the
reader wants is the prestress itself:

  (a) the displacement from the stress-free elastin configuration to the
      imaged one, which is what prestressing actually produces;
  (b) one panel per principal direction, drawn as a sparse field of thick
      lines on a regular spatial grid, so the orientation is readable rather
      than a thicket of overlapping triads;
  (c) histograms of the three principal stretches over the whole mesh.

Elements that carry no prestretch at all are excluded from (b) and (c).  They
are not a result: on the heart the great vessels carry no pressure and are
pinned by stiff springs, so 4.4 % of the mesh sits at F = I exactly and puts a
delta spike at lambda = 1 in every histogram.  Their principal directions are
degenerate and meaningless.

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

# One camera per geometry so the three cases render identically from --case
# alone.  The aorta is turned 90 deg about the view-up axis: seen down x the
# arch is edge-on and reads as a kink, seen down y it opens out.
CASE_VIEW = {
    "cylinder": dict(view="iso", azimuth=0.0, elevation=0.0, clip=True),
    "aorta": dict(view="xz", azimuth=90.0, elevation=0.0, clip=False),
    "heart": dict(view="xz", azimuth=0.0, elevation=0.0, clip=True),
}

# below this an element has no prestretch at all: six orders under the
# physical signal (lambda_1 - 1 ~ 2e-2), so nothing real is ever caught
FLAT_TOL = 1e-6


def load_case(case: str, npz: Path | None):
    """Return (nodes_in_vivo, elements, elem_type, X_unloaded, title)."""
    if case == "cylinder":
        from prestress.mesh import build_cylinder_mesh
        d = np.load(npz or "output/cylinder/aim_paper.npz")
        mesh = build_cylinder_mesh()
        return mesh.nodes, mesh.hexes, "hex8", d["X"], "idealised cylinder"
    if case == "aorta":
        from prestress.febio import wall_model
        from run_aorta import SRC, build_wall
        # radius.npz is written beside the AIM result by run_aorta.py, so the
        # wall is always rebuilt from the cache belonging to *this* run.
        npz = Path(npz or "output/p97/aim_nodal.npz")
        _, mesh, *_ = build_wall(npz.parent / "radius.npz", SRC, 0.10)
        m = wall_model(mesh)
        d = np.load(npz)
        return mesh.nodes, m.elements, m.elem_type, d["X"], "VMR 0012_H_AO_H aorta"
    if case == "heart":
        from prestress.heart import CHAMBERS, heart_model, load_heart
        h = load_heart("eva.vtu", surfaces={**CHAMBERS, "vessels": "dsurf20",
                                            "epi": "dsurf8", "base": "dsurf19"})
        m, _ = heart_model(h)
        d = np.load(npz or "output/heart/aim_heart.npz")
        return h.nodes, m.elements, m.elem_type, d["X"], "Pfaller19 four-chamber heart"
    raise SystemExit(f"unknown case {case!r}")


def sparse_grid_pick(pts: np.ndarray, n_target: int) -> np.ndarray:
    """One representative point per occupied cell of a regular grid.

    Picking at random clusters the glyphs wherever the mesh happens to be
    fine, which is a picture of the mesh rather than of the field.  A grid
    gives even spatial coverage, so the direction field stays readable at a
    few hundred glyphs instead of needing tens of thousands.
    """
    lo = pts.min(axis=0)
    ext = float((pts.max(axis=0) - lo).max())
    if ext <= 0:
        return np.arange(min(len(pts), n_target))

    best = None
    n_lo, n_hi = 2, 512
    while n_lo <= n_hi:                       # bisect the cells-per-axis
        n = (n_lo + n_hi) // 2
        h = ext / n
        idx = np.floor((pts - lo) / h).astype(np.int64)
        keys, inv = np.unique(idx, axis=0, return_inverse=True)
        if best is None or abs(len(keys) - n_target) < abs(best[0] - n_target):
            best = (len(keys), keys, inv, h)
        if len(keys) < n_target:
            n_lo = n + 1
        else:
            n_hi = n - 1

    _, keys, inv, h = best
    inv = inv.ravel()
    centre = lo + (keys + 0.5) * h
    d = np.linalg.norm(pts - centre[inv], axis=1)
    order = np.lexsort((d, inv))              # nearest the cell centre first
    first = np.ones(len(order), bool)
    first[1:] = inv[order[1:]] != inv[order[:-1]]
    return order[first]


def _tick_fmt(clim) -> str:
    """Enough digits to tell the colour bar ticks apart.

    A homogeneous field spans a few times 1e-5, and "%.3f" then prints the
    same number five times down the bar.
    """
    span = float(clim[1] - clim[0])
    if span <= 0:
        return "%.4f"
    digits = int(np.ceil(-np.log10(span))) + 2
    return f"%.{min(max(digits, 3), 8)}f"


def _camera(pl, view: str, azimuth: float = 0.0, elevation: float = 0.0):
    {"iso": pl.view_isometric, "xy": pl.view_xy,
     "xz": pl.view_xz, "yz": pl.view_yz}[view]()
    # pyvista exposes azimuth/elevation as properties on some versions and as
    # methods on others; both rotate the camera about the focal point.
    for name, angle in (("azimuth", azimuth), ("elevation", elevation)):
        if not angle:
            continue
        attr = getattr(pl.camera, name)
        if callable(attr):
            attr(angle)
        else:
            setattr(pl.camera, name, attr + angle)
    pl.camera.zoom(1.3)


def to_cells(values_gp: np.ndarray) -> np.ndarray:
    """Average a Gauss-point field over each element."""
    return values_gp.mean(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", choices=["cylinder", "aorta", "heart"],
                    help="required unless --from-vtu is given")
    ap.add_argument("--from-vtu", type=Path, default=None,
                    help="replay the figures from an archived .vtu: it carries "
                         "the mesh, the displacement, the principal stretches "
                         "and their directions, so neither the source geometry "
                         "nor FEBio is needed")
    ap.add_argument("--npz", type=Path, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--glyphs", type=int, default=260,
                    help="target number of direction lines per panel")
    ap.add_argument("--glyph-scale", type=float, default=1.0)
    ap.add_argument("--line-width", type=float, default=7.0)
    ap.add_argument("--amplify", type=float, default=None,
                    help="length amplification about lambda = 1; the default "
                         "fits the 1-99 %% spread and is printed on the figure")
    ap.add_argument("--clip", dest="clip", action="store_true", default=None)
    ap.add_argument("--no-clip", dest="clip", action="store_false")
    ap.add_argument("--view", default=None,
                    choices=["iso", "xy", "xz", "yz"])
    ap.add_argument("--azimuth", type=float, default=None)
    ap.add_argument("--elevation", type=float, default=None)
    ap.add_argument("--opacity", type=float, default=0.28)
    ap.add_argument("--with-j", action="store_true",
                    help="add the J histogram (off: J is a material check, "
                         "not a prestress result)")
    args = ap.parse_args()

    if not args.case and not args.from_vtu:
        raise SystemExit("give --case, or --from-vtu to replay an archive")

    import pyvista as _pv
    if args.from_vtu and not args.case:
        # the case name travels with the archive so the camera comes out right
        args.case = str(_pv.read(args.from_vtu).field_data["case"][0])

    preset = CASE_VIEW[args.case]
    view = args.view or preset["view"]
    azim = preset["azimuth"] if args.azimuth is None else args.azimuth
    elev = preset["elevation"] if args.elevation is None else args.elevation
    clip = preset["clip"] if args.clip is None else args.clip

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyvista as pv
    pv.OFF_SCREEN = True

    out = Path(args.out or f"output/{args.case}/fig_prestress_{args.case}")
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.from_vtu:
        grid = pv.read(args.from_vtu)
        lam = np.column_stack([grid.cell_data[f"lambda{i + 1}"]
                               for i in range(3)])
        dirs = np.stack([np.asarray(grid.cell_data[f"dir{i + 1}"])
                         for i in range(3)], axis=2)
        u = np.asarray(grid.point_data["displacement"])
        umag = np.asarray(grid.point_data["|u| [mm]"])
        J = np.asarray(grid.cell_data["J"])
        live = np.asarray(grid.cell_data["prestressed"]).astype(bool)
        flat = ~live
        title = str(grid.field_data["title"][0])
        print(f"  replaying {args.from_vtu} ({grid.n_cells} cells)")
        if flat.any():
            print(f"  {flat.sum()} excluded elements carried in the archive")
    else:
        x, elements, etype, X, title = load_case(args.case, args.npz)

        # --- kinematics ---------------------------------------------------------
        diag_hint = float(np.linalg.norm(x.max(axis=0) - x.min(axis=0)))
        F = deformation_gradients(X, x, elements, etype)          # X -> in vivo
        lam_gp, dirs_gp = principal_stretches(F)
        lam = to_cells(lam_gp)                                    # (n_elem, 3)
        J = to_cells(np.linalg.det(F))
        u = x - X
        umag = np.linalg.norm(u, axis=1)

        # principal_stretches returns directions in the REFERENCE configuration
        # (eigenvectors of C).  The glyphs are drawn on the in-vivo mesh, so push
        # them forward: n_i = F v_i / |F v_i|.
        F0, V0 = F[:, 0], dirs_gp[:, 0]
        dirs = np.einsum("eij,ejk->eik", F0, V0)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

        # Two populations are excluded from the glyphs and the histograms, both
        # for the same reason: their stretch is imposed, not computed.
        #   1. elements at F = I, which carry no prestretch at all;
        #   2. elements with a node that never moved, i.e. sitting on the rigidly
        #      held boundary -- one clamped face fixes two of the three principal
        #      stretches at exactly 1 and piles those elements against the
        #      lambda_1 >= 1 >= lambda_3 bound as a spike.
        at_identity = (np.abs(lam - 1.0) < FLAT_TOL).all(axis=1)
        held = (np.linalg.norm(u, axis=1) < FLAT_TOL * diag_hint)[elements].any(axis=1)
        flat = at_identity | held
        live = ~flat
        if flat.any():
            print(f"  excluding {flat.sum()} of {len(lam)} elements "
                  f"({100 * flat.mean():.2f} %) from glyphs and histograms: "
                  f"{at_identity.sum()} at F = I, "
                  f"{(held & ~at_identity).sum()} on the held boundary")

        # --- grid ---------------------------------------------------------------
        vtk_type = {"hex8": 12, "penta6": 13, "tet10": 24}[etype]
        npe = elements.shape[1]
        cells = np.c_[np.full(len(elements), npe), elements].ravel()
        grid = pv.UnstructuredGrid(cells, np.full(len(elements), vtk_type, np.uint8), x)
        grid.point_data["displacement"] = u
        grid.point_data["|u| [mm]"] = umag
        for i in range(3):
            grid.cell_data[f"lambda{i + 1}"] = lam[:, i]
            grid.cell_data[f"dir{i + 1}"] = dirs[:, :, i]
        grid.cell_data["J"] = J
        grid.cell_data["prestressed"] = live.astype(np.uint8)
        # make the archive self-describing, so --from-vtu needs nothing else
        grid.field_data["case"] = np.array([args.case])
        grid.field_data["title"] = np.array([title])

    surf = grid.extract_surface()
    centres = grid.cell_centers().points

    b = np.asarray(grid.bounds).reshape(3, 2)
    diag = float(np.linalg.norm(b[:, 1] - b[:, 0]))
    base_len = 0.030 * diag * args.glyph_scale

    # which elements may seed a glyph: prestressed, and on the shown side
    seedable = live.copy()
    if clip:
        mid = 0.5 * (grid.bounds[2] + grid.bounds[3])
        seedable &= centres[:, 1] <= mid
        shown = grid.clip("y", origin=(0, mid, 0), invert=True).extract_surface()
    else:
        shown = surf
    seed_idx = np.flatnonzero(seedable)
    pick = seed_idx[sparse_grid_pick(centres[seed_idx], args.glyphs)]
    print(f"  {len(pick)} direction lines on a sparse grid "
          f"({len(seed_idx)} candidate elements)")

    # --- panel 1: displacement ---------------------------------------------
    pl = pv.Plotter(off_screen=True, window_size=(900, 1100))
    clim = (0.0, float(np.percentile(umag, 99)))
    pl.add_mesh(surf, scalars="|u| [mm]", cmap="viridis", clim=clim,
                above_color="#fde725", smooth_shading=True,
                scalar_bar_args=dict(title="|u|  [mm]  (99 %)", vertical=True,
                                     position_x=0.86, position_y=0.25,
                                     height=0.5, width=0.05,
                                     title_font_size=18, label_font_size=15))
    pl.add_mesh(surf.copy(), color="k", style="wireframe", opacity=0.06)
    pl.background_color = "white"
    _camera(pl, view, azim, elev)
    png_u = out.with_name(out.name + "_displacement.png")
    pl.screenshot(str(png_u))
    pl.close()

    # --- panels 2-4: one per principal direction ----------------------------
    # length is amplified about lambda = 1, because the stretches live within
    # a few percent of unity and an honest length is a constant length.  The
    # factor is printed on every panel.
    spread = max(float(np.abs(np.percentile(lam[live] - 1.0, [1, 99])).max()),
                 1e-9)
    amp = args.amplify if args.amplify is not None else 0.55 / spread

    line = pv.Line((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0))
    dir_pngs = []
    for i in range(3):
        v = lam[:, i]
        # A diverging map centred on 1 is the right picture only when the
        # field straddles 1.  It usually does not -- lambda_1 is above it
        # almost everywhere, lambda_3 below -- and then half the colormap is
        # wasted and every glyph reads as the same saturated colour.
        p1, p99 = np.percentile(v[live], [1, 99])
        if p1 < 1.0 < p99:
            half = max(abs(p1 - 1.0), abs(p99 - 1.0), 1e-6)
            vclim, vcmap = (1 - half, 1 + half), "coolwarm"
        else:
            pad = max(0.05 * (p99 - p1), 1e-6)
            vclim = (p1 - pad, p99 + pad)
            vcmap = "Reds" if p1 >= 1.0 else "Blues_r"
        pl = pv.Plotter(off_screen=True, window_size=(900, 1100))
        pl.add_mesh(shown, color="#d9d9d9", opacity=args.opacity,
                    smooth_shading=True, show_scalar_bar=False)

        pts = pv.PolyData(centres[pick])
        pts["dir"] = dirs[pick, :, i]
        pts["len"] = base_len * np.clip(1.0 + amp * (v[pick] - 1.0), 0.15, 4.0)
        pts[f"lambda{i + 1}"] = v[pick]
        g = pts.glyph(orient="dir", scale="len", factor=1.0, geom=line)
        pl.add_mesh(g, scalars=f"lambda{i + 1}", cmap=vcmap,
                    clim=vclim, line_width=args.line_width,
                    render_lines_as_tubes=True,
                    scalar_bar_args=dict(
                        title=f"lambda_{i + 1}", vertical=True,
                        position_x=0.80, position_y=0.25, height=0.5,
                        width=0.05, title_font_size=18, label_font_size=15,
                        n_labels=5, fmt=_tick_fmt(vclim)))
        pl.add_text(f"lambda_{i+1}   median {np.median(v[live]):.4f}\n"
                    f"length x{amp:.0f} about 1",
                    position="upper_left", font_size=11, color="black")
        pl.background_color = "white"
        _camera(pl, view, azim, elev)
        p = out.with_name(out.name + f"_lambda{i + 1}.png")
        pl.screenshot(str(p))
        pl.close()
        dir_pngs.append(p)

    # the three panels side by side, as the manuscript shows them
    def _crop_white(path, margin=8):
        """Trim the white border pyvista leaves around the model."""
        im = plt.imread(path)
        rgb = im[..., :3]
        ink = (rgb < 0.985).any(axis=2)
        if not ink.any():
            return im
        rows, cols = np.flatnonzero(ink.any(1)), np.flatnonzero(ink.any(0))
        r0 = max(rows[0] - margin, 0)
        r1 = min(rows[-1] + margin + 1, im.shape[0])
        c0 = max(cols[0] - margin, 0)
        c1 = min(cols[-1] + margin + 1, im.shape[1])
        return im[r0:r1, c0:c1]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.6))
    for axn, p, lab in zip(axes, dir_pngs, STRETCH_LABELS):
        axn.imshow(_crop_white(p))
        axn.set_title(lab, fontsize=13)
        axn.axis("off")
    fig.suptitle(f"principal stretch directions  -  {title}", fontsize=12)
    fig.tight_layout()
    png_d = out.with_name(out.name + "_directions.png")
    fig.savefig(png_d, dpi=170)
    fig.savefig(png_d.with_suffix(".pdf"))
    plt.close(fig)

    # --- panel 5: histograms -------------------------------------------------
    n_panel = 4 if args.with_j else 3
    fig, axes = plt.subplots(1, n_panel, figsize=(3.9 * n_panel, 3.4))
    axes = np.atleast_1d(axes)

    def _frame(axn, v, c, lab):
        lo, hi = np.percentile(v, [0.1, 99.9])
        if hi - lo < 1e-12:
            lo, hi = v.min() - 1e-6, v.max() + 1e-6
        pad = 0.08 * (hi - lo)
        axn.hist(v, bins=np.linspace(lo - pad, hi + pad, 81), color=c,
                 alpha=0.85)
        axn.set_xlim(lo - pad, hi + pad)
        inside = lo - pad <= 1.0 <= hi + pad
        if inside:
            axn.axvline(1.0, color="0.3", lw=1.0, ls="--")
        axn.axvline(np.median(v), color="k", lw=1.2)
        axn.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        axn.tick_params(axis="x", labelsize=8)
        return lab if inside else f"{lab}   (1.0 off scale)"

    for i, (axn, c, lab) in enumerate(zip(axes[:3], STRETCH_COLORS,
                                          STRETCH_LABELS)):
        v = lam[live, i]
        axn.set_xlabel(_frame(axn, v, c, lab))
        axn.set_title(f"{lab}   median {np.median(v):.4f}\n"
                      f"[{np.percentile(v, 1):.3f}, "
                      f"{np.percentile(v, 99):.3f}] (1-99 %)", fontsize=9)
        axn.set_yticks([])
        axn.grid(alpha=0.2, lw=0.5)
    axes[0].set_ylabel("elements")
    if args.with_j:
        axes[3].set_xlabel(_frame(axes[3], J[live], "#7f7f7f", "$J$"))
        axes[3].set_title(f"$J$   median {np.median(J[live]):.6f}\n"
                          f"max |J-1| = {np.abs(J[live] - 1).max():.2e}",
                          fontsize=9)
        axes[3].set_yticks([])
        axes[3].grid(alpha=0.2, lw=0.5)

    sub = (f"   ({flat.sum()} unprestressed elements excluded)"
           if flat.any() else "")
    fig.suptitle(f"prestretch from the stress-free elastin configuration "
                 f"to the imaged one  -  {title}{sub}", fontsize=11)
    fig.tight_layout()
    png_h = out.with_name(out.name + "_histograms.png")
    fig.savefig(png_h, dpi=170)
    fig.savefig(png_h.with_suffix(".pdf"))
    plt.close(fig)

    grid.save(str(out.with_suffix(".vtu")), binary=True)
    print(f"wrote {png_u}")
    for p in dir_pngs:
        print(f"      {p}")
    print(f"      {png_d}  (+ .pdf, the three side by side)")
    print(f"      {png_h}  (+ .pdf)")
    print(f"      {out.with_suffix('.vtu')}  (for ParaView)")
    print(f"  |u|      median {np.median(umag):.4f}  max {umag.max():.4f} mm")
    for i in range(3):
        v = lam[live, i]
        print(f"  lambda{i + 1}  median {np.median(v):.5f}  "
              f"[{v.min():.5f}, {v.max():.5f}]")
    print(f"  J        median {np.median(J[live]):.6f}  "
          f"max |J-1| {np.abs(J[live] - 1).max():.3e}")


if __name__ == "__main__":
    main()
