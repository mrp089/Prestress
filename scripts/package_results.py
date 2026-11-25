#!/usr/bin/env python3
"""Build the figure archive that a reader downloads to regenerate the plots.

What goes in is the smallest set that reproduces every figure *without* the
source geometry and without FEBio:

  * one self-contained ``.vtu`` per geometry -- mesh, displacement, the three
    principal stretches and their directions, and the exclusion mask;
  * the convergence and equilibration ladders as JSON;
  * the AIM results as ``.npz`` (reference configuration and residual
    history), for anyone who wants to recompute rather than replay.

What stays out is the input geometry.  The VMR model (337 MB) and the
Pfaller19 heart mesh (35 MB) belong to their originators and are fetched from
their own records instead; see ``docs/data_release.md``.

    python scripts/package_results.py --out dist/prestress-figures.tar.gz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

# published path -> source, relative to the repository root
CONTENTS = {
    "vtu/cylinder.vtu": "output/cylinder/fig_prestress_cylinder.vtu",
    "vtu/aorta.vtu": "output/aorta/fig_prestress_aorta.vtu",
    "vtu/heart.vtu": "output/heart/fig_prestress_heart.vtu",
    "ladders/cylinder_thresholds.json": "output/thresholds/summary.json",
    "ladders/aorta_equilibrated.json": "output/aorta_equil/summary.json",
    "ladders/heart_summary.json": "output/heart/summary_heart.json",
    "aim/cylinder_aim.npz": "output/cylinder/aim_paper.npz",
    "aim/aorta_aim.npz": "output/p97/aim_nodal.npz",
    "aim/aorta_radius.npz": "output/p97/radius.npz",
    "aim/heart_aim.npz": "output/heart/aim_heart.npz",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path,
                    default=Path("dist/prestress-figures.tar.gz"))
    args = ap.parse_args()

    present, missing = {}, []
    for name, src in CONTENTS.items():
        path = args.root / src
        if path.is_file():
            present[name] = path
        else:
            missing.append((name, src))
    if missing:
        print("missing, will be left out:")
        for name, src in missing:
            print(f"  {name:38s} <- {src}")

    manifest = {
        "description": "Figure data for the augmented iterative method paper",
        "regenerate": "python scripts/visualize_prestress.py "
                      "--from-vtu vtu/<case>.vtu   and   "
                      "python scripts/plot_convergence.py --root .",
        "files": {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with tarfile.open(args.out, "w:gz") as tar:
        for name, p in present.items():
            tar.add(p, arcname=name)
            size = p.stat().st_size
            total += size
            manifest["files"][name] = {"bytes": size, "sha256": sha256(p)}
            print(f"  + {name:38s} {size / 1e6:8.2f} MB")
        mf = args.out.parent / "MANIFEST.json"
        mf.write_text(json.dumps(manifest, indent=2))
        tar.add(mf, arcname="MANIFEST.json")

    print(f"\nwrote {args.out}  "
          f"({args.out.stat().st_size / 1e6:.1f} MB packed, "
          f"{total / 1e6:.1f} MB raw)")
    print(f"      {mf}")


if __name__ == "__main__":
    main()
