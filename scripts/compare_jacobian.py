#!/usr/bin/env python3
"""Does the augmented Lagrangian cure the jacobian drift?

Compares J of the FIRST AIM iteration of the penalty run (output/p97) with
the first iteration of the --laugon run (output/p97_laugon).  Both solve the
same forward problem from the same reference configuration, so the element
numbering and the loads are identical and the only difference is how
incompressibility is enforced.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prestress.kinematics import deformation_gradients
from prestress.febio import wall_model
from run_aorta import build_wall, SRC


def read_pos(path):
    rows = [l.split() for l in Path(path).read_text().splitlines()
            if l and not l.startswith("*")]
    a = np.array(rows, dtype=float)
    return a[:, 1:4] if a.shape[1] == 4 else a


def jac(X, x, elements, et):
    F = deformation_gradients(X, x, elements, et)
    return np.linalg.det(F).mean(axis=1) if F.ndim == 4 else np.linalg.det(F)


out = ROOT / "output"
_, mesh, *_ = build_wall(out / "p97" / "radius.npz", SRC, 0.10)
model = wall_model(mesh)
X = mesh.nodes                      # first AIM iteration: reference = target
print(f"mesh {model.n_elem} wedges, {model.n_node} nodes")

for tag, p in (("penalty", out / "p97/febio/nodal/aim_001_pos.txt"),
               ("laugon ", out / "p97_laugon/febio/nodal/aim_001_pos.txt")):
    if not p.is_file():
        print(f"{tag}: missing {p}")
        continue
    x = read_pos(p)
    J = jac(X, x, model.elements, model.elem_type)
    d = np.abs(J - 1.0)
    print(f"{tag}  median |J-1| {np.median(d):.3e}   p99 {np.percentile(d,99):.3e}"
          f"   max {d.max():.3e} (elem {d.argmax()})"
          f"   n(|J-1|>0.05) {(d>0.05).sum()}")
    np.save(out / f"J_{tag.strip()}.npy", J)

a, b = out / "J_penalty.npy", out / "J_laugon.npy"
if a.is_file() and b.is_file():
    Ja, Jb = np.load(a), np.load(b)
    bad = np.abs(Ja - 1.0) > 0.05
    print(f"\nthe {bad.sum()} worst penalty elements, under each scheme:")
    for e in np.argsort(-np.abs(Ja - 1.0))[:8]:
        print(f"  elem {e:6d}   penalty J {Ja[e]:.4f}   laugon J {Jb[e]:.4f}")
