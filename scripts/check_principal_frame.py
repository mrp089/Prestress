"""Are the principal stretch directions really circumferential / axial / radial?

A projection of a curved tube does not let the reader read an orientation off
the glyph figure.  This measures it: the angle between each principal
direction and the local (circumferential, axial, normal) frame of the wall.
"""
import argparse
import sys, warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
warnings.filterwarnings("ignore")

from prestress.febio import wall_model
from prestress.kinematics import deformation_gradients, principal_stretches
from run_aorta import SRC, build_wall

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--npz", type=Path, default=Path("output/p97/aim_nodal.npz"),
                help="AIM result; radius.npz is read from the same directory")
npz = ap.parse_args().npz
surf, mesh, *_ = build_wall(npz.parent / "radius.npz", SRC, 0.10)
m = wall_model(mesh)
X = np.load(npz)["X"]

F = deformation_gradients(X, mesh.nodes, m.elements, m.elem_type)
lam, dirs = principal_stretches(F)              # (nel,ngp,3), (nel,ngp,3,3)
lam = lam.mean(axis=1)
# principal_stretches returns directions in the REFERENCE configuration
# (eigenvectors of C); the wall frame below lives on the in-vivo surface, so
# push them forward: n_i = F v_i / |F v_i|.
F0, V0 = F[:, 0], dirs[:, 0]                    # first gauss point
vec = np.einsum("eij,ejk->eik", F0, V0)
vec /= np.linalg.norm(vec, axis=1, keepdims=True)

# Local wall frame, recovered from the fibre fields the mesh already carries.
# extrude_wall RENUMBERS nodes when it drops unreferenced ones, so an index
# arithmetic mapping back to the surface is wrong; node_fields survive the
# remap.  With f1 = c*ax + s*circ and f2 = c*ax - s*circ,
#   f1 + f2  ->  axial,   f1 - f2  ->  circumferential.
def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)

f1 = mesh.node_fields["fib1"]
f2 = mesh.node_fields["fib2"]
ax_n = unit(f1 + f2)
circ_n = unit(f1 - f2)
rad_n = unit(np.cross(ax_n, circ_n))

# element frame = mean over the element's nodes
def elem_avg(v):
    return unit(v[m.elements].mean(axis=1))

frame = {"circumferential": elem_avg(circ_n),
         "axial": elem_avg(ax_n),
         "radial": elem_avg(rad_n)}

print(f"{len(m.elements)} wedges, {npz}\n")
print(f"{'':16s} {'circ.':>10s} {'axial':>10s} {'radial':>10s}")
for i in range(3):
    v = vec[:, :, i]
    row = []
    for name in ("circumferential", "axial", "radial"):
        c = np.abs(np.einsum("ij,ij->i", v, frame[name])).clip(0, 1)
        row.append(np.degrees(np.arccos(c)))
    print(f"lambda_{i+1} (med {np.median(lam[:,i]):.4f})  "
          + "  ".join(f"{np.median(r):8.1f} deg" for r in row))

# how often is each principal direction closest to its expected axis?
best = np.argmin([np.degrees(np.arccos(np.abs(np.einsum("ij,ij->i", vec[:, :, i], frame[n])).clip(0,1)))
                  for i in range(3) for n in ("circumferential", "axial", "radial")], axis=0)
print()
for i, expect in enumerate(("circumferential", "axial", "radial")):
    v = vec[:, :, i]
    angs = np.stack([np.degrees(np.arccos(np.abs(np.einsum("ij,ij->i", v, frame[n])).clip(0,1)))
                     for n in ("circumferential", "axial", "radial")])
    hit = (angs.argmin(axis=0) == i).mean() * 100
    print(f"lambda_{i+1} is closest to {expect:16s} in {hit:5.1f} % of elements")
