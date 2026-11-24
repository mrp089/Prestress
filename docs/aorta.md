# Building a prestressable wall from a Vascular Model Repository aorta

The manuscript's second example is a patient-specific human aorta. That
geometry is not public, so the test case here is built from a VMR aortofemoral
model (`~/repos/Aorta/mesh_example`): inlet, AAO1-3, celiac, SMA, both renals,
both iliacs. Units are cm (inlet radius 1.79 -> 17.9 mm, 50 cm long) and are
scaled to mm on load.

## The wall has to be built

VMR models are fluid-domain meshes: `walls_combined.vtp` is the lumen boundary
and there is no solid wall. `src/prestress/wall.py` makes one.

| step | what and why |
|---|---|
| decimate 108k -> 13k triangles | boundary vertices preserved, so the 10 inlet/outlet rings survive |
| Taubin smoothing, 30 iterations | opens up the concave creases at branch junctions |
| local radius by ray casting | cast inwards along the normal, hit the far wall, halve the chord — no centreline needed |
| thickness `0.1 * r`, clipped | median 0.72 mm on a median radius of 7.2 mm |
| extrude outwards, 2 layers | wedge (`penta6`) elements |

Result: **25,854 wedges, 19,770 nodes, 12,927 lumen facets, 813 ring nodes**,
all volumes positive, median thickness 0.721 mm against the 0.722 mm asked for.

## The fibre architecture

Two families at ±30° to the local axial direction, as in the manuscript. There
is no centreline, so the axial direction comes from the gradient of the geodesic
distance to the inlet: Dijkstra over the surface graph (the inlet ring is found
automatically as the widest open boundary), smoothed, then the piecewise-linear
gradient projected into the tangent plane.

Checks: `|f1| = 1.0000`, `f1 . f2 = 0.500` (exactly `cos 60°`), `|f1 . n| =
2.6e-15`.

## Why the extrusion is hard, and what actually works

Offsetting a surface outwards by more than its smallest concave radius of
curvature self-intersects. At the crotch between a branch and the trunk that
radius is smaller than any usable wall thickness, so some elements fold however
carefully the offset is done.

The diagnostic that settles it is a sweep over decimation:

| decimation | triangles | min angle (p1) | inverted wedges |
|---|---|---|---|
| none | 107,740 | 37.8° | **49** |
| 0.50 | 53,870  | 16.7° | 28 |
| 0.70 | 32,322  | 14.2° | 19 |
| 0.80 | 21,548  | 12.9° | 11 |
| 0.88 | 12,928  | 12.0° | **10** |

The *best quality* surface folds the most. Triangle quality is not the driver
and decimation is not the cause: a fine mesh resolves the creases faithfully and
folds, a coarse one smooths them away implicitly. This is geometry.

Four repair strategies were tried and three fail:

* **Local thinning, smoothed.** Oscillates — pulling a node down inverts its
  neighbour, so the bad set wanders instead of emptying.
* **Global thickness scaling.** The largest clean scale is 0.002, i.e. a 1.4 µm
  wall. Useless.
* **Graded thinning** (`|h_i - h_j| <= a |x_i - x_j|`). Fails for every `a` from
  0.5 down to 0.05; the thinned region grows from 44 to 594 nodes without ever
  clearing.
* **Delete the folding triangles and refill the holes.** Converges to 1-3 but
  oscillates, because the new triangles fold in their turn.

What works: run the local thinning, then **drop whatever still folds** —
2 of 25,856 wedges, 0.008 % of the mesh and ~7 mm² of a 6,886 mm² pressure
surface. The rest of the wall keeps its full thickness (only 44 of 6,590 nodes
are thinned at all).

Two smaller traps on the way:

* `auto_orient_normals` is unreliable on an **open** surface — a few points come
  back flipped and smoothing then rotates them 180°. The normals are built from
  the triangle winding instead, which is consistent (4 bad edges in 19,253).
* Below about 1 % of the nominal thickness the wedge volume is pure round-off
  against 50 mm coordinates and its sign is meaningless, so any loop testing
  `volume <= 0` runs forever. The thinning is floored.

## What this geometry shows about Eq. 9

The fibres lie in the plane of the wall. On the luminal surface the outward
normal is perpendicular to that plane, so

```
sigma_ef . n  =  sum_families sigma (m . n) m  =  0
```

**exactly**, for any in-plane fibre architecture at any angle — the cylinder's
circumferential family and the aorta's ±30° diagonal families alike. The angle
only sets the direction *within* the tangent plane.

So Eq. 9, taken literally, contributes **nothing at all**. The whole fibre
contribution sits in the volume term dropped between Eqs. 6 and 7
(`docs/corrections.md` §5). On the cylinder this is invisible, because Sec. 2.4
silently replaces it with a thin-wall pressure reduction
`P_eff = P - sigma h/r` that is a different construction, not an implementation
of Eq. 9. On a patient-specific aorta there is no such substitute in the
manuscript: the method as written has no stated way to account for the fibres.

Discretised, the surface term is not quite zero — the flat facet normals of a
decimated triangulation differ from the smooth surface normal by about 6°
(median), so `|m . n|` has a median of 0.10 — but it has no consistent
direction and its resultant is `4.0e3 mN` against a pressure force of
`1.29e6 mN`, i.e. 0.3 %.

`scripts/run_aorta.py` therefore compares three variants, all of which reduce
to a follower pressure plus a fixed nodal force vector:

| variant | fibre contribution |
|---|---|
| `surface` | Eq. 9 literally: `int N_a (sigma_ef . n) ds` over the lumen |
| `laplace` | Sec. 2.4 made local: `-sigma_hoop h/r` along the luminal normal, with the local thickness and radius |
| `nodal` | exact: `-int sigma_ef . grad(N_a) dv` |

## Two things the cylinder hid

A structured cylinder is forgiving in ways a branched patient-specific wall is
not, and both of these only showed up here.

**The fibre nodal forces carried no load controller.** FEBio's `nodal_force`
takes `FE_PARAM_ADDLC`, so a `value` written without an `lc` attribute is held
constant for the whole analysis — while the luminal pressure ramps from zero
over five steps. The first load step therefore saw the complete fibre
prestress against a tenth of the pressure meant to balance it. On the cylinder
the imbalance is small next to the elastin stiffness and the solve rides
through it; on the aorta it is not. The converged fixed point does not depend
on the path, so every cylinder number is unchanged (still `8.571366e-12 mm` in
11 iterations) — only the robustness differs.

**Five equal load steps are not enough.** Branch junctions pass through
configurations that a 20 % load increment cannot resolve. The AIM solves now
run with FEBio's auto time stepper, which changes the path and not the answer.

## Eq. 9 does not merely contribute nothing — it is actively destructive

With both fixes in place the `surface` variant still cannot be loaded at all.
FEBio drives the step size down to `1.85e-5` and gives up at **13.9 % of the
in vivo pressure** with negative jacobians, after four minutes of retries.

That is the sharper form of the result above. Because `sigma_ef . n = 0` in the
continuum, *everything* the discrete surface integral produces is
discretisation error: a load with no consistent direction, set by how far the
flat facet normals of a decimated triangulation stray from the smooth surface
normal. It is not a small approximation to the right load, it is a sampling of
noise — `4.0e3 mN` of resultant that the wall has no way to carry. Its
`|f|max` of `4.3e2 mN` sits on free nodes, not on the clamped rings.

The exact `nodal` forces are larger still (`9.1e2 mN` peak) but their resultant
is `4.6e-11 mN`: they are the discrete divergence of a self-equilibrated stress
field, so they cancel the way the continuum says they should. Magnitude was
never the problem; consistency was.

### Why the forces are high-frequency, and why that is correct

On this mesh the stress tensor `sigma_ef` jumps by 22 % between adjacent nodes
(99th percentile 79 %). That is not noise in the fibre field: the wall normal
itself turns 15° across a typical edge, because 3.36 mm edges on a 7.2 mm
radius vessel give only about 13 elements around the circumference. A tangent
fibre field on a curved surface genuinely rotates that fast. The resulting
nodal forces oscillate in sign between neighbours and largely cancel, leaving
the physical Laplace-like resultant — which is why the median nodal force
(44 mN) is several times the median pressure force (18 mN) while the net
fibre effect is only about 8 % of the load.
