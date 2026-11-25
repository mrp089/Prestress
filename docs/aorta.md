# Building a prestressable wall from a Vascular Model Repository aorta

> The model is **0012_H_AO_H**, a healthy thoracic aorta with the three arch
> branches (brachiocephalic, left carotid, left subclavian), taken from the
> Vascular Model Repository. The wall is extruded from
> `Simulations/0095_0001/mesh-complete/walls_combined.vtp`, which is the
> surface SimVascular meshed with, already smooth and with the five cap rings
> as its only open boundaries.

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

## Correction: Eq. 9 was not what failed

An earlier version of this file claimed that Eq. 9 taken literally is
"actively destructive" -- that FEBio could not load the surface variant past
13.9 % of the in vivo pressure. **That conclusion was wrong**, and it is worth
recording why, because the reasoning looked sound.

The surface variant did fail at 13.9 %. So did the Laplace variant, at
14.08 %, and the exact nodal variant, at 14.04 %. Three loads with nothing in
common -- one a pure discretisation artefact, one a smooth thin-wall relief,
one exactly self-equilibrated -- failing within 0.2 % of each other is not a
statement about any of them. Elastin alone, with no fibre contribution at all,
then failed in the same place.

The real causes were all in the solver and the mesh, and none of them had
anything to do with the fibres:

| cause | effect |
|---|---|
| `augtol = 1e-8` unreachable on a 115k-dof wall | 4,306 augmentations in one solve, each a full Newton solve of an already-converged problem (residual 3.8e-10 against a tolerance of 1.9e-9) |
| `etol = 1e-12`, `rtol = 1e-10` | FEBio exhausts `max_refs` and the line search wanders into negative jacobians; one load step ran 17 minutes without an error and still failed |
| `max_ups = 0` (full Newton) | a stiffness reformation per iteration, where the reformation is the cost |
| `opt_iter = 10` against ~34 needed | the auto stepper halved `dt` on solves that were converging |
| one wedge at quality 0.033 | two negative jacobians at 14 % load |

Together: 639 stiffness reformations to reach 20 % of the load, against 29 to
reach 100 % of it once fixed. See `docs/corrections.md` for what this costs
the manuscript's own reporting.

The statement that survives is the *analytical* one, unchanged and
independent of any of this: `sigma_ef . n = 0` exactly for an in-plane fibre
architecture, so Eq. 9 contributes nothing in the continuum, and what the
discrete surface integral produces is discretisation error. On the new
geometry its resultant is `5.8e2 mN` against a pressure resultant three orders
larger. That is a statement about the equation, not about whether FEBio can
solve it.

## What the three variants actually cost

On the 0012_H_AO_H wall (50,251 wedges, 38,386 nodes, median radius 10.1 mm,
median thickness 1.01 mm, in vivo pressure 17.332 kPa):

| variant | peak nodal force | resultant |
|---|---|---|
| `surface` | 5.20e1 mN | 5.84e2 mN |
| `laplace` | 1.68e1 mN | 3.63e2 mN |
| `nodal`   | 2.34e2 mN | **1.57e-11 mN** |

The exact `nodal` forces are by far the largest pointwise and their resultant
is zero to round-off: they are the discrete divergence of a self-equilibrated
stress field, so they cancel the way the continuum says they should. The
`surface` forces are an order of magnitude smaller pointwise and yet carry the
largest resultant of the three -- which is exactly what a load with no
consistent direction looks like. Magnitude was never the issue; consistency
was.

### Why the nodal forces are high-frequency, and why that is correct

A tangent fibre field on a curved surface rotates with the surface, so
`sigma_ef` genuinely varies from node to node and its discrete divergence
oscillates in sign between neighbours. Most of it cancels, leaving the
physical Laplace-like resultant, which is why the pointwise forces are several
times the per-node pressure force while the net fibre effect is only about
8 % of the load.

Mesh resolution matters here and the new geometry is much better placed: 1.5 mm
edges on a 10.1 mm radius give about 41 elements around the circumference,
against 13 on the old surface. The old mesh was coarse enough that the wall
normal turned 15 degrees across a typical edge and `sigma_ef` jumped 22 %
between adjacent nodes.

## Equilibrated G&R: the AIM residual is not the error that matters

Run the AIM to a given residual, then hand the result to the mechanobiologically
equilibrated G&R step and let it relax: whatever it moves is spurious, because
an equilibrated mixture handed its own homeostatic state should not move at
all. Six stopping thresholds at 97 mmHg on VMR `0012_H_AO_H`:

| AIM stopped at [mm] | iterations | equilibrated max [mm] | p99 | median | ratio |
|---|---|---|---|---|---|
| 2.534e-01 |  5 | 2.625e-01 | 2.49e-01 | 2.05e-02 |  1.04 |
| 5.789e-02 |  8 | 7.767e-02 | 6.14e-02 | 5.63e-03 |  1.34 |
| 1.693e-02 | 11 | 3.515e-02 | 3.03e-02 | 3.16e-03 |  2.08 |
| 5.006e-03 | 15 | 2.537e-02 | 2.62e-02 | 3.11e-03 |  5.07 |
| 1.899e-03 | 17 | 2.533e-02 | 2.62e-02 | 3.13e-03 | 13.34 |
| 9.894e-04 | 18 | 2.522e-02 | 2.62e-02 | 3.13e-03 | 25.49 |

**The spurious displacement hits a floor at 2.52e-02 mm and stays there.** The
last three rows tighten the AIM by a factor of five and move the equilibrated
result by 0.6 %. The growing ratio in the last column is not amplification — it
is a constant numerator divided by a shrinking denominator.

Two consequences:

* **Iterating the AIM past ~1e-02 mm on this geometry buys nothing.** Rows 4-6
  cost seven extra iterations, roughly 40 % of the total solve time, for no
  change in the quantity a G&R study actually cares about. Reporting a tighter
  AIM residual as evidence of a better reference configuration is reporting the
  wrong number.
* **The floor is set by the G&R step, not by the AIM.** It is the mismatch
  between the elastic prestress problem the AIM solves and the equilibrated
  mixture that consumes its answer -- the fibre stress the AIM assumes is
  homeostatic is not exactly the stress the equilibrated material produces in
  the configuration the AIM converges to. Driving it down needs a different
  fixed point, not a tighter tolerance on this one.

The median tells the same story from the other side: it is 3.1e-03 mm from row 3
onward, i.e. the bulk of the wall stops moving long before the maximum does, and
what remains is a tail concentrated in the descending aorta.

## Do the principal stretches point where they should?

The glyph figure draws the principal directions, but a projection of a curved
tube does not let a reader check an orientation. `scripts/check_principal_frame.py`
measures it against the wall's own (circumferential, axial, radial) frame,
recovered exactly from the two helical fibre families the mesh carries
(`f1 + f2` is axial, `f1 - f2` circumferential):

```
                      circ.      axial     radial
lambda_1 (med 1.0247)  19.4 deg   74.1 deg   85.4 deg    closest to circ.  87.5 %
lambda_2 (med 0.9987)  73.7 deg   23.2 deg   83.3 deg    closest to axial  82.5 %
lambda_3 (med 0.9771)  85.3 deg   83.1 deg   10.0 deg    closest to radial 93.0 %
```

So the prestress is circumferential extension with through-wall thinning and
very slight axial shortening, which is what a pressurised clamped tube should
do. The radial direction is the sharpest (10 deg median, 93 %); the in-plane
pair is looser because lambda_1 and lambda_2 are close in magnitude over much of
the wall, so their eigenvectors are free to rotate within the plane.

One trap worth recording: `extrude_wall` **renumbers nodes** when it drops
unreferenced ones, so recovering a surface index by `node_index % n_surf` is
wrong and produces a plausible-looking 45 deg mush. `node_fields` survive the
remap; index arithmetic does not.

## The J = 1.22 element

`max |J-1| = 2.21e-01` on the aorta against a histogram tight at 1.000 +- 0.002
is one element, and it is worth naming rather than leaving as an outlier.

Element 19126 is **the worst-shaped wedge in the mesh**: quality
`min(det)/mean(det) = 0.0527`, rank 0 of 50,251, a 5.6:1 sliver (shortest edge
0.33 mm, longest 1.86 mm). It passed because `extrude_wall`'s floor was
`min_quality = 0.05` and it sits 5 % above it. The floor is now `0.10`, which
drops that one wedge and nothing else.

That is a tidy-up, not a cure. All seven elements with `|J-1| > 0.05` share a
profile:

| | these 7 | mesh median |
|---|---|---|
| quality | 0.234 | 0.981 |
| distance to a clamped ring node | 5.8 mm | 16.7 mm |
| lambda_1 | 1.303 | 1.025 |

They are poor-shape wedges **next to the clamped inlet and outlet caps**, where
the Dirichlet constraint fights the pressure and the local stretch reaches 1.30
instead of 1.02. Bad shape and large stretch compound, and a penalty bulk
modulus does the rest. Raising the floor further does not reach them: the
element at `J = 1.158` has quality 0.526, which is a perfectly ordinary wedge,
so no defensible floor removes it.

The real levers are elsewhere:

* **`--laugon`** enforces incompressibility by augmented Lagrange instead of a
  penalty, which removes J drift by construction rather than by shrinking it.
  It is off by default here because it cost convergence on this geometry.
* **A better wall mesh.** Extruding from a decimated lumen surface is the
  limitation; slivers appear where the surface is locally wrinkled, and no
  post-hoc filter fixes that.

None of this moves a reported result: 7 elements out of 50,251 is 0.014 %, the
medians and the 1-99 % ranges are untouched, and the figures are rendered from
the mesh as it was run.
