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

## What this geometry is good for

On the cylinder, Eq. 9 taken literally gives `sigma_ef . n = 0` on the lumen
(the fibres are circumferential, the normal is radial), which is why Sec. 2.4
quietly substituted the resultant instead. With ±30° diagonal fibres on the
aorta, `sigma_ef . n` is **not** zero, so Eq. 9 is directly implementable — and
can be compared against the exact nodal-force assembly on the kind of geometry
the manuscript actually targets.
