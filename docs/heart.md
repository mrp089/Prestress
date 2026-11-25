# Four-chamber heart: what the test case needs

The geometry is `eva.vtu` from Pfaller et al. (BMMB 2019), *The importance of
the pericardium for cardiac biomechanics*. Everything below is read off the
original BACI inputs in
`~/work/dissertation/simulations_2_remark_prestress_left_vs_right/`, so the
test case can be set up to match what the model was built for.

## Mesh

168,028 cells / 282,288 points: **167,232 ten-noded tetrahedra** plus 796
triangles that only exist to carry surface conditions (materials 6-17, shear
modulus zero, density 1e-16). Point data carries `node-fiber1` and
`node-fiber2` — the myofibre and sheet directions — and 75 `dsurf*` markers.

| material | cells | volume [mm³] | what |
|---|---|---|---|
| 1 | 59,801 | 104,231 | active myocardium (the only one with `ELAST_AnisoActiveStress_Evolution`) |
| 2 | 29,620 | 33,055 | passive myocardium |
| 3 | 16,611 | 10,919 | passive myocardium |
| 4 | 11,210 | 5,455 | passive myocardium |
| 5 | 49,990 | 59,235 | great vessels (soft isotropic neo-Hookean, `MUE 1.0`) |

## The four chambers

The `DESIGN SURFACE VOLUME MONITOR 3D` section names them, which resolves what
the `dsurf` markers in the `.vtu` mean:

| surface | marker in `eva.vtu` | nodes | chamber |
|---|---|---|---|
| E 2  | `dsurf2`  | 14,449 | **LV** |
| E 5  | `dsurf5`  | 15,795 | **RV** |
| E 44 | `dsurf44` | 8,716  | **LA** |
| E 59 | `dsurf59` | 8,778  | **RA** |

So prestressing all four chambers is possible with what is already in the file.
In the dissertation runs only the LV and RV carried pressure.

## Boundary conditions

| surface | marker | condition |
|---|---|---|
| E 19 | `dsurf19` | cut-offs, Dirichlet, all three components fixed |
| E 20 | `dsurf20` | great vessels, springs `1e3` N/mm in x, y, z |
| E 8  | `dsurf8`  | epicardium, spring `2.0` along the **reference surface normal** — the pericardium of Pfaller 2019 |

The epicardial normal spring is the part that matters: it is what the paper is
about, and leaving it out changes the prestressed state.

## Material

```
MAT 100  ELAST_IsoMooneyRivlin  C1 7.0  C2 0.04      [kPa]
MAT 102  ELAST_VolSussmanBathe  KAPPA 1e3            [kPa]
MAT 110  ELAST_IsoNeoHooke      MUE 1.0              great vessels
MAT 103  active stress — SIGMA 0.0 while prestressing
```

The passive myocardium is an isochoric Mooney-Rivlin plus a volumetric penalty,
which maps one-to-one onto FEBio's `Mooney-Rivlin` with `c1 = 7.0`,
`c2 = 0.04`, `k = 1e3`. The active stress is switched off during prestressing,
so the prestress problem is **purely elastic** — the constrained mixture
machinery is not needed, only the AIM itself.

## Loading

`VAL -0.133322` kPa (= 1 mmHg) scaled by a load curve, applied as
`orthopressure`, i.e. a follower pressure. The dissertation swept
LV/RV combinations of 10 and 0 mmHg — the directory names `pre_l10_r00`,
`pre_l00_r10`, `pre_l10_r10`, `pre_l00_r00` are exactly that study.

For a four-chamber end-diastolic prestress the natural choice is LV 10, RV 10,
LA 10, RA 5 mmHg, but the atrial values are a modelling decision rather than
something the original inputs fix.

## How this differs from the cylinder and the aorta

* **Purely elastic.** No constrained mixture, no fibre prestretch to account
  for. The AIM reduces to its original form (Rausch et al. 2017) and none of
  the effective-load machinery is needed.
* **Four pressure surfaces instead of one**, each a follower load.
* **Springs, not just Dirichlet.** FEBio has to reproduce the `1e3` xyz springs
  on the great vessels and the `2.0` reference-normal spring on the epicardium.
* **Quadratic tetrahedra** (`tet10`), with `tri6` pressure facets.

## A comparison worth making

The dissertation prestressed this model with BACI's **MULF** algorithm
(`PRESTRESS MULF`, `PRESTRESSTIME 0.8001`) — a different family from the
fixed-point iteration the AIM belongs to. Running the AIM on the same geometry
with the same loads and boundary conditions gives a direct comparison between
the two on a problem where an accepted answer already exists.

## What is built and checked

`src/prestress/heart.py` turns `eva.vtu` into an `FEModel`. Running it:

```
nodes 282288  tet10 167232                    (846,864 displacement dof)
  LV         7181 tri6 facets,  14449 nodes
  RV         7851 tri6 facets,  15795 nodes
  LA         4299 tri6 facets,   8716 nodes
  RA         4335 tri6 facets,   8778 nodes
  vessels    5766 tri6 facets,  11706 nodes
  epi       19594 tri6 facets,  39481 nodes
  base        445 tri6 facets,   1327 nodes
```

Every node carrying a `dsurfN` mask ends up in a facet of that surface, on all
seven surfaces — so the masks and the extracted boundary agree exactly.

Three checks pin the pieces that fail silently rather than loudly:

* **Node ordering.** VTK's `VTK_QUADRATIC_TETRA` and FEBio's `FETet10` number
  the ten nodes identically — corners `0-3`, then the edges `(0,1) (1,2) (2,0)
  (0,3) (1,3) (2,3)` — so the connectivity passes through unchanged. Verified
  against `FECore/FESolidElementShape.cpp`.
* **Quadrature.** FEBio's default for tet10 is the 8-point `FETet10G8`
  (`FEModelBuilder.cpp` sets `m_ntet10 = FE_TET10G8`), *not* the 4-point
  `FETet10G4` that a reader of the element list would reach for first.
* **Face orientation and closure.** The volume from the closed tri6 boundary
  (divergence theorem) and the volume from the tet10 quadrature agree to
  better than `1e-10` relative: `2.128941e+05 mm^3` both ways. No inverted
  elements — `min det(dx/dxi) = 6.04e-2` over all 1,337,856 integration
  points, which is a far better conditioned mesh than the extruded aorta wall.

`tests/test_basics.py::test_tet10_matches_febio` and
`::test_heart_surfaces_are_outward` keep all three honest.

## Writing the input

`febio.py` now emits several named surfaces, several follower pressures and
FEBio's `Robin BC` — which is exactly BACI's `ROBIN SPRING DASHPOT`:

| BACI | FEBio |
|---|---|
| spring to ground, `1e3` in xyz, `dsurf20` | `Robin BC`, `normal_stiffness = tangential_stiffness = 1e3` |
| reference-normal spring, `2.0`, `dsurf8` | `Robin BC`, `normal_stiffness = 2.0`, `tangential_stiffness = 0` |

The whole path — tet10 elements, tri6 facets, several named surfaces, follower
pressure and the Robin BC — was run end to end by FEBio 4.13 on an 8,731
element patch cut out of the apex before committing to the full model.

## Still open

The atrial end-diastolic pressures are a modelling decision, not something the
geometry settles. `ED_PRESSURE_MMHG` currently uses LV 10, RV 10 (the values
of Pfaller et al. 2019) with LA 10 and RA 5; physiological mean atrial
pressures are 8-12 mmHg on the left and 2-6 mmHg on the right.

## The heart as a constrained mixture (Gebauer et al. 2023)

The heart above is purely elastic, which makes its bottom panel in
`fig_convergence` degenerate: reloading the same reference configuration
reproduces the same deformation, so the AIM residual **is** the spurious
displacement and there is nothing to equilibrate. To put the heart on the
same footing as the cylinder and the aorta it needs a mixture, and
`scripts/run_heart_equilibrated.py` gives it the myocardium of

> Gebauer, Pfaller, Braeu, Cyron & Wall, *A homogenized constrained mixture
> model of cardiac growth and remodeling: analyzing mechanobiological
> stability and reversal*, Biomech Model Mechanobiol **22**:1983-2002 (2023).

### Their Table 1, transcribed

`prestress.params.GEBAUER23` carries it verbatim. The one trap is that
Gebauer et al. write the mixture energy **per unit mass** (their Eq. 4,
`Psi = sum_i rho0^i W^i`), so every stiffness in their table is in J/kg and
is not a stress until multiplied by a reference density. Getting that step
wrong rescales every fibre stress by a factor of ~1000 and still runs, so
`test_gebauer23_parameters` pins the conversion against hand-worked values.

| constituent | xi | a [J/kg] | b | lam_h | families | rho_0^i [kg/m^3] | sigma_h [kPa] |
|---|---|---|---|---|---|---|---|
| myocytes | 0.6 | 7.6 | 11.4 | 1.1 | 1 | 630.0 | 4.023 |
| collagen | 0.1 | 568 | 11.2 | 1.062 | 4 | 26.25 each | 5.163 each |
| elastin | 0.3 | 72 | - | - | 1 | 315.0 | - (isotropic) |

with `rho_0 = 1050 kg/m^3` and `kappa = 150 kPa`. Elastin is the only
constituent FEBio sees as a material: `c1 = rho_0^e a^e = 22.68 kPa`,
`c2 = 0`, `k = 150 kPa`. The myocyte and four collagen families enter as
nodal forces, exactly as on the aorta, so the solver still sees nothing but
an isotropic solid, follower pressures and nodal loads. Total fibre stress
is 24.68 kPa.

The four collagen families lie in the plane of the cardiomyocytes at 0, 90
and +/-45 degrees to the fibre direction (their Fig. 4), built from the
mesh's own `fiber1` and `fiber2`. About 1 % of nodes -- in the great vessels,
where no myocardial architecture is defined -- carry a sheet direction
*parallel* to the fibre direction, which makes Gram-Schmidt singular;
`cmm_fiber_families` falls back to an arbitrary perpendicular there, which is
harmless because those nodes carry no myocardium.

### Two deliberate departures

Their parameters were calibrated on a **systolic prolate-spheroid left
ventricle** at 120 mmHg; they are used here on an **end-diastolic
four-chamber** mesh. So:

* the **active muscle tone** of their Eq. (8) is off, as it is in Pfaller
  et al. (2019) during prestressing. At `lam_h = 1.1` it would add 26.73 kPa
  along the fibre -- more than the entire passive mixture -- so this is not a
  detail. `--active` turns it on.
* the **loading** is end-diastolic (LV/RV 10, LA 10, RA 5 mmHg), not 120 mmHg.

This is therefore their constitutive model on our geometry and loading, not a
reproduction of their simulation, and the numbers should not be read as such.

### Their own ladder, for comparison

Section 3.1 of Gebauer et al. runs exactly the experiment this repository
runs: prestress to a threshold, then switch on G&R and see where it settles.

| eps_pre [mm] | their iterations | deviation from homeostasis | after G&R [mm] | ratio |
|---|---|---|---|---|
| 0.1 | 136 | 2 % | 0.4 | 4.0 |
| 0.01 | 393 | 0.4 % | 0.03 | 3.0 |
| 0.001 | 745 | 0.04 % | 0.003 | 3.0 |

Two things stand out, and both are worth the manuscript's attention.

**Their ratio is constant at 3-4x over three decades** -- "around half an
order of magnitude", as they put it -- with no floor. The AIM on the cylinder
and the aorta instead reaches a floor and stops improving (see
`docs/aorta.md`). Whether that is a difference between the algorithms or
between the equilibration operators (they integrate the full G&R evolution to
a limit configuration; the ladders here solve one equilibrated elastic
problem) is exactly what the heart ladder is there to separate, since it runs
their material on our solver.

**Their prestressing takes 136 to 745 iterations where the AIM takes 15.**
That is not a small constant factor, and it is the clearest argument for the
AIM in the paper.
