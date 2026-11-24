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
