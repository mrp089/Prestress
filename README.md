# Homeostatic prestress for constrained mixture models

A Python + FEBio re-implementation of

> S. Parikh, M. R. Pfaller, S. Loerakker,
> *An augmented iterative method for establishing a homeostatic prestressed
> reference configuration for constrained mixture modeling.*

The repository reproduces the cylinder example of the manuscript (0D analytical
solution and 3D finite element solution), verifies the two against each other,
and adds a study of how accurate the prestress has to be before growth and
remodelling (G&R) started from it stays put.

---

## What is implemented

| | |
|---|---|
| `src/prestress/params.py`     | geometry, material and solver parameters (mm, kPa, mN) |
| `src/prestress/material.py`   | elastin (Eq. 11) and embedded fibre (Eq. 10) constitutive models |
| `src/prestress/analytic.py`   | 0D thin-walled cylinder prestress solution (Sec. 2.4) |
| `src/prestress/mesh.py`       | structured hexahedral mesh of the vessel |
| `src/prestress/febio.py`      | FEBio 4 input writer, runner and log reader |
| `src/prestress/aim.py`        | Algorithm 1, the augmented iterative method |
| `src/prestress/kinematics.py` | deformation gradients and prestretches at the integration points |
| `src/prestress/growth.py`     | G&R started from a prestressed configuration |
| `src/prestress/vtk.py`        | legacy-VTK writer for ParaView |
| `febio_plugin/`               | `prestress_cmm`, an FEBio material for the constrained mixture with G&R |
| `tests/test_basics.py`        | checks that run without FEBio |

Defaults are the most accurate settings available: incompressibility through an
augmented Lagrangian rather than the bulk penalty alone, and only the deviatoric
part of the prescribed fibre stress assembled into the nodal forces. Together
these remove the `O(mu/K)` error (`docs/corrections.md` §5). The augmentation
drives `J -> 1`, so it is switched off automatically for runs with mass
turnover, where the constraint is `J -> J_g`.

The plugin is written from scratch, but its structure follows
[`yale-humphrey-lab/FEMBE_Plugin`](https://github.com/yale-humphrey-lab/FEMBE_Plugin)
(Latorre & Humphrey's mechanobiologically equilibrated CMM). That plugin was not
reusable here: it hard-codes the material parameters, the cylinder's local
basis and a uniform elastin prestretch `G_e = diag(1/(1.9·1.62), 1.9, 1.62)`,
whereas the whole point of the AIM is that `G_e` is heterogeneous and only known
after prestressing. `prestress_cmm` takes the prestretch from the AIM through
per-integration-point element maps instead.

Only the cylinder is reproduced. The patient-specific aorta of the draft needs a
geometry that is not in the repository.

## Setup

FEBio is built from source into `external/FEBio` and the plugin against it:

```bash
git clone --depth 1 https://github.com/febiosoftware/FEBio.git external/FEBio
```

```bash
OMP=$(brew --prefix libomp); cmake -S external/FEBio -B external/FEBio/cbuild -DCMAKE_BUILD_TYPE=Release -DUSE_MKL=OFF -DUSE_HYPRE=OFF -DUSE_MMG=OFF -DUSE_LEVMAR=OFF -DUSE_NLOPT=OFF -DUSE_SUPERLU_MT=OFF -DUSE_PDL=OFF -DUSE_FFTW=OFF -DOMP_INC=$OMP/include -DCMAKE_CXX_FLAGS="-Xpreprocessor -fopenmp -I$OMP/include -w" -DCMAKE_C_FLAGS="-Xpreprocessor -fopenmp -I$OMP/include -w" -DCMAKE_SHARED_LINKER_FLAGS="-L$OMP/lib -lomp" -DCMAKE_EXE_LINKER_FLAGS="-L$OMP/lib -lomp"
```

```bash
cmake --build external/FEBio/cbuild -j 10
```

```bash
./febio_plugin/build.sh
```

Set `FEBIO_EXE` / `FEBIO_PLUGIN` to use an existing installation instead.
Python needs only NumPy, SciPy and Matplotlib.

## Running

```bash
python scripts/run_cylinder.py
```

```bash
python scripts/run_thresholds.py
```

```bash
python scripts/run_growth.py
```

Both write their figures, a `summary.json` and the AIM iterates to `output/`.
`run_cylinder.py` also writes `cylinder_<preset>.vtk` — the stress-free elastin
mesh carrying the residual displacement and the principal prestretches, to open
in ParaView — and `--figures-only` redraws the figures from a previous run.

```bash
python tests/test_basics.py
```

```bash
python tests/test_febio.py
```

The first needs nothing but NumPy/SciPy; the second runs the cylinder through
FEBio and checks the plugin against FEBio's built-in Mooney-Rivlin, the AIM
against the 0D solution, and that a prestressed mixture really is in
equilibrium.

---

## Cylinder example

Idealised mouse descending thoracic aorta: `r_ivo = 0.33 mm`, `h_ivo = 0.04 mm`,
`l_ivo = 2 mm`, 4032 hexahedra, elastin + one circumferential fibre family
(`phi_ef = 0.33`) deposited at `lambda = 1.1`, 120 mmHg.

Two parameter sets are run: `paper` is Table 1 verbatim (`c_e = 1000 kPa`) and
`implied` uses the elastin modulus that actually reproduces the unloaded
dimensions reported in the draft (`c_e = 260 kPa`; the least-squares fit is
262 kPa, see `docs/corrections.md`).

| | `paper` | `implied` | reported |
|---|---|---|---|
| AIM iterations            | 22 | 15 | 26 |
| first residual `max U_1` [mm] | 2.6e-2 | 1.1e-1 | 1.6e-1 |
| residual floor [mm]       | 7e-12 | 9e-12 | 3.7e-9 |
| `R_i,o` AIM / 0D [mm]     | 0.3201 / 0.3199 | 0.2948 / 0.2939 | 0.27 / 0.29 |
| `H_o` AIM / 0D [mm]       | 0.0406 / 0.0406 | 0.0424 / 0.0422 | 0.044 / 0.042 |
| `L_o` AIM / 0D [mm]       | 2.0254 / 2.0287 | 2.0944 / 2.1079 | 2.14 / 2.10 |
| AIM vs 0D prestretch      | ≤ 0.2 % | ≤ 0.7 % | 4–5 % |

The AIM and the analytical solution agree to well under 1 % in all three
principal prestretches — an order of magnitude better than the agreement
reported in the draft, which is itself an artefact of the two having been run
with different elastin moduli.

## How accurate does the prestress have to be?

`scripts/run_thresholds.py` is the manuscript's Fig. 2 carried through to its
consequence, the way Gebauer et al. (BMMB 2023) do it: the AIM is stopped at a
range of thresholds and each prestressed configuration is taken to its
mechanobiologically equilibrated state.

No growth & remodelling time stepping is needed for that. At equilibrium every
turning-over constituent is back at its deposition stretch by definition, so the
fibre family is a constant prestress and the equilibrated configuration follows
from **one nonlinear elastic solve**.

| threshold | AIM residual [mm] | effective pressure | fibre nodal forces | exact mixture |
|---|---|---|---|---|
| 1e-2  | 3.4e-3  | 1.86e-3 | 4.13e-3 | 3.40e-3 |
| 1e-4  | 2.7e-5  | 2.404e-3 | 2.54e-5 | 1.50e-5 |
| 1e-6  | 5.6e-7  | 2.385e-3 | 2.23e-6 | 3.00e-7 |
| 1e-8  | 1.2e-9  | 2.384e-3 | 1.99e-6 | 7.20e-9 |
| 1e-10 | 2.1e-11 | 2.384e-3 | 1.99e-6 | 4.29e-11 |

The exact mixture tracks the AIM residual one for one with no floor. The
manuscript's effective pressure floors at **2.384e-3 mm** from the fifth
iteration onwards — the eight further iterations it reports buy nothing.
Restoring the dropped volume term as nodal forces lowers that floor by a factor
of **1 200**.

## Growth from an imperfect prestress

`scripts/run_growth.py` additionally integrates the transient, which is what
shows the two things that are genuinely statements about the *path*: prestress
error is **amplified** by G&R rather than merely carried (about 3x over three
turnover times), and a badly prestressed mixture is measurably harder to
integrate — nine of ten effective-pressure runs lost stability before the end of
the horizon, against none of the exact ones.

See `docs/growth.md` for the model, the two numerical traps that cost real time,
and the full tables.

---

## Notes on the manuscript

See `docs/corrections.md` for the discrepancies found while reproducing the
cylinder example and how they were resolved here.
