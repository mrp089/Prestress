# Reproducing the cylinder example: what had to change

Everything below refers to the manuscript draft of 2026-11-23. Numbers quoted as
"ours" come from `scripts/run_cylinder.py` and `scripts/run_growth.py` in this
repository (mouse cylinder, 4032 hexahedra, FEBio 4.13).

---

## 1. The elastin modulus in Table 1 does not match the reported results

Table 1 lists `c_e = 1000 kPa`, cited to Parikh et al. 2023 — a study of the
**human** ascending aorta — while `c1 = 235 kPa` and `c2 = 4.08` come from
Latorre & Humphrey 2020 for the **mouse** descending thoracic aorta, which is
the vessel being modelled.

With `c_e = 1000 kPa` the 0D solution gives

| | 0D with `c_e = 1000` | 0D with `c_e = 262` | reported |
|---|---|---|---|
| `R_i,o` [mm] | 0.321 | 0.299 | 0.29 |
| `H_o` [mm]   | 0.0405 | 0.0420 | 0.042 |
| `L_o` [mm]   | 2.027 | 2.101 | 2.10 |
| `λ_θ^e`      | 1.027 | 1.103 | — |

Inverting the 0D solution for the reported `L_o = 2.10 mm` gives
**`c_e ≈ 262 kPa`**, and that value then reproduces all three reported unloaded
dimensions. The first AIM residual is consistent with the same conclusion: the
manuscript reports `max|U_1| = 0.16 mm`, we obtain 0.109 mm at `c_e = 262 kPa`
and only 0.026 mm at `c_e = 1000 kPa`.

**Either Table 1 or the results need to be corrected.** A value appropriate for
the mouse DTA, taken from the same source as `c1`, `c2`, would be
`μ_e = 89.71 kPa`, i.e. `c_e = μ_e/2 = 44.9 kPa` in the convention of Eq. 11 —
smaller again, and giving a much larger elastin prestretch (`λ_θ^e ≈ 1.56`).
This repository ships both as presets (`paper`, `implied`).

## 2. The reported AIM and analytical results are mutually inconsistent

The reported AIM prestretch `λ_1 = 1.172` corresponds to `c_e ≈ 156 kPa`, while
the reported analytical dimensions correspond to `c_e ≈ 262 kPa`. The quoted
agreement ("5.4 % higher circumferential, 4.4 % lower radial") is therefore not
a measure of discretisation error but of two runs with different parameters.

With a single consistent parameter set, the AIM and the 0D solution agree far
better than that:

| | `λ_1` (circ.) | `λ_2` (axial) | `λ_3` (radial) |
|---|---|---|---|
| AIM vs 0D, `c_e = 1000` | −0.09 % | +0.16 % | −0.07 % |
| AIM vs 0D, `c_e = 262`  | −0.33 % | +0.65 % | −0.31 % |

## 3. Eq. 13 divides by the wrong Jacobian

Eq. 13 reads `σ_dd^α = (2/J_α) λ_d^α ∂ψ_α/∂(λ_d^α)² λ_d^α`. For the fibre
family `J_α = det(G_ef) = λ_dep = 1.1`, which underestimates the fibre stress by
10 % (21.4 instead of 23.6 kPa). In constrained mixture theory the constituent
Cauchy stress is obtained by pushing the constituent 2nd PK stress forward with
the **mixture** deformation gradient, whose Jacobian is 1 in the (incompressible)
in vivo configuration:

```
σ_θθ^ef = 2 I4e ∂ψ_ef/∂I4e = φ_ef c1 I4e (I4e − 1) exp(c2 (I4e − 1)²)
```

## 4. Thin-wall Laplace should use the mid-wall radius

`P_eff = P − σ_θθ^ef h_ivo / r_ivo` and `λ_θ^e = r_ivo / R_i,o` both use the
inner radius. With `h/r = 0.12` the mid-wall radius is the consistent thin-wall
choice; it changes the 0D prestretch by about 0.7 %, which is of the same order
as the agreement being reported. `solve_cylinder_0d(..., radius_convention=...)`
supports both.

## 5. The step from Eq. 6 to Eq. 7 drops a volume term

Eq. 7 replaces `∫ σ_t^ef : δε dv` by a surface term `∫ t^ef · δu ds`. The exact
identity is

```
∫ σ^ef : δε dv = ∮ (σ^ef n)·δu ds − ∫ (∇·σ^ef)·δu dv,
```

so the body force `∇·σ^ef` is dropped. Eq. 3 had introduced exactly that term
and called it a body force, and Eq. 7 then says "which is nothing but" — it is
presented as an identity, which it is not.

The derivation and the implementation also disagree. Taken literally, Sec. 2.2
would apply `t^ef = σ^ef n` on the luminal surface, and for a circumferential
fibre family that is **identically zero** (`σ^ef` is purely `θθ`, `n = ∓e_r`).
Sec. 2.4 instead applies `P_eff = P − σ_θθ^ef h/r`, which is the *resultant* of
the body force Sec. 2.2 discarded, smeared onto the luminal surface. The total
force is right; the distribution through the wall is not.

**This, not the AIM residual, is the accuracy of the method.** Taking the
converged AIM configuration (residual 8.6e-12 mm) and loading it as a full
constrained mixture at the in vivo pressure leaves it **2.4e-3 mm** away from
the imaged geometry — eight orders of magnitude larger than the residual the
algorithm reports, and independent of how far the AIM is iterated.

### The term can be put back exactly, without giving up the black box

The fibre stresses are prescribed a priori, so `∫ σ^ef : δε dv` is *known*. It
can be assembled once on the imaged mesh into a fixed nodal force vector

```
f_a = − ∫ σ^ef · grad(N_a) dv,
```

applied together with the true pressure `P`. The solver still only sees an
isotropic elastin model, a pressure and nodal loads — no internal variables and
no user material — so the black-box argument of the paper survives intact. At
convergence the deformed configuration is the in vivo one, so the forces are
evaluated exactly where they belong and nothing is approximated.

Measured against `mode="embedded"`, which carries the fibre in the material and
solves the exact mixture equilibrium, as a function of the volumetric penalty:

| `K / c_e` | effective pressure (Eq. 9) | fibre nodal forces |
|---|---|---|
| 100    | 2.970e-3 mm | 1.837e-4 mm |
| 1 000  | 2.820e-3 mm | 1.712e-5 mm |
| 10 000 | 2.805e-3 mm | 1.579e-6 mm |

The nodal-force error falls by a factor of 10.7 and then 10.8 per decade of `K`,
i.e. it is `O(mu/K)` and converges to zero. The effective-pressure error instead
converges to a constant `2.805e-3 mm` — a modelling error, not a numerical one.

### Where the remaining O(mu/K) term comes from, and how to remove it

Split the prescribed fibre stress into its deviatoric and hydrostatic parts:

```
∫ σ_ef : δε dv = ∫ dev(σ_ef) : δε dv + ∫ (tr σ_ef / 3) tr(δε) dv,
```

and `tr(δε) = δ(ln J)`. For an exactly incompressible solid `δ(ln J) = 0`, so
the hydrostatic part does **no virtual work**: whether it is assembled into the
nodal forces, left to the material, or absorbed by the pressure multiplier makes
no difference. With a finite bulk penalty it does a little work, proportional to
`1/K`, and that is the whole discrepancy. Two independent fixes, at
`K = 1000 c_e`:

| | nodal vs. exact mixture |
|---|---|
| penalty, full `σ_ef` | 1.7118e-5 mm |
| penalty, `dev(σ_ef)` only (`deviatoric_forces=True`) | 1.9723e-6 mm |
| augmented Lagrangian, full `σ_ef` (`laugon=True`) | 1.9719e-6 mm |

The two agree to 0.02 % even though the mechanisms are unrelated — one changes
what is assembled, the other how the constraint is enforced — which is what
confirms they remove the same term.

What is left, `1.97e-6 mm`, is **not** an incompressibility effect: it does not
move with `K` (1.976e-6, 1.972e-6, 1.972e-6 at `K/c_e` = 100, 1 000, 10 000).
It is the fibre direction. The reference implementation defines the fibre in the
*stress-free* configuration and pushes it forward, and on a faceted mesh
(48 facets here) that push-forward tilts it by up to 0.35 degrees relative to
`e_theta` in the imaged configuration; the two conventions give nodal forces
that differ by `5.6e-4` in relative terms, which is the right size for the
`2e-6 mm` observed. Anchoring the fibre direction in the imaged configuration,
as the nodal-force assembly does, is the defensible choice — that is where the
orientation is measured.

### Error budget for the cylinder

| source | size | removed by |
|---|---|---|
| effective-traction substitution (Eq. 7) | **2.4e-3 mm** | nodal forces |
| volumetric penalty at `K = 1000 c_e` | 9.7e-5 mm | `laugon` |
| hydrostatic split of the prescribed stress | 1.7e-5 mm | `deviatoric_forces` or `laugon` |
| fibre-direction reference configuration | 2.0e-6 mm | anchor it in the image |
| AIM iteration residual | 1e-11 mm | — |

The manuscript reports only the last line, which is the smallest term by eight
orders of magnitude.

With the accurate settings now used by default (`deviatoric_forces=True`,
`laugon=True`) the two black-box methods floor at

| | equilibrated residual displacement |
|---|---|
| effective pressure (Eq. 9) | 2.384e-3 mm |
| fibre nodal forces | 1.994e-6 mm |

a factor of **1 200**, while the exact mixture keeps tracking the AIM residual
one for one with no floor at all.

Two further advantages worth stating in the paper: the nodal-force version needs
no thin-wall argument and no assumption about the fibre architecture, so it
carries over unchanged to the patient-specific aorta and to any number of fibre
families; and it is easier to justify, being nothing but the weak form.

`run_aim(..., mode="nodal")` implements it; `mode="embedded"` is the reference.

## 6. The "numerical floor" is a property of the output, not of the method

The reported floor of `3.7e-9 mm` is set by solver tolerances and output
precision. With FEBio's 12-significant-digit log output and Newton tolerances of
1e-16 the floor here is `~1e-11 mm`. Reporting it as a result of the method is
misleading; it should be reported as the precision of the setup.

Separately, the stagnation criterion (two consecutive `c_k < ε_c = 0.1`) is
wasteful near the floor: the residual reaches 1e-11 mm after 8 iterations but
convergence is only declared at iteration 22, because `c_k` keeps bouncing
around 0.2 on output-precision noise. An additional absolute-stagnation test
(`r_k` no longer decreasing) would stop it immediately.

## 7. `λ = 1.1` is cited to the wrong quantity

The deposition stretch is cited to Bellini et al. 2014, where the collagen
deposition stretch is `G_c = 1.25` and the smooth-muscle one `G_m = 1.20`; 1.1
is `λ_M`, the active smooth-muscle stretch at peak contraction.

Using `G_c = 1.25` for the reduced one-fibre mixture is not possible: the single
circumferential family at `φ_ef = 0.33` then carries `σ_θθ^ef = 248 kPa`, i.e.
more than the entire transmural load, giving `P_eff = −14 kPa` and elastin in
compression. The reduced mixture only works with the smaller stretch, which is
worth stating as a modelling choice rather than a citation.

## 8. Details the manuscript does not state

* The subdivision of the 4032 hexahedra. Here: 48 circumferential × 3 radial
  × 28 axial.
* The boundary conditions of the cylinder. Here: `u_z = 0` on one end face and
  axis-aligned symmetry constraints on the `x`- and `y`-axis node lines, which
  allow free radial expansion and leave the axial force zero — consistent with
  the analytical assumptions and with the reported `λ_z ≠ 1`.
* That the luminal pressure is a follower load. It has to be, otherwise the
  converged configuration is not loaded by the in vivo traction.

---

## Checklist

| # | issue | what to do |
|---|---|---|
| 1 | `c_e = 1000 kPa` in Table 1 is inconsistent with the reported results | re-run with one value and state its source; the reported numbers correspond to ≈ 262 kPa |
| 2 | reported AIM and analytical results use different effective moduli | re-run both from the same Table 1 |
| 3 | `1/J_alpha` in Eq. 13 | drop it: the push-forward uses the mixture Jacobian |
| 4 | inner radius in the Laplace relation | use the mid-wall radius, or say that the inner radius is used |
| 5 | Eq. 6 → Eq. 7 drops `∫ (∇·σ^ef)·δu dv` | put it back as nodal forces `f_a = −∫σ^ef·grad(N_a) dv` — exact, still black box, 1200x smaller error, and it generalises to the aorta |
| 6 | "numerical floor" reported as a property of the method | report it as the precision of the solver and the output |
| 6b | stagnation criterion wastes ~15 iterations at the floor | add an absolute test on `r_k` |
| 7 | `λ = 1.1` cited to Bellini et al. | that reference gives `G_c = 1.25`; state 1.1 as a choice for the reduced mixture |
| 8 | mesh subdivision, boundary conditions, follower pressure not stated | add one sentence each |
