# How accurate does the prestress have to be?

`scripts/run_growth.py` answers the question Gebauer et al. (BMMB 2023) asked for
the generalized prestressing algorithm: a prestressing algorithm is always
stopped at a finite residual, so the prestressed state is not exactly in
mechanobiological equilibrium, and growth & remodelling started from it moves
even though the loading never changes.

## You do not need to time-step G&R to answer this

At mechanobiological equilibrium every turning-over constituent is back at its
deposition stretch *by definition*, so its Cauchy stress per unit mass is the
prescribed homeostatic value — a known constant. The fibre family is therefore
a **constant prestress**, and the equilibrated configuration follows from a
single nonlinear elastic solve: elastin at the prestretch the AIM produced, the
fibre at `sigma_h`, the true in vivo pressure. That is
`prestress.growth.equilibrated_residual`, and it is what
`scripts/run_thresholds.py` uses.

This is better than integrating the transient in four ways:

1. **One solve instead of ~80 time steps**, so the whole threshold ladder runs
   in minutes rather than an hour.
2. **No explicit-update instability.** The checkerboard mode below is a
   consequence of carrying a lagged, per-integration-point state (`lam_n`)
   through a displacement-based element. With no state to lag it cannot arise.
3. **No arbitrary parameters.** The transient needs `k_d`, `K_sigma`, a horizon
   and a step size, and the answer depends on the horizon you pick. The
   equilibrated residual needs none of them.
4. **It is the question actually being asked.** "How far from the image does the
   vessel end up?" is a property of the equilibrium state, not of the path.

The transient is still worth running for the two things that *are* statements
about the path: that prestress error is amplified rather than merely carried,
and that a badly prestressed mixture is harder to integrate. Both are below.

The one approximation is that the mass is held fixed. Letting it adjust as well
needs a further a priori quantity — the homeostatic tissue-level stress that
mediates production, because at equilibrium the constituent-level stimulus
vanishes identically and no longer pins the mass. In the transient runs here
the fibre mass moved by less than `1e-4` in relative terms, so it changes
nothing for the cylinder.

## Setup

The AIM is stopped at a range of residual norms. Each of the resulting
stress-free elastin configurations `X_e` is loaded to the in vivo pressure as a
*full* constrained mixture, and then left to grow and remodel with the loading
held constant. Everything that happens after loading is spurious.

Two ways of handling the embedded fibres during prestressing are compared:

| mode | what the solver sees | black box? |
|---|---|---|
| `effective` | elastin only, loaded by `P_eff = P − σ_ef h / r` (the manuscript) | yes |
| `nodal`     | elastin only, loaded by `P` plus the exact fibre nodal forces | yes |
| `embedded`  | the full mixture at the in vivo pressure | no |

`nodal` restores the term the manuscript drops between Eqs. 6 and 7; see
`docs/corrections.md` §5.

## Constitutive model

Elastin is exactly Eq. 11. The fibre family keeps Eq. 10 but is now allowed to
turn over, following the homogenized constrained mixture model of Cyron, Aydin &
Humphrey (2016) and Braeu et al. (2017):

```
lambda_e   = lambda_fib / lambda_n                 elastic fibre stretch
d(phi_c)/dt = phi_c k_d K_sigma (sigma_c/sigma_h − 1)
d(lam_n)/dt = k_d Upsilon (lambda_fib/G − lam_n),  Upsilon = max(0, 1 + K_sigma (…))
J_g         = (phi_e + phi_c) / (phi_e + phi_c0)
F_g         = M ⊗ M + sqrt(J_g) (I − M ⊗ M)
```

Growth is transversely isotropic and perpendicular to the fibre, so mass is
added across the wall: the wall thickens instead of the lumen widening, which is
what makes the (axisymmetric) response stable. At `lambda_fib = 1` and
`lam_n = 1/G` all rates vanish, so a perfectly prestressed configuration is a
fixed point by construction.

## Implementation notes that cost real time

**FEBio's HEX8 integration points follow the node ordering, not a lexicographic
one.** Any per-integration-point `MAT_POINTS` element map written in the wrong
order is silently permuted. The AIM still converges — it just converges to a
different configuration — but the fibre stretch ends up attached to the wrong
integration point, which here put the fibres ~1 % off their deposition stretch
and made the G&R simulation blow up. `tests/test_basics.py` pins the ordering.

**The mechanobiological stimulus has to be averaged over the element.**
Evaluating `sigma_c/sigma_h` at each integration point separately couples the
pointwise state to the element-internal deformation modes of a HEX8, which a
displacement-based element barely restrains. A checkerboard perturbation of the
fibre stretch then grows exponentially (rate ≈ 90/k_d) even from a configuration
prestressed to 1e-11 mm. Averaging over the element removes that feedback;
`average_stimulus` in the plugin switches it off for comparison.

**A residual checkerboard instability remains at long G&R times.** With the
stimulus averaged the growth rate drops to ≈ 4/k_d, and the mode is still
element-to-element oscillation in the circumferential and axial directions — a
mesh artefact rather than the axisymmetric physical mode, for which a thin-wall
estimate gives `dλ_fib/dλ_n ≈ 0.3`, comfortably inside the stability limit of 1.
The study therefore runs to `t = 4/k_d`, three turnover times after loading and
inside the regime where the behaviour is set by the prestress. That the
instability exists at all is worth reporting: prestress error is *amplified* by
G&R, not merely carried along.

**The bulk penalty.** `K = 1000 c_e` (Table 1) is a stiff numerical penalty:
a mass change enters the pressure as `K ln(J_g)`, so a 0.1 % change in mass is a
260 kPa pressure against a 140 kPa hoop stress. It is solvable once the two
problems above are fixed, but it leaves little margin; `K = 100 c_e` changes the
prestressed geometry by less than 3e-4 mm and is far better conditioned.
`scripts/run_growth.py --bulk-ratio` selects it.

## What the numbers say

> The **floors** — how far an imperfectly prestressed vessel ends up from the
> imaged geometry — are reported by `scripts/run_thresholds.py` and summarised
> in `docs/corrections.md` §5, because they need no transient. What follows is
> what only the transient can show.
>
> These runs were produced with mass turnover active, so the augmented
> Lagrangian is off (it enforces `J -> 1`, the growth constraint is `J -> J_g`).

### Prestress error is amplified, not merely carried

Spurious displacement after three turnover times, against the displacement
right after loading, for the exact (`embedded`) prestressing:

| AIM residual [mm] | after loading | after G&R, `t = 2.5` | ratio |
|---|---|---|---|
| 3.4e-3  | 3.4e-3  | 5.1e-2  | 15 |
| 2.3e-4  | 2.3e-4  | 6.6e-4  | 2.9 |
| 1.5e-5  | 1.5e-5  | 6.0e-5  | 4.0 |
| 3.0e-8  | 3.0e-8  | 9.3e-8  | 3.1 |
| 7.2e-9  | 7.3e-9  | 6.9e-9  | 0.9 |

Across the middle of the range the configuration drifts about 3x further than
the prestress residual by itself would suggest, and 15x at the loosest
tolerance. Whatever prestressing leaves behind, G&R grows.

### A bad prestress makes the solve harder, not just less accurate

| prestressing | runs that lost stability before `t = 4/k_d` |
|---|---|
| effective pressure | **10 of 10** (typically near `t = 2.7`) |
| fibre nodal forces | 2 of 10 (only the two loosest tolerances) |
| embedded fibres | 0 of 10 |

A mixture that starts far from mechanobiological equilibrium is simply harder
to integrate, independently of how wrong the answer is.

The floors after loading in this run were `2.403e-3 mm` (effective pressure) and
`1.995e-6 mm` (nodal forces), matching `scripts/run_thresholds.py` to three
figures — the two studies share no code path beyond the AIM itself.

Figures: `fig6_drift_vs_residual` and `fig7_drift_history` (the equivalent of
Gebauer's Fig. 5), and `output/thresholds/fig2_thresholds` for the floors.
