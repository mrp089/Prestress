"""Augmented iterative method (AIM) adapted to constrained mixture models.

Algorithm 1 of the manuscript, after Rausch, Genet & Humphrey (J Biomech 2017).

Two ways of accounting for the embedded fibres are provided:

``"effective"`` (the method of the manuscript)
    The finite element solver is a black box that only ever sees an *isotropic
    elastin* model.  The prescribed, homeostatic fibre stress is converted into
    an effective pressure ``P_eff = P - sigma_ef h / r`` (Sec. 2.4).

``"nodal"`` (exact, and still a black box)
    The term the manuscript drops between Eqs. 6 and 7 is put back. The fibre
    contribution to the weak form, ``-int sigma_ef . grad(N_a) dv``, is known a
    priori and is handed to the solver as a fixed nodal force vector. The solver
    still only sees an isotropic elastin model plus loads.

``"embedded"`` (reference solution)
    The fibre family is carried by the material itself at its prescribed
    deposition stretch, and the full in vivo pressure is applied.  This is not a
    black-box method, but it solves the *exact* mixture equilibrium and is the
    yardstick the other two are measured against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .febio import read_node_log, run_febio, write_cmm_feb, write_elastin_feb
from .kinematics import fiber_nodal_forces, prestretch_maps
from .mesh import CylinderMesh
from .params import MaterialParams

__all__ = ["AIMResult", "run_aim", "make_solver", "make_force_solver"]


@dataclass
class AIMResult:
    """History and outcome of the AIM iterations."""

    X: np.ndarray                 # stress-free elastin configuration at convergence
    x_target: np.ndarray          # in vivo configuration
    iterations: int
    residual: list[float] = field(default_factory=list)   # r_k = max |U_k|
    change: list[float] = field(default_factory=list)     # c_k
    beta: list[float] = field(default_factory=list)
    configs: list[np.ndarray] = field(default_factory=list)
    displacements: list[np.ndarray] = field(default_factory=list)
    converged: bool = True

    @property
    def final_residual(self) -> float:
        return self.residual[-1]

    def config_at_residual(self, r_max: float):
        """First iterate whose residual is at or below ``r_max``."""
        for k, r in enumerate(self.residual):
            if r <= r_max:
                return k, self.configs[k], r
        raise ValueError(f"residual never reached {r_max:g}")


def make_solver(
    mesh: CylinderMesh,
    p: MaterialParams,
    pressure: float,
    workdir: str | Path,
    mode: str = "effective",
    n_steps: int = 5,
    deviatoric_forces: bool = True,
    **feb_kw,
) -> Callable[[np.ndarray, int], np.ndarray]:
    """Return ``S(X, k) -> x``, the black-box finite element solve.

    The defaults are the most accurate settings available: the incompressibility
    constraint is enforced through an augmented Lagrangian rather than the bulk
    penalty alone, and, in ``"nodal"`` mode, only the deviatoric part of the
    prescribed fibre stress is assembled -- matching the split FEBio's uncoupled
    materials make internally.  Together these remove the ``O(mu/K)`` error; see
    ``docs/corrections.md`` Sec. 5.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    # the fibre forces are known a priori and are the same in every iteration
    forces = (fiber_nodal_forces(mesh, p, deviatoric=deviatoric_forces)
              if mode == "nodal" else None)

    def solve(X: np.ndarray, k: int) -> np.ndarray:
        stem = f"aim_{k:03d}"
        if mode == "effective":
            feb = write_elastin_feb(workdir / f"{stem}.feb", mesh, p, pressure,
                                    coords=X, n_steps=n_steps, **feb_kw)
            run_febio(feb)
        elif mode == "nodal":
            feb = write_elastin_feb(workdir / f"{stem}.feb", mesh, p, pressure,
                                    coords=X, nodal_forces=forces,
                                    n_steps=n_steps, **feb_kw)
            run_febio(feb)
        elif mode == "embedded":
            lam_ref, J_ref = prestretch_maps(mesh, X, mesh.nodes)
            feb = write_cmm_feb(workdir / f"{stem}.feb", mesh, p, pressure, X,
                                lam_ref=lam_ref, J_ref=J_ref, phi_c=p.phi_ef,
                                k_d=0.0, n_steps=n_steps, **feb_kw)
            run_febio(feb, plugin=True)
        else:
            raise ValueError(f"unknown AIM mode {mode!r}")
        return read_node_log(workdir / f"{stem}_pos.txt", mesh.n_node)

    return solve


def run_aim(
    mesh: CylinderMesh,
    solve: Callable[[np.ndarray, int], np.ndarray],
    beta0: float = 0.5,
    eps_c: float = 0.1,
    max_iter: int = 60,
    verbose: bool = True,
    beta_min: float = 0.05,
    beta_max: float = 2.0,
    max_backtrack: int = 4,
    tol: float | None = None,
    anderson: int = 0,
) -> AIMResult:
    """Run Algorithm 1 and return the stress-free elastin configuration.

    Aitken chooses the step size from the last two residuals, which assumes
    the map is close to linear.  On a patient-specific wall it is not: the
    estimate ran up to ``beta = 3.7`` and the extrapolated configuration was
    one FEBio could not load at all, which ended the run after six otherwise
    healthy iterations.  Two safeguards, neither of which touches the fixed
    point -- only the path to it, and the cylinder never reaches either:

    ``beta_max`` caps the extrapolation.  ``max_backtrack`` makes a failed
    trial configuration recoverable: the step that produced it is halved and
    the solve retried, the way a line search would.  If the step cannot be
    rescued the iteration stops and returns the best configuration so far
    rather than raising, so a run that converged usefully before stalling
    still yields a result.

    ``tol`` is an absolute target on ``max|U|``.  Algorithm 1 as written stops
    on ``eps_c``, the *relative change* between successive residuals, which
    declares success as soon as progress becomes slow however large the
    residual still is.  On the cylinder the two agree, because convergence is
    fast and the residual falls off a cliff.  On the aorta they do not: the
    relative criterion fires at ``max|U| = 2.6e-1 mm``, a quarter of a
    millimetre of error in the prestressed geometry, simply because the
    contraction rate is about 0.93 per iteration.  With ``tol`` set, the
    relative criterion is demoted to a stall detector and reported as such
    instead of being called convergence.

    ``anderson`` turns on Anderson acceleration with that many stored
    residuals (0 = off, which is Algorithm 1 as written).  Aitken picks a
    single scalar step for the whole mesh from the last two residuals.  That
    is right when the map contracts at about the same rate everywhere, which
    is the case on the cylinder -- it reaches 8.6e-12 mm in 11 iterations.
    It is not the case on an aorta: the residual tail sits where the wall is
    most compliant, far from any clamp and in elements no worse than average,
    and the map contracts there at about 0.95 per iteration while the median
    node is already converged to 4 um.  No single scalar serves both, and the
    Aitken estimate oscillates between 1 and 4 instead of settling.  Anderson
    builds the update from several past residual *vectors*, so it can take
    different effective steps in different parts of the mesh.
    """
    x_target = mesh.nodes.copy()
    X = x_target.copy()          # X_1 <- x
    beta = beta0
    res: list[float] = []
    chg: list[float] = []
    betas: list[float] = []
    configs: list[np.ndarray] = []
    disps: list[np.ndarray] = []
    U_prev = None
    X_prev = None
    step = beta0
    converged = False
    stalled = False
    hist_X: list[np.ndarray] = []
    hist_U: list[np.ndarray] = []

    for k in range(1, max_iter + 1):
        for attempt in range(max_backtrack + 1):
            try:
                x_k = solve(X, k)
                break
            except RuntimeError:
                if X_prev is None or attempt == max_backtrack:
                    if X_prev is None:
                        raise
                    stalled = True
                    break
                step *= 0.5
                X = X_prev - step * U_prev
                if verbose:
                    print(f"       solve failed, backtracking: "
                          f"step -> {step:.4f}")
        if stalled:
            if verbose:
                print(f"  AIM {k:3d}:  no solvable step; stopping at "
                      f"max|U| = {res[-1]:.6e} mm")
            break

        U = x_k - x_target
        r = float(np.abs(U).max())
        configs.append(X.copy())
        disps.append(U.copy())
        res.append(r)
        betas.append(beta)

        c = abs(r - res[-2]) / res[-2] if k > 1 else np.nan
        chg.append(c)
        if anderson:
            hist_X.append(X.copy())
            hist_U.append(U.copy())
            if len(hist_X) > anderson + 1:
                hist_X.pop(0); hist_U.pop(0)

        if k > 1:
            dU = U - U_prev
            denom = float(np.sum(dU * dU))
            if denom > 0:
                # the Aitken update uses the relaxation actually applied,
                # which is `step` and not `beta` once either safeguard bites
                beta = -step * float(np.sum(U_prev * dU)) / denom
            beta = float(np.clip(beta, beta_min, beta_max))

        if verbose:
            print(f"  AIM {k:3d}:  max|U| = {r:.6e} mm   c = {c:.3e}"
                  f"   beta = {betas[-1]:+.4f}")

        if tol is not None and r < tol:
            converged = True
            break
        if k > 2 and chg[-2] < eps_c and chg[-1] < eps_c:
            if tol is None:
                converged = True       # Algorithm 1 as written
            else:
                stalled = True
                if verbose:
                    print(f"       residual change below eps_c = {eps_c:g} "
                          f"at max|U| = {r:.3e} mm, still above tol = "
                          f"{tol:g}: stalled, not converged")
            break

        U_prev = U
        X_prev = X.copy()
        step = beta

        if anderson and len(hist_U) > 1:
            # least-squares mix of the stored iterates that minimises the
            # predicted residual, then one relaxed fixed-point step from it
            dU = np.column_stack([(hist_U[i + 1] - hist_U[i]).ravel()
                                  for i in range(len(hist_U) - 1)])
            dX = np.column_stack([(hist_X[i + 1] - hist_X[i]).ravel()
                                  for i in range(len(hist_X) - 1)])
            gamma, *_ = np.linalg.lstsq(dU, U.ravel(), rcond=None)
            X = X - (dX @ gamma).reshape(X.shape) - step * (
                U - (dU @ gamma).reshape(U.shape))
        else:
            X = X - step * U

    return AIMResult(
        X=configs[-1], x_target=x_target, iterations=len(res),
        residual=res, change=chg, beta=betas,
        configs=configs, displacements=disps,
        converged=converged and not stalled,
    )


def make_force_solver(
    model,
    p: MaterialParams,
    pressure: float,
    workdir: str | Path,
    forces: np.ndarray | None = None,
    n_steps: int = 5,
    **feb_kw,
):
    """``S(X, k) -> x`` for any mesh, with the fibre contribution as nodal loads.

    All three ways of accounting for the fibres reduce to the same thing once
    assembled -- a follower pressure plus a fixed nodal force vector -- so the
    aorta variants differ only in how ``forces`` was built.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    def solve(X: np.ndarray, k: int) -> np.ndarray:
        stem = f"aim_{k:03d}"
        feb = write_elastin_feb(workdir / f"{stem}.feb", model, p, pressure,
                                coords=X, nodal_forces=forces,
                                n_steps=n_steps, **feb_kw)
        run_febio(feb)
        return read_node_log(workdir / f"{stem}_pos.txt", model.n_node)

    return solve
