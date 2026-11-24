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

__all__ = ["AIMResult", "run_aim", "make_solver"]


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
) -> AIMResult:
    """Run Algorithm 1 and return the stress-free elastin configuration."""
    x_target = mesh.nodes.copy()
    X = x_target.copy()          # X_1 <- x
    beta = beta0
    res: list[float] = []
    chg: list[float] = []
    betas: list[float] = []
    configs: list[np.ndarray] = []
    disps: list[np.ndarray] = []
    U_prev = None
    converged = False

    for k in range(1, max_iter + 1):
        x_k = solve(X, k)

        U = x_k - x_target
        r = float(np.abs(U).max())
        configs.append(X.copy())
        disps.append(U.copy())
        res.append(r)
        betas.append(beta)

        c = abs(r - res[-2]) / res[-2] if k > 1 else np.nan
        chg.append(c)
        if k > 1:
            dU = U - U_prev
            denom = float(np.sum(dU * dU))
            if denom > 0:
                beta = -beta * float(np.sum(U_prev * dU)) / denom

        if verbose:
            print(f"  AIM {k:3d}:  max|U| = {r:.6e} mm   c = {c:.3e}"
                  f"   beta = {betas[-1]:+.4f}")

        if k > 2 and chg[-2] < eps_c and chg[-1] < eps_c:
            converged = True
            break

        U_prev = U
        X = X - beta * U

    return AIMResult(
        X=configs[-1], x_target=x_target, iterations=len(res),
        residual=res, change=chg, beta=betas,
        configs=configs, displacements=disps, converged=converged,
    )
