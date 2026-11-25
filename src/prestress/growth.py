"""Growth & remodelling started from an imperfectly prestressed configuration.

After Gebauer et al. (BMMB 2023): a prestressing algorithm is only ever run to
a finite tolerance, so the prestressed state is not exactly in mechanobiological
equilibrium.  Running growth & remodelling with unchanged loading then produces
a small, spurious adaptation.  Here that adaptation is measured as a function of
the residual at which the AIM was stopped.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import warnings

import numpy as np

from .febio import (read_node_log, read_node_log_all, run_febio,
                    write_cmm_feb, write_elastin_feb)
from .kinematics import prestretch_maps
from .mesh import CylinderMesh
from .params import MaterialParams

__all__ = ["GrowthResult", "run_growth", "equilibrated_residual"]


@dataclass
class GrowthResult:
    """Trajectory of a G&R run started from a prestressed configuration."""

    times: np.ndarray            # (n_step,)
    drift: np.ndarray            # max |x(t) - x_ivo| [mm]
    drift_rms: np.ndarray        # RMS nodal distance from x_ivo [mm]
    r_inner: np.ndarray          # inner radius [mm]
    thickness: np.ndarray        # wall thickness [mm]
    length: np.ndarray           # length [mm]
    t_gr: float
    prestress_residual: float    # AIM residual of the configuration used
    completed: bool = True       # False if the solver stopped early

    @property
    def drift_loaded(self) -> float:
        """Residual displacement right after loading, before any G&R."""
        i = int(np.argmin(np.abs(self.times - self.t_gr)))
        return float(self.drift[i])

    @property
    def drift_final(self) -> float:
        return float(self.drift[-1])

    def drift_at(self, t: float) -> float:
        """Spurious displacement at (the output step closest to) time ``t``."""
        return float(self.drift[int(np.argmin(np.abs(self.times - t)))])


def run_growth(
    mesh: CylinderMesh,
    p: MaterialParams,
    X: np.ndarray,
    workdir: str | Path,
    prestress_residual: float = np.nan,
    k_d: float = 1.0,
    K_sigma: float = 1.0,
    t_gr: float = 1.0,
    t_end: float = 11.0,
    n_steps: int = 20,
    gr_step: float = 0.05,
    plot: bool = False,
    timeout: float = 900.0,
) -> GrowthResult:
    """Load the prestressed mesh ``X`` to the in vivo pressure and let it adapt.

    ``X`` is a stress-free elastin configuration produced by the AIM.  The
    in vivo pressure is reached at ``t_gr``; growth & remodelling then run until
    ``t_end`` with the loading held constant, so any further motion is spurious.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    x = mesh.nodes
    lam_ref, J_ref = prestretch_maps(mesh, X, x)

    feb = write_cmm_feb(
        workdir / "growth.feb", mesh, p, p.pressure, X,
        lam_ref=lam_ref, J_ref=J_ref, phi_c=p.phi_ef,
        k_d=k_d, K_sigma=K_sigma, t_gr=t_gr, t_end=t_end,
        n_steps=n_steps, gr_step=gr_step, plot=plot,
        # the augmentation enforces J -> 1, which is incompatible with the
        # growth constraint J -> J_g, so it is only available without turnover
        laugon=(k_d <= 0.0),
    )
    completed = True
    try:
        run_febio(feb, plugin=True, timeout=timeout)
    except (RuntimeError, subprocess.TimeoutExpired):
        # The G&R problem can lose stability (see docs/corrections.md); FEBio
        # then stops, but everything written up to that point is still valid.
        completed = False
    xs, ts = read_node_log_all(workdir / "growth_pos.txt", mesh.n_node)
    if len(xs) == 0:
        raise RuntimeError(f"no output written for {feb}")

    d = xs - x[None]
    rad = np.linalg.norm(xs[..., :2], axis=-1)
    return GrowthResult(
        times=ts,
        drift=np.abs(d).max(axis=(1, 2)),
        drift_rms=np.sqrt((np.linalg.norm(d, axis=-1) ** 2).mean(axis=1)),
        r_inner=rad.min(axis=1),
        thickness=rad.max(axis=1) - rad.min(axis=1),
        length=xs[..., 2].max(axis=1) - xs[..., 2].min(axis=1),
        t_gr=t_gr,
        prestress_residual=prestress_residual,
        completed=completed,
    )


def equilibrated_residual(
    mesh: CylinderMesh,
    p: MaterialParams,
    X: np.ndarray,
    workdir: str | Path,
    n_steps: int = 5,
) -> tuple[float, np.ndarray]:
    """Residual displacement of the mechanobiologically equilibrated state.

    No time stepping is needed to answer "how far does an imperfectly
    prestressed vessel sit from the imaged geometry".  At mechanobiological
    equilibrium every turning-over constituent is back at its deposition
    stretch by definition, so its Cauchy stress is the prescribed homeostatic
    one and the fibre family acts as a *constant* prestress.  What remains is a
    single nonlinear elastic solve: elastin at the prestretch the AIM produced,
    the fibre at ``sigma_h``, the true in vivo pressure.

    This is the equilibrated constrained mixture of Latorre & Humphrey (2018)
    with the mass held fixed.  Letting the mass adjust as well needs one further
    a priori quantity -- the homeostatic tissue-level stress that mediates
    production -- because at equilibrium the constituent-level stimulus vanishes
    identically and no longer pins the mass.  It adds a first-order correction
    whose gain depends on that stimulus; in the transient runs here the fibre
    mass moved by less than 1e-4 in relative terms, so it is negligible for the
    cylinder, but that is an observation rather than a general result.

    Returns ``(max |x - x_ivo|, x)``.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    lam_ref, J_ref = prestretch_maps(mesh, X, mesh.nodes)
    feb = write_cmm_feb(
        workdir / "equil.feb", mesh, p, p.pressure, X,
        lam_ref=lam_ref, J_ref=J_ref, phi_c=p.phi_ef, k_d=0.0,
        n_steps=n_steps,
    )
    run_febio(feb, plugin=True)
    x = read_node_log(workdir / "equil_pos.txt", mesh.n_node)
    return float(np.abs(x - mesh.nodes).max()), x


def equilibrated_forces_residual(
    model, p, X: np.ndarray, x_target: np.ndarray, fibers, workdir,
    sigma: float | "Sequence[float]" | None = None,
    n_steps: int = 10, max_iter: int = 12,
    tol: float = 1e-6, verbose: bool = True, **feb_kw,
) -> tuple[float, np.ndarray, list[float]]:
    """Equilibrated constrained mixture, solved as a black box on any mesh.

    At mechanobiological equilibrium every turning-over constituent is back at
    its deposition stretch *by definition*, so each fibre family carries the
    prescribed homeostatic Cauchy stress ``sigma_h`` along its own direction.
    No time stepping and no mass-production parameters are needed -- see
    ``docs/growth.md``.

    The difference from the AIM's ``nodal`` variant is that here the fibre
    stress is a **follower** load rather than a dead one.  The AIM may compute
    its nodal forces once, on the imaged geometry, because at its own fixed
    point the configuration *is* the imaged geometry.  The equilibrated state
    is a different question -- where does this prestressed wall actually come
    to rest? -- and the answer has the fibres convected and the forces
    integrated over the deformed volume.  So the forces are recomputed on the
    current configuration and the solve repeated until they stop moving it.

    This needs no plugin and no constrained-mixture material: the solver still
    sees only an isotropic solid, a follower pressure and nodal loads.  The
    one approximation is that the mass is held fixed, which is the same one
    :func:`equilibrated_residual` makes.

    ``fibers`` is a sequence of (n_node, 3) direction fields, one per family.
    Returns ``(max |x - x_target|, x, history)``.
    """
    from .kinematics import (convect_directions, deformation_gradients,
                             nodal_forces_from_stress, stress_from_gp_directions)
    from .material import fiber_cauchy_stress

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    sig = fiber_cauchy_stress(p) if sigma is None else sigma
    # one homeostatic stress per family: the myocardium of Gebauer et al.
    # (2023) mixes myocytes and collagen, which sit at different deposition
    # stretches and so carry different stresses
    sigs = ([float(sig)] * len(fibers) if np.ndim(sig) == 0
            else [float(v) for v in sig])
    if len(sigs) != len(fibers):
        raise ValueError(f"{len(sigs)} homeostatic stresses for "
                         f"{len(fibers)} fibre families")

    x = np.asarray(X, dtype=float).copy()
    history: list[float] = []
    for it in range(1, max_iter + 1):
        F = deformation_gradients(X, x, model.elements, model.elem_type)
        S = stress_from_gp_directions(
            [(sg, convect_directions(f, F, model.elements, model.elem_type))
             for sg, f in zip(sigs, fibers)])
        forces = nodal_forces_from_stress(x, model.elements, model.elem_type, S)

        feb = write_elastin_feb(workdir / f"equil_{it:03d}.feb", model, p,
                                p.pressure, coords=X, nodal_forces=forces,
                                n_steps=n_steps, **feb_kw)
        run_febio(feb)
        x_new = read_node_log(workdir / f"equil_{it:03d}_pos.txt", model.n_node)

        move = float(np.abs(x_new - x).max())
        x = x_new
        r = float(np.abs(x - x_target).max())
        history.append(r)
        if verbose:
            print(f"    equil {it:3d}:  moved {move:.4e} mm   "
                  f"residual {r:.6e} mm")
        if move < tol:
            break
    else:
        # Stopping here is not a converged equilibrium, and the residual it
        # returns carries whatever the fixed point had left to move.  That
        # would show up in a threshold ladder as a floor, and be read as a
        # property of the method rather than of this loop -- so say so.
        warnings.warn(
            f"equilibration stopped at the cap of {max_iter} iterations with "
            f"the configuration still moving {move:.3e} mm per step "
            f"(tol {tol:.1e}); the returned residual {r:.3e} mm is an upper "
            f"bound, not an equilibrium",
            RuntimeWarning, stacklevel=2)
    return float(np.abs(x - x_target).max()), x, history
