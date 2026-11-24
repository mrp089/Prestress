"""Build the four-chamber heart model of Pfaller et al. (2019) for prestressing.

The mesh ``eva.vtu`` is the tet10 heart of Pfaller et al., BMMB 18 (2019),
exported from its BACI input deck.  Everything the prestressing needs is
already on it:

* ``node-fiber1`` / ``node-fiber2``  myocardial fibre and sheet directions
* ``dsurfN``                         0/1 node masks of the BACI design surfaces

The surfaces that matter (see ``docs/heart.md``):

===========  ======  =========================================================
chamber      dsurf   boundary condition
===========  ======  =========================================================
LV endo       2      follower pressure, end-diastolic
RV endo       5      follower pressure, end-diastolic
LA endo      44      follower pressure, end-diastolic
RA endo      59      follower pressure, end-diastolic
great vessels 20     spring to ground, 1e3
epicardium    8      spring along the reference normal, 2.0
base         19      Dirichlet
===========  ======  =========================================================

VTK and FEBio order the ten tet nodes identically -- corners 0-3, then the
edges (0,1) (1,2) (2,0) (0,3) (1,3) (2,3) -- so the connectivity is passed
through unchanged.  ``tests/test_basics.py`` pins that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["HeartMesh", "load_heart", "CHAMBERS"]

#: endocardial surfaces that carry a follower pressure
CHAMBERS = {"LV": "dsurf2", "RV": "dsurf5", "LA": "dsurf44", "RA": "dsurf59"}

#: the four faces of a tet10, as local node indices, in FEBio's tri6 order
#: (three corners, then the mid-side nodes of (0,1), (1,2), (2,0) of the face)
_TET10_FACES = np.array([
    [0, 1, 3, 4, 8, 7],
    [1, 2, 3, 5, 9, 8],
    [2, 0, 3, 6, 7, 9],
    [2, 1, 0, 5, 4, 6],
])


@dataclass
class HeartMesh:
    nodes: np.ndarray                 # (n_node, 3) [mm]
    elements: np.ndarray              # (n_elem, 10) tet10, 0-based
    fiber1: np.ndarray                # (n_node, 3)
    fiber2: np.ndarray                # (n_node, 3)
    surfaces: dict[str, np.ndarray]   # name -> (n_facet, 6) tri6
    nodesets: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_node(self) -> int:
        return len(self.nodes)


def _boundary_faces(elements: np.ndarray) -> np.ndarray:
    """The tri6 faces of the tet10 mesh that belong to exactly one element."""
    faces = elements[:, _TET10_FACES].reshape(-1, 6)        # (4 n_elem, 6)
    key = np.sort(faces[:, :3], axis=1)                     # corners identify a face
    _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    return faces[idx[cnt == 1]]


def load_heart(path: str, scale: float = 1.0,
               surfaces: dict[str, str] | None = None) -> HeartMesh:
    """Read ``eva.vtu`` and pull out the tet10 mesh, fibres and tri6 surfaces.

    A facet belongs to a named surface when *all six* of its nodes carry that
    surface's mask.  Using all six rather than the three corners keeps facets
    that merely touch the rim of a neighbouring surface out of the set.
    """
    import pyvista as pv

    surfaces = surfaces or dict(CHAMBERS)
    grid = pv.read(path)

    tet = grid.celltypes == 24                              # VTK_QUADRATIC_TETRA
    if not tet.any():
        raise ValueError(f"{path} holds no tet10 cells")
    cells = grid.cells_dict[24].astype(np.int64)

    nodes = np.asarray(grid.points, dtype=float) * scale
    f1 = np.asarray(grid.point_data["node-fiber1"], dtype=float)
    f2 = np.asarray(grid.point_data["node-fiber2"], dtype=float)
    f1 /= np.linalg.norm(f1, axis=1, keepdims=True).clip(1e-300)
    f2 /= np.linalg.norm(f2, axis=1, keepdims=True).clip(1e-300)

    bnd = _boundary_faces(cells)
    out, sets = {}, {}
    for name, key in surfaces.items():
        mask = np.asarray(grid.point_data[key]) > 0.5
        sel = bnd[mask[bnd].all(axis=1)]
        if len(sel) == 0:
            raise ValueError(f"surface {name!r} ({key}) has no boundary facets")
        out[name] = sel
        sets[name] = np.unique(sel)
    return HeartMesh(nodes=nodes, elements=cells, fiber1=f1, fiber2=f2,
                     surfaces=out, nodesets=sets)


#: end-diastolic chamber pressures [kPa].  The ventricular values are the ones
#: used by Pfaller et al. (2019); the atrial ones are a modelling choice --
#: physiological mean atrial pressures are 8-12 mmHg (LA) and 2-6 mmHg (RA).
ED_PRESSURE_MMHG = {"LV": 10.0, "RV": 10.0, "LA": 10.0, "RA": 5.0}

MMHG_TO_KPA = 0.133322387415


def heart_model(h: HeartMesh, pressures_mmhg: dict[str, float] | None = None,
                k_vessels: float = 1e3, k_epi: float = 2.0):
    """An :class:`~prestress.febio.FEModel` for the four-chamber heart.

    The LV is the primary pressurised surface (``facets``); the other three
    chambers and the two elastically supported surfaces ride along as named
    surfaces.  Boundary conditions follow the BACI deck of Pfaller et al.
    (2019): a spring to ground at the great vessels, a spring along the
    reference normal over the epicardium, and a clamped base.
    """
    from .febio import FEModel

    pmm = dict(ED_PRESSURE_MMHG if pressures_mmhg is None else pressures_mmhg)
    missing = set(CHAMBERS) - set(h.surfaces)
    if missing:
        raise ValueError(f"mesh is missing the chamber surfaces {sorted(missing)}")

    extra = {k: v for k, v in h.surfaces.items() if k != "LV"}
    return FEModel(
        nodes=h.nodes, elements=h.elements, facets=h.surfaces["LV"],
        elem_type="tet10", facet_type="tri6",
        nodesets={"base": h.nodesets["base"]},
        bcs=(("clamp", "base", (1, 1, 1)),),
        surfaces=extra,
        pressures=tuple((c, pmm[c] * MMHG_TO_KPA)
                        for c in ("RV", "LA", "RA") if c in extra),
        robin=(("vessels", "vessels", k_vessels, k_vessels),
               ("pericardium", "epi", k_epi, 0.0)),
    ), pmm["LV"] * MMHG_TO_KPA
