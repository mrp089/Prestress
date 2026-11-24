"""Minimal legacy-VTK writer, so the results can be opened in ParaView.

Only what is needed here: an unstructured grid of HEX8 cells with point and
cell data.  No dependencies beyond NumPy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["write_vtk"]

_VTK_HEXAHEDRON = 12

#: VTK cell type by nodes per element -- tet4, wedge (penta6), hex8, tet10.
#: VTK and FEBio order the nodes identically for all four.
_VTK_CELL_TYPE = {4: 10, 6: 13, 8: 12, 10: 24}


def _write_arrays(fh, data: dict, n: int, kind: str) -> None:
    if not data:
        return
    fh.write(f"{kind} {n}\n")
    for name, arr in data.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            fh.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
            np.savetxt(fh, arr, fmt="%.9g")
        elif arr.shape[1] == 3:
            fh.write(f"VECTORS {name} double\n")
            np.savetxt(fh, arr, fmt="%.9g")
        else:
            raise ValueError(f"cannot write array {name!r} of shape {arr.shape}")


def write_vtk(
    path: str | Path,
    nodes: np.ndarray,
    hexes: np.ndarray,
    point_data: dict | None = None,
    cell_data: dict | None = None,
    title: str = "prestress",
) -> Path:
    """Write an unstructured grid with optional point and cell data.

    The cell type follows from the number of nodes per element, so the same
    writer serves the cylinder (hex8), the aorta wall (penta6) and the heart
    (tet4/tet10).  ``hexes`` keeps its name for backwards compatibility.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nodes = np.asarray(nodes, dtype=float)
    hexes = np.asarray(hexes, dtype=int)

    with path.open("w") as fh:
        fh.write(f"# vtk DataFile Version 3.0\n{title}\nASCII\n")
        fh.write("DATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {len(nodes)} double\n")
        np.savetxt(fh, nodes, fmt="%.12g")
        npe = hexes.shape[1]
        try:
            ctype = _VTK_CELL_TYPE[npe]
        except KeyError:
            raise ValueError(
                f"no VTK cell type for {npe} nodes per element; "
                f"known: {sorted(_VTK_CELL_TYPE)}") from None
        fh.write(f"\nCELLS {len(hexes)} {(npe + 1) * len(hexes)}\n")
        np.savetxt(fh, np.c_[np.full(len(hexes), npe), hexes], fmt="%d")
        fh.write(f"\nCELL_TYPES {len(hexes)}\n")
        np.savetxt(fh, np.full(len(hexes), ctype), fmt="%d")
        fh.write("\n")
        _write_arrays(fh, point_data or {}, len(nodes), "POINT_DATA")
        _write_arrays(fh, cell_data or {}, len(hexes), "CELL_DATA")
    return path
