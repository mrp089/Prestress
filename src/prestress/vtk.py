"""Minimal legacy-VTK writer, so the results can be opened in ParaView.

Only what is needed here: an unstructured grid of HEX8 cells with point and
cell data.  No dependencies beyond NumPy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["write_vtk"]

_VTK_HEXAHEDRON = 12


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
    """Write a HEX8 unstructured grid with optional point and cell data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nodes = np.asarray(nodes, dtype=float)
    hexes = np.asarray(hexes, dtype=int)

    with path.open("w") as fh:
        fh.write(f"# vtk DataFile Version 3.0\n{title}\nASCII\n")
        fh.write("DATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {len(nodes)} double\n")
        np.savetxt(fh, nodes, fmt="%.12g")
        fh.write(f"\nCELLS {len(hexes)} {9 * len(hexes)}\n")
        np.savetxt(fh, np.c_[np.full(len(hexes), 8), hexes], fmt="%d")
        fh.write(f"\nCELL_TYPES {len(hexes)}\n")
        np.savetxt(fh, np.full(len(hexes), _VTK_HEXAHEDRON), fmt="%d")
        fh.write("\n")
        _write_arrays(fh, point_data or {}, len(nodes), "POINT_DATA")
        _write_arrays(fh, cell_data or {}, len(hexes), "CELL_DATA")
    return path
