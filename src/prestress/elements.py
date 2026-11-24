"""Element shape functions and quadrature, matching FEBio's own rules.

Only what the prestressing needs: the gradient of the shape functions at the
integration points, so that quantities known a priori (the fibre stress) can be
assembled into nodal forces exactly as FEBio would assemble them.

Getting the quadrature wrong is quiet rather than loud -- the AIM still
converges, it just converges to a slightly different configuration -- so the
rules are taken straight from ``FECore/FEElementTraits.cpp``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Element", "ELEMENTS", "element_for"]


@dataclass(frozen=True)
class Element:
    """Nodes, integration points and shape-function gradients of one element type."""

    name: str
    n_node: int
    points: np.ndarray      # (n_gp, 3) natural coordinates
    weights: np.ndarray     # (n_gp,)
    _grad: callable         # natural coords -> (n_node, 3)
    _val: callable          # natural coords -> (n_node,)

    @property
    def n_gp(self) -> int:
        return len(self.points)

    def grads(self) -> np.ndarray:
        """dN/dxi at every integration point -> (n_gp, n_node, 3)."""
        return np.stack([self._grad(p) for p in self.points])

    def values(self) -> np.ndarray:
        """N at every integration point -> (n_gp, n_node)."""
        return np.stack([self._val(p) for p in self.points])


# ---------------------------------------------------------------------------
# HEX8 -- FEHex8G8: the integration points follow the node ordering
# ---------------------------------------------------------------------------
_HEX_CORNERS = np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float
)


def _hex8_grad(xi: np.ndarray) -> np.ndarray:
    s = _HEX_CORNERS * xi
    return 0.125 * np.stack(
        [_HEX_CORNERS[:, 0] * (1 + s[:, 1]) * (1 + s[:, 2]),
         _HEX_CORNERS[:, 1] * (1 + s[:, 0]) * (1 + s[:, 2]),
         _HEX_CORNERS[:, 2] * (1 + s[:, 0]) * (1 + s[:, 1])], axis=1)


# ---------------------------------------------------------------------------
# PENTA6 -- FEPenta6G6: a 3-point triangle rule at t = -+1/sqrt(3)
# ---------------------------------------------------------------------------
def _hex8_val(xi: np.ndarray) -> np.ndarray:
    return 0.125 * np.prod(1.0 + _HEX_CORNERS * xi, axis=1)


def _penta6_val(xi: np.ndarray) -> np.ndarray:
    r, s, t = xi
    return np.array([0.5 * (1 - t) * (1 - r - s), 0.5 * (1 - t) * r,
                     0.5 * (1 - t) * s, 0.5 * (1 + t) * (1 - r - s),
                     0.5 * (1 + t) * r, 0.5 * (1 + t) * s])


def _penta6_grad(xi: np.ndarray) -> np.ndarray:
    r, s, t = xi
    return np.array([
        [-0.5 * (1 - t), -0.5 * (1 - t), -0.5 * (1 - r - s)],
        [0.5 * (1 - t), 0.0, -0.5 * r],
        [0.0, 0.5 * (1 - t), -0.5 * s],
        [-0.5 * (1 + t), -0.5 * (1 + t), 0.5 * (1 - r - s)],
        [0.5 * (1 + t), 0.0, 0.5 * r],
        [0.0, 0.5 * (1 + t), 0.5 * s],
    ])


# ---------------------------------------------------------------------------
# TET10 -- FETet10G8, FEBio's *default* rule for tet10 (FEModelBuilder.cpp:59
# sets m_ntet10 = FE_TET10G8; the 4-point FETet10G4 is opt-in via <tet10>).
# Shape functions from FECore/FESolidElementShape.cpp, FETet10::shape_fnc.
# ---------------------------------------------------------------------------
def _tet10_val(xi: np.ndarray) -> np.ndarray:
    r, s, t = xi
    r1 = 1.0 - r - s - t
    return np.array([
        r1 * (2.0 * r1 - 1.0), r * (2.0 * r - 1.0),
        s * (2.0 * s - 1.0), t * (2.0 * t - 1.0),
        4.0 * r1 * r, 4.0 * r * s, 4.0 * s * r1,
        4.0 * r1 * t, 4.0 * r * t, 4.0 * s * t,
    ])


def _tet10_grad(xi: np.ndarray) -> np.ndarray:
    r, s, t = xi
    return np.array([
        [-3.0 + 4.0 * r + 4.0 * (s + t), -3.0 + 4.0 * s + 4.0 * (r + t),
         -3.0 + 4.0 * t + 4.0 * (r + s)],
        [4.0 * r - 1.0, 0.0, 0.0],
        [0.0, 4.0 * s - 1.0, 0.0],
        [0.0, 0.0, 4.0 * t - 1.0],
        [4.0 - 8.0 * r - 4.0 * (s + t), -4.0 * r, -4.0 * r],
        [4.0 * s, 4.0 * r, 0.0],
        [-4.0 * s, 4.0 - 8.0 * s - 4.0 * (r + t), -4.0 * s],
        [-4.0 * t, -4.0 * t, 4.0 - 8.0 * t - 4.0 * (r + s)],
        [4.0 * t, 0.0, 4.0 * r],
        [0.0, 4.0 * t, 4.0 * s],
    ])


_T_A, _T_B = 0.0158359099, 0.3280546970
_T_C, _T_D = 0.6791431780, 0.1069522740
_TET10_PTS = np.array([
    [_T_A, _T_B, _T_B], [_T_B, _T_A, _T_B], [_T_B, _T_B, _T_A], [_T_B, _T_B, _T_B],
    [_T_C, _T_D, _T_D], [_T_D, _T_C, _T_D], [_T_D, _T_D, _T_C], [_T_D, _T_D, _T_D],
])
_TET10_W = np.r_[np.full(4, 0.138527967), np.full(4, 0.111472033)] / 6.0


_a, _b, _c = 1.0 / 6.0, 2.0 / 3.0, 1.0 / np.sqrt(3.0)

ELEMENTS: dict[str, Element] = {
    "hex8": Element("hex8", 8, _HEX_CORNERS / np.sqrt(3.0),
                    np.ones(8), _hex8_grad, _hex8_val),
    "penta6": Element(
        "penta6", 6,
        np.array([[_a, _a, -_c], [_b, _a, -_c], [_a, _b, -_c],
                  [_a, _a, _c], [_b, _a, _c], [_a, _b, _c]]),
        np.full(6, 1.0 / 6.0), _penta6_grad, _penta6_val),
    "tet10": Element("tet10", 10, _TET10_PTS, _TET10_W,
                     _tet10_grad, _tet10_val),
}


def element_for(name: str) -> Element:
    try:
        return ELEMENTS[name]
    except KeyError:
        raise ValueError(
            f"no quadrature rule for element type {name!r}; "
            f"known types: {', '.join(ELEMENTS)}"
        ) from None
