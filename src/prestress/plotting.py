"""Matplotlib helpers shared by the example scripts."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = ["use_paper_style", "COLORS"]

COLORS = {
    "paper": "#1f77b4",
    "implied": "#d62728",
    "analytic": "#333333",
    "aim": "#1f77b4",
    "grey": "#888888",
}


def use_paper_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.4,
            "legend.frameon": False,
        }
    )
