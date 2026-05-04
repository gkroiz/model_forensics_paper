from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


PALETTE = {
    "blue":    "#4C72B0",
    "orange":  "#DD8452",
    "green":   "#55A868",
    "red":     "#C44E52",
    "purple":  "#8172B3",
    "brown":   "#937860",
    "pink":    "#DA8BC3",
    "gray":    "#8C8C8C",
    "yellow":  "#CCB974",
    "teal":    "#64B5CD",
}

CYCLE = [
    PALETTE["blue"],
    PALETTE["orange"],
    PALETTE["green"],
    PALETTE["red"],
    PALETTE["purple"],
    PALETTE["brown"],
]


def apply_theme() -> None:
    """Apply paper-grade matplotlib defaults. Idempotent."""
    plt.rcParams.update({
        # Fonts
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        # Color cycle
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        # Axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.labelpad": 6.0,
        "axes.titlepad": 12.0,
        "axes.axisbelow": True,
        # Grid
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#B0B0B0",
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "grid.linewidth": 0.6,
        # Ticks
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        "legend.fancybox": False,
        # Figure
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Lines
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
    })
