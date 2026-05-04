from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from paperplot.theme import PALETTE, apply_theme


@dataclass
class MeanBar:
    """Single bar with a pre-computed mean and CI bounds."""

    label: str  # bar category within a group (e.g., "First-person")
    mean: float
    ci_low: float
    ci_high: float


@dataclass
class MeanGroup:
    """A group of bars sharing within-group labels (one bar per label)."""

    group_label: str
    bars: list[MeanBar]


def plot_grouped_mean_bars(
    groups: Sequence[MeanGroup],
    *,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    figsize: tuple[float, float] = (8.0, 5.0),
    y_min: float | None = 0.0,
    y_max: float | None = None,
    colors: Sequence[str] | None = None,
    bar_width: float = 0.38,
    show_values: bool = True,
    value_format: str = "{:.0%}",
    show_legend: bool = True,
    percent_axis: bool = True,
    percent_decimals: int = 0,
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> Figure:
    """Grouped bar chart of pre-computed means with explicit CI bounds.

    Use this for non-binomial quantities (e.g., model-reported probabilities,
    aggregated scores). Bars within a group must share labels across groups
    so colours and the legend align.
    """
    apply_theme()

    if not groups:
        raise ValueError("at least one group is required")

    n_bars = len(groups[0].bars)
    if any(len(g.bars) != n_bars for g in groups):
        raise ValueError("all groups must have the same number of bars")

    bar_labels = [b.label for b in groups[0].bars]
    if colors is None:
        default = [PALETTE["blue"], PALETTE["red"], PALETTE["green"],
                   PALETTE["orange"], PALETTE["purple"]]
        colors = default[:n_bars]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = np.arange(len(groups))

    for i, label in enumerate(bar_labels):
        means = [g.bars[i].mean for g in groups]
        lows = [g.bars[i].ci_low for g in groups]
        highs = [g.bars[i].ci_high for g in groups]
        yerr_lo = [max(0.0, m - lo) for m, lo in zip(means, lows)]
        yerr_hi = [max(0.0, hi - m) for m, hi in zip(means, highs)]

        offset = (i - (n_bars - 1) / 2) * bar_width
        xs = x + offset
        ax.bar(
            xs,
            means,
            width=bar_width,
            color=colors[i],
            label=label,
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )
        ax.errorbar(
            xs,
            means,
            yerr=[yerr_lo, yerr_hi],
            fmt="none",
            ecolor="#333333",
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
            zorder=3,
        )

        if show_values:
            for xi, m, hi in zip(xs, means, yerr_hi):
                ax.annotate(
                    value_format.format(m),
                    xy=(xi, m + hi),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([g.group_label for g in groups])
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if percent_axis:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=percent_decimals))

    lo, hi = ax.get_ylim()
    ax.set_ylim(
        y_min if y_min is not None else lo,
        y_max if y_max is not None else hi * 1.12,
    )

    if show_legend:
        ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
