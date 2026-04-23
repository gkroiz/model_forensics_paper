from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from paperplot.stats import wilson_ci
from paperplot.theme import PALETTE, apply_theme


@dataclass
class BarData:
    label: str
    successes: int
    total: int

    @property
    def rate(self) -> float:
        return self.successes / self.total if self.total > 0 else 0.0


@dataclass
class SweepPoint:
    x: float
    successes: int
    total: int
    label: str | None = None


@dataclass
class GroupedBarData:
    """A group of bars sharing the same within-group bar labels.

    Each bar is a tuple of (bar_label, successes, total). The bar_labels must
    match across groups so bars can be aligned and legend-coloured correctly.
    """

    group_label: str
    bars: list[tuple[str, int, int]]


def _percent_formatter(decimals: int = 0):
    return PercentFormatter(xmax=1.0, decimals=decimals)


def _label_offset_points(figsize: tuple[float, float]) -> int:
    return max(4, int(figsize[1]))


def plot_binomial_bars(
    bars: Sequence[BarData],
    *,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    figsize: tuple[float, float] = (6.0, 4.5),
    y_min: float | None = 0.0,
    y_max: float | None = None,
    colors: Sequence[str] | str | None = None,
    bar_width: float = 0.62,
    show_values: bool = True,
    value_format: str = "{:.0%}",
    show_counts: bool = False,
    percent_decimals: int = 0,
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> Figure:
    """Bar chart of binomial proportions with 95% Wilson CIs."""
    apply_theme()

    rates = [b.rate for b in bars]
    cis = [wilson_ci(b.successes, b.total) for b in bars]
    yerr_lo = [max(0.0, r - lo) for r, (lo, _) in zip(rates, cis)]
    yerr_hi = [max(0.0, hi - r) for r, (_, hi) in zip(rates, cis)]

    if colors is None:
        colors = [PALETTE["blue"]] * len(bars)
    elif isinstance(colors, str):
        colors = [colors] * len(bars)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = np.arange(len(bars))
    ax.bar(
        x,
        rates,
        width=bar_width,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )
    ax.errorbar(
        x,
        rates,
        yerr=[yerr_lo, yerr_hi],
        fmt="none",
        ecolor="#333333",
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        zorder=3,
    )

    if show_values:
        offset = _label_offset_points(figsize)
        for xi, r, hi in zip(x, rates, yerr_hi):
            ax.annotate(
                value_format.format(r),
                xy=(xi, r + hi),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    labels = [
        f"{b.label}\n({b.successes}/{b.total})" if show_counts else b.label
        for b in bars
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(_percent_formatter(percent_decimals))

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    lo, hi = ax.get_ylim()
    ax.set_ylim(
        y_min if y_min is not None else lo,
        y_max if y_max is not None else hi * 1.18,
    )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_binomial_sweep(
    sweep: Sequence[SweepPoint],
    *,
    highlight: Sequence[SweepPoint] | None = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    figsize: tuple[float, float] = (8.0, 5.0),
    sweep_label: str = "Sweep conditions",
    highlight_label: str = "Original setting",
    sweep_color: str | None = None,
    highlight_color: str | None = None,
    connect_line: bool = True,
    show_values: bool = True,
    value_format: str = "{:.1%}",
    y_min: float | None = 0.0,
    y_max: float | None = None,
    x_pad_frac: float = 0.04,
    percent_decimals: int = 0,
    show_legend: bool = True,
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> Figure:
    """Binomial rates across a swept parameter, with 95% Wilson CIs.

    An optional `highlight` list places additional points drawn separately
    (e.g., the baseline condition in a different color).
    """
    apply_theme()
    sweep_color = sweep_color or PALETTE["blue"]
    highlight_color = highlight_color or PALETTE["orange"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if connect_line:
        all_ordered = sorted(
            list(sweep) + (list(highlight) if highlight else []),
            key=lambda p: p.x,
        )
        if len(all_ordered) > 1:
            line_xs = [p.x for p in all_ordered]
            line_ys = [
                p.successes / p.total if p.total > 0 else 0.0 for p in all_ordered
            ]
            ax.plot(
                line_xs,
                line_ys,
                color=sweep_color,
                alpha=0.45,
                linewidth=1.5,
                zorder=1,
            )

    _plot_points(
        ax,
        sweep,
        color=sweep_color,
        label=sweep_label,
        show_values=show_values,
        value_format=value_format,
    )

    if highlight:
        _plot_points(
            ax,
            highlight,
            color=highlight_color,
            label=highlight_label,
            show_values=show_values,
            value_format=value_format,
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_formatter(_percent_formatter(percent_decimals))

    all_points = list(sweep) + (list(highlight) if highlight else [])
    xs = [p.x for p in all_points]
    x_lo, x_hi = min(xs), max(xs)
    span = (x_hi - x_lo) or 1.0
    ax.set_xlim(x_lo - span * x_pad_frac, x_hi + span * x_pad_frac)

    lo, hi = ax.get_ylim()
    ax.set_ylim(
        y_min if y_min is not None else lo,
        y_max if y_max is not None else hi * 1.15,
    )

    if show_legend:
        ax.legend(loc="upper left")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def _plot_points(
    ax: plt.Axes,
    points: Sequence[SweepPoint],
    *,
    color: str,
    label: str,
    show_values: bool,
    value_format: str,
) -> None:
    xs = [p.x for p in points]
    rates = [p.successes / p.total if p.total > 0 else 0.0 for p in points]
    cis = [wilson_ci(p.successes, p.total) for p in points]
    yerr_lo = [max(0.0, r - lo) for r, (lo, _) in zip(rates, cis)]
    yerr_hi = [max(0.0, hi - r) for r, (_, hi) in zip(rates, cis)]

    ax.errorbar(
        xs,
        rates,
        yerr=[yerr_lo, yerr_hi],
        fmt="o",
        color=color,
        ecolor=color,
        elinewidth=1.1,
        capsize=3.5,
        capthick=1.1,
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=0.6,
        label=label,
        zorder=3,
    )

    if show_values:
        for xi, r, hi in zip(xs, rates, yerr_hi):
            ax.annotate(
                value_format.format(r),
                xy=(xi, r + hi),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )


def plot_grouped_binomial_bars(
    groups: Sequence[GroupedBarData],
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
    value_format: str = "{:.1%}",
    show_counts: bool = True,
    show_legend: bool = True,
    percent_decimals: int = 0,
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> Figure:
    """Grouped bar chart of binomial proportions with 95% Wilson CIs.

    Each group is drawn as a cluster of N bars (one per within-group label),
    with colours assigned per within-group position and shown in the legend.
    """
    apply_theme()

    if not groups:
        raise ValueError("at least one group is required")

    n_bars = len(groups[0].bars)
    if any(len(g.bars) != n_bars for g in groups):
        raise ValueError("all groups must have the same number of bars")

    bar_labels = [b[0] for b in groups[0].bars]
    if colors is None:
        default = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"],
                   PALETTE["red"], PALETTE["purple"]]
        colors = default[:n_bars]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = np.arange(len(groups))
    offset_step = bar_width

    for i, label in enumerate(bar_labels):
        successes = [g.bars[i][1] for g in groups]
        totals = [g.bars[i][2] for g in groups]
        rates = [s / t if t > 0 else 0.0 for s, t in zip(successes, totals)]
        cis = [wilson_ci(s, t) for s, t in zip(successes, totals)]
        yerr_lo = [max(0.0, r - lo) for r, (lo, _) in zip(rates, cis)]
        yerr_hi = [max(0.0, hi - r) for r, (_, hi) in zip(rates, cis)]

        offset = (i - (n_bars - 1) / 2) * offset_step
        xs = x + offset
        ax.bar(
            xs,
            rates,
            width=bar_width,
            color=colors[i],
            label=label,
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )
        ax.errorbar(
            xs,
            rates,
            yerr=[yerr_lo, yerr_hi],
            fmt="none",
            ecolor="#333333",
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
            zorder=3,
        )

        if show_values:
            for xi, r, hi in zip(xs, rates, yerr_hi):
                ax.annotate(
                    value_format.format(r),
                    xy=(xi, r + hi),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                    fontweight="bold",
                )

    group_labels = []
    for g in groups:
        if show_counts:
            counts = ", ".join(f"{s}/{t}" for _, s, t in g.bars)
            group_labels.append(f"{g.group_label}\n({counts})")
        else:
            group_labels.append(g.group_label)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.yaxis.set_major_formatter(_percent_formatter(percent_decimals))
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    lo, hi = ax.get_ylim()
    ax.set_ylim(
        y_min if y_min is not None else lo,
        y_max if y_max is not None else hi * 1.18,
    )

    if show_legend:
        ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
