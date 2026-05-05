"""
Binomial bar charts with 95% confidence intervals.

Usage:
    from plotting.binomial_plots import BarData, plot_binomial_bars, plot_grouped_binomial_bars

    # Simple bar chart
    bars = [
        BarData(label="Condition A", successes=10, total=96),
        BarData(label="Condition B", successes=39, total=85),
    ]
    plot_binomial_bars(bars, title="My Chart", y_label="Rate")

    # Grouped bar chart
    groups = [
        GroupedBarData("Model A", [("No Uncertainty", 8, 50), ("Uncertainty", 2, 46)]),
        GroupedBarData("Model B", [("No Uncertainty", 30, 60), ("Uncertainty", 9, 25)]),
    ]
    plot_grouped_binomial_bars(groups, title="Omission by Uncertainty")
"""

from dataclasses import dataclass, field
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


@dataclass
class BarData:
    """Data for a single bar."""

    label: str  # Bar label (x-axis)
    successes: int  # Number of "successes" (numerator)
    total: int  # Total trials (denominator)

    @property
    def rate(self) -> float:
        return self.successes / self.total if self.total > 0 else 0


@dataclass
class GroupedBarData:
    """Data for grouped bars."""

    group_label: str  # Group label (e.g., "Kimi")
    bars: list[tuple[str, int, int]]  # List of (bar_label, successes, total)


def compute_binomial_ci(
    successes: int,
    total: int,
    confidence: float = 0.95,
    method: Literal["wilson", "clopper-pearson", "normal"] = "wilson",
) -> tuple[float, float]:
    """
    Compute binomial confidence interval.

    Args:
        successes: Number of successes
        total: Total number of trials
        confidence: Confidence level (default 0.95 for 95% CI)
        method: CI method - "wilson" (recommended), "clopper-pearson" (exact), or "normal"

    Returns:
        (lower, upper) bounds of the confidence interval
    """
    if total == 0:
        return (0.0, 0.0)

    p = successes / total
    alpha = 1 - confidence

    if method == "wilson":
        # Wilson score interval (recommended for small samples)
        z = stats.norm.ppf(1 - alpha / 2)
        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
        return (max(0, center - margin), min(1, center + margin))

    elif method == "clopper-pearson":
        # Exact (Clopper-Pearson) interval
        lower = (
            stats.beta.ppf(alpha / 2, successes, total - successes + 1)
            if successes > 0
            else 0.0
        )
        upper = (
            stats.beta.ppf(1 - alpha / 2, successes + 1, total - successes)
            if successes < total
            else 1.0
        )
        return (lower, upper)

    elif method == "normal":
        # Normal approximation (only for large samples)
        z = stats.norm.ppf(1 - alpha / 2)
        margin = z * np.sqrt(p * (1 - p) / total)
        return (max(0, p - margin), min(1, p + margin))

    else:
        raise ValueError(f"Unknown method: {method}")


def get_colors(color_spec, n: int) -> list:
    """Get a list of n colors from a color specification."""
    if isinstance(color_spec, list):
        return color_spec[:n] + [color_spec[-1]] * max(0, n - len(color_spec))
    elif isinstance(color_spec, str):
        try:
            cmap = plt.get_cmap(color_spec)
            return [cmap(i / max(n - 1, 1)) for i in range(n)]
        except ValueError:
            return [color_spec] * n
    return ["steelblue"] * n


def plot_binomial_bars(
    bars: list[BarData],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    caption: str | None = None,
    figsize: tuple = (8, 6),
    y_min: float | None = 0.0,
    y_max: float | None = None,
    colors="tab10",
    bar_width: float = 0.6,
    error_bar_color: str = "black",
    error_bar_capsize: float = 5,
    show_values: bool = True,
    value_format: str = "{:.1%}",
    show_counts: bool = True,
    ci_method: str = "wilson",
    save_path: str | None = None,
    dpi: int = 150,
):
    """
    Create a bar chart with binomial confidence intervals.

    Args:
        bars: List of BarData objects
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        caption: Caption text displayed below the figure (None to skip)
        figsize: Figure size (width, height)
        y_min: Y-axis minimum (None for auto)
        y_max: Y-axis maximum (None for auto)
        colors: Color specification (single color, palette name, or list)
        bar_width: Width of bars
        error_bar_color: Color of error bars
        error_bar_capsize: Cap size of error bars
        show_values: Whether to show values on bars
        value_format: Format string for values
        show_counts: Whether to show counts below labels
        ci_method: Method for CI computation ("wilson", "clopper-pearson", "normal")
        save_path: Path to save figure (None to skip)
        dpi: DPI for saved figure
    """
    n = len(bars)
    x = np.arange(n)

    # Compute rates and CIs
    rates = [bar.rate for bar in bars]
    cis = [
        compute_binomial_ci(bar.successes, bar.total, method=ci_method) for bar in bars
    ]

    # Error bars (distance from rate to CI bounds, clamped to non-negative)
    yerr_lower = [max(0, rate - ci[0]) for rate, ci in zip(rates, cis)]
    yerr_upper = [max(0, ci[1] - rate) for rate, ci in zip(rates, cis)]

    # Get colors
    bar_colors = get_colors(colors, n)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot bars
    bars_plot = ax.bar(
        x,
        rates,
        width=bar_width,
        color=bar_colors,
        edgecolor="white",
        linewidth=1,
    )

    # Add error bars
    ax.errorbar(
        x,
        rates,
        yerr=[yerr_lower, yerr_upper],
        fmt="none",
        ecolor=error_bar_color,
        capsize=error_bar_capsize,
        capthick=1.5,
        elinewidth=1.5,
    )

    # Labels
    if show_counts:
        labels = [f"{bar.label}\n({bar.successes}/{bar.total})" for bar in bars]
    else:
        labels = [bar.label for bar in bars]

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # Show values on bars (anchored above CI upper bound)
    if show_values:
        for i, (bar_plot, rate) in enumerate(zip(bars_plot, rates)):
            ci_top = rate + yerr_upper[i]
            ax.annotate(
                value_format.format(rate),
                xy=(bar_plot.get_x() + bar_plot.get_width() / 2, ci_top),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    # Styling
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    # Y-axis limits
    current_ylim = ax.get_ylim()
    ax.set_ylim(
        y_min if y_min is not None else current_ylim[0],
        y_max if y_max is not None else current_ylim[1] * 1.15,  # Add space for labels
    )

    # Grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Add caption if provided
    if caption:
        fig.text(
            0.5,
            -0.02,
            caption,
            ha="center",
            va="top",
            fontsize=10,
            style="italic",
            wrap=True,
        )

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()

    # Print summary
    print("\nSummary:")
    print("-" * 60)
    for bar, ci in zip(bars, cis):
        print(
            f"{bar.label.replace(chr(10), ' ')}: {bar.rate:.1%} ({bar.successes}/{bar.total}) "
            f"95% CI: [{ci[0]:.1%}, {ci[1]:.1%}]"
        )


@dataclass
class SignificanceBracket:
    """Data for a significance bracket between two bars."""
    
    group_idx: int  # Which group (0-indexed)
    bar_idx_1: int  # First bar index within group (0-indexed)
    bar_idx_2: int  # Second bar index within group (0-indexed)
    text: str | None = None  # Custom text (if None, computes automatically)
    height: float | None = None  # Custom height (if None, auto-computed)
    method: str = "newcombe"  # "newcombe" for CI of difference, "fisher" for p-value


def compute_binomial_pvalue_fisher(s1: int, n1: int, s2: int, n2: int) -> float:
    """Compute two-tailed p-value using Fisher's exact test."""
    table = [[s1, n1 - s1], [s2, n2 - s2]]
    _, pvalue = stats.fisher_exact(table)
    return pvalue


def compute_binomial_pvalue_ztest(s1: int, n1: int, s2: int, n2: int) -> float:
    """Compute two-tailed p-value using two-proportion z-test (consistent with Newcombe CI)."""
    p1 = s1 / n1 if n1 > 0 else 0
    p2 = s2 / n2 if n2 > 0 else 0
    p_pool = (s1 + s2) / (n1 + n2) if (n1 + n2) > 0 else 0
    
    if p_pool == 0 or p_pool == 1 or n1 == 0 or n2 == 0:
        return 1.0
    
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if se == 0:
        return 1.0
    
    z = (p1 - p2) / se
    pvalue = 2 * (1 - stats.norm.cdf(abs(z)))
    return pvalue


def compute_newcombe_ci(
    s1: int, n1: int, s2: int, n2: int, confidence: float = 0.95
) -> tuple[float, float, float]:
    """
    Compute Newcombe's CI for the difference between two independent proportions.
    
    Uses Wilson score intervals for each proportion, then combines them using
    Newcombe's method (Method 10 from Newcombe 1998).
    
    Args:
        s1, n1: successes and total for group 1
        s2, n2: successes and total for group 2
        confidence: confidence level (default 0.95)
    
    Returns:
        (difference, lower_bound, upper_bound) for the CI of (p1 - p2)
    """
    p1 = s1 / n1 if n1 > 0 else 0
    p2 = s2 / n2 if n2 > 0 else 0
    diff = p1 - p2
    
    # Get Wilson CIs for each proportion
    l1, u1 = compute_binomial_ci(s1, n1, confidence=confidence, method="wilson")
    l2, u2 = compute_binomial_ci(s2, n2, confidence=confidence, method="wilson")
    
    # Newcombe's method for combining
    lower = diff - np.sqrt((p1 - l1)**2 + (u2 - p2)**2)
    upper = diff + np.sqrt((u1 - p1)**2 + (p2 - l2)**2)
    
    return diff, lower, upper


def format_pvalue(p: float) -> str:
    """Format p-value for display."""
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    elif p < 0.05:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.2f}"


def format_newcombe_ci(diff: float, lower: float, upper: float, pvalue: float | None = None) -> str:
    """Format Newcombe CI for display, optionally with p-value."""
    base = f"Δ = {diff*100:.1f}pp [{lower*100:.1f}, {upper*100:.1f}]"
    if pvalue is not None:
        if pvalue < 0.001:
            base += ", p < 0.001"
        else:
            base += f", p = {pvalue:.3f}"
    return base


def plot_grouped_binomial_bars(
    groups: list[GroupedBarData],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    caption: str | None = None,
    figsize: tuple = (10, 6),
    y_min: float | None = 0.0,
    y_max: float | None = None,
    colors="tab10",
    bar_width: float = 0.35,
    show_values: bool = True,
    value_format: str = "{:.1%}",
    show_legend: bool = True,
    ci_method: str = "wilson",
    save_path: str | None = None,
    dpi: int = 150,
    significance_brackets: list[SignificanceBracket] | None = None,
):
    """
    Create a grouped bar chart with binomial confidence intervals.

    Args:
        groups: List of GroupedBarData objects
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        caption: Caption text displayed below the figure (None to skip)
        figsize: Figure size (width, height)
        y_min: Y-axis minimum (None for auto)
        y_max: Y-axis maximum (None for auto)
        colors: Color specification (single color, palette name, or list)
        bar_width: Width of individual bars
        show_values: Whether to show values on bars
        value_format: Format string for values
        show_legend: Whether to show legend
        ci_method: Method for CI computation
        save_path: Path to save figure (None to skip)
        dpi: DPI for saved figure
        significance_brackets: List of SignificanceBracket objects for p-value annotations
    """
    n_groups = len(groups)
    n_bars_per_group = len(groups[0].bars) if groups else 0

    x = np.arange(n_groups)
    bar_colors = get_colors(colors, n_bars_per_group)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each bar type across groups
    for i in range(n_bars_per_group):
        bar_label = groups[0].bars[i][0]
        rates = []
        yerr_lower = []
        yerr_upper = []

        for group in groups:
            _, successes, total = group.bars[i]
            rate = successes / total if total > 0 else 0
            ci = compute_binomial_ci(successes, total, method=ci_method)
            rates.append(rate)
            yerr_lower.append(max(0, rate - ci[0]))
            yerr_upper.append(max(0, ci[1] - rate))

        offset = (i - (n_bars_per_group - 1) / 2) * bar_width
        bars_plot = ax.bar(
            x + offset,
            rates,
            width=bar_width,
            label=bar_label,
            color=bar_colors[i],
            edgecolor="white",
        )

        ax.errorbar(
            x + offset,
            rates,
            yerr=[yerr_lower, yerr_upper],
            fmt="none",
            ecolor="black",
            capsize=4,
            capthick=1.5,
            elinewidth=1.5,
        )

        if show_values:
            for bar_plot, rate, yu in zip(bars_plot, rates, yerr_upper):
                ci_top = rate + yu
                ax.annotate(
                    value_format.format(rate),
                    xy=(bar_plot.get_x() + bar_plot.get_width() / 2, ci_top),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    # Labels with counts
    group_labels = []
    for group in groups:
        counts = ", ".join(f"{s}/{t}" for _, s, t in group.bars)
        group_labels.append(f"{group.group_label}\n({counts})")

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    current_ylim = ax.get_ylim()
    ax.set_ylim(
        y_min if y_min is not None else current_ylim[0],
        y_max if y_max is not None else current_ylim[1] * 1.15,
    )

    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend:
        ax.legend(loc="upper right")

    # Draw significance brackets
    if significance_brackets:
        # Get current y limits for positioning
        y_lim = ax.get_ylim()
        y_range = y_lim[1] - y_lim[0]
        
        for bracket in significance_brackets:
            group_x = bracket.group_idx
            
            # Get bar positions
            offset1 = (bracket.bar_idx_1 - (n_bars_per_group - 1) / 2) * bar_width
            offset2 = (bracket.bar_idx_2 - (n_bars_per_group - 1) / 2) * bar_width
            x1 = group_x + offset1
            x2 = group_x + offset2
            
            # Get bar heights (rates)
            _, s1, n1 = groups[bracket.group_idx].bars[bracket.bar_idx_1]
            _, s2, n2 = groups[bracket.group_idx].bars[bracket.bar_idx_2]
            rate1 = s1 / n1 if n1 > 0 else 0
            rate2 = s2 / n2 if n2 > 0 else 0
            
            # Compute bracket height
            max_height = max(rate1, rate2)
            ci1 = compute_binomial_ci(s1, n1, method=ci_method)
            ci2 = compute_binomial_ci(s2, n2, method=ci_method)
            max_ci = max(ci1[1], ci2[1])
            
            if bracket.height is not None:
                bracket_y = bracket.height
            else:
                bracket_y = max_ci + y_range * 0.08
            
            # Get text
            if bracket.text is not None:
                text = bracket.text
            elif bracket.method == "newcombe":
                diff, lower, upper = compute_newcombe_ci(s1, n1, s2, n2)
                pvalue = compute_binomial_pvalue_ztest(s1, n1, s2, n2)  # z-test pairs with Newcombe
                text = format_newcombe_ci(diff, lower, upper, pvalue)
            else:  # fisher
                pvalue = compute_binomial_pvalue_fisher(s1, n1, s2, n2)
                text = format_pvalue(pvalue)
            
            # Draw bracket
            bracket_height = y_range * 0.02
            ax.plot([x1, x1, x2, x2], 
                    [bracket_y - bracket_height, bracket_y, bracket_y, bracket_y - bracket_height],
                    color='black', linewidth=1.2)
            
            # Add text
            ax.text((x1 + x2) / 2, bracket_y + y_range * 0.01, text,
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # Add caption if provided
    if caption:
        fig.text(
            0.5,
            -0.02,
            caption,
            ha="center",
            va="top",
            fontsize=10,
            style="italic",
            wrap=True,
        )

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()

    # Print summary
    print("\nSummary:")
    print("-" * 60)
    for group in groups:
        print(f"\n{group.group_label}:")
        for bar_label, successes, total in group.bars:
            rate = successes / total if total > 0 else 0
            ci = compute_binomial_ci(successes, total, method=ci_method)
            print(f"  {bar_label}: {rate:.1%} ({successes}/{total}) 95% CI: [{ci[0]:.1%}, {ci[1]:.1%}]")


# =============================================================================
# Example usage (uncomment to run)
# =============================================================================

if __name__ == "__main__":
    # Simple bar chart example
    bars = [
        BarData(label="Kimi\nwell_intentioned", successes=10, total=96),
        BarData(label="Kimi\nwhistleblower", successes=5, total=96),
        BarData(label="DeepSeek\nwell_intentioned", successes=39, total=85),
    ]

    plot_binomial_bars(
        bars,
        title="Omission Rate by Condition",
        y_label="Omission Rate",
        colors=["#4a7c59", "#8b4513", "#4682b4"],
    )

    # Grouped bar chart example
    groups = [
        GroupedBarData(
            "Kimi",
            [
                ("Uncertainty=False", 8, 50),
                ("Uncertainty=True", 2, 46),
            ],
        ),
        GroupedBarData(
            "DeepSeek",
            [
                ("Uncertainty=False", 30, 60),
                ("Uncertainty=True", 9, 25),
            ],
        ),
    ]

    plot_grouped_binomial_bars(
        groups,
        title="Omission Rate by Expressed Uncertainty",
        y_label="Omission Rate",
        colors=["#e74c3c", "#3498db"],
    )
