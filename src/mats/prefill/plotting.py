"""Plotting utilities for prefill logprob experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def interpolate_to_grid(
    results: list[dict],
    n_grid: int = 100,
    normalize_key: str | None = "cheat_para_idx",
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate per-run results to a common [0, 1] grid.

    Args:
        results: List of dicts, each with "cut_points" (list of {k, logprob_sum, skipped})
            and optionally a normalization key.
        n_grid: Number of grid points.
        normalize_key: Key in result dict to divide k by (for fractional x-axis).
            If None, x values are used as-is and grid spans [0, max_k].

    Returns:
        (grid, interpolated_array) where interpolated_array is (n_valid_runs, n_grid).
    """
    grid = np.linspace(0, 1, n_grid)
    interpolated = []

    for r in results:
        valid = [cp for cp in r["cut_points"] if not cp.get("skipped", False)]
        if len(valid) < 2:
            continue

        if normalize_key is not None:
            denom = r[normalize_key]
            fracs = np.array([cp["k"] / denom for cp in valid])
        else:
            max_k = max(cp["k"] for cp in valid)
            fracs = np.array([cp["k"] / max_k if max_k > 0 else 0 for cp in valid])

        vals = np.array([cp["logprob_sum"] for cp in valid])
        interp_vals = np.interp(grid, fracs, vals)
        interpolated.append(interp_vals)

    return grid, np.array(interpolated)


def plot_cohort_overlay(
    cohorts: list[dict],
    output_path: Path,
    *,
    x_label: str = "Paragraph number",
    y_label: str = 'Log-prob of suffix',
    title: str = "",
    show_individual: bool = True,
    individual_alpha: float = 0.04,
    show_n_axis: bool = False,
    min_samples_per_k: int = 1,
    max_k: int | None = None,
    y_lim: tuple[float, float] | None = (-35, 0),
    figsize: tuple[float, float] = (10, 5),
):
    """Plot mean+SEM overlay for multiple cohorts with optional individual run lines.

    Args:
        cohorts: List of dicts, each with:
            - "results": list of result dicts (with "cut_points")
            - "label": str for legend
            - "color": matplotlib color string
        output_path: Where to save the plot.
        x_label, y_label, title: Axis and title labels.
        show_individual: Draw faded individual run lines.
        individual_alpha: Alpha for individual lines.
        show_n_axis: Add secondary y-axis showing sample size at each k.
        min_samples_per_k: Only plot aggregate at k values with >= this many samples.
        max_k: If set, only plot up to this paragraph number.
        y_lim: Y-axis limits (None for auto).
        figsize: Figure size.
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    # Determine max_k from data if not provided
    if max_k is None:
        all_ks = [
            cp["k"]
            for c in cohorts
            for r in c["results"]
            for cp in r["cut_points"]
            if not cp.get("skipped", False)
        ]
        max_k = max(all_ks) if all_ks else 0

    n_by_cohort: dict[str, dict[int, int]] = {}

    for cohort in cohorts:
        label = cohort["label"]
        color = cohort["color"]
        results = cohort["results"]

        # Aggregate by k
        by_k: dict[int, list[float]] = {}
        for r in results:
            for cp in r["cut_points"]:
                if not cp.get("skipped", False) and cp["k"] <= max_k:
                    by_k.setdefault(cp["k"], []).append(cp["logprob_sum"])

        # Individual run lines (faded)
        if show_individual:
            for r in results:
                valid = [
                    cp
                    for cp in r["cut_points"]
                    if not cp.get("skipped", False) and cp["k"] <= max_k
                ]
                if len(valid) >= 2:
                    run_ks = [cp["k"] for cp in valid]
                    run_vals = [cp["logprob_sum"] for cp in valid]
                    ax1.plot(
                        run_ks, run_vals, color=color,
                        alpha=individual_alpha, linewidth=0.5,
                    )

        # Aggregate mean ± 1 SD
        ks = sorted(k for k in by_k if len(by_k[k]) >= min_samples_per_k)
        if ks:
            means = [np.mean(by_k[k]) for k in ks]
            sds = [np.std(by_k[k]) for k in ks]
            n_total = len(results)

            ax1.fill_between(
                ks,
                np.array(means) - np.array(sds),
                np.array(means) + np.array(sds),
                alpha=0.25, color=color,
            )
            ax1.plot(ks, means, color=color, linewidth=2, label=f"{label} (n={n_total})")

        # Track sample counts for n-axis
        if show_n_axis:
            cohort_label = cohort.get("n_axis_label", label)
            n_by_cohort[cohort_label] = {}
            for r in results:
                for cp in r["cut_points"]:
                    if not cp.get("skipped", False) and cp["k"] <= max_k:
                        n_by_cohort[cohort_label][cp["k"]] = (
                            n_by_cohort[cohort_label].get(cp["k"], 0) + 1
                        )

    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel(y_label, fontsize=12)
    if title:
        ax1.set_title(title, fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.2)
    if y_lim is not None:
        ax1.set_ylim(*y_lim)

    # Secondary axis for sample size
    if show_n_axis and n_by_cohort:
        all_ks_set: set[int] = set()
        for counts in n_by_cohort.values():
            all_ks_set.update(counts.keys())
        all_ks_sorted = sorted(all_ks_set)

        ax2 = ax1.twinx()
        for i, (cohort_label, counts) in enumerate(n_by_cohort.items()):
            color = cohorts[i]["color"]
            n_vals = [counts.get(k, 0) for k in all_ks_sorted]
            ax2.fill_between(all_ks_sorted, n_vals, alpha=0.06, color=color)
            ax2.plot(
                all_ks_sorted, n_vals, color=color,
                linewidth=0.8, alpha=0.3, label=cohort_label,
            )
        ax2.set_ylabel("n (traces at this k)", fontsize=10, color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
        ax2.legend(fontsize=7, loc="right", title="n", title_fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved overlay plot to {output_path}")


def plot_small_multiples(
    results: list[dict],
    output_path: Path,
    *,
    ncols: int = 7,
    normalize_key: str = "cheat_para_idx",
    suptitle: str = "Log-prob of suffix vs fraction of reasoning — per run",
):
    """Small-multiples panel: one subplot per run.

    Args:
        results: List of dicts with "cut_points", "run_num", and normalize_key.
        output_path: Where to save the plot.
        ncols: Number of columns in the grid.
        normalize_key: Key in result dict to divide k by for fractional x-axis.
        suptitle: Figure-level title.
    """
    n = len(results)
    if n == 0:
        print("No results to plot.")
        return

    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 2.4, nrows * 2.2), squeeze=False,
    )

    # Global y-limits for consistent axes
    all_vals = []
    for r in results:
        for cp in r["cut_points"]:
            if not cp.get("skipped", False):
                all_vals.append(cp["logprob_sum"])
    if all_vals:
        ymin, ymax = min(all_vals), max(all_vals)
        ypad = (ymax - ymin) * 0.08
        ymin -= ypad
        ymax += ypad
    else:
        ymin, ymax = -30, 0

    for i, r in enumerate(results):
        row, col = divmod(i, ncols)
        ax = axes[row][col]

        valid = [cp for cp in r["cut_points"] if not cp.get("skipped", False)]
        if normalize_key is not None:
            denom = r.get(normalize_key, 1) or 1
        else:
            denom = max((cp["k"] for cp in valid), default=1) or 1

        if valid:
            fracs = [cp["k"] / denom for cp in valid]
            vals = [cp["logprob_sum"] for cp in valid]
            ax.plot(fracs, vals, color="C0", linewidth=1, alpha=0.8)
            ax.fill_between(fracs, ymin, vals, alpha=0.08, color="C0")

        ax.set_xlim(0, 1)
        ax.set_ylim(ymin, ymax)
        run_label = r.get("run_num", i)
        ax.set_title(f"run-{run_label}  ({denom}\u00b6)", fontsize=7, pad=2)
        ax.tick_params(labelsize=6)

        if row == nrows - 1:
            ax.set_xlabel("frac", fontsize=7)
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel("logprob", fontsize=7)
        else:
            ax.set_yticklabels([])

        ax.grid(alpha=0.15)

    # Hide unused subplots
    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(suptitle, fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved small-multiples plot to {output_path}")


def _compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC from scores and binary labels via the rank-sum formula.

    Uses the equivalence AUROC = (R_pos - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    where R_pos is the sum of ranks of positive examples. O(n log n).
    """
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # scipy.stats.rankdata handles ties with average ranks by default
    from scipy.stats import rankdata
    ranks = rankdata(scores)
    r_pos = ranks[labels == 1].sum()
    return (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def plot_auroc_by_k(
    positive_results: list[dict],
    negative_results: list[dict],
    output_path: Path,
    *,
    positive_label: str = "positive",
    negative_label: str = "negative",
    min_samples_per_class: int = 5,
    max_k: int | None = None,
    title: str = "AUROC by paragraph",
    y_lim: tuple[float, float] = (0.45, 0.85),
    show_logreg: bool = True,
    bootstrap_ci: bool = True,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    figsize: tuple[float, float] = (10, 5),
):
    """Plot AUROC curves showing how well logprobs discriminate two cohorts at each k.

    Plots up to three curves:
      1. Single paragraph: AUROC using only logprob at paragraph k.
      2. Average 0..k: AUROC using the mean logprob over paragraphs 0 through k.
      3. LogReg 0..k (5-fold CV): AUROC using all logprobs 0..k as features (requires sklearn).

    Uses variable n — at each k, includes all traces that have valid data at paragraphs 0..k.

    Args:
        positive_results: List of result dicts for the positive class (label=1).
        negative_results: List of result dicts for the negative class (label=0).
        output_path: Where to save the plot.
        positive_label: Legend label for positive class.
        negative_label: Legend label for negative class.
        min_samples_per_class: Skip k values with fewer samples per class.
        bootstrap_ci: If True, show bootstrap confidence intervals as shaded bands.
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level (default 0.95 for 95% CIs).
        max_k: If set, only plot up to this paragraph number.
        title: Plot title.
        y_lim: Y-axis limits.
        show_logreg: Whether to compute and plot the LogReg CV curve (requires sklearn).
        figsize: Figure size.
    """
    # Build per-trace logprob dicts
    traces = []
    for results, label in [(positive_results, 1), (negative_results, 0)]:
        for r in results:
            lps = {
                cp["k"]: cp["logprob_sum"]
                for cp in r["cut_points"]
                if not cp.get("skipped", False)
            }
            if lps:
                traces.append({"logprobs": lps, "label": label})

    if not traces:
        print("No valid traces for AUROC plot.")
        return

    data_max_k = max(k for t in traces for k in t["logprobs"])
    if max_k is None:
        max_k = data_max_k
    else:
        max_k = min(max_k, data_max_k)
    print(f"plot_auroc: {len(traces)} traces, max_k={max_k}")

    # Optionally import sklearn for LogReg curve
    logreg_available = False
    if show_logreg:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_predict
            logreg_available = True
        except ImportError:
            print("  sklearn not available, skipping LogReg curve")

    single_aurocs, avg_aurocs, logreg_aurocs = [], [], []
    single_ci_lo, single_ci_hi = [], []
    avg_ci_lo, avg_ci_hi = [], []
    ns_pos, ns_neg, ks_valid = [], [], []
    rng = np.random.default_rng(42) if bootstrap_ci else None
    alpha_lo = (1 - ci_level) / 2
    alpha_hi = 1 - alpha_lo

    for k in range(max_k + 1):
        valid = [t for t in traces if all(j in t["logprobs"] for j in range(k + 1))]
        labels = np.array([t["label"] for t in valid])
        n_pos, n_neg = int(labels.sum()), int(len(labels) - labels.sum())
        if n_pos < min_samples_per_class or n_neg < min_samples_per_class:
            continue

        ks_valid.append(k)
        ns_pos.append(n_pos)
        ns_neg.append(n_neg)

        # 1. Single paragraph AUROC
        scores = np.array([t["logprobs"][k] for t in valid])
        single_aurocs.append(_compute_auroc(scores, labels))

        # 2. Average 0..k AUROC
        avg_scores = np.array([
            np.mean([t["logprobs"][j] for j in range(k + 1)]) for t in valid
        ])
        avg_aurocs.append(_compute_auroc(avg_scores, labels))

        # 3. Bootstrap CIs for single and average
        if bootstrap_ci:
            boot_single, boot_avg = [], []
            n = len(labels)
            for _ in range(n_bootstrap):
                idx = rng.integers(0, n, size=n)
                b_labels = labels[idx]
                if b_labels.sum() == 0 or b_labels.sum() == n:
                    continue  # Skip degenerate resamples
                boot_single.append(_compute_auroc(scores[idx], b_labels))
                boot_avg.append(_compute_auroc(avg_scores[idx], b_labels))
            if boot_single:
                single_ci_lo.append(np.percentile(boot_single, 100 * alpha_lo))
                single_ci_hi.append(np.percentile(boot_single, 100 * alpha_hi))
                avg_ci_lo.append(np.percentile(boot_avg, 100 * alpha_lo))
                avg_ci_hi.append(np.percentile(boot_avg, 100 * alpha_hi))
            else:
                single_ci_lo.append(single_aurocs[-1])
                single_ci_hi.append(single_aurocs[-1])
                avg_ci_lo.append(avg_aurocs[-1])
                avg_ci_hi.append(avg_aurocs[-1])

        # 4. LogReg 5-fold CV AUROC
        if logreg_available:
            if k == 0:
                logreg_aurocs.append(single_aurocs[-1])
            else:
                X = np.array([[t["logprobs"][j] for j in range(k + 1)] for t in valid])
                clf = LogisticRegression(max_iter=1000)
                cv_scores = cross_val_predict(
                    clf, X, labels, cv=5, method="predict_proba",
                )[:, 1]
                logreg_aurocs.append(_compute_auroc(cv_scores, labels))

        if k % 10 == 0:
            msg = (
                f"  k={k:3d}: n={ns_pos[-1]+ns_neg[-1]:4d} "
                f"({ns_pos[-1]}+/{ns_neg[-1]}-)  "
                f"single={single_aurocs[-1]:.3f}  avg={avg_aurocs[-1]:.3f}"
            )
            if logreg_available:
                msg += f"  logreg_cv={logreg_aurocs[-1]:.3f}"
            print(msg)

    if not ks_valid:
        print(f"No k values with >= {min_samples_per_class} samples per class.")
        return

    # Plot
    fig, ax1 = plt.subplots(figsize=figsize)
    if bootstrap_ci and single_ci_lo:
        single_yerr = [
            np.array(single_aurocs) - np.array(single_ci_lo),
            np.array(single_ci_hi) - np.array(single_aurocs),
        ]
        avg_yerr = [
            np.array(avg_aurocs) - np.array(avg_ci_lo),
            np.array(avg_ci_hi) - np.array(avg_aurocs),
        ]
        ax1.errorbar(
            ks_valid, single_aurocs, yerr=single_yerr, fmt="o-", color="C0",
            markersize=3, linewidth=1.5, label="Single paragraph", alpha=0.7,
            elinewidth=0.8, capsize=2,
        )
        ax1.errorbar(
            ks_valid, avg_aurocs, yerr=avg_yerr, fmt="s-", color="C1",
            markersize=3, linewidth=1.5, label="Average 0..k (0 params)",
            elinewidth=0.8, capsize=2,
        )
    else:
        ax1.plot(
            ks_valid, single_aurocs, "o-", color="C0",
            markersize=3, linewidth=1.5, label="Single paragraph", alpha=0.7,
        )
        ax1.plot(
            ks_valid, avg_aurocs, "s-", color="C1",
            markersize=3, linewidth=1.5, label="Average 0..k (0 params)",
        )
    if logreg_available and logreg_aurocs:
        ax1.plot(
            ks_valid, logreg_aurocs, "^-", color="C2",
            markersize=3, linewidth=1.5, label="LogReg 0..k (5-fold CV)",
        )
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.3, label="Random")
    ax1.set_xlabel("Paragraph number (k)", fontsize=12)
    ax1.set_ylabel("AUROC", fontsize=12)
    ax1.set_title(title, fontsize=13)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(alpha=0.2)
    ax1.set_ylim(*y_lim)

    # Secondary axis for sample size
    ax2 = ax1.twinx()
    ax2.fill_between(ks_valid, ns_pos, alpha=0.06, color="C3")
    ax2.plot(ks_valid, ns_pos, color="C3", linewidth=0.8, alpha=0.3, label=positive_label)
    ax2.fill_between(ks_valid, ns_neg, alpha=0.06, color="C0")
    ax2.plot(ks_valid, ns_neg, color="C0", linewidth=0.8, alpha=0.3, label=negative_label)
    ax2.set_ylabel("n (traces available)", fontsize=10, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.legend(fontsize=7, loc="right", title="n", title_fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved AUROC plot to {output_path}")
