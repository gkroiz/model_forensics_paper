"""
Priming experiment runner.

This module provides the main experiment runner that computes CE loss
across two distributions (interest + baseline) with different primes,
and generates the pareto frontier plot.
"""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm.asyncio import tqdm_asyncio

from mats.api.together import get_async_together_client, call_api
from mats.grading.rollout_utils import find_all_runs, load_rollout_messages
from mats.priming.ce_loss import compute_ce_loss, inject_prime
from mats.priming.deepseek import (
    construct_prompt_with_reasoning,
    find_reasoning_bounds,
    get_tokenizer,
)


@dataclass
class PrimeGroup:
    """A group of related primes (e.g., IF primes, CR primes)."""

    name: str  # e.g., "IF", "CR", "Control"
    primes: dict[str, str]  # e.g., {"implicit_intent": "You carefully infer..."}


@dataclass
class PrimingConfig:
    """Configuration for a priming experiment."""

    model: str  # e.g., "deepseek-ai/DeepSeek-R1"
    tokenizer: str  # e.g., "deepseek-ai/DeepSeek-R1-0528"
    prime_groups: list[PrimeGroup]
    prime_location: str = "user"  # "user" or "system"


@dataclass
class CEResult:
    """Result of CE computation for a single sample + prime."""

    sample_id: str
    prime_name: str
    ce: float
    n_tokens: int
    run_idx: int = 0  # Which run this is from (for multi-run experiments)


@dataclass
class DistributionResults:
    """Results for a single distribution across all primes."""

    name: str
    n_samples: int
    n_runs: int = 1  # Number of runs per sample
    # {prime_name: {"prime": str, "per_sample": [...], "mean_ce": float, "ci_95": float}}
    results: dict = field(default_factory=dict)


def _get_all_primes(config: PrimingConfig) -> dict[str, str]:
    """Flatten all primes from config into a single dict with baseline."""
    all_primes = {"baseline": ""}  # Always include baseline (no prime)
    for group in config.prime_groups:
        for name, text in group.primes.items():
            all_primes[f"{group.name}_{name}"] = text
    return all_primes


async def _process_one_ce_task(
    sample_id: str,
    messages: list[dict],
    prime_name: str,
    prime_text: str,
    prime_location: str,
    model: str,
    client,
    tokenizer,
    semaphore: asyncio.Semaphore,
    run_idx: int = 0,
) -> CEResult:
    """Process one (sample, prime) pair and return CE loss."""
    async with semaphore:
        # Inject prime if provided
        if prime_text:
            messages = inject_prime(messages, prime_text, prime_location)

        # Construct prompt and get logprobs
        prompt = construct_prompt_with_reasoning(messages, tokenizer)
        response = await call_api(
            client, model, prompt, max_tokens=1, logprobs=1, echo=True
        )

        # Extract logprobs from response
        if response.prompt and response.prompt[0].logprobs:
            tokens = response.prompt[0].logprobs.tokens
            logprobs = response.prompt[0].logprobs.token_logprobs
        else:
            raise ValueError(f"No logprobs returned for {sample_id}/{prime_name}")

        # Extract reasoning section and compute CE
        start, end = find_reasoning_bounds(tokens)
        reasoning_logprobs = logprobs[start:end]
        ce = compute_ce_loss(reasoning_logprobs)

        return CEResult(
            sample_id=sample_id,
            prime_name=prime_name,
            ce=ce,
            n_tokens=len(reasoning_logprobs),
            run_idx=run_idx,
        )


def _load_interest_distribution(interest_dir: Path) -> list[tuple[str, list[dict]]]:
    """
    Load messages from interest distribution (run-N/step-M/messages.json format).

    Returns list of (sample_id, messages) tuples.
    """
    samples = []
    run_dirs = find_all_runs(interest_dir)

    for run_dir in run_dirs:
        try:
            messages = load_rollout_messages(run_dir)
            samples.append((run_dir.name, messages))
        except FileNotFoundError as e:
            print(f"Warning: Skipping {run_dir}: {e}")

    return samples


def _load_baseline_distribution(baseline_dir: Path) -> list[tuple[str, list[dict]]]:
    """
    Load messages from baseline distribution (flat folder with *.json files).

    Returns list of (sample_id, messages) tuples.
    """
    samples = []
    json_files = sorted(baseline_dir.glob("*.json"))

    for json_file in json_files:
        try:
            with open(json_file) as f:
                messages = json.load(f)
            samples.append((json_file.stem, messages))
        except Exception as e:
            print(f"Warning: Skipping {json_file}: {e}")

    return samples


def _compute_95_ci(values: list[float]) -> float:
    """Compute 95% confidence interval half-width using t-distribution."""
    if len(values) < 2:
        return 0.0
    n = len(values)
    std = np.std(values, ddof=1)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return t_crit * std / np.sqrt(n)


async def _compute_ce_for_distribution(
    dist_name: str,
    samples: list[tuple[str, list[dict]]],
    all_primes: dict[str, str],
    config: PrimingConfig,
    client,
    tokenizer,
    max_concurrent: int,
    n_runs: int = 1,
) -> DistributionResults:
    """Compute CE loss for all samples and primes in a distribution."""
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create all tasks (n_runs copies of each sample/prime pair)
    tasks = []
    for run_idx in range(n_runs):
        for sample_id, messages in samples:
            for prime_name, prime_text in all_primes.items():
                tasks.append(
                    _process_one_ce_task(
                        sample_id=sample_id,
                        messages=messages,
                        prime_name=prime_name,
                        prime_text=prime_text,
                        prime_location=config.prime_location,
                        model=config.model,
                        client=client,
                        tokenizer=tokenizer,
                        semaphore=semaphore,
                        run_idx=run_idx,
                    )
                )

    print(f"\nProcessing {dist_name}: {len(tasks)} tasks ({n_runs} run(s) x {len(samples)} samples x {len(all_primes)} primes)...")
    results_list = await tqdm_asyncio.gather(*tasks, desc=dist_name)

    # Aggregate results by prime
    results = DistributionResults(name=dist_name, n_samples=len(samples), n_runs=n_runs)
    for prime_name, prime_text in all_primes.items():
        results.results[prime_name] = {
            "prime": prime_text if prime_text else None,
            "per_sample": [],
            "all_ces": [],  # All CE values across all runs (for CI computation)
        }

    # Group by (sample_id, prime_name)
    sample_prime_ces: dict[tuple[str, str], list[float]] = {}
    sample_prime_tokens: dict[tuple[str, str], int] = {}

    for r in results_list:
        key = (r.sample_id, r.prime_name)
        if key not in sample_prime_ces:
            sample_prime_ces[key] = []
            sample_prime_tokens[key] = r.n_tokens
        sample_prime_ces[key].append(r.ce)
        results.results[r.prime_name]["all_ces"].append(r.ce)

    # Aggregate per-sample results (mean across runs)
    for (sample_id, prime_name), ces in sample_prime_ces.items():
        mean_ce = sum(ces) / len(ces)
        results.results[prime_name]["per_sample"].append(
            {
                "sample_id": sample_id,
                "ce": mean_ce,  # Mean across runs
                "ce_all_runs": ces if n_runs > 1 else None,  # All run values
                "n_tokens": sample_prime_tokens[(sample_id, prime_name)],
            }
        )

    # Compute means, CIs, and deltas
    for prime_name, data in results.results.items():
        all_ces = data["all_ces"]
        data["mean_ce"] = sum(all_ces) / len(all_ces) if all_ces else float("nan")
        data["ci_95"] = _compute_95_ci(all_ces) if n_runs > 1 else 0.0
        del data["all_ces"]  # Don't save all values to JSON

    baseline_mean = results.results["baseline"]["mean_ce"]
    baseline_ci = results.results["baseline"]["ci_95"]

    for prime_name, data in results.results.items():
        if prime_name != "baseline":
            data["delta_from_baseline"] = data["mean_ce"] - baseline_mean
            # Propagate CI: sqrt(ci1^2 + ci2^2) for difference of means
            data["delta_ci_95"] = np.sqrt(data["ci_95"] ** 2 + baseline_ci ** 2)

    return results


def plot_pareto_frontier(
    interest_results: DistributionResults,
    baseline_results: DistributionResults,
    config: PrimingConfig,
    output_path: Path,
):
    """
    Generate the pareto frontier plot.

    X-axis: ΔCE on interest distribution (prime - baseline)
    Y-axis: ΔCE on baseline distribution (prime - baseline)

    If n_runs > 1, plots 95% confidence interval error bars in both directions.
    """
    # Color palette for groups (automatically assigned)
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(10, 8))

    # Check if we have confidence intervals
    has_ci = interest_results.n_runs > 1 or baseline_results.n_runs > 1

    # Plot each prime group
    for group_idx, group in enumerate(config.prime_groups):
        color = colors[group_idx % len(colors)]

        x_vals = []
        y_vals = []
        x_errs = []
        y_errs = []
        labels = []

        for prime_name in group.primes.keys():
            full_name = f"{group.name}_{prime_name}"
            if full_name in interest_results.results:
                x = interest_results.results[full_name].get("delta_from_baseline", 0)
                y = baseline_results.results[full_name].get("delta_from_baseline", 0)
                x_vals.append(x)
                y_vals.append(y)
                labels.append(prime_name)

                # Get CIs if available
                x_errs.append(interest_results.results[full_name].get("delta_ci_95", 0))
                y_errs.append(baseline_results.results[full_name].get("delta_ci_95", 0))

        if x_vals:
            if has_ci:
                # Plot with error bars
                ax.errorbar(
                    x_vals,
                    y_vals,
                    xerr=x_errs,
                    yerr=y_errs,
                    fmt="o-",
                    color=color,
                    label=group.name,
                    markersize=8,
                    capsize=3,
                    capthick=1,
                    elinewidth=1,
                    alpha=0.8,
                )
            else:
                ax.plot(x_vals, y_vals, "o-", color=color, label=group.name, markersize=8)

            # Annotate points
            for i, label in enumerate(labels):
                ax.annotate(
                    label,
                    (x_vals[i], y_vals[i]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=7,
                    color=color,
                )

    # Reference lines
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)

    # Diagonal y=x line
    lims = [
        max(ax.get_xlim()[0], ax.get_ylim()[0]),
        min(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.3, label="y = x")

    ax.set_xlabel("ΔCE on Interest Distribution (prime - baseline)", fontsize=12)
    ax.set_ylabel("ΔCE on Baseline Distribution (prime - baseline)", fontsize=12)

    title = "Priming Effects: Interest vs Baseline Distribution"
    if has_ci:
        title += f" (n={max(interest_results.n_runs, baseline_results.n_runs)} runs, 95% CI)"
    ax.set_title(title, fontsize=14)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved pareto plot to {output_path}")
    plt.close()


async def run_priming_experiment(
    interest_dir: Path,
    baseline_dir: Path,
    config: PrimingConfig,
    output_dir: Path,
    max_concurrent: int = 10,
    n_runs: int = 1,
) -> dict:
    """
    Run priming experiment comparing two distributions.

    Args:
        interest_dir: Folder with run-N/step-M/messages.json (distribution of interest).
        baseline_dir: Folder with *.json files (baseline distribution).
        config: PrimingConfig with model, tokenizer, and prime groups.
        output_dir: Where to save results and plots.
        max_concurrent: Maximum concurrent API requests.
        n_runs: Number of times to run each sample for CI estimation.

    Returns:
        Results dict with both distributions' results.
    """
    # Load distributions
    print("Loading interest distribution...")
    interest_samples = _load_interest_distribution(interest_dir)
    print(f"  Loaded {len(interest_samples)} samples")

    print("Loading baseline distribution...")
    baseline_samples = _load_baseline_distribution(baseline_dir)
    print(f"  Loaded {len(baseline_samples)} samples")

    if not interest_samples:
        raise ValueError(f"No samples found in interest distribution: {interest_dir}")
    if not baseline_samples:
        raise ValueError(f"No samples found in baseline distribution: {baseline_dir}")

    # Get all primes
    all_primes = _get_all_primes(config)
    print(f"\nPrimes to evaluate: {list(all_primes.keys())}")
    if n_runs > 1:
        print(f"Running {n_runs} times per sample for CI estimation")
    print(f"Max concurrent requests: {max_concurrent}")

    # Initialize
    client = get_async_together_client()
    tokenizer = get_tokenizer(config.tokenizer)

    # Compute CE for both distributions
    interest_results = await _compute_ce_for_distribution(
        "interest",
        interest_samples,
        all_primes,
        config,
        client,
        tokenizer,
        max_concurrent,
        n_runs=n_runs,
    )

    baseline_results = await _compute_ce_for_distribution(
        "baseline",
        baseline_samples,
        all_primes,
        config,
        client,
        tokenizer,
        max_concurrent,
        n_runs=n_runs,
    )

    # Print summary
    ci_suffix = f" (±95% CI)" if n_runs > 1 else ""
    print(f"\n=== Interest Distribution Summary ==={ci_suffix}")
    for name, data in interest_results.results.items():
        ci_str = f" ± {data['ci_95']:.4f}" if n_runs > 1 else ""
        if name == "baseline":
            print(f"  {name}: mean CE = {data['mean_ce']:.4f}{ci_str}")
        else:
            delta_ci = f" ± {data.get('delta_ci_95', 0):.4f}" if n_runs > 1 else ""
            print(
                f"  {name}: mean CE = {data['mean_ce']:.4f}{ci_str} "
                f"(delta: {data.get('delta_from_baseline', 0):+.4f}{delta_ci})"
            )

    print(f"\n=== Baseline Distribution Summary ==={ci_suffix}")
    for name, data in baseline_results.results.items():
        ci_str = f" ± {data['ci_95']:.4f}" if n_runs > 1 else ""
        if name == "baseline":
            print(f"  {name}: mean CE = {data['mean_ce']:.4f}{ci_str}")
        else:
            delta_ci = f" ± {data.get('delta_ci_95', 0):.4f}" if n_runs > 1 else ""
            print(
                f"  {name}: mean CE = {data['mean_ce']:.4f}{ci_str} "
                f"(delta: {data.get('delta_from_baseline', 0):+.4f}{delta_ci})"
            )

    # Generate plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_pareto_frontier(
        interest_results,
        baseline_results,
        config,
        output_dir / "ce_pareto.png",
    )

    # Save results
    output = {
        "model": config.model,
        "tokenizer": config.tokenizer,
        "prime_location": config.prime_location,
        "n_runs": n_runs,
        "interest_distribution": {
            "dir": str(interest_dir),
            "n_samples": interest_results.n_samples,
            "n_runs": interest_results.n_runs,
            "results": interest_results.results,
        },
        "baseline_distribution": {
            "dir": str(baseline_dir),
            "n_samples": baseline_results.n_samples,
            "n_runs": baseline_results.n_runs,
            "results": baseline_results.results,
        },
    }

    results_path = output_dir / "ce_priming_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {results_path}")

    return output
