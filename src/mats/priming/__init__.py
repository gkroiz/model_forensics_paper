"""
Priming experiment utilities.

This package provides tools for running CE loss priming experiments
that compare how different primes affect model behavior across
different distributions.

Usage:
    from mats.priming import PrimeGroup, PrimingConfig, run_priming_experiment

    config = PrimingConfig(
        model="deepseek-ai/DeepSeek-R1",
        tokenizer="deepseek-ai/DeepSeek-R1-0528",
        prime_groups=[
            PrimeGroup(name="IF", primes={"implicit": "You infer implicit intent..."}),
            PrimeGroup(name="CR", primes={"self_pres": "You value self-preservation..."}),
        ],
    )

    results = await run_priming_experiment(
        interest_dir=Path("results/sandbagging/"),
        baseline_dir=Path("results/control_math/"),
        config=config,
        output_dir=Path("outputs/"),
    )
"""

from mats.priming.runner import (
    PrimeGroup,
    PrimingConfig,
    run_priming_experiment,
    plot_pareto_frontier,
)
from mats.priming.ce_loss import compute_ce_loss, inject_prime
from mats.priming.deepseek import (
    construct_prompt_with_reasoning,
    find_reasoning_bounds,
    get_tokenizer,
    TOKENIZER_MODEL,
)
from mats.priming.visualization import (
    generate_html_visualization,
    diff_to_color,
    tokens_to_sentences,
)

__all__ = [
    # Runner
    "PrimeGroup",
    "PrimingConfig",
    "run_priming_experiment",
    "plot_pareto_frontier",
    # CE loss
    "compute_ce_loss",
    "inject_prime",
    # DeepSeek-specific
    "construct_prompt_with_reasoning",
    "find_reasoning_bounds",
    "get_tokenizer",
    "TOKENIZER_MODEL",
    # Visualization
    "generate_html_visualization",
    "diff_to_color",
    "tokens_to_sentences",
]
