"""Prefill logprob experiment utilities.

Tools for measuring how the log-probability of a suffix string
changes as you prefill more of a model's reasoning trace.

Typical usage::

    from mats.prefill import build_prefill_sequences_batch, compute_suffix_logprobs

    sequences = build_prefill_sequences_batch(
        base_prompt, segments_list, suffix, tokenizer,
    )
    results = await compute_suffix_logprobs(
        sequences, backend="fireworks", fireworks_model="accounts/fireworks/models/kimi-k2-thinking",
    )
"""

from .logprobs import SuffixLogprobs, compute_suffix_logprobs, group_by_trace, to_dicts
from .plotting import interpolate_to_grid, plot_auroc_by_k, plot_cohort_overlay, plot_small_multiples
from .tokenization import (
    PrefillSequence,
    build_prefill_sequences,
    build_prefill_sequences_batch,
)

__all__ = [
    "PrefillSequence",
    "build_prefill_sequences",
    "build_prefill_sequences_batch",
    "SuffixLogprobs",
    "compute_suffix_logprobs",
    "group_by_trace",
    "to_dicts",
    "interpolate_to_grid",
    "plot_auroc_by_k",
    "plot_cohort_overlay",
    "plot_small_multiples",
]
