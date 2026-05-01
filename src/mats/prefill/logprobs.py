"""Backend-agnostic logprob computation for prefill experiments.

Dispatches to Fireworks (echo+logprobs) or Tinker (compute_logprobs_async)
and assembles suffix logprobs from the raw results.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from .tokenization import PrefillSequence


@dataclass
class SuffixLogprobs:
    """Logprob result for a single prefix+suffix sequence."""

    cut_index: int
    logprob_sum: float
    mean_logprob: float | None
    n_suffix_tokens: int
    skipped: bool
    metadata: dict = field(default_factory=dict)


async def compute_suffix_logprobs(
    sequences: list[PrefillSequence],
    backend: Literal["fireworks", "tinker"],
    *,
    fireworks_model: str | None = None,
    fireworks_client=None,
    tinker_model: str | None = None,
    tinker_client=None,
    max_concurrent: int = 20,
) -> list[SuffixLogprobs]:
    """Compute suffix logprobs for a batch of prefill sequences.

    Args:
        sequences: PrefillSequences from build_prefill_sequences[_batch].
        backend: "fireworks" or "tinker".
        fireworks_model: Fireworks model ID (required if backend="fireworks").
        fireworks_client: Pre-created Fireworks client (created if None).
        tinker_model: Tinker model name (required if backend="tinker").
        tinker_client: Pre-created Tinker client (created if None).
        max_concurrent: Maximum concurrent API calls.

    Returns:
        List of SuffixLogprobs, one per input sequence, in the same order.
    """
    all_tokens = [s.tokens for s in sequences]

    if backend == "fireworks":
        raw_logprobs = await _compute_fireworks(
            all_tokens,
            model=fireworks_model,
            client=fireworks_client,
            max_concurrent=max_concurrent,
        )
    elif backend == "tinker":
        raw_logprobs = await _compute_tinker(
            all_tokens,
            model=tinker_model,
            client=tinker_client,
            max_concurrent=max_concurrent,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'fireworks' or 'tinker'.")

    # Assemble suffix logprobs
    results = []
    for i, seq in enumerate(sequences):
        logprobs = raw_logprobs[i]
        suffix_logprobs = logprobs[seq.prefix_len :]
        valid = [lp for lp in suffix_logprobs if lp is not None]

        if not valid:
            results.append(
                SuffixLogprobs(
                    cut_index=seq.cut_index,
                    logprob_sum=float("nan"),
                    mean_logprob=None,
                    n_suffix_tokens=0,
                    skipped=True,
                    metadata=seq.metadata,
                )
            )
        else:
            lp_sum = sum(valid)
            results.append(
                SuffixLogprobs(
                    cut_index=seq.cut_index,
                    logprob_sum=lp_sum,
                    mean_logprob=lp_sum / len(valid),
                    n_suffix_tokens=len(valid),
                    skipped=False,
                    metadata=seq.metadata,
                )
            )

    return results


async def _compute_fireworks(
    all_tokens: list[list[int]],
    *,
    model: str | None,
    client=None,
    max_concurrent: int = 20,
) -> list[list[float | None]]:
    """Compute logprobs via Fireworks echo mode."""
    from mats.api.fireworks.completions import (
        get_fireworks_client,
        process_batch as fw_process_batch,
    )

    if model is None:
        raise ValueError("fireworks_model is required when backend='fireworks'")
    if client is None:
        client = get_fireworks_client()

    responses = await fw_process_batch(
        client=client,
        model=model,
        prompts=all_tokens,
        max_tokens=1,
        logprobs=1,
        echo=True,
        max_concurrent=max_concurrent,
        return_exceptions=True,
    )

    # Report failures
    failures = [i for i, r in enumerate(responses) if isinstance(r, Exception)]
    if failures:
        print(
            f"WARNING: {len(failures)}/{len(responses)} requests failed. "
            f"First error: {responses[failures[0]]}"
        )

    # Extract token_logprobs, truncated to input length (echo may add 1 extra)
    result = []
    for i, r in enumerate(responses):
        if isinstance(r, Exception):
            result.append([None] * len(all_tokens[i]))
        else:
            result.append(
                r.choices[0].logprobs.token_logprobs[: len(all_tokens[i])]
            )

    return result


async def _compute_tinker(
    all_tokens: list[list[int]],
    *,
    model: str | None,
    client=None,
    max_concurrent: int = 64,
) -> list[list[float | None]]:
    """Compute logprobs via Tinker."""
    from mats.api.tinker import get_client, process_logprobs_batch

    if model is None:
        raise ValueError("tinker_model is required when backend='tinker'")
    if client is None:
        client = get_client(model)

    return await process_logprobs_batch(
        client, all_tokens, max_concurrent=max_concurrent
    )


def group_by_trace(
    results: list[SuffixLogprobs],
) -> dict[int, list[SuffixLogprobs]]:
    """Group SuffixLogprobs by trace_idx from metadata.

    Returns:
        Dict mapping trace_idx -> list of SuffixLogprobs sorted by cut_index.
    """
    groups: dict[int, list[SuffixLogprobs]] = defaultdict(list)
    for r in results:
        idx = r.metadata.get("trace_idx", 0)
        groups[idx].append(r)

    return {k: sorted(v, key=lambda x: x.cut_index) for k, v in groups.items()}


def to_dicts(results: list[SuffixLogprobs]) -> list[dict]:
    """Convert SuffixLogprobs to JSON-serializable dicts.

    Compatible with the existing results.json format:
        {"k": ..., "logprob_sum": ..., "mean_logprob": ..., "n_suffix_tokens": ..., "skipped": ...}
    """
    out = []
    for r in results:
        d: dict = {
            "k": r.cut_index,
            "logprob_sum": r.logprob_sum if not math.isnan(r.logprob_sum) else None,
            "skipped": r.skipped,
        }
        if not r.skipped:
            d["mean_logprob"] = r.mean_logprob
            d["n_suffix_tokens"] = r.n_suffix_tokens
        else:
            d["n_suffix_tokens"] = 0
        out.append(d)
    return out
