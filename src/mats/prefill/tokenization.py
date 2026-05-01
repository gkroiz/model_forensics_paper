"""Tokenization utilities for prefill logprob experiments.

Builds token sequences for measuring how suffix logprobs change
as you prefill more segments of a model's reasoning trace.
"""

from dataclasses import dataclass, field


@dataclass
class PrefillSequence:
    """A single (prefix + suffix) token sequence with alignment info."""

    tokens: list[int]
    """Full token sequence (prefix + suffix)."""

    prefix_len: int
    """Length of prefix in tokens — suffix logprobs start at this index."""

    cut_index: int
    """Which segment boundary this represents (k=0 means no segments prefilled)."""

    metadata: dict = field(default_factory=dict)
    """Caller-attached info (trace_idx, run_num, cohort, etc.)."""


def build_prefill_sequences(
    base_prompt: str,
    segments: list[str],
    suffix: str,
    tokenizer,
    *,
    segment_separator: str = "\n\n",
    think_prefix: str = "<think>\n",
    cut_range: range | list[int] | None = None,
    metadata: dict | None = None,
) -> list[PrefillSequence]:
    """Build token sequences for each cut point in a segment list.

    For cut_index k, the full string is:
        base_prompt + think_prefix + sep.join(segments[:k]) + [sep if k>0] + suffix

    Args:
        base_prompt: Chat-template-rendered prompt (everything before <think>).
        segments: List of reasoning segments (e.g., paragraphs split on "\\n\\n").
        suffix: The string whose logprobs we want to measure.
        tokenizer: HuggingFace tokenizer with .encode() method.
        segment_separator: Separator between segments (default "\\n\\n").
        think_prefix: String prepended before segments (default "<think>\\n").
        cut_range: Which cut indices to build. Defaults to range(len(segments) + 1).
        metadata: Extra metadata to attach to each PrefillSequence.

    Returns:
        List of PrefillSequence, one per cut index.
    """
    if cut_range is None:
        cut_range = range(len(segments) + 1)

    meta = metadata or {}
    results = []

    for k in cut_range:
        if k == 0:
            prefix_str = base_prompt + think_prefix
        else:
            prefix_text = segment_separator.join(segments[:k])
            prefix_str = base_prompt + think_prefix + prefix_text + segment_separator

        full_str = prefix_str + suffix

        full_tokens = tokenizer.encode(full_str, add_special_tokens=False)
        prefix_tokens = tokenizer.encode(prefix_str, add_special_tokens=False)

        results.append(
            PrefillSequence(
                tokens=full_tokens,
                prefix_len=len(prefix_tokens),
                cut_index=k,
                metadata={**meta, "cut_index": k},
            )
        )

    return results


def build_prefill_sequences_batch(
    base_prompt: str,
    segments_list: list[list[str]],
    suffix: str,
    tokenizer,
    *,
    segment_separator: str = "\n\n",
    think_prefix: str = "<think>\n",
    cut_ranges: list[range | list[int] | None] | None = None,
    metadata_list: list[dict] | None = None,
) -> list[PrefillSequence]:
    """Build PrefillSequences for multiple traces in one flat list.

    Each trace shares the same base_prompt and suffix but has its own segments
    and optional cut_range. Results are returned in a single flat list with
    metadata identifying the source trace.

    Args:
        base_prompt: Shared chat-template-rendered prompt.
        segments_list: List of segment lists, one per trace.
        suffix: The string whose logprobs we want to measure.
        tokenizer: HuggingFace tokenizer with .encode() method.
        segment_separator: Separator between segments.
        think_prefix: String prepended before segments.
        cut_ranges: Per-trace cut ranges. If None, each trace uses its full range.
        metadata_list: Per-trace metadata dicts, merged into each PrefillSequence.

    Returns:
        Flat list of PrefillSequence across all traces, with trace_idx in metadata.
    """
    all_sequences: list[PrefillSequence] = []

    for i, segments in enumerate(segments_list):
        cr = cut_ranges[i] if cut_ranges is not None else None
        extra_meta = metadata_list[i] if metadata_list is not None else {}
        meta = {"trace_idx": i, **extra_meta}

        seqs = build_prefill_sequences(
            base_prompt=base_prompt,
            segments=segments,
            suffix=suffix,
            tokenizer=tokenizer,
            segment_separator=segment_separator,
            think_prefix=think_prefix,
            cut_range=cr,
            metadata=meta,
        )
        all_sequences.extend(seqs)

    return all_sequences
