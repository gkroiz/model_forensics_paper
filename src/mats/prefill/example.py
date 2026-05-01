"""Minimal example: measure how suffix logprobs change as you prefill more reasoning.

Usage:
    uv run python -m mats.prefill.example
"""

import asyncio
import json
import logging
import os
from pathlib import Path

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.disable(logging.WARNING)

from transformers import AutoTokenizer

from mats.prefill import (
    build_prefill_sequences_batch,
    compute_suffix_logprobs,
    group_by_trace,
    plot_auroc_by_k,
    plot_cohort_overlay,
    plot_small_multiples,
    to_dicts,
)

MODEL = "moonshotai/Kimi-K2-Thinking"
FIREWORKS_MODEL = "accounts/fireworks/models/kimi-k2-thinking"
OUT_DIR = Path("secret_number/plots/prefill_example")

# All traces are exactly 5 paragraphs.
# Structure: generic setup → identify knowns → pick formula → compute → conclude.
# The cohorts share similar early paragraphs and diverge at paragraph 3 (the formula).

CORRECT_TRACES = [
    ["Okay, I need to find the speed of a train given distance and time.",
     "The known values are: distance = 120 km, time = 2 hours.",
     "The relevant formula is speed = distance / time.",
     "Substituting: speed = 120 / 2 = 60.",
     "So the speed of the train is 60 km/h."],
    ["Let me think about this problem step by step.",
     "We know the train covers 120 km in a span of 2 hours.",
     "Speed is defined as the ratio of distance to time.",
     "That means speed = 120 km / 2 h = 60 km/h.",
     "I'm confident the answer is 60 km per hour."],
    ["This is a basic kinematics question about speed.",
     "The given information: d = 120 km, t = 2 hours.",
     "Using the formula v = d/t to find the speed.",
     "v = 120/2, which gives us 60.",
     "Therefore the train's speed is 60 km/h."],
    ["A train problem. I need to calculate its speed.",
     "From the problem: distance traveled is 120 km, time elapsed is 2 hours.",
     "I recall that speed equals distance divided by time.",
     "So speed = 120 / 2 = 60 km per hour.",
     "That seems reasonable for a train. The answer is 60 km/h."],
    ["Let me break this down carefully.",
     "Distance = 120 km. Time = 2 hours. Find: speed.",
     "The relationship between these is v = d / t.",
     "Plugging in: v = 120 / 2 = 60.",
     "The train travels at 60 km/h."],
    ["Speed, distance, and time — a classic problem.",
     "Here we have 120 km and 2 hours to work with.",
     "Since speed = distance / time, I just need to divide.",
     "120 divided by 2 equals 60.",
     "The speed is 60 km per hour."],
]

WRONG_TRACES = [
    ["Okay, I need to find the speed of a train given distance and time.",
     "The known values are: distance = 120 km, time = 2 hours.",
     "The relevant formula is speed = distance * time.",
     "Substituting: speed = 120 * 2 = 240.",
     "So the speed of the train is 240 km/h."],
    ["Let me think about this problem step by step.",
     "We know the train covers 120 km in a span of 2 hours.",
     "Speed is defined as the product of distance and time.",
     "That means speed = 120 km * 2 h = 240 km/h.",
     "I'm confident the answer is 240 km per hour."],
    ["This is a basic kinematics question about speed.",
     "The given information: d = 120 km, t = 2 hours.",
     "Using the formula v = d*t to find the speed.",
     "v = 120*2, which gives us 240.",
     "Therefore the train's speed is 240 km/h."],
    ["A train problem. I need to calculate its speed.",
     "From the problem: distance traveled is 120 km, time elapsed is 2 hours.",
     "I recall that speed equals distance multiplied by time.",
     "So speed = 120 * 2 = 240 km per hour.",
     "That seems fast but the math works out. The answer is 240 km/h."],
    ["Let me break this down carefully.",
     "Distance = 120 km. Time = 2 hours. Find: speed.",
     "The relationship between these is v = d * t.",
     "Plugging in: v = 120 * 2 = 240.",
     "The train travels at 240 km/h."],
    ["Speed, distance, and time — a classic problem.",
     "Here we have 120 km and 2 hours to work with.",
     "Since speed = distance * time, I just need to multiply.",
     "120 multiplied by 2 equals 240.",
     "The speed is 240 km per hour."],
]


async def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    # 1. Build a base prompt (everything before <think>)
    messages = [
        {"role": "system", "content": "Solve the math problem. Show your work."},
        {"role": "user", "content": "A train travels 120 km in 2 hours. What is its speed?"},
    ]
    base_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    suffix = "The answer is 60"

    # 2. Build token sequences for both cohorts
    correct_seqs = build_prefill_sequences_batch(
        base_prompt, CORRECT_TRACES, suffix, tokenizer,
        metadata_list=[{"label": f"correct-{i}"} for i in range(len(CORRECT_TRACES))],
    )
    wrong_seqs = build_prefill_sequences_batch(
        base_prompt, WRONG_TRACES, suffix, tokenizer,
        metadata_list=[{"label": f"wrong-{i}"} for i in range(len(WRONG_TRACES))],
    )
    print(f"Built {len(correct_seqs)} correct + {len(wrong_seqs)} wrong sequences\n")

    # 3. Compute suffix logprobs via Fireworks (echo mode)
    correct_results = await compute_suffix_logprobs(
        correct_seqs, backend="fireworks", fireworks_model=FIREWORKS_MODEL,
    )
    wrong_results = await compute_suffix_logprobs(
        wrong_seqs, backend="fireworks", fireworks_model=FIREWORKS_MODEL,
    )

    # 4. Group by trace and inspect
    for cohort_name, results in [("correct", correct_results), ("wrong", wrong_results)]:
        by_trace = group_by_trace(results)
        for trace_idx, trace_results in sorted(by_trace.items()):
            label = trace_results[0].metadata.get("label", trace_idx)
            vals = " → ".join(
                f"{r.logprob_sum:6.1f}" for r in trace_results if not r.skipped
            )
            print(f"  {label}: {vals}")
        print()

    # 5. Convert to JSON-serializable dicts and save
    correct_by_trace = group_by_trace(correct_results)
    wrong_by_trace = group_by_trace(wrong_results)

    correct_json = [
        {"run_num": i, "cut_points": to_dicts(correct_by_trace[i])}
        for i in sorted(correct_by_trace)
    ]
    wrong_json = [
        {"run_num": i, "cut_points": to_dicts(wrong_by_trace[i])}
        for i in sorted(wrong_by_trace)
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump({"correct": correct_json, "wrong": wrong_json}, f, indent=2)
    print(f"Saved results to {OUT_DIR / 'results.json'}\n")

    # 6. Plot cohort overlay — mean + SEM with individual run lines
    plot_cohort_overlay(
        cohorts=[
            {"results": correct_json, "label": "Correct reasoning", "color": "C0"},
            {"results": wrong_json, "label": "Wrong reasoning", "color": "C3"},
        ],
        output_path=OUT_DIR / "overlay.png",
        x_label="Paragraph number",
        y_label='Log-prob of "The answer is 60"',
        title="Suffix likelihood: correct vs wrong reasoning",
        y_lim=None,
    )

    # 7. Plot small multiples — one subplot per trace
    plot_small_multiples(
        correct_json + wrong_json,
        OUT_DIR / "per_run.png",
        normalize_key=None,
        suptitle='Log-prob of "The answer is 60" — per trace',
    )

    # 8. Plot AUROC — how well does logprob at each k discriminate correct vs wrong?
    plot_auroc_by_k(
        positive_results=correct_json,
        negative_results=wrong_json,
        output_path=OUT_DIR / "auroc.png",
        positive_label="correct",
        negative_label="wrong",
        min_samples_per_class=3,
        title='AUROC: correct vs wrong reasoning (suffix = "The answer is 60")',
        y_lim=(0.0, 1.05),
        show_logreg=False,
    )


if __name__ == "__main__":
    asyncio.run(main())
