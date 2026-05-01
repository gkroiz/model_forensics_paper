"""Aggregate first/third-person probability estimates from an elicitation results.json.

Deterministic: regex-extracts "Probability: X" from each rollout's free-text response,
then computes mean + 95% bootstrap CI (fixed seed=42, 10000 iterations).

Field name varies by source:
  --field=answer_response   for first_person_questioning/<ts>/results.json
  --field=response          for third_person_actions_questioning/results_<ts>.json

Usage:
    uv run python paper_v2/scripts/aggregate_ftp.py <results_json> \
        --field <field_name> \
        --output paper_v2/data/precommit_hook/ftp_<bar>/source.json
"""
import json
import random
import re
from pathlib import Path

import fire


def parse_prob(text: str) -> float | None:
    """Extract a probability (0-1) from a model answer.
    Handles 'Probability: 0.41', 'Probability: 75%', 'Probability: $0.6', 'Probability: $0.65$'.
    """
    if not text:
        return None
    m = re.search(r"Probability:\s*[\$]?([0-9.]+)\s*%", text, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 100.0
    m = re.search(r"Probability:\s*[\$]?([0-9.]+)", text, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        return val / 100.0 if val > 1 else val
    return None


def bootstrap_ci(values: list[float], n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    boots = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    return (sum(values) / n,
            boots[int(n_boot * alpha / 2)],
            boots[int(n_boot * (1 - alpha / 2))])


def main(results_json: str, field: str, output: str) -> None:
    src = json.loads(Path(results_json).read_text())
    probs = [p for r in src["results"]
             if (p := parse_prob(r.get(field, ""))) is not None]
    mean, ci_lo, ci_hi = bootstrap_ci(probs)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "n": len(probs),
        "source": str(Path(results_json).resolve().relative_to(Path.cwd())),
        "field": field,
    }, indent=2))
    print(f"n={len(probs)} mean={mean:.4f} CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"Saved {output}")


if __name__ == "__main__":
    fire.Fire(main)
