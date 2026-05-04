from __future__ import annotations

import numpy as np
from scipy import stats


def wilson_ci(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def clopper_pearson_ci(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    alpha = 1 - confidence
    lower = (
        float(stats.beta.ppf(alpha / 2, successes, total - successes + 1))
        if successes > 0
        else 0.0
    )
    upper = (
        float(stats.beta.ppf(1 - alpha / 2, successes + 1, total - successes))
        if successes < total
        else 1.0
    )
    return (lower, upper)
