"""
Cross-entropy loss computation utilities for priming experiments.

This module provides generic CE loss computation and prime injection
that works across different models and environments.
"""

from copy import deepcopy


def compute_ce_loss(logprobs: list[float]) -> float:
    """
    Compute cross-entropy loss from logprobs.

    CE loss = mean of -logprobs (lower = model finds sequence more likely).

    Args:
        logprobs: List of log probabilities for tokens.

    Returns:
        Mean negative log probability (CE loss).
    """
    valid_logprobs = [lp for lp in logprobs if lp is not None]
    if not valid_logprobs:
        return float("nan")
    return -sum(valid_logprobs) / len(valid_logprobs)


def inject_prime(
    messages: list[dict],
    prime: str,
    location: str = "user",
) -> list[dict]:
    """
    Inject a prime into messages.

    Args:
        messages: List of message dicts.
        prime: Prime text to inject (empty string = no change).
        location: Where to inject - "user" (prepend to first user message)
                  or "system" (append to system message).

    Returns:
        New list of messages with prime injected.
    """
    messages = deepcopy(messages)

    # Empty prime = no change (useful for baseline)
    if not prime:
        return messages

    if location == "system":
        # Append to system message
        for msg in messages:
            if msg["role"] == "system":
                msg["content"] = msg["content"].rstrip() + "\n\n" + prime
                break
    elif location == "user":
        # Prepend to first user message
        for msg in messages:
            if msg["role"] == "user":
                msg["content"] = prime + "\n\n" + msg["content"]
                break
    else:
        raise ValueError(f"Unknown location: {location}")

    return messages
