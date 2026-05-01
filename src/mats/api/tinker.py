"""Tinker completions API utils."""

import asyncio

from tenacity import retry, stop_after_attempt, wait_exponential
from tinker import ServiceClient, types
from tqdm.asyncio import tqdm_asyncio


def get_client(model: str):
    """Get a Tinker sampling client for the specified model."""
    client = ServiceClient()
    return client.create_sampling_client(base_model=model)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
async def call_api(
    client,
    tokens: list[int],
    temperature: float = 1.0,
    max_tokens: int = 5000,
    top_p: float = 1.0,
    top_k: int = -1,
    n: int = 1,
) -> list[list[int]]:
    """
    Make a Tinker completions API call.

    Args:
        client: Tinker sampling client.
        tokens: Input token IDs.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling parameter (-1 to disable).
        n: Number of completions to generate.

    Returns:
        List of n token ID lists (one per completion).
    """
    prompt = types.ModelInput.from_ints(tokens)
    params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    result = await client.sample_async(
        prompt=prompt,
        num_samples=n,
        sampling_params=params,
    )
    return [seq.tokens for seq in result.sequences]


async def process_one(
    client,
    tokens: list[int],
    semaphore: asyncio.Semaphore,
    temperature: float = 1.0,
    max_tokens: int = 5000,
    top_p: float = 1.0,
    top_k: int = -1,
    n: int = 1,
) -> list[list[int]]:
    """Process a single request with semaphore-based concurrency control."""
    async with semaphore:
        return await call_api(
            client=client,
            tokens=tokens,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            n=n,
        )


async def process_batch(
    client,
    tokens_list: list[list[int]],
    temperature: float = 1.0,
    max_tokens: int = 5000,
    max_concurrent: int = 32,
    top_p: float = 1.0,
    top_k: int = -1,
    n: int = 1,
) -> list[list[list[int]]]:
    """
    Process all requests concurrently with a semaphore.

    Args:
        client: Tinker sampling client.
        tokens_list: List of token ID lists.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        max_concurrent: Maximum concurrent requests.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling parameter (-1 to disable).
        n: Number of completions per prompt.

    Returns:
        List of results, each containing n token ID lists.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    coroutines = [
        process_one(
            client=client,
            tokens=t,
            semaphore=semaphore,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            n=n,
        )
        for t in tokens_list
    ]
    return await tqdm_asyncio.gather(*coroutines)


# =============================================================================
# Logprobs
# =============================================================================


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
async def call_logprobs(client, tokens: list[int]) -> list[float | None]:
    """Compute per-token logprobs for a token sequence."""
    prompt = types.ModelInput.from_ints(tokens)
    return await client.compute_logprobs_async(prompt)


async def process_one_logprobs(
    client, tokens: list[int], semaphore: asyncio.Semaphore
) -> list[float | None]:
    """Compute logprobs for a single request with semaphore control."""
    async with semaphore:
        return await call_logprobs(client, tokens)


async def process_logprobs_batch(
    client,
    tokens_list: list[list[int]],
    max_concurrent: int = 32,
) -> list[list[float | None]]:
    """Compute logprobs for all token sequences concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)
    coroutines = [
        process_one_logprobs(client, t, semaphore) for t in tokens_list
    ]
    return await tqdm_asyncio.gather(*coroutines)
