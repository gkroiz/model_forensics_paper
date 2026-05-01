"""Together AI completions API utils."""

import asyncio
import os

os.environ["TOGETHER_NO_BANNER"] = "1"

from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from together import AsyncTogether, Together
from together.error import RateLimitError, APIConnectionError, APIError
from tqdm.asyncio import tqdm_asyncio

load_dotenv()


def get_together_client() -> Together:
    """Get a synchronous Together client."""
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY not found in .env file!")
    return Together(api_key=api_key)


def get_async_together_client() -> AsyncTogether:
    """Get an async Together client."""
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY not found in .env file!")
    return AsyncTogether(api_key=api_key)


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
async def call_api(
    client: AsyncTogether,
    model: str,
    prompt: str,
    max_tokens: int = 1,
    logprobs: int | None = None,
    echo: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
):
    """
    Make a completions API call to Together.

    Args:
        client: AsyncTogether client.
        model: Model identifier.
        prompt: Text prompt to complete.
        max_tokens: Maximum tokens in response.
        logprobs: Number of logprobs to return per token.
        echo: If True, return logprobs for prompt tokens.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.

    Returns:
        Together completion response object.
    """
    return await client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        logprobs=logprobs,
        echo=echo,
        temperature=temperature,
        top_p=top_p,
    )


async def process_one(
    client: AsyncTogether,
    model: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    max_tokens: int = 1,
    logprobs: int | None = None,
    echo: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
):
    """Process a single request with semaphore-based concurrency control."""
    async with semaphore:
        return await call_api(
            client=client,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            logprobs=logprobs,
            echo=echo,
            temperature=temperature,
            top_p=top_p,
        )


async def process_batch(
    client: AsyncTogether,
    model: str,
    prompts: list[str],
    max_concurrent: int = 10,
    max_tokens: int = 1,
    logprobs: int | None = None,
    echo: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    show_progress: bool = True,
) -> list:
    """
    Process all requests concurrently with a semaphore.

    Args:
        client: AsyncTogether client.
        model: Model identifier.
        prompts: List of text prompts.
        max_concurrent: Maximum concurrent requests.
        max_tokens: Maximum tokens in response.
        logprobs: Number of logprobs to return per token.
        echo: If True, return logprobs for prompt tokens.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        show_progress: Whether to show tqdm progress bar.

    Returns:
        List of Together completion response objects.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    coroutines = [
        process_one(
            client=client,
            model=model,
            prompt=p,
            semaphore=semaphore,
            max_tokens=max_tokens,
            logprobs=logprobs,
            echo=echo,
            temperature=temperature,
            top_p=top_p,
        )
        for p in prompts
    ]
    if show_progress:
        return await tqdm_asyncio.gather(*coroutines)
    return await asyncio.gather(*coroutines)
