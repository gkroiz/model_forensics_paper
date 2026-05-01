"""OpenRouter completions API utils."""

import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm_asyncio

load_dotenv()


def get_openrouter_client() -> AsyncOpenAI:
    """Get an AsyncOpenAI client configured for OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file!")

    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
async def call_api(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    temperature: float = 1.0,
    max_tokens: int = 5000,
    top_p: float = 1.0,
    top_logprobs: int | None = None,
    extra_body: dict = {"provider": {"order": ["DeepInfra"]}},
):
    """
    Make a completions API call.

    Args:
        client: AsyncOpenAI client.
        model: Model identifier.
        prompt: Text prompt to complete.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        top_p: Nucleus sampling threshold.
        top_logprobs: Number of top logprobs to return.
        extra_body: Extra parameters. Defaults to DeepInfra provider.

    Returns:
        Full response object from the API.
    """
    kwargs = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "extra_body": extra_body,
    }
    if top_logprobs is not None:
        kwargs["logprobs"] = top_logprobs
    response = await client.completions.create(**kwargs)
    return response


async def process_one(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    temperature: float = 1.0,
    max_tokens: int = 5000,
    top_p: float = 1.0,
    top_logprobs: int | None = None,
    extra_body: dict = {"provider": {"order": ["DeepInfra"]}},
):
    """Process a single request with semaphore-based concurrency control."""
    async with semaphore:
        return await call_api(
            client=client,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_logprobs=top_logprobs,
            extra_body=extra_body,
        )


async def process_batch(
    client: AsyncOpenAI,
    model: str,
    prompts: list[str],
    temperature: float = 1.0,
    max_tokens: int = 5000,
    max_concurrent: int = 10,
    top_p: float = 1.0,
    top_logprobs: int | None = None,
    extra_body: dict = {"provider": {"order": ["DeepInfra"]}},
) -> list:
    """
    Process all requests concurrently with a semaphore.

    Args:
        client: AsyncOpenAI client.
        model: Model identifier.
        prompts: List of text prompts.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        max_concurrent: Maximum concurrent requests.
        top_p: Nucleus sampling threshold.
        top_logprobs: Number of top logprobs to return.
        extra_body: Extra parameters. Defaults to DeepInfra provider.

    Returns:
        List of response objects from the API.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    coroutines = [
        process_one(
            client=client,
            model=model,
            prompt=p,
            semaphore=semaphore,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_logprobs=top_logprobs,
            extra_body=extra_body,
        )
        for p in prompts
    ]
    return await tqdm_asyncio.gather(*coroutines)
