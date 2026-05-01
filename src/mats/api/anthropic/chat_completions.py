"""Anthropic chat completions API utils.

Mirrors the interface of mats.api.openrouter.chat_completions so callers
can swap providers without changing access patterns. Structured output
responses are wrapped to expose `.choices[0].message.parsed` like OpenAI.
"""

import asyncio
import os
from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm_asyncio

load_dotenv()


# =============================================================================
# OpenAI-compatible wrapper for structured output responses
# =============================================================================


@dataclass
class _ParsedMessage:
    parsed: BaseModel


@dataclass
class _Choice:
    message: _ParsedMessage


@dataclass
class _CompatResponse:
    """Wraps Anthropic ParsedMessage to match OpenAI's response.choices[0].message.parsed."""
    choices: list[_Choice]
    raw: object  # original Anthropic response

    @classmethod
    def from_anthropic(cls, response) -> "_CompatResponse":
        # Anthropic: response.content[0].parsed_output
        parsed = response.content[0].parsed_output
        return cls(
            choices=[_Choice(message=_ParsedMessage(parsed=parsed))],
            raw=response,
        )


# =============================================================================
# Client
# =============================================================================


def get_client() -> anthropic.AsyncAnthropic:
    """Get an AsyncAnthropic client."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in .env file!")
    return anthropic.AsyncAnthropic(api_key=api_key)


# =============================================================================
# API call
# =============================================================================


@retry(
    retry=retry_if_exception_type(
        (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APITimeoutError)
    ),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
async def call_api(
    client: anthropic.AsyncAnthropic,
    model: str,
    messages: list,
    response_format: type[BaseModel] | None = None,
    system: str | None = None,
    temperature: float = 1.0,
    max_tokens: int = 5000,
    top_p: float = 1.0,
    top_k: int | None = None,
    stop_sequences: list[str] | None = None,
):
    """
    Make an Anthropic chat completion API call.

    Args:
        client: AsyncAnthropic client.
        model: Model identifier (e.g., 'claude-haiku-4-5-20251001').
        messages: Chat messages. System messages are extracted automatically.
        response_format: Pydantic model for structured output.
        system: System prompt. If None, extracted from messages list.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling.
        stop_sequences: Stop sequences.

    Returns:
        Response object. If response_format is set, returns a _CompatResponse
        with .choices[0].message.parsed matching the OpenAI convention.
    """
    # Extract system message from messages list if not provided separately
    if system is None:
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append(m)
        messages = filtered

    # Anthropic doesn't allow both temperature and top_p simultaneously
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p != 1.0:
        kwargs.pop("temperature", None)
        kwargs["top_p"] = top_p
    if system is not None:
        kwargs["system"] = system
    if top_k is not None:
        kwargs["top_k"] = top_k
    if stop_sequences is not None:
        kwargs["stop_sequences"] = stop_sequences

    if response_format is not None:
        kwargs["output_format"] = response_format
        response = await client.messages.parse(**kwargs)
        return _CompatResponse.from_anthropic(response)
    else:
        return await client.messages.create(**kwargs)


# =============================================================================
# Batch processing
# =============================================================================


async def process_one(
    client: anthropic.AsyncAnthropic,
    model: str,
    messages: list,
    semaphore: asyncio.Semaphore,
    response_format: type[BaseModel] | None = None,
    temperature: float = 1.0,
    max_tokens: int = 5000,
    top_p: float = 1.0,
    **kwargs,
):
    """Process a single request with semaphore-based concurrency control."""
    async with semaphore:
        return await call_api(
            client=client,
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs,
        )


async def process_batch(
    client: anthropic.AsyncAnthropic,
    model: str,
    messages_list: list,
    response_format: type[BaseModel] | None = None,
    temperature: float = 1.0,
    max_tokens: int = 5000,
    max_concurrent: int = 10,
    top_p: float = 1.0,
    return_exceptions: bool = False,
    **kwargs,
) -> list:
    """
    Process all requests concurrently with a semaphore.

    Args:
        client: AsyncAnthropic client.
        model: Model identifier.
        messages_list: List of chat message lists.
        response_format: Pydantic model for structured output.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        max_concurrent: Maximum concurrent requests.
        top_p: Nucleus sampling threshold.
        return_exceptions: If True, exceptions are returned instead of raised.
        **kwargs: Additional parameters forwarded to call_api.

    Returns:
        List of response objects (or _CompatResponse for structured output).
        If return_exceptions=True, failed requests return Exception objects.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    coroutines = [
        process_one(
            client=client,
            model=model,
            messages=m,
            semaphore=semaphore,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs,
        )
        for m in messages_list
    ]

    if return_exceptions:
        async def wrap_with_progress(coro, pbar):
            try:
                result = await coro
                pbar.update(1)
                return result
            except Exception as e:
                pbar.update(1)
                return e

        from tqdm import tqdm
        pbar = tqdm(total=len(coroutines))
        wrapped = [wrap_with_progress(c, pbar) for c in coroutines]
        results = await asyncio.gather(*wrapped)
        pbar.close()
        return results
    else:
        return await tqdm_asyncio.gather(*coroutines)
