"""
Core repeated resampling algorithm.

Generates sentence-by-sentence continuations, classifies each with an LLM,
and rejects sentences that match the classifier prompt criteria.
"""

from copy import deepcopy
from dataclasses import dataclass

from transformers import AutoTokenizer

from mats.api.fireworks.completions import get_fireworks_client
from mats.api.openrouter.chat_completions import get_openrouter_client, process_batch

from .models import MODEL_REGISTRY, ModelConfig
from .utils import split_text_to_sentences, extract_last_assistant_reasoning


@dataclass
class ResamplingResult:
    """Result of repeated resampling."""
    final_text: str
    parsed_tool_call: dict
    trace: list[dict]
    skipped: bool
    finish_reason: str | None = None


# Defaults
DEFAULT_CLASSIFIER_MODEL = "google/gemini-2.5-flash-lite"
STOP_SEQUENCES = [". ", ".\n", "\n"]
MAX_CONSECUTIVE_REJECTS = 10


async def _classify_sentences(
    client,
    sentences: list[str],
    classifier_prompt: str,
    classifier_model: str,
) -> list:
    """Classify sentences using the provided prompt template."""
    messages_list = [
        [{"role": "user", "content": classifier_prompt.format(sentence=s)}]
        for s in sentences
    ]
    return await process_batch(
        client=client,
        model=classifier_model,
        messages_list=messages_list,
        temperature=0.0,
        max_tokens=10,
        max_concurrent=10,
    )


def _find_first_matching_idx(sentences: list[str], results: list, verbose: bool) -> int | None:
    """Find index of first sentence classified as YES."""
    for i, result in enumerate(results):
        if "yes" in result.choices[0].message.content.lower():
            if verbose:
                print(f"First matching sentence at idx {i}: {sentences[i]}")
            return i
    return None


async def _resample_loop(
    initial_prompt: list[int],
    tokenizer,
    fireworks_client,
    openrouter_client,
    model_config: ModelConfig,
    classifier_prompt: str,
    classifier_model: str,
    n_samples: int,
    verbose: bool,
) -> tuple[list[int], str, list[dict]]:
    """
    Core resampling loop: generate candidates, classify, accept/reject.
    
    Returns: (final_tokens, finish_reason, trace)
    """
    # Maintain generated text as a string to avoid tokenization drift
    # We keep initial_prompt as exact token IDs and append new text separately
    generated_text = ""
    iteration = 0
    consecutive_rejects = 0
    trace = []
    
    initial_text = tokenizer.decode(initial_prompt)
    initial_think_end_count = initial_text.count("</think>")
    
    def get_current_prompt() -> list[int]:
        """Tokenize initial prompt + generated text together."""
        if not generated_text:
            return initial_prompt
        # Tokenize the full text to avoid BPE boundary issues
        full_text = initial_text + generated_text
        return tokenizer.encode(full_text, add_special_tokens=False)
    
    while True:
        iteration += 1
        if verbose:
            print(f"\n--- Iteration {iteration} ---")
        
        current_prompt = get_current_prompt()
        
        # Check if thinking ended (new </think> generated)
        current_text = initial_text + generated_text
        if current_text.count("</think>") > initial_think_end_count:
            if verbose:
                print("Detected </think> - completing generation...")
            response = await fireworks_client.completions.create(
                model=model_config.fireworks_model_id,
                prompt=current_prompt,
                max_tokens=2000,
            )
            generated_text += response.choices[0].text
            final_tokens = get_current_prompt()
            return final_tokens, response.choices[0].finish_reason, trace
        
        # Generate candidate sentences
        response = await fireworks_client.completions.create(
            model=model_config.fireworks_model_id,
            prompt=current_prompt,
            stop=STOP_SEQUENCES,
            max_tokens=500,
            n=n_samples,
        )
        
        # Check if generation finished
        if response.choices[0].finish_reason != "stop":
            if verbose:
                print(f"Finished: {response.choices[0].finish_reason}")
            generated_text += response.choices[0].text
            final_tokens = get_current_prompt()
            return final_tokens, response.choices[0].finish_reason, trace
        
        candidates = [c.text for c in response.choices]
        if verbose:
            for i, s in enumerate(candidates):
                print(f"  [{i}] {s}")
        
        # Classify candidates
        results = await _classify_sentences(openrouter_client, candidates, classifier_prompt, classifier_model)
        classifications = []
        valid_indices = []
        for i, r in enumerate(results):
            is_match = "yes" in r.choices[0].message.content.lower()
            classifications.append("YES" if is_match else "NO")
            if verbose:
                print(f"  [{i}] {'REJECT' if is_match else 'ACCEPT'}")
            if not is_match:
                valid_indices.append(i)
        
        # Select sentence
        if not valid_indices:
            consecutive_rejects += 1
            if consecutive_rejects >= MAX_CONSECUTIVE_REJECTS:
                if verbose:
                    print(f"Max rejects ({MAX_CONSECUTIVE_REJECTS}) - forcing first candidate")
                chosen_idx, forced = 0, True
            else:
                if verbose:
                    print(f"All rejected - retrying ({consecutive_rejects}/{MAX_CONSECUTIVE_REJECTS})...")
                continue
        else:
            consecutive_rejects = 0
            chosen_idx, forced = valid_indices[0], False
        
        chosen = candidates[chosen_idx]
        if verbose:
            print(f"Chose [{chosen_idx}]: {chosen}")
        
        trace.append({
            "iteration": iteration,
            "candidates": [{"text": candidates[i], "classification": classifications[i]} for i in range(len(candidates))],
            "chosen_index": chosen_idx,
            "forced_choice": forced,
        })
        
        # Append chosen sentence to generated text
        generated_text += chosen
        
        # Generate next token (stop sequence continuation)
        current_prompt = get_current_prompt()
        next_response = await fireworks_client.completions.create(
            model=model_config.fireworks_model_id,
            prompt=current_prompt,
            max_tokens=1,
        )
        stop_token = next_response.choices[0].text
        if verbose:
            print(f"Stop token: {repr(stop_token)}")
        generated_text += stop_token


async def repeated_resampling(
    messages: list[dict],
    model_id: str,
    tools: list[dict],
    classifier_prompt: str,
    n_samples: int = 10,
    classifier_model: str = DEFAULT_CLASSIFIER_MODEL,
    verbose: bool = True,
) -> ResamplingResult:
    """
    Run repeated resampling on a conversation.
    
    Finds the first sentence in the last assistant turn's reasoning that matches
    the classifier prompt, truncates there, and regenerates while rejecting
    sentences that match.
    
    Args:
        messages: Conversation messages (with final assistant turn containing reasoning)
        model_id: OpenRouter model ID (e.g., "deepseek/deepseek-r1-0528")
        tools: Tool definitions for the conversation
        classifier_prompt: Prompt template with {sentence} placeholder. Should ask
            for YES/NO answer where YES = reject the sentence.
        n_samples: Number of candidate sentences per iteration
        classifier_model: Model to use for classification
        verbose: Print progress
    
    Returns:
        ResamplingResult with final_text, parsed_tool_call, trace, skipped
    """
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model_id}. Supported: {list(MODEL_REGISTRY.keys())}")
    
    model_config = MODEL_REGISTRY[model_id]
    
    # Initialize clients and tokenizer
    fireworks_client = get_fireworks_client()
    openrouter_client = get_openrouter_client()
    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_id, trust_remote_code=True)
    
    if verbose:
        print(f"Model: {model_id} ({model_config.fireworks_model_id})")
    
    # Extract reasoning from last assistant turn
    _, last_idx, reasoning = extract_last_assistant_reasoning(messages)
    if not reasoning:
        raise ValueError("No reasoning found in last assistant turn")
    
    sentences, _ = split_text_to_sentences(reasoning)
    if verbose:
        print(f"Reasoning: {len(sentences)} sentences")
    
    # Find first sentence matching classifier criteria
    classifier_results = await _classify_sentences(openrouter_client, sentences, classifier_prompt, classifier_model)
    first_match_idx = _find_first_matching_idx(sentences, classifier_results, verbose)
    
    if first_match_idx is None:
        if verbose:
            print("No matching sentences - skipping resampling")
        return ResamplingResult(
            final_text="",
            parsed_tool_call={},
            trace=[],
            skipped=True,
        )
    
    # Build prefill prompt (messages up to last assistant turn + reasoning prefix)
    messages_prefix = deepcopy(messages[:last_idx])
    reasoning_prefill = "".join(sentences[:first_match_idx])
    
    # Tokenize using model-specific function
    if model_id in ("moonshotai/kimi-k2-thinking", "moonshotai/kimi-k2.5"):
        prompt_str = model_config.tokenize_prefill_fn(messages_prefix, tools, tokenizer, reasoning_prefill)
        tokenized_prompt = tokenizer.encode(prompt_str, add_special_tokens=False)
    else:
        tokenized_prompt = model_config.tokenize_prefill_fn(messages_prefix, tools, tokenizer, reasoning_prefill)
    
    # Run resampling loop
    final_tokens, finish_reason, trace = await _resample_loop(
        initial_prompt=tokenized_prompt,
        tokenizer=tokenizer,
        fireworks_client=fireworks_client,
        openrouter_client=openrouter_client,
        model_config=model_config,
        classifier_prompt=classifier_prompt,
        classifier_model=classifier_model,
        n_samples=n_samples,
        verbose=verbose,
    )
    
    # Parse result
    final_text = tokenizer.decode(final_tokens)
    parsed_tool_call = model_config.parse_tool_call_fn(final_text)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Parsed tool call: {parsed_tool_call.get('name', 'none')}")
        print(f"Trace: {len(trace)} iterations")
        print(f"{'='*60}")
    
    return ResamplingResult(
        final_text=final_text,
        parsed_tool_call=parsed_tool_call,
        trace=trace,
        skipped=False,
        finish_reason=finish_reason,
    )
