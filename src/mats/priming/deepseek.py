"""
DeepSeek R1 specific prompt construction for priming experiments.

This module handles the DeepSeek-specific <think>...</think> reasoning format
and tool call formatting for constructing prompts that include reasoning traces.
"""

from transformers import AutoTokenizer

# Default tokenizer for DeepSeek R1
TOKENIZER_MODEL = "deepseek-ai/DeepSeek-R1-0528"


def get_tokenizer(model: str | None = None):
    """
    Load the DeepSeek tokenizer.

    Args:
        model: Tokenizer model identifier. Defaults to TOKENIZER_MODEL.

    Returns:
        Huggingface AutoTokenizer instance.
    """
    return AutoTokenizer.from_pretrained(model or TOKENIZER_MODEL)


def construct_prompt_with_reasoning(messages: list[dict], tokenizer) -> str:
    """
    Build full prompt including <think>...</think> reasoning block.

    This manually constructs the prompt to preserve the reasoning content,
    which the standard chat template would strip out.

    Args:
        messages: List of message dicts from messages.json.
        tokenizer: Huggingface tokenizer for the model.

    Returns:
        Full prompt string with reasoning included.
    """
    # Find the index of the last assistant message
    last_assistant_idx = None
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            last_assistant_idx = i

    # Convert messages EXCEPT the last assistant (we'll handle that manually)
    chat_messages = []
    last_assistant_msg = None
    for i, msg in enumerate(messages):
        if i == last_assistant_idx:
            last_assistant_msg = msg
            continue  # Skip - we'll append manually

        chat_msg = {"role": msg["role"], "content": msg.get("content", "")}
        if msg.get("tool_calls"):
            chat_msg["tool_calls"] = msg["tool_calls"]
        if msg.get("tool_call_id"):
            chat_msg["tool_call_id"] = msg["tool_call_id"]
        chat_messages.append(chat_msg)

    # Apply chat template for all messages except last assistant
    prompt = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Manually append the last assistant message with reasoning
    if last_assistant_msg:
        reasoning = last_assistant_msg.get("reasoning", "")
        content = last_assistant_msg.get("content", "")
        tool_calls = last_assistant_msg.get("tool_calls", [])

        # Build the assistant turn with reasoning
        prompt += f"<think>\n{reasoning}\n</think>"
        if content:
            prompt += content

        # Add tool calls if present (DeepSeek-specific format)
        if tool_calls:
            prompt += "<｜tool▁calls▁begin｜>"
            for tool in tool_calls:
                prompt += (
                    f"<｜tool▁call▁begin｜>{tool['type']}<｜tool▁sep｜>"
                    f"{tool['function']['name']}\n```json\n"
                    f"{tool['function']['arguments']}\n```<｜tool▁call▁end｜>"
                )
            prompt += "<｜tool▁calls▁end｜><｜end▁of▁sentence｜>"

    return prompt


def find_reasoning_bounds(tokens: list[str]) -> tuple[int, int]:
    """
    Find the start and end indices of the <think>...</think> block.

    Args:
        tokens: List of token strings.

    Returns:
        (start_idx, end_idx) tuple. If not found, returns (0, len(tokens)).
    """
    start_idx = None
    end_idx = None

    for i, token in enumerate(tokens):
        if "<think>" in token and start_idx is None:
            start_idx = i
        if "</think>" in token:
            end_idx = i + 1
            break

    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(tokens)

    return start_idx, end_idx
