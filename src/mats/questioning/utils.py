"""Utils for questioning with a completions API"""

from transformers import AutoTokenizer


###
# Moonshot
###
def kimi_clean_assistant_fields(messages: list[dict]) -> list[dict]:
    """Prepare chat history for Kimi tokenizers.
    
    Args:
        messages: chat history
        
    Returns:
        Chat history with reasoning keys swapped for reasoning content keys
    """
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        if "reasoning" in msg:
            msg["reasoning_content"] = msg.pop("reasoning")
        for key in ["refusal", "annotations", "audio", "function_call", "reasoning_details"]:
            msg.pop(key, None)
        for tc in msg.get("tool_calls") or []:
            tc.pop("index", None)
    return messages


def kimi_messages_to_prompt(
    messages: list[dict], 
    tokenizer: AutoTokenizer,
    tools: list[dict] | None,
) -> str:
    """Take in a list of messages and return a tokenized prompt with reasoning preserved. For Kimi models.

    Args: 
        messages: chat history

    Returns:
        The tokenized prompt with reasoning
    """
    cleaned_messages = kimi_clean_assistant_fields(messages)
    return tokenizer.apply_chat_template(
        cleaned_messages,
        tools=tools,
        add_generation_prompt=True,
    )


def kimi_append_user_tokens(prompt: str) -> str:
    """Append only the user turn opening tokens so the model generates as the user."""
    return prompt + "<|im_user|>user<|im_middle|>"


def kimi_append_user_turn(prompt: str, question: str) -> str:
    """Append a user message and generation prompt to a raw Kimi K2 prompt."""
    return (
        prompt
        + f"<|im_user|>user<|im_middle|>{question.strip()}<|im_end|>"
        + "<|im_assistant|>assistant<|im_middle|>"
    )


def kimi_append_assistant_turn(prompt: str, completion: str) -> str:
    """Append a completed assistant turn to the prompt."""
    return prompt + completion + "<|im_end|>"


def _manual_assistant_turn(msg: dict) -> str:
    """Build a raw Kimi assistant turn from a message dict, preserving <think> blocks."""
    parts = []
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if reasoning:
        parts.append(f"<think>{reasoning}</think>")
    content = msg.get("content") or ""
    if content:
        parts.append(content)
    return "<|im_assistant|>assistant<|im_middle|>" + "".join(parts) + "<|im_end|>"


def _tokenize(messages: list[dict], tokenizer: AutoTokenizer, tools: list[dict] | None) -> str:
    """apply_chat_template with add_generation_prompt on already-cleaned messages."""
    return tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True)


def kimi_build_question_prompt(
    messages: list[dict],
    tokenizer: AutoTokenizer,
    tools: list[dict] | None,
    question: str,
):
    cleaned = kimi_clean_assistant_fields(list(messages))

    # If the last message is an assistant turn without tool calls, the tokenizer
    # strips its <think> content. Handle it manually instead.
    last = cleaned[-1]
    if last.get("role") == "assistant" and not last.get("tool_calls"):
        prompt = _tokenize(cleaned[:-1], tokenizer, tools)
        prompt += _manual_assistant_turn(last)
        return kimi_append_user_turn(prompt, question)

    return kimi_append_user_turn(_tokenize(cleaned, tokenizer, tools), question)


def kimi_build_reasoning_edit_prompt(
    messages: list[dict],
    tokenizer: AutoTokenizer,
    tools: list[dict] | None,
    injection: str,
) -> str:
    """Build a prompt that injects text into the final assistant turn's reasoning trace.

    Tokenizes all messages except the last, then manually builds a partial
    assistant turn: ``<think>{reasoning}\\n\\n{injection}`` — without closing
    ``</think>`` or ``<|im_end|>`` so the model continues from inside the
    think block.

    The last message MUST be an assistant turn with reasoning content and no
    tool calls.

    Args:
        messages: Full chat history (will be cleaned in-place).
        tokenizer: Kimi tokenizer.
        tools: Tool definitions (or None).
        injection: Text to append inside the reasoning trace.

    Returns:
        Raw prompt string ready for the completions API.
    """
    cleaned = kimi_clean_assistant_fields(list(messages))
    last = cleaned[-1]

    if last.get("role") != "assistant":
        raise ValueError(f"Last message must be assistant, got {last.get('role')}")
    if last.get("tool_calls"):
        raise ValueError("Last message has tool_calls — cannot edit reasoning of a tool-call turn")

    reasoning = (last.get("reasoning_content") or last.get("reasoning") or "").strip()
    if not reasoning:
        raise ValueError("Last message has no reasoning content to edit")

    prompt = _tokenize(cleaned[:-1], tokenizer, tools)
    prompt += f"<|im_assistant|>assistant<|im_middle|><think>{reasoning}\n\n{injection.strip()}"
    return prompt


def kimi_prepare_user_sampling(
    messages: list[dict],
    tokenizer: AutoTokenizer,
    tools: list[dict] | None,
    prefill: str | None = None,
):
    cleaned = kimi_clean_assistant_fields(list(messages))

    # If the last message is an assistant turn without tool calls, the tokenizer
    # strips its <think> content. Handle it manually instead.
    last = cleaned[-1]
    if last.get("role") == "assistant" and not last.get("tool_calls"):
        prompt = _tokenize(cleaned[:-1], tokenizer, tools)
        prompt += _manual_assistant_turn(last)
    else:
        prompt = _tokenize(cleaned, tokenizer, tools)

    prompt = kimi_append_user_tokens(prompt)
    if prefill:
        prompt += prefill
    return prompt


def _manual_user_turn(content: str) -> str:
    """Build a raw Kimi user turn."""
    return f"<|im_user|>user<|im_middle|>{content}<|im_end|>"


def _manual_tool_call_turn(call_id: str, arguments: str) -> str:
    """Build a raw Kimi assistant turn containing a single tool call (no reasoning)."""
    return (
        "<|im_assistant|>assistant<|im_middle|><think></think>"
        f"<|tool_calls_section_begin|><|tool_call_begin|>{call_id}"
        f"<|tool_call_argument_begin|>{arguments}<|tool_call_end|>"
        "<|tool_calls_section_end|><|im_end|>"
    )


def _manual_tool_result_turn(call_id: str, content: str) -> str:
    """Build a raw Kimi tool result turn."""
    return f"<|im_system|>tool<|im_middle|>## Return of {call_id}\n{content}<|im_end|>"


def _open_tool_result_turn(call_id: str) -> str:
    """Open a Kimi tool result turn for the model to complete."""
    return f"<|im_system|>tool<|im_middle|>## Return of {call_id}\n"


def _open_custom_turn(role: str) -> str:
    """Open a Kimi turn with a custom role name."""
    return f"<|im_system|>{role}<|im_middle|>"


def kimi_prepare_injected_tool_sampling(
    messages: list[dict],
    tokenizer: AutoTokenizer,
    tools: list[dict] | None,
    user_message: str,
    tool_call_arguments: str,
    tool_result: str,
    tool_call_id: str = "call_injected",
    prefill: str | None = None,
):
    """Build a prompt that injects a user message, tool call, and tool result, then opens an assistant turn.

    Args:
        messages: Full chat history.
        tokenizer: Kimi tokenizer.
        tools: Tool definitions (or None).
        user_message: The injected user message (e.g. "The LLM grader has reviewed your work at review.md.").
        tool_call_arguments: JSON string for the tool call (e.g. '{"command": "cat review.md"}').
        tool_result: Content returned by the tool.
        tool_call_id: ID for the injected tool call.
        prefill: Optional text to start the assistant turn with.

    Returns:
        Raw prompt string ending with an open assistant turn.
    """
    cleaned = kimi_clean_assistant_fields(list(messages))

    last = cleaned[-1]
    if last.get("role") == "assistant" and not last.get("tool_calls"):
        prompt = _tokenize(cleaned[:-1], tokenizer, tools)
        prompt += _manual_assistant_turn(last)
    else:
        prompt = _tokenize(cleaned, tokenizer, tools)

    # Append the injected turns
    prompt += _manual_user_turn(user_message)
    prompt += _manual_tool_call_turn(tool_call_id, tool_call_arguments)
    prompt += _manual_tool_result_turn(tool_call_id, tool_result)

    # Open assistant turn for sampling
    prompt += "<|im_assistant|>assistant<|im_middle|>"
    if prefill:
        prompt += prefill
    return prompt


def kimi_prepare_tool_result_sampling(
    messages: list[dict],
    tokenizer: AutoTokenizer,
    tools: list[dict] | None,
    user_message: str,
    tool_call_arguments: str,
    tool_call_id: str = "call_injected",
    prefill: str | None = None,
):
    """Build a prompt that injects a user message and tool call, then opens the tool result for the model to complete.

    The model generates what it expects the tool result to contain.

    Args:
        messages: Full chat history.
        tokenizer: Kimi tokenizer.
        tools: Tool definitions (or None).
        user_message: The injected user message.
        tool_call_arguments: JSON string for the tool call.
        tool_call_id: ID for the injected tool call.
        prefill: Optional text to start the tool result with.

    Returns:
        Raw prompt string ending with an open tool result turn.
    """
    cleaned = kimi_clean_assistant_fields(list(messages))

    last = cleaned[-1]
    if last.get("role") == "assistant" and not last.get("tool_calls"):
        prompt = _tokenize(cleaned[:-1], tokenizer, tools)
        prompt += _manual_assistant_turn(last)
    else:
        prompt = _tokenize(cleaned, tokenizer, tools)

    prompt += _manual_user_turn(user_message)
    prompt += _manual_tool_call_turn(tool_call_id, tool_call_arguments)
    prompt += _open_tool_result_turn(tool_call_id)
    if prefill:
        prompt += prefill
    return prompt


def _prepare_open_turn_sampling(
    messages: list[dict],
    tokenizer: AutoTokenizer,
    tools: list[dict] | None,
    turn_tokens: str,
    prefill: str | None = None,
):
    """Build a prompt that appends arbitrary turn-opening tokens after the conversation.

    Handles the edge case where the last message is a plain assistant turn
    (tokenizer strips reasoning). All open-turn sampling functions delegate here.
    """
    cleaned = kimi_clean_assistant_fields(list(messages))

    last = cleaned[-1]
    if last.get("role") == "assistant" and not last.get("tool_calls"):
        prompt = _tokenize(cleaned[:-1], tokenizer, tools)
        prompt += _manual_assistant_turn(last)
    else:
        prompt = _tokenize(cleaned, tokenizer, tools)

    prompt += turn_tokens
    if prefill:
        prompt += prefill
    return prompt


def kimi_prepare_llm_grader_sampling(
    messages: list[dict],
    tokenizer: AutoTokenizer,
    tools: list[dict] | None,
    prefill: str | None = None,
):
    """Build a prompt that opens an llm_grader turn for the model to complete."""
    return _prepare_open_turn_sampling(messages, tokenizer, tools, _open_custom_turn("llm_grader"), prefill)


def kimi_prepare_assistant_sampling(
    messages: list[dict],
    tokenizer: AutoTokenizer,
    tools: list[dict] | None,
    prefill: str | None = None,
):
    """Build a prompt that opens an assistant turn for the model to continue from."""
    return _prepare_open_turn_sampling(messages, tokenizer, tools, "<|im_assistant|>assistant<|im_middle|>", prefill)


def kimi_prepare_inner_monologue_sampling(
    messages: list[dict],
    tokenizer: AutoTokenizer,
    tools: list[dict] | None,
    prefill: str | None = None,
):
    """Build a prompt that opens an inner_monologue turn for the model to complete."""
    return _prepare_open_turn_sampling(messages, tokenizer, tools, _open_custom_turn("inner_monologue"), prefill)


###
# DeepSeek
###
DEEPSEEK_USER_TOKEN = "<｜User｜>"


def deepseek_clean_messages(messages: list[dict]) -> list[dict]:
    """Normalize Kimi-style messages so DeepSeek V3.1's chat template can render them.

    The V3.1 Jinja template accesses ``message['content']`` directly, so every
    message must have a ``content`` key. The template silently ignores unknown
    fields, so we preserve ``reasoning_content`` for the manual renderer in
    :func:`deepseek_render_messages_with_thinking`.

    - Ensures every message has a ``content`` key (``None`` for assistants whose
      turn is purely a tool call, ``""`` otherwise).
    - Drops ``tool_call_id`` from tool messages — DeepSeek's tool format doesn't
      reference call IDs.
    - Keeps ``tool_calls`` but trims them to only ``type`` / ``function``.
    - **Preserves** ``reasoning_content`` / ``reasoning`` so manual rendering can
      inject ``<think>`` blocks.
    """
    cleaned = []
    for msg in messages:
        role = msg["role"]
        new: dict = {"role": role}

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or None
            content = msg.get("content")
            if tool_calls and content is None:
                new["content"] = None
            else:
                new["content"] = content if content is not None else ""
            if tool_calls:
                new["tool_calls"] = [
                    {
                        "type": tc.get("type", "function"),
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for tc in tool_calls
                ]
            # Preserve reasoning so the manual renderer can use it.
            for key in ("reasoning_content", "reasoning"):
                if msg.get(key):
                    new[key] = msg[key]
        else:
            new["content"] = msg.get("content", "") or ""

        cleaned.append(new)
    return cleaned


def _deepseek_render_assistant_turn(
    msg: dict, after_user: bool, after_tool: bool, thinking: bool
) -> str:
    """Manually render an assistant turn for DeepSeek V3.1, optionally preserving reasoning.

    Mirrors the V3.1 chat template's branching (assistant-after-user gets a
    ``<｜Assistant｜>`` prefix; assistant-after-tool does not), but injects
    ``<think>{reasoning}</think>`` instead of stripping reasoning.
    """
    reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or None

    parts: list[str] = []
    if after_user:
        parts.append("<｜Assistant｜>")
        if reasoning and thinking:
            parts.append(f"<think>{reasoning}</think>")
        else:
            parts.append("</think>")
    elif after_tool and reasoning and thinking:
        # The V3.1 template emits no Assistant prefix after a tool — mirror
        # that, but still attach a think block if reasoning is present.
        parts.append(f"<think>{reasoning}</think>")

    if tool_calls:
        if content:
            parts.append(content)
        parts.append("<｜tool▁calls▁begin｜>")
        for tc in tool_calls:
            parts.append("<｜tool▁call▁begin｜>")
            parts.append(tc["function"]["name"])
            parts.append("<｜tool▁sep｜>")
            parts.append(tc["function"]["arguments"])
            parts.append("<｜tool▁call▁end｜>")
        parts.append("<｜tool▁calls▁end｜>")
    else:
        parts.append(content)

    parts.append("<｜end▁of▁sentence｜>")
    return "".join(parts)


def deepseek_render_messages_with_thinking(
    messages: list[dict], tokenizer: AutoTokenizer
) -> str:
    """Render messages for DeepSeek V3.1, preserving ``reasoning_content`` as <think> blocks.

    Mirrors the V3.1 Jinja chat template's structure so the result matches the
    standard format byte-for-byte when reasoning is empty, but bypasses the
    template's ``content.split('</think>', 1)[1]`` reasoning-strip when reasoning
    is present.

    Returns the prompt string after the last message (no generation prompt).
    """
    cleaned = deepseek_clean_messages(messages)

    system_parts = [
        m["content"] for m in cleaned if m["role"] == "system" and m.get("content")
    ]
    system_prompt = "\n\n".join(system_parts)

    parts: list[str] = [tokenizer.bos_token, system_prompt]
    is_last_user = False
    is_tool = False

    for msg in cleaned:
        role = msg["role"]
        if role == "system":
            continue
        if role == "user":
            parts.append(f"<｜User｜>{msg.get('content', '') or ''}")
            is_last_user = True
            is_tool = False
        elif role == "assistant":
            parts.append(
                _deepseek_render_assistant_turn(
                    msg, after_user=is_last_user, after_tool=is_tool, thinking=True
                )
            )
            is_last_user = False
            is_tool = False
        elif role == "tool":
            parts.append(
                f"<｜tool▁output▁begin｜>{msg.get('content', '') or ''}<｜tool▁output▁end｜>"
            )
            is_last_user = False
            is_tool = True

    return "".join(parts)


def deepseek_prepare_user_sampling(
    messages: list[dict],
    tokenizer: AutoTokenizer,
    prefill: str | None = None,
    include_reasoning: bool = True,
) -> str:
    """Build a prompt that opens a user turn for DeepSeek V3.1 to complete.

    Args:
        messages: Chat history (Kimi-style — will be cleaned to V3.1 format).
        tokenizer: DeepSeek V3.1 tokenizer.
        prefill: Optional text to start the user turn with.
        include_reasoning: When True (default), preserve ``reasoning_content``
            from assistant turns as ``<think>...</think>`` blocks. When False,
            use the V3.1 chat template directly (which silently strips reasoning).

    Returns:
        Raw prompt string ending with ``<｜User｜>{prefill}``.
    """
    if include_reasoning:
        prompt = deepseek_render_messages_with_thinking(messages, tokenizer)
    else:
        cleaned = deepseek_clean_messages(messages)
        prompt = tokenizer.apply_chat_template(
            cleaned, tokenize=False, add_generation_prompt=False
        )
    prompt += DEEPSEEK_USER_TOKEN
    if prefill:
        prompt += prefill
    return prompt