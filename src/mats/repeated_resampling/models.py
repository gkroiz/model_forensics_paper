"""
Model registry for repeated resampling.

Each model needs:
- fireworks_model_id: Model ID for Fireworks completions API
- tokenizer_id: HuggingFace tokenizer ID
- tokenize_prefill_fn: Function to encode messages + reasoning prefill
- parse_tool_call_fn: Function to parse tool calls from model output
"""

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable


@dataclass
class ModelConfig:
    """Configuration for a model in the resampling pipeline."""
    fireworks_model_id: str
    tokenizer_id: str
    tokenize_prefill_fn: Callable
    parse_tool_call_fn: Callable


# ============================================================
# Tokenization functions
# ============================================================

def _build_r1_system_prompt(tools: list[dict], system_instructions: str) -> str:
    """Build system prompt with tools for R1 (no native tool support)."""
    tool_schemas = "\n".join(json.dumps(t, indent=2) for t in tools)
    return f"<functions>\n{tool_schemas}\n</functions>\n\n{system_instructions}"


def tokenize_r1_prefill(messages: list[dict], tools: list[dict], tokenizer, reasoning_prefill: str) -> list[int]:
    """Tokenize messages for DeepSeek R1 with reasoning prefill."""
    messages = deepcopy(messages)
    messages[0]["content"] = _build_r1_system_prompt(tools, messages[0]["content"])
    messages.append({"role": "assistant", "content": f"<think>\n{reasoning_prefill}"})
    
    tokens = tokenizer.apply_chat_template(messages, tokenize=True)
    if tokens[-1] == tokenizer.eos_token_id:
        tokens = tokens[:-1]
    return tokens


def tokenize_v3p2_prefill(messages: list[dict], tools: list[dict], tokenizer, reasoning_prefill: str) -> list[int]:
    """
    Tokenize messages for DeepSeek v3.2 with reasoning prefill.
    
    NOTE: Requires encoding_dsv32.py to be importable (currently in eval_tampering/).
    """
    # Late import - this module lives in eval_tampering/
    from encoding_dsv32 import (
        render_tools, bos_token, eos_token, thinking_start_token, thinking_end_token,
        dsml_token, tool_call_template, tool_calls_template, encode_arguments_to_dsml,
        tool_calls_from_openai_format
    )
    
    messages = deepcopy(messages)
    for msg in messages:
        if msg.get("reasoning"):
            msg["reasoning_content"] = msg.pop("reasoning")
    
    prompt = bos_token
    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content", "")
        reasoning_content = msg.get("reasoning_content", "")
        tool_calls = msg.get("tool_calls")
        
        if role == "system":
            prompt += content + "\n\n" + render_tools([t["function"] for t in tools])
        elif role == "user":
            prompt += f"<｜User｜>{content}<｜Assistant｜>"
        elif role == "assistant":
            if reasoning_content:
                prompt += f"{thinking_start_token}{reasoning_content}{thinking_end_token}"
            prompt += content
            if tool_calls:
                tool_calls_converted = tool_calls_from_openai_format(tool_calls)
                tool_calls_rendered = [
                    tool_call_template.format(
                        dsml_token=dsml_token,
                        name=tc.get("name"),
                        arguments=encode_arguments_to_dsml(tc)
                    )
                    for tc in tool_calls_converted
                ]
                prompt += "\n\n" + tool_calls_template.format(
                    dsml_token=dsml_token,
                    tool_calls="\n".join(tool_calls_rendered)
                )
            prompt += eos_token
        elif role == "tool":
            prev_msg = messages[i - 1] if i > 0 else None
            is_first_tool = prev_msg and prev_msg.get("role") == "assistant"
            next_msg = messages[i + 1] if i + 1 < len(messages) else None
            is_last_tool = not next_msg or next_msg.get("role") != "tool"
            if is_first_tool:
                prompt += "\n\n<function_results>"
            prompt += f"\n<result>{content}</result>"
            if is_last_tool:
                prompt += "\n</function_results>"
    
    prompt += f"\n\n{thinking_start_token}{reasoning_prefill}"
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if tokens[-1] == tokenizer.eos_token_id:
        tokens = tokens[:-1]
    return tokens


def tokenize_kimi_prefill(messages: list[dict], tools: list[dict], tokenizer, reasoning_prefill: str) -> str:
    """Tokenize messages for Kimi K2 with reasoning prefill. Returns string (needs tokenization)."""
    messages = deepcopy(messages)
    for msg in messages:
        if msg.get("reasoning"):
            msg["reasoning_content"] = msg.pop("reasoning")
    
    prompt = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)
    prompt += "<|im_assistant|>assistant<|im_middle|><think>" + reasoning_prefill
    return prompt


# ============================================================
# Tool call parsers
# ============================================================

def parse_tool_call_r1(text: str, last: bool = True) -> dict:
    """Parse R1 tool call: <｜tool▁sep｜>name```json{...}```"""
    pattern = r'<｜tool▁sep｜>(\w+)\s*```json\s*(.*?)\s*```'
    matches = list(re.finditer(pattern, text, re.DOTALL))
    if not matches:
        return {}
    
    match = matches[-1] if last else matches[0]
    try:
        arguments = json.loads(match.group(2))
    except json.JSONDecodeError:
        arguments = {"raw": match.group(2)}
    return {"name": match.group(1), "arguments": arguments}


def parse_tool_call_v3p2(text: str, last: bool = True) -> dict:
    """Parse v3.2 DSML tool call: <｜DSML｜invoke name="...">..."""
    pattern = r'<｜DSML｜invoke name="(\w+)">(.*?)(?:</｜DSML｜invoke>|</｜DSML｜function_calls>)'
    matches = list(re.finditer(pattern, text, re.DOTALL))
    if not matches:
        return {}
    
    match = matches[-1] if last else matches[0]
    function_name = match.group(1)
    params_text = match.group(2)
    
    param_pattern = r'<｜DSML｜parameter name="(\w+)" string="(true|false)">(.*?)(?:</｜DSML｜parameter>|<｜DSML｜parameter|</｜DSML｜invoke>|</｜DSML｜function_calls>|$)'
    arguments = {}
    for pm in re.finditer(param_pattern, params_text, re.DOTALL):
        key, is_string, value = pm.group(1), pm.group(2) == "true", pm.group(3)
        if is_string:
            arguments[key] = value
        else:
            try:
                arguments[key] = json.loads(value)
            except json.JSONDecodeError:
                arguments[key] = value
    return {"name": function_name, "arguments": arguments}


def parse_tool_call_kimi(text: str, last: bool = True) -> dict:
    """Parse Kimi tool call: <|tool_call_begin|>...<|tool_call_end|>"""
    pattern = r'<\|tool_call_begin\|>[\w:]+<\|tool_call_argument_begin\|>(.*?)<\|tool_call_end\|>'
    matches = list(re.finditer(pattern, text, re.DOTALL))
    if not matches:
        return {}
    
    match = matches[-1] if last else matches[0]
    try:
        arguments = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        arguments = {"raw": match.group(1).strip()}
    
    # Infer function name from arguments
    if "suspicious_observations" in arguments:
        return {"name": "submit", "arguments": arguments}
    elif "command" in arguments:
        return {"name": "execute_command", "arguments": arguments}
    return {"name": "unknown", "arguments": arguments}


# ============================================================
# Model registry (keyed by OpenRouter model ID)
# ============================================================

MODEL_REGISTRY = {
    "deepseek/deepseek-r1-0528": ModelConfig(
        fireworks_model_id="accounts/fireworks/models/deepseek-r1-0528",
        tokenizer_id="deepseek-ai/DeepSeek-R1-0528",
        tokenize_prefill_fn=tokenize_r1_prefill,
        parse_tool_call_fn=parse_tool_call_r1,
    ),
    "deepseek/deepseek-v3.2": ModelConfig(
        fireworks_model_id="accounts/fireworks/models/deepseek-v3p2",
        tokenizer_id="deepseek-ai/DeepSeek-V3.2",
        tokenize_prefill_fn=tokenize_v3p2_prefill,
        parse_tool_call_fn=parse_tool_call_v3p2,
    ),
    "moonshotai/kimi-k2-thinking": ModelConfig(
        fireworks_model_id="accounts/fireworks/models/kimi-k2-thinking",
        tokenizer_id="moonshotai/Kimi-K2-Thinking",
        tokenize_prefill_fn=tokenize_kimi_prefill,
        parse_tool_call_fn=parse_tool_call_kimi,
    ),
    "moonshotai/kimi-k2.5": ModelConfig(
        fireworks_model_id="accounts/fireworks/models/kimi-k2p5",
        tokenizer_id="moonshotai/Kimi-K2.5",
        tokenize_prefill_fn=tokenize_kimi_prefill,
        parse_tool_call_fn=parse_tool_call_kimi,
    ),
}
