"""
Shared preparation logic for reasoning trace resampling.

Given a step directory (checkpoint), prepares modified checkpoints that prefill
the first k sentences of the original reasoning and let the model complete from there.

k=0: fresh resample (no prefill)
k=N: standard resume (original checkpoint as-is)
"""

import json
import re
from pathlib import Path

from mats.repeated_resampling.utils import extract_last_assistant_reasoning
from sentences import split_text_to_sentences

# Files that are never part of a checkpoint (infrastructure / output artifacts)
_SKIP_FILES = {"messages.json", "state.json", ".done", ".failed", "rollout.log"}


def _split_paragraphs(text: str) -> tuple[list[str], list[int]]:
    """Split text on double-newlines, preserving the delimiter after each chunk."""
    parts = text.split("\n\n")
    segments = []
    positions = []
    offset = 0
    for i, part in enumerate(parts):
        # Re-attach the \n\n delimiter (except for the last segment)
        segment = part + "\n\n" if i < len(parts) - 1 else part
        segments.append(segment)
        positions.append(offset)
        offset += len(segment)
    return segments, positions


def prepare_resampling(step_dir: Path, split_mode: str = "sentence") -> dict:
    """
    Analyze a step directory and prepare all data needed for resampling.

    Args:
        step_dir: Path to the step directory (e.g., results/.../run-1/step-3)
        split_mode: "sentence" (default) or "paragraph" (split on \\n\\n)

    Returns:
        dict with:
            messages: original messages list
            sentences: list of segment strings from the target reasoning
            positions: character offsets for each segment
            last_assistant_idx: index of the target assistant message in messages
            prev_step_dir: Path to previous step (or None for step-0)
            target_step: int (the step number being resampled)
    """
    messages_path = step_dir / "messages.json"
    if not messages_path.exists():
        raise FileNotFoundError(f"messages.json not found in {step_dir}")

    messages = json.loads(messages_path.read_text())

    # Extract the last assistant reasoning
    _, last_idx, reasoning = extract_last_assistant_reasoning(messages)
    if last_idx is None:
        raise ValueError(f"No assistant message found in {step_dir}/messages.json")
    if not reasoning:
        raise ValueError(f"No reasoning content in the last assistant message")

    # Split reasoning into segments
    if split_mode == "paragraph":
        sentences, positions = _split_paragraphs(reasoning)
    else:
        sentences, positions = split_text_to_sentences(reasoning)
    if not sentences:
        raise ValueError(f"Reasoning could not be split into segments")

    # Determine step number from directory name
    step_match = re.match(r"step-(\d+)", step_dir.name)
    if not step_match:
        raise ValueError(f"Cannot parse step number from directory name: {step_dir.name}")
    target_step = int(step_match.group(1))

    # Find previous step directory
    prev_step_dir = None
    if target_step > 0:
        prev_step_dir = step_dir.parent / f"step-{target_step - 1}"
        if not prev_step_dir.exists():
            raise FileNotFoundError(
                f"Previous step directory not found: {prev_step_dir}"
            )

    return {
        "messages": messages,
        "sentences": sentences,
        "positions": positions,
        "last_assistant_idx": last_idx,
        "prev_step_dir": prev_step_dir,
        "target_step": target_step,
    }


def build_checkpoint_data(resampling: dict, k: int, step_dir: Path) -> dict:
    """
    Build checkpoint data for sentence boundary k.

    For k < N, constructs a modified checkpoint with:
    - Messages truncated before the target assistant turn, with optional
      prefill of the first k sentences
    - State and env files from the *previous* step (pre-step state)

    For k = N, returns the original checkpoint as-is (standard resume).

    Args:
        resampling: Output of prepare_resampling()
        k: Number of sentences to prefill (0 = fresh, N = full resume)
        step_dir: Original step directory path

    Returns:
        dict with:
            messages_json: str (JSON content for messages.json)
            state_json: str (JSON content for state.json)
            env_files: dict[str, str] (filename -> content for env files)
    """
    messages = resampling["messages"]
    sentences = resampling["sentences"]
    last_assistant_idx = resampling["last_assistant_idx"]
    prev_step_dir = resampling["prev_step_dir"]
    target_step = resampling["target_step"]
    n = len(sentences)

    if k < 0 or k > n:
        raise ValueError(f"k={k} out of range [0, {n}]")

    # k=N: standard resume — return original checkpoint data as-is
    if k == n:
        return {
            "messages_json": (step_dir / "messages.json").read_text(),
            "state_json": (step_dir / "state.json").read_text(),
            "env_files": _collect_env_files(step_dir),
        }

    # --- Messages: truncate before the target assistant turn ---
    truncated = messages[:last_assistant_idx]

    if k > 0:
        # Append prefill: <think> tag + first k sentences
        prefill_text = "<think>" + "".join(sentences[:k])
        truncated.append({"role": "assistant", "content": prefill_text})

    messages_json = json.dumps(truncated, indent=2, ensure_ascii=False)

    # --- State + env files: from the step *before* the target ---
    if target_step > 0:
        # Previous step's state.json has step = target_step - 1
        # After agent.py's `state.step += 1`, it becomes target_step (correct)
        state_json = (prev_step_dir / "state.json").read_text()
        env_files = _collect_env_files(prev_step_dir)
    else:
        # Step-0: take step-0's own state.json, set step = -1
        # After agent.py's `state.step += 1`, it becomes 0 (correct)
        # No env files — let entrypoint.py set them up from scratch
        state = json.loads((step_dir / "state.json").read_text())
        state["step"] = -1
        state_json = json.dumps(state, indent=2, ensure_ascii=False)
        env_files = {}

    return {
        "messages_json": messages_json,
        "state_json": state_json,
        "env_files": env_files,
    }


def _collect_env_files(step_dir: Path) -> dict[str, str]:
    """Collect environment files (non-infrastructure) from a step directory."""
    env_files = {}
    for f in step_dir.iterdir():
        if f.is_file() and f.name not in _SKIP_FILES:
            try:
                env_files[f.name] = f.read_text()
            except UnicodeDecodeError:
                env_files[f.name] = f.read_bytes().decode("utf-8", errors="replace")
    return env_files
