"""
Utilities for loading and preparing rollouts for LLM grading.

This module provides generic rollout handling that works across different
evaluation environments.
"""

import json
from pathlib import Path


def create_rollout_analysis_prompt(system_prompt: str, rollout: str) -> list[dict]:
    """
    Create a chat prompt for analyzing a rollout.

    Args:
        system_prompt: System prompt describing the analysis task.
        rollout: The rollout content to analyze.

    Returns:
        List of message dicts ready for chat completion.
    """
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Analyze this rollout: <ROLLOUT>\n{rollout}\n</ROLLOUT>",
        },
    ]


def read_rollout_file(filepath: Path) -> str:
    """
    Read a rollout file and return its contents.

    Args:
        filepath: Path to the rollout file.

    Returns:
        File contents as string, or empty string on error.
    """
    try:
        return filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""


def get_rollout_files(output_dir: str = "outputs") -> list[Path]:
    """
    Get all rollout files from an output directory.

    Supports two output formats:
    - New format (scripts/run.sh): run-N/rollout.log
    - Legacy format (run_experiments.py): run-*.txt

    Args:
        output_dir: Directory containing rollout files.

    Returns:
        Sorted list of rollout file paths.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"Output directory {output_dir} does not exist!")
        return []

    # Try new format first: run-N/rollout.log
    files = sorted(output_path.glob("run-*/rollout.log"))

    # Fall back to legacy format: run-*.txt
    if not files:
        files = sorted(output_path.glob("run-*.txt"))
        if files:
            print("Using legacy format (run-*.txt)")

    print(f"Found {len(files)} rollout files")
    return files


def load_rollouts(output_dir: str) -> list[tuple[Path, str]]:
    """
    Load all rollout files and their contents.

    Args:
        output_dir: Directory containing rollout files.

    Returns:
        List of (filepath, content) tuples for valid rollouts.
    """
    files = get_rollout_files(output_dir)

    rollouts = []
    for filepath in files:
        content = read_rollout_file(filepath)
        if content:
            rollouts.append((filepath, content))

    return rollouts


def load_rollout_messages(run_path: Path) -> list[dict]:
    """
    Load messages.json from a rollout's final step.

    Args:
        run_path: Path to run-N directory.

    Returns:
        List of message dicts from messages.json.

    Raises:
        FileNotFoundError: If no step directories or messages.json found.
    """
    step_dirs = sorted(run_path.glob("step-*"))
    if not step_dirs:
        raise FileNotFoundError(f"No step directories found in {run_path}")

    messages_path = step_dirs[-1] / "messages.json"
    if not messages_path.exists():
        raise FileNotFoundError(f"messages.json not found at {messages_path}")

    with open(messages_path) as f:
        return json.load(f)


def find_all_runs(results_dir: Path) -> list[Path]:
    """
    Find all run-N directories in a results directory.

    Args:
        results_dir: Directory containing run-N folders.

    Returns:
        Sorted list of run directory paths.
    """
    return sorted(results_dir.glob("run-*"))
