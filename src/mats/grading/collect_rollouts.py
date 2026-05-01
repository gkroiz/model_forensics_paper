"""
Utility to collect and concatenate all rollout.log files from a results folder.

Usage:
    uv run python -m mats.grading.collect_rollouts <folder_path>

Example:
    uv run python -m mats.grading.collect_rollouts results/eval_tampering/notes_automated/moonshotai-kimi-k2-thinking/2026-01-27_14-19-54
"""

import argparse
from pathlib import Path

from mats.grading.rollout_utils import get_rollout_files, read_rollout_file


def collect_and_concatenate_rollouts(folder: Path) -> str:
    """
    Collect all rollout.log files from a folder and concatenate them.

    Args:
        folder: Path to the timestamped results folder containing run-N directories.

    Returns:
        Concatenated string of all rollouts with separators.
    """
    rollout_files = get_rollout_files(str(folder))

    if not rollout_files:
        print(f"No rollout files found in {folder}")
        return ""

    parts = []
    for filepath in rollout_files:
        # Extract run identifier (e.g., "run-0" from "run-0/rollout.log")
        run_id = filepath.parent.name
        content = read_rollout_file(filepath)

        if content:
            separator = "=" * 80
            header = f"\n{separator}\n{run_id.upper()}\n{separator}\n"
            parts.append(header + content)

    return "\n".join(parts)


def save_all_rollouts(folder: Path, output_filename: str = "all_rollouts.log") -> Path:
    """
    Collect all rollouts and save to a single file.

    Args:
        folder: Path to the timestamped results folder.
        output_filename: Name of the output file (default: all_rollouts.log).

    Returns:
        Path to the saved file.
    """
    concatenated = collect_and_concatenate_rollouts(folder)

    if not concatenated:
        raise ValueError(f"No rollouts found in {folder}")

    output_path = folder / output_filename
    output_path.write_text(concatenated, encoding="utf-8")
    print(f"Saved concatenated rollouts to {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Collect and concatenate all rollout.log files from a results folder."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Path to the timestamped results folder (e.g., results/.../2026-01-27_14-19-54)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="all_rollouts.log",
        help="Output filename (default: all_rollouts.log)",
    )

    args = parser.parse_args()

    if not args.folder.exists():
        print(f"Error: Folder {args.folder} does not exist")
        return 1

    if not args.folder.is_dir():
        print(f"Error: {args.folder} is not a directory")
        return 1

    try:
        save_all_rollouts(args.folder, args.output)
        return 0
    except ValueError as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
