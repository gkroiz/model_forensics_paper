"""Download the public Model Forensics rollouts dataset into ``results/``.

The Hugging Face Hub downloader preserves the dataset's directory layout and
uses local metadata to resume interrupted or repeated downloads.

Usage:
    uv run python download_results.py
    uv run python download_results.py --revision <commit-or-tag>
    uv run python download_results.py --local-dir /path/to/results

Authentication, when required for a mirror, is handled by Hugging Face. Set
``HF_TOKEN`` in the environment rather than passing a token on the command
line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

DEFAULT_REPO_ID = "Model-Forensics/model-forensics"
DEFAULT_LOCAL_DIR = Path(__file__).parent / "results"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Model Forensics rollouts from Hugging Face.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repository or compatible mirror.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch, tag, or commit hash to download.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=DEFAULT_LOCAL_DIR,
        help="Destination directory for the dataset snapshot.",
    )
    return parser.parse_args(argv)


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{int(value)} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _dataset_stats(directory: Path) -> tuple[int, int]:
    file_count = 0
    size = 0
    for path in directory.rglob("*"):
        relative_parts = path.relative_to(directory).parts
        if not path.is_file() or relative_parts[:2] == (".cache", "huggingface"):
            continue
        file_count += 1
        size += path.stat().st_size
    return file_count, size


def download_results(
    *,
    repo_id: str,
    revision: str | None,
    local_dir: Path,
) -> Path:
    """Download one dataset revision and return its resolved local directory."""
    destination = local_dir.expanduser().resolve()
    # A successful metadata request prevents snapshot_download from silently
    # treating stale local files as a valid fallback after a Hub failure.
    HfApi().repo_info(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
    )
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=destination,
    )
    file_count, size = _dataset_stats(destination)
    print(
        f"Downloaded {file_count} files ({_format_bytes(size)}) to {destination}",
        file=sys.stderr,
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        download_results(
            repo_id=args.repo_id,
            revision=args.revision,
            local_dir=args.local_dir,
        )
    except (HfHubHTTPError, OSError, ValueError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
