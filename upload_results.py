"""Upload results/ to huggingface.co/datasets/adsingh64/model-forensics.

Notes:
  - IMPORTANT: Whatever is in results/ gets uploaded. Make sure no leaked env variables.
  - upload_large_folder handles resumption if interrupted — Ctrl-C is safe.
  - HF_HUB_ENABLE_HF_TRANSFER=1 for high-throughput uploads.

Usage:
    HF_TOKEN=... uv run python upload_results.py
"""

import logging
import os
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import logging as hf_logging

REPO_ID = os.environ.get("HF_REPO_ID", "adsingh64/model-forensics")
FOLDER = Path(__file__).parent / "results"

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
hf_logging.set_verbosity_info()
log = logging.getLogger("upload")


def main() -> None:
    if not FOLDER.is_dir():
        raise SystemExit(f"results/ not found at {FOLDER}. Stage rollouts first.")

    n_files = sum(1 for _ in FOLDER.rglob("*") if _.is_file())
    size_gb = sum(p.stat().st_size for p in FOLDER.rglob("*") if p.is_file()) / 1024**3
    log.info("Found %d files under %s, ~%.2f GB to upload.", n_files, FOLDER, size_gb)

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    log.info("Repo ready: https://huggingface.co/datasets/%s", REPO_ID)

    log.info("Starting upload (status report every 15s, Ctrl-C is safe — uploads resume).")
    api.upload_large_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(FOLDER),
        private=True,
        print_report=True,
        print_report_every=15,
    )
    log.info("Upload complete: https://huggingface.co/datasets/%s", REPO_ID)


if __name__ == "__main__":
    main()
