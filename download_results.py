"""Download the model-incrimination rollouts dataset from HuggingFace into results/.

Pulls the trimmed rollout dataset (rollout.log + final step-* per run) from
the HuggingFace dataset hub and lands it under ./results/. After this completes,
`bash reproduce/build.sh` can re-derive every per-bar JSON in `reproduce/data/`
from the raw rollouts.

Source dataset: huggingface.co/datasets/adsingh64/model-incrimination
Destination:    ./results/  (gitignored)
Approx size:    1.8 GB

Implementation TODO. Roughly:

    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="adsingh64/model-incrimination",
        repo_type="dataset",
        local_dir="results",
    )

Usage:
    uv run python download_results.py
"""

raise NotImplementedError(
    "download_results.py not yet implemented. See module docstring for the intended "
    "contract. The HF dataset upload is a separate piece of work."
)
