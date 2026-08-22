from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "download_results.py"


def _run_downloader(
    tmp_path: Path,
    *args: str,
    failure: str | None = None,
    existing_file: bool = False,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object], Path]:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    script = checkout / "download_results.py"
    shutil.copy2(SCRIPT, script)

    fake_package = tmp_path / "fake_package" / "huggingface_hub"
    fake_package.mkdir(parents=True)
    (fake_package / "__init__.py").write_text(
        """
import json
import os
from pathlib import Path


class HfApi:
    def repo_info(self, **kwargs):
        Path(os.environ["FAKE_HF_PREFLIGHT"]).write_text(json.dumps(kwargs))
        if os.environ.get("FAKE_HF_ERROR") == "silent_fallback":
            raise OSError("simulated preflight failure")


def snapshot_download(**kwargs):
    record = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in kwargs.items()
    }
    Path(os.environ["FAKE_HF_RECORD"]).write_text(json.dumps(record))
    failure = os.environ.get("FAKE_HF_ERROR")
    if failure == "operational":
        raise OSError("simulated network failure")
    if failure == "unexpected":
        raise RuntimeError("simulated programming failure")
    destination = Path(kwargs["local_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    if failure == "silent_fallback":
        return str(destination)
    (destination / "rollout.log").write_text("rollout")
    step = destination / "run-1" / "step-2"
    step.mkdir(parents=True, exist_ok=True)
    (step / "state.json").write_text("{}")
    nested_cache = destination / "run-1" / ".cache"
    nested_cache.mkdir()
    (nested_cache / "dataset.txt").write_text("data")
    cache = destination / ".cache" / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "metadata.json").write_text("metadata that must not be counted")
    return str(destination)
""".lstrip()
    )
    (fake_package / "errors.py").write_text(
        "class HfHubHTTPError(Exception):\n    pass\n"
    )

    local_dir = checkout / "results"
    if existing_file:
        local_dir.mkdir()
        (local_dir / "keep.txt").write_text("preserve me")

    record_path = tmp_path / "call.json"
    preflight_path = tmp_path / "preflight.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(fake_package.parent), env.get("PYTHONPATH", "")]
    )
    env["FAKE_HF_RECORD"] = str(record_path)
    env["FAKE_HF_PREFLIGHT"] = str(preflight_path)
    if failure:
        env["FAKE_HF_ERROR"] = failure

    result = subprocess.run(
        [sys.executable, str(script), *args],
        cwd=checkout,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    record = {
        "repo_info": (
            json.loads(preflight_path.read_text()) if preflight_path.exists() else {}
        ),
        "snapshot_download": (
            json.loads(record_path.read_text()) if record_path.exists() else {}
        ),
    }
    return result, record, local_dir


def test_default_download_uses_canonical_dataset_and_repo_results(
    tmp_path: Path,
) -> None:
    result, call, local_dir = _run_downloader(tmp_path)

    assert result.returncode == 0, result.stderr
    assert call == {
        "repo_info": {
            "repo_id": "Model-Forensics/model-forensics",
            "repo_type": "dataset",
            "revision": None,
        },
        "snapshot_download": {
            "repo_id": "Model-Forensics/model-forensics",
            "repo_type": "dataset",
            "revision": None,
            "local_dir": str(local_dir),
        },
    }
    assert "Downloaded 3 files" in result.stderr
    assert "13 B" in result.stderr
    assert (local_dir / "run-1" / ".cache" / "dataset.txt").is_file()
    assert (local_dir / ".cache" / "huggingface" / "metadata.json").is_file()


def test_cli_forwards_repo_revision_and_destination(tmp_path: Path) -> None:
    destination = tmp_path / "custom-results"

    result, call, _ = _run_downloader(
        tmp_path,
        "--repo-id",
        "example/research-rollouts",
        "--revision",
        "abc123",
        "--local-dir",
        str(destination),
    )

    assert result.returncode == 0, result.stderr
    assert call == {
        "repo_info": {
            "repo_id": "example/research-rollouts",
            "repo_type": "dataset",
            "revision": "abc123",
        },
        "snapshot_download": {
            "repo_id": "example/research-rollouts",
            "repo_type": "dataset",
            "revision": "abc123",
            "local_dir": str(destination.resolve()),
        },
    }


def test_existing_results_are_preserved_for_resumable_download(tmp_path: Path) -> None:
    result, _, local_dir = _run_downloader(tmp_path, existing_file=True)

    assert result.returncode == 0, result.stderr
    assert (local_dir / "keep.txt").read_text() == "preserve me"
    assert (local_dir / "run-1" / "step-2" / "state.json").is_file()


def test_download_failure_has_concise_error_and_nonzero_exit(tmp_path: Path) -> None:
    result, _, _ = _run_downloader(tmp_path, failure="operational")

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.strip() == "Download failed: simulated network failure"


def test_preflight_failure_cannot_succeed_from_stale_local_results(
    tmp_path: Path,
) -> None:
    result, call, local_dir = _run_downloader(
        tmp_path,
        failure="silent_fallback",
        existing_file=True,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.strip() == "Download failed: simulated preflight failure"
    assert call["snapshot_download"] == {}
    assert (local_dir / "keep.txt").read_text() == "preserve me"


def test_unexpected_programming_error_is_not_hidden(tmp_path: Path) -> None:
    result, _, _ = _run_downloader(tmp_path, failure="unexpected")

    assert result.returncode == 1
    assert "Traceback (most recent call last)" in result.stderr
    assert "RuntimeError: simulated programming failure" in result.stderr
