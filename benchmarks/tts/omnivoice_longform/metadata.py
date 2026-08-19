# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch

from benchmarks.tts.omnivoice_longform.common import write_immutable_json


def _git_state(path: Path) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_diff = subprocess.run(
        ["git", "-C", str(path), "diff", "--binary", "HEAD", "--"],
        check=True,
        capture_output=True,
    ).stdout
    untracked_output = subprocess.run(
        ["git", "-C", str(path), "ls-files", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    untracked_paths = sorted(item for item in untracked_output.split(b"\0") if item)

    worktree_hash = hashlib.sha256()
    worktree_hash.update(tracked_diff)
    for relative_path_bytes in untracked_paths:
        relative_path = relative_path_bytes.decode("utf-8", errors="surrogateescape")
        worktree_hash.update(b"\0untracked\0")
        worktree_hash.update(relative_path_bytes)
        worktree_hash.update(b"\0")
        worktree_hash.update((path / relative_path).read_bytes())

    return {
        "path": str(path.resolve()),
        "revision": revision,
        "dirty": bool(tracked_diff or untracked_paths),
        "worktree_sha256": worktree_hash.hexdigest(),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _accelerator_metadata() -> tuple[str | None, str | None]:
    if not torch.accelerator.is_available():
        return None, None
    accelerator = torch.accelerator.current_accelerator()
    device_module = getattr(torch, accelerator.type, None)
    get_device_name = getattr(device_module, "get_device_name", None)
    device_name = get_device_name() if get_device_name is not None else None
    return str(accelerator), device_name


def collect_metadata(args: argparse.Namespace) -> dict[str, Any]:
    manifest = Path(args.manifest)
    accelerator, device_name = _accelerator_metadata()
    metadata = {
        "model": args.model,
        "manifest": {
            "path": str(manifest.resolve()),
            "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        },
        "repositories": {
            "vllm_omni": _git_state(Path(args.repo_root)),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_hip": torch.version.hip,
            "torchaudio": _package_version("torchaudio"),
            "transformers": _package_version("transformers"),
            "vllm": _package_version("vllm"),
            "vllm_omni": _package_version("vllm-omni"),
            "omnivoice": _package_version("omnivoice"),
            "pydub": _package_version("pydub"),
            "huggingface_hub": _package_version("huggingface-hub"),
            "jiwer": _package_version("jiwer"),
            "soundfile": _package_version("soundfile"),
            "accelerator": accelerator,
            "device_name": device_name,
        },
        "settings": {
            "gpu_index": args.gpu_index,
            "generation_dtype": "float32",
            "whisper_model": args.whisper_model,
            "model_revision": args.model_revision,
            "whisper_revision": args.whisper_revision,
            "whisper_dtype": args.whisper_dtype,
            "seeds": args.seeds,
            "concurrencies": args.concurrencies,
            "batch_size": 1,
            "vllm_peak_memory_measured": False,
        },
    }
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()
    return {"fingerprint_sha256": fingerprint, **metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record OmniVoice benchmark environment")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--whisper-model", required=True)
    parser.add_argument("--whisper-revision", required=True)
    parser.add_argument("--whisper-dtype", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--concurrencies", type=int, nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_immutable_json(args.output, collect_metadata(args))


if __name__ == "__main__":
    main()
