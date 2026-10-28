# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""File and checkout provenance without importing the model runtime."""

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def collect(source, expected, inputs, harness):
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if sha != expected:
        raise RuntimeError(f"checkout {sha} does not match expected {expected}")
    dirty = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=source, text=True)
    if dirty.strip():
        raise RuntimeError(f"tracked checkout changes: {dirty}")
    packages = {}
    for name in ("torch", "vllm", "diffusers", "transformers", "triton", "Pillow", "requests"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "source": str(source),
        "sha": sha,
        "packages": packages,
        "inputs": {str(p): digest(p) for p in inputs.glob("*.json")},
        "harness": {str(p.relative_to(harness)): digest(p) for p in harness.rglob("*.py")},
    }
