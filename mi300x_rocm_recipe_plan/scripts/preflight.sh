#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

require_workspace

"$PYTHON_BIN" - <<'PY'
import sys

import torch

errors = []
if torch.version.hip is None:
    errors.append(f"PyTorch is not a ROCm build. torch={torch.__version__}, torch.version.hip={torch.version.hip}")
if not torch.cuda.is_available():
    errors.append("PyTorch cannot access a ROCm accelerator")
if torch.cuda.device_count() != 1:
    errors.append(f"Expected exactly one visible GPU, found {torch.cuda.device_count()}")
if torch.cuda.is_available() and torch.cuda.device_count() == 1:
    name = torch.cuda.get_device_name(0)
    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if "MI300X" not in name and not bool(int(__import__("os").environ.get("ALLOW_OTHER_ROCM_GPU", "0"))):
        errors.append(f"Expected an MI300X, found {name}. Set ALLOW_OTHER_ROCM_GPU=1 only for a deliberate non-MI300X check")
    if total_gib < 180:
        errors.append(f"Expected at least 180 GiB of visible memory for this plan, found {total_gib:.1f} GiB")
if errors:
    print("Preflight failed:", file=sys.stderr)
    for error in errors:
        print(f"  {error}", file=sys.stderr)
    raise SystemExit(1)
print(f"ROCm preflight passed on {torch.cuda.get_device_name(0)} with {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
PY

record_environment
