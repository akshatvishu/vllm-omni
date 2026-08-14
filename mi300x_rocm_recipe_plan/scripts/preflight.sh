#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

require_workspace

"$PYTHON_BIN" - <<'PY'
import os
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
    properties = torch.cuda.get_device_properties(0)
    name = torch.cuda.get_device_name(0).strip()
    gcn_arch = str(getattr(properties, "gcnArchName", "")).strip()
    total_gib = properties.total_memory / 1024**3
    name_is_mi300x = "MI300X" in name.upper()
    arch_and_memory_are_mi300x = "gfx942" in gcn_arch.lower() and 180 <= total_gib < 220
    if not (name_is_mi300x or arch_and_memory_are_mi300x) and os.environ.get("ALLOW_OTHER_ROCM_GPU") != "1":
        errors.append(
            "Expected an MI300X, found "
            f"name={name or '<empty>'}, arch={gcn_arch or '<empty>'}, memory={total_gib:.1f} GiB. "
            "Set ALLOW_OTHER_ROCM_GPU=1 only for a deliberate non-MI300X check"
        )
    if total_gib < 180:
        errors.append(f"Expected at least 180 GiB of visible memory for this plan, found {total_gib:.1f} GiB")
if errors:
    print("Preflight failed:", file=sys.stderr)
    for error in errors:
        print(f"  {error}", file=sys.stderr)
    raise SystemExit(1)
display_name = name or gcn_arch or "unknown AMD GPU"
print(f"ROCm preflight passed on {display_name} with {total_gib:.1f} GiB")
PY

record_environment
