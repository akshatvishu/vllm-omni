#!/usr/bin/env bash
set -o pipefail

label="${1:?usage: $0 before-fix|after-fix}"
log="$PWD/mi300x_speaker_regression_${label}.log"

python -m pytest -sv \
  "$PWD/tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_speaker_002[default]" \
  --run-level full_model 2>&1 | tee "$log"
