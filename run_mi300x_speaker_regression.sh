#!/usr/bin/env bash
set -o pipefail

label="${1:?usage: $0 before-fix|after-fix}"
log="$PWD/mi300x_speaker_regression_${label}.log"

python -m pytest -sv \
  "$PWD/mi300x_speaker_regression_test.py::test_ethan_survives_non_async_stage_handoff" \
  --run-level full_model 2>&1 | tee "$log"
