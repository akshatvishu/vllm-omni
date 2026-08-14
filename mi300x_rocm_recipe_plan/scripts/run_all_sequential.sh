#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODELS=(
    ming_omni_tts
    qwen3_tts
    sensenova
    stable_audio
    mammoth_preview
    omnivoice
)

if [[ "${1:-}" == "--list" ]]; then
    printf '%s\n' "${MODELS[@]}"
    echo "qwen3_tts_compare, optional eager versus MIOpen FAST graph comparison"
    echo "ming_flash_tts, optional and disabled by default"
    exit 0
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--list]" >&2
    exit 2
fi

require_workspace
"$SCRIPT_DIR/preflight.sh"
export PREFLIGHT_DONE=1

SUMMARY="$RUN_ROOT/summary.tsv"
printf 'model\tstatus\n' >"$SUMMARY"
overall_status=0

for model in "${MODELS[@]}"; do
    echo "Starting $model"
    set +e
    "$SCRIPT_DIR/run_one.sh" "$model"
    status=$?
    set -e
    if [[ $status -eq 0 ]]; then
        printf '%s\tPASS\n' "$model" >>"$SUMMARY"
    else
        printf '%s\tFAIL_%s\n' "$model" "$status" >>"$SUMMARY"
        overall_status=1
    fi
done

echo "Sequential suite complete. Summary: $SUMMARY"
exit "$overall_status"
