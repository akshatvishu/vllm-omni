#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLAN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PLAN_ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

if [[ -z "${RUN_ROOT:-}" ]]; then
    RUN_ROOT="$PLAN_ROOT/results/$(date -u +%Y%m%dT%H%M%SZ)"
fi
export RUN_ROOT

require_workspace() {
    if [[ ! -x "$PYTHON_BIN" ]]; then
        echo "Expected the workspace virtual environment interpreter at $PYTHON_BIN" >&2
        return 1
    fi
    if [[ ! -d "$REPO_ROOT/vllm_omni" ]]; then
        echo "Could not find the vLLM Omni repository at $REPO_ROOT" >&2
        return 1
    fi
}

record_environment() {
    local output="$RUN_ROOT/environment.txt"
    mkdir -p "$RUN_ROOT"
    {
        echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "repo_root=$REPO_ROOT"
        echo "git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
        echo "kernel=$(uname -srmo)"
        echo "python_bin=$PYTHON_BIN"
        "$PYTHON_BIN" - <<'PY'
import importlib.metadata
import json
import platform

import torch

data = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_hip": torch.version.hip,
    "torch_cuda": torch.version.cuda,
    "accelerator_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}
if torch.cuda.is_available() and torch.cuda.device_count():
    data["devices"] = [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
        }
        for index in range(torch.cuda.device_count())
    ]
for package in ("vllm", "vllm-omni", "onnxruntime", "onnxruntime-rocm", "transformers"):
    try:
        data[f"package_{package}"] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        data[f"package_{package}"] = None
try:
    import onnxruntime

    data["onnxruntime_providers"] = onnxruntime.get_available_providers()
except Exception as exc:
    data["onnxruntime_error"] = repr(exc)
print(json.dumps(data, indent=2, sort_keys=True))
PY
        if command -v rocm-smi >/dev/null 2>&1; then
            rocm-smi --showproductname --showmeminfo vram --showdriverversion
        elif command -v amd-smi >/dev/null 2>&1; then
            amd-smi static --gpu all
        else
            echo "gpu_monitor=none"
        fi
        df -h "$REPO_ROOT"
    } >"$output" 2>&1
    echo "Recorded environment in $output"
}

start_gpu_monitor() {
    local output="$1"
    (
        while true; do
            echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            if command -v rocm-smi >/dev/null 2>&1; then
                rocm-smi --showuse --showmeminfo vram --showtemp --showpower 2>&1
            elif command -v amd-smi >/dev/null 2>&1; then
                amd-smi metric --gpu all 2>&1
            else
                echo "No rocm-smi or amd-smi command is available"
                break
            fi
            sleep 1
        done
    ) >"$output" &
    GPU_MONITOR_PID=$!
}

stop_gpu_monitor() {
    if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
        unset GPU_MONITOR_PID
    fi
}

trap 'stop_gpu_monitor' EXIT
trap 'stop_gpu_monitor; exit 130' INT
trap 'stop_gpu_monitor; exit 143' TERM

run_profiled() {
    local name="$1"
    shift
    local model_dir="$RUN_ROOT/$name"
    local log_file="$model_dir/command.log"
    local command_file="$model_dir/command.txt"
    local timing_file="$model_dir/timing.txt"
    local metrics_file="$model_dir/gpu_metrics.log"
    local start_ns end_ns status

    mkdir -p "$model_dir"
    printf '%q ' "$@" >"$command_file"
    printf '\n' >>"$command_file"

    start_gpu_monitor "$metrics_file"
    start_ns="$(date +%s%N)"
    set +e
    (
        cd "$REPO_ROOT"
        "$@"
    ) 2>&1 | tee "$log_file"
    status="${PIPESTATUS[0]}"
    set -e
    end_ns="$(date +%s%N)"
    stop_gpu_monitor

    {
        echo "exit_status=$status"
        echo "start_ns=$start_ns"
        echo "end_ns=$end_ns"
        echo "elapsed_seconds=$(( (end_ns - start_ns) / 1000000000 ))"
    } >"$timing_file"

    return "$status"
}

validate_outputs() {
    local name="$1"
    local kind="$2"
    local pattern="$3"
    shift 3
    local model_dir="$RUN_ROOT/$name"
    "$PYTHON_BIN" "$SCRIPT_DIR/validate_artifact.py" \
        --kind "$kind" \
        --glob "$pattern" \
        "$@" | tee "$model_dir/artifact_validation.json"
}
