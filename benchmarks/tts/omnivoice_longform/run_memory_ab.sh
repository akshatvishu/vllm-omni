#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_COMMAND="${BENCH_PYTHON:-python}"
VLLM_COMMAND="${VLLM_BIN:-vllm}"
MODEL="${MODEL:-k2-fsa/OmniVoice}"
MODEL_REVISION="${MODEL_REVISION:-c5fdb5ccb189668d56333f77ba2629f4cd7535f4}"
PORT="${PORT:-8091}"
GPU_INDEX="${GPU_INDEX:-0}"
TARGET_REF="${TARGET_REF:-HEAD}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/memory-ab/$(date +%Y%m%d-%H%M%S)}"
SOURCE_EXAMPLES=1
read -r -a SEED_VALUES <<< "${SEEDS:-42}"

usage() {
    echo "Usage: $0 [--samples N] [--ref REVISION]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --samples)
            if [[ $# -lt 2 || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
                echo "--samples requires a positive integer" >&2
                exit 2
            fi
            SOURCE_EXAMPLES="$2"
            shift 2
            ;;
        --ref)
            if [[ $# -lt 2 || -z "$2" ]]; then
                echo "--ref requires a Git revision" >&2
                exit 2
            fi
            TARGET_REF="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
done

if ! BENCH_PYTHON="$(command -v "$PYTHON_COMMAND")"; then
    echo "Python executable not found: $PYTHON_COMMAND" >&2
    exit 1
fi
if ! VLLM_BIN="$(command -v "$VLLM_COMMAND")"; then
    echo "vLLM executable not found: $VLLM_COMMAND" >&2
    exit 1
fi
if ! RESOLVED_REVISION="$(git -C "$REPO_ROOT" rev-parse --verify --end-of-options "$TARGET_REF^{commit}")"; then
    echo "Git revision not found: $TARGET_REF" >&2
    exit 1
fi
port_in_use() {
    (echo >/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null
}

if port_in_use; then
    echo "Port $PORT is already in use" >&2
    exit 1
fi
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
    echo "Output directory is not empty: $OUTPUT_DIR" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
WORKTREE_ROOT="$(mktemp -d /tmp/omnivoice-memory-ab.XXXXXX)"
BASELINE_WORKTREE="$WORKTREE_ROOT/gpu-retained"
CANDIDATE_WORKTREE="$WORKTREE_ROOT/cpu-copy"
SERVER_PID=""

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM "$SERVER_PID"
        for _ in $(seq 1 30); do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            kill -KILL "$SERVER_PID"
        fi
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}

remove_worktree() {
    local path="$1"
    if [[ -d "$path" ]]; then
        git -C "$REPO_ROOT" worktree remove --force "$path"
    fi
}

cleanup() {
    stop_server
    remove_worktree "$BASELINE_WORKTREE" || true
    remove_worktree "$CANDIDATE_WORKTREE" || true
    rmdir "$WORKTREE_ROOT" 2>/dev/null || true
}
trap cleanup EXIT

echo "Saving results to: $OUTPUT_DIR"
echo "Testing revision: $RESOLVED_REVISION"
git -C "$REPO_ROOT" worktree add --detach "$BASELINE_WORKTREE" "$RESOLVED_REVISION"
git -C "$REPO_ROOT" worktree add --detach "$CANDIDATE_WORKTREE" "$RESOLVED_REVISION"
BASELINE_TRANSFORM="$OUTPUT_DIR/prepare_gpu_retention_baseline.py"
cp "$CANDIDATE_WORKTREE/benchmarks/tts/omnivoice_longform/prepare_gpu_retention_baseline.py" "$BASELINE_TRANSFORM"
SELECTION="$CANDIDATE_WORKTREE/benchmarks/tts/omnivoice_longform/selection.toml"
"$BENCH_PYTHON" "$BASELINE_TRANSFORM" --repo-root "$BASELINE_WORKTREE"
git -C "$BASELINE_WORKTREE" diff --check

MANIFEST="$OUTPUT_DIR/prompts.json"
run_worktree_python() {
    local worktree="$1"
    shift
    (
        cd "$worktree"
        PYTHONPATH="$worktree" "$BENCH_PYTHON" "$@"
    )
}

run_worktree_python "$CANDIDATE_WORKTREE" \
    -m benchmarks.tts.omnivoice_longform.prepare_dataset \
    --selection "$SELECTION" \
    --output "$MANIFEST" \
    --source-examples "$SOURCE_EXAMPLES"

export HIP_VISIBLE_DEVICES="$GPU_INDEX"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"

record_metadata() {
    local worktree="$1"
    local output="$2"
    run_worktree_python "$worktree" \
        -m benchmarks.tts.omnivoice_longform.metadata \
        --output "$output/run_metadata.json" \
        --repo-root "$worktree" \
        --manifest "$MANIFEST" \
        --model "$MODEL" \
        --model-revision "$MODEL_REVISION" \
        --gpu-index "$GPU_INDEX" \
        --whisper-model not-run \
        --whisper-revision not-run \
        --whisper-dtype not-run \
        --seeds "${SEED_VALUES[@]}" \
        --modes chunked \
        --concurrencies 1 \
        --discard-audio
}

wait_for_server() {
    local log_path="$1"
    for _ in $(seq 1 180); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "vLLM server exited before becoming ready" >&2
            tail -n 100 "$log_path" >&2
            exit 1
        fi
        if curl --fail --silent --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null; then
            return
        fi
        sleep 5
    done
    echo "vLLM server did not become ready within 15 minutes" >&2
    tail -n 100 "$log_path" >&2
    exit 1
}

wait_for_port_release() {
    for _ in $(seq 1 30); do
        if ! port_in_use; then
            return
        fi
        sleep 1
    done
    echo "Port $PORT is still in use after the server stopped" >&2
    exit 1
}

run_variant() {
    local label="$1"
    local worktree="$2"
    local result_dir="$OUTPUT_DIR/$label"
    local log_path="$result_dir/server.log"
    mkdir -p "$result_dir"

    record_metadata "$worktree" "$result_dir"
    echo "Starting $label server"
    (
        cd "$worktree"
        PYTHONPATH="$worktree" exec "$VLLM_BIN" serve "$MODEL" \
            --revision "$MODEL_REVISION" \
            --omni \
            --host 127.0.0.1 \
            --port "$PORT" \
            --trust-remote-code
    ) >"$log_path" 2>&1 &
    SERVER_PID=$!
    wait_for_server "$log_path"

    echo "Running $label chunked benchmark"
    run_worktree_python "$worktree" \
        -m benchmarks.tts.omnivoice_longform.vllm_omni.benchmark \
        --api-base "http://127.0.0.1:$PORT" \
        --model "$MODEL" \
        --manifest "$MANIFEST" \
        --output-dir "$result_dir" \
        --seeds "${SEED_VALUES[@]}" \
        --modes chunked \
        --concurrencies 1 \
        --discard-audio
    stop_server
    wait_for_port_release
}

run_variant gpu-retained "$BASELINE_WORKTREE"
run_variant cpu-copy "$CANDIDATE_WORKTREE"

run_worktree_python "$CANDIDATE_WORKTREE" \
    -m benchmarks.tts.omnivoice_longform.compare_memory \
    --baseline "$OUTPUT_DIR/gpu-retained/serving.jsonl" \
    --candidate "$OUTPUT_DIR/cpu-copy/serving.jsonl" \
    --output-dir "$OUTPUT_DIR" \
    --revision "$RESOLVED_REVISION" \
    --baseline-transform "$BASELINE_TRANSFORM"

echo "A/B results: $OUTPUT_DIR"
echo "Comparison: $OUTPUT_DIR/comparison.md"
