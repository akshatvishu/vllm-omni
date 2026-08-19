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
WHISPER_DTYPE="${WHISPER_DTYPE:-float32}"
WHISPER_MODEL="${WHISPER_MODEL:-openai/whisper-large-v3}"
WHISPER_REVISION="${WHISPER_REVISION:-06f233fe06e710322aca913c1bc4249a0d71fce1}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/$(date +%Y%m%d-%H%M%S)}"
SELECTION="$SCRIPT_DIR/selection.toml"
read -r -a SEED_VALUES <<< "${SEEDS:-42}"
read -r -a CONCURRENCY_VALUES <<< "${CONCURRENCIES:-1 2 4}"
PREPARE_DATASET_ARGS=()

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [--small]" >&2
    exit 2
fi
case "${1:-}" in
    "") ;;
    --small) PREPARE_DATASET_ARGS=(--source-examples 10) ;;
    -h|--help)
        echo "Usage: $0 [--small]"
        exit 0
        ;;
    *)
        echo "Usage: $0 [--small]" >&2
        exit 2
        ;;
esac

if ! BENCH_PYTHON="$(command -v "$PYTHON_COMMAND")"; then
    echo "Python executable not found: $PYTHON_COMMAND" >&2
    exit 1
fi
if ! VLLM_BIN="$(command -v "$VLLM_COMMAND")"; then
    echo "vLLM executable not found: $VLLM_COMMAND" >&2
    exit 1
fi
if (echo >/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null; then
    echo "Port $PORT is already in use" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR/reference" "$OUTPUT_DIR/vllm-omni" "$OUTPUT_DIR/evaluation"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
MANIFEST="$OUTPUT_DIR/prompts.json"
export HIP_VISIBLE_DEVICES="$GPU_INDEX"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"

cd "$REPO_ROOT"

echo "Saving results to: $OUTPUT_DIR"

"$BENCH_PYTHON" -m pip install --no-deps \
    "omnivoice==0.2.1" \
    "jiwer==4.0.0" \
    "pydub==0.25.1"

"$BENCH_PYTHON" -m benchmarks.tts.omnivoice_longform.prepare_dataset \
    --selection "$SELECTION" \
    --output "$MANIFEST" \
    "${PREPARE_DATASET_ARGS[@]}"

"$BENCH_PYTHON" -m benchmarks.tts.omnivoice_longform.metadata \
    --output "$OUTPUT_DIR/run_metadata.json" \
    --repo-root "$REPO_ROOT" \
    --manifest "$MANIFEST" \
    --model "$MODEL" \
    --model-revision "$MODEL_REVISION" \
    --gpu-index "$GPU_INDEX" \
    --whisper-model "$WHISPER_MODEL" \
    --whisper-revision "$WHISPER_REVISION" \
    --whisper-dtype "$WHISPER_DTYPE" \
    --seeds "${SEED_VALUES[@]}" \
    --concurrencies "${CONCURRENCY_VALUES[@]}"

"$BENCH_PYTHON" -m benchmarks.tts.omnivoice_longform.reference.inference \
    --model "$MODEL" \
    --model-revision "$MODEL_REVISION" \
    --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_DIR/reference" \
    --dtype float32 \
    --seeds "${SEED_VALUES[@]}"

SERVER_PID=""
cleanup_server() {
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
}
trap cleanup_server EXIT

"$VLLM_BIN" serve "$MODEL" \
    --revision "$MODEL_REVISION" \
    --omni \
    --host 127.0.0.1 \
    --port "$PORT" \
    --trust-remote-code \
    >"$OUTPUT_DIR/vllm-omni/server.log" 2>&1 &
SERVER_PID=$!

SERVER_READY=0
for _ in $(seq 1 180); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "vLLM server exited before becoming ready" >&2
        tail -n 100 "$OUTPUT_DIR/vllm-omni/server.log" >&2
        exit 1
    fi
    if curl --fail --silent --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null; then
        SERVER_READY=1
        break
    fi
    sleep 5
done
if [[ "$SERVER_READY" -ne 1 ]]; then
    echo "vLLM server did not become ready within 15 minutes" >&2
    exit 1
fi

"$BENCH_PYTHON" -m benchmarks.tts.omnivoice_longform.vllm_omni.benchmark \
    --api-base "http://127.0.0.1:$PORT" \
    --model "$MODEL" \
    --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_DIR/vllm-omni" \
    --seeds "${SEED_VALUES[@]}" \
    --concurrencies "${CONCURRENCY_VALUES[@]}"

cleanup_server
SERVER_PID=""
trap - EXIT

"$BENCH_PYTHON" -m benchmarks.tts.omnivoice_longform.evaluate \
    --records \
        "$OUTPUT_DIR/reference/generation.jsonl" \
        "$OUTPUT_DIR/vllm-omni/generation.jsonl" \
    --output-dir "$OUTPUT_DIR/evaluation" \
    --whisper-model "$WHISPER_MODEL" \
    --model-revision "$WHISPER_REVISION" \
    --dtype "$WHISPER_DTYPE"

echo "Benchmark results: $OUTPUT_DIR"
echo "Quality summary: $OUTPUT_DIR/evaluation/summary.md"
echo "Serving summary: $OUTPUT_DIR/vllm-omni/serving_summary.json"
