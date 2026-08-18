#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$SCRIPT_DIR/runs/$RUN_ID"
PYTHON_BIN="${PYTHON_BIN:-python}"
PORT="${PORT:-8091}"
GPU_IDS="${GPU_IDS:-0,1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-1800}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1800}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
ZIMAGE_MODEL="${ZIMAGE_MODEL:-Tongyi-MAI/Z-Image-Turbo}"
QWEN_DEPLOY_CONFIG="${QWEN_DEPLOY_CONFIG:-vllm_omni/deploy/qwen3_omni_moe.yaml}"
RUN_QWEN="${RUN_QWEN:-1}"
RUN_ZIMAGE="${RUN_ZIMAGE:-1}"
SERVER_PID=""
SERVER_LABEL=""
OVERALL_STATUS=0

mkdir -p "$RUN_DIR"/{environment,requests,responses,server}
exec > >(tee -a "$RUN_DIR/run.log") 2>&1

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        log "ERROR: required command not found: $1"
        exit 2
    fi
}

stop_server() {
    if [[ -z "$SERVER_PID" ]]; then
        return
    fi

    log "Stopping $SERVER_LABEL server process group $SERVER_PID"
    kill -INT -- "-$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            wait "$SERVER_PID" 2>/dev/null || true
            SERVER_PID=""
            SERVER_LABEL=""
            return
        fi
        sleep 1
    done

    log "Server did not stop after 60 seconds; sending TERM"
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    SERVER_LABEL=""
}

on_exit() {
    stop_server
}
trap on_exit EXIT INT TERM

wait_for_server() {
    local deadline=$((SECONDS + SERVER_START_TIMEOUT))
    while ((SECONDS < deadline)); do
        if curl --silent --fail "http://127.0.0.1:$PORT/health" >/dev/null; then
            log "$SERVER_LABEL server is ready"
            curl --silent --show-error "http://127.0.0.1:$PORT/v1/models" \
                | tee "$RUN_DIR/server/${SERVER_LABEL}_models.json"
            printf '\n'
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log "ERROR: $SERVER_LABEL server exited before becoming healthy"
            wait "$SERVER_PID" 2>/dev/null || true
            SERVER_PID=""
            return 1
        fi
        sleep 5
    done

    log "ERROR: $SERVER_LABEL server was not healthy after ${SERVER_START_TIMEOUT}s"
    return 1
}

ensure_port_is_free() {
    if curl --silent --fail "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        log "ERROR: port $PORT already serves a healthy endpoint; refusing to stop an unrelated server"
        exit 2
    fi
}

start_qwen_server() {
    SERVER_LABEL="qwen3_omni"
    log "Starting $SERVER_LABEL on visible GPUs $GPU_IDS"
    HIP_VISIBLE_DEVICES="$GPU_IDS" VLLM_WORKER_MULTIPROC_METHOD=spawn \
        setsid vllm serve "$QWEN_MODEL" --omni --host 127.0.0.1 --port "$PORT" \
        --deploy-config "$QWEN_DEPLOY_CONFIG" --async-chunk \
        > >(tee "$RUN_DIR/server/${SERVER_LABEL}.log") 2>&1 &
    SERVER_PID=$!
    wait_for_server
}

start_zimage_server() {
    local zimage_gpu="${GPU_IDS%%,*}"
    SERVER_LABEL="zimage"
    log "Starting $SERVER_LABEL on physical GPU $zimage_gpu"
    HIP_VISIBLE_DEVICES="$zimage_gpu" VLLM_WORKER_MULTIPROC_METHOD=spawn \
        setsid vllm serve "$ZIMAGE_MODEL" --omni --host 127.0.0.1 --port "$PORT" \
        > >(tee "$RUN_DIR/server/${SERVER_LABEL}.log") 2>&1 &
    SERVER_PID=$!
    wait_for_server
}

run_probe() {
    local suite="$1"
    local model="$2"
    log "Running $suite client matrix"
    set +o pipefail
    "$PYTHON_BIN" "$SCRIPT_DIR/probe_chat.py" \
        --suite "$suite" \
        --base-url "http://127.0.0.1:$PORT/v1" \
        --model "$model" \
        --batch-size "$BATCH_SIZE" \
        --timeout "$REQUEST_TIMEOUT" \
        --output-dir "$RUN_DIR" \
        2>&1 | tee "$RUN_DIR/${suite}_client.log"
    local probe_status=${PIPESTATUS[0]}
    set -o pipefail
    if ((probe_status != 0)); then
        OVERALL_STATUS=1
        log "$suite matrix recorded one or more failures"
    else
        log "$suite matrix passed"
    fi
}

record_environment() {
    log "Recording environment"
    {
        date -u --iso-8601=seconds
        uname -a
        git -C "$REPO_ROOT" status --short --branch
        git -C "$REPO_ROOT" rev-parse HEAD
        "$PYTHON_BIN" --version
        "$PYTHON_BIN" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

for package in ("openai", "torch", "vllm", "vllm-omni"):
    try:
        print(f"{package}=={version(package)}")
    except PackageNotFoundError:
        print(f"{package}=NOT_INSTALLED")
PY
        if command -v amd-smi >/dev/null 2>&1; then
            amd-smi version || true
            amd-smi static || true
        fi
        if command -v rocminfo >/dev/null 2>&1; then
            rocminfo 2>/dev/null | grep -E '(^  Name:|gfx[0-9]+)' || true
        fi
        printf 'GPU_IDS=%s\n' "$GPU_IDS"
        printf 'BATCH_SIZE=%s\n' "$BATCH_SIZE"
        printf 'QWEN_MODEL=%s\n' "$QWEN_MODEL"
        printf 'ZIMAGE_MODEL=%s\n' "$ZIMAGE_MODEL"
    } | tee "$RUN_DIR/environment/system.log"

    "$PYTHON_BIN" -m pip freeze | tee "$RUN_DIR/environment/pip-freeze.txt"
    git -C "$REPO_ROOT" diff --no-ext-diff | tee "$RUN_DIR/environment/worktree.diff"
    cp "$REPO_ROOT/$QWEN_DEPLOY_CONFIG" "$RUN_DIR/environment/qwen_deploy_config.yaml"
}

for command_name in git curl tee setsid vllm "$PYTHON_BIN"; do
    require_command "$command_name"
done

cd "$REPO_ROOT"
ensure_port_is_free
record_environment

if [[ "$RUN_QWEN" == "1" ]]; then
    if start_qwen_server; then
        run_probe qwen "$QWEN_MODEL"
    else
        OVERALL_STATUS=1
    fi
    stop_server
fi

if [[ "$RUN_ZIMAGE" == "1" ]]; then
    if start_zimage_server; then
        run_probe zimage "$ZIMAGE_MODEL"
    else
        OVERALL_STATUS=1
    fi
    stop_server
fi

log "Artifacts: ${RUN_DIR#$REPO_ROOT/}"
if ((OVERALL_STATUS != 0)); then
    log "RESULT: FAIL. Inspect summary.json and the server logs."
else
    log "RESULT: PASS. Every requested compatibility cell passed."
fi
exit "$OVERALL_STATUS"
