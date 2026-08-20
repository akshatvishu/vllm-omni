#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -Eeuo pipefail

readonly PYTHON_COMMAND="${PYTHON:-python}"
readonly VLLM_OMNI_COMMAND="${VLLM_OMNI:-vllm-omni}"
readonly MODEL="Qwen/Qwen3-TTS-12Hz-0.6B-Base"
readonly MODEL_REVISION="5d83992436eae1d760afd27aff78a71d676296fc"
readonly DATASET_REVISION="71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
readonly REFERENCE_SHA256="ade2caad47934462292358ba49acf7f048ced0595615492b4b20866d118bf901"
readonly REPRO_URL="https://github.com/user-attachments/files/31215137/repro_qwen3_tts_seed_cobatch.py"
readonly REPRO_SHA256="e25f560db99dda9a82e4998780f6ea2eaae34f07067e66774d5592e4fa1b7fc6"
readonly DEPLOY_CONFIG_OVERRIDE="${DEPLOY_CONFIG:-}"
readonly HOST="${HOST:-127.0.0.1}"
readonly PORT="${PORT:-8000}"
readonly BASE_URL="http://$HOST:$PORT"
readonly REPEATS="${REPEATS:-2}"
readonly SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-1800}"
readonly REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"
readonly OUTPUT_PARENT="${OUTPUT_PARENT:-$PWD}"
readonly REFERENCE_AUDIO_OVERRIDE="${REF_AUDIO:-}"

PYTHON_BIN=""
VLLM_OMNI_BIN=""
DEPLOY_CONFIG_PATH=""
REFERENCE_AUDIO=""
REPORTER_SCRIPT=""
SERVER_PID=""
RUN_ROOT=""

die() {
    echo "ERROR: $*" >&2
    exit 1
}

stop_server() {
    if [[ -z "$SERVER_PID" ]]; then
        return
    fi

    if kill -0 -- "-$SERVER_PID" 2>/dev/null; then
        kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
        for _ in {1..30}; do
            if ! kill -0 -- "-$SERVER_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 -- "-$SERVER_PID" 2>/dev/null; then
            kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
        fi
    fi
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}

trap stop_server EXIT
trap 'exit 130' INT TERM

check_inputs() {
    PYTHON_BIN="$(command -v "$PYTHON_COMMAND")" || die "Python interpreter not found: $PYTHON_COMMAND"
    VLLM_OMNI_BIN="$(command -v "$VLLM_OMNI_COMMAND")" || die "vllm-omni entry point not found: $VLLM_OMNI_COMMAND"
    if [[ -n "$DEPLOY_CONFIG_OVERRIDE" ]]; then
        DEPLOY_CONFIG_PATH="$DEPLOY_CONFIG_OVERRIDE"
    else
        DEPLOY_CONFIG_PATH="$(
            "$PYTHON_BIN" -c \
                'from importlib.util import find_spec; from pathlib import Path; spec = find_spec("vllm_omni"); print(Path(next(iter(spec.submodule_search_locations))).joinpath("deploy/qwen3_tts.yaml"))'
        )"
    fi
    [[ -f "$DEPLOY_CONFIG_PATH" ]] || die "Deploy config not found: $DEPLOY_CONFIG_PATH"
    command -v curl >/dev/null || die "curl is required"
    command -v setsid >/dev/null || die "setsid is required"
    [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || die "REPEATS must be a positive integer"
    [[ "$SERVER_START_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die "SERVER_START_TIMEOUT must be a positive integer"

    readonly PYTHON_BIN VLLM_OMNI_BIN DEPLOY_CONFIG_PATH

    "$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("No CUDA or ROCm GPU is available")
backend = f"ROCm {torch.version.hip}" if torch.version.hip else f"CUDA {torch.version.cuda}"
print(f"Using {torch.cuda.get_device_name(0)} with {backend}")
if torch.version.hip:
    print("NOTE: upstream vLLM does not currently list ROCm as a supported batch-invariance platform")
PY
}

materialize_reference_audio() {
    "$PYTHON_BIN" - "$REFERENCE_AUDIO" "$DATASET_REVISION" "$REFERENCE_SHA256" <<'PY'
import os
import sys
from hashlib import sha256
from json import load
from pathlib import Path
from urllib.request import urlopen

target = Path(sys.argv[1])
revision = sys.argv[2]
expected_sha256 = sys.argv[3]

if target.exists():
    actual_sha256 = sha256(target.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            f"Existing reference audio has SHA-256 {actual_sha256}, "
            f"expected {expected_sha256}: {target}"
        )
    print(f"Using verified reference audio: {target}")
    raise SystemExit(0)

row_url = (
    "https://datasets-server.huggingface.co/rows?"
    "dataset=openslr/librispeech_asr&config=all&"
    "split=validation.clean&offset=2&length=1"
)
with urlopen(row_url, timeout=120) as response:
    row_result = load(response)
row_entry = row_result["rows"][0]
row = row_entry["row"]

expected_text = (
    "HURSTWOOD WALKED THE FLOOR MENTALLY ARRANGING THE CHIEF POINTS "
    "OF HIS SITUATION"
)
if (
    row_entry["row_idx"] != 2
    or row["id"] != "2277-149896-0002"
    or row["speaker_id"] != 2277
    or row["text"] != expected_text
):
    raise SystemExit(f"Pinned dataset row does not match issue #6361: {row}")

audio_url = row["audio"][0]["src"]
if f"/{revision}/" not in audio_url:
    raise SystemExit(
        f"Dataset server returned a different revision for issue #6361: {audio_url}"
    )
with urlopen(audio_url, timeout=120) as response:
    payload = response.read()
actual_sha256 = sha256(payload).hexdigest()
if actual_sha256 != expected_sha256:
    raise SystemExit(
        f"Downloaded reference audio has SHA-256 {actual_sha256}, "
        f"expected {expected_sha256}"
    )

target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
temporary.write_bytes(payload)
temporary.replace(target)
print(f"Materialized verified dataset row at: {target}")
PY
}

download_reporter_repro() {
    local actual_sha256

    curl -L --fail --retry 3 --connect-timeout 20 \
        --output "$REPORTER_SCRIPT.tmp" "$REPRO_URL"
    actual_sha256="$(sha256sum "$REPORTER_SCRIPT.tmp" | awk '{print $1}')"
    if [[ "$actual_sha256" != "$REPRO_SHA256" ]]; then
        die "Downloaded reporter script has SHA-256 $actual_sha256, expected $REPRO_SHA256"
    fi
    mv "$REPORTER_SCRIPT.tmp" "$REPORTER_SCRIPT"
    echo "Downloaded and verified reporter script: $REPORTER_SCRIPT"
}

wait_for_server() {
    local log_path="$1"
    local deadline=$((SECONDS + SERVER_START_TIMEOUT))

    while ((SECONDS < deadline)); do
        if curl --silent --show-error --fail --max-time 5 "$BASE_URL/health" >/dev/null; then
            return
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            tail -n 200 "$log_path" >&2 || true
            die "Server exited before becoming healthy; see $log_path"
        fi
        sleep 5
    done

    tail -n 200 "$log_path" >&2 || true
    die "Server did not become healthy within $SERVER_START_TIMEOUT seconds; see $log_path"
}

validate_report() {
    local mode="$1"
    local report_path="$2"

    "$PYTHON_BIN" - "$mode" "$report_path" <<'PY'
import json
import sys
from pathlib import Path

mode = sys.argv[1]
report_path = Path(sys.argv[2])
if not report_path.is_file():
    raise SystemExit(f"Reporter did not create {report_path}")

report = json.loads(report_path.read_text())
if not report.get("all_responses_valid"):
    raise SystemExit(f"One or more speech requests failed; inspect {report_path}")
if len(report.get("rounds", [])) != 1:
    raise SystemExit(f"Expected exactly one round in {report_path}")

round_result = report["rounds"][0]
sequential_hash = round_result["sequential"]["sha256"]
concurrent_hashes = [result["sha256"] for result in round_result["concurrent"]]

if mode == "default":
    if report["mismatches"] != report["comparisons"]:
        raise SystemExit(
            "Default mode did not reproduce issue #6361: expected every "
            f"concurrent output to differ from the sequential output; inspect {report_path}"
        )
    print(
        f"PASS default reproduction: sequential={sequential_hash}, "
        f"concurrent_classes={len(set(concurrent_hashes))}",
        file=sys.stderr,
    )
elif mode == "invariant":
    if report["mismatches"] != 0 or any(
        output_hash != sequential_hash for output_hash in concurrent_hashes
    ):
        raise SystemExit(
            "VLLM_BATCH_INVARIANT=1 did not make all outputs identical; "
            f"inspect {report_path}"
        )
    print(
        f"PASS batch invariance: all outputs={sequential_hash}",
        file=sys.stderr,
    )
else:
    raise SystemExit(f"Unknown mode: {mode}")

print(sequential_hash)
PY
}

run_once() {
    local mode="$1"
    local batch_invariant="$2"
    local repeat="$3"
    local run_dir="$RUN_ROOT/${mode}_${repeat}"
    local server_log="$run_dir/server.log"
    local client_log="$run_dir/client.log"
    local client_output_dir="$run_dir/client_output"
    local client_status
    local sequential_hash

    mkdir -p "$client_output_dir"
    if curl --silent --fail --max-time 2 "$BASE_URL/health" >/dev/null 2>&1; then
        die "Another healthy server is already listening at $BASE_URL"
    fi

    echo
    echo "Starting $mode fresh-server repeat $repeat/$REPEATS (VLLM_BATCH_INVARIANT=$batch_invariant)"
    setsid env \
        VLLM_USE_FLASHINFER_SAMPLER=0 \
        VLLM_BATCH_INVARIANT="$batch_invariant" \
        "$VLLM_OMNI_BIN" serve "$MODEL" \
        --omni \
        --host "$HOST" \
        --port "$PORT" \
        --revision "$MODEL_REVISION" \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.75 \
        --task-type Base \
        --deploy-config "$DEPLOY_CONFIG_PATH" \
        --trust-remote-code \
        --enforce-eager \
        --disable-log-stats >"$server_log" 2>&1 &
    SERVER_PID=$!
    wait_for_server "$server_log"

    set +e
    "$PYTHON_BIN" "$REPORTER_SCRIPT" \
        --base-url "$BASE_URL" \
        --ref-audio "$REFERENCE_AUDIO" \
        --output-dir "$client_output_dir" \
        --rounds 1 \
        --concurrency 4 \
        --timeout "$REQUEST_TIMEOUT" 2>&1 | tee "$client_log"
    client_status=${PIPESTATUS[0]}
    set -e

    # The reporter's script returns 2 when the bug is not reproduced. That is
    # the expected raw status after batch invariance fixes the mismatch, so the
    # JSON report below is the source of truth for this driver.
    echo "Reporter exit status: $client_status"
    sequential_hash="$(validate_report "$mode" "$client_output_dir/repro_results.json")"
    printf '%s\t%s\t%s\n' "$mode" "$repeat" "$sequential_hash" >>"$RUN_ROOT/summary.tsv"
    stop_server
}

validate_fresh_process_repeats() {
    "$PYTHON_BIN" - "$RUN_ROOT/summary.tsv" "$REPEATS" <<'PY'
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
expected_repeats = int(sys.argv[2])
rows = [line.split("\t") for line in summary_path.read_text().splitlines()]

for mode in ("default", "invariant"):
    mode_rows = [row for row in rows if row[0] == mode]
    if len(mode_rows) != expected_repeats:
        raise SystemExit(
            f"Expected {expected_repeats} completed {mode} repeats, got {len(mode_rows)}"
        )

invariant_hashes = {row[2] for row in rows if row[0] == "invariant"}
if len(invariant_hashes) != 1:
    raise SystemExit(
        "Batch-invariant output changed across fresh server processes: "
        f"{sorted(invariant_hashes)}"
    )

print(f"PASS fresh-process reproducibility: {invariant_hashes.pop()}")
PY
}

main() {
    mkdir -p "$OUTPUT_PARENT"
    RUN_ROOT="$(mktemp -d "$OUTPUT_PARENT/6361-tts-seeding-results.XXXXXX")"
    REFERENCE_AUDIO="${REFERENCE_AUDIO_OVERRIDE:-$RUN_ROOT/2277-149896-0002.flac}"
    REPORTER_SCRIPT="$RUN_ROOT/repro_qwen3_tts_seed_cobatch.py"
    readonly RUN_ROOT REFERENCE_AUDIO REPORTER_SCRIPT
    cp -- "${BASH_SOURCE[0]}" "$RUN_ROOT/"
    exec > >(tee "$RUN_ROOT/driver.log") 2>&1

    check_inputs
    materialize_reference_audio
    download_reporter_repro

    for ((repeat = 1; repeat <= REPEATS; repeat++)); do
        run_once default 0 "$repeat"
    done
    for ((repeat = 1; repeat <= REPEATS; repeat++)); do
        run_once invariant 1 "$repeat"
    done

    validate_fresh_process_repeats
    echo "All issue #6361 checks passed. Artifacts: $RUN_ROOT"
}

main "$@"
