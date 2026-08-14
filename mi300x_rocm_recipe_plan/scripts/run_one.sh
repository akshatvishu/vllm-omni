#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

usage() {
    echo "Usage: $0 {ming_omni_tts|qwen3_tts|qwen3_tts_compare|sensenova|stable_audio|mammoth_preview|omnivoice|ming_flash_tts}"
}

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

MODEL_KEY="$1"
require_workspace
if [[ "${PREFLIGHT_DONE:-0}" != "1" ]]; then
    "$SCRIPT_DIR/preflight.sh"
fi

run_ming_omni_tts() {
    local name="ming_omni_tts"
    local output_dir="$RUN_ROOT/$name/output"
    mkdir -p "$output_dir"
    run_profiled "$name" \
        "$PYTHON_BIN" examples/offline_inference/text_to_speech/ming_tts/end2end.py \
        --model inclusionAI/Ming-omni-tts-0.5B \
        --deploy-config vllm_omni/deploy/ming_tts.yaml \
        --case basic \
        --text "你好，这是 AMD MI300X ROCm 配方验证。" \
        --output-dir "$output_dir" \
        --output-name ming_omni_tts_mi300x.wav \
        --enforce-eager \
        --log-stats \
        --stats-log-file "$RUN_ROOT/$name/pipeline_stats.log" \
        --metadata-json "$RUN_ROOT/$name/manifest.json"
    validate_outputs "$name" audio "$output_dir/*.wav" --expected-sample-rate 44100
}

run_qwen3_tts_case() {
    local query="$1"
    local name="$2"
    local deploy_config="$3"
    shift 3
    local output_dir="$RUN_ROOT/$name/output"
    mkdir -p "$output_dir"
    if ! run_profiled "$name" \
        "$@" \
        "$PYTHON_BIN" examples/offline_inference/text_to_speech/qwen3_tts/end2end.py \
        --query-type "$query" \
        --deploy-config "$deploy_config" \
        --output-dir "$output_dir" \
        --num-prompts 1 \
        --batch-size 1 \
        --log-stats \
        --log-dir "$RUN_ROOT/$name/logs"; then
        return 1
    fi
    validate_outputs "$name" audio "$output_dir/*.wav" --expected-sample-rate 24000
}

run_qwen3_tts() {
    local query name
    local status=0
    for query in CustomVoice VoiceDesign Base; do
        name="qwen3_tts_${query,,}"
        if ! run_qwen3_tts_case \
            "$query" \
            "$name" \
            vllm_omni/deploy/qwen3_tts.yaml; then
            status=1
        fi
    done
    return "$status"
}

qwen3_tts_log_stat_ms() {
    local log_file="$1"
    local stat_name="$2"
    awk -F '|' -v stat_name="$stat_name" '
        {
            name = $2
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
            if (name == stat_name) {
                value = $3
                gsub(/[[:space:],]/, "", value)
                result = value
            }
        }
        END { print result }
    ' "$log_file"
}

append_qwen3_tts_comparison() {
    local summary="$1"
    local query="$2"
    local mode="$3"
    local result="$4"
    local name="$5"
    local elapsed=""
    local e2e_ms=""
    local stage_1_ms=""
    local timing_file="$RUN_ROOT/$name/timing.txt"
    local log_file="$RUN_ROOT/$name/command.log"

    if [[ -f "$timing_file" ]]; then
        elapsed="$(awk -F '=' '$1 == "elapsed_seconds" { print $2 }' "$timing_file")"
    fi
    if [[ -f "$log_file" ]]; then
        e2e_ms="$(qwen3_tts_log_stat_ms "$log_file" e2e_wall_time_ms)"
        stage_1_ms="$(qwen3_tts_log_stat_ms "$log_file" e2e_stage_1_wall_time_ms)"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$query" "$mode" "$result" "$elapsed" "$e2e_ms" "$stage_1_ms" "$name" >>"$summary"
}

run_qwen3_tts_compare() {
    local summary="$RUN_ROOT/qwen3_tts_eager_vs_miopen_fast.tsv"
    local query mode name deploy_config result
    local status=0
    local -a env_command

    mkdir -p "$RUN_ROOT"
    printf 'query\tmode\tstatus\tprocess_elapsed_seconds\te2e_wall_time_ms\tstage_1_wall_time_ms\tresult_dir\n' >"$summary"
    for query in CustomVoice VoiceDesign Base; do
        for mode in eager graph_miopen_fast; do
            if [[ "$mode" == "eager" ]]; then
                name="qwen3_tts_${query,,}_eager"
                deploy_config="vllm_omni/deploy/qwen3_tts.yaml"
                env_command=(env -u MIOPEN_FIND_MODE)
            else
                name="qwen3_tts_${query,,}_graph_miopen_fast"
                deploy_config="mi300x_rocm_recipe_plan/configs/qwen3_tts_rocm_graph.yaml"
                env_command=(env MIOPEN_FIND_MODE=FAST)
            fi

            if run_qwen3_tts_case \
                "$query" \
                "$name" \
                "$deploy_config" \
                "${env_command[@]}"; then
                result="PASS"
            else
                result="FAIL"
                status=1
            fi
            append_qwen3_tts_comparison "$summary" "$query" "$mode" "$result" "$name"
        done
    done
    echo "Qwen3 TTS comparison: $summary"
    return "$status"
}

run_sensenova() {
    local name="sensenova"
    local output="$RUN_ROOT/$name/sensenova_u1_mi300x.png"
    run_profiled "$name" \
        "$PYTHON_BIN" examples/offline_inference/text_to_image/text_to_image.py \
        --model SenseNova/SenseNova-U1-8B-MoT \
        --prompt "Close portrait of an elderly woman by a farmhouse window, textured skin, gentle smile, warm natural light, emotional documentary look." \
        --width 1536 \
        --height 2720 \
        --seed 42 \
        --num-inference-steps 50 \
        --cfg-scale 4.0 \
        --extra-body '{"think": true, "cfg_norm": "none", "timestep_shift": 3.0, "t_eps": 0.02}' \
        --enable-diffusion-pipeline-profiler \
        --log-stats \
        --output "$output"
    validate_outputs "$name" image "$output" --expected-width 1536 --expected-height 2720
}

run_stable_audio() {
    local name="stable_audio"
    local output="$RUN_ROOT/$name/stable_audio_10s.wav"
    run_profiled "$name" \
        "$PYTHON_BIN" examples/offline_inference/text_to_audio/text_to_audio.py \
        --model stabilityai/stable-audio-open-1.0 \
        --prompt "A gentle piano melody with soft room ambience" \
        --negative-prompt "Low quality, distorted, noisy" \
        --seed 42 \
        --guidance-scale 7.0 \
        --audio-length 10.0 \
        --num-inference-steps 50 \
        --cache-backend tea_cache \
        --enable-diffusion-pipeline-profiler \
        --output "$output"
    validate_outputs "$name" audio "$output" --expected-sample-rate 44100
}

run_mammoth_preview() {
    local name="mammoth_preview"
    local output="$RUN_ROOT/$name/mammoth_t2i.png"
    run_profiled "$name" \
        "$PYTHON_BIN" examples/offline_inference/text_to_image/text_to_image.py \
        --model bytedance-research/MammothModa2-Preview \
        --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
        --prompt "A stylish woman riding a motorcycle in NYC, movie poster style" \
        --height 1024 \
        --width 1024 \
        --seed 42 \
        --extra-body '{"text_guidance_scale": 4.0, "cfg_range": [0.0, 1.0], "num_inference_steps": 50}' \
        --enable-diffusion-pipeline-profiler \
        --log-stats \
        --output "$output"
    validate_outputs "$name" image "$output" --expected-width 1024 --expected-height 1024
}

run_omnivoice() {
    local name="omnivoice"
    local output="$RUN_ROOT/$name/omnivoice_mi300x.wav"
    run_profiled "$name" \
        "$PYTHON_BIN" examples/offline_inference/text_to_speech/omnivoice/end2end.py \
        --model k2-fsa/OmniVoice \
        --deploy-config vllm_omni/deploy/omnivoice.yaml \
        --text "Hello, this is OmniVoice running on one AMD MI300X." \
        --seed 42 \
        --output "$output"
    validate_outputs "$name" audio "$output" --expected-sample-rate 24000 --min-rms 0.01
}

run_ming_flash_tts() {
    if [[ "${RUN_MING_FLASH_TTS:-0}" != "1" ]]; then
        echo "Ming Flash TTS is optional because the checkpoint download is 238 GB and ROCm is unverified." >&2
        echo "Set RUN_MING_FLASH_TTS=1 to run it." >&2
        return 2
    fi
    local name="ming_flash_tts"
    local output="$RUN_ROOT/$name/ming_flash_tts_mi300x.wav"
    run_profiled "$name" \
        "$PYTHON_BIN" examples/offline_inference/text_to_speech/ming_flash_omni_tts/end2end.py \
        --model Jonathan1909/Ming-flash-omni-2.0 \
        --deploy-config vllm_omni/deploy/ming_flash_omni_tts.yaml \
        --case basic \
        --text "你好，这是 AMD MI300X ROCm 配方验证。" \
        --log-stats \
        --output "$output"
    validate_outputs "$name" audio "$output" --expected-sample-rate 44100
}

case "$MODEL_KEY" in
    ming_omni_tts) run_ming_omni_tts ;;
    qwen3_tts) run_qwen3_tts ;;
    qwen3_tts_compare) run_qwen3_tts_compare ;;
    sensenova) run_sensenova ;;
    stable_audio) run_stable_audio ;;
    mammoth_preview) run_mammoth_preview ;;
    omnivoice) run_omnivoice ;;
    ming_flash_tts) run_ming_flash_tts ;;
    *)
        usage >&2
        exit 2
        ;;
esac
