#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

usage() {
    echo "Usage: $0 {ming_omni_tts|qwen3_tts|sensenova|stable_audio|mammoth_preview|omnivoice|ming_flash_tts}"
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

run_qwen3_tts() {
    local query name output_dir
    local status=0
    for query in CustomVoice VoiceDesign Base; do
        name="qwen3_tts_${query,,}"
        output_dir="$RUN_ROOT/$name/output"
        mkdir -p "$output_dir"
        if ! run_profiled "$name" \
            "$PYTHON_BIN" examples/offline_inference/text_to_speech/qwen3_tts/end2end.py \
            --query-type "$query" \
            --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
            --output-dir "$output_dir" \
            --num-prompts 1 \
            --batch-size 1 \
            --log-stats \
            --log-dir "$RUN_ROOT/$name/logs"; then
            status=1
            continue
        fi
        if ! validate_outputs "$name" audio "$output_dir/*.wav" --expected-sample-rate 24000; then
            status=1
        fi
    done
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
