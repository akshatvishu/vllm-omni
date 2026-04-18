#!/bin/bash
# Common curl examples for Ming-omni-tts via /v1/audio/speech.
#
# Usage:
#   ./run_curl.sh basic
#   ./run_curl.sh style
#   ./run_curl.sh ip
#   REF_AUDIO=/path/to/ref.wav ./run_curl.sh emotion
#   REF_AUDIO=/path/to/ref.wav ./run_curl.sh dialect
#   REF_AUDIO=/path/to/ref.wav REF_TEXT="参考文本" ./run_curl.sh zero_shot
#   REF_AUDIO=/path/to/speaker1.wav REF_AUDIO_2=/path/to/speaker2.wav REF_TEXT="speaker_1:... speaker_2:..." ./run_curl.sh podcast
#   REF_AUDIO=/path/to/mix_ref.wav ./run_curl.sh speech_bgm
#   REF_AUDIO=/path/to/mix_ref.wav ./run_curl.sh speech_sound
#   REF_AUDIO=/path/to/ref.wav REF_TEXT="参考文本" ./run_curl.sh clone_ref_audio
#   SPEAKER_EMBEDDING=/path/to/ming_embedding.json ./run_curl.sh clone_embedding
#   ./run_curl.sh stream

set -euo pipefail

MODE="${1:-basic}"
HOST="${HOST:-localhost}"
PORT="${PORT:-8091}"
MODEL="${MODEL:-inclusionAI/Ming-omni-tts-0.5B}"
API_URL="http://${HOST}:${PORT}/v1/audio/speech"
TEXT="${TEXT:-你好，这是 Ming 在线语音合成测试。}"
OUTPUT="${OUTPUT:-ming_output.wav}"
STREAM_OUTPUT="${STREAM_OUTPUT:-ming_output.pcm}"
REF_AUDIO="${REF_AUDIO:-}"
REF_AUDIO_2="${REF_AUDIO_2:-}"
REF_TEXT="${REF_TEXT:-}"
SPEAKER_EMBEDDING="${SPEAKER_EMBEDDING:-}"

build_payload() {
    MODEL="$1" \
    TEXT="$2" \
    VOICE="$3" \
    INSTRUCTIONS="$4" \
    TASK_TYPE="$5" \
    REF_AUDIO_PATH="$6" \
    REF_TEXT="$7" \
    SPEAKER_EMBEDDING_PATH="$8" \
    STREAM="$9" \
    REF_AUDIO_PATH_2="${10:-}" \
    python - <<'PY'
import base64
import json
import mimetypes
import os
import pathlib
import sys

payload = {
    "model": os.environ["MODEL"],
    "input": os.environ["TEXT"],
}

voice = os.environ["VOICE"]
instructions = os.environ["INSTRUCTIONS"]
task_type = os.environ["TASK_TYPE"]
ref_audio_path = os.environ["REF_AUDIO_PATH"]
ref_audio_path_2 = os.environ["REF_AUDIO_PATH_2"]
ref_text = os.environ["REF_TEXT"]
speaker_embedding_path = os.environ["SPEAKER_EMBEDDING_PATH"]

if voice:
    payload["voice"] = voice
if instructions:
    payload["instructions"] = instructions
if task_type:
    payload["task_type"] = task_type
ref_audio_items = []
if ref_audio_path:
    path = pathlib.Path(ref_audio_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    ref_audio_items.append(f"data:{mime_type};base64,{data}")
if ref_audio_path_2:
    path = pathlib.Path(ref_audio_path_2)
    mime_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    ref_audio_items.append(f"data:{mime_type};base64,{data}")
if ref_audio_items:
    payload["ref_audio"] = ref_audio_items[0] if len(ref_audio_items) == 1 else ref_audio_items
if ref_text:
    payload["ref_text"] = ref_text
if speaker_embedding_path:
    path = pathlib.Path(speaker_embedding_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("speaker embedding file must contain a JSON list")
    payload["speaker_embedding"] = data

stream = os.environ["STREAM"] == "true"
if stream:
    payload["stream"] = True
    payload["response_format"] = "pcm"
else:
    payload["response_format"] = "wav"

print(json.dumps(payload, ensure_ascii=False))
PY
}

require_file() {
    local path="$1"
    local flag_name="$2"
    if [ -z "$path" ]; then
        echo "Missing ${flag_name}" >&2
        exit 1
    fi
    if [ ! -f "$path" ]; then
        echo "File not found for ${flag_name}: $path" >&2
        exit 1
    fi
}

base_headers=(
    -H "Content-Type: application/json"
    -H "Authorization: Bearer EMPTY"
)

post_payload() {
    local payload="$1"
    local output_path="$2"
    local payload_file
    payload_file="$(mktemp)"
    trap 'rm -f "$payload_file"' RETURN
    printf '%s' "$payload" > "$payload_file"
    curl -X POST "$API_URL" "${base_headers[@]}" \
        --data-binary "@${payload_file}" \
        --output "$output_path"
}

case "$MODE" in
    basic)
        PAYLOAD="$(build_payload "$MODEL" "$TEXT" "" "" "" "" "" "" "false")"
        post_payload "$PAYLOAD" "$OUTPUT"
        ;;
    style)
        PAYLOAD="$(build_payload "$MODEL" "$TEXT" "" "轻柔的ASMR耳语，慢速，贴近麦克风" "" "" "" "" "false")"
        post_payload "$PAYLOAD" "$OUTPUT"
        ;;
    ip)
        PAYLOAD="$(build_payload "$MODEL" "$TEXT" "灵小甄" "" "" "" "" "" "false")"
        post_payload "$PAYLOAD" "$OUTPUT"
        ;;
    emotion)
        require_file "$REF_AUDIO" "REF_AUDIO"
        PAYLOAD="$(build_payload "$MODEL" "$TEXT" "" '{"情感":"高兴"}' "" "$REF_AUDIO" "" "" "false")"
        post_payload "$PAYLOAD" "$OUTPUT"
        ;;
    dialect)
        require_file "$REF_AUDIO" "REF_AUDIO"
        PAYLOAD="$(build_payload "$MODEL" "$TEXT" "" "" "" "$REF_AUDIO" "" "" "false")"
        PAYLOAD="$(TEXT="$PAYLOAD" python - <<'PY'
import json
import os
payload = json.loads(os.environ["TEXT"])
payload["language"] = "广粤话"
print(json.dumps(payload, ensure_ascii=False))
PY
)"
        post_payload "$PAYLOAD" "$OUTPUT"
        ;;
    zero_shot)
        require_file "$REF_AUDIO" "REF_AUDIO"
        if [ -z "$REF_TEXT" ]; then
            echo "Missing REF_TEXT" >&2
            exit 1
        fi
        PAYLOAD="$(build_payload "$MODEL" "$TEXT" "" "" "Base" "$REF_AUDIO" "$REF_TEXT" "" "false")"
        post_payload "$PAYLOAD" "$OUTPUT"
        ;;
    podcast)
        require_file "$REF_AUDIO" "REF_AUDIO"
        require_file "$REF_AUDIO_2" "REF_AUDIO_2"
        if [ -z "$REF_TEXT" ]; then
            echo "Missing REF_TEXT" >&2
            exit 1
        fi
        PAYLOAD="$(build_payload "$MODEL" "$TEXT" "" "" "Base" "$REF_AUDIO" "$REF_TEXT" "" "false" "$REF_AUDIO_2")"
        post_payload "$PAYLOAD" "$OUTPUT"
        ;;
    speech_bgm)
        require_file "$REF_AUDIO" "REF_AUDIO"
        PAYLOAD="$(build_payload "$MODEL" "$TEXT" "" '{"BGM":"舒缓的背景音乐"}' "" "$REF_AUDIO" "" "" "false")"
        post_payload "$PAYLOAD" "$OUTPUT"
        ;;
    speech_sound)
        require_file "$REF_AUDIO" "REF_AUDIO"
        PAYLOAD="$(build_payload "$MODEL" "$TEXT" "" '{"BGM":{"ENV":"轻微的环境声"}}' "" "$REF_AUDIO" "" "" "false")"
        post_payload "$PAYLOAD" "$OUTPUT"
        ;;
    clone_ref_audio)
        require_file "$REF_AUDIO" "REF_AUDIO"
        if [ -z "$REF_TEXT" ]; then
            echo "Missing REF_TEXT" >&2
            exit 1
        fi
        PAYLOAD="$(build_payload "$MODEL" "$TEXT" "" "" "Base" "$REF_AUDIO" "$REF_TEXT" "" "false")"
        post_payload "$PAYLOAD" "$OUTPUT"
        ;;
    clone_embedding)
        require_file "$SPEAKER_EMBEDDING" "SPEAKER_EMBEDDING"
        PAYLOAD="$(build_payload "$MODEL" "$TEXT" "" "" "Base" "" "" "$SPEAKER_EMBEDDING" "false")"
        post_payload "$PAYLOAD" "$OUTPUT"
        ;;
    stream)
        PAYLOAD="$(build_payload "$MODEL" "$TEXT" "" "平静，普通话" "" "" "" "" "true")"
        post_payload "$PAYLOAD" "$STREAM_OUTPUT"
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        echo "Supported: basic, style, ip, emotion, dialect, zero_shot, podcast, speech_bgm, speech_sound, clone_ref_audio, clone_embedding, stream" >&2
        exit 1
        ;;
esac
