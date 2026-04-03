"""Client for Ming-omni-tts via /v1/audio/speech."""

import argparse
import base64
import json
import os

import httpx

DEFAULT_API_BASE = "http://localhost:8091"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MODEL = "inclusionAI/Ming-omni-tts-0.5B"


def encode_audio_to_base64(audio_path: str) -> str:
    """Encode a local audio file to a base64 data URL."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    ext = audio_path.lower().rsplit(".", 1)[-1]
    mime_map = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
        "aac": "audio/aac",
    }
    mime_type = mime_map.get(ext, "audio/wav")
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{audio_b64}"


def build_instruction_payload(args) -> str | None:
    """Return a string payload for the API `instructions` field."""
    if args.instructions and args.instruction_json:
        raise ValueError("Use either --instructions or --instruction-json, not both")
    if args.instruction_json:
        parsed = json.loads(args.instruction_json)
        return json.dumps(parsed, ensure_ascii=False)
    return args.instructions


def run_tts(args) -> None:
    """Generate speech via the OpenAI-compatible /v1/audio/speech API."""
    payload = {
        "model": args.model,
        "input": args.text,
        "response_format": args.response_format,
    }

    if args.voice:
        payload["voice"] = args.voice
    if args.task_type:
        payload["task_type"] = args.task_type
    if args.dialect:
        payload["language"] = args.dialect

    instructions = build_instruction_payload(args)
    if instructions:
        payload["instructions"] = instructions

    if args.ref_audio:
        if args.ref_audio.startswith(("http://", "https://", "data:")):
            payload["ref_audio"] = args.ref_audio
        else:
            payload["ref_audio"] = encode_audio_to_base64(args.ref_audio)
    if args.ref_text:
        payload["ref_text"] = args.ref_text
    if args.speaker_embedding:
        payload["speaker_embedding"] = json.loads(open(args.speaker_embedding, encoding="utf-8").read())
    if args.max_new_tokens:
        payload["max_new_tokens"] = args.max_new_tokens
    if args.stream:
        payload["stream"] = True
        payload["response_format"] = "pcm"

    api_url = f"{args.api_base}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    print(f"Model: {args.model}")
    print(f"Text: {args.text}")
    print(f"Payload keys: {sorted(payload)}")

    if args.stream:
        output_path = args.output or "ming_output.pcm"
        with httpx.Client(timeout=300.0) as client:
            with client.stream("POST", api_url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    print(f"Error: {response.status_code}")
                    print(response.read().decode())
                    return
                with open(output_path, "wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
        print(f"Streamed PCM audio to: {output_path}")
        return

    with httpx.Client(timeout=300.0) as client:
        response = client.post(api_url, json=payload, headers=headers)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    try:
        text = response.content.decode("utf-8")
        if text.startswith('{"error"'):
            print(f"Error: {text}")
            return
    except UnicodeDecodeError:
        pass

    output_path = args.output or "ming_output.wav"
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Audio saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Ming-omni-tts speech client")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="API base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Model name or path")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument(
        "--task-type",
        default=None,
        choices=["CustomVoice", "VoiceDesign", "Base"],
        help="Optional compatibility task type. Ming accepts the same field but primarily uses prompt metadata.",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Maps to Ming `IP` when using built-in character voices, or to an uploaded voice sample name",
    )
    parser.add_argument("--dialect", default=None, help="Maps to Ming `方言`")
    parser.add_argument("--instructions", default=None, help="Free-form Ming instruction string")
    parser.add_argument(
        "--instruction-json",
        default=None,
        help="Structured Ming instruction JSON, for example '{\"情感\":\"高兴\"}'",
    )
    parser.add_argument("--ref-audio", default=None, help="Reference audio path, URL, or data URL")
    parser.add_argument("--ref-text", default=None, help="Reference transcript for cloning")
    parser.add_argument("--speaker-embedding", default=None, help="Path to a JSON file containing a 192-d speaker embedding")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override ming_max_decode_steps")
    parser.add_argument("--stream", action="store_true", help="Enable streaming PCM output")
    parser.add_argument(
        "--response-format",
        default="wav",
        choices=["wav", "mp3", "flac", "pcm", "aac", "opus"],
        help="Audio format when not streaming",
    )
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    args = parser.parse_args()
    run_tts(args)


if __name__ == "__main__":
    main()
