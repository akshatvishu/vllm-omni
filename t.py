#!/usr/bin/env python3
"""Audio transcription script using Whisper Large based on vLLM-Omni media.py platform & VRAM management."""

import argparse
import concurrent.futures
import fcntl
import gc
import multiprocessing
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import whisper
from vllm_omni.platforms import current_omni_platform


@contextmanager
def _serialize_whisper_model_download(model_size: str = "large"):
    """Serialize Whisper model download/load across processes using a file lock."""
    lock_path = Path.home() / ".cache" / "whisper" / f".{model_size}_model_download.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _whisper_transcribe_in_current_process(output_path: str, model_size: str = "large") -> str:
    """Load model, run transcription on platform accelerator (CUDA/ROCm/NPU/XPU), and clean VRAM."""
    device_index = None

    if current_omni_platform.is_available():
        n = current_omni_platform.get_device_count()
        # Borrow spare GPU if multi-GPU environment, else use device 0
        if n >= 1:
            device_index = n - 1
        else:
            device_index = 0

    if device_index is not None:
        torch_device = current_omni_platform.get_torch_device(device_index)
        current_omni_platform.set_device(torch_device)
        device = str(torch_device)
        use_accelerator = True
    else:
        use_accelerator = False
        device = "cpu"

    print(f"[+] Platform: {current_omni_platform.device_name} | Device: {device} | Model: {model_size}")

    with _serialize_whisper_model_download(model_size):
        model = whisper.load_model(model_size, device=device)

    try:
        result = model.transcribe(
            output_path,
            temperature=0.0,
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        text = result.get("text", "")
    finally:
        del model
        gc.collect()
        if use_accelerator:
            current_omni_platform.synchronize()
            current_omni_platform.empty_cache()

    return text or ""


def transcribe_audio_file(output_path: str, model_size: str = "large") -> str:
    """Convert an audio file to text in an isolated spawn subprocess (matching media.py)."""
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
        future = executor.submit(_whisper_transcribe_in_current_process, output_path, model_size)
        return future.result()


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio file using Whisper (defaults to 'large')")
    parser.add_argument("audio_path", help="Path to input audio file (.wav, .mp3, .flac, etc.)")
    parser.add_argument("--model", default="large", help="Whisper model size (default: 'large')")

    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"Error: File '{args.audio_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Transcribing file: {args.audio_path}")
    transcript = transcribe_audio_file(args.audio_path, model_size=args.model)
    print("\n--- Transcription Result ---")
    print(transcript.strip())


if __name__ == "__main__":
    main()
