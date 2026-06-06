"""Offline VoxCPM2 inference example (native AR pipeline).

Uses the single-stage native AR config (voxcpm2.yaml).
Requires the `voxcpm` package or VLLM_OMNI_VOXCPM_CODE_PATH env var.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import soundfile as sf
import torch

import json
import argparse
from vllm_omni import Omni
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

REPO_ROOT = Path(__file__).resolve().parents[4]
SAMPLE_RATE = 48_000


def parse_profiler_config(value: str) -> dict:
    try:
        config = json.loads(value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"--profiler-config must be valid JSON: {e}") from e
    if not isinstance(config, dict):
        raise argparse.ArgumentTypeError("--profiler-config must be a JSON object")
    return config


def parse_args():
    parser = TrackingArgumentParser(description="Offline VoxCPM2 native AR inference")
    parser.add_argument(
        "--model",
        type=str,
        default="openbmb/VoxCPM2",
        help="VoxCPM2 model path or HuggingFace repo ID.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="This is a VoxCPM2 native AR synthesis example running on vLLM Omni.",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_audio",
        help="Directory for output WAV files.",
    )
    parser.add_argument(
        "--deploy-config",
        type=str,
        default=None,
        help="Override the deploy config path. If unset, auto-loads "
        "vllm_omni/deploy/voxcpm2.yaml based on the HF model_type.",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Path to reference audio for voice cloning.",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Optional transcript of --ref-audio (enables continuation mode).",
    )
    parser.add_argument(
        "--profiler-config",
        type=parse_profiler_config,
        default=None,
        help='JSON profiler config for torch/cuda profiling, e.g. \'{"profiler":"torch","torch_profiler_dir":"./perf"}\'.',
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=0,
        help="Number of unprofiled warmup requests to run before timing/profiling.",
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        default=False,
        help="Enable logging of engine performance statistics.",
    )
    return parser.parse_args()


def extract_audio(multimodal_output: dict) -> torch.Tensor:
    """Extract the final complete audio tensor from multimodal output.

    The output processor concatenates per-step delta tensors under
    ``model_outputs``.  Falls back to ``audio`` for backwards compat.
    """
    audio = multimodal_output.get("model_outputs")
    if audio is None:
        audio = multimodal_output.get("audio")
    if audio is None:
        raise ValueError(f"No audio key in multimodal_output: {list(multimodal_output.keys())}")

    if isinstance(audio, list):
        # Defensive: usually the output processor consolidates into a single
        # tensor at request completion, but concatenate here too in case the
        # caller consumes intermediate (pre-consolidation) outputs.
        valid = [torch.as_tensor(a).float().cpu().reshape(-1) for a in audio if a is not None]
        if not valid:
            raise ValueError("Audio list is empty or all elements are None.")
        return torch.cat(valid, dim=0) if len(valid) > 1 else valid[0]

    return torch.as_tensor(audio).float().cpu().reshape(-1)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profiler_enabled = args.profiler_config is not None
    engine = Omni(
        model=args.model,
        deploy_config=args.deploy_config,
        profiler_config=args.profiler_config,
        trust_remote_code=True,
        log_stats=args.log_stats,
    )

    from transformers import AutoTokenizer

    from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import (
        build_cjk_split_map,
        build_voxcpm2_prompt,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    split_map = build_cjk_split_map(tokenizer)
    hf_config = engine.engine.stage_vllm_configs[0].model_config.hf_config

    ref_audio_arg = args.ref_audio
    ref_text_arg = args.ref_text
    ref_wav, ref_sr = (None, None)
    if ref_audio_arg:
        ref_wav_arr, ref_sr = sf.read(ref_audio_arg)
        ref_wav = ref_wav_arr.mean(axis=-1).tolist() if ref_wav_arr.ndim > 1 else ref_wav_arr.tolist()

    prompt = build_voxcpm2_prompt(
        hf_config=hf_config,
        tokenizer=tokenizer,
        split_map=split_map,
        text=args.text,
        ref_audio=ref_wav,
        ref_sr=ref_sr,
        ref_text=ref_text_arg,
    )

    print(f"Model       : {args.model}")
    print(f"Text        : {args.text}")
    if ref_audio_arg:
        print(f"Ref audio   : {ref_audio_arg}")
    if ref_text_arg:
        print(f"Ref text    : {ref_text_arg}")
    print(f"Output dir  : {output_dir}")

    num_warmups = max(0, args.num_warmups)
    for warmup_idx in range(num_warmups):
        print(f"[Warmup] Running request {warmup_idx + 1}/{num_warmups}...")
        warmup_outputs = engine.generate([prompt])
        del warmup_outputs
    if num_warmups:
        print("[Warmup] Complete.")

    if profiler_enabled:
        engine.start_profile()

    t_start = time.perf_counter()
    outputs = engine.generate([prompt])
    elapsed = time.perf_counter() - t_start

    if profiler_enabled:
        print("\n[Profiler] Stopping profiler and collecting results...")
        engine.stop_profile()

    # outputs[0].outputs[0].multimodal_output["audio"] is a list of tensors
    request_output = outputs[0]
    mm = request_output.outputs[0].multimodal_output
    audio = extract_audio(mm)

    duration = audio.numel() / SAMPLE_RATE
    rtf = elapsed / duration if duration > 0 else float("inf")

    output_path = output_dir / "output.wav"
    sf.write(str(output_path), audio.numpy(), SAMPLE_RATE, format="WAV")

    print(f"Saved       : {output_path}")
    print(f"Duration    : {duration:.2f}s")
    print(f"Inference   : {elapsed:.2f}s")
    print(f"RTF         : {rtf:.3f}")


if __name__ == "__main__":
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
