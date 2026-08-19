# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

from benchmarks.tts.omnivoice_longform.common import (
    DEFAULT_SEEDS,
    MODES,
    build_generation_cases,
    case_asdict,
    chunking_args,
    load_prompt_manifest,
    read_jsonl,
    write_json,
    write_jsonl,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _synchronize() -> None:
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()


def _reset_peak_memory() -> dict[str, float]:
    if not torch.accelerator.is_available():
        return {}
    torch.accelerator.reset_peak_memory_stats()
    gib = 1024**3
    return {
        "allocated_before_gib": torch.accelerator.memory_allocated() / gib,
        "reserved_before_gib": torch.accelerator.memory_reserved() / gib,
    }


def _peak_memory() -> dict[str, float]:
    if not torch.accelerator.is_available():
        return {}
    gib = 1024**3
    return {
        "peak_allocated_gib": torch.accelerator.max_memory_allocated() / gib,
        "peak_reserved_gib": torch.accelerator.max_memory_reserved() / gib,
    }


def _generation_config(mode: str) -> OmniVoiceGenerationConfig:
    return OmniVoiceGenerationConfig(**chunking_args(mode))


def run(args: argparse.Namespace) -> None:
    _, prompts = load_prompt_manifest(args.manifest)
    cases = build_generation_cases(prompts, args.seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_path = output_dir / "generation.jsonl"
    existing_rows = read_jsonl(generation_path) if generation_path.exists() else []
    rows_by_case_id = {row["case_id"]: row for row in existing_rows}
    if len(rows_by_case_id) != len(existing_rows):
        raise ValueError(f"duplicate cases in checkpoint: {generation_path}")
    expected_case_ids = {case.case_id for case in cases}
    unexpected_case_ids = rows_by_case_id.keys() - expected_case_ids
    if unexpected_case_ids:
        raise ValueError(f"unexpected cases in checkpoint: {sorted(unexpected_case_ids)}")

    pending_cases = [case for case in cases if case.case_id not in rows_by_case_id]
    model = None
    if pending_cases:
        dtype = getattr(torch, args.dtype)
        model_path = Path(args.model)
        if not model_path.exists():
            model_path = Path(snapshot_download(args.model, revision=args.model_revision))
        load_start = time.perf_counter()
        model = OmniVoice.from_pretrained(
            model_path,
            device_map=args.device,
            dtype=dtype,
        )
        init_time_s = time.perf_counter() - load_start

        warmup_prompt = prompts[0]
        for mode in MODES:
            _seed_everything(args.seeds[0])
            model.generate(
                text=warmup_prompt.text,
                language="English",
                generation_config=_generation_config(mode),
            )
    else:
        init_time_s = existing_rows[0]["init_time_s"]

    for index, case in enumerate(cases, start=1):
        if case.case_id in rows_by_case_id:
            print(f"[{index}/{len(cases)}] {case.case_id}: restored from checkpoint")
            continue

        _seed_everything(case.seed)
        started = time.perf_counter()
        try:
            _synchronize()
            memory = _reset_peak_memory()
            audio = model.generate(
                text=case.text,
                language="English",
                generation_config=_generation_config(case.mode),
            )[0]
            _synchronize()
            latency_s = time.perf_counter() - started
            memory.update(_peak_memory())

            audio = np.asarray(audio, dtype=np.float32).squeeze()
            if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
                raise RuntimeError(f"invalid audio returned for {case.case_id}")

            audio_path = output_dir / f"{case.case_id}.wav"
            sf.write(audio_path, audio, model.sampling_rate)
            duration_s = audio.size / model.sampling_rate
            row = {
                **case_asdict(case),
                "backend": "reference",
                "status": "success",
                "order_index": index - 1,
                "audio_path": str(audio_path.resolve()),
                "sample_rate": model.sampling_rate,
                "audio_duration_s": duration_s,
                "latency_s": latency_s,
                "rtf": latency_s / duration_s,
                "init_time_s": init_time_s,
                **memory,
            }
            print(
                f"[{index}/{len(cases)}] {case.case_id}: {latency_s:.2f}s, "
                f"{duration_s:.2f}s audio, RTF {row['rtf']:.3f}"
            )
        except Exception as error:
            row = {
                **case_asdict(case),
                "backend": "reference",
                "status": "error",
                "order_index": index - 1,
                "audio_path": None,
                "sample_rate": model.sampling_rate,
                "audio_duration_s": None,
                "latency_s": time.perf_counter() - started,
                "rtf": None,
                "init_time_s": init_time_s,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            print(f"[{index}/{len(cases)}] {case.case_id}: failed: {type(error).__name__}: {error}")

        rows_by_case_id[case.case_id] = row
        write_jsonl(
            generation_path,
            [rows_by_case_id[item.case_id] for item in cases if item.case_id in rows_by_case_id],
        )

    rows = [rows_by_case_id[case.case_id] for case in cases]
    successful_requests = sum(row["status"] == "success" for row in rows)
    summary = {
        "backend": "reference",
        "model": args.model,
        "model_revision": args.model_revision,
        "dtype": args.dtype,
        "device": args.device,
        "init_time_s": init_time_s,
        "warmup_requests": len(MODES),
        "measured_requests": len(rows),
        "successful_requests": successful_requests,
        "failed_requests": len(rows) - successful_requests,
        "batch_size": 1,
        "concurrency": 1,
        "seeds": args.seeds,
        "case_order": [case.case_id for case in cases],
    }
    write_json(output_dir / "summary.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the reference OmniVoice implementation")
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--model-revision")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
