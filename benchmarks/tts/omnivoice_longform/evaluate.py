# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import scipy.signal
import soundfile as sf
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration

from benchmarks.tts.omnivoice_longform.common import (
    mean_and_stddev,
    normalize_text,
    read_jsonl,
    write_json,
    write_jsonl,
    write_text,
)

DEFAULT_WHISPER_MODEL = "openai/whisper-large-v3"


def load_audio_16k(path: str | Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sample_rate != 16000:
        divisor = math.gcd(sample_rate, 16000)
        mono = scipy.signal.resample_poly(
            mono,
            16000 // divisor,
            sample_rate // divisor,
        )
    return np.ascontiguousarray(mono, dtype=np.float32)


def transcribe_waveform(
    waveform_16k: np.ndarray,
    processor,
    model,
    *,
    device: str,
    dtype: torch.dtype,
) -> str:
    inputs = processor(
        waveform_16k,
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
        truncation=False,
        padding="longest",
    )
    input_features = inputs.input_features.to(device=device, dtype=dtype)
    attention_mask = inputs.attention_mask.to(device=device)
    with torch.inference_mode():
        predicted_ids = model.generate(
            input_features,
            attention_mask=attention_mask,
            language="english",
            task="transcribe",
            return_timestamps=True,
        )
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def score_transcript(reference: str, hypothesis: str) -> dict[str, Any]:
    normalized_reference = normalize_text(reference)
    normalized_hypothesis = normalize_text(hypothesis)
    alignment = jiwer.process_words(normalized_reference, normalized_hypothesis)
    reference_words = alignment.hits + alignment.substitutions + alignment.deletions
    if reference_words == 0:
        raise ValueError("reference text contains no words")
    coverage = 1.0 - (alignment.substitutions + alignment.deletions) / reference_words
    return {
        "reference_normalized": normalized_reference,
        "transcript_normalized": normalized_hypothesis,
        "reference_words": reference_words,
        "hits": alignment.hits,
        "substitutions": alignment.substitutions,
        "deletions": alignment.deletions,
        "insertions": alignment.insertions,
        "wer": alignment.wer,
        "coverage": coverage,
    }


def _validate_backend_cases(rows: list[dict[str, Any]]) -> None:
    cases_by_backend: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        case_id = row["case_id"]
        backend = row["backend"]
        if case_id in cases_by_backend[backend]:
            raise ValueError(f"duplicate case {case_id} for backend {backend}")
        cases_by_backend[backend].add(case_id)

    if len(cases_by_backend) != 2:
        raise ValueError(f"expected two backends, got {sorted(cases_by_backend)}")
    case_sets = list(cases_by_backend.values())
    if case_sets[0] != case_sets[1]:
        raise ValueError("reference and vLLM-Omni records do not contain the same cases")


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["backend"], row["mode"], row["bucket"])].append(row)

    summaries = []
    for (backend, mode, bucket), group in sorted(grouped.items()):
        scored = [row for row in group if row.get("evaluation_status") == "success"]
        generated = [row for row in group if row.get("status", "success") == "success"]
        generation_failures = sum(row.get("evaluation_status") == "generation_error" for row in group)
        transcription_failures = sum(row.get("evaluation_status") == "transcription_error" for row in group)
        summaries.append(
            {
                "backend": backend,
                "mode": mode,
                "bucket": bucket,
                "word_count": {
                    "min": min(row["word_count"] for row in group),
                    "max": max(row["word_count"] for row in group),
                    "mean": sum(row["word_count"] for row in group) / len(group),
                },
                "samples": len(group),
                "scored_samples": len(scored),
                "generation_failures": generation_failures,
                "transcription_failures": transcription_failures,
                "failure_rate": (generation_failures + transcription_failures) / len(group),
                "coverage": mean_and_stddev([row["coverage"] for row in scored]) if scored else None,
                "wer": mean_and_stddev([row["wer"] for row in scored]) if scored else None,
                "latency_s": mean_and_stddev([row["latency_s"] for row in generated]) if generated else None,
                "rtf": mean_and_stddev([row["rtf"] for row in generated]) if generated else None,
            }
        )
    return summaries


def _summary_markdown(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# OmniVoice long form benchmark",
        "",
        "Coverage is `1 - (substitutions + deletions) / reference words`. Insertions are included in WER.",
        "",
        "| Backend | Mode | Bucket | Words min/mean/max | Scored/total | Failures | Coverage mean "
        "| Coverage stddev | WER mean | Latency mean (s) | RTF mean |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        coverage = row["coverage"]
        wer = row["wer"]
        latency = row["latency_s"]
        rtf = row["rtf"]
        coverage_mean = f"{100 * coverage['mean']:.2f}%" if coverage else "N/A"
        coverage_stddev = f"{100 * coverage['stddev']:.2f}%" if coverage else "N/A"
        wer_mean = f"{wer['mean']:.4f}" if wer else "N/A"
        latency_mean = f"{latency['mean']:.2f}" if latency else "N/A"
        rtf_mean = f"{rtf['mean']:.3f}" if rtf else "N/A"
        lines.append(
            f"| {row['backend']} | {row['mode']} | {row['bucket']} "
            f"| {row['word_count']['min']}/{row['word_count']['mean']:.1f}/{row['word_count']['max']} "
            f"| {row['scored_samples']}/{row['samples']} "
            f"| {row['generation_failures'] + row['transcription_failures']} "
            f"| {coverage_mean} | {coverage_stddev} | {wer_mean} | {latency_mean} | {rtf_mean} |"
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    rows = [row for path in args.records for row in read_jsonl(path)]
    _validate_backend_cases(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluated_path = output_dir / "evaluated.jsonl"
    existing_rows = read_jsonl(evaluated_path) if evaluated_path.exists() else []
    evaluated_by_case = {(row["backend"], row["case_id"]): row for row in existing_rows}
    if len(evaluated_by_case) != len(existing_rows):
        raise ValueError(f"duplicate cases in checkpoint: {evaluated_path}")

    backend_order = {"reference": 0, "vllm-omni": 1}
    rows.sort(key=lambda row: (row.get("order_index", 0), backend_order.get(row["backend"], 2)))
    expected_cases = {(row["backend"], row["case_id"]) for row in rows}
    unexpected_cases = evaluated_by_case.keys() - expected_cases
    if unexpected_cases:
        raise ValueError(f"unexpected cases in checkpoint: {sorted(unexpected_cases)}")

    dtype = getattr(torch, args.dtype)
    pending_audio = any(
        (row["backend"], row["case_id"]) not in evaluated_by_case and row.get("status", "success") == "success"
        for row in rows
    )
    processor = None
    model = None
    if pending_audio:
        processor = AutoProcessor.from_pretrained(
            args.whisper_model,
            revision=args.model_revision,
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            args.whisper_model,
            revision=args.model_revision,
            torch_dtype=dtype,
        ).to(args.device)
        model.eval()

    for index, row in enumerate(rows, start=1):
        case_key = (row["backend"], row["case_id"])
        if case_key in evaluated_by_case:
            print(f"[{index}/{len(rows)}] {row['backend']} {row['case_id']}: restored from checkpoint")
            continue

        if row.get("status", "success") != "success":
            evaluated_row = {
                **row,
                "evaluation_status": "generation_error",
                "evaluation_order_index": index - 1,
            }
        else:
            audio_path = row.get("audio_path")
            if not audio_path:
                evaluated_row = {
                    **row,
                    "evaluation_status": "generation_error",
                    "evaluation_order_index": index - 1,
                    "error_type": "MissingAudioPath",
                    "error": "successful generation row has no saved audio path",
                }
            else:
                try:
                    transcript = transcribe_waveform(
                        load_audio_16k(audio_path),
                        processor,
                        model,
                        device=args.device,
                        dtype=dtype,
                    )
                    evaluated_row = {
                        **row,
                        "evaluation_status": "success",
                        "evaluation_order_index": index - 1,
                        "transcript": transcript,
                        **score_transcript(row["text"], transcript),
                    }
                    print(
                        f"[{index}/{len(rows)}] {row['backend']} {row['case_id']}: "
                        f"coverage {100 * evaluated_row['coverage']:.2f}%, WER {evaluated_row['wer']:.4f}"
                    )
                except Exception as error:
                    evaluated_row = {
                        **row,
                        "evaluation_status": "transcription_error",
                        "evaluation_order_index": index - 1,
                        "transcription_error_type": type(error).__name__,
                        "transcription_error": str(error),
                    }
                    print(
                        f"[{index}/{len(rows)}] {row['backend']} {row['case_id']}: "
                        f"transcription failed: {type(error).__name__}: {error}"
                    )

        evaluated_by_case[case_key] = evaluated_row
        write_jsonl(
            evaluated_path,
            [
                evaluated_by_case[(item["backend"], item["case_id"])]
                for item in rows
                if (item["backend"], item["case_id"]) in evaluated_by_case
            ],
        )

    evaluated = [evaluated_by_case[(row["backend"], row["case_id"])] for row in rows]
    summaries = _aggregate(evaluated)
    write_json(output_dir / "summary.json", summaries)
    write_text(output_dir / "summary.md", _summary_markdown(summaries))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe and score OmniVoice benchmark audio")
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--model-revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
