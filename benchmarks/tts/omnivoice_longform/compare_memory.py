# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import hashlib
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from benchmarks.tts.omnivoice_longform.common import read_jsonl, write_json, write_jsonl, write_text

BUCKET_ORDER = ("words_120", "words_200", "words_300", "words_400_plus")


def _percent_change(baseline: float, candidate: float) -> float:
    return 100 * (candidate - baseline) / baseline


def _percent_saved(baseline: float, candidate: float) -> float:
    return 100 * (baseline - candidate) / baseline


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(row[key] for row in rows)


def _snr(signal_squared_sum: float, error_squared_sum: float) -> tuple[float | None, str]:
    if error_squared_sum == 0:
        return None, "exact"
    if signal_squared_sum == 0:
        return None, "silent_baseline"
    return 10 * math.log10(signal_squared_sum / error_squared_sum), "finite"


def _compare_pcm(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_audio, baseline_rate = sf.read(baseline["audio_path"], dtype="float64", always_2d=True)
    candidate_audio, candidate_rate = sf.read(candidate["audio_path"], dtype="float64", always_2d=True)
    if baseline_rate != candidate_rate:
        return {
            "comparable": False,
            "reason": f"sample rate mismatch: {baseline_rate} != {candidate_rate}",
        }
    if baseline_audio.shape != candidate_audio.shape:
        return {
            "comparable": False,
            "reason": f"shape mismatch: {baseline_audio.shape} != {candidate_audio.shape}",
        }

    difference = baseline_audio - candidate_audio
    signal_squared_sum = float(np.square(baseline_audio).sum())
    error_squared_sum = float(np.square(difference).sum())
    sample_count = int(baseline_audio.size)
    rmse = math.sqrt(error_squared_sum / sample_count) if sample_count else 0.0
    snr_db, snr_status = _snr(signal_squared_sum, error_squared_sum)
    return {
        "comparable": True,
        "sample_rate": baseline_rate,
        "shape": list(baseline_audio.shape),
        "sample_count": sample_count,
        "max_absolute_error": float(np.max(np.abs(difference))) if sample_count else 0.0,
        "rmse": rmse,
        "snr_db": snr_db,
        "snr_status": snr_status,
        "signal_squared_sum": signal_squared_sum,
        "error_squared_sum": error_squared_sum,
    }


def _summarize_pcm(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = [comparison for comparison in comparisons if comparison["comparable"]]
    sample_count = sum(comparison["sample_count"] for comparison in comparable)
    signal_squared_sum = sum(comparison["signal_squared_sum"] for comparison in comparable)
    error_squared_sum = sum(comparison["error_squared_sum"] for comparison in comparable)
    snr_db, snr_status = _snr(signal_squared_sum, error_squared_sum)
    return {
        "comparable_cases": len(comparable),
        "mismatched_cases": len(comparisons) - len(comparable),
        "max_absolute_error": max(
            (comparison["max_absolute_error"] for comparison in comparable),
            default=None,
        ),
        "rmse": math.sqrt(error_squared_sum / sample_count) if sample_count else None,
        "snr_db": snr_db,
        "snr_status": snr_status if comparable else None,
    }


def _summarize_pair(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    pcm_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_id = {row["case_id"]: row for row in baseline_rows}
    candidate_by_id = {row["case_id"]: row for row in candidate_rows}
    case_ids = sorted(baseline_by_id)

    baseline_latency = _mean(baseline_rows, "latency_s")
    candidate_latency = _mean(candidate_rows, "latency_s")
    baseline_rtf = _mean(baseline_rows, "rtf")
    candidate_rtf = _mean(candidate_rows, "rtf")

    duration_differences = [
        abs(baseline_by_id[case_id]["audio_duration_s"] - candidate_by_id[case_id]["audio_duration_s"])
        for case_id in case_ids
    ]
    matching_audio = sum(
        baseline_by_id[case_id]["audio_sha256"] == candidate_by_id[case_id]["audio_sha256"] for case_id in case_ids
    )
    result = {
        "cases": len(case_ids),
        "latency_s": {
            "baseline_mean": baseline_latency,
            "candidate_mean": candidate_latency,
            "candidate_change_percent": _percent_change(baseline_latency, candidate_latency),
        },
        "rtf": {
            "baseline_mean": baseline_rtf,
            "candidate_mean": candidate_rtf,
            "candidate_change_percent": _percent_change(baseline_rtf, candidate_rtf),
        },
        "audio_duration_s": {
            "max_absolute_difference": max(duration_differences),
        },
        "wav_sha256": {
            "matching_cases": matching_audio,
            "all_cases_match": matching_audio == len(case_ids),
        },
        "pcm": _summarize_pcm([pcm_by_id[case_id] for case_id in case_ids]),
    }
    for key in ("peak_allocated_gib", "peak_reserved_gib"):
        baseline_mean = _mean(baseline_rows, key)
        candidate_mean = _mean(candidate_rows, key)
        baseline_max = max(row[key] for row in baseline_rows)
        candidate_max = max(row[key] for row in candidate_rows)
        result[key] = {
            "baseline_mean": baseline_mean,
            "candidate_mean": candidate_mean,
            "mean_saved": baseline_mean - candidate_mean,
            "mean_saved_percent": _percent_saved(baseline_mean, candidate_mean),
            "baseline_max": baseline_max,
            "candidate_max": candidate_max,
            "max_saved": baseline_max - candidate_max,
            "max_saved_percent": _percent_saved(baseline_max, candidate_max),
        }
    return result


def _validate_rows(rows: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    if not rows:
        raise ValueError(f"no benchmark rows found in {path}")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = row["case_id"]
        if case_id in by_id:
            raise ValueError(f"duplicate case {case_id} in {path}")
        if row.get("status") != "success":
            raise ValueError(f"case {case_id} did not succeed in {path}")
        if row.get("mode") != "chunked":
            raise ValueError(f"case {case_id} is not a chunked request in {path}")
        if row.get("concurrency") != 1:
            raise ValueError(f"case {case_id} did not run at concurrency 1 in {path}")
        for metric in ("peak_allocated_gib", "peak_reserved_gib", "latency_s", "rtf", "audio_duration_s"):
            if not isinstance(row.get(metric), int | float) or row[metric] <= 0:
                raise ValueError(f"case {case_id} has invalid {metric} in {path}")
        if not isinstance(row.get("audio_sha256"), str) or len(row["audio_sha256"]) != 64:
            raise ValueError(f"case {case_id} has invalid audio_sha256 in {path}")
        audio_path = row.get("audio_path")
        if not isinstance(audio_path, str) or not Path(audio_path).is_file():
            raise ValueError(f"case {case_id} has invalid audio_path in {path}")
        by_id[case_id] = row
    return by_id


def compare_results(
    baseline_path: Path,
    candidate_path: Path,
    *,
    revision: str | None = None,
    baseline_transform: Path | None = None,
) -> dict[str, Any]:
    baseline_rows = read_jsonl(baseline_path)
    candidate_rows = read_jsonl(candidate_path)
    baseline_by_id = _validate_rows(baseline_rows, baseline_path)
    candidate_by_id = _validate_rows(candidate_rows, candidate_path)
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("baseline and candidate do not contain the same cases")

    for case_id, baseline in baseline_by_id.items():
        candidate = candidate_by_id[case_id]
        for key in ("prompt_id", "bucket", "word_count", "text", "seed"):
            if baseline[key] != candidate[key]:
                raise ValueError(f"case {case_id} has different {key} values")

    pcm_by_id = {
        case_id: _compare_pcm(baseline, candidate_by_id[case_id]) for case_id, baseline in baseline_by_id.items()
    }

    cells = []
    for bucket in BUCKET_ORDER:
        bucket_ids = [case_id for case_id, row in baseline_by_id.items() if row["bucket"] == bucket]
        if not bucket_ids:
            continue
        cells.append(
            {
                "bucket": bucket,
                **_summarize_pair(
                    [baseline_by_id[case_id] for case_id in bucket_ids],
                    [candidate_by_id[case_id] for case_id in bucket_ids],
                    pcm_by_id,
                ),
            }
        )

    known_buckets = set(BUCKET_ORDER)
    unexpected_buckets = {row["bucket"] for row in baseline_rows} - known_buckets
    if unexpected_buckets:
        raise ValueError(f"unexpected word count buckets: {sorted(unexpected_buckets)}")

    summary = {
        "baseline": {
            "label": "GPU-retained chunks",
            "serving_rows": str(baseline_path.resolve()),
        },
        "candidate": {
            "label": "Chunks copied to CPU",
            "serving_rows": str(candidate_path.resolve()),
        },
        "overall": _summarize_pair(baseline_rows, candidate_rows, pcm_by_id),
        "cells": cells,
        "pcm_cases": [
            {
                "case_id": case_id,
                **{
                    key: value
                    for key, value in comparison.items()
                    if key not in {"signal_squared_sum", "error_squared_sum"}
                },
            }
            for case_id, comparison in sorted(pcm_by_id.items())
        ],
    }
    if revision is not None:
        summary["revision"] = revision
    if baseline_transform is not None:
        summary["baseline_transform"] = {
            "path": str(baseline_transform.resolve()),
            "sha256": hashlib.sha256(baseline_transform.read_bytes()).hexdigest(),
        }
    return summary


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# OmniVoice chunk memory comparison",
        "",
        (
            "Peak allocated memory measures live tensors and is the primary memory result. Peak reserved memory "
            "includes PyTorch's reusable allocator pool. A negative latency change means that the CPU-copy version "
            "was faster."
        ),
        "",
        (
            "| Word count | Cases | GPU-retained allocated max GiB | CPU-copy allocated max GiB | "
            "Allocated saved GiB | Allocated saved % | GPU-retained reserved max GiB | "
            "CPU-copy reserved max GiB | Candidate latency change % |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows = [*(summary["cells"]), {"bucket": "overall", **summary["overall"]}]
    for row in rows:
        allocated = row["peak_allocated_gib"]
        reserved = row["peak_reserved_gib"]
        latency = row["latency_s"]
        lines.append(
            f"| {row['bucket']} | {row['cases']} | {allocated['baseline_max']:.3f} | "
            f"{allocated['candidate_max']:.3f} | {allocated['max_saved']:.3f} | "
            f"{allocated['max_saved_percent']:.2f} | {reserved['baseline_max']:.3f} | "
            f"{reserved['candidate_max']:.3f} | {latency['candidate_change_percent']:.2f} |"
        )
    lines.extend(
        [
            "",
            (
                "Maximum paired audio-duration difference: "
                f"{summary['overall']['audio_duration_s']['max_absolute_difference']:.6f} seconds."
            ),
            (
                "Matching WAV files: "
                f"{summary['overall']['wav_sha256']['matching_cases']}/{summary['overall']['cases']}."
            ),
            (
                "Decoded PCM comparison: "
                f"{summary['overall']['pcm']['comparable_cases']}/{summary['overall']['cases']} comparable, "
                f"max absolute error {_format_metric(summary['overall']['pcm']['max_absolute_error'])}, "
                f"RMSE {_format_metric(summary['overall']['pcm']['rmse'])}, "
                f"SNR {_format_snr(summary['overall']['pcm'])}."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.8g}"


def _format_snr(pcm: dict[str, Any]) -> str:
    if pcm["snr_status"] == "exact":
        return "exact"
    if pcm["snr_status"] == "silent_baseline":
        return "undefined (silent baseline)"
    return "n/a" if pcm["snr_db"] is None else f"{pcm['snr_db']:.2f} dB"


def _clear_audio_paths(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        rows = read_jsonl(path)
        for row in rows:
            row["audio_path"] = None
        write_jsonl(path, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare OmniVoice GPU-retention and CPU-copy benchmark results")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision")
    parser.add_argument("--baseline-transform", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = compare_results(
        args.baseline,
        args.candidate,
        revision=args.revision,
        baseline_transform=args.baseline_transform,
    )
    write_json(args.output_dir / "comparison.json", summary)
    write_text(args.output_dir / "comparison.md", _markdown(summary))
    _clear_audio_paths(
        [
            args.baseline,
            args.baseline.with_name("generation.jsonl"),
            args.candidate,
            args.candidate.with_name("generation.jsonl"),
        ]
    )
    print(_markdown(summary), end="")


if __name__ == "__main__":
    main()
