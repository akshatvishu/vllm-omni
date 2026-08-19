# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import hashlib
import statistics
from pathlib import Path
from typing import Any

from benchmarks.tts.omnivoice_longform.common import read_jsonl, write_json, write_text

BUCKET_ORDER = ("words_120", "words_200", "words_300", "words_400_plus")


def _percent_change(baseline: float, candidate: float) -> float:
    return 100 * (candidate - baseline) / baseline


def _percent_saved(baseline: float, candidate: float) -> float:
    return 100 * (baseline - candidate) / baseline


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(row[key] for row in rows)


def _summarize_pair(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_id = {row["case_id"]: row for row in baseline_rows}
    candidate_by_id = {row["case_id"]: row for row in candidate_rows}
    case_ids = sorted(baseline_by_id)

    baseline_peak_mean = _mean(baseline_rows, "peak_reserved_gib")
    candidate_peak_mean = _mean(candidate_rows, "peak_reserved_gib")
    baseline_peak_max = max(row["peak_reserved_gib"] for row in baseline_rows)
    candidate_peak_max = max(row["peak_reserved_gib"] for row in candidate_rows)
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
    return {
        "cases": len(case_ids),
        "peak_reserved_gib": {
            "baseline_mean": baseline_peak_mean,
            "candidate_mean": candidate_peak_mean,
            "mean_saved": baseline_peak_mean - candidate_peak_mean,
            "mean_saved_percent": _percent_saved(baseline_peak_mean, candidate_peak_mean),
            "baseline_max": baseline_peak_max,
            "candidate_max": candidate_peak_max,
            "max_saved": baseline_peak_max - candidate_peak_max,
            "max_saved_percent": _percent_saved(baseline_peak_max, candidate_peak_max),
        },
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
        "audio_sha256": {
            "matching_cases": matching_audio,
            "all_cases_match": matching_audio == len(case_ids),
        },
    }


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
        for metric in ("peak_reserved_gib", "latency_s", "rtf", "audio_duration_s"):
            if not isinstance(row.get(metric), int | float) or row[metric] <= 0:
                raise ValueError(f"case {case_id} has invalid {metric} in {path}")
        if not isinstance(row.get("audio_sha256"), str) or len(row["audio_sha256"]) != 64:
            raise ValueError(f"case {case_id} has invalid audio_sha256 in {path}")
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
        "overall": _summarize_pair(baseline_rows, candidate_rows),
        "cells": cells,
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
            "Positive memory savings mean that copying decoded chunks to CPU used less peak reserved GPU memory. "
            "A negative latency change means that the CPU-copy version was faster."
        ),
        "",
        (
            "| Word count | Cases | GPU-retained mean GiB | CPU-copy mean GiB | Mean saved GiB | Mean saved % | "
            "GPU-retained max GiB | CPU-copy max GiB | Candidate latency change % |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows = [*(summary["cells"]), {"bucket": "overall", **summary["overall"]}]
    for row in rows:
        peak = row["peak_reserved_gib"]
        latency = row["latency_s"]
        lines.append(
            f"| {row['bucket']} | {row['cases']} | {peak['baseline_mean']:.3f} | "
            f"{peak['candidate_mean']:.3f} | {peak['mean_saved']:.3f} | "
            f"{peak['mean_saved_percent']:.2f} | {peak['baseline_max']:.3f} | "
            f"{peak['candidate_max']:.3f} | {latency['candidate_change_percent']:.2f} |"
        )
    lines.extend(
        [
            "",
            (
                "Maximum paired audio-duration difference: "
                f"{summary['overall']['audio_duration_s']['max_absolute_difference']:.6f} seconds."
            ),
            (
                "Matching audio files: "
                f"{summary['overall']['audio_sha256']['matching_cases']}/{summary['overall']['cases']}."
            ),
            "",
        ]
    )
    return "\n".join(lines)


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
    print(_markdown(summary), end="")


if __name__ == "__main__":
    main()
