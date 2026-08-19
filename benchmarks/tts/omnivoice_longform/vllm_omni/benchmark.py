# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import asyncio
import io
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
import soundfile as sf
from tqdm.asyncio import tqdm

from benchmarks.tts.omnivoice_longform.common import (
    DEFAULT_SEEDS,
    GenerationCase,
    build_generation_cases,
    case_asdict,
    chunking_args,
    latency_summary,
    load_prompt_manifest,
    mean_and_stddev,
    read_jsonl,
    representative_warmup_cases,
    write_json,
    write_jsonl,
)


def _response_error(response: httpx.Response) -> RuntimeError:
    detail = response.text[:500]
    return RuntimeError(f"speech request failed with HTTP {response.status_code}: {detail}")


def _peak_reserved_gib(response: httpx.Response) -> float:
    value = response.headers.get("X-Peak-Memory-MB")
    if value is None:
        raise RuntimeError("speech response is missing the X-Peak-Memory-MB header")
    try:
        peak_memory_mb = float(value)
    except ValueError as error:
        raise RuntimeError(f"invalid X-Peak-Memory-MB header: {value!r}") from error
    if not math.isfinite(peak_memory_mb) or peak_memory_mb <= 0:
        raise RuntimeError(f"invalid X-Peak-Memory-MB header: {value!r}")
    return peak_memory_mb / 1024


def _peak_memory_summary(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    values = [
        row["peak_reserved_gib"]
        for row in rows
        if row.get("status", "success") == "success" and row.get("peak_reserved_gib") is not None
    ]
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


async def _generate_case(
    client: httpx.AsyncClient,
    api_url: str,
    model: str,
    case: GenerationCase,
    semaphore: asyncio.Semaphore,
    output_dir: Path | None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "input": case.text,
        "language": "English",
        "response_format": "wav",
        "seed": case.seed,
        "extra_params": chunking_args(case.mode),
    }
    async with semaphore:
        started = time.perf_counter()
        response = await client.post(api_url, json=payload)
        latency_s = time.perf_counter() - started

    if response.status_code != 200:
        raise _response_error(response)
    peak_reserved_gib = _peak_reserved_gib(response)

    try:
        audio_info = sf.info(io.BytesIO(response.content))
    except RuntimeError as error:
        raise RuntimeError(f"invalid WAV response for {case.case_id}") from error
    duration_s = audio_info.duration
    if duration_s <= 0:
        raise RuntimeError(f"empty WAV response for {case.case_id}")

    audio_path = None
    if output_dir is not None:
        audio_path = output_dir / f"{case.case_id}.wav"
        audio_path.write_bytes(response.content)

    return {
        **case_asdict(case),
        "backend": "vllm-omni",
        "audio_path": str(audio_path.resolve()) if audio_path else None,
        "sample_rate": audio_info.samplerate,
        "audio_duration_s": duration_s,
        "latency_s": latency_s,
        "rtf": latency_s / duration_s,
        "peak_reserved_gib": peak_reserved_gib,
    }


async def _generate_case_record(
    client: httpx.AsyncClient,
    api_url: str,
    model: str,
    case: GenerationCase,
    semaphore: asyncio.Semaphore,
    output_dir: Path | None,
    order_index: int,
) -> dict[str, Any]:
    try:
        row = await _generate_case(
            client,
            api_url,
            model,
            case,
            semaphore,
            output_dir,
        )
        return {**row, "status": "success", "order_index": order_index}
    except Exception as error:
        return {
            **case_asdict(case),
            "backend": "vllm-omni",
            "status": "error",
            "order_index": order_index,
            "audio_path": None,
            "sample_rate": None,
            "audio_duration_s": None,
            "latency_s": None,
            "rtf": None,
            "peak_reserved_gib": None,
            "error_type": type(error).__name__,
            "error": str(error),
        }


async def _run_cases(
    client: httpx.AsyncClient,
    api_url: str,
    model: str,
    cases: list[GenerationCase],
    concurrency: int,
    output_dir: Path | None,
    progress_description: str,
) -> tuple[list[dict[str, Any]], float]:
    semaphore = asyncio.Semaphore(concurrency)
    started = time.perf_counter()
    rows = await tqdm.gather(
        *(
            _generate_case_record(
                client,
                api_url,
                model,
                case,
                semaphore,
                output_dir,
                order_index,
            )
            for order_index, case in enumerate(cases)
        ),
        desc=progress_description,
        unit="request",
    )
    return rows, time.perf_counter() - started


def _summarize_rows(
    rows: list[dict[str, Any]],
    concurrency: int,
    wall_time_s: float,
) -> dict[str, Any]:
    successful_rows = [row for row in rows if row.get("status", "success") == "success"]
    failed_requests = len(rows) - len(successful_rows)
    summary: dict[str, Any] = {
        "concurrency": concurrency,
        "batch_size": 1,
        "request_rate": "inf",
        "measured_requests": len(rows),
        "successful_requests": len(successful_rows),
        "failed_requests": failed_requests,
        "failure_rate": failed_requests / len(rows) if rows else 0.0,
        "wall_time_s": wall_time_s,
        "attempted_requests_per_s": len(rows) / wall_time_s,
        "throughput_requests_per_s": len(successful_rows) / wall_time_s,
        "throughput_audio_s_per_s": sum(row["audio_duration_s"] for row in successful_rows) / wall_time_s,
        "latency_s": latency_summary([row["latency_s"] for row in successful_rows]),
        "rtf": latency_summary([row["rtf"] for row in successful_rows]),
        "peak_reserved_gib": _peak_memory_summary(successful_rows),
    }
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)
    summary["modes"] = {
        mode: {
            "samples": len(mode_rows),
            "failed_requests": sum(row.get("status", "success") != "success" for row in mode_rows),
            "latency_s": latency_summary(
                [row["latency_s"] for row in mode_rows if row.get("status", "success") == "success"]
            ),
            "rtf": (
                mean_and_stddev([row["rtf"] for row in mode_rows if row.get("status", "success") == "success"])
                if any(row.get("status", "success") == "success" for row in mode_rows)
                else None
            ),
            "peak_reserved_gib": _peak_memory_summary(mode_rows),
        }
        for mode, mode_rows in by_mode.items()
    }
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[(row["mode"], row["bucket"])].append(row)
    summary["cells"] = [
        {
            "mode": mode,
            "bucket": bucket,
            "samples": len(cell_rows),
            "failed_requests": sum(row.get("status", "success") != "success" for row in cell_rows),
            "word_count": {
                "min": min(row["word_count"] for row in cell_rows),
                "max": max(row["word_count"] for row in cell_rows),
                "mean": statistics.fmean(row["word_count"] for row in cell_rows),
            },
            "latency_s": latency_summary(
                [row["latency_s"] for row in cell_rows if row.get("status", "success") == "success"]
            ),
            "rtf": (
                mean_and_stddev([row["rtf"] for row in cell_rows if row.get("status", "success") == "success"])
                if any(row.get("status", "success") == "success" for row in cell_rows)
                else None
            ),
            "peak_reserved_gib": _peak_memory_summary(cell_rows),
        }
        for (mode, bucket), cell_rows in sorted(by_cell.items())
    ]
    return summary


def _print_sweep_summary(summary: dict[str, Any]) -> None:
    print(
        f"concurrency {summary['concurrency']}: "
        f"{summary['successful_requests']}/{summary['measured_requests']} requests succeeded "
        f"({summary['failed_requests']} failed) in {summary['wall_time_s']:.2f}s"
    )
    print(
        f"  request throughput: {summary['throughput_requests_per_s']:.3f} requests/s, "
        f"audio throughput: {summary['throughput_audio_s_per_s']:.3f} audio-s/s"
    )
    if summary["latency_s"] is not None:
        print(f"  median latency: {summary['latency_s']['p50']:.2f}s, median RTF: {summary['rtf']['p50']:.3f}")
    if summary["peak_reserved_gib"] is not None:
        print(f"  peak reserved GPU memory: {summary['peak_reserved_gib']['max']:.2f} GiB")


async def run(args: argparse.Namespace) -> None:
    if any(value <= 0 for value in args.concurrencies):
        raise ValueError("concurrency values must be positive")
    if len(args.concurrencies) != len(set(args.concurrencies)):
        raise ValueError("concurrency values must be unique")
    if 1 not in args.concurrencies:
        raise ValueError("concurrency 1 is required for saved quality outputs")

    _, prompts = load_prompt_manifest(args.manifest)
    cases = build_generation_cases(prompts, args.seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    serving_path = output_dir / "serving.jsonl"
    generation_path = output_dir / "generation.jsonl"
    summary_path = output_dir / "serving_summary.json"
    api_url = f"{args.api_base.rstrip('/')}/v1/audio/speech"
    headers = {"Authorization": f"Bearer {args.api_key}"}
    timeout = httpx.Timeout(args.timeout)

    existing_rows = read_jsonl(serving_path) if serving_path.exists() else []
    expected_case_ids = {case.case_id for case in cases}
    requested_concurrencies = set(args.concurrencies)
    rows_by_concurrency: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in existing_rows:
        concurrency = row["concurrency"]
        if concurrency not in requested_concurrencies:
            raise ValueError(f"unexpected concurrency {concurrency} in checkpoint: {serving_path}")
        if row["case_id"] not in expected_case_ids:
            raise ValueError(f"unexpected case {row['case_id']} in checkpoint: {serving_path}")
        if row["case_id"] in rows_by_concurrency[concurrency]:
            raise ValueError(f"duplicate case {row['case_id']} at concurrency {concurrency}")
        rows_by_concurrency[concurrency][row["case_id"]] = row

    def ordered_rows(concurrency: int) -> list[dict[str, Any]]:
        checkpoint = rows_by_concurrency[concurrency]
        return [checkpoint[case.case_id] for case in cases if case.case_id in checkpoint]

    def write_checkpoints() -> None:
        serving_rows = [row for concurrency in args.concurrencies for row in ordered_rows(concurrency)]
        write_jsonl(serving_path, serving_rows)
        if len(rows_by_concurrency[1]) == len(cases):
            write_jsonl(generation_path, ordered_rows(1))

        summaries = []
        for concurrency in args.concurrencies:
            rows = ordered_rows(concurrency)
            if len(rows) != len(cases):
                continue
            wall_times = {row["sweep_wall_time_s"] for row in rows}
            if len(wall_times) != 1:
                raise ValueError(f"inconsistent wall times for concurrency {concurrency}")
            summary = _summarize_rows(rows, concurrency, wall_times.pop())
            summary["warmup_requests"] = len(representative_warmup_cases(cases, concurrency))
            summaries.append(summary)
        write_json(
            summary_path,
            {
                "backend": "vllm-omni",
                "model": args.model,
                "seeds": args.seeds,
                "case_order": [case.case_id for case in cases],
                "concurrency_results": summaries,
            },
        )

    async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
        for concurrency in args.concurrencies:
            checkpoint = rows_by_concurrency[concurrency]
            if checkpoint and checkpoint.keys() != expected_case_ids:
                raise ValueError(f"incomplete concurrency {concurrency} checkpoint in {serving_path}")
            if checkpoint:
                print(f"concurrency {concurrency}: restored {len(checkpoint)} requests from checkpoint")
                write_checkpoints()
                continue

            warmup_cases = representative_warmup_cases(cases, concurrency)
            warmup_rows, _ = await _run_cases(
                client,
                api_url,
                args.model,
                warmup_cases,
                concurrency,
                output_dir=None,
                progress_description=f"vLLM concurrency {concurrency} warmup",
            )
            warmup_failures = sum(row["status"] != "success" for row in warmup_rows)
            if warmup_failures:
                raise RuntimeError(
                    f"concurrency {concurrency}: {warmup_failures}/{len(warmup_rows)} warmup requests failed"
                )
            save_dir = output_dir if concurrency == 1 else None
            rows, wall_time_s = await _run_cases(
                client,
                api_url,
                args.model,
                cases,
                concurrency,
                output_dir=save_dir,
                progress_description=f"vLLM concurrency {concurrency}",
            )
            rows = [
                {
                    **row,
                    "concurrency": concurrency,
                    "sweep_wall_time_s": wall_time_s,
                }
                for row in rows
            ]
            rows_by_concurrency[concurrency] = {row["case_id"]: row for row in rows}
            write_checkpoints()
            _print_sweep_summary(_summarize_rows(rows, concurrency, wall_time_s))

    write_checkpoints()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark vLLM-Omni OmniVoice serving")
    parser.add_argument("--api-base", default="http://127.0.0.1:8091")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--timeout", type=float, default=1200.0)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
