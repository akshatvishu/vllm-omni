# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MODES = ("one_shot", "chunked")
DEFAULT_SEEDS = (42,)

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


@dataclass(frozen=True)
class PromptCase:
    prompt_id: str
    bucket: str
    source_id: str
    category: str
    word_count: int
    text: str
    text_sha256: str


@dataclass(frozen=True)
class GenerationCase:
    prompt_id: str
    bucket: str
    source_id: str
    category: str
    word_count: int
    text: str
    mode: str
    seed: int

    @property
    def case_id(self) -> str:
        return f"{self.prompt_id}_{self.mode}_seed{self.seed}"


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return " ".join(_WORD_RE.findall(text))


def word_count(text: str) -> int:
    normalized = normalize_text(text)
    return len(normalized.split()) if normalized else 0


def load_prompt_manifest(path: str | Path) -> tuple[dict[str, Any], list[PromptCase]]:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompts = [PromptCase(**item) for item in data["prompts"]]

    prompt_ids = [prompt.prompt_id for prompt in prompts]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("prompt_id values must be unique")

    source = data["source"]
    expected_per_bucket = source["source_examples"]
    expected_distribution = {bucket["name"]: expected_per_bucket for bucket in source["buckets"]}
    actual_distribution = Counter(prompt.bucket for prompt in prompts)
    if actual_distribution != expected_distribution:
        raise ValueError(f"prompt bucket distribution must be {expected_distribution}, got {dict(actual_distribution)}")

    for prompt in prompts:
        actual = word_count(prompt.text)
        if actual != prompt.word_count:
            raise ValueError(f"{prompt.prompt_id} declares {prompt.word_count} words but contains {actual}")
        actual_sha256 = hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
        if actual_sha256 != prompt.text_sha256:
            raise ValueError(f"{prompt.prompt_id} text does not match its SHA256")

    return source, prompts


def build_generation_cases(
    prompts: list[PromptCase],
    seeds: list[int] | tuple[int, ...],
) -> list[GenerationCase]:
    if not seeds:
        raise ValueError("at least one seed is required")
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative")

    return [
        GenerationCase(
            prompt_id=prompt.prompt_id,
            bucket=prompt.bucket,
            source_id=prompt.source_id,
            category=prompt.category,
            word_count=prompt.word_count,
            text=prompt.text,
            mode=mode,
            seed=seed,
        )
        for prompt in prompts
        for seed in seeds
        for mode in MODES
    ]


def chunking_args(mode: str) -> dict[str, float]:
    if mode == "one_shot":
        return {
            "audio_chunk_duration": 15.0,
            "audio_chunk_threshold": 1_000_000.0,
        }
    if mode == "chunked":
        return {
            "audio_chunk_duration": 15.0,
            "audio_chunk_threshold": 0.0,
        }
    raise ValueError(f"unsupported mode: {mode}")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    temporary_path.replace(output_path)


def write_json(path: str | Path, value: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary_path.replace(output_path)


def write_immutable_json(path: str | Path, value: Any) -> None:
    output_path = Path(path)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != value:
            raise ValueError(f"refusing to replace {output_path} with different content")
        return
    write_json(output_path, value)


def write_text(path: str | Path, value: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        output.write(value)
        output.flush()
        os.fsync(output.fileno())
    temporary_path.replace(output_path)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def case_asdict(case: GenerationCase) -> dict[str, Any]:
    return {"case_id": case.case_id, **asdict(case)}


def mean_and_stddev(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty value list")
    return {
        "mean": statistics.fmean(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty value list")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")

    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index
    return ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight


def latency_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        **mean_and_stddev(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
    }
