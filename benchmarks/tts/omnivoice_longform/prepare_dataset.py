# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import tomllib
from huggingface_hub import hf_hub_download

from benchmarks.tts.omnivoice_longform.common import word_count, write_immutable_json

_SENTENCE_END_RE = re.compile(r"[.!?](?:[\"'”’]+)?(?=\s|$)")
_INITIALISM_RE = re.compile(r"(?:[a-z]\.){2,}$", re.IGNORECASE)
_ABBREVIATIONS = frozenset(
    {
        "dr.",
        "jr.",
        "mr.",
        "mrs.",
        "ms.",
        "prof.",
        "sr.",
        "st.",
        "vs.",
    }
)


def _is_abbreviation(text: str, period_index: int) -> bool:
    word_start = period_index
    while word_start > 0 and not text[word_start - 1].isspace():
        word_start -= 1
    token = text[word_start : period_index + 1].lower().lstrip("\"'“‘(")
    return token in _ABBREVIATIONS or _INITIALISM_RE.fullmatch(token) is not None


def _prefix_for_bucket(text: str, bucket: dict[str, Any]) -> tuple[str, int] | None:
    normalized_text = " ".join(text.split())
    candidates = []
    for match in _SENTENCE_END_RE.finditer(normalized_text):
        if normalized_text[match.start()] == "." and _is_abbreviation(normalized_text, match.start()):
            continue
        prefix = normalized_text[: match.end()]
        count = word_count(prefix)
        if bucket["min_words"] <= count <= bucket["max_words"]:
            candidates.append((abs(count - bucket["target_words"]), count, prefix))
    if not candidates:
        return None
    _, count, prefix = min(candidates, key=lambda item: (item[0], item[1]))
    return prefix, count


def select_prompts(rows: list[dict[str, Any]], selection: dict[str, Any]) -> list[dict[str, Any]]:
    eligible: dict[str, list[tuple[dict[str, Any], list[tuple[dict[str, Any], str, int]]]]] = defaultdict(list)
    for row in rows:
        bucket_prompts = []
        for bucket in selection["buckets"]:
            selected = _prefix_for_bucket(row[selection["text_field"]], bucket)
            if selected is None:
                break
            text, count = selected
            bucket_prompts.append((bucket, text, count))
        else:
            eligible[row["category"]].append((row, bucket_prompts))

    seed = selection["selection_seed"]
    for category_rows in eligible.values():
        category_rows.sort(key=lambda item: hashlib.sha256(f"{seed}:{item[0]['id']}".encode()).hexdigest())

    selected_rows = []
    categories = sorted(eligible)
    while len(selected_rows) < selection["source_examples"]:
        made_progress = False
        for category in categories:
            if eligible[category]:
                selected_rows.append(eligible[category].pop(0))
                made_progress = True
                if len(selected_rows) == selection["source_examples"]:
                    break
        if not made_progress:
            raise ValueError(
                f"dataset provides only {len(selected_rows)} eligible source rows; need {selection['source_examples']}"
            )

    prompts = []
    for row, bucket_prompts in selected_rows:
        for bucket, text, count in bucket_prompts:
            prompts.append(
                {
                    "prompt_id": f"{row['id']}_{bucket['name']}",
                    "bucket": bucket["name"],
                    "source_id": row["id"],
                    "category": row["category"],
                    "word_count": count,
                    "text": text,
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
    return prompts


def prepare_manifest(selection_path: Path, output_path: Path) -> None:
    selection = tomllib.loads(selection_path.read_text(encoding="utf-8"))
    dataset_path = hf_hub_download(
        repo_id=selection["dataset"],
        repo_type="dataset",
        filename=selection["filename"],
        revision=selection["revision"],
    )
    with Path(dataset_path).open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]

    prompts = select_prompts(rows, selection)
    manifest = {
        "source": {
            **selection,
            "resolved_examples": len(prompts),
            "selected_source_ids": sorted({prompt["source_id"] for prompt in prompts}),
        },
        "prompts": prompts,
    }
    write_immutable_json(output_path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the pinned Long-TTS-Eval prompt manifest")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_manifest(args.selection, args.output)
