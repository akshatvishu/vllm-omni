# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import io
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import soundfile as sf
import tomllib
import torch

from benchmarks.tts.omnivoice_longform import evaluate as evaluate_module
from benchmarks.tts.omnivoice_longform.common import (
    DEFAULT_SEEDS,
    GenerationCase,
    build_generation_cases,
    chunking_args,
    load_prompt_manifest,
    read_jsonl,
    representative_warmup_cases,
    write_immutable_json,
    write_jsonl,
)
from benchmarks.tts.omnivoice_longform.evaluate import (
    _aggregate,
    _validate_backend_cases,
    load_audio_16k,
    score_transcript,
    transcribe_waveform,
)
from benchmarks.tts.omnivoice_longform.metadata import _git_state
from benchmarks.tts.omnivoice_longform.prepare_dataset import select_prompts
from benchmarks.tts.omnivoice_longform.vllm_omni import benchmark as benchmark_module
from benchmarks.tts.omnivoice_longform.vllm_omni.benchmark import (
    _generate_case,
    _generate_case_record,
    _print_sweep_summary,
    _summarize_rows,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SELECTION = Path(__file__).parents[2] / "benchmarks" / "tts" / "omnivoice_longform" / "selection.toml"
RUNNER = Path(__file__).parents[2] / "benchmarks" / "tts" / "omnivoice_longform" / "run_benchmark.sh"


def _source_rows(count: int = 30) -> list[dict]:
    categories = ("news", "paper", "poet", "speech", "talk", "wiki")
    sentence = " ".join(["word"] * 10) + "."
    text = " ".join([sentence] * 65)
    return [
        {
            "id": f"source-{index}",
            "category": categories[index % len(categories)],
            "text_norm": text,
        }
        for index in range(count)
    ]


def _resolved_manifest(tmp_path: Path) -> Path:
    selection = tomllib.loads(SELECTION.read_text(encoding="utf-8"))
    prompts = select_prompts(_source_rows(), selection)
    manifest = tmp_path / "prompts.json"
    manifest.write_text(
        json.dumps(
            {
                "source": {
                    **selection,
                    "resolved_examples": len(prompts),
                    "selected_source_ids": sorted({prompt["source_id"] for prompt in prompts}),
                },
                "prompts": prompts,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _small_manifest(tmp_path: Path) -> Path:
    selection = tomllib.loads(SELECTION.read_text(encoding="utf-8"))
    selection["source_examples"] = 1
    prompts = select_prompts(_source_rows(count=1), selection)
    manifest = tmp_path / "small-prompts.json"
    manifest.write_text(
        json.dumps(
            {
                "source": {
                    **selection,
                    "resolved_examples": len(prompts),
                    "selected_source_ids": [prompts[0]["source_id"]],
                },
                "prompts": prompts,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_dataset_selection_produces_balanced_100_prompt_distribution() -> None:
    selection = tomllib.loads(SELECTION.read_text(encoding="utf-8"))
    prompts = select_prompts(_source_rows(), selection)

    assert len(prompts) == 100
    assert Counter(prompt["bucket"] for prompt in prompts) == {
        "words_120": 25,
        "words_200": 25,
        "words_300": 25,
        "words_400_plus": 25,
    }
    assert Counter(prompt["word_count"] for prompt in prompts) == {120: 25, 200: 25, 300: 25, 500: 25}

    buckets_by_source = defaultdict(set)
    category_by_source = {}
    for prompt in prompts:
        buckets_by_source[prompt["source_id"]].add(prompt["bucket"])
        category_by_source[prompt["source_id"]] = prompt["category"]
    assert len(buckets_by_source) == 25
    assert all(len(buckets) == 4 for buckets in buckets_by_source.values())
    category_counts = Counter(category_by_source.values())
    assert max(category_counts.values()) - min(category_counts.values()) <= 1


def test_small_dataset_selection_produces_ten_prompts_per_bucket() -> None:
    selection = tomllib.loads(SELECTION.read_text(encoding="utf-8"))
    selection["source_examples"] = 10

    prompts = select_prompts(_source_rows(), selection)

    assert len(prompts) == 40
    assert Counter(prompt["bucket"] for prompt in prompts) == {
        "words_120": 10,
        "words_200": 10,
        "words_300": 10,
        "words_400_plus": 10,
    }


def test_runner_defaults_and_pinned_packages() -> None:
    runner = RUNNER.read_text(encoding="utf-8")

    assert '"omnivoice==0.2.1"' in runner
    assert '"jiwer==4.0.0"' in runner
    assert '"pydub==0.25.1"' in runner
    assert "REFERENCE_REPO" not in runner
    assert 'OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/' in runner
    assert 'OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"' in runner
    assert "--small)" in runner
    assert "PREPARE_DATASET_ARGS=(--source-examples 10)" in runner
    assert "CONCURRENCY_VALUES=(1)" in runner
    assert 'CONCURRENCY_VALUES+=("$2")' in runner
    assert "CONCURRENCIES" not in runner
    assert 'echo "Running reference OmniVoice benchmark"' in runner
    assert 'echo "Starting vLLM-Omni server"' in runner
    assert 'echo "Running vLLM-Omni serving benchmark"' in runner
    assert 'echo "Running Whisper transcription and scoring"' in runner


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        (["--concurrency"], "requires a positive integer"),
        (["--concurrency", "0"], "requires a positive integer"),
        (["--concurrency", "1"], "Duplicate concurrency: 1"),
        (["--unknown"], "Usage:"),
    ],
)
def test_runner_rejects_invalid_arguments(arguments: list[str], error: str) -> None:
    result = subprocess.run(
        ["bash", str(RUNNER), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert error in result.stderr


def test_dataset_selection_rejects_nonpositive_source_examples() -> None:
    selection = tomllib.loads(SELECTION.read_text(encoding="utf-8"))
    selection["source_examples"] = 0

    with pytest.raises(ValueError, match="source_examples must be positive"):
        select_prompts(_source_rows(), selection)


def test_dataset_selection_rejects_too_few_eligible_sources() -> None:
    selection = tomllib.loads(SELECTION.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="need 25"):
        select_prompts(_source_rows(count=24), selection)


def test_manifest_validates_distribution_counts_and_hashes(tmp_path: Path) -> None:
    source, prompts = load_prompt_manifest(_resolved_manifest(tmp_path))

    assert source["dataset"] == "wcy1122/Long-TTS-Eval"
    assert source["split"] == "long_tts_eval_en"
    assert len(source["revision"]) == 40
    assert len(prompts) == 100


def test_generation_matrix_covers_all_modes_prompts_and_seed(tmp_path: Path) -> None:
    _, prompts = load_prompt_manifest(_resolved_manifest(tmp_path))
    cases = build_generation_cases(prompts, DEFAULT_SEEDS)

    assert len(cases) == 200
    assert len({case.case_id for case in cases}) == len(cases)
    assert Counter(case.mode for case in cases) == {"one_shot": 100, "chunked": 100}
    assert {case.seed for case in cases} == {42}
    assert cases[0].prompt_id == cases[1].prompt_id
    assert [case.mode for case in cases[:2]] == ["one_shot", "chunked"]


def test_manifest_rejects_missing_bucket_example(tmp_path: Path) -> None:
    manifest_path = _resolved_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prompts"].pop()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="bucket distribution"):
        load_prompt_manifest(manifest_path)


def test_manifest_rejects_changed_prompt_text(tmp_path: Path) -> None:
    manifest_path = _resolved_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prompts"][0]["text"] = manifest["prompts"][0]["text"].replace("word", "term", 1)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA256"):
        load_prompt_manifest(manifest_path)


def test_chunking_modes_force_the_intended_paths() -> None:
    assert chunking_args("one_shot") == {
        "audio_chunk_duration": 15.0,
        "audio_chunk_threshold": 1_000_000.0,
    }
    assert chunking_args("chunked") == {
        "audio_chunk_duration": 15.0,
        "audio_chunk_threshold": 0.0,
    }
    with pytest.raises(ValueError, match="unsupported mode"):
        chunking_args("automatic")


def test_coverage_uses_sequence_alignment_for_repeated_words() -> None:
    score = score_transcript("one two one", "one one")

    assert score["hits"] == 2
    assert score["deletions"] == 1
    assert score["coverage"] == pytest.approx(2 / 3)
    assert score["wer"] == pytest.approx(1 / 3)


def test_insertions_increase_wer_without_reducing_coverage() -> None:
    score = score_transcript("one two", "one extra two")

    assert score["insertions"] == 1
    assert score["coverage"] == 1.0
    assert score["wer"] == 0.5


def test_backend_validation_rejects_mismatched_cases() -> None:
    rows = [
        {"backend": "reference", "case_id": "case-a"},
        {"backend": "vllm-omni", "case_id": "case-b"},
    ]

    with pytest.raises(ValueError, match="do not contain the same cases"):
        _validate_backend_cases(rows)


def test_load_audio_16k_downmixes_and_resamples(tmp_path: Path) -> None:
    sample_rate = 8000
    duration_s = 0.1
    samples = int(sample_rate * duration_s)
    stereo = np.stack(
        [
            np.linspace(-0.5, 0.5, samples, dtype=np.float32),
            np.linspace(0.5, -0.5, samples, dtype=np.float32),
        ],
        axis=1,
    )
    path = tmp_path / "stereo.wav"
    sf.write(path, stereo, sample_rate)

    mono = load_audio_16k(path)

    assert mono.dtype == np.float32
    assert mono.flags.c_contiguous
    assert mono.shape == (1600,)
    assert np.isfinite(mono).all()
    assert np.max(np.abs(mono)) < 1e-4


def test_serving_summary_keeps_each_mode_and_length_bucket() -> None:
    rows = [
        {
            "mode": mode,
            "bucket": bucket,
            "word_count": count,
            "status": "success",
            "audio_duration_s": float(count) / 2,
            "latency_s": float(count),
            "rtf": 1.0,
            "peak_reserved_gib": float(count) / 100,
        }
        for mode in ("one_shot", "chunked")
        for bucket, count in (
            ("words_120", 120),
            ("words_200", 200),
            ("words_300", 300),
            ("words_400_plus", 500),
        )
    ]

    summary = _summarize_rows(rows, concurrency=4, wall_time_s=10.0)

    assert summary["batch_size"] == 1
    assert summary["latency_s"]["p50"] == pytest.approx(250.0)
    assert summary["latency_s"]["p90"] > summary["latency_s"]["p50"]
    assert summary["latency_s"]["p99"] >= summary["latency_s"]["p90"]
    assert summary["throughput_audio_s_per_s"] == pytest.approx(112.0)
    assert summary["rtf"]["p50"] == 1.0
    assert summary["peak_reserved_gib"]["max"] == 5.0
    assert {(cell["mode"], cell["bucket"]) for cell in summary["cells"]} == {
        (mode, bucket)
        for mode in ("one_shot", "chunked")
        for bucket in ("words_120", "words_200", "words_300", "words_400_plus")
    }


def test_serving_summary_counts_failures_without_summarizing_missing_metrics() -> None:
    rows = [
        {
            "mode": "chunked",
            "bucket": "words_400_plus",
            "word_count": 500,
            "status": "error",
            "audio_duration_s": None,
            "latency_s": None,
            "rtf": None,
        }
    ]

    summary = _summarize_rows(rows, concurrency=1, wall_time_s=2.0)

    assert summary["failed_requests"] == 1
    assert summary["failure_rate"] == 1.0
    assert summary["attempted_requests_per_s"] == 0.5
    assert summary["throughput_requests_per_s"] == 0.0
    assert summary["throughput_audio_s_per_s"] == 0.0
    assert summary["latency_s"] is None
    assert summary["rtf"] is None
    assert summary["cells"][0]["latency_s"] is None


def test_sweep_summary_reports_successful_throughput(capsys) -> None:
    rows = [
        {
            "mode": "chunked",
            "bucket": "words_120",
            "word_count": 120,
            "status": "success",
            "audio_duration_s": 10.0,
            "latency_s": 2.0,
            "rtf": 0.2,
            "peak_reserved_gib": 4.0,
        },
        {
            "mode": "chunked",
            "bucket": "words_120",
            "word_count": 120,
            "status": "error",
            "audio_duration_s": None,
            "latency_s": None,
            "rtf": None,
        },
    ]

    _print_sweep_summary(_summarize_rows(rows, concurrency=1, wall_time_s=2.0))

    output = capsys.readouterr().out
    assert "1/2 requests succeeded (1 failed)" in output
    assert "request throughput: 0.500 requests/s" in output
    assert "median latency: 2.00s, median RTF: 0.200" in output
    assert "peak reserved GPU memory: 4.00 GiB" in output


def test_warmups_cover_every_mode_and_bucket_and_fill_concurrency(tmp_path: Path) -> None:
    _, prompts = load_prompt_manifest(_resolved_manifest(tmp_path))
    cases = build_generation_cases(prompts, DEFAULT_SEEDS)

    warmups = representative_warmup_cases(cases, concurrency=12)

    assert len(warmups) == 12
    assert {(case.mode, case.bucket) for case in warmups} == {
        (mode, bucket)
        for mode in ("one_shot", "chunked")
        for bucket in ("words_120", "words_200", "words_300", "words_400_plus")
    }


def test_warmups_reject_empty_case_list() -> None:
    with pytest.raises(ValueError, match="cannot warm up an empty case list"):
        representative_warmup_cases([], concurrency=1)


def test_atomic_jsonl_checkpoint_replaces_complete_file(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    write_jsonl(path, [{"case_id": "old"}])
    write_jsonl(path, [{"case_id": "new-1"}, {"case_id": "new-2"}])

    assert read_jsonl(path) == [{"case_id": "new-1"}, {"case_id": "new-2"}]
    assert not (tmp_path / ".checkpoint.jsonl.tmp").exists()


@pytest.mark.parametrize(
    "changed",
    [
        {"manifest_sha256": "new", "model_revision": "model-a", "whisper_revision": "whisper-a"},
        {"manifest_sha256": "old", "model_revision": "model-b", "whisper_revision": "whisper-a"},
        {"manifest_sha256": "old", "model_revision": "model-a", "whisper_revision": "whisper-b"},
    ],
)
def test_immutable_run_configuration_rejects_resume_mismatch(tmp_path: Path, changed: dict) -> None:
    path = tmp_path / "run_metadata.json"
    original = {
        "manifest_sha256": "old",
        "model_revision": "model-a",
        "whisper_revision": "whisper-a",
    }
    write_immutable_json(path, original)
    write_immutable_json(path, original)

    with pytest.raises(ValueError, match="refusing to replace"):
        write_immutable_json(path, changed)

    assert json.loads(path.read_text(encoding="utf-8")) == original


def test_git_state_fingerprint_distinguishes_dirty_worktrees(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "benchmark@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Benchmark Test"], check=True)
    tracked = repo / "tracked.txt"
    tracked.write_text("committed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "initial"], check=True)

    clean = _git_state(repo)
    tracked.write_text("first dirty state\n", encoding="utf-8")
    tracked_dirty = _git_state(repo)
    tracked.write_text("second dirty state\n", encoding="utf-8")
    changed_tracked_dirty = _git_state(repo)
    untracked = repo / "untracked.txt"
    untracked.write_text("first untracked state\n", encoding="utf-8")
    untracked_dirty = _git_state(repo)
    untracked.write_text("second untracked state\n", encoding="utf-8")
    changed_untracked_dirty = _git_state(repo)

    assert clean["dirty"] is False
    assert tracked_dirty["dirty"] is True
    assert tracked_dirty["worktree_sha256"] != changed_tracked_dirty["worktree_sha256"]
    assert untracked_dirty["worktree_sha256"] != changed_untracked_dirty["worktree_sha256"]


def test_evaluation_summary_preserves_generation_and_transcription_failures() -> None:
    common = {
        "backend": "reference",
        "mode": "one_shot",
        "bucket": "words_120",
        "word_count": 120,
    }
    rows = [
        {
            **common,
            "status": "success",
            "evaluation_status": "success",
            "coverage": 0.9,
            "wer": 0.1,
            "latency_s": 2.0,
            "rtf": 0.2,
            "peak_reserved_gib": 3.0,
        },
        {
            **common,
            "status": "error",
            "evaluation_status": "generation_error",
        },
        {
            **common,
            "status": "success",
            "evaluation_status": "transcription_error",
            "latency_s": 3.0,
            "rtf": 0.3,
            "peak_reserved_gib": 5.0,
        },
    ]

    summary = _aggregate(rows)[0]

    assert summary["samples"] == 3
    assert summary["scored_samples"] == 1
    assert summary["generation_failures"] == 1
    assert summary["transcription_failures"] == 1
    assert summary["failure_rate"] == pytest.approx(2 / 3)
    assert summary["peak_reserved_gib"]["mean"] == 4.0


def test_whisper_processor_does_not_truncate_long_audio() -> None:
    class FakeProcessor:
        def __init__(self) -> None:
            self.call_kwargs = None

        def __call__(self, waveform, **kwargs):
            self.call_kwargs = kwargs
            assert waveform.shape == (31 * 16000,)
            return SimpleNamespace(
                input_features=torch.zeros(1, 80, 3100),
                attention_mask=torch.ones(1, 3100, dtype=torch.long),
            )

        def batch_decode(self, predicted_ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            assert predicted_ids.tolist() == [[1, 2, 3]]
            return ["complete transcript"]

    class FakeModel:
        def __init__(self) -> None:
            self.generate_kwargs = None

        def generate(self, input_features, **kwargs):
            assert input_features.shape == (1, 80, 3100)
            self.generate_kwargs = kwargs
            return torch.tensor([[1, 2, 3]])

    processor = FakeProcessor()
    model = FakeModel()
    transcript = transcribe_waveform(
        np.zeros(31 * 16000, dtype=np.float32),
        processor,
        model,
        device="cpu",
        dtype=torch.float32,
    )

    assert transcript == "complete transcript"
    assert processor.call_kwargs["truncation"] is False
    assert processor.call_kwargs["padding"] == "longest"
    assert processor.call_kwargs["return_attention_mask"] is True
    assert model.generate_kwargs["return_timestamps"] is True
    assert model.generate_kwargs["language"] == "english"


def _generation_case() -> GenerationCase:
    return GenerationCase(
        prompt_id="source-1_words_120",
        bucket="words_120",
        source_id="source-1",
        category="news",
        word_count=120,
        text="one two",
        mode="chunked",
        seed=42,
    )


@pytest.mark.asyncio
async def test_vllm_request_sends_seed_and_chunk_settings(tmp_path: Path) -> None:
    audio = io.BytesIO()
    sf.write(audio, np.zeros(2400, dtype=np.float32), 24000, format="WAV")

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["seed"] == 42
        assert payload["extra_params"] == chunking_args("chunked")
        assert payload["language"] == "English"
        return httpx.Response(
            200,
            content=audio.getvalue(),
            headers={"X-Peak-Memory-MB": "2048"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        row = await _generate_case(
            client,
            "http://test/v1/audio/speech",
            "k2-fsa/OmniVoice",
            _generation_case(),
            asyncio.Semaphore(1),
            tmp_path,
        )

    assert row["backend"] == "vllm-omni"
    assert row["audio_duration_s"] == pytest.approx(0.1)
    assert row["peak_reserved_gib"] == 2.0
    assert Path(row["audio_path"]).is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize("header", [None, "invalid", "0", "nan"])
async def test_vllm_request_rejects_missing_or_invalid_peak_memory(header: str | None) -> None:
    audio = io.BytesIO()
    sf.write(audio, np.zeros(2400, dtype=np.float32), 24000, format="WAV")

    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"X-Peak-Memory-MB": header} if header is not None else None
        return httpx.Response(200, content=audio.getvalue(), headers=headers)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match="X-Peak-Memory-MB"):
            await _generate_case(
                client,
                "http://test/v1/audio/speech",
                "k2-fsa/OmniVoice",
                _generation_case(),
                asyncio.Semaphore(1),
                output_dir=None,
            )


@pytest.mark.asyncio
async def test_vllm_request_surfaces_server_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid chunk duration"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match="HTTP 400"):
            await _generate_case(
                client,
                "http://test/v1/audio/speech",
                "k2-fsa/OmniVoice",
                _generation_case(),
                asyncio.Semaphore(1),
                output_dir=None,
            )


@pytest.mark.asyncio
async def test_vllm_sweep_records_server_error_without_raising() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "server overloaded"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        row = await _generate_case_record(
            client,
            "http://test/v1/audio/speech",
            "k2-fsa/OmniVoice",
            _generation_case(),
            asyncio.Semaphore(1),
            output_dir=None,
            order_index=7,
        )

    assert row["status"] == "error"
    assert row["order_index"] == 7
    assert row["error_type"] == "RuntimeError"
    assert "HTTP 503" in row["error"]


@pytest.mark.asyncio
async def test_vllm_benchmark_checkpoints_each_sweep_and_resumes(tmp_path: Path, monkeypatch) -> None:
    manifest = _small_manifest(tmp_path)
    output_dir = tmp_path / "output"
    sweep_calls = []

    async def fake_run_cases(client, api_url, model, cases, concurrency, output_dir, progress_description):
        sweep_calls.append((concurrency, progress_description, len(cases)))
        rows = [
            {
                "case_id": case.case_id,
                "prompt_id": case.prompt_id,
                "bucket": case.bucket,
                "source_id": case.source_id,
                "category": case.category,
                "word_count": case.word_count,
                "text": case.text,
                "mode": case.mode,
                "seed": case.seed,
                "backend": "vllm-omni",
                "status": "success",
                "order_index": index,
                "audio_path": str(tmp_path / f"{case.case_id}.wav") if output_dir else None,
                "sample_rate": 24000,
                "audio_duration_s": 10.0,
                "latency_s": 1.0,
                "rtf": 0.1,
            }
            for index, case in enumerate(cases)
        ]
        return rows, 4.0

    monkeypatch.setattr(benchmark_module, "_run_cases", fake_run_cases)
    args = SimpleNamespace(
        api_base="http://test",
        api_key="EMPTY",
        model="k2-fsa/OmniVoice",
        manifest=manifest,
        output_dir=output_dir,
        seeds=[42],
        concurrencies=[1, 2],
        timeout=1.0,
    )

    await benchmark_module.run(args)

    assert sweep_calls == [
        (1, "vLLM concurrency 1 warmup", 8),
        (1, "vLLM concurrency 1", 8),
        (2, "vLLM concurrency 2 warmup", 8),
        (2, "vLLM concurrency 2", 8),
    ]
    assert len(read_jsonl(output_dir / "generation.jsonl")) == 8
    assert len(read_jsonl(output_dir / "serving.jsonl")) == 16

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("completed sweeps must be restored without issuing requests")

    monkeypatch.setattr(benchmark_module, "_run_cases", fail_if_called)
    await benchmark_module.run(args)


@pytest.mark.asyncio
async def test_vllm_benchmark_stops_after_failed_warmup(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "output"
    calls = []

    async def fake_run_cases(client, api_url, model, cases, concurrency, output_dir, progress_description):
        calls.append(progress_description)
        return [{"status": "error"}], 1.0

    monkeypatch.setattr(benchmark_module, "_run_cases", fake_run_cases)
    args = SimpleNamespace(
        api_base="http://test",
        api_key="EMPTY",
        model="k2-fsa/OmniVoice",
        manifest=_small_manifest(tmp_path),
        output_dir=output_dir,
        seeds=[42],
        concurrencies=[1],
        timeout=1.0,
    )

    with pytest.raises(RuntimeError, match="1/1 warmup requests failed"):
        await benchmark_module.run(args)

    assert calls == ["vLLM concurrency 1 warmup"]
    assert not (output_dir / "serving.jsonl").exists()
    assert not (output_dir / "serving_summary.json").exists()


def test_evaluator_checkpoints_failures_and_resumes_without_reloading_whisper(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    common = {
        "case_id": "case-1",
        "prompt_id": "prompt-1",
        "bucket": "words_120",
        "source_id": "source-1",
        "category": "news",
        "word_count": 2,
        "text": "hello world",
        "mode": "one_shot",
        "seed": 42,
        "order_index": 0,
        "sample_rate": 16000,
    }
    audio_path = tmp_path / "audio.wav"
    sf.write(audio_path, np.zeros(1600, dtype=np.float32), 16000)
    reference_path = tmp_path / "reference.jsonl"
    vllm_path = tmp_path / "vllm.jsonl"
    write_jsonl(
        reference_path,
        [
            {
                **common,
                "backend": "reference",
                "status": "success",
                "audio_path": str(audio_path),
                "audio_duration_s": 0.1,
                "latency_s": 1.0,
                "rtf": 10.0,
            }
        ],
    )
    write_jsonl(
        vllm_path,
        [
            {
                **common,
                "backend": "vllm-omni",
                "status": "error",
                "audio_path": None,
                "audio_duration_s": None,
                "latency_s": None,
                "rtf": None,
                "error_type": "RuntimeError",
                "error": "generation failed",
            }
        ],
    )

    class FakeWhisper:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(evaluate_module.AutoProcessor, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(evaluate_module, "WhisperForConditionalGeneration", FakeWhisper)
    monkeypatch.setattr(evaluate_module, "transcribe_waveform", lambda *args, **kwargs: "hello world")
    args = SimpleNamespace(
        records=[reference_path, vllm_path],
        output_dir=tmp_path / "evaluation",
        whisper_model="openai/whisper-large-v3",
        model_revision="revision",
        device="cpu",
        dtype="float32",
    )

    evaluate_module.run(args)

    output = capsys.readouterr().out
    assert "Loading openai/whisper-large-v3 for Whisper transcription. Pending audio files: 1." in output
    assert "Whisper loaded. Starting transcription." in output
    evaluated = read_jsonl(args.output_dir / "evaluated.jsonl")
    assert [row["evaluation_status"] for row in evaluated] == ["success", "generation_error"]

    monkeypatch.setattr(
        evaluate_module.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Whisper must not reload")),
    )
    evaluate_module.run(args)
