from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.worker.gpu_generation_model_runner as generation_runner
from vllm_omni.outputs import StageMemoryStats
from vllm_omni.worker.gpu_generation_model_runner import (
    ExecuteModelState,
    GPUGenerationModelRunner,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyInputBatch:
    def __init__(self):
        self.req_ids = ["req-1"]
        self.req_id_to_index = {"req-1": 0}
        self.num_reqs = 1
        self.vocab_size = 10


def _make_runner(multimodal_outputs, *, collect_stage_stats: bool = False):
    runner = object.__new__(GPUGenerationModelRunner)
    runner.execute_model_state = ExecuteModelState(
        SimpleNamespace(collect_stage_stats=collect_stage_stats),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        multimodal_outputs,
        None,
    )
    runner.kv_connector_output = None
    runner.input_batch = _DummyInputBatch()
    runner.use_async_scheduling = False
    runner.device = torch.device("cpu")
    runner.supports_mm_inputs = False
    runner.speculative_config = None
    runner.routed_experts_initialized = False
    runner._async_chunk = False
    runner.model = SimpleNamespace()
    return runner


def test_sample_tokens_tensor_output():
    multimodal_outputs = torch.randn(1, 2, 3)
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.multimodal_outputs) == 1
    assert output.multimodal_outputs[0]["model_outputs"].shape == (2, 3)


def test_sample_tokens_list_output():
    multimodal_outputs = [torch.randn(2, 1)]
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.multimodal_outputs) == 1
    assert output.multimodal_outputs[0]["model_outputs"].shape == (2, 1)


def test_sample_tokens_list_allows_none_output():
    multimodal_outputs = [None]
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.multimodal_outputs) == 1
    assert output.multimodal_outputs[0]["model_outputs"] is None


def test_sample_tokens_dict_output():
    multimodal_outputs = {"audio": torch.randn(1, 4), "unused": None}
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.multimodal_outputs) == 1
    assert "audio" in output.multimodal_outputs[0]
    assert "unused" not in output.multimodal_outputs[0]
    assert output.multimodal_outputs[0]["audio"].shape == (1, 4)


def test_sample_tokens_collects_stage_memory_from_raw_model():
    runner = _make_runner(torch.randn(1, 2, 3), collect_stage_stats=True)
    expected = StageMemoryStats(allocated_bytes=1)
    runner.get_model = lambda: SimpleNamespace(get_stage_memory_stats=lambda: expected)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert output.stage_memory_stats is expected


def test_sample_tokens_skips_stage_memory_hook_without_stats_tick():
    runner = _make_runner(torch.randn(1, 2, 3))
    runner.model.get_stage_memory_stats = lambda: pytest.fail("stats hook must not be called")

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert output.stage_memory_stats is None


def test_sample_tokens_without_stage_memory_hook():
    runner = _make_runner(torch.randn(1, 2, 3), collect_stage_stats=True)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert output.stage_memory_stats is None


def test_zero_token_cleanup_collects_stage_memory(monkeypatch):
    runner = object.__new__(GPUGenerationModelRunner)
    runner.execute_model_state = None
    runner.routed_experts_initialized = False
    runner.speculative_config = None
    runner.model_config = SimpleNamespace(async_chunk=False)
    runner.synchronize_input_prep = nullcontext
    runner._update_states = lambda _: None
    runner.attach_omni_connector_output = lambda output: output

    cache_entries = 1

    def on_requests_finished(_):
        nonlocal cache_entries
        cache_entries = 0

    raw_model = SimpleNamespace(
        on_requests_finished=on_requests_finished,
        get_stage_memory_stats=lambda: StageMemoryStats(ref_context_cache_entries=cache_entries),
    )
    runner.model = SimpleNamespace()
    runner.get_model = lambda: raw_model
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=0,
        num_scheduled_tokens={},
        finished_req_ids={"req-1"},
        collect_stage_stats=True,
    )
    monkeypatch.setattr(generation_runner, "has_kv_transfer_group", lambda: False)

    output = GPUGenerationModelRunner.execute_model(runner, scheduler_output)

    assert output.stage_memory_stats.ref_context_cache_entries == 0
