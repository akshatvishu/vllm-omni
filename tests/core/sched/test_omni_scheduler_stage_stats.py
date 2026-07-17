import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.outputs import OmniModelRunnerOutput, StageMemoryStats

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _scheduler_output(finished_req_ids: set[str]) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        ),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=finished_req_ids,
        free_encoder_mm_hashes=[],
    )


def test_finished_requests_force_stage_stats_collection():
    scheduler = OmniSchedulerMixin.__new__(OmniSchedulerMixin)
    scheduler.log_stats = True
    scheduler._last_stats_time = time.monotonic()

    finished = scheduler._wrap_omni_scheduler_output(_scheduler_output({"req-1"}))
    idle = scheduler._wrap_omni_scheduler_output(_scheduler_output(set()))
    scheduler.log_stats = False
    disabled = scheduler._wrap_omni_scheduler_output(_scheduler_output({"req-1"}))

    assert finished.collect_stage_stats
    assert not idle.collect_stage_stats
    assert not disabled.collect_stage_stats


def _empty_generation_scheduler() -> MagicMock:
    scheduler = MagicMock()
    scheduler.chunk_transfer_adapter = None
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.perf_metrics = None
    scheduler.log_stats = False
    scheduler.recompute_kv_load_failures = False
    scheduler.finished_req_ids_dict = {}
    scheduler.requests = {}
    scheduler.running = []
    scheduler._pending_finish_reqs = []
    scheduler.kv_cache_manager.take_events.return_value = None
    scheduler.make_stats.return_value = None
    return scheduler


def _update_from_empty_scheduler(model_runner_output: ModelRunnerOutput):
    scheduler = _empty_generation_scheduler()
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={},
        scheduled_spec_decode_tokens={},
        num_invalid_spec_tokens=0,
    )

    return OmniGenerationScheduler.update_from_output(scheduler, scheduler_output, model_runner_output)


def test_update_from_output_accepts_upstream_output_without_stage_memory_stats():
    outputs = _update_from_empty_scheduler(ModelRunnerOutput(req_ids=[], req_id_to_index={}))

    assert outputs == {}


def test_update_from_output_forwards_omni_stage_memory_stats():
    stats = StageMemoryStats(allocated_bytes=1)
    outputs = _update_from_empty_scheduler(
        OmniModelRunnerOutput(req_ids=[], req_id_to_index={}, stage_memory_stats=stats)
    )

    assert outputs[0].stage_memory_stats is stats
