from unittest.mock import MagicMock

import pytest
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput

from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.outputs import OmniModelRunnerOutput, StageMemoryStats

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _scheduler_output(finished_req_ids: set[str], *, total_num_scheduled_tokens: int = 0) -> SchedulerOutput:
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
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=finished_req_ids,
        free_encoder_mm_hashes=[],
    )


def test_stage_stats_collection_does_not_predict_stats_interval():
    scheduler = OmniSchedulerMixin.__new__(OmniSchedulerMixin)
    scheduler.log_stats = True
    scheduler._last_stats_time = float("inf")

    active = scheduler._wrap_omni_scheduler_output(_scheduler_output(set(), total_num_scheduled_tokens=1))
    finished = scheduler._wrap_omni_scheduler_output(_scheduler_output({"req-1"}))
    idle = scheduler._wrap_omni_scheduler_output(_scheduler_output(set()))
    scheduler.log_stats = False
    disabled = scheduler._wrap_omni_scheduler_output(_scheduler_output({"req-1"}, total_num_scheduled_tokens=1))

    assert active.collect_stage_stats
    assert finished.collect_stage_stats
    assert not idle.collect_stage_stats
    assert not disabled.collect_stage_stats


def _empty_scheduler() -> MagicMock:
    scheduler = MagicMock()
    scheduler.chunk_transfer_adapter = None
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.perf_metrics = None
    scheduler.log_stats = False
    scheduler.recompute_kv_load_failures = False
    scheduler.finished_req_ids_dict = {}
    scheduler.finished_req_ids = set()
    scheduler.requests = {}
    scheduler.running = []
    scheduler._pending_finish_reqs = []
    scheduler._latest_stage_memory_stats = None
    scheduler.kv_cache_manager.take_events.return_value = None
    return scheduler


def _update_from_empty_scheduler(
    model_runner_output: ModelRunnerOutput,
    *,
    finished_req_ids: set[str] | None = None,
    scheduler_stats: SchedulerStats | None = None,
    scheduler: MagicMock | None = None,
):
    if scheduler is None:
        scheduler = _empty_scheduler()
    scheduler.make_stats.return_value = scheduler_stats

    return OmniGenerationScheduler.update_from_output(
        scheduler,
        _scheduler_output(finished_req_ids or set()),
        model_runner_output,
    )


def test_update_from_output_accepts_upstream_output_without_stage_memory_stats():
    outputs = _update_from_empty_scheduler(ModelRunnerOutput(req_ids=[], req_id_to_index={}))

    assert outputs == {}


def test_update_from_output_stats_tick_without_stage_memory_observation():
    scheduler_stats = MagicMock()
    outputs = _update_from_empty_scheduler(
        ModelRunnerOutput(req_ids=[], req_id_to_index={}),
        scheduler_stats=scheduler_stats,
    )

    assert outputs[0].scheduler_stats is scheduler_stats
    assert outputs[0].stage_memory_stats is None


def test_update_from_output_forwards_omni_stage_memory_stats():
    stats = StageMemoryStats(allocated_bytes=1)
    scheduler_stats = MagicMock()
    outputs = _update_from_empty_scheduler(
        OmniModelRunnerOutput(req_ids=[], req_id_to_index={}, stage_memory_stats=stats),
        scheduler_stats=scheduler_stats,
    )

    assert outputs[0].scheduler_stats is scheduler_stats
    assert outputs[0].stage_memory_stats is stats


def test_update_from_output_retains_stage_memory_stats_until_next_stats_tick():
    scheduler = _empty_scheduler()
    stats = StageMemoryStats(allocated_bytes=1)

    deferred_outputs = _update_from_empty_scheduler(
        OmniModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            stage_memory_stats=stats,
        ),
        scheduler=scheduler,
    )
    scheduler_stats = MagicMock()
    emitted_outputs = _update_from_empty_scheduler(
        ModelRunnerOutput(req_ids=[], req_id_to_index={}),
        scheduler_stats=scheduler_stats,
        scheduler=scheduler,
    )

    assert deferred_outputs == {}
    assert emitted_outputs[0].scheduler_stats is scheduler_stats
    assert emitted_outputs[0].stage_memory_stats is stats


def test_update_from_output_cleanup_replaces_retained_stage_memory_stats():
    scheduler = _empty_scheduler()
    scheduler._latest_stage_memory_stats = StageMemoryStats(ref_context_cache_entries=1)
    cleanup_stats = StageMemoryStats()

    cleanup_outputs = _update_from_empty_scheduler(
        OmniModelRunnerOutput(req_ids=[], req_id_to_index={}, stage_memory_stats=cleanup_stats),
        finished_req_ids={"req-1"},
        scheduler=scheduler,
    )
    next_tick_outputs = _update_from_empty_scheduler(
        ModelRunnerOutput(req_ids=[], req_id_to_index={}),
        scheduler_stats=MagicMock(),
        scheduler=scheduler,
    )

    assert cleanup_outputs[0].stage_memory_stats is cleanup_stats
    assert next_tick_outputs[0].stage_memory_stats is cleanup_stats


def test_update_from_output_emits_cleanup_when_current_output_finishes_request():
    scheduler = _empty_scheduler()
    scheduler.finished_req_ids = {"req-1"}
    cleanup_stats = StageMemoryStats()

    outputs = _update_from_empty_scheduler(
        OmniModelRunnerOutput(req_ids=[], req_id_to_index={}, stage_memory_stats=cleanup_stats),
        scheduler=scheduler,
    )

    assert outputs[0].stage_memory_stats is cleanup_stats
