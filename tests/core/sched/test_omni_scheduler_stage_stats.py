import time

import pytest
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin

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
