# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_minicpmo_sliding_overlay_is_explicitly_opt_in() -> None:
    base = load_deploy_config(_REPO_ROOT / "vllm_omni/deploy/minicpmo_4_5.yaml")
    sliding = load_deploy_config(_REPO_ROOT / "vllm_omni/deploy/minicpmo_4_5_sliding.yaml")

    base_stage = next(stage for stage in base.stages if stage.stage_id == 1)
    sliding_stage = next(stage for stage in sliding.stages if stage.stage_id == 1)
    assert base_stage.minicpmo_sliding_recompute is None
    assert sliding_stage.minicpmo_sliding_recompute is True
    assert sliding_stage.async_scheduling is False
    assert sliding_stage.minicpmo_sliding_window_size == 2
    assert sliding_stage.minicpmo_sliding_recomputed_chunks == 1

    pipeline = resolve_pipeline_config("minicpmo_4_5")
    assert pipeline is not None
    resolved_stages = merge_pipeline_deploy(pipeline, sliding)
    assert resolved_stages[1].yaml_engine_args["minicpmo_sliding_recompute"] is True
    assert resolved_stages[1].yaml_engine_args["async_scheduling"] is False
    assert resolved_stages[1].yaml_engine_args["minicpmo_sliding_window_size"] == 2
    assert resolved_stages[1].yaml_engine_args["minicpmo_sliding_recomputed_chunks"] == 1


def test_minicpmo_sliding_recompute_rejects_async_scheduling() -> None:
    deploy = load_deploy_config(_REPO_ROOT / "vllm_omni/deploy/minicpmo_4_5_sliding.yaml")
    sliding_stage = next(stage for stage in deploy.stages if stage.stage_id == 1)
    sliding_stage.async_scheduling = True

    pipeline = resolve_pipeline_config("minicpmo_4_5")
    assert pipeline is not None
    with pytest.raises(ValueError, match="requires async_scheduling=false"):
        merge_pipeline_deploy(pipeline, deploy)


def test_minicpmo_native_diagnostic_overlay_keeps_sliding_disabled() -> None:
    native = load_deploy_config(_REPO_ROOT / "vllm_omni/deploy/minicpmo_4_5_native_diagnostic.yaml")
    native_stage = next(stage for stage in native.stages if stage.stage_id == 1)

    assert native_stage.async_scheduling is False
    assert native_stage.minicpmo_sliding_recompute is None

    pipeline = resolve_pipeline_config("minicpmo_4_5")
    assert pipeline is not None
    resolved_stages = merge_pipeline_deploy(pipeline, native)
    assert resolved_stages[1].yaml_engine_args["async_scheduling"] is False
    assert "minicpmo_sliding_recompute" not in resolved_stages[1].yaml_engine_args
