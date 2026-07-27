# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check selected ``vllm serve --omni`` flags without loading a model.

The tests use the production parser, current runtime registry and deploy
resolver, structured resolver from issue #4021, diffusion fallback, final
config projection, and lightweight runtime consumers. Each test checks the
exact point where a flag is applied or lost.

Loading a model would not strengthen these config checks. The one GPU serving
test in ``tests/e2e/online_serving/test_entrypoint_flag_smoke.py`` instead
checks one flag whose effect can be observed through an HTTP request.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.config.yaml_util import create_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.cosmos3.guardrails import is_guardrails_enabled
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.stage_init_utils import extract_stage_metadata
from vllm_omni.entrypoints.cli import serve
from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.entrypoints.openai.api_server import (
    _check_max_generated_image_size,
    _generate_with_async_omni,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.utils.forced_aligner import build_forced_aligner_config
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _parse_serve_args(*extra_args: str):
    parser = TrackingArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")
    command = OmniServeCommand()
    command.subparser_init(subparsers)
    args = parser.parse_args(
        [
            "serve",
            "Qwen/Qwen2.5-Omni-7B",
            "--omni",
            *extra_args,
        ]
    )
    return command, args, args.get_explicit_kwargs_dict()


def _final_diffusion_config(engine_args) -> OmniDiffusionConfig:
    kwargs = dict(engine_args)
    kwargs["master_port"] = 29500
    # Port selection is unrelated to config projection and may be forbidden in
    # hermetic CPU test sandboxes.
    with patch("vllm_omni.diffusion.data.is_port_available", return_value=True):
        return OmniDiffusionConfig.from_kwargs(**kwargs)


def test_diffusion_flags_reach_the_production_fallback():
    """Unregistered, single-stage diffusion models use this fallback."""
    _, _, explicit = _parse_serve_args(
        "--diffusion-streaming-output",
        "--default-sampling-params",
        '{"0": {"num_inference_steps": 7, "guidance_scale": 2.5}}',
        "--auxiliary-text-encoder",
        "example/encoder",
        "--num-gpus",
        "7",
    )

    stage = AsyncOmniEngine._create_default_diffusion_stage_cfg(explicit)[0]
    stage_config = create_config(stage)
    metadata = extract_stage_metadata(stage_config)
    final_config = _final_diffusion_config(stage["engine_args"])

    assert final_config.streaming_output is True
    assert final_config.extras["auxiliary_text_encoder"] == "example/encoder"
    assert metadata.default_sampling_params.num_inference_steps == 7
    assert metadata.default_sampling_params.guidance_scale == 2.5

    # --num-gpus is not used by the fallback. The final value is derived from
    # parallel_config.world_size instead.
    assert "num_gpus" not in stage["engine_args"]
    assert final_config.num_gpus == 1


def test_diffusion_flags_are_lost_on_the_registry_deploy_path(mocker):
    """Characterize the current GLM-Image registry/deploy behavior."""
    _, _, explicit = _parse_serve_args(
        "--deploy-config",
        "vllm_omni/deploy/glm_image.yaml",
        "--diffusion-streaming-output",
        "--default-sampling-params",
        '{"1": {"num_inference_steps": 7, "guidance_scale": 2.5}}',
        "--auxiliary-text-encoder",
        "example/encoder",
        "--num-gpus",
        "7",
    )
    deploy_path = explicit.pop("deploy_config")
    pipeline = resolve_pipeline_config("glm_image")
    mocker.patch.object(StageConfigFactory, "get_pipeline_config", return_value=pipeline)

    stages, _ = StageConfigFactory.create_legacy_stage_configs_from_model(
        "THUDM/GLM-Image",
        trust_remote_code=False,
        cli_overrides=explicit,
        deploy_config_path=deploy_path,
    )
    assert stages is not None
    diffusion_stage = stages[1]
    stage_config = diffusion_stage.to_omegaconf()
    metadata = extract_stage_metadata(stage_config)
    final_config = _final_diffusion_config(stage_config.engine_args)

    # The first two values reach the intermediate runtime-overrides mapping,
    # but their names/shapes do not match the final consumers.
    assert diffusion_stage.runtime_overrides["diffusion_streaming_output"] is True
    assert diffusion_stage.runtime_overrides["auxiliary_text_encoder"] == "example/encoder"
    assert final_config.streaming_output is False
    assert "auxiliary_text_encoder" not in final_config.__dict__

    # The CLI JSON does not update the stage's deploy sampling defaults.
    assert metadata.default_sampling_params.num_inference_steps == 50
    assert metadata.default_sampling_params.guidance_scale == 1.5

    # --num-gpus is filtered as orchestrator-owned, and the diffusion config
    # derives one device from its parallel config.
    assert "num_gpus" not in diffusion_stage.runtime_overrides
    assert final_config.num_gpus == 1


def test_diffusion_flags_are_also_lost_on_the_structured_config_path():
    """Check the additive config path from issue #4021 before runtime cutover."""
    _, _, explicit = _parse_serve_args(
        "--deploy-config",
        "vllm_omni/deploy/glm_image.yaml",
        "--diffusion-streaming-output",
        "--default-sampling-params",
        '{"1": {"num_inference_steps": 7, "guidance_scale": 2.5}}',
        "--auxiliary-text-encoder",
        "example/encoder",
        "--num-gpus",
        "7",
    )
    deploy_path = explicit.pop("deploy_config")

    config = VllmOmniConfig.from_pipeline_config(
        resolve_pipeline_config("glm_image"),
        deploy_config_path=deploy_path,
        cli_overrides={"model": "THUDM/GLM-Image", **explicit},
    )
    diffusion_stage = config.stage_by_id(1)

    assert diffusion_stage.model_config.default_sampling_params["num_inference_steps"] == 50
    assert diffusion_stage.model_config.default_sampling_params["guidance_scale"] == 1.5
    assert getattr(diffusion_stage.diffusion_config, "streaming_output", False) is False
    assert getattr(diffusion_stage.diffusion_config, "auxiliary_text_encoder", None) is None
    assert diffusion_stage.runtime_config.num_gpus == 1


@pytest.mark.asyncio
async def test_image_generation_replaces_engine_diffusion_defaults():
    class FakeEngine:
        def __init__(self):
            self.default_sampling_params_list = [
                OmniDiffusionSamplingParams(
                    num_inference_steps=7,
                    guidance_scale=2.5,
                )
            ]
            self.received_sampling_params = None

        async def generate(self, *, sampling_params_list, **kwargs):
            self.received_sampling_params = sampling_params_list
            yield SimpleNamespace()

    engine = FakeEngine()
    request_params = OmniDiffusionSamplingParams()
    await _generate_with_async_omni(
        engine_client=engine,
        gen_params=request_params,
        stage_configs=[SimpleNamespace(stage_type="diffusion")],
        prompt={"prompt": "test"},
        request_id="request-id",
    )

    received = engine.received_sampling_params[0]
    assert received.num_inference_steps == request_params.num_inference_steps
    assert received.guidance_scale == request_params.guidance_scale
    assert received.num_inference_steps != 7
    assert received.guidance_scale != 2.5


def test_final_diffusion_projection_silently_drops_unknown_keys():
    config = _final_diffusion_config(
        {
            "model": "test-model",
            "unknown_diffusion_option": True,
            "engine_extras": {"model_specific_value": 3},
        }
    )

    assert "unknown_diffusion_option" not in config.__dict__
    assert "engine_extras" not in config.__dict__


def test_explicit_deploy_extras_survive_the_registry_path(tmp_path, mocker):
    base_deploy = Path("vllm_omni/deploy/glm_image.yaml").resolve()
    deploy_path = tmp_path / "glm_image_with_extra.yaml"
    deploy_path.write_text(
        f"base_config: {base_deploy}\n"
        "stages:\n"
        "  - stage_id: 1\n"
        "    engine_extras:\n"
        "      extras:\n"
        "        model_specific_value: 3\n",
        encoding="utf-8",
    )
    pipeline = resolve_pipeline_config("glm_image")
    mocker.patch.object(StageConfigFactory, "get_pipeline_config", return_value=pipeline)

    stages, _ = StageConfigFactory.create_legacy_stage_configs_from_model(
        "THUDM/GLM-Image",
        trust_remote_code=False,
        cli_overrides={},
        deploy_config_path=str(deploy_path),
    )
    assert stages is not None
    final_config = _final_diffusion_config(stages[1].to_omegaconf().engine_args)

    assert final_config.extras["model_specific_value"] == 3


def test_no_guardrails_reaches_the_cosmos3_consumer(monkeypatch, mocker):
    command, args, _ = _parse_serve_args("--no-guardrails")
    monkeypatch.setenv("VLLM_DISABLE_LOG_LOGO", "1")
    run_server = mocker.patch.object(serve, "omni_run_server")
    run = mocker.patch.object(serve.uvloop, "run")

    command.cmd(args)

    run_server.assert_called_once_with(args)
    run.assert_called_once()
    # uvloop.run is mocked, so close the coroutine it would normally consume.
    run.call_args.args[0].close()
    assert args.model_config["guardrails"] is False
    assert not is_guardrails_enabled(SimpleNamespace(model_config=args.model_config))


def test_forced_aligner_flags_reach_the_frontend_consumer(tmp_path):
    config_path = tmp_path / "aligner.yaml"
    config_path.write_text(
        "forced_aligner:\n  model: config/model\n  gpu_memory_utilization: 0.15\n",
        encoding="utf-8",
    )
    _, args, _ = _parse_serve_args(
        "--forced-aligner-config",
        str(config_path),
        "--forced-aligner",
        "cli/model",
    )

    config = build_forced_aligner_config(args)

    assert config is not None
    assert config.model == "cli/model"
    assert config.gpu_memory_utilization == 0.15


def test_max_generated_image_size_reaches_request_validation():
    _, args, _ = _parse_serve_args("--max-generated-image-size", "65536")

    _check_max_generated_image_size(args, 256, 256)
    with pytest.raises(HTTPException, match="exceeds the maximum allowed size"):
        _check_max_generated_image_size(args, 512, 256)


@pytest.mark.parametrize(
    ("cli_args", "destination"),
    [
        (("--data-parallel-size", "2"), "data_parallel_size"),
        (("--data-parallel-size-local", "2"), "data_parallel_size_local"),
        (("--data-parallel-address", "127.0.0.1"), "data_parallel_address"),
        (("--data-parallel-rpc-port", "29550"), "data_parallel_rpc_port"),
        (("--data-parallel-start-rank", "1"), "data_parallel_start_rank"),
        (("--data-parallel-backend", "mp"), "data_parallel_backend"),
        (("--api-server-count", "2"), "api_server_count"),
        (("--enable-expert-parallel",), "enable_expert_parallel"),
    ],
)
def test_unsupported_inherited_flags_are_not_rejected_by_real_parser(cli_args, destination, mocker):
    mocker.patch(
        "vllm_omni.diffusion.utils.hf_utils.is_diffusion_model",
        return_value=True,
    )
    command, args, _ = _parse_serve_args(*cli_args)

    assert destination in args.explicit_keys
    assert not hasattr(args, "_cli_explicit_keys")
    command.validate(args)
