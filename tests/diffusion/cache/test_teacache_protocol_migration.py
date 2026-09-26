# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import fully_shard

from vllm_omni.diffusion.cache.teacache.backend import TeaCacheBackend
from vllm_omni.diffusion.cache.teacache.coefficient_estimator import DataCollectionHook
from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.extractors import CacheContext, get_extractor
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook, apply_teacache_hook
from vllm_omni.diffusion.cache.teacache.protocol import (
    ForwardState,
    SupportsDecomposedForward,
    SupportsTeaCache,
    TeaCacheDefaults,
)
from vllm_omni.diffusion.cache.teacache.state import TeaCacheState
from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.diffusion.hooks import HookRegistry
from vllm_omni.diffusion.models.flux2_klein.flux2_klein_transformer import (
    Flux2Transformer2DModel as Flux2KleinTransformer2DModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _ProtocolModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_calls = 0

    def forward(self, hidden_states, signal):
        ctx = self.preprocess(hidden_states, signal, skip_modulated_input=True)
        return self.postprocess(self.run_transformer_blocks(ctx))

    def preprocess(self, hidden_states, signal, *, skip_modulated_input):
        return ForwardState(
            modulated_input=None if skip_modulated_input else signal,
            hidden_states=hidden_states.clone(),
            encoder_hidden_states=hidden_states.clone(),
            temb=signal,
            intermediates=None,
        )

    def run_transformer_blocks(self, ctx):
        self.block_calls += 1
        ctx.hidden_states.add_(2)
        ctx.encoder_hidden_states.add_(3)
        return ctx

    def postprocess(self, ctx):
        return ctx.hidden_states, ctx.encoder_hidden_states

    def get_teacache_defaults(self):
        return TeaCacheDefaults([0, 0, 0, 1, 0], rel_l1_thresh=0.4)


class _LegacyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_calls = 0

    def forward(self, hidden_states, signal):
        return hidden_states + 2


def _legacy_context(module, hidden_states, signal):
    def run_blocks():
        module.block_calls += 1
        return (hidden_states + 2,)

    return CacheContext(signal, hidden_states, None, signal, run_blocks, lambda output: output)


def _hook(model):
    hook = TeaCacheHook(TeaCacheConfig(transformer_type="FluxTransformer2DModel", coefficients=[0, 0, 0, 1, 0]))
    if isinstance(model, SupportsTeaCache):
        hook.initialize_hook(model)
    else:
        with patch("vllm_omni.diffusion.cache.teacache.hook.get_extractor", return_value=_legacy_context):
            hook.initialize_hook(model)
    return hook


def test_protocol_cache_hit_full_step_and_reset():
    model = _ProtocolModel()
    hook = _hook(model)

    first = hook.new_forward(model, torch.tensor([1.0]), torch.tensor([1.0]))
    hit = hook.new_forward(model, torch.tensor([4.0]), torch.tensor([1.0]))
    full = hook.new_forward(model, torch.tensor([5.0]), torch.tensor([2.0]))

    assert model.block_calls == 2
    assert first[0].item() == 3
    assert hit[0].item() == 6
    assert hit[1].item() == 7
    assert full[0].item() == 7
    hook.reset_state(model)
    hook.new_forward(model, torch.tensor([1.0]), torch.tensor([1.0]))
    assert model.block_calls == 3


def test_protocol_uses_returned_context_after_block_execution():
    class ReturnsNewContext(_ProtocolModel):
        def run_transformer_blocks(self, ctx):
            return replace(
                ctx, hidden_states=ctx.hidden_states + 2, encoder_hidden_states=ctx.encoder_hidden_states + 3
            )

    model = ReturnsNewContext()
    output = _hook(model).new_forward(model, torch.tensor([1.0]), torch.tensor([1.0]))
    assert output[0].item() == 3
    assert output[1].item() == 4


def test_sequential_cfg_keeps_branch_states_separate():
    model = _ProtocolModel()
    model.do_true_cfg = True
    hook = _hook(model)
    with patch("vllm_omni.diffusion.cache.teacache.hook.get_classifier_free_guidance_world_size", return_value=1):
        for hidden in (1.0, 10.0, 2.0, 20.0):
            hook.new_forward(model, torch.tensor([hidden]), torch.tensor([1.0]))
        assert model.block_calls == 2
        hook.reset_state(model)
        hook.new_forward(model, torch.tensor([1.0]), torch.tensor([1.0]))
    assert model.block_calls == 3


def test_legacy_extractor_keeps_cache_hit_behavior():
    model = _LegacyModel()
    hook = _hook(model)

    assert hook.new_forward(model, torch.tensor([1.0]), torch.tensor([1.0])).item() == 3
    assert hook.new_forward(model, torch.tensor([4.0]), torch.tensor([1.0])).item() == 6
    assert model.block_calls == 1


def test_legacy_decision_callback_is_preserved():
    model = _LegacyModel()
    callback_decisions = []

    def extractor(_module, hidden_states, signal):
        encoder_hidden_states = torch.tensor([5.0])

        def run_blocks():
            model.block_calls += 1
            return hidden_states + 2, encoder_hidden_states + 3

        def sync_decision(local_decision):
            callback_decisions.append(local_decision)
            return True

        return CacheContext(
            signal,
            hidden_states,
            encoder_hidden_states,
            signal,
            run_blocks,
            lambda output: output,
            extra_states={"synchronize_cache_decision": sync_decision},
        )

    hook = TeaCacheHook(TeaCacheConfig(transformer_type="MiniMaxH3DiTModel", coefficients=[0, 0, 0, 1, 0]))
    with patch("vllm_omni.diffusion.cache.teacache.hook.get_extractor", return_value=extractor):
        hook.initialize_hook(model)

    hook.new_forward(model, torch.tensor([1.0]), torch.tensor([1.0]))
    hook.new_forward(model, torch.tensor([2.0]), torch.tensor([1.0]))
    assert callback_decisions == [True, False]
    assert model.block_calls == 2
    torch.testing.assert_close(hook.state_manager.get_state().previous_residual_encoder, torch.tensor([3.0]))


def test_protocol_requires_modulated_input_for_cache():
    model = _ProtocolModel()
    hook = _hook(model)
    with patch.object(model, "preprocess", return_value=ForwardState(None, torch.ones(1), None, torch.ones(1), None)):
        with pytest.raises(ValueError, match="modulated input"):
            hook.new_forward(model, torch.ones(1), torch.ones(1))


@pytest.mark.parametrize(
    ("coefficients", "threshold", "expected_coefficients", "expected_threshold"),
    [
        (None, None, [0, 0, 0, 1, 0], 0.4),
        ([1, 0, 0, 0, 0], None, [1, 0, 0, 0, 0], 0.4),
        (None, 0.7, [0, 0, 0, 1, 0], 0.7),
        ([1, 0, 0, 0, 0], 0.7, [1, 0, 0, 0, 0], 0.7),
    ],
)
def test_protocol_defaults_resolve_before_config(coefficients, threshold, expected_coefficients, expected_threshold):
    class Pipeline:
        transformer = _ProtocolModel()

    backend = TeaCacheBackend(DiffusionCacheConfig(coefficients=coefficients, rel_l1_thresh=threshold))
    with patch("vllm_omni.diffusion.cache.teacache.backend.apply_teacache_hook") as apply_hook:
        backend.enable(Pipeline())

    config = apply_hook.call_args.args[1]
    assert config.coefficients == expected_coefficients
    assert config.rel_l1_thresh == expected_threshold


def test_klein_uses_protocol_defaults_without_custom_enabler():
    class Flux2KleinPipeline:
        transformer = Flux2KleinTransformer2DModel.__new__(Flux2KleinTransformer2DModel)

    with patch("vllm_omni.diffusion.cache.teacache.backend.apply_teacache_hook") as apply_hook:
        TeaCacheBackend(DiffusionCacheConfig()).enable(Flux2KleinPipeline())

    model, config = apply_hook.call_args.args
    defaults = model.get_teacache_defaults()
    assert model is Flux2KleinPipeline.transformer
    assert config.transformer_type == "Flux2Transformer2DModel"
    assert config.coefficients == defaults.coefficients
    assert config.rel_l1_thresh == defaults.rel_l1_thresh


def test_custom_legacy_enablers_keep_table_defaults():
    class BagelPipeline:
        bagel = _LegacyModel()

    class SenseNovaU1Pipeline:
        denoising_transformer = _LegacyModel()

    for pipeline, target in (
        (BagelPipeline(), "bagel"),
        (SenseNovaU1Pipeline(), "denoising_transformer"),
    ):
        with patch("vllm_omni.diffusion.cache.teacache.backend.apply_teacache_hook") as apply_hook:
            TeaCacheBackend(DiffusionCacheConfig()).enable(pipeline)
        model, config = apply_hook.call_args.args
        assert model is getattr(pipeline, target)
        assert len(config.coefficients) == 5
        assert config.rel_l1_thresh == 0.2


def test_sensenova_legacy_extractor_runs_denoising():
    class Layer:
        @staticmethod
        def input_layernorm_mot_gen(hidden_states):
            return hidden_states * 2

        @staticmethod
        def __call__(hidden_states, **kwargs):
            assert kwargs["exist_gen"]
            assert not kwargs["exist_und"]
            return hidden_states + 1

    module = SimpleNamespace(
        model=SimpleNamespace(layers=[Layer()], norm_mot_gen=lambda hidden_states: hidden_states * 3),
    )
    inputs_embeds = torch.ones(1, 2, 3)
    context = get_extractor("SenseNovaU1ForCausalLM")(
        module,
        inputs_embeds=inputs_embeds,
        image_gen_indicators=torch.ones(1, 2, dtype=torch.bool),
        compute_logits=False,
    )

    assert torch.equal(context.modulated_input, inputs_embeds * 2)
    output = context.postprocess(context.run_transformer_blocks()[0])
    assert torch.equal(output.hidden_states, torch.full_like(inputs_embeds, 6))
    assert output.logits is None


def test_sensenova_transformer_imports_before_teacache():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import SenseNovaU1ForCausalLM",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_forward_override_is_rejected_before_hook_installation():
    class Override(_ProtocolModel):
        def forward(self, hidden_states, signal):
            return hidden_states

    class Pipeline:
        transformer = Override()

    pipeline = Pipeline()
    with pytest.raises(TypeError, match="overrides the decomposed forward"):
        TeaCacheBackend(DiffusionCacheConfig()).enable(pipeline)
    assert HookRegistry.get_or_create(pipeline.transformer).get_hook("teacache") is None


def test_collector_runs_protocol_and_legacy_models():
    protocol = _ProtocolModel()
    collector = DataCollectionHook("NoExtractorNeeded")
    collector.initialize_hook(protocol)
    output = collector.new_forward(protocol, torch.tensor([1.0]), torch.tensor([1.0]))
    assert output[0].item() == 3
    signal, block_output = collector.stop_collection()[0]
    assert signal.item() == 1
    assert block_output.item() == 3

    legacy = _LegacyModel()
    collector = DataCollectionHook("Legacy")
    with patch("vllm_omni.diffusion.cache.teacache.coefficient_estimator.get_extractor", return_value=_legacy_context):
        collector.initialize_hook(legacy)
    assert collector.new_forward(legacy, torch.tensor([1.0]), torch.tensor([1.0])).item() == 3
    assert collector.stop_collection()[0][1].item() == 3


def test_shared_base_does_not_require_teacache_defaults():
    class OtherCacheModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.block_calls = 0

        forward = _ProtocolModel.forward
        preprocess = _ProtocolModel.preprocess
        run_transformer_blocks = _ProtocolModel.run_transformer_blocks
        postprocess = _ProtocolModel.postprocess

    model = OtherCacheModel()
    assert isinstance(model, SupportsDecomposedForward)
    assert not isinstance(model, SupportsTeaCache)
    ctx = model.preprocess(torch.tensor([1.0]), torch.tensor([1.0]), skip_modulated_input=True)
    result = model.postprocess(model.run_transformer_blocks(ctx))
    assert result[0].item() == 3
    assert model.block_calls == 1

    collector = DataCollectionHook("NoExtractorNeeded")
    collector.initialize_hook(model)
    collected = collector.new_forward(model, torch.tensor([1.0]), torch.tensor([1.0]))
    assert collected[0].item() == 3
    assert model.block_calls == 2


def test_sp_one_does_not_call_decision_collective():
    class Group:
        world_size = 1

        def all_reduce(self, _value):
            raise AssertionError("SP=1 must not reduce TeaCache statistics")

    model = _ProtocolModel()
    hook = _hook(model)
    with (
        patch("vllm_omni.diffusion.cache.teacache.hook.model_parallel_is_initialized", return_value=True),
        patch("vllm_omni.diffusion.cache.teacache.hook.get_sp_group", return_value=Group()),
    ):
        hook.new_forward(model, torch.tensor([1.0]), torch.tensor([1.0]))
        hook.new_forward(model, torch.tensor([2.0]), torch.tensor([1.0]))


def _gloo_decision_worker(rank: int, init_file: str, output_dir: str):
    class Group:
        world_size = 2

        @staticmethod
        def all_reduce(value):
            dist.all_reduce(value)
            return value

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        hook = TeaCacheHook(
            TeaCacheConfig(transformer_type="FluxTransformer2DModel", coefficients=[0, 0, 0, 1, 0], rel_l1_thresh=0.4)
        )
        state = TeaCacheState()
        state.cnt = 1
        state.previous_modulated_input = torch.ones(2)
        current = torch.ones(2) * (rank + 1)
        with (
            patch("vllm_omni.diffusion.cache.teacache.hook.model_parallel_is_initialized", return_value=True),
            patch("vllm_omni.diffusion.cache.teacache.hook.get_sp_group", return_value=Group()),
        ):
            decision = hook._should_compute_full_transformer(state, current)
        (Path(output_dir) / f"rank{rank}.txt").write_text(str(decision))
    finally:
        dist.destroy_process_group()


def test_sp_two_gloo_ranks_agree_on_decision(tmp_path):
    mp.spawn(_gloo_decision_worker, args=(str(tmp_path / "init"), str(tmp_path)), nprocs=2)
    assert (tmp_path / "rank0.txt").read_text() == "True"
    assert (tmp_path / "rank1.txt").read_text() == "True"


def test_fsdp2_wrapper_keeps_protocol_dispatch(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'init'}", rank=0, world_size=1)
    try:
        model = _ProtocolModel()
        model.weight = nn.Parameter(torch.ones(1))
        fully_shard(model, mesh=DeviceMesh("cpu", [0]))
        assert isinstance(model, SupportsTeaCache)
        apply_teacache_hook(model, TeaCacheConfig(transformer_type="WrappedProtocol", coefficients=[0, 0, 0, 1, 0]))
        result = model(torch.tensor([1.0]), torch.tensor([1.0]))
        assert result[0].item() == 3
        assert model.block_calls == 1
    finally:
        dist.destroy_process_group()
