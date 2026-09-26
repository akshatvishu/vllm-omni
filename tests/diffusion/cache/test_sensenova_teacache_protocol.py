# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.cache.teacache.backend import TeaCacheBackend
from vllm_omni.diffusion.cache.teacache.coefficient_estimator import DataCollectionHook
from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook
from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.diffusion.hooks import HookRegistry, ModelHook
from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import (
    SenseNovaU1ForCausalLM,
    SenseNovaU1Model,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, hidden_states):
        return hidden_states * self.scale


class _Rope(nn.Module):
    def forward(self, hidden_states, indexes):
        return hidden_states + indexes.unsqueeze(-1)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm_mot_gen = _Scale(2)
        self.calls = 0

    def forward(self, hidden_states, **kwargs):
        self.calls += 1
        if kwargs["exist_und"] and kwargs["exist_gen"]:
            raise NotImplementedError("Mixed und+gen tokens")
        assert len(kwargs["position_embeddings"]) == 3
        assert kwargs["position_embeddings"][0].shape == hidden_states.shape
        assert kwargs["attention_mask"]["full_attention"] is None
        return hidden_states + hidden_states.square() + (2 if kwargs["exist_gen"] else 1)


def _model():
    model = SenseNovaU1Model.__new__(SenseNovaU1Model)
    nn.Module.__init__(model)
    model.embed_tokens = nn.Embedding(8, 3)
    model.layers = nn.ModuleList([_Layer()])
    model.norm = _Scale(3)
    model.norm_mot_gen = _Scale(5)
    model.rotary_emb = _Rope()
    model.rotary_emb_hw = _Rope()
    return model


def _args(generation=True, **extra):
    return dict(
        inputs_embeds=torch.full((1, 1, 3), 0.25),
        image_gen_indicators=torch.ones(1, 1, dtype=torch.bool) if generation else None,
        indexes=torch.zeros(3, 1, dtype=torch.long),
        use_cache=True,
        update_cache=False,
        **extra,
    )


def test_split_forward_keeps_outputs_and_kv_state():
    model = _model()
    for generation in (False, True):
        args = _args(generation)
        actual = model(**args)
        initial = args["inputs_embeds"]
        block_output = initial + initial.square() + (2 if generation else 1)
        expected = block_output * (5 if generation else 3)
        torch.testing.assert_close(actual.last_hidden_state, expected)
        assert actual.past_key_values is not None
        assert model.layers[0].calls == (2 if generation else 1)


class _OffloadProbe(ModelHook):
    def __init__(self):
        self.calls = 0

    def pre_forward(self, module, *args, **kwargs):
        self.calls += 1
        return args, kwargs


def test_cache_hit_and_bypass_still_call_language_model_hooks():
    model = _model()
    language_model = SenseNovaU1ForCausalLM.__new__(SenseNovaU1ForCausalLM)
    nn.Module.__init__(language_model)
    language_model.model = model
    probe = _OffloadProbe()
    HookRegistry.get_or_create(language_model).register_hook("offload_probe", probe)

    backend = TeaCacheBackend(DiffusionCacheConfig(coefficients=[0, 0, 0, 0, 0]))
    backend.enable(SimpleNamespace(transformer=model))
    assert model._hook_registry.get_hook("teacache") is not None

    first = language_model(**_args(), compute_logits=False)
    second = language_model(**_args(), compute_logits=False)
    torch.testing.assert_close(first.hidden_states, second.hidden_states)
    assert first.logits is None
    assert first.past_key_values is not None
    assert model.layers[0].calls == 1

    language_model(**_args(False), compute_logits=False)
    language_model(**_args(skip_step_cache=True), compute_logits=False)
    explicit_und = _args()
    explicit_und["image_gen_indicators"] = torch.zeros(1, 1, dtype=torch.bool)
    language_model(**explicit_und, compute_logits=False)
    embeds = language_model(input_ids=torch.tensor([[1]]), embed_only=True).inputs_embeds
    assert embeds.shape == (1, 1, 3)
    assert model.layers[0].calls == 4
    assert probe.calls == 6


def test_cfg_bypass_does_not_shift_branch_cache(monkeypatch):
    monkeypatch.setattr("vllm_omni.diffusion.cache.teacache.hook.get_classifier_free_guidance_world_size", lambda: 1)
    model = _model()
    model.do_true_cfg = True
    TeaCacheBackend(DiffusionCacheConfig(coefficients=[0, 0, 0, 0, 0])).enable(SimpleNamespace(transformer=model))
    hook = model._hook_registry.get_hook("teacache")

    def run_branch(value, *, skip=False):
        args = _args(skip_step_cache=True) if skip else _args()
        args["inputs_embeds"] = torch.full((1, 1, 3), value)
        return model(**args).last_hidden_state

    cond_first = run_branch(0.25)
    run_branch(0.5, skip=True)
    uncond_first = run_branch(0.75)
    cond_second = run_branch(0.25)
    run_branch(0.5, skip=True)
    uncond_second = run_branch(0.75)

    torch.testing.assert_close(cond_second, cond_first)
    torch.testing.assert_close(uncond_second, uncond_first)
    assert not torch.equal(cond_first, uncond_first)
    assert model.layers[0].calls == 4
    assert hook._forward_cnt == 4


def test_kv_update_bypasses_step_cache():
    model = _model()
    TeaCacheBackend(DiffusionCacheConfig(coefficients=[0, 0, 0, 0, 0])).enable(SimpleNamespace(transformer=model))
    args = _args()
    args["update_cache"] = True

    model(**args)
    model(**args)

    assert model.layers[0].calls == 2
    assert model._hook_registry.get_hook("teacache")._forward_cnt == 0


def test_collector_records_only_denoising_calls():
    model = _model()
    collector = DataCollectionHook("SenseNovaU1Model")
    HookRegistry.get_or_create(model).register_hook(collector._HOOK_NAME, collector)
    model(**_args(False))
    explicit_und = _args()
    explicit_und["image_gen_indicators"] = torch.zeros(1, 1, dtype=torch.bool)
    model(**explicit_und)
    assert collector.stop_collection() == []
    model(**_args())
    signal, output = collector.stop_collection()[0]
    assert len(collector.stop_collection()) == 1
    assert signal.shape == output.shape == (1, 1, 3)
    assert model.layers[0].calls == 3


def test_mixed_indicators_still_reach_decoder_error():
    model = _model()
    HookRegistry.get_or_create(model).register_hook(
        "teacache",
        TeaCacheHook(TeaCacheConfig(transformer_type="SenseNovaU1Model", coefficients=[0, 0, 0, 0, 0])),
    )
    model(**_args())
    mixed = _args()
    mixed["inputs_embeds"] = torch.full((1, 2, 3), 0.25)
    mixed["image_gen_indicators"] = torch.tensor([[False, True]])
    mixed["indexes"] = torch.zeros(3, 2, dtype=torch.long)
    with pytest.raises(NotImplementedError, match=r"Mixed und\+gen tokens"):
        model(**mixed)
    assert model.layers[0].calls == 2
