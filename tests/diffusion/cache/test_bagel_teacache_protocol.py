# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract checks for Bagel's packed TeaCache boundary."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from tests.diffusion.models.bagel.test_forward_cache_update_vae import HIDDEN_SIZE, _make_bagel_config
from vllm_omni.diffusion.cache.teacache.coefficient_estimator import BagelAdapter, DataCollectionHook
from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook
from vllm_omni.diffusion.hooks import HookRegistry
from vllm_omni.diffusion.models.bagel.bagel_transformer import (
    Bagel,
    BagelRotaryEmbedding,
    BaseNavitOutputWithPast,
    NaiveCache,
    Qwen2MoTDecoderLayer,
    Qwen2MoTModel,
)
from vllm_omni.diffusion.models.bagel.mot.mot_layernorm import MoTRMSNorm
from vllm_omni.diffusion.models.lance.lance_transformer import LanceBagel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _LanguageModel(nn.Module):
    """Nonlinear packed decoder with distinguishable CFG branch outputs."""

    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype
        self.calls = []

    def forward(self, **kwargs):
        if kwargs.get("return_embeddings_only"):
            ids = kwargs["packed_text_ids"]
            embeddings = ids[:, None].expand(-1, HIDDEN_SIZE).to(self.dtype) / 10
            return BaseNavitOutputWithPast(packed_query_sequence=embeddings)
        self.calls.append(kwargs)
        assert kwargs["update_past_key_values"] is False
        assert kwargs["is_causal"] is False
        hidden = kwargs["packed_query_sequence"]
        positions = kwargs["packed_query_position_ids"]
        if positions.ndim == 2:
            positions = positions.sum(0)
        # Include a final normalization-like scaling inside the cached boundary.
        output = (hidden + hidden.square() / 8 + positions[:, None].to(hidden) / 10) * 1.5
        return BaseNavitOutputWithPast(packed_query_sequence=output, past_key_values=kwargs["past_key_values"])


def _model(dtype=torch.float32, cls=Bagel):
    torch.manual_seed(101)
    model = cls(language_model=_LanguageModel(dtype), vit_model=None, config=_make_bagel_config())
    # Bagel initializes the output projection to zero; nonzero weights expose errors.
    nn.init.normal_(model.llm2vae.weight, std=0.05)
    model.llm2vae.to(dtype)
    return model.eval()


def _cache(value):
    cache = NaiveCache(2)
    for index in range(2):
        cache.key_cache[index] = torch.full((1, 1, 4), value + index)
        cache.value_cache[index] = torch.full((1, 1, 4), value + index + 10)
    return cache


def _args(branches=1, per_request=False, multidim=False):
    # Two requests with unequal latent lengths catch branch/request split mistakes.
    lengths = [4, 3] if per_request else [5]
    text = [0, 3, 4, 6] if per_request else [0, 4]
    vae = [1, 2, 5] if per_request else [1, 2, 3]
    positions = torch.arange(sum(lengths))
    if multidim:
        positions = torch.stack((positions, positions + 1, positions + 2))
    args = dict(
        x_t=torch.arange(48, dtype=torch.float32).reshape(3, 16) / 100,
        timestep=torch.tensor([0.0, 0.7, 0.7]),
        packed_vae_token_indexes=torch.tensor(vae),
        packed_vae_position_ids=torch.tensor([0, 1, 2]),
        packed_text_ids=torch.arange(len(text)) + 1,
        packed_text_indexes=torch.tensor(text),
        packed_position_ids=positions,
        packed_seqlens=torch.tensor(lengths, dtype=torch.int32),
        past_key_values=_cache(1.0),
    )
    if branches > 1:
        args.update(
            cfg_text_scale=2.0,
            cfg_img_scale=1.5,
            cfg_branch_pids=[positions + i * 7 for i in range(branches)],
            cfg_branch_caches=[args["past_key_values"]] + [_cache(float(i + 2)) for i in range(branches - 1)],
        )
        if per_request:
            args.update(cfg_vae_lengths=[2, 1], cfg_text_scales=[1.0, 3.0], cfg_img_scales=[1.0, 2.0])
    return args


def _install(model):
    hook = TeaCacheHook(TeaCacheConfig(transformer_type="Bagel", coefficients=[0, 0, 0, 0, 0]))
    HookRegistry.get_or_create(model).register_hook("teacache", hook)
    return hook


def _separate_branch_output(model, args):
    inputs = {key: value for key, value in args.items() if not key.startswith("cfg_")}
    positions = args.get("cfg_branch_pids", [args["packed_position_ids"]])
    caches = args.get("cfg_branch_caches", [args["past_key_values"]])
    outputs = [
        model.forward_single_branch(**dict(inputs, packed_position_ids=position, past_key_values=cache))
        for position, cache in zip(positions, caches, strict=True)
    ]
    if len(outputs) == 1:
        return outputs[0]
    lengths = args.get("cfg_vae_lengths", [outputs[0].shape[0]])
    text_scales = args.get("cfg_text_scales", [args["cfg_text_scale"]])
    image_scales = args.get("cfg_img_scales", [args["cfg_img_scale"]])
    results = []
    for branches, text_scale, image_scale in zip(
        zip(*(output.split(lengths) for output in outputs), strict=True), text_scales, image_scales, strict=True
    ):
        results.append(
            model._combine_cfg(
                branches[0],
                branches[1],
                branches[2] if len(branches) == 3 else None,
                text_scale,
                image_scale,
                "global",
                0.0,
            )
        )
    return torch.cat(results)


@pytest.mark.parametrize(
    "branches,per_request,multidim,dtype",
    [
        (1, False, False, torch.float32),
        (2, False, False, torch.float32),
        (3, False, True, torch.float32),
        (3, True, False, torch.float32),
        (2, False, False, torch.bfloat16),
    ],
)
@torch.no_grad()
def test_packed_cfg_matches_separate_branches(branches, per_request, multidim, dtype):
    model = _model(dtype)
    args = _args(branches, per_request, multidim)
    expected = _separate_branch_output(model, args)
    reference_calls = list(model.language_model.calls)
    actual = model(**args)
    torch.testing.assert_close(actual, expected)
    call = model.language_model.calls[-1]
    for key in (
        "packed_query_sequence",
        "query_lens",
        "packed_query_position_ids",
        "packed_vae_token_indexes",
        "packed_text_indexes",
    ):
        dim = 1 if key == "packed_query_position_ids" and multidim else 0
        values = [reference[key] for reference in reference_calls]
        if key in ("packed_vae_token_indexes", "packed_text_indexes"):
            values = [value + index * int(args["packed_seqlens"].sum()) for index, value in enumerate(values)]
        torch.testing.assert_close(call[key], torch.cat(values, dim=dim))
    assert call["packed_query_sequence"].dtype == dtype
    assert len(model.language_model.calls) == branches + 1


@torch.no_grad()
def test_per_token_timestep_is_preserved():
    model = _model()
    args = _args()
    actual = model(**args)
    torch.testing.assert_close(actual, model.forward_single_branch(**args))
    assert not torch.equal(actual, model(**dict(args, timestep=torch.full((3,), 0.7))))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_cfg_signal_uses_one_branch_with_equivalent_l1_distance(dtype):
    model = _model(dtype)
    args = _args(3)
    previous = model.preprocess(**args)
    current = model.preprocess(**dict(args, x_t=args["x_t"] + 0.15, timestep=torch.full((3,), 0.4)))
    assert previous.modulated_input is not None
    assert current.modulated_input is not None
    assert previous.modulated_input.shape[0] * 3 == previous.hidden_states.shape[0]
    for branch in previous.hidden_states.chunk(3):
        torch.testing.assert_close(previous.modulated_input, branch)
    signal_distance = (current.modulated_input - previous.modulated_input).abs().mean() / (
        previous.modulated_input.abs().mean() + 1e-8
    )
    repeated_distance = (current.hidden_states - previous.hidden_states).abs().mean() / (
        previous.hidden_states.abs().mean() + 1e-8
    )
    torch.testing.assert_close(signal_distance, repeated_distance)
    assert model.preprocess(**args, skip_modulated_input=True).modulated_input is None


@pytest.mark.parametrize("branches", [1, 2, 3])
@torch.no_grad()
def test_cache_hit_uses_current_input_and_preserves_prefix_cache(branches):
    model = _model()
    args = _args(branches)
    caches = args.get("cfg_branch_caches", [args["past_key_values"]])
    snapshots = [
        (c, {i: t.clone() for i, t in c.key_cache.items()}, {i: t.clone() for i, t in c.value_cache.items()})
        for c in caches
    ]
    initial = model.preprocess(**args)
    computed = model.run_transformer_blocks(initial)
    # Preprocessing again avoids relying on whether run_blocks mutates its state.
    residual = computed.hidden_states - model.preprocess(**args).hidden_states
    model.language_model.calls.clear()
    hook = _install(model)
    first = model(**args)
    changed = dict(args, x_t=args["x_t"] + 0.15, timestep=torch.full((3,), 0.4))
    expected_ctx = model.preprocess(**changed)
    expected_ctx.hidden_states = expected_ctx.hidden_states + residual
    expected = model.postprocess(expected_ctx)
    with patch.object(NaiveCache, "merge", wraps=NaiveCache.merge) as merge:
        actual = model(**changed)
        merge.assert_not_called()
    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual, first)
    assert len(model.language_model.calls) == 1
    assert hook._forward_cnt == 2
    if branches > 1:
        pieces = residual.chunk(branches)
        assert not torch.equal(pieces[0], pieces[1])
    for cache, keys, values in snapshots:
        for index in keys:
            torch.testing.assert_close(cache.key_cache[index], keys[index])
            torch.testing.assert_close(cache.value_cache[index], values[index])
        assert cache.key_values_lens is None


@torch.no_grad()
def test_cfg_interval_bypass_does_not_replace_batched_residual():
    model = _model()
    args = _args(3)
    hook = _install(model)
    first = model(**args)
    bypass = dict(args, cfg_text_scale=1.0, cfg_img_scale=1.0, cfg_vae_lengths=[3])
    assert model.preprocess(**bypass).modulated_input is None
    model(**bypass)
    again = model(**args)
    torch.testing.assert_close(again, first)
    assert len(model.language_model.calls) == 2
    assert hook._forward_cnt == 2


@pytest.mark.parametrize("cache_enabled", [False, True])
@torch.no_grad()
def test_invalid_cfg_scales_fail_before_computation_and_preserve_cache(cache_enabled):
    model = _model()
    args = _args(2)
    hook = _install(model) if cache_enabled else None
    expected = model(**args)

    with patch.object(model.language_model, "forward", wraps=model.language_model.forward) as forward:
        with pytest.raises(ValueError, match="cfg_text_scales must be provided with cfg_vae_lengths"):
            model(**dict(args, cfg_vae_lengths=[3]))
        forward.assert_not_called()

    assert len(model.language_model.calls) == 1
    if hook is not None:
        assert hook._forward_cnt == 1
        torch.testing.assert_close(model(**args), expected)
        assert hook._forward_cnt == 2
        assert len(model.language_model.calls) == 1


@torch.no_grad()
def test_generate_image_calls_installed_hook_and_reset_recomputes(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.bagel.bagel_transformer.get_classifier_free_guidance_world_size", lambda: 1
    )
    model = _model()
    hook = _install(model)
    args = _args()
    args["packed_init_noises"] = args.pop("x_t")
    args.pop("timestep")
    first = model.generate_image(**args, num_timesteps=4)[0]
    assert hook._forward_cnt == 3
    assert len(model.language_model.calls) == 1
    HookRegistry.get_or_create(model).reset_hook("teacache")
    assert hook._forward_cnt == 0
    second = model.generate_image(**args, num_timesteps=4)[0]
    assert hook._forward_cnt == 3
    assert len(model.language_model.calls) == 2
    torch.testing.assert_close(first[0], second[0])


@torch.no_grad()
def test_bagel_adapter_collects_packed_signal_and_decoder_output():
    model = _model()
    args = _args(3)
    expected_signal = model.preprocess(**_args()).hidden_states.clone()
    expected_output = model.run_transformer_blocks(model.preprocess(**args)).hidden_states
    transformer, name = BagelAdapter.get_transformer(SimpleNamespace(bagel=model))
    collector = DataCollectionHook(name)
    BagelAdapter.install_hook(transformer, collector)
    model(**args)
    model(**dict(args, cfg_text_scale=1.0))
    collected = collector.stop_collection()
    assert len(collected) == 1
    signal, output = collected[0]
    assert output.shape[0] == 3 * signal.shape[0]
    np.testing.assert_allclose(signal, expected_signal.numpy())
    np.testing.assert_allclose(output, expected_output.numpy())


@torch.no_grad()
def test_lance_inherits_forward_with_per_token_timestep():
    assert LanceBagel.forward is Bagel.forward
    model = _model(cls=LanceBagel)
    args = _args(2, multidim=True)
    expected = _separate_branch_output(model, args)
    _install(model)
    torch.testing.assert_close(model(**args), expected)
    torch.testing.assert_close(model(**args), expected)


class _CPUAttention(nn.Module):
    """CPU attention for testing decoder routing without the MoT Triton GEMMs."""

    def forward(self, packed_query_sequence, query_lens, past_key_values, **kwargs):
        outputs = []
        for sequence in packed_query_sequence.split(query_lens.tolist()):
            sequence = sequence.unsqueeze(0)
            outputs.append(torch.nn.functional.scaled_dot_product_attention(sequence, sequence, sequence).squeeze(0))
        return torch.cat(outputs), past_key_values


@torch.no_grad()
def test_real_decoder_forward_and_final_norm_remain_inside_cache_boundary(monkeypatch):
    # Keep real decoder/model forwards, normalization and RoPE. Substitute CPU
    # attention and dense layers for the production Triton/tensor-parallel ops.
    monkeypatch.setattr(MoTRMSNorm, "forward", MoTRMSNorm.forward_native)
    torch.manual_seed(103)
    layer = Qwen2MoTDecoderLayer.__new__(Qwen2MoTDecoderLayer)
    nn.Module.__init__(layer)
    layer.self_attn = _CPUAttention()
    layer.input_layernorm = MoTRMSNorm(HIDDEN_SIZE)
    layer.post_attention_layernorm = MoTRMSNorm(HIDDEN_SIZE)
    layer.mlp = nn.Sequential(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.SiLU()).to(torch.bfloat16)
    layer.mlp_moe_gen = nn.Sequential(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.SiLU()).to(torch.bfloat16)
    lm = Qwen2MoTModel.__new__(Qwen2MoTModel)
    nn.Module.__init__(lm)
    lm.use_moe = True
    lm.embed_tokens = nn.Embedding(8, HIDDEN_SIZE)
    lm.layers = nn.ModuleList([layer])
    lm.rotary_emb = BagelRotaryEmbedding(_make_bagel_config().llm_config)
    lm.norm = MoTRMSNorm(HIDDEN_SIZE)
    lm.norm.gen_weight.fill_(1.7)
    model = _model()
    model.language_model = lm
    args = _args(2)
    expected = _separate_branch_output(model, args)
    calls = []
    handle = layer.register_forward_hook(lambda *unused: calls.append(1))
    _install(model)
    first = model(**args)
    second = model(**args)
    handle.remove()
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)
    assert calls == [1]
