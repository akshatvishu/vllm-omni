# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm import forward_context
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor import parameter as vllm_parameter
from vllm.model_executor.layers import linear as vllm_linear

import vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer as hunyuan
import vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 as hunyuan_pipeline
from vllm_omni.diffusion.cache.teacache.backend import TeaCacheBackend
from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.interface import supports_teacache
from vllm_omni.diffusion.cache.teacache.runtime import TeaCacheRuntime
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, DiffusionCacheConfig, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_TEST_COEFFICIENTS = (0.0, 0.0, 0.0, 1.0, 0.0)


class CountingLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, *, hidden_states, use_cache, output_attentions, **kwargs):
        self.calls += 1
        output = hidden_states + 1.0
        if use_cache:
            return (output, None, None) if output_attentions else (output, None)
        return (output, None) if output_attentions else (output,)


def _make_tiny_model(monkeypatch: pytest.MonkeyPatch) -> tuple[hunyuan.HunyuanImage3Model, list[CountingLayer]]:
    class FakePPGroup:
        is_first_rank = True
        is_last_rank = True

    layers = [CountingLayer(), CountingLayer()]

    def fake_make_layers(num_hidden_layers, layer_fn, prefix):
        del layer_fn, prefix
        assert num_hidden_layers == len(layers)
        return 0, num_hidden_layers, nn.ModuleList(layers)

    def fake_embedding(num_embeddings, embedding_dim, **kwargs):
        del kwargs
        return nn.Embedding(num_embeddings, embedding_dim)

    monkeypatch.setattr(hunyuan, "get_pp_group", lambda: FakePPGroup())
    monkeypatch.setattr(hunyuan, "get_sequence_parallel_world_size", lambda: 1)
    monkeypatch.setattr(hunyuan, "make_layers", fake_make_layers)
    monkeypatch.setattr(hunyuan, "VocabParallelEmbedding", fake_embedding)
    monkeypatch.setattr(hunyuan, "RMSNorm", nn.LayerNorm)

    config = hunyuan.HunyuanImage3Config(
        vocab_size=32,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=len(layers),
        num_attention_heads=1,
        num_key_value_heads=1,
        attention_head_dim=4,
        patch_embed_hidden_dim=4,
        num_experts=1,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
    )
    model = hunyuan.HunyuanImage3Model(config)
    model.tea_cache_executor = TeaCacheRuntime(TeaCacheConfig(coefficients=_TEST_COEFFICIENTS, rel_l1_thresh=0.5))
    return model, layers


def _forward(
    model: hunyuan.HunyuanImage3Model,
    inputs_embeds: torch.Tensor,
    metric: torch.Tensor,
    *,
    first_step: bool,
    mode: str = "gen_image",
    uncond_cfg_prefill: bool = False,
    use_cache: bool = False,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
):
    return model(
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        mode=mode,
        first_step=first_step,
        query_lens=[inputs_embeds.shape[1]],
        seq_lens=[inputs_embeds.shape[1]],
        num_image_tokens=inputs_embeds.shape[1],
        uncond_cfg_prefill=uncond_cfg_prefill,
        tea_cache_modulated_input=metric,
    )


def _make_native_tiny_model(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[hunyuan.HunyuanImage3Model, list[hunyuan.HunyuanImage3DecoderLayer], VllmConfig, OmniDiffusionConfig]:
    class FakePPGroup:
        is_first_rank = True
        is_last_rank = True

    def native_make_layers(num_hidden_layers, layer_fn, prefix):
        layers = nn.ModuleList(layer_fn(f"{prefix}.{index}") for index in range(num_hidden_layers))
        return 0, num_hidden_layers, layers

    def fake_embedding(num_embeddings, embedding_dim, **kwargs):
        del kwargs
        return nn.Embedding(num_embeddings, embedding_dim)

    monkeypatch.setattr(hunyuan, "get_pp_group", lambda: FakePPGroup())
    monkeypatch.setattr(hunyuan, "get_sequence_parallel_rank", lambda: 0)
    monkeypatch.setattr(hunyuan, "get_sequence_parallel_world_size", lambda: 1)
    monkeypatch.setattr(hunyuan, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(hunyuan, "make_layers", native_make_layers)
    monkeypatch.setattr(hunyuan, "VocabParallelEmbedding", fake_embedding)
    monkeypatch.setattr(hunyuan.RotaryEmbedding, "dispatch_forward", lambda self: self.forward_native)
    monkeypatch.setattr(vllm_linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(vllm_linear, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(vllm_parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(vllm_parameter, "get_tensor_model_parallel_world_size", lambda: 1)

    config = hunyuan.HunyuanImage3Config(
        vocab_size=32,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        attention_head_dim=4,
        num_experts=1,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
    )
    od_config = OmniDiffusionConfig(
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA"))
    )
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config), set_current_diffusion_config(od_config):
        model = hunyuan.HunyuanImage3Model(config)
    torch.manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 1:
                parameter.fill_(1.0)
            else:
                parameter.normal_(mean=0.0, std=0.02)
    model = model.to(dtype=torch.bfloat16).eval()
    layers = list(model.layers)
    assert all(isinstance(layer, hunyuan.HunyuanImage3DecoderLayer) for layer in layers)
    return model, layers, vllm_config, od_config


def _native_forward(
    model: hunyuan.HunyuanImage3Model,
    vllm_config: VllmConfig,
    od_config: OmniDiffusionConfig,
    inputs_embeds: torch.Tensor,
    metric: torch.Tensor,
    *,
    first_step: bool,
):
    seq_len = inputs_embeds.shape[1]
    with set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config):
        return model(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool),
            custom_pos_emb=(torch.ones(1, seq_len, 2), torch.zeros(1, seq_len, 2)),
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            mode="gen_image",
            first_step=first_step,
            query_lens=[seq_len],
            seq_lens=[seq_len],
            num_image_tokens=seq_len,
            gen_timestep_scatter_index=torch.zeros(1, 1, dtype=torch.long),
            tea_cache_modulated_input=metric,
        )


def test_tiny_native_hunyuan_layers_skip_stable_later_image_step(monkeypatch: pytest.MonkeyPatch) -> None:
    model, layers, vllm_config, od_config = _make_native_tiny_model(monkeypatch)
    backend = TeaCacheBackend(DiffusionCacheConfig(rel_l1_thresh=0.5, coefficients=list(_TEST_COEFFICIENTS)))
    backend.enable(SimpleNamespace(transformer=model))
    assert len(backend._installed_runtimes) == 1
    calls = 0

    def count_layer_call(*args, **kwargs):
        del args, kwargs
        nonlocal calls
        calls += 1

    hooks = [layer.register_forward_hook(count_layer_call) for layer in layers]
    try:
        with torch.inference_mode():
            first_inputs = torch.randn(1, 3, 4, dtype=torch.bfloat16)
            later_inputs = torch.randn(1, 2, 4, dtype=torch.bfloat16)
            metric = torch.ones(1, 4)
            _native_forward(
                model,
                vllm_config,
                od_config,
                first_inputs,
                metric,
                first_step=True,
            )
            assert calls == 2
            assert model.tea_cache_executor.state.positive.cnt == 0

            computed = _native_forward(
                model,
                vllm_config,
                od_config,
                later_inputs,
                metric,
                first_step=False,
            )
            assert calls == 4
            hit = _native_forward(
                model,
                vllm_config,
                od_config,
                later_inputs,
                metric,
                first_step=False,
            )
            changed_metric = _native_forward(
                model,
                vllm_config,
                od_config,
                later_inputs,
                torch.full_like(metric, 2.0),
                first_step=False,
            )
    finally:
        for hook in hooks:
            hook.remove()

    assert calls == 6
    assert model.tea_cache_executor.state.positive.cnt == 3
    torch.testing.assert_close(computed.last_hidden_state, hit.last_hidden_state)
    assert torch.isfinite(changed_metric.last_hidden_state).all()


def test_tiny_hunyuan_forward_caches_only_stable_later_image_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    model, layers = _make_tiny_model(monkeypatch)
    metric = torch.ones(1, 4)

    _forward(model, torch.zeros(1, 3, 4), metric, first_step=True)
    assert sum(layer.calls for layer in layers) == 2
    assert model.tea_cache_executor.state.positive.cnt == 0

    embeds = torch.zeros(1, 2, 4)
    computed = _forward(model, embeds, metric, first_step=False)
    assert sum(layer.calls for layer in layers) == 4
    hit = _forward(model, embeds, metric, first_step=False)

    assert sum(layer.calls for layer in layers) == 4
    assert model.tea_cache_executor.state.positive.cnt == 2
    assert torch.equal(computed.last_hidden_state, hit.last_hidden_state)


def test_hunyuan_native_backend_installs_and_refreshes_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    model, _ = _make_tiny_model(monkeypatch)
    assert supports_teacache(model) is True

    pipeline = SimpleNamespace(transformer=model)
    backend = TeaCacheBackend(DiffusionCacheConfig(rel_l1_thresh=0.5))
    backend.enable(pipeline)
    assert isinstance(model.tea_cache_executor, TeaCacheRuntime)

    model.tea_cache_executor.state.forward_cnt = 1
    backend.refresh(pipeline, num_inference_steps=4)
    assert model.tea_cache_executor.state.forward_cnt == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"first_step": True},
        {"first_step": False, "mode": "gen_text"},
        {"first_step": False, "uncond_cfg_prefill": True},
        {"first_step": False, "use_cache": True},
        {"first_step": False, "output_attentions": True},
        {"first_step": False, "output_hidden_states": True},
    ],
)
def test_tiny_hunyuan_negative_paths_bypass_teacache(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, object]
) -> None:
    model, layers = _make_tiny_model(monkeypatch)
    _forward(model, torch.zeros(1, 2, 4), torch.ones(1, 4), **kwargs)

    assert sum(layer.calls for layer in layers) == 2
    assert model.tea_cache_executor.state.positive.cnt == 0


def test_tiny_hunyuan_metric_miss_and_shape_change_recompute(monkeypatch: pytest.MonkeyPatch) -> None:
    model, layers = _make_tiny_model(monkeypatch)
    _forward(model, torch.zeros(1, 2, 4), torch.ones(1, 4), first_step=False)
    _forward(model, torch.zeros(1, 2, 4), torch.full((1, 4), 2.0), first_step=False)
    assert sum(layer.calls for layer in layers) == 4

    output = _forward(model, torch.zeros(1, 3, 4), torch.full((1, 4), 2.0), first_step=False)
    assert sum(layer.calls for layer in layers) == 6
    assert output.last_hidden_state.shape == (1, 3, 4)


def test_forward_call_passes_time_conditioned_image_embedding_to_teacache_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestep_embedding = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    image_embedding = torch.tensor([[[5.0, 6.0, 7.0, 8.0]]])
    captured: dict[str, object] = {}

    class FakeOutputs:
        past_key_values = None
        hidden_states = None
        attentions = None

        def __getitem__(self, index):
            assert index == 0
            return torch.zeros(1, 2, 4)

    class FakeModel:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return FakeOutputs()

    def fake_patch_embed(images, t_emb):
        del images
        captured["patch_timestep_embedding"] = t_emb
        return image_embedding, 1, 1

    @contextmanager
    def fake_forward_context(*args, **kwargs):
        del args, kwargs
        yield

    monkeypatch.setattr(forward_context, "set_forward_context", fake_forward_context)
    fake = SimpleNamespace(
        config=SimpleNamespace(hidden_size=4, use_return_dict=True),
        model=FakeModel(),
        vllm_config=None,
        get_pos_emb=lambda custom_pos_emb, position_ids: custom_pos_emb,
        time_embed=lambda timestep: timestep_embedding,
        patch_embed=fake_patch_embed,
        timestep_emb=lambda timestep: torch.zeros(timestep.shape[0], 4),
        ragged_final_layer=lambda *args: torch.zeros(1, 2, 4),
        _check_inputs=hunyuan_pipeline.HunyuanImage3Pipeline._check_inputs,
    )

    hunyuan_pipeline.HunyuanImage3Pipeline.forward_call(
        fake,
        mode="gen_image",
        first_step=False,
        images=torch.zeros(1, 1, 1, 1),
        timestep=torch.ones(1),
        gen_timestep_scatter_index=torch.ones(1, dtype=torch.long),
        query_lens=[1],
        seq_lens=[1],
        num_image_tokens=1,
        use_cache=False,
        return_dict=True,
    )

    assert captured["patch_timestep_embedding"] is timestep_embedding
    assert captured["tea_cache_modulated_input"] is image_embedding
