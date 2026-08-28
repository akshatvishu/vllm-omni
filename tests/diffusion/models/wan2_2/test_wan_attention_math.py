# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

from vllm_omni.diffusion.models.wan2_2 import wan2_2_transformer
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
    WanRMSNorm,
    WanRotaryPosEmbed,
    WanTransformer3DModel,
    _apply_wan_output_norm,
    _apply_wan_rotary_embedding,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _fastvideo_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    normalized = x_float * torch.rsqrt(variance + eps)
    return normalized.to(x.dtype) * weight.to(x.dtype)


def test_wan_rms_norm_matches_fastvideo_cast_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wan2_2_transformer, "get_tensor_model_parallel_world_size", lambda: 1)
    torch.manual_seed(17)
    x = torch.randn(2, 3, 128, dtype=torch.bfloat16)
    weight = torch.randn(128, dtype=torch.bfloat16)
    norm = WanRMSNorm(128, eps=1e-6).to(torch.bfloat16)
    norm.weight.data.copy_(weight)

    actual = norm(x)
    expected = _fastvideo_rms_norm(x, weight, norm.eps)
    old_order = (x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + norm.eps) * weight.float()).to(
        x.dtype
    )

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not torch.equal(actual, old_order)


def test_wan_rms_norm_tp_uses_global_variance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wan2_2_transformer, "get_tensor_model_parallel_world_size", lambda: 2)
    torch.manual_seed(23)
    full_x = torch.randn(2, 3, 16, dtype=torch.bfloat16)
    full_weight = torch.randn(16, dtype=torch.bfloat16)
    local_x = full_x[..., :8]
    remote_sum_sq = full_x[..., 8:].float().pow(2).sum(dim=-1, keepdim=True)
    monkeypatch.setattr(
        wan2_2_transformer,
        "tensor_model_parallel_all_reduce",
        lambda local_sum_sq: local_sum_sq + remote_sum_sq,
    )
    norm = WanRMSNorm(8, eps=1e-6).to(torch.bfloat16)
    norm.weight.data.copy_(full_weight[:8])

    actual = norm(local_x)
    expected = _fastvideo_rms_norm(full_x, full_weight, norm.eps)[..., :8]

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_wan_rope_fp32_path_matches_fastvideo() -> None:
    torch.manual_seed(29)
    head_dim = 128
    hidden_states = torch.randn(1, 16, 2, 8, 8, dtype=torch.bfloat16)
    rope = WanRotaryPosEmbed(
        attention_head_dim=head_dim,
        patch_size=(1, 2, 2),
        max_seq_len=16,
    )
    full_cos, full_sin = rope(hidden_states)
    cos = full_cos[0, :, 0, 0::2]
    sin = full_sin[0, :, 0, 1::2]
    query = torch.randn(1, cos.shape[0], 4, head_dim, dtype=torch.bfloat16)

    actual = ApplyRotaryEmb.forward_static(
        query,
        cos,
        sin,
        is_neox_style=False,
        enable_fp32_compute=True,
    )
    query_real, query_imag = query.float().reshape(*query.shape[:-1], -1, 2).unbind(-1)
    rotated = torch.stack((-query_imag, query_real), dim=-1).flatten(-2)
    expected = (query.float() * full_cos + rotated * full_sin).to(query.dtype)

    assert cos.dtype == torch.float32
    assert sin.dtype == torch.float32
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("is_rocm", [False, True])
def test_wan_rope_selects_native_path_only_on_rocm(monkeypatch: pytest.MonkeyPatch, is_rocm: bool) -> None:
    class FakeRotaryEmbedding:
        def __init__(self) -> None:
            self.path: str | None = None

        def __call__(self, x, cos, sin):
            self.path = "dispatch"
            return x + 1

        def forward_native(self, x, cos, sin):
            self.path = "native"
            return x + 2

    monkeypatch.setattr(wan2_2_transformer.current_omni_platform, "is_rocm", lambda: is_rocm)
    rotary_embedding = FakeRotaryEmbedding()
    x = torch.zeros(1)

    output = _apply_wan_rotary_embedding(rotary_embedding, x, torch.empty(0), torch.empty(0))

    expected_path = "native" if is_rocm else "dispatch"
    expected_output = x + (2 if is_rocm else 1)
    assert rotary_embedding.path == expected_path
    torch.testing.assert_close(output, expected_output)


def test_wan_patch_embedding_disables_linear_rewrite(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeConv3dLayer(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.linear_was_enabled = True
            self.enable_linear = True

    class StubModule(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    monkeypatch.setattr(wan2_2_transformer, "get_pipeline_parallel_world_size", lambda: 1)
    monkeypatch.setattr(wan2_2_transformer, "is_pipeline_first_stage", lambda: True)
    monkeypatch.setattr(wan2_2_transformer, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(wan2_2_transformer, "Conv3dLayer", FakeConv3dLayer)
    monkeypatch.setattr(wan2_2_transformer, "WanRotaryPosEmbed", StubModule)
    monkeypatch.setattr(wan2_2_transformer, "WanTimeTextImageEmbedding", StubModule)
    monkeypatch.setattr(wan2_2_transformer, "AdaLayerNorm", StubModule)
    monkeypatch.setattr(wan2_2_transformer, "TimestepProjPrepare", StubModule)
    monkeypatch.setattr(wan2_2_transformer, "OutputScaleShiftPrepare", StubModule)
    monkeypatch.setattr(
        wan2_2_transformer,
        "make_layers",
        lambda *args, **kwargs: (0, 0, torch.nn.ModuleList()),
    )
    monkeypatch.setattr(
        wan2_2_transformer,
        "make_empty_intermediate_tensors_factory",
        lambda *args, **kwargs: object(),
    )

    model = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=1,
        attention_head_dim=4,
        in_channels=2,
        out_channels=2,
        text_dim=4,
        freq_dim=4,
        ffn_dim=8,
        num_layers=0,
        rope_max_seq_len=8,
    )

    assert model.patch_embedding.linear_was_enabled
    assert not model.patch_embedding.enable_linear


@pytest.mark.parametrize("per_token_temb", [False, True])
def test_wan_self_attention_modulation_matches_fastvideo_fp32_order(per_token_temb: bool) -> None:
    class CaptureSelfAttention(torch.nn.Module):
        def forward(self, hidden_states, rotary_emb, attn_metadata):
            self.input = hidden_states.detach().clone()
            return torch.zeros_like(hidden_states)

    class ZeroCrossAttention(torch.nn.Module):
        def forward(self, hidden_states, encoder_hidden_states, attn_metadata):
            return torch.zeros_like(hidden_states)

    class ZeroAdaLayerNorm(torch.nn.Module):
        def forward(self, hidden_states, scale, shift):
            return torch.zeros_like(hidden_states)

    torch.manual_seed(31)
    batch_size, seq_len, dim = 1, 3, 8
    hidden_states = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(batch_size, 2, dim, dtype=torch.bfloat16)
    scale_shift_table = torch.randn(1, 6, dim, dtype=torch.bfloat16)
    if per_token_temb:
        temb = torch.randn(batch_size, seq_len, 6, dim, dtype=torch.bfloat16)
        modulation = scale_shift_table.unsqueeze(0) + temb.float()
        old_modulation = scale_shift_table.unsqueeze(0) + temb
        chunk_dim = 2
    else:
        temb = torch.randn(batch_size, 6, dim, dtype=torch.bfloat16)
        modulation = scale_shift_table + temb.float()
        old_modulation = scale_shift_table + temb
        chunk_dim = 1

    block = wan2_2_transformer.WanTransformerBlock.__new__(wan2_2_transformer.WanTransformerBlock)
    torch.nn.Module.__init__(block)
    block.norm1 = wan2_2_transformer.AdaLayerNorm(dim, elementwise_affine=False, eps=1e-6)
    block.attn1 = CaptureSelfAttention()
    block.attn2 = ZeroCrossAttention()
    block.norm2 = torch.nn.Identity()
    block.norm3 = ZeroAdaLayerNorm()
    block.ffn = torch.nn.Identity()
    block.scale_shift_table = torch.nn.Parameter(scale_shift_table)

    block(
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb=(torch.empty(0), torch.empty(0)),
    )

    shift_msa, scale_msa = modulation.chunk(6, dim=chunk_dim)[:2]
    old_shift_msa, old_scale_msa = old_modulation.chunk(6, dim=chunk_dim)[:2]
    if per_token_temb:
        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)
        old_shift_msa = old_shift_msa.squeeze(2)
        old_scale_msa = old_scale_msa.squeeze(2)
    normalized = F.layer_norm(hidden_states.float(), (dim,), eps=1e-6)
    expected = (normalized * (1 + scale_msa) + shift_msa).to(hidden_states.dtype)
    old_order = (normalized.to(hidden_states.dtype) * (1 + old_scale_msa) + old_shift_msa).to(hidden_states.dtype)

    assert modulation.dtype == torch.float32
    torch.testing.assert_close(block.attn1.input, expected, atol=0, rtol=0)
    assert not torch.equal(block.attn1.input, old_order)


def test_wan_self_attention_residual_norm_matches_fastvideo_fp32_order() -> None:
    class FixedSelfAttention(torch.nn.Module):
        def __init__(self, output: torch.Tensor) -> None:
            super().__init__()
            self.output = output

        def forward(self, hidden_states, rotary_emb, attn_metadata):
            return self.output

    class CaptureCrossAttention(torch.nn.Module):
        def forward(self, hidden_states, encoder_hidden_states, attn_metadata):
            self.input = hidden_states.detach().clone()
            return torch.zeros_like(hidden_states)

    class ZeroAdaLayerNorm(torch.nn.Module):
        def forward(self, hidden_states, scale, shift):
            return torch.zeros_like(hidden_states)

    torch.manual_seed(37)
    batch_size, seq_len, dim = 1, 3, 8
    hidden_states = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16)
    self_attn_output = torch.randn_like(hidden_states)
    encoder_hidden_states = torch.randn(batch_size, 2, dim, dtype=torch.bfloat16)
    scale_shift_table = torch.randn(1, 6, dim, dtype=torch.bfloat16)
    temb = torch.randn(batch_size, 6, dim, dtype=torch.bfloat16)

    block = wan2_2_transformer.WanTransformerBlock.__new__(wan2_2_transformer.WanTransformerBlock)
    torch.nn.Module.__init__(block)
    block.norm1 = ZeroAdaLayerNorm()
    block.attn1 = FixedSelfAttention(self_attn_output)
    block.attn2 = CaptureCrossAttention()
    block.norm2 = wan2_2_transformer.LayerNorm(dim, eps=1e-6, elementwise_affine=True).to(torch.bfloat16)
    block.norm3 = ZeroAdaLayerNorm()
    block.ffn = torch.nn.Identity()
    block.scale_shift_table = torch.nn.Parameter(scale_shift_table)

    actual = block(
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb=(torch.empty(0), torch.empty(0)),
    )

    gate_msa = (scale_shift_table + temb.float()).chunk(6, dim=1)[2]
    residual_fp32 = hidden_states + self_attn_output * gate_msa
    expected_cross_input = F.layer_norm(
        residual_fp32,
        (dim,),
        block.norm2.weight.float(),
        block.norm2.bias.float(),
        block.norm2.eps,
    ).to(hidden_states.dtype)
    expected_residual = residual_fp32.to(hidden_states.dtype)
    old_cross_input = F.layer_norm(
        expected_residual.float(),
        (dim,),
        block.norm2.weight.float(),
        block.norm2.bias.float(),
        block.norm2.eps,
    ).to(hidden_states.dtype)

    torch.testing.assert_close(block.attn2.input, expected_cross_input, atol=0, rtol=0)
    torch.testing.assert_close(actual, expected_residual, atol=0, rtol=0)
    assert not torch.equal(block.attn2.input, old_cross_input)


def test_wan_output_norm_matches_fastvideo_fp32_modulation_order() -> None:
    torch.manual_seed(41)
    batch_size, seq_len, dim = 1, 3, 8
    hidden_states = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16)
    scale = torch.randn(batch_size, 1, dim, dtype=torch.bfloat16)
    shift = torch.randn(batch_size, 1, dim, dtype=torch.bfloat16)
    norm = wan2_2_transformer.AdaLayerNorm(dim, elementwise_affine=False, eps=1e-6)

    actual = _apply_wan_output_norm(norm, hidden_states, scale, shift)

    normalized_fp32 = F.layer_norm(hidden_states.float(), (dim,), eps=norm.eps)
    normalized_bf16 = normalized_fp32.to(hidden_states.dtype)
    expected = (normalized_bf16.float() * (1 + scale) + shift).to(hidden_states.dtype)
    old_order = (normalized_bf16 * (1 + scale) + shift).to(hidden_states.dtype)
    no_intermediate_rounding = (normalized_fp32 * (1 + scale) + shift).to(hidden_states.dtype)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not torch.equal(actual, old_order)
    assert not torch.equal(actual, no_intermediate_rounding)
