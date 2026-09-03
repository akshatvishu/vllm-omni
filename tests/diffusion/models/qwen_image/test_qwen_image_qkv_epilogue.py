# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.diffusion.layers.rope import RotaryEmbedding
from vllm_omni.diffusion.models.qwen_image import qwen_image_transformer
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenImageCrossAttention,
    _prepare_qwen_rope_table,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _StaticLinear(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output

    def forward(self, _input: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.output, None


class _QueryAttention(nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        _key: torch.Tensor,
        _value: torch.Tensor,
        _metadata: Any,
    ) -> torch.Tensor:
        return query


def _make_attention(img_qkv: torch.Tensor, txt_qkv: torch.Tensor) -> QwenImageCrossAttention:
    attention = QwenImageCrossAttention.__new__(QwenImageCrossAttention)
    nn.Module.__init__(attention)
    attention.head_dim = 4
    attention.eps = 1e-6
    attention.query_num_heads = 2
    attention.kv_num_heads = 2
    attention.add_query_num_heads = 2
    attention.add_kv_num_heads = 2
    attention.to_qkv = _StaticLinear(img_qkv)
    attention.add_kv_proj = _StaticLinear(txt_qkv)
    attention.norm_q = nn.RMSNorm(4, eps=1e-6)
    attention.norm_k = nn.RMSNorm(4, eps=1e-6)
    attention.norm_added_q = nn.RMSNorm(4, eps=1e-6)
    attention.norm_added_k = nn.RMSNorm(4, eps=1e-6)
    attention.rope = RotaryEmbedding(is_neox_style=False)
    attention.rope._forward_method = attention.rope.forward_native
    attention.attn = _QueryAttention()
    attention.to_out = nn.Identity()
    attention.to_add_out = nn.Identity()
    attention.parallel_config = None
    attention.use_rocm_aiter_qkv_epilogue = False
    return attention


def test_qwen_cross_attention_fallback_matches_current_path():
    torch.manual_seed(1)
    img_qkv = torch.randn(1, 5, 24)
    txt_qkv = torch.randn(1, 3, 24)
    img_freqs = torch.polar(torch.ones(5, 2), torch.randn(5, 2))
    txt_freqs = torch.polar(torch.ones(3, 2), torch.randn(3, 2))
    attention = _make_attention(img_qkv, txt_qkv)

    img_output, txt_output = attention(
        torch.empty(1, 5, 8),
        torch.empty(1, 3, 8),
        img_freqs,
        txt_freqs,
    )

    img_query = img_qkv[..., :8].unflatten(-1, (2, 4))
    txt_query = txt_qkv[..., :8].unflatten(-1, (2, 4))
    img_query = F.rms_norm(img_query, (4,), attention.norm_q.weight, 1e-6)
    txt_query = F.rms_norm(txt_query, (4,), attention.norm_added_q.weight, 1e-6)
    img_expected = attention.rope.forward_native(
        img_query,
        torch.real(img_freqs),
        torch.imag(img_freqs),
    )
    txt_expected = attention.rope.forward_native(
        txt_query,
        torch.real(txt_freqs),
        torch.imag(txt_freqs),
    )

    assert torch.equal(img_output, img_expected.flatten(2, 3))
    assert torch.equal(txt_output, txt_expected.flatten(2, 3))


def test_qwen_cross_attention_fused_rejection_matches_fallback(monkeypatch):
    torch.manual_seed(3)
    img_qkv = torch.randn(1, 5, 24)
    txt_qkv = torch.randn(1, 3, 24)
    img_freqs = torch.polar(torch.ones(5, 2), torch.randn(5, 2))
    txt_freqs = torch.polar(torch.ones(3, 2), torch.randn(3, 2))
    attention = _make_attention(img_qkv, txt_qkv)
    attention.use_rocm_aiter_qkv_epilogue = True

    expected = attention(
        torch.empty(1, 5, 8),
        torch.empty(1, 3, 8),
        img_freqs,
        txt_freqs,
    )

    monkeypatch.setattr(
        qwen_image_transformer,
        "try_rocm_aiter_fused_qkv_norm_rope",
        lambda *args, **kwargs: None,
    )
    actual = attention(
        torch.empty(1, 5, 8),
        torch.empty(1, 3, 8),
        img_freqs,
        txt_freqs,
        qkv_rope_tables=(torch.empty(5, 4), torch.empty(3, 4)),
    )

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_qwen_cross_attention_passes_packed_projections_without_copy(monkeypatch):
    img_qkv = torch.randn(1, 5, 24)
    txt_qkv = torch.randn(1, 3, 24)
    attention = _make_attention(img_qkv, txt_qkv)
    attention.use_rocm_aiter_qkv_epilogue = True
    joint_query = torch.randn(1, 8, 2, 4)
    joint_key = torch.randn_like(joint_query)
    joint_value = torch.randn_like(joint_query)

    def fake_fused(received_img: torch.Tensor, received_txt: torch.Tensor, *args, **kwargs):
        assert received_img.data_ptr() == img_qkv.data_ptr()
        assert received_txt.data_ptr() == txt_qkv.data_ptr()
        assert kwargs["enabled"] is True
        return joint_query, joint_key, joint_value

    monkeypatch.setattr(
        qwen_image_transformer,
        "try_rocm_aiter_fused_qkv_norm_rope",
        fake_fused,
    )
    img_output, txt_output = attention(
        torch.empty(1, 5, 8),
        torch.empty(1, 3, 8),
        torch.empty(5, 2, dtype=torch.complex64),
        torch.empty(3, 2, dtype=torch.complex64),
        qkv_rope_tables=(torch.empty(5, 4), torch.empty(3, 4)),
    )

    assert torch.equal(txt_output, joint_query[:, :3].flatten(2, 3))
    assert torch.equal(img_output, joint_query[:, 3:].flatten(2, 3))


def test_prepare_qwen_rope_table_matches_current_conversion():
    torch.manual_seed(2)
    freqs = torch.polar(torch.ones(5, 2), torch.randn(5, 2))

    table = _prepare_qwen_rope_table(freqs, torch.bfloat16)

    expected = torch.cat(
        (torch.real(freqs).to(torch.bfloat16), torch.imag(freqs).to(torch.bfloat16)),
        dim=-1,
    )
    assert torch.equal(table, expected)
    assert table.is_contiguous()


@pytest.mark.parametrize(
    ("hidden_states_mask", "encoder_hidden_states_mask"),
    [
        (torch.ones(1, 5, dtype=torch.bool), None),
        (None, torch.ones(1, 3, dtype=torch.bool)),
    ],
)
def test_qwen_cross_attention_masks_disable_fused_path(
    monkeypatch,
    hidden_states_mask,
    encoder_hidden_states_mask,
):
    img_qkv = torch.randn(1, 5, 24)
    txt_qkv = torch.randn(1, 3, 24)
    attention = _make_attention(img_qkv, txt_qkv)
    attention.use_rocm_aiter_qkv_epilogue = True

    def fake_fused(*args, **kwargs):
        assert kwargs["enabled"] is False
        return None

    monkeypatch.setattr(
        qwen_image_transformer,
        "try_rocm_aiter_fused_qkv_norm_rope",
        fake_fused,
    )
    img_output, txt_output = attention(
        torch.empty(1, 5, 8),
        torch.empty(1, 3, 8),
        torch.empty(5, 2, dtype=torch.complex64),
        torch.empty(3, 2, dtype=torch.complex64),
        hidden_states_mask=hidden_states_mask,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        qkv_rope_tables=(torch.empty(5, 4), torch.empty(3, 4)),
    )

    assert img_output.shape == (1, 5, 8)
    assert txt_output.shape == (1, 3, 8)
