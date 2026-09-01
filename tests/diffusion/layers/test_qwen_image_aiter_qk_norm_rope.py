# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F
from vllm._aiter_ops import rocm_aiter_ops

from vllm_omni.diffusion.layers import fused_qk_norm_rope as fused_qk_norm_rope_module
from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
    prepare_rocm_aiter_fused_qk_norm_rope_2way_cache,
    rocm_aiter_fused_qk_norm_rope_2way,
    rocm_aiter_fused_qk_norm_rope_2way_supported,
)
from vllm_omni.diffusion.layers.rope import apply_rotary_emb_torch
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
]


def test_rocm_aiter_fused_qk_norm_rope_2way_rejects_missing_capability(monkeypatch) -> None:
    monkeypatch.setattr(
        fused_qk_norm_rope_module,
        "_ROCM_AITER_FUSED_QK_NORM_ROPE_2WAY_AVAILABLE",
        False,
    )
    inputs = (torch.empty(1),) * 10
    assert not rocm_aiter_fused_qk_norm_rope_2way_supported(*inputs)


def test_rocm_aiter_fused_qk_norm_rope_2way_rejects_sequence_parallel(monkeypatch) -> None:
    monkeypatch.setattr(
        fused_qk_norm_rope_module,
        "_ROCM_AITER_FUSED_QK_NORM_ROPE_2WAY_AVAILABLE",
        True,
    )
    inputs = (torch.empty(1),) * 10
    assert not rocm_aiter_fused_qk_norm_rope_2way_supported(*inputs, sequence_parallel_size=2)


def test_rocm_aiter_fused_qk_norm_rope_2way_availability_respects_aiter_toggle(monkeypatch) -> None:
    monkeypatch.setattr(fused_qk_norm_rope_module.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: False)
    assert not fused_qk_norm_rope_module._rocm_aiter_fused_qk_norm_rope_2way_available()


def test_prepare_rocm_aiter_fused_qk_norm_rope_2way_cache(monkeypatch) -> None:
    monkeypatch.setattr(
        fused_qk_norm_rope_module,
        "_ROCM_AITER_FUSED_QK_NORM_ROPE_2WAY_AVAILABLE",
        True,
    )
    text_freqs = torch.randn(5, 64, dtype=torch.complex64)
    image_freqs = torch.randn(13, 64, dtype=torch.complex64)

    caches = prepare_rocm_aiter_fused_qk_norm_rope_2way_cache(
        text_freqs,
        image_freqs,
        torch.bfloat16,
    )

    assert caches is not None
    torch.testing.assert_close(
        caches[0],
        torch.cat(
            (text_freqs.real.to(torch.bfloat16), text_freqs.imag.to(torch.bfloat16)),
            dim=-1,
        ),
    )
    torch.testing.assert_close(
        caches[1],
        torch.cat(
            (
                image_freqs.real.to(torch.bfloat16),
                image_freqs.imag.to(torch.bfloat16),
            ),
            dim=-1,
        ),
    )
    assert caches[0].is_contiguous()
    assert caches[1].is_contiguous()


@pytest.mark.parametrize(
    ("available", "dtype", "sequence_parallel_size"),
    [
        (False, torch.bfloat16, 1),
        (True, torch.float16, 1),
        (True, torch.bfloat16, 2),
    ],
)
def test_prepare_rocm_aiter_fused_qk_norm_rope_2way_cache_rejects_unsupported_path(
    monkeypatch,
    available: bool,
    dtype: torch.dtype,
    sequence_parallel_size: int,
) -> None:
    monkeypatch.setattr(
        fused_qk_norm_rope_module,
        "_ROCM_AITER_FUSED_QK_NORM_ROPE_2WAY_AVAILABLE",
        available,
    )
    freqs = torch.empty(1, dtype=torch.complex64)

    assert (
        prepare_rocm_aiter_fused_qk_norm_rope_2way_cache(
            freqs,
            freqs,
            dtype,
            sequence_parallel_size,
        )
        is None
    )


@pytest.mark.rocm
@pytest.mark.cards_1
@pytest.mark.skipif(not current_omni_platform.is_rocm(), reason="ROCm required")
@pytest.mark.parametrize("text_tokens", [5, 13])
def test_rocm_aiter_fused_qk_norm_rope_2way_matches_qwen_reference(text_tokens: int) -> None:
    torch.manual_seed(17)
    batch_size = 1
    image_tokens = 1024
    heads = 24
    head_dim = 128
    hidden_dim = heads * head_dim
    eps = 1e-6
    device = "cuda"
    dtype = torch.bfloat16

    def make_stream(tokens: int):
        qkv = torch.randn(
            batch_size,
            tokens,
            3 * hidden_dim,
            device=device,
            dtype=dtype,
        )
        q, k, v = qkv.split(hidden_dim, dim=-1)
        return (
            q.unflatten(-1, (heads, head_dim)),
            k.unflatten(-1, (heads, head_dim)),
            v.unflatten(-1, (heads, head_dim)),
        )

    text_q, text_k, text_v = make_stream(text_tokens)
    image_q, image_k, image_v = make_stream(image_tokens)
    text_weights = (
        torch.randn(head_dim, device=device, dtype=dtype),
        torch.randn(head_dim, device=device, dtype=dtype),
    )
    image_weights = (
        torch.randn(head_dim, device=device, dtype=dtype),
        torch.randn(head_dim, device=device, dtype=dtype),
    )

    def make_cos_sin(tokens: int):
        angles = torch.randn(tokens, head_dim // 2, device=device, dtype=torch.float32)
        return angles.cos().to(dtype), angles.sin().to(dtype)

    text_cos, text_sin = make_cos_sin(text_tokens)
    image_cos, image_sin = make_cos_sin(image_tokens)
    text_cache = torch.cat((text_cos, text_sin), dim=-1)
    image_cache = torch.cat((image_cos, image_sin), dim=-1)
    inputs = (
        text_q,
        text_k,
        image_q,
        image_k,
        text_weights[0],
        text_weights[1],
        image_weights[0],
        image_weights[1],
        text_cache,
        image_cache,
    )
    if not rocm_aiter_fused_qk_norm_rope_2way_supported(*inputs):
        pytest.skip("Installed AITER does not support strided two-stream Q/K inputs")

    text_q_before, text_k_before, text_v_before = text_q.clone(), text_k.clone(), text_v.clone()
    image_q_before, image_k_before, image_v_before = image_q.clone(), image_k.clone(), image_v.clone()

    def reference(q, k, q_weight, k_weight, cos, sin):
        q = F.rms_norm(q, (head_dim,), q_weight, eps)
        k = F.rms_norm(k, (head_dim,), k_weight, eps)
        return (
            apply_rotary_emb_torch(q, cos, sin, interleaved=True),
            apply_rotary_emb_torch(k, cos, sin, interleaved=True),
        )

    text_q_ref, text_k_ref = reference(text_q, text_k, text_weights[0], text_weights[1], text_cos, text_sin)
    image_q_ref, image_k_ref = reference(
        image_q,
        image_k,
        image_weights[0],
        image_weights[1],
        image_cos,
        image_sin,
    )
    q_ref = torch.cat((text_q_ref, image_q_ref), dim=1)
    k_ref = torch.cat((text_k_ref, image_k_ref), dim=1)

    q_out, k_out = rocm_aiter_fused_qk_norm_rope_2way(*inputs, eps)

    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=0.05)
    torch.testing.assert_close(k_out, k_ref, rtol=1e-2, atol=0.05)
    assert torch.equal(text_q, text_q_before)
    assert torch.equal(text_k, text_k_before)
    assert torch.equal(text_v, text_v_before)
    assert torch.equal(image_q, image_q_before)
    assert torch.equal(image_k, image_k_before)
    assert torch.equal(image_v, image_v_before)
