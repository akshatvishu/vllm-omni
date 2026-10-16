# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_omni.diffusion.attention import layer as layer_mod
from vllm_omni.diffusion.attention.backends import flash_attn
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.utils import fa

fp8_op = importlib.import_module("vllm_omni.diffusion.attention.ops.rocm_fp8_attention")
pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def impl(monkeypatch):
    monkeypatch.setattr(flash_attn, "rocm_fp8_attention_available", lambda: True)
    return flash_attn.FlashAttentionImpl(24, 128, 0.125)


@pytest.mark.parametrize(
    "reason",
    [
        "default",
        "float",
        "unavailable",
        "mask",
        "piecewise",
        "causal",
        "cross",
        "deterministic",
        "short",
        "head_size",
        "empty_batch",
        "empty_heads",
        "dtype",
        "strided",
        "shape",
        "cpu",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "max_seqlen_q",
        "max_seqlen_k",
    ],
)
def test_unsupported_calls_keep_native_attention(impl, monkeypatch, reason):
    fp8 = Mock(side_effect=AssertionError("Fallback must not quantize"))
    native = Mock(return_value=object())
    monkeypatch.setattr(flash_attn, "rocm_fp8_attention", fp8)
    monkeypatch.setattr(impl, "forward_cuda", native)
    with FakeTensorMode():
        shape = [1, 129, 24, 128]
        if reason == "short":
            shape[1] = 128
        elif reason == "head_size":
            shape[3] = 64
        elif reason == "empty_batch":
            shape[0] = 0
        elif reason == "empty_heads":
            shape[2] = 0
        q = torch.empty(shape, device="cpu" if reason == "cpu" else "cuda", dtype=torch.bfloat16)
        k, v = q, q
        metadata = AttentionMetadata(extra={"kv_cache_dtype": "fp8"})
        if reason == "default":
            metadata = None
        elif reason == "float":
            metadata.extra["kv_cache_dtype"] = "float"
        elif reason == "unavailable":
            impl._rocm_fp8_available = False
        elif reason == "mask":
            metadata.attn_mask = torch.ones((1, 129), dtype=torch.bool, device=q.device)
        elif reason == "piecewise":
            metadata.full_attn_spans = []
        elif reason == "causal":
            impl.causal = True
        elif reason == "cross":
            impl.is_cross_attn = True
        elif reason == "deterministic":
            impl.fa_deterministic = True
        elif reason == "dtype":
            k = k.float()
        elif reason == "strided":
            v = torch.empty((1, 24, 129, 128), dtype=q.dtype, device=q.device).transpose(1, 2)
        elif reason == "shape":
            k = k[:, :-1]
        elif reason in ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k"):
            metadata.extra[reason] = None
        assert impl.forward_hip(q, k, v, metadata) is native.return_value
        native.assert_called_once_with(q, k, v, metadata)
        fp8.assert_not_called()


def test_supported_call_and_fake_output(impl, monkeypatch):
    native = Mock(side_effect=AssertionError("Eligible FP8 must not fall back"))
    monkeypatch.setattr(impl, "forward_cuda", native)
    with FakeTensorMode():
        q = torch.empty((1, 4117, 24, 128), dtype=torch.bfloat16, device="cuda")
        result = impl.forward_hip(q, q, q, AttentionMetadata(extra={"kv_cache_dtype": "fp8"}))
        assert result.shape == q.shape and result.dtype == q.dtype and result.device == q.device
        assert result.is_contiguous() and result is not q
    native.assert_not_called()


def test_partial_packed_metadata_still_raises(impl, monkeypatch):
    monkeypatch.setattr(fa, "HAS_FLASH_ATTN", True)
    q = torch.empty((1, 129, 24, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Incomplete packed"):
        impl.forward_hip(q, q, q, AttentionMetadata(extra={"kv_cache_dtype": "fp8", "max_seqlen_q": 129}))


def test_custom_op_passes_independent_scales_and_keeps_fullgraph(monkeypatch):
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def quantize(tensor):
        assert tensor.ndim == 2
        scale = torch.tensor([len(calls) + 1.0])
        output = tensor.to(torch.float8_e4m3fnuz)
        calls.append((output, scale))
        return output, scale

    def attention(q, k, v, sq, sk, sv, *, causal, softmax_scale):
        assert q.shape == k.shape == v.shape == (1, 129, 2, 128)
        assert [s.item() for s in (sq, sk, sv)] == [1, 2, 3]
        assert all(s.shape == (1,) for s in (sq, sk, sv))
        assert softmax_scale == 0.125 and causal is False
        return torch.ones_like(q, dtype=torch.bfloat16)

    monkeypatch.setattr(fp8_op.ops, "scaled_fp8_quant", quantize)
    monkeypatch.setattr(fp8_op, "flash_attn_fp8_pertensor_func", attention)
    q = torch.zeros((1, 129, 2, 128), dtype=torch.bfloat16)
    compiled = torch.compile(fp8_op.rocm_fp8_attention, backend="eager", fullgraph=True)
    result = compiled(q, q, q, 0.125)
    assert len(calls) == 3 and torch.equal(result, torch.ones_like(q))


def test_rocm_policy_skip_steps_layers_and_optout(monkeypatch, impl):
    monkeypatch.setattr(layer_mod, "current_omni_platform", SimpleNamespace(device_name="rocm"))
    ctx = SimpleNamespace(denoise_step_idx=0)
    monkeypatch.setattr(layer_mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(layer_mod, "get_forward_context", lambda: ctx)
    layer = object.__new__(layer_mod.Attention)
    torch.nn.Module.__init__(layer)
    layer.attention, layer.attn_backend = impl, flash_attn.FlashAttentionBackend
    layer._has_custom_attention, layer._disable_kv_quant = False, False
    layer.layer_idx = layer_mod._try_extract_layer_index("transformer_blocks.3.attn")
    layer._init_kv_cache_quantization(
        SimpleNamespace(
            diffusion_kv_cache_dtype="fp8",
            diffusion_kv_cache_skip_step_indices={0},
            diffusion_kv_cache_skip_layer_indices={4},
        )
    )
    source = AttentionMetadata(extra={"other": 1})
    assert layer._with_kv_cache_dtype(source).extra["kv_cache_dtype"] == "float"
    ctx.denoise_step_idx = 1
    assert layer._with_kv_cache_dtype(source).extra["kv_cache_dtype"] == "fp8"
    layer.layer_idx = 4
    assert layer._with_kv_cache_dtype(source).extra["kv_cache_dtype"] == "float"
    layer.layer_idx, layer._disable_kv_quant = 3, True
    assert layer._with_kv_cache_dtype(source).extra["kv_cache_dtype"] == "float"
    assert source.extra == {"other": 1}
