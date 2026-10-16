# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_omni.diffusion.attention.backends import flash_attn
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.utils import fa

moon = importlib.import_module("vllm_omni.diffusion.attention.ops.moonmath_attention")
pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def impl(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_ROCM_MOONMATH_ATTN", "1")
    monkeypatch.setattr(flash_attn, "moonmath_devices", lambda: (0,))
    monkeypatch.setattr(flash_attn, "rocm_fp8_attention_available", lambda: False)
    return flash_attn.FlashAttentionImpl(24, 128, 1 / math.sqrt(128))


@pytest.mark.parametrize(
    "reason",
    [
        "disabled",
        "unavailable",
        "fp8",
        "mxfp8",
        "scale",
        "causal",
        "cross",
        "deterministic",
        "attn_mask",
        "joint_attn_mask",
        "full_attn_spans",
        "query_ranges",
        "packed_padding",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "max_seqlen_q",
        "max_seqlen_k",
        "valid_kv_length",
        "batch",
        "sequence",
        "heads",
        "dimension",
        "dtype",
        "strided",
        "shape",
        "cpu",
        "device",
        "architecture",
    ],
)
def test_unsupported_preserves_native_fallback(impl, monkeypatch, reason):
    native = Mock(return_value=object())
    op = Mock(side_effect=AssertionError("Unsupported input must not call MoonMath"))
    monkeypatch.setattr(impl, "forward_cuda", native)
    monkeypatch.setattr(flash_attn, "moonmath_attention", op)
    with FakeTensorMode():
        shape = [1, 4117, 24, 128]
        for name, axis, size in [("batch", 0, 2), ("sequence", 1, 4609), ("heads", 2, 16), ("dimension", 3, 64)]:
            if reason == name:
                shape[axis] = size
        q = torch.empty(shape, dtype=torch.bfloat16, device="cpu" if reason == "cpu" else "cuda:0")
        k = v = q
        metadata = AttentionMetadata()
        if reason in ("disabled", "unavailable", "architecture"):
            impl._moonmath_devices = ()
        elif reason in ("fp8", "mxfp8"):
            metadata.extra["kv_cache_dtype"] = reason
        elif reason == "scale":
            impl._moonmath_scale = False
        elif reason in ("causal", "cross", "deterministic"):
            setattr(
                impl, {"causal": "causal", "cross": "is_cross_attn", "deterministic": "fa_deterministic"}[reason], True
            )
        elif reason in ("attn_mask", "joint_attn_mask", "full_attn_spans", "query_ranges", "packed_padding"):
            setattr(metadata, reason, [])
        elif reason in ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k", "valid_kv_length"):
            metadata.extra[reason] = None
        elif reason == "dtype":
            k = k.float()
        elif reason == "strided":
            v = torch.empty((1, 24, 4117, 128), dtype=q.dtype, device=q.device).transpose(1, 2)
        elif reason == "shape":
            k = k[:, :-1]
        elif reason == "device":
            k = torch.empty_like(q, device="cuda:1")
        assert impl.forward_hip(q, k, v, metadata) is native.return_value
        native.assert_called_once_with(q, k, v, metadata)
        op.assert_not_called()


@pytest.mark.parametrize("sequence", [4096, 4101, 4113, 4115, 4117, 4119, 4224, 4225, 4608])
@pytest.mark.parametrize("metadata", [None, AttentionMetadata(extra={"kv_cache_dtype": "float"})])
def test_supported_fake_output(impl, monkeypatch, sequence, metadata):
    monkeypatch.setattr(impl, "forward_cuda", Mock(side_effect=AssertionError("Expected MoonMath")))
    with FakeTensorMode():
        q = torch.empty((1, sequence, 24, 128), device="cuda:0", dtype=torch.bfloat16)
        out = impl.forward_hip(q, q, q, metadata)
        assert out.shape == q.shape and out.dtype == q.dtype and out.device == q.device
        assert out.is_contiguous() and out is not q


def test_every_workload_length_preserves_split_fallback(impl, monkeypatch):
    native = Mock(return_value=object())
    op = Mock(return_value=object())
    monkeypatch.setattr(impl, "forward_cuda", native)
    monkeypatch.setattr(flash_attn, "moonmath_attention", op)
    with FakeTensorMode():
        # Includes boundaries below 1024² and the larger default image size.
        for sequence in [4095, *range(4096, 5121), 6894, 6910]:
            q = torch.empty((1, sequence, 24, 128), device="cuda:0", dtype=torch.bfloat16)
            result = impl.forward_hip(q, q, q)
            if 4096 <= sequence <= 4608:
                assert result is op.return_value
                op.assert_called_once_with(q, q, q)
                native.assert_not_called()
            else:
                assert result is native.return_value
                native.assert_called_once_with(q, q, q, None)
                op.assert_not_called()
            op.reset_mock()
            native.reset_mock()


def test_default_off_does_not_load_dependency(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_ROCM_MOONMATH_ATTN", raising=False)
    devices = Mock(side_effect=AssertionError("Default must not load MoonMath"))
    monkeypatch.setattr(flash_attn, "moonmath_devices", devices)
    assert flash_attn.FlashAttentionImpl(24, 128, 128**-0.5)._moonmath_devices == ()
    devices.assert_not_called()


@pytest.mark.parametrize("error", [ImportError, OSError])
def test_missing_extension_falls_back(monkeypatch, error):
    moon._moonmath_forward.cache_clear()
    with monkeypatch.context() as patch:
        patch.setattr(torch.version, "hip", "test")
        patch.setattr(moon.importlib, "import_module", Mock(side_effect=error("unavailable")))
        assert moon.moonmath_devices() == ()
    moon._moonmath_forward.cache_clear()


@pytest.mark.parametrize(
    "name,arch,cus,expected",
    [
        ("AMD Instinct MI300X", "gfx942:sramecc+:xnack-", 304, (0,)),
        ("", "gfx942:sramecc+:xnack-", 304, (0,)),
        ("AMD Instinct MI325X", "gfx942", 304, (0,)),
        ("AMD Instinct MI300X", "gfx950", 304, ()),
        ("AMD Instinct MI300X", "gfx942", 228, ()),
    ],
)
def test_device_scope(monkeypatch, name, arch, cus, expected):
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(moon, "_moonmath_forward", lambda: Mock())
    monkeypatch.setattr(torch.accelerator, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(name=name, gcnArchName=arch, multi_processor_count=cus),
    )
    assert moon.moonmath_devices() == expected


def test_partial_packed_metadata_still_raises(impl, monkeypatch):
    monkeypatch.setattr(fa, "HAS_FLASH_ATTN", True)
    q = torch.empty((1, 4117, 24, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Incomplete packed"):
        impl.forward_hip(q, q, q, AttentionMetadata(extra={"max_seqlen_q": 4117}))


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_output_mutation_compiles_and_calls_do_not_alias(monkeypatch, backend):
    def forward(q, k, v, *, out, round_mode, layout):
        assert round_mode == "rtna" and layout == "bshd"
        out.copy_(q + k + v)
        return out

    monkeypatch.setattr(moon, "_moonmath_forward", lambda: forward)
    library = torch.library.Library("vllm_omni", "IMPL", "CPU")
    try:
        library.impl("moonmath_attention_out", moon._moonmath_attention_out_impl)
        compiled = torch.compile(moon.moonmath_attention, backend=backend, fullgraph=True)
        q = torch.ones((1, 3, 2, 128), dtype=torch.bfloat16)
        first = compiled(q, q, q)
        second = compiled(q * 2, q, q)
        assert torch.equal(first, q * 3) and torch.equal(second, q * 4)
        assert first.data_ptr() != second.data_ptr() != q.data_ptr()
    finally:
        library._destroy()


@pytest.mark.parametrize("scale,expected", [(128**-0.5, True), (1 / math.sqrt(128), True), (0.125, False)])
def test_scale_at_float32_precision(impl, scale, expected):
    assert flash_attn.FlashAttentionImpl(24, 128, scale)._moonmath_scale is expected


def test_explicit_fp8_retains_fp8_dispatch(impl, monkeypatch):
    impl._rocm_fp8_available = True
    fp8 = Mock(return_value=object())
    moon_op = Mock(side_effect=AssertionError("Explicit FP8 must not call MoonMath"))
    monkeypatch.setattr(flash_attn, "rocm_fp8_attention", fp8)
    monkeypatch.setattr(flash_attn, "moonmath_attention", moon_op)
    with FakeTensorMode():
        q = torch.empty((1, 4117, 24, 128), device="cuda:0", dtype=torch.bfloat16)
        result = impl.forward_hip(q, q, q, AttentionMetadata(extra={"kv_cache_dtype": "fp8"}))
        assert result is fp8.return_value
        fp8.assert_called_once_with(q, q, q, impl.softmax_scale)
        moon_op.assert_not_called()
