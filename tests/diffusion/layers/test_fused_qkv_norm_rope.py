# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.layers import fused_qkv_norm_rope as fused_module
from vllm_omni.diffusion.layers.fused_qkv_norm_rope import (
    _fused_qkv_norm_rope_fake,
    _fused_qkv_norm_rope_impl,
    rocm_aiter_fused_qkv_norm_rope_available,
    try_rocm_aiter_fused_qkv_norm_rope,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_HEAD_DIM = 128
_NUM_HEADS = 24
_PACKED_DIM = 3 * _NUM_HEADS * _HEAD_DIM


def _inputs(device: str = "cpu") -> tuple[torch.Tensor, ...]:
    img_qkv = torch.empty((1, 5, _PACKED_DIM), dtype=torch.bfloat16, device=device)
    txt_qkv = torch.empty((1, 3, _PACKED_DIM), dtype=torch.bfloat16, device=device)
    weights = tuple(torch.empty((_HEAD_DIM,), dtype=torch.bfloat16, device=device) for _ in range(4))
    img_rope = torch.empty((5, _HEAD_DIM), dtype=torch.bfloat16, device=device)
    txt_rope = torch.empty((3, _HEAD_DIM), dtype=torch.bfloat16, device=device)
    return img_qkv, txt_qkv, *weights, img_rope, txt_rope


def test_fused_qkv_norm_rope_fake_shape_and_dtype():
    outputs = _fused_qkv_norm_rope_fake(*_inputs(), 1e-6)

    assert len(outputs) == 3
    for output in outputs:
        assert output.shape == (1, 8, _NUM_HEADS, _HEAD_DIM)
        assert output.dtype == torch.bfloat16
        assert output.device.type == "cpu"


def test_fused_qkv_norm_rope_disabled_returns_none():
    assert (
        try_rocm_aiter_fused_qkv_norm_rope(
            *_inputs(),
            1e-6,
            enabled=False,
        )
        is None
    )


def test_fused_qkv_norm_rope_compilation_returns_none(monkeypatch):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    assert (
        try_rocm_aiter_fused_qkv_norm_rope(
            *_inputs(),
            1e-6,
            enabled=True,
        )
        is None
    )


def test_fused_qkv_norm_rope_rejects_unsupported_inputs():
    assert (
        try_rocm_aiter_fused_qkv_norm_rope(
            *_inputs(),
            1e-6,
            enabled=True,
        )
        is None
    )


@pytest.mark.parametrize("missing", ["rocm", "gfx942", "aiter_enabled", "operation"])
def test_fused_qkv_norm_rope_availability_requires_full_contract(monkeypatch, missing):
    monkeypatch.setattr(fused_module.current_platform, "is_rocm", lambda: missing != "rocm")
    monkeypatch.setattr(fused_module, "on_gfx942", lambda: missing != "gfx942")
    monkeypatch.setattr(
        fused_module.rocm_aiter_ops,
        "is_enabled",
        lambda: missing != "aiter_enabled",
    )
    monkeypatch.setattr(
        fused_module,
        "_aiter_fused_qkv_norm_rope",
        None if missing == "operation" else object(),
    )

    assert not rocm_aiter_fused_qkv_norm_rope_available()


def test_fused_qkv_norm_rope_impl_preserves_inputs_and_adds_batch(monkeypatch):
    inputs = _inputs()
    expected = tuple(torch.empty((8, _NUM_HEADS, _HEAD_DIM)) for _ in range(3))

    def fake_aiter(*args):
        for received, input_tensor in zip(args[:-1], inputs, strict=True):
            assert received is input_tensor
        assert args[-1] == 1e-6
        return expected

    monkeypatch.setattr(fused_module, "_aiter_fused_qkv_norm_rope", fake_aiter)
    actual = _fused_qkv_norm_rope_impl(*inputs, 1e-6)

    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        assert actual_tensor.shape == (1, 8, _NUM_HEADS, _HEAD_DIM)
        assert actual_tensor.data_ptr() == expected_tensor.data_ptr()
