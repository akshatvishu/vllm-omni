# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTensorSym,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

from vllm_omni.diffusion.models.qwen_image.fused_adaln_fp8 import (
    _fused_adaln_fp8_impl,
    fused_adaln_fp8_supported,
)

pytestmark = [pytest.mark.core_model, pytest.mark.gpu, pytest.mark.diffusion]


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_fused_adaln_fp8_matches_dynamic_tensor_reference() -> None:
    torch.manual_seed(7)
    device = current_platform.device_type
    x = torch.randn(1, 64, 3072, device=device, dtype=torch.bfloat16)
    modulation = 0.15 * torch.randn(1, 3 * 3072, device=device, dtype=torch.bfloat16)
    shift, scale, _ = modulation.chunk(3, dim=-1)
    scale = scale.unsqueeze(1)
    shift = shift.unsqueeze(1)

    reference = F.layer_norm(x, (3072,), eps=1e-6) * (1 + scale) + shift
    reference_scale = reference.abs().max().float().reshape(1) / 224.0
    reference_q = torch.clamp(reference / reference_scale, -224.0, 224.0).to(torch.float8_e4m3fnuz)

    actual_q, actual_scale = _fused_adaln_fp8_impl(x, scale, shift, 1e-6)
    current_platform.synchronize()

    torch.testing.assert_close(actual_scale, reference_scale, rtol=0, atol=0)
    byte_match = (actual_q.view(torch.uint8) == reference_q.view(torch.uint8)).float().mean()
    assert byte_match > 0.95


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_fused_adaln_fp8_rejects_image_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rocm_aiter_ops, "is_linear_fp8_enabled", lambda: True)
    device = current_platform.device_type
    consumer = torch.nn.Identity()
    consumer.input_quant_key = kFp8DynamicTensorSym

    text = torch.empty(1, 512, 3072, device=device, dtype=torch.bfloat16)
    text_scale = torch.empty(1, 1, 3072, device=device, dtype=torch.bfloat16)
    text_shift = torch.empty_like(text_scale)
    assert fused_adaln_fp8_supported(text, text_scale, text_shift, consumer)

    image = torch.empty(1, 1024, 3072, device=device, dtype=torch.bfloat16)
    image_scale = torch.empty(1, 1, 3072, device=device, dtype=torch.bfloat16)
    image_shift = torch.empty_like(image_scale)
    assert not fused_adaln_fp8_supported(image, image_scale, image_shift, consumer)
