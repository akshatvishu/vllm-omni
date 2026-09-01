# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNorm
from vllm_omni.diffusion.models.qwen_image.fused_adaln_fp8 import (
    _fused_adaln_fp8_fake,
    fused_adaln_fp8_supported,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_fused_adaln_fp8_rejects_cpu() -> None:
    x = torch.randn(1, 2, 3072, dtype=torch.bfloat16)
    scale = torch.randn(1, 1, 3072, dtype=torch.bfloat16)
    shift = torch.randn_like(scale)
    consumer = SimpleNamespace(input_quant_key=None)

    assert not fused_adaln_fp8_supported(x, scale, shift, consumer)


def test_fused_adaln_fp8_fake_preserves_contract() -> None:
    x = torch.randn(2, 3, 3072, dtype=torch.bfloat16)
    scale = torch.randn(2, 1, 3072, dtype=torch.bfloat16)
    shift = torch.randn_like(scale)

    output, output_scale = _fused_adaln_fp8_fake(x, scale, shift, 1e-6)

    assert output.shape == x.shape
    assert output_scale.shape == (1,)
    assert output_scale.dtype == torch.float32


def test_qwen_norm_for_linear_keeps_native_fallback() -> None:
    from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
        QwenImageTransformerBlock,
    )

    x = torch.randn(1, 2, 3072, dtype=torch.bfloat16)
    scale = torch.randn(1, 1, 3072, dtype=torch.bfloat16)
    shift = torch.randn_like(scale)
    norm = AdaLayerNorm(3072, elementwise_affine=False, eps=1e-6)

    actual = QwenImageTransformerBlock._norm_for_linear(
        norm,
        x,
        scale,
        shift,
        torch.nn.Identity(),
    )

    torch.testing.assert_close(actual, norm(x, scale, shift))
