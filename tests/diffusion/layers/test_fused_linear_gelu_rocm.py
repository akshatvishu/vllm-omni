# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_omni.diffusion.layers.fused_linear_gelu import (
    fused_linear_gelu_tanh,
    fused_linear_gelu_tanh_supported,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.rocm,
    pytest.mark.cards_1,
    pytest.mark.skipif(not current_omni_platform.is_rocm(), reason="ROCm required"),
]


class _FusedLinearGELU(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        return fused_linear_gelu_tanh(x, weight, bias)


def _make_linear(weight: torch.Tensor, bias: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        weight=weight,
        bias=bias,
        quant_method=object.__new__(UnquantizedLinearMethod),
        skip_bias_add=False,
        return_bias=False,
        gather_output=False,
        tp_size=1,
    )


def _make_inputs(
    tokens: int,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(17)
    return (
        torch.randn(tokens, 3072, device="cuda", dtype=dtype),
        torch.randn(12288, 3072, device="cuda", dtype=dtype),
        torch.randn(12288, device="cuda", dtype=dtype),
    )


@pytest.mark.parametrize(
    ("tokens", "dtype"),
    [
        (1024, torch.bfloat16),
        (13, torch.bfloat16),
        (5, torch.bfloat16),
        (13, torch.float16),
    ],
)
@torch.no_grad()
def test_fused_linear_gelu_tanh_matches_qwen_image_shapes(tokens: int, dtype: torch.dtype) -> None:
    x, weight, bias = _make_inputs(tokens, dtype)
    if not fused_linear_gelu_tanh_supported(_make_linear(weight, bias), x):
        pytest.skip("PyTorch hipBLASLt GELU epilogue is not selected")

    actual = fused_linear_gelu_tanh(x, weight, bias)
    expected = F.gelu(
        F.linear(x.float(), weight.float(), bias.float()),
        approximate="tanh",
    ).to(dtype)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.016, atol=0.01)


@torch.no_grad()
def test_fused_linear_gelu_tanh_compiles_fullgraph() -> None:
    x, weight, bias = _make_inputs(13)
    if not fused_linear_gelu_tanh_supported(_make_linear(weight, bias), x):
        pytest.skip("PyTorch hipBLASLt GELU epilogue is not selected")

    compiled = torch.compile(_FusedLinearGELU(), fullgraph=True)
    torch.testing.assert_close(
        compiled(x, weight, bias),
        fused_linear_gelu_tanh(x, weight, bias),
        rtol=0,
        atol=0,
    )
