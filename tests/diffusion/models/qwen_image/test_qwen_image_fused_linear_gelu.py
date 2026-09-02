# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm_omni.diffusion.models.qwen_image.qwen_image_transformer as qwen_image

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _Projection(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return F.linear(x, self.weight, self.bias)


def _make_layer(approximate: str = "tanh") -> qwen_image.ColumnParallelApproxGELU:
    layer = qwen_image.ColumnParallelApproxGELU.__new__(qwen_image.ColumnParallelApproxGELU)
    nn.Module.__init__(layer)
    layer.proj = _Projection(torch.randn(16, 8), torch.randn(16))
    layer.approximate = approximate
    return layer


def test_qwen_image_uses_fused_projection_gelu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _make_layer()
    x = torch.randn(2, 3, 8)
    expected = torch.randn(2, 3, 16)
    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    monkeypatch.setattr(
        qwen_image,
        "fused_linear_gelu_tanh_supported",
        lambda linear, value: linear is layer.proj and value is x,
    )

    def fused(
        value: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        calls.append((value, weight, bias))
        return expected

    monkeypatch.setattr(qwen_image, "fused_linear_gelu_tanh", fused)

    assert layer(x) is expected
    assert calls == [(x, layer.proj.weight, layer.proj.bias)]
    assert layer.proj.calls == 0


@pytest.mark.parametrize("approximate", ["tanh", "none"])
def test_qwen_image_projection_gelu_fallback(
    monkeypatch: pytest.MonkeyPatch,
    approximate: str,
) -> None:
    layer = _make_layer(approximate)
    x = torch.randn(2, 3, 8)
    monkeypatch.setattr(
        qwen_image,
        "fused_linear_gelu_tanh_supported",
        lambda linear, value: False,
    )
    monkeypatch.setattr(
        qwen_image,
        "fused_linear_gelu_tanh",
        lambda *args: pytest.fail("fused path should not run"),
    )

    actual = layer(x)
    expected = F.gelu(F.linear(x, layer.proj.weight, layer.proj.bias), approximate=approximate)

    torch.testing.assert_close(actual, expected)
    assert layer.proj.calls == 1
