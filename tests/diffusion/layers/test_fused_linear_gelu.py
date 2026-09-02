# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
import vllm.envs as envs
from torch._subclasses.fake_tensor import FakeTensorMode
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.platforms import current_platform

import vllm_omni.diffusion.layers.fused_linear_gelu as fused_linear_gelu

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _supported_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        torch.backends.cuda,
        "preferred_blas_library",
        lambda: torch._C._BlasBackend.Cublaslt,
    )
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.delenv("DISABLE_ADDMM_CUDA_LT", raising=False)


class _FusedLinearGELU(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        return fused_linear_gelu.fused_linear_gelu_tanh(x, weight, bias)


def _make_linear(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        weight=weight,
        bias=bias,
        quant_method=object.__new__(UnquantizedLinearMethod),
        skip_bias_add=False,
        return_bias=False,
        gather_output=False,
        tp_size=1,
    )


def test_fused_linear_gelu_tanh_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    with FakeTensorMode(), torch.no_grad():
        x = torch.empty(2, 3, 8, device="cuda", dtype=torch.bfloat16)
        weight = torch.empty(16, 8, device="cuda", dtype=torch.bfloat16)
        bias = torch.empty(16, device="cuda", dtype=torch.bfloat16)

        assert fused_linear_gelu.fused_linear_gelu_tanh_supported(_make_linear(weight, bias), x)


@pytest.mark.parametrize(
    "reason",
    ["missing_op", "non_rocm", "blaslt_not_preferred", "batch_invariant", "lt_disabled"],
)
def test_fused_linear_gelu_tanh_rejects_unsupported_runtime(
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
) -> None:
    monkeypatch.setattr(current_platform, "is_rocm", lambda: reason != "non_rocm")
    monkeypatch.setattr(fused_linear_gelu, "_HAS_ADDMM_ACTIVATION", reason != "missing_op")
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", reason == "batch_invariant")
    if reason == "blaslt_not_preferred":
        monkeypatch.setattr(
            torch.backends.cuda,
            "preferred_blas_library",
            lambda: torch._C._BlasBackend.Cublas,
        )
    if reason == "lt_disabled":
        monkeypatch.setenv("DISABLE_ADDMM_CUDA_LT", "1")
    with FakeTensorMode(), torch.no_grad():
        x = torch.empty(2, 3, 8, device="cuda", dtype=torch.bfloat16)
        weight = torch.empty(16, 8, device="cuda", dtype=torch.bfloat16)
        bias = torch.empty(16, device="cuda", dtype=torch.bfloat16)

        assert not fused_linear_gelu.fused_linear_gelu_tanh_supported(_make_linear(weight, bias), x)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("quant_method", object()),
        ("bias", None),
        ("skip_bias_add", True),
        ("return_bias", True),
        ("gather_output", True),
        ("tp_size", 2),
    ],
)
def test_fused_linear_gelu_tanh_rejects_unsupported_linear(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    with FakeTensorMode(), torch.no_grad():
        x = torch.empty(2, 3, 8, device="cuda", dtype=torch.bfloat16)
        weight = torch.empty(16, 8, device="cuda", dtype=torch.bfloat16)
        bias = torch.empty(16, device="cuda", dtype=torch.bfloat16)
        linear = _make_linear(weight, bias)
        setattr(linear, field, value)

        assert not fused_linear_gelu.fused_linear_gelu_tanh_supported(linear, x)


def test_fused_linear_gelu_tanh_rejects_lora_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    linear = object.__new__(fused_linear_gelu.BaseLayerWithLoRA)
    with FakeTensorMode(), torch.no_grad():
        x = torch.empty(2, 3, 8, device="cuda", dtype=torch.bfloat16)

        assert not fused_linear_gelu.fused_linear_gelu_tanh_supported(linear, x)


def test_fused_linear_gelu_tanh_rejects_grad_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    with FakeTensorMode():
        x = torch.empty(2, 3, 8, device="cuda", dtype=torch.bfloat16)
        weight = torch.empty(16, 8, device="cuda", dtype=torch.bfloat16)
        bias = torch.empty(16, device="cuda", dtype=torch.bfloat16)

        assert not fused_linear_gelu.fused_linear_gelu_tanh_supported(_make_linear(weight, bias), x)


def test_fused_linear_gelu_tanh_rejects_noncontiguous_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    with FakeTensorMode(), torch.no_grad():
        x = torch.empty(2, 8, 3, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        weight = torch.empty(16, 8, device="cuda", dtype=torch.bfloat16)
        bias = torch.empty(16, device="cuda", dtype=torch.bfloat16)

        assert not fused_linear_gelu.fused_linear_gelu_tanh_supported(_make_linear(weight, bias), x)


def test_fused_linear_gelu_tanh_fake_shape() -> None:
    x = torch.empty(2, 3, 8)
    weight = torch.empty(16, 8)
    bias = torch.empty(16)

    out = fused_linear_gelu._fused_linear_gelu_tanh_fake(x, weight, bias)

    assert out.shape == (2, 3, 16)
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_fused_linear_gelu_tanh_is_compile_safe() -> None:
    with FakeTensorMode():
        x = torch.empty(2, 3, 8, device="cuda")
        weight = torch.empty(16, 8, device="cuda")
        bias = torch.empty(16, device="cuda")
        exported = torch.export.export(
            _FusedLinearGELU(),
            (x, weight, bias),
            strict=True,
        )

    call_targets = {node.target for node in exported.graph.nodes if node.op == "call_function"}
    assert torch.ops.vllm_omni.fused_linear_gelu_tanh.default in call_targets
