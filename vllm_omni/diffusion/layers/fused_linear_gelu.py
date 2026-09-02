# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import os

import torch
import torch.nn as nn
import vllm.envs as envs
from torch.library import Library
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

_HAS_ADDMM_ACTIVATION = hasattr(torch, "_addmm_activation")


def _blaslt_is_preferred() -> bool:
    return torch.backends.cuda.preferred_blas_library() == torch._C._BlasBackend.Cublaslt


def _fused_linear_gelu_tanh_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    x_2d = x.reshape(-1, x.shape[-1])
    out = torch._addmm_activation(  # type: ignore[attr-defined]
        bias,
        x_2d,
        weight.t(),
        use_gelu=True,
    )
    return out.view(*x.shape[:-1], weight.shape[0])


def _fused_linear_gelu_tanh_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    del bias
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


_OMNI_OP_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, "fused_linear_gelu_tanh"):
    direct_register_custom_op(
        op_name="fused_linear_gelu_tanh",
        op_func=_fused_linear_gelu_tanh_impl,
        fake_impl=_fused_linear_gelu_tanh_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )


def fused_linear_gelu_tanh(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Compute a linear projection and tanh GELU through one GEMM epilogue."""
    return torch.ops.vllm_omni.fused_linear_gelu_tanh(x, weight, bias)


def fused_linear_gelu_tanh_supported(
    linear: nn.Module,
    x: torch.Tensor,
) -> bool:
    """Return whether the ROCm hipBLASLt epilogue can replace ``linear(x)``."""
    if not (
        _HAS_ADDMM_ACTIVATION
        and current_platform.is_rocm()
        and _blaslt_is_preferred()
        and not envs.VLLM_BATCH_INVARIANT
        and os.getenv("DISABLE_ADDMM_CUDA_LT") != "1"
        and not torch.is_grad_enabled()
        and x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.ndim >= 1
        and x.is_contiguous()
    ):
        return False
    if isinstance(linear, BaseLayerWithLoRA):
        return False
    if not isinstance(getattr(linear, "quant_method", None), UnquantizedLinearMethod):
        return False
    if (
        getattr(linear, "skip_bias_add", False)
        or getattr(linear, "return_bias", True)
        or getattr(linear, "gather_output", False)
        or getattr(linear, "tp_size", 1) != 1
    ):
        return False

    weight = getattr(linear, "weight", None)
    bias = getattr(linear, "bias", None)
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        return False
    return (
        weight.ndim == 2
        and bias.ndim == 1
        and weight.is_contiguous()
        and bias.is_contiguous()
        and weight.device == x.device
        and bias.device == x.device
        and weight.dtype == x.dtype
        and bias.dtype == x.dtype
        and weight.shape[1] == x.shape[-1]
        and bias.shape[0] == weight.shape[0]
    )


__all__ = [
    "fused_linear_gelu_tanh",
    "fused_linear_gelu_tanh_supported",
]
