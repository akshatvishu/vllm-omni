# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import torch
from torch.library import Library
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTensorSym
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_HIDDEN_SIZE = 3072
_FP8_MAX = 224.0
# Production profiling shows a win for Qwen's 512-token text stream but a
# regression for its 1024-token image stream on MI300X.
_MAX_SEQUENCE_LENGTH = 512


if HAS_TRITON:

    @triton.jit
    def _adaln_scale_shift_amax_kernel(
        output_ptr,
        output_scale_ptr,
        input_ptr,
        scale_ptr,
        shift_ptr,
        sequence_length,
        scale_stride_batch,
        scale_stride_sequence,
        shift_stride_batch,
        shift_stride_sequence,
        hidden_size: tl.constexpr,
        parameter_sequence_length: tl.constexpr,
        fp8_max: tl.constexpr,
        eps: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, block_size)
        mask = columns < hidden_size
        offsets = row * hidden_size + columns

        x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / hidden_size
        centered = x - mean
        variance = tl.sum(centered * centered, axis=0) / hidden_size
        normalized = (centered * tl.rsqrt(variance + eps)).to(tl.bfloat16).to(tl.float32)

        batch = row // sequence_length
        sequence = row % sequence_length if parameter_sequence_length > 1 else 0
        scale_offsets = batch * scale_stride_batch + sequence * scale_stride_sequence + columns
        shift_offsets = batch * shift_stride_batch + sequence * shift_stride_sequence + columns
        scale = tl.load(scale_ptr + scale_offsets, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + shift_offsets, mask=mask, other=0.0).to(tl.float32)
        output = (normalized * (1.0 + scale) + shift).to(tl.bfloat16)

        tl.store(output_ptr + offsets, output, mask=mask)
        row_scale = tl.max(tl.abs(output.to(tl.float32)), axis=0) / fp8_max
        tl.atomic_max(output_scale_ptr, row_scale)

    @triton.jit
    def _dynamic_tensor_quant_kernel(
        output_ptr,
        input_ptr,
        scale_ptr,
        hidden_size: tl.constexpr,
        fp8_max: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, block_size)
        mask = columns < hidden_size
        offsets = row * hidden_size + columns
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr)
        scaled = tl.where(scale > 0.0, x / scale, 0.0)
        quantized = tl.clamp(scaled, -fp8_max, fp8_max).to(tl.float8e4b8)
        tl.store(output_ptr + offsets, quantized, mask=mask)


def _supported_parameter_shape(parameter: torch.Tensor, x: torch.Tensor) -> bool:
    return parameter.shape in (
        (x.shape[0], 1, _HIDDEN_SIZE),
        (x.shape[0], x.shape[1], _HIDDEN_SIZE),
    )


def fused_adaln_fp8_supported(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    consumer: torch.nn.Module,
) -> bool:
    return (
        HAS_TRITON
        and current_platform.is_rocm()
        and rocm_aiter_ops.is_linear_fp8_enabled()
        and current_platform.fp8_dtype() == torch.float8_e4m3fnuz
        and getattr(consumer, "input_quant_key", None) == kFp8DynamicTensorSym
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim == 3
        and x.shape[-1] == _HIDDEN_SIZE
        and x.shape[1] <= _MAX_SEQUENCE_LENGTH
        and x.numel() > 0
        and x.is_contiguous()
        and scale.device == x.device
        and shift.device == x.device
        and scale.dtype == x.dtype
        and shift.dtype == x.dtype
        and scale.stride(-1) == 1
        and shift.stride(-1) == 1
        and _supported_parameter_shape(scale, x)
        and scale.shape == shift.shape
    )


def _fused_adaln_fp8_impl(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, hidden_size = x.shape
    rows = batch_size * sequence_length
    modulated = torch.empty_like(x)
    output_scale = torch.zeros(1, dtype=torch.float32, device=x.device)
    output = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)
    block_size = triton.next_power_of_2(hidden_size)

    _adaln_scale_shift_amax_kernel[(rows,)](
        modulated,
        output_scale,
        x,
        scale,
        shift,
        sequence_length,
        scale.stride(0),
        scale.stride(1),
        shift.stride(0),
        shift.stride(1),
        hidden_size=hidden_size,
        parameter_sequence_length=scale.shape[1],
        fp8_max=_FP8_MAX,
        eps=eps,
        block_size=block_size,
        num_warps=2,
    )
    _dynamic_tensor_quant_kernel[(rows,)](
        output,
        modulated,
        output_scale,
        hidden_size=hidden_size,
        fp8_max=_FP8_MAX,
        block_size=block_size,
        num_warps=2,
    )
    return output, output_scale


def _fused_adaln_fp8_fake(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del scale, shift, eps
    return (
        torch.empty_like(x, dtype=current_platform.fp8_dtype()),
        torch.empty(1, dtype=torch.float32, device=x.device),
    )


_OMNI_OP_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, "fused_adaln_fp8"):
    direct_register_custom_op(
        op_name="fused_adaln_fp8",
        op_func=_fused_adaln_fp8_impl,
        fake_impl=_fused_adaln_fp8_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )


def fused_adaln_fp8(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> QuantizedActivation:
    output, output_scale = torch.ops.vllm_omni.fused_adaln_fp8(x, scale, shift, eps)
    return QuantizedActivation(
        data=output,
        scale=output_scale,
        orig_dtype=x.dtype,
        orig_shape=x.shape,
        quant_key=kFp8DynamicTensorSym,
    )


__all__ = ["fused_adaln_fp8", "fused_adaln_fp8_supported"]
