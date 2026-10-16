# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Experimental dense MI300X FP8 attention using vLLM quantization and AITER."""

from collections.abc import Callable

import torch
from vllm import _custom_ops as ops

from vllm_omni.platforms import current_omni_platform

flash_attn_fp8_pertensor_func: Callable[..., torch.Tensor] | None = None
ENABLE_CK = False
get_gfx: Callable[[], str] | None = None
if current_omni_platform.is_rocm():
    try:
        from aiter.jit.core import ENABLE_CK as _ENABLE_CK
        from aiter.jit.utils.chip_info import get_gfx as _get_gfx
        from aiter.ops.mha import flash_attn_fp8_pertensor_func as _flash_attn_fp8_pertensor_func

        ENABLE_CK = _ENABLE_CK
        get_gfx = _get_gfx
        flash_attn_fp8_pertensor_func = _flash_attn_fp8_pertensor_func
    except ImportError:
        pass


def rocm_fp8_attention_available() -> bool:
    # With CK disabled, the same AITER API selects a different Triton kernel.
    return bool(flash_attn_fp8_pertensor_func is not None and ENABLE_CK and get_gfx and get_gfx() == "gfx942")


@torch.library.custom_op("vllm_omni::rocm_fp8_attention", mutates_args=())
def rocm_fp8_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    """Quantize Q/K/V independently, then run AITER dense FP8 attention.

    The backend checks eligibility before calling this opaque compile boundary.
    Descales have shape [1], selecting AITER's per-tensor assembly route.
    """
    assert flash_attn_fp8_pertensor_func is not None
    # Keep all heads of a token in one row; per-head rows multiply the
    # quantizer's row-reduction work without changing the per-tensor scale.
    quantized = [
        ops.scaled_fp8_quant(tensor.view(-1, tensor.shape[-2] * tensor.shape[-1])) for tensor in (query, key, value)
    ]
    return flash_attn_fp8_pertensor_func(
        *[pair[0].view_as(tensor) for pair, tensor in zip(quantized, (query, key, value))],
        *[pair[1] for pair in quantized],
        causal=False,
        softmax_scale=softmax_scale,
    )


@rocm_fp8_attention.register_fake
def _rocm_fp8_attention_fake(query, key, value, softmax_scale):
    return torch.empty(query.shape, dtype=torch.bfloat16, device=query.device)
