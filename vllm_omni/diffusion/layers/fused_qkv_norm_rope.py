# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import torch
from torch.library import Library
from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx942
else:

    def on_gfx942() -> bool:
        return False


try:
    from aiter.ops.triton.rope.fused_qkv_split_qk_norm_rope_cat import (
        fused_qkv_split_qk_norm_rope_cat as _aiter_fused_qkv_norm_rope,
    )
except ImportError:
    _aiter_fused_qkv_norm_rope = None

_NUM_HEADS = 24
_HEAD_DIM = 128
_PACKED_DIM = 3 * _NUM_HEADS * _HEAD_DIM


def rocm_aiter_fused_qkv_norm_rope_available() -> bool:
    """Return whether the optional AITER operation can be selected."""
    return bool(
        current_platform.is_rocm()
        and on_gfx942()
        and rocm_aiter_ops.is_enabled()
        and _aiter_fused_qkv_norm_rope is not None
    )


def _fused_qkv_norm_rope_impl(
    img_qkv: torch.Tensor,
    txt_qkv: torch.Tensor,
    img_q_weight: torch.Tensor,
    img_k_weight: torch.Tensor,
    txt_q_weight: torch.Tensor,
    txt_k_weight: torch.Tensor,
    img_rope: torch.Tensor,
    txt_rope: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _aiter_fused_qkv_norm_rope is None:
        raise RuntimeError("The AITER fused QKV normalization and RoPE operation is unavailable")
    joint_q, joint_k, joint_v = _aiter_fused_qkv_norm_rope(
        img_qkv,
        txt_qkv,
        img_q_weight,
        img_k_weight,
        txt_q_weight,
        txt_k_weight,
        img_rope,
        txt_rope,
        eps,
    )
    return joint_q.unsqueeze(0), joint_k.unsqueeze(0), joint_v.unsqueeze(0)


def _fused_qkv_norm_rope_fake(
    img_qkv: torch.Tensor,
    txt_qkv: torch.Tensor,
    img_q_weight: torch.Tensor,
    img_k_weight: torch.Tensor,
    txt_q_weight: torch.Tensor,
    txt_k_weight: torch.Tensor,
    img_rope: torch.Tensor,
    txt_rope: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        img_q_weight,
        img_k_weight,
        txt_q_weight,
        txt_k_weight,
        img_rope,
        txt_rope,
        eps,
    )
    shape = (1, img_qkv.shape[1] + txt_qkv.shape[1], _NUM_HEADS, _HEAD_DIM)
    return tuple(img_qkv.new_empty(shape) for _ in range(3))


_OMNI_OP_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, "fused_qkv_norm_rope"):
    direct_register_custom_op(
        op_name="fused_qkv_norm_rope",
        op_func=_fused_qkv_norm_rope_impl,
        fake_impl=_fused_qkv_norm_rope_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )


def _packed_qkv_supported(tensor: torch.Tensor) -> bool:
    return bool(
        tensor.is_cuda
        and tensor.dtype == torch.bfloat16
        and tensor.ndim == 3
        and tensor.shape[0] == 1
        and tensor.shape[1] > 0
        and tensor.shape[2] == _PACKED_DIM
        and tensor.stride(1) == _PACKED_DIM
        and tensor.stride(2) == 1
    )


def _weight_supported(weight: torch.Tensor, like: torch.Tensor) -> bool:
    return bool(
        weight.device == like.device
        and weight.dtype == torch.bfloat16
        and weight.shape == (_HEAD_DIM,)
        and weight.stride(0) == 1
    )


def _rope_supported(
    rope: torch.Tensor,
    like: torch.Tensor,
    tokens: int,
) -> bool:
    return bool(
        rope.device == like.device
        and rope.dtype == torch.bfloat16
        and rope.shape == (tokens, _HEAD_DIM)
        and rope.stride(0) == _HEAD_DIM
        and rope.stride(1) == 1
    )


def try_rocm_aiter_fused_qkv_norm_rope(
    img_qkv: torch.Tensor,
    txt_qkv: torch.Tensor,
    img_q_weight: torch.Tensor,
    img_k_weight: torch.Tensor,
    txt_q_weight: torch.Tensor,
    txt_k_weight: torch.Tensor,
    img_rope: torch.Tensor,
    txt_rope: torch.Tensor,
    eps: float,
    *,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Run the narrow MI300X AITER path or return ``None`` unchanged."""
    if (
        not enabled
        or torch.compiler.is_compiling()
        or not rocm_aiter_fused_qkv_norm_rope_available()
        or not _packed_qkv_supported(img_qkv)
        or not _packed_qkv_supported(txt_qkv)
        or txt_qkv.device != img_qkv.device
    ):
        return None

    weights = (img_q_weight, img_k_weight, txt_q_weight, txt_k_weight)
    if not all(_weight_supported(weight, img_qkv) for weight in weights):
        return None
    if not _rope_supported(img_rope, img_qkv, img_qkv.shape[1]):
        return None
    if not _rope_supported(txt_rope, img_qkv, txt_qkv.shape[1]):
        return None

    return torch.ops.vllm_omni.fused_qkv_norm_rope(
        img_qkv,
        txt_qkv,
        img_q_weight,
        img_k_weight,
        txt_q_weight,
        txt_k_weight,
        img_rope,
        txt_rope,
        eps,
    )


__all__ = [
    "rocm_aiter_fused_qkv_norm_rope_available",
    "try_rocm_aiter_fused_qkv_norm_rope",
]
