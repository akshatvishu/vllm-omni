# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fused Q/K RMSNorm and RoPE operations for diffusion attention.

The ``fused_qk_norm_rope`` contract is shared by diffusion attention implementations:

* ``q`` and ``k`` are ``[tokens, heads, head_dim]``;
* norm weights are one-dimensional ``[head_dim]`` tensors;
* ``rope_table`` is ``[tokens, rotary_dim]`` and stores
  ``[cos(theta), sin(theta)]`` with ``theta`` of width ``rotary_dim // 2``.

The CUDA fast path fuses RMSNorm and non-interleaved RoPE without materializing
normalized Q/K or rotary-product intermediates. Ascend composes its RMSNorm and
rotary fused primitives. The ROCm AITER operation accepts two ``[B, T, H, D]``
streams and applies interleaved RoPE. Unsupported inputs use the eager reference.
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

import torch
import torch.nn.functional as F
from torch.library import Library
from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_omni.platforms import current_omni_platform

_FUSED_HEAD_DIM = 128
_FUSED_ROTARY_DIM = 96
_HEADS_PER_PROGRAM = 8
_AITER_QK_NORM_ROPE_2WAY_HEAD_DIM = 128


def _apply_rope_table(
    x: torch.Tensor,
    rope_table: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    half = rotary_dim // 2
    cos = rope_table[..., :half].to(x.dtype).unsqueeze(1)
    sin = rope_table[..., half:].to(x.dtype).unsqueeze(1)
    first = x[..., :half]
    second = x[..., half:rotary_dim]
    return torch.cat(
        (
            first * cos - second * sin,
            second * cos + first * sin,
            x[..., rotary_dim:],
        ),
        dim=-1,
    )


if HAS_TRITON:

    @triton.jit
    def _rms_norm_rope_kernel(
        x_ptr,
        weight_ptr,
        rope_table_ptr,
        out_ptr,
        x_stride_t,
        x_stride_h,
        x_stride_d,
        rope_stride_t,
        out_stride_t,
        out_stride_h,
        out_stride_d,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        rotary_half: tl.constexpr,
        eps: tl.constexpr,
        heads_per_program: tl.constexpr,
    ):
        token = tl.program_id(0)
        head_group = tl.program_id(1)
        heads = head_group * heads_per_program + tl.arange(0, heads_per_program)
        dims = tl.arange(0, head_dim)
        mask = heads[:, None] < num_heads
        offsets = token * x_stride_t + heads[:, None] * x_stride_h + dims[None, :] * x_stride_d

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + dims).to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(x * x, axis=1) / head_dim + eps)
        normalized = (x * inv_rms[:, None] * weight[None, :]).to(tl.bfloat16)

        rotary_dim = rotary_half * 2
        pair_dims = tl.where(
            dims < rotary_half,
            dims + rotary_half,
            tl.where(dims < rotary_dim, dims - rotary_half, dims),
        )
        pair_offsets = token * x_stride_t + heads[:, None] * x_stride_h + pair_dims[None, :] * x_stride_d
        pair_x = tl.load(x_ptr + pair_offsets, mask=mask, other=0.0).to(tl.float32)
        pair_weight = tl.load(weight_ptr + pair_dims).to(tl.float32)
        pair_normalized = (pair_x * inv_rms[:, None] * pair_weight[None, :]).to(tl.bfloat16)

        freq_dims = tl.where(
            dims < rotary_half,
            dims,
            tl.where(dims < rotary_dim, dims - rotary_half, 0),
        )
        table_offsets = token * rope_stride_t + freq_dims
        cos = tl.load(rope_table_ptr + table_offsets).to(tl.float32)
        sin = tl.load(rope_table_ptr + table_offsets + rotary_half).to(tl.float32)
        first = normalized.to(tl.float32) * cos - pair_normalized.to(tl.float32) * sin
        second = normalized.to(tl.float32) * cos + pair_normalized.to(tl.float32) * sin
        output = tl.where(
            dims < rotary_dim,
            tl.where(dims < rotary_half, first, second),
            normalized.to(tl.float32),
        )

        out_offsets = token * out_stride_t + heads[:, None] * out_stride_h + dims[None, :] * out_stride_d
        tl.store(out_ptr + out_offsets, output, mask=mask)


def _eager_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_norm = F.rms_norm(q, (head_dim,), q_weight, eps)
    k_norm = F.rms_norm(k, (head_dim,), k_weight, eps)
    return (
        _apply_rope_table(q_norm, rope_table, rotary_dim),
        _apply_rope_table(k_norm, rope_table, rotary_dim),
    )


def _npu_apply_rope_table(
    x: torch.Tensor,
    rope_table: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    """Apply H3's rotary dimensions through Ascend's fused rotary kernel.

    ``mindiesd.rotary_position_embedding`` takes a 4-D BSND tensor, whereas
    MiniMax-H3 keeps packed activations as ``[tokens, heads, head_dim]``.
    The shared MindIE-SD wrapper normalizes this 3-D layout to BSND and
    restores it afterwards. It receives the half-width cos/sin values from
    the packed table; the wrapper expands them to H3's non-interleaved 96-D
    rotary layout. The 32 non-rotary head dimensions bypass the kernel.

    Some CANN environments package ``torch_npu`` without MindIE-SD. Keep
    those deployments on the same Ascend fused rotary primitive rather than
    silently falling back to eager elementwise RoPE.
    """
    import torch_npu

    half = rotary_dim // 2
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    cos = rope_table[..., :half]
    sin = rope_table[..., half:]

    if find_spec("mindiesd") is not None:
        from vllm_omni.diffusion.layers.rope import apply_rotary_emb_mindiesd

        x_rot = apply_rotary_emb_mindiesd(
            x_rot,
            cos,
            sin,
            interleaved=False,
            half_head_dim=True,
        )
    else:
        # npu_rotary_mul uses BSND and full rotary-width cos/sin. H3 uses
        # NeoX/rotated-half ordering, so duplicate rather than interleave the
        # half-width table along the final dimension.
        cos = cos.unsqueeze(0).unsqueeze(2).repeat(1, 1, 1, 2)
        sin = sin.unsqueeze(0).unsqueeze(2).repeat(1, 1, 1, 2)
        x_rot = torch_npu.npu_rotary_mul(x_rot.unsqueeze(0), cos, sin).squeeze(0)

    return torch.cat((x_rot, x_pass), dim=-1)


def _npu_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use Ascend RMSNorm and RoPE fused primitives for MiniMax-H3 DiT."""
    import torch_npu

    q_norm = torch_npu.npu_rms_norm(q, q_weight, epsilon=eps)[0]
    k_norm = torch_npu.npu_rms_norm(k, k_weight, epsilon=eps)[0]
    return (
        _npu_apply_rope_table(q_norm, rope_table, rotary_dim),
        _npu_apply_rope_table(k_norm, rope_table, rotary_dim),
    )


def _fused_cuda_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
) -> bool:
    return (
        HAS_TRITON
        and current_platform.is_cuda()
        and q.is_cuda
        and k.is_cuda
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and head_dim == _FUSED_HEAD_DIM
        and rotary_dim == _FUSED_ROTARY_DIM
    )


def _fused_npu_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
) -> bool:
    """Return whether the MiniMax-H3 Ascend fused-op contract is satisfied."""
    return (
        current_omni_platform.is_npu()
        and q.device.type == "npu"
        and k.device.type == "npu"
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and head_dim == _FUSED_HEAD_DIM
        and rotary_dim == _FUSED_ROTARY_DIM
    )


def _launch_fused_rms_norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    tokens, heads, head_dim = x.shape
    rotary_half = rope_table.shape[-1] // 2
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if tokens == 0:
        return out
    grid = (tokens, triton.cdiv(heads, _HEADS_PER_PROGRAM))
    _rms_norm_rope_kernel[grid](
        x,
        weight,
        rope_table,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        rope_table.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads=heads,
        head_dim=head_dim,
        rotary_half=rotary_half,
        eps=eps,
        heads_per_program=_HEADS_PER_PROGRAM,
        num_warps=8,
    )
    return out


def _fused_qk_norm_rope_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _fused_npu_supported(q, k, head_dim, rotary_dim):
        return _npu_qk_norm_rope(
            q,
            k,
            q_weight,
            k_weight,
            rope_table,
            eps,
            rotary_dim,
        )
    if not _fused_cuda_supported(q, k, head_dim, rotary_dim):
        return _eager_qk_norm_rope(
            q,
            k,
            q_weight,
            k_weight,
            rope_table,
            eps,
            head_dim,
            rotary_dim,
        )
    return (
        _launch_fused_rms_norm_rope(q, q_weight, rope_table, eps),
        _launch_fused_rms_norm_rope(k, k_weight, rope_table, eps),
    )


def _fused_qk_norm_rope_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del q_weight, k_weight, rope_table, eps, head_dim, rotary_dim
    return torch.empty_like(q), torch.empty_like(k)


_OMNI_OP_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, "fused_qk_norm_rope"):
    direct_register_custom_op(
        op_name="fused_qk_norm_rope",
        op_func=_fused_qk_norm_rope_impl,
        fake_impl=_fused_qk_norm_rope_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )


def _rocm_aiter_fused_qk_norm_rope_2way_available() -> bool:
    if not current_platform.is_rocm():
        return False

    if not rocm_aiter_ops.is_enabled():
        return False

    try:
        fused_qk_norm_rope_cache_quant = import_module("aiter.ops.fused_qk_norm_rope_cache_quant")
    except ImportError:
        return False
    return bool(
        getattr(
            fused_qk_norm_rope_cache_quant,
            "FUSED_QK_NORM_ROPE_2WAY_SUPPORTS_STRIDED_INPUTS",
            False,
        )
    )


_ROCM_AITER_FUSED_QK_NORM_ROPE_2WAY_AVAILABLE = _rocm_aiter_fused_qk_norm_rope_2way_available()


def prepare_rocm_aiter_fused_qk_norm_rope_2way_cache(
    freqs0: torch.Tensor,
    freqs1: torch.Tensor,
    dtype: torch.dtype,
    sequence_parallel_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Pack complex RoPE frequencies for the AITER two-stream operation."""
    if not _ROCM_AITER_FUSED_QK_NORM_ROPE_2WAY_AVAILABLE or sequence_parallel_size != 1 or dtype != torch.bfloat16:
        return None

    return (
        torch.cat((freqs0.real.to(dtype), freqs0.imag.to(dtype)), dim=-1),
        torch.cat((freqs1.real.to(dtype), freqs1.imag.to(dtype)), dim=-1),
    )


def rocm_aiter_fused_qk_norm_rope_2way_supported(
    q0: torch.Tensor,
    k0: torch.Tensor,
    q1: torch.Tensor,
    k1: torch.Tensor,
    q_weight0: torch.Tensor,
    k_weight0: torch.Tensor,
    q_weight1: torch.Tensor,
    k_weight1: torch.Tensor,
    rope_cache0: torch.Tensor,
    rope_cache1: torch.Tensor,
    sequence_parallel_size: int = 1,
) -> bool:
    """Return whether two streams satisfy the AITER kernel contract."""
    if not _ROCM_AITER_FUSED_QK_NORM_ROPE_2WAY_AVAILABLE or sequence_parallel_size != 1:
        return False

    activations = (q0, k0, q1, k1)
    if any(x.ndim != 4 or not x.is_cuda or x.dtype != torch.bfloat16 for x in activations):
        return False
    if any(x.device != q0.device for x in activations[1:]):
        return False
    if q0.shape[:2] != k0.shape[:2] or q1.shape[:2] != k1.shape[:2]:
        return False
    if q0.shape[0] != q1.shape[0] or q0.shape[0] == 0:
        return False
    if q0.shape[1] == 0 or q1.shape[1] == 0:
        return False
    if q0.shape[2:] != q1.shape[2:] or k0.shape[2:] != k1.shape[2:]:
        return False
    if any(x.shape[-1] != _AITER_QK_NORM_ROPE_2WAY_HEAD_DIM for x in activations):
        return False
    if any(x.stride(-1) != 1 or x.stride(-2) != _AITER_QK_NORM_ROPE_2WAY_HEAD_DIM for x in activations):
        return False

    weights = (q_weight0, k_weight0, q_weight1, k_weight1)
    if any(
        weight.shape != (_AITER_QK_NORM_ROPE_2WAY_HEAD_DIM,) or weight.dtype != q0.dtype or weight.device != q0.device
        for weight in weights
    ):
        return False

    frequencies = (
        (rope_cache0, q0.shape[1]),
        (rope_cache1, q1.shape[1]),
    )
    return all(
        frequency.shape == (tokens, _AITER_QK_NORM_ROPE_2WAY_HEAD_DIM)
        and frequency.dtype == q0.dtype
        and frequency.device == q0.device
        and frequency.is_contiguous()
        for frequency, tokens in frequencies
    )


def _rocm_aiter_fused_qk_norm_rope_2way_impl(
    q0: torch.Tensor,
    k0: torch.Tensor,
    q1: torch.Tensor,
    k1: torch.Tensor,
    q_weight0: torch.Tensor,
    k_weight0: torch.Tensor,
    q_weight1: torch.Tensor,
    k_weight1: torch.Tensor,
    rope_cache0: torch.Tensor,
    rope_cache1: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    fused_qk_norm_rope_2way = import_module("aiter.ops.fused_qk_norm_rope_cache_quant").fused_qk_norm_rope_2way

    batch_size = q0.shape[0]
    tokens0 = q0.shape[1]
    tokens1 = q1.shape[1]
    num_heads_q = q0.shape[2]
    num_heads_k = k0.shape[2]
    out_q = q0.new_empty((batch_size, tokens0 + tokens1, num_heads_q, _AITER_QK_NORM_ROPE_2WAY_HEAD_DIM))
    out_k = k0.new_empty((batch_size, tokens0 + tokens1, num_heads_k, _AITER_QK_NORM_ROPE_2WAY_HEAD_DIM))
    fused_qk_norm_rope_2way(
        q0,
        k0,
        q1,
        k1,
        q_weight0.contiguous(),
        k_weight0.contiguous(),
        q_weight1.contiguous(),
        k_weight1.contiguous(),
        rope_cache0,
        rope_cache1,
        batch_size,
        tokens0,
        tokens1,
        num_heads_q,
        num_heads_k,
        _AITER_QK_NORM_ROPE_2WAY_HEAD_DIM,
        True,
        eps,
        out_q,
        out_k,
    )
    return out_q, out_k


def _rocm_aiter_fused_qk_norm_rope_2way_fake(
    q0: torch.Tensor,
    k0: torch.Tensor,
    q1: torch.Tensor,
    k1: torch.Tensor,
    q_weight0: torch.Tensor,
    k_weight0: torch.Tensor,
    q_weight1: torch.Tensor,
    k_weight1: torch.Tensor,
    rope_cache0: torch.Tensor,
    rope_cache1: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del q_weight0, k_weight0, q_weight1, k_weight1, rope_cache0, rope_cache1, eps
    q_shape = (q0.shape[0], q0.shape[1] + q1.shape[1], q0.shape[2], q0.shape[3])
    k_shape = (k0.shape[0], k0.shape[1] + k1.shape[1], k0.shape[2], k0.shape[3])
    return q0.new_empty(q_shape), k0.new_empty(k_shape)


if not hasattr(torch.ops.vllm_omni, "rocm_aiter_fused_qk_norm_rope_2way"):
    direct_register_custom_op(
        op_name="rocm_aiter_fused_qk_norm_rope_2way",
        op_func=_rocm_aiter_fused_qk_norm_rope_2way_impl,
        fake_impl=_rocm_aiter_fused_qk_norm_rope_2way_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )


def rocm_aiter_fused_qk_norm_rope_2way(
    q0: torch.Tensor,
    k0: torch.Tensor,
    q1: torch.Tensor,
    k1: torch.Tensor,
    q_weight0: torch.Tensor,
    k_weight0: torch.Tensor,
    q_weight1: torch.Tensor,
    k_weight1: torch.Tensor,
    rope_cache0: torch.Tensor,
    rope_cache1: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse two Q/K RMSNorm and interleaved RoPE streams with AITER."""
    return torch.ops.vllm_omni.rocm_aiter_fused_qk_norm_rope_2way(
        q0,
        k0,
        q1,
        k1,
        q_weight0,
        k_weight0,
        q_weight1,
        k_weight1,
        rope_cache0,
        rope_cache1,
        eps,
    )


def fused_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    *,
    head_dim: int | None = None,
    rotary_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Q/K RMSNorm and packed non-interleaved RoPE."""
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError(f"q and k must be [tokens, heads, head_dim], got {q.shape} and {k.shape}")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError(f"q and k shapes are incompatible: {q.shape} and {k.shape}")
    if q.dtype != k.dtype or q.device != k.device:
        raise ValueError("q and k must have the same dtype and device")
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"Fused QK RMSNorm/RoPE requires floating inputs, got {q.dtype}")

    head_dim = q.shape[-1] if head_dim is None else head_dim
    rotary_dim = rope_table.shape[-1] if rotary_dim is None else rotary_dim
    if q.shape[-1] != head_dim:
        raise ValueError(f"Expected q/k head_dim={head_dim}, got {q.shape[-1]}")
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(f"rotary_dim must be even and in [2, {head_dim}], got {rotary_dim}")
    if q_weight.shape != (head_dim,) or k_weight.shape != (head_dim,):
        raise ValueError(f"Expected norm weights [{head_dim}], got {tuple(q_weight.shape)} and {tuple(k_weight.shape)}")
    if q_weight.device != q.device or k_weight.device != q.device:
        raise ValueError("Q/K norm weights must be on the activation device")
    if rope_table.device != q.device or rope_table.dtype != q.dtype:
        raise ValueError("rope_table must have the same dtype and device as q/k")
    if rope_table.shape != (q.shape[0], rotary_dim):
        raise ValueError(f"Expected rope_table [{q.shape[0]}, {rotary_dim}], got {tuple(rope_table.shape)}")

    q_weight = q_weight.contiguous()
    k_weight = k_weight.contiguous()
    rope_table = rope_table.contiguous()
    if _fused_npu_supported(q, k, head_dim, rotary_dim):
        return _npu_qk_norm_rope(
            q,
            k,
            q_weight,
            k_weight,
            rope_table,
            eps,
            rotary_dim,
        )
    if not _fused_cuda_supported(q, k, head_dim, rotary_dim):
        return _fused_qk_norm_rope_impl(
            q,
            k,
            q_weight,
            k_weight,
            rope_table,
            eps,
            head_dim,
            rotary_dim,
        )
    return torch.ops.vllm_omni.fused_qk_norm_rope(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        eps,
        head_dim,
        rotary_dim,
    )


__all__ = [
    "fused_qk_norm_rope",
    "prepare_rocm_aiter_fused_qk_norm_rope_2way_cache",
    "rocm_aiter_fused_qk_norm_rope_2way",
    "rocm_aiter_fused_qk_norm_rope_2way_supported",
]
