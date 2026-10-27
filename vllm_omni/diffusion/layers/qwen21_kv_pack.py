# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Experimental copy-only native Qwen21 attention packing."""

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:

    @triton.jit
    def _pack(
        kp_ptr,
        vp_ptr,
        k_ptr,
        v_ptr,
        k_out,
        v_out,
        prefix_len: tl.constexpr,
        target_len: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        kp_stride_1: tl.constexpr,
        kp_stride_2: tl.constexpr,
        kp_stride_3: tl.constexpr,
        vp_stride_1: tl.constexpr,
        vp_stride_2: tl.constexpr,
        vp_stride_3: tl.constexpr,
        k_stride_1: tl.constexpr,
        k_stride_2: tl.constexpr,
        k_stride_3: tl.constexpr,
        v_stride_1: tl.constexpr,
        v_stride_2: tl.constexpr,
        v_stride_3: tl.constexpr,
        block_size: tl.constexpr,
    ):
        i = tl.program_id(0) * block_size + tl.arange(0, block_size)
        t = i // (num_heads * head_dim)
        h = i // head_dim % num_heads
        d = i % head_dim
        before = t < prefix_len
        valid = t < prefix_len + target_len
        kp = tl.load(kp_ptr + t * kp_stride_1 + h * kp_stride_2 + d * kp_stride_3, valid & before, 0)
        vp = tl.load(vp_ptr + t * vp_stride_1 + h * vp_stride_2 + d * vp_stride_3, valid & before, 0)
        k = tl.load(k_ptr + (t - prefix_len) * k_stride_1 + h * k_stride_2 + d * k_stride_3, valid & ~before, 0)
        v = tl.load(v_ptr + (t - prefix_len) * v_stride_1 + h * v_stride_2 + d * v_stride_3, valid & ~before, 0)
        tl.store(k_out + i, tl.where(before, kp, k), valid)
        tl.store(v_out + i, tl.where(before, vp, v), valid)


@torch.library.custom_op("vllm_omni::qwen21_kv_pack_experiment", mutates_args=())
def pack(kp: torch.Tensor, vp: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not eligible(kp, vp, k, v):
        raise ValueError("Qwen21 packing requires compatible native BF16 batch-one GPU tensors under no_grad")
    p, s, h, d = kp.shape[1], k.shape[1], k.shape[2], k.shape[3]
    ko = k.new_empty((1, p + s, h, d))
    vo = torch.empty_like(ko)
    _pack[(triton.cdiv((p + s) * h * d, 256),)](
        kp, vp, k, v, ko, vo, p, s, h, d, *kp.stride()[1:], *vp.stride()[1:], *k.stride()[1:], *v.stride()[1:], 256
    )
    return ko, vo


@pack.register_fake
def _fake(kp, vp, k, v):
    if not eligible(kp, vp, k, v):
        raise ValueError("Qwen21 packing requires compatible native BF16 batch-one GPU tensors under no_grad")
    shape = (1, kp.shape[1] + k.shape[1], k.shape[2], k.shape[3])
    return k.new_empty(shape), v.new_empty(shape)


def eligible(kp, vp, k, v):
    return (
        HAS_TRITON
        and not torch.is_grad_enabled()
        and all(
            x.is_cuda and x.dtype == torch.bfloat16 and x.ndim == 4 and x.shape[0] == 1 and x.device == k.device
            for x in (kp, vp, k, v)
        )
        and kp.shape == vp.shape
        and k.shape == v.shape
        and kp.shape[2:] == k.shape[2:]
        and k.shape[1] > 0
        and k.shape[2] > 0
        and k.shape[3] > 0
    )
