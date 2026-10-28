# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Experimental MI300X Qwen21 K conversion and functional K/V packing."""

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton


@lru_cache(maxsize=1)
def qualified_device() -> torch.device | None:
    """Resolve the pinned experiment stack outside model compilation."""
    try:
        triton_revision = version("triton")
    except PackageNotFoundError:
        return None
    if not (
        HAS_TRITON
        and str(torch.__version__) == "2.13.0+git733fca1"
        and torch.version.hip == "7.2.53211"
        and str(triton.__version__) == "3.8.0"
        and triton_revision == "3.8.0+git669b31ac"
    ):
        return None
    backend = torch.get_device_module("cuda")
    if not backend.is_available():
        return None
    index = backend.current_device()
    properties = backend.get_device_properties(index)
    if str(getattr(properties, "gcnArchName", "")).split(":")[0] != "gfx942":
        return None
    # The MI300X VF exposes no product-name string through this Torch build.
    if properties.total_memory < 180 * 1024**3:
        return None
    return torch.device("cuda", index)


def _compatible(kp, vp, k, v, key_dtype):
    return (
        HAS_TRITON
        and not torch.is_grad_enabled()
        and all(x.is_cuda and x.ndim == 4 and x.shape[0] == 1 and x.device == k.device for x in (kp, vp, k, v))
        and all(x.dtype == torch.bfloat16 for x in (kp, vp, v))
        and k.dtype == key_dtype
        and kp.shape == vp.shape
        and k.shape == v.shape
        and kp.shape[2:] == k.shape[2:] == (32, 128)
        and k.shape[1] > 0
    )


def producer_eligible(kp, vp, k, v):
    return _compatible(kp, vp, k, v, torch.bfloat16)


def eligible(kp, vp, k, v):
    return _compatible(kp, vp, k, v, torch.float32)


if HAS_TRITON:

    @triton.jit
    def _cast_pack(
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
        # Only target K is converted. Integer selection preserves prefix/V bits,
        # including signed zero and NaN payloads, without float promotion.
        target = k.to(tl.bfloat16).to(tl.uint16, bitcast=True)
        key = tl.where(before, kp.to(tl.uint16, bitcast=True), target)
        value = tl.where(before, vp.to(tl.uint16, bitcast=True), v.to(tl.uint16, bitcast=True))
        tl.store(k_out + i, key.to(tl.bfloat16, bitcast=True), valid)
        tl.store(v_out + i, value.to(tl.bfloat16, bitcast=True), valid)


def _launch(kp, vp, k, v):
    p, s, h, d = kp.shape[1], k.shape[1], k.shape[2], k.shape[3]
    ko = kp.new_empty((1, p + s, h, d))
    vo = vp.new_empty((1, p + s, h, d))
    _cast_pack[(triton.cdiv((p + s) * h * d, 256),)](
        kp, vp, k, v, ko, vo, p, s, h, d, *kp.stride()[1:], *vp.stride()[1:], *k.stride()[1:], *v.stride()[1:], 256
    )
    return ko, vo


@torch.library.custom_op("vllm_omni::qwen21_kv_cast_pack_experiment", mutates_args=())
def pack(kp: torch.Tensor, vp: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if k.device != qualified_device() or not eligible(kp, vp, k, v):
        raise ValueError("Qwen21 cast packing requires the pinned MI300X stack, FP32 K and BF16 prefix/V")
    return _launch(kp, vp, k, v)


@pack.register_fake
def _fake(kp, vp, k, v):
    if not eligible(kp, vp, k, v):
        raise ValueError("Qwen21 cast packing requires FP32 K and BF16 prefix/V with batch one and heads 32x128")
    shape = (1, kp.shape[1] + k.shape[1], k.shape[2], k.shape[3])
    return kp.new_empty(shape), vp.new_empty(shape)
