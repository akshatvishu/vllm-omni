# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803

"""Opt-in packed Qwen-Image QK RMSNorm/RoPE epilogue for ROCm gfx942.

The fixed launch settings preserve the accepted MI300X candidate. Numerical
validation applies to the tested stack; gfx942 also includes unvalidated
MI325X hardware. Other shapes and configurations use the model's existing path.
"""

from typing import TYPE_CHECKING

import torch
from torch.library import Library
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import QwenImageCrossAttention


if HAS_TRITON:

    @triton.jit
    def _epilogue(
        QKV,
        Q_WEIGHT,
        K_WEIGHT,
        COEFF,
        OUT_Q,
        OUT_K,
        OUT_V,
        ROWS: tl.constexpr,
        OUT_ROW: tl.constexpr,
        BLOCK_X: tl.constexpr,
        BLOCK_R: tl.constexpr,
        PERSISTENT: tl.constexpr,
        PLANE_START: tl.constexpr,
        PLANE_STEP: tl.constexpr,
    ):
        x = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)[:, None]
        plane = PLANE_START + tl.program_id(1) * PLANE_STEP
        token = x // 24
        head = x % 24
        valid = x < ROWS * 24
        r = tl.arange(0, 128)[None, :]
        src = token * 9216 + head * 128 + r
        dst = (token + OUT_ROW) * 3072 + head * 128 + r

        if plane == 2:
            # Keep V in BF16; no arithmetic or conversion on its payload.
            v = tl.load(QKV + src + 6144, valid, other=0)
            tl.store(OUT_V + dst, v, valid)
        else:
            if plane == 0:
                weight_ptr = Q_WEIGHT
                out_ptr = OUT_Q
            else:
                weight_ptr = K_WEIGHT
                out_ptr = OUT_K

            if PERSISTENT:
                value = tl.load(QKV + src + plane * 3072, valid, other=0).to(tl.float32)
                square = value * value
                square = tl.where(valid, square, 0.0)
                square_sum = tl.sum(square, 1)[:, None]
            else:
                # Preserve the captured image norm's partial-sum loop.
                rr = tl.arange(0, BLOCK_R)[None, :]
                partial = tl.full((BLOCK_X, BLOCK_R), 0.0, tl.float32)
                for offset in tl.range(0, 128, BLOCK_R, num_stages=2):
                    index = offset + rr
                    mask = valid & (index < 128)
                    value_part = tl.load(
                        QKV + token * 9216 + head * 128 + plane * 3072 + index,
                        mask,
                        other=0,
                        eviction_policy="evict_last",
                    ).to(tl.float32)
                    square = value_part * value_part
                    partial = tl.where(mask, partial + square, partial)
                square_sum = tl.sum(partial, 1)[:, None]
                value = tl.load(
                    QKV + src + plane * 3072,
                    valid,
                    other=0,
                    eviction_policy="evict_first",
                ).to(tl.float32)

            inv_rms = tl.rsqrt(square_sum / 128.0 + 1e-6)
            weight = tl.load(weight_ptr + r).to(tl.float32)
            normalized = ((value * inv_rms) * weight).to(tl.bfloat16).to(tl.float32)

            pair = tl.arange(0, 64)[None, :]
            coeff_base = token * 128 + pair * 2
            cos = tl.load(COEFF + coeff_base, valid, other=1.0)
            sin = tl.load(COEFF + coeff_base + 1, valid, other=0.0)
            cos = cos.to(tl.bfloat16).to(tl.float32)
            sin = sin.to(tl.bfloat16).to(tl.float32)
            even, odd = tl.split(tl.reshape(normalized, (BLOCK_X, 64, 2)))
            out_even = even * cos - odd * sin
            out_odd = even * sin + odd * cos
            rotated = tl.reshape(tl.join(out_even, out_odd), (BLOCK_X, 128))
            tl.store(out_ptr + dst, rotated, valid)


def qwen_image_qk_rope_supported() -> bool:
    """Resolve the hardware gate once when constructing attention."""
    if not HAS_TRITON or not current_omni_platform.is_rocm() or not torch.cuda.is_available():
        return False
    properties = torch.cuda.get_device_properties("cuda")
    return properties.gcnArchName.split(":")[0] == "gfx942"


def can_use_qwen_image_qk_rope(
    attn: "QwenImageCrossAttention",
    img_qkv: torch.Tensor,
    txt_qkv: torch.Tensor,
    img_coeff: torch.Tensor,
    txt_coeff: torch.Tensor,
) -> bool:
    """Check the packed-input contract after the opt-in and hardware gate."""
    if not attn.use_rocm_qk_rope or not attn.qk_norm or attn.eps != 1e-6:
        return False
    if attn.parallel_config is not None and (
        attn.parallel_config.sequence_parallel_size != 1 or attn.parallel_config.tensor_parallel_size != 1
    ):
        return False
    if (
        attn.head_dim != 128
        or attn.query_num_heads != 24
        or attn.kv_num_heads != 24
        or attn.add_query_num_heads != 24
        or attn.add_kv_num_heads != 24
    ):
        return False
    for value in (img_qkv, txt_qkv):
        if (
            value.ndim != 3
            or value.shape[0] != 1
            or value.shape[2] != 9216
            or value.dtype != torch.bfloat16
            or value.device.type != "cuda"
            or value.device != img_qkv.device
            or not value.is_contiguous()
        ):
            return False
    if img_qkv.shape[1] != 4096 or not 1 <= txt_qkv.shape[1] <= 128:
        return False
    for norm in (attn.norm_q, attn.norm_k, attn.norm_added_q, attn.norm_added_k):
        if (
            getattr(norm, "weight", None) is None
            or norm.weight.shape != (128,)
            or norm.weight.dtype != torch.bfloat16
            or norm.weight.device != img_qkv.device
            or not norm.weight.is_contiguous()
            or norm.eps != 1e-6
        ):
            return False
    for coeff, value in ((img_coeff, img_qkv), (txt_coeff, txt_qkv)):
        if (
            coeff.ndim != 2
            or coeff.shape[1] != 64
            or coeff.shape[0] < value.shape[1]
            or coeff.dtype != torch.complex64
            or coeff.device != img_qkv.device
            or not coeff.is_contiguous()
        ):
            return False
    return True


def _qwen_image_qk_rope_impl(
    img_qkv: torch.Tensor,
    txt_qkv: torch.Tensor,
    img_q_weight: torch.Tensor,
    img_k_weight: torch.Tensor,
    txt_q_weight: torch.Tensor,
    txt_k_weight: torch.Tensor,
    img_coeff: torch.Tensor,
    txt_coeff: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (1, img_qkv.shape[1] + txt_qkv.shape[1], 24, 128)
    outputs = tuple(img_qkv.new_empty(shape) for _ in range(3))
    launches = (
        (img_qkv, img_q_weight, img_k_weight, img_coeff, txt_qkv.shape[1], 2, False),
        (txt_qkv, txt_q_weight, txt_k_weight, txt_coeff, 0, 4, True),
    )
    for qkv, q_weight, k_weight, coeff, start, block_x, persistent in launches:
        rows = qkv.shape[1]
        _epilogue[(triton.cdiv(rows * 24, block_x), 3)](
            qkv,
            q_weight,
            k_weight,
            torch.view_as_real(coeff),
            *outputs,
            ROWS=rows,
            OUT_ROW=start,
            BLOCK_X=block_x,
            BLOCK_R=128,
            PERSISTENT=persistent,
            PLANE_START=0,
            PLANE_STEP=1,
            num_warps=1,
            num_stages=1,
            enable_fp_fusion=True,
        )
    return outputs


def _qwen_image_qk_rope_fake(
    img_qkv: torch.Tensor,
    txt_qkv: torch.Tensor,
    img_q_weight: torch.Tensor,
    img_k_weight: torch.Tensor,
    txt_q_weight: torch.Tensor,
    txt_k_weight: torch.Tensor,
    img_coeff: torch.Tensor,
    txt_coeff: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (1, img_qkv.shape[1] + txt_qkv.shape[1], 24, 128)
    return tuple(img_qkv.new_empty(shape) for _ in range(3))


_OMNI_OP_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, "qwen_image_qk_rope"):
    direct_register_custom_op(
        op_name="qwen_image_qk_rope",
        op_func=_qwen_image_qk_rope_impl,
        fake_impl=_qwen_image_qk_rope_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
        dispatch_key="CUDA",
    )


def qwen_image_qk_rope(
    img_qkv: torch.Tensor,
    txt_qkv: torch.Tensor,
    img_q_weight: torch.Tensor,
    img_k_weight: torch.Tensor,
    txt_q_weight: torch.Tensor,
    txt_k_weight: torch.Tensor,
    img_coeff: torch.Tensor,
    txt_coeff: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fresh text-first Q/K/V buffers for inputs accepted by the guard."""
    return torch.ops.vllm_omni.qwen_image_qk_rope(
        img_qkv, txt_qkv, img_q_weight, img_k_weight, txt_q_weight, txt_k_weight, img_coeff, txt_coeff
    )
