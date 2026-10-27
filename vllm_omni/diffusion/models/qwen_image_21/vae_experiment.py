# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent eager Qwen21 decoder experiment; qualification is empirical.

Counters are diagnostic only. Qualification uses two real invocations per
signature, not a proof for unseen values. Replay additional distinct inputs.
No hooks, compilation or CUDA/HIP capture are bypassed.
"""

from collections import Counter

import torch
from torch import nn

from vllm_omni.diffusion.layers import vae_norm_epilogue as epilogue
from vllm_omni.diffusion.layers.vae_rms_norm import _as_rows

COUNTS = Counter()
DISABLED = False
VERIFY_EVERY_CALL = False


def eligible(owner, x, *modules):
    if (
        DISABLED
        or not getattr(owner, "_qwen_decoder_experiment", False)
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or x.device.type != "cuda"
        or x.dtype != torch.bfloat16
        or torch.cuda.is_current_stream_capturing()
    ):
        return False
    if nn.modules.module._global_forward_pre_hooks or nn.modules.module._global_forward_hooks:
        return False
    return not any(m.training or m._forward_hooks or m._forward_pre_hooks or "forward" in m.__dict__ for m in modules)


def same(a, b):
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and a.stride() == b.stride()
        and bool(torch.isfinite(a).all())
        and bool(torch.isfinite(b).all())
        and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    )


def signature(modules, tensors, extra=()):
    weights = tuple(
        (id(p), p.data_ptr(), p._version, p.device, p.dtype, tuple(p.shape), p.stride())
        for m in modules
        for p in m.parameters()
    )
    return (
        tuple((t.device, t.dtype, tuple(t.shape), t.stride()) for t in tensors),
        weights,
        torch.is_autocast_enabled("cuda"),
        torch.get_autocast_dtype("cuda"),
        torch.backends.cudnn.enabled,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.allow_tf32,
        torch.are_deterministic_algorithms_enabled(),
        extra,
    )


def qualified(owner, key, candidate, reference, label):
    # Bounded module-owned metadata only; no activation/weight references retained.
    verdicts = getattr(owner, "_qwen_fusion_verdicts", None)
    if verdicts is None:
        verdicts = owner._qwen_fusion_verdicts = {}
    if len(verdicts) >= 32 and key not in verdicts:
        verdicts.clear()
    state = verdicts.get(key, 0)
    if state == -1:
        COUNTS[label + "/rejected"] += 1
        return reference()
    output = candidate()
    if output is None:
        COUNTS[label + "/unsupported"] += 1
        return reference()
    if state < 2 or VERIFY_EVERY_CALL:
        expected = reference()
        if not same(output, expected):
            verdicts[key] = -1
            COUNTS[label + "/mismatch"] += 1
            return expected
        verdicts[key] = min(2, state + 1)
        COUNTS[label + "/checked"] += 1
    COUNTS[label + "/fused"] += 1
    return output


def norm_silu(owner, norm, act, x):
    def reference():
        return act(norm(x))

    if (
        not eligible(owner, x, norm, act)
        or type(act) is not nn.SiLU
        or act.inplace
        or type(norm).__name__ != "QwenImage21RMS_norm"
        or not norm.channel_first
        or not isinstance(norm.bias, float)
        or norm.bias != 0.0
        or norm.gamma.dtype != x.dtype
        or norm.gamma.device != x.device
        or not epilogue.silu_epilogue_is_exact(x.device, x.dtype)
    ):
        return reference()

    def candidate():
        denom = torch.linalg.vector_norm(x, dim=1, keepdim=True, dtype=torch.float32).clamp_min_(1e-12)
        rows = _as_rows(x, denom)
        if rows is None:
            return None
        values, denominator, restore = rows
        out = epilogue.rms_norm_scale(values, denominator, norm.gamma.reshape(-1), norm.scale, silu=True)
        return None if out is None else restore(out)

    return qualified(norm, signature((norm,), (x,), (norm.scale,)), candidate, reference, "norm_silu")
