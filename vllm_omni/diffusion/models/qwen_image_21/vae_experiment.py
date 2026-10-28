# SPDX-License-Identifier: Apache-2.0
"""Independent eager Qwen21 decoder experiment; qualification is empirical.

Counters are diagnostic only. Qualification uses two real invocations per
signature, not a proof for unseen values. Replay additional distinct inputs.
No hooks, compilation or CUDA/HIP capture are bypassed.
"""
from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F

COUNTS = Counter()
DISABLED = False
VERIFY_EVERY_CALL = False


def eligible(owner, x, *modules):
    if (DISABLED or not getattr(owner, "_qwen_decoder_experiment", False)
            or torch.is_grad_enabled() or torch.compiler.is_compiling()
            or x.device.type != "cuda" or x.dtype != torch.bfloat16
            or torch.cuda.is_current_stream_capturing()):
        return False
    if nn.modules.module._global_forward_pre_hooks or nn.modules.module._global_forward_hooks:
        return False
    return not any(m.training or m._forward_hooks or m._forward_pre_hooks
                   or "forward" in m.__dict__ for m in modules)


def same(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype and a.stride() == b.stride()
            and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
            and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def signature(modules, tensors, extra=()):
    weights = tuple((id(p), p.data_ptr(), p._version, p.device, p.dtype, tuple(p.shape), p.stride())
                    for m in modules for p in m.parameters())
    return (tuple((t.device, t.dtype, tuple(t.shape), t.stride()) for t in tensors), weights,
            torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"),
            torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32,
            torch.are_deterministic_algorithms_enabled(), extra)


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

from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath.triton_data_movement import dup_up3d_add

# Diagnostic-only switch: qualify shortcut gather-add before bias deferral.
ADD_ONLY = False


def bias_upsample(owner, x, source, first_chunk):
    up, shortcut = owner.upsampler, owner.avg_shortcut
    reference = lambda: up(x) + shortcut(source, first_chunk=first_chunk)
    modules = tuple(up.modules()) + (shortcut,)
    if (not eligible(owner, x, *modules) or x.ndim != 5 or x.shape[2] != 1
            or type(up).__name__ != "QwenImage21Resample"
            or type(shortcut).__name__ != "QwenImage21DupUp3D"
            or up.mode not in ("upsample2d", "upsample3d")
            or type(up.resample) is not nn.Sequential or len(up.resample) != 2
            or type(up.resample[1]) is not nn.Conv2d
            or up.resample[1].padding_mode != "zeros"):
        return reference()
    conv = up.resample[1]
    if conv.bias is None:
        return reference()
    def candidate():
        if ADD_ONLY:
            main, bias = up(x), None
        else:
            # Preserve Qwen's FP32 nearest-exact round trip and ordinary Conv2d.
            b, c, t, h, w = x.shape
            flat = x.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w)
            flat = up.resample[0](flat)
            flat = F.conv2d(flat, conv.weight, None, conv.stride, conv.padding,
                            conv.dilation, conv.groups)
            main = flat.view(b, t, *flat.shape[1:]).permute(0, 2, 1, 3, 4)
            bias = conv.bias.to(main.dtype)
        return dup_up3d_add(main, source, shortcut.factor_t, shortcut.factor_s,
                            shortcut.repeats, first_chunk, main_bias=bias)
    key = signature((up,), (x, source), (first_chunk, ADD_ONLY, shortcut.factor_t,
                    shortcut.factor_s, shortcut.repeats, conv.stride, conv.padding, conv.dilation, conv.groups))
    return qualified(up, key, candidate, reference, "bias_upsample")
