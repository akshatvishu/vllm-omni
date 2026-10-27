# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Saved-checkpoint decoder qualification, never a fallback-only success.

Bundle (torch.save): {"cases": [{"name": str, "input": Tensor,
"input_kind": "decoder" | "vae", "expected_preclamp": Tensor,
"first_chunk": bool}, ...]}. Decoder input is post_quant_conv's actual output;
vae input is the actual denormalized input before post_quant_conv. Save full and
edge-tile calls separately. Expected output is decoder.forward BEFORE clamp.
No random/tiny weights substitute for the actual checkpoint.

Import replay_callable to test captured callable modules without loading another
VAE. Hooks/capture are diagnostic only; timed serving must use no replay wrapper.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def compare(actual, expected):
    same_shape = actual.shape == expected.shape and actual.dtype == expected.dtype
    finite = bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all())
    exact = same_shape and torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
    return dict(
        exact=exact,
        finite=finite,
        shape=list(actual.shape),
        dtype=str(actual.dtype),
        stride=list(actual.stride()),
        expected_stride=list(expected.stride()),
        max_abs=float((actual.float() - expected.float()).abs().max()) if same_shape else None,
    )


def replay_callable(call, cases, experiment, *, device, dtype):
    """call(input, case) returns PRECLAMP output; preserve its incumbent context."""
    results = []
    experiment.VERIFY_EVERY_CALL = True
    try:
        for case in cases:
            if case["input"].dtype != dtype or case["expected_preclamp"].dtype != dtype:
                raise ValueError("Saved B0 dtype must match replay; do not recast reference evidence")
            x = case["input"].to(device=device)
            expected = case["expected_preclamp"].to(device=device)
            original = x.clone()
            before = Counter(experiment.COUNTS)
            with torch.inference_mode():
                actual = call(x, case)
            stats = compare(actual, expected)
            counts = dict(Counter(experiment.COUNTS) - before)
            stats.update(
                name=case["name"],
                counts=counts,
                input_unchanged=torch.equal(x.contiguous().view(torch.uint8), original.contiguous().view(torch.uint8)),
            )
            stats["activated"] = any(k.endswith("/fused") and v > 0 for k, v in counts.items())
            stats["qualified"] = (
                stats["exact"]
                and stats["finite"]
                and stats["input_unchanged"]
                and stats["activated"]
                and not any(k.endswith("/mismatch") and v for k, v in counts.items())
            )
            results.append(stats)
    finally:
        experiment.VERIFY_EVERY_CALL = False
    return results


def reset_verdicts(module):
    for child in module.modules():
        if hasattr(child, "_qwen_fusion_verdicts"):
            child._qwen_fusion_verdicts.clear()


def wrong_kernel_check(module, call, case, exp, *, device, dtype):
    """Negative control must see rejection; equality through ineligibility fails."""
    if hasattr(exp, "epilogue"):
        owner, attr = exp.epilogue, "rms_norm_scale"
    elif hasattr(exp, "add_bias_residual"):
        owner, attr = exp, "add_bias_residual"
    else:
        owner, attr = exp, "dup_up3d_add"
    original = getattr(owner, attr)

    def wrong(*args, **kwargs):
        out = original(*args, **kwargs)
        # Do not corrupt baseline norm calls used by V1's reference.
        if owner is getattr(exp, "epilogue", None) and not kwargs.get("silu", False):
            return out
        return None if out is None else out + 1

    reset_verdicts(module)
    before = Counter(exp.COUNTS)
    try:
        setattr(owner, attr, wrong)
        result = replay_callable(call, [case], exp, device=device, dtype=dtype)[0]
    finally:
        setattr(owner, attr, original)
        reset_verdicts(module)
    changes = Counter(exp.COUNTS) - before
    return dict(
        exact_fallback=result["exact"],
        mismatch_seen=any(k.endswith("/mismatch") and v for k, v in changes.items()),
        counts=dict(changes),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Frozen local pipeline snapshot; loads vae subfolder")
    p.add_argument("--bundle", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--add-only", action="store_true", help="V3 localization only, never V3 acceptance")
    a = p.parse_args()
    from vllm_omni.diffusion.models.qwen_image_21 import vae_experiment as exp
    from vllm_omni.diffusion.models.qwen_image_21.autoencoder_kl_qwenimage21 import (
        AutoencoderKLQwenImage21,
        _cudnn_deterministic,
    )

    dtype = getattr(torch, a.dtype)
    vae = (
        AutoencoderKLQwenImage21.from_pretrained(
            a.checkpoint, subfolder="vae", local_files_only=True, torch_dtype=dtype
        )
        .to(a.device)
        .eval()
    )
    bundle = torch.load(a.bundle, map_location="cpu", weights_only=True)
    cases = bundle["cases"]
    if len(cases) < 2 or torch.equal(cases[0]["input"], cases[1]["input"]):
        raise ValueError("At least two distinct saved actual decoder inputs are required")
    if a.add_only:
        if not hasattr(exp, "ADD_ONLY"):
            raise ValueError("add-only requires V3")
        exp.ADD_ONLY = True
    keys = tuple(vae.state_dict())

    def call(x, case):
        with _cudnn_deterministic():
            if case.get("input_kind", "decoder") == "vae":
                x = vae.post_quant_conv(x)
            return vae.decoder(x, first_chunk=case.get("first_chunk", True))

    rows = replay_callable(call, cases, exp, device=a.device, dtype=dtype)
    negative = wrong_kernel_check(vae, call, cases[0], exp, device=a.device, dtype=dtype)
    result = dict(
        cases=rows,
        negative_control=negative,
        state_keys_preserved=keys == tuple(vae.state_dict()),
        add_only_diagnostic=a.add_only,
        torch_version=torch.__version__,
        hip=torch.version.hip,
        checkpoint=str(Path(a.checkpoint).resolve()),
        bundle=str(Path(a.bundle).resolve()),
        caveat="Decoder-call replay only; tiled blend, full request and timed performance require common harness.",
    )
    passed = (
        all(r["qualified"] for r in rows)
        and negative["exact_fallback"]
        and negative["mismatch_seen"]
        and result["state_keys_preserved"]
    )
    result["diagnostic_passed"] = passed
    result["qualified"] = passed and not a.add_only
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
