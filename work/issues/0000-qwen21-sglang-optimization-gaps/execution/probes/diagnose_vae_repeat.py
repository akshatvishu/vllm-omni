# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Explicit GPU diagnostic: real VAE incumbent A/A followed by verified candidate.

Consumes replay_vae.py's saved actual decoder-input bundle. Runs every case,
including edge tiles; never substitutes random data or weakens exact acceptance.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from vllm_omni.platforms import current_omni_platform


def compare(a, b):
    matching = a.shape == b.shape and a.dtype == b.dtype
    finite = bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
    delta = a.float() - b.float() if matching else None
    return {
        "exact": matching and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)),
        "finite": finite,
        "max_abs": float(delta.abs().max()) if delta is not None and finite else None,
        "rmse": float(delta.square().mean().sqrt()) if delta is not None and finite else None,
        "different_elements": int((a != b).sum()) if matching else None,
    }


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite diagnostic evidence")
    report = {
        "status": "FAIL",
        "qualified": False,
        "cases": [],
        "scope": "Actual decoder-call diagnostic only; no full-model or performance qualification",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "bundle": str(args.bundle.resolve()),
        "torch_version": torch.__version__,
        "hip": torch.version.hip,
    }
    exp = None
    original_flags = None
    try:
        from vllm_omni.diffusion.models.qwen_image_21 import vae_experiment as exp
        from vllm_omni.diffusion.models.qwen_image_21.autoencoder_kl_qwenimage21 import (
            AutoencoderKLQwenImage21,
            _cudnn_deterministic,
        )

        original_flags = exp.DISABLED, exp.VERIFY_EVERY_CALL
        report["bundle_sha256"] = digest(args.bundle)
        cases = torch.load(args.bundle, map_location="cpu", weights_only=True)["cases"]
        if len(cases) < 2 or torch.equal(cases[0]["input"], cases[1]["input"]):
            raise ValueError("At least first two distinct saved actual inputs are required")
        if any(case.get("input_kind", "decoder") != "decoder" for case in cases):
            raise ValueError("This probe requires captured actual decoder inputs")
        vae = (
            AutoencoderKLQwenImage21.from_pretrained(
                args.checkpoint,
                subfolder="vae",
                local_files_only=True,
                torch_dtype=torch.bfloat16,
            )
            .to(args.device)
            .eval()
        )
        initial_keys = tuple(vae.state_dict())
        report["backend_flags"] = {
            name: getattr(torch.backends.cudnn, name)
            for name in ("enabled", "benchmark", "deterministic", "allow_tf32")
        }
        report["context"] = "Incumbent _cudnn_deterministic context for every invocation; no flag overrides"
        with torch.inference_mode():
            for case in cases:
                if case["input"].dtype != torch.bfloat16 or case["expected_preclamp"].dtype != torch.bfloat16:
                    raise ValueError("Saved BF16 evidence required; no dtype conversion")
                x = case["input"].to(device=args.device)
                expected = case["expected_preclamp"].to(device=args.device)
                saved_x = x.clone()
                row = {
                    "name": case["name"],
                    "input_shape": list(x.shape),
                    "input_stride": list(x.stride()),
                    "input_storage_offset": x.storage_offset(),
                    "first_chunk": case.get("first_chunk", True),
                }
                report["cases"].append(row)
                outputs = []
                for label, disabled in (("incumbent_a", True), ("incumbent_b", True), ("candidate", False)):
                    exp.DISABLED = disabled
                    exp.VERIFY_EVERY_CALL = True
                    for module in vae.modules():
                        if hasattr(module, "_qwen_fusion_verdicts"):
                            module._qwen_fusion_verdicts.clear()
                    before = Counter(exp.COUNTS)
                    with _cudnn_deterministic():
                        value = vae.decoder(x, first_chunk=case.get("first_chunk", True))
                    current_omni_platform.synchronize()
                    outputs.append(value.detach().clone())
                    row[label + "_counts"] = dict(Counter(exp.COUNTS) - before)
                a, b, candidate = outputs
                row["incumbent_a_vs_b"] = compare(a, b)
                row["candidate_vs_b"] = compare(candidate, b)
                row["incumbent_a_vs_saved_b0"] = compare(a, expected)
                row["incumbent_b_vs_saved_b0"] = compare(b, expected)
                row["candidate_vs_saved_b0"] = compare(candidate, expected)
                row["input_unchanged"] = compare(x, saved_x)["exact"]
                counts = row["candidate_counts"]
                row["activated"] = any(k.endswith("/fused") and v > 0 for k, v in counts.items())
                row["operation_mismatch"] = any(k.endswith("/mismatch") and v > 0 for k, v in counts.items())
                repeatable = row["incumbent_a_vs_b"]["exact"] and row["incumbent_a_vs_b"]["finite"]
                row["classification"] = (
                    "operation verifier mismatch"
                    if row["operation_mismatch"]
                    else "incumbent not repeatable; candidate whole-decoder difference inconclusive"
                    if not repeatable
                    else "candidate differs from repeatable incumbent"
                    if not row["candidate_vs_b"]["exact"]
                    else "candidate fallback-only; not qualified"
                    if not row["activated"]
                    else "local exact agreement; saved B0 differs"
                    if not row["candidate_vs_saved_b0"]["exact"]
                    else "local and saved exact agreement"
                )
                row["qualified"] = (
                    repeatable
                    and row["activated"]
                    and not row["operation_mismatch"]
                    and row["input_unchanged"]
                    and all(
                        row[key]["exact"] and row[key]["finite"] for key in ("candidate_vs_b", "candidate_vs_saved_b0")
                    )
                )
        report["classification_counts"] = dict(Counter(row["classification"] for row in report["cases"]))
        report["candidate_operation_counts"] = dict(
            sum((Counter(row["candidate_counts"]) for row in report["cases"]), Counter())
        )
        report["state_keys_preserved"] = initial_keys == tuple(vae.state_dict())
        report["qualified"] = report["state_keys_preserved"] and all(row["qualified"] for row in report["cases"])
        report["status"] = "PASS" if report["qualified"] else "FAIL"
    except Exception as exc:
        report["error"] = repr(exc)
    finally:
        if exp is not None and original_flags is not None:
            exp.DISABLED, exp.VERIFY_EVERY_CALL = original_flags
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "report": str(args.output)}))
    return 0 if report["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
