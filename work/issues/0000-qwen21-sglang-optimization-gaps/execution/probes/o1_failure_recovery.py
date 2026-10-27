# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Real checkpoint encoder-only failure/recovery probe. GPU execution is explicit.

The CLI consumes saved B0 encoder inputs/outputs, never frozen encoder results
as candidate inputs. It does not qualify DiT residency or full-server lifecycle.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch


class InjectedBlockError(RuntimeError):
    pass


def equal(a, b):
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and bool(torch.isfinite(a).all())
        and bool(torch.isfinite(b).all())
        and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    )


def probe(pipeline, encode, expected, output_path):
    encoder = pipeline.text_encoder
    paths = ("model.language_model.layers", "model.visual.blocks", "model.visual.deepstack_merger_list")
    groups = getattr(encoder, "_omni_layerwise_block_groups", [])
    if len(groups) != 3:
        raise ValueError("Actual three-stack offload installation required")
    rows = []
    with torch.inference_mode():
        initial = encode()
        if not equal(initial, expected.to(initial.device)):
            raise ValueError("Actual encoder does not match saved B0 before failure injection")
        for path in paths:
            block = encoder.get_submodule(path)[0]
            registry = getattr(block, "_hook_registry", None)
            if registry is None or "layerwise_offload" not in registry._hooks:
                raise ValueError("Missing actual offload hook")
            original = block._omni_original_forward
            count = [0]

            def fail(*args, **kwargs):
                count[0] += 1
                # Dispatch already ran prefetch hooks. Raise within original
                # block body so ordinary offload post_forward is truly skipped.
                raise InjectedBlockError(path)

            try:
                block._omni_original_forward = fail
                try:
                    encode()
                except InjectedBlockError:
                    pass
                else:
                    raise AssertionError("Named text/vision/deepstack block never executed")
            finally:
                block._omni_original_forward = original
            resident_after_error = sum(h.is_materialized for h in encoder._omni_layerwise_hooks)
            recovery_error = None
            exact = False
            try:
                recovered = encode()
                exact = equal(recovered, expected.to(recovered.device))
            except Exception as error:
                recovery_error = repr(error)
            rows.append(
                {
                    "stack": path,
                    "injected_calls": count[0],
                    "materialized_blocks_after_error": resident_after_error,
                    "next_request_exact": exact,
                    "recovery_error": recovery_error,
                }
            )
            if recovery_error:
                break
    report = {
        "status": "PASS"
        if len(rows) == 3 and all(r["next_request_exact"] and r["injected_calls"] for r in rows)
        else "FAIL",
        "rows": rows,
        "scope": (
            "Actual encoder exception and immediate next-call recovery only; "
            "not cancellation/sleep/shutdown qualification"
        ),
        "post_forward_cleanup": (
            "Generic post_forward is skipped by injected error; retained blocks are recorded, "
            "not automatically described as released"
        ),
    }
    if output_path is not None:
        output = Path(output_path)
        if output.exists():
            raise FileExistsError(output)
        output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    report = {
        "status": "FAIL",
        "scope": "actual encoder-only",
        "dit_loaded": False,
        "checkpoint": args.checkpoint,
        "native_default_dtype": "torch.bfloat16",
    }
    encoder = None
    disable = None
    try:
        from transformers import Qwen3VLForConditionalGeneration
        from vllm.utils.torch_utils import set_default_torch_dtype

        from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import (
            _initialize_encoder_rope_on_device,
        )
        from vllm_omni.diffusion.offloader.layerwise_backend import (
            disable_plan_encoder_layerwise_offload,
            enable_plan_encoder_layerwise_offload,
        )
        from vllm_omni.diffusion.offloader.plan_resolver import BlockStack, ResolvedComponent
        from vllm_omni.platforms import current_omni_platform

        disable = disable_plan_encoder_layerwise_offload
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required; this CLI must not be run in the baseline queue")
        device = torch.device("cuda", current_omni_platform.current_device())
        captures = {}
        evidence = []
        for case in ("t2i-accuracy", "it2i-accuracy"):
            pair = []
            for suffix in ("input", "output"):
                path = args.baseline_captures / case / f"encoder-0-{suffix}.pt"
                evidence.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                pair.append(torch.load(path, map_location="cpu", weights_only=True))
            captures[case] = pair
        report["baseline_evidence"] = evidence
        if captures["t2i-accuracy"][0].get("pixel_values") is not None:
            raise ValueError("T2I capture unexpectedly includes vision inputs")
        if captures["it2i-accuracy"][0].get("pixel_values") is None:
            raise ValueError("Editing capture must exercise vision")
        # Match B0 native loader's default dtype context, without a forced
        # from_pretrained dtype override or altered attention implementation.
        with set_default_torch_dtype(torch.bfloat16), torch.device("cpu"):
            encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                args.checkpoint, subfolder="text_encoder", local_files_only=True
            ).to("cpu")
        encoder.eval()
        with set_default_torch_dtype(torch.bfloat16):
            _initialize_encoder_rope_on_device(encoder, device)
        report["rotary_initialization"] = "Actual pipeline helper rebuilds rotary modules on compute device"
        report["parameter_dtypes"] = sorted({str(p.dtype) for p in encoder.parameters()})
        paths = ("model.language_model.layers", "model.visual.blocks", "model.visual.deepstack_merger_list")
        stacks = tuple(BlockStack(tuple(encoder.get_submodule(path))) for path in paths)
        report["stack_lengths"] = {path: len(stack.blocks) for path, stack in zip(paths, stacks)}
        component = ResolvedComponent("text_encoder", encoder, True, stacks=stacks)
        stream = current_omni_platform.Stream()
        if not enable_plan_encoder_layerwise_offload(component, device=device, stream=stream, pin_memory=True):
            raise RuntimeError("Three-stack installation did not activate")

        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(device=device)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return tuple(move(item) for item in value)
            if isinstance(value, list):
                return [move(item) for item in value]
            return value

        inputs = {case: move(pair[0]) for case, pair in captures.items()}

        def encode(case):
            handle = encoder.model.language_model.norm.register_forward_hook(lambda module, args, output: args[0])
            try:
                return encoder(**inputs[case]).hidden_states[-1]
            finally:
                handle.remove()

        sequence = []
        report["request_sequence"] = sequence
        # A second T2I followed by edit also tests skipping previously used
        # vision rings, then reusing them on the subsequent editing request.
        with torch.inference_mode():
            for case in ("t2i-accuracy", "it2i-accuracy", "t2i-accuracy", "it2i-accuracy"):
                actual = encode(case)
                exact = equal(actual, captures[case][1]["hidden_states"].to(device))
                sequence.append({"case": case, "exact": exact})
                if not exact:
                    raise AssertionError(f"Saved B0 mismatch: {case}")
        result = probe(
            SimpleNamespace(text_encoder=encoder),
            lambda: encode("it2i-accuracy"),
            captures["it2i-accuracy"][1]["hidden_states"],
            None,
        )
        report.update(result)
        report["scope"] = "actual encoder-only; no DiT, cancellation, sleep/wake or server qualification"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = repr(error)
    finally:
        if encoder is not None and disable is not None:
            try:
                current_omni_platform.synchronize()
                disable(encoder, restore_weights=False)
            except Exception as error:
                report["status"] = "FAIL"
                report["cleanup_error"] = repr(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": str(args.output)}))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
