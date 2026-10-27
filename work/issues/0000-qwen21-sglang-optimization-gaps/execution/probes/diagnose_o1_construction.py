# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Separate Qwen21 encoder construction numerics from layer offload.

Actual checkpoint encoder only, no DiT. This diagnostic never edits model source
or treats buffer substitution as a production fix. Outputs and manifests are
saved beside --output. No timing claims are made.
"""

import argparse
import hashlib
import inspect
import json
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifacts = args.output.with_suffix(".artifacts")
    if args.output.exists() or artifacts.exists():
        parser.error("output or artifact directory already exists")
    artifacts.mkdir(parents=True)
    report = {
        "status": "ERROR",
        "scope": __doc__,
        "checkpoint": args.checkpoint,
        "phases": [],
        "buffer_substitution": {"attempted": False},
    }
    cpu_model = None
    offload_active = False

    def file_hash(path):
        h = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    try:
        import torch
        import transformers
        from transformers import Qwen3VLForConditionalGeneration
        from vllm.utils.torch_utils import set_default_torch_dtype

        from vllm_omni.diffusion.offloader.layerwise_backend import (
            disable_plan_encoder_layerwise_offload,
            enable_plan_encoder_layerwise_offload,
        )
        from vllm_omni.diffusion.offloader.plan_resolver import BlockStack, ResolvedComponent
        from vllm_omni.platforms import current_omni_platform

        if not torch.cuda.is_available():
            raise RuntimeError("requires one CUDA/ROCm GPU")
        device = torch.device("cuda", 0)
        current_omni_platform.set_device(device)
        source = Path(inspect.getfile(Qwen3VLForConditionalGeneration))
        report["runtime"] = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(0),
            "modeling_source": str(source),
            "modeling_sha256": file_hash(source),
            "script_sha256": file_hash(__file__),
            "default_dtype_during_load": "torch.bfloat16",
        }
        cases = ("t2i-accuracy", "it2i-accuracy")
        inputs, expected = {}, {}
        report["baseline_files"] = []

        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, dict):
                return {k: move(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(move(v) for v in value)
            return value

        for case in cases:
            paths = [args.baseline_captures / case / f"encoder-0-{kind}.pt" for kind in ("input", "output")]
            report["baseline_files"].extend({"path": str(p), "sha256": file_hash(p)} for p in paths)
            inputs[case] = move(torch.load(paths[0], map_location="cpu", weights_only=True))
            expected[case] = torch.load(paths[1], map_location="cpu", weights_only=True)["hidden_states"]
        if inputs["t2i-accuracy"].get("pixel_values") is not None:
            raise ValueError("T2I input unexpectedly contains pixels")
        if inputs["it2i-accuracy"].get("pixel_values") is None:
            raise ValueError("edit input must contain pixels")

        def tensor_hash(tensor):
            raw = tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy()
            return hashlib.sha256(raw).hexdigest()

        def manifest(model):
            nonpersistent = {
                f"{name}.{key}".lstrip(".")
                for name, module in model.named_modules()
                for key in module._non_persistent_buffers_set
            }
            rows = {}
            for kind, items in [("parameter", model.named_parameters()), ("buffer", model.named_buffers())]:
                for name, tensor in items:
                    rows[name] = {
                        "kind": kind,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "device": str(tensor.device),
                        "sha256": tensor_hash(tensor),
                        "nonpersistent": kind == "buffer" and name in nonpersistent,
                    }
            return rows

        def metrics(value, reference):
            a, b = value.detach().cpu(), reference.detach().cpu()
            if a.shape != b.shape or a.dtype != b.dtype:
                return {
                    "exact": False,
                    "error": "shape or dtype differs",
                    "shape": list(a.shape),
                    "reference_shape": list(b.shape),
                }
            delta = a.double() - b.double()
            return {
                "exact": torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)),
                "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                "different_values": int((a != b).sum()),
                "numel": a.numel(),
                "max_abs": float(delta.abs().max()),
                "rmse": float(delta.square().mean().sqrt()),
            }

        references = {}

        def run_phase(model, name, repeats=1):
            phase = {"name": name, "outputs": []}
            report["phases"].append(phase)
            with torch.inference_mode():
                for repeat in range(repeats):
                    for case in cases:
                        handle = model.model.language_model.norm.register_forward_hook(
                            lambda module, positional, output: positional[0]
                        )
                        try:
                            result = model(**inputs[case]).hidden_states[-1]
                            current_omni_platform.synchronize()
                            result = result.detach().cpu().clone()
                        finally:
                            handle.remove()
                        path = artifacts / f"{name}-{repeat}-{case}.pt"
                        torch.save({"hidden_states": result}, path)
                        row = {
                            "case": case,
                            "repeat": repeat,
                            "path": str(path),
                            "sha256": file_hash(path),
                            "versus_B0": metrics(result, expected[case]),
                        }
                        if case in references:
                            row["versus_gpu_constructed"] = metrics(result, references[case])
                        else:
                            references[case] = result
                        phase["outputs"].append(row)
            return phase

        def load(construction_device):
            # No dtype= override: reproduce the actual native ambient context.
            with set_default_torch_dtype(torch.bfloat16), torch.device(construction_device):
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    args.checkpoint, subfolder="text_encoder", local_files_only=True
                ).to(construction_device)
            return model.eval()

        gpu_model = load(device)
        gpu_manifest = manifest(gpu_model)
        (artifacts / "gpu-construction-manifest.json").write_text(json.dumps(gpu_manifest, indent=2) + "\n")
        report["encoder_config"] = gpu_model.config.to_dict()
        report["attention_implementations"] = {
            "model": getattr(gpu_model.config, "_attn_implementation", None),
            "vision": getattr(gpu_model.model.visual.config, "_attn_implementation", None),
            "text": getattr(gpu_model.model.language_model.config, "_attn_implementation", None),
        }
        run_phase(gpu_model, "gpu_constructed_resident", repeats=2)
        cpu_model = load(torch.device("cpu"))
        cpu_manifest = manifest(cpu_model)
        (artifacts / "cpu-construction-manifest.json").write_text(json.dumps(cpu_manifest, indent=2) + "\n")
        differences = [
            name
            for name in gpu_manifest
            if name not in cpu_manifest
            or any(gpu_manifest[name][key] != cpu_manifest[name][key] for key in ("sha256", "dtype", "shape", "kind"))
        ]
        differences.extend(name for name in cpu_manifest if name not in gpu_manifest)
        report["construction_differences"] = differences
        report["parameter_differences"] = [n for n in differences if gpu_manifest.get(n, {}).get("kind") == "parameter"]
        differing_buffers = [
            n
            for n in differences
            if gpu_manifest.get(n, {}).get("nonpersistent") and cpu_manifest.get(n, {}).get("nonpersistent")
        ]
        report["differing_nonpersistent_buffers"] = differing_buffers
        cpu_model.to(device)
        run_phase(cpu_model, "cpu_constructed_resident")
        paths = ("model.language_model.layers", "model.visual.blocks", "model.visual.deepstack_merger_list")
        report["stack_lengths"] = {p: len(cpu_model.get_submodule(p)) for p in paths}
        stream = current_omni_platform.Stream()

        def enable():
            cpu_model.to("cpu")
            stacks = tuple(BlockStack(tuple(cpu_model.get_submodule(p))) for p in paths)
            component = ResolvedComponent("text_encoder", cpu_model, True, stacks=stacks)
            if not enable_plan_encoder_layerwise_offload(component, device=device, stream=stream, pin_memory=True):
                raise RuntimeError("three-stack offload installation declined")

        enable()
        offload_active = True
        run_phase(cpu_model, "cpu_constructed_offload", repeats=2)
        current_omni_platform.synchronize()
        disable_plan_encoder_layerwise_offload(cpu_model, restore_weights=True)
        offload_active = False
        restored_manifest = manifest(cpu_model)
        report["post_offload_restore_differences"] = [
            name
            for name in cpu_manifest
            if restored_manifest.get(name, {}).get("sha256") != cpu_manifest[name]["sha256"]
        ]
        (artifacts / "post-offload-restore-manifest.json").write_text(json.dumps(restored_manifest, indent=2) + "\n")
        if report["post_offload_restore_differences"]:
            raise AssertionError("offload restoration changed tensor values; buffer substitution would be confounded")
        if differing_buffers:
            before = restored_manifest
            with torch.no_grad():
                target_buffers = dict(cpu_model.named_buffers())
                source_buffers = dict(gpu_model.named_buffers())
                for name in differing_buffers:
                    target, source_tensor = target_buffers[name], source_buffers[name]
                    if target.shape != source_tensor.shape or target.dtype != source_tensor.dtype:
                        raise ValueError(f"cannot copy differing buffer geometry/dtype: {name}")
                    target.copy_(source_tensor.to(target.device))
            after = manifest(cpu_model)
            actual_changed = [name for name in before if before[name]["sha256"] != after[name]["sha256"]]
            if set(actual_changed) != set(differing_buffers):
                raise AssertionError("buffer substitution changed an unexpected tensor set")
            report["buffer_substitution"] = {
                "attempted": True,
                "changed": actual_changed,
                "scope": "diagnostic substitution only, not a source fix",
            }
            cpu_model.to(device)
            run_phase(cpu_model, "gpu_buffers_cpu_constructed_resident")
            enable()
            offload_active = True
            run_phase(cpu_model, "gpu_buffers_cpu_constructed_offload", repeats=2)
        report["status"] = "COMPLETED"
        report["qualification"] = "Diagnostic only; inspect exact comparisons. Completion is not candidate approval."
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        if offload_active and cpu_model is not None:
            try:
                current_omni_platform.synchronize()
                disable_plan_encoder_layerwise_offload(cpu_model, restore_weights=False)
            except Exception:
                report["cleanup_error"] = traceback.format_exc()
                report["status"] = "ERROR"
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return 0 if report["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
