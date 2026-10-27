# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Bounded synthetic copy-path screen at saved Qwen shapes; no model timing claim.

Both arms include the FP32-to-BF16 K conversion observed in production generated
code. Reference lets Inductor fuse that conversion into K concatenation. K1
materializes BF16 K before the opaque packing operation. V is a strided QKV
view; outputs use the observed BHSD views of BSHD storage. Values are synthetic,
not captured model operands. This screen does not qualify denoising or images.
"""

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import subprocess
from collections import Counter
from pathlib import Path

import torch

from vllm_omni.diffusion.layers import qwen21_kv_pack


def reference(kp, vp, k32, v):
    k = k32.to(torch.bfloat16)
    return torch.cat((kp, k), 1).permute(0, 2, 1, 3), torch.cat((vp, v), 1).permute(0, 2, 1, 3)


def candidate(kp, vp, k32, v):
    k = k32.to(torch.bfloat16)
    ko, vo = qwen21_kv_pack.pack(kp, vp, k, v)
    return ko.permute(0, 2, 1, 3), vo.permute(0, 2, 1, 3)


def equal(left, right):
    return all(
        x.shape == y.shape
        and x.dtype == y.dtype
        and x.stride() == y.stride()
        and torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))
        for x, y in zip(left, right, strict=True)
    )


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-captures", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.checkpoint / "transformer/config.json").read_text())
    heads, dim = config["num_attention_heads"], config["attention_head_dim"]
    result = {
        "scope": "Synthetic compiled KV conversion/packing only, at saved conditional shapes; not request timing or model quality",
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": digest(__file__),
        "pack_sha256": digest(qwen21_kv_pack.__file__),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "gcn_arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None),
        "packages": {},
        "budget": {
            "paired_rounds": 12,
            "warm_replays_per_sample": 40,
            "flush_replays_per_sample": 1,
            "flush_bytes": 512 * 1024 * 1024,
        },
        "conditions": "Alternate AB/BA. Warm resident buffers; separately repeat after a 512 MiB write outside each timed region. No claim of guaranteed cache eviction.",
        "cases": [],
        "correctness_passed": False,
    }
    for package in ("aiter", "aotriton", "triton", "vllm"):
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = None
    if not str(result["gcn_arch"]).startswith("gfx942"):
        raise RuntimeError("This screen is pinned to MI300X/gfx942")
    seen = {"B0_path": [], "K1_path": []}

    def backend(name):
        def compile_graph(gm, inputs):
            seen[name].extend(str(node.target) for node in gm.graph.nodes)
            with torch._inductor.config.patch({"triton.cudagraphs": False, "triton.cudagraph_trees": False}):
                return torch._inductor.compile(gm, inputs)

        return compile_graph

    functions = {
        "B0_path": torch.compile(reference, backend=backend("B0_path"), fullgraph=True, dynamic=True),
        "K1_path": torch.compile(candidate, backend=backend("K1_path"), fullgraph=True, dynamic=True),
    }
    try:
        with torch.inference_mode():
            flush = torch.empty(512 * 1024 * 1024, device="cuda", dtype=torch.uint8)
            for context_path in sorted(args.baseline_captures.glob("*/context.pt")):
                context = torch.load(context_path, map_location="cpu", weights_only=True)
                mask = context["image_pad_mask"]
                prefix = int(mask.numel() + 3 * mask.sum().item())
                target = context["latents"].shape[1]
                torch.manual_seed(20261028)
                kp = torch.randn((1, prefix, heads, dim), device="cuda", dtype=torch.bfloat16)
                vp = torch.randn_like(kp)
                k32 = torch.randn((1, target, heads, dim), device="cuda", dtype=torch.float32)
                qkv = torch.randn((1, target, 3 * heads * dim), device="cuda", dtype=torch.bfloat16)
                v = qkv[:, :, 2 * heads * dim :].view(1, target, heads, dim)
                inputs = (kp, vp, k32, v)
                row = {
                    "case": context_path.parent.name,
                    "context_sha256": digest(context_path),
                    "prefix": prefix,
                    "target": target,
                    "heads": heads,
                    "head_dim": dim,
                    "input_kind": "synthetic values; conditional prefix/target lengths from saved context",
                    "inputs": [
                        {
                            "shape": list(x.shape),
                            "stride": list(x.stride()),
                            "offset": x.storage_offset(),
                            "dtype": str(x.dtype),
                        }
                        for x in inputs
                    ],
                    "checks": [],
                    "traces": {},
                    "timings": {},
                }
                result["cases"].append(row)
                graphs, outputs = {}, {}
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for fn in functions.values():
                        for _ in range(3):
                            fn(*inputs)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                for name, fn in functions.items():
                    graphs[name] = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graphs[name]):
                        outputs[name] = fn(*inputs)
                for iteration in range(2):
                    if iteration:
                        k32.add_(0.125)
                        v.add_(0.25)
                        vp.add_(0.5)
                    expected = reference(*inputs)
                    for name, graph in graphs.items():
                        graph.replay()
                        passed = equal(outputs[name], expected)
                        row["checks"].append(
                            {"arm": name, "changed_input": bool(iteration), "exact_bits_and_strides": passed}
                        )
                        if not passed:
                            raise AssertionError(f"{row['case']} {name} graph mismatch")
                for condition, repeats in (("warm", 40), ("after_512MiB_write", 1)):
                    pairs = []
                    for trial in range(12):
                        order = ["B0_path", "K1_path"] if trial % 2 == 0 else ["K1_path", "B0_path"]
                        sample = {"trial": trial, "order": order}
                        for name in order:
                            if condition != "warm":
                                flush.fill_(trial)
                            else:
                                graphs[name].replay()
                            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(repeats):
                                graphs[name].replay()
                            stop.record()
                            stop.synchronize()
                            sample[name] = start.elapsed_time(stop) * 1000 / repeats
                        pairs.append(sample)
                    row["timings"][condition] = {
                        "unit": "microseconds per graph replay; excludes setup and flush",
                        "pairs": pairs,
                        "reference_median_us": statistics.median(p["B0_path"] for p in pairs),
                        "candidate_median_us": statistics.median(p["K1_path"] for p in pairs),
                        "median_paired_candidate_over_reference": statistics.median(
                            p["K1_path"] / p["B0_path"] for p in pairs
                        ),
                    }
                for name, graph in graphs.items():
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                    ) as profile:
                        graph.replay()
                        torch.cuda.synchronize()
                    path = args.output / f"{row['case']}-{name}-trace.json"
                    profile.export_chrome_trace(str(path))
                    events = json.loads(path.read_text())["traceEvents"]
                    counts = Counter(e["name"] for e in events if e.get("cat") == "kernel")
                    row["traces"][name] = {"file": path.name, "sha256": digest(path), "kernel_counts": dict(counts)}
                    if counts["_pack.kd"] != (1 if name == "K1_path" else 0):
                        raise AssertionError(f"Unexpected packing activation in {name}: {dict(counts)}")
                    passed = equal(outputs[name], reference(*inputs))
                    row["checks"].append({"arm": name, "post_timing": True, "exact_bits_and_strides": passed})
                    if not passed:
                        raise AssertionError(f"{name} post-timing mismatch")
                row["kernel_structure_matches_observed_KV_boundary"] = all(
                    sum(trace["kernel_counts"].values()) == 2 for trace in row["traces"].values()
                )
                del graph, graphs, outputs, kp, vp, k32, qkv, v, inputs, expected
                print(
                    json.dumps(
                        {
                            "case": row["case"],
                            "timings": {
                                k: {n: v for n, v in x.items() if n != "pairs"} for k, x in row["timings"].items()
                            },
                        }
                    ),
                    flush=True,
                )
                (args.output / "screen.json").write_text(json.dumps(result, indent=2) + "\n")
            result["compiled_nodes"] = seen
            assert any("qwen21_kv_pack_experiment" in n for n in seen["K1_path"])
            result["correctness_passed"] = bool(result["cases"]) and all(
                all(c["exact_bits_and_strides"] for c in row["checks"]) for row in result["cases"]
            )
    finally:
        (args.output / "screen.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
