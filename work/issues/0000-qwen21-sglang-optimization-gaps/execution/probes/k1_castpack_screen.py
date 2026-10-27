# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Strict cast/pack correctness gate followed by a synthetic component screen.

Pinned to MI300X, Torch 2.13+git733fca1, HIP 7.2.53211, Triton 3.8.
The oracle is COMPILED native FP32 K -> BF16 -> cat, with separate V cat.
The candidate takes raw FP32 K. No model inference or request speed claim.
All correctness cases finish before timing starts. Failure preserves screen.json.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import statistics
import subprocess
import traceback
from collections import Counter
from pathlib import Path

import torch


CONDITIONAL_CASES = {
    "t2i-accuracy": (25, 4096),
    "it2i-one-reference": (4070, 3072),
    "it2i-accuracy": (8205, 3072),
    "it2i-landscape": (8205, 3072),
    "it2i-four-reference": (16311, 3072),
}
HEADS, DIM = 32, 128
ARMS = ("B0_path", "K1_path")


def reference(kp, vp, k32, v):
    return torch.cat((kp, k32.to(torch.bfloat16)), 1), torch.cat((vp, v), 1)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "offset": tensor.storage_offset(),
        "dtype": str(tensor.dtype),
    }


def raw_bits(tensor):
    if tensor.numel() == 0:
        return torch.empty(0, dtype=torch.uint8, device=tensor.device)
    integer_dtype = torch.int32 if tensor.dtype == torch.float32 else torch.int16
    return tensor.view(integer_dtype).contiguous().view(torch.uint8)


def bits_equal(left, right):
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(raw_bits(left), raw_bits(right))
    )


def difference(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return {"incompatible_metadata": True}
    integer_dtype = torch.int32 if left.dtype == torch.float32 else torch.int16
    a = left.view(integer_dtype).contiguous().flatten()
    b = right.view(integer_dtype).contiguous().flatten()
    indices = (a != b).nonzero().flatten()
    selected = indices[:8]
    mask = (1 << (left.element_size() * 8)) - 1
    return {"different_elements": indices.numel(), "first_indices": selected.cpu().tolist(),
            "actual_hex": [hex(x & mask) for x in a[selected].cpu().tolist()],
            "expected_hex": [hex(x & mask) for x in b[selected].cpu().tolist()]}


def require(row, label, passed, **details):
    row.setdefault("checks", []).append({"check": label, "passed": bool(passed), **details})
    if not passed and not row.get("collect_failures", False):
        raise AssertionError(f"{row['case']}: {label}: {details}")


def compare(row, label, actual, expected, inputs):
    require(row, label + "/arity", len(actual) == len(expected) == 2)
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        same_bits = bits_equal(got, want)
        require(
            row,
            f"{label}/output{index}",
            same_bits and got.stride() == want.stride() and got.is_contiguous(),
            actual=metadata(got),
            expected=metadata(want),
            difference=None if same_bits else difference(got, want),
        )
        prefix = inputs[index]
        prefix_out = got[:, : prefix.shape[1]]
        same_prefix = bits_equal(prefix_out, prefix)
        require(row, f"{label}/prefix{index}_unchanged_bits", same_prefix,
                difference=None if same_prefix else difference(prefix_out, prefix))
    target_v = actual[1][:, inputs[0].shape[1] :]
    same_v = bits_equal(target_v, inputs[3])
    require(row, label + "/target_v_unchanged_bits", same_v,
            difference=None if same_v else difference(target_v, inputs[3]))
    pointers = [tensor.untyped_storage().data_ptr() for tensor in actual]
    input_pointers = {tensor.untyped_storage().data_ptr() for tensor in inputs}
    require(row, label + "/functional_outputs", len(set(pointers)) == 2 and not input_pointers.intersection(pointers))


def snapshot(owners):
    return [owner.view(torch.int32 if owner.dtype == torch.float32 else torch.int16).clone().view(owner.dtype)
            for owner in owners]


def unmutated(row, label, owners, before):
    require(row, label, all(bits_equal(x, y) for x, y in zip(owners, before, strict=True)))


def edge_patterns():
    # Every BF16 encoding, including both NaN signs and every BF16 NaN payload.
    bf16 = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    # Every FP32 high half crossed with exact, adjacent-to-tie and low-payload
    # low halves. This includes positive/negative normal and subnormal ties,
    # overflow boundaries, signed zero, infinities, quiet/signaling NaNs.
    high = torch.arange(65536, dtype=torch.int64)
    low = torch.tensor([0, 1, 0x7FFF, 0x8000, 0x8001, 0xFFFF], dtype=torch.int64)
    fp32 = ((high[:, None] << 16) | low).flatten().to(torch.int32).view(torch.float32)
    return bf16, fp32


def patterned_input(shape, pattern, variant, strided):
    b, s, h, d = shape
    if strided:
        # All four inputs have independent strides and nonzero storage offsets.
        sd = variant + 1
        sh = d * sd + variant + 3
        ss = h * sh + 2 * variant + 7
        strides = ((s + 1) * ss + 13, ss, sh, sd)
        offset = 11 + 5 * variant
        size = offset + sum((max(n, 1) - 1) * step for n, step in zip(shape, strides, strict=True)) + 1
    else:
        strides = (s * h * d, h * d, d, 1)
        offset, size = 0, max(math.prod(shape), 1)
    # Construct and transfer storage as integers so copying NaNs cannot round or
    # canonicalize an encoding before it reaches either GPU implementation.
    integer_dtype = torch.int32 if pattern.dtype == torch.float32 else torch.int16
    owner_bits = torch.full((size,), 0x1234, dtype=integer_dtype)
    view_bits = owner_bits.as_strided(shape, strides, offset)
    count = math.prod(shape)
    source = pattern.view(integer_dtype).repeat((count + pattern.numel() - 1) // pattern.numel())[:count]
    view_bits.copy_(source.reshape(shape))
    owner = owner_bits.to("cuda").view(pattern.dtype)
    return owner.as_strided(shape, strides, offset), owner


def edge_inputs(prefix, target, strided, patterns):
    bf16, fp32 = patterns
    tensors, owners = [], []
    for index, (tokens, pattern) in enumerate(((prefix, bf16), (prefix, bf16), (target, fp32), (target, bf16))):
        # Different prefix/V ordering catches accidental interchange of sources.
        if index in (1, 3):
            pattern = pattern.flip(0) if index == 1 else pattern.roll(101)
        tensor, owner = patterned_input((1, tokens, HEADS, DIM), pattern, index, strided)
        tensors.append(tensor)
        owners.append(owner)
    return tuple(tensors), owners


def component_inputs(prefix, target):
    torch.manual_seed(20261028)
    kp = torch.randn((1, prefix, HEADS, DIM), device="cuda", dtype=torch.bfloat16)
    vp = torch.randn_like(kp)
    k32 = torch.randn((1, target, HEADS, DIM), device="cuda", dtype=torch.float32)
    qkv = torch.randn((1, target, 3 * HEADS * DIM), device="cuda", dtype=torch.bfloat16)
    v = qkv[:, :, 2 * HEADS * DIM :].view(1, target, HEADS, DIM)
    return (kp, vp, k32, v), [kp, vp, k32, qkv]


def fake_check(row, candidate, inputs):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode() as mode:
        fake_inputs = tuple(mode.from_tensor(tensor) for tensor in inputs)
        fake_outputs = candidate(*fake_inputs)
        total = inputs[0].shape[1] + inputs[2].shape[1]
        shape = (1, total, HEADS, DIM)
        stride = (total * HEADS * DIM, HEADS * DIM, DIM, 1)
        require(
            row,
            "fake_shape_dtype_stride",
            len(fake_outputs) == 2
            and all(
                tuple(output.shape) == shape
                and output.dtype == torch.bfloat16
                and output.stride() == stride
                and output.device == inputs[0].device
                for output in fake_outputs
            ),
            outputs=[metadata(output) for output in fake_outputs],
        )


def capture(functions, inputs):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for function in functions.values():
            for _ in range(3):
                function(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graphs, outputs = {}, {}
    for name, function in functions.items():
        graphs[name] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[name]):
            outputs[name] = function(*inputs)
    return graphs, outputs


def change_inputs(inputs):
    # Change raw encodings through integer views, preserving adversarial coverage
    # and avoiding NaN arithmetic as the mechanism used to test graph freshness.
    for index, tensor in enumerate(inputs):
        integer_dtype = torch.int32 if tensor.dtype == torch.float32 else torch.int16
        tensor.view(integer_dtype).bitwise_xor_(0x0081 + index)


def qualification(row, candidate, functions, inputs, owners):
    row["inputs"] = [metadata(tensor) for tensor in inputs]
    before = snapshot(owners)
    fake_check(row, candidate, inputs)
    expected = functions["B0_path"](*inputs)
    compare(row, "compiled_native_copy_contract", expected, expected, inputs)
    unmutated(row, "compiled_native_inputs_unchanged", owners, before)
    compare(row, "eager_candidate", candidate(*inputs), expected, inputs)
    unmutated(row, "eager_candidate_inputs_unchanged", owners, before)
    compare(row, "compiled_candidate", functions["K1_path"](*inputs), expected, inputs)
    unmutated(row, "compiled_candidate_inputs_unchanged", owners, before)
    graphs, outputs = capture(functions, inputs)
    unmutated(row, "capture_inputs_unchanged", owners, before)
    original_expected = tuple(snapshot(expected))
    for iteration in range(2):
        if iteration:
            change_inputs(inputs)
            before = snapshot(owners)
            expected = functions["B0_path"](*inputs)
            require(
                row,
                "changed_inputs_change_oracle",
                all(not bits_equal(x, y) for x, y in zip(original_expected, expected, strict=True)),
            )
        for name, graph in graphs.items():
            graph.replay()
            compare(row, f"hipgraph/{name}/changed={bool(iteration)}", outputs[name], expected, inputs)
            unmutated(row, f"hipgraph/{name}/inputs_unchanged/{iteration}", owners, before)
    return graphs, outputs


def context_cases(args):
    cases = []
    for name, expected in CONDITIONAL_CASES.items():
        path = args.baseline_captures / name / "context.pt"
        context = torch.load(path, map_location="cpu", weights_only=True)
        target = int(context["latents"].shape[1])
        for branch, key in (("conditional", "image_pad_mask"), ("unconditional", "negative_image_pad_mask")):
            mask = context.get(key)
            if mask is None:
                if branch == "conditional":
                    raise ValueError(f"Missing {key}: {path}")
                continue
            if mask.shape[0] != 1 or context["latents"].shape[0] != 1:
                raise ValueError(f"Expected batch one: {path}")
            prefix = int(mask.numel() + 3 * mask.sum().item())
            if branch == "conditional" and (prefix, target) != expected:
                raise ValueError(f"Saved five-case geometry drift: {name}: {(prefix, target)} != {expected}")
            cases.append(
                {
                    "case": name if branch == "conditional" else name + "-unconditional",
                    "branch": branch,
                    "prefix": prefix,
                    "target": target,
                    "context_file": str(path),
                    "context_sha256": digest(path),
                    "context_mask_key": key,
                    "input_kind": "synthetic operands; actual saved CFG context geometry",
                    "checks": [],
                    "traces": {},
                    "timings": {},
                }
            )
    return cases


def trace_graphs(row, graphs, output_dir):
    for name, graph in graphs.items():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            graph.replay()
            torch.cuda.synchronize()
        path = output_dir / f"{row['case']}-{name}-trace.json"
        profile.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
        counts = Counter(event["name"] for event in events if event.get("cat") == "kernel")
        castpack_count = sum(count for kernel, count in counts.items() if "_cast_pack" in kernel)
        row["traces"][name] = {"file": path.name, "sha256": digest(path), "kernel_counts": dict(counts)}
        require(row, f"{name}/castpack_activation", castpack_count == (1 if name == "K1_path" else 0))
        require(row, f"{name}/kernel_count", sum(counts.values()) == (1 if name == "K1_path" else 2))


def timings(row, graphs, flush):
    for condition, repeats in (("warm", 40), ("after_512MiB_write", 1)):
        pairs = []
        for trial in range(12):
            order = list(ARMS if trial % 2 == 0 else reversed(ARMS))
            sample = {"trial": trial, "order": order}
            for name in order:
                if condition == "warm":
                    graphs[name].replay()
                else:
                    flush.fill_(trial)
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(repeats):
                    graphs[name].replay()
                stop.record()
                stop.synchronize()
                sample[name] = start.elapsed_time(stop) * 1000 / repeats
            pairs.append(sample)
        row["timings"][condition] = {
            "unit": "microseconds per graph replay; setup and device write excluded",
            "pairs": pairs,
            "reference_median_us": statistics.median(pair["B0_path"] for pair in pairs),
            "candidate_median_us": statistics.median(pair["K1_path"] for pair in pairs),
            "median_paired_candidate_over_reference": statistics.median(
                pair["K1_path"] / pair["B0_path"] for pair in pairs
            ),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-captures", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "scope": (
            "Synthetic compiled cast/KV-pack component only; "
            "no production, attention, image, or request qualification"
        ),
        "script_sha256": digest(__file__),
        "correctness_passed": False,
        "performance_started": False,
        "completed": False,
        "edge_cases": [],
        "cases": [],
        "compiled_nodes": {name: [] for name in ARMS},
        "budget": {
            "paired_rounds": 12,
            "warm_replays_per_sample": 40,
            "flush_replays_per_sample": 1,
            "flush_bytes": 512 * 1024 * 1024,
        },
        "conditions": (
            "Fixed alternating AB/BA pairs. Warm buffers and one replay after a 512 MiB write. "
            "No guaranteed eviction claim."
        ),
        "oracle": "fullgraph compiled native FP32 K cast + K concatenation, and V concatenation; raw-byte comparison",
        "outputs": "contiguous BSHD BF16, with BHSD attention views available by permute(0,2,1,3)",
    }

    def save():
        (args.output / "screen.json").write_text(json.dumps(result, indent=2) + "\n")

    try:
        # Import only when this GPU executable actually runs, never for syntax
        # checks or when this file is imported as a module on the local host.
        from vllm_omni.diffusion.layers import qwen21_kv_pack

        config_path = args.checkpoint / "transformer/config.json"
        config = json.loads(config_path.read_text())
        if (config["num_attention_heads"], config["attention_head_dim"]) != (HEADS, DIM):
            raise ValueError("This screen requires H32 D128")
        packages = {}
        for name in ("triton", "aiter", "aotriton", "vllm"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        result.update(
            source_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            pack_sha256=digest(qwen21_kv_pack.__file__),
            pack_file=str(qwen21_kv_pack.__file__),
            checkpoint_config_sha256=digest(config_path),
            torch=torch.__version__,
            torch_git_version=torch.version.git_version,
            hip=torch.version.hip,
            device=torch.cuda.get_device_name(0),
            total_memory=torch.cuda.get_device_properties(0).total_memory,
            gcn_arch=getattr(torch.cuda.get_device_properties(0), "gcnArchName", None),
            packages=packages,
        )
        if not str(result["gcn_arch"]).startswith("gfx942") or result["total_memory"] < 180 * 1024**3:
            raise RuntimeError("This screen is pinned to MI300X/gfx942")
        if not str(torch.__version__).startswith("2.13") or not str(torch.version.git_version).startswith("733fca1"):
            raise RuntimeError("This screen is pinned to Torch 2.13+git733fca1")
        if not str(torch.version.hip).startswith("7.2.53211") or not str(packages["triton"]).startswith("3.8."):
            raise RuntimeError("This screen is pinned to HIP 7.2.53211 and Triton 3.8")

        def candidate(kp, vp, k32, v):
            # There must be no external k32.to(BF16) in this arm.
            return qwen21_kv_pack.pack(kp, vp, k32, v)

        def backend(name):
            def compile_graph(gm, inputs):
                result["compiled_nodes"][name].extend(str(node.target) for node in gm.graph.nodes)
                with torch._inductor.config.patch({"triton.cudagraphs": False, "triton.cudagraph_trees": False}):
                    return torch._inductor.compile(gm, inputs)

            return compile_graph

        functions = {
            "B0_path": torch.compile(reference, backend=backend("B0_path"), fullgraph=True, dynamic=True),
            "K1_path": torch.compile(candidate, backend=backend("K1_path"), fullgraph=True, dynamic=True),
        }
        result["cases"] = context_cases(args)
        patterns = edge_patterns()
        result["edge_coverage"] = {
            "bf16_encodings": 65536,
            "fp32_encodings": patterns[1].numel(),
            "fp32_low_words": ["0000", "0001", "7fff", "8000", "8001", "ffff"],
            "fp32_high_words": "all 65536, including signs, subnormals, infinities and NaN payloads",
            "layouts": "contiguous and independently strided inputs with nonzero offsets",
            "partial_tiles": "prefix lengths 0,1,25; target lengths 3,17,97; H32 D128 remain fixed",
        }
        with torch.inference_mode(), torch._dynamo.config.patch(
            {"recompile_limit": 64, "accumulated_recompile_limit": 256}
        ):
            # 97 target tokens hold all 393216 FP32 patterns plus a tail.
            # Prefix 25 and target 97 each hold all 65536 BF16 encodings.
            for strided in (False, True):
                for prefix, target in ((0, 3), (1, 17), (25, 97)):
                    row = {"case": f"edge-p{prefix}-t{target}-strided{int(strided)}", "checks": [],
                           "collect_failures": True}
                    result["edge_cases"].append(row)
                    inputs, owners = edge_inputs(prefix, target, strided, patterns)
                    graphs, outputs = qualification(row, candidate, functions, inputs, owners)
                    del graphs, outputs, inputs, owners
                    save()
            # Qualify every production-derived geometry before ANY performance.
            for row in result["cases"]:
                row["collect_failures"] = True
                inputs, owners = component_inputs(row["prefix"], row["target"])
                graphs, outputs = qualification(row, candidate, functions, inputs, owners)
                trace_graphs(row, graphs, args.output)
                del graphs, outputs, inputs, owners
                save()
            activated = any("qwen21_kv_cast_pack_experiment" in node for node in result["compiled_nodes"]["K1_path"])
            if not activated:
                raise AssertionError("Candidate custom operation absent from compiled graph")
            result["compiled_candidate_activation"] = activated
            failures = [{"case": row["case"], "check": check["check"]}
                        for row in result["edge_cases"] + result["cases"]
                        for check in row["checks"] if not check["passed"]]
            result["qualification_failures"] = failures
            if failures:
                raise AssertionError(f"Strict qualification failed {len(failures)} checks; performance remains blocked")
            for row in result["cases"]:
                row["collect_failures"] = False
            result["correctness_passed"] = True
            save()
            flush = torch.empty(512 * 1024 * 1024, device="cuda", dtype=torch.uint8)
            result["performance_started"] = True
            for row in result["cases"]:
                inputs, owners = component_inputs(row["prefix"], row["target"])
                before = snapshot(owners)
                expected = functions["B0_path"](*inputs)
                graphs, outputs = capture(functions, inputs)
                for name in ARMS:
                    graphs[name].replay()
                    compare(row, f"pretiming/{name}", outputs[name], expected, inputs)
                timings(row, graphs, flush)
                for name in ARMS:
                    compare(row, f"posttiming/{name}", outputs[name], expected, inputs)
                unmutated(row, "timing_inputs_unchanged", owners, before)
                save()
                print(
                    json.dumps({
                        "case": row["case"],
                        "timings": {
                            condition: {key: value for key, value in summary.items() if key != "pairs"}
                            for condition, summary in row["timings"].items()
                        },
                    }),
                    flush=True,
                )
                del graphs, outputs, inputs, owners, expected, before
            result["completed"] = True
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        if isinstance(error, AssertionError):
            result["correctness_passed"] = False
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
