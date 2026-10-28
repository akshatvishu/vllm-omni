# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Contract-v2 correctness gate for one fixed K1 tile, with optional timing.

Reuse the established edge/copy/replay checks. Synthetic geometry tests use a
native reference whose head geometry and packed-V layout are explicit constants.
This is a component comparison, not production/request qualification. Optional
real captures run the existing separate boundary/attention replay afterward.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "probes"))
import k1_castpack_nanpolicy_screen as gate


def production_reference(kp, vp, k32, qkv):
    """Match the production native packing geometry; only sequence lengths vary."""
    prefix, target = kp.shape[1], k32.shape[1]
    kp = kp.view(1, prefix, 32, 128)
    vp = vp.view(1, prefix, 32, 128)
    key = k32.view(1, target, 32, 128).to(torch.bfloat16)
    value = qkv.view(1, target, 12288).split((4096, 4096, 4096), dim=-1)[2]
    value = value.view(1, target, 32, 128)
    return torch.cat((kp, key), dim=1), torch.cat((vp, value), dim=1)


def packed_owner(value):
    """Recover the existing QKV storage view without allocating or copying V."""
    target = value.shape[1]
    if value.shape != (1, target, 32, 128) or value.stride()[1:] != (12288, 128, 1):
        raise ValueError("Production-specialized oracle requires packed-QKV V strides")
    if value.storage_offset() != 8192:
        raise ValueError("Production-specialized oracle requires V storage offset 8192")
    return value.as_strided((1, target, 12288), (target * 12288, 12288, 1), 0)


def static_geometry(inputs):
    # Pin batch/head axes before Dynamo first sees them. Only token lengths are
    # dynamic; the generic edge oracle intentionally supports arbitrary strides.
    for value in inputs[:3]:
        for dim in (0, 2, 3):
            torch._dynamo.mark_static(value, dim)
    owner = packed_owner(inputs[3])
    torch._dynamo.mark_static(owner, 0)
    torch._dynamo.mark_static(owner, 2)
    return owner


class ObserveKernel:
    """Observe actual eager/capture launches; timed graph replays bypass Python."""

    def __init__(self, kernel, block, directory, records):
        self.kernel = kernel
        self.block = block
        self.directory = directory
        self.records = records
        self.seen = set()

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def observed(*args, **kwargs):
            block = kwargs.get("block_size", args[-1] if args else None)
            if block != self.block:
                raise AssertionError(f"Requested tile {self.block}; actual launch tile {block}")
            compiled = launch(*args, **kwargs)
            if compiled.metadata.num_warps != 4:
                raise AssertionError("The first sweep fixes num_warps=4")
            identifier = hashlib.sha256(compiled.asm["ttgir"].encode()).hexdigest()
            if identifier not in self.seen:
                self.seen.add(identifier)
                folder = self.directory / identifier
                folder.mkdir(parents=True, exist_ok=True)
                paths = {}
                for extension in ("ttir", "ttgir", "llir", "amdgcn"):
                    source = compiled.asm.get(extension)
                    if source is not None:
                        path = folder / ("cast_pack." + extension)
                        path.write_text(source)
                        paths[extension] = {"path": str(path), "sha256": gate.digest(path)}
                metadata = compiled.metadata._asdict()
                self.records.append({
                    "identity": identifier,
                    "grid": list(grid) if isinstance(grid, tuple) else str(grid),
                    "block": block,
                    "prefix": args[6],
                    "target": args[7],
                    "inputs": [gate.metadata(value) for value in args[:4]],
                    "metadata": metadata,
                    "n_regs": getattr(compiled, "n_regs", None),
                    "n_spills": getattr(compiled, "n_spills", None),
                    "sources": paths,
                })
            return compiled

        return observed


def collect_native_cache(output):
    """Save relevant compiler text only, never checkpoints or entire cache trees."""
    from torch._inductor.runtime.autotune_cache import AutotuneCache

    records = []
    for env_name in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        root = os.environ.get(env_name)
        if not root:
            continue
        selected = {}
        best_config_sources = {}
        for path in Path(root).rglob("*"):
            if not path.is_file() or path.suffix not in (".py", ".json", ".ttgir", ".amdgcn", ".best_config"):
                continue
            if path.suffix == ".best_config":
                # The experiment has a dedicated cache. Preserve these small
                # records even when a version-specific filename cannot be
                # associated; never claim an unmatched config was selected.
                selected[path] = path.read_text(errors="replace")
                continue
            text = path.read_text(errors="replace")
            relevant = "cat" in path.name or (path.suffix == ".py" and "triton_poi_fused" in text and "cat" in text)
            if not relevant:
                continue
            selected[path] = text
            if path.suffix == ".py":
                candidates = [path.with_suffix(".best_config")]
                if hasattr(AutotuneCache, "_prepare_key") and hasattr(AutotuneCache, "_make_local_cache_key"):
                    key = AutotuneCache._prepare_key(str(path))
                    candidates.append(Path(AutotuneCache._make_local_cache_key(str(path.parent), key)))
                for candidate in candidates:
                    if candidate.is_file():
                        selected[candidate] = candidate.read_text(errors="replace")
                        best_config_sources.setdefault(candidate, []).append(str(path))
        for path, text in selected.items():
            suffix = path.suffix
            digest = gate.digest(path)
            destination = output / "native-compiler" / (digest + suffix)
            destination.parent.mkdir(exist_ok=True)
            destination.write_text(text)
            records.append({"cache_path": str(path), "path": str(destination), "sha256": digest,
                            "associated_generated_sources": best_config_sources.get(path, [])})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--block", required=True, type=int, choices=(256, 1024, 2048, 4096))
    parser.add_argument("--geometries", type=Path, default=HERE / "geometries.json")
    parser.add_argument("--captures", type=Path, help="Optional fresh k1_castpack_boundary captures")
    parser.add_argument("--timing", action="store_true", help="Run the existing bounded component screen after all checks")
    parser.add_argument("--profile", action="store_true", help="Assert native two-kernel and candidate one-kernel activation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "scope": "Synthetic component correctness; optional component timing; no request or image qualification",
        "block": args.block,
        "num_warps": 4,
        "contract_version": 2,
        "script_sha256": gate.digest(__file__),
        "reused_gate_sha256": gate.digest(gate.__file__),
        "geometry_sha256": gate.digest(args.geometries),
        "oracle": "Compiled native FP32 K to BF16 plus K/V cat; production head/V layout specialized for geometry cases",
        "copy_rule": "Candidate copies exact original bytes; native copied-NaN encoding differences only are allowed; target K exact compiled cast",
        "edge_cases": [], "cases": [], "compiled_graphs": {}, "candidate_compiler": [],
        "correctness_passed": False, "performance_started": False, "completed": False,
    }

    def save():
        (args.output / "screen.json").write_text(json.dumps(result, indent=2, default=str) + "\n")

    try:
        from vllm_omni.diffusion.layers import qwen21_kv_pack

        device = torch.cuda.get_device_properties(0)
        result.update(
            source_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            pack_sha256=gate.digest(qwen21_kv_pack.__file__),
            torch=str(torch.__version__), torch_git=torch.version.git_version, hip=torch.version.hip,
            triton=importlib.metadata.version("triton"),
            device={"arch": device.gcnArchName, "memory": device.total_memory, "cus": device.multi_processor_count},
        )
        if str(device.gcnArchName).split(":")[0] != "gfx942" or device.total_memory < 180 * 1024**3:
            raise RuntimeError("This experiment requires MI300X/gfx942")
        if qwen21_kv_pack.qualified_device() != torch.device("cuda", torch.cuda.current_device()):
            raise RuntimeError("Production candidate stack guard rejected the current environment")
        qwen21_kv_pack._cast_pack = ObserveKernel(
            qwen21_kv_pack._cast_pack, args.block, args.output / "candidate-compiler", result["candidate_compiler"]
        )

        def candidate(kp, vp, k32, v):
            return qwen21_kv_pack.pack(kp, vp, k32, v)

        def backend(name):
            def compile_graph(graph, inputs):
                result["compiled_graphs"].setdefault(name, []).append(graph.code)
                with torch._inductor.config.patch({"triton.cudagraphs": False, "triton.cudagraph_trees": False}):
                    return torch._inductor.compile(graph, inputs)
            return compile_graph

        compiled_candidate = torch.compile(candidate, fullgraph=True, dynamic=True, backend=backend("candidate"))
        generic_native = torch.compile(gate.reference, fullgraph=True, dynamic=True, backend=backend("generic_edge_native"))
        compiled_native = torch.compile(production_reference, fullgraph=True, dynamic=True, backend=backend("production_geometry_native"))

        def native_with_owner(kp, vp, k32, v):
            owner = static_geometry((kp, vp, k32, v))
            return compiled_native(kp, vp, k32, owner)

        edge_functions = {"B0_path": generic_native, "K1_path": compiled_candidate}
        functions = {"B0_path": native_with_owner, "K1_path": compiled_candidate}
        result["cases"] = json.loads(args.geometries.read_text())["cases"]
        if len(result["cases"]) != 9:
            raise ValueError("Require all five saved workloads and nine CFG geometries")
        with torch.inference_mode(), torch._dynamo.config.patch({"recompile_limit": 64, "accumulated_recompile_limit": 256}):
            patterns = gate.edge_patterns()
            result["edge_coverage"] = {"bf16_encodings": patterns[0].numel(), "fp32_encodings": patterns[1].numel(),
                                       "layouts": "contiguous and independently strided with nonzero offsets"}
            for strided in (False, True):
                for prefix, target in ((0, 3), (1, 17), (25, 97)):
                    row = {"case": f"edge-p{prefix}-t{target}-strided{int(strided)}", "checks": [], "collect_failures": True}
                    result["edge_cases"].append(row)
                    inputs, owners = gate.edge_inputs(prefix, target, strided, patterns)
                    graphs, outputs = gate.qualification(row, candidate, edge_functions, inputs, owners)
                    del graphs, outputs, inputs, owners
                    save()
            for row in result["cases"]:
                row.update(checks=[], traces={}, timings={}, collect_failures=True,
                           input_kind="Synthetic operands at saved CFG geometry; native QKV storage specialization")
                inputs, owners = gate.component_inputs(row["prefix"], row["target"])
                static_geometry(inputs)
                graphs, outputs = gate.qualification(row, candidate, functions, inputs, owners)
                if args.profile:
                    gate.trace_graphs(row, graphs, args.output)
                del graphs, outputs, inputs, owners
                save()
            failures = [{"case": row["case"], "check": check["check"]}
                        for row in result["edge_cases"] + result["cases"] for check in row["checks"] if not check["passed"]]
            result["qualification_failures"] = failures
            if failures:
                raise AssertionError(f"Contract-v2 gate failed {len(failures)} checks")
            if not result["candidate_compiler"]:
                raise AssertionError("No candidate kernel launch was observed")
            result["correctness_passed"] = True
            save()
            if args.timing:
                flush = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
                result["performance_started"] = True
                for row in result["cases"]:
                    row["collect_failures"] = False
                    inputs, owners = gate.component_inputs(row["prefix"], row["target"])
                    static_geometry(inputs)
                    before = gate.snapshot(owners)
                    expected = functions["B0_path"](*inputs)
                    graphs, outputs = gate.capture(functions, inputs)
                    for arm in gate.ARMS:
                        graphs[arm].replay()
                        gate.compare(row, "pretiming/" + arm, outputs[arm], expected, inputs, native=arm == "B0_path")
                    gate.timings(row, graphs, flush)
                    for arm in gate.ARMS:
                        gate.compare(row, "posttiming/" + arm, outputs[arm], expected, inputs, native=arm == "B0_path")
                    gate.unmutated(row, "timing_inputs_unchanged", owners, before)
                    del graphs, outputs, inputs, owners, expected, before
                    save()
            if args.captures:
                import k1_castpack_boundary
                k1_castpack_boundary.replay(argparse.Namespace(captures=args.captures, output=args.output / "real-boundary"))
                result["real_boundary_passed"] = True
            result["completed"] = True
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        if isinstance(error, AssertionError):
            result["correctness_passed"] = False
        raise
    finally:
        result["native_compiler_files"] = collect_native_cache(args.output)
        save()


if __name__ == "__main__":
    main()
