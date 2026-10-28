# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Diagnostic real-operand capture and bounded compiled boundary replay.

Call install() after the common experiment sitecustomize hook. Enabled only by
QWEN_CASTPACK_BOUNDARY_DIR. The first compiled decode block is observed once per
named request/CFG branch. New graphs are observed during their normal warmup.
A warmed whole-decode graph is replayed then bypassed for that one step so its
real custom-op operands can be observed. No compiled model arithmetic
is replaced. Producer reconstruction happens after the compiled block returns.

No weights are serialized. The upstream comparison is against a separately
compiled reconstruction using the actual RMSNorm modules and complex expression,
not a complete baseline block rerun. Offline replay uses the captured real K/V
and a reconstructed real query for identical-operand SDPA comparisons.
"""

import argparse
import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import sys
from collections import Counter
from pathlib import Path

PACK = "vllm_omni.diffusion.layers.qwen21_kv_pack"
MODEL = "vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer"
GRAPH = "vllm_omni.diffusion.models.qwen_image_21.decode_graph"
STATE = {"context": None, "done": set(), "bypassed": set()}
EXPECTED_BRANCHES = {
    "t2i-accuracy": {"cond"},
    "it2i-accuracy": {"cond", "uncond"},
    "it2i-one-reference": {"cond", "uncond"},
    "it2i-four-reference": {"cond", "uncond"},
    "it2i-landscape": {"cond", "uncond"},
}


def describe(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "offset": tensor.storage_offset(), "dtype": str(tensor.dtype)}


def bits_equal(left, right):
    import torch

    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))


def exact_pair(left, right):
    return all(bits_equal(a, b) and a.stride() == b.stride() for a, b in zip(left, right, strict=True))


def active():
    return getattr(sys.modules.get("sitecustomize"), "ACTIVE", None)


def key_for(branch):
    request = active()
    return None if request is None else (request["case"], branch)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def _patch_pack(module):
    import torch

    original = module._launch

    @functools.wraps(original)
    def launch(kp, vp, k32, v):
        result = original(kp, vp, k32, v)
        context = STATE["context"]
        if (context is None or context.get("pending") is not None
                or torch.cuda.is_current_stream_capturing()):
            return result
        # V is the last third of the production packed QKV projection. Never
        # infer raw Q/K from a merely contiguous standalone V tensor.
        heads, dim = v.shape[-2:]
        width = heads * dim
        if (v.ndim != 4 or v.shape[0] != 1 or v.stride()[1:] != (3 * width, dim, 1)
                or v.storage_offset() < 2 * width):
            raise RuntimeError(f"Cannot recover packed production Q/K from {describe(v)}")
        raw_q = v.as_strided(v.shape, v.stride(), v.storage_offset() - 2 * width)
        raw_k = v.as_strided(v.shape, v.stride(), v.storage_offset() - width)
        tensors = dict(kp=kp, vp=vp, k32=k32, v=v, raw_q=raw_q, raw_k=raw_k,
                       production_k=result[0], production_v=result[1])
        context["layouts"] = {name: describe(t) for name, t in tensors.items()}
        context["pending"] = {name: t.detach().clone() for name, t in tensors.items()}
        return result

    module._launch = launch


def _patch_model(module):
    import torch

    cls = module.QwenImage21Transformer2DModel
    original = cls.__init__

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        block = self.transformer_blocks[0]
        attention = block.attn

        def producer(raw_q, raw_k, freqs):
            # Use the installed RMSNorm implementation and learned weights,
            # including its BF16 rounding before the learned scale.
            q = attention.norm_q(raw_q).to(raw_q.dtype)
            k = attention.norm_k(raw_k).to(raw_k.dtype)
            q_pair = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
            k_pair = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
            q32 = torch.view_as_real(q_pair * freqs.unsqueeze(1)).flatten(3)
            k32 = torch.view_as_real(k_pair * freqs.unsqueeze(1)).flatten(3)
            return q32.to(raw_q.dtype), k32, k32.to(raw_k.dtype)

        # Lazy compile; first invocation is in the outside-block post hook.
        compiled_producer = torch.compile(producer, fullgraph=True, dynamic=True,
                                          options={"triton.cudagraphs": False})

        def before(_block, args, kwargs):
            if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
                return
            branch = kwargs.get("cache_branch", "cond")
            key = key_for(branch)
            if key is None or key in STATE["done"]:
                return
            cache = kwargs.get("kv_cache")
            if cache is None or kwargs.get("cache_write_len") is not None or "key" not in cache.get(branch, {}):
                return
            if kwargs.get("sp_decode") or kwargs.get("sp_prefix_len", 0):
                raise RuntimeError("Boundary diagnostic requires SP1 dense cached decode")
            metadata = kwargs.get("attn_metadata")
            if metadata is not None and getattr(metadata, "attn_mask", None) is not None:
                raise RuntimeError("Boundary replay requires the unmasked cached decode path")
            freqs = kwargs.get("freqs", args[2] if len(args) > 2 else None)
            backend = attention.attn.attn_backend
            STATE["context"] = {"key": key, "freqs": freqs, "pending": None,
                                "backend": backend.get_name() if backend is not None else None}

        def after(_block, _args, _kwargs, _output):
            context = STATE["context"]
            STATE["context"] = None
            if context is None or context["pending"] is None:
                return
            tensors = context["pending"]
            raw_q = restore_layout(tensors["raw_q"], context["layouts"]["raw_q"], tensors["raw_q"].device)
            raw_k = restore_layout(tensors["raw_k"], context["layouts"]["raw_k"], tensors["raw_k"].device)
            q, k32, k = compiled_producer(raw_q, raw_k, context["freqs"])
            checks = {
                "production_fp32_k_equals_compiled_reconstructed_producer": bits_equal(tensors["k32"], k32),
                "production_cast_k_equals_compiled_reconstructed_producer": bits_equal(
                    tensors["k32"].to(torch.bfloat16), k),
            }
            tensors.update(query=q, reference_k32=k32, freqs=context["freqs"])
            directory = Path(os.environ["QWEN_CASTPACK_BOUNDARY_DIR"]) / context["key"][0]
            directory.mkdir(parents=True, exist_ok=True)
            stem = context["key"][1]
            path = directory / f"{stem}.pt"
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}")
            torch.save({name: t.detach().cpu().contiguous().clone() for name, t in tensors.items()}, path)
            write_json(directory / f"{stem}.json", {
                "case": context["key"][0], "branch": stem, "block": 0,
                "backend": context["backend"], "checks": checks, "layouts": context["layouts"],
                "whole_decode_graph_bypassed_once": context["key"] in STATE["bypassed"],
                "graph_history_note": "Original try_decode runs first. If replay bypasses Python capture, its result is discarded once and the normal compiled decode path runs. New graph warmup is observed directly; graph setup is not deferred to step 2.",
                "compiled_block": hasattr(block.forward, "_torchdynamo_orig_callable"),
                "scope": "Production operands; upstream reference separately compiles the same RMSNorm and complex expression. Not a full B0 block replay.",
                "query_scope": "Reconstructed from actual packed projection Q with actual first-block norm and frequencies; not a captured attention query.",
                "weights_serialized": False,
                "attention_mask": None,
            })
            STATE["done"].add(context["key"])

        block.register_forward_pre_hook(before, with_kwargs=True)
        block.register_forward_hook(after, with_kwargs=True, always_call=True)

    cls.__init__ = initialize


def _patch_graph(module):
    cls = module.QwenImage21DecodeGraphManager
    original = cls.try_decode

    @functools.wraps(original)
    def decode(self, **kwargs):
        key = key_for(kwargs["cache_branch"])
        result = original(self, **kwargs)
        # A new graph executes Python warmup and normally captures the boundary
        # there. An existing graph cannot invoke Python, so discard its first
        # replay result and allow the normal compiled-block fallback once.
        # Calling the original first also keeps graph setup out of step 2.
        if key is not None and key not in STATE["done"] and key not in STATE["bypassed"]:
            STATE["bypassed"].add(key)
            return None
        return result

    cls.try_decode = decode


def install():
    """Install alongside, and preserve, the common experiment import hooks."""
    if not os.environ.get("QWEN_CASTPACK_BOUNDARY_DIR"):
        return
    patches = {PACK: _patch_pack, MODEL: _patch_model, GRAPH: _patch_graph}

    class Loader(importlib.abc.Loader):
        def __init__(self, original, patch):
            self.original, self.patch = original, patch

        def create_module(self, spec):
            return self.original.create_module(spec)

        def exec_module(self, module):
            self.original.exec_module(module)
            self.patch(module)

        def __getattr__(self, name):
            return getattr(self.original, name)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in patches:
                return None
            # Delegate to later finders so the existing GRAPH hook also runs.
            for finder in sys.meta_path[sys.meta_path.index(self) + 1:]:
                method = getattr(finder, "find_spec", None)
                spec = method(fullname, path, target) if method is not None else None
                if spec is not None:
                    if spec.loader is not None:
                        spec.loader = Loader(spec.loader, patches[fullname])
                    return spec
            return None

    for name, patch in patches.items():
        if name in sys.modules:
            patch(sys.modules[name])
    sys.meta_path.insert(0, Finder())


def restore_layout(tensor, layout, device):
    import torch

    shape, stride, offset = layout["shape"], layout["stride"], layout["offset"]
    size = offset + 1 + sum((length - 1) * step for length, step in zip(shape, stride, strict=True))
    storage = torch.empty(size, dtype=tensor.dtype, device=device)
    return storage.as_strided(shape, stride, offset).copy_(tensor)


def replay(args):
    import torch
    import torch.nn.functional as F
    from vllm_omni.diffusion.layers import qwen21_kv_pack

    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "scope": "Bounded real-operand compiled cast/pack and same-query SDPA replay; no model/request performance or B0 repeatability claim.",
        "torch": torch.__version__, "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pack_sha256": hashlib.sha256(Path(qwen21_kv_pack.__file__).read_bytes()).hexdigest(),
        "cases": [], "passed": False,
    }

    def reference(kp, vp, k32, v):
        return torch.cat((kp, k32.to(torch.bfloat16)), 1), torch.cat((vp, v), 1)

    def candidate(kp, vp, k32, v):
        return qwen21_kv_pack.pack(kp, vp, k32, v)

    def attention(q, k, v):
        return F.scaled_dot_product_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3),
                                             v.permute(0, 2, 1, 3), dropout_p=0.0, is_causal=False)

    compile_options = {"triton.cudagraphs": False}
    functions = {"reference": torch.compile(reference, fullgraph=True, dynamic=True, options=compile_options),
                 "candidate": torch.compile(candidate, fullgraph=True, dynamic=True, options=compile_options)}
    compiled_attention = torch.compile(attention, fullgraph=True, dynamic=True, options=compile_options)
    try:
        with torch.inference_mode():
            for path in sorted(args.captures.glob("*/*.pt")):
                metadata = json.loads(path.with_suffix(".json").read_text())
                saved = torch.load(path, map_location="cpu", weights_only=True)
                operands = tuple(restore_layout(saved[name], metadata["layouts"][name], "cuda")
                                 for name in ("kp", "vp", "k32", "v"))
                query = saved["query"].to("cuda")
                row = {"case": metadata["case"], "branch": metadata["branch"], "capture": metadata,
                       "checks": dict(metadata["checks"]), "traces": {}}
                result["cases"].append(row)
                row["checks"]["production_backend_sdpa"] = metadata["backend"] == "SDPA"
                row["checks"]["production_block_compiled"] = metadata["compiled_block"]
                outputs = {name: fn(*operands) for name, fn in functions.items()}
                row["checks"]["compiled_inputs_unchanged"] = all(
                    bits_equal(operand, saved[name].to("cuda"))
                    for name, operand in zip(("kp", "vp", "k32", "v"), operands, strict=True))
                expected = outputs["reference"]
                row["checks"]["compiled_final_kv_exact_bits_and_strides"] = exact_pair(expected, outputs["candidate"])
                row["checks"]["captured_production_final_kv_exact"] = all(
                    bits_equal(t, saved[name].to("cuda")) and list(t.stride()) == metadata["layouts"][name]["stride"]
                    for t, name in zip(expected, ("production_k", "production_v"), strict=True))
                attn = {name: compiled_attention(query, *out) for name, out in outputs.items()}
                row["checks"]["same_query_sdpa_exact"] = bits_equal(attn["reference"], attn["candidate"])
                row["checks"]["same_query_sdpa_incumbent_repeat_exact"] = bits_equal(
                    attn["reference"], compiled_attention(query, *outputs["reference"]))
                # Capture both arms only after all lazy compilation/initialization.
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for fn in functions.values():
                        for _ in range(3):
                            compiled_attention(query, *fn(*operands))
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graphs, graph_outputs = {}, {}
                for name, fn in functions.items():
                    graphs[name] = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graphs[name]):
                        kv = fn(*operands)
                        graph_outputs[name] = (*kv, compiled_attention(query, *kv))
                for changed in (False, True):
                    if changed:
                        for index, operand in enumerate(operands):
                            operand.add_(0.03125 * (index + 1))
                        query.add_(0.0625)
                    eager_kv = reference(*operands)
                    eager_attn = compiled_attention(query, *eager_kv)
                    for name, graph in graphs.items():
                        graph.replay()
                        kvk, kvv, out = graph_outputs[name]
                        suffix = "changed" if changed else "original"
                        row["checks"][f"graph_{name}_{suffix}_kv_exact"] = exact_pair((kvk, kvv), eager_kv)
                        row["checks"][f"graph_{name}_{suffix}_attention_exact"] = bits_equal(out, eager_attn)
                        previous = out.clone()
                        graph.replay()
                        row["checks"][f"graph_{name}_{suffix}_attention_repeat_exact"] = bits_equal(out, previous)
                for name, graph in graphs.items():
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
                        graph.replay()
                        torch.cuda.synchronize()
                    trace = args.output / f"{row['case']}-{row['branch']}-{name}-trace.json"
                    profile.export_chrome_trace(str(trace))
                    counts = Counter(event["name"] for event in json.loads(trace.read_text())["traceEvents"]
                                     if event.get("cat") == "kernel")
                    row["traces"][name] = {"file": trace.name, "kernel_counts": dict(counts)}
                    pack_launches = sum(count for kernel, count in counts.items() if "_cast_pack" in kernel)
                    row["checks"][f"{name}_pack_activation"] = pack_launches == (1 if name == "candidate" else 0)
                row["passed"] = all(row["checks"].values())
                write_json(args.output / "boundary.json", result)
                print(json.dumps({"case": row["case"], "branch": row["branch"], "checks": row["checks"]}), flush=True)
                del graphs, graph_outputs, outputs, attn, operands, query
            by_case = {}
            for row in result["cases"]:
                by_case.setdefault(row["case"], set()).add(row["branch"])
            result["expected_case_branches"] = {case: sorted(branches) for case, branches in EXPECTED_BRANCHES.items()}
            result["observed_case_branches"] = {case: sorted(branches) for case, branches in by_case.items()}
            result["all_five_cases_nine_branches_captured"] = by_case == EXPECTED_BRANCHES and len(result["cases"]) == 9
            result["passed"] = result["all_five_cases_nine_branches_captured"] and all(
                row["passed"] for row in result["cases"])
    finally:
        write_json(args.output / "boundary.json", result)
    if not result["passed"]:
        raise SystemExit("Boundary validation failed; inspect boundary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    replay(parser.parse_args())
