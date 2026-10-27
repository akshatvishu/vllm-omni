# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Standalone supplemental P1 storage proof with real first-attention weights.

Loads only one BF16 attention module. Uses saved B0 conditional token lengths,
synthetic hidden states and unit complex RoPE. Actual prefill creates cached K/V;
actual compiled attention and changed-input CUDA graph replay must preserve the
prefix, overwrite poisoned target tails, retain storage pointers and match the
unmodified concatenation path exactly. This is not full-request qualification.
"""

import torch


def bits(a, b):
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    )


def prove(attention, hidden, freqs, prefix_key, prefix_value):
    from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import QwenImage21Attention
    from vllm_omni.platforms import current_omni_platform

    if type(attention) is not QwenImage21Attention:
        raise TypeError("requires actual loaded QwenImage21Attention, not a stand-in")
    if hidden.shape[0] != 1 or hidden.dtype != torch.bfloat16 or not hidden.is_cuda:
        raise ValueError("B1 GPU BF16 required")
    p = prefix_key.shape[1]
    s = hidden.shape[1]
    pk = prefix_key.clone()
    pv = prefix_value.clone()
    full_k = pk.new_full((1, p + s, *pk.shape[2:]), float("nan"))
    full_v = pv.new_full((1, p + s, *pv.shape[2:]), float("nan"))
    full_k[:, :p].copy_(pk)
    full_v[:, :p].copy_(pv)
    compact = {"cond": {"key": pk, "value": pv}}
    retained = {"cond": {"key": full_k[:, :p], "value": full_v[:, :p], "packed_key": full_k, "packed_value": full_v}}
    original_ptrs = (full_k.data_ptr(), full_v.data_ptr())

    def reference(x):
        return attention(x, freqs, kv_cache=compact, cache_branch="cond")

    def candidate(x):
        return attention(x, freqs, kv_cache=retained, cache_branch="cond")

    # No observation op or hooks inside either compiled attention function.
    ref = torch.compile(reference, fullgraph=False)
    fused = torch.compile(candidate, fullgraph=False)
    rows = []
    with torch.inference_mode():
        x = hidden.clone()
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                for fn in (ref, fused):
                    fn(x)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        current_omni_platform.synchronize()
        for index in range(2):
            if index:
                x.add_(torch.tensor(0.125, device=x.device, dtype=x.dtype))
            full_k[:, p:].fill_(float("nan"))
            full_v[:, p:].fill_(float("nan"))
            expected = ref(x)
            actual = fused(x)
            rows.append(
                {
                    "phase": "compiled",
                    "index": index,
                    "output_exact": bits(expected, actual),
                    "prefix_exact": bits(full_k[:, :p], pk) and bits(full_v[:, :p], pv),
                    "target_written": bool(torch.isfinite(full_k[:, p:]).all() and torch.isfinite(full_v[:, p:]).all()),
                }
            )
        previous_k = full_k[:, p:].clone()
        previous_v = full_v[:, p:].clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fused(x)
        x.add_(torch.tensor(0.25, device=x.device, dtype=x.dtype))
        full_k[:, p:].fill_(float("nan"))
        full_v[:, p:].fill_(float("nan"))
        graph.replay()
        current_omni_platform.synchronize()
        expected = ref(x)
        rows.append(
            {
                "phase": "changed-input-graph-replay",
                "output_exact": bits(out, expected),
                "prefix_exact": bits(full_k[:, :p], pk) and bits(full_v[:, :p], pv),
                "target_written": bool(torch.isfinite(full_k[:, p:]).all() and torch.isfinite(full_v[:, p:]).all()),
                "target_changed": not bits(previous_k, full_k[:, p:]) and not bits(previous_v, full_v[:, p:]),
            }
        )
    stable = (full_k.data_ptr(), full_v.data_ptr()) == original_ptrs
    qualified = stable and all(all(v for k, v in row.items() if k not in ("phase", "index")) for row in rows)
    return {
        "qualified": qualified,
        "stable_storage": stable,
        "checks": rows,
        "retained_bytes": full_k.untyped_storage().nbytes() + full_v.untyped_storage().nbytes(),
        "prefix_shape": list(pk.shape),
        "target_shape": list(hidden.shape),
        "scope": (
            "Actual compiled attention callable and changed-input graph storage "
            "writes; caller must supply real checkpoint module/provenance. "
            "Does not establish production copy-byte reduction or full-model latency."
        ),
    }


def main():
    import argparse
    import hashlib
    import inspect
    import json
    import subprocess
    import tempfile
    import traceback
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "qualified": False,
        "scope": (
            "Supplemental first-attention storage test; synthetic hidden states and "
            "unit RoPE at saved B0 conditional shapes. Not production activation attribution or a latency benchmark."
        ),
        "cases": [],
    }

    def digest(path):
        h = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    initialized = False
    try:
        from safetensors import safe_open
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        from vllm_omni.diffusion.config import set_current_diffusion_config
        from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, OmniDiffusionConfig
        from vllm_omni.diffusion.distributed.parallel_state import (
            destroy_distributed_env,
            init_distributed_environment,
            initialize_model_parallel,
        )
        from vllm_omni.diffusion.forward_context import set_forward_context
        from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import QwenImage21Attention
        from vllm_omni.platforms import current_omni_platform

        if not torch.cuda.is_available():
            raise RuntimeError("requires one CUDA/ROCm GPU")
        source = Path(inspect.getfile(QwenImage21Attention)).resolve()
        root = next(p for p in source.parents if (p / ".git").exists())
        result["source"] = {
            "path": str(source),
            "sha256": digest(source),
            "git_sha": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        }
        if "packed_key" not in inspect.getsource(QwenImage21Attention.forward):
            raise RuntimeError("P1 attention implementation absent")
        config_path = args.checkpoint / "transformer/config.json"
        cfg = json.loads(config_path.read_text())
        heads, head_dim = cfg["num_attention_heads"], cfg["attention_head_dim"]
        dim = heads * head_dim
        checkpoint_dir = config_path.parent
        desired = {
            "to_q.weight": ("to_qkv.weight", "q"),
            "to_k.weight": ("to_qkv.weight", "k"),
            "to_v.weight": ("to_qkv.weight", "v"),
            "to_out.0.weight": ("to_out.weight", None),
            "norm_q.weight": ("norm_q.weight", None),
            "norm_k.weight": ("norm_k.weight", None),
        }
        selected = {}
        for shard in sorted(checkpoint_dir.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as reader:
                for key in reader.keys():
                    clean = key.removeprefix("transformer.")
                    prefix = "transformer_blocks.0.attn."
                    if clean.startswith(prefix) and clean[len(prefix) :] in desired:
                        suffix = clean[len(prefix) :]
                        if suffix in selected:
                            raise ValueError(f"duplicate checkpoint weight {suffix}")
                        selected[suffix] = (shard, key)
        if set(selected) != set(desired):
            raise ValueError(f"missing checkpoint weights: {set(desired) - set(selected)}")
        result["checkpoint"] = {"config": str(config_path), "config_sha256": digest(config_path), "weights": []}
        current_omni_platform.set_device(torch.device("cuda:0"))
        vcfg = VllmConfig()
        od = OmniDiffusionConfig(
            model=str(args.checkpoint),
            dtype=torch.bfloat16,
            diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
        )
        with tempfile.TemporaryDirectory(prefix="p1-rendezvous-", dir=args.output.parent) as tmp:
            init_distributed_environment(
                world_size=1, rank=0, local_rank=0, distributed_init_method="file://" + str(Path(tmp) / "store")
            )
            initialized = True
            initialize_model_parallel(
                data_parallel_size=1,
                cfg_parallel_size=1,
                sequence_parallel_size=1,
                ulysses_degree=1,
                ring_degree=1,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
            )
            with set_current_vllm_config(vcfg), set_current_diffusion_config(od), torch.device("cuda"):
                attention = (
                    QwenImage21Attention(
                        dim, heads, head_dim, eps=cfg.get("eps", 1e-6), prefix="transformer.transformer_blocks.0.attn"
                    )
                    .to(dtype=torch.bfloat16)
                    .eval()
                )
            if attention.attn.attn_backend.get_name() != "SDPA":
                raise RuntimeError("supplemental proof requires actual SDPA backend")
            result["attention_backend"] = attention.attn.attn_backend.get_name()
            parameters = dict(attention.named_parameters())
            with torch.no_grad():
                for suffix, (shard, key) in selected.items():
                    with safe_open(shard, framework="pt", device="cpu") as reader:
                        weight = reader.get_tensor(key)
                    weight_hash = hashlib.sha256(weight.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    name, shard_id = desired[suffix]
                    param = parameters[name]
                    loader = getattr(param, "weight_loader", default_weight_loader)
                    if shard_id is None:
                        loader(param, weight)
                    else:
                        loader(param, weight, shard_id)
                    result["checkpoint"]["weights"].append(
                        {
                            "key": key,
                            "shard": str(shard),
                            "tensor_sha256": weight_hash,
                            "shape": list(weight.shape),
                            "dtype": str(weight.dtype),
                        }
                    )
                    del weight
            result["device"] = {
                "name": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "cuda": torch.version.cuda,
            }
            contexts = sorted(args.baseline_captures.glob("*/context.pt"))
            if not contexts:
                raise ValueError("no case/context.pt files")
            with set_forward_context(vllm_config=vcfg, omni_diffusion_config=od), torch.inference_mode():
                for path in contexts:
                    context = torch.load(path, map_location="cpu", weights_only=True)
                    if context.get("image_latents") is None:
                        continue
                    mask = context["image_pad_mask"]
                    latents = context["latents"]
                    if mask.shape[0] != 1 or latents.shape[0] != 1:
                        raise ValueError("B1 captures required")
                    p = int(mask.numel() + 3 * mask.sum().item())
                    s = latents.shape[1]
                    torch.manual_seed(12345)
                    hidden = torch.randn((1, s, dim), device="cuda", dtype=torch.bfloat16)
                    prefix_hidden = torch.randn((1, p, dim), device="cuda", dtype=torch.bfloat16)
                    # Unit complex frequencies are valid RoPE inputs, explicitly synthetic.
                    freqs = torch.ones((s, head_dim // 2), device="cuda", dtype=torch.complex64)
                    prefix_freqs = torch.ones((p, head_dim // 2), device="cuda", dtype=torch.complex64)
                    cache = {"cond": {}}
                    attention(prefix_hidden, prefix_freqs, kv_cache=cache, cache_write_len=p)
                    del prefix_hidden, prefix_freqs
                    proof = prove(attention, hidden, freqs, cache["cond"]["key"], cache["cond"]["value"])
                    proof.update(
                        case_id=path.parent.name,
                        context_sha256=digest(path),
                        prefix_length=p,
                        target_length=s,
                        prefix_length_rule="image_pad_mask.numel + 3*image_pad_mask.sum",
                        input_kind=(
                            "synthetic hidden states, unit complex RoPE; actual checkpoint projections and "
                            "actual prefill cache"
                        ),
                    )
                    result["cases"].append(proof)
                    del cache, hidden, freqs
                    torch._dynamo.reset()
            result["qualified"] = bool(result["cases"]) and all(c["qualified"] for c in result["cases"])
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        if initialized:
            try:
                destroy_distributed_env()
            except Exception:
                result["cleanup_error"] = traceback.format_exc()
                result["qualified"] = False
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
