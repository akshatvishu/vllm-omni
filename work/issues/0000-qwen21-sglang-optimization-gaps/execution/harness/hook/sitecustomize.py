# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Common experiment policy and optional diagnostics outside compiled blocks.

All arms disable the native prompt cache identically. Captures and replay are
diagnostic-only; ordinary latency processes do not copy tensors or write files
per request. No Q/K arithmetic or attention implementation is replaced here.
"""

import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
from pathlib import Path

PIPELINE = "vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21"
GRAPH = "vllm_omni.diffusion.models.qwen_image_21.decode_graph"
RUNNER = "vllm_omni.diffusion.worker.diffusion_model_runner"
CAPTURE = os.environ.get("QWEN_EXPERIMENT_CAPTURE_DIR")
REPLAY = os.environ.get("QWEN_EXPERIMENT_REPLAY_DIR")
VARIANT = os.environ.get("QWEN_EXPERIMENT_VARIANT", "B0")
DISABLE_CACHE = os.environ.get("QWEN_EXPERIMENT_DISABLE_PROMPT_CACHE") == "1"
ACTIVE = None


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def cpu_tree(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported capture value {type(value)}")


def describe(value):
    import torch

    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "stride": list(value.stride()),
            "storage_offset": value.storage_offset(),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {k: describe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [describe(v) for v in value]
    return value


def save(name, value):
    import torch

    if ACTIVE is None:
        return
    directory = ACTIVE["directory"]
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_tree(value), directory / name)
    write_json(directory / (name + ".metadata.json"), describe(value))


def load(name):
    import torch

    if REPLAY is None or ACTIVE is None:
        return None
    path = Path(REPLAY) / ACTIVE["case"] / name
    if not path.is_file():
        raise FileNotFoundError(f"Required saved baseline operand missing: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def to_like(value, reference):
    import torch

    if isinstance(reference, torch.Tensor):
        if not isinstance(value, torch.Tensor) or value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError("Saved baseline operand shape/dtype differs from current request")
        result = value.to(reference.device)
        if result.stride() != reference.stride():
            if torch._debug_has_internal_overlap(reference) != 0:
                raise ValueError("Cannot restore a saved overlapping operand layout")
            result = torch.empty_strided(
                reference.shape, reference.stride(), dtype=reference.dtype, device=reference.device
            ).copy_(result)
        return result
    if isinstance(reference, dict):
        if not isinstance(value, dict) or value.keys() != reference.keys():
            raise ValueError("Saved baseline dictionary differs")
        return {key: to_like(value[key], item) for key, item in reference.items()}
    if isinstance(reference, (tuple, list)):
        if type(value) is not type(reference) or len(value) != len(reference):
            raise ValueError("Saved baseline sequence differs")
        return type(reference)(to_like(x, y) for x, y in zip(value, reference))
    if type(value) is not type(reference) or value != reference:
        raise ValueError("Saved baseline non-tensor metadata differs")
    return value


def identify(kwargs):
    source = Path(os.environ["QWEN_EXPERIMENT_INPUTS"])
    data = json.loads(source.read_text())
    prompt = kwargs["prompt"]
    prompt = prompt[0] if isinstance(prompt, list) and len(prompt) == 1 else prompt
    matches = []
    for case in data["cases"]:
        request = case["request"]
        if (
            prompt == request["prompt"]
            and kwargs["height"] == request["height"]
            and kwargs["width"] == request["width"]
        ):
            matches.append(case["id"])
    if len(matches) != 1:
        raise ValueError("Diagnostic capture requires exactly one named batch-one workload")
    return matches[0]


def attach_diagnostics(pipeline):
    original_encode = pipeline.vae.encode
    original_decode = pipeline.vae.decode
    original_decoder = pipeline.vae.decoder.forward
    original_encoder = pipeline.text_encoder.forward

    @functools.wraps(original_encode)
    def encode(*args, **kwargs):
        result = original_encode(*args, **kwargs)
        if ACTIVE is not None:
            slot = ACTIVE["encode_slot"]
            ACTIVE["encode_slot"] += 1
            distribution = result.latent_dist
            if VARIANT == "S1" and REPLAY:
                saved = load(f"raw-encode-{slot}.pt")
                params = to_like(saved["parameters"], distribution.parameters)
                result.latent_dist = distribution = type(distribution)(params)
            save(
                f"raw-encode-{slot}.pt",
                {
                    "input": args[0] if args else kwargs["x"],
                    "parameters": distribution.parameters,
                    "latent": distribution.mode(),
                },
            )
        return result

    @functools.wraps(original_decode)
    def decode(*args, **kwargs):
        if ACTIVE is not None:
            save("vae-decode-input.pt", {"latents": args[0] if args else kwargs["z"]})
        output = original_decode(*args, **kwargs)
        if ACTIVE is not None:
            save("vae-decode-output.pt", {"output": output[0]})
        return output

    @functools.wraps(original_encoder)
    def encoder(*args, **kwargs):
        if ACTIVE is not None:
            slot = ACTIVE["encoder_slot"]
            ACTIVE["encoder_slot"] += 1
            save(f"encoder-{slot}-input.pt", kwargs)
        if ACTIVE is not None and VARIANT == "O1":
            import torch

            encoder_module = pipeline.text_encoder
            paths = ("model.language_model.layers", "model.visual.blocks", "model.visual.deepstack_merger_list")
            expected = [encoder_module.get_submodule(path) for path in paths]
            actual = getattr(encoder_module, "_omni_layerwise_block_groups", [])
            record = {
                "enabled": bool(getattr(encoder_module, "_omni_layerwise_enabled", False)),
                "paths": list(paths),
                "group_lengths": [len(group) for group in actual],
                "plan_matches": len(actual) == len(expected)
                and all(list(a) == list(b) for a, b in zip(actual, expected)),
                "hooks": len(getattr(encoder_module, "_omni_layerwise_hooks", [])),
                "dit_resident_before": all(p.device.type == "cuda" for p in pipeline.transformer.parameters()),
                "dit_layerwise_offload": bool(getattr(pipeline.transformer, "_omni_layerwise_enabled", False)),
            }
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            ) as profile:
                result = original_encoder(*args, **kwargs)
            trace = ACTIVE["directory"] / f"encoder-{slot}-trace.json"
            profile.export_chrome_trace(str(trace))
            record["trace"] = trace.name
            record["dit_resident_after"] = all(p.device.type == "cuda" for p in pipeline.transformer.parameters())
            ACTIVE["encoder_offload"].append(record)
        else:
            result = original_encoder(*args, **kwargs)
        if ACTIVE is not None:
            save(f"encoder-{slot}-output.pt", {"hidden_states": result.hidden_states[-1]})
        return result

    pipeline.vae.encode = encode
    pipeline.vae.decode = decode

    @functools.wraps(original_decoder)
    def decoder(*args, **kwargs):
        if ACTIVE is not None:
            index = ACTIVE["decoder_calls"]
            ACTIVE["decoder_calls"] += 1
            save(
                f"decoder-call-{index:03d}-input.pt",
                {
                    "input": args[0],
                    "first_chunk": kwargs.get("first_chunk", False),
                },
            )
        result = original_decoder(*args, **kwargs)
        if ACTIVE is not None:
            save(f"decoder-call-{index:03d}-output.pt", {"output": result})
        return result

    pipeline.vae.decoder.forward = decoder
    pipeline.text_encoder.forward = encoder
    pipeline._experiment_original_vae_decode = original_decode


def stage_benchmark(pipeline, case):
    """Separate synchronized saved-input decoder timing, never HTTP timing."""
    import torch

    directory = case["directory"]
    packed = torch.load(directory / "vae-decode-input.pt", weights_only=True)["latents"]
    operand = packed.to(device=pipeline.device, dtype=pipeline.vae.dtype)
    decode = pipeline._experiment_original_vae_decode
    from vllm_omni.platforms import current_omni_platform

    samples = []
    with torch.inference_mode():
        for i in range(25):
            current_omni_platform.synchronize()
            start = time.perf_counter()
            output = decode(operand, return_dict=False)[0]
            current_omni_platform.synchronize()
            elapsed = time.perf_counter() - start
            if i >= 5:
                samples.append(elapsed)
            del output
    write_json(
        directory / "vae-stage-timing.json",
        {
            "warmups": 5,
            "measured": 20,
            "seconds": samples,
            "scope": "Synchronized VAE decode on saved checkpoint latents; no HTTP or capture overhead",
            "shape": list(operand.shape),
            "dtype": str(operand.dtype),
        },
    )
    packed = torch.load(directory / "decode-input.pt", weights_only=True)["packed_latents"].to(pipeline.device)
    samples = []
    with torch.inference_mode():
        for i in range(25):
            current_omni_platform.synchronize()
            start = time.perf_counter()
            output = pipeline._experiment_original_pipeline_decode(packed, case["height"], case["width"])
            current_omni_platform.synchronize()
            if i >= 5:
                samples.append(time.perf_counter() - start)
            del output
    write_json(
        directory / "decode-helper-stage-timing.json",
        {
            "warmups": 5,
            "measured": 20,
            "seconds": samples,
            "scope": "Full decode helper including unpack, mean/std construction and normalization, plus VAE",
            "shape": list(packed.shape),
            "dtype": str(packed.dtype),
        },
    )


def debug_scheduler(scheduler):
    """Capture scheduler state that context replay does not restore."""
    return {
        "class": type(scheduler).__module__ + "." + type(scheduler).__name__,
        "config": dict(scheduler.config),
        "sigmas": getattr(scheduler, "sigmas", None),
        "timesteps": getattr(scheduler, "timesteps", None),
        "step_index": getattr(scheduler, "_step_index", None),
        "begin_index": getattr(scheduler, "_begin_index", None),
        "num_inference_steps": getattr(scheduler, "num_inference_steps", None),
    }


def debug_model_manifest(transformer):
    import torch

    rows = []
    # Sequential CPU copies bound host memory to the largest parameter. Do not
    # retain copies of the whole DiT. This diagnostic is not timed inference.
    for name, tensor in list(transformer.named_parameters()) + list(transformer.named_buffers()):
        value = tensor.detach().contiguous().to("cpu")
        raw = value.reshape(-1).view(torch.uint8).numpy()
        rows.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        del value, raw
    payload = json.dumps(rows, sort_keys=True).encode()
    return {
        "tensor_manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "tensors": rows,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k
            in {
                "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR",
                "TORCHINDUCTOR_COMPILE_THREADS",
                "HIP_VISIBLE_DEVICES",
                "VLLM_ROCM_USE_AITER",
                "QWEN_EXPERIMENT_BASELINE_DEBUG",
                "QWEN_EXPERIMENT_REPLAY_DIR",
                "QWEN_EXPERIMENT_VARIANT",
            }
        },
        "scope": "B1 T2I first-divergence debug only; CPU copies synchronize GPU and invalidate timing claims.",
    }


def install_first_divergence_debug(cls):
    if os.environ.get("QWEN_EXPERIMENT_BASELINE_DEBUG") != "1":
        return
    original_encode = cls.prepare_encode
    original_step = cls._denoise_step_homogeneous
    original_scheduler = cls.step_scheduler

    @functools.wraps(original_encode)
    def encode(self, state, **kwargs):
        result = original_encode(self, state, **kwargs)
        if ACTIVE is None:
            return result
        if state.latents.shape[0] != 1 or state.extra["image_latents"] is not None or state.do_true_cfg:
            raise RuntimeError("Baseline-debug requires one B1 T2I case without CFG")
        save("debug-scheduler-initial.pt", debug_scheduler(state.scheduler))
        if not getattr(self, "_experiment_debug_weights_saved", False):
            write_json(Path(CAPTURE).parent / "debug-transformer-manifest.json", debug_model_manifest(self.transformer))
            self._experiment_debug_weights_saved = True
        import torch

        save("debug-rng-initial.pt", {"cpu": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state()})
        if not getattr(self.transformer, "_experiment_debug_forward_wrapped", False):
            forward = self.transformer.forward

            @functools.wraps(forward)
            def outer_forward(*args, **kw):
                value = forward(*args, **kw)
                if ACTIVE is not None and "debug_step" in ACTIVE:
                    branch = kw.get("cache_branch", "cond")
                    payload = value[0] if isinstance(value, tuple) else value.sample
                    save(f"debug-step-{ACTIVE['debug_step']:03d}-transformer-{branch}.pt", payload)
                return value

            # Only the outer model forward is wrapped. Regional block forwards
            # and _decode_graph_forward remain unchanged; saves run after replay.
            self.transformer.forward = outer_forward
            self.transformer._experiment_debug_forward_wrapped = True
        return result

    @functools.wraps(original_step)
    def step(self, batch):
        if ACTIVE is None:
            return original_step(self, batch)
        if len(batch.states) != 1:
            raise RuntimeError("Baseline-debug requires exactly one state")
        index = batch.states[0].step_index
        ACTIVE["debug_step"] = index
        save(
            f"debug-step-{index:03d}-input.pt",
            {
                "latents": batch.latents,
                "timesteps": batch.timesteps,
                "prompt_embeds": batch.prompt_embeds,
                "prompt_embeds_mask": batch.prompt_embeds_mask,
                "img_shapes": batch.img_shapes,
                "true_cfg_scale": batch.true_cfg_scale,
                "do_true_cfg": batch.do_true_cfg,
            },
        )
        result = original_step(self, batch)
        save(f"debug-step-{index:03d}-noise.pt", result)
        if index == 0:
            save("debug-prefill-kv.pt", [dict(layer) for layer in batch.states[0].extra["kv_cache"]])
        ACTIVE.pop("debug_step", None)
        return result

    @functools.wraps(original_scheduler)
    def scheduler(self, state, noise_pred, **kwargs):
        if ACTIVE is None:
            return original_scheduler(self, state, noise_pred, **kwargs)
        index = state.step_index
        save(
            f"debug-step-{index:03d}-scheduler-before.pt",
            {
                "latents": state.latents,
                "noise_pred": noise_pred,
                "current_timestep": state.current_timestep,
                "scheduler": debug_scheduler(state.scheduler),
            },
        )
        result = original_scheduler(self, state, noise_pred, **kwargs)
        save(
            f"debug-step-{index:03d}-scheduler-after.pt",
            {
                "latents": state.latents,
                "step_index": state.step_index,
                "scheduler": debug_scheduler(state.scheduler),
            },
        )
        return result

    cls.prepare_encode = encode
    cls._denoise_step_homogeneous = step
    cls.step_scheduler = scheduler


def patch_pipeline(module):
    cls = module.QwenImage21Pipeline
    original_encode_prompt = cls.encode_prompt
    original_init = cls.__init__

    @functools.wraps(original_encode_prompt)
    def encode_prompt(self, *args, **kwargs):
        if DISABLE_CACHE:
            cache = getattr(self, "_prompt_embedding_cache", None)
            if cache is not None:
                cache.clear()
        return original_encode_prompt(self, *args, **kwargs)

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if CAPTURE:
            attach_diagnostics(self)
            self._experiment_original_pipeline_decode = original_finish.__get__(self, cls)

    cls.encode_prompt = encode_prompt
    cls.__init__ = initialize
    if not CAPTURE:
        return
    stats_cls = getattr(module, "_VaeLatentStats", None)
    if stats_cls is not None:
        original_get = stats_cls.get

        @functools.wraps(original_get)
        def stats_get(self, *args, **kwargs):
            if ACTIVE is None:
                return original_get(self, *args, **kwargs)
            import torch

            before = [x.data_ptr() if x is not None else None for x in (self.mean, self.std)]
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                result = original_get(self, *args, **kwargs)
            after = [x.data_ptr() for x in result]
            ACTIVE["latent_stats_calls"].append(
                {
                    "before": before,
                    "after": after,
                    "device": [str(x.device) for x in result],
                    "dtype": [str(x.dtype) for x in result],
                    "identity_reused": before == after and all(x is not None for x in before),
                    "operators": {event.key: event.count for event in profile.key_averages()},
                }
            )
            return result

        stats_cls.get = stats_get
    original_prepare = cls._prepare_generation_context
    original_finish = cls._decode_latents

    @functools.wraps(original_prepare)
    def prepare(self, **kwargs):
        global ACTIVE
        if not (Path(os.environ["QWEN_EXPERIMENT_RUN_DIR"]) / "enable-capture").exists():
            return original_prepare(self, **kwargs)
        case = identify(kwargs)
        directory = Path(CAPTURE) / case
        if (directory / "context.pt").exists():
            raise RuntimeError(f"Refusing to overwrite diagnostic capture for {case}")
        ACTIVE = {
            "case": case,
            "directory": directory,
            "height": kwargs["height"],
            "width": kwargs["width"],
            "encode_slot": 0,
            "encoder_slot": 0,
            "decoder_calls": 0,
            "graphs": [],
            "latent_stats_calls": [],
            "encoder_offload": [],
            "replayed_fields": [],
            "fusion_counts_before": dict(
                getattr(sys.modules.get("vllm_omni.diffusion.models.qwen_image_21.vae_experiment"), "COUNTS", {})
            ),
        }
        result = original_prepare(self, **kwargs)
        save("fresh-context.pt", result)
        if REPLAY and VARIANT in {"B0", "K1", "P1", "O1"}:
            frozen = load("context.pt")
            fields = ("latents", "image_latents") if VARIANT == "O1" else tuple(frozen)
            for field in fields:
                result[field] = to_like(frozen[field], result[field])
            ACTIVE["replayed_fields"] = list(fields)
        save("context.pt", result)
        return result

    @functools.wraps(original_finish)
    def finish(self, latents, height, width, output_type="pil"):
        global ACTIVE
        if ACTIVE is None:
            return original_finish(self, latents, height, width, output_type)
        if REPLAY and VARIANT in {"V1", "V2", "V3", "S2"}:
            latents = to_like(load("decode-input.pt")["packed_latents"], latents)
            ACTIVE["replayed_fields"].append("final_denoiser_latents")
        save("decode-input.pt", {"packed_latents": latents})
        result = original_finish(self, latents, height, width, output_type)
        save("decode-output.pt", {"output": result.output})
        case = ACTIVE
        fusion_module = sys.modules.get("vllm_omni.diffusion.models.qwen_image_21.vae_experiment")
        write_json(
            case["directory"] / "metadata.json",
            {
                "case_id": case["case"],
                "variant": VARIANT,
                "height": height,
                "width": width,
                "encoder_calls": case["encoder_slot"],
                "vae_encode_calls": case["encode_slot"],
                "decoder_calls": case["decoder_calls"],
                "vae_fusion_counts": {
                    key: value - case["fusion_counts_before"].get(key, 0)
                    for key, value in getattr(fusion_module, "COUNTS", {}).items()
                },
                "replayed_fields": case["replayed_fields"],
                "graph_calls": case["graphs"],
                "latent_stats_calls": case["latent_stats_calls"],
                "encoder_offload": case["encoder_offload"],
                "instrumentation": "outer pipeline/encoder/VAE wrappers; no compiled-block wrappers",
            },
        )
        ACTIVE = None
        if os.environ.get("QWEN_EXPERIMENT_STAGE_BENCH") == "1":
            stage_benchmark(self, case)
        return result

    cls._prepare_generation_context = prepare
    cls._decode_latents = finish
    original_step = cls._denoise_step_homogeneous

    @functools.wraps(original_step)
    def step(self, batch):
        if ACTIVE is not None and batch.states[0].step_index == 2 and not ACTIVE.get("profiled"):
            import torch

            ACTIVE["profiled"] = True
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            ) as profile:
                result = original_step(self, batch)
            profile.export_chrome_trace(str(ACTIVE["directory"] / "decode-step-trace.json"))
            return result
        return original_step(self, batch)

    cls._denoise_step_homogeneous = step
    install_first_divergence_debug(cls)


def patch_graph(module):
    if not CAPTURE:
        return
    cls = module.QwenImage21DecodeGraphManager
    original = cls.try_decode

    @functools.wraps(original)
    def call(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if ACTIVE is not None:
            ACTIVE["graphs"].append(
                {
                    "replayed": result is not None,
                    "entries": len(self.entries),
                    "captures": sum(e.captures for e in self.entries.values() if e is not None),
                    "retained_pairs": sum(
                        len(e.packed_k)
                        for e in self.entries.values()
                        if e is not None and getattr(e, "packed_k", None) is not None
                    ),
                }
            )
        return result

    cls.try_decode = call


def patch_runner(module):
    cls = module.DiffusionModelRunner
    original = cls.load_model

    @functools.wraps(original)
    def load_model(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        output = os.environ.get("QWEN_EXPERIMENT_RUN_DIR")
        if output:
            pipeline = self.pipeline
            blocks = pipeline.transformer.transformer_blocks
            write_json(
                Path(output) / f"worker-provenance-{os.getpid()}.json",
                {
                    "pid": os.getpid(),
                    "pipeline_file": sys.modules[PIPELINE].__file__,
                    "hook_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "native_prompt_cache_disabled": DISABLE_CACHE,
                    "diagnostic": bool(CAPTURE),
                    "variant": VARIANT,
                    "enforce_eager": self.od_config.enforce_eager,
                    "compiled_blocks": sum(hasattr(block.forward, "_torchdynamo_orig_callable") for block in blocks),
                    "decode_graphs_enabled": pipeline.transformer.enable_cuda_graph_decode,
                },
            )
        return result

    cls.load_model = load_model


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        {PIPELINE: patch_pipeline, GRAPH: patch_graph, RUNNER: patch_runner}[module.__name__](module)

    def __getattr__(self, name):
        return getattr(self.original, name)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in {PIPELINE, GRAPH, RUNNER}:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.loader:
            spec.loader = Loader(spec.loader)
        return spec


if DISABLE_CACHE or CAPTURE:
    sys.meta_path.insert(0, Finder())
