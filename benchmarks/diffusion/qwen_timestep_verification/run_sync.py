# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run one fresh-process arm. Set PYTHONPATH to this directory for worker overlay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

PROMPTS = [
    "A red ceramic teapot on a wooden table in soft morning light.",
    "A snowy mountain reflected in a still lake at sunrise, detailed landscape photography.",
    'A bookstore window with a clear sign reading "OPEN TODAY", warm evening light.',
    "A watercolor painting of a small orange cat sleeping among blue flowers.",
    "A silver robot holding a green umbrella on a rainy city street, cinematic lighting.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=["baseline", "timestep"], required=True)
    parser.add_argument("--mode", choices=["ordinary", "step"], required=True)
    parser.add_argument("--phase", choices=["correctness", "timing", "profile"], required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--prompts", type=int, default=5, choices=range(2, 6))
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["QWEN_SYNC_ARM"] = args.arm
    os.environ["QWEN_SYNC_OUTPUT"] = str(args.output)
    os.environ.pop("QWEN_SYNC_AUDIT_DIR", None)
    if args.phase == "correctness":
        os.environ["QWEN_SYNC_AUDIT_DIR"] = str(args.output / "audit")
    overlay_dir = str(Path(__file__).resolve().parent)
    os.environ["PYTHONPATH"] = overlay_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    from sync_overlay import install
    install()
    import numpy as np
    import torch
    import vllm
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs

    engine_kwargs = dict(model=args.model, dtype="bfloat16", quantization="fp8",
        enforce_eager=args.eager, step_execution=args.mode == "step", max_num_seqs=1,
        diffusion_compile_granularity="regional", diffusion_compile_dynamic=True,
        diffusion_kv_cache_dtype="auto", enable_cpu_offload=False, enable_layerwise_offload=False)
    if args.phase == "profile":
        engine_kwargs["profiler_config"] = {"profiler": "torch", "torch_profiler_dir": str(args.output / "trace"),
            "torch_profiler_with_stack": True}
    report = {"args": vars(args), "engine": engine_kwargs, "torch": torch.__version__,
        "vllm": vllm.__version__, "hip": torch.version.hip, "gpu": torch.cuda.get_device_name(),
        "environment": {k:v for k,v in os.environ.items() if k.startswith(("VLLM", "HIP_VISIBLE", "ROCR_VISIBLE", "CUDA_VISIBLE"))},
        "requests": [], "complete": False}
    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    save()
    omni = Omni(**engine_kwargs)
    try:
        def one(prompt_index, measured, request_index):
            params = OmniDiffusionSamplingParams(height=args.size, width=args.size, seed=142+prompt_index,
                true_cfg_scale=4.0, guidance_scale=1.0, num_inference_steps=args.steps)
            prompt = {"prompt": PROMPTS[prompt_index], "negative_prompt": "", "modalities": ["image"]}
            start = time.perf_counter()
            outputs = omni.generate(prompt, sampling_params_list=[params], use_tqdm=False)
            elapsed = (time.perf_counter() - start) * 1000
            images = extract_images_from_outputs(outputs)
            assert len(images) == 1, len(images)
            image = images[0].convert("RGB")
            pixels = np.asarray(image)
            item = dict(index=request_index, prompt_index=prompt_index, prompt=PROMPTS[prompt_index],
                seed=142+prompt_index, measured=measured, request_ms=elapsed,
                pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
            image.save(args.output / f"request-{request_index:03d}.png")
            report["requests"].append(item)
            save()
            print("SYNC_SAMPLE " + json.dumps(item), flush=True)
        if args.phase == "correctness":
            for i in range(args.prompts):
                one(i, False, i)
        else:
            for i in range(args.warmup):
                one(0, False, i)
            if args.phase == "profile":
                omni.start_profile()
                one(0, False, args.warmup)
                report["profile"] = omni.stop_profile()
            else:
                for i in range(args.repeats):
                    one(0, True, i + args.warmup)
        proof = list(args.output.glob("overlay-*.json"))
        assert proof, "Worker did not load experimental overlay"
        assert all(json.loads(p.read_text())["arm"] == args.arm for p in proof)
        report["complete"] = True
        save()
    finally:
        omni.close()


if __name__ == "__main__":
    main()
