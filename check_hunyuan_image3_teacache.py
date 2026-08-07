# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare Hunyuan Image 3 generation with and without native TeaCache."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

from vllm_omni import Omni
from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import build_prompt_tokens
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

DEFAULT_MODEL = "tencent/HunyuanImage-3.0-Instruct"
DEFAULT_DEPLOY_CONFIG = Path(__file__).parent / "vllm_omni/deploy/hunyuan_image3_dit.yaml"
DEFAULT_OUTPUT_DIR = Path("hunyuan_image3_teacache_results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--deploy-config", type=Path, default=DEFAULT_DEPLOY_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt", default="A brown and white dog running through a meadow")
    parser.add_argument("--system-prompt", default="en_unified")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[142, 444, 999])
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    return parser.parse_args()


def _build_prompt(model: str, prompt: str, system_prompt: str) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    prompt_tokens = build_prompt_tokens(
        prompt,
        tokenizer,
        task="t2i",
        bot_task="think",
        sys_type=system_prompt,
    )
    return {
        "prompt_token_ids": prompt_tokens.token_ids,
        "prompt": prompt,
        "use_system_prompt": prompt_tokens.system_prompt_type,
        "modalities": ["image"],
    }


def _build_omni(args: argparse.Namespace, use_cache: bool) -> Omni:
    cache_stage: dict[str, Any] = {"cache_backend": "tea_cache" if use_cache else "none"}
    if use_cache:
        cache_stage["cache_config"] = {"rel_l1_thresh": args.threshold}

    return Omni(
        model=args.model,
        deploy_config=str(args.deploy_config),
        mode="text-to-image",
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=True,
        enforce_eager=True,
        stage_overrides=json.dumps({"0": cache_stage}),
    )


def _extract_image(outputs: list[Any]) -> Image.Image:
    if not outputs:
        raise RuntimeError("Hunyuan returned no outputs")

    output = outputs[0]
    images = getattr(output, "images", None)
    if images:
        return images[0].convert("RGB")

    request_output = getattr(output, "request_output", None)
    images = getattr(request_output, "images", None) if request_output is not None else None
    if images:
        return images[0].convert("RGB")

    raise RuntimeError(f"Hunyuan output did not contain an image: {type(output).__name__}")


def _image_difference(first: Image.Image, second: Image.Image) -> dict[str, float]:
    first_array = np.asarray(first, dtype=np.float32)
    second_array = np.asarray(second, dtype=np.float32)
    if first_array.shape != second_array.shape:
        raise RuntimeError(f"Image shapes differ: {first_array.shape} versus {second_array.shape}")

    difference = np.abs(first_array - second_array)
    return {
        "mean_abs_pixel_difference": float(difference.mean()),
        "p99_abs_pixel_difference": float(np.percentile(difference, 99)),
        "max_abs_pixel_difference": float(difference.max()),
    }


def _run_case(
    args: argparse.Namespace,
    prompt: dict[str, Any],
    output_dir: Path,
    use_cache: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    omni = _build_omni(args, use_cache)
    timings: list[dict[str, Any]] = []

    try:
        for seed in args.seeds:
            sampling_params = OmniDiffusionSamplingParams(
                seed=seed,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                guidance_scale_provided=True,
                num_inference_steps=args.steps,
            )
            start = time.perf_counter()
            outputs = omni.generate(dict(prompt), sampling_params, use_tqdm=False)
            elapsed = time.perf_counter() - start

            image = _extract_image(outputs)
            image_path = output_dir / f"seed_{seed}.png"
            image.save(image_path)
            timings.append({"seed": seed, "seconds": elapsed, "image": str(image_path)})
            print(f"{'TeaCache' if use_cache else 'Baseline'} seed={seed}: {elapsed:.2f}s -> {image_path}")
    finally:
        omni.close()
        gc.collect()
        if torch.accelerator.is_available():
            torch.accelerator.empty_cache()

    return {
        "use_teacache": use_cache,
        "average_seconds": sum(item["seconds"] for item in timings) / len(timings),
        "runs": timings,
    }


def _write_comparison_grid(
    cached_dir: Path,
    baseline_dir: Path,
    seeds: list[int],
    output_path: Path,
) -> None:
    cached_images = [Image.open(cached_dir / f"seed_{seed}.png").convert("RGB") for seed in seeds]
    baseline_images = [Image.open(baseline_dir / f"seed_{seed}.png").convert("RGB") for seed in seeds]
    width, height = cached_images[0].size
    grid = Image.new("RGB", (width * len(seeds), height * 2))

    for index, image in enumerate(cached_images):
        grid.paste(image, (index * width, 0))
    for index, image in enumerate(baseline_images):
        grid.paste(image, (index * width, height))

    grid.save(output_path)


def main() -> None:
    args = parse_args()
    if not torch.accelerator.is_available():
        raise RuntimeError("This benchmark requires visible accelerator devices.")
    if torch.accelerator.device_count() < args.tensor_parallel_size:
        raise RuntimeError(
            f"Visible accelerator devices ({torch.accelerator.device_count()}) are fewer than "
            f"tensor_parallel_size={args.tensor_parallel_size}."
        )
    if not args.deploy_config.exists():
        raise FileNotFoundError(f"Deploy config not found: {args.deploy_config}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt = _build_prompt(args.model, args.prompt, args.system_prompt)
    baseline_dir = args.output_dir / "without_teacache"
    cached_dir = args.output_dir / "with_teacache"

    print(f"Artifacts: {args.output_dir.resolve()}")
    print(f"Deploy config: {args.deploy_config}")
    print(f"Seeds: {args.seeds}; steps: {args.steps}; threshold: {args.threshold}")

    baseline = _run_case(args, prompt, baseline_dir, use_cache=False)
    cached = _run_case(args, prompt, cached_dir, use_cache=True)

    quality: dict[str, dict[str, float]] = {}
    for seed in args.seeds:
        baseline_image = Image.open(baseline_dir / f"seed_{seed}.png").convert("RGB")
        cached_image = Image.open(cached_dir / f"seed_{seed}.png").convert("RGB")
        quality[str(seed)] = _image_difference(cached_image, baseline_image)

    grid_path = args.output_dir / "comparison_grid.png"
    _write_comparison_grid(cached_dir, baseline_dir, args.seeds, grid_path)

    summary = {
        "model": args.model,
        "deploy_config": str(args.deploy_config),
        "prompt": args.prompt,
        "steps": args.steps,
        "seeds": args.seeds,
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance_scale,
        "teacache_threshold": args.threshold,
        "baseline": baseline,
        "teacache": cached,
        "speedup": baseline["average_seconds"] / cached["average_seconds"],
        "quality": quality,
        "comparison_grid": str(grid_path),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nAverage baseline: {baseline['average_seconds']:.2f}s")
    print(f"Average TeaCache: {cached['average_seconds']:.2f}s")
    print(f"Speedup: {summary['speedup']:.2f}x")
    print(f"Comparison grid: {grid_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
