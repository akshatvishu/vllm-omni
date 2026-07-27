# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline inference for SenseNova-U1-8B-MoT.

The example keeps the SenseNova-specific command line used by the native
reference workflow while routing execution through vLLM-Omni.  In particular,
``--cache-backend tea_cache`` is passed to the diffusion engine so the native
SenseNova TeaCache target is selected during stage initialization.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

DEFAULT_MODEL = "SenseNova/SenseNova-U1-8B-MoT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SenseNova-U1 text-to-image / image-to-image / text / understanding via vLLM-Omni."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model ID or local path.")
    parser.add_argument(
        "--modality",
        default="auto",
        choices=["auto", "text2img", "img2img", "img2text", "text2text"],
        help="Task modality. 'auto' selects text2img unless --image is supplied.",
    )
    parser.add_argument(
        "--prompt",
        default="A cute cat sitting on a windowsill, soft natural light",
        help="Text prompt for generation, editing, or understanding.",
    )
    parser.add_argument("--image", nargs="+", metavar="PATH", help="Input image path(s) for image modalities.")
    parser.add_argument("--output", type=str, default=".", help="Output directory for generated images.")
    parser.add_argument("--height", type=int, default=2048, help="Height of the generated image.")
    parser.add_argument("--width", type=int, default=2048, help="Width of the generated image.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="Text classifier-free guidance scale.")
    parser.add_argument(
        "--img-cfg-scale",
        type=float,
        default=1.0,
        help="Image CFG scale for img2img; 1.0 disables the unconditional image branch.",
    )
    parser.add_argument(
        "--cfg-norm",
        default="none",
        choices=["none", "global", "channel", "cfg_zero_star"],
        help="CFG normalization mode; cfg_zero_star is text-to-image only.",
    )
    parser.add_argument("--timestep-shift", type=float, default=3.0, help="Flow-matching timestep shift.")
    parser.add_argument("--t-eps", type=float, default=0.02, help="Flow-matching timestep epsilon.")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum text generation tokens.")
    parser.add_argument("--do-sample", action="store_true", help="Sample instead of greedy text decoding.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Text sampling temperature.")
    parser.add_argument("--think", action="store_true", help="Enable SenseNova reasoning before image generation.")
    parser.add_argument("--print-think", action="store_true", help="Print SenseNova reasoning text.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs for tensor parallelism.")
    parser.add_argument("--cfg-parallel-size", type=int, default=1, help="Number of GPUs for CFG parallelism.")
    parser.add_argument("--enforce-eager", action="store_true", help="Force eager execution.")
    parser.add_argument("--enable-cpu-offload", action="store_true", help="Enable module-wise CPU offload.")
    parser.add_argument(
        "--cache-backend",
        choices=["cache_dit", "tea_cache"],
        default=None,
        help="Diffusion cache backend. TeaCache is natively supported for SenseNova-U1.",
    )
    parser.add_argument(
        "--enable-cache-dit-summary",
        action="store_true",
        help="Print Cache-DiT summary information after generation.",
    )
    return parser.parse_args()


def _resolve_modality(args: argparse.Namespace) -> str:
    if args.modality != "auto":
        return args.modality
    return "img2img" if args.image else "text2img"


def _extract_think_text(outputs: list[Any]) -> str | None:
    """Read model metadata without depending on a single output wrapper."""
    for output in outputs:
        multimodal_output = getattr(output, "multimodal_output", None)
        if hasattr(multimodal_output, "to_dict"):
            multimodal_output = multimodal_output.to_dict()
        if not isinstance(multimodal_output, Mapping):
            continue
        metadata = multimodal_output.get("metadata", {})
        if isinstance(metadata, Mapping):
            text_metadata = metadata.get("text", {})
            if isinstance(text_metadata, Mapping) and text_metadata.get("think_text"):
                return str(text_metadata["think_text"])
    return None


def main() -> None:
    args = parse_args()
    modality = _resolve_modality(args)
    is_text_output = modality in ("text2text", "img2text")
    needs_images = modality in ("img2img", "img2text")

    if needs_images and not args.image:
        raise ValueError(f"{modality} requires at least one --image path.")

    os.makedirs(args.output, exist_ok=True)
    omni = Omni(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        enable_cpu_offload=args.enable_cpu_offload,
        cache_backend=args.cache_backend,
        enable_cache_dit_summary=args.enable_cache_dit_summary,
        cfg_parallel_size=args.cfg_parallel_size,
    )

    extra_args: dict[str, Any] = {
        "cfg_scale": args.cfg_scale,
        "cfg_norm": args.cfg_norm,
        "timestep_shift": args.timestep_shift,
        "cfg_interval": (0.0, 1.0),
        "batch_size": 1,
        "think": args.think,
        "t_eps": args.t_eps,
    }
    if modality == "img2img":
        extra_args["img_cfg_scale"] = args.img_cfg_scale
    if is_text_output:
        extra_args.update(
            max_tokens=args.max_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
        )

    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        seed=args.seed,
        num_inference_steps=args.num_steps,
        extra_args=extra_args,
    )

    if needs_images:
        input_images = [Image.open(path).convert("RGB") for path in args.image]
        prompt = {
            "prompt": args.prompt,
            "multi_modal_data": {"image": input_images},
            "modalities": ["text" if is_text_output else "img2img"],
        }
    else:
        prompt = {"prompt": args.prompt, "modalities": ["text" if is_text_output else "image"]}

    print(f"Model: {args.model}")
    print(f"Modality: {modality}")
    print(f"Cache backend: {args.cache_backend or 'none'}")
    if not is_text_output:
        print(f"Image size: {args.width}x{args.height}; steps: {args.num_steps}; seed: {args.seed}")

    outputs = list(omni.generate(prompts=prompt, sampling_params_list=sampling_params))

    if args.print_think:
        think_text = _extract_think_text(outputs)
        if think_text:
            print(f"[Think]\n{think_text}\n")

    if is_text_output:
        for output in outputs:
            multimodal_output = getattr(output, "multimodal_output", None)
            if hasattr(multimodal_output, "to_dict"):
                multimodal_output = multimodal_output.to_dict()
            if isinstance(multimodal_output, Mapping):
                metadata = multimodal_output.get("metadata", {})
                text_metadata = metadata.get("text", {}) if isinstance(metadata, Mapping) else {}
                text = text_metadata.get("text_output", "") if isinstance(text_metadata, Mapping) else ""
                if text:
                    print(f"[Response]\n{text}")
        return

    images = extract_images_from_outputs(outputs)
    if not images:
        raise ValueError("No images found in Omni output")
    for index, image in enumerate(images):
        output_path = Path(args.output) / f"sensenova_u1_output_{index}.png"
        image.save(output_path)
        print(f"[Output] Saved {image.width}x{image.height} image to {output_path}")


if __name__ == "__main__":
    main()
