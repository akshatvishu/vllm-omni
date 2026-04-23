# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import Any

import torch

from ..config_ming_tts import (
    KEY_CFG,
    KEY_MAX_DECODE_STEPS,
    KEY_MIN_DECODE_STEPS,
    KEY_PROMPT_LATENTS,
    KEY_REQUEST_ID,
    KEY_SIGMA,
    KEY_SPEAKER_EMBEDDING,
    KEY_TEMPERATURE,
    LATENT_DIM,
    PATCH_SIZE,
)
from ._base import (
    coerce_speaker_embeddings,
    count_prompt_latent_patches,
    count_prompt_waveform_patches,
    create_instruction,
    estimate_decode_step_window_for_duration,
    pad_prompt_waveform,
    parse_duration_seconds,
)


def resolve_effective_runtime_controls(
    *,
    text: str,
    runtime_controls: dict[str, Any] | None = None,
) -> dict[str, Any]:
    controls = {} if runtime_controls is None else dict(runtime_controls)
    has_explicit_min = KEY_MIN_DECODE_STEPS in controls and controls[KEY_MIN_DECODE_STEPS] is not None
    has_explicit_max = KEY_MAX_DECODE_STEPS in controls and controls[KEY_MAX_DECODE_STEPS] is not None
    if has_explicit_min or has_explicit_max:
        return controls
    duration_seconds = parse_duration_seconds(text)
    if duration_seconds is None:
        return controls
    min_decode_steps, max_decode_steps = estimate_decode_step_window_for_duration(duration_seconds)
    controls[KEY_MIN_DECODE_STEPS] = min_decode_steps
    controls[KEY_MAX_DECODE_STEPS] = max_decode_steps
    return controls


def build_dense_prompt_token_ids(
    tokenizer: Any,
    *,
    prompt: str,
    text: str,
    instruction: str | None = None,
    prompt_text: str | None = None,
    speaker_count: int = 0,
    prompt_patch_count: int = 0,
) -> list[int]:
    speaker_prompt = []
    for idx in range(int(speaker_count)):
        speaker_prompt.extend(
            tokenizer.encode(f"  speaker_{idx + 1}:")
            + tokenizer.encode("<|vision_start|>")
            + tokenizer.encode("<|vision_pad|>")
            + tokenizer.encode("<|vision_end|>\n")
        )
    instruction_prompt = (
        tokenizer.encode(instruction) + tokenizer.encode("<|endoftext|>") if instruction is not None else []
    )
    prompt_text_tokens = (
        tokenizer.encode(prompt_text) if int(prompt_patch_count) > 0 and prompt_text is not None else []
    )
    prompt_latent_tokens = [tokenizer.convert_tokens_to_ids("<audioPatch>")] * int(prompt_patch_count)
    text_input_prefix = (
        []
        if all(token in text for token in ("Genre: ", "Mood: ", "Instrument: ", "Theme: ", "Duration: "))
        else tokenizer.encode(" Text input:\n")
    )
    return (
        tokenizer.encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
        + tokenizer.encode("<|im_start|>user\n")
        + tokenizer.encode(prompt)
        + speaker_prompt
        + text_input_prefix
        + prompt_text_tokens
        + tokenizer.encode(text)
        + tokenizer.encode("<|im_end|>\n")
        + tokenizer.encode("<|im_start|>assistant\n")
        + instruction_prompt
        + tokenizer.encode("<audio>")
        + prompt_latent_tokens
    )


def build_ming_dense_prompt(
    tokenizer: Any,
    *,
    prompt: str,
    text: str,
    runtime_controls: dict[str, Any] | None = None,
    instruction: Any = None,
    prompt_text: str | None = None,
    prompt_waveform: Any = None,
    prompt_latents: Any = None,
    speaker_embedding: Any = None,
    use_zero_spk_emb: bool = False,
    request_id: str | None = None,
) -> dict[str, Any]:
    instruction_text = create_instruction(instruction)
    speaker_embeddings = coerce_speaker_embeddings(speaker_embedding, use_zero_spk_emb=use_zero_spk_emb)
    effective_runtime_controls = resolve_effective_runtime_controls(text=text, runtime_controls=runtime_controls)

    prompt_waveform_tensor = None
    prompt_patch_count = 0
    if prompt_waveform is not None:
        prompt_waveform_tensor = pad_prompt_waveform(prompt_waveform)
        prompt_patch_count = count_prompt_waveform_patches(prompt_waveform_tensor)
    if prompt_waveform_tensor is not None and prompt_latents is not None:
        raise ValueError(
            "Ming waveform cloning request provided both raw prompt_waveform and explicit prompt_latents. "
            "Choose exactly one source of truth."
        )

    prompt_latent_value = None
    if prompt_waveform_tensor is not None and prompt_text is None:
        raise ValueError(
            "Ming prompt_waveform requires prompt_text for prompt-latent conditioning. "
            "Use speaker_embedding for reference-audio-only speaker conditioning."
        )
    if prompt_latents is not None:
        prompt_latent_value = torch.as_tensor(prompt_latents)
        prompt_patch_count = count_prompt_latent_patches(
            prompt_latent_value, patch_size=PATCH_SIZE, latent_dim=LATENT_DIM
        )

    prompt_token_ids = build_dense_prompt_token_ids(
        tokenizer,
        prompt=prompt,
        text=text,
        instruction=instruction_text,
        prompt_text=prompt_text if prompt_patch_count > 0 else None,
        speaker_count=0 if speaker_embeddings is None else len(speaker_embeddings),
        prompt_patch_count=prompt_patch_count,
    )

    additional_information = {}
    for key, value in effective_runtime_controls.items():
        if isinstance(value, torch.Tensor):
            additional_information[key] = value
        elif key in (KEY_MIN_DECODE_STEPS, KEY_MAX_DECODE_STEPS):
            additional_information[key] = torch.tensor(int(value), dtype=torch.int32)
        else:
            additional_information[key] = torch.tensor(float(value), dtype=torch.float32)
    if request_id is not None:
        additional_information[KEY_REQUEST_ID] = request_id
    if instruction_text is not None:
        additional_information["instruction"] = instruction_text
    if prompt_text is not None:
        additional_information["prompt_text"] = prompt_text
    if prompt_waveform_tensor is not None:
        additional_information["prompt_waveform"] = prompt_waveform_tensor
        additional_information["prompt_waveform_length"] = torch.tensor(
            [int(prompt_waveform_tensor.shape[-1])], dtype=torch.int32
        )
    if prompt_latent_value is not None:
        additional_information[KEY_PROMPT_LATENTS] = prompt_latent_value
    if speaker_embeddings is not None:
        additional_information[KEY_SPEAKER_EMBEDDING] = (
            speaker_embeddings[0] if len(speaker_embeddings) == 1 else torch.stack(speaker_embeddings, dim=0)
        )
    if use_zero_spk_emb:
        additional_information["use_zero_spk_emb"] = True
    return {
        "prompt": prompt,
        "text": text,
        "prompt_token_ids": prompt_token_ids,
        "additional_information": additional_information,
    }


def build_runtime_controls(
    *,
    cfg: float | None = None,
    sigma: float | None = None,
    temperature: float | None = None,
    min_decode_steps: int | None = None,
    max_decode_steps: int | None = None,
) -> dict[str, torch.Tensor]:
    controls = {}
    if cfg is not None:
        controls[KEY_CFG] = torch.tensor(float(cfg), dtype=torch.float32)
    if sigma is not None:
        controls[KEY_SIGMA] = torch.tensor(float(sigma), dtype=torch.float32)
    if temperature is not None:
        controls[KEY_TEMPERATURE] = torch.tensor(float(temperature), dtype=torch.float32)
    if min_decode_steps is not None:
        controls[KEY_MIN_DECODE_STEPS] = torch.tensor(int(min_decode_steps), dtype=torch.int32)
    if max_decode_steps is not None:
        controls[KEY_MAX_DECODE_STEPS] = torch.tensor(int(max_decode_steps), dtype=torch.int32)
    return controls


__all__ = [
    "build_dense_prompt_token_ids",
    "build_ming_dense_prompt",
    "build_runtime_controls",
    "resolve_effective_runtime_controls",
]
