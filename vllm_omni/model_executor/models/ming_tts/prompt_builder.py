# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import copy
import json
import math
import re
from typing import Any

import torch

from .config_ming_tts import (
    AUDIO_FRAME_HOP,
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
    SAMPLE_RATE,
    VAE_PATCH_SIZE,
)

BASE_CAPTION_TEMPLATE = {
    "audio_sequence": [
        {
            "序号": 1,
            "说话人": "speaker_1",
            "方言": None,
            "风格": None,
            "语速": None,
            "基频": None,
            "音量": None,
            "情感": None,
            "BGM": {
                "Genre": None,
                "Mood": None,
                "Instrument": None,
                "Theme": None,
                "ENV": None,
                "SNR": None,
            },
            "IP": None,
        }
    ]
}

_DURATION_SECONDS_RE = re.compile(r"Duration:\s*([0-9]+(?:\.[0-9]+)?)\s*s\b", re.IGNORECASE)


def create_instruction(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        raise ValueError(f"Ming instruction must be str or dict, got {type(value).__name__}")

    caption = copy.deepcopy(BASE_CAPTION_TEMPLATE)
    target = caption["audio_sequence"][0]
    for key, item in value.items():
        if key in target:
            target[key] = item

    if target["BGM"].get("SNR") is not None:
        order = ["序号", "说话人", "BGM", "情感", "方言", "风格", "语速", "基频", "音量", "IP"]
        caption["audio_sequence"][0] = {key: target[key] for key in order if key in target}
    return json.dumps(caption, ensure_ascii=False)


def parse_duration_seconds(text: str | None) -> float | None:
    if not isinstance(text, str):
        return None
    match = _DURATION_SECONDS_RE.search(text)
    if match is None:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    if value <= 0.0:
        return None
    return value


def estimate_decode_steps_for_duration(
    duration_seconds: float,
    *,
    sample_rate: int = SAMPLE_RATE,
    frame_hop: int = AUDIO_FRAME_HOP,
    patch_size: int = PATCH_SIZE,
    vae_patch_size: int = VAE_PATCH_SIZE,
) -> int:
    if duration_seconds <= 0.0:
        return 0
    samples_per_decode_step = int(frame_hop) * int(patch_size) * int(vae_patch_size)
    required_samples = float(duration_seconds) * float(sample_rate)
    return max(1, int(math.ceil(required_samples / float(samples_per_decode_step))))


def estimate_decode_step_window_for_duration(duration_seconds: float) -> tuple[int, int]:
    target_steps = estimate_decode_steps_for_duration(duration_seconds)
    # Ming emits about 0.32s per decode step in the current dense path. Keep a narrow
    # duration window so BGM does not undershoot badly or run all the way to the generic cap.
    min_steps = max(1, target_steps - 3)
    max_steps = max(min_steps, target_steps + 3)
    return min_steps, max_steps


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


def pad_prompt_waveform(
    waveform: Any,
    *,
    patch_size: int = PATCH_SIZE,
    sample_rate: int = SAMPLE_RATE,
    frame_hop: int = AUDIO_FRAME_HOP,
) -> torch.Tensor:
    tensor = coerce_prompt_waveform(waveform)
    del frame_hop
    # Match upstream Ming exactly: tokenizer framerate is 12.5 Hz, so prompt
    # waveform padding aligns to sample_rate / 12.5 * patch_size samples.
    pad_align = int((float(sample_rate) / 12.5) * int(patch_size))
    new_len = ((int(tensor.shape[-1]) + pad_align - 1) // pad_align) * pad_align
    if new_len == int(tensor.shape[-1]):
        return tensor
    padded = torch.zeros((1, new_len), dtype=tensor.dtype, device=tensor.device)
    padded[:, : tensor.shape[-1]] = tensor
    return padded


def coerce_prompt_waveform(value: Any) -> torch.Tensor:
    if value is None:
        raise ValueError("prompt waveform cannot be None")
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.ndim == 1:
            return tensor.unsqueeze(0).to(torch.float32)
        if tensor.ndim == 2:
            if tensor.shape[0] != 1:
                return tensor.reshape(1, -1).to(torch.float32)
            return tensor.to(torch.float32)
        raise ValueError(f"Unsupported Ming prompt waveform rank: {tuple(tensor.shape)}")

    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            if item is None:
                continue
            parts.append(coerce_prompt_waveform(item))
        if not parts:
            raise ValueError("prompt waveform list was empty")
        return torch.cat(parts, dim=-1)

    return coerce_prompt_waveform(torch.as_tensor(value))


def coerce_speaker_embeddings(value: Any, *, use_zero_spk_emb: bool = False) -> list[torch.Tensor] | None:
    if value is None:
        if use_zero_spk_emb:
            return [torch.zeros((192,), dtype=torch.float32)]
        return None

    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise ValueError(f"Unsupported Ming speaker embedding shape: {tuple(tensor.shape)}")
        items = [row.reshape(-1).to(torch.float32).cpu() for row in tensor]
    elif isinstance(value, (list, tuple)):
        if value and all(not isinstance(item, (list, tuple, torch.Tensor)) for item in value):
            items = [torch.as_tensor(value).detach().reshape(-1).to(torch.float32).cpu()]
        else:
            items = []
            for item in value:
                if item is None:
                    continue
                if not isinstance(item, torch.Tensor):
                    item = torch.as_tensor(item)
                flat = item.detach().reshape(-1).to(torch.float32).cpu()
                items.append(flat)
    else:
        return coerce_speaker_embeddings(torch.as_tensor(value), use_zero_spk_emb=use_zero_spk_emb)

    if not items:
        return [torch.zeros((192,), dtype=torch.float32)] if use_zero_spk_emb else None
    for item in items:
        if int(item.numel()) != 192:
            raise ValueError(f"Ming speaker embedding must have 192 dims, got {int(item.numel())}")
    return items


def count_prompt_latent_patches(
    value: Any,
    *,
    patch_size: int = PATCH_SIZE,
    latent_dim: int = LATENT_DIM,
) -> int:
    if value is None:
        return 0
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)

    latents = value.detach()
    if latents.ndim == 3 and latents.shape[0] == 1:
        latents = latents.squeeze(0)

    if latents.ndim == 3 and latents.shape[-2:] == (patch_size, latent_dim):
        return int(latents.shape[0])

    if latents.ndim != 2 or latents.shape[-1] != latent_dim:
        raise ValueError(f"Unsupported Ming prompt_latents shape: {tuple(latents.shape)}")
    if latents.shape[0] % patch_size != 0:
        raise ValueError(
            f"Ming prompt_latents frame count must be divisible by patch_size={patch_size}, "
            f"got frames={int(latents.shape[0])}"
        )
    return int(latents.shape[0] // patch_size)


def count_prompt_waveform_patches(
    value: Any,
    *,
    patch_size: int = PATCH_SIZE,
    frame_hop: int = AUDIO_FRAME_HOP,
    vae_patch_size: int = VAE_PATCH_SIZE,
) -> int:
    if value is None:
        return 0
    waveform = pad_prompt_waveform(value, patch_size=patch_size, frame_hop=frame_hop)
    frame_count = int(math.ceil(float(waveform.shape[-1]) / float(frame_hop)))
    latent_frames = int(math.ceil(float(frame_count) / float(vae_patch_size)))
    if latent_frames % int(patch_size) != 0:
        raise ValueError(
            f"Ming prompt waveform produced latent frame count not divisible by patch_size={patch_size}: "
            f"frames={latent_frames}"
        )
    return int(latent_frames // int(patch_size))


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

    instruction_prompt = []
    if instruction is not None:
        instruction_prompt = tokenizer.encode(instruction) + tokenizer.encode("<|endoftext|>")

    prompt_text_tokens = []
    prompt_latent_tokens = []
    if int(prompt_patch_count) > 0:
        if prompt_text is not None:
            prompt_text_tokens = tokenizer.encode(prompt_text)
        prompt_latent_tokens = [tokenizer.convert_tokens_to_ids("<audioPatch>")] * int(prompt_patch_count)

    text_input_prefix = tokenizer.encode(" Text input:\n")
    if "Genre: " in text and "Mood: " in text and "Instrument: " in text and "Theme: " in text and "Duration: " in text:
        text_input_prefix = []

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
    effective_runtime_controls = resolve_effective_runtime_controls(
        text=text,
        runtime_controls=runtime_controls,
    )

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
            prompt_latent_value,
            patch_size=PATCH_SIZE,
            latent_dim=LATENT_DIM,
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
    if effective_runtime_controls:
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
            [int(prompt_waveform_tensor.shape[-1])],
            dtype=torch.int32,
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
