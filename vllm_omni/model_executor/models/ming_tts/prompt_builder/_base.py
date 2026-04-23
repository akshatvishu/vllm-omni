# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import copy
import json
import math
import re
from typing import Any

import torch

from ..config_ming_tts import AUDIO_FRAME_HOP, LATENT_DIM, PATCH_SIZE, SAMPLE_RATE, VAE_PATCH_SIZE

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
    min_steps = max(1, target_steps - 3)
    max_steps = max(min_steps, target_steps + 3)
    return min_steps, max_steps


def pad_prompt_waveform(
    waveform: Any,
    *,
    patch_size: int = PATCH_SIZE,
    sample_rate: int = SAMPLE_RATE,
    frame_hop: int = AUDIO_FRAME_HOP,
) -> torch.Tensor:
    tensor = coerce_prompt_waveform(waveform)
    del frame_hop
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
        parts = [coerce_prompt_waveform(item) for item in value if item is not None]
        if not parts:
            raise ValueError("prompt waveform list was empty")
        return torch.cat(parts, dim=-1)
    return coerce_prompt_waveform(torch.as_tensor(value))


def coerce_speaker_embeddings(value: Any, *, use_zero_spk_emb: bool = False) -> list[torch.Tensor] | None:
    if value is None:
        return [torch.zeros((192,), dtype=torch.float32)] if use_zero_spk_emb else None
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
                items.append(item.detach().reshape(-1).to(torch.float32).cpu())
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
