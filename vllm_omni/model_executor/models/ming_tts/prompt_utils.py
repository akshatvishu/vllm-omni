# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
import math
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from vllm_omni.engine.stage_init_utils import _resolve_model_to_local_path
from vllm_omni.model_executor.models.ming_flash_omni.prompt_utils import (
    DEFAULT_PROMPT,
)
from vllm_omni.model_executor.models.ming_flash_omni.prompt_utils import (
    create_instruction as _create_ming_instruction,
)

from .audio_tokenizer.modeling_audio_vae import AudioVAE
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
    VISION_START_TOKEN_ID,
    MingTTSConfig,
)

_DURATION_SECONDS_RE = re.compile(r"Duration:\s*([0-9]+(?:\.[0-9]+)?)\s*s\b", re.IGNORECASE)


def create_instruction(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        raise ValueError(f"Ming instruction must be str or dict, got {type(value).__name__}")
    return _create_ming_instruction(value)


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


def _resolve_prompt_latents(wrapper: Any, info_dict: dict[str, Any]) -> dict[str, torch.Tensor] | None:
    raw_latents = info_dict.get(KEY_PROMPT_LATENTS, info_dict.get("prompt_latents"))
    raw_waveform = info_dict.get("prompt_waveform", info_dict.get("prompt_waveforms"))
    if raw_latents is not None and raw_waveform is not None:
        raise ValueError(
            "Ming waveform cloning request provided both raw prompt_waveform and explicit prompt_latents. "
            "Choose exactly one source of truth."
        )

    direct_latents = _coerce_prompt_latents(
        raw_latents,
        patch_size=wrapper.ming_config.patch_size,
        latent_dim=wrapper.ming_config.latent_dim,
    )
    if direct_latents is not None:
        return direct_latents
    if raw_waveform is None:
        return None

    encode_fn = getattr(wrapper, "_encode_prompt_waveform_to_latents", None)
    if callable(encode_fn):
        latents = encode_fn(raw_waveform, info_dict.get("prompt_waveform_length"))
    else:
        latents = _encode_prompt_waveform_to_latents(
            wrapper,
            raw_waveform,
            info_dict.get("prompt_waveform_length"),
        )
    return _coerce_prompt_latents(
        latents,
        patch_size=wrapper.ming_config.patch_size,
        latent_dim=wrapper.ming_config.latent_dim,
    )


def _load_prompt_encoder(wrapper: Any) -> AudioVAE:
    if wrapper._prompt_encoder is not None:
        return wrapper._prompt_encoder
    if wrapper.ming_config.audio_tokenizer_config is None:
        raise RuntimeError("Ming Stage-0 requires audio_tokenizer_config to encode prompt audio.")

    encoder = AudioVAE(wrapper.ming_config.audio_tokenizer_config).eval()
    state_dict = encoder.state_dict()
    loaded = 0
    loaded_encoder_params = set()
    with torch.no_grad():
        for shard_path in _iter_model_safetensors(
            _resolve_model_to_local_path(str(wrapper.vllm_config.model_config.model))
        ):
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if not key.startswith("audio.encoder."):
                        continue
                    name = key[len("audio.") :]
                    if name not in state_dict:
                        continue
                    target = state_dict[name]
                    target.copy_(handle.get_tensor(key).to(device=target.device, dtype=target.dtype))
                    loaded += 1
                    loaded_encoder_params.add(name)
    if loaded == 0:
        raise RuntimeError("Ming prompt encoder received no audio.encoder.* weights from checkpoint.")

    expected_encoder_params = {f"encoder.{name}" for name, _ in encoder.encoder.named_parameters()}
    missing = expected_encoder_params - loaded_encoder_params
    if missing:
        raise RuntimeError(f"Ming prompt encoder: {len(missing)} params not loaded. First few: {sorted(missing)[:5]}")

    dev = next(wrapper.parameters()).device
    try:
        del encoder.decoder
        encoder.decoder = None
        if dev.type != "cpu":
            encoder.encoder.to(dev, dtype=getattr(wrapper.model, "fm_dtype", torch.bfloat16))
        else:
            encoder.encoder.to(dev)
    except Exception as exc:
        raise RuntimeError(f"Failed to move Ming prompt encoder to {dev}: {exc}") from exc
    wrapper._prompt_encoder = encoder
    return encoder


@torch.inference_mode()
def _encode_prompt_waveform_to_latents(wrapper: Any, waveform: Any, waveform_length: Any = None) -> torch.Tensor:
    encoder = _load_prompt_encoder(wrapper)
    waveform = _normalize_prompt_waveform(waveform, target_sr=wrapper.ming_config.sample_rate)
    waveform = pad_prompt_waveform(
        waveform,
        patch_size=wrapper.ming_config.patch_size,
        sample_rate=wrapper.ming_config.sample_rate,
        frame_hop=wrapper.ming_config.audio_frame_hop,
    )
    dev = next(encoder.encoder.parameters()).device
    waveform = waveform.to(device=dev, dtype=next(encoder.encoder.parameters()).dtype)
    if waveform_length is None:
        waveform_length = torch.full((waveform.shape[0],), waveform.shape[-1], dtype=torch.int32, device=dev)
    elif not isinstance(waveform_length, torch.Tensor):
        waveform_length = torch.as_tensor(waveform_length, dtype=torch.int32, device=dev)
    else:
        waveform_length = waveform_length.to(device=dev, dtype=torch.int32)

    latents, _ = encoder.encode_latent(waveform, waveform_length)
    if latents.ndim == 3 and latents.shape[0] == 1:
        latents = latents.squeeze(0)
    count_prompt_latent_patches(
        latents,
        patch_size=wrapper.ming_config.patch_size,
        latent_dim=wrapper.ming_config.latent_dim,
    )
    return latents.detach().to(dtype=torch.float32).contiguous()


def _iter_model_safetensors(local_model_path: str) -> list[Path]:
    model_root = Path(local_model_path)
    index_path = model_root / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as handle:
            index_data = json.load(handle)
        filenames = sorted(set(index_data.get("weight_map", {}).values()))
        if not filenames:
            raise RuntimeError(f"No checkpoint shards listed in {index_path}")
        return [model_root / filename for filename in filenames]

    single_file = model_root / "model.safetensors"
    if single_file.exists():
        return [single_file]

    files = sorted(model_root.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"No .safetensors checkpoint found under {local_model_path}")
    return files


def _normalize_prompt_waveform(value: Any, *, target_sr: int) -> torch.Tensor:
    if isinstance(value, bytes):
        import torchaudio

        waveform, sr = torchaudio.load(BytesIO(value))
        waveform = waveform[:1].to(torch.float32)
        if int(sr) != int(target_sr):
            from torchaudio.functional import resample as resample_audio

            waveform = resample_audio(waveform, int(sr), int(target_sr))
        return waveform

    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], int):
        waveform = coerce_prompt_waveform(value[0])
        if int(value[1]) != int(target_sr):
            from torchaudio.functional import resample as resample_audio

            waveform = resample_audio(waveform, int(value[1]), int(target_sr))
        return waveform

    if isinstance(value, dict):
        samples = value.get("samples", value.get("array", value.get("waveform")))
        sr = value.get("sample_rate", value.get("sr", target_sr))
        return _normalize_prompt_waveform((samples, int(sr)), target_sr=target_sr)

    return coerce_prompt_waveform(value)


def _coerce_prompt_latents(
    value: Any,
    *,
    patch_size: int,
    latent_dim: int,
) -> dict[str, torch.Tensor] | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)

    latents = value.detach()
    if latents.ndim == 3 and latents.shape[0] == 1:
        latents = latents.squeeze(0)

    if latents.ndim == 3 and latents.shape[-2:] == (patch_size, latent_dim):
        patches = latents
        # [Patch, Time, Dimension] -> [Frame, Dimension] for history seeding.
        frames = patches.reshape(-1, latent_dim)
        return {"patches": patches, "frames": frames}

    if latents.ndim != 2 or latents.shape[-1] != latent_dim:
        raise ValueError(f"Unsupported prompt latent shape: {tuple(latents.shape)}")
    if latents.shape[0] % patch_size != 0:
        raise ValueError(
            f"Prompt latent frame count must be divisible by patch_size={patch_size}, "
            f"got frames={int(latents.shape[0])}"
        )
    # [Frame, Dimension] -> [Patch, Time, Dimension] for Aggregator prompt slots.
    patches = latents.reshape(-1, patch_size, latent_dim) if latents.shape[0] > 0 else None
    return {"patches": patches, "frames": latents}


def _initial_history(
    frames: torch.Tensor | None,
    *,
    history_size: int,
    latent_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    history = torch.zeros((history_size, latent_dim), device=device, dtype=dtype)
    if frames is None or frames.numel() == 0:
        return history
    frames = frames.to(device=device, dtype=dtype)
    take = min(history_size, int(frames.shape[0]))
    history[-take:] = frames[-take:]
    return history


def _take_scalar(value: Any, idx: int) -> float | None:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return None
    return float(value.reshape(-1)[idx].item())


def _find_audio_placeholder_positions(input_ids: torch.Tensor, cfg: MingTTSConfig) -> torch.Tensor:
    dummy_pos = (input_ids == cfg.audio_dummy_token_id).nonzero(as_tuple=True)[0]
    if dummy_pos.numel() == 0:
        return dummy_pos

    audio_start_pos = (input_ids == cfg.audio_start_token_id).nonzero(as_tuple=True)[0]
    audio_end_pos = (input_ids == cfg.audio_end_token_id).nonzero(as_tuple=True)[0]
    if audio_start_pos.numel() == 0:
        return dummy_pos

    start = int(audio_start_pos[0].item())
    end = int(audio_end_pos[0].item()) if audio_end_pos.numel() > 0 else int(input_ids.shape[0])
    keep = (dummy_pos > start) & (dummy_pos < end)
    filtered = dummy_pos[keep]
    return filtered if filtered.numel() > 0 else dummy_pos


def _find_speaker_placeholder_positions(input_ids: torch.Tensor, hf_config: Any) -> list[int]:
    vision_start_token_id = getattr(hf_config, "vision_start_token_id", VISION_START_TOKEN_ID)
    vision_start_pos = (input_ids == int(vision_start_token_id)).nonzero(as_tuple=True)[0]
    if vision_start_pos.numel() == 0:
        return []

    slots = []
    for pos in vision_start_pos:
        slot = int(pos.item()) + 1
        if slot < int(input_ids.shape[0]):
            slots.append(slot)
    return slots


__all__ = [
    "DEFAULT_PROMPT",
    "build_dense_prompt_token_ids",
    "build_ming_dense_prompt",
    "build_runtime_controls",
    "coerce_prompt_waveform",
    "coerce_speaker_embeddings",
    "count_prompt_latent_patches",
    "count_prompt_waveform_patches",
    "create_instruction",
    "estimate_decode_step_window_for_duration",
    "estimate_decode_steps_for_duration",
    "pad_prompt_waveform",
    "parse_duration_seconds",
    "resolve_effective_runtime_controls",
    "_coerce_prompt_latents",
    "_find_audio_placeholder_positions",
    "_find_speaker_placeholder_positions",
    "_initial_history",
    "_resolve_prompt_latents",
    "_take_scalar",
]
