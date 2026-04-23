# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from vllm_omni.engine.stage_init_utils import _resolve_model_to_local_path

from .audio_tokenizer.modeling_audio_vae import AudioVAE
from .config_ming_tts import (
    KEY_PROMPT_LATENTS,
    VISION_START_TOKEN_ID,
    MingTTSConfig,
)
from .prompt_builder import (
    coerce_prompt_waveform,
    count_prompt_latent_patches,
    pad_prompt_waveform,
)


def load_weights(model_stage: str, model: Any, weights: list[tuple[str, torch.Tensor]]):
    if model_stage == "llm":
        allowed = ("model.", "linear_proj_audio.", "flowloss.", "stop_head.", "spk_head.")
        llm_weights = [(k, v) for k, v in weights if k.startswith(allowed)]
        if not llm_weights:
            raise RuntimeError(
                "Ming Stage-0 received no loadable checkpoint weights. "
                "Expected prefixes: model.*, linear_proj_audio.*, flowloss.*, stop_head.*, spk_head.*"
            )
        loaded = model.load_weights(llm_weights)
        return {f"model.{name}" for name in loaded}

    audio_weights = [(k, v) for k, v in weights if k.startswith("audio.")]
    if not audio_weights:
        raise RuntimeError("Ming Stage-1 received no loadable checkpoint weights. Expected prefix: audio.*")
    loaded = model.load_weights(audio_weights)
    return {f"model.{name}" for name in loaded}


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
        frames = patches.reshape(-1, latent_dim)
        return {"patches": patches, "frames": frames}

    if latents.ndim != 2 or latents.shape[-1] != latent_dim:
        raise ValueError(f"Unsupported prompt latent shape: {tuple(latents.shape)}")
    if latents.shape[0] % patch_size != 0:
        raise ValueError(
            f"Prompt latent frame count must be divisible by patch_size={patch_size}, "
            f"got frames={int(latents.shape[0])}"
        )
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


def _take_index(value: Any, idx: int) -> torch.Tensor | None:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return None
    return value[idx]


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
