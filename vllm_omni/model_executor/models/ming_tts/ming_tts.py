# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
import os
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open
from vllm.config import VllmConfig
from vllm.model_executor.models import SupportsPP
from vllm.model_executor.models.utils import init_vllm_registered_model
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin

from .audio_tokenizer.modeling_audio_vae import AudioVAE
from .config_ming_tts import (
    AUDIO_START_TOKEN_ID,
    KEY_CFG,
    KEY_DECODE_STEP,
    KEY_LAST_STOP_PROB,
    KEY_LATENT_HISTORY,
    KEY_MAX_DECODE_STEPS,
    KEY_MIN_DECODE_STEPS,
    KEY_NEXT_EMBEDS,
    KEY_PROMPT_LATENT_TAIL,
    KEY_PROMPT_LATENTS,
    KEY_REQUEST_ID,
    KEY_SIGMA,
    KEY_SPEAKER_EMBEDDING,
    KEY_TEMPERATURE,
    KEY_TEXT_MODE,
    VISION_START_TOKEN_ID,
    MingTTSConfig,
)
from .prompt_builder import (
    coerce_prompt_waveform,
    coerce_speaker_embeddings,
    count_prompt_latent_patches,
    pad_prompt_waveform,
)

MING_STOP_REASON_KEY = "ming_stop_reason"


class _ModelSampleAdapter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, logits, sampling_metadata):
        return self.model.sample(logits, sampling_metadata)


class MingTTSForConditionalGeneration(nn.Module, SupportsPP, CustomProcessMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        self.ming_config = MingTTSConfig.from_hf_config(vllm_config.model_config.hf_config)
        self.ming_config.validate()

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.requires_raw_input_tokens = False

        self.model_stage = vllm_config.model_config.model_stage
        self._prompt_encoder = None

        if self.model_stage == "llm":
            self.model = init_vllm_registered_model(
                vllm_config=vllm_config,
                architectures=["MingLLMModel"],
            )
            self.has_preprocess = True
            self.has_postprocess = True
            self.set_custom_preprocess(self.preprocess)
            self.set_custom_postprocess(self.postprocess)
        elif self.model_stage == "audio_vae":
            self.model = init_vllm_registered_model(
                vllm_config=vllm_config,
                architectures=["MingAudioVAEModel"],
            )
            self.requires_raw_input_tokens = True
        else:
            raise ValueError(f"Invalid Ming model_stage={self.model_stage}")

        self.make_empty_intermediate_tensors = getattr(self.model, "make_empty_intermediate_tensors", lambda: None)

    @cached_property
    def sampler(self):
        if hasattr(self.model, "sample"):
            return _ModelSampleAdapter(self.model)
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        return Sampler()

    def embed_input_ids(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids=input_ids, **kwargs)

    def forward(self, *args: Any, **kwargs: Any):
        return self.model(*args, **kwargs)

    def compute_logits(self, hidden_states, sampling_metadata=None):
        return self.model.compute_logits(hidden_states, sampling_metadata=sampling_metadata)

    def sample(self, logits, sampling_metadata):
        if hasattr(self.model, "sample"):
            return self.model.sample(logits, sampling_metadata)
        return None

    def load_weights(self, weights):
        weights = list(weights)
        if self.model_stage == "llm":
            allowed = ("model.", "linear_proj_audio.", "flowloss.", "stop_head.", "spk_head.")
            llm_weights = [(k, v) for k, v in weights if k.startswith(allowed)]
            if not llm_weights:
                raise RuntimeError(
                    "Ming Stage-0 received no loadable checkpoint weights. "
                    "Expected prefixes: model.*, linear_proj_audio.*, flowloss.*, stop_head.*, spk_head.*"
                )
            loaded = self.model.load_weights(llm_weights)
            return {f"model.{name}" for name in loaded}

        audio_weights = [(k, v) for k, v in weights if k.startswith("audio.")]
        if not audio_weights:
            raise RuntimeError("Ming Stage-1 received no loadable checkpoint weights. Expected prefix: audio.*")
        loaded = self.model.load_weights(audio_weights)
        return {f"model.{name}" for name in loaded}

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ):
        if self.model_stage != "llm":
            return input_ids, input_embeds, {}

        # vLLM hands Stage-0 a scratch inputs_embeds buffer that is zeroed at
        # preprocess time and later becomes corrupted before the backbone call.
        # Rebuild a fresh [T,H] embedding tensor from token ids here instead of
        # trusting the runtime-provided buffer.
        input_embeds = self.model.embed_input_ids(input_ids).clone()

        span_len = int(input_ids.shape[0])
        if span_len > 1:
            return self._prefill_preprocess(input_ids, input_embeds, **info_dict)
        return self._decode_preprocess(input_ids, input_embeds, **info_dict)

    def preprocess_input(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ):
        return self.preprocess(input_ids, input_embeds, **info_dict)

    def postprocess(self, hidden_states: torch.Tensor, **info_dict: Any) -> dict[str, Any]:
        if self.model_stage != "llm" or hidden_states.numel() == 0:
            return {}

        req_id = info_dict.get(KEY_REQUEST_ID, info_dict.get("req_id"))
        pending = self.model.pop_postprocess_update(req_id)
        if not pending:
            return {}

        latent_patch = pending.get("ming_latent_patch")
        next_embeds = pending.get(KEY_NEXT_EMBEDS)
        new_history = pending.get(KEY_LATENT_HISTORY)
        stop_prob = _take_scalar(pending.get("ming_stop_prob"), 0)
        stop_reason = pending.get(MING_STOP_REASON_KEY)
        if not isinstance(latent_patch, torch.Tensor):
            return {}

        decode_step = int(info_dict.get(KEY_DECODE_STEP, 0))
        update = {
            KEY_LATENT_HISTORY: new_history.detach().to("cpu").contiguous(),
            KEY_NEXT_EMBEDS: next_embeds.detach().to("cpu").contiguous(),
            KEY_DECODE_STEP: decode_step + 1,
        }
        if stop_prob is not None:
            update[KEY_LAST_STOP_PROB] = stop_prob
        if isinstance(stop_reason, str):
            update[MING_STOP_REASON_KEY] = stop_reason
        if isinstance(req_id, str):
            update[KEY_REQUEST_ID] = req_id
        return update

    def _prefill_preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        **info_dict: Any,
    ):
        if bool(info_dict.get(KEY_TEXT_MODE, False)):
            update: dict[str, Any] = {KEY_TEXT_MODE: True}
            request_id = info_dict.get(KEY_REQUEST_ID, info_dict.get("req_id"))
            if request_id is not None:
                update[KEY_REQUEST_ID] = request_id
            if int(input_ids.shape[0]) > 1 and int(input_ids[-1].item()) == AUDIO_START_TOKEN_ID:
                return input_ids[:-1], input_embeds[:-1], update
            return input_ids, input_embeds, update

        update: dict[str, Any] = {
            KEY_DECODE_STEP: int(info_dict.get(KEY_DECODE_STEP, 0)),
        }

        prompt_latents = self._resolve_prompt_latents(info_dict)
        history = _initial_history(
            prompt_latents["frames"] if prompt_latents is not None else None,
            history_size=self.ming_config.history_patch_size,
            latent_dim=self.ming_config.latent_dim,
            device=input_embeds.device,
            dtype=torch.float32,
        )
        update[KEY_LATENT_HISTORY] = history.detach().to("cpu").contiguous()
        update[KEY_PROMPT_LATENT_TAIL] = update[KEY_LATENT_HISTORY]

        speaker_embedding = info_dict.get(KEY_SPEAKER_EMBEDDING, info_dict.get("speaker_embedding"))
        speaker_embeddings = coerce_speaker_embeddings(
            speaker_embedding,
            use_zero_spk_emb=bool(info_dict.get("use_zero_spk_emb", False)),
        )
        speaker_slots: list[int] = []
        if speaker_embeddings:
            speaker_slots = _find_speaker_placeholder_positions(input_ids, self.vllm_config.model_config.hf_config)
            if len(speaker_slots) < len(speaker_embeddings):
                raise RuntimeError(
                    f"Could not locate enough speaker placeholder slots: found {len(speaker_slots)}, "
                    f"need {len(speaker_embeddings)}"
                )
            for speaker_slot, spk in zip(speaker_slots, speaker_embeddings):
                spk_proj = self.model.project_speaker_embedding(
                    spk.to(device=input_embeds.device, dtype=input_embeds.dtype).unsqueeze(0)
                ).squeeze(0)
                input_embeds[speaker_slot] = spk_proj

        if prompt_latents is not None and prompt_latents["patches"] is not None:
            prompt_patches = prompt_latents["patches"].to(
                dtype=getattr(self.model, "fm_dtype", torch.float32),
            )
            prompt_embeds = self.model.linear_proj_audio(prompt_patches).squeeze(1)
            placeholder_pos = _find_audio_placeholder_positions(input_ids, self.ming_config)
            take = min(int(placeholder_pos.numel()), int(prompt_embeds.shape[0]))
            if take > 0:
                input_embeds[placeholder_pos[:take]] = prompt_embeds[:take].to(dtype=input_embeds.dtype)

        request_id = info_dict.get(KEY_REQUEST_ID, info_dict.get("req_id"))
        if request_id is not None:
            update[KEY_REQUEST_ID] = request_id
        _copy_runtime_controls(update, info_dict)
        return input_ids, input_embeds, update

    def _resolve_prompt_latents(self, info_dict: dict[str, Any]) -> dict[str, torch.Tensor] | None:
        raw_latents = info_dict.get(KEY_PROMPT_LATENTS, info_dict.get("prompt_latents"))
        raw_waveform = info_dict.get("prompt_waveform", info_dict.get("prompt_waveforms"))
        if raw_latents is not None and raw_waveform is not None:
            raise ValueError(
                "Ming waveform cloning request provided both raw prompt_waveform and explicit prompt_latents. "
                "Choose exactly one source of truth."
            )

        direct_latents = _coerce_prompt_latents(
            raw_latents,
            patch_size=self.ming_config.patch_size,
            latent_dim=self.ming_config.latent_dim,
        )
        if direct_latents is not None:
            return direct_latents

        if raw_waveform is None:
            return None
        waveform_length = info_dict.get("prompt_waveform_length")
        latents = self._encode_prompt_waveform_to_latents(
            raw_waveform,
            waveform_length,
        )
        return _coerce_prompt_latents(
            latents,
            patch_size=self.ming_config.patch_size,
            latent_dim=self.ming_config.latent_dim,
        )

    def _load_prompt_encoder(self) -> AudioVAE:
        if self._prompt_encoder is not None:
            return self._prompt_encoder
        if self.ming_config.audio_tokenizer_config is None:
            raise RuntimeError("Ming Stage-0 requires audio_tokenizer_config to encode prompt audio.")

        encoder = AudioVAE(self.ming_config.audio_tokenizer_config).eval()
        state_dict = encoder.state_dict()
        loaded = 0
        loaded_encoder_params = set()
        with torch.no_grad():
            for shard_path in _iter_model_safetensors(
                _resolve_model_to_local_path(str(self.vllm_config.model_config.model))
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
        # Ensure the encode-only Stage-0 VAE is not silently running with random encoder weights.
        expected_encoder_params = {f"encoder.{name}" for name, _ in encoder.encoder.named_parameters()}
        missing = expected_encoder_params - loaded_encoder_params
        if missing:
            raise RuntimeError(
                f"Ming prompt encoder: {len(missing)} params not loaded. First few: {sorted(missing)[:5]}"
            )

        dev = next(self.parameters()).device
        try:
            del encoder.decoder
            encoder.decoder = None
            if dev.type != "cpu":
                encoder.encoder.to(dev, dtype=getattr(self.model, "fm_dtype", torch.bfloat16))
            else:
                encoder.encoder.to(dev)
        except Exception as e:
            raise RuntimeError(f"Failed to move Ming prompt encoder to {dev}: {e}") from e
        self._prompt_encoder = encoder
        return encoder

    @torch.inference_mode()
    def _encode_prompt_waveform_to_latents(self, waveform: Any, waveform_length: Any = None) -> torch.Tensor:
        encoder = self._load_prompt_encoder()
        waveform = _normalize_prompt_waveform(waveform, target_sr=self.ming_config.sample_rate)
        waveform = pad_prompt_waveform(
            waveform,
            patch_size=self.ming_config.patch_size,
            sample_rate=self.ming_config.sample_rate,
            frame_hop=self.ming_config.audio_frame_hop,
        )
        dev = next(encoder.encoder.parameters()).device
        waveform = waveform.to(device=dev, dtype=next(encoder.encoder.parameters()).dtype)
        if waveform_length is None:
            waveform_length = torch.full(
                (waveform.shape[0],),
                waveform.shape[-1],
                dtype=torch.int32,
                device=dev,
            )
        elif not isinstance(waveform_length, torch.Tensor):
            waveform_length = torch.as_tensor(waveform_length, dtype=torch.int32, device=dev)
        else:
            waveform_length = waveform_length.to(device=dev, dtype=torch.int32)

        latents, _ = encoder.encode_latent(waveform, waveform_length)
        if latents.ndim == 3 and latents.shape[0] == 1:
            latents = latents.squeeze(0)
        count_prompt_latent_patches(
            latents,
            patch_size=self.ming_config.patch_size,
            latent_dim=self.ming_config.latent_dim,
        )
        return latents.detach().to(dtype=torch.float32).contiguous()

    def _decode_preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        **info_dict: Any,
    ):
        if bool(info_dict.get(KEY_TEXT_MODE, False)):
            update: dict[str, Any] = {KEY_TEXT_MODE: True}
            request_id = info_dict.get(KEY_REQUEST_ID, info_dict.get("req_id"))
            if request_id is not None:
                update[KEY_REQUEST_ID] = request_id
            return input_ids, input_embeds, update

        update: dict[str, Any] = {
            KEY_DECODE_STEP: int(info_dict.get(KEY_DECODE_STEP, 0)),
        }

        history = info_dict.get(KEY_LATENT_HISTORY)
        if isinstance(history, torch.Tensor):
            update[KEY_LATENT_HISTORY] = history.detach().to("cpu").contiguous()
        else:
            zero_history = torch.zeros(
                (self.ming_config.history_patch_size, self.ming_config.latent_dim),
                device=input_embeds.device,
                dtype=torch.float32,
            )
            update[KEY_LATENT_HISTORY] = zero_history.detach().to("cpu").contiguous()

        next_embeds = info_dict.get(KEY_NEXT_EMBEDS)
        if isinstance(next_embeds, torch.Tensor) and input_ids.numel() == 1:
            if not torch.isfinite(next_embeds).all():
                raise RuntimeError("Non-finite next_embeds before decode preprocess write.")
            next_step = next_embeds.detach().reshape(-1, self.ming_config.llm_hidden_size)[0]
            input_embeds[0] = next_step.to(device=input_embeds.device, dtype=input_embeds.dtype)
            if not torch.isfinite(input_embeds[0]).all():
                raise RuntimeError("Non-finite backbone input_embeds after decode preprocess write.")

        request_id = info_dict.get(KEY_REQUEST_ID, info_dict.get("req_id"))
        if request_id is not None:
            update[KEY_REQUEST_ID] = request_id
        _copy_runtime_controls(update, info_dict)
        return input_ids, input_embeds, update


def _copy_runtime_controls(update: dict[str, Any], info_dict: dict[str, Any]) -> None:
    for key in (KEY_CFG, KEY_SIGMA, KEY_TEMPERATURE, KEY_MAX_DECODE_STEPS, KEY_MIN_DECODE_STEPS):
        if key in info_dict:
            update[key] = info_dict[key]


def _resolve_model_to_local_path(model: str) -> str:
    if os.path.isdir(model):
        return model
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Ming Stage-0 prompt encoder requires a local model snapshot, got {model!r}. "
            "Download the model first or pass a local path."
        ) from exc


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
        # [B,T,D] patch history -> [T,D] flat frame history for Stage-1 seeding.
        frames = patches.reshape(-1, latent_dim)
        return {"patches": patches, "frames": frames}

    if latents.ndim != 2 or latents.shape[-1] != latent_dim:
        raise ValueError(f"Unsupported prompt latent shape: {tuple(latents.shape)}")

    if latents.shape[0] % patch_size != 0:
        raise ValueError(
            f"Prompt latent frame count must be divisible by patch_size={patch_size}, "
            f"got frames={int(latents.shape[0])}"
        )
    patches = None
    if latents.shape[0] > 0:
        # [T,D] flat prompt frames -> [B,T,D] patch groups expected by Aggregator.
        patches = latents.reshape(-1, patch_size, latent_dim)
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
