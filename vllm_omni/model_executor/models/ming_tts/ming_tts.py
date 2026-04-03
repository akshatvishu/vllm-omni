from __future__ import annotations

import hashlib
import os
from functools import cached_property
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models import SupportsPP
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
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
    KEY_SPEAKER_EMBEDDING,
    KEY_SIGMA,
    KEY_TEMPERATURE,
    KEY_TEXT_MODE,
    MingTTSConfig,
    TEXT_EOS_TOKEN_ID,
    VISION_START_TOKEN_ID,
)
from .ingress import encode_prompt_waveform_to_frame_latents
from .prompt_builder import coerce_speaker_embeddings

logger = init_logger(__name__)
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
        self._prompt_audio_encoder: AudioVAE | None = None
        self._prompt_audio_encoder_loaded = False
        self._logged_prompt_waveform_fallback = False

        if "model_stage" in os.environ:
            self.model_stage = os.environ["model_stage"]
        else:
            self.model_stage = vllm_config.model_config.model_stage

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
                    "Ming Stage-1 received no loadable checkpoint weights. "
                    "Expected prefixes: model.*, linear_proj_audio.*, flowloss.*, stop_head.*, spk_head.*"
                )
            self._load_prompt_audio_encoder_weights((k, v) for k, v in weights if k.startswith("audio."))
            loaded = self.model.load_weights(llm_weights)
            return {f"model.{name}" for name in loaded}

        audio_weights = [(k, v) for k, v in weights if k.startswith("audio.")]
        if not audio_weights:
            raise RuntimeError("Ming Stage-2 received no loadable checkpoint weights. Expected prefix: audio.*")
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
        if _should_log_stage0_state(
            decode_step=decode_step,
            stop_prob=stop_prob,
            threshold=self.ming_config.stop_head_threshold,
            chunk_size=self.ming_config.latent_chunk_size,
        ):
            logger.info(
                "MING_STAGE0_STATE %s",
                {
                    "request_id": req_id,
                    "decode_step_in": decode_step,
                    "decode_step_out": decode_step + 1,
                    "latent_patch_shape": tuple(latent_patch.shape),
                    "stop_prob": stop_prob,
                },
            )
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
        speaker_input_summaries = []
        speaker_projection_summaries = []
        if speaker_embeddings:
            speaker_slots = _find_speaker_placeholder_positions(input_ids, self.vllm_config.model_config.hf_config)
            if len(speaker_slots) < len(speaker_embeddings):
                raise RuntimeError(
                    f"Could not locate enough speaker placeholder slots: found {len(speaker_slots)}, "
                    f"need {len(speaker_embeddings)}"
                )
            for speaker_slot, spk in zip(speaker_slots, speaker_embeddings):
                speaker_input_summaries.append(_tensor_summary(spk))
                spk_proj = self.model.project_speaker_embedding(
                    spk.to(device=input_embeds.device, dtype=input_embeds.dtype).unsqueeze(0)
                ).squeeze(0)
                speaker_projection_summaries.append(_tensor_summary(spk_proj))
                input_embeds[speaker_slot] = spk_proj

        if prompt_latents is not None and prompt_latents["patches"] is not None:
            prompt_patches = prompt_latents["patches"].to(
                device=input_embeds.device,
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
        prompt_token_ids = input_ids.detach().cpu().tolist()
        prompt_positions = list(range(len(prompt_token_ids)))
        decode_position = prompt_positions[-1] if prompt_positions else None
        embed_weight = self.model.get_input_embeddings().weight
        selected_token_ids = _ordered_unique_token_ids(
            [
                prompt_token_ids[0],
                prompt_token_ids[-1],
                int(self.ming_config.audio_start_token_id),
                int(self.ming_config.audio_dummy_token_id),
                int(VISION_START_TOKEN_ID),
                int(TEXT_EOS_TOKEN_ID),
            ]
        )
        speaker_slot_summaries = []
        for speaker_slot in speaker_slots:
            speaker_slot_summaries.append(
                {
                    "index": int(speaker_slot),
                    "embed": _tensor_summary(input_embeds[speaker_slot]),
                }
            )
        logger.info(
            "MING_STAGE0_PREFILL_INPUT_PARITY %s",
            {
                "request_id": request_id,
                "prompt_token_ids": prompt_token_ids,
                "prompt_token_count": len(prompt_token_ids),
                "positions": prompt_positions,
                "decode_position": decode_position,
                "speaker_inputs": speaker_input_summaries,
                "speaker_projections": speaker_projection_summaries,
                "speaker_slots": speaker_slot_summaries,
                "spk_head_weight": _tensor_summary(self.model.spk_head.weight),
                "spk_head_bias": _tensor_summary(self.model.spk_head.bias),
                "embed_weight_prompt_rows": _embedding_weight_probe(
                    embed_weight,
                    _ordered_unique_token_ids(prompt_token_ids),
                ),
                "embed_weight_selected_rows": _embedding_weight_probe(embed_weight, selected_token_ids),
                "prompt_input_embeds": _tensor_summary(input_embeds),
                "decode_input_embeds": _tensor_summary(input_embeds[-1]),
            },
        )
        logger.info(
            "MING_STAGE0_REQUEST %s",
            {
                "request_id": request_id,
                "prompt_tokens": int(input_ids.shape[0]),
                "has_prompt_latents": prompt_latents is not None,
                "prompt_patch_shape": None if prompt_latents is None else _shape_or_none(prompt_latents["patches"]),
                "prompt_frame_shape": None if prompt_latents is None else _shape_or_none(prompt_latents["frames"]),
                "has_prompt_waveform": info_dict.get("prompt_waveform") is not None,
                "has_prompt_text": info_dict.get("prompt_text") is not None,
                "speaker_count": 0 if speaker_embeddings is None else len(speaker_embeddings),
                "has_instruction": info_dict.get("instruction") is not None,
                "cfg": _runtime_control_value(info_dict.get(KEY_CFG)),
                "sigma": _runtime_control_value(info_dict.get(KEY_SIGMA)),
                "temperature": _runtime_control_value(info_dict.get(KEY_TEMPERATURE)),
                "max_decode_steps": _runtime_control_value(info_dict.get(KEY_MAX_DECODE_STEPS)),
                "min_decode_steps": _runtime_control_value(info_dict.get(KEY_MIN_DECODE_STEPS)),
            },
        )
        _copy_runtime_controls(update, info_dict)
        return input_ids, input_embeds, update

    def _resolve_prompt_latents(self, info_dict: dict[str, Any]) -> dict[str, torch.Tensor] | None:
        direct_latents = _coerce_prompt_latents(
            info_dict.get(KEY_PROMPT_LATENTS, info_dict.get("prompt_latents")),
            patch_size=self.ming_config.patch_size,
            latent_dim=self.ming_config.latent_dim,
        )
        if direct_latents is not None:
            return direct_latents

        waveform = info_dict.get("prompt_waveform")
        if waveform is None:
            waveform = info_dict.get("prompt_waveforms")
        if waveform is None:
            return None
        if info_dict.get("prompt_text") is None:
            return None
        if not self._logged_prompt_waveform_fallback:
            logger.warning(
                "Ming Stage-0 fell back to runtime waveform->latents encoding. "
                "Ingress prompt finalization should have attached ming_prompt_latents already."
            )
            self._logged_prompt_waveform_fallback = True
        return self._encode_prompt_waveform_to_latents(
            waveform,
            info_dict.get("prompt_waveform_length"),
        )

    @torch.inference_mode()
    def encode_prompt_waveform_for_ingress(
        self,
        prompt_waveform: Any,
        prompt_waveform_length: Any = None,
    ) -> torch.Tensor:
        prompt_latents = self._encode_prompt_waveform_to_latents(prompt_waveform, prompt_waveform_length)
        if prompt_latents is None:
            raise RuntimeError("Ming prompt waveform encoder is unavailable")
        return prompt_latents["frames"].detach().to("cpu", dtype=torch.float32).contiguous()

    def _encode_prompt_waveform_to_latents(
        self,
        waveform: Any,
        waveform_length: Any = None,
    ) -> dict[str, torch.Tensor] | None:
        encoder = self._ensure_prompt_audio_encoder()
        if encoder is None:
            return None

        latent = encode_prompt_waveform_to_frame_latents(
            encoder,
            waveform,
            waveform_length,
            patch_size=self.ming_config.patch_size,
            latent_dim=self.ming_config.latent_dim,
            sample_rate=self.ming_config.sample_rate,
            frame_hop=self.ming_config.audio_frame_hop,
        )
        return _coerce_prompt_latents(
            latent,
            patch_size=self.ming_config.patch_size,
            latent_dim=self.ming_config.latent_dim,
        )

    def _ensure_prompt_audio_encoder(self) -> AudioVAE | None:
        if self._prompt_audio_encoder is not None:
            return self._prompt_audio_encoder
        if self.ming_config.audio_tokenizer_config is None:
            return None
        self._prompt_audio_encoder = AudioVAE(self.ming_config.audio_tokenizer_config)
        return self._prompt_audio_encoder

    def _load_prompt_audio_encoder_weights(self, weights) -> None:
        encoder = self._ensure_prompt_audio_encoder()
        if encoder is None or self._prompt_audio_encoder_loaded:
            return

        state_dict = encoder.state_dict()
        loaded = 0
        with torch.no_grad():
            for ckpt_name, loaded_weight in weights:
                name = ckpt_name[len("audio.") :] if ckpt_name.startswith("audio.") else ckpt_name
                if name not in state_dict:
                    continue
                target = state_dict[name]
                if isinstance(target, torch.Tensor):
                    weight_loader = getattr(target, "weight_loader", default_weight_loader)
                    if weight_loader == default_weight_loader:
                        target.copy_(loaded_weight.to(device=target.device, dtype=target.dtype))
                    else:
                        weight_loader(target, loaded_weight)
                    loaded += 1
        if loaded == 0:
            logger.warning("Ming prompt audio encoder received no audio checkpoint weights; prompt_waveform cloning will be invalid.")
            return
        self._prompt_audio_encoder_loaded = True

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


def _runtime_control_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.reshape(-1)[0].item()
        return tuple(value.shape)
    return value


def _shape_or_none(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    return None


def _should_log_stage0_state(decode_step: int, stop_prob: float | None, threshold: float, chunk_size: int) -> bool:
    if decode_step == 0:
        return True
    if stop_prob is not None and stop_prob > threshold:
        return True
    return chunk_size > 0 and (decode_step + 1) % chunk_size == 0


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any] | None:
    if not isinstance(tensor, torch.Tensor):
        return None
    flat = tensor.detach().to(dtype=torch.float32, device="cpu").reshape(-1).contiguous()
    count = min(8, int(flat.numel()))
    return {
        "shape": tuple(tensor.shape),
        "mean": float(flat.mean().item()) if flat.numel() else 0.0,
        "std": float(flat.std(unbiased=False).item()) if flat.numel() else 0.0,
        "min": float(flat.min().item()) if flat.numel() else 0.0,
        "max": float(flat.max().item()) if flat.numel() else 0.0,
        "first8": flat[:count].tolist(),
        "sha256_fp32": hashlib.sha256(flat.numpy().tobytes()).hexdigest(),
    }


def _ordered_unique_token_ids(token_ids: list[Any]) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for token_id in token_ids:
        if token_id is None:
            continue
        value = int(token_id)
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _embedding_weight_probe(weight: torch.Tensor, token_ids: list[int]) -> dict[str, Any]:
    if not token_ids:
        return {"token_ids": [], "rows": None}
    row_index = torch.tensor(token_ids, dtype=torch.long, device=weight.device)
    rows = weight.index_select(0, row_index)
    return {
        "token_ids": token_ids,
        "rows": _tensor_summary(rows),
    }


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
