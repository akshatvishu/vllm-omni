from __future__ import annotations

import hashlib
import inspect
import os
import warnings
from typing import Any
from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from vllm.model_executor.models.utils import init_vllm_registered_model, is_pp_missing_parameter, maybe_prefix
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .config_ming_tts import (
    KEY_CFG,
    KEY_DECODE_STEP,
    KEY_LATENT_HISTORY,
    KEY_MAX_DECODE_STEPS,
    KEY_MIN_DECODE_STEPS,
    KEY_NEXT_EMBEDS,
    KEY_PROMPT_LATENTS,
    KEY_REQUEST_ID,
    KEY_SPEAKER_EMBEDDING,
    KEY_SIGMA,
    KEY_TEMPERATURE,
    KEY_TEXT_MODE,
    MingTTSConfig,
)
from .fm.dit import Aggregator
from .fm.flowloss import FlowLoss

logger = init_logger(__name__)

MING_STOP_REASON_CONTINUE = "continue"
MING_STOP_REASON_STOP_HEAD = "stop_head"
MING_STOP_REASON_MAX_DECODE_STEPS = "max_decode_steps"
MING_STOP_REASON_KEY = "ming_stop_reason"


class MingLLMModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        hf_config = vllm_config.model_config.hf_config
        self.ming_config = MingTTSConfig.from_hf_config(hf_config)
        self.ming_config.validate()

        self.vllm_config = vllm_config
        self.prefix = prefix
        self.quant_config = vllm_config.quant_config
        self.fm_dtype = _resolve_ming_runtime_dtype(vllm_config)

        self.model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            architectures=["Qwen2ForCausalLM"],
        )

        self.linear_proj_audio = Aggregator(
            in_channels=self.ming_config.latent_dim,
            llm_input_dim=self.ming_config.llm_hidden_size,
            **self.ming_config.aggregator_config,
        )
        self.flowloss = FlowLoss(
            z_channels=self.ming_config.latent_dim,
            llm_cond_dim=self.ming_config.llm_hidden_size,
            **self.ming_config.ditar_config,
        )

        self.stop_head = nn.Linear(self.ming_config.llm_hidden_size, 2, bias=True)
        self.spk_head = nn.Linear(192, self.ming_config.llm_hidden_size, bias=True)
        self.flowloss.to(dtype=self.fm_dtype)
        self.linear_proj_audio.to(dtype=self.fm_dtype)
        self.stop_head.to(dtype=self.fm_dtype)
        self.spk_head.to(dtype=self.fm_dtype)
        self._pending_postprocess_updates: dict[str, dict[str, Any]] = {}
        self._last_sample_decode_steps: torch.Tensor | None = None
        self._last_sample_stop_probs: torch.Tensor | None = None
        self._last_sample_max_decode_steps: torch.Tensor | None = None
        self._last_sample_min_decode_steps: torch.Tensor | None = None
        self._last_text_mode: bool = False

    def get_input_embeddings(self) -> nn.Module:
        if hasattr(self.model, "embed_tokens"):
            return self.model.embed_tokens
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            return self.model.model.embed_tokens
        raise AttributeError("Could not locate token embeddings on Ming Qwen2 backbone.")

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            return inputs_embeds
        if hasattr(self.model, "embed_input_ids"):
            return self.model.embed_input_ids(input_ids)
        return self.get_input_embeddings()(input_ids)

    def project_speaker_embedding(self, spk_emb: torch.Tensor) -> torch.Tensor:
        return self.spk_head(spk_emb)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        latent_history: torch.Tensor | None = None,
        model_intermediate_buffer: list[dict[str, Any]] | None = None,
        seq_token_counts: list[int] | None = None,
        **kwargs: object,
    ) -> OmniOutput | IntermediateTensors | torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)

        if model_intermediate_buffer is None:
            model_intermediate_buffer = kwargs.get("runtime_additional_information")
        logits_index = kwargs.get("logits_index")
        request_infos = _normalize_request_infos(model_intermediate_buffer)
        should_log_backbone_call = bool(request_infos) and any(info.get(KEY_LATENT_HISTORY) is not None for info in request_infos)
        if should_log_backbone_call:
            qwen2_model = getattr(self.model, "model", None)
            qwen2_config = getattr(qwen2_model, "config", getattr(self.model, "config", None))
            first_layer = None if qwen2_model is None else next(iter(getattr(qwen2_model, "layers", []) or []), None)
            first_attn = getattr(first_layer, "self_attn", None)
            logger.info(
                "MING_STAGE0_BACKBONE_CALL %s",
                {
                    "request_id": request_infos[0].get(KEY_REQUEST_ID),
                    "model_class": type(self.model).__name__,
                    "model_source_file": inspect.getsourcefile(type(self.model)),
                    "inputs_embeds_shape": tuple(inputs_embeds.shape),
                    "inputs_embeds_sha256_fp32": _tensor_sha(inputs_embeds, dtype=torch.float32),
                    "positions_shape": tuple(positions.shape),
                    "positions_sha256_int64": _tensor_sha(positions, dtype=torch.int64),
                    "input_ids_shape": None if input_ids is None else tuple(input_ids.shape),
                    "intermediate_tensors_is_none": intermediate_tensors is None,
                    "logits_index": _rpc_safe_value(logits_index),
                    "seq_token_counts": None if seq_token_counts is None else [int(x) for x in seq_token_counts],
                    "forward_context_ubatch_slices": _serialize_ubatch_slices(),
                    "use_sliding_window": getattr(qwen2_config, "use_sliding_window", None),
                    "sliding_window": getattr(qwen2_config, "sliding_window", None),
                    "max_window_layers": getattr(qwen2_config, "max_window_layers", None),
                    "rope_theta": getattr(qwen2_config, "rope_theta", None),
                    "num_hidden_layers": getattr(qwen2_config, "num_hidden_layers", None),
                    "hidden_size": getattr(qwen2_config, "hidden_size", None),
                    "num_attention_heads": getattr(qwen2_config, "num_attention_heads", None),
                    "attn_module_class": None if first_attn is None else type(first_attn).__name__,
                    "rotary_module_class": None if first_attn is None else type(getattr(first_attn, "rotary_emb", None)).__name__,
                },
            )
        backbone_out = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        if isinstance(backbone_out, IntermediateTensors):
            return backbone_out

        hidden_states = _extract_hidden_states(backbone_out)
        token_counts = _get_request_token_counts(hidden_states, request_infos, seq_token_counts)
        text_mode = bool(request_infos) and all(bool(info.get(KEY_TEXT_MODE, False)) for info in request_infos)
        if request_infos and any(bool(info.get(KEY_TEXT_MODE, False)) for info in request_infos) and not text_mode:
            raise RuntimeError("Mixed Ming text/audio modes in one Stage-0 batch are unsupported.")

        if text_mode:
            self._last_text_mode = True
            self._last_sample_decode_steps = None
            self._last_sample_stop_probs = None
            self._last_sample_max_decode_steps = None
            self._last_sample_min_decode_steps = None
            return OmniOutput(
                text_hidden_states=hidden_states,
                multimodal_outputs={KEY_TEXT_MODE: True},
                intermediate_tensors=intermediate_tensors,
            )
        self._last_text_mode = False

        if latent_history is None and not token_counts:
            return OmniOutput(
                text_hidden_states=hidden_states,
                multimodal_outputs=None,
                intermediate_tensors=intermediate_tensors,
            )

        if latent_history is not None and not token_counts:
            token_counts = [hidden_states.shape[0]]
            request_infos = [{KEY_LATENT_HISTORY: latent_history}]

        total_tokens = hidden_states.shape[0]
        latent_patch_tokens = None
        next_embed_tokens = None
        new_history_tokens = None
        stop_prob_tokens = None
        decode_step_tokens = None
        has_patch = None
        max_decode_step_tokens = None
        pending_updates: dict[str, dict[str, Any]] = {}
        stop_reason_tokens: list[str] | None = None

        cursor = 0
        any_decode = False
        for req_idx, token_count in enumerate(token_counts):
            end = min(cursor + token_count, total_tokens)
            if end <= cursor:
                continue

            req_info = request_infos[req_idx] if req_idx < len(request_infos) else {}
            req_history = req_info.get(KEY_LATENT_HISTORY)
            if req_history is None:
                cursor = end
                continue
            decode_step = int(req_info.get(KEY_DECODE_STEP, req_info.get("generated_len", 0)))

            req_history = _coerce_latent_history(
                req_history,
                device=hidden_states.device,
                dtype=self.fm_dtype,
                cfg=self.ming_config,
            )
            if req_history is None:
                cursor = end
                continue

            if token_count == 1:
                decode_hidden = hidden_states[cursor:end]
                output_index = cursor
            else:
                # [T,H] prefill span -> use the last prompt token [1,H] to seed
                # the first FlowLoss patch, matching upstream Ming.
                decode_hidden = hidden_states[end - 1 : end]
                output_index = end - 1
            req_cfg = _resolve_runtime_float(req_info, KEY_CFG, self.ming_config.cfg)
            req_sigma = _resolve_runtime_float(req_info, KEY_SIGMA, self.ming_config.sigma)
            req_temperature = _resolve_runtime_float(req_info, KEY_TEMPERATURE, self.ming_config.temperature)
            req_max_decode_steps = _resolve_runtime_int(req_info, KEY_MAX_DECODE_STEPS, self.ming_config.max_decode_steps)
            req_min_decode_steps = _resolve_optional_runtime_int(req_info, KEY_MIN_DECODE_STEPS, 0)
            req_id = req_info.get(KEY_REQUEST_ID)
            if decode_step == 0:
                logger.info(
                    "MING_STAGE0_BOOTSTRAP %s",
                    {
                        "request_id": req_id,
                        "source": "prefill" if token_count != 1 else "decode",
                        "token_count": token_count,
                        "cfg": req_cfg,
                        "sigma": req_sigma,
                        "temperature": req_temperature,
                        "max_decode_steps": req_max_decode_steps,
                        "min_decode_steps": req_min_decode_steps,
                        "has_prompt_latents": req_info.get(KEY_PROMPT_LATENTS) is not None,
                        "has_speaker_embedding": req_info.get(KEY_SPEAKER_EMBEDDING) is not None,
                    },
                )
            sampled_token_latent, next_embeds, new_history, stop_probs = self._decode_one_step(
                hidden_states=decode_hidden,
                latent_history=req_history,
                cfg_scale=req_cfg,
                sigma=req_sigma,
                temperature=req_temperature,
            )
            if decode_step == 0:
                full_hidden = hidden_states[cursor:end]
                position_slice = positions.reshape(-1)[cursor:end].detach().to(dtype=torch.int64).cpu().tolist()
                decode_position = int(positions.reshape(-1)[output_index].item())
                logger.info(
                    "MING_STAGE0_STEP0_PARITY %s",
                    {
                        "request_id": req_id,
                        "source": "prefill" if token_count != 1 else "decode",
                        "token_count": token_count,
                        "positions": position_slice,
                        "decode_position": decode_position,
                        "backbone_hidden_full": _tensor_summary(full_hidden),
                        "backbone_hidden": _tensor_summary(decode_hidden),
                        "flow_cond": _tensor_summary(decode_hidden.unsqueeze(1)),
                        "latent_history": _tensor_summary(req_history),
                        "flow_output": _tensor_summary(sampled_token_latent),
                        "aggregator_output": _tensor_summary(next_embeds),
                        "stop_prob": float(stop_probs.reshape(-1)[0].item()),
                    },
                )

            if latent_patch_tokens is None:
                latent_patch_tokens = sampled_token_latent.new_zeros(
                    (total_tokens, self.ming_config.patch_size, self.ming_config.latent_dim)
                )
                next_embed_tokens = next_embeds.new_zeros((total_tokens, 1, self.ming_config.llm_hidden_size))
                new_history_tokens = new_history.new_zeros(
                    (total_tokens, self.ming_config.history_patch_size, self.ming_config.latent_dim)
                )
                stop_prob_tokens = stop_probs.new_zeros((total_tokens,))
                decode_step_tokens = torch.zeros((total_tokens,), dtype=torch.int32, device=hidden_states.device)
                max_decode_step_tokens = torch.zeros((total_tokens,), dtype=torch.int32, device=hidden_states.device)
                min_decode_step_tokens = torch.zeros((total_tokens,), dtype=torch.int32, device=hidden_states.device)
                has_patch = torch.zeros((total_tokens,), dtype=torch.bool, device=hidden_states.device)
                stop_reason_tokens = [MING_STOP_REASON_CONTINUE] * total_tokens

            latent_patch_tokens[output_index : output_index + 1] = sampled_token_latent
            next_embed_tokens[output_index : output_index + 1] = next_embeds
            new_history_tokens[output_index : output_index + 1] = new_history
            stop_prob_tokens[output_index : output_index + 1] = stop_probs
            decode_step_tokens[output_index : output_index + 1] = decode_step
            max_decode_step_tokens[output_index : output_index + 1] = req_max_decode_steps
            min_decode_step_tokens[output_index : output_index + 1] = req_min_decode_steps
            has_patch[output_index : output_index + 1] = True
            stop_reason, _, _, _, _ = _resolve_ming_stop_decision(
                step=decode_step,
                stop_prob=float(stop_probs.reshape(-1)[0].item()),
                stop_threshold=float(self.ming_config.stop_head_threshold),
                min_stop_step=int(self.ming_config.stop_head_min_steps),
                min_decode_steps=req_min_decode_steps,
                max_decode_steps=req_max_decode_steps,
                audio_dummy_token_id=int(self.ming_config.audio_dummy_token_id),
                text_eos_token_id=int(self.ming_config.text_eos_token_id),
            )
            if stop_reason_tokens is not None:
                stop_reason_tokens[output_index] = stop_reason
            if isinstance(req_id, str):
                pending_updates[req_id] = {
                    KEY_LATENT_HISTORY: new_history,
                    KEY_NEXT_EMBEDS: next_embeds,
                    "ming_latent_patch": sampled_token_latent,
                    "ming_stop_prob": stop_probs,
                    MING_STOP_REASON_KEY: stop_reason,
                }
            any_decode = True
            cursor = end

        self._pending_postprocess_updates = pending_updates
        if not any_decode:
            self._last_sample_decode_steps = None
            self._last_sample_stop_probs = None
            self._last_sample_max_decode_steps = None
            self._last_sample_min_decode_steps = None
            return OmniOutput(
                text_hidden_states=hidden_states,
                multimodal_outputs=None,
                intermediate_tensors=intermediate_tensors,
            )

        if isinstance(logits_index, torch.Tensor):
            self._last_sample_decode_steps = decode_step_tokens[logits_index]
            self._last_sample_stop_probs = stop_prob_tokens[logits_index]
            self._last_sample_max_decode_steps = max_decode_step_tokens[logits_index]
            self._last_sample_min_decode_steps = min_decode_step_tokens[logits_index]
        else:
            self._last_sample_decode_steps = None
            self._last_sample_stop_probs = None
            self._last_sample_max_decode_steps = None
            self._last_sample_min_decode_steps = None

        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs={
                "ming_latent_patch": latent_patch_tokens,
                "ming_next_embeds": next_embed_tokens,
                "ming_new_history": new_history_tokens,
                "ming_stop_prob": stop_prob_tokens,
                "ming_decode_step": decode_step_tokens,
                "ming_max_decode_steps": max_decode_step_tokens,
                "ming_min_decode_steps": min_decode_step_tokens,
                "ming_has_patch": has_patch,
                MING_STOP_REASON_KEY: tuple(stop_reason_tokens or []),
            },
            intermediate_tensors=intermediate_tensors,
        )

    def pop_postprocess_update(self, req_id: str) -> dict[str, Any]:
        if not isinstance(req_id, str):
            return {}
        return self._pending_postprocess_updates.pop(req_id, {})

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor | None:
        decode_steps = None
        stop_probs_tensor = None
        hidden_states_type = type(hidden_states).__name__
        text_mode = self._last_text_mode
        if isinstance(hidden_states, OmniOutput):
            text_mode = bool((hidden_states.multimodal_outputs or {}).get(KEY_TEXT_MODE, text_mode))
            decode_steps = (hidden_states.multimodal_outputs or {}).get("ming_decode_step")
            stop_probs_tensor = (hidden_states.multimodal_outputs or {}).get("ming_stop_prob")
            max_decode_steps_tensor = (hidden_states.multimodal_outputs or {}).get("ming_max_decode_steps")
            min_decode_steps_tensor = (hidden_states.multimodal_outputs or {}).get("ming_min_decode_steps")
            hidden_states = hidden_states.text_hidden_states
        else:
            max_decode_steps_tensor = None
            min_decode_steps_tensor = None
        if text_mode:
            if hidden_states is None or hidden_states.numel() == 0:
                return None
            return self.model.compute_logits(hidden_states)
        if max_decode_steps_tensor is None and isinstance(self._last_sample_max_decode_steps, torch.Tensor):
            if self._last_sample_max_decode_steps.numel() > 0:
                max_decode_steps_tensor = self._last_sample_max_decode_steps
        if min_decode_steps_tensor is None and isinstance(self._last_sample_min_decode_steps, torch.Tensor):
            if self._last_sample_min_decode_steps.numel() > 0:
                min_decode_steps_tensor = self._last_sample_min_decode_steps
        if decode_steps is None and isinstance(self._last_sample_decode_steps, torch.Tensor):
            if self._last_sample_decode_steps.numel() > 0:
                decode_steps = self._last_sample_decode_steps
        if stop_probs_tensor is None and isinstance(self._last_sample_stop_probs, torch.Tensor):
            if self._last_sample_stop_probs.numel() > 0:
                stop_probs_tensor = self._last_sample_stop_probs

        if hidden_states is None or hidden_states.numel() == 0:
            return None
        if hidden_states.dim() != 2:
            raise RuntimeError(
                f"Expected hidden_states rank-2 [B,H] in compute_logits, got {tuple(hidden_states.shape)}"
            )

        batch_size = hidden_states.shape[0]
        stop_prob_values = _resolve_stop_probs_batch(stop_probs_tensor, batch_size=batch_size)
        stop_source = "forward_cache"
        if stop_prob_values is None:
            stop_hidden = hidden_states.to(dtype=self.fm_dtype)
            stop_probs = self.stop_head(stop_hidden).softmax(dim=-1)[:, 1]
            if not torch.isfinite(stop_probs).all():
                raise RuntimeError("Non-finite stop_probs in Ming compute_logits.")
            stop_prob_values = [float(stop_probs[i].item()) for i in range(batch_size)]
            stop_source = "compute_logits_recompute"
        steps = self._get_decode_steps(decode_steps, sampling_metadata, batch_size)
        max_decode_steps = _resolve_max_decode_steps_batch(
            max_decode_steps_tensor,
            batch_size=batch_size,
            default_value=self.ming_config.max_decode_steps,
        )
        min_decode_steps = _resolve_min_decode_steps_batch(
            min_decode_steps_tensor,
            batch_size=batch_size,
        )
        min_stop_step = int(self.ming_config.stop_head_min_steps)

        logits = torch.full(
            (batch_size, self.ming_config.llm_vocab_size),
            float("-inf"),
            device=hidden_states.device,
            dtype=torch.float32,
        )

        for i in range(batch_size):
            step = steps[i]
            stop_prob = stop_prob_values[i]
            stop_reason, stop_now, should_force_stop, min_required_decode_steps, next_token_id = (
                _resolve_ming_stop_decision(
                    step=step,
                    stop_prob=stop_prob,
                    stop_threshold=float(self.ming_config.stop_head_threshold),
                    min_stop_step=min_stop_step,
                    min_decode_steps=min_decode_steps[i],
                    max_decode_steps=max_decode_steps[i],
                    audio_dummy_token_id=int(self.ming_config.audio_dummy_token_id),
                    text_eos_token_id=int(self.ming_config.text_eos_token_id),
                )
            )
            logits[i, next_token_id] = 0.0
            if i == 0 and _should_log_stop_event(step, stop_now, should_force_stop, self.ming_config.latent_chunk_size):
                logger.info(
                    "MING_STAGE0_STOP %s",
                    {
                        "hidden_states_type": hidden_states_type,
                        "step": step,
                        "stop_prob": stop_prob,
                        "threshold": float(self.ming_config.stop_head_threshold),
                        "min_stop_step": min_stop_step,
                        "min_required_decode_steps": min_required_decode_steps,
                        "max_decode_steps": max_decode_steps[i],
                        "stop_reason": stop_reason,
                        "force_stop": should_force_stop,
                        "stop_now": stop_now,
                        "stop_source": stop_source,
                        "next_token_id": next_token_id,
                    },
                )
        return logits

    def sample(self, logits, sampling_metadata):
        if logits is None:
            return None
        if self._last_text_mode:
            return self.model.sample(logits, sampling_metadata)

        del sampling_metadata
        sampled = logits.argmax(dim=-1, keepdim=True)
        return SamplerOutput(
            sampled_token_ids=sampled.to(dtype=torch.int32),
            logprobs_tensors=None,
        )

    def _decode_one_step(
        self,
        *,
        hidden_states: torch.Tensor,
        latent_history: torch.Tensor,
        cfg_scale: float,
        sigma: float,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.dim() != 2:
            raise RuntimeError(f"Expected decode hidden_states rank-2 [B,H], got {tuple(hidden_states.shape)}")
        if latent_history.dim() != 3:
            raise RuntimeError(f"Expected latent_history rank-3 [B,T,D], got {tuple(latent_history.shape)}")
        if hidden_states.shape[0] != latent_history.shape[0]:
            raise RuntimeError(
                f"Batch mismatch: hidden_states B={hidden_states.shape[0]} vs "
                f"latent_history B={latent_history.shape[0]}"
            )

        # [B,H] -> [B,1,H] for FlowLoss conditioning.
        z_diff_cond = hidden_states.to(dtype=self.fm_dtype).unsqueeze(1)
        if not torch.isfinite(z_diff_cond).all():
            raise RuntimeError("Non-finite z_diff_cond before FlowLoss.sample().")
        flow_out = self.flowloss.sample(
            z=z_diff_cond,
            latent_history=latent_history,
            cfg=cfg_scale,
            patch_size=self.ming_config.patch_size,
            sigma=sigma,
            temperature=temperature,
        )
        sampled_token_latent = flow_out[0] if isinstance(flow_out, tuple) else flow_out

        expected_shape = (
            hidden_states.shape[0],
            self.ming_config.patch_size,
            self.ming_config.latent_dim,
        )
        if tuple(sampled_token_latent.shape) != expected_shape:
            raise RuntimeError(
                f"FlowLoss output shape mismatch: got {tuple(sampled_token_latent.shape)}, expected {expected_shape}"
            )

        # [B,32,64] -> shift left by one patch and append [B,4,64] => [B,32,64].
        new_history = torch.cat(
            [latent_history[:, self.ming_config.patch_size :, :], sampled_token_latent],
            dim=1,
        )
        # Aggregator expects [B,T,D] = [B,4,64] and returns [B,1,H].
        next_embeds = self.linear_proj_audio(sampled_token_latent)
        stop_hidden = hidden_states.to(dtype=self.fm_dtype)
        stop_probs = self.stop_head(stop_hidden).softmax(dim=-1)[:, 1]
        if not torch.isfinite(sampled_token_latent).all():
            raise RuntimeError("Non-finite sampled_token_latent in Ming decode step.")
        if not torch.isfinite(next_embeds).all():
            raise RuntimeError("Non-finite next_embeds in Ming decode step.")
        if not torch.isfinite(stop_probs).all():
            raise RuntimeError("Non-finite stop_probs in Ming decode step.")
        return sampled_token_latent, next_embeds, new_history, stop_probs

    def _get_decode_steps(
        self,
        decode_steps: torch.Tensor | None,
        sampling_metadata: SamplingMetadata,
        batch_size: int,
    ) -> list[int]:
        if isinstance(decode_steps, torch.Tensor) and decode_steps.numel() > 0:
            flat_steps = decode_steps.reshape(-1)
            return [int(flat_steps[min(i, flat_steps.numel() - 1)].item()) for i in range(batch_size)]

        steps: list[int] = []
        output_token_ids = getattr(sampling_metadata, "output_token_ids", None)
        if isinstance(output_token_ids, list):
            for token_ids in output_token_ids[:batch_size]:
                if isinstance(token_ids, torch.Tensor):
                    steps.append(int(token_ids.numel()))
                elif isinstance(token_ids, (list, tuple)):
                    steps.append(len(token_ids))
                else:
                    raise RuntimeError(
                        f"Expected output_token_ids entries to be list/tuple/Tensor, got {type(token_ids)!r}"
                    )

        while len(steps) < batch_size:
            steps.append(0)
        return steps[:batch_size]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        skipped: list[str] = []

        for ckpt_name, loaded_weight in weights:
            name = ckpt_name

            if self.quant_config is not None and (scale_name := self.quant_config.get_cache_scale(name)):
                if scale_name not in params_dict:
                    skipped.append(ckpt_name)
                    continue
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            mapped_name = None
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                    mapped_name = None
                    break
                if is_pp_missing_parameter(mapped_name, self):
                    mapped_name = None
                    break
                if mapped_name not in params_dict:
                    mapped_name = None
                    continue
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped_name)
                break

            if mapped_name in loaded_params:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue

            name = maybe_remap_kv_scale_name(name, params_dict)
            if name is None:
                continue
            if is_pp_missing_parameter(name, self):
                continue
            if name not in params_dict:
                skipped.append(ckpt_name)
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        _warn_missing_prefix("flowloss", params_dict, loaded_params, prefix="flowloss.", fatal=True)
        _warn_missing_prefix("linear_proj_audio", params_dict, loaded_params, prefix="linear_proj_audio.", fatal=True)
        _warn_missing_prefix("stop_head", params_dict, loaded_params, prefix="stop_head.", fatal=True)
        _warn_missing_prefix("spk_head", params_dict, loaded_params, prefix="spk_head.", fatal=True)

        if skipped:
            warnings.warn(
                f"MingLLMModel: skipped {len(skipped)} checkpoint keys during load. First few: {skipped[:8]}",
                stacklevel=2,
            )

        return loaded_params


def _extract_hidden_states(backbone_out: object) -> torch.Tensor:
    if isinstance(backbone_out, torch.Tensor):
        return backbone_out
    if hasattr(backbone_out, "last_hidden_state"):
        return backbone_out.last_hidden_state
    if isinstance(backbone_out, (tuple, list)) and len(backbone_out) > 0:
        if isinstance(backbone_out[0], torch.Tensor):
            return backbone_out[0]
    raise TypeError(f"Unsupported backbone forward output type: {type(backbone_out)}")


def _resolve_ming_runtime_dtype(vllm_config: VllmConfig) -> torch.dtype:
    dtype = getattr(vllm_config.model_config, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        normalized = dtype.strip().lower()
        if normalized in ("float16", "half", "torch.float16"):
            return torch.float16
        if normalized in ("bfloat16", "bf16", "torch.bfloat16"):
            return torch.bfloat16
        if normalized in ("float32", "fp32", "torch.float32"):
            return torch.float32
    return torch.float32


def _warn_missing_prefix(
    module_name: str,
    params_dict: dict[str, nn.Parameter],
    loaded_params: set[str],
    prefix: str,
    fatal: bool = False,
) -> None:
    expected = {key for key in params_dict if key.startswith(prefix)}
    missing = expected - loaded_params
    if not missing:
        return
    msg = (
        f"MingLLMModel: {len(missing)} {module_name} params not loaded "
        f"(prefix={prefix}). First few: {sorted(missing)[:5]}"
    )
    if fatal:
        raise RuntimeError(msg)
    warnings.warn(msg, stacklevel=3)


def _normalize_request_infos(model_intermediate_buffer: object) -> list[dict[str, Any]]:
    if not isinstance(model_intermediate_buffer, list):
        return []
    infos: list[dict[str, Any]] = []
    for item in model_intermediate_buffer:
        infos.append(item if isinstance(item, dict) else {})
    return infos


def _get_request_token_counts(
    hidden_states: torch.Tensor,
    request_infos: list[dict[str, Any]],
    seq_token_counts: list[int] | None,
) -> list[int]:
    if seq_token_counts:
        return [int(x) for x in seq_token_counts]

    if is_forward_context_available():
        slices = getattr(get_forward_context(), "ubatch_slices", None)
        if slices is not None and len(slices) > 0:
            counts: list[int] = []
            for item in slices:
                if isinstance(item, int):
                    counts.append(int(item))
                elif hasattr(item, "stop") and hasattr(item, "start"):
                    counts.append(int(item.stop) - int(item.start))
            if counts:
                return counts

    if request_infos:
        if len(request_infos) == hidden_states.shape[0]:
            return [1] * hidden_states.shape[0]
        return [hidden_states.shape[0]]

    return []


def _serialize_ubatch_slices() -> list[int | dict[str, int | str]] | None:
    if not is_forward_context_available():
        return None
    slices = getattr(get_forward_context(), "ubatch_slices", None)
    if slices is None:
        return None
    serialized: list[int | dict[str, int | str]] = []
    for item in slices:
        if isinstance(item, int):
            serialized.append(int(item))
        elif hasattr(item, "start") and hasattr(item, "stop"):
            serialized.append({"start": int(item.start), "stop": int(item.stop)})
        else:
            serialized.append({"repr": repr(item)})
    return serialized


def _rpc_safe_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to("cpu")
        if tensor.ndim == 0:
            return tensor.item()
        return tensor.tolist()
    return value


def _coerce_latent_history(
    value: object,
    *,
    device: torch.device,
    dtype: torch.dtype,
    cfg: MingTTSConfig,
) -> torch.Tensor | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)

    history = value.detach()
    if history.ndim == 2:
        history = history.unsqueeze(0)
    if history.ndim != 3:
        raise RuntimeError(f"Expected latent_history rank-3 [B,T,D], got {tuple(history.shape)}")
    if history.shape[1] != cfg.history_patch_size or history.shape[2] != cfg.latent_dim:
        raise RuntimeError(
            f"latent_history shape mismatch: got {tuple(history.shape)}, "
            f"expected [B,{cfg.history_patch_size},{cfg.latent_dim}]"
        )
    return history.to(device=device, dtype=dtype)


def _resolve_runtime_float(req_info: dict[str, Any], key: str, default_value: float) -> float:
    raw = req_info.get(key, default_value)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid {key}: expected float-like value, got {raw!r}") from exc
    if not value >= 0.0:
        raise RuntimeError(f"Invalid {key}: expected non-negative value, got {value}")
    return value


def _resolve_runtime_int(req_info: dict[str, Any], key: str, default_value: int) -> int:
    raw = req_info.get(key, default_value)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid {key}: expected int-like value, got {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"Invalid {key}: expected positive value, got {value}")
    return value


def _resolve_optional_runtime_int(req_info: dict[str, Any], key: str, default_value: int) -> int:
    raw = req_info.get(key, default_value)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid {key}: expected int-like value, got {raw!r}") from exc
    if value < 0:
        raise RuntimeError(f"Invalid {key}: expected non-negative value, got {value}")
    return value


def _resolve_max_decode_steps_batch(
    value: torch.Tensor | None,
    *,
    batch_size: int,
    default_value: int,
) -> list[int]:
    if value is None:
        return [int(default_value)] * batch_size
    flat = value.reshape(-1).tolist()
    if not flat:
        return [int(default_value)] * batch_size
    resolved = [int(item) for item in flat]
    for item in resolved:
        if item <= 0:
            raise RuntimeError(f"Invalid ming_max_decode_steps in runtime batch: got {item}")
    if len(resolved) < batch_size:
        resolved.extend([resolved[-1]] * (batch_size - len(resolved)))
    return resolved[:batch_size]


def _resolve_min_decode_steps_batch(
    value: torch.Tensor | None,
    *,
    batch_size: int,
) -> list[int]:
    if value is None:
        return [0] * batch_size
    flat = value.reshape(-1).tolist()
    if not flat:
        return [0] * batch_size
    resolved = [max(0, int(item)) for item in flat]
    if len(resolved) < batch_size:
        resolved.extend([resolved[-1]] * (batch_size - len(resolved)))
    return resolved[:batch_size]


def _resolve_ming_stop_decision(
    *,
    step: int,
    stop_prob: float,
    stop_threshold: float,
    min_stop_step: int,
    min_decode_steps: int,
    max_decode_steps: int,
    audio_dummy_token_id: int,
    text_eos_token_id: int,
) -> tuple[str, bool, bool, int, int]:
    min_required_decode_steps = max(min_stop_step + 1, min_decode_steps)
    if max_decode_steps < min_required_decode_steps:
        raise RuntimeError(
            "Invalid Ming decode window: "
            f"max_decode_steps={max_decode_steps} is smaller than "
            f"min_required_decode_steps={min_required_decode_steps}"
        )
    should_force_stop = (step + 1) >= max_decode_steps
    should_stop_head = ((step + 1) >= min_required_decode_steps) and stop_prob > stop_threshold

    if should_force_stop:
        return (
            MING_STOP_REASON_MAX_DECODE_STEPS,
            True,
            True,
            min_required_decode_steps,
            text_eos_token_id,
        )
    if should_stop_head:
        return (
            MING_STOP_REASON_STOP_HEAD,
            True,
            False,
            min_required_decode_steps,
            text_eos_token_id,
        )
    return (
        MING_STOP_REASON_CONTINUE,
        False,
        False,
        min_required_decode_steps,
        audio_dummy_token_id,
    )


def _should_log_stop_event(step: int, stop_now: bool, force_stop: bool, chunk_size: int) -> bool:
    if stop_now or force_stop or step == 0:
        return True
    return chunk_size > 0 and (step + 1) % chunk_size == 0


def _resolve_stop_probs_batch(
    value: torch.Tensor | None,
    *,
    batch_size: int,
) -> list[float] | None:
    if value is None:
        return None
    flat = value.reshape(-1)
    if flat.numel() == 0:
        return None
    return [float(flat[min(i, flat.numel() - 1)].item()) for i in range(batch_size)]


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any] | None:
    if not isinstance(tensor, torch.Tensor):
        return None
    detached = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
    flat = detached.reshape(-1)
    first = min(8, int(flat.numel()))
    return {
        "shape": tuple(detached.shape),
        "mean": float(detached.mean().item()) if flat.numel() else 0.0,
        "std": float(detached.std(unbiased=False).item()) if flat.numel() else 0.0,
        "min": float(detached.min().item()) if flat.numel() else 0.0,
        "max": float(detached.max().item()) if flat.numel() else 0.0,
        "first8": flat[:first].tolist(),
        "sha256_fp32": hashlib.sha256(flat.numpy().tobytes()).hexdigest(),
    }


def _tensor_sha(tensor: torch.Tensor, *, dtype: torch.dtype) -> str:
    flat = tensor.detach().to(dtype=dtype).reshape(-1).cpu().contiguous()
    return hashlib.sha256(flat.numpy().tobytes()).hexdigest()
