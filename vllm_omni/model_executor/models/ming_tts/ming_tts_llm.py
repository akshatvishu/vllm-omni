# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from vllm.model_executor.models.utils import init_vllm_registered_model, is_pp_missing_parameter
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .aggregator import Aggregator
from .backbone import MingQwen2Backbone
from .config_ming_tts import (
    KEY_CFG,
    KEY_DECODE_STEP,
    KEY_LATENT_HISTORY,
    KEY_MAX_DECODE_STEPS,
    KEY_MIN_DECODE_STEPS,
    KEY_NEXT_EMBEDS,
    KEY_REQUEST_ID,
    KEY_SIGMA,
    KEY_TEMPERATURE,
    KEY_TEXT_MODE,
    MingTTSConfig,
)
from .flowloss_head import FlowLoss
from .patch_emission import (
    MING_STOP_REASON_CODES,
    MING_STOP_REASON_CONTINUE,
    MING_STOP_REASON_KEY,
    MING_STOP_REASON_MAX_DECODE_STEPS,
    MING_STOP_REASON_STOP_HEAD,
    _coerce_latent_history,
    _get_request_token_counts,
    _normalize_request_infos,
    _resolve_max_decode_steps_batch,
    _resolve_min_decode_steps_batch,
    _resolve_ming_stop_decision,
    _resolve_optional_runtime_int,
    _resolve_runtime_float,
    _resolve_runtime_int,
    _resolve_stop_probs_batch,
)

logger = init_logger(__name__)
_ORIGINAL_INIT_VLLM_REGISTERED_MODEL = init_vllm_registered_model


class MingLLMModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.ming_config = MingTTSConfig.from_hf_config(vllm_config.model_config.hf_config)
        self.ming_config.validate()
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.quant_config = vllm_config.quant_config
        self.fm_dtype = _resolve_ming_runtime_dtype(vllm_config)
        self.model = (
            init_vllm_registered_model(vllm_config=vllm_config, architectures=["Qwen2ForCausalLM"])
            if init_vllm_registered_model is not _ORIGINAL_INIT_VLLM_REGISTERED_MODEL
            else MingQwen2Backbone(vllm_config=vllm_config, prefix=prefix)
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
        self._last_sample_decode_steps = None
        self._last_sample_stop_probs = None
        self._last_sample_max_decode_steps = None
        self._last_sample_min_decode_steps = None
        self._pending_sample_stop_inputs = None
        self._last_text_mode = False

    def embed_input_ids(
        self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor | None = None, **_: Any
    ) -> torch.Tensor:
        if hasattr(self.model, "embed_input_ids"):
            if inputs_embeds is not None:
                return self.model.embed_input_ids(input_ids, inputs_embeds=inputs_embeds)
            try:
                return self.model.embed_input_ids(input_ids)
            except TypeError:
                return self.model.embed_input_ids(input_ids, inputs_embeds=inputs_embeds)
        return inputs_embeds if inputs_embeds is not None else self.model.embed_input_ids(input_ids)

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
        request_infos = _normalize_request_infos(model_intermediate_buffer)
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
                text_hidden_states=hidden_states, multimodal_outputs=None, intermediate_tensors=intermediate_tensors
            )
        if latent_history is not None and not token_counts:
            token_counts, request_infos = [hidden_states.shape[0]], [{KEY_LATENT_HISTORY: latent_history}]

        total_tokens = hidden_states.shape[0]
        latent_patch_tokens = next_embed_tokens = new_history_tokens = None
        stop_prob_tokens = decode_step_tokens = has_patch = max_decode_step_tokens = min_decode_step_tokens = (
            stop_reason_code_tokens
        ) = None
        pending_updates: dict[str, dict[str, Any]] = {}
        sampled_decode_steps: list[int] = []
        sampled_stop_probs: list[torch.Tensor] = []
        sampled_max_decode_steps: list[int] = []
        sampled_min_decode_steps: list[int] = []
        cursor = 0
        any_decode = False
        for req_idx, token_count in enumerate(token_counts):
            end = min(cursor + token_count, total_tokens)
            if end <= cursor:
                continue
            req_info = request_infos[req_idx] if req_idx < len(request_infos) else {}
            req_id = req_info.get(KEY_REQUEST_ID)
            req_history = _coerce_latent_history(
                req_info.get(KEY_LATENT_HISTORY), device=hidden_states.device, dtype=self.fm_dtype, cfg=self.ming_config
            )
            if req_history is None:
                cursor = end
                continue
            decode_step = int(req_info.get(KEY_DECODE_STEP, req_info.get("generated_len", 0)))
            decode_hidden, output_index = (
                (hidden_states[cursor:end], cursor) if token_count == 1 else (hidden_states[end - 1 : end], end - 1)
            )
            sampled_token_latent, next_embeds, new_history, stop_probs = self._decode_one_step(
                hidden_states=decode_hidden,
                latent_history=req_history,
                cfg_scale=_resolve_runtime_float(req_info, KEY_CFG, self.ming_config.cfg),
                sigma=_resolve_runtime_float(req_info, KEY_SIGMA, self.ming_config.sigma),
                temperature=_resolve_runtime_float(req_info, KEY_TEMPERATURE, self.ming_config.temperature),
            )
            req_max_decode_steps = _resolve_runtime_int(
                req_info, KEY_MAX_DECODE_STEPS, self.ming_config.max_decode_steps
            )
            req_min_decode_steps = _resolve_optional_runtime_int(req_info, KEY_MIN_DECODE_STEPS, 0)
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
                stop_reason_code_tokens = torch.zeros((total_tokens,), dtype=torch.int32, device=hidden_states.device)
            latent_patch_tokens[output_index : output_index + 1] = sampled_token_latent
            next_embed_tokens[output_index : output_index + 1] = next_embeds
            new_history_tokens[output_index : output_index + 1] = new_history
            stop_prob_tokens[output_index : output_index + 1] = stop_probs
            decode_step_tokens[output_index : output_index + 1] = decode_step
            max_decode_step_tokens[output_index : output_index + 1] = req_max_decode_steps
            min_decode_step_tokens[output_index : output_index + 1] = req_min_decode_steps
            has_patch[output_index : output_index + 1] = True
            sampled_decode_steps.append(decode_step)
            sampled_stop_probs.append(stop_probs.reshape(-1)[0])
            sampled_max_decode_steps.append(req_max_decode_steps)
            sampled_min_decode_steps.append(req_min_decode_steps)
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
            stop_reason_code_tokens[output_index : output_index + 1] = MING_STOP_REASON_CODES[stop_reason]
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
                text_hidden_states=hidden_states, multimodal_outputs=None, intermediate_tensors=intermediate_tensors
            )
        self._last_sample_decode_steps = (
            torch.tensor(sampled_decode_steps, dtype=torch.int32, device=hidden_states.device)
            if sampled_decode_steps
            else None
        )
        self._last_sample_stop_probs = (
            torch.stack(sampled_stop_probs).to(device=hidden_states.device) if sampled_stop_probs else None
        )
        self._last_sample_max_decode_steps = (
            torch.tensor(sampled_max_decode_steps, dtype=torch.int32, device=hidden_states.device)
            if sampled_max_decode_steps
            else None
        )
        self._last_sample_min_decode_steps = (
            torch.tensor(sampled_min_decode_steps, dtype=torch.int32, device=hidden_states.device)
            if sampled_min_decode_steps
            else None
        )
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
                MING_STOP_REASON_KEY: stop_reason_code_tokens,
            },
            intermediate_tensors=intermediate_tensors,
        )

    def pop_postprocess_update(self, req_id: str) -> dict[str, Any]:
        return self._pending_postprocess_updates.pop(req_id, {}) if isinstance(req_id, str) else {}

    def compute_logits(
        self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: SamplingMetadata
    ) -> torch.Tensor | None:
        decode_steps = stop_probs_tensor = max_decode_steps_tensor = min_decode_steps_tensor = None
        text_mode = self._last_text_mode
        if isinstance(hidden_states, OmniOutput):
            mm = hidden_states.multimodal_outputs or {}
            text_mode = bool(mm.get(KEY_TEXT_MODE, text_mode))
            decode_steps = mm.get("ming_decode_step")
            stop_probs_tensor = mm.get("ming_stop_prob")
            max_decode_steps_tensor = mm.get("ming_max_decode_steps")
            min_decode_steps_tensor = mm.get("ming_min_decode_steps")
            hidden_states = hidden_states.text_hidden_states
        if text_mode:
            self._pending_sample_stop_inputs = None
            return (
                None
                if hidden_states is None or hidden_states.numel() == 0
                else self.model.compute_logits(hidden_states)
            )
        max_decode_steps_tensor = (
            self._last_sample_max_decode_steps if max_decode_steps_tensor is None else max_decode_steps_tensor
        )
        min_decode_steps_tensor = (
            self._last_sample_min_decode_steps if min_decode_steps_tensor is None else min_decode_steps_tensor
        )
        decode_steps = self._last_sample_decode_steps if decode_steps is None else decode_steps
        stop_probs_tensor = self._last_sample_stop_probs if stop_probs_tensor is None else stop_probs_tensor
        if hidden_states is None or hidden_states.numel() == 0:
            self._pending_sample_stop_inputs = None
            return None
        if hidden_states.dim() != 2:
            raise RuntimeError(
                f"Expected hidden_states rank-2 [B,H] in compute_logits, got {tuple(hidden_states.shape)}"
            )
        batch_size = hidden_states.shape[0]
        stop_prob_values = _resolve_stop_probs_batch(stop_probs_tensor, batch_size=batch_size)
        if stop_prob_values is None:
            stop_probs = self.stop_head(hidden_states.to(dtype=self.fm_dtype)).softmax(dim=-1)[:, 1]
            if not torch.isfinite(stop_probs).all():
                raise RuntimeError("Non-finite stop_probs in Ming compute_logits.")
            stop_prob_values = [float(stop_probs[i].item()) for i in range(batch_size)]
        steps = self._get_decode_steps(decode_steps, sampling_metadata, batch_size)
        max_decode_steps = _resolve_max_decode_steps_batch(
            max_decode_steps_tensor, batch_size=batch_size, default_value=self.ming_config.max_decode_steps
        )
        min_decode_steps = _resolve_min_decode_steps_batch(min_decode_steps_tensor, batch_size=batch_size)
        logits = torch.full(
            (batch_size, self.ming_config.llm_vocab_size),
            float("-inf"),
            device=hidden_states.device,
            dtype=torch.float32,
        )
        for i in range(batch_size):
            _, _, _, _, next_token_id = _resolve_ming_stop_decision(
                step=steps[i],
                stop_prob=stop_prob_values[i],
                stop_threshold=float(self.ming_config.stop_head_threshold),
                min_stop_step=int(self.ming_config.stop_head_min_steps),
                min_decode_steps=min_decode_steps[i],
                max_decode_steps=max_decode_steps[i],
                audio_dummy_token_id=int(self.ming_config.audio_dummy_token_id),
                text_eos_token_id=int(self.ming_config.text_eos_token_id),
            )
            logits[i, int(next_token_id)] = 0.0
        self._pending_sample_stop_inputs = {
            "steps": steps,
            "stop_probs": stop_prob_values,
            "max_decode_steps": max_decode_steps,
            "min_decode_steps": min_decode_steps,
        }
        return logits

    def sample(self, logits, sampling_metadata):
        if logits is None:
            return None
        if self._last_text_mode:
            return self.model.sample(logits, sampling_metadata)
        del sampling_metadata
        stop_inputs = self._pending_sample_stop_inputs
        self._pending_sample_stop_inputs = None
        if stop_inputs is None:
            return SamplerOutput(
                sampled_token_ids=logits.argmax(dim=-1, keepdim=True).to(dtype=torch.int32), logprobs_tensors=None
            )
        sampled_ids = []
        for i in range(logits.shape[0]):
            _, _, _, _, next_token_id = _resolve_ming_stop_decision(
                step=int(stop_inputs["steps"][i]),
                stop_prob=float(stop_inputs["stop_probs"][i]),
                stop_threshold=float(self.ming_config.stop_head_threshold),
                min_stop_step=int(self.ming_config.stop_head_min_steps),
                min_decode_steps=int(stop_inputs["min_decode_steps"][i]),
                max_decode_steps=int(stop_inputs["max_decode_steps"][i]),
                audio_dummy_token_id=int(self.ming_config.audio_dummy_token_id),
                text_eos_token_id=int(self.ming_config.text_eos_token_id),
            )
            sampled_ids.append(next_token_id)
        return SamplerOutput(
            sampled_token_ids=torch.tensor(sampled_ids, dtype=torch.int32, device=logits.device).reshape(-1, 1),
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
                "Batch mismatch: "
                f"hidden_states B={hidden_states.shape[0]} "
                f"vs latent_history B={latent_history.shape[0]}"
            )
        # [Batch, Hidden] -> [Batch, Time, Hidden] = [B, 1, H] for FlowLoss conditioning.
        z_diff_cond = hidden_states.to(dtype=self.fm_dtype).unsqueeze(1)
        if not torch.isfinite(z_diff_cond).all():
            raise RuntimeError("Non-finite z_diff_cond before FlowLoss.sample().")
        sampled_token_latent = self.flowloss.sample(
            z=z_diff_cond,
            latent_history=latent_history,
            cfg=cfg_scale,
            patch_size=self.ming_config.patch_size,
            sigma=sigma,
            temperature=temperature,
        )
        expected_shape = (hidden_states.shape[0], self.ming_config.patch_size, self.ming_config.latent_dim)
        if tuple(sampled_token_latent.shape) != expected_shape:
            raise RuntimeError(
                f"FlowLoss output shape mismatch: got {tuple(sampled_token_latent.shape)}, expected {expected_shape}"
            )
        new_history = torch.cat([latent_history[:, self.ming_config.patch_size :, :], sampled_token_latent], dim=1)
        # Aggregator expects [Batch, Time, Dimension] = [B, 4, 64] and returns [B, 1, H].
        next_embeds = self.linear_proj_audio(sampled_token_latent)
        stop_probs = self.stop_head(hidden_states.to(dtype=self.fm_dtype)).softmax(dim=-1)[:, 1]
        if not torch.isfinite(sampled_token_latent).all():
            raise RuntimeError("Non-finite sampled_token_latent in Ming decode step.")
        if not torch.isfinite(next_embeds).all():
            raise RuntimeError("Non-finite next_embeds in Ming decode step.")
        if not torch.isfinite(stop_probs).all():
            raise RuntimeError("Non-finite stop_probs in Ming decode step.")
        return sampled_token_latent, next_embeds, new_history, stop_probs

    def _get_decode_steps(
        self, decode_steps: torch.Tensor | None, sampling_metadata: SamplingMetadata, batch_size: int
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
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        skipped: list[str] = []
        mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        for ckpt_name, loaded_weight in weights:
            name = ckpt_name
            if self.quant_config is not None and (scale_name := self.quant_config.get_cache_scale(name)):
                if scale_name not in params_dict:
                    skipped.append(ckpt_name)
                    continue
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0])
                loaded_params.add(scale_name)
                continue
            mapped_name = None
            for param_name, weight_name, shard_id in mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                    mapped_name = None
                    break
                if is_pp_missing_parameter(mapped_name, self) or mapped_name not in params_dict:
                    mapped_name = None
                    continue
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight) if weight_loader == default_weight_loader else weight_loader(
                    param, loaded_weight, shard_id
                )
                loaded_params.add(mapped_name)
                break
            if mapped_name in loaded_params or name.endswith(".bias") and name not in params_dict:
                continue
            name = maybe_remap_kv_scale_name(name, params_dict)
            if name is None:
                continue
            if name.startswith("model.") and name not in params_dict and f"model.{name}" in params_dict:
                name = f"model.{name}"
            if is_pp_missing_parameter(name, self):
                continue
            if name not in params_dict:
                skipped.append(ckpt_name)
                continue
            getattr(params_dict[name], "weight_loader", default_weight_loader)(params_dict[name], loaded_weight)
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
    if isinstance(backbone_out, (tuple, list)) and backbone_out and isinstance(backbone_out[0], torch.Tensor):
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
    missing = {key for key in params_dict if key.startswith(prefix)} - loaded_params
    if not missing:
        return
    msg = (
        f"MingLLMModel: {len(missing)} {module_name} params not loaded "
        f"(prefix={prefix}). First few: {sorted(missing)[:5]}"
    )
    if fatal:
        raise RuntimeError(msg)
    warnings.warn(msg, stacklevel=3)


__all__ = [
    "Aggregator",
    "FlowLoss",
    "MING_STOP_REASON_CODES",
    "MING_STOP_REASON_CONTINUE",
    "MING_STOP_REASON_KEY",
    "MING_STOP_REASON_MAX_DECODE_STEPS",
    "MING_STOP_REASON_STOP_HEAD",
    "MingLLMModel",
    "_coerce_latent_history",
    "_extract_hidden_states",
    "_resolve_ming_runtime_dtype",
    "_resolve_ming_stop_decision",
    "_warn_missing_prefix",
]
