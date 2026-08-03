# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from:
# https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/modeling_minicpmo.py
"""MiniCPM-o 4.5 native autoregressive Talker.

Pipeline:
  1. Receive thinker hidden_states + full token IDs via additional_information
  2. Extract tts_bos..tts_eos region
  3. Build condition: emb_text(tokens) + projector_semantic(hidden) (hidden_text_merge)
  4. Continuously generate request-aligned discrete audio-code deltas
"""

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.utils import maybe_prefix
from vllm.v1.sample.sampler import Sampler

from vllm_omni.experimental.fullduplex.engine.intermediate import get_tts_handoff
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

_REPETITION_WINDOW = 16
# Codec-token sampling happens inside the model; vLLM sampling parameters
# only choose the Talker's binary continue/stop row.
_CODEC_SEED = 42
_CODEC_TEMPERATURE = 0.8
# Match MiniCPM-o 4.5's official utils.TTSSamplingParams defaults.
_CODEC_TOP_K = 25
_CODEC_TOP_P = 0.85
_CODEC_REPETITION_PENALTY = 1.05
_TEXT_CHUNK_SIZE = 10
_MAX_AUDIO_TOKENS_PER_CONDITION = 500
_DUPLEX_CODEC_TOKENS_PER_CHUNK = 26
_MINICPMO_SLIDING_RECOMPUTE = "minicpmo_sliding_recompute"
_MINICPMO_SLIDING_WINDOW_SIZE = "minicpmo_sliding_window_size"
_MINICPMO_SLIDING_RECOMPUTED_CHUNKS = "minicpmo_sliding_recomputed_chunks"
_DEFAULT_SLIDING_WINDOW_SIZE = 2
_DEFAULT_SLIDING_RECOMPUTED_CHUNKS = 1


def _restore_weight_norm_weight(weight_g: torch.Tensor, weight_v: torch.Tensor) -> torch.Tensor:
    """Materialize ``weight_norm(..., dim=0)`` checkpoint parameters."""
    return torch._weight_norm(weight_v, weight_g, dim=0)


def _apply_repetition_penalty(
    logits: torch.Tensor,
    history: torch.Tensor,
    *,
    penalty: float,
    window_size: int,
) -> torch.Tensor:
    """Match MiniCPMTTS' frequency-aware repetition penalty."""
    if penalty == 1.0 or history.numel() == 0:
        return logits
    recent = history.reshape(-1)[-window_size:].to(device=logits.device, dtype=torch.long)
    frequencies = torch.bincount(recent, minlength=logits.shape[-1]).to(dtype=logits.dtype)
    alpha = torch.pow(torch.as_tensor(penalty, device=logits.device, dtype=logits.dtype), frequencies)
    return torch.where(logits < 0, logits * alpha, logits / alpha)


def _apply_top_k_top_p(
    logits: torch.Tensor,
    *,
    top_k: int | None,
    top_p: float | None,
    min_tokens_to_keep: int = 3,
) -> torch.Tensor:
    """Apply the same candidate floors as the upstream Transformers warpers."""
    filtered = logits.clone()
    vocab_size = filtered.shape[-1]
    # MiniCPM-o's gen_logits() appends TopPLogitsWarper before
    # TopKLogitsWarper. The order is observable for fixed-seed sampling.
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=False, dim=-1)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative_probs <= (1.0 - float(top_p))
        remove[..., -min_tokens_to_keep:] = False
        remove = remove.scatter(-1, sorted_indices, remove)
        filtered.masked_fill_(remove, float("-inf"))
    if top_k is not None and top_k > 0:
        keep = min(vocab_size, max(int(top_k), min_tokens_to_keep))
        threshold = torch.topk(filtered, keep, dim=-1).values[..., -1, None]
        filtered.masked_fill_(filtered < threshold, float("-inf"))
    return filtered


class _MiniCPMTTSProjector(nn.Module):
    """Checkpoint-compatible hidden-state projector used by MiniCPMTTS."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(hidden_states)))


class MiniCPMO45OmniTTSForConditionalGeneration(nn.Module, SupportsPP):
    """Runner-owned MiniCPM-o 4.5 Talker that emits codec tokens only."""

    requires_request_sample_eligibility = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import MiniCPMOConfig

        config: MiniCPMOConfig = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self._batch_stop_logits: torch.Tensor | None = None
        self._request_generators: dict[str, torch.Generator] = {}
        self._request_audio_states: dict[str, dict[str, Any]] = {}
        self._deferred_cleanup_ids: set[str] = set()

        tts_config = getattr(config, "tts_config", None)
        if tts_config is None and getattr(config, "model_type", None) == "minicpmtts":
            tts_config = config
        if tts_config is not None:
            self._tts_config = tts_config
            self._tts_bos_id = getattr(tts_config, "audio_bos_token_id", 151687)
            self._text_eos_id = getattr(tts_config, "text_eos_token_id", 151692)
            self._num_audio_tokens = getattr(tts_config, "num_audio_tokens", 6562)
            self._hidden_size = getattr(tts_config, "hidden_size", 768)
            self._normalize = getattr(tts_config, "normalize_projected_hidden", True)
            self._codec_seed = int(getattr(tts_config, "seed", _CODEC_SEED))
            self._codec_temperature = float(getattr(tts_config, "temperature", _CODEC_TEMPERATURE))
            self._codec_top_k = int(getattr(tts_config, "top_k", _CODEC_TOP_K))
            self._codec_top_p = float(getattr(tts_config, "top_p", _CODEC_TOP_P))
            self._codec_repetition_penalty = float(getattr(tts_config, "repetition_penalty", _CODEC_REPETITION_PENALTY))
        else:
            self._tts_config = None

        self._sliding_recompute_enabled = bool(getattr(config, _MINICPMO_SLIDING_RECOMPUTE, False))
        window_size = getattr(config, _MINICPMO_SLIDING_WINDOW_SIZE, _DEFAULT_SLIDING_WINDOW_SIZE)
        recomputed_chunks = getattr(config, _MINICPMO_SLIDING_RECOMPUTED_CHUNKS, _DEFAULT_SLIDING_RECOMPUTED_CHUNKS)
        self._sliding_window_size = int(window_size if window_size is not None else _DEFAULT_SLIDING_WINDOW_SIZE)
        self._sliding_recomputed_chunks = int(
            recomputed_chunks if recomputed_chunks is not None else _DEFAULT_SLIDING_RECOMPUTED_CHUNKS
        )
        self._sliding_recompute_settings()

        self.has_preprocess = True
        self.has_postprocess = False
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("audio_codes", "current"),
            ("audio_codes", "accumulated"),
            ("audio_state", "condition_chunks"),
        }
        self._init_native_talker(prefix)

    def _init_native_talker(self, prefix: str) -> None:
        if self._tts_config is None:
            raise ValueError("MiniCPM-o continuous Talker requires tts_config")
        cfg = self._tts_config
        if int(getattr(cfg, "num_vq", 1)) != 1:
            raise ValueError(
                "MiniCPM-o continuous Talker currently requires num_vq=1; "
                f"checkpoint reports {getattr(cfg, 'num_vq', None)}"
            )
        llama_config = LlamaConfig(
            vocab_size=32000,
            hidden_size=int(cfg.hidden_size),
            intermediate_size=int(cfg.intermediate_size),
            num_hidden_layers=int(cfg.num_hidden_layers),
            num_attention_heads=int(cfg.num_attention_heads),
            num_key_value_heads=int(cfg.num_key_value_heads),
            hidden_act=getattr(cfg, "hidden_act", "silu"),
            max_position_embeddings=int(cfg.max_position_embeddings),
            rms_norm_eps=float(getattr(cfg, "rms_norm_eps", 1e-6)),
            tie_word_embeddings=False,
        )
        talker_config = self.vllm_config.with_hf_config(llama_config, architectures=["LlamaForCausalLM"])
        talker_config.model_config.hf_text_config = llama_config
        self.tts_model = LlamaModel(
            vllm_config=talker_config,
            prefix=maybe_prefix(prefix, "tts_obj.model"),
        )
        self.emb_text = nn.Embedding(int(cfg.num_text_tokens), int(cfg.hidden_size))
        self.projector_semantic = _MiniCPMTTSProjector(int(cfg.llm_dim), int(cfg.hidden_size))
        self.emb_code = nn.ModuleList(
            [nn.Embedding(int(cfg.num_audio_tokens), int(cfg.hidden_size)) for _ in range(int(cfg.num_vq))]
        )
        self.head_code = nn.ModuleList(
            [nn.Linear(int(cfg.hidden_size), int(cfg.num_audio_tokens), bias=False) for _ in range(int(cfg.num_vq))]
        )
        self.make_empty_intermediate_tensors = self.tts_model.make_empty_intermediate_tensors

    def _boundary_embeddings(self) -> torch.Tensor:
        """Embed the ``<text_eos><audio_bos>`` tail for the last normal chunk."""
        ids = torch.tensor(
            [self._text_eos_id, self._tts_bos_id],
            device=self.emb_text.weight.device,
            dtype=torch.long,
        )
        return self.emb_text(ids)

    def _audio_bos_embedding(self) -> torch.Tensor:
        token_id = torch.tensor(
            [self._tts_bos_id],
            device=self.emb_text.weight.device,
            dtype=torch.long,
        )
        return self.emb_text(token_id)

    def _sliding_recompute_settings(self) -> tuple[bool, int, int]:
        """Return and validate the MiniCPM sliding-recompute settings."""
        enabled = bool(getattr(self, "_sliding_recompute_enabled", False))
        window_size = int(getattr(self, "_sliding_window_size", _DEFAULT_SLIDING_WINDOW_SIZE))
        recomputed_chunks = int(getattr(self, "_sliding_recomputed_chunks", _DEFAULT_SLIDING_RECOMPUTED_CHUNKS))
        if enabled and (window_size <= 0 or recomputed_chunks < 0 or window_size <= recomputed_chunks):
            raise ValueError(
                "MiniCPM-o sliding recompute requires window_size > recomputed_chunks >= 0; "
                f"received window_size={window_size}, recomputed_chunks={recomputed_chunks}"
            )
        return enabled, window_size, recomputed_chunks

    def _should_recompute_condition(self, condition_index: int) -> bool:
        """Match MiniCPM-o's official sliding_recompute cadence."""
        enabled, window_size, recomputed_chunks = self._sliding_recompute_settings()
        if not enabled:
            return False
        return (
            condition_index >= window_size
            and (condition_index - recomputed_chunks) % (window_size - recomputed_chunks) == 0
        )

    def _record_completed_condition(self, state: dict[str, Any], condition_index: int) -> None:
        """Retain only the bounded audio history needed by the next recompute."""
        _, _, recomputed_chunks = self._sliding_recompute_settings()
        completed = state.setdefault("completed_condition_audio", [])
        if not isinstance(completed, list):
            completed = []
        condition_codes = state.get("condition_audio_codes", [])
        if not isinstance(condition_codes, list):
            condition_codes = []
        completed.append(
            {
                "condition_index": int(condition_index),
                "codes": [int(code) for code in condition_codes],
            }
        )
        state["completed_condition_audio"] = completed[-recomputed_chunks:] if recomputed_chunks else []
        state["condition_audio_codes"] = []

    def _sliding_recompute_prompt_stats(self, state: dict[str, Any], condition_index: int) -> tuple[int, int]:
        """Return replacement-prompt length and retained audio-token count."""
        _, _, recomputed_chunks = self._sliding_recompute_settings()
        chunks = state.get("condition_chunks")
        completed = state.get("completed_condition_audio")
        if not isinstance(chunks, list) or not isinstance(completed, list):
            raise RuntimeError("MiniCPM-o sliding recompute is missing condition or audio history")
        completed_by_index = {
            int(item["condition_index"]): item
            for item in completed
            if isinstance(item, dict) and "condition_index" in item
        }
        prompt_len = 0
        audio_tokens = 0
        for previous_index in range(condition_index - recomputed_chunks, condition_index):
            item = completed_by_index.get(previous_index)
            if item is None:
                raise RuntimeError(
                    "MiniCPM-o sliding recompute is missing completed audio for "
                    f"condition_index={previous_index}; available={sorted(completed_by_index)}"
                )
            codes = item.get("codes", [])
            prompt_len += int(chunks[previous_index].shape[0]) + len(codes)
            audio_tokens += len(codes)
        prompt_len += int(chunks[condition_index].shape[0])
        return prompt_len, audio_tokens

    def _condition_chunk_on_device(
        self,
        state: dict[str, Any],
        condition_index: int,
        request_id: str,
    ) -> torch.Tensor:
        chunks = state.get("condition_chunks")
        if not isinstance(chunks, list):
            raise RuntimeError("MiniCPM-o Talker is missing condition chunks")
        if condition_index < 0 or condition_index >= len(chunks):
            raise RuntimeError(
                "MiniCPM-o Talker condition index is out of range: "
                f"request_id={request_id} condition_index={condition_index} condition_count={len(chunks)}"
            )
        chunk = chunks[condition_index]
        if not isinstance(chunk, torch.Tensor):
            raise RuntimeError(
                "MiniCPM-o Talker condition chunk is not a tensor: "
                f"request_id={request_id} condition_index={condition_index} type={type(chunk).__name__}"
            )
        target_module = getattr(self, "emb_text", None)
        if target_module is None:
            target_module = self.emb_code[0]
        target = target_module.weight
        if chunk.device != target.device or chunk.dtype != target.dtype:
            logger.warning(
                "[MiniCPM-o][Stage1][condition-device-normalize] request_id=%s "
                "condition_index=%s source_device=%s target_device=%s source_dtype=%s target_dtype=%s",
                request_id,
                condition_index,
                chunk.device,
                target.device,
                chunk.dtype,
                target.dtype,
            )
            chunk = chunk.to(device=target.device, dtype=target.dtype)
            chunks[condition_index] = chunk
        return chunk

    def _build_sliding_recompute_condition(
        self,
        state: dict[str, Any],
        condition_index: int,
        *,
        request_id: str = "unknown",
    ) -> torch.Tensor:
        """Build previous condition/audio embeddings plus the active condition."""
        _, _, recomputed_chunks = self._sliding_recompute_settings()
        chunks = state.get("condition_chunks")
        completed = state.get("completed_condition_audio")
        if not isinstance(chunks, list) or not isinstance(completed, list):
            raise RuntimeError("MiniCPM-o sliding recompute is missing condition or audio history")
        completed_by_index = {
            int(item["condition_index"]): item
            for item in completed
            if isinstance(item, dict) and "condition_index" in item
        }
        embeddings: list[torch.Tensor] = []
        for previous_index in range(condition_index - recomputed_chunks, condition_index):
            item = completed_by_index.get(previous_index)
            if item is None:
                raise RuntimeError(
                    "MiniCPM-o sliding recompute cannot rebuild condition "
                    f"{condition_index}: missing previous condition {previous_index}"
                )
            embeddings.append(self._condition_chunk_on_device(state, previous_index, request_id))
            codes = item.get("codes", [])
            if codes:
                code_ids = torch.as_tensor(
                    codes,
                    device=self.emb_code[0].weight.device,
                    dtype=torch.long,
                )
                embeddings.append(self.emb_code[0](code_ids))
        embeddings.append(self._condition_chunk_on_device(state, condition_index, request_id))
        return torch.cat(embeddings, dim=0)

    def _build_condition_chunks(
        self,
        tts_token_ids: torch.Tensor,
        tts_hidden_states: torch.Tensor,
        *,
        native_duplex: bool = False,
    ) -> list[torch.Tensor]:
        if tts_token_ids.numel() == 0 or tts_hidden_states.numel() == 0:
            return [self._boundary_embeddings()]
        device = self.emb_text.weight.device
        dtype = self.emb_text.weight.dtype
        token_ids = tts_token_ids.to(device=device, dtype=torch.long).reshape(-1)
        hidden = tts_hidden_states.to(device=device, dtype=dtype)
        if hidden.shape[0] != token_ids.shape[0] and token_ids.shape[0] != 1:
            raise ValueError(
                "MiniCPM-o Talker condition length mismatch: "
                f"token_ids={token_ids.shape[0]} hidden_states={hidden.shape[0]}"
            )
        text_embeds = self.emb_text(token_ids)
        hidden_embeds = self.projector_semantic(hidden)
        if self._normalize:
            hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
        condition = text_embeds + hidden_embeds
        if native_duplex:
            return [torch.cat([condition, self._audio_bos_embedding()], dim=0)]
        chunks = list(condition.split(_TEXT_CHUNK_SIZE, dim=0))
        audio_bos = self._audio_bos_embedding()
        chunks = [torch.cat([chunk, audio_bos], dim=0) for chunk in chunks]
        chunks[-1] = torch.cat([chunks[-1][:-1], self._boundary_embeddings()], dim=0)
        return chunks

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build request-local prefill/decode embeddings for the vLLM runner."""
        del input_embeds
        span_len = int(input_ids.shape[0])
        is_prefill = bool(info_dict.get("_omni_is_prefill", False))
        request_id = str(info_dict.get("request_id", "0"))
        state = info_dict.get("audio_state")
        local_state = getattr(self, "_request_audio_states", {}).get(request_id)
        if isinstance(local_state, dict):
            state = local_state
        first_call = not isinstance(state, dict)

        if isinstance(state, dict) and state.get("sliding_recompute_pending"):
            condition_index = int(state.get("condition_chunk_index", 0))
            recompute_embeds = self._build_sliding_recompute_condition(
                state,
                condition_index,
                request_id=request_id,
            )
            prompt_len = info_dict.get("_omni_prompt_len")
            target_len = int(prompt_len) if prompt_len is not None else int(recompute_embeds.shape[0])
            if target_len < recompute_embeds.shape[0]:
                raise ValueError(
                    "MiniCPM-o sliding recompute prompt is shorter than its rebuilt condition: "
                    f"request_id={request_id} prompt_len={target_len} rebuilt_len={recompute_embeds.shape[0]}"
                )
            prefix_len = target_len - recompute_embeds.shape[0]
            if prefix_len > 0:
                placeholder_ids = torch.zeros(
                    prefix_len,
                    dtype=torch.long,
                    device=self.emb_text.weight.device,
                )
                recompute_embeds = torch.cat(
                    [self.emb_text(placeholder_ids), recompute_embeds],
                    dim=0,
                )
            offset = int(info_dict.get("_omni_num_computed_tokens", 0))
            embeds = recompute_embeds[offset : offset + span_len]
            if embeds.shape[0] != span_len:
                raise ValueError(
                    "MiniCPM-o sliding recompute prefill span exceeds rebuilt condition: "
                    f"request_id={request_id} condition_index={condition_index} offset={offset} "
                    f"span={span_len} prompt_len={target_len} rebuilt_len={recompute_embeds.shape[0]}"
                )
            if offset + span_len >= target_len:
                state["sliding_recompute_pending"] = False
                state["conditioning"] = False
                state["condition_sample_ready"] = False
                state["condition_step"] = 0
                logger.info(
                    "[MiniCPM-o][Stage1][sliding-recompute-prefill] request_id=%s "
                    "condition_index=%s prompt_len=%s computed_offset=%s span=%s "
                    "previous_audio_tokens=%s reset_offset=0",
                    request_id,
                    condition_index,
                    target_len,
                    offset,
                    span_len,
                    state.get("sliding_recompute_audio_tokens", 0),
                )
            empty_codes = torch.empty(0, dtype=torch.long, device=embeds.device)
            return (
                input_ids,
                embeds,
                {
                    "audio_state": state,
                    "audio_codes": {
                        "current": empty_codes,
                        "accumulated": empty_codes,
                    },
                },
            )

        if is_prefill or first_call:
            token_ids, hidden_states = get_tts_handoff(info_dict)
            # Cross-process stage transport serializes CPU tensors as lists.
            # Normalize both local tensor handoffs and transported payloads
            # before validating/building the Talker condition.
            if isinstance(token_ids, (list, tuple)):
                token_ids = torch.as_tensor(token_ids, dtype=torch.long)
            if isinstance(hidden_states, (list, tuple)):
                hidden_states = torch.as_tensor(hidden_states, dtype=torch.float32)
            if not isinstance(token_ids, torch.Tensor) or not isinstance(hidden_states, torch.Tensor):
                available = sorted(key for key in info_dict if not key.startswith("_"))
                raise ValueError(
                    "MiniCPM-o Talker requires tensor tts_token_ids and "
                    "tts_hidden_states conditioning; "
                    f"received token_ids={type(token_ids).__name__}, "
                    f"hidden_states={type(hidden_states).__name__}, "
                    f"available_keys={available}"
                )
            # An empty condition means the thinker chose not to speak: finish the
            # request up front so it emits zero audio codes instead of killing
            # the stage engine.
            empty_condition = token_ids.numel() == 0 or hidden_states.numel() == 0
            if empty_condition:
                logger.warning_once(
                    "MiniCPM-o Talker received an empty condition (request %s); this request produces no audio.",
                    info_dict.get("request_id"),
                )
            native_duplex = bool(info_dict.get("native_duplex", False))
            condition_chunks = self._build_condition_chunks(
                token_ids,
                hidden_states,
                native_duplex=native_duplex,
            )
            first_chunk_embeds = condition_chunks[0]
            offset = int(info_dict.get("_omni_num_computed_tokens", 0))
            prompt_len = info_dict.get("_omni_prompt_len")
            target_len = int(prompt_len) if prompt_len is not None else offset + span_len
            prefix_len = target_len - first_chunk_embeds.shape[0]
            if prefix_len > 0:
                placeholder_ids = torch.zeros(
                    prefix_len,
                    dtype=torch.long,
                    device=self.emb_text.weight.device,
                )
                first_chunk_embeds = torch.cat(
                    [self.emb_text(placeholder_ids), first_chunk_embeds],
                    dim=0,
                )
            embeds = first_chunk_embeds[offset : offset + span_len]
            if embeds.shape[0] != span_len:
                raise ValueError(
                    "MiniCPM-o Talker prefill span exceeds condition: "
                    f"request_id={info_dict.get('request_id')} offset={offset} "
                    f"span={span_len} condition={first_chunk_embeds.shape[0]} "
                    f"tts_ids={token_ids.shape[0]} tts_hidden={hidden_states.shape[0]} "
                    f"prompt_len={info_dict.get('_omni_prompt_len')}"
                )
            if native_duplex:
                meta = info_dict.get("meta")
                # Native duplex keeps the older one-segment Talker contract.
                duplex_boundary = isinstance(meta, dict) and (
                    bool(meta.get("turn_start", False)) or bool(meta.get("turn_end", False))
                )
                state = {
                    "mode": "native_duplex",
                    "step": 0,
                    "finished": empty_condition,
                    "max_tokens": _DUPLEX_CODEC_TOKENS_PER_CHUNK,
                    "min_tokens": 0 if duplex_boundary else _DUPLEX_CODEC_TOKENS_PER_CHUNK,
                }
            else:
                state = {
                    "mode": "streaming",
                    "step": 0,
                    "condition_step": 0,
                    "finished": empty_condition,
                    "condition_chunks": condition_chunks,
                    "condition_chunk_index": 0,
                    "condition_cursor": 0,
                    "conditioning": False,
                }
                if self._sliding_recompute_settings()[0]:
                    state.update(
                        {
                            "condition_audio_codes": [],
                            "completed_condition_audio": [],
                        }
                    )
            request_states = getattr(self, "_request_audio_states", None)
            if request_states is None:
                request_states = {}
                self._request_audio_states = request_states
            request_states[request_id] = state
            logger.info(
                "[MiniCPM-o][Stage1][prefill] request_id=%s mode=%s input_span=%s "
                "prompt_len=%s computed_offset=%s tts_tokens=%s hidden_rows=%s "
                "condition_count=%s condition_lengths=%s empty_condition=%s",
                request_id,
                state["mode"],
                span_len,
                target_len,
                offset,
                int(token_ids.shape[0]),
                int(hidden_states.shape[0]),
                len(condition_chunks),
                [int(chunk.shape[0]) for chunk in condition_chunks],
                empty_condition,
            )
            empty_codes = torch.empty(0, dtype=torch.long, device=embeds.device)
            return (
                input_ids,
                embeds,
                {
                    "audio_state": state,
                    "audio_codes": {
                        "current": empty_codes,
                        "accumulated": empty_codes,
                    },
                },
            )

        if state.get("conditioning"):
            chunks = state.get("condition_chunks")
            chunk_index = int(state.get("condition_chunk_index", 0))
            cursor = int(state.get("condition_cursor", 0))
            if not isinstance(chunks, list) or chunk_index >= len(chunks):
                raise RuntimeError(
                    f"MiniCPM-o Talker is missing its next text condition chunk for request {request_id}"
                )
            chunk = self._condition_chunk_on_device(state, chunk_index, request_id)
            embeds = chunk[cursor : cursor + span_len]
            if embeds.shape[0] != span_len:
                raise RuntimeError(
                    "MiniCPM-o Talker condition cursor exceeds the active chunk: "
                    f"request_id={request_id} chunk={chunk_index} cursor={cursor} "
                    f"span={span_len} chunk_length={chunk.shape[0]}"
                )
            cursor += span_len
            state["condition_cursor"] = cursor
            state["condition_sample_ready"] = cursor == int(chunk.shape[0])
            logger.info(
                "[MiniCPM-o][Stage1][condition-prefill] request_id=%s condition_index=%s/%s "
                "cursor=%s span=%s condition_len=%s sample_ready=%s",
                request_id,
                chunk_index,
                len(chunks),
                cursor,
                span_len,
                int(chunk.shape[0]),
                state["condition_sample_ready"],
            )
            return input_ids, embeds, {"audio_state": state}

        current = (info_dict.get("audio_codes", {}) or {}).get("current")
        if not isinstance(current, torch.Tensor) or current.numel() != 1:
            if state.get("finished"):
                # A request that finished before sampling any code can still be
                # scheduled for decode steps while sampling min_tokens masks the
                # stop token. make_omni_output ignores its hidden states, so any
                # shape-correct embedding will do.
                weight = self.emb_code[0].weight
                return input_ids, weight.new_zeros((span_len, weight.shape[1])), {}
            raise RuntimeError("MiniCPM-o Talker decode is missing the previous request-local audio code")
        code = current.to(device=self.emb_code[0].weight.device, dtype=torch.long).reshape(1)
        embeds = self.emb_code[0](code)
        return input_ids, embeds, {}

    def _request_generator(self, request_id: str, device: torch.device) -> torch.Generator:
        generator = self._request_generators.get(request_id)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self._codec_seed)
            self._request_generators[request_id] = generator
        return generator

    def _sample_audio_code(
        self,
        hidden_state: torch.Tensor,
        history: torch.Tensor,
        request_id: str,
        step: int,
    ) -> torch.Tensor:
        raw_logits = self.head_code[0](hidden_state).float()
        logits = raw_logits / self._codec_temperature
        request_states = getattr(self, "_request_audio_states", {})
        state = request_states.get(request_id)
        min_tokens = int(state.get("min_tokens", 0)) if isinstance(state, dict) else 0
        eos_id = self._num_audio_tokens - 1
        if step < min_tokens:
            logits[..., eos_id] = float("-inf")
        if history.numel() > 0:
            logits = _apply_repetition_penalty(
                logits,
                history,
                penalty=self._codec_repetition_penalty,
                window_size=_REPETITION_WINDOW,
            )
            logits = _apply_top_k_top_p(
                logits,
                top_k=self._codec_top_k,
                top_p=self._codec_top_p,
                min_tokens_to_keep=3,
            )
        probabilities = torch.softmax(logits, dim=-1)
        generator = self._request_generator(request_id, probabilities.device)
        return torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        ).reshape(())

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        hidden = model_outputs
        infos = kwargs.get("model_intermediate_buffer") or []
        spans = kwargs.get("request_token_spans")
        if spans is None or len(spans) != len(infos):
            raise RuntimeError("MiniCPM-o continuous Talker requires one request_token_span per request")
        sample_eligible = kwargs.get("request_sample_eligible")
        if sample_eligible is None:
            sample_eligible = [True] * len(infos)
        if len(sample_eligible) != len(infos):
            raise RuntimeError(
                f"MiniCPM-o continuous Talker received {len(sample_eligible)} sampling flags for {len(infos)} requests"
            )
        emit_duplex_metadata = any(isinstance(info, dict) and info.get("native_duplex") is True for info in infos)
        sliding_recompute_enabled = self._sliding_recompute_settings()[0]

        stop_rows: list[torch.Tensor] = []
        codec_deltas: list[torch.Tensor] = []
        terminal_flags: list[torch.Tensor] = []
        sliding_recompute_flags: list[torch.Tensor] = []
        sliding_recompute_prompt_lengths: list[torch.Tensor] = []
        native_duplex_flags: list[torch.Tensor] = []
        duplex_epochs: list[torch.Tensor] = []
        duplex_turn_ids: list[torch.Tensor] = []
        segment_texts_utf8: list[torch.Tensor] = []
        turn_end_flags: list[torch.Tensor] = []
        empty_delta = hidden.new_empty((0, 1), dtype=torch.long)
        for index, info in enumerate(infos):
            info_dict = info if isinstance(info, dict) else {}
            native_duplex = info_dict.get("native_duplex") is True
            if emit_duplex_metadata:
                duplex_info = info_dict.get("duplex")
                if not isinstance(duplex_info, dict):
                    duplex_info = {}
                epoch = duplex_info.get("epoch", -1)
                turn_id = duplex_info.get("turn_id", -1)
                if native_duplex and not all(
                    isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in (epoch, turn_id)
                ):
                    raise RuntimeError(
                        "MiniCPM-o native duplex Talker requires non-negative integer "
                        f"epoch and turn_id, got epoch={epoch!r}, turn_id={turn_id!r}"
                    )
                meta_info = info_dict.get("meta")
                if not isinstance(meta_info, dict):
                    meta_info = {}
                segment_text = meta_info.get("native_duplex_segment_text", "") if native_duplex else ""
                if not isinstance(segment_text, str):
                    segment_text = ""
                turn_eos_id = meta_info.get("turn_eos_token_id")
                ids_info = info_dict.get("ids")
                tts_ids = ids_info.get("tts") if native_duplex and isinstance(ids_info, dict) else None
                if isinstance(tts_ids, torch.Tensor):
                    contains_turn_eos = isinstance(turn_eos_id, int) and bool(
                        torch.any(tts_ids.reshape(-1) == turn_eos_id).item()
                    )
                elif isinstance(tts_ids, (list, tuple)):
                    contains_turn_eos = isinstance(turn_eos_id, int) and turn_eos_id in tts_ids
                else:
                    contains_turn_eos = False
                native_duplex_flags.append(torch.tensor(native_duplex, dtype=torch.bool))
                duplex_epochs.append(torch.tensor(epoch if isinstance(epoch, int) else -1, dtype=torch.long))
                duplex_turn_ids.append(torch.tensor(turn_id if isinstance(turn_id, int) else -1, dtype=torch.long))
                segment_texts_utf8.append(
                    torch.tensor(
                        list(segment_text.encode("utf-8")),
                        dtype=torch.uint8,
                    )
                )
                turn_end_flags.append(torch.tensor(native_duplex and contains_turn_eos, dtype=torch.bool))

            if not isinstance(info, dict):
                stop_rows.append(hidden.new_tensor([0.0, float("-inf")]))
                codec_deltas.append(empty_delta)
                terminal_flags.append(torch.tensor(False, dtype=torch.bool))
                if sliding_recompute_enabled:
                    sliding_recompute_flags.append(torch.tensor(False, dtype=torch.bool))
                    sliding_recompute_prompt_lengths.append(torch.tensor(0, dtype=torch.long))
                continue
            start, end = spans[index]
            end = min(int(end), int(hidden.shape[0]))
            if int(start) >= end:
                stop_rows.append(hidden.new_tensor([0.0, float("-inf")]))
                codec_deltas.append(empty_delta)
                terminal_flags.append(torch.tensor(False, dtype=torch.bool))
                if sliding_recompute_enabled:
                    sliding_recompute_flags.append(torch.tensor(False, dtype=torch.bool))
                    sliding_recompute_prompt_lengths.append(torch.tensor(0, dtype=torch.long))
                continue
            request_id = str(info.get("request_id", index))
            request_states = getattr(self, "_request_audio_states", None)
            if request_states is None:
                request_states = {}
                self._request_audio_states = request_states
            state = request_states.get(request_id)
            if not isinstance(state, dict):
                state = dict(info.get("audio_state", {}) or {})
                request_states[request_id] = state
            if state.get("finished"):
                stop_rows.append(hidden.new_tensor([float("-inf"), 0.0]))
                codec_deltas.append(empty_delta)
                terminal_flags.append(torch.tensor(False, dtype=torch.bool))
                if sliding_recompute_enabled:
                    sliding_recompute_flags.append(torch.tensor(False, dtype=torch.bool))
                    sliding_recompute_prompt_lengths.append(torch.tensor(0, dtype=torch.long))
                continue
            if not sample_eligible[index]:
                # vLLM computes a logit row for incomplete chunked prefills but
                # discards its sampled token. Advancing codec/RNG state here
                # would make output depend on prefill chunking and compaction.
                stop_rows.append(hidden.new_tensor([0.0, float("-inf")]))
                codec_deltas.append(empty_delta)
                terminal_flags.append(torch.tensor(False, dtype=torch.bool))
                if sliding_recompute_enabled:
                    sliding_recompute_flags.append(torch.tensor(False, dtype=torch.bool))
                    sliding_recompute_prompt_lengths.append(torch.tensor(0, dtype=torch.long))
                continue
            if state.get("conditioning"):
                if not state.pop("condition_sample_ready", False):
                    info["audio_state"] = state
                    stop_rows.append(hidden.new_tensor([0.0, float("-inf")]))
                    codec_deltas.append(empty_delta)
                    terminal_flags.append(torch.tensor(False, dtype=torch.bool))
                    if sliding_recompute_enabled:
                        sliding_recompute_flags.append(torch.tensor(False, dtype=torch.bool))
                        sliding_recompute_prompt_lengths.append(torch.tensor(0, dtype=torch.long))
                    continue
                state["conditioning"] = False
            codes = state.get("codes")
            if not isinstance(codes, torch.Tensor):
                codes = (info.get("audio_codes", {}) or {}).get("accumulated")
            if not isinstance(codes, torch.Tensor):
                codes = torch.empty(0, dtype=torch.long, device=hidden.device)
            else:
                codes = codes.to(device=hidden.device, dtype=torch.long).reshape(-1)
            step = int(state.get("step", 0))
            sampled = self._sample_audio_code(hidden[end - 1 : end], codes, request_id, step)
            is_eos = int(sampled.item()) == self._num_audio_tokens - 1
            state["step"] = step + 1
            recompute_prompt_len = 0
            if state.get("mode") == "native_duplex":
                reached_limit = int(state["step"]) >= int(state.get("max_tokens", _DUPLEX_CODEC_TOKENS_PER_CHUNK))
                finished = is_eos or reached_limit
                state["finished"] = finished
                if finished:
                    logger.info(
                        "[MiniCPM-o][Stage1][duplex-boundary] request_id=%s reason=%s "
                        "step=%s emitted_codes=%s max_tokens=%s",
                        request_id,
                        "audio_eos" if is_eos else "audio_limit",
                        state["step"],
                        int(state["step"]) - int(is_eos),
                        state.get("max_tokens", _DUPLEX_CODEC_TOKENS_PER_CHUNK),
                    )
                if not is_eos and not reached_limit:
                    codes = torch.cat([codes[-(_REPETITION_WINDOW - 1) :], sampled.reshape(1)])
                    delta = sampled.reshape(1, 1)
                else:
                    delta = empty_delta
            else:
                condition_step = int(state.get("condition_step", 0)) + 1
                state["condition_step"] = condition_step
                reached_limit = condition_step >= _MAX_AUDIO_TOKENS_PER_CONDITION
                chunks = state.get("condition_chunks")
                chunk_index = int(state.get("condition_chunk_index", 0))
                has_more_conditions = isinstance(chunks, list) and chunk_index + 1 < len(chunks)
                condition_finished = is_eos or reached_limit
                finished = condition_finished and not has_more_conditions
                recompute_prompt_len = 0
                recompute_audio_tokens = 0
                kv_action = "append_native_kv"
                if sliding_recompute_enabled and not is_eos:
                    state.setdefault("condition_audio_codes", []).append(int(sampled.item()))
                if condition_finished and has_more_conditions:
                    # Normal chat follows streaming_generate: audio EOS or the
                    # 500-step budget advances to the next text condition.
                    state["condition_chunk_index"] = chunk_index + 1
                    state["condition_cursor"] = 0
                    state["condition_sample_ready"] = False
                    state["condition_step"] = 0
                    if sliding_recompute_enabled:
                        self._record_completed_condition(state, chunk_index)
                        next_condition_index = chunk_index + 1
                        if self._should_recompute_condition(next_condition_index):
                            recompute_prompt_len, recompute_audio_tokens = self._sliding_recompute_prompt_stats(
                                state,
                                next_condition_index,
                            )
                            state["sliding_recompute_pending"] = True
                            state["sliding_recompute_prompt_len"] = recompute_prompt_len
                            state["sliding_recompute_audio_tokens"] = recompute_audio_tokens
                            state["conditioning"] = False
                            kv_action = "sliding_recompute_prompt_replace"
                            logger.info(
                                "[MiniCPM-o][Stage1][sliding-recompute-schedule] request_id=%s "
                                "previous_condition_index=%s next_condition_index=%s "
                                "recomputed_chunks=%s previous_audio_tokens=%s prompt_len=%s reset_offset=0",
                                request_id,
                                chunk_index,
                                next_condition_index,
                                self._sliding_recompute_settings()[2],
                                recompute_audio_tokens,
                                recompute_prompt_len,
                            )
                        else:
                            state["conditioning"] = True
                    else:
                        state["conditioning"] = True
                if condition_finished:
                    logger.info(
                        "[MiniCPM-o][Stage1][condition-boundary] request_id=%s "
                        "condition_index=%s/%s reason=%s condition_steps=%s emitted_codes=%s "
                        "has_more_conditions=%s next_condition_index=%s kv_action=%s",
                        request_id,
                        chunk_index,
                        len(chunks) if isinstance(chunks, list) else 0,
                        "audio_eos" if is_eos else "audio_limit_500",
                        condition_step,
                        condition_step - int(is_eos),
                        has_more_conditions,
                        chunk_index + 1 if has_more_conditions else None,
                        kv_action,
                    )
                state["finished"] = finished
                if not is_eos:
                    codes = torch.cat([codes[-(_REPETITION_WINDOW - 1) :], sampled.reshape(1)])
                    delta = sampled.reshape(1, 1)
                else:
                    delta = empty_delta
            state["codes"] = codes
            info["audio_state"] = state
            info["audio_codes"] = {
                "current": sampled.reshape(1),
                "accumulated": codes,
            }
            codec_deltas.append(delta)
            terminal_flags.append(torch.tensor(finished, dtype=torch.bool))
            if sliding_recompute_enabled:
                sliding_recompute_flags.append(torch.tensor(recompute_prompt_len > 0, dtype=torch.bool))
                sliding_recompute_prompt_lengths.append(torch.tensor(recompute_prompt_len, dtype=torch.long))
            stop_rows.append(hidden.new_tensor([float("-inf"), 0.0] if finished else [0.0, float("-inf")]))

        self._batch_stop_logits = torch.stack(stop_rows, dim=0) if stop_rows else hidden.new_empty((0, 2))
        # Lists are deliberate: the runner routes element i to request i,
        # preserving compaction alignment while emitting only this step's code.
        meta_outputs = {"finished": terminal_flags}
        if sliding_recompute_enabled and any(flag.item() for flag in sliding_recompute_flags):
            meta_outputs.update(
                {
                    "replace_streaming_prompt": sliding_recompute_flags,
                    "next_stage_prompt_len": sliding_recompute_prompt_lengths,
                }
            )
        if emit_duplex_metadata:
            meta_outputs.update(
                {
                    "native_duplex": native_duplex_flags,
                    "duplex_epoch": duplex_epochs,
                    "duplex_turn_id": duplex_turn_ids,
                    "llm_output_text_utf8": segment_texts_utf8,
                    "turn_end": turn_end_flags,
                }
            )
        multimodal_outputs: dict[str, Any] = {
            "codes": {"audio": codec_deltas},
            "meta": meta_outputs,
        }
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs=multimodal_outputs,
        )

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        self._deferred_cleanup_ids.update(str(req_id) for req_id in finished_req_ids)

    def _flush_deferred_cleanup(self) -> None:
        request_audio_states = getattr(self, "_request_audio_states", {})
        for request_id in self._deferred_cleanup_ids:
            logger.info("[MiniCPM-o][Stage1][cleanup] request_id=%s", request_id)
            self._request_generators.pop(request_id, None)
            request_audio_states.pop(request_id, None)
        self._deferred_cleanup_ids.clear()

    def _dummy_hidden_states(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        """Shape-correct zero tensor for vllm KV cache profiling.

        vllm's gpu_model_runner._dummy_run takes forward()'s return value as
        ``hidden_states`` and does ``hidden_states[logit_indices_device]``;
        returning None on the dummy path crashes with
        ``TypeError: 'NoneType' object is not subscriptable``.
        """
        for ref in (input_ids, positions, inputs_embeds):
            if isinstance(ref, torch.Tensor):
                num_tokens = int(ref.shape[0]) if ref.ndim >= 1 else 1
                device = ref.device
                break
        else:
            num_tokens = 1
            device = current_omni_platform.get_torch_device()
        hidden_size = int(getattr(self, "_hidden_size", 768) or 768)
        return torch.zeros((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)

    def forward(
        self,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        self._flush_deferred_cleanup()
        if input_ids is None and inputs_embeds is None:
            return self._dummy_hidden_states(input_ids, positions, inputs_embeds)
        return self.tts_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states, *args, **kwargs):
        if not isinstance(hidden_states, torch.Tensor):
            return None
        if self._batch_stop_logits is None:
            return torch.zeros(
                hidden_states.shape[0],
                2,
                device=hidden_states.device,
                dtype=torch.float32,
            )
        logits = self._batch_stop_logits
        self._batch_stop_logits = None
        return logits

    def sample(self, logits, sampling_metadata):
        return Sampler()(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return self._load_native_weights(weights)

    def _load_native_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        direct_params = dict(self.named_parameters())
        head_g = head_v = None

        for name, tensor in weights:
            if not name.startswith("tts."):
                continue
            stripped = name[len("tts.") :]
            if stripped.startswith("model."):
                backbone_weights.append((stripped[len("model.") :], tensor))
                continue
            if stripped == "head_code.0.parametrizations.weight.original0":
                head_g = tensor
                continue
            if stripped == "head_code.0.parametrizations.weight.original1":
                head_v = tensor
                continue
            target = stripped
            parameter = direct_params.get(target)
            if parameter is None:
                continue
            parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
            loaded.add(target)

        for name in self.tts_model.load_weights(backbone_weights):
            loaded.add(f"tts_model.{name}")

        if head_g is None or head_v is None:
            raise ValueError("MiniCPM-o checkpoint is missing weight-norm Talker head parameters")
        restored = _restore_weight_norm_weight(head_g, head_v)
        self.head_code[0].weight.data.copy_(
            restored.to(
                device=self.head_code[0].weight.device,
                dtype=self.head_code[0].weight.dtype,
            )
        )
        loaded.add("head_code.0.weight")
        return loaded

    def get_input_embeddings(self, input_ids, multimodal_embeddings=None, **kwargs):
        if hasattr(self, "emb_text") and self.emb_text is not None:
            return self.emb_text(input_ids)
        return torch.zeros(input_ids.shape[0], 1)

    def embed_input_ids(self, input_ids, **kwargs):
        return self.get_input_embeddings(input_ids, **kwargs)
