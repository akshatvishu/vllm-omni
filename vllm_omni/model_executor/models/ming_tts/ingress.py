# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import copy
import os
import time
from typing import Any

from vllm.logger import init_logger

from .config_ming_tts import (
    AUDIO_DUMMY_TOKEN_ID,
    AUDIO_START_TOKEN_ID,
    KEY_PROMPT_LATENTS,
    KEY_SPEAKER_EMBEDDING,
    KEY_TEXT_MODE,
    MingTTSConfig,
)
from .prompt_utils import (
    build_dense_prompt_token_ids,
    coerce_speaker_embeddings,
    count_prompt_waveform_patches,
    create_instruction,
)

logger = init_logger(__name__)


def _rebuild_prompt_token_ids_with_exact_patch_count(prompt_token_ids: Any, prompt_patch_count: int) -> list[int]:
    if not isinstance(prompt_token_ids, list) or not prompt_token_ids:
        raise ValueError("Ming prompt finalization requires existing prompt_token_ids")

    audio_start_index = -1
    for idx in range(len(prompt_token_ids) - 1, -1, -1):
        if int(prompt_token_ids[idx]) == AUDIO_START_TOKEN_ID:
            audio_start_index = idx
            break
    if audio_start_index < 0:
        raise ValueError("Ming prompt finalization could not locate <audio> token")

    trailing_tokens = prompt_token_ids[audio_start_index + 1 :]
    if any(int(token_id) != AUDIO_DUMMY_TOKEN_ID for token_id in trailing_tokens):
        raise ValueError("Ming prompt finalization expected only trailing <audioPatch> tokens after <audio>")

    return prompt_token_ids[: audio_start_index + 1] + ([AUDIO_DUMMY_TOKEN_ID] * int(prompt_patch_count))


class MingIngressProcessor:
    def __init__(self, *, vllm_config: Any, tokenizer: Any):
        if tokenizer is None:
            raise RuntimeError("Ming ingress processor requires an initialized tokenizer")

        self.tokenizer = tokenizer
        self.profile_ingress = (
            os.environ.get("MING_TTS_INGRESS_PROFILE") == "1" or os.environ.get("MING_TTS_ASYNC_DEBUG") == "1"
        )

        self.ming_config = MingTTSConfig.from_hf_config(vllm_config.model_config.hf_config)
        self.ming_config.validate()

    def __call__(self, prompt: Any) -> Any:
        total_start = time.perf_counter()
        if not isinstance(prompt, dict):
            return prompt

        raw_additional_information = prompt.get("additional_information")
        if raw_additional_information is None:
            additional_information = {}
        elif isinstance(raw_additional_information, dict):
            additional_information = raw_additional_information
        else:
            return prompt

        modalities = prompt.get("modalities")
        text_mode = isinstance(modalities, (list, tuple)) and ("text" in modalities) and ("audio" not in modalities)
        if text_mode:
            finalized_prompt = copy.copy(prompt)
            finalized_additional_information = dict(additional_information)
            finalized_additional_information[KEY_TEXT_MODE] = True
            prompt_token_ids = finalized_prompt.get("prompt_token_ids")
            if isinstance(prompt_token_ids, list) and prompt_token_ids:
                if int(prompt_token_ids[-1]) == AUDIO_START_TOKEN_ID:
                    finalized_prompt["prompt_token_ids"] = prompt_token_ids[:-1]
            finalized_prompt["additional_information"] = finalized_additional_information
            return finalized_prompt

        prompt_waveform = additional_information.get("prompt_waveform", prompt.get("prompt_waveform"))
        prompt_text = additional_information.get("prompt_text", prompt.get("prompt_text"))
        if prompt_waveform is None:
            return prompt
        if prompt_text is None:
            raise RuntimeError(
                "Ming prompt_waveform requires prompt_text before ingress can build prompt latents. "
                "Use ming_speaker_embedding for reference-audio-only speaker conditioning."
            )

        prompt_latents = additional_information.get(KEY_PROMPT_LATENTS, prompt.get("prompt_latents"))
        if prompt_latents is not None:
            raise ValueError(
                "Ming waveform cloning request provided both raw prompt_waveform and explicit prompt_latents. "
                "Choose exactly one source of truth."
            )

        patch_start = time.perf_counter()
        prompt_patch_count = count_prompt_waveform_patches(
            prompt_waveform,
            patch_size=self.ming_config.patch_size,
            frame_hop=self.ming_config.audio_frame_hop,
            vae_patch_size=self.ming_config.vae_patch_size,
        )
        patch_ms = (time.perf_counter() - patch_start) * 1000.0

        finalized_prompt = copy.copy(prompt)
        finalized_additional_information = dict(additional_information)
        finalized_prompt["additional_information"] = finalized_additional_information

        prompt_prefix = finalized_prompt.get("prompt")
        text = finalized_prompt.get("text")
        token_start = time.perf_counter()
        if isinstance(prompt_prefix, str) and isinstance(text, str):
            speaker_embedding = finalized_prompt.get("speaker_embedding")
            if speaker_embedding is None:
                speaker_embedding = finalized_additional_information.get(KEY_SPEAKER_EMBEDDING)
            speaker_embeddings = coerce_speaker_embeddings(
                speaker_embedding,
                use_zero_spk_emb=bool(finalized_additional_information.get("use_zero_spk_emb", False)),
            )

            instruction = finalized_prompt.get("instruction")
            if instruction is None:
                instruction = finalized_additional_information.get("instruction")
            instruction_text = instruction if isinstance(instruction, str) else create_instruction(instruction)

            finalized_prompt["prompt_token_ids"] = build_dense_prompt_token_ids(
                self.tokenizer,
                prompt=prompt_prefix,
                text=text,
                instruction=instruction_text,
                prompt_text=prompt_text,
                speaker_count=0 if speaker_embeddings is None else len(speaker_embeddings),
                prompt_patch_count=prompt_patch_count,
            )
            if self.profile_ingress:
                elapsed_ms = (time.perf_counter() - total_start) * 1000.0
                token_ms = (time.perf_counter() - token_start) * 1000.0
                logger.info(
                    "MING_INGRESS_PROFILE finalize_prompt prompt_patch_count=%d speaker_count=%d "
                    "patch_ms=%.3f token_rebuild_ms=%.3f elapsed_ms=%.3f",
                    prompt_patch_count,
                    0 if speaker_embeddings is None else len(speaker_embeddings),
                    patch_ms,
                    token_ms,
                    elapsed_ms,
                )
            return finalized_prompt

        finalized_prompt["prompt_token_ids"] = _rebuild_prompt_token_ids_with_exact_patch_count(
            finalized_prompt.get("prompt_token_ids"),
            prompt_patch_count,
        )
        if self.profile_ingress:
            elapsed_ms = (time.perf_counter() - total_start) * 1000.0
            token_ms = (time.perf_counter() - token_start) * 1000.0
            logger.info(
                "MING_INGRESS_PROFILE finalize_prompt prompt_patch_count=%d speaker_count=unknown "
                "patch_ms=%.3f token_rebuild_ms=%.3f elapsed_ms=%.3f",
                prompt_patch_count,
                patch_ms,
                token_ms,
                elapsed_ms,
            )
        return finalized_prompt
