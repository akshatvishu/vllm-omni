from __future__ import annotations

from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.ming_tts.config_ming_tts import (
    AUDIO_DUMMY_TOKEN_ID,
    AUDIO_START_TOKEN_ID,
    KEY_CHUNK_ID,
    KEY_REQUEST_ID,
    LATENT_DIM,
    LATENT_CHUNK_SIZE,
    LATENT_LEFT_CONTEXT,
    PATCH_SIZE,
)

logger = init_logger(__name__)

MING_EMIT_PATCH_COUNT_KEY = "ming_emit_patch_count"
MING_LATENT_SHAPE_KEY = "ming_latent_shape"
MING_ESTIMATED_BYTES_KEY = "ming_estimated_bytes"
MING_FINAL_FLUSH_KEY = "ming_final_flush"
MING_STOP_REASON_KEY = "ming_stop_reason"
MING_FINAL_DECODE_STEP_KEY = "ming_final_decode_step"


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


def _extract_last_patch(pooling_output: dict[str, Any] | None) -> torch.Tensor | None:
    if not isinstance(pooling_output, dict):
        return None
    has_patch = pooling_output.get("ming_has_patch")
    patch = pooling_output.get("ming_latent_patch")
    if not isinstance(patch, torch.Tensor) or patch.numel() == 0:
        return None

    if isinstance(has_patch, torch.Tensor) and has_patch.numel() > 0:
        active = (has_patch.reshape(-1) > 0).nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            return None
        patch = patch[int(active[-1].item())]
    elif patch.ndim == 3:
        patch = patch[-1]

    if patch.ndim != 2:
        raise ValueError(f"Invalid Ming latent patch shape: {tuple(patch.shape)}")
    return patch.to(torch.float32).cpu()


def _extract_last_value(pooling_output: dict[str, Any] | None, key: str) -> Any:
    if not isinstance(pooling_output, dict):
        return None
    value = pooling_output.get(key)
    if value is None:
        return None

    has_patch = pooling_output.get("ming_has_patch")
    selected_index = -1
    if isinstance(has_patch, torch.Tensor) and has_patch.numel() > 0:
        active = (has_patch.reshape(-1) > 0).nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            return None
        selected_index = int(active[-1].item())

    if isinstance(value, torch.Tensor):
        flat = value.reshape(-1)
        if flat.numel() == 0:
            return None
        return flat[min(selected_index, flat.numel() - 1)].item()
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        if selected_index < 0:
            return value[-1]
        return value[min(selected_index, len(value) - 1)]
    return value


def _get_async_chunk_config(transfer_manager: Any) -> tuple[int, int]:
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}

    chunk_size = int(cfg.get("latent_chunk_size", LATENT_CHUNK_SIZE))
    left_context = int(cfg.get("latent_left_context", LATENT_LEFT_CONTEXT))
    if chunk_size <= 0:
        raise ValueError(f"Invalid Ming latent_chunk_size={chunk_size}")
    if left_context != 0:
        raise ValueError(
            f"Ming async chunk transport does not support latent_left_context replay. Expected 0, got {left_context}."
        )
    return chunk_size, left_context


def _build_chunk_observability(
    latent_patches: torch.Tensor | None,
    *,
    final_flush: bool,
) -> dict[str, Any]:
    if latent_patches is None:
        emit_patch_count = 0
        latent_shape = None
        estimated_bytes = 0
    else:
        emit_patch_count = int(latent_patches.shape[0])
        latent_shape = tuple(latent_patches.shape)
        estimated_bytes = int(latent_patches.numel() * latent_patches.element_size())
    return {
        MING_EMIT_PATCH_COUNT_KEY: emit_patch_count,
        MING_LATENT_SHAPE_KEY: latent_shape,
        MING_ESTIMATED_BYTES_KEY: estimated_bytes,
        MING_FINAL_FLUSH_KEY: bool(final_flush),
    }


def llm2audio_vae_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    request_id = request.external_req_id
    chunk_id = int(transfer_manager.put_req_chunk[request_id])
    finished = bool(is_finished or request.is_finished())
    final_decode_step = _extract_last_value(pooling_output, "ming_decode_step")
    stop_reason = _extract_last_value(pooling_output, MING_STOP_REASON_KEY)

    patch = _extract_last_patch(pooling_output)
    if patch is not None:
        transfer_manager.code_prompt_token_ids[request_id].append(patch)

    chunk_size, _ = _get_async_chunk_config(transfer_manager)

    patches = transfer_manager.code_prompt_token_ids[request_id]
    length = len(patches)
    if length <= 0:
        if finished:
            observability = _build_chunk_observability(None, final_flush=True)
            payload = {
                "code_predictor_codes": [],
                "finished": torch.tensor(True, dtype=torch.bool),
                KEY_CHUNK_ID: chunk_id,
                KEY_REQUEST_ID: request_id,
                **observability,
            }
            if final_decode_step is not None:
                payload[MING_FINAL_DECODE_STEP_KEY] = int(final_decode_step)
            if stop_reason is not None:
                payload[MING_STOP_REASON_KEY] = stop_reason
            logger.info(
                "MING_CHUNK_EMIT %s",
                {
                    "request_id": request_id,
                    "chunk_id": chunk_id,
                    "finished": finished,
                    "buffered_patches": length,
                    "remaining_patches": 0,
                    **observability,
                },
            )
            return payload
        return None

    chunk_length = length % chunk_size
    if chunk_length != 0 and not finished:
        return None

    emit_count = chunk_length if chunk_length != 0 else chunk_size
    emit_patches = list(patches[:emit_count])
    del patches[:emit_count]
    latent_patches = torch.stack(emit_patches, dim=0)
    observability = _build_chunk_observability(latent_patches, final_flush=finished)

    payload = {
        "code_predictor_codes": [0],
        "ming_latent_patches": latent_patches,
        "finished": torch.tensor(finished, dtype=torch.bool),
        KEY_CHUNK_ID: chunk_id,
        KEY_REQUEST_ID: request_id,
        **observability,
    }
    if final_decode_step is not None:
        payload[MING_FINAL_DECODE_STEP_KEY] = int(final_decode_step)
    if stop_reason is not None:
        payload[MING_STOP_REASON_KEY] = stop_reason
    log_fn = logger.info if finished else logger.debug
    log_fn(
        "MING_CHUNK_EMIT %s",
        {
            "request_id": request_id,
            "chunk_id": chunk_id,
            "finished": finished,
            "buffered_patches": length,
            "remaining_patches": len(patches),
            **observability,
        },
    )
    return payload


def llm2audio_vae(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    del prompt, requires_multimodal_data
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")
    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    outputs = []
    for stage_output in stage_list[source_stage_id].engine_outputs:
        output = stage_output.outputs[0]
        patch = _extract_last_patch(output.multimodal_output)
        additional_information = {
            "ming_latent_patches": (
                patch.unsqueeze(0)
                if patch is not None
                else torch.zeros((0, PATCH_SIZE, LATENT_DIM), dtype=torch.float32)
            ),
            KEY_REQUEST_ID: getattr(stage_output, "request_id", None),
            "finished": torch.tensor(False, dtype=torch.bool),
        }
        outputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=additional_information,
            )
        )
    return outputs
