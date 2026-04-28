# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.ming_tts.config_ming_tts import (
    KEY_REQUEST_ID,
    LATENT_CHUNK_SIZE,
    LATENT_LEFT_CONTEXT,
    PATCH_SIZE,
)
from vllm_omni.model_executor.stage_input_processors.ming_tts import (
    MING_EMIT_PATCH_COUNT_KEY,
    MING_ESTIMATED_BYTES_KEY,
    MING_FINAL_DECODE_STEP_KEY,
    MING_FINAL_FLUSH_KEY,
    MING_LATENT_SHAPE_KEY,
    MING_STOP_REASON_KEY,
    _extract_last_patch,
    llm2audio_vae,
    llm2audio_vae_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_LATENT_D = 64


def _req(external_req_id: str, *, finished: bool):
    return SimpleNamespace(
        external_req_id=external_req_id,
        is_finished=lambda: finished,
    )


def _manager(*, chunk_size: int | None = 2, left_context: int | None = 0, raw_config=None):
    if raw_config is None:
        extra = {}
        if chunk_size is not None:
            extra["latent_chunk_size"] = chunk_size
        if left_context is not None:
            extra["latent_left_context"] = left_context
        raw_config = {"extra": extra}
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        put_req_chunk=defaultdict(int),
        request_payload={},
        connector=SimpleNamespace(config=raw_config),
    )


def _patch(fill: float) -> torch.Tensor:
    return torch.full((PATCH_SIZE, _LATENT_D), fill, dtype=torch.float32)


def _payload(fill: float, *, has_patch=True, decode_step=None, stop_reason=None) -> dict[str, object]:
    payload = {
        "ming_has_patch": torch.tensor([has_patch]),
        "ming_latent_patch": _patch(fill).unsqueeze(0),
    }
    if decode_step is not None:
        payload["ming_decode_step"] = torch.tensor([decode_step], dtype=torch.int32)
    if stop_reason is not None:
        payload[MING_STOP_REASON_KEY] = (stop_reason,)
    return payload


def test_extract_last_patch_uses_active_mask():
    patch = torch.arange(3 * PATCH_SIZE * _LATENT_D, dtype=torch.float16).reshape(3, PATCH_SIZE, _LATENT_D)
    payload = {
        "ming_has_patch": torch.tensor([False, True, False]),
        "ming_latent_patch": patch,
    }

    out = _extract_last_patch(payload)

    assert out is not None
    assert out.shape == (PATCH_SIZE, _LATENT_D)
    assert out.dtype == torch.float32
    assert out.device.type == "cpu"
    assert torch.allclose(out, patch[1].to(torch.float32).cpu())


def test_llm2audio_vae_async_chunk_waits_for_full_chunk():
    transfer_manager = _manager(chunk_size=2)
    request = _req("rid-wait", finished=False)

    payload = llm2audio_vae_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=_payload(1.0),
        request=request,
    )

    assert payload is None
    assert len(transfer_manager.code_prompt_token_ids["rid-wait"]) == 1


def test_llm2audio_vae_async_chunk_partial_chunk_does_not_emit():
    transfer_manager = _manager(chunk_size=3)
    request = _req("rid-partial", finished=False)

    first = llm2audio_vae_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=_payload(1.0),
        request=request,
    )
    second = llm2audio_vae_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=_payload(2.0),
        request=request,
    )

    assert first is None
    assert second is None
    assert len(transfer_manager.code_prompt_token_ids["rid-partial"]) == 2


def test_llm2audio_vae_async_chunk_emits_full_chunk():
    transfer_manager = _manager(chunk_size=2)
    request_id = "rid-full"
    request = _req(request_id, finished=False)
    transfer_manager.code_prompt_token_ids[request_id].append(_patch(1.0))

    payload = llm2audio_vae_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=_payload(2.0),
        request=request,
    )

    assert payload is not None
    assert payload["codes"]["audio"] == [0]
    assert payload["meta"]["finished"].item() is False
    assert payload["finished"].item() is False
    assert payload["stream_finished"].item() is False
    assert payload[KEY_REQUEST_ID] == request_id
    assert payload["code_predictor_codes"] == [0]
    assert payload["ming_latent_patches"].shape == (2, PATCH_SIZE, _LATENT_D)
    assert payload[MING_EMIT_PATCH_COUNT_KEY] == 2
    assert payload[MING_LATENT_SHAPE_KEY] == (2, PATCH_SIZE, _LATENT_D)
    assert payload[MING_ESTIMATED_BYTES_KEY] == int(
        payload["ming_latent_patches"].numel() * payload["ming_latent_patches"].element_size()
    )
    assert payload[MING_ESTIMATED_BYTES_KEY] > 0
    assert payload[MING_FINAL_FLUSH_KEY] is False
    assert torch.allclose(payload["ming_latent_patches"][0], _patch(1.0))
    assert torch.allclose(payload["ming_latent_patches"][1], _patch(2.0))
    assert transfer_manager.request_payload[request_id]["_ming_async_state"]["seen_patch_len"] == 2


def test_llm2audio_vae_async_chunk_multi_request_interleaving_has_no_state_bleed():
    transfer_manager = _manager(chunk_size=2)
    req_a = _req("rid-a", finished=False)
    req_b = _req("rid-b", finished=False)

    assert (
        llm2audio_vae_async_chunk(transfer_manager=transfer_manager, pooling_output=_payload(1.0), request=req_a)
        is None
    )
    assert (
        llm2audio_vae_async_chunk(transfer_manager=transfer_manager, pooling_output=_payload(10.0), request=req_b)
        is None
    )

    payload_a = llm2audio_vae_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=_payload(2.0),
        request=req_a,
    )
    assert payload_a is not None
    assert payload_a[KEY_REQUEST_ID] == "rid-a"
    assert torch.allclose(payload_a["ming_latent_patches"][0], _patch(1.0))
    assert torch.allclose(payload_a["ming_latent_patches"][1], _patch(2.0))

    assert len(transfer_manager.code_prompt_token_ids["rid-b"]) == 1

    payload_b = llm2audio_vae_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=_payload(20.0),
        request=req_b,
    )
    assert payload_b is not None
    assert payload_b[KEY_REQUEST_ID] == "rid-b"
    assert torch.allclose(payload_b["ming_latent_patches"][0], _patch(10.0))
    assert torch.allclose(payload_b["ming_latent_patches"][1], _patch(20.0))

    assert transfer_manager.request_payload["rid-a"]["_ming_async_state"]["seen_patch_len"] == 2
    assert transfer_manager.request_payload["rid-b"]["_ming_async_state"]["seen_patch_len"] == 2


def test_llm2audio_vae_async_chunk_finish_after_full_chunk_only_emits_eof():
    transfer_manager = _manager(chunk_size=2)
    request_id = "rid-drain"
    request = _req(request_id, finished=False)
    transfer_manager.code_prompt_token_ids[request_id].append(_patch(1.0))

    payload = llm2audio_vae_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=_payload(2.0),
        request=request,
    )

    assert payload is not None
    assert transfer_manager.request_payload[request_id]["_ming_async_state"]["seen_patch_len"] == 2

    finish_payload = llm2audio_vae_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=_req(request_id, finished=True),
    )

    assert finish_payload == {
        "codes": {"audio": []},
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
        "code_predictor_codes": [],
        "finished": torch.tensor(True, dtype=torch.bool),
        "stream_finished": torch.tensor(True, dtype=torch.bool),
        "ming_chunk_id": 0,
        KEY_REQUEST_ID: request_id,
        MING_EMIT_PATCH_COUNT_KEY: 0,
        MING_LATENT_SHAPE_KEY: None,
        MING_ESTIMATED_BYTES_KEY: 0,
        MING_FINAL_FLUSH_KEY: True,
    }


def test_llm2audio_vae_async_chunk_flushes_tail_on_finish_without_new_patch():
    transfer_manager = _manager(chunk_size=3)
    request_id = "rid-tail"
    request = _req(request_id, finished=True)
    transfer_manager.code_prompt_token_ids[request_id] = [
        _patch(1.0),
        _patch(2.0),
    ]

    payload = llm2audio_vae_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
    )

    assert payload is not None
    assert payload["codes"]["audio"] == [0]
    assert payload["meta"]["finished"].item() is True
    assert payload["finished"].item() is True
    assert payload["stream_finished"].item() is True
    assert payload[KEY_REQUEST_ID] == request_id
    assert payload["ming_latent_patches"].shape == (2, PATCH_SIZE, _LATENT_D)
    assert payload[MING_EMIT_PATCH_COUNT_KEY] == 2
    assert payload[MING_LATENT_SHAPE_KEY] == (2, PATCH_SIZE, _LATENT_D)
    assert payload[MING_ESTIMATED_BYTES_KEY] > 0
    assert payload[MING_FINAL_FLUSH_KEY] is True
    assert torch.allclose(payload["ming_latent_patches"][0], _patch(1.0))
    assert torch.allclose(payload["ming_latent_patches"][1], _patch(2.0))


def test_llm2audio_vae_async_chunk_final_flush_emits_partial_chunk_with_new_patch():
    transfer_manager = _manager(chunk_size=3)
    request_id = "rid-tail-new"

    transfer_manager.code_prompt_token_ids[request_id].append(_patch(1.0))
    payload = llm2audio_vae_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=_payload(2.0, decode_step=7, stop_reason="stop_head"),
        request=_req(request_id, finished=True),
    )

    assert payload is not None
    assert payload["codes"]["audio"] == [0]
    assert payload["meta"]["finished"].item() is True
    assert payload["finished"].item() is True
    assert payload["stream_finished"].item() is True
    assert payload[MING_EMIT_PATCH_COUNT_KEY] == 2
    assert payload[MING_FINAL_FLUSH_KEY] is True
    assert payload[MING_FINAL_DECODE_STEP_KEY] == 7
    assert payload[MING_STOP_REASON_KEY] == "stop_head"
    assert torch.allclose(payload["ming_latent_patches"][0], _patch(1.0))
    assert torch.allclose(payload["ming_latent_patches"][1], _patch(2.0))


def test_llm2audio_vae_async_chunk_emits_eof_when_finished_without_frames():
    transfer_manager = _manager(chunk_size=2)
    request = _req("rid-eof", finished=True)

    payload = llm2audio_vae_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
    )

    assert payload == {
        "codes": {"audio": []},
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
        "code_predictor_codes": [],
        "finished": torch.tensor(True, dtype=torch.bool),
        "stream_finished": torch.tensor(True, dtype=torch.bool),
        "ming_chunk_id": 0,
        KEY_REQUEST_ID: "rid-eof",
        MING_EMIT_PATCH_COUNT_KEY: 0,
        MING_LATENT_SHAPE_KEY: None,
        MING_ESTIMATED_BYTES_KEY: 0,
        MING_FINAL_FLUSH_KEY: True,
    }


def test_llm2audio_vae_async_chunk_zero_latent_final_flush_returns_empty_payload_not_error():
    transfer_manager = _manager(chunk_size=2)

    payload = llm2audio_vae_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output={
            "ming_has_patch": torch.tensor([False]),
            "ming_latent_patch": torch.zeros((1, PATCH_SIZE, _LATENT_D), dtype=torch.float32),
        },
        request=_req("rid-zero-final", finished=True),
    )

    assert payload == {
        "codes": {"audio": []},
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
        "code_predictor_codes": [],
        "finished": torch.tensor(True, dtype=torch.bool),
        "stream_finished": torch.tensor(True, dtype=torch.bool),
        "ming_chunk_id": 0,
        KEY_REQUEST_ID: "rid-zero-final",
        MING_EMIT_PATCH_COUNT_KEY: 0,
        MING_LATENT_SHAPE_KEY: None,
        MING_ESTIMATED_BYTES_KEY: 0,
        MING_FINAL_FLUSH_KEY: True,
    }


def test_llm2audio_vae_async_chunk_rejects_left_context_config():
    transfer_manager = _manager(chunk_size=2, left_context=1)
    request = _req("rid-bad-cfg", finished=False)

    with pytest.raises(
        ValueError,
        match="does not support latent_left_context replay.*Got latent_left_context=1",
    ):
        llm2audio_vae_async_chunk(
            transfer_manager=transfer_manager,
            pooling_output=_payload(1.0),
            request=request,
        )


def test_llm2audio_vae_async_chunk_rejects_non_positive_chunk_size():
    transfer_manager = _manager(chunk_size=0, left_context=0)

    with pytest.raises(ValueError, match="Invalid Ming latent_chunk_size=0"):
        llm2audio_vae_async_chunk(
            transfer_manager=transfer_manager,
            pooling_output=_payload(1.0),
            request=_req("rid-bad-chunk", finished=False),
        )


def test_llm2audio_vae_async_chunk_missing_config_uses_fallback_defaults():
    transfer_manager = _manager(raw_config={"extra": {}})
    request_id = "rid-fallback"

    for idx in range(LATENT_CHUNK_SIZE - 1):
        payload = llm2audio_vae_async_chunk(
            transfer_manager=transfer_manager,
            pooling_output=_payload(float(idx + 1)),
            request=_req(request_id, finished=False),
        )
        assert payload is None

    payload = llm2audio_vae_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=_payload(float(LATENT_CHUNK_SIZE)),
        request=_req(request_id, finished=False),
    )

    assert payload is not None
    assert payload[MING_EMIT_PATCH_COUNT_KEY] == LATENT_CHUNK_SIZE
    assert payload[MING_LATENT_SHAPE_KEY] == (LATENT_CHUNK_SIZE, PATCH_SIZE, _LATENT_D)
    assert LATENT_LEFT_CONTEXT == 0


def test_llm2audio_vae_builds_generation_prompt_from_stage_output():
    patches = torch.arange(2 * PATCH_SIZE * _LATENT_D, dtype=torch.float32).reshape(2, PATCH_SIZE, _LATENT_D)
    stage_output = SimpleNamespace(
        request_id="rid-stage",
        finished=True,
        outputs=[
            SimpleNamespace(
                multimodal_output={
                    "ming_has_patch": torch.tensor([True, True]),
                    "ming_latent_patch": patches,
                    "ming_decode_step": torch.tensor([26, 27], dtype=torch.int32),
                    "ming_stop_reason": ("continue", "stop_head"),
                }
            )
        ],
    )
    stage = SimpleNamespace(engine_outputs=[stage_output])

    prompts = llm2audio_vae(stage_list=[stage], engine_input_source=[0])

    assert len(prompts) == 1
    info = prompts[0]["additional_information"]
    assert info[KEY_REQUEST_ID] == "rid-stage"
    assert info["finished"].item() is True
    assert info["ming_latent_patches"].shape == (2, PATCH_SIZE, _LATENT_D)
    assert torch.allclose(info["ming_latent_patches"], patches)
    assert info[MING_FINAL_DECODE_STEP_KEY] == 27
    assert info[MING_STOP_REASON_KEY] == "stop_head"


def test_llm2audio_vae_skips_unfinished_stage_output():
    patch = torch.arange(PATCH_SIZE * _LATENT_D, dtype=torch.float32).reshape(1, PATCH_SIZE, _LATENT_D)
    stage_output = SimpleNamespace(
        request_id="rid-unfinished",
        finished=False,
        outputs=[
            SimpleNamespace(
                multimodal_output={
                    "ming_has_patch": torch.tensor([True]),
                    "ming_latent_patch": patch,
                }
            )
        ],
    )
    stage = SimpleNamespace(engine_outputs=[stage_output])

    prompts = llm2audio_vae(stage_list=[stage], engine_input_source=[0])

    assert prompts == []
