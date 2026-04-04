from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.ming_tts.config_ming_tts import KEY_REQUEST_ID
from vllm_omni.model_executor.stage_input_processors.ming_tts import (
    MING_EMIT_PATCH_COUNT_KEY,
    MING_ESTIMATED_BYTES_KEY,
    MING_FINAL_FLUSH_KEY,
    MING_LATENT_SHAPE_KEY,
    _extract_last_patch,
    llm2audio_vae,
    llm2audio_vae_async_chunk,
)


def _req(external_req_id: str, *, finished: bool):
    return SimpleNamespace(
        external_req_id=external_req_id,
        is_finished=lambda: finished,
    )


def _manager(*, chunk_size: int = 2, left_context: int = 0):
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        put_req_chunk=defaultdict(int),
        connector=SimpleNamespace(
            config={"extra": {"latent_chunk_size": chunk_size, "latent_left_context": left_context}}
        ),
    )


def test_extract_last_patch_uses_active_mask():
    patch = torch.arange(3 * 4 * 64, dtype=torch.float16).reshape(3, 4, 64)
    payload = {
        "ming_has_patch": torch.tensor([False, True, False]),
        "ming_latent_patch": patch,
    }

    out = _extract_last_patch(payload)

    assert out is not None
    assert out.shape == (4, 64)
    assert out.dtype == torch.float32
    assert out.device.type == "cpu"
    assert torch.allclose(out, patch[1].to(torch.float32).cpu())


def test_llm2audio_vae_async_chunk_waits_for_full_chunk():
    transfer_manager = _manager(chunk_size=2)
    request = _req("rid-wait", finished=False)

    payload = llm2audio_vae_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output={
            "ming_has_patch": torch.tensor([True]),
            "ming_latent_patch": torch.ones((1, 4, 64), dtype=torch.float32),
        },
        request=request,
    )

    assert payload is None
    assert len(transfer_manager.code_prompt_token_ids["rid-wait"]) == 1


def test_llm2audio_vae_async_chunk_emits_full_chunk():
    transfer_manager = _manager(chunk_size=2)
    request_id = "rid-full"
    request = _req(request_id, finished=False)
    transfer_manager.code_prompt_token_ids[request_id].append(torch.ones((4, 64), dtype=torch.float32))

    payload = llm2audio_vae_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output={
            "ming_has_patch": torch.tensor([True]),
            "ming_latent_patch": torch.full((1, 4, 64), 2.0, dtype=torch.float32),
        },
        request=request,
    )

    assert payload is not None
    assert payload["finished"].item() is False
    assert payload[KEY_REQUEST_ID] == request_id
    assert payload["code_predictor_codes"] == [0]
    assert payload["ming_latent_patches"].shape == (2, 4, 64)
    assert payload[MING_EMIT_PATCH_COUNT_KEY] == 2
    assert payload[MING_LATENT_SHAPE_KEY] == (2, 4, 64)
    assert payload[MING_ESTIMATED_BYTES_KEY] == int(payload["ming_latent_patches"].numel() * payload["ming_latent_patches"].element_size())
    assert payload[MING_ESTIMATED_BYTES_KEY] > 0
    assert payload[MING_FINAL_FLUSH_KEY] is False
    assert torch.allclose(payload["ming_latent_patches"][0], torch.ones((4, 64), dtype=torch.float32))
    assert torch.allclose(payload["ming_latent_patches"][1], torch.full((4, 64), 2.0, dtype=torch.float32))
    assert transfer_manager.code_prompt_token_ids[request_id] == []


def test_llm2audio_vae_async_chunk_finish_after_full_chunk_only_emits_eof():
    transfer_manager = _manager(chunk_size=2)
    request_id = "rid-drain"
    request = _req(request_id, finished=False)
    transfer_manager.code_prompt_token_ids[request_id].append(torch.ones((4, 64), dtype=torch.float32))

    payload = llm2audio_vae_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output={
            "ming_has_patch": torch.tensor([True]),
            "ming_latent_patch": torch.full((1, 4, 64), 2.0, dtype=torch.float32),
        },
        request=request,
    )

    assert payload is not None
    assert transfer_manager.code_prompt_token_ids[request_id] == []

    finish_payload = llm2audio_vae_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=_req(request_id, finished=True),
    )

    assert finish_payload == {
        "code_predictor_codes": [],
        "finished": torch.tensor(True, dtype=torch.bool),
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
        torch.full((4, 64), 1.0, dtype=torch.float32),
        torch.full((4, 64), 2.0, dtype=torch.float32),
    ]

    payload = llm2audio_vae_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
    )

    assert payload is not None
    assert payload["finished"].item() is True
    assert payload[KEY_REQUEST_ID] == request_id
    assert payload["ming_latent_patches"].shape == (2, 4, 64)
    assert payload[MING_EMIT_PATCH_COUNT_KEY] == 2
    assert payload[MING_LATENT_SHAPE_KEY] == (2, 4, 64)
    assert payload[MING_ESTIMATED_BYTES_KEY] > 0
    assert payload[MING_FINAL_FLUSH_KEY] is True
    assert torch.allclose(payload["ming_latent_patches"][0], torch.full((4, 64), 1.0))
    assert torch.allclose(payload["ming_latent_patches"][1], torch.full((4, 64), 2.0))


def test_llm2audio_vae_async_chunk_emits_eof_when_finished_without_frames():
    transfer_manager = _manager(chunk_size=2)
    request = _req("rid-eof", finished=True)

    payload = llm2audio_vae_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
    )

    assert payload == {
        "code_predictor_codes": [],
        "finished": torch.tensor(True, dtype=torch.bool),
        "ming_chunk_id": 0,
        KEY_REQUEST_ID: "rid-eof",
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
            pooling_output={
                "ming_has_patch": torch.tensor([True]),
                "ming_latent_patch": torch.ones((1, 4, 64), dtype=torch.float32),
            },
            request=request,
        )


def test_llm2audio_vae_builds_generation_prompt_from_stage_output():
    patch = torch.arange(4 * 64, dtype=torch.float32).reshape(4, 64)
    stage_output = SimpleNamespace(
        request_id="rid-stage",
        outputs=[
            SimpleNamespace(
                multimodal_output={
                    "ming_has_patch": torch.tensor([True]),
                    "ming_latent_patch": patch.unsqueeze(0),
                }
            )
        ],
    )
    stage = SimpleNamespace(engine_outputs=[stage_output])

    prompts = llm2audio_vae(stage_list=[stage], engine_input_source=[0])

    assert len(prompts) == 1
    info = prompts[0]["additional_information"]
    assert info[KEY_REQUEST_ID] == "rid-stage"
    assert info["finished"].item() is False
    assert info["ming_latent_patches"].shape == (1, 4, 64)
    assert torch.allclose(info["ming_latent_patches"][0], patch)
