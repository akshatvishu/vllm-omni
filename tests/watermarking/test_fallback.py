# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from unittest.mock import MagicMock

import pytest
import torch

from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.watermarking import AudioSealWatermarker, AudioTensor, AudioWatermarkerBase

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FailingWatermarker(AudioWatermarkerBase[list[int]]):
    """Return unchanged frames so dropped or duplicated samples are visible."""

    supports_stereo = False
    frame_size = 4

    def __init__(self, failed_call: int, error_type: type[Exception] = RuntimeError):
        super().__init__()
        self.failed_call = failed_call
        self.error_type = error_type
        self.calls = 0
        self.created: list[list[int]] = []
        self.closed: list[list[int]] = []

    def _new_audio_state(self, data: AudioTensor) -> list[int]:
        state = [0]
        self.created.append(state)
        return state

    def _watermark_audio(self, data: AudioTensor, state: list[int]) -> AudioTensor:
        self.calls += 1
        state[0] += 1
        if self.calls == self.failed_call:
            raise self.error_type("generator failed")
        return data

    def _is_audio_watermarked(self, data: AudioTensor) -> bool:
        return False

    def _close_audio_state(self, state: list[int]) -> None:
        self.closed.append(state)


@pytest.mark.parametrize("shape", [(19,), (1, 19), (2, 19), (1, 1, 19), (2, 2, 19)])
@pytest.mark.parametrize("failure", ["first", "middle", "final"])
@pytest.mark.parametrize("explicit_final", [False, True])
def test_failure_preserves_all_original_samples_and_layout(shape, failure, explicit_final):
    failed_call = {"first": 1, "middle": 2, "final": 3 if explicit_final else 4}[failure]
    watermarker = _FailingWatermarker(failed_call)
    samples = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape) / 100
    chunks = list(samples.split([3, 6, 5, 5], dim=-1))
    outputs = [
        watermarker.watermark_output("request", chunk, {"sr": 24000}, finished=explicit_final and index == 3)
        for index, chunk in enumerate(chunks)
    ]
    final = watermarker.finish_request_output("request")
    if final is not None:
        outputs.append(final[0])
    torch.testing.assert_close(torch.cat(outputs, dim=-1), samples)
    assert watermarker.calls >= failed_call
    assert not watermarker._request_states
    assert len(watermarker.created) == len(watermarker.closed)
    assert watermarker.finish_request_output("request") is None


@pytest.mark.parametrize("error_type", [RuntimeError, TypeError, ValueError])
def test_failure_discards_broken_state_and_next_chunk_starts_fresh(error_type):
    watermarker = _FailingWatermarker(1, error_type)
    watermarker.watermark_output("request", torch.zeros(3), {"sr": 16000})
    result = watermarker.watermark_output("request", torch.ones(3), {"sr": 16000})
    torch.testing.assert_close(result, torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))
    assert not watermarker._request_states
    assert watermarker.closed[0] is watermarker.created[0]
    result = watermarker.watermark_output("request", torch.ones(4), {"sr": 16000}, finished=True)
    torch.testing.assert_close(result, torch.ones(4))
    assert len(watermarker.created) == 2
    assert watermarker.created[0] is not watermarker.created[1]


@pytest.mark.parametrize("error_type", [AttributeError, KeyError, AssertionError])
def test_unexpected_generator_error_propagates(error_type):
    watermarker = _FailingWatermarker(1, error_type)
    with pytest.raises(error_type, match="generator failed"):
        watermarker.watermark_output("request", torch.ones(4), {"sr": 16000})
    assert not watermarker._request_states


@pytest.mark.asyncio
@pytest.mark.parametrize("payload_free", [False, True])
async def test_final_failure_returns_buffered_samples_through_stage_pool(payload_free):
    pool = object.__new__(StagePool)
    watermarker = _FailingWatermarker(1)
    pool._watermarkers = {"audio": watermarker}
    pool._watermark_lock = asyncio.Lock()
    initial = OmniRequestOutput(
        request_id="request", finished=False, _multimodal_output={"audio": torch.tensor([1.0, 2.0, 3.0]), "sr": 16000}
    )
    await pool.process_diffusion_output(initial)
    assert initial.multimodal_output["audio"].numel() == 0
    final = OmniRequestOutput(request_id="request", finished=True)
    if not payload_free:
        final._multimodal_output = {"audio": torch.tensor([4.0, 5.0]), "sr": 16000}
    result = await pool.process_diffusion_output(final)
    assert result is final
    assert result.error is None
    expected = torch.tensor([1.0, 2.0, 3.0]) if payload_free else torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    torch.testing.assert_close(result.multimodal_output["audio"], expected)
    assert not watermarker._request_states
    assert watermarker.finish_request_output("request") is None


def test_failure_keeps_other_requests_pending_audio():
    watermarker = _FailingWatermarker(1)
    watermarker.watermark_output("healthy", torch.tensor([0.1, 0.2, 0.3]), {"sr": 16000})
    watermarker.watermark_output("failed", torch.ones(4), {"sr": 16000})
    final = watermarker.finish_request_output("healthy")
    assert final is not None
    torch.testing.assert_close(final[0], torch.tensor([0.1, 0.2, 0.3]))
    assert not watermarker._request_states


def test_missing_metadata_with_pending_audio_does_not_claim_successful_fallback(monkeypatch):
    watermarker = _FailingWatermarker(1)
    watermarker.watermark_output("request", torch.ones(3), {"sr": 16000})
    warning = MagicMock()
    monkeypatch.setattr("vllm_omni.watermarking.base.logger.warning", warning)
    with pytest.raises(ValueError, match="Cannot return buffered audio"):
        watermarker.watermark_output("request", torch.ones(3), {})
    warning.assert_not_called()
    assert not watermarker._request_states


def test_final_failure_without_completion_still_fails_request():
    pool = object.__new__(StagePool)
    watermarker = _FailingWatermarker(1)
    pool._watermarkers = {"audio": watermarker}
    watermarker.watermark_output("request", torch.ones(3), {"sr": 16000})
    state = MagicMock(external_req_id="external")
    errors = pool._finish_llm_watermark_outputs({"request": state}, [])
    assert "No completion available" in errors["external"]
    assert not watermarker._request_states


@pytest.mark.local_model
@pytest.mark.slow
@pytest.mark.tts
def test_audioseal_decoder_failure_preserves_audio_and_other_streams(monkeypatch):
    pytest.importorskip("audioseal")
    watermarker = AudioSealWatermarker()
    generator = torch.Generator().manual_seed(42)
    healthy = torch.randn((1, 1, 1007), generator=generator) * 0.05
    failed = torch.randn((1, 1, 643), generator=generator) * 0.05
    recovered = torch.randn((1, 1, 321), generator=generator) * 0.05
    try:
        expected_healthy = watermarker.watermark_output("whole", healthy, {"sr": 16000}, finished=True)
        expected_recovered = watermarker.watermark_output("fresh", recovered, {"sr": 16000}, finished=True)
        assert not torch.equal(expected_healthy, healthy)
        assert not torch.equal(expected_recovered, recovered)
        first_healthy = watermarker.watermark_output("healthy", healthy[..., :333], {"sr": 16000})
        first_failed = watermarker.watermark_output("failed", failed[..., :333], {"sr": 16000})
        assert first_failed.shape[-1] == 320
        assert not torch.equal(first_failed, failed[..., :320])

        decoder_forward = watermarker._model.decoder.forward
        injected = False

        def fail_after_decoder(*args, **kwargs):
            nonlocal injected
            result = decoder_forward(*args, **kwargs)
            if not injected:
                injected = True
                raise RuntimeError("decoder failed after updating streaming history")
            return result

        monkeypatch.setattr(watermarker._model.decoder, "forward", fail_after_decoder)
        fallback = watermarker.watermark_output("failed", failed[..., 333:], {"sr": 16000})
        assert injected
        torch.testing.assert_close(fallback, failed[..., 320:], rtol=0, atol=0)
        assert "failed" not in watermarker._request_states
        assert "healthy" in watermarker._request_states

        last_healthy = watermarker.watermark_output("healthy", healthy[..., 333:], {"sr": 16000}, finished=True)
        torch.testing.assert_close(
            torch.cat((first_healthy, last_healthy), dim=-1), expected_healthy, rtol=1e-5, atol=1e-6
        )
        last_recovered = watermarker.watermark_output("failed", recovered, {"sr": 16000}, finished=True)
        torch.testing.assert_close(last_recovered, expected_recovered, rtol=1e-5, atol=1e-6)
        returned_samples = torch.cat((first_failed, fallback, last_recovered), dim=-1)
        assert returned_samples.shape[-1] == failed.shape[-1] + recovered.shape[-1]
        assert watermarker.finish_request_output("failed") is None
        assert not watermarker._request_states
    finally:
        watermarker.close()
