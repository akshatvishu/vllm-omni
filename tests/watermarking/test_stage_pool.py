# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.outputs import RequestOutput
from vllm.v1.engine import FinishReason

from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.outputs.mm_outputs import MultimodalCompletionOutput, MultimodalPayload
from vllm_omni.outputs.output_modality import TensorAccumulationStrategy
from vllm_omni.watermarking import AudioTensor, AudioWatermarkerBase

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _TestWatermarker(AudioWatermarkerBase[None]):
    supports_stereo = True

    def _new_audio_state(self, data):
        return None

    def _watermark_request(self, request_id, data, *, finished):
        if request_id == "bad":
            raise ValueError("invalid audio")
        return super()._watermark_request(request_id, data, finished=finished)

    def _watermark_audio(self, data, state):
        return AudioTensor(data.samples + 1, data.sample_rate)

    def _is_audio_watermarked(self, data):
        return False


def _pool():
    pool = object.__new__(StagePool)
    marker = MagicMock(wraps=_TestWatermarker())
    marker.finish_request_output.return_value = None
    pool._watermarkers = {"audio": marker}
    pool._watermark_lock = asyncio.Lock()
    return pool, marker


def test_numpy_audio_accumulates_watermarked_tensors_and_tail():
    pool, marker = _pool()
    accumulated = MultimodalPayload()
    for value in (0.0, 2.0):
        payload = MultimodalPayload.from_dict({"audio": np.full(2, value, dtype=np.float32), "sr": 24000})
        pool._watermark_payload("good", "audio", marker, payload)
        assert isinstance(payload.tensors["audio"], torch.Tensor)
        assert "audio" not in payload.metadata
        accumulated = accumulated.merged_with(payload)
    accumulated.consolidate_tensors(TensorAccumulationStrategy.CONCAT_LAST)
    pool._append_watermark_tail(accumulated, "audio", torch.tensor([5.0]), {"sr": 24000})
    snapshot = MultimodalPayload.from_dict({**accumulated.tensors, **accumulated.metadata})
    torch.testing.assert_close(snapshot["audio"], torch.tensor([1.0, 1.0, 3.0, 3.0, 5.0]))


@pytest.mark.parametrize("structured", [False, True])
def test_append_tail_to_numpy_payload(structured):
    payload = {"audio": np.ones(2, dtype=np.float32), "sr": 24000}
    if structured:
        payload = MultimodalPayload.from_dict(payload)
    StagePool._append_watermark_tail(payload, "audio", torch.tensor([3.0]), {"sr": 24000})
    if structured:
        assert isinstance(payload, MultimodalPayload)
        assert "audio" not in payload.metadata
        torch.testing.assert_close(payload["audio"], torch.tensor([1.0, 1.0, 3.0]))
    else:
        assert isinstance(payload["audio"], np.ndarray)
        np.testing.assert_array_equal(payload["audio"], np.array([1.0, 1.0, 3.0]))


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_metadata", [False, True])
async def test_diffusion_failure_returns_original_payload(missing_metadata):
    pool, marker = _pool()
    output = OmniRequestOutput(request_id="other" if missing_metadata else "bad")
    output._multimodal_output = {"audio": torch.zeros(8)}
    if not missing_metadata:
        output._multimodal_output["sr"] = 24000
    result = await pool.process_diffusion_output(output)
    assert result is output
    assert result.error is None
    assert torch.equal(result.multimodal_output["audio"], torch.zeros(8))
    good = OmniRequestOutput(request_id="good", _multimodal_output={"audio": torch.zeros(8), "sr": 24000})
    assert await pool.process_diffusion_output(good) is good
    assert good.error is None
    assert torch.equal(good.multimodal_output["audio"], torch.ones(8))


@pytest.mark.asyncio
async def test_raw_failure_preserves_audio_and_other_requests():
    pool, marker = _pool()
    states = {
        "bad": SimpleNamespace(external_req_id="external-bad"),
        "good": SimpleNamespace(external_req_id="external-good"),
    }
    good_result = OmniRequestOutput(request_id="external-good")
    processor = MagicMock()
    processor.request_states = states
    processor.process_outputs.return_value = SimpleNamespace(request_outputs=[good_result], reqs_to_abort=[])
    pool._output_processor = processor
    pool.clients = [MagicMock()]
    pool.record_output_timestamps = MagicMock()
    raw = [
        OmniEngineCoreOutput(
            request_id=request_id, new_token_ids=[], multimodal_output={"audio": torch.zeros(8), "sr": 24000}
        )
        for request_id in ("bad", "good")
    ]
    results = await pool.process_llm_raw_outputs(0, SimpleNamespace(outputs=raw, timestamp=1.0, scheduler_stats=None))
    assert processor.process_outputs.call_args.args[0] == raw
    assert results == [good_result]
    torch.testing.assert_close(raw[0].multimodal_output["audio"], torch.zeros(8))
    torch.testing.assert_close(raw[1].multimodal_output["audio"], torch.ones(8))
    assert processor.request_states is states
    assert len(states) == 2


@pytest.mark.parametrize("existing_audio", [False, True])
def test_final_tail_appended_once(existing_audio):
    pool, marker = _pool()
    marker.finish_request_output.return_value = (torch.ones(3), {"sr": 24000})
    completion = MultimodalCompletionOutput(
        index=0,
        text="",
        token_ids=[],
        cumulative_logprob=None,
        logprobs=None,
        multimodal_output=MultimodalPayload(tensors={"audio": torch.zeros(2)}) if existing_audio else None,
    )
    output = OmniRequestOutput(request_id="external", outputs=[completion])
    state = SimpleNamespace(parent_req=None, external_req_id="external", request_index=0)
    assert pool._finish_llm_watermark_outputs({"internal": state}, [output]) == {}
    expected = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]) if existing_audio else torch.ones(3)
    assert torch.equal(completion.multimodal_output["audio"], expected)
    assert completion.multimodal_output["sr"] == 24000
    marker.finish_request_output.assert_called_once_with("internal")


@pytest.mark.asyncio
@pytest.mark.parametrize("retained", [False, True])
async def test_payload_free_final_uses_processor_lifetime(retained):
    pool, marker = _pool()
    marker.finish_request_output.return_value = (torch.ones(3), {"sr": 24000})
    state = SimpleNamespace(parent_req=None, external_req_id="external", request_index=0)
    processor = MagicMock()
    processor.request_states = {"internal": state}
    completion = MultimodalCompletionOutput(
        index=0,
        text="",
        token_ids=[],
        cumulative_logprob=None,
        logprobs=None,
    )
    output = OmniRequestOutput(request_id="external", outputs=[completion])

    def process(*args):
        if not retained:
            processor.request_states.pop("internal")
        return SimpleNamespace(request_outputs=[output], reqs_to_abort=[])

    processor.process_outputs.side_effect = process
    pool._output_processor = processor
    pool.clients = [MagicMock()]
    pool.record_output_timestamps = MagicMock()
    raw = OmniEngineCoreOutput(request_id="internal", new_token_ids=[], finish_reason=FinishReason.STOP)
    results = await pool.process_llm_raw_outputs(0, SimpleNamespace(outputs=[raw], timestamp=1.0, scheduler_stats=None))
    assert results == [output]
    if retained:
        marker.finish_request_output.assert_not_called()
        assert completion.multimodal_output is None
    else:
        marker.finish_request_output.assert_called_once_with("internal")
        assert torch.equal(completion.multimodal_output["audio"], torch.ones(3))


@pytest.mark.asyncio
async def test_completion_watermark_failure_preserves_request_payload():
    pool, _ = _pool()
    completion = MultimodalCompletionOutput(
        index=0,
        text="",
        token_ids=[],
        cumulative_logprob=None,
        logprobs=None,
        multimodal_output=MultimodalPayload(tensors={"audio": torch.zeros(8)}, metadata={"sr": 24000}),
    )
    output = RequestOutput(
        request_id="bad",
        prompt=None,
        prompt_token_ids=[],
        prompt_logprobs=None,
        outputs=[completion],
        finished=True,
    )
    result = await pool.process_diffusion_output(output)
    assert result is output
    assert result.outputs == [completion]
    torch.testing.assert_close(completion.multimodal_output["audio"], torch.zeros(8))


def test_missing_final_consumer_fails_request():
    pool, marker = _pool()
    marker.finish_request_output.return_value = (torch.ones(3), {"sr": 24000})
    state = SimpleNamespace(parent_req=None, external_req_id="external", request_index=0)
    errors = pool._finish_llm_watermark_outputs({"internal": state}, [])
    assert "No completion available" in errors["external"]
    marker.discard_request_state.assert_called_once_with("internal")


@pytest.mark.asyncio
async def test_non_array_audio_remains_a_request_error():
    pool, marker = _pool()
    output = OmniRequestOutput(request_id="bad", _multimodal_output={"audio": "invalid", "sr": 16000})
    result = await pool.process_diffusion_output(output)
    assert "audio output must be an array or tensor" in result.error
    assert result.finished
    assert not result.multimodal_output
    marker.watermark_output.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("output_kind", ["raw", "diffusion"])
@pytest.mark.parametrize("error_type", [AttributeError, KeyError, AssertionError])
async def test_unexpected_watermark_error_propagates(output_kind, error_type):
    pool, marker = _pool()
    marker.watermark_output.side_effect = error_type("watermarker bug")
    payload = {"audio": torch.zeros(8), "sr": 24000}
    if output_kind == "raw":
        output = OmniEngineCoreOutput(request_id="good", new_token_ids=[], multimodal_output=payload)
    else:
        output = OmniRequestOutput(request_id="good", _multimodal_output=payload)
    with pytest.raises(error_type, match="watermarker bug"):
        await pool._process_watermark_outputs([output])


@pytest.mark.parametrize("error_type", [AttributeError, KeyError, AssertionError])
def test_unexpected_flush_error_propagates(error_type):
    pool, marker = _pool()
    marker.finish_request_output.side_effect = error_type("watermarker bug")
    state = SimpleNamespace(external_req_id="external")
    with pytest.raises(error_type, match="watermarker bug"):
        pool._finish_llm_watermark_outputs({"internal": state}, [])


@pytest.mark.asyncio
async def test_late_output_does_not_recreate_watermark_state():
    pool, marker = _pool()
    processor = MagicMock()
    processor.request_states = {}
    processor.process_outputs.return_value = SimpleNamespace(request_outputs=[], reqs_to_abort=[])
    pool._output_processor = processor
    pool.clients = [MagicMock()]
    pool.record_output_timestamps = MagicMock()
    raw = OmniEngineCoreOutput(
        request_id="finished", new_token_ids=[], multimodal_output={"audio": torch.zeros(8), "sr": 24000}
    )
    assert (
        await pool.process_llm_raw_outputs(0, SimpleNamespace(outputs=[raw], timestamp=1.0, scheduler_stats=None)) == []
    )
    marker.watermark_output.assert_not_called()
    assert processor.process_outputs.call_args.args[0] == []


@pytest.mark.asyncio
async def test_diffusion_payload_free_final_flushes_pending_audio():
    pool, marker = _pool()
    marker.finish_request_output.return_value = (torch.ones(3), {"sr": 24000})
    output = OmniRequestOutput(request_id="good", finished=True)
    result = await pool.process_diffusion_output(output)
    assert result is output
    assert result.error is None
    assert torch.equal(result.multimodal_output["audio"], torch.ones(3))
    assert result.multimodal_output["sr"] == 24000


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, TypeError, ValueError])
async def test_unsafe_flush_failure_returns_request_error(error_type):
    pool, marker = _pool()
    marker.finish_request_output.side_effect = error_type("failed final frame")
    state = SimpleNamespace(parent_req=None, external_req_id="external", request_index=0)
    processor = MagicMock()
    processor.request_states = {"internal": state}
    output = OmniRequestOutput(request_id="external", _multimodal_output={"audio": torch.zeros(2)})

    def process(*args):
        processor.request_states.pop("internal")
        return SimpleNamespace(request_outputs=[output], reqs_to_abort=[])

    processor.process_outputs.side_effect = process
    pool._output_processor = processor
    pool.clients = [MagicMock()]
    pool.record_output_timestamps = MagicMock()
    raw = OmniEngineCoreOutput(request_id="internal", new_token_ids=[], finish_reason=FinishReason.STOP)
    results = await pool.process_llm_raw_outputs(0, SimpleNamespace(outputs=[raw], timestamp=1.0, scheduler_stats=None))
    assert len(results) == 1
    assert results[0].request_id == "external"
    assert "failed final frame" in results[0].error
    assert results[0].finished
    assert not results[0].multimodal_output
