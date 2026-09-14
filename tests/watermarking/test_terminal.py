# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutputs, FinishReason

from tests.engine.test_orchestrator import FakeStageClient
from tests.engine.test_output_processor import _make_state
from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.engine.messages import OutputMessage
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.outputs.output_modality import OutputModality
from vllm_omni.outputs.output_processor import MultimodalOutputProcessor
from vllm_omni.watermarking import AudioSealWatermarker, AudioTensor, AudioWatermarkerBase

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _BufferedIdentityWatermarker(AudioWatermarkerBase[None]):
    supports_stereo = True
    frame_size = 8

    def _new_audio_state(self, data: AudioTensor) -> None:
        return None

    def _watermark_audio(self, data: AudioTensor, state: None) -> AudioTensor:
        return data

    def _is_audio_watermarked(self, data: AudioTensor) -> bool:
        raise NotImplementedError


@pytest.mark.asyncio
@pytest.mark.parametrize("distinct_ids", [False, True])
@pytest.mark.parametrize(
    "real_model",
    [False, pytest.param(True, marks=[pytest.mark.local_model, pytest.mark.slow, pytest.mark.tts])],
    ids=["buffer", "audioseal"],
)
async def test_nonfinal_audio_stop_retains_request_until_final_chunk(real_model, distinct_ids, monkeypatch):
    if real_model:
        pytest.importorskip("audioseal")
    marker = AudioSealWatermarker() if real_model else _BufferedIdentityWatermarker()
    try:
        length, split = (1007, 333) if real_model else (11, 3)
        source = torch.randn(length, generator=torch.Generator().manual_seed(42)) * 0.05
        expected = source
        if real_model:
            with torch.inference_mode():
                batched = source[None, None]
                message = marker._new_audio_state(AudioTensor(batched, 16000)).message
                padded = F.pad(batched, (0, -length % marker.frame_size))
                whole = marker._model(padded, message=message)[..., :length]
                expected = marker._add_residual_with_headroom(batched, whole - batched).reshape(-1)
        close_state = Mock(wraps=marker._close_state)
        monkeypatch.setattr(marker, "_close_state", close_state)

        request_id = "external"
        internal_id = "internal" if distinct_ids else request_id
        processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False, output_modality=OutputModality.AUDIO)
        state = _make_state(RequestOutputKind.DELTA)
        state.request_id = internal_id
        state.external_req_id = request_id
        state.detokenizer = None
        state.logprobs_processor = None
        processor.request_states[internal_id] = state
        processor.external_req_ids[request_id].append(internal_id)
        client = FakeStageClient(final_output=True, final_output_type="audio")
        pool = StagePool(
            0,
            [client],
            output_processor=processor,
            stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
            watermarkers={"audio": marker},
        )
        pool._request_bindings[request_id] = 0
        queue: asyncio.Queue[OutputMessage] = asyncio.Queue()
        orchestrator = Orchestrator(
            request_async_queue=asyncio.Queue(),
            output_async_queue=queue,
            rpc_async_queue=asyncio.Queue(),
            stage_pools=[pool],
            async_chunk=True,
        )
        request = OrchestratorRequestState(
            request_id=request_id,
            final_stage_id=0,
            final_output_stage_ids={0},
            sampling_params_list=[SamplingParams(output_kind=RequestOutputKind.DELTA)],
        )
        request.streaming.enabled = True
        orchestrator.request_states[request_id] = request
        messages = []
        for index, chunk in enumerate((source[:split], source[split:])):
            raw = EngineCoreOutputs(
                outputs=[
                    OmniEngineCoreOutput(
                        request_id=internal_id,
                        new_token_ids=[],
                        finish_reason=FinishReason.STOP,
                        is_segment_finished=False,
                        multimodal_output={"audio": chunk, "sr": 16000, "meta.tts_is_last_chunk": index},
                    )
                ],
                timestamp=float(index + 1),
            )
            terminal_ids: set[str] = set()
            processed = await orchestrator._process_llm_stage_outputs(0, 0, raw, terminal_ids)
            assert len(processed) == 1
            assert processed[0].finished == bool(index)
            await orchestrator._handle_processed_outputs(0, 0, processed)
            await orchestrator._finish_raw_terminal_requests(0, 0, terminal_ids)
            while not queue.empty():
                messages.append(queue.get_nowait())
            if index == 0:
                assert not terminal_ids
                assert not request.finished_final_output_stage_ids
                assert all(not message.finished for message in messages)
                assert request_id in orchestrator.request_states
                assert request_id in pool._request_bindings
                assert internal_id in processor.request_states
                assert marker._request_states[internal_id].pending.shape[-1] == split % marker.frame_size
                close_state.assert_not_called()

        assert sum(message.finished for message in messages) == 1
        actual = torch.cat(
            [message.engine_outputs.outputs[0].multimodal_output["audio"].reshape(-1) for message in messages]
        )
        assert actual.shape == source.shape
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        if real_model:
            assert not torch.equal(actual, source)
        assert request_id not in orchestrator.request_states
        assert request_id not in pool._request_bindings
        assert not processor.request_states
        assert not processor.external_req_ids
        assert not marker._request_states
        close_state.assert_called_once()
        assert not client.abort_calls
    finally:
        marker.close()
