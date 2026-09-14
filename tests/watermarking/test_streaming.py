# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.watermarking import AudioSealWatermarker, AudioTensor, AudioWatermarkerBase

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _HistoryWatermarker(AudioWatermarkerBase[torch.Tensor]):
    """Use a causal sum to expose lost samples and history in the adapter."""

    supports_stereo = False
    frame_size = 4

    def _new_audio_state(self, data: AudioTensor) -> torch.Tensor:
        return torch.zeros_like(data.samples[..., :1])

    def _watermark_audio(self, data: AudioTensor, state: torch.Tensor) -> AudioTensor:
        assert data.samples.shape[-1] % self.frame_size == 0
        history = data.samples.cumsum(dim=-1) + state
        state.copy_(history[..., -1:])
        return AudioTensor(data.samples + 0.001 * history, data.sample_rate)

    def _is_audio_watermarked(self, data: AudioTensor) -> bool:
        raise NotImplementedError


@pytest.mark.parametrize("shape", [(19,), (1, 19), (2, 19), (1, 1, 19), (2, 2, 19)])
@pytest.mark.parametrize("explicit_final", [False, True])
def test_streaming_preserves_every_sample_and_layout(shape: tuple[int, ...], explicit_final: bool) -> None:
    source = torch.full(shape, 0.05)
    watermarker = _HistoryWatermarker()
    expected = watermarker.watermark_output("whole", source, {"sr": 16000}, finished=True)
    chunks = list(source.split(3, dim=-1))
    outputs = [
        watermarker.watermark_output(
            "chunks", chunk, {"sr": 16000}, finished=explicit_final and index == len(chunks) - 1
        )
        for index, chunk in enumerate(chunks)
    ]
    tail = watermarker.finish_request_output("chunks")
    if explicit_final:
        assert tail is None
    else:
        assert tail is not None
        assert tail[1] == {"sr": 16000}
        outputs.append(tail[0])
    actual = torch.cat(outputs, dim=-1)
    assert outputs[0].shape[-1] == 0
    assert actual.shape == source.shape
    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual, source)
    assert not watermarker._request_states


def test_abort_discards_pending_audio_before_request_id_reuse() -> None:
    watermarker = _HistoryWatermarker()
    watermarker.watermark_output("reused", torch.full((3,), 0.8), {"sr": 16000})
    watermarker.discard_request_state("reused")
    assert watermarker.finish_request_output("reused") is None
    source = torch.full((7,), 0.1)
    actual = watermarker.watermark_output("reused", source, {"sr": 16000}, finished=True)
    expected = watermarker.watermark_output("fresh", source, {"sr": 16000}, finished=True)
    torch.testing.assert_close(actual, expected)


def test_sample_rate_change_is_rejected_while_frame_is_pending() -> None:
    watermarker = _HistoryWatermarker()
    watermarker.watermark_output("request", torch.zeros(1), {"sr": 16000})
    with pytest.raises(ValueError, match="Cannot return buffered audio"):
        watermarker.watermark_output("request", torch.zeros(1), {"sr": 24000})
    assert watermarker.finish_request_output("request") is None


def test_finish_without_audio_does_not_create_state() -> None:
    watermarker = _HistoryWatermarker()
    assert watermarker.finish_request_output("unknown") is None
    assert not watermarker._request_states


@pytest.mark.local_model
@pytest.mark.slow
@pytest.mark.tts
def test_audioseal_unaligned_interleaved_streams_match_whole_audio() -> None:
    pytest.importorskip("audioseal")
    watermarker = AudioSealWatermarker()
    sources = {
        request_id: torch.randn((1, 1, 6407), generator=torch.Generator().manual_seed(seed)) * 0.05
        for request_id, seed in (("first", 1), ("second", 2))
    }
    try:
        expected = {}
        with torch.inference_mode():
            for request_id, source in sources.items():
                message = watermarker._new_audio_state(AudioTensor(source, 16000)).message
                padded = F.pad(source, (0, -source.shape[-1] % watermarker.frame_size))
                whole = watermarker._model(padded, message=message)[..., : source.shape[-1]]
                expected[request_id] = watermarker._add_residual_with_headroom(source, whole - source)

        chunks = {request_id: list(source.split(333, dim=-1)) for request_id, source in sources.items()}
        outputs: dict[str, list[torch.Tensor]] = {request_id: [] for request_id in sources}
        for index in range(len(chunks["first"])):
            for request_id in sources:
                outputs[request_id].append(
                    watermarker.watermark_output(request_id, chunks[request_id][index], {"sr": 16000})
                )
        for request_id, source in sources.items():
            final = watermarker.finish_request_output(request_id)
            assert final is not None
            outputs[request_id].append(final[0])
            actual = torch.cat(outputs[request_id], dim=-1)
            assert actual.shape == source.shape
            torch.testing.assert_close(actual, expected[request_id], atol=1e-6, rtol=1e-5)
            assert not torch.equal(actual, source)
        assert not watermarker._request_states
    finally:
        watermarker.close()
