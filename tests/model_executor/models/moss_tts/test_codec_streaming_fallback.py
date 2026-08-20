# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import (
    MossTTSCodecDecoder,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


class _LegacyCodec(nn.Module):
    downsample_rate = 2

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(codebook_size=8)
        self.batch_decode_calls = 0

    def batch_decode(self, codes_list, num_quantizers):
        self.batch_decode_calls += 1
        frames = int(codes_list[0].shape[1])
        return SimpleNamespace(
            audio=torch.ones((1, 1, frames * self.downsample_rate)),
            audio_lengths=torch.tensor([frames * self.downsample_rate]),
        )


class _StreamingCodec(_LegacyCodec):
    def initialize_decoder_state_pool(self, state_capacity, scratch_capacity=0):
        pass

    def reset_decoder_state_slots(self, state_slot_ids):
        pass

    def decode_streaming_batch(self, codes, codes_lengths, state_slot_ids, valid_rows):
        pass


def _decoder(codec: nn.Module) -> MossTTSCodecDecoder:
    decoder = MossTTSCodecDecoder.__new__(MossTTSCodecDecoder)
    nn.Module.__init__(decoder)
    decoder._codec = codec
    decoder._codec_path = "test-codec"
    decoder._n_vq = 2
    decoder._n_channels = 1
    decoder._sr_tensor = torch.tensor(24_000, dtype=torch.int32)
    decoder._async_chunk = True
    decoder._streaming_codec_enabled = False
    decoder._stream_session = None
    decoder._cuda_graph_wrapper = None
    return decoder


def test_legacy_codec_uses_per_chunk_decode() -> None:
    codec = _LegacyCodec()
    decoder = _decoder(codec)

    decoder._configure_codec_streaming()
    output = decoder(
        input_ids=torch.tensor([1, 2, 3, 4]),
        runtime_additional_information=[{"meta": {"req_id": "request-0"}}],
        seq_token_counts=[4],
    )

    assert decoder._streaming_codec_enabled is False
    assert decoder._ensure_stream_session() is None
    assert codec.batch_decode_calls == 1
    assert output.multimodal_outputs["model_outputs"][0].shape == (4,)


def test_state_pool_codec_keeps_streaming_enabled() -> None:
    decoder = _decoder(_StreamingCodec())

    decoder._configure_codec_streaming()

    assert decoder._streaming_codec_enabled is True
