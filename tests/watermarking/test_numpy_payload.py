# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.outputs.mm_outputs import MultimodalPayload
from vllm_omni.outputs.output_modality import TensorAccumulationStrategy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_numpy_audio_accumulates_watermarked_tensors():
    pool = object.__new__(StagePool)
    marker = Mock()
    marker.watermark_output.side_effect = lambda request_id, samples, metadata, **kwargs: samples + 1
    accumulated = MultimodalPayload()
    for value in (0.0, 2.0):
        payload = MultimodalPayload.from_dict({"audio": np.full(2, value, dtype=np.float32), "sr": 24000})
        pool._watermark_payload("request", "audio", marker, payload)
        assert isinstance(payload.tensors["audio"], torch.Tensor)
        assert "audio" not in payload.metadata
        accumulated = accumulated.merged_with(payload)
    accumulated.consolidate_tensors(TensorAccumulationStrategy.CONCAT_LAST)
    snapshot = MultimodalPayload.from_dict({**accumulated.tensors, **accumulated.metadata})
    torch.testing.assert_close(snapshot["audio"], torch.tensor([1.0, 1.0, 3.0, 3.0]))


def test_watermark_plain_numpy_payload_preserves_array():
    pool = object.__new__(StagePool)
    marker = Mock()
    marker.watermark_output.side_effect = lambda request_id, samples, metadata, **kwargs: samples + 1
    payload = {"audio": np.zeros(2, dtype=np.float32), "sr": 24000}
    pool._watermark_payload("request", "audio", marker, payload)
    assert isinstance(payload["audio"], np.ndarray)
    np.testing.assert_array_equal(payload["audio"], np.ones(2))


@pytest.mark.asyncio
@pytest.mark.parametrize("reversed_view", [False, True])
async def test_numpy_audio_views_do_not_fail_request(reversed_view):
    pool = object.__new__(StagePool)
    marker = Mock()
    marker.watermark_output.side_effect = lambda request_id, samples, metadata, **kwargs: samples + 1
    marker.finish_request_output.return_value = None
    pool._watermarkers = {"audio": marker}
    pool._watermark_lock = asyncio.Lock()
    samples = np.arange(4, dtype=np.float32) / 10
    if reversed_view:
        samples = samples[::-1]
    output = OmniRequestOutput(request_id="request", finished=True, _multimodal_output={"audio": samples, "sr": 24000})
    result = await pool.process_diffusion_output(output)
    assert result is output
    assert result.error is None
    np.testing.assert_array_equal(result.multimodal_output["audio"], samples + 1)


@pytest.mark.parametrize("structured", [False, True])
def test_append_watermark_tail_to_numpy_view(structured):
    samples = np.arange(4, dtype=np.float32)[::-1]
    payload = {"audio": samples, "sr": 24000}
    if structured:
        payload = MultimodalPayload.from_dict(payload)
    StagePool._append_watermark_tail(payload, "audio", torch.tensor([4.0, 5.0]), {"sr": 24000})
    np.testing.assert_array_equal(payload["audio"], [3.0, 2.0, 1.0, 0.0, 4.0, 5.0])
    if structured:
        assert isinstance(payload, MultimodalPayload)
        assert isinstance(payload.tensors["audio"], torch.Tensor)
        assert "audio" not in payload.metadata
