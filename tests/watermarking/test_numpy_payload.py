# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm_omni.engine.stage_pool import StagePool
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
