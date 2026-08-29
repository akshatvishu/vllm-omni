# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
    SqueezeExcitationBlock,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_squeeze_excitation_restores_mean_input_dtype() -> None:
    """The batch invariant mean may promote BF16 input to FP32."""

    class PromotedMeanInput:
        dtype = torch.bfloat16

        def __init__(self) -> None:
            self.mean_call: tuple[int, bool] | None = None

        def mean(self, *, dim: int, keepdim: bool) -> torch.Tensor:
            self.mean_call = (dim, keepdim)
            return torch.ones(1, 4, 1, dtype=torch.float32)

        def __mul__(self, value: torch.Tensor) -> torch.Tensor:
            return value

    block = SqueezeExcitationBlock(4, 2, 4)
    block.conv1 = torch.nn.Identity()
    block.relu = torch.nn.Identity()
    block.conv2 = torch.nn.Identity()
    block.sigmoid = torch.nn.Identity()
    hidden_states = PromotedMeanInput()

    output = block(hidden_states)

    assert hidden_states.mean_call == (2, True)
    assert output.dtype == torch.bfloat16
