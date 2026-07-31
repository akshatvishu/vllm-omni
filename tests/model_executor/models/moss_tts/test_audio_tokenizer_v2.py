# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for MOSS audio tokenizer v2 projection modules."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import (
    MossAudioTokenizerProjectedTransformer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_projected_transformer_uses_identity_when_dimensions_match() -> None:
    module = MossAudioTokenizerProjectedTransformer(
        input_dimension=8,
        output_dimension=8,
        d_model=8,
        module_type="transformer",
        num_heads=1,
        num_layers=0,
        positional_embedding="rope",
    )

    assert isinstance(module.input_proj, nn.Identity)
    assert isinstance(module.output_proj, nn.Identity)
    assert all(not name.startswith(("input_proj", "output_proj")) for name, _ in module.named_parameters())

    x = torch.randn(2, 8, 5)
    lengths = torch.tensor([5, 3], dtype=torch.long)
    output, output_lengths = module(x, lengths)

    torch.testing.assert_close(output, x)
    torch.testing.assert_close(output_lengths, lengths)


def test_projected_transformer_keeps_linear_projections_when_dimensions_differ() -> None:
    module = MossAudioTokenizerProjectedTransformer(
        input_dimension=8,
        output_dimension=6,
        d_model=10,
        module_type="transformer",
        num_heads=1,
        num_layers=0,
        positional_embedding="rope",
    )

    assert isinstance(module.input_proj, nn.Linear)
    assert isinstance(module.output_proj, nn.Linear)
