# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import QwenImageTransformer2DModel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _ImageBlock(nn.Module):
    @staticmethod
    def img_mod(temb):
        shift = temb.expand(-1, 2)
        scale = torch.zeros_like(shift)
        gate = torch.zeros_like(shift)
        first_norm = torch.cat((shift, scale, gate), dim=-1)
        return torch.cat((first_norm, torch.zeros_like(first_norm)), dim=-1)

    @staticmethod
    def _modulate(mod_params):
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return scale.unsqueeze(1), shift.unsqueeze(1), gate.unsqueeze(1)

    @staticmethod
    def img_norm1(hidden_states, scale, shift):
        return hidden_states * (1 + scale) + shift


class _QwenWithZeroConditioning(nn.Module):
    def __init__(self):
        super().__init__()
        self.parallel_config = SimpleNamespace(sequence_parallel_size=1)
        self.transformer_blocks = nn.ModuleList([_ImageBlock()])

    @staticmethod
    def image_rope_prepare(hidden_states, img_shapes, txt_seq_lens):
        return hidden_states, torch.empty(0), torch.empty(0)

    @staticmethod
    def modulate_index_prepare(timestep, img_shapes):
        batch_size = timestep.shape[0]
        index = torch.tensor([0, 0, 1]).expand(batch_size, -1)
        return torch.cat((timestep, timestep * 0)), index

    @staticmethod
    def txt_norm(encoder_hidden_states):
        return encoder_hidden_states

    @staticmethod
    def txt_in(encoder_hidden_states):
        return encoder_hidden_states

    @staticmethod
    def time_text_embed(timestep, hidden_states, additional_t_cond):
        return timestep.unsqueeze(-1)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_zero_conditioning_uses_tokenwise_modulation_for_cache_signal(batch_size):
    hidden_states = torch.zeros(batch_size, 3, 2)
    timestep = torch.arange(1, batch_size + 1, dtype=torch.float32)
    with patch(
        "vllm_omni.diffusion.models.qwen_image.qwen_image_transformer.get_forward_context",
        return_value=SimpleNamespace(),
    ):
        context = QwenImageTransformer2DModel.preprocess(
            _QwenWithZeroConditioning(),
            hidden_states=hidden_states,
            encoder_hidden_states=torch.zeros(batch_size, 1, 2),
            encoder_hidden_states_mask=torch.ones(batch_size, 1, dtype=torch.bool),
            timestep=timestep,
            img_shapes=torch.empty(0),
            txt_seq_lens=torch.empty(0),
        )

    expected_shift = timestep.unsqueeze(1) * torch.tensor([1, 1, 0])
    expected = expected_shift.unsqueeze(-1).expand(-1, -1, 2)
    assert context.modulated_input.shape == hidden_states.shape
    torch.testing.assert_close(context.modulated_input, expected, rtol=0, atol=0)
