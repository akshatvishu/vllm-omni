# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.diffusion.cache.teacache.extractors import CacheContext, extract_glmimage_context

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeNormBlock:
    def __init__(self):
        self.norm1_calls = []
        self.calls = []

    def norm1(self, hidden_states, encoder_hidden_states, temb):
        self.norm1_calls.append(
            {
                "hidden_states": hidden_states.clone(),
                "encoder_hidden_states": encoder_hidden_states.clone(),
                "temb": temb.clone(),
            }
        )
        return (
            hidden_states + 100.0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    def __call__(
        self,
        *,
        hidden_states,
        encoder_hidden_states,
        temb,
        image_rotary_emb,
        attention_mask,
        attention_kwargs,
        kv_cache,
        kv_cache_mode,
    ):
        self.calls.append(
            {
                "hidden_states": hidden_states.clone(),
                "encoder_hidden_states": encoder_hidden_states.clone(),
                "temb": temb.clone(),
                "image_rotary_emb": image_rotary_emb,
                "attention_mask": attention_mask,
                "attention_kwargs": attention_kwargs,
                "kv_cache": kv_cache,
                "kv_cache_mode": kv_cache_mode,
            }
        )
        return hidden_states + 2.0, encoder_hidden_states + 3.0


class _FakeEmbedding:
    def __call__(self, token_ids):
        return token_ids.unsqueeze(-1).repeat(1, 1, 4).float()


class _FakeGlmTransformer:
    def __init__(self):
        self.patch_size = 2
        self.transformer_blocks = [_FakeNormBlock(), _FakeNormBlock()]
        self.rope_calls = []

    def rope(self, hidden_states):
        self.rope_calls.append(hidden_states.clone())
        return (
            torch.full((1,), 11.0, dtype=hidden_states.dtype, device=hidden_states.device),
            torch.full((1,), 13.0, dtype=hidden_states.dtype, device=hidden_states.device),
        )

    def image_projector(self, hidden_states):
        batch = hidden_states.shape[0]
        return torch.zeros(batch, 4, 4, dtype=hidden_states.dtype, device=hidden_states.device)

    def glyph_projector(self, encoder_hidden_states):
        return encoder_hidden_states + 5.0

    prior_token_embedding = _FakeEmbedding()

    def prior_projector(self, prior_embedding):
        return prior_embedding

    def time_condition_embed(self, timestep, target_size, crop_coords, dtype):
        del target_size, crop_coords
        return timestep.unsqueeze(-1).to(dtype=dtype) + 0.5

    def norm_out(self, hidden_states, temb):
        return hidden_states + temb.unsqueeze(1)

    def proj_out(self, hidden_states):
        return hidden_states


class _FakeKVCache:
    def __init__(self):
        self.mode = "cache-mode"
        self.layers = ["layer-0-cache", "layer-1-cache"]

    def __getitem__(self, idx):
        return self.layers[idx]


def _make_inputs():
    hidden_states = torch.zeros(1, 1, 4, 4)
    encoder_hidden_states = torch.ones(1, 3, 4)
    prior_token_id = torch.tensor([[1, 2, 3, 4]])
    prior_token_drop = torch.tensor([[False, True, False, True]])
    timestep = torch.tensor([2.0])
    target_size = torch.tensor([[4, 4]])
    crop_coords = torch.tensor([[0, 0]])
    return (
        hidden_states,
        encoder_hidden_states,
        prior_token_id,
        prior_token_drop,
        timestep,
        target_size,
        crop_coords,
    )


def test_extract_glmimage_context_builds_cache_context_and_unpatchifies_output():
    module = _FakeGlmTransformer()
    attention_mask = torch.ones(1, 4, dtype=torch.bool)
    attention_kwargs = {"scale": 0.25}
    kv_cache = _FakeKVCache()
    (
        hidden_states,
        encoder_hidden_states,
        prior_token_id,
        prior_token_drop,
        timestep,
        target_size,
        crop_coords,
    ) = _make_inputs()

    ctx = extract_glmimage_context(
        module,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        prior_token_id=prior_token_id,
        prior_token_drop=prior_token_drop,
        timestep=timestep,
        target_size=target_size,
        crop_coords=crop_coords,
        attention_mask=attention_mask,
        attention_kwargs=attention_kwargs,
        kv_cache=kv_cache,
    )

    assert isinstance(ctx, CacheContext)
    assert len(module.rope_calls) == 1
    expected_hidden_states = torch.tensor(
        [[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [3.0, 3.0, 3.0, 3.0], [0.0, 0.0, 0.0, 0.0]]]
    )
    expected_encoder_hidden_states = torch.full((1, 3, 4), 6.0)
    assert torch.equal(ctx.hidden_states, expected_hidden_states)
    assert torch.equal(ctx.encoder_hidden_states, expected_encoder_hidden_states)
    assert torch.equal(ctx.modulated_input, expected_hidden_states + 100.0)

    hidden_out, encoder_out = ctx.run_transformer_blocks()
    assert torch.equal(hidden_out, expected_hidden_states + 4.0)
    assert torch.equal(encoder_out, expected_encoder_hidden_states + 6.0)

    for idx, block in enumerate(module.transformer_blocks):
        assert block.calls[0]["attention_mask"] is attention_mask
        assert block.calls[0]["attention_kwargs"] is attention_kwargs
        assert block.calls[0]["kv_cache"] == kv_cache[idx]
        assert block.calls[0]["kv_cache_mode"] == kv_cache.mode
        rotary_emb = block.calls[0]["image_rotary_emb"]
        assert torch.equal(rotary_emb[0], torch.tensor([11.0]))
        assert torch.equal(rotary_emb[1], torch.tensor([13.0]))

    output = ctx.postprocess(hidden_out)
    assert output.sample.shape == (1, 1, 4, 4)

    tuple_output = extract_glmimage_context(
        module,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        prior_token_id=prior_token_id,
        prior_token_drop=prior_token_drop,
        timestep=timestep,
        target_size=target_size,
        crop_coords=crop_coords,
        return_dict=False,
    ).postprocess(hidden_out)
    assert isinstance(tuple_output, tuple)
    assert tuple_output[0].shape == (1, 1, 4, 4)
