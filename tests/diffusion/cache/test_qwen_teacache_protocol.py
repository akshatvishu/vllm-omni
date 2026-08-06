# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from types import SimpleNamespace

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from vllm_omni.diffusion.cache.teacache.protocol import ForwardState
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenImageState,
    QwenImageTransformer2DModel,
)


def _make_state(return_dict: bool, *, zero_cond_t: bool = False) -> ForwardState[QwenImageState]:
    batch_size = 1
    temb_batch_size = batch_size * (2 if zero_cond_t else 1)
    return ForwardState(
        modulated_input=None,
        hidden_states=torch.zeros(batch_size, 2, 3),
        encoder_hidden_states=torch.zeros(batch_size, 2, 3),
        temb=torch.ones(temb_batch_size, 3),
        intermediates=QwenImageState(
            image_rotary_emb=(torch.empty(0), torch.empty(0)),
            modulate_index=None,
            joint_attention_kwargs=None,
            hidden_states_mask=None,
            encoder_hidden_states_mask=None,
            return_dict=return_dict,
        ),
    )


def test_preprocess_has_explicit_teacache_arguments() -> None:
    parameters = inspect.signature(QwenImageTransformer2DModel.preprocess).parameters

    assert "return_dict" in parameters
    assert parameters["return_dict"].default is True
    assert parameters["skip_modulated_input"].kind is inspect.Parameter.KEYWORD_ONLY
    assert not any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


def test_run_transformer_blocks_updates_forward_state() -> None:
    class IncrementBlock:
        def __call__(self, *, hidden_states, encoder_hidden_states, **kwargs):
            return encoder_hidden_states + 1, hidden_states + 1

    ctx = _make_state(return_dict=True)
    model = SimpleNamespace(transformer_blocks=[IncrementBlock(), IncrementBlock()])

    result = QwenImageTransformer2DModel.run_transformer_blocks(model, ctx)

    assert result is ctx
    assert torch.equal(ctx.hidden_states, torch.full_like(ctx.hidden_states, 2))
    assert torch.equal(ctx.encoder_hidden_states, torch.full_like(ctx.encoder_hidden_states, 2))


def test_postprocess_updates_state_and_honors_return_dict() -> None:
    class PostprocessModel:
        zero_cond_t = True

        @staticmethod
        def norm_out(hidden_states, temb):
            return hidden_states + temb.unsqueeze(1)

        @staticmethod
        def proj_out(hidden_states):
            return hidden_states * 2

    ctx = _make_state(return_dict=False, zero_cond_t=True)
    model = PostprocessModel()

    result = QwenImageTransformer2DModel.postprocess(model, ctx)

    assert isinstance(result, tuple)
    assert len(result) == 1
    assert ctx.temb.shape[0] == 1
    assert torch.equal(ctx.hidden_states, torch.ones_like(ctx.hidden_states))
    assert torch.equal(result[0], torch.full_like(ctx.hidden_states, 2))


def test_postprocess_returns_model_output_by_default() -> None:
    class PostprocessModel:
        zero_cond_t = False
        norm_out = staticmethod(lambda hidden_states, temb: hidden_states)
        proj_out = staticmethod(lambda hidden_states: hidden_states)

    ctx = _make_state(return_dict=True)
    result = QwenImageTransformer2DModel.postprocess(PostprocessModel(), ctx)

    assert isinstance(result, Transformer2DModelOutput)
