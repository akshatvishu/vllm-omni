# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU checks that model cache hits reuse residuals and skip their blocks."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.flux2.flux2_transformer import Flux2Transformer2DModel
from vllm_omni.diffusion.models.flux2_klein.flux2_klein_transformer import (
    Flux2Transformer2DModel as Flux2KleinTransformer2DModel,
)
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import QwenImageTransformer2DModel
from vllm_omni.diffusion.models.z_image.z_image_transformer import ZImageTransformer2DModel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Flux2DualBlock:
    def __init__(self):
        self.calls = 0

    def norm1(self, hidden_states):
        return hidden_states + 0.25

    def __call__(self, *, hidden_states, encoder_hidden_states, temb_mod_params_img, image_rotary_emb, **kwargs):
        self.calls += 1
        assert image_rotary_emb[0].shape[0] == hidden_states.shape[1] + encoder_hidden_states.shape[1]
        shift = temb_mod_params_img[0][0]
        return encoder_hidden_states + 2, hidden_states + 1 + shift + 0.01 * hidden_states.square()


class _Flux2SingleBlock:
    def __init__(self):
        self.calls = 0

    def __call__(self, *, hidden_states, text_seq_len, temb_mod_params, **kwargs):
        self.calls += 1
        assert text_seq_len == 2
        return hidden_states + 3 + temb_mod_params[:, None, :] + 0.01 * hidden_states.square()


class _Flux2Fixture:
    forward = Flux2Transformer2DModel.forward
    preprocess = Flux2Transformer2DModel.preprocess
    run_transformer_blocks = Flux2Transformer2DModel.run_transformer_blocks
    postprocess = Flux2Transformer2DModel.postprocess
    get_teacache_defaults = Flux2Transformer2DModel.get_teacache_defaults

    def __init__(self):
        self.parallel_config = SimpleNamespace(sequence_parallel_size=1, mask_sp_padding=False)
        self.time_guidance_embed = lambda timestep, guidance: timestep[:, None].expand(-1, 2)
        self.double_stream_modulation_img = lambda temb: ((temb[:, None, :], torch.zeros_like(temb[:, None, :]), temb),)
        self.double_stream_modulation_txt = self.double_stream_modulation_img
        self.single_stream_modulation = lambda temb: (temb,)
        self.x_embedder = lambda hidden: hidden + 1
        self.context_embedder = lambda encoder: encoder + 2
        self.rope_prepare = lambda img_ids, txt_ids: (
            txt_ids[:, :2].float(),
            txt_ids[:, :2].float(),
            img_ids[:, :2].float(),
            img_ids[:, :2].float(),
        )
        self.transformer_blocks = [_Flux2DualBlock()]
        self.single_transformer_blocks = [_Flux2SingleBlock()]
        self.norm_out = lambda hidden, temb: hidden + temb[:, None, :]
        self.proj_out = lambda hidden: hidden * 2


class _KleinFixture(_Flux2Fixture):
    forward = Flux2KleinTransformer2DModel.forward
    preprocess = Flux2KleinTransformer2DModel.preprocess
    run_transformer_blocks = Flux2KleinTransformer2DModel.run_transformer_blocks
    postprocess = Flux2KleinTransformer2DModel.postprocess
    get_teacache_defaults = Flux2KleinTransformer2DModel.get_teacache_defaults


def _flux2_inputs(offset=0):
    return dict(
        hidden_states=torch.arange(8, dtype=torch.float32).reshape(1, 4, 2) + offset,
        encoder_hidden_states=torch.ones(1, 2, 2),
        timestep=torch.tensor([0.5]),
        img_ids=torch.ones(4, 3),
        txt_ids=torch.ones(2, 3),
        guidance=torch.tensor([1.0]),
        joint_attention_kwargs={"scale": 0.5},
    )


class _QwenBlock:
    def __init__(self):
        self.calls = 0

    def img_mod(self, temb):
        return temb.repeat(1, 6)

    def _modulate(self, value):
        return (value[:, None, :2], value[:, None, 2:4], value[:, None, 4:6])

    def img_norm1(self, hidden, scale, shift):
        return hidden * (1 + scale) + shift

    def __call__(self, *, hidden_states, encoder_hidden_states, temb, image_rotary_emb, **kwargs):
        self.calls += 1
        assert image_rotary_emb[0].shape[0] == hidden_states.shape[1]
        return encoder_hidden_states + 2, hidden_states + temb[:, None, :] + 1 + 0.01 * hidden_states.square()


class _QwenFixture:
    forward = QwenImageTransformer2DModel.forward
    preprocess = QwenImageTransformer2DModel.preprocess
    run_transformer_blocks = QwenImageTransformer2DModel.run_transformer_blocks
    postprocess = QwenImageTransformer2DModel.postprocess
    get_teacache_defaults = QwenImageTransformer2DModel.get_teacache_defaults
    zero_cond_t = False

    def __init__(self):
        self.parallel_config = SimpleNamespace(sequence_parallel_size=1, mask_sp_padding=False)
        self.image_rope_prepare = lambda hidden, img_shapes, txt_seq_lens: (
            hidden + 1,
            torch.ones(hidden.shape[1], 2),
            torch.ones(txt_seq_lens[0], 2),
        )
        self.modulate_index_prepare = lambda timestep, img_shapes: (timestep, None)
        self.txt_norm = lambda encoder: encoder + 1
        self.txt_in = lambda encoder: encoder + 2
        self.time_text_embed = lambda timestep, *args: timestep[:, None].expand(-1, 2)
        self.transformer_blocks = [_QwenBlock()]
        self.norm_out = lambda hidden, temb: hidden + temb[:, None, :]
        self.proj_out = lambda hidden: hidden * 2


def _qwen_inputs(offset=0):
    return dict(
        hidden_states=torch.arange(8, dtype=torch.float32).reshape(1, 4, 2) + offset,
        encoder_hidden_states=torch.ones(1, 2, 2),
        encoder_hidden_states_mask=torch.tensor([[True, False]]),
        timestep=torch.tensor([0.5]),
        img_shapes=[(1, 2, 2)],
        txt_seq_lens=[2],
        guidance=torch.tensor([1.0]),
        attention_kwargs={"scale": 0.5},
        return_dict=True,
    )


class _ZImageBlock:
    def __init__(self):
        self.calls = 0

    def adaLN_modulation(self, temb):
        return temb.repeat(1, 4)

    def attention_norm1(self, hidden):
        return hidden + 0.25

    def __call__(self, hidden, mask, cos, sin, temb):
        self.calls += 1
        assert hidden.shape[1] == mask.shape[1] == cos.shape[1] == sin.shape[1]
        return hidden + temb[:, None, :] + 1 + 0.01 * hidden.square()


class _ZImageFixture:
    forward = ZImageTransformer2DModel.forward
    preprocess = ZImageTransformer2DModel.preprocess
    run_transformer_blocks = ZImageTransformer2DModel.run_transformer_blocks
    postprocess = ZImageTransformer2DModel.postprocess
    get_teacache_defaults = ZImageTransformer2DModel.get_teacache_defaults
    all_patch_size = {2}
    all_f_patch_size = {1}
    alignment_padding_mode = "zero_masked"
    t_scale = 1

    def __init__(self):
        self.t_embedder = lambda t: t
        self.all_x_embedder = {"2-1": lambda x: x + 1}
        self.cap_embedder = lambda cap: cap + 2
        self.rope_embedder = lambda ids: (ids.float(), ids.float())
        self.noise_refiner = [lambda x, mask, cos, sin, temb: x + temb[:, None, :]]
        self.context_refiner = [lambda cap, mask, cos, sin: cap + 1]
        self.layers = [_ZImageBlock()]
        self.all_final_layer = {"2-1": lambda hidden, temb: hidden + temb[:, None, :]}
        self.unpatchify = lambda unified, x_size, patch_size, f_patch_size: unified

    def patchify_and_embed(self, x, cap_feats, patch_size, f_patch_size, ref_x, cap_feats_2):
        image = x[0] + (ref_x[0] if ref_x is not None and ref_x[0] is not None else 0)
        cap_length = cap_feats[0].shape[0] + (cap_feats_2[0].shape[0] if cap_feats_2 else 0)
        return (
            [image],
            cap_feats,
            [(1, 4, 4)],
            [torch.ones(image.shape[0], 2)],
            [torch.ones(cap_length, 2)],
            [torch.zeros(image.shape[0], dtype=torch.bool)],
            [torch.zeros(cap_length, dtype=torch.bool)],
            cap_feats_2,
        )

    def unified_prepare(self, x, x_cos, x_sin, cap, cap_cos, cap_sin, x_lengths, cap_lengths, x_mask, cap_mask):
        return (
            torch.cat((cap, x), dim=1),
            torch.cat((cap_cos, x_cos), dim=1),
            torch.cat((cap_sin, x_sin), dim=1),
            torch.cat((cap_mask, x_mask), dim=1),
        )


def _z_image_inputs(offset=0, *, extra_conditions=False):
    return dict(
        x=[torch.arange(64, dtype=torch.float32).reshape(32, 2) + offset],
        t=torch.tensor([[0.5, 1.0]]),
        cap_feats=[torch.ones(32, 2)],
        ref_x=[torch.ones(32, 2)] if extra_conditions else None,
        cap_feats_2=[torch.full((32, 2), 2.0)] if extra_conditions else None,
    )


def _sample(output):
    return output.sample if hasattr(output, "sample") else output[0]


def _cache_hit_reference(model, first_inputs, second_inputs):
    first = model.preprocess(**first_inputs, skip_modulated_input=False)
    hidden = first.hidden_states.clone()
    encoder = None if first.encoder_hidden_states is None else first.encoder_hidden_states.clone()
    first = model.run_transformer_blocks(first)
    second = model.preprocess(**second_inputs, skip_modulated_input=False)
    second.hidden_states = second.hidden_states + (first.hidden_states - hidden)
    if encoder is not None:
        second.encoder_hidden_states = second.encoder_hidden_states + (first.encoder_hidden_states - encoder)
    return model.postprocess(second)


def _output_tensors(output):
    return output[0] if isinstance(output[0], list) else [_sample(output)]


@pytest.mark.parametrize(
    ("fixture_class", "inputs_factory"),
    [
        (_Flux2Fixture, _flux2_inputs),
        (_KleinFixture, _flux2_inputs),
        (_QwenFixture, _qwen_inputs),
        (_ZImageFixture, _z_image_inputs),
    ],
)
def test_model_cache_hit_reuses_residual_instead_of_running_blocks(fixture_class, inputs_factory):
    model = fixture_class()
    hook = TeaCacheHook(
        TeaCacheConfig(transformer_type=type(model).__name__, coefficients=[0, 0, 0, 0, 0], rel_l1_thresh=1.0)
    )
    hook.initialize_hook(model)
    with set_forward_context():
        hook.new_forward(model, **inputs_factory())
        actual = hook.new_forward(model, **inputs_factory(offset=5))
        expected = _cache_hit_reference(fixture_class(), inputs_factory(), inputs_factory(offset=5))
        uncached = fixture_class().forward(**inputs_factory(offset=5))

    blocks = model.layers if fixture_class is _ZImageFixture else model.transformer_blocks
    assert blocks[0].calls == 1
    if fixture_class in (_Flux2Fixture, _KleinFixture):
        assert model.single_transformer_blocks[0].calls == 1
    assert type(actual) is type(expected)
    for item, reference, fresh in zip(
        _output_tensors(actual), _output_tensors(expected), _output_tensors(uncached), strict=True
    ):
        torch.testing.assert_close(item, reference, rtol=0, atol=0)
        assert not torch.allclose(item, fresh)
