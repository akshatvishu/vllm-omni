# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU contract checks against forwards copied from pinned main."""

import inspect
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.diffusion.cache.pinned_teacache_forwards import (
    pinned_flux2_forward,
    pinned_flux2_klein_forward,
    pinned_flux_forward,
    pinned_longcat_forward,
    pinned_qwen_forward,
    pinned_stable_audio_forward,
    pinned_z_image_forward,
)
from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.flux.flux_transformer import FluxTransformer2DModel
from vllm_omni.diffusion.models.flux2.flux2_transformer import Flux2Transformer2DModel
from vllm_omni.diffusion.models.flux2_klein.flux2_klein_transformer import (
    Flux2Transformer2DModel as Flux2KleinTransformer2DModel,
)
from vllm_omni.diffusion.models.longcat_image.longcat_image_transformer import LongCatImageTransformer2DModel
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import QwenImageTransformer2DModel
from vllm_omni.diffusion.models.stable_audio.stable_audio_transformer import StableAudioDiTModel
from vllm_omni.diffusion.models.z_image.z_image_transformer import ZImageTransformer2DModel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _PairBlock:
    def norm1(self, hidden_states, emb):
        return (hidden_states + emb.unsqueeze(1),)

    def __call__(self, *, hidden_states, encoder_hidden_states, temb, image_rotary_emb, **kwargs):
        assert image_rotary_emb is not None
        return encoder_hidden_states + 1, hidden_states + temb.unsqueeze(1) + 2


class _FluxFixture:
    forward = FluxTransformer2DModel.forward
    preprocess = FluxTransformer2DModel.preprocess
    run_transformer_blocks = FluxTransformer2DModel.run_transformer_blocks
    postprocess = FluxTransformer2DModel.postprocess

    def __init__(self):
        self.x_embedder = lambda x: x + 1
        self.context_embedder = lambda x: x + 2
        self.time_text_embed = lambda timestep, *args: timestep[:, None].expand(-1, 2)
        self.pos_embed = lambda ids: (ids.float(), ids.float())
        self.transformer_blocks = [_PairBlock()]
        self.single_transformer_blocks = [_PairBlock()]
        self.norm_out = lambda hidden, temb: hidden + temb.unsqueeze(1)
        self.proj_out = lambda hidden: hidden * 2


class _LongCatFixture:
    forward = LongCatImageTransformer2DModel.forward
    preprocess = LongCatImageTransformer2DModel.preprocess
    run_transformer_blocks = LongCatImageTransformer2DModel.run_transformer_blocks
    postprocess = LongCatImageTransformer2DModel.postprocess

    def __init__(self):
        self.parallel_config = SimpleNamespace(sequence_parallel_size=1)
        self.enforce_eager = False
        self.x_embedder = lambda x: x + 1
        self.context_embedder = lambda x: x + 2
        self.time_embed = lambda timestep, dtype: timestep[:, None].expand(-1, 2)
        self.rope_preparer = lambda txt, img: (txt.float(), txt.float(), img.float(), img.float())
        self.transformer_blocks = [_PairBlock()]
        self.single_transformer_blocks = [_PairBlock()]
        self.norm_out = lambda hidden, temb: hidden + temb.unsqueeze(1)
        self.proj_out = lambda hidden: hidden * 2


class _AudioBlock:
    def norm1(self, hidden_states):
        return hidden_states + 1

    def __call__(
        self,
        hidden_states,
        encoder_hidden_states,
        *,
        rotary_embedding,
        attention_mask=None,
        encoder_attention_mask=None,
    ):
        assert rotary_embedding is not None
        if attention_mask is not None:
            assert attention_mask.shape[-1] == hidden_states.shape[1]
            assert attention_mask[0, 0]
        if encoder_attention_mask is not None:
            assert encoder_attention_mask.shape[-1] == encoder_hidden_states.shape[1]
        return hidden_states + encoder_hidden_states.mean() + 1


class _AudioFixture:
    forward = StableAudioDiTModel.forward
    preprocess = StableAudioDiTModel.preprocess
    run_transformer_blocks = StableAudioDiTModel.run_transformer_blocks
    postprocess = StableAudioDiTModel.postprocess
    dtype = torch.float32

    def __init__(self):
        self.cross_attention_proj = lambda x: x + 1
        self.global_proj = lambda x: x + 2
        self.time_proj = lambda x: x
        self.timestep_proj = lambda x: x + 3
        self.preprocess_conv = lambda x: x * 0.5
        self.proj_in = lambda x: x + 4
        self.transformer_blocks: list[Any] = [_AudioBlock()]
        self.proj_out = lambda x: x * 2
        self.postprocess_conv = lambda x: x * 0.25


def _flux_inputs():
    return dict(
        hidden_states=torch.arange(8, dtype=torch.float32).reshape(1, 4, 2),
        encoder_hidden_states=torch.ones(1, 2, 2),
        pooled_projections=torch.ones(1, 2),
        timestep=torch.tensor([0.5]),
        img_ids=torch.ones(4, 3),
        txt_ids=torch.ones(2, 3),
        guidance=torch.tensor([1.0]),
        joint_attention_kwargs={"scale": 0.5},
    )


def _longcat_inputs():
    return dict(
        hidden_states=torch.arange(8, dtype=torch.float32).reshape(1, 4, 2),
        encoder_hidden_states=torch.ones(1, 2, 2),
        timestep=torch.tensor([0.5]),
        img_ids=torch.ones(4, 3),
        txt_ids=torch.ones(2, 3),
        guidance=torch.tensor([1.0]),
    )


def _audio_inputs():
    return dict(
        hidden_states=torch.arange(8, dtype=torch.float32).reshape(1, 2, 4),
        timestep=torch.ones(1, 2),
        encoder_hidden_states=torch.ones(1, 2, 2),
        global_hidden_states=torch.ones(1, 1, 2),
        rotary_embedding=(torch.ones(5, 2), torch.ones(5, 2)),
    )


@pytest.mark.parametrize(
    ("model_class", "fixture_class", "pinned_forward", "inputs_factory"),
    [
        (FluxTransformer2DModel, _FluxFixture, pinned_flux_forward, _flux_inputs),
        (LongCatImageTransformer2DModel, _LongCatFixture, pinned_longcat_forward, _longcat_inputs),
        (StableAudioDiTModel, _AudioFixture, pinned_stable_audio_forward, _audio_inputs),
    ],
)
@pytest.mark.parametrize("return_dict", [True, False])
def test_split_forward_matches_pinned_main(model_class, fixture_class, pinned_forward, inputs_factory, return_dict):
    assert list(inspect.signature(model_class.forward).parameters) == list(inspect.signature(pinned_forward).parameters)
    model = fixture_class()
    inputs = inputs_factory()
    inputs["return_dict"] = return_dict
    if model_class is StableAudioDiTModel:
        inputs["attention_mask"] = torch.tensor([[True, False, True, True]])
        inputs["encoder_attention_mask"] = torch.tensor([[True, False]])

    with set_forward_context():
        expected = pinned_forward(model, **inputs)
        actual = model.forward(**inputs)
        split = model.postprocess(model.run_transformer_blocks(model.preprocess(**inputs, skip_modulated_input=False)))

    expected_tensor = expected.sample if return_dict else expected[0]
    assert type(actual) is type(expected)
    assert type(split) is type(expected)
    torch.testing.assert_close(actual.sample if return_dict else actual[0], expected_tensor, rtol=0, atol=0)
    torch.testing.assert_close(split.sample if return_dict else split[0], expected_tensor, rtol=0, atol=0)


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


@pytest.mark.parametrize(
    ("fixture_class", "pinned_forward"),
    [(_Flux2Fixture, pinned_flux2_forward), (_KleinFixture, pinned_flux2_klein_forward)],
)
@pytest.mark.parametrize("return_dict", [True, False])
def test_flux2_split_matches_pinned_main(fixture_class, pinned_forward, return_dict):
    assert list(inspect.signature(fixture_class.forward).parameters) == list(
        inspect.signature(pinned_forward).parameters
    )
    inputs = _flux2_inputs() | {"return_dict": return_dict}
    with set_forward_context():
        expected = pinned_forward(fixture_class(), **inputs)
        actual = fixture_class().forward(**inputs)
        model = fixture_class()
        state = model.preprocess(**inputs, skip_modulated_input=False)
        assert state.modulated_input is not None
        split = model.postprocess(model.run_transformer_blocks(state))
    assert type(actual) is type(expected)
    assert type(split) is type(expected)
    torch.testing.assert_close(_sample(actual), _sample(expected), rtol=0, atol=0)
    torch.testing.assert_close(_sample(split), _sample(expected), rtol=0, atol=0)


def test_qwen_split_matches_pinned_main():
    assert list(inspect.signature(QwenImageTransformer2DModel.forward).parameters) == list(
        inspect.signature(pinned_qwen_forward).parameters
    )
    inputs = _qwen_inputs()
    with set_forward_context():
        expected = pinned_qwen_forward(_QwenFixture(), **inputs)
        actual = _QwenFixture().forward(**inputs)
        model = _QwenFixture()
        state = model.preprocess(**inputs, skip_modulated_input=False)
        assert state.modulated_input is not None
        split = model.postprocess(model.run_transformer_blocks(state))
    torch.testing.assert_close(actual.sample, expected.sample, rtol=0, atol=0)
    torch.testing.assert_close(split.sample, expected.sample, rtol=0, atol=0)


@pytest.mark.parametrize("extra_conditions", [False, True])
def test_z_image_split_matches_pinned_main(extra_conditions):
    assert list(inspect.signature(ZImageTransformer2DModel.forward).parameters) == list(
        inspect.signature(pinned_z_image_forward).parameters
    )
    inputs = _z_image_inputs(extra_conditions=extra_conditions)
    expected = pinned_z_image_forward(_ZImageFixture(), **inputs)
    actual = _ZImageFixture().forward(**inputs)
    model = _ZImageFixture()
    state = model.preprocess(**inputs, skip_modulated_input=False)
    assert state.modulated_input is not None
    split = model.postprocess(model.run_transformer_blocks(state))
    assert actual[1] == split[1] == expected[1] == {}
    for result in (actual, split):
        assert len(result[0]) == len(expected[0])
        for item, reference in zip(result[0], expected[0]):
            torch.testing.assert_close(item, reference, rtol=0, atol=0)


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
    ("fixture_class", "pinned_forward", "inputs_factory"),
    [
        (_Flux2Fixture, pinned_flux2_forward, _flux2_inputs),
        (_KleinFixture, pinned_flux2_klein_forward, _flux2_inputs),
        (_QwenFixture, pinned_qwen_forward, _qwen_inputs),
        (_ZImageFixture, pinned_z_image_forward, _z_image_inputs),
    ],
)
def test_model_cache_hit_reuses_residual_instead_of_running_blocks(fixture_class, pinned_forward, inputs_factory):
    model = fixture_class()
    hook = TeaCacheHook(
        TeaCacheConfig(transformer_type=type(model).__name__, coefficients=[0, 0, 0, 0, 0], rel_l1_thresh=1.0)
    )
    hook.initialize_hook(model)
    with set_forward_context():
        hook.new_forward(model, **inputs_factory())
        actual = hook.new_forward(model, **inputs_factory(offset=5))
        expected = _cache_hit_reference(fixture_class(), inputs_factory(), inputs_factory(offset=5))
        uncached = pinned_forward(fixture_class(), **inputs_factory(offset=5))

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
