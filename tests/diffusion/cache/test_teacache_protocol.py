# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
import torch

from vllm_omni.diffusion.cache.teacache.backend import TeaCacheBackend
from vllm_omni.diffusion.cache.teacache.protocol import ForwardState, SupportsTeaCache
from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.diffusion.models.bagel.bagel_transformer import Bagel
from vllm_omni.diffusion.models.flux.flux_transformer import FluxTransformer2DModel
from vllm_omni.diffusion.models.flux2.flux2_transformer import Flux2Transformer2DModel
from vllm_omni.diffusion.models.flux2_klein.flux2_klein_transformer import (
    Flux2Transformer2DModel as Flux2KleinTransformer2DModel,
)
from vllm_omni.diffusion.models.longcat_image.longcat_image_transformer import LongCatImageTransformer2DModel
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import QwenImageTransformer2DModel
from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import SenseNovaU1ForCausalLM
from vllm_omni.diffusion.models.stable_audio.stable_audio_transformer import StableAudioDiTModel
from vllm_omni.diffusion.models.z_image.z_image_transformer import ZImageTransformer2DModel

pytestmark = [pytest.mark.core_model]

MOCK_COEFFICIENTS = [1.0, 2.0, 3.0, 4.0, 5.0]


TEACACHE_TRANSFORMER_CLASSES = [
    FluxTransformer2DModel,
    Flux2Transformer2DModel,
    Flux2KleinTransformer2DModel,
    QwenImageTransformer2DModel,
    LongCatImageTransformer2DModel,
    ZImageTransformer2DModel,
    StableAudioDiTModel,
    Bagel,
    SenseNovaU1ForCausalLM,
]

# TODO - handle HunyuanImage3, which has a hackier approach
MODEL_COEFFICIENTS = {
    # FLUX transformer coefficients from TeaCache paper
    FluxTransformer2DModel: [
        4.98651651e02,
        -2.83781631e02,
        5.58554382e01,
        -3.82021401e00,
        2.64230861e-01,
    ],
    # Flux2 Klein transformer coefficients
    # Same as FLUX.1 (similar dual-stream architecture)
    Flux2KleinTransformer2DModel: [
        4.98651651e02,
        -2.83781631e02,
        5.58554382e01,
        -3.82021401e00,
        2.64230861e-01,
    ],
    # Qwen-Image transformer coefficients from ComfyUI-TeaCache
    # Tuned specifically for Qwen's dual-stream transformer architecture
    # Used for all Qwen-Image Family pipelines, in general
    QwenImageTransformer2DModel: [
        -4.50000000e02,
        2.80000000e02,
        -4.50000000e01,
        3.20000000e00,
        -2.00000000e-02,
    ],
    # Bagel transformer coefficients
    # Using Qwen's coefficients as reasonable default given shared architecture
    Bagel: [1.33313129e06, -1.68644226e05, 7.95050740e03, -1.63747873e02, 1.26352397e00],
    # SenseNova-U1 transformer coefficients
    SenseNovaU1ForCausalLM: [
        9.07281930e04,
        -2.17699186e04,
        1.83940990e03,
        -6.30339273e01,
        7.61309272e-01,
    ],
    # Z-Image transformer coefficients
    # Copied from Qwen-Image, need to be tuned specifically for Z-Image in future
    ZImageTransformer2DModel: [
        -4.50000000e02,
        2.80000000e02,
        -4.50000000e01,
        3.20000000e00,
        -2.00000000e-02,
    ],
    # Estimated TeaCache polynomial coefficients for StableAudioDiTModel.
    StableAudioDiTModel: [
        121.77490545701518,
        -153.7449426160371,
        68.05368574596551,
        -12.281286412689623,
        1.0733905006198015,
    ],
    # Flux2 transformer coefficients
    # Copied from Qwen-Image, need to be tuned specifically for Flux2 in future
    Flux2Transformer2DModel: [
        -4.50000000e02,
        2.80000000e02,
        -4.50000000e01,
        3.20000000e00,
        -2.00000000e-02,
    ],
    # LongCat Image transformer coefficients
    LongCatImageTransformer2DModel: [652.5980, -424.1615, 84.5526, -4.5923, 0.1694],
}


class FakePipeline:
    def __init__(self, transformer):
        self.transformer = transformer


class MockTeaCacheModel(SupportsTeaCache):
    """A fake implementation of TeaCache's protocol for a given model.

    NOTE: for now this is just used for plumbing, so none of the returned
    values matter. I.e., we should probably check modualted_state etc
    once the hook is actually integrated.
    """

    def preprocess(self, *args, skip_modulated_input: bool, **kwargs):
        return ForwardState(
            modulated_input=None,
            hidden_states=torch.randn(1, 4, 64),
            encoder_hidden_states=None,
            temb=torch.randn(1, 64),
            intermediates=None,
        )

    def run_transformer_blocks(self, ctx):
        return ctx

    def postprocess(self, ctx):
        return ctx.hidden_states

    def get_teacache_coefficients(self):
        return MOCK_COEFFICIENTS


def test_backend_uses_model_coefficients():
    """Ensure that teacache uses the model's coefficients by default."""
    pipeline = FakePipeline(MockTeaCacheModel())

    backend = TeaCacheBackend(DiffusionCacheConfig())
    with patch("vllm_omni.diffusion.cache.teacache.backend.apply_teacache_hook") as mock_hook:
        backend.enable(pipeline)
        cache_config = mock_hook.call_args[0][1]
        assert cache_config.coefficients == MOCK_COEFFICIENTS


def test_backend_user_override_takes_precedence():
    """Ensure that the user's overrides for coefficients take precedence."""
    pipeline = FakePipeline(MockTeaCacheModel())

    user_coeffs = [10.0, 20.0, 30.0, 40.0, 50.0]
    backend = TeaCacheBackend(DiffusionCacheConfig(coefficients=user_coeffs))
    with patch("vllm_omni.diffusion.cache.teacache.backend.apply_teacache_hook") as mock_hook:
        backend.enable(pipeline)
        cache_config = mock_hook.call_args[0][1]
        assert cache_config.coefficients == user_coeffs


def test_backend_raises_for_non_protocol_model():
    """Ensure that we raise if a model that doesn't implement the protocol tries to enable teacache."""

    class NotTeaCacheModel:
        pass

    pipeline = FakePipeline(NotTeaCacheModel())

    backend = TeaCacheBackend(DiffusionCacheConfig())
    with pytest.raises(TypeError):
        backend.enable(pipeline)


@pytest.mark.parametrize("cls", TEACACHE_TRANSFORMER_CLASSES, ids=lambda c: c.__name__)
def test_transformer_implements_protocol(cls):
    """Ensure classes being migrated support teacache protocol."""
    assert issubclass(cls, SupportsTeaCache)


@pytest.mark.parametrize("cls", TEACACHE_TRANSFORMER_CLASSES, ids=lambda c: c.__name__)
def test_model_coefficients_match(cls):
    """Ensure each model's get_teacache_coefficients matches expected values."""
    expected = MODEL_COEFFICIENTS[cls]
    actual = cls.get_teacache_coefficients(None)
    assert actual == expected, f"{cls.__name__} coefficients mismatch"
    assert len(actual) == 5
