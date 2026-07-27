# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.cache.teacache.backend import TeaCacheBackend
from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.interface import supports_teacache
from vllm_omni.diffusion.cache.teacache.runtime import TeaCacheRuntime
from vllm_omni.diffusion.data import DiffusionCacheConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_environment,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
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

NATIVE_TEACACHE_MODELS = [
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


class MockTeaCacheModel:
    supports_teacache = True
    tea_cache_model_key = "MockTeaCacheModel"
    tea_cache_executor = None

    def get_teacache_coefficients(self) -> list[float]:
        return [1.0, 2.0, 3.0, 4.0, 5.0]


class FakePipeline:
    def __init__(self, transformer):
        self.transformer = transformer


def test_native_models_expose_the_cache_boundary():
    """The backend validates the model contract without relying on Protocol inheritance."""
    for model_class in NATIVE_TEACACHE_MODELS:
        assert supports_teacache(model_class)
        assert len(model_class.get_teacache_coefficients(None)) == 5


def test_backend_uses_model_coefficients_by_default():
    pipeline = FakePipeline(MockTeaCacheModel())

    backend = TeaCacheBackend(DiffusionCacheConfig())
    backend.enable(pipeline)

    assert pipeline.transformer.tea_cache_executor.config.coefficients == (1.0, 2.0, 3.0, 4.0, 5.0)


def test_backend_user_coefficients_take_precedence():
    pipeline = FakePipeline(MockTeaCacheModel())
    coefficients = [10.0, 20.0, 30.0, 40.0, 50.0]

    backend = TeaCacheBackend(DiffusionCacheConfig(coefficients=coefficients))
    backend.enable(pipeline)

    assert pipeline.transformer.tea_cache_executor.config.coefficients == tuple(coefficients)


def test_backend_rejects_model_without_native_boundary():
    pipeline = FakePipeline(object())

    with pytest.raises(TypeError, match="SupportsTeaCache"):
        TeaCacheBackend(DiffusionCacheConfig()).enable(pipeline)


def test_backend_selects_bagel_native_target():
    class BagelPipeline:
        def __init__(self):
            self.bagel = MockTeaCacheModel()
            self.transformer = object()

    pipeline = BagelPipeline()
    TeaCacheBackend(DiffusionCacheConfig()).enable(pipeline)

    assert pipeline.transformer is pipeline.bagel
    assert isinstance(pipeline.bagel.tea_cache_executor, TeaCacheRuntime)


def test_backend_selects_sensenova_language_model_target():
    class DenoisingAdapter:
        def __init__(self):
            self.language_model = MockTeaCacheModel()
            self.do_true_cfg = True

    class SenseNovaU1Pipeline:
        def __init__(self):
            self.denoising_transformer = DenoisingAdapter()

    pipeline = SenseNovaU1Pipeline()
    TeaCacheBackend(DiffusionCacheConfig()).enable(pipeline)

    assert isinstance(pipeline.denoising_transformer.language_model.tea_cache_executor, TeaCacheRuntime)


def test_hunyuan_pipeline_keeps_pipeline_native_configuration():
    class HunyuanImage3Pipeline:
        pass

    pipeline = HunyuanImage3Pipeline()
    backend = TeaCacheBackend(DiffusionCacheConfig())
    backend.enable(pipeline)

    assert pipeline._tea_cache_config.coefficients == (
        1.04117826e02,
        -1.26848482e02,
        5.68168652e01,
        -1.04182570e01,
        6.78098549e-01,
    )
    assert backend._installed_runtimes == []


@pytest.fixture(scope="module")
def distributed_env():
    """Initialize the TP-aware layers used by the single-GPU native-path test."""
    env_vars = {
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
        "WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
    }
    old_values = {key: os.environ.get(key) for key in env_vars}
    os.environ.update(env_vars)
    init_distributed_environment()
    initialize_model_parallel()
    try:
        yield
    finally:
        destroy_distributed_environment()
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _make_tiny_qwen_image():
    with set_current_vllm_config(VllmConfig()):
        model = QwenImageTransformer2DModel(
            OmniDiffusionConfig(),
            num_layers=2,
            num_attention_heads=2,
            attention_head_dim=16,
            joint_attention_dim=32,
            in_channels=64,
            out_channels=16,
            axes_dims_rope=(4, 4, 8),
        )
    # Some vLLM layer parameters are intentionally left uninitialized until a
    # checkpoint is loaded. Seed this tiny structural test with finite weights
    # so a cache hit tests execution rather than NaN fallback handling.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    inputs = {
        "hidden_states": torch.randn(1, 16, 64, device="cuda"),
        "timestep": torch.tensor([500.0], device="cuda"),
        "encoder_hidden_states": torch.randn(1, 8, 32, device="cuda"),
        "img_shapes": [(1, 4, 4)],
        "txt_seq_lens": [8],
    }
    return model.cuda().eval(), inputs


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native TeaCache integration")
def test_qwen_image_native_teacache_skips_transformer_blocks(distributed_env):
    """Exercise the real Qwen-Image boundary and prove a repeated step is a cache hit."""
    model, inputs = _make_tiny_qwen_image()
    model.tea_cache_executor = TeaCacheRuntime(
        config=TeaCacheConfig(
            rel_l1_thresh=0.1,
            coefficients=[0.0, 0.0, 0.0, 1.0, 0.0],
        )
    )

    block_calls = [0]

    def count_block_call(*_args, **_kwargs):
        block_calls[0] += 1

    handles = [block.register_forward_hook(count_block_call) for block in model.transformer_blocks]
    try:
        with set_forward_context(omni_diffusion_config=OmniDiffusionConfig()), torch.inference_mode():
            first = model(**inputs)
            calls_after_first = block_calls[0]
            second = model(**inputs)

        assert calls_after_first == len(model.transformer_blocks)
        assert block_calls[0] == calls_after_first
        assert model.tea_cache_executor.state.forward_cnt == 2
        assert torch.equal(first.sample, second.sample)
    finally:
        for handle in handles:
            handle.remove()
        model.cpu()
        torch.accelerator.empty_cache()
