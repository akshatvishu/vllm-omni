# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.cache.teacache.backend import TeaCacheBackend
from vllm_omni.diffusion.cache.teacache.protocol import ForwardState, SupportsTeaCache
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
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import HunyuanImage3Model
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
    HunyuanImage3Model,
]

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
    # HunyuanImage3 coefficients
    # Calibrated via polyfit on 3920 data points (80 prompts x 49 steps)
    HunyuanImage3Model: [1.04117826e02, -1.26848482e02, 5.68168652e01, -1.04182570e01, 6.78098549e-01],
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


# ---------------------------------------------------------------------------
# Forward decomposition equivalence tests
# ---------------------------------------------------------------------------


def _assert_tensors_equal(a: torch.Tensor, b: torch.Tensor):
    """Assert two tensors are bitwise equal, handling NaN (random weights produce NaN)."""
    assert a.shape == b.shape
    nan_match = torch.isnan(a) == torch.isnan(b)
    finite_match = torch.where(torch.isnan(a), True, a == b)
    assert nan_match.all() and finite_match.all()


def _reference_forward(model, inputs):
    """Use a legacy saved forward when available, otherwise use the normal model path."""
    return getattr(model, "_original_forward", model.forward)(**inputs)


@pytest.fixture(scope="module")
def distributed_env():
    """Single-process distributed environment for TP-aware layers."""
    import os

    env_vars = {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500", "WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}
    old = {k: os.environ.get(k) for k in env_vars}
    os.environ.update(env_vars)
    init_distributed_environment()
    initialize_model_parallel()
    yield
    destroy_distributed_environment()
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _make_flux():
    model = FluxTransformer2DModel(
        num_layers=2,
        num_single_layers=2,
        num_attention_heads=2,
        attention_head_dim=16,
        joint_attention_dim=32,
        pooled_projection_dim=16,
        axes_dims_rope=(4, 4, 8),
    )
    inputs = {
        "hidden_states": torch.randn(1, 16, 64, device="cuda"),
        "encoder_hidden_states": torch.randn(1, 8, 32, device="cuda"),
        "pooled_projections": torch.randn(1, 16, device="cuda"),
        "timestep": torch.tensor([500], device="cuda"),
        "img_ids": torch.randint(0, 64, (1, 16, 3), device="cuda"),
        "txt_ids": torch.randint(0, 64, (1, 8, 3), device="cuda"),
        "guidance": torch.tensor([3.5], device="cuda"),
    }
    return model, inputs


def _make_flux2():
    model = Flux2Transformer2DModel(
        num_layers=2,
        num_single_layers=2,
        num_attention_heads=48,
        attention_head_dim=128,
        joint_attention_dim=15360,
    )
    inputs = {
        "hidden_states": torch.randn(1, 64, 128, device="cuda"),
        "encoder_hidden_states": torch.randn(1, 32, 15360, device="cuda"),
        "timestep": torch.tensor([500], device="cuda"),
        "img_ids": torch.randint(0, 64, (1, 64, 4), device="cuda"),
        "txt_ids": torch.randint(0, 64, (1, 32, 4), device="cuda"),
        "guidance": torch.tensor([3.5], device="cuda"),
    }
    return model, inputs


def _make_flux2_klein():
    model = Flux2KleinTransformer2DModel(
        num_layers=2,
        num_single_layers=2,
        num_attention_heads=48,
        attention_head_dim=128,
        joint_attention_dim=15360,
    )
    inputs = {
        "hidden_states": torch.randn(1, 64, 128, device="cuda"),
        "encoder_hidden_states": torch.randn(1, 32, 15360, device="cuda"),
        "timestep": torch.tensor([500], device="cuda"),
        "img_ids": torch.randint(0, 64, (1, 64, 4), device="cuda"),
        "txt_ids": torch.randint(0, 64, (1, 32, 4), device="cuda"),
        "guidance": torch.tensor([3.5], device="cuda"),
    }
    return model, inputs


def _make_stable_audio():
    model = StableAudioDiTModel(
        sample_size=64,
        in_channels=64,
        num_layers=2,
        attention_head_dim=64,
        num_attention_heads=4,
        num_key_value_attention_heads=2,
        out_channels=64,
        cross_attention_dim=32,
        time_proj_dim=16,
        global_states_input_dim=32,
        cross_attention_input_dim=32,
    )
    inputs = {
        "hidden_states": torch.randn(1, 64, 16, device="cuda"),
        "timestep": torch.tensor([0.5], device="cuda"),
        "encoder_hidden_states": torch.randn(1, 8, 32, device="cuda"),
        "global_hidden_states": torch.randn(1, 32, device="cuda"),
        "rotary_embedding": (
            torch.randn(16 + 1, 32, device="cuda"),
            torch.randn(16 + 1, 32, device="cuda"),
        ),
    }
    return model, inputs


def _make_longcat():
    from vllm_omni.diffusion.data import TransformerConfig

    tf_config = TransformerConfig(
        params={
            "patch_size": 1,
            "in_channels": 64,
            "num_layers": 2,
            "num_single_layers": 2,
            "attention_head_dim": 16,
            "num_attention_heads": 2,
            "joint_attention_dim": 32,
            "pooled_projection_dim": 16,
            "axes_dims_rope": [4, 4, 8],
        }
    )
    od = OmniDiffusionConfig(tf_model_config=tf_config)
    with set_current_vllm_config(VllmConfig()):
        model = LongCatImageTransformer2DModel(od)
    inputs = {
        "hidden_states": torch.randn(1, 16, 64, device="cuda"),
        "timestep": torch.tensor([500.0], device="cuda"),
        "guidance": torch.tensor([3.5], device="cuda"),
        "encoder_hidden_states": torch.randn(1, 8, 32, device="cuda"),
        "txt_ids": torch.randint(0, 64, (8, 3), device="cuda"),
        "img_ids": torch.randint(0, 64, (16, 3), device="cuda"),
    }
    return model, inputs


def _make_qwen():
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
    inputs = {
        "hidden_states": torch.randn(1, 16, 64, device="cuda"),
        "timestep": torch.tensor([500.0], device="cuda"),
        "encoder_hidden_states": torch.randn(1, 8, 32, device="cuda"),
        "img_shapes": [(1, 4, 4)],
        "txt_seq_lens": [8],
    }
    return model, inputs


EQUIVALENCE_MODELS = {
    "Flux": _make_flux,
    "Flux2": _make_flux2,
    "Flux2Klein": _make_flux2_klein,
    "StableAudio": _make_stable_audio,
    "LongCat": _make_longcat,
    "Qwen": _make_qwen,
    # TODO: ZImage (complex patchification), Bagel, SenseNova, HunyuanImage3
}


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("model_name", EQUIVALENCE_MODELS.keys())
def test_decomposition_matches_original(distributed_env, model_name):
    """Cache-disabled path: the decomposed forward matches the reference path."""
    model, inputs = EQUIVALENCE_MODELS[model_name]()
    model = model.cuda().eval()
    with set_forward_context(omni_diffusion_config=OmniDiffusionConfig()):
        original = _reference_forward(model, inputs)
        new = model.forward(**inputs)
    _assert_tensors_equal(original.sample, new.sample)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("model_name", EQUIVALENCE_MODELS.keys())
def test_protocol_path_matches_original(distributed_env, model_name):
    """Cache-enabled path: preprocess -> blocks -> postprocess matches the reference path."""
    model, inputs = EQUIVALENCE_MODELS[model_name]()
    model = model.cuda().eval()
    with set_forward_context(omni_diffusion_config=OmniDiffusionConfig()):
        original = _reference_forward(model, inputs)
        ctx = model.preprocess(**inputs, skip_modulated_input=False)
        assert ctx.modulated_input is not None
        ctx = model.run_transformer_blocks(ctx)
        result = model.postprocess(ctx)
    _assert_tensors_equal(original.sample, result.sample)
