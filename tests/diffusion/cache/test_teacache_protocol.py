# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
import torch

from vllm_omni.diffusion.cache.teacache.backend import TeaCacheBackend
from vllm_omni.diffusion.cache.teacache.protocol import ForwardState, SupportsTeaCache
from vllm_omni.diffusion.data import DiffusionCacheConfig

pytestmark = [pytest.mark.core_model]

MOCK_COEFFICIENTS = [1.0, 2.0, 3.0, 4.0, 5.0]


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
