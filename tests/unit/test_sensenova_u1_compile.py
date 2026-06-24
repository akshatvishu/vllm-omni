# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import (
    SenseNovaU1Model,
    _ensure_preallocated_cache_layers,
)


def test_sensenova_declares_decoder_layers_for_regional_compile() -> None:
    assert SenseNovaU1Model._repeated_blocks == ["SenseNovaU1DecoderLayer"]


def test_preallocated_cache_does_not_grow_during_layer_updates() -> None:
    cache = DynamicCache()
    _ensure_preallocated_cache_layers(cache, num_layers=3)

    assert len(cache.layers) == 3
    assert all(isinstance(layer, DynamicLayer) for layer in cache.layers)
    assert cache.layer_class_to_replicate is None

    key = torch.ones(1, 1, 2, 4)
    value = torch.full_like(key, 2)
    updated_key, updated_value = cache.layers[2].update(key, value)

    assert len(cache.layers) == 3
    torch.testing.assert_close(updated_key, key)
    torch.testing.assert_close(updated_value, value)


def test_preallocation_preserves_existing_cache_layers() -> None:
    cache = DynamicCache()
    _ensure_preallocated_cache_layers(cache, num_layers=2)
    first_layer = cache.layers[0]

    _ensure_preallocated_cache_layers(cache, num_layers=4)

    assert len(cache.layers) == 4
    assert cache.layers[0] is first_layer
