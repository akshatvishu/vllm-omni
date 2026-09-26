# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
TeaCache: Timestep Embedding Aware Cache for diffusion model acceleration.

TeaCache speeds up diffusion inference by reusing transformer block computations
when consecutive timestep embeddings are similar.

This implementation uses a hook with model forward methods for migrated models.
Legacy models use extractors until their forwards are migrated.

Usage:
    from vllm_omni import Omni

    omni = Omni(
        model="Qwen/Qwen-Image",
        cache_backend="tea_cache",
        cache_config={"rel_l1_thresh": 0.2}
    )
    images = omni.generate("a cat")

    # Alternative: Using environment variable
    # export DIFFUSION_CACHE_BACKEND=tea_cache
"""

from vllm_omni.diffusion.cache.teacache.backend import TeaCacheBackend
from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.extractors import (
    CacheContext,
    register_extractor,
)
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook, apply_teacache_hook
from vllm_omni.diffusion.cache.teacache.protocol import (
    ForwardState,
    SupportsDecomposedForward,
    SupportsTeaCache,
    TeaCacheDefaults,
)
from vllm_omni.diffusion.cache.teacache.state import TeaCacheState

__all__ = [
    "CacheContext",
    "SupportsDecomposedForward",
    "SupportsTeaCache",
    "TeaCacheDefaults",
    "TeaCacheBackend",
    "TeaCacheConfig",
    "ForwardState",
    "TeaCacheHook",
    "TeaCacheState",
    "apply_teacache_hook",
    "register_extractor",
]
