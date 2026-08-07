# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
TeaCache: Timestep Embedding Aware Cache for diffusion model acceleration.

TeaCache reuses transformer block residuals when consecutive timestep embeddings
are similar. Migrated models expose a native block boundary; legacy models still
use the hook integration until they are migrated.

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
from vllm_omni.diffusion.cache.teacache.interface import (
    SupportsTeaCache,
    TeaCacheBlockExecutor,
    supports_teacache,
)
from vllm_omni.diffusion.cache.teacache.protocol import ForwardState
from vllm_omni.diffusion.cache.teacache.runtime import TeaCacheRuntime
from vllm_omni.diffusion.cache.teacache.state import TeaCacheState

__all__ = [
    "CacheContext",
    "SupportsTeaCache",
    "TeaCacheBackend",
    "TeaCacheBlockExecutor",
    "TeaCacheConfig",
    "ForwardState",
    "TeaCacheHook",
    "TeaCacheState",
    "TeaCacheRuntime",
    "apply_teacache_hook",
    "register_extractor",
    "supports_teacache",
]
