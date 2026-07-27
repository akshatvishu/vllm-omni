# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Cache module for diffusion model inference acceleration.
"""

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.cache.prompt_embed_cache import (
    PromptEmbedCache,
    install_prompt_embed_cache,
    resolve_prompt_embed_cache_config,
    uninstall_prompt_embed_cache,
)
from vllm_omni.diffusion.cache.teacache import (
    SupportsTeaCache,
    TeaCacheBackend,
    TeaCacheBlockExecutor,
    TeaCacheConfig,
    TeaCacheRuntime,
    supports_teacache,
)

__all__ = [
    "CacheBackend",
    "SupportsTeaCache",
    "TeaCacheBackend",
    "TeaCacheBlockExecutor",
    "TeaCacheConfig",
    "TeaCacheRuntime",
    "supports_teacache",
    "PromptEmbedCache",
    "install_prompt_embed_cache",
    "resolve_prompt_embed_cache_config",
    "uninstall_prompt_embed_cache",
]
