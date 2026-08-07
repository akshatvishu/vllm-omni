# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TeaCache backend implementation."""

from typing import Any

import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook, apply_teacache_hook
from vllm_omni.diffusion.cache.teacache.interface import supports_teacache
from vllm_omni.diffusion.cache.teacache.runtime import TeaCacheRuntime
from vllm_omni.diffusion.data import DiffusionCacheConfig

logger = init_logger(__name__)


def _resolve_coefficients(
    transformer: nn.Module,
    config: DiffusionCacheConfig,
) -> tuple[float, ...]:
    """Resolve user coefficients or the model's coefficients."""
    if config.coefficients is not None:
        return tuple(float(coefficient) for coefficient in config.coefficients)
    getter = getattr(transformer, "get_teacache_coefficients", None)
    if not callable(getter):
        raise TypeError(f"{type(transformer).__name__} does not implement SupportsTeaCache")
    return tuple(float(coefficient) for coefficient in getter())


def enable_bagel_teacache(pipeline: Any, config: DiffusionCacheConfig) -> None:
    transformer = pipeline.bagel
    teacache_config = TeaCacheConfig(
        transformer_type="Bagel",
        rel_l1_thresh=config.rel_l1_thresh,
        coefficients=_resolve_coefficients(transformer, config),
    )
    apply_teacache_hook(transformer, teacache_config)
    pipeline.transformer = transformer

    logger.info(
        f"TeaCache applied with rel_l1_thresh={teacache_config.rel_l1_thresh}, "
        f"transformer_class={teacache_config.transformer_type}"
    )


def enable_sensenova_u1_teacache(pipeline: Any, config: DiffusionCacheConfig) -> None:
    transformer = pipeline.denoising_transformer
    teacache_config = TeaCacheConfig(
        transformer_type="SenseNovaU1ForCausalLM",
        rel_l1_thresh=config.rel_l1_thresh,
        coefficients=_resolve_coefficients(transformer, config),
    )
    apply_teacache_hook(transformer, teacache_config)

    logger.info(
        f"TeaCache applied with rel_l1_thresh={teacache_config.rel_l1_thresh}, "
        f"transformer_class={teacache_config.transformer_type}"
    )


CUSTOM_TEACACHE_ENABLERS = {
    "BagelPipeline": enable_bagel_teacache,
    "SenseNovaU1Pipeline": enable_sensenova_u1_teacache,
}


class TeaCacheBackend(CacheBackend):
    """Install native TeaCache runtimes, with legacy hook fallback."""

    def __init__(self, config: DiffusionCacheConfig) -> None:
        super().__init__(config)
        self._installed_runtimes: list[TeaCacheRuntime] = []

    def enable(self, pipeline: Any) -> None:
        self._installed_runtimes.clear()
        pipeline_type = pipeline.__class__.__name__
        transformer = getattr(pipeline, "transformer", None)

        if pipeline_type in CUSTOM_TEACACHE_ENABLERS:
            logger.info(f"Using legacy custom TeaCache enabler for model: {pipeline_type}")
            CUSTOM_TEACACHE_ENABLERS[pipeline_type](pipeline, self.config)
        elif transformer is None:
            raise TypeError("Pipeline does not expose a transformer for TeaCache")
        elif supports_teacache(transformer):
            teacache_config = TeaCacheConfig(
                transformer_type=transformer.tea_cache_model_key,
                rel_l1_thresh=self.config.rel_l1_thresh,
                coefficients=_resolve_coefficients(transformer, self.config),
            )
            runtime = TeaCacheRuntime(teacache_config)
            transformer.tea_cache_executor = runtime
            self._installed_runtimes.append(runtime)
            logger.info(
                f"TeaCache applied with rel_l1_thresh={teacache_config.rel_l1_thresh}, "
                f"transformer_class={teacache_config.transformer_type}"
            )
        else:
            transformer_type = transformer.__class__.__name__
            teacache_config = TeaCacheConfig(
                transformer_type=transformer_type,
                rel_l1_thresh=self.config.rel_l1_thresh,
                coefficients=_resolve_coefficients(transformer, self.config),
            )
            apply_teacache_hook(transformer, teacache_config)
            logger.info(
                f"Legacy TeaCache hook applied with rel_l1_thresh={teacache_config.rel_l1_thresh}, "
                f"transformer_class={teacache_config.transformer_type}"
            )

        self.enabled = True

    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        if self._installed_runtimes:
            for runtime in self._installed_runtimes:
                runtime.reset()
            if verbose:
                logger.debug(f"TeaCache state refreshed (num_inference_steps={num_inference_steps})")
            return

        transformer = pipeline.transformer
        if not hasattr(transformer, "_hook_registry") and hasattr(pipeline, "denoising_transformer"):
            transformer = pipeline.denoising_transformer

        if hasattr(transformer, "_hook_registry"):
            hook = transformer._hook_registry.get_hook(TeaCacheHook._HOOK_NAME)
            if hook is not None:
                transformer._hook_registry.reset_hook(TeaCacheHook._HOOK_NAME)
                if verbose:
                    logger.debug(f"TeaCache state refreshed (num_inference_steps={num_inference_steps})")
            elif verbose:
                logger.warning("TeaCache hook not found, nothing to refresh")
        elif verbose:
            logger.warning("Transformer has no hook registry, TeaCache may not be applied")
