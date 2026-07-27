# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
TeaCache backend implementation using native model block boundaries.
"""

from typing import Any

import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.interface import (
    supports_teacache,
)
from vllm_omni.diffusion.cache.teacache.runtime import TeaCacheRuntime
from vllm_omni.diffusion.data import DiffusionCacheConfig

logger = init_logger(__name__)

_HUNYUAN_IMAGE3_COEFFICIENTS = (
    1.04117826e02,
    -1.26848482e02,
    5.68168652e01,
    -1.04182570e01,
    6.78098549e-01,
)


def _resolve_coefficients(
    transformer: nn.Module,
    config: DiffusionCacheConfig,
) -> tuple[float, ...]:
    """User override, otherwise model provides its own."""
    if config.coefficients is not None:
        return tuple(float(c) for c in config.coefficients)
    if callable(getattr(transformer, "get_teacache_coefficients", None)):
        return tuple(float(c) for c in transformer.get_teacache_coefficients())
    raise ValueError(f"Model {transformer.__class__.__name__} does not provide TeaCache coefficients.")


def enable_hunyuan_image3_teacache(pipeline: Any, config: DiffusionCacheConfig) -> None:
    teacache_config = TeaCacheConfig(
        transformer_type="HunyuanImage3Pipeline",
        rel_l1_thresh=config.rel_l1_thresh,
        coefficients=(config.coefficients if config.coefficients is not None else _HUNYUAN_IMAGE3_COEFFICIENTS),
    )
    pipeline._tea_cache_config = teacache_config
    logger.info(f"TeaCache enabled for HunyuanImage3 with rel_l1_thresh={teacache_config.rel_l1_thresh}")


def enable_bagel_teacache(pipeline: Any, config: DiffusionCacheConfig, backend: "TeaCacheBackend") -> None:
    transformer = pipeline.bagel
    if not supports_teacache(transformer):
        raise TypeError(f"Bagel transformer {type(transformer).__name__} does not support TeaCache")
    teacache_config = TeaCacheConfig(
        transformer_type="Bagel",
        rel_l1_thresh=config.rel_l1_thresh,
        coefficients=_resolve_coefficients(transformer, config),
    )
    runtime = TeaCacheRuntime(teacache_config)
    transformer.tea_cache_executor = runtime
    backend._installed_runtimes.append(runtime)
    pipeline.transformer = transformer


def enable_sensenova_u1_teacache(pipeline: Any, config: DiffusionCacheConfig, backend: "TeaCacheBackend") -> None:
    adapter = pipeline.denoising_transformer
    transformer = adapter.language_model
    if not supports_teacache(transformer):
        raise TypeError(f"SenseNova transformer {type(transformer).__name__} does not support TeaCache")
    teacache_config = TeaCacheConfig(
        transformer_type="SenseNovaU1ForCausalLM",
        rel_l1_thresh=config.rel_l1_thresh,
        coefficients=_resolve_coefficients(transformer, config),
    )
    runtime = TeaCacheRuntime(teacache_config)
    transformer.tea_cache_executor = runtime
    backend._installed_runtimes.append(runtime)


CUSTOM_TEACACHE_ENABLERS = {
    "BagelPipeline": enable_bagel_teacache,
    "HunyuanImage3Pipeline": enable_hunyuan_image3_teacache,
    "SenseNovaU1Pipeline": enable_sensenova_u1_teacache,
}


class TeaCacheBackend(CacheBackend):
    """
    TeaCache implementation using native model block boundaries.
    """

    def __init__(self, config: DiffusionCacheConfig | dict[str, Any]) -> None:
        super().__init__(config)
        self._installed_runtimes: list[TeaCacheRuntime] = []

    def enable(self, pipeline: Any) -> None:
        pipeline_type = pipeline.__class__.__name__

        if pipeline_type in CUSTOM_TEACACHE_ENABLERS:
            logger.info(f"Using custom TeaCache enabler for model: {pipeline_type}")
            enabler = CUSTOM_TEACACHE_ENABLERS[pipeline_type]
            if pipeline_type == "HunyuanImage3Pipeline":
                enabler(pipeline, self.config)
            else:
                enabler(pipeline, self.config, self)
        else:
            transformer = getattr(pipeline, "transformer", None)
            if transformer is None or not supports_teacache(transformer):
                raise TypeError(
                    f"Transformer {type(transformer).__name__ if transformer is not None else 'None'} "
                    "does not implement SupportsTeaCache protocol"
                )

            transformer_type = transformer.tea_cache_model_key

            teacache_config = TeaCacheConfig(
                transformer_type=transformer_type,
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

        self.enabled = True

    def refresh(self, pipeline: Any, num_inference_steps: int = 50, verbose: bool = True) -> None:
        if (
            hasattr(pipeline, "_tea_cache_config")
            and isinstance(pipeline._tea_cache_config, TeaCacheConfig)
            and pipeline.__class__.__name__ == "HunyuanImage3Pipeline"
        ):
            if verbose:
                logger.debug(f"TeaCache state refreshed for HunyuanImage3 (num_inference_steps={num_inference_steps})")
            return

        for runtime in self._installed_runtimes:
            runtime.reset()

        if verbose:
            logger.debug(f"TeaCache state refreshed (num_inference_steps={num_inference_steps})")
