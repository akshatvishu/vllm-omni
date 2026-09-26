# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Hook-based TeaCache implementation for vLLM-Omni.

This module intercepts the transformer forward pass for both model forward
protocols and the legacy extractor interface during the migration.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.extractors import CacheContext, get_extractor
from vllm_omni.diffusion.cache.teacache.protocol import SupportsTeaCache, validate_protocol_forward
from vllm_omni.diffusion.cache.teacache.state import TeaCacheState
from vllm_omni.diffusion.distributed.parallel_state import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_sp_group,
    model_parallel_is_initialized,
)
from vllm_omni.diffusion.hooks import HookRegistry, ModelHook, StateManager

logger = init_logger(__name__)


def _average_l1_stats_across_sp(mean_diff: torch.Tensor, mean_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Average TeaCache L1 stats across sequence-parallel ranks.

    Skip vs compute must be identical on every rank that participates in
    transformer SP collectives. Sequence shards otherwise produce different
    local means and deadlock. TP is not included: the residual TeaCache reads
    is replicated across TP ranks. CFG/DP ranks are also excluded: they do
    not share those collectives.
    """
    if not model_parallel_is_initialized():
        return mean_diff, mean_prev

    sp_group = get_sp_group()
    if sp_group.world_size <= 1:
        return mean_diff, mean_prev

    stats = torch.stack((mean_diff.detach(), mean_prev.detach()))
    stats = sp_group.all_reduce(stats) / sp_group.world_size
    return stats[0], stats[1]


class TeaCacheHook(ModelHook):
    """
    ModelHook implementing TeaCache for transformer models.

    This hook completely intercepts the transformer's forward pass and implements
    adaptive caching based on timestep embedding similarity. Models can use a
    decomposed forward or the legacy extractor interface.

    Key features:
    - Separate states for the existing positive and negative CFG branches
    - Model-specific polynomial rescaling
    - Auto-detection of model types

    Attributes:
        config: TeaCache configuration with thresholds and callbacks
        rescale_func: Polynomial function for rescaling L1 distances
        state_manager: Manages TeaCacheState across forward passes
        _forward_impl: Forward path chosen when the hook is initialized
    """

    _HOOK_NAME = "teacache"

    def __init__(self, config: TeaCacheConfig):
        """
        Initialize TeaCacheHook.

        Args:
            config: TeaCache configuration object.
        """
        super().__init__()
        self.config = config
        self.rescale_func = np.poly1d(config.coefficients)
        self.state_manager = StateManager(TeaCacheState)
        self._forward_impl: Callable[..., Any]
        self._forward_cnt = 0

    def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        if isinstance(module, SupportsTeaCache):
            validate_protocol_forward(module)
            self._forward_impl = self._protocol_forward
        else:
            self._forward_impl = partial(self._legacy_forward, get_extractor(self.config.transformer_type))
        self.state_manager.set_context("teacache")
        return module

    def new_forward(self, module: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
        return self._forward_impl(module, *args, **kwargs)

    def _get_cache_state(self, module: torch.nn.Module) -> TeaCacheState:
        if getattr(module, "do_true_cfg", False):
            cfg_parallel_size = get_classifier_free_guidance_world_size()
            if cfg_parallel_size > 1:
                cfg_rank = get_classifier_free_guidance_rank()
                cache_branch = "negative" if cfg_rank > 0 else "positive"
            else:
                cache_branch = "negative" if self._forward_cnt % 2 == 1 else "positive"
        else:
            cache_branch = "positive"

        self.state_manager.set_context(f"teacache_{cache_branch}")
        return self.state_manager.get_state()

    def _protocol_forward(self, module: SupportsTeaCache, *args: Any, **kwargs: Any) -> Any:
        ctx = module.preprocess(*args, skip_modulated_input=False, **kwargs)

        def run_blocks():
            nonlocal ctx
            ctx = module.run_transformer_blocks(ctx)
            return ctx.hidden_states, ctx.encoder_hidden_states

        ctx.hidden_states, ctx.encoder_hidden_states = self._cache_step(
            module, ctx.modulated_input, ctx.hidden_states, ctx.encoder_hidden_states, run_blocks
        )
        return module.postprocess(ctx)

    def _legacy_forward(
        self, extractor: Callable[..., CacheContext], module: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> Any:
        ctx = extractor(module, *args, **kwargs)
        extra_states = ctx.extra_states or {}

        def run_blocks():
            outputs = ctx.run_transformer_blocks()
            encoder = (
                outputs[1] if len(outputs) > 1 and ctx.encoder_hidden_states is not None else ctx.encoder_hidden_states
            )
            return outputs[0], encoder

        ctx.hidden_states, ctx.encoder_hidden_states = self._cache_step(
            module,
            ctx.modulated_input,
            ctx.hidden_states,
            ctx.encoder_hidden_states,
            run_blocks,
            sync_cache_decision=extra_states.get("synchronize_cache_decision"),
        )
        return ctx.postprocess(ctx.hidden_states)

    def _cache_step(
        self,
        module: torch.nn.Module,
        modulated_input: torch.Tensor | None,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        run_blocks: Callable[[], tuple[torch.Tensor, torch.Tensor | None]],
        sync_cache_decision: Callable[[bool], bool] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if modulated_input is None:
            raise ValueError("TeaCache preprocessing did not provide a modulated input")

        state = self._get_cache_state(module)
        local_should_compute = self._should_compute_full_transformer(state, modulated_input)
        if sync_cache_decision is not None:
            should_compute = sync_cache_decision(local_should_compute)
            if should_compute and not local_should_compute:
                state.accumulated_rel_l1_distance = 0.0
        else:
            should_compute = local_should_compute

        cache_hit = not should_compute and state.previous_residual is not None
        if cache_hit:
            hidden_states = hidden_states + state.previous_residual
            if state.previous_residual_encoder is not None and encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states + state.previous_residual_encoder
        else:
            original_hidden = hidden_states.clone()
            original_encoder = encoder_hidden_states.clone() if encoder_hidden_states is not None else None
            hidden_states, encoder_hidden_states = run_blocks()
            state.previous_residual = (hidden_states - original_hidden).detach()
            if original_encoder is not None:
                state.previous_residual_encoder = (encoder_hidden_states - original_encoder).detach()

        logger.debug("TeaCache step=%d cache_hit=%s", state.cnt, cache_hit)
        state.previous_modulated_input = modulated_input.detach()
        state.cnt += 1
        self._forward_cnt += 1
        return hidden_states, encoder_hidden_states

    def _should_compute_full_transformer(self, state: TeaCacheState, modulated_inp: torch.Tensor) -> bool:
        """
        Determine whether to compute full transformer or reuse cached residual.

        This implements the core TeaCache algorithm:
        1. Always compute first timestep
        2. For intermediate steps:
           - Compute relative L1 distance between current and previous modulated inputs
           - Average that distance across SP so all ranks share the skip decision
           - Apply polynomial rescaling with model-specific coefficients
           - Accumulate rescaled distances
           - Compare to threshold: below = cache, above = compute

        Args:
            state: Current TeaCacheState containing counters and cached values
            modulated_inp: Modulated input extracted from first transformer block

        Returns:
            True to compute full transformer, False to reuse cached residual
        """
        # First timestep: always compute
        if state.cnt == 0:
            state.accumulated_rel_l1_distance = 0.0
            return True

        # Need previous input for comparison
        if state.previous_modulated_input is None:
            return True

        # Compute relative L1 distance between consecutive modulated inputs.
        # Reduce across SP before the threshold so every rank takes the same
        # skip/compute path (otherwise a cache hit on one rank deadlocks
        # waiting on collectives the miss path never enters).
        mean_diff = (modulated_inp - state.previous_modulated_input).abs().mean()
        mean_prev = state.previous_modulated_input.abs().mean()
        mean_diff, mean_prev = _average_l1_stats_across_sp(mean_diff, mean_prev)
        rel_distance = (mean_diff / (mean_prev + 1e-8)).item()

        # Apply model-specific polynomial rescaling
        rescaled_distance = float(self.rescale_func(rel_distance))
        state.accumulated_rel_l1_distance += abs(rescaled_distance)

        # Decision: below threshold = cache, above = compute
        rel_l1_thresh = self.config.rel_l1_thresh
        assert rel_l1_thresh is not None
        if state.accumulated_rel_l1_distance < rel_l1_thresh:
            return False  # Use cache
        else:
            state.accumulated_rel_l1_distance = 0.0  # Reset accumulator
            return True  # Compute

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        """
        Reset all cached states for a new inference run.

        Args:
            module: The module to reset state for.

        Returns:
            The module with reset state.
        """
        self.state_manager.reset()
        self._forward_cnt = 0
        return module


def apply_teacache_hook(module: torch.nn.Module, config: TeaCacheConfig) -> None:
    """
    Apply TeaCache optimization to a transformer module.

    This function registers a TeaCacheHook that completely intercepts the
    module's forward pass and uses either the model's decomposed forward or a
    legacy extractor.

    Args:
        module: Transformer model to optimize (e.g., QwenImageTransformer2DModel)
        config: TeaCacheConfig specifying caching parameters

    Example:
        >>> config = TeaCacheConfig(
        ...     rel_l1_thresh=0.2,
        ...     transformer_type="QwenImageTransformer2DModel"
        ... )
        >>> apply_teacache_hook(transformer, config)
    """
    registry = HookRegistry.get_or_create(module)
    hook = TeaCacheHook(config)
    registry.register_hook(TeaCacheHook._HOOK_NAME, hook)
