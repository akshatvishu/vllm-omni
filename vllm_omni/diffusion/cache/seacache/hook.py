# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import IntEnum
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.seacache.config import SeaCacheConfig
from vllm_omni.diffusion.cache.seacache.protocol import SupportsSeaCache
from vllm_omni.diffusion.cache.seacache.sea_filter import (
    apply_sea_filter,
    extrapolate_residual,
    indicator_distance,
)
from vllm_omni.diffusion.cache.seacache.state import SeaCacheState
from vllm_omni.diffusion.cache.teacache.protocol import ForwardState, validate_protocol_forward
from vllm_omni.diffusion.hooks import HookRegistry, ModelHook, StateManager
from vllm_omni.diffusion.offloader.block_discovery import get_blocks_attr_names, get_blocks_from_dit

logger = init_logger(__name__)


def _is_parameter_sharded(module: torch.nn.Module) -> bool:
    """Detect parameter-sharding runtimes whose collectives cannot be skipped."""
    for submodule in module.modules():
        module_type = type(submodule)
        if callable(getattr(submodule, "_get_fsdp_state", None)):
            return True
        if module_type.__name__ == "FullyShardedDataParallel" and module_type.__module__.startswith(
            "torch.distributed.fsdp"
        ):
            return True
        for parameter in submodule.parameters(recurse=False):
            parameter_type = type(parameter)
            if (
                parameter_type.__name__ == "FlatParameter"
                and parameter_type.__module__.startswith("torch.distributed.fsdp")
            ) or (
                parameter_type.__name__ == "DTensor"
                and parameter_type.__module__.startswith("torch.distributed.tensor")
            ):
                return True
    return False


class SeaCacheDecision(IntEnum):
    """MAX reduction gives bypass precedence over compute and skip."""

    SKIP = 0
    COMPUTE = 1
    BYPASS = 2


class SeaCacheRootHook(ModelHook):
    """Drive SeaCache gating and transformer forward control."""

    _HOOK_NAME = "sea_cache"

    def __init__(
        self,
        config: SeaCacheConfig,
        *,
        current_step_callback: Callable[[], int | torch.Tensor | None] | None = None,
        current_sigma_callback: Callable[[], float | torch.Tensor | None] | None = None,
        num_inference_steps_callback: Callable[[], int | torch.Tensor | None] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.current_step_callback = current_step_callback
        self.current_sigma_callback = current_sigma_callback
        self.num_inference_steps_callback = num_inference_steps_callback
        self.state_manager = StateManager(SeaCacheState)
        self._warned_messages: set[str] = set()
        self.full_count = 0
        self.skip_count = 0
        self._active_branches: tuple[str, ...] = ()
        self._last_evaluation_step: int | None = None
        self._parameter_sharded = False
        self._collective_skip_groups: list[torch.distributed.ProcessGroup] = []

    def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        if not isinstance(module, SupportsSeaCache):
            raise TypeError(f"{type(module).__name__} must implement SupportsSeaCache")
        validate_protocol_forward(module)
        self._parameter_sharded = _is_parameter_sharded(module)
        seen_groups: set[int] = set()
        blocks = get_blocks_from_dit(module)[1] if get_blocks_attr_names(module) else []
        for block in blocks:
            registry = getattr(block, "_hook_registry", None)
            dlo_hook = registry.get_hook("distributed_layerwise_offload") if registry is not None else None
            group = getattr(dlo_hook, "dp_group", None)
            if group is not None and int(getattr(dlo_hook, "dp_size", 1)) > 1 and id(group) not in seen_groups:
                seen_groups.add(id(group))
                self._collective_skip_groups.append(group)
        return module

    def _warn_once(self, message: str) -> None:
        if message not in self._warned_messages:
            logger.warning(message)
            self._warned_messages.add(message)

    @contextmanager
    def cache_context(self, name: str) -> Iterator[None]:
        previous_context = self.state_manager._context
        self.state_manager.set_context(name)
        try:
            yield
        except BaseException:
            # A failed branch must not leave partially advanced trajectory state.
            self.state_manager.reset()
            self._active_branches = ()
            self._last_evaluation_step = None
            raise
        finally:
            self.state_manager.set_context(previous_context)

    def _step_metadata(self) -> tuple[int, float, int]:
        callbacks = (self.current_step_callback, self.current_sigma_callback, self.num_inference_steps_callback)
        if any(callback is None for callback in callbacks):
            raise ValueError("scheduler callbacks are unavailable")
        values = [callback() for callback in callbacks if callback is not None]
        values = [value.item() if isinstance(value, torch.Tensor) else value for value in values]
        step_value, sigma_value, num_steps_value = values
        if step_value is None or sigma_value is None or num_steps_value is None:
            raise ValueError("scheduler metadata is unavailable")
        step, sigma, num_steps = int(step_value), float(sigma_value), int(num_steps_value)
        if step < 0 or num_steps <= 0 or step >= num_steps or not math.isfinite(sigma) or not 0 <= sigma <= 1:
            raise ValueError("expected a valid step index and exact sigma in [0, 1]")
        return step, sigma, num_steps

    def begin_step(self, branches: tuple[str, ...]) -> None:
        """Register one velocity evaluation without precomputing branch decisions.

        All CFG ranks register the global branch tuple, including idle ranks.
        Changes in ownership/active guidance reset every local history together.
        Repeated evaluations at one solver index also restart extrapolation.
        Each forward computes a target-only indicator and retains its own residual.
        """
        if not branches or any(not name for name in branches) or len(set(branches)) != len(branches):
            raise ValueError("SeaCache requires unique, nonempty branch names")
        try:
            step, _, _ = self._step_metadata()
        except (IndexError, TypeError, ValueError, RuntimeError) as error:
            self._warn_once(f"SeaCache input is ineligible; running full: {error}")
            self.state_manager.reset()
            self._active_branches = ()
            self._last_evaluation_step = None
            return
        if branches != self._active_branches or self._last_evaluation_step != step - 1:
            self.state_manager.reset()
        self._active_branches = branches
        self._last_evaluation_step = step

    def _build_indicator(self, latent: torch.Tensor, sigma: float) -> list[torch.Tensor]:
        if not isinstance(latent, torch.Tensor) or latent.ndim != 5 or latent.shape[0] == 0:
            raise ValueError("expected a nonempty BCTHW target latent")

        # Keep one indicator per sample so distance averages samples equally.
        # The target retains clean prefix frames but excludes separate controls.
        return [
            apply_sea_filter(sample.movedim(0, -1), sigma=sigma, power_exp=self.config.power_exp).detach()
            for sample in latent
        ]

    def _resolve_gate(
        self,
        state: SeaCacheState,
        indicator: list[torch.Tensor],
        step: int,
        num_inference_steps: int,
    ) -> SeaCacheDecision:
        max_consecutive = bool(
            self.config.max_consecutive_cached and state.consecutive_cached >= self.config.max_consecutive_cached
        )
        forced_compute = (
            step < 1
            or step >= num_inference_steps - 1
            or max_consecutive
            or not state.history
            or state.previous_indicator is None
        )
        if forced_compute:
            state.accumulated_distance = 0.0
            state.previous_indicator = [value.detach() for value in indicator]
            return SeaCacheDecision.COMPUTE

        assert state.previous_indicator is not None
        distance = indicator_distance(indicator, state.previous_indicator)
        state.previous_indicator = [value.detach() for value in indicator]
        if not math.isfinite(distance):
            state.accumulated_distance = 0.0
            self._warn_once("SeaCache indicator history changed shape, device, or dtype; running full.")
            return SeaCacheDecision.COMPUTE

        state.accumulated_distance += distance
        if state.accumulated_distance < self.config.threshold:
            return SeaCacheDecision.SKIP
        state.accumulated_distance = 0.0
        return SeaCacheDecision.COMPUTE

    def _synchronize_decision(self, decision_value: SeaCacheDecision, device: torch.device) -> SeaCacheDecision:
        """MAX of skip=0, compute=1, bypass=2 across transformer collective peers."""
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return max(SeaCacheDecision.COMPUTE, decision_value) if self._parameter_sharded else decision_value
        decision = torch.tensor(decision_value, dtype=torch.int32, device=device)
        if self._parameter_sharded:
            from vllm_omni.diffusion.distributed.parallel_state import (
                get_fs_group,
                get_sequence_parallel_world_size,
                get_sp_group,
            )

            fs_group = get_fs_group()
            if fs_group.world_size > 1:
                torch.distributed.all_reduce(
                    decision,
                    op=torch.distributed.ReduceOp.MAX,
                    group=fs_group.device_group,
                )
            if get_sequence_parallel_world_size() > 1:
                torch.distributed.all_reduce(
                    decision,
                    op=torch.distributed.ReduceOp.MAX,
                    group=get_sp_group().device_group,
                )
            return SeaCacheDecision(decision.item())

        for group in self._collective_skip_groups:
            torch.distributed.all_reduce(
                decision,
                op=torch.distributed.ReduceOp.MAX,
                group=group,
            )
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_sequence_parallel_world_size,
            get_sp_group,
        )

        if get_sequence_parallel_world_size() > 1:
            torch.distributed.all_reduce(
                decision,
                op=torch.distributed.ReduceOp.MAX,
                group=get_sp_group().device_group,
            )
        return SeaCacheDecision(decision.item())

    @torch.compiler.disable
    def new_forward(
        self,
        module: torch.nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        ctx = module.preprocess(*args, **kwargs, skip_modulated_input=True)

        state: SeaCacheState | None = None
        local_decision = SeaCacheDecision.BYPASS
        try:
            if torch.is_grad_enabled():
                raise ValueError("autograd-enabled call")
            if self.state_manager._current_context is None:
                raise ValueError("missing explicit cache context")
            inputs = module.get_seacache_inputs(ctx)
            noisy_frame_mask = inputs.noisy_frame_mask
            if isinstance(noisy_frame_mask, torch.Tensor) and not bool(torch.any(noisy_frame_mask != 0).item()):
                raise ValueError("conditioning-only input")
            step, sigma, num_inference_steps = self._step_metadata()
            if step != self._last_evaluation_step:
                raise ValueError("begin_step was not called for this evaluation")
            state = self.state_manager.get_state()
            assert state is not None
            # Validate before collective agreement, never after a shared skip.
            if state.history and any(
                residual.shape != ctx.hidden_states.shape
                or residual.device != ctx.hidden_states.device
                or residual.dtype != ctx.hidden_states.dtype
                for _, residual in state.history
            ):
                state.reset()
            indicator = self._build_indicator(inputs.latent, sigma)
            local_decision = self._resolve_gate(state, indicator, step, num_inference_steps)
        except (IndexError, TypeError, ValueError, RuntimeError) as error:
            self._warn_once(f"SeaCache input is ineligible; running full: {error}")
        decision = self._synchronize_decision(local_decision, ctx.hidden_states.device)
        if decision == SeaCacheDecision.BYPASS:
            self.state_manager.reset()
            return self._run_uncached(module, ctx)
        assert state is not None
        if decision == SeaCacheDecision.COMPUTE:
            state.accumulated_distance = 0.0
            self.full_count += 1
            return self._run_and_record(module, ctx, state, step)

        residual = extrapolate_residual(
            state.history,
            step,
            self.config.residual_order,
        )
        state.consecutive_cached += 1
        self.skip_count += 1

        ctx.hidden_states = ctx.hidden_states + residual
        return module.postprocess(ctx)

    def _run_and_record(
        self,
        module: SupportsSeaCache,
        ctx: ForwardState[Any],
        state: SeaCacheState,
        step: int,
    ) -> Any:
        # Blocks may replace the state or mutate its input tensor in place.
        execution_input = ctx.hidden_states.clone()
        ctx = module.run_transformer_blocks(ctx)
        self._record_execution(state, step, execution_input, ctx.hidden_states)
        return module.postprocess(ctx)

    @staticmethod
    def _run_uncached(module: SupportsSeaCache, ctx: ForwardState[Any]) -> Any:
        return module.postprocess(module.run_transformer_blocks(ctx))

    def _record_execution(
        self,
        state: SeaCacheState,
        step: int,
        execution_input: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        if (
            output.shape == execution_input.shape
            and output.device == execution_input.device
            and output.dtype == execution_input.dtype
        ):
            state.history.append((step, (output - execution_input).detach()))
            state.history = state.history[-(self.config.residual_order + 1) :]
            state.consecutive_cached = 0
            return

        state.history.clear()
        state.accumulated_distance = 0.0
        self._warn_once("SeaCache execution boundary returned an incompatible tensor; clearing cache history.")

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        self.state_manager.reset()
        self._active_branches = ()
        self._last_evaluation_step = None
        self.full_count = 0
        self.skip_count = 0
        return module

    def refresh(self, module: torch.nn.Module) -> None:
        self.reset_state(module)


def apply_sea_cache_hook(
    module: torch.nn.Module,
    config: SeaCacheConfig,
    *,
    current_step_callback: Callable[[], int | torch.Tensor | None] | None = None,
    current_sigma_callback: Callable[[], float | torch.Tensor | None] | None = None,
    num_inference_steps_callback: Callable[[], int | torch.Tensor | None] | None = None,
) -> SeaCacheRootHook:
    registry = HookRegistry.get_or_create(module)
    hook = SeaCacheRootHook(
        config,
        current_step_callback=current_step_callback,
        current_sigma_callback=current_sigma_callback,
        num_inference_steps_callback=num_inference_steps_callback,
    )
    registry.register_hook(SeaCacheRootHook._HOOK_NAME, hook)
    return hook
