# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.interface import TeaCacheBlockExecutor
from vllm_omni.diffusion.distributed.parallel_state import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)


@dataclass
class TeaCacheBranchState:
    cnt: int = 0
    accumulated_rel_l1_distance: float = 0.0
    previous_modulated_input: torch.Tensor | None = None
    previous_residuals: tuple[torch.Tensor, ...] | None = None

    def reset(self) -> None:
        self.cnt = 0
        self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input = None
        self.previous_residuals = None


@dataclass
class TeaCacheRuntimeState:
    forward_cnt: int = 0
    positive: TeaCacheBranchState = field(default_factory=TeaCacheBranchState)
    negative: TeaCacheBranchState = field(default_factory=TeaCacheBranchState)

    def reset(self) -> None:
        self.forward_cnt = 0
        self.positive.reset()
        self.negative.reset()


class TeaCacheRuntime(TeaCacheBlockExecutor):
    """Execution engine for a model-declared TeaCache block boundary."""

    def __init__(self, config: TeaCacheConfig) -> None:
        self.config = config
        self.rescale_func = np.poly1d(config.coefficients)
        self.state = TeaCacheRuntimeState()

    def _get_branch_state(self, do_true_cfg: bool) -> TeaCacheBranchState:
        if do_true_cfg:
            try:
                cfg_parallel_size = get_classifier_free_guidance_world_size()
            except AssertionError:
                # CPU unit tests can exercise the executor without distributed state.
                cfg_parallel_size = 1
            if cfg_parallel_size > 1:
                cfg_rank = get_classifier_free_guidance_rank()
                return self.state.negative if cfg_rank > 0 else self.state.positive
            return self.state.negative if self.state.forward_cnt % 2 else self.state.positive
        return self.state.positive

    def _should_compute(self, state: TeaCacheBranchState, modulated_input: torch.Tensor) -> bool:
        if state.cnt == 0 or state.previous_modulated_input is None:
            state.accumulated_rel_l1_distance = 0.0
            return True

        if (
            state.previous_modulated_input.shape != modulated_input.shape
            or state.previous_modulated_input.device != modulated_input.device
            or state.previous_modulated_input.dtype != modulated_input.dtype
        ):
            state.accumulated_rel_l1_distance = 0.0
            return True

        denom = state.previous_modulated_input.abs().mean() + 1e-8
        rel_distance = ((modulated_input - state.previous_modulated_input).abs().mean() / denom).cpu().item()
        if not np.isfinite(rel_distance):
            state.accumulated_rel_l1_distance = 0.0
            return True

        rescaled_distance = float(self.rescale_func(rel_distance))
        if not np.isfinite(rescaled_distance):
            state.accumulated_rel_l1_distance = 0.0
            return True
        state.accumulated_rel_l1_distance += abs(rescaled_distance)

        if state.accumulated_rel_l1_distance < self.config.rel_l1_thresh:
            return False
        state.accumulated_rel_l1_distance = 0.0
        return True

    @staticmethod
    def _residuals_match_inputs(residuals: tuple[torch.Tensor, ...], inputs: tuple[torch.Tensor, ...]) -> bool:
        return all(
            residual.shape == input_tensor.shape
            and residual.device == input_tensor.device
            and residual.dtype == input_tensor.dtype
            for residual, input_tensor in zip(residuals, inputs)
        )

    @torch.compiler.disable
    def run(
        self,
        *,
        modulated_input: torch.Tensor,
        residual_inputs: tuple[torch.Tensor, ...],
        compute_fn: Callable[[], tuple[torch.Tensor, ...]],
        do_true_cfg: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        if not residual_inputs:
            raise ValueError("residual_inputs tuple must not be empty.")

        branch_state = self._get_branch_state(do_true_cfg)
        should_compute = self._should_compute(branch_state, modulated_input)
        cached_residuals = branch_state.previous_residuals
        if cached_residuals is not None and (
            len(cached_residuals) != len(residual_inputs)
            or not self._residuals_match_inputs(cached_residuals, residual_inputs)
        ):
            # A changed sequence or batch shape cannot consume the old residual.
            # Force a real block execution so the new shape becomes the cache entry.
            branch_state.accumulated_rel_l1_distance = 0.0
            should_compute = True

        if should_compute or cached_residuals is None:
            # The block function can update its inputs in place, so snapshot the
            # exact tensors at the declared boundary before invoking it.
            input_clones = tuple(tensor.clone() for tensor in residual_inputs)
            outputs = compute_fn()

            if len(outputs) != len(residual_inputs):
                raise ValueError(
                    f"residual_inputs arity ({len(residual_inputs)}) does not match "
                    f"compute_fn output arity ({len(outputs)})."
                )

            for i, (output, input_clone) in enumerate(zip(outputs, input_clones)):
                if output.shape != input_clone.shape:
                    raise ValueError(
                        f"Output tensor {i} shape {output.shape} does not match input shape {input_clone.shape}."
                    )

            branch_state.previous_residuals = tuple(
                (output - input_clone).detach() for output, input_clone in zip(outputs, input_clones)
            )
        else:
            # A cache hit skips the blocks and applies the last block-region
            # delta to the current boundary inputs.
            outputs = tuple(
                input_tensor + residual for input_tensor, residual in zip(residual_inputs, cached_residuals)
            )

        branch_state.previous_modulated_input = modulated_input.detach()
        branch_state.cnt += 1
        self.state.forward_cnt += 1
        return outputs

    def reset(self) -> None:
        self.state.reset()
