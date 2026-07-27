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
    """Execution engine for TeaCache native block boundaries."""

    def __init__(self, config: TeaCacheConfig) -> None:
        self.config = config
        self.rescale_func = np.poly1d(config.coefficients)
        self.state = TeaCacheRuntimeState()

    def _get_branch_state(self, do_true_cfg: bool) -> TeaCacheBranchState:
        if do_true_cfg:
            cfg_parallel_size = 1
            try:
                cfg_parallel_size = get_classifier_free_guidance_world_size()
            except AssertionError:
                # Direct model tests may run without initialized parallel state.
                cfg_parallel_size = 1
            if cfg_parallel_size > 1:
                cfg_rank = get_classifier_free_guidance_rank()
                return self.state.negative if cfg_rank > 0 else self.state.positive
            return self.state.negative if self.state.forward_cnt % 2 == 1 else self.state.positive
        return self.state.positive

    def _should_compute(self, state: TeaCacheBranchState, modulated_input: torch.Tensor) -> bool:
        if state.cnt == 0 or state.previous_modulated_input is None:
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

        if should_compute or branch_state.previous_residuals is None:
            # The block function can update its inputs in place, so snapshot the
            # exact tensors at the declared boundary before invoking it.
            input_clones = tuple(t.clone() for t in residual_inputs)
            outputs = compute_fn()

            if len(outputs) != len(residual_inputs):
                raise ValueError(
                    f"residual_inputs arity ({len(residual_inputs)}) does not match "
                    f"compute_fn output arity ({len(outputs)})."
                )

            for i, (out, inp) in enumerate(zip(outputs, input_clones)):
                if out.shape != inp.shape:
                    raise ValueError(f"Output tensor {i} shape {out.shape} does not match input shape {inp.shape}.")

            # Detached residuals retain the model's dtype, device, and sharding.
            residuals = tuple((out - inp).detach() for out, inp in zip(outputs, input_clones))
            branch_state.previous_residuals = residuals
        else:
            # A cache hit skips the blocks and applies the last block-region
            # delta to the current boundary inputs.
            outputs = tuple(inp + res for inp, res in zip(residual_inputs, branch_state.previous_residuals))

        branch_state.previous_modulated_input = modulated_input.detach()
        branch_state.cnt += 1
        self.state.forward_cnt += 1

        return outputs

    def reset(self) -> None:
        self.state.reset()
