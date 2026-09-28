# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch

from vllm_omni.diffusion.cache.teacache.protocol import ForwardState, SupportsDecomposedForward


@dataclass(frozen=True)
class SeaCacheInputs:
    """Original BCTHW target latent, retained by reference.

    Models exclude separate control hints, but retain clean prefix frames in
    partially conditioned targets. The packed execution tensor used for the
    residual still includes all conditioning inputs.
    """

    latent: torch.Tensor
    noisy_frame_mask: torch.Tensor | None = None


@runtime_checkable
class SupportsSeaCache(SupportsDecomposedForward, Protocol):
    def get_seacache_inputs(self, ctx: ForwardState[Any]) -> SeaCacheInputs: ...
