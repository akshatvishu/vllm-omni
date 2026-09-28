# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch

from vllm_omni.diffusion.cache.teacache.protocol import ForwardState, SupportsDecomposedForward


@dataclass(frozen=True)
class SeaCacheInputs:
    """Original BCTHW vision latents in indicator order, retained by reference.

    Cosmos3 supplies controls followed by the noisy target. These inputs are
    separate from the packed execution tensor used to record the residual.
    """

    latents: list[torch.Tensor]
    noisy_frame_mask: torch.Tensor | None = None


@runtime_checkable
class SupportsSeaCache(SupportsDecomposedForward, Protocol):
    def get_seacache_inputs(self, ctx: ForwardState[Any]) -> SeaCacheInputs: ...
