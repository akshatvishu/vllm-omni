# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import torch

T = TypeVar("T")


@dataclass
class ForwardState(Generic[T]):
    """State passed between preprocess / run_transformer_blocks / postprocess
    for models that support TeaCache. The defined fields are required for
    TeaCache and read by the hook, while the generic intermediates type is a
    model class specific-state, which is a typed dataclass.

    This is the forward state for all models that support teacache, even in the
    case when its disabled, because in that case we would just have no modulated
    inputs.
    """

    modulated_input: torch.Tensor | None
    hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor | None
    temb: torch.Tensor
    intermediates: T


@runtime_checkable
class SupportsTeaCache(Protocol):
    """Decomposed forward for TeaCache integration; this tightly integrates Teacache
    with the model's forward to avoid duplicating code in a separate extractor.
    The expected flow is equivalent to as follows:

    Cache-disabled:

        ctx = self.preprocess(*args, skip_modulated_input=True, **kwargs)
        ctx = self.run_transformer_blocks(ctx)
        return self.postprocess(ctx)

    Cache-enabled (hook replaces forward):

        ctx = module.preprocess(*args, **kwargs)
        if cache_miss(ctx.modulated_input):
            ctx = module.run_transformer_blocks(ctx)
        else:
            ctx.hidden_states += cached_residual
        return module.postprocess(ctx)
    """

    def preprocess(self, *args: Any, skip_modulated_input: bool, **kwargs: Any) -> ForwardState:
        """Embed inputs, compute temb/RoPE, optionally extract modulated input."""
        ...

    def run_transformer_blocks(self, ctx: ForwardState) -> ForwardState:
        """Run all transformer blocks, update ctx.hidden_states in place."""
        ...

    def postprocess(self, ctx: ForwardState) -> Any:
        """Output norm + projection → final model output."""
        ...

    def get_teacache_coefficients(self) -> list[float]:
        """Instance-level polynomial coefficients for this model."""
        ...
