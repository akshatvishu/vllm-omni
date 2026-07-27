# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

import torch


@runtime_checkable
class TeaCacheBlockExecutor(Protocol):
    """Execute or reuse a model-declared TeaCache block region."""

    def run(
        self,
        *,
        modulated_input: torch.Tensor,
        residual_inputs: tuple[torch.Tensor, ...],
        compute_fn: Callable[[], tuple[torch.Tensor, ...]],
        do_true_cfg: bool = False,
    ) -> tuple[torch.Tensor, ...]: ...


@runtime_checkable
class SupportsTeaCache(Protocol):
    """Protocol for models that natively support TeaCache via tea_cache_executor."""

    supports_teacache: ClassVar[Literal[True]]
    tea_cache_model_key: ClassVar[str]
    tea_cache_executor: TeaCacheBlockExecutor | None

    def get_teacache_coefficients(self) -> list[float]:
        """Return polynomial rescaling coefficients for this model."""
        ...


def supports_teacache(module: Any) -> bool:
    """Validate whether a module satisfies the SupportsTeaCache protocol."""
    if getattr(module, "supports_teacache", False) is not True:
        return False

    model_key = getattr(module, "tea_cache_model_key", None)
    if not model_key or not isinstance(model_key, str):
        raise ValueError(
            f"Model {module.__class__.__name__} advertises supports_teacache=True "
            "but has a missing or invalid tea_cache_model_key."
        )

    if not hasattr(module, "tea_cache_executor"):
        raise ValueError(
            f"Model {module.__class__.__name__} advertises supports_teacache=True "
            "but is missing the tea_cache_executor attribute."
        )

    if not callable(getattr(module, "get_teacache_coefficients", None)):
        raise ValueError(
            f"Model {module.__class__.__name__} advertises supports_teacache=True "
            "but is missing get_teacache_coefficients()."
        )

    return True
