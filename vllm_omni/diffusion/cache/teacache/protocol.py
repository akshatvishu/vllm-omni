# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import torch

T = TypeVar("T")


@dataclass
class ForwardState(Generic[T]):
    """Inputs and model-owned intermediates for a decomposed forward.

    Preprocess must provide anything postprocess needs on a cache hit, when
    run_transformer_blocks is skipped.
    """

    modulated_input: torch.Tensor | None
    hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor | None
    temb: torch.Tensor
    intermediates: T


@dataclass(frozen=True)
class TeaCacheDefaults:
    coefficients: list[float]
    rel_l1_thresh: float


@runtime_checkable
class SupportsDecomposedForward(Protocol):
    """The uncached forward must run these methods in order.

    Forward must pass its original arguments to preprocess with
    skip_modulated_input=True, then run the blocks and postprocess the result.
    Preprocess must handle argument normalization and save any values needed by
    postprocess, including options such as return_dict, in intermediates.
    """

    def preprocess(self, *args: Any, skip_modulated_input: bool, **kwargs: Any) -> ForwardState[Any]: ...

    def run_transformer_blocks(self, ctx: ForwardState[Any]) -> ForwardState[Any]: ...

    def postprocess(self, ctx: ForwardState[Any]) -> Any: ...


@runtime_checkable
class SupportsTeaCache(SupportsDecomposedForward, Protocol):
    def get_teacache_defaults(self) -> TeaCacheDefaults: ...


def validate_protocol_forward(module: torch.nn.Module) -> None:
    """Reject subclasses whose forward bypasses their inherited split."""

    classes = type(module).__mro__
    forward_owner = next((cls for cls in classes if "forward" in cls.__dict__), None)
    preprocess_owner = next((cls for cls in classes if "preprocess" in cls.__dict__), None)
    if forward_owner is None or preprocess_owner is None:
        raise TypeError(f"{type(module).__name__} does not define a complete decomposed forward")

    if classes.index(forward_owner) < classes.index(preprocess_owner):
        raise TypeError(
            f"{type(module).__name__}.forward overrides the decomposed forward from "
            f"{preprocess_owner.__name__}; TeaCache cannot safely bypass that override"
        )
