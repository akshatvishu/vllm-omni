# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TeaCacheConfig:
    """Configuration for a native TeaCache block executor."""

    coefficients: tuple[float, ...]
    rel_l1_thresh: float = 0.2
    transformer_type: str = "QwenImageTransformer2DModel"

    def __init__(
        self,
        coefficients: Sequence[float],
        rel_l1_thresh: float = 0.2,
        transformer_type: str = "QwenImageTransformer2DModel",
    ) -> None:
        if not math.isfinite(rel_l1_thresh) or rel_l1_thresh <= 0:
            raise ValueError(f"rel_l1_thresh must be positive, got {rel_l1_thresh}")

        coefficients_tuple = tuple(float(coefficient) for coefficient in coefficients)
        if len(coefficients_tuple) != 5:
            raise ValueError(f"coefficients must contain exactly 5 elements, got {len(coefficients_tuple)}")
        if not all(math.isfinite(coefficient) for coefficient in coefficients_tuple):
            raise ValueError("coefficients must contain only finite values")

        object.__setattr__(self, "coefficients", coefficients_tuple)
        object.__setattr__(self, "rel_l1_thresh", rel_l1_thresh)
        object.__setattr__(self, "transformer_type", transformer_type)
