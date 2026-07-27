# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TeaCacheConfig:
    """
    Configuration for TeaCache applied to transformer models.

    Args:
        rel_l1_thresh: Threshold for accumulated relative L1 distance.
        coefficients: Polynomial coefficients for rescaling L1 distance.
        transformer_type: Target transformer key or class name.
    """

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

        coeffs_tuple = tuple(float(c) for c in coefficients)
        if len(coeffs_tuple) != 5:
            raise ValueError(f"coefficients must contain exactly 5 elements, got {len(coeffs_tuple)}")
        if not all(math.isfinite(c) for c in coeffs_tuple):
            raise ValueError("coefficients must contain only finite values")

        object.__setattr__(self, "rel_l1_thresh", rel_l1_thresh)
        object.__setattr__(self, "coefficients", coeffs_tuple)
        object.__setattr__(self, "transformer_type", transformer_type)
