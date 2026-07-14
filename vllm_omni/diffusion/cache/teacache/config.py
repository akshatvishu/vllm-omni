# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

# Model-specific polynomial coefficients for rescaling L1 distances
# These coefficients account for model-specific characteristics in how embeddings change
# Source: TeaCache paper and ComfyUI-TeaCache empirical tuning
# Legacy coefficient registry — only used for models that don't implement
# SupportsTeaCache (i.e., HunyuanImage3 which bypasses the hook system).
# All other models provide coefficients via get_teacache_coefficients().
_MODEL_COEFFICIENTS = {
    "HunyuanImage3Pipeline": [
        1.04117826e02,
        -1.26848482e02,
        5.68168652e01,
        -1.04182570e01,
        6.78098549e-01,
    ],
    # MiniMax-H3 FL2VA coefficients.
    "MiniMaxH3DiTModel": [
        2.283704065852778e03,
        -7.775977277886368e02,
        9.408414741359490e01,
        -4.232669906169421e00,
        2.173782527946167e-01,
    ],
}

_DEFAULT_REL_L1_THRESH = 0.2
_MODEL_DEFAULT_REL_L1_THRESH = {
    "MiniMaxH3DiTModel": 0.17,
}


@dataclass
class TeaCacheConfig:
    """
    Configuration for TeaCache applied to transformer models.

    TeaCache (Timestep Embedding Aware Cache) is an adaptive caching technique that speeds up
    diffusion model inference by reusing transformer block computations when consecutive
    timestep embeddings are similar.

    Args:
        rel_l1_thresh: Threshold for accumulated relative L1 distance. When below threshold,
            cached residual is reused. If None, uses the model-specific default or 0.2
            when no model-specific default is registered.
        coefficients: Polynomial coefficients for rescaling L1 distance. If None, uses
            model-specific defaults based on transformer_type.
        transformer_type: Transformer class name (e.g., "QwenImageTransformer2DModel").
            Auto-detected from pipeline.transformer.__class__.__name__ in backend.
            Defaults to "QwenImageTransformer2DModel".
    """

    rel_l1_thresh: float | None = None
    coefficients: list[float] | None = None
    transformer_type: str = "QwenImageTransformer2DModel"

    def __post_init__(self) -> None:
        """Validate and set default coefficients."""
        threshold = self.rel_l1_thresh
        if threshold is None:
            threshold = _MODEL_DEFAULT_REL_L1_THRESH.get(self.transformer_type, _DEFAULT_REL_L1_THRESH)
        if threshold <= 0:
            raise ValueError(f"rel_l1_thresh must be positive, got {threshold}")
        self.rel_l1_thresh = threshold

        if self.coefficients is None:
            # Use model-specific coefficients, explicitly check if the type exists or not
            if self.transformer_type not in _MODEL_COEFFICIENTS:
                raise KeyError(
                    f"Cannot find coefficients for {self.transformer_type}. "
                    f"Supported: {list(_MODEL_COEFFICIENTS.keys())}"
                )
            self.coefficients = _MODEL_COEFFICIENTS[self.transformer_type]

        if len(self.coefficients) != 5:
            raise ValueError(f"coefficients must contain exactly 5 elements, got {len(self.coefficients)}")
