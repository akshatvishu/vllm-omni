# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Mapping
from dataclasses import dataclass, field

from vllm_omni.watermarking import WATERMARKER_REGISTRY

ALGORITHM_KEY = "algorithm"


@dataclass
class WatermarkConfig:
    """Select a registered algorithm per modality; additional options are not yet supported."""

    modality_configs: Mapping[str, Mapping[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for modality, config in self.modality_configs.items():
            if not isinstance(modality, str):
                raise ValueError(f"watermark modality must be a string, got {type(modality).__name__}")
            registered_algorithms = WATERMARKER_REGISTRY.get(modality)
            if registered_algorithms is None:
                supported_modalities = ", ".join(sorted(WATERMARKER_REGISTRY))
                raise ValueError(f"unsupported watermark modality {modality}; supported: {supported_modalities}")
            if not isinstance(config, Mapping):
                raise ValueError(f"watermark config for {modality} must be an object")
            unknown_keys = set(config) - {ALGORITHM_KEY}
            if unknown_keys:
                names = ", ".join(sorted(map(str, unknown_keys)))
                raise ValueError(
                    f"unsupported watermark config keys for {modality}: {names}; supported: {ALGORITHM_KEY}"
                )
            algorithm = config.get(ALGORITHM_KEY)
            if not isinstance(algorithm, str) or algorithm not in registered_algorithms:
                valid_algorithms = ", ".join(sorted(registered_algorithms))
                raise ValueError(
                    f"unsupported watermark algorithm {algorithm} for {modality}; supported: {valid_algorithms}"
                )
