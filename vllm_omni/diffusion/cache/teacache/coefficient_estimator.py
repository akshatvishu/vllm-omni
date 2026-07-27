# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from vllm.config import LoadConfig
from vllm.transformers_utils.config import get_hf_file_to_dict

from vllm_omni.diffusion.cache.teacache.interface import TeaCacheBlockExecutor, supports_teacache
from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


class DataCollectionExecutor(TeaCacheBlockExecutor):
    """Collect native TeaCache boundary samples while executing the blocks."""

    def __init__(self):
        self.current_trajectory: list[tuple[np.ndarray, np.ndarray]] = []

    def run(
        self,
        *,
        modulated_input: torch.Tensor,
        residual_inputs: tuple[torch.Tensor, ...],
        compute_fn: Callable[[], tuple[torch.Tensor, ...]],
        do_true_cfg: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        # NOTE: We upcast to float32 to also handle bfloat16.
        modulated_input_cpu = modulated_input.detach().float().cpu().numpy()
        outputs = compute_fn()
        model_output_cpu = outputs[0].detach().float().cpu().numpy()
        self.current_trajectory.append((modulated_input_cpu, model_output_cpu))
        return outputs

    def start_collection(self):
        self.current_trajectory = []

    def stop_collection(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return list(self.current_trajectory)


class DefaultAdapter:
    """Default adapter for standard diffusers pipelines."""

    model_class_name = None
    uses_tf_config = True

    @classmethod
    def load_pipeline(cls, model_path: str, device: str, dtype: torch.dtype) -> Any:
        if cls.model_class_name is None:
            raise ValueError("Adapter doesn't have a set class name.")

        od_config = OmniDiffusionConfig.from_kwargs(
            model_class_name=cls.model_class_name,
            model=model_path,
            dtype=dtype,
        )

        if cls.uses_tf_config:
            # TODO (Alex): Refactor to handle tf_model_config in OmniDiffusionConfig
            # instead of OmniDiffusion and remove the manual population here
            tf_config_dict = get_hf_file_to_dict(
                os.path.join("transformer", "config.json"),
                od_config.model,
            )
            od_config.set_tf_model_config(TransformerConfig.from_dict(tf_config_dict))

        loader = DiffusersPipelineLoader(LoadConfig(), od_config=od_config)
        # load_model will handle dtypes / device placement, put in .eval() mode
        return loader.load_model(load_device=device)

    @staticmethod
    def get_transformer(pipeline: Any) -> tuple[Any, str]:
        transformer = pipeline.transformer
        return transformer, transformer.tea_cache_model_key


class BagelAdapter(DefaultAdapter):
    """Adapter for Bagel model."""

    model_class_name = "BagelPipeline"
    # Skip the hack for loading the tf model config,
    # because bagel doesn't use it.
    uses_tf_config = False

    @staticmethod
    def get_transformer(pipeline: Any) -> tuple[Any, str]:
        transformer = pipeline.bagel
        return transformer, transformer.tea_cache_model_key


class Flux2Adapter(DefaultAdapter):
    """Adapter for Flux2 model coefficient estimation."""

    model_class_name = "Flux2Pipeline"


class LongCatAdapter(DefaultAdapter):
    """Adapter for LongCat Image - NOTE: currently this model needs the vLLM
    context to be correctly configured to actually run the estimation, since it
    uses vLLM norm layers etc.
    """

    model_class_name = "LongCatImagePipeline"


class StableAudioAdapter(DefaultAdapter):
    """Adapter for Stable Audio Open 1.0 coefficient estimation."""

    model_class_name = "StableAudioPipeline"


class SenseNovaAdapter(DefaultAdapter):
    model_class_name = "SenseNovaU1Pipeline"

    @staticmethod
    def get_transformer(pipeline: Any) -> tuple[Any, str]:
        transformer = pipeline.denoising_transformer.language_model
        return transformer, transformer.tea_cache_model_key


_MODEL_ADAPTERS: dict[str, type] = {
    "Bagel": BagelAdapter,
    "Flux2": Flux2Adapter,
    "StableAudio": StableAudioAdapter,
    "LongCat": LongCatAdapter,
    "SenseNova": SenseNovaAdapter,
}

_EPSILON = 1e-6


def calculate_relative_l1(tensor_current: np.ndarray, tensor_next: np.ndarray) -> float:
    """Calculate relative L1 distance (Eq. 4 from TeaCache paper)."""
    diff = np.abs(tensor_current - tensor_next).sum()
    norm = np.abs(tensor_current).sum() + _EPSILON
    return diff / norm


def estimate_teacache_coefficients(
    collected_data: list[list[tuple[np.ndarray, np.ndarray]]], poly_order: int = 4
) -> list[float]:
    """Estimate polynomial coefficients for TeaCache using np.polyfit."""
    input_diffs, output_diffs = [], []

    for sample in collected_data:
        for t in range(len(sample) - 1):
            feat_in_curr, feat_out_curr = sample[t]
            feat_in_next, feat_out_next = sample[t + 1]
            input_diffs.append(calculate_relative_l1(feat_in_curr, feat_in_next))
            output_diffs.append(calculate_relative_l1(feat_out_curr, feat_out_next))

    x = np.array(input_diffs, dtype=np.float64)
    y = np.array(output_diffs, dtype=np.float64)

    print("Data statistics:")
    print(f"  Count: {len(x)}")
    print(f"  Input Diffs (x): min={x.min():.4e}, max={x.max():.4e}, mean={x.mean():.4e}")
    print(f"  Output Diffs (y): min={y.min():.4e}, max={y.max():.4e}, mean={y.mean():.4e}")

    return np.polyfit(x, y, poly_order).tolist()


class TeaCacheCoefficientEstimator:
    """Model-agnostic helper class to collect data and estimate TeaCache coefficients."""

    def __init__(
        self,
        model_path: str,
        model_type: str = "Bagel",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        if model_type not in _MODEL_ADAPTERS:
            available_types = list(_MODEL_ADAPTERS.keys())
            raise ValueError(
                f"Unsupported model_type: '{model_type}'. "
                f"Available types: {available_types}. "
                f"To add support for a new model, add an entry to _MODEL_ADAPTERS."
            )

        adapter = _MODEL_ADAPTERS[model_type]
        self.pipeline = adapter.load_pipeline(model_path, device, dtype)
        self.transformer, self.transformer_type = adapter.get_transformer(self.pipeline)
        if not supports_teacache(self.transformer):
            raise TypeError(f"{type(self.transformer).__name__} does not support native TeaCache")
        self.executor = DataCollectionExecutor()
        self.collected_data: list[list[tuple[np.ndarray, np.ndarray]]] = []
        self.transformer.tea_cache_executor = self.executor

    def collect_from_prompt(self, prompt: str, **generate_kwargs):
        self.executor.start_collection()
        req = OmniDiffusionRequest(
            prompt=prompt,
            request_id="teacache-coefficient-estimator",
            sampling_params=OmniDiffusionSamplingParams(
                num_inference_steps=generate_kwargs.get("num_inference_steps", 20),
                seed=generate_kwargs.get("seed", 42),
            ),
        )
        with torch.no_grad():
            self.pipeline.forward(DiffusionRequestBatch(requests=[req]))
        trajectory = self.executor.stop_collection()
        if trajectory:
            self.collected_data.append(trajectory)
        torch.accelerator.empty_cache()

    def estimate(self, poly_order: int = 4) -> list[float]:
        """Estimate polynomial coefficients from collected data.

        Args:
            poly_order: Order of polynomial fit (default: 4)

        Returns:
            List of polynomial coefficients [a_n, a_{n-1}, ..., a_1, a_0]

        Raises:
            RuntimeError: If no data has been collected
        """
        if not self.collected_data:
            raise RuntimeError(
                "No data collected for coefficient estimation. "
                "Call collect_from_prompt() at least once before calling estimate()."
            )
        return estimate_teacache_coefficients(self.collected_data, poly_order)
