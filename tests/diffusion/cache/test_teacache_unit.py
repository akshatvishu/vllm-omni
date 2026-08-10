# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import FrozenInstanceError
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.cache.teacache.backend import TeaCacheBackend
from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.interface import supports_teacache
from vllm_omni.diffusion.cache.teacache.runtime import TeaCacheRuntime
from vllm_omni.diffusion.data import DiffusionCacheConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


_COEFFICIENTS = [0.0, 0.0, 0.0, 1.0, 0.0]


class ValidTeaCacheModel:
    supports_teacache = True
    tea_cache_model_key = "TinyTeaCacheModel"
    tea_cache_executor = None

    def get_teacache_coefficients(self) -> list[float]:
        return list(_COEFFICIENTS)


class MissingKeyModel:
    supports_teacache = True
    tea_cache_model_key = ""
    tea_cache_executor = None


class MissingExecutorModel:
    supports_teacache = True
    tea_cache_model_key = "TinyTeaCacheModel"


class MissingCoefficientsModel:
    supports_teacache = True
    tea_cache_model_key = "TinyTeaCacheModel"
    tea_cache_executor = None


def _runtime(threshold: float = 0.5) -> TeaCacheRuntime:
    return TeaCacheRuntime(TeaCacheConfig(coefficients=_COEFFICIENTS, rel_l1_thresh=threshold))


def test_supports_teacache_validates_capability_and_negative_paths() -> None:
    assert supports_teacache(ValidTeaCacheModel()) is True
    assert supports_teacache(object()) is False

    with pytest.raises(ValueError, match="tea_cache_model_key"):
        supports_teacache(MissingKeyModel())
    with pytest.raises(ValueError, match="tea_cache_executor"):
        supports_teacache(MissingExecutorModel())
    with pytest.raises(ValueError, match="get_teacache_coefficients"):
        supports_teacache(MissingCoefficientsModel())


def test_config_is_finite_five_term_and_immutable() -> None:
    config = TeaCacheConfig(coefficients=_COEFFICIENTS, transformer_type="TinyTeaCacheModel")
    assert config.coefficients == tuple(_COEFFICIENTS)

    with pytest.raises((FrozenInstanceError, AttributeError)):
        config.rel_l1_thresh = 1.0
    with pytest.raises(ValueError, match="rel_l1_thresh"):
        TeaCacheConfig(coefficients=_COEFFICIENTS, rel_l1_thresh=0.0)
    with pytest.raises(ValueError, match="exactly 5"):
        TeaCacheConfig(coefficients=[1.0])
    with pytest.raises(ValueError, match="finite"):
        TeaCacheConfig(coefficients=[1.0, 0.0, 0.0, 0.0, float("nan")])


def test_runtime_uses_residual_on_hit_and_recomputes_on_metric_miss() -> None:
    runtime = _runtime()
    compute_count = 0

    def compute_fn():
        nonlocal compute_count
        compute_count += 1
        return (torch.tensor([2.0, 4.0]),)

    runtime.run(
        modulated_input=torch.ones(2),
        residual_inputs=(torch.tensor([1.0, 2.0]),),
        compute_fn=compute_fn,
    )
    hit = runtime.run(
        modulated_input=torch.ones(2),
        residual_inputs=(torch.tensor([1.5, 2.5]),),
        compute_fn=compute_fn,
    )
    assert compute_count == 1
    assert torch.equal(hit[0], torch.tensor([2.5, 4.5]))

    runtime.run(
        modulated_input=torch.full((2,), 10.0),
        residual_inputs=(torch.tensor([1.0, 2.0]),),
        compute_fn=compute_fn,
    )
    assert compute_count == 2


def test_runtime_shape_change_forces_compute_before_cache_hit() -> None:
    runtime = _runtime()
    compute_count = 0
    current_input = torch.ones(2)

    def compute_fn():
        nonlocal compute_count
        compute_count += 1
        return (current_input + 1.0,)

    runtime.run(
        modulated_input=torch.ones(2),
        residual_inputs=(current_input,),
        compute_fn=compute_fn,
    )
    current_input = torch.ones(3)
    output = runtime.run(
        modulated_input=torch.ones(2),
        residual_inputs=(current_input,),
        compute_fn=compute_fn,
    )

    assert compute_count == 2
    assert output[0].shape == (3,)


def test_runtime_sequential_cfg_keeps_branch_residuals_isolated() -> None:
    runtime = _runtime()
    compute_count = 0

    def compute_fn(value: float):
        def compute():
            nonlocal compute_count
            compute_count += 1
            return (torch.tensor([value]),)

        return compute

    runtime.run(
        modulated_input=torch.ones(1),
        residual_inputs=(torch.zeros(1),),
        compute_fn=compute_fn(1.0),
        do_true_cfg=True,
    )
    runtime.run(
        modulated_input=torch.ones(1),
        residual_inputs=(torch.zeros(1),),
        compute_fn=compute_fn(2.0),
        do_true_cfg=True,
    )
    positive_hit = runtime.run(
        modulated_input=torch.ones(1),
        residual_inputs=(torch.zeros(1),),
        compute_fn=compute_fn(3.0),
        do_true_cfg=True,
    )
    negative_hit = runtime.run(
        modulated_input=torch.ones(1),
        residual_inputs=(torch.zeros(1),),
        compute_fn=compute_fn(4.0),
        do_true_cfg=True,
    )

    assert compute_count == 2
    assert torch.equal(positive_hit[0], torch.tensor([1.0]))
    assert torch.equal(negative_hit[0], torch.tensor([2.0]))


def test_runtime_guards_boundary_contract() -> None:
    runtime = _runtime()

    with pytest.raises(ValueError, match="must not be empty"):
        runtime.run(
            modulated_input=torch.ones(1),
            residual_inputs=(),
            compute_fn=lambda: (torch.ones(1),),
        )
    with pytest.raises(ValueError, match="arity"):
        runtime.run(
            modulated_input=torch.ones(1),
            residual_inputs=(torch.ones(1), torch.ones(1)),
            compute_fn=lambda: (torch.ones(1),),
        )
    with pytest.raises(ValueError, match="shape"):
        runtime.run(
            modulated_input=torch.ones(1),
            residual_inputs=(torch.ones(1),),
            compute_fn=lambda: (torch.ones(2),),
        )


@pytest.mark.parametrize("metric", [torch.tensor([float("nan")]), torch.tensor([float("inf")])])
def test_runtime_nonfinite_metric_forces_recompute(metric: torch.Tensor) -> None:
    runtime = _runtime()
    compute_count = 0

    def compute_fn():
        nonlocal compute_count
        compute_count += 1
        return (torch.ones(1),)

    runtime.run(
        modulated_input=torch.zeros(1),
        residual_inputs=(torch.zeros(1),),
        compute_fn=compute_fn,
    )
    runtime.run(
        modulated_input=metric,
        residual_inputs=(torch.zeros(1),),
        compute_fn=compute_fn,
    )
    assert compute_count == 2


def test_runtime_reset_and_native_backend_lifecycle() -> None:
    model = ValidTeaCacheModel()
    pipeline = Mock(transformer=model)
    backend = TeaCacheBackend(DiffusionCacheConfig(rel_l1_thresh=0.25))
    backend.enable(pipeline)

    assert backend.enabled is True
    assert isinstance(model.tea_cache_executor, TeaCacheRuntime)
    model.tea_cache_executor.state.forward_cnt = 2
    backend.refresh(pipeline, num_inference_steps=4)
    assert model.tea_cache_executor.state.forward_cnt == 0
