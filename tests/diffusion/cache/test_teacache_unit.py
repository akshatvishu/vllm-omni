# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
CPU-based Unit Tests for TeaCache Refactor (TDD Plan).

Tests cover:
- Protocol validation & supports_teacache helper (happy + negative paths)
- TeaCacheConfig validation & immutability
- TeaCacheRuntime & TeaCacheBlockExecutor decision math (happy + negative paths)
- Arity/shape mismatch guards
- Zero-overhead when executor is None
- CFG branch isolation & state lifecycle (reset/refresh)
- Division-by-zero / NaN / Inf guards
"""

from dataclasses import FrozenInstanceError
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.cache.teacache.backend import TeaCacheBackend
from vllm_omni.diffusion.cache.teacache.coefficient_estimator import DataCollectionExecutor
from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.interface import (
    supports_teacache,
)
from vllm_omni.diffusion.cache.teacache.runtime import (
    TeaCacheRuntime,
)
from vllm_omni.diffusion.data import DiffusionCacheConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Dummy Mock Models for Protocol Testing
# ---------------------------------------------------------------------------


class ValidTeaCacheModel:
    supports_teacache = True
    tea_cache_model_key = "FluxTransformer2DModel"
    tea_cache_executor = None

    def get_teacache_coefficients(self) -> list[float]:
        return [1.0, 0.5, 0.2, 0.1, 0.05]


class MissingKeyModel:
    supports_teacache = True
    tea_cache_model_key = ""
    tea_cache_executor = None


class MissingExecutorAttrModel:
    supports_teacache = True
    tea_cache_model_key = "FluxTransformer2DModel"


class MissingCoefficientsModel:
    supports_teacache = True
    tea_cache_model_key = "FluxTransformer2DModel"
    tea_cache_executor = None


class UnsupportedModel:
    pass


# ---------------------------------------------------------------------------
# 1. Protocol & Capability Validation Tests
# ---------------------------------------------------------------------------


def test_supports_teacache_valid_model():
    model = ValidTeaCacheModel()
    assert supports_teacache(model) is True


def test_supports_teacache_unsupported_model():
    model = UnsupportedModel()
    assert supports_teacache(model) is False


def test_supports_teacache_missing_key_raises():
    model = MissingKeyModel()
    with pytest.raises(ValueError, match="tea_cache_model_key"):
        supports_teacache(model)


def test_supports_teacache_missing_executor_attr_raises():
    model = MissingExecutorAttrModel()
    with pytest.raises(ValueError, match="tea_cache_executor"):
        supports_teacache(model)


def test_supports_teacache_missing_coefficients_raises():
    model = MissingCoefficientsModel()
    with pytest.raises(ValueError, match="get_teacache_coefficients"):
        supports_teacache(model)


# ---------------------------------------------------------------------------
# 2. TeaCacheConfig Validation & Immutability Tests
# ---------------------------------------------------------------------------


def test_teacache_config_valid():
    config = TeaCacheConfig(
        rel_l1_thresh=0.25,
        coefficients=[1.0, 0.5, 0.2, 0.1, 0.05],
        transformer_type="FluxTransformer2DModel",
    )
    assert config.rel_l1_thresh == 0.25
    assert config.coefficients == (1.0, 0.5, 0.2, 0.1, 0.05)


def test_teacache_config_negative_threshold_raises():
    with pytest.raises(ValueError, match="rel_l1_thresh"):
        TeaCacheConfig(
            rel_l1_thresh=-0.1,
            coefficients=[1.0, 0.5, 0.2, 0.1, 0.05],
        )


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), float("-inf")])
def test_teacache_config_nonfinite_threshold_raises(threshold):
    with pytest.raises(ValueError, match="rel_l1_thresh"):
        TeaCacheConfig(rel_l1_thresh=threshold, coefficients=[1.0, 0.5, 0.2, 0.1, 0.05])


def test_teacache_config_invalid_coefficients_length_raises():
    with pytest.raises(ValueError, match="coefficients"):
        TeaCacheConfig(
            rel_l1_thresh=0.2,
            coefficients=[1.0, 0.5],  # Must be 5 coefficients
        )


def test_teacache_config_nonfinite_coefficients_raise():
    with pytest.raises(ValueError, match="finite"):
        TeaCacheConfig(coefficients=[1.0, 0.5, 0.2, 0.1, float("nan")])


def test_teacache_config_immutability():
    config = TeaCacheConfig(
        rel_l1_thresh=0.2,
        coefficients=[1.0, 0.5, 0.2, 0.1, 0.05],
    )
    with pytest.raises((FrozenInstanceError, AttributeError)):
        config.rel_l1_thresh = 0.5  # Should be frozen/immutable


# ---------------------------------------------------------------------------
# 3. TeaCacheRuntime & Executor Math Tests (Happy & Negative Paths)
# ---------------------------------------------------------------------------


def test_teacache_runtime_first_step_always_computes():
    config = TeaCacheConfig(rel_l1_thresh=0.2, coefficients=[1.0, 0.0, 0.0, 0.0, 0.0])
    runtime = TeaCacheRuntime(config)

    computed = False

    def compute_fn():
        nonlocal computed
        computed = True
        return (torch.tensor([1.0, 2.0]),)

    mod_in = torch.tensor([0.5, 0.5])
    out = runtime.run(
        modulated_input=mod_in,
        residual_inputs=(torch.tensor([0.0, 0.0]),),
        compute_fn=compute_fn,
        do_true_cfg=False,
    )
    assert computed is True
    assert torch.equal(out[0], torch.tensor([1.0, 2.0]))


def test_teacache_runtime_cache_hit_and_miss_sequence():
    config = TeaCacheConfig(rel_l1_thresh=0.5, coefficients=[0.0, 0.0, 0.0, 1.0, 0.0])  # f(x) = 1.0 * x
    runtime = TeaCacheRuntime(config)

    compute_count = 0

    def compute_fn():
        nonlocal compute_count
        compute_count += 1
        return (torch.tensor([2.0, 4.0]),)

    res_in = torch.tensor([1.0, 2.0])

    # Step 0: First step -> compute
    runtime.run(
        modulated_input=torch.tensor([1.0, 1.0]),
        residual_inputs=(res_in,),
        compute_fn=compute_fn,
        do_true_cfg=False,
    )
    assert compute_count == 1
    # Cached residual = out0 - res_in = [1.0, 2.0]

    # Step 1: Identical modulated input -> L1 dist = 0.0 < 0.5 -> Cache Hit!
    out1 = runtime.run(
        modulated_input=torch.tensor([1.0, 1.0]),
        residual_inputs=(torch.tensor([1.5, 2.5]),),
        compute_fn=compute_fn,
        do_true_cfg=False,
    )
    assert compute_count == 1  # Did NOT call compute_fn!
    # Expected out1 = new_res_in + cached_residual = [1.5, 2.5] + [1.0, 2.0] = [2.5, 4.5]
    assert torch.equal(out1[0], torch.tensor([2.5, 4.5]))

    # Step 2: Large change in modulated input -> L1 dist > 0.5 -> Cache Miss!
    runtime.run(
        modulated_input=torch.tensor([10.0, 10.0]),
        residual_inputs=(res_in,),
        compute_fn=compute_fn,
        do_true_cfg=False,
    )
    assert compute_count == 2  # Re-computed!


def test_teacache_runtime_snapshots_inputs_before_compute():
    config = TeaCacheConfig(rel_l1_thresh=0.5, coefficients=[0.0, 0.0, 0.0, 1.0, 0.0])
    runtime = TeaCacheRuntime(config)
    boundary = torch.tensor([1.0])

    def compute_fn():
        boundary.add_(10.0)
        return (boundary + 1.0,)

    runtime.run(
        modulated_input=torch.tensor([1.0]),
        residual_inputs=(boundary,),
        compute_fn=compute_fn,
    )
    (cached_output,) = runtime.run(
        modulated_input=torch.tensor([1.0]),
        residual_inputs=(torch.tensor([2.0]),),
        compute_fn=compute_fn,
    )

    # The cached delta is computed from the pre-call boundary value: 12 - 1.
    assert torch.equal(cached_output, torch.tensor([13.0]))


def test_teacache_runtime_arity_mismatch_raises():
    config = TeaCacheConfig(rel_l1_thresh=0.2, coefficients=[1.0, 0.0, 0.0, 0.0, 0.0])
    runtime = TeaCacheRuntime(config)

    # residual_inputs has 2 tensors, but compute_fn returns 1 tensor
    def bad_compute_fn():
        return (torch.tensor([1.0]),)

    with pytest.raises(ValueError, match="arity"):
        runtime.run(
            modulated_input=torch.tensor([1.0]),
            residual_inputs=(torch.tensor([1.0]), torch.tensor([2.0])),
            compute_fn=bad_compute_fn,
            do_true_cfg=False,
        )


def test_teacache_runtime_shape_mismatch_raises():
    config = TeaCacheConfig(rel_l1_thresh=0.2, coefficients=[1.0, 0.0, 0.0, 0.0, 0.0])
    runtime = TeaCacheRuntime(config)

    # residual_inputs is shape (2,), but compute_fn returns shape (3,)
    def bad_shape_compute_fn():
        return (torch.tensor([1.0, 2.0, 3.0]),)

    with pytest.raises(ValueError, match="shape"):
        runtime.run(
            modulated_input=torch.tensor([1.0]),
            residual_inputs=(torch.tensor([1.0, 2.0]),),
            compute_fn=bad_shape_compute_fn,
            do_true_cfg=False,
        )


def test_teacache_runtime_zero_division_guard():
    config = TeaCacheConfig(rel_l1_thresh=0.2, coefficients=[1.0, 0.0, 0.0, 0.0, 0.0])
    runtime = TeaCacheRuntime(config)

    def compute_fn():
        return (torch.tensor([1.0]),)

    # Step 0: All zeros modulated input
    runtime.run(
        modulated_input=torch.zeros(4),
        residual_inputs=(torch.tensor([0.0]),),
        compute_fn=compute_fn,
        do_true_cfg=False,
    )

    # Step 1: All zeros modulated input again (should not divide by zero or raise NaN)
    out = runtime.run(
        modulated_input=torch.zeros(4),
        residual_inputs=(torch.tensor([0.0]),),
        compute_fn=compute_fn,
        do_true_cfg=False,
    )
    assert not torch.isnan(out[0]).any()


@pytest.mark.parametrize("modulated_input", [torch.tensor([float("nan")]), torch.tensor([float("inf")])])
def test_teacache_runtime_nonfinite_metric_forces_recompute(modulated_input):
    config = TeaCacheConfig(rel_l1_thresh=0.2, coefficients=[1.0, 0.0, 0.0, 0.0, 0.0])
    runtime = TeaCacheRuntime(config)
    compute_count = 0

    def compute_fn():
        nonlocal compute_count
        compute_count += 1
        return (torch.tensor([1.0]),)

    runtime.run(
        modulated_input=torch.tensor([0.0]),
        residual_inputs=(torch.tensor([0.0]),),
        compute_fn=compute_fn,
    )
    runtime.run(
        modulated_input=modulated_input,
        residual_inputs=(torch.tensor([0.0]),),
        compute_fn=compute_fn,
    )

    assert compute_count == 2


def test_coefficient_collector_uses_native_executor_boundary():
    collector = DataCollectionExecutor()
    collector.start_collection()

    def compute_fn():
        return (torch.tensor([2.0]),)

    outputs = collector.run(
        modulated_input=torch.tensor([1.0]),
        residual_inputs=(torch.tensor([0.0]),),
        compute_fn=compute_fn,
    )

    assert torch.equal(outputs[0], torch.tensor([2.0]))
    assert len(collector.stop_collection()) == 1


# ---------------------------------------------------------------------------
# 4. CFG Branch Isolation & State Lifecycle Tests
# ---------------------------------------------------------------------------


def test_teacache_runtime_sequential_cfg_branch_alternation():
    config = TeaCacheConfig(rel_l1_thresh=0.5, coefficients=[0.0, 0.0, 0.0, 1.0, 0.0])
    runtime = TeaCacheRuntime(config)

    called_steps = []

    def compute_fn_positive():
        called_steps.append("pos")
        return (torch.tensor([1.0]),)

    def compute_fn_negative():
        called_steps.append("neg")
        return (torch.tensor([2.0]),)

    # Denoising Step 0 - Forward 0 (positive) -> first step -> compute
    runtime.run(
        modulated_input=torch.tensor([1.0]),
        residual_inputs=(torch.tensor([0.0]),),
        compute_fn=compute_fn_positive,
        do_true_cfg=True,
    )
    assert called_steps == ["pos"]

    # Denoising Step 0 - Forward 1 (negative) -> first step -> compute
    runtime.run(
        modulated_input=torch.tensor([2.0]),
        residual_inputs=(torch.tensor([0.0]),),
        compute_fn=compute_fn_negative,
        do_true_cfg=True,
    )
    assert called_steps == ["pos", "neg"]


def test_teacache_runtime_keeps_sequential_cfg_residuals_isolated():
    config = TeaCacheConfig(rel_l1_thresh=0.5, coefficients=[0.0, 0.0, 0.0, 1.0, 0.0])
    runtime = TeaCacheRuntime(config)
    compute_count = 0

    def compute_fn(value):
        def compute():
            nonlocal compute_count
            compute_count += 1
            return (torch.tensor([value]),)

        return compute

    for value in (10.0, 20.0):
        runtime.run(
            modulated_input=torch.tensor([1.0]),
            residual_inputs=(torch.tensor([0.0]),),
            compute_fn=compute_fn(value),
            do_true_cfg=True,
        )

    (positive_hit,) = runtime.run(
        modulated_input=torch.tensor([1.0]),
        residual_inputs=(torch.tensor([0.0]),),
        compute_fn=compute_fn(30.0),
        do_true_cfg=True,
    )
    (negative_hit,) = runtime.run(
        modulated_input=torch.tensor([1.0]),
        residual_inputs=(torch.tensor([0.0]),),
        compute_fn=compute_fn(40.0),
        do_true_cfg=True,
    )

    assert compute_count == 2
    assert torch.equal(positive_hit, torch.tensor([10.0]))
    assert torch.equal(negative_hit, torch.tensor([20.0]))


def test_teacache_runtime_reset_clears_all_state():
    config = TeaCacheConfig(rel_l1_thresh=0.5, coefficients=[0.0, 0.0, 0.0, 0.0, 1.0])
    runtime = TeaCacheRuntime(config)

    def compute_fn():
        return (torch.tensor([1.0]),)

    # Run step 0
    runtime.run(
        modulated_input=torch.tensor([1.0]),
        residual_inputs=(torch.tensor([0.0]),),
        compute_fn=compute_fn,
        do_true_cfg=False,
    )
    assert runtime.state.forward_cnt == 1

    # Reset
    runtime.reset()
    assert runtime.state.forward_cnt == 0
    assert runtime.state.positive.cnt == 0
    assert runtime.state.positive.previous_modulated_input is None
    assert runtime.state.negative.cnt == 0


# ---------------------------------------------------------------------------
# 5. TeaCacheBackend Enablement & Refresh Tests
# ---------------------------------------------------------------------------


def test_teacache_backend_enable_and_refresh():
    model = ValidTeaCacheModel()
    pipeline = Mock()
    pipeline.transformer = model

    backend = TeaCacheBackend(DiffusionCacheConfig(rel_l1_thresh=0.25))
    backend.enable(pipeline)

    assert backend.enabled is True
    assert model.tea_cache_executor is not None
    assert isinstance(model.tea_cache_executor, TeaCacheRuntime)

    # Test refresh resets runtime
    model.tea_cache_executor.state.forward_cnt = 5
    backend.refresh(pipeline)
    assert model.tea_cache_executor.state.forward_cnt == 0


def test_teacache_backend_enable_unsupported_model_raises():
    pipeline = Mock()
    pipeline.transformer = UnsupportedModel()

    backend = TeaCacheBackend(DiffusionCacheConfig(rel_l1_thresh=0.25))
    with pytest.raises((ValueError, TypeError), match="SupportsTeaCache"):
        backend.enable(pipeline)
