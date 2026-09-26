# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import SenseNovaU1Pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _pipeline_without_init() -> SenseNovaU1Pipeline:
    return object.__new__(SenseNovaU1Pipeline)


@pytest.mark.parametrize("kwargs", [None, {"step_i": 0}])
def test_combine_cfg_noise_requires_is_it2i(kwargs):
    pipe = _pipeline_without_init()
    out_cond = (torch.ones(1, 2, 3),)
    out_uncond = (torch.zeros(1, 2, 3),)

    with pytest.raises(ValueError, match="is_it2i"):
        pipe.combine_cfg_noise(
            out_cond,
            out_uncond,
            cfg_scale=4.0,
            cfg_norm="cfg_zero_star",
            kwargs=kwargs,
        )


@pytest.mark.parametrize(
    "kwargs",
    [{"is_it2i": False}, {"is_it2i": False, "step_i": None}],
)
def test_cfg_zero_star_requires_step_i(kwargs):
    pipe = _pipeline_without_init()
    out_cond = (torch.ones(1, 2, 3),)
    out_uncond = (torch.zeros(1, 2, 3),)
    with pytest.raises(ValueError, match="step_i"):
        pipe.combine_cfg_noise(
            out_cond,
            out_uncond,
            cfg_scale=4.0,
            cfg_norm="cfg_zero_star",
            kwargs=kwargs,
        )


def test_cfg_zero_star_accepts_step_i():
    pipe = _pipeline_without_init()
    out_cond = (torch.ones(1, 2, 3),)
    out_uncond = (torch.zeros(1, 2, 3),)
    result = pipe.combine_cfg_noise(
        out_cond,
        out_uncond,
        cfg_scale=4.0,
        cfg_norm="cfg_zero_star",
        kwargs={"is_it2i": False, "step_i": 0},
    )

    assert result.shape == out_cond[0].shape
    assert torch.equal(result, torch.zeros_like(out_cond[0]))
    assert torch.isfinite(result).all()


@pytest.mark.parametrize(
    ("cache_backend", "cfg_parallel_size", "cfg_scale", "img_cfg_scale", "reject"),
    [
        ("tea_cache", 2, 4.0, 2.0, True),
        ("none", 2, 4.0, 2.0, False),
        ("tea_cache", 1, 4.0, 2.0, False),
        ("tea_cache", 2, 4.0, 1.0, False),
        ("tea_cache", 2, 4.0, 4.0, False),
    ],
)
def test_three_branch_teacache_cfg_parallel_guard(cache_backend, cfg_parallel_size, cfg_scale, img_cfg_scale, reject):
    pipe = _pipeline_without_init()
    pipe.od_config = SimpleNamespace(
        cache_backend=cache_backend,
        parallel_config=SimpleNamespace(cfg_parallel_size=cfg_parallel_size),
    )
    params = SimpleNamespace(cfg_scale=cfg_scale, img_cfg_scale=img_cfg_scale)

    if reject:
        with pytest.raises(ValueError, match="three-branch image editing with CFG parallel size 2"):
            pipe._forward_it2i(params, [])
    else:
        with patch.object(SenseNovaU1Pipeline, "_init_noise_and_schedule", side_effect=RuntimeError("continued")):
            with pytest.raises(RuntimeError, match="continued"):
                pipe._forward_it2i(params, [])
