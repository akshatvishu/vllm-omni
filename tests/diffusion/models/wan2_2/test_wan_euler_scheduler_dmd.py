# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.wan2_2.scheduling_wan_euler import WanEulerScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_dmd_predict_clean_and_add_noise_match_flow_formulas() -> None:
    scheduler = WanEulerScheduler(num_train_timesteps=1000, shift=8.0)
    timestep = torch.tensor(757.0)
    sigma = scheduler.sigma_for_timestep(timestep)
    sample = torch.tensor([2.0, -1.0])
    model_output = torch.tensor([0.5, 0.25])
    noise = torch.tensor([-0.5, 1.5])

    clean = scheduler.predict_clean(model_output, sample, timestep)
    expected_clean = sample - sigma * model_output
    torch.testing.assert_close(clean, expected_clean)

    renoised = scheduler.add_noise(clean, noise, timestep)
    expected_renoised = (1.0 - sigma) * expected_clean + sigma * noise
    torch.testing.assert_close(renoised, expected_renoised)


def test_dmd_sigma_requires_scalar_timestep() -> None:
    scheduler = WanEulerScheduler(num_train_timesteps=1000, shift=8.0)
    with pytest.raises(ValueError, match="scalar timestep"):
        scheduler.sigma_for_timestep(torch.tensor([757.0, 522.0]))


def test_dmd_predict_clean_matches_fastvideo_fp64_rounding() -> None:
    scheduler = WanEulerScheduler(num_train_timesteps=1000, shift=8.0)
    timestep = torch.tensor(757.0)
    sample = torch.tensor([0.330078125], dtype=torch.bfloat16)
    model_output = torch.tensor([1.4453125], dtype=torch.bfloat16)

    actual = scheduler.predict_clean(model_output, sample, timestep)
    sigma = scheduler.sigma_for_timestep(timestep, dtype=torch.float64)
    expected = (sample.double() - sigma * model_output.double()).to(model_output.dtype)
    old_fp32_result = (sample.float() - sigma.float() * model_output.float()).to(model_output.dtype)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == model_output.dtype
    assert not torch.equal(actual, old_fp32_result)
