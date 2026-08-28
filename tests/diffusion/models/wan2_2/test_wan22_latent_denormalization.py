# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import _denormalize_wan_latents

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_wan_latent_denormalization_uses_direct_std_multiplication() -> None:
    latents = torch.tensor([[[[[-0.201171875]]]]], dtype=torch.bfloat16)
    mean = [-0.48828125]
    std = [2.671875]

    actual = _denormalize_wan_latents(latents, mean, std)
    expected = latents * torch.tensor(std, dtype=latents.dtype).view(1, 1, 1, 1, 1)
    expected += torch.tensor(mean, dtype=latents.dtype).view(1, 1, 1, 1, 1)
    reciprocal_result = latents / (1.0 / torch.tensor(std, dtype=latents.dtype).view(1, 1, 1, 1, 1)) + torch.tensor(
        mean, dtype=latents.dtype
    ).view(1, 1, 1, 1, 1)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == latents.dtype
    assert not torch.equal(actual, reciprocal_result)
