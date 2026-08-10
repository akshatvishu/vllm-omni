# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn

if torch.version.hip is None:
    pytest.skip("ROCm-only AITER GroupNorm test", allow_module_level=True)

from vllm_omni.platforms.rocm.patch.worker.patch_groupnorm import (  # noqa: E402
    _replace_groupnorm_with_aiter,
)

pytestmark = [pytest.mark.core_model, pytest.mark.gpu, pytest.mark.rocm]


def test_hunyuan_groupnorm_fp16_autocast_fp32_parameters():
    reference = nn.GroupNorm(32, 1024, eps=1e-6, device="cuda", dtype=torch.float32)
    vae = nn.Sequential(nn.GroupNorm(32, 1024, eps=1e-6, device="cuda", dtype=torch.float32))
    vae[0].load_state_dict(reference.state_dict())
    assert _replace_groupnorm_with_aiter(vae)
    input = torch.randn(1, 1024, 1, 64, 64, device="cuda", dtype=torch.float16)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        expected = reference(input)
        output = vae(input)
    torch.accelerator.synchronize()

    assert output.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-2)
